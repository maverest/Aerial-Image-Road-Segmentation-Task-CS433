import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from sklearn.metrics import f1_score
import cv2
from mask_to_submission import masks_to_submission


def prediction_on_test_set(model, test_loader, device,threshold = 0.5, display = False):
    model.eval()
    all_predictions = []
    paths = []
    with torch.no_grad():
        for images, path in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_predictions.append(outputs.cpu().squeeze(1))
            paths.append(path)
    all_predictions= torch.cat(all_predictions, dim=0)


    prob = [sigmoid(i.numpy()) for i in all_predictions]
    if display:
        for i in range(5):
            plt.imshow(prob[i])
            plt.show()
    final_mask = []
    for prob, path in zip(prob, paths):
        prediction = (prob > threshold).astype(np.uint8)
        filename = os.path.basename(path).replace(".png", "_mask.png")
        output_path = os.path.join("predictions", filename)
        plt.imsave(output_path, prediction, cmap="gray")
        final_mask.append(prediction)
    return final_mask



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fill_hole_masks(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_opened = mask.astype(np.uint8)
    for i in range(4):
        mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    return mask_closed

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def get_unique_filepath(filepath):
    """
    Generates a unique file path by appending a counter to the base name if the file already exists.
    """
    base, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(filepath):
        filepath = f"{base}_{counter}{ext}"
        counter += 1

    return filepath

def extract_val_image_mask(val_img_dir, val_mask_dir):
    val_img = []
    val_mask = []
    for img in os.listdir(val_img_dir):
        val_img.append(load_image(os.path.join(val_img_dir,img)))
    for mask in os.listdir(val_mask_dir):
        mask_array = load_image(os.path.join(val_mask_dir, mask))  # Load the mask
        binary_mask = (mask_array > 0.5).astype(int)  # Binarize the mask and convert to integer
        val_mask.append(binary_mask)
    return val_img, val_mask


def plot_images_with_predictions(images, masks, predictions, rows=3):
    """
    Plot figure with rows rows, where each row contains an image, its mask, and a prediction.
    """
    assert len(images) == len(masks) == len(predictions)

    num_to_display = min(rows, len(images))

    fig, axes = plt.subplots(num_to_display, 3, figsize=(15, 5 * num_to_display))

    if num_to_display == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_to_display):

        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')


        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 1].set_title("Original mask")
        axes[i, 1].axis('off')


        im = axes[i, 2].imshow(predictions[i], cmap='viridis')
        axes[i, 2].set_title("Predicted mask")
        axes[i, 2].axis('off')
        cbar = fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    return 0


def prediction_and_prob_from_saved_model(model, loader, device):
    all_preds = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1)
            outputs = model(images)
            all_preds.append(outputs.cpu().squeeze(1))
    all_preds = torch.cat(all_preds, dim=0)
    prob_mask = []
    for i in range(len(all_preds)):
        image = torch.sigmoid(all_preds[i]).numpy()
        prob_mask.append(image)
    return prob_mask, all_preds


# VALIDATION FUNCTIONS
def mask_to_submission_format(mask, patch_size=16, threshold=0.25):
    """
    Convert a mask into patch labels.
    Each patch is labeled as 1 (road) if more than `threshold` percent of its pixels are road (1).
    """
    #h, w = mask.shape #! cause erreur
    h, w = (400,400)
    patch_labels = np.zeros((h // patch_size, w // patch_size), dtype=int)

    for j in range(0, w, patch_size):
        for i in range(0, h, patch_size):
            patch = mask[i:i + patch_size, j:j + patch_size]
            if np.mean(patch) > threshold:  # Compute mean pixel value in the patch
                patch_labels[i // patch_size, j // patch_size] = 1
            else:
                patch_labels[i // patch_size, j // patch_size] = 0

    return patch_labels

def reconstruct_full_mask_from_patches(patch_labels, patch_size):
    """
    Reconstruct the full mask from patch labels.
    Each patch label (1 or 0) is expanded into a patch of size `patch_size`.
    """
    h_patches, w_patches = patch_labels.shape
    h_full = h_patches * patch_size
    w_full = w_patches * patch_size

    # Create an empty full mask
    full_mask = np.zeros((h_full, w_full), dtype=int)

    # Fill in the patches
    for i in range(h_patches):
        for j in range(w_patches):
            full_mask[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size
            ] = patch_labels[i, j]

    return full_mask


def compute_patch_tp_fp_fn(true_masks, pred_probs, threshold = 0.5, patch_size=16, patch_threshold=0.25):
    """
    Compute TP, FP, FN, and TN globally for patch-based labels.
    """
    tp, fp, fn, tn = 0, 0, 0, 0
    for true_mask, pred_prob in zip(true_masks, pred_probs):
        # Convert masks to patch labels
        true_patch_labels = mask_to_submission_format(true_mask, patch_size, patch_threshold)
        pred_patch_labels = mask_to_submission_format((pred_prob > threshold).astype(int), patch_size, patch_threshold)

        # Update TP, FP, FN, and TN globally
        tp += np.sum((true_patch_labels == 1) & (pred_patch_labels == 1))
        fp += np.sum((true_patch_labels == 0) & (pred_patch_labels == 1))
        fn += np.sum((true_patch_labels == 1) & (pred_patch_labels == 0))
        tn += np.sum((true_patch_labels == 0) & (pred_patch_labels == 0))  # Add TN calculation

    return tp, fp, fn, tn

def compute_f1(true_masks, pred_probs, patch_size=16, patch_threshold=0.25):
    """
    Compute F1 score globally for patch-based labels.
    """
    tp, fp, fn, tn = compute_patch_tp_fp_fn(true_masks = true_masks,pred_probs = pred_probs)

        # Compute Precision, Recall, and F1 globally
    precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1



def find_optimal_threshold_global(true_masks, pred_probs, thresholds, patch_size=16, patch_threshold=0.25):
    """
    Find the threshold that maximizes the global F1 score and compute accuracy for the optimal threshold.
    """
    best_f1 = 0
    best_threshold = 0
    best_accuracy = 0  # To store accuracy for the best threshold

    for threshold in thresholds:
        # Compute global TP, FP, FN, and TN
        tp, fp, fn, tn = compute_patch_tp_fp_fn(true_masks, pred_probs, threshold, patch_size, patch_threshold)

        # Compute Precision, Recall, and F1 globally
        precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Compute Accuracy globally
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

        # Update the best threshold and corresponding F1/Accuracy
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_accuracy = accuracy

    return best_threshold, best_f1, best_accuracy


def compare_masks_with_images(images, masks, num_display=5, alpha=0.5):
    # Determine grid size for subplots
    cols = min(3, num_display)  # Limit to 3 columns
    rows = (num_display + cols - 1) // cols  # Calculate rows dynamically

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # Flatten axes for easier indexing
    axes = axes.ravel() if num_display > 1 else [axes]

    for i in range(num_display):
        # Get the image and mask
        image = images[i]
        mask = masks[i]

        # Normalize image to [0, 1] if necessary
        if torch.is_tensor(image):
            image = image.cpu().numpy().transpose(1, 2, 0)  # Convert [C, H, W] to [H, W, C]
        if image.max() > 1:
            image = image / 255.0

        # Ensure mask is a numpy array
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        # Ensure the image is in RGB format
        if len(image.shape) == 2:  # Grayscale to RGB
            image = np.stack([image] * 3, axis=-1)

        # Create the overlay
        overlay = image.copy()
        overlay[:, :, 0] = np.maximum(overlay[:, :, 0], mask * alpha)  # Increase red channel
        overlay[:, :, 1] *= 1 - mask * alpha  # Dim green channel
        overlay[:, :, 2] *= 1 - mask * alpha  # Dim blue channel

        # Plot the overlay
        ax = axes[i]
        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(f"Image {i + 1}")

    # Turn off unused subplots
    for j in range(num_display, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def reconstruct_all_masks(patches, num_images, image_size, patch_size, stride):
    """
    Reconstruct multiple binary masks with uniform dimensions from a flat list of patches.
    :param patches: Flat list of all binary mask patches as PyTorch tensors (e.g., [patch1, patch2, ...]),
                    each with shape [1, 1, H, W] or already [H, W].
    :param num_images: Number of masks to reconstruct.
    :param image_size: Tuple (H, W) representing the uniform size of all masks.
    :param patch_size: Size of each patch (assumes square patches).
    :param stride: Stride used when creating the patches.
    :return: List of reconstructed binary masks as NumPy arrays.
    """
    h, w = image_size
    if stride is None:
        stride = patch_size

    # Calculate the number of patches per image
    patches_per_row = (h - patch_size) // stride + 1
    patches_per_col = (w - patch_size) // stride + 1
    patches_per_image = patches_per_row * patches_per_col

    reconstructed_masks = []  # To store reconstructed masks
    patch_idx = 0  # To track the current patch index

    for _ in range(num_images):
        # Initialize arrays for reconstruction
        reconstructed = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        # Reconstruct this mask
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                # Convert patch to NumPy and handle dimensions
                patch = patches[patch_idx].cpu().numpy() if torch.is_tensor(patches[patch_idx]) else patches[patch_idx]

                # Handle different shapes
                if patch.ndim == 4:  # If shape is [1, 1, H, W], remove batch and channel dimensions
                    patch = patch.squeeze(0).squeeze(0)
                elif patch.ndim == 3:  # If shape is [1, H, W], remove channel dimension
                    patch = patch.squeeze(0)

                reconstructed[i:i+patch_size, j:j+patch_size] += patch
                counts[i:i+patch_size, j:j+patch_size] += 1
                patch_idx += 1

        # Average overlapping regions
        reconstructed /= counts

        reconstructed_masks.append(reconstructed)

    return reconstructed_masks