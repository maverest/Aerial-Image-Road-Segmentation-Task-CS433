import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import random

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, threshold=128, patch_size=None, stride = None, num_augmentations=(0,0), CLAHE = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.threshold = threshold
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.num_rotation, self.num_flip = num_augmentations
        if self.num_rotation > 1:
            self.angle = np.linspace(25,90, self.num_rotation + 1).tolist()
        else :
            self.angle = [25,90]

        self.CLAHE = CLAHE

        # Generate samples
        self.samples = self._generate_samples()


    def get_path(self, index):
        return os.path.join(self.image_dir, self.images[index]), os.path.join(self.mask_dir, self.images[index])


    def get_patches(self, image, mask, patch_size, stride):
        h, w, _ = image.shape
        patches_img = []
        patches_mask = []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch_img = image[i:i + patch_size, j:j + patch_size]
                patch_mask = mask[i:i + patch_size, j:j + patch_size]
                patches_img.append(patch_img)
                patches_mask.append(patch_mask)
        return patches_img, patches_mask

    def img_float_to_uint8(self, img):
        rimg = img - np.min(img)
        rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
        return rimg

    def apply_CLAHE(self, image):
        # Convert image to uint8 (CLAHE works with uint8 data)
        image_uint8 = self.img_float_to_uint8(image)
        # Split into R, G, B channels
        r, g, b = cv2.split(image_uint8)
        # Create CLAHE model
        clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(8, 8))
        # Apply CLAHE to each channel
        r_clahe = clahe.apply(r)
        g_clahe = clahe.apply(g)
        b_clahe = clahe.apply(b)
        # Merge channels back
        return cv2.merge((r_clahe, g_clahe, b_clahe))


    def _generate_samples(self):
        all_image= []
        all_mask = []
        flip_list = ["vert", "horz"]
        for i in range(self.num_flip-2):
            flip_list.append(random.choice(["vert", "horz"]))
        for idx in range(len(self.images)):
            img_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.images[idx])

            # Load the image and mask
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

            # Binarize the mask
            mask = (mask > self.threshold).astype(np.float32)

            if self.CLAHE:
                image = self.apply_CLAHE(image)

            # Compute rotation angles for augmentations
            # rotation = [self.angle * i for i in range(self.num_augmentations)]

            # Generate augmented patches if patch_size is specified
            if self.patch_size:
                patches_img, patches_mask = self.get_patches(image, mask, self.patch_size, self.stride)

                for patch_img, patch_mask in zip(patches_img, patches_mask):
                    for n in range(self.num_rotation + self.num_flip+ 1):
                        if n == 0:
                            dynamic_transform = self.transform # if only 1 image, specify the transformation we want
                        elif n <= self.num_rotation:
                            dynamic_transform = A.Compose(
                                [
                                    A.Rotate(limit=(self.angle[n - 1], self.angle[n]),border_mode =cv2.BORDER_REFLECT, p=1.0), #Specify the reflected border mode
                                    A.VerticalFlip(p=0.4),
                                    A.HorizontalFlip(p=0.4),
                                ]
                                + (self.transform.transforms if self.transform else [])
                            )
                        else:
                            if self.num_flip == 1 :
                                flip_to_do = np.random.choice(["vert", "horz"])
                            else :
                                flip_to_do = flip_list[n - self.num_rotation - 1]
                            if flip_to_do == "vert":
                                dynamic_transform = A.Compose(
                                    [
                                        A.VerticalFlip(p=1.0),
                                        A.OneOf([
                                            A.Rotate(limit=[90, 90], p=1.0),  # Rotate by pi/2 (90 degrees)
                                            A.Rotate(limit=[180, 180], p=1.0),  # Rotate by pi (180 degrees)
                                            A.Rotate(limit=[-90, -90], p=1.0),  # Rotate by -pi/2 (-90 degrees)
                                        ], p=1.0),  # Ensure one of the rotations is always applied
                                    ] + (self.transform.transforms if self.transform else [])
                                )
                            else:
                                dynamic_transform = A.Compose(
                                    [
                                        A.HorizontalFlip(p=1.0),
                                        A.OneOf([
                                            A.Rotate(limit=[90, 90], p=1.0),  # Rotate by pi/2 (90 degrees)
                                            A.Rotate(limit=[180, 180], p=1.0),  # Rotate by pi (180 degrees)
                                            A.Rotate(limit=[-90, -90], p=1.0),  # Rotate by -pi/2 (-90 degrees)
                                        ], p=1.0),  # Ensure one of the rotations is always applied
                                    ] + (self.transform.transforms if self.transform else [])
                                )

                        # Ensure dynamic_transform is not None
                        if dynamic_transform:
                            augmented = dynamic_transform(image=patch_img, mask=patch_mask)
                            all_image.append(augmented["image"])
                            all_mask.append(augmented["mask"])
                        else:
                            # If no transform, append original patch
                            all_image.append(patch_img)
                            all_mask.append(patch_mask)

            else:
                # Generate augmentations for the full image
                for n in range(self.num_augmentations):
                    if n == 0:
                        dynamic_transform = self.transform
                    else:
                        dynamic_transform = A.Compose(
                            [A.Rotate(limit=(self.angle[n - 1], self.angle[n]), p=1.0)]
                            + (self.transform.transforms if self.transform else [])
                        )

                    # Ensure dynamic_transform is not None
                    if dynamic_transform:
                        augmented = dynamic_transform(image=image, mask=mask)
                        all_image.append(augmented["image"])
                        all_mask.append(augmented["mask"])
                    else:
                        # If no transform, append original image
                        all_image.append(image)
                        all_mask.append(mask)


        return all_image, all_mask

    def __len__(self):
        # Return the number of augmented samples
        return len(self.samples[0])

    def __getitem__(self, idx):
        # Return the idx-th augmented image and mask
        return self.samples[0][idx], self.samples[1][idx]

    

        
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None, patch_size=None, stride=None, CLAHE=False):
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride or patch_size  # Default stride is equal to patch size
        self.images = self._collect_image_paths()
        self.CLAHE = CLAHE
        self.samples = self._generate_samples()  # Precompute patches if patch_size is specified

    def img_float_to_uint8(self, img):
        rimg = img - np.min(img)
        rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
        return rimg

    def _collect_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(".png"):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _generate_samples(self):
        samples = []
        for img_path in self.images:
            image = np.array(Image.open(img_path).convert("RGB"))

            # Apply CLAHE if enabled
            if self.CLAHE:
                image = self.apply_CLAHE(image)

            # Extract patches if patch_size is provided
            if self.patch_size:
                patches_img = self.get_patches(image, self.patch_size, self.stride)
                for patch_img in patches_img:
                    if self.transform:
                        augmented = self.transform(image=patch_img)
                        samples.append((augmented["image"], img_path))  # Store transformed patch and path
                    else:
                        samples.append((patch_img, img_path))
            else:
                # Use the full image if no patch_size is specified
                if self.transform:
                    augmented = self.transform(image=image)
                    samples.append((augmented["image"], img_path))
                else:
                    samples.append((image, img_path))
        return samples

    def get_patches(self, image, patch_size, stride):
        h, w, _ = image.shape
        patches_img = []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch_img = image[i:i + patch_size, j:j + patch_size]
                patches_img.append(patch_img)
        return patches_img


    def apply_CLAHE(self, image):
        # Convert image to uint8 (CLAHE works with uint8 data)
        image_uint8 = self.img_float_to_uint8(image)
        # Split into R, G, B channels
        r, g, b = cv2.split(image_uint8)
        # Create CLAHE model
        clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(8, 8))
        # Apply CLAHE to each channel
        r_clahe = clahe.apply(r)
        g_clahe = clahe.apply(g)
        b_clahe = clahe.apply(b)
        # Merge channels back
        return cv2.merge((r_clahe, g_clahe, b_clahe))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

