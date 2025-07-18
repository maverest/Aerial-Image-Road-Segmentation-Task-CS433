import json
import matplotlib.pyplot as plt

class TrainingTracker:
    def __init__(self, hyperparams=None):
        self.hyperparams = hyperparams if hyperparams else {}
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": {},
            "val_metrics": {},
            "lr": [],
        }

    def log_hyperparams(self, hyperparams):
        self.hyperparams.update(hyperparams)

    def log_epoch(self, train_loss = None, val_loss = None, train_metrics = None, val_metrics = None):
        try :
            self.history["train_metrics"].append(train_metrics)
            self.history["val_metrics"].append(val_metrics)
        except :
            pass
        try :
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

        except :
            pass


    def log_lr(self, lr):
        self.history["lr"].append(lr)

    def save(self, filepath = "history/history.json" ):
        data = {
            "hyperparams": self.hyperparams,
            "history": self.history,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"History saved in {filepath}.")

    def plot_metrics(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Configuration pour l'axe y basé uniquement sur la perte d'entraînement
        min_train_loss = min(self.history["train_loss"])
        max_train_loss = max(self.history["train_loss"])
        margin = (max_train_loss - min_train_loss) * 0.1  # Ajout d'une marge de 10% pour l'affichage
        ax1.set_ylim(min_train_loss - margin, max_train_loss + margin)

        # Configuration de l'axe x et des courbes
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="tab:blue")
        ax1.plot(epochs, self.history["train_loss"], label="Train Loss", color="tab:blue")
        ax1.plot(epochs, self.history["val_loss"], label="Validation Loss", color="tab:orange")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.legend(loc="upper left")
        ax1.grid(True)

        # Ajout de la courbe de taux d'apprentissage (learning rate) si disponible
        if "lr" in self.history and len(self.history["lr"]) > 0:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Learning Rate")
            lr_epochs = range(1, len(self.history["lr"]) + 1)
            ax2.plot(lr_epochs, self.history["lr"], color="tab:green", linestyle="--", label="Learning Rate")
            ax2.tick_params(axis="y", labelcolor="tab:green")
            ax2.legend(loc="upper right")

        # Titre et mise en page finale
        plt.title("Training Progress: Loss and Learning Rate")
        fig.tight_layout()
        plt.show()

