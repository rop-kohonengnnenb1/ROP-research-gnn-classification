
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from datetime import datetime
from tqdm import tqdm

from utils.DatasetHandler import GraphImageDataset
from model.KohonenGNNenb1 import FusionClassifier


def create_fold_indices(val_fold, test_fold=4):
    """
    Based on the id of validation fold and test fold assign the training fold ids.
    """
    train_folds = [i for i in range(4) if i != val_fold]
    return {
        "train": train_folds,
        "val": [val_fold],
        "test": [test_fold]
    }


class StdoutLogger:
    def log(self, msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


class GNNTrainer:
    def __init__(self, graph_dir, image_dir, patches_dir, fold_indices, 
            model_save_path="weights.pth", batch_size=12, lr=0.0001, logger=None):
        """
        Graph neural network trainer.
        :param graph_dir: path to the folder with graphs (nodes positionings).
        :param image_dir: path to the folder with images.
        :param patches_dir: path to the folder with patches.
        :param fold_indices: dict with the folds ids.
        :param model_save_path: path where to save model weights.
        :param batch_size:
        :param lr:
        :param logger:
        """
        self.graph_dir = graph_dir
        self.image_dir = image_dir
        self.patches_dir = patches_dir
        self.fold_indices = fold_indices  # {"train": [...], "val": [...], "test": [...]}
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or StdoutLogger()

        self._prepare_data()
        self._prepare_model()

    def _prepare_data(self):
        self.train_loader = DataLoader(
            GraphImageDataset(self.graph_dir, self.image_dir, self.patches_dir, 
            self.fold_indices["train"]), batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            GraphImageDataset(self.graph_dir, self.image_dir, self.patches_dir,
            self.fold_indices["val"]), batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            GraphImageDataset(self.graph_dir, self.image_dir, self.patches_dir, 
            self.fold_indices["test"]), batch_size=self.batch_size, shuffle=False
        )

        self.class_weights = self.train_loader.dataset.get_class_weights().to(self.device)
        self.num_classes = len(self.train_loader.dataset.unique_labels)

        self.logger.log(f"Data prepared: train={len(self.train_loader.dataset)}, val={len(self.val_loader.dataset)}, test={len(self.test_loader.dataset)}")
        self.logger.log(f"Class weights: {self.class_weights.cpu().numpy()}")

    def _prepare_model(self):

        self.model = FusionClassifier(num_classes=self.num_classes)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.logger.log(f"Model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters")

    def train(self, epochs=0):
        best_val = 0

        for epoch in range(epochs):
            self.model.train()
            train_preds, train_labels, train_losses = [], [], []

            for geo_data in self.train_loader:

                geo_data = geo_data.to(self.device)

                outputs = self.model(geo_data)
                loss = self.criterion(outputs, geo_data.label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(geo_data.label.cpu().numpy())

            train_metrics = self._calculate_metrics(train_labels, train_preds, train_losses, "Train", epoch)

            val_metrics = self._evaluate(self.val_loader, epoch, phase="Val")
            test_metrics = self._evaluate(self.test_loader, epoch, phase="Test")

            if test_metrics["acc"] > best_val:
                best_val = test_metrics["acc"]
                torch.save(self.model.state_dict(), self.model_save_path)
                self.logger.log("Model saved.")
        
        self._evaluate(self.train_loader, -1, phase="Train")
        self._evaluate(self.val_loader, -1, phase="Val")
        self._evaluate(self.test_loader, -1, phase="Test")

    def _evaluate(self, loader, epoch, phase="Val"):
        self.model.eval()
        losses, preds, labels = [], [], []

        with torch.no_grad():
            for geo_data in loader:

                geo_data = geo_data.to(self.device)
                outputs = self.model(geo_data)
                loss = self.criterion(outputs, geo_data.label)

                losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                labels.extend(geo_data.label.cpu().numpy())

        return self._calculate_metrics(labels, preds, losses, phase, epoch)

    def _calculate_metrics(self, labels, preds, losses, phase, epoch):
        weights = self.class_weights[labels].cpu().numpy()
        acc = accuracy_score(labels, preds, sample_weight=weights)
        precision = precision_score(labels, preds, average='weighted', sample_weight=weights)
        recall = recall_score(labels, preds, average='weighted', sample_weight=weights)
        f1 = f1_score(labels, preds, average='weighted', sample_weight=weights)
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        self.logger.log(f"{phase} Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

        return {"loss": avg_loss, "acc": acc, "precision": precision, "recall": recall, "f1": f1}



if __name__ == "__main__":
    graph_dir = # PATH TO THE GRAPHS FOLDER
    image_dir = # PATH TO THE IMAGES FOLDER
    patches_dir = # PATH TO THE PATCHES FOLDER
    
    val_fold = 3  # validation folder ID

    trainer = GNNTrainer(
        graph_dir=graph_dir, image_dir=image_dir, patches_dir=patches_dir,
        fold_indices=create_fold_indices(val_fold), 
        model_save_path=f"weights_{val_fold}_norm.pth"
    )
    trainer.train(epochs=100)
