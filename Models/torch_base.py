# Models/torch_base.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from Models.base import BaseModel


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TorchBaseClassifier(BaseModel):

    def __init__(self, model, name=None, lr=1e-3, weight_decay=0.0, batch_size=64, epochs=20,
                 device=None, early_stopping_patience=5, verbose=True, auto_remap_labels=True):
        BaseModel.__init__(self, name=name or model.__class__.__name__)
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.auto_remap_labels = auto_remap_labels

        self.le = None
        self.classes_ = None
        self.model.to(self.device)
        self.best_state = None

    def _encode_labels(self, y):
        if not self.auto_remap_labels:
            return y
        if self.le is None:
            self.le = LabelEncoder()
            y_enc = self.le.fit_transform(y)
            self.classes_ = self.le.classes_
        else:
            y_enc = self.le.transform(y)
        return y_enc

    def _decode_labels(self, y_pred):
        if self.auto_remap_labels and self.le is not None:
            return self.le.inverse_transform(y_pred)
        return y_pred


    def _create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _create_criterion(self, class_weights=None):
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        return torch.nn.CrossEntropyLoss(weight=class_weights)

    def fit(self, X, y, X_val=None, y_val=None, class_weights=None):
        y_enc = self._encode_labels(y)
        y_val_enc = self._encode_labels(y_val) if y_val is not None else None

        train_ds = NumpyDataset(X, y_enc)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_ds = NumpyDataset(X_val, y_val_enc)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        optim = self._create_optimizer()
        criterion = self._create_criterion(class_weights)

        best_val_loss = np.inf
        patience = 0

        for _ in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        logits = self.model(xb)
                        loss = criterion(logits, yb)
                        val_loss += loss.item() * xb.size(0)
                val_loss /= len(val_loader.dataset)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= self.early_stopping_patience:
                        break

        self.is_fitted = True
        return self

    def predict(self, X):
        self.model.eval()
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return self._decode_labels(preds)

    def predict_proba(self, X):
        self.model.eval()
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    
    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "classes": self.classes_,
            "label_encoder": self.le
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.classes_ = checkpoint["classes"]
        self.le = checkpoint["label_encoder"]
        self.is_fitted = True
        return self