import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from Models.base import BaseModel

class HFTextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.texts[idx]
        return self.texts[idx], self.labels[idx]


class HFTransformerClassifier(BaseModel):
    def __init__(self,
                 model_name="camembert-base",
                 max_length=512,
                 lr=2e-5,
                 weight_decay=0.0,
                 batch_size=8,
                 epochs=3,
                 warmup_ratio=0.06,
                 gradient_accumulation_steps=1,
                 early_stopping_patience=2,
                 eval_every=1,
                 device=None,
                 fp16=False,
                 auto_remap_labels=True,
                 name=None,
                 verbose=True):

        BaseModel.__init__(self, name=name or f"HFTransformer({model_name})")

        self.model_name = model_name
        self.max_length = max_length
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.eval_every = eval_every
        self.verbose = verbose
        self.fp16 = fp16

        self.auto_remap_labels = auto_remap_labels
        self.le = None
        self.classes_ = None

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.best_state_dict = None

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

    def _collate_fn(self, batch):
        if isinstance(batch[0], tuple):
            texts, labels = zip(*batch)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            texts = batch
            labels = None

        tok = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        if labels is not None:
            tok["labels"] = labels
        return tok

    def fit(self, X, y, X_val=None, y_val=None, class_weights=None):
        y_enc = self._encode_labels(y)
        num_labels = len(np.unique(y_enc))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        ).to(self.device)

        train_ds = HFTextDataset(X, y_enc)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=self._collate_fn
        )

        if X_val is not None and y_val is not None:
            y_val_enc = self._encode_labels(y_val)
            val_ds = HFTextDataset(X_val, y_val_enc)
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False,
                collate_fn=self._collate_fn
            )
        else:
            val_loader = None

        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        warmup_steps = int(self.warmup_ratio * total_steps) if total_steps > 0 else 0
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        best_val_loss = np.inf
        patience = 0
        global_step = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0

            for step, batch in enumerate(train_loader, start=1):
                for k in batch:
                    batch[k] = batch[k].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if step % self.gradient_accumulation_steps == 0:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
                    scheduler.step()

                running_loss += loss.item() * batch["input_ids"].size(0)
                global_step += 1


            if val_loader is not None and (epoch % self.eval_every == 0):
                val_loss = self._evaluate_loss(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= self.early_stopping_patience:
                        break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        self.is_fitted = True
        return self

    def _evaluate_loss(self, loader):
        self.model.eval()
        loss_sum = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                bs = batch["input_ids"].size(0)
                loss_sum += loss.item() * bs
                n += bs
        return loss_sum / n

    def predict(self, X):
        self.model.eval()
        ds = HFTextDataset(X, labels=None)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)

        preds = []
        with torch.no_grad():
            for batch in loader:
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                outputs = self.model(**batch)
                logits = outputs.logits
                p = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(p)
        preds = np.concatenate(preds)
        return self._decode_labels(preds)

    def predict_proba(self, X):
        self.model.eval()
        ds = HFTextDataset(X, labels=None)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)

        probs = []
        with torch.no_grad():
            for batch in loader:
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                outputs = self.model(**batch)
                logits = outputs.logits
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(p)
        probs = np.concatenate(probs)
        return probs

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save({
            "classes_": self.classes_,
            "le": self.le
        }, os.path.join(path, "label_encoder.pt"))

    def load(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)

        ckpt = torch.load(os.path.join(path, "label_encoder.pt"), map_location=self.device)
        self.classes_ = ckpt["classes_"]
        self.le = ckpt["le"]
        self.is_fitted = True
        return self