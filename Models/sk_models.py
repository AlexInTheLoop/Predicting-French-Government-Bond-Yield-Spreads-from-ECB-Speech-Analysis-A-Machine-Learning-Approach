import joblib
import numpy as np
from Models.base import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb #type: ignore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import RidgeClassifier, SGDClassifier
import lightgbm as lgb


class LogReg(BaseModel):
    def __init__(self, **kwargs):
        BaseModel.__init__(self, name="LogisticRegression")
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        return self


class SVM(BaseModel):
    def __init__(self, kernel="linear", probability=False, **kwargs):
        BaseModel.__init__(self, name="SVM")
        if kernel == "linear":
            self.model = LinearSVC(**kwargs)
            self._linear = True
            self._probability = False
        else:
            self.model = SVC(kernel=kernel, probability=probability, **kwargs)
            self._linear = False
            self._probability = probability

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self._linear and self._probability:
            return self.model.predict_proba(X)
        return None

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        return self


class RandomForest(BaseModel):
    def __init__(self, **kwargs):
        BaseModel.__init__(self, name="RandomForest")
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        return self


class XGBoostClassifier(BaseModel):
    def __init__(self, auto_remap_labels=True, **kwargs):
        BaseModel.__init__(self, name="XGBoost")
        self.model = xgb.XGBClassifier(**kwargs)
        self.auto_remap_labels = auto_remap_labels
        self.le = None
        self.classes_ = None 

    def _maybe_encode(self, y):
        if not self.auto_remap_labels:
            return y
        if self.le is None:
            self.le = LabelEncoder()
            y_enc = self.le.fit_transform(y)
            self.classes_ = self.le.classes_
        else:
            y_enc = self.le.transform(y)
        return y_enc

    def _maybe_decode(self, y_pred):
        if self.auto_remap_labels and self.le is not None:
            return self.le.inverse_transform(y_pred)
        return y_pred

    def _maybe_decode_proba(self, proba):
        return proba

    def fit(self, X, y, X_val=None, y_val=None):
        y_enc = self._maybe_encode(y)

        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_enc = self._maybe_encode(y_val)
            eval_set = [(X_val, y_val_enc)]

        self.model.fit(X, y_enc, eval_set=eval_set, verbose=False)
        self.is_fitted = True
        return self

    def predict(self, X):
        y_pred_enc = self.model.predict(X)
        return self._maybe_decode(y_pred_enc.astype(int))

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        return self._maybe_decode_proba(proba)

    def save(self, path):
        self.model.save_model(path + ".xgb")
        meta = {
            "auto_remap_labels": self.auto_remap_labels,
            "classes_": self.classes_.tolist() if self.classes_ is not None else None
        }
        joblib.dump({"meta": meta, "label_encoder": self.le}, path + ".meta")

    def load(self, path):
        self.model = xgb.XGBClassifier()
        self.model.load_model(path + ".xgb")
        payload = joblib.load(path + ".meta")
        self.auto_remap_labels = payload["meta"]["auto_remap_labels"]
        self.classes_ = np.array(payload["meta"]["classes_"]) if payload["meta"]["classes_"] is not None else None
        self.le = payload["label_encoder"]
        self.is_fitted = True
        return self


class KNNClassifier(BaseModel):
    def __init__(self, n_neighbors=5, weights='distance', metric='cosine', **kwargs):
        BaseModel.__init__(self, name="KNN")
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, 
            weights=weights, 
            metric=metric,
            **kwargs
        )

    def fit(self, X, y, X_val=None, y_val=None):
        if hasattr(X, "toarray"):
            X = X.toarray()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        return self


class NaiveBayesClassifier(BaseModel):
    def __init__(self, variant="multinomial", **kwargs):
        BaseModel.__init__(self, name=f"NaiveBayes({variant})")
        if variant == "multinomial":
            self.model = MultinomialNB(**kwargs)
        elif variant == "complement":
            self.model = ComplementNB(**kwargs)
        else:
            raise ValueError("variant must be 'multinomial' or 'complement'")

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        return self


class LightGBMClassifier(BaseModel):
    def __init__(self, auto_remap_labels=True, **kwargs):
        BaseModel.__init__(self, name="LightGBM")
        default_params = {
            'objective': 'multiclass',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        default_params.update(kwargs)
        self.model = lgb.LGBMClassifier(**default_params)
        self.auto_remap_labels = auto_remap_labels
        self.le = None
        self.classes_ = None

    def _encode(self, y):
        if not self.auto_remap_labels:
            return y
        if self.le is None:
            self.le = LabelEncoder()
            y_enc = self.le.fit_transform(y)
            self.classes_ = self.le.classes_
        else:
            y_enc = self.le.transform(y)
        return y_enc

    def _decode(self, y_pred):
        if self.auto_remap_labels and self.le is not None:
            return self.le.inverse_transform(y_pred)
        return y_pred

    def fit(self, X, y, X_val=None, y_val=None):
        y_enc = self._encode(y)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_enc = self._encode(y_val)
            eval_set = [(X_val, y_val_enc)]

        self.model.fit(X, y_enc, eval_set=eval_set, callbacks=[lgb.early_stopping(50)])
        self.is_fitted = True
        return self

    def predict(self, X):
        y_pred_enc = self.model.predict(X)
        return self._decode(y_pred_enc.astype(int))

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        self.model.booster_.save_model(path + ".lgb")
        meta = {
            "auto_remap_labels": self.auto_remap_labels,
            "classes_": self.classes_.tolist() if self.classes_ is not None else None
        }
        joblib.dump({"meta": meta, "label_encoder": self.le}, path + ".meta")

    def load(self, path):
        self.model = lgb.Booster(model_file=path + ".lgb")
        payload = joblib.load(path + ".meta")
        self.auto_remap_labels = payload["meta"]["auto_remap_labels"]
        self.classes_ = np.array(payload["meta"]["classes_"]) if payload["meta"]["classes_"] is not None else None
        self.le = payload["label_encoder"]
        self.is_fitted = True
        return self


class ExtraTreesClassifierWrapper(BaseModel):
    def __init__(self, **kwargs):
        BaseModel.__init__(self, name="ExtraTrees")
        self.model = ExtraTreesClassifier(**kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        return self


class AdaBoostClassifierWrapper(BaseModel):
    def __init__(self, **kwargs):
        BaseModel.__init__(self, name="AdaBoost")
        self.model = AdaBoostClassifier(**kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        return self


class RidgeClassifierWrapper(BaseModel):
    def __init__(self, **kwargs):
        BaseModel.__init__(self, name="Ridge")
        self.model = RidgeClassifier(**kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return None

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        return self


class SGDClassifierWrapper(BaseModel):
    def __init__(self, **kwargs):
        BaseModel.__init__(self, name="SGD")
        default_params = {
            'loss': 'log_loss',
            'alpha': 0.0001,
            'max_iter': 1000,
            'tol': 1e-3
        }
        default_params.update(kwargs)
        self.model = SGDClassifier(**default_params)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        return self