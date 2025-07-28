from Models.metrics import compute_all_metrics

class BaseModel:
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.is_fitted = False

    def fit(self, X, y, X_val=None, y_val=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        return None

    def evaluate(self, X, y, average="macro", labels=None, output_dict=False):
        y_pred = self.predict(X)
        y_proba = None
        try:
            y_proba = self.predict_proba(X)
        except Exception:
            pass
        return compute_all_metrics(y, y_pred, average=average, labels=labels, output_dict=output_dict)

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
