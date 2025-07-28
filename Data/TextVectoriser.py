import pandas as pd
import numpy as np

from scipy.sparse import hstack, csr_matrix, issparse

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.model_selection import train_test_split

from pathlib import Path

BASE_DIR = Path(__file__).parent

class TextVectorizer:
    def __init__(self,
                 dataset_csv="dataset.csv",
                 text_cols=("speakers", "title", "subtitle", "contents"),
                 target_cols=("target_1", "target_2"),
                 id_col="date",
                 concat_text=True):

        self.dataset_csv = BASE_DIR / dataset_csv
        self.text_cols = list(text_cols)
        self.target_cols = list(target_cols)
        self.id_col = id_col
        self.concat_text = concat_text

        self.df = pd.DataFrame()
        self.loaded = False

    def load(self):
        self.df = pd.read_csv(self.dataset_csv)
        if self.id_col in self.df.columns:
            try:
                self.df[self.id_col] = pd.to_datetime(self.df[self.id_col])
            except Exception:
                pass
        else:
            self.df.index = pd.to_datetime(self.df.index)
        self.loaded = True

    def _check_loaded(self):
        if not self.loaded:
            raise ValueError("Call load() first.")

    def _prepare_text_series(self):
        self._check_loaded()

        self.text_cols = [c for c in self.text_cols if c in self.df.columns]

        if self.concat_text:
            ser = self.df[self.text_cols].fillna("").astype(str).agg(" ".join, axis=1)
            return ser
        else:
            out = {}
            for c in self.text_cols:
                out[c] = self.df[c].fillna("").astype(str)
            return out

    def _get_y(self, target):
        if target not in self.df.columns:
            raise ValueError(f"Target '{target}' is not in the dataset.")
        return self.df[target].values


    def tfidf(self, target="target_1", max_features=20000, ngram_range=(1,2),
              min_df=2, sublinear_tf=True, lowercase=True, analyzer="word"):
        y = self._get_y(target)

        if self.concat_text:
            corpus = self._prepare_text_series()
            vec = TfidfVectorizer(max_features=max_features,
                                  ngram_range=ngram_range,
                                  min_df=min_df,
                                  sublinear_tf=sublinear_tf,
                                  lowercase=lowercase,
                                  analyzer=analyzer)
            X = vec.fit_transform(corpus)
            return X, y, vec
        else:
            matrices = []
            vectorizers = {}
            texts = self._prepare_text_series()
            for col, ser in texts.items():
                vec = TfidfVectorizer(max_features=max_features,
                                      ngram_range=ngram_range,
                                      min_df=min_df,
                                      sublinear_tf=sublinear_tf,
                                      lowercase=lowercase,
                                      analyzer=analyzer)
                Xc = vec.fit_transform(ser)
                matrices.append(Xc)
                vectorizers[col] = vec
            X = hstack(matrices).tocsr()
            return X, y, vectorizers

    def count(self, target="target_1", max_features=20000, ngram_range=(1,1),
              min_df=2, lowercase=True, analyzer="word"):
        y = self._get_y(target)
        if self.concat_text:
            corpus = self._prepare_text_series()
            vec = CountVectorizer(max_features=max_features,
                                  ngram_range=ngram_range,
                                  min_df=min_df,
                                  lowercase=lowercase,
                                  analyzer=analyzer)
            X = vec.fit_transform(corpus)
            return X, y, vec
        else:
            matrices = []
            vectorizers = {}
            texts = self._prepare_text_series()
            for col, ser in texts.items():
                vec = CountVectorizer(max_features=max_features,
                                      ngram_range=ngram_range,
                                      min_df=min_df,
                                      lowercase=lowercase,
                                      analyzer=analyzer)
                Xc = vec.fit_transform(ser)
                matrices.append(Xc)
                vectorizers[col] = vec
            X = hstack(matrices).tocsr()
            return X, y, vectorizers

    def hashing(self, target="target_1", n_features=2**18, ngram_range=(1,2),
                alternate_sign=True, lowercase=True, analyzer="word"):
        y = self._get_y(target)
        if self.concat_text:
            corpus = self._prepare_text_series()
            vec = HashingVectorizer(n_features=n_features,
                                    ngram_range=ngram_range,
                                    alternate_sign=alternate_sign,
                                    lowercase=lowercase,
                                    analyzer=analyzer)
            X = vec.transform(corpus)
            return X, y, vec
        else:
            matrices = []
            vectorizers = {}
            texts = self._prepare_text_series()
            for col, ser in texts.items():
                vec = HashingVectorizer(n_features=n_features,
                                        ngram_range=ngram_range,
                                        alternate_sign=alternate_sign,
                                        lowercase=lowercase,
                                        analyzer=analyzer)
                Xc = vec.transform(ser)
                matrices.append(Xc)
                vectorizers[col] = vec
            X = hstack(matrices).tocsr()
            return X, y, vectorizers


    def standardize(self, X, with_mean=False):
        if issparse(X) and with_mean:
            with_mean = False

        scaler = StandardScaler(with_mean=with_mean)
        if issparse(X):
            X_std = scaler.fit_transform(X)
        else:
            X_std = scaler.fit_transform(X)
        return X_std, scaler

    def reduce(self, X, method="pca", n_components=50, random_state=42, **kwargs):
        method = method.lower()

        if method == "pca":
            if issparse(X):
                reducer = TruncatedSVD(n_components=n_components, random_state=random_state, **kwargs)
                X_red = reducer.fit_transform(X)
            else:
                reducer = PCA(n_components=n_components, random_state=random_state, **kwargs)
                X_red = reducer.fit_transform(X)
            return X_red, reducer

        elif method == "tsne":
            X_dense = X.toarray() if issparse(X) else X
            reducer = TSNE(n_components=n_components, random_state=random_state, **kwargs)
            X_red = reducer.fit_transform(X_dense)
            return X_red, reducer

        elif method == "mds":
            X_dense = X.toarray() if issparse(X) else X
            reducer = MDS(n_components=n_components, random_state=random_state, **kwargs)
            X_red = reducer.fit_transform(X_dense)
            return X_red, reducer

        else:
            raise ValueError("Unknown method. Choose from: 'pca', 'tsne', 'mds'.")

    def temporal_split(self, X, y, split_date, stratify=False):
        if self.id_col in self.df.columns:
            dates = pd.to_datetime(self.df[self.id_col], errors="coerce")
        else:
            dates = pd.to_datetime(self.df.index, errors="coerce")

        mask_train = dates < pd.Timestamp(split_date)
        mask_test = ~mask_train

        if issparse(X):
            X_train = X[mask_train]
            X_test = X[mask_test]
        else:
            X_train = X[mask_train]
            X_test = X[mask_test]

        y_train = y[mask_train]
        y_test = y[mask_test]

        return X_train, X_test, y_train, y_test

    def random_split(self, X, y, test_size=0.2, random_state=42, stratify=True):
        if stratify:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        else:
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
