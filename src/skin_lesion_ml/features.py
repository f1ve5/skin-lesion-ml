"""Fold-local transforms adapted from the course project's screening ideas."""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def channel_group(name):
    """Group color channels and texture/shape families using original naming."""
    match = re.search(r"(?:^|_)(RGB_[RGB]|HSV_[HSV]|Lab_[Lab]|YCrCb_(?:Y|Cr|Cb))(?:_|$)", name)
    if match:
        return match[1]
    match = re.match(r"glcm_([A-Za-z]+)_", name)
    if match:
        return "GLCM_" + match[1]
    for prefix in ("lbp", "gabor", "wave", "spec", "bnd", "asym", "hu", "fd",
                   "sig", "topo", "fractal", "rel", "ccr", "specific", "bord"):
        if name.startswith(prefix + "_"):
            return prefix
    if name.startswith(("reg_convex", "reg_solidity")):
        return "reg_shape"
    return "reg_general" if name.startswith("reg_") else "other"


class FeatureTransform(TransformerMixin, BaseEstimator):
    """Impute, drop constants, optionally apply within-channel PCA, then rank.

    mRMR uses MI relevance with an absolute Pearson correlation penalty,
    matching the original project's approximation rather than pairwise MI.
    """
    def __init__(self, k=30, selection="mrmr", pca=False, variance=0.99, random_state=42):
        self.k = k
        self.selection = selection
        self.pca = pca
        self.variance = variance
        self.random_state = random_state

    def _project(self, values):
        return np.column_stack([
            transform.transform(values[:, ids]) if transform is not None else values[:, ids]
            for ids, transform in self.blocks_
        ])

    def fit(self, X, y):
        if self.k < 1 or self.selection not in {"f", "mrmr"}:
            raise ValueError("k must be positive; selection must be f or mrmr.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        values = self.imputer_.fit_transform(X)
        self.nonconstant_ = np.var(values, axis=0) > 1e-12
        if not self.nonconstant_.any():
            raise ValueError("All training features are constant or missing.")
        names = self.feature_names_in_[self.nonconstant_]
        values = values[:, self.nonconstant_]
        self.blocks_, projected_names = [], []
        groups = {}
        for idx, name in enumerate(names):
            groups.setdefault(channel_group(name), []).append(idx)
        for group, ids in sorted(groups.items()):
            transform = None
            if self.pca and len(ids) >= 3:
                from sklearn.pipeline import make_pipeline
                transform = make_pipeline(StandardScaler(), PCA(n_components=self.variance, svd_solver="full"))
                transform.fit(values[:, ids])
                width = transform[-1].n_components_
                projected_names.extend(f"PCA_{group}_{i}" for i in range(width))
            else:
                projected_names.extend(names[ids])
            self.blocks_.append((ids, transform))
        values = self._project(values)
        k = min(self.k, values.shape[1])
        if self.selection == "f":
            scores = np.nan_to_num(f_classif(values, y)[0], nan=0, posinf=1e30)
            chosen = np.argsort(-scores, kind="stable")[:k]
        else:
            scores = mutual_info_classif(values, y, random_state=self.random_state)
            corr = np.atleast_2d(np.nan_to_num(np.abs(np.corrcoef(values.T))))
            chosen = []
            for _ in range(k):
                utility = scores.copy()
                if chosen:
                    utility *= 1 - corr[:, chosen].mean(axis=1)
                    utility[chosen] = -np.inf
                chosen.append(int(np.argmax(utility)))
        self.selected_ = np.asarray(chosen)
        self.selected_names_ = np.asarray(projected_names)[self.selected_].tolist()
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_")
        values = self.imputer_.transform(X.loc[:, self.feature_names_in_])[:, self.nonconstant_]
        return self._project(values)[:, self.selected_]


class InteractionFeatures(TransformerMixin, BaseEstimator):
    """Learn complementary feature pairs on training rows and freeze them.

    Family names in the original code are F/RF rank heuristics; they are not
    anatomical annotations. Denominator floors are learned once at fit time.
    """
    def __init__(self, max_keep=10, candidates=5, random_state=42):
        self.max_keep = max_keep
        self.candidates = candidates
        self.random_state = random_state

    def _values(self, X):
        values = self.imputer_.transform(X.loc[:, self.feature_names_in_])
        cols = []
        for left, right, op in self.pairs_:
            a, b = values[:, left], values[:, right]
            if op == "product":
                result = a * b
            elif op == "difference":
                result = a - b
            else:
                denom = np.where(b >= 0, 1, -1) * np.maximum(np.abs(b), self.floors_[right])
                result = a / denom
            cols.append(np.clip(result, -1e12, 1e12))
        return np.column_stack(cols)

    def fit(self, X, y):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        values = self.imputer_.fit_transform(X)
        self.floors_ = np.maximum(1e-8, 1e-4 * np.mean(np.abs(values), axis=0))
        f = np.nan_to_num(f_classif(values, y)[0], nan=0, posinf=1e30)
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    random_state=self.random_state, n_jobs=1).fit(values, y)
        def normalize(a):
            return (a - a.min()) / max(float(np.ptp(a)), 1e-12)
        delta = normalize(rf.feature_importances_) - normalize(f)
        order = np.argsort(delta, kind="stable")
        m = min(self.candidates, max(1, X.shape[1] // 2))
        self.pairs_ = [(int(a), int(b), op) for a in order[:m] for b in order[-m:]
                       if a != b for op in ("product", "ratio", "difference")]
        if not self.pairs_:
            self.selected_ = np.array([], dtype=int)
            return self
        interactions = self._values(X)
        valid = np.flatnonzero(np.var(interactions, axis=0) > 1e-12)
        if len(valid):
            mi = mutual_info_classif(interactions[:, valid], y, random_state=self.random_state)
            self.selected_ = valid[np.argsort(-mi, kind="stable")[:self.max_keep]]
        else:
            self.selected_ = np.array([], dtype=int)
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_")
        if not len(self.selected_):
            return np.empty((len(X), 0))
        return self._values(X)[:, self.selected_]
