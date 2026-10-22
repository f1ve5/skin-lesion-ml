"""Five flat baselines and a feature-space-asymmetric hierarchical model."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from .features import FeatureTransform, InteractionFeatures


class HierarchicalClassifier(ClassifierMixin, BaseEstimator):
    """L1: vasc/rest with LR; L2: mel/nv with SVM and optional interactions.

    All learned transformations live inside fit, including when the complete
    estimator is cloned by inner GridSearchCV. predict never accepts labels.
    """
    def __init__(self, k=30, selection="mrmr", l1_pca=True, interactions=True,
                 l1_C=1.0, l2_C=2.0, random_state=42):
        self.k = k
        self.selection = selection
        self.l1_pca = l1_pca
        self.interactions = interactions
        self.l1_C = l1_C
        self.l2_C = l2_C
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        if set(y) != {"mel", "nv", "vasc"}:
            raise ValueError("Hierarchical training requires mel, nv and vasc in every training fold.")
        self.classes_ = np.array(["mel", "nv", "vasc"])
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        def transform(pca=False):
            return FeatureTransform(self.k, self.selection, pca, random_state=self.random_state)
        self.layer1_ = Pipeline([
            ("features", transform(self.l1_pca)), ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=self.l1_C, class_weight="balanced", max_iter=3000,
                                       random_state=self.random_state))])
        layer2_features = transform()
        if self.interactions:
            layer2_features = FeatureUnion([
                ("base", layer2_features),
                ("interactions", InteractionFeatures(random_state=self.random_state))])
        self.layer2_ = Pipeline([
            ("features", layer2_features), ("scale", StandardScaler()),
            ("clf", SVC(C=self.l2_C, kernel="rbf", class_weight="balanced",
                        random_state=self.random_state))])
        self.layer1_.fit(X, (y == "vasc").astype(int))
        mask = y != "vasc"
        self.layer2_.fit(X.iloc[np.flatnonzero(mask)], y[mask])
        return self

    def predict(self, X):
        check_is_fitted(self, "layer2_")
        prediction = np.full(len(X), "vasc", dtype=object)
        routed = np.flatnonzero(self.layer1_.predict(X) == 0)
        if len(routed):
            prediction[routed] = self.layer2_.predict(X.iloc[routed])
        return prediction


def model_spec(name, k=30, selection="mrmr", seed=42):
    if name.startswith("hierarchical"):
        model = HierarchicalClassifier(k=k, selection=selection, random_state=seed,
                                       l1_pca=name != "hierarchical-no-pca",
                                       interactions=name != "hierarchical-no-interactions")
        return model, {"l1_C": [0.1, 1.0], "l2_C": [1.0, 2.0]}
    classifiers = {
        "logreg": (LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed),
                   {"clf__C": [0.1, 1.0]}),
        "svm": (SVC(kernel="rbf", class_weight="balanced", random_state=seed),
                {"clf__C": [1.0, 2.0]}),
        "random-forest": (RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                  n_jobs=1, random_state=seed),
                          {"clf__max_depth": [None, 10]}),
        "gradient-boosting": (GradientBoostingClassifier(n_estimators=150, random_state=seed),
                              {"clf__max_depth": [2, 3]}),
        "knn": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 7]}),
    }
    if name not in classifiers:
        raise ValueError(f"Unknown model: {name}")
    classifier, grid = classifiers[name]
    return Pipeline([("features", FeatureTransform(k=k, selection=selection, random_state=seed)),
                     ("scale", StandardScaler()), ("clf", classifier)]), grid


MODEL_NAMES = ["logreg", "svm", "random-forest", "gradient-boosting", "knn",
               "hierarchical", "hierarchical-no-pca", "hierarchical-no-interactions"]
