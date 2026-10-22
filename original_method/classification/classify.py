"""
分类与评估
  - 多分类器对比：LR / SVM / RandomForest / GradientBoost / KNN
  - 严格按image_id分组防泄漏（同id的原图与增强不能跨训练-测试）
  - 5-fold CV
  - 混淆矩阵、各类PR
  - 特征重要性（RF）
  - 【v3 新增】HierarchicalClassifier: 两层层次分类 + 嵌套 GroupKFold
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, GridSearchCV, cross_val_predict, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from analysis.viz_features import setup_style


def get_classifiers(random_state=42):
    return {
        "LogReg": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=random_state)),
        ]),
        "SVM-RBF": Pipeline([
            ("sc", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale",
                        probability=True, class_weight="balanced",
                        random_state=random_state)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=random_state),
        "GradientBoost": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=random_state),
        "KNN": Pipeline([
            ("sc", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
    }


def evaluate_classifiers(X: pd.DataFrame, y: pd.Series,
                          groups: pd.Series,
                          top_k_features: int = 30,
                          n_splits: int = 5,
                          random_state: int = 42,
                          logger=None) -> dict:
    """
    返回 dict of {classifier_name: {metrics, predictions, y_true}}
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y.values)
    classes = le.classes_

    # 选取K个最优特征（在整个数据集上做F-test - 对比模型间公平）
    if top_k_features and top_k_features < X.shape[1]:
        selector = SelectKBest(f_classif, k=min(top_k_features, X.shape[1]))
        selector.fit(X.values, y_enc)
        keep_mask = selector.get_support()
        X_sel = X.loc[:, keep_mask].copy()
        if logger:
            logger.info(f"特征选择: 从 {X.shape[1]} -> {X_sel.shape[1]} 维")
    else:
        X_sel = X.copy()

    # GroupKFold保证image_id不跨折
    n_groups = groups.nunique()
    n_splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    results = {}
    classifiers = get_classifiers(random_state)
    for name, model in classifiers.items():
        if logger: logger.info(f"训练 {name} ...")
        try:
            y_pred = cross_val_predict(
                model, X_sel.values, y_enc,
                groups=groups.values, cv=gkf, n_jobs=1,
            )
        except Exception as e:
            if logger: logger.warning(f"{name} 失败: {e}")
            continue
        acc  = accuracy_score(y_enc, y_pred)
        bacc = balanced_accuracy_score(y_enc, y_pred)
        f1m  = f1_score(y_enc, y_pred, average="macro", zero_division=0)
        cm   = confusion_matrix(y_enc, y_pred)
        results[name] = {
            "accuracy": acc, "balanced_acc": bacc, "f1_macro": f1m,
            "confusion_matrix": cm, "y_true": y_enc, "y_pred": y_pred,
            "classes": classes,
            "feature_names": X_sel.columns.tolist(),
        }
        if logger:
            logger.info(f"  {name}: acc={acc:.3f}, b_acc={bacc:.3f}, "
                         f"f1_macro={f1m:.3f}")
    return results, X_sel


def viz_classifier_comparison(results: dict, save_path):
    setup_style()
    rows = []
    for name, r in results.items():
        rows.append({"clf": name,
                     "Accuracy": r["accuracy"],
                     "Balanced Acc": r["balanced_acc"],
                     "F1 (macro)": r["f1_macro"]})
    df = pd.DataFrame(rows).set_index("clf")
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A467"])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title("Classifier Comparison (5-fold GroupKFold CV)")
    ax.set_xticklabels(df.index, rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return df


def viz_confusion_matrices(results: dict, save_path):
    setup_style()
    n = len(results)
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 4.0*n_rows))
    axes = np.array(axes).reshape(-1)
    for ax, (name, r) in zip(axes, results.items()):
        cm = r["confusion_matrix"]
        # 归一化
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=r["classes"], yticklabels=r["classes"],
                    ax=ax, cbar=False)
        ax.set_title(f"{name}\n(acc={r['accuracy']:.3f})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices (normalized by row, annotated counts)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def viz_feature_importance(X: pd.DataFrame, y: pd.Series, save_path,
                             top_k=25, random_state=42):
    setup_style()
    le = LabelEncoder(); y_enc = le.fit_transform(y.values)
    rf = RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                  class_weight="balanced",
                                  random_state=random_state)
    rf.fit(X.values, y_enc)
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False).head(top_k)
    # 为颜色按特征分类着色
    def cat(n):
        if n.startswith(("bnd_", "fd_", "sig_", "reg_", "hu_", "topo_", "asym_")):
            return "Shape"
        if n.startswith("int_"): return "Intensity"
        if n.startswith("glcm_"): return "GLCM"
        if n.startswith("lbp_"): return "LBP"
        if n.startswith("gabor_"): return "Gabor"
        if n.startswith("wave_"): return "Wavelet"
        if n.startswith("spec_"): return "Spectral"
        return "Other"
    cats = [cat(n) for n in imp.index]
    palette = {"Shape": "#4C72B0", "Intensity": "#DD8452", "GLCM": "#55A467",
               "LBP": "#C44E52", "Gabor": "#8172B2", "Wavelet": "#937860",
               "Spectral": "#DA8BC3", "Other": "#999999"}
    colors = [palette[c] for c in cats]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.32*top_k)))
    ax.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
    ax.set_xlabel("RF Feature Importance")
    ax.set_title(f"Top-{top_k} Most Important Features (Random Forest)")
    # 自定义图例
    handles = [plt.Rectangle((0,0),1,1, color=palette[k]) for k in palette]
    ax.legend(handles, palette.keys(), loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return imp


# ============================================================
# 层次分类器 (v3 新增)
# ============================================================
class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """
    两层层次分类器，严格嵌套 GroupKFold 防泄漏。

    结构：
      Layer 1: vasc vs non-vasc (nv + mel)
        - 模型: LogisticRegression(L2, class_weight='balanced')
        - 超参搜索: C ∈ {0.01, 0.1, 1, 10}
      Layer 2: mel vs nv
        - 模型: 比较 XGBClassifier 与 SVC(rbf)，选更优者
        - 超参搜索: XGBoost(max_depth, n_estimators) / SVM(C, gamma)

    嵌套交叉验证：
      外循环: GroupKFold(n_splits=5) 按 image_id 分组
        内循环: GroupKFold(n_splits=4) GridSearchCV，显式传入 groups=groups_train
    """

    def __init__(self, random_state=42, logger=None,
                 force_include_features=None,
                 layer1_feature_names=None,
                 layer2_feature_names=None):
        self.random_state = random_state
        self.logger = logger
        self.force_include_features = force_include_features or []
        self.layer1_model_ = None
        self.layer2_model_ = None
        self.layer2_type_ = None  # 'xgboost' or 'svm'
        self.classes_ = np.array(["mel", "nv", "vasc"])
        self.layer1_selector_ = None
        self.layer2_selector_ = None
        self.layer2_interact_cols_ = []  # 训练时使用的交互特征列名
        # v4: 预选特征名列表 (外部做特征筛选+交互工程时传入)
        self.layer1_feature_names_ = layer1_feature_names
        self.layer2_feature_names_ = layer2_feature_names

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def _prepare_layer_data(self, X, y, layer):
        """为各层准备二分类标签和数据"""
        if layer == 1:
            # vasc vs non-vasc
            y_bin = np.where(np.array(y) == "vasc", 1, 0)
        else:
            # mel vs nv (排除 vasc)
            mask = np.array(y) != "vasc"
            if isinstance(X, pd.DataFrame):
                X = X.loc[mask].reset_index(drop=True)
            else:
                X = X[mask]
            y = np.array(y)[mask]
            y_bin = np.where(y == "mel", 1, 0)
        return X, y_bin

    def fit(self, X, y, groups=None, joint_selector_cls=None,
            layer1_selector=None, layer2_selector=None,
            X2_interact=None):
        """
        使用嵌套 GroupKFold 训练两层分类器。

        Parameters
        ----------
        X : pd.DataFrame
        y : array-like
        groups : array-like, image_id
        joint_selector_cls : JointFeatureSelector class (未 fit 的实例)
            — v3 兼容参数, 与 layer1/2_selector 二选一
        layer1_selector : 已 fit 的 Layer 1 特征选择器 (v4 新增)
        layer2_selector : 已 fit 的 Layer 2 特征选择器 (v4 新增)
        X2_interact : pd.DataFrame — Layer 2 交互特征矩阵 (v4 新增),
            与 X2 样本对应, 会被拼接到 X2_sel 后
        """
        y_arr = np.array(y)
        groups_arr = np.array(groups) if groups is not None else np.arange(len(y))

        # ====== Layer 1: vasc vs non-vasc ======
        self._log("Training Layer 1: vasc vs non-vasc (LogisticRegression)")
        X1, y1 = self._prepare_layer_data(X, y_arr, layer=1)

        # 特征选择: 优先使用 v4 预选特征名/选择器
        if layer1_selector is not None:
            self.layer1_selector_ = layer1_selector
            X1_sel = X1[self.layer1_selector_.selected_features_]
        elif self.layer1_feature_names_ is not None:
            X1_sel = X1[[f for f in self.layer1_feature_names_ if f in X1.columns]]
        elif joint_selector_cls is not None:
            sel1 = clone(joint_selector_cls)
            sel1.fit(X1, y1)
            X1_sel = sel1.transform(X1)
            self.layer1_selector_ = sel1
        else:
            X1_sel = X1

        X1_arr = X1_sel.values if isinstance(X1_sel, pd.DataFrame) else X1_sel

        # 内循环 GridSearchCV
        inner_cv = GroupKFold(n_splits=min(4, len(np.unique(groups_arr))))
        pipe1 = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", class_weight="balanced",
                                       max_iter=2000,
                                       random_state=self.random_state)),
        ])
        grid1 = GridSearchCV(
            pipe1, param_grid={"clf__C": [0.01, 0.1, 1, 10]},
            cv=inner_cv, scoring="balanced_accuracy", n_jobs=1,
        )
        grid1.fit(X1_arr, y1, groups=groups_arr)
        self.layer1_model_ = grid1.best_estimator_
        self._log(f"  Layer 1 best C={grid1.best_params_['clf__C']}, "
                  f"score={grid1.best_score_:.3f}")

        # Layer 1 阈值调优 (Phase 7): 内层 CV 扫描最优阈值
        self.layer1_threshold_ = 0.5
        try:
            y1_proba_cv = cross_val_predict(
                self.layer1_model_, X1_arr, y1, groups=groups_arr,
                cv=inner_cv, method="predict_proba",
            )[:, 1]
            best_thresh = 0.5
            best_bacc = -1
            for t in np.arange(0.30, 0.71, 0.05):
                y1_pred_t = (y1_proba_cv >= t).astype(int)
                bacc = balanced_accuracy_score(y1, y1_pred_t)
                if bacc > best_bacc:
                    best_bacc = bacc
                    best_thresh = t
            self.layer1_threshold_ = best_thresh
            self._log(f"  Layer 1 threshold tuned: {best_thresh:.2f} "
                      f"(bacc={best_bacc:.3f})")
        except Exception:
            self.layer1_threshold_ = 0.5

        # ====== Layer 2: mel vs nv ======
        self._log("Training Layer 2: mel vs nv (XGBoost vs SVM)")
        X2, y2 = self._prepare_layer_data(X, y_arr, layer=2)
        groups2 = groups_arr[np.array(y_arr) != "vasc"]

        # 特征选择: 优先使用 v4 预选特征/选择器
        if layer2_selector is not None:
            self.layer2_selector_ = layer2_selector
            X2_sel = X2[[f for f in self.layer2_selector_.selected_features_
                        if f in X2.columns]]
        elif self.layer2_feature_names_ is not None:
            selected = [f for f in self.layer2_feature_names_ if f in X2.columns]
            X2_sel = X2[selected]
        elif joint_selector_cls is not None:
            sel2 = clone(joint_selector_cls)
            sel2.fit(X2, y2)
            selected = sel2.selected_features_
            for f in self.force_include_features:
                if f in X2.columns and f not in selected:
                    selected.append(f)
            sel2.selected_features_ = selected
            X2_sel = X2[selected]
            self.layer2_selector_ = sel2
        else:
            X2_sel = X2

        # v4: 拼接交互特征
        if X2_interact is not None and X2_interact.shape[0] == X2_sel.shape[0]:
            X2_concat = [X2_sel.reset_index(drop=True),
                         X2_interact.reset_index(drop=True)]
            X2_sel = pd.concat(X2_concat, axis=1)
            self.layer2_interact_cols_ = list(X2_interact.columns)
            self._log(f"  Layer 2: appended {X2_interact.shape[1]} interaction "
                      f"features, total={X2_sel.shape[1]}")

        X2_arr = X2_sel.values if isinstance(X2_sel, pd.DataFrame) else X2_sel

        # 尝试 XGBoost (保守固定参数，避免小样本 grid search 过拟合)
        xgb_score = -1
        svm_score = -1
        self.layer2_type_ = "svm"  # fallback

        try:
            import xgboost as xgb
            from sklearn.model_selection import cross_val_score
            scale_pos_weight = (y2 == 0).sum() / max((y2 == 1).sum(), 1)
            xgb_fixed = Pipeline([
                ("sc", StandardScaler()),
                ("clf", xgb.XGBClassifier(
                    objective="binary:logistic",
                    max_depth=3, n_estimators=200,
                    learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.random_state,
                    use_label_encoder=False, eval_metric="logloss", verbosity=0)),
            ])
            inner_gkf = GroupKFold(n_splits=min(4, len(np.unique(groups2))))
            xgb_scores = cross_val_score(
                xgb_fixed, X2_arr, y2, groups=groups2,
                cv=inner_gkf, scoring="balanced_accuracy",
            )
            xgb_score = xgb_scores.mean()
            xgb_fixed.fit(X2_arr, y2)  # 固定参数，直接全量训练
            self._log(f"  XGBoost fixed: score={xgb_score:.3f} "
                      f"(±{xgb_scores.std():.3f})")
        except ImportError:
            self._log("  XGBoost not available, using SVM only")

        # SVM 搜索 (对过拟合不敏感，保留 grid search)
        svm_pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True,
                        class_weight="balanced",
                        random_state=self.random_state)),
        ])
        svm_grid = GridSearchCV(
            svm_pipe,
            param_grid={
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.01, 0.1],
            },
            cv=GroupKFold(n_splits=min(4, len(np.unique(groups2)))),
            scoring="balanced_accuracy", n_jobs=1,
        )
        svm_grid.fit(X2_arr, y2, groups=groups2)
        svm_score = svm_grid.best_score_
        self._log(f"  SVM best: {svm_grid.best_params_}, score={svm_score:.3f}")

        # 选更优者
        if xgb_score > svm_score:
            self.layer2_model_ = xgb_fixed
            self.layer2_type_ = "xgboost"
            self._log(f"  Layer 2 selected: XGBoost (score={xgb_score:.3f})")
        else:
            self.layer2_model_ = svm_grid.best_estimator_
            self.layer2_type_ = "svm"
            self._log(f"  Layer 2 selected: SVM (score={svm_score:.3f})")

        # 阈值调优: 内层 CV 扫描最优阈值
        self.layer2_threshold_ = 0.5
        try:
            from sklearn.model_selection import cross_val_predict
            inner_gkf = GroupKFold(n_splits=min(4, len(np.unique(groups2))))
            y2_proba_cv = cross_val_predict(
                self.layer2_model_, X2_arr, y2, groups=groups2,
                cv=inner_gkf, method="predict_proba",
            )[:, 1]
            best_thresh = 0.5
            best_bacc = -1
            for t in np.arange(0.30, 0.71, 0.05):
                y2_pred_t = (y2_proba_cv >= t).astype(int)
                bacc = balanced_accuracy_score(y2, y2_pred_t)
                if bacc > best_bacc:
                    best_bacc = bacc
                    best_thresh = t
            self.layer2_threshold_ = best_thresh
            self._log(f"  Layer 2 threshold tuned: {best_thresh:.2f} "
                      f"(bacc={best_bacc:.3f})")
        except Exception:
            self.layer2_threshold_ = 0.5

        return self

    def predict(self, X, groups=None):
        """预测类别标签"""
        if self.layer1_model_ is None or self.layer2_model_ is None:
            raise RuntimeError("Must call fit() before predict()")

        # Layer 1: 特征选择 + 预测
        if self.layer1_selector_ is not None:
            sel_feats = [f for f in self.layer1_selector_.selected_features_
                        if f in X.columns]
            X1 = X[sel_feats]
        elif self.layer1_feature_names_ is not None:
            sel_feats = [f for f in self.layer1_feature_names_ if f in X.columns]
            X1 = X[sel_feats]
        else:
            X1 = X
        X1_arr = X1.values if isinstance(X1, pd.DataFrame) else X1
        l1_thresh = getattr(self, "layer1_threshold_", 0.5)
        p1 = self.layer1_model_.predict_proba(X1_arr)[:, 1]
        y1_pred = (p1 >= l1_thresh).astype(int)

        results = np.full(len(X), "vasc", dtype=object)

        # 非 vasc 的样本进入 Layer 2
        non_vasc_idx = np.where(y1_pred == 0)[0]
        if len(non_vasc_idx) > 0:
            X_non_vasc = X.iloc[non_vasc_idx] if isinstance(X, pd.DataFrame) else X[non_vasc_idx]

            # Layer 2 特征选择 + 交互特征 (使用训练时保存的列名)
            if self.layer2_selector_ is not None:
                sel_feats = [f for f in self.layer2_selector_.selected_features_
                            if f in X_non_vasc.columns]
                inter_feats = [c for c in self.layer2_interact_cols_
                              if c in X_non_vasc.columns]
                X2 = X_non_vasc[sel_feats + inter_feats]
            elif self.layer2_feature_names_ is not None:
                sel_feats = [f for f in self.layer2_feature_names_ if f in X_non_vasc.columns]
                X2 = X_non_vasc[sel_feats]
            else:
                X2 = X_non_vasc

            X2_arr = X2.values if isinstance(X2, pd.DataFrame) else X2
            p2 = self.layer2_model_.predict_proba(X2_arr)[:, 1]
            y2_pred = (p2 >= self.layer2_threshold_).astype(int)
            results[non_vasc_idx] = np.where(y2_pred == 1, "mel", "nv")

        return results

    def predict_proba(self, X):
        """预测概率 (3 类)"""
        if self.layer1_model_ is None or self.layer2_model_ is None:
            raise RuntimeError("Must call fit() before predict_proba()")

        n = len(X)
        proba = np.zeros((n, 3), dtype=np.float64)  # mel, nv, vasc

        # Layer 1
        if self.layer1_selector_ is not None:
            sel_feats = [f for f in self.layer1_selector_.selected_features_
                        if f in X.columns]
            X1 = X[sel_feats]
        elif self.layer1_feature_names_ is not None:
            sel_feats = [f for f in self.layer1_feature_names_ if f in X.columns]
            X1 = X[sel_feats]
        else:
            X1 = X
        X1_arr = X1.values if isinstance(X1, pd.DataFrame) else X1
        p1 = self.layer1_model_.predict_proba(X1_arr)  # [P(non-vasc), P(vasc)]
        proba[:, 2] = p1[:, 1]  # vasc prob

        # Layer 2 for non-vasc
        non_vasc_weight = p1[:, 0]  # P(non-vasc)
        if self.layer2_selector_ is not None:
            sel_feats = [f for f in self.layer2_selector_.selected_features_
                        if f in X.columns]
            inter_feats = [c for c in self.layer2_interact_cols_
                          if c in X.columns]
            X2 = X[sel_feats + inter_feats]
        elif self.layer2_feature_names_ is not None:
            sel_feats = [f for f in self.layer2_feature_names_ if f in X.columns]
            X2 = X[sel_feats]
        else:
            X2 = X

        X2_arr = X2.values if isinstance(X2, pd.DataFrame) else X2
        p2 = self.layer2_model_.predict_proba(X2_arr)  # [P(nv), P(mel)]

        proba[:, 1] = p2[:, 0] * non_vasc_weight  # nv
        proba[:, 0] = p2[:, 1] * non_vasc_weight  # mel

        # 归一化
        row_sums = proba.sum(axis=1, keepdims=True)
        proba = proba / (row_sums + 1e-12)
        return proba


def evaluate_hierarchical(X: pd.DataFrame, y: pd.Series,
                          groups: pd.Series,
                          rob_df=None,
                          clinical_priority_map=None,
                          force_include_features=None,
                          n_splits=5,
                          random_state=42,
                          logger=None,
                          # v4: 每折独立做特征筛选 + 交互生成 (CV-leak-free)
                          screen_strategy="borda",
                          l1_k=30, l2_k=30, l1_icc=0.50, l2_icc=0.75,
                          interact_m=5, interact_n=5,
                          interact_types=("product", "ratio", "difference"),
                          interact_max_keep=10,
                          max_per_channel=None,
                          # v5.0: 非对称 PCA — L1/L2 独立控制
                          l1_pca=True,
                          l1_pca_var=0.99,
                          l2_pca=False,
                          l2_pca_var=0.99) -> dict:
    """
    用嵌套 GroupKFold 评估 HierarchicalClassifier。

    v4 CV-leak-free 设计：
      每折内部独立完成 F-test + RF importance + 特征筛选 + 交互生成，
      确保测试数据在任何阶段都不参与特征选择或交互排序。

    v5.0 非对称 PCA:
      Layer 1 (vasc vs non-vasc): PCA 去噪 → vasc_rec↑
      Layer 2 (mel vs nv): 保留原始特征 → mel_rec↑
      两层独立控制, 各取所需。
    """
    from analysis.feature_screening import FeatureScreener, \
        compute_layer_specific_rf_importance, apply_within_group_pca
    from analysis.feature_interaction import (
        generate_layer2_interactions, apply_interaction_pairs,
    )

    le = LabelEncoder()
    y_enc = le.fit_transform(y.values)
    classes = le.classes_

    force_feats = force_include_features or []

    n_groups = groups.nunique()
    n_splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    all_y_true = []
    all_y_pred = []
    all_test_indices = []  # 记录每折 test 样本在原始 DataFrame 中的位置
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y_enc, groups)):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        groups_train = groups.iloc[train_idx].values

        # 每折 train 内构建二分类标签
        y1_train = np.where(y_train.values == "vasc", 1, 0)
        mel_nv_train = y_train.isin(["mel", "nv"])
        mel_nv_train_idx = np.where(mel_nv_train)[0]

        if logger:
            logger.info(f"  Fold {fold_i+1}/{n_splits}: "
                        f"train={len(X_train)}, test={len(X_test)}")

        # ---- v5.0: 非对称 PCA (L1/L2 独立控制) ----
        # 保存原始特征 (L2 始终使用原始特征, 避免 PCA 丢弃 mel/nv 判别信号)
        X_train_orig = X_train.copy()
        X_test_orig = X_test.copy()

        pca_info = None
        X_train_l1 = X_train_orig.copy()  # L1 默认用原始特征 (copy 防别名)
        X_test_l1 = X_test_orig.copy()

        if l1_pca:
            X_train_arr = X_train.values
            X_test_arr = X_test.values
            feat_names = X_train.columns.tolist()
            X_train_l1, X_test_l1, pca_info = apply_within_group_pca(
                X_train_arr, X_test_arr, feat_names,
                variance_threshold=l1_pca_var, min_group_size=3,
                logger=logger if fold_i == 0 else None,
            )

        # ---- Per-fold Layer 1 Feature Screener (用 L1 特征空间) ----
        l1_rf_imp_fold, _ = compute_layer_specific_rf_importance(
            X_train_l1, y1_train, X_train_l1.columns.tolist(),
            random_state=random_state,
        )
        l1_screener = FeatureScreener(
            rf_importance_dict=l1_rf_imp_fold, rob_df=rob_df,
            strategy=screen_strategy, top_k=l1_k, layer=1, icc_min=l1_icc,
            clinical_priority_map=clinical_priority_map or {},
            channel_diversity=False,  # Layer 1: vasc 颜色特征即可, 无需多样性约束
        )
        l1_screener.fit(X_train_l1, y1_train)

        # ---- Per-fold Layer 2 Feature Screener (仅 mel/nv, 始终用原始特征) ----
        X2_train = X_train_orig.iloc[mel_nv_train_idx].reset_index(drop=True)
        y2_train = np.where(y_train.iloc[mel_nv_train_idx].values == "mel", 1, 0)
        groups2_train = groups_train[mel_nv_train_idx]

        l2_rf_imp_fold, _ = compute_layer_specific_rf_importance(
            X2_train, y2_train, X2_train.columns.tolist(),
            random_state=random_state,
        )
        l2_screener = FeatureScreener(
            rf_importance_dict=l2_rf_imp_fold, rob_df=rob_df,
            strategy=screen_strategy, top_k=l2_k, layer=2, icc_min=l2_icc,
            clinical_priority_map=clinical_priority_map or {},
            channel_diversity=True,   # Layer 2: 防止 HSV-H 等单一通道垄断
            max_per_channel=max_per_channel if max_per_channel is not None else max(2, l2_k // 10),
        )
        l2_screener.fit(X2_train, y2_train)

        # ---- Per-fold Interaction Features (仅 mel/nv) ----
        interact_report = {}
        X2_interact_train = None
        X2_interact_test = None
        interact_cols = []
        if interact_m > 0 and interact_n > 0:
            X2_interact_train, interact_report = generate_layer2_interactions(
                X2_train, y2_train,
                n_structural=interact_m, n_color=interact_n,
                interaction_types=interact_types, max_keep=interact_max_keep,
                random_state=random_state, logger=logger if fold_i == 0 else None,
            )
            # 将相同交互对应用到 test, 仅 mel/nv 行有值, vasc 行填 0
            sel_pairs = interact_report.get("selected_pairs", [])
            if sel_pairs:
                # test 中 mel/nv 行的位置 (相对于 X_test)
                test_mn_positions = [j for j, idx in enumerate(test_idx)
                                    if y.iloc[idx] in ("mel", "nv")]
                X_test_mn = X_test.iloc[test_mn_positions].reset_index(drop=True)
                X_interact_mn = apply_interaction_pairs(X_test_mn, sel_pairs)
                interact_cols = X_interact_mn.columns.tolist()
                # 构建全尺寸 DataFrame: non-mel/nv 行填 0
                X2_interact_test = pd.DataFrame(
                    0.0, index=np.arange(len(X_test)), columns=interact_cols,
                )
                X2_interact_test.iloc[test_mn_positions] = X_interact_mn.values

        if logger and fold_i == 0:
            logger.info(f"  [Fold 1] L1 selected: {len(l1_screener.selected_features_)}")
            logger.info(f"  [Fold 1] L2 selected: {len(l2_screener.selected_features_)}")
            logger.info(f"  [Fold 1] Interactions: "
                        f"{X2_interact_train.shape[1] if X2_interact_train is not None else 0}")

        # ---- Build hybrid X: PCA features (L1) + raw features (L2) ----
        X_train_hybrid = pd.concat([
            X_train_l1.reset_index(drop=True),
            X_train_orig.reset_index(drop=True),
        ], axis=1)

        X_test_hybrid = pd.concat([
            X_test_l1.reset_index(drop=True),
            X_test_orig.reset_index(drop=True),
        ], axis=1)

        # ---- Fit HierarchicalClassifier on hybrid X ----
        clf = HierarchicalClassifier(
            random_state=random_state, logger=logger,
            force_include_features=force_feats,
        )
        clf.fit(X_train_hybrid, y_train.values, groups=groups_train,
                layer1_selector=l1_screener,
                layer2_selector=l2_screener,
                X2_interact=X2_interact_train)

        # ---- Predict on test (join interaction features to hybrid X) ----
        X_test_eval = X_test_hybrid.copy()
        if X2_interact_test is not None and X2_interact_test.shape[1] > 0:
            X_test_eval = pd.concat(
                [X_test_eval.reset_index(drop=True),
                 X2_interact_test.reset_index(drop=True)],
                axis=1,
            )

        y_pred = clf.predict(X_test_eval)
        all_y_true.extend(y_test.values)
        all_y_pred.extend(y_pred)
        all_test_indices.extend(test_idx)

        fold_acc = accuracy_score(y_test, y_pred)
        fold_bacc = balanced_accuracy_score(y_test, y_pred)
        fold_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        fold_results.append({
            "fold": fold_i + 1,
            "accuracy": fold_acc,
            "balanced_acc": fold_bacc,
            "f1_macro": fold_f1,
            "layer2_type": clf.layer2_type_,
            "layer2_threshold": getattr(clf, "layer2_threshold_", 0.5),
        })

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    acc = accuracy_score(all_y_true, all_y_pred)
    bacc = balanced_accuracy_score(all_y_true, all_y_pred)
    f1m = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred)

    precisions = precision_score(all_y_true, all_y_pred, average=None,
                                  labels=np.unique(all_y_true), zero_division=0)
    recalls = recall_score(all_y_true, all_y_pred, average=None,
                            labels=np.unique(all_y_true), zero_division=0)

    results = {
        "accuracy": acc,
        "balanced_acc": bacc,
        "f1_macro": f1m,
        "confusion_matrix": cm,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "classes": classes,
        "fold_details": pd.DataFrame(fold_results),
        "per_class_precision": dict(zip(classes, precisions)),
        "per_class_recall": dict(zip(classes, recalls)),
        "test_indices": all_test_indices,  # GroupKFold 顺序→原始 DataFrame 位置的映射
    }

    if logger:
        logger.info(f"  Hierarchical: acc={acc:.3f}, b_acc={bacc:.3f}, "
                     f"f1_macro={f1m:.3f}")
        logger.info(f"    Per-class precision: {results['per_class_precision']}")
        logger.info(f"    Per-class recall: {results['per_class_recall']}")

    return results
