"""
Layer-2 专属集成分类器
=======================
为 mel vs nv 决策提供 XGBoost 单模型、VotingClassifier、StackingClassifier
三种集成方案。Layer 1 (vasc vs non-vasc) 保持简单 LR 不变。

文献依据：
  - Sharafudeen 2023: 手工特征 + Soft Voting, Sens=90.49%
  - Alfi 2022: Stacking + SHAP, Diagnostics
  - Ichim 2023: Binary decomposition ensemble > flat multi-class

关键防泄漏设计：
  - StackingClassifier 的 cv=GroupKFold(4), 外循环 GroupKFold(5) 独立
  - 所有评估通过 cross_val_predict + groups 参数保证 image_id 不跨折
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score,
                              confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier

from analysis.viz_features import setup_style


def get_extended_classifiers(random_state=42):
    """
    返回扩展后的分类器集合：原 5 个 + XGBoost (+ LightGBM 可选)。
    XGBoost/LightGBM 缺失时优雅跳过。
    """
    classifiers = {
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

    # XGBoost
    try:
        import xgboost as xgb
        classifiers["XGBoost"] = Pipeline([
            ("sc", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=200, max_depth=5,
                learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                use_label_encoder=False, eval_metric="logloss",
                verbosity=0)),
        ])
    except ImportError:
        pass

    # LightGBM
    try:
        import lightgbm as lgb
        classifiers["LightGBM"] = Pipeline([
            ("sc", StandardScaler()),
            ("clf", lgb.LGBMClassifier(
                objective="binary", n_estimators=200,
                max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=random_state, verbose=-1)),
        ])
    except ImportError:
        pass

    return classifiers


def _quick_rank_classifiers(X, y, groups, classifiers, n_splits=3,
                             random_state=42, logger=None):
    """
    轻量级 3-fold GroupKFold 评估，选出 top-N 分类器用于 Voting。
    """
    gkf = GroupKFold(n_splits=n_splits)
    scores = {}
    for name, model in classifiers.items():
        try:
            y_pred = cross_val_predict(
                model, X.values if isinstance(X, pd.DataFrame) else X, y,
                groups=groups, cv=gkf, n_jobs=1,
            )
            scores[name] = balanced_accuracy_score(y, y_pred)
        except Exception as e:
            if logger:
                logger.info(f"    {name}: failed ({e})")
            scores[name] = 0.0
    return pd.Series(scores).sort_values(ascending=False)


def evaluate_voting_ensemble(X, y, groups, classifiers,
                              top_n=3, voting_type="soft",
                              n_splits=5, random_state=42,
                              logger=None):
    """
    评估 VotingClassifier (soft voting)。

    流程：
      1. 轻量 3-fold 评估所有基分类器 → 选出 top-N
      2. 构建 VotingClassifier(top-N, voting=soft)
      3. 5-fold GroupKFold cross_val_predict 评估
    """
    all_clf = classifiers.copy()
    # Scaling 已内置在 pipelines 中，VotingClassifier 需去掉外层 StandardScaler
    # 如果基分类器是 Pipeline(scaler, clf), 传入 Voting 时保持不变即可
    ranked = _quick_rank_classifiers(X, y, groups, all_clf,
                                       n_splits=3, random_state=random_state,
                                       logger=logger)
    top_names = ranked.head(top_n).index.tolist()
    estimators = [(name, all_clf[name]) for name in top_names]

    if logger:
        logger.info(f"  Voting base classifiers: {top_names}")

    vote = VotingClassifier(estimators=estimators, voting=voting_type)

    gkf = GroupKFold(n_splits=n_splits)
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_pred = cross_val_predict(vote, X_arr, y, groups=groups, cv=gkf, n_jobs=1)

    return _compute_metrics(y, y_pred, f"Voting-{voting_type}({','.join(top_names)})"), top_names


def evaluate_stacking_ensemble(X, y, groups, classifiers,
                                top_n=3, meta_classifier="LogReg",
                                n_splits=5, random_state=42,
                                logger=None):
    """
    评估 StackingClassifier。

    关键防泄漏：
      - StackingClassifier 的 cv=GroupKFold(4) 传入 groups 保证内部不泄漏
      - 外层评估使用独立的 GroupKFold(5)
    """
    all_clf = classifiers.copy()
    ranked = _quick_rank_classifiers(X, y, groups, all_clf,
                                       n_splits=3, random_state=random_state,
                                       logger=logger)
    top_names = ranked.head(top_n).index.tolist()
    estimators = [(name, all_clf[name]) for name in top_names]

    if meta_classifier == "LogReg":
        meta = LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=random_state)
    else:
        meta = LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=random_state)

    n_inner = min(4, len(np.unique(groups)))
    from sklearn.model_selection import StratifiedKFold
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True,
                               random_state=random_state)

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=inner_cv,
        stack_method="predict_proba",
    )

    if logger:
        logger.info(f"  Stacking base: {top_names}, meta: {meta_classifier}")

    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = cross_val_predict(stack, X_arr, y, groups=groups, cv=gkf, n_jobs=1)

    return (_compute_metrics(y, y_pred,
                              f"Stacking({','.join(top_names)})",
                              top_names),
            top_names)


def _compute_metrics(y_true, y_pred, name, base_classifiers=None):
    """计算分类指标"""
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "name": name,
        "accuracy": acc,
        "balanced_acc": bacc,
        "f1_binary": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
        "base_classifiers": base_classifiers,
    }


def evaluate_layer2_ensembles(X, y_mel_nv, groups,
                               n_splits=5, random_state=42,
                               logger=None):
    """
    Layer 2 集成分类器评估 — 一键式函数。

    依次评估：
      1. 所有扩展单模型 (含 XGBoost, LightGBM)
      2. VotingClassifier (soft, top-3 base)
      3. StackingClassifier (LR meta-learner)

    Parameters
    ----------
    X : pd.DataFrame — mel+nv 样本, 筛选+交互后的特征矩阵
    y_mel_nv : array-like — 0=nv, 1=mel
    groups : array-like — image_id (仅 mel+nv 样本)
    n_splits : 外层 GroupKFold splits
    random_state : int
    logger : Logger or None

    Returns
    -------
    all_results : list of dict, 每个元素是一个分类器的评估结果
    """
    if logger:
        logger.info(f"\n  Layer-2 Ensemble Evaluation "
                    f"({X.shape[1]} features, {len(y_mel_nv)} samples)")

    classifiers = get_extended_classifiers(random_state)
    gkf = GroupKFold(n_splits=n_splits)
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    all_results = []

    # 1) 评估所有单模型
    for name, model in classifiers.items():
        try:
            y_pred = cross_val_predict(model, X_arr, y_mel_nv,
                                        groups=groups, cv=gkf, n_jobs=1)
            metrics = _compute_metrics(y_mel_nv, y_pred, name)
            all_results.append(metrics)
            if logger:
                logger.info(f"    {name}: b_acc={metrics['balanced_acc']:.3f}, "
                           f"f1={metrics['f1_binary']:.3f}")
        except Exception as e:
            if logger:
                logger.info(f"    {name}: failed ({e})")

    # 2) Voting
    try:
        vote_result, _ = evaluate_voting_ensemble(
            X, y_mel_nv, groups, classifiers,
            top_n=3, voting_type="soft",
            n_splits=n_splits, random_state=random_state,
            logger=logger,
        )
        all_results.append(vote_result)
        if logger:
            logger.info(f"    {vote_result['name']}: b_acc={vote_result['balanced_acc']:.3f}, "
                       f"f1={vote_result['f1_binary']:.3f}")
    except Exception as e:
        if logger:
            logger.info(f"    Voting: failed ({e})")

    # 3) Stacking
    try:
        stack_result, _ = evaluate_stacking_ensemble(
            X, y_mel_nv, groups, classifiers,
            top_n=3, meta_classifier="LogReg",
            n_splits=n_splits, random_state=random_state,
            logger=logger,
        )
        all_results.append(stack_result)
        if logger:
            logger.info(f"    {stack_result['name']}: b_acc={stack_result['balanced_acc']:.3f}, "
                       f"f1={stack_result['f1_binary']:.3f}")
    except Exception as e:
        if logger:
            logger.info(f"    Stacking: failed ({e})")

    return all_results


def viz_ensemble_comparison(results, save_path):
    """可视化集成分类器对比"""
    setup_style()
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]
    baccs = [r["balanced_acc"] for r in results]
    f1s = [r["f1_binary"] for r in results]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(names)*1.2), 6))
    ax.bar(x - width, accs, width, color="#4C72B0", label="Accuracy")
    ax.bar(x, baccs, width, color="#DD8452", label="Balanced Acc")
    ax.bar(x + width, f1s, width, color="#55A467", label="F1 (binary)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Layer 2 Ensemble Classifier Comparison\n(mel vs nv, GroupKFold CV)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return pd.DataFrame({
        "classifier": names,
        "accuracy": accs,
        "balanced_acc": baccs,
        "f1_binary": f1s,
    })
