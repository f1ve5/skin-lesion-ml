"""
鲁棒特征筛选实验（Robustness-aware Feature Selection）
===========================================================
文献依据：
  - Xue et al. 2021 Quant Imaging Med Surg 系统综述: 用 ICC>0.75 阈值
    过滤掉不稳健的特征是 radiomics 最佳实践。
  - Edwards-Salzano et al.: filtering threshold ICC>0.95 显著提高预测
    模型的可泛化性。

实验：
  在已有 features.csv 和 robustness_per_feature.csv 基础上，
  分别用 ICC 阈值 = [无, 0.5, 0.75, 0.90] 筛选特征训练分类器，
  比较 5-fold GroupKFold 性能。

期望结果（按文献预期）：
  随阈值升高，特征数减少 → 模型可能略降 acc 但提升泛化稳定性。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from analysis.viz_features import setup_style
from classification.classify import get_classifiers


def filter_features_by_icc(feat_cols: list,
                            rob_df: pd.DataFrame,
                            icc_threshold: float) -> list:
    """返回 ICC ≥ threshold 的特征名列表"""
    if icc_threshold is None or icc_threshold <= 0:
        return list(feat_cols)
    keep = rob_df[rob_df["icc"] >= icc_threshold]["feature"].tolist()
    # 只保留同时存在于 feat_cols 中的
    return [f for f in feat_cols if f in keep]


def robustness_filter_experiment(X: pd.DataFrame,
                                   y: pd.Series,
                                   groups: pd.Series,
                                   rob_df: pd.DataFrame,
                                   thresholds=(0.0, 0.5, 0.75, 0.90),
                                   classifier_name: str = "SVM-RBF",
                                   n_splits: int = 5,
                                   random_state: int = 42,
                                   logger=None) -> pd.DataFrame:
    """
    对每个 ICC 阈值，跑一次完整 5-fold GroupKFold，并记录指标。
    """
    le = LabelEncoder(); y_enc = le.fit_transform(y.values)
    n_groups = groups.nunique()
    n_splits = min(n_splits, n_groups)

    rows = []
    for thr in thresholds:
        kept = filter_features_by_icc(X.columns.tolist(), rob_df, thr)
        if len(kept) < 5:
            if logger:
                logger.warning(f"  阈值 {thr}: 只剩 {len(kept)} 个特征，跳过")
            continue
        Xf = X[kept]
        if logger:
            logger.info(f"  ICC ≥ {thr}: 保留 {len(kept)}/{X.shape[1]} 维")

        model = get_classifiers(random_state)[classifier_name]
        try:
            gkf = GroupKFold(n_splits=n_splits)
            y_pred = cross_val_predict(model, Xf.values, y_enc,
                                         groups=groups.values, cv=gkf, n_jobs=1)
            acc  = accuracy_score(y_enc, y_pred)
            bacc = balanced_accuracy_score(y_enc, y_pred)
            f1m  = f1_score(y_enc, y_pred, average="macro", zero_division=0)
        except Exception as e:
            if logger:
                logger.warning(f"  阈值 {thr} 训练失败: {e}")
            continue

        rows.append({
            "icc_threshold": thr,
            "n_features":    len(kept),
            "accuracy":      acc,
            "balanced_acc":  bacc,
            "f1_macro":      f1m,
        })
        if logger:
            logger.info(f"    acc={acc:.3f}, b_acc={bacc:.3f}, f1m={f1m:.3f}")

    return pd.DataFrame(rows)


def viz_robustness_filter_experiment(result_df: pd.DataFrame, save_path):
    """画 ICC 阈值 vs 性能 vs 特征数 的双轴图"""
    setup_style()
    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = result_df["icc_threshold"].astype(str).tolist()

    ax1.plot(x, result_df["accuracy"],      "o-",  lw=2, label="Accuracy",      color="#4C72B0")
    ax1.plot(x, result_df["balanced_acc"],  "s-",  lw=2, label="Balanced Acc",  color="#DD8452")
    ax1.plot(x, result_df["f1_macro"],      "^-",  lw=2, label="F1 (macro)",    color="#55A467")
    ax1.set_xlabel("ICC threshold (≥)")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0.5, 1.0)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(x, result_df["n_features"], alpha=0.2, color="gray",
            label="# features")
    ax2.set_ylabel("# features retained", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # 数值标签
    for xi, ni in zip(x, result_df["n_features"]):
        ax2.text(xi, ni, str(int(ni)), ha="center", va="bottom",
                 fontsize=9, color="gray")

    plt.title("Robustness-aware Feature Selection — Effect on Classifier Performance",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
