"""
鲁棒性分析（v2）—— 加入对新增特征类别的识别
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .viz_features import setup_style
from .dataset_analysis import _assign_category


def compute_icc21(df_wide: pd.DataFrame) -> float:
    """ICC(2,1) two-way random absolute agreement single rater"""
    Y = df_wide.values.astype(np.float64)
    n, k = Y.shape
    if n < 2 or k < 2:
        return np.nan
    grand_mean = Y.mean()
    row_mean = Y.mean(axis=1); col_mean = Y.mean(axis=0)
    SS_total   = ((Y - grand_mean) ** 2).sum()
    SS_between = k * ((row_mean - grand_mean) ** 2).sum()
    SS_judge   = n * ((col_mean - grand_mean) ** 2).sum()
    SS_error   = SS_total - SS_between - SS_judge
    if (n - 1) <= 0 or (k - 1) <= 0 or (n - 1) * (k - 1) <= 0:
        return np.nan
    MS_between = SS_between / (n - 1)
    MS_judge   = SS_judge   / (k - 1)
    MS_error   = SS_error   / ((n - 1) * (k - 1))
    denom = MS_between + (k - 1) * MS_error + k * (MS_judge - MS_error) / n
    if denom <= 0:
        return np.nan
    return float((MS_between - MS_error) / denom)


def build_paired_table(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    sub = df[["image_id", "aug_tag", feature]].copy()
    wide = sub.pivot_table(index="image_id", columns="aug_tag",
                            values=feature, aggfunc="mean")
    return wide.dropna()


def compute_robustness_per_feature(df: pd.DataFrame,
                                    feature_cols: list) -> pd.DataFrame:
    rows = []
    for feat in feature_cols:
        wide = build_paired_table(df, feat)
        if wide.empty or wide.shape[0] < 3 or wide.shape[1] < 2:
            rows.append({"feature": feat, "icc": np.nan,
                         "pearson": np.nan, "n_paired": wide.shape[0]})
            continue
        icc = compute_icc21(wide)
        if "orig" in wide.columns:
            ref = wide["orig"].values
            other_cols = [c for c in wide.columns if c != "orig"]
            if other_cols:
                other_mean = wide[other_cols].mean(axis=1).values
                if np.std(ref) > 1e-12 and np.std(other_mean) > 1e-12:
                    pear = np.corrcoef(ref, other_mean)[0, 1]
                else:
                    pear = np.nan
            else:
                pear = np.nan
        else:
            pear = np.nan
        rows.append({"feature": feat, "icc": icc,
                     "pearson": pear, "n_paired": wide.shape[0]})
    return pd.DataFrame(rows)


def viz_robustness_overview(rob_df: pd.DataFrame, save_path):
    setup_style()
    rob = rob_df.dropna(subset=["icc"]).copy()
    rob["category"] = rob["feature"].apply(_assign_category)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：ICC 直方图
    axes[0].hist(rob["icc"].clip(-0.2, 1.0), bins=40,
                  color="steelblue", edgecolor="black", lw=0.5)
    axes[0].axvline(0.75, color="red", ls="--", lw=1.2,
                     label="ICC=0.75 (good)")
    axes[0].axvline(0.90, color="green", ls="--", lw=1.2,
                     label="ICC=0.90 (excellent)")
    axes[0].set_xlabel("ICC(2,1)"); axes[0].set_ylabel("# features")
    axes[0].set_title(
        f"ICC distribution  (median = {rob['icc'].median():.3f}, "
        f"≥0.75: {(rob['icc']>=0.75).mean():.1%}, "
        f"≥0.90: {(rob['icc']>=0.90).mean():.1%})")
    axes[0].legend(loc="upper left"); axes[0].grid(alpha=0.3)

    # 右：按分类的箱线图
    order = sorted(rob["category"].unique(),
                   key=lambda c: -rob[rob["category"] == c]["icc"].median())
    palette = sns.color_palette("Set3", n_colors=len(order))
    sns.boxplot(data=rob, x="category", y="icc", order=order,
                palette=palette, ax=axes[1])
    sns.stripplot(data=rob, x="category", y="icc", order=order,
                  color="black", size=2, alpha=0.4, ax=axes[1])
    axes[1].axhline(0.75, color="red", ls="--", lw=1)
    axes[1].set_title("ICC by feature category")
    axes[1].set_xlabel(""); axes[1].set_ylabel("ICC(2,1)")
    plt.setp(axes[1].get_xticklabels(), rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Robustness Analysis - Original vs Augmented (v2)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path); plt.close(fig)


def viz_robustness_scatter_examples(df: pd.DataFrame, rob_df: pd.DataFrame,
                                      save_path, n_top=3, n_bot=3):
    setup_style()
    rob = rob_df.dropna(subset=["icc"]).copy()
    rob_sorted = rob.sort_values("icc", ascending=False)
    top_feats = rob_sorted.head(n_top)["feature"].tolist()
    bot_feats = rob_sorted.tail(n_bot)["feature"].tolist()
    feats = top_feats + bot_feats

    n = len(feats)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 4*n_rows))
    axes = axes.flatten()
    for i, feat in enumerate(feats):
        ax = axes[i]
        wide = build_paired_table(df, feat)
        if wide.empty or "orig" not in wide.columns:
            ax.axis("off"); continue
        ref = wide["orig"].values
        others = [c for c in wide.columns if c != "orig"]
        other_mean = wide[others].mean(axis=1).values
        ax.scatter(ref, other_mean, s=24, alpha=0.6, edgecolor="white")
        mn = min(ref.min(), other_mean.min())
        mx = max(ref.max(), other_mean.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="y=x")
        ax.set_xlabel("orig"); ax.set_ylabel("mean(aug)")
        icc_v = rob.loc[rob["feature"] == feat, "icc"].iloc[0]
        ax.set_title(f"{feat}\nICC={icc_v:.3f}", fontsize=9)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Most stable (top {n_top}) and least stable (bottom {n_bot}) features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path); plt.close(fig)
