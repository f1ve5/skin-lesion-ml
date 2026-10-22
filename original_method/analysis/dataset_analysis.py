"""
特征分析（数据集级别）—— 升级版 v2
==========================================
新增对 reg_inner50_*, reg_border_*, reg_outer_*, rel_*, bord_* 类别的识别。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import f_classif

from .viz_features import setup_style


META_COLS = ["image_id", "aug_tag", "dx", "filename"]


def split_meta_features(df):
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].copy()
    y = df["dx"].copy()
    return X, y, feat_cols


def clean_features(X: pd.DataFrame, logger=None) -> pd.DataFrame:
    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if logger and const_cols:
        logger.info(f"删除 {len(const_cols)} 个常量特征")
    X = X.drop(columns=const_cols)
    X = X.replace([np.inf, -np.inf], np.nan)
    n_nan = X.isna().sum().sum()
    if n_nan > 0:
        if logger:
            logger.info(f"填补 {n_nan} 个 NaN/Inf 为列中位数")
        X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    return X


def viz_class_distribution(df, save_path):
    setup_style()
    counts = df.groupby(["dx", "aug_tag"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", stacked=True, ax=ax,
                color=["#4C72B0", "#DD8452", "#55A467"])
    ax.set_xlabel("Diagnosis (dx)"); ax.set_ylabel("Count")
    ax.set_title("Class Distribution (stacked by augmentation tag)")
    ax.legend(title="aug_tag", loc="upper right")
    for i, (idx, row) in enumerate(counts.iterrows()):
        total = row.sum()
        ax.text(i, total + 0.5, str(int(total)), ha="center", fontsize=9)
    plt.xticks(rotation=0); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path); plt.close(fig)


def _assign_category(name: str) -> str:
    """统一的特征 → 类别 映射（v2: 含新增特征）"""
    # 新增：区域分割特征
    if name.startswith("reg_inner50_"):  return "Region-Inner50"
    if name.startswith("reg_inner75_"):  return "Region-Inner75"
    if name.startswith("reg_border_"):   return "Region-Border"
    if name.startswith("reg_outer_"):    return "Region-Outer"
    # 新增：相对色彩特征
    if name.startswith("rel_"):          return "Relative-Color"
    # 新增：边界对比度
    if name.startswith("bord_"):         return "Border-Contrast"
    # 原有：注意 reg_ 不带子区域的是 region shape feature
    if name.startswith(("bnd_", "fd_", "sig_")):  return "Shape-Boundary"
    if name.startswith("reg_"):          return "Shape-Region"
    if name.startswith("hu_"):           return "Shape-Hu"
    if name.startswith("topo_"):         return "Shape-Topo"
    if name.startswith("asym_"):         return "Shape-Asymmetry"
    if name.startswith("int_"):          return "Intensity"
    if name.startswith("glcm_"):         return "GLCM"
    if name.startswith("lbp_"):          return "LBP"
    if name.startswith("gabor_"):        return "Gabor"
    if name.startswith("wave_"):         return "Wavelet"
    if name.startswith("spec_"):         return "Spectral"
    return "Other"


def viz_feature_category_count(feat_cols, save_path):
    setup_style()
    counts = {}
    for c in feat_cols:
        cat = _assign_category(c)
        counts[cat] = counts.get(cat, 0) + 1
    counts = dict(sorted(counts.items(), key=lambda x: -x[1]))
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.cm.tab20
    bars = ax.barh(list(counts.keys()), list(counts.values()),
                   color=[cmap(i) for i in range(len(counts))])
    for bar, v in zip(bars, counts.values()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=10)
    ax.set_xlabel("Number of features")
    ax.set_title(f"Feature Categories ({sum(counts.values())} features total, v2 upgrade)")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path); plt.close(fig)
    return counts


def viz_correlation_heatmap(X: pd.DataFrame, save_path, top_k=50):
    setup_style()
    variances = X.var().sort_values(ascending=False)
    cols = variances.head(top_k).index.tolist()
    Xs = StandardScaler().fit_transform(X[cols])
    corr = pd.DataFrame(Xs, columns=cols).corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f"Correlation Heatmap of Top-{top_k} Highest-variance Features",
                 fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, fontsize=6); plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path); plt.close(fig)
    return corr


def viz_pca_tsne(X: pd.DataFrame, y: pd.Series, save_path, random_state=42):
    setup_style()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=random_state)
    Xp = pca.fit_transform(Xs)
    n = len(Xs)
    perp = min(30, max(5, (n - 1) // 3))
    try:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state,
                    init="pca", learning_rate="auto")
        Xt = tsne.fit_transform(Xs)
    except Exception:
        Xt = Xp

    classes = sorted(y.unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(classes), 3)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for c, col in zip(classes, colors):
        sel = (y == c).values
        axes[0].scatter(Xp[sel, 0], Xp[sel, 1], c=[col], s=30,
                        alpha=0.7, edgecolor="white", lw=0.4, label=c)
        axes[1].scatter(Xt[sel, 0], Xt[sel, 1], c=[col], s=30,
                        alpha=0.7, edgecolor="white", lw=0.4, label=c)
    axes[0].set_title(f"PCA (var explained: {pca.explained_variance_ratio_.sum():.1%})")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(loc="best", fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].set_title(f"t-SNE (perplexity={perp})")
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(loc="best", fontsize=9); axes[1].grid(alpha=0.3)
    fig.suptitle("Low-dim Embedding of Extracted Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path); plt.close(fig)


def viz_pca_variance(X: pd.DataFrame, save_path, max_pc=30):
    setup_style()
    Xs = StandardScaler().fit_transform(X)
    n = min(max_pc, min(Xs.shape))
    pca = PCA(n_components=n); pca.fit(Xs)
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(1, n+1), var, color="steelblue", alpha=0.7, label="individual")
    ax.plot(range(1, n+1), cum, "r-o", lw=1.2, markersize=4, label="cumulative")
    ax.axhline(0.9, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Principal Component"); ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA - explained variance"); ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path); plt.close(fig)


def viz_top_discriminative_features(X: pd.DataFrame, y: pd.Series,
                                      save_path, top_k=15):
    setup_style()
    classes = sorted(y.unique())
    code = {c: i for i, c in enumerate(classes)}
    y_enc = y.map(code).values
    try:
        f_vals, p_vals = f_classif(X.values, y_enc)
    except ValueError:
        return None
    rank_df = pd.DataFrame({
        "feature": X.columns, "F": f_vals, "p": p_vals
    }).sort_values("F", ascending=False)
    top_feats = rank_df.head(top_k)["feature"].tolist()

    n_cols = 3
    n_rows = (top_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2*n_cols, 3.2*n_rows))
    axes = axes.flatten()
    palette = sns.color_palette("Set2", n_colors=len(classes))
    for ax, fname in zip(axes, top_feats):
        data = [X.loc[(y == c).values, fname].values for c in classes]
        bp = ax.boxplot(data, labels=classes, patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], palette):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_title(fname, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(top_feats):]:
        ax.axis("off")
    fig.suptitle(f"Top-{top_k} discriminative features (ANOVA F-test)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path); plt.close(fig)
    return rank_df
