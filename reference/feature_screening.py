"""
分层双排序融合特征筛选
=======================
为 Layer 1 (vasc vs non-vasc) 和 Layer 2 (mel vs nv) 分别做
binary F-test + RF importance 双排序融合，叠加 ICC 鲁棒性惩罚。

文献依据：
  - Saratxaga 2023: 颜色对 mel/nv 鉴别不足，需结构/纹理特征
  - Liopyris 2023: 高一致性 dermoscopic 特征全为结构性
  - Benyahia 2022: 层次分类各层应使用不同特征子集

融合策略：
  - RRF (Reciprocal Rank Fusion): 1/(k+rank_F) + 1/(k+rank_RF)
  - Borda Count: (N-rank_F) + (N-rank_RF)
  - Weighted Sum: w_F*F_norm + w_RF*RF_norm
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .dataset_analysis import _assign_category


# 颜色空间+通道组合 (用于多样性惩罚的通道分组)
_COLOR_SPACES = ["RGB", "HSV", "Lab", "YCrCb"]
_COLOR_CHANNELS = {
    "RGB": ["R", "G", "B"],
    "HSV": ["H", "S", "V"],
    "Lab": ["L", "a", "b"],
    "YCrCb": ["Y", "Cr", "Cb"],
}


def _extract_channel_group(feat_name: str) -> str:
    """
    从特征名中提取通道分组标签，用于多样性惩罚。

    分组逻辑:
      - 颜色特征: 按 {ColorSpace}_{Channel} 分组 (如 HSV_H, RGB_R, Lab_a)
        匹配模式: {prefix}_{CS}_{CH}_{stat} 其中 CS∈{RGB,HSV,Lab,YCrCb}
      - GLCM: 按属性分组 (如 GLCM_contrast, GLCM_homogeneity)
      - LBP/Gabor/Wavelet/Spectral: 各自成组
      - 形状: 按子类型分组 (bnd, asym, hu, fd, sig, reg_shape)

    Returns
    -------
    group_label : str — 通道分组标签
    """
    name = feat_name

    # 0) PCA 变换后的特征: PCA_HSV_H_0 → group "HSV_H", PCA_GLCM_contrast_2 → group "GLCM_contrast"
    if name.startswith("PCA_"):
        # 去掉 PCA_ 前缀和 _\d+ 后缀
        base = name[4:]  # "HSV_H_0", "GLCM_contrast_2"
        # 去掉末尾的数字索引
        base = re.sub(r"_\d+$", "", base)  # "HSV_H", "GLCM_contrast"
        return base

    # 1) 交互特征: inter_X_x_Y, inter_X_div_Y, inter_X_sub_Y
    if name.startswith("inter_"):
        return "interaction"

    # 2) 颜色特征: 查找 {ColorSpace}_{Channel} 模式
    #    使用 (?:_|^) 和 (?:_|$) 做边界而非 \b, 因为 \b 不匹配 _
    #    否则 reg_inner50_Lab_b_range 中的 Lab_b 后面是 _range,
    #    _ 是 \w 字符, \b 不触发 → 全部落入 reg_general (288 个特征误分类)
    for cs in _COLOR_SPACES:
        for ch in _COLOR_CHANNELS[cs]:
            if re.search(rf"(?:^|_){cs}_{ch}(?:$|_)", name):
                return f"{cs}_{ch}"

    # 3) GLCM 纹理: glcm_{property}_d{N}_{a{N}|mean|range|std}
    m = re.match(r"glcm_([a-z]+)_d\d+_(a\d+|mean|range|std|max|min|skew|kurt)", name)
    if m:
        return f"GLCM_{m.group(1)}"

    # 4) LBP
    if name.startswith("lbp_"):
        return "LBP"

    # 5) Gabor
    if name.startswith("gabor_"):
        return "Gabor"

    # 6) Wavelet
    if name.startswith("wave_"):
        return "Wavelet"

    # 7) Spectral
    if name.startswith("spec_"):
        return "Spectral"

    # 8) 形状特征子类型
    for shape_prefix in ["bnd_", "asym_", "hu_", "fd_", "sig_",
                          "reg_convex", "reg_solidity",
                          "topo_", "fractal_"]:
        if name.startswith(shape_prefix):
            return f"Shape_{shape_prefix.rstrip('_')}"

    # 9) 相对色彩 rel_
    if name.startswith("rel_"):
        return "rel_color"

    # 10) 颜色恒常性 ccr_
    if name.startswith("ccr_"):
        return "ccr"

    # 11) 特异性检测器 specific_
    if name.startswith("specific_"):
        return "specific"

    # 12) 边界对比度 bord_
    if name.startswith("bord_"):
        return "bord"

    # 13) 区域通用 reg_ (未匹配到 inner50/border/outer 的)
    if name.startswith("reg_"):
        return "reg_general"

    # 14) 一阶强度 (未匹配到具体颜色空间的 int_)
    if name.startswith("int_"):
        return "intensity_other"

    # fallback
    return "other"


def apply_within_group_pca(X_train, X_test, feat_names,
                           variance_threshold=0.95, min_group_size=3,
                           logger=None):
    """
    组内 PCA 降维：对每个特征通道组独立拟合 PCA，保留指定方差比例。

    在 CV 每折内调用：PCA 仅在 train 上拟合，train 和 test 同时变换，
    确保 CV-leak-free。

    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame, shape (n_train, n_features)
    X_test : np.ndarray or pd.DataFrame, shape (n_test, n_features)
    feat_names : list of str, 特征名列表
    variance_threshold : float, PCA 保留的方差比例 (默认 0.95)
    min_group_size : int, 少于此特征数的组不做 PCA, 直接保留原特征
    logger : optional logging function

    Returns
    -------
    X_train_pca : pd.DataFrame
    X_test_pca : pd.DataFrame
    pca_info : dict with keys:
        - "group_components": {group_name: n_components}
        - "total_before": int, 原始特征数
        - "total_after": int, PCA 后特征数
        - "pca_objects": {group_name: fitted PCA}  (for inspection)
    """
    if isinstance(X_train, pd.DataFrame):
        X_train_arr = X_train.values
    else:
        X_train_arr = X_train
    if isinstance(X_test, pd.DataFrame):
        X_test_arr = X_test.values
    else:
        X_test_arr = X_test

    n_samples_train = X_train_arr.shape[0]
    n_samples_test = X_test_arr.shape[0]

    # 分组
    groups = {}
    for i, name in enumerate(feat_names):
        grp = _extract_channel_group(name)
        groups.setdefault(grp, []).append(i)

    train_parts = []
    test_parts = []
    new_cols = []
    group_components = {}
    pca_objects = {}
    scaler_objects = {}  # v5.0: 保存 StandardScaler 供测试时复用

    for grp_name, indices in sorted(groups.items()):
        n_in_group = len(indices)
        if n_in_group < min_group_size:
            # 小组合并取平均值或直接保留: 这里直接保留
            train_parts.append(pd.DataFrame(
                X_train_arr[:, indices],
                columns=[feat_names[i] for i in indices],
            ))
            test_parts.append(pd.DataFrame(
                X_test_arr[:, indices],
                columns=[feat_names[i] for i in indices],
            ))
            new_cols.extend([feat_names[i] for i in indices])
            group_components[grp_name] = n_in_group
            continue

        # 提取组内特征
        X_grp_train = X_train_arr[:, indices]
        X_grp_test = X_test_arr[:, indices]

        # Standardize
        scaler = StandardScaler()
        X_grp_train_scaled = scaler.fit_transform(X_grp_train)
        X_grp_test_scaled = scaler.transform(X_grp_test)

        # PCA
        n_max = min(n_samples_train, n_in_group)
        pca = PCA(n_components=min(n_in_group, n_max))
        pca.fit(X_grp_train_scaled)

        # 确定保留的组件数
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumsum, variance_threshold) + 1)
        n_keep = max(1, min(n_keep, n_max))  # 至少保留 1 个, 最多 n_max

        # 用选定的组件数重新拟合 (或直接截断)
        X_grp_train_pca = pca.transform(X_grp_train_scaled)[:, :n_keep]
        X_grp_test_pca = pca.transform(X_grp_test_scaled)[:, :n_keep]

        col_names = [f"PCA_{grp_name}_{j}" for j in range(n_keep)]
        train_parts.append(pd.DataFrame(X_grp_train_pca, columns=col_names))
        test_parts.append(pd.DataFrame(X_grp_test_pca, columns=col_names))
        new_cols.extend(col_names)
        group_components[grp_name] = n_keep
        pca_objects[grp_name] = pca
        scaler_objects[grp_name] = scaler  # v5.0: 保存 scaler

    X_train_pca = pd.concat(train_parts, axis=1)
    X_test_pca = pd.concat(test_parts, axis=1)

    pca_info = {
        "group_components": group_components,
        "total_before": len(feat_names),
        "total_after": len(new_cols),
        "pca_objects": pca_objects,
        "scaler_objects": scaler_objects,  # v5.0
    }

    if logger:
        logger.info(
            f"  Within-group PCA: {len(feat_names)} → {len(new_cols)} features "
            f"({len(new_cols) / len(feat_names) * 100:.1f}%), "
            f"variance_threshold={variance_threshold}"
        )

    return X_train_pca, X_test_pca, pca_info


def _normalize_ranking(f_vals, rf_importance_dict, feature_names,
                       strategy="rrf", w_f=0.5, w_rf=0.5):
    """
    融合 F-test 排名和 RF 重要性排名。

    Parameters
    ----------
    f_vals : F-test 值数组 [n_features]
    rf_importance_dict : {feature_name: importance} — 可能只覆盖 top-N
    feature_names : list of str
    strategy : "rrf" | "borda" | "weighted"
    w_f, w_rf : Weighted 策略下的权重

    Returns
    -------
    fusion_scores : np.ndarray [n_features], 值越高越优
    rank_F : np.ndarray [n_features], rank=1 最优
    rank_RF : np.ndarray [n_features], rank=1 最优
    """
    n = len(feature_names)

    # F-test 排名 (降序: 最大F → rank 1)
    order_F = np.argsort(-f_vals)
    rank_F = np.zeros(n, dtype=np.float64)
    for r, idx in enumerate(order_F):
        rank_F[idx] = r + 1

    # RF 重要性排名 (降序)
    rank_RF = np.zeros(n, dtype=np.float64)
    rf_present = 0
    for i, name in enumerate(feature_names):
        if name in rf_importance_dict:
            rank_RF[i] = 0  # 临时，后续统一排名
            rf_present += 1
        else:
            rank_RF[i] = n  # 不在 top-N 的给最差排名

    # 对 RF 覆盖的特征排名
    rf_covered_names = [name for name in feature_names if name in rf_importance_dict]
    if rf_covered_names:
        rf_vals_for_rank = np.array([rf_importance_dict[name] for name in rf_covered_names])
        order_RF = np.argsort(-rf_vals_for_rank)
        for r, name_idx in enumerate(order_RF):
            feat_name = rf_covered_names[name_idx]
            i = feature_names.index(feat_name)
            rank_RF[i] = r + 1

    if strategy == "rrf":
        k = 60
        fusion = 1.0 / (k + rank_F) + 1.0 / (k + rank_RF)
    elif strategy == "borda":
        fusion = (n - rank_F) + (n - rank_RF)
    else:  # weighted
        f_norm = (n - rank_F) / (n - 1)
        rf_norm = np.where(rank_RF < n,
                           (n - rank_RF) / (n - 1),
                           0.0)  # 不在 RF top-N 的得 0
        fusion = w_f * f_norm + w_rf * rf_norm

    return fusion, rank_F.astype(int), rank_RF.astype(int)


def _icc_penalty_multiplier(icc_values, layer=1):
    """
    将 ICC 值映射为惩罚乘数 [0, 1]。
    Layer 2 比 Layer 1 更严格。
    """
    multipliers = np.ones(len(icc_values), dtype=np.float64)
    for i, icc in enumerate(icc_values):
        if np.isnan(icc):
            m = 0.60 if layer == 1 else 0.40
        elif icc >= 0.90:
            m = 1.00
        elif icc >= 0.75:
            m = 0.95 if layer == 2 else 1.00
        elif icc >= 0.50:
            m = 0.80 if layer == 2 else 0.90
        else:
            m = 0.60 if layer == 2 else 0.75
        multipliers[i] = m
    return multipliers


def _mrmr_scores(X_arr, y, feat_names, top_k=30):
    """
    mRMR (minimal Redundancy Maximum Relevance) 特征评分。

    用 MI 衡量相关性, 用 Pearson |r| 衡量冗余,
    贪心选择最大化 (relevance - redundancy) 的特征。

    Returns
    -------
    scores : np.ndarray [n_features], 值越高越优
    rank_mrmr : np.ndarray [n_features], rank=1 最优
    """
    n = len(feat_names)
    k = min(top_k, n)

    # Relevance: MI with class
    relevance = mutual_info_classif(X_arr, y, random_state=42, n_neighbors=3)

    # Redundancy proxy: pre-compute correlation matrix
    # 用 |r| 近似 MI, 比 pairwise MI 快 ~100x
    corr = np.abs(np.corrcoef(X_arr.T))
    np.fill_diagonal(corr, 0)  # 不自相关

    selected = []
    remaining = set(range(n))

    # First feature: max relevance
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)

    # Greedy: maximize relevance - mean(corr with selected)
    for _ in range(1, k):
        best_score = -np.inf
        best_i = -1
        for i in remaining:
            red = corr[i, selected].mean()
            score = relevance[i] - red * relevance[i]  # 冗余惩罚与相关性成比例
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            break
        selected.append(best_i)
        remaining.remove(best_i)

    # Build scores: selected get descending values, unselected get -inf
    scores = np.full(n, -np.inf, dtype=np.float64)
    for rank, idx in enumerate(selected):
        scores[idx] = k - rank  # k, k-1, ..., 1

    rank_mrmr = np.full(n, n, dtype=int)
    for rank, idx in enumerate(selected):
        rank_mrmr[idx] = rank + 1  # 1-indexed

    return scores, rank_mrmr


class FeatureScreener(BaseEstimator, TransformerMixin):
    """
    sklearn 兼容的分层特征筛选器。

    在 fit() 阶段：
      1. 在 (X, y_binary) 上计算 F-test (或 mRMR 评分)
      2. 使用预加载的 RF importance 排名
      3. 按 strategy 融合双排名 (或直接用 mRMR)
      4. 叠加 ICC 鲁棒性惩罚
      5. 选出 Top-k 特征

    transform() 返回选中的特征子集。

    strategy 选项:
      - "borda" / "rrf" / "weighted": F-test + RF 双排序融合
      - "mrmr": 最小冗余最大相关 (MI relevance + |r| redundancy)
    """

    def __init__(self, rf_importance_dict=None, rob_df=None,
                 strategy="rrf", top_k=50, layer=2,
                 w_f=0.5, w_rf=0.5, icc_min=0.75,
                 clinical_priority_map=None,
                 channel_diversity=None, max_per_channel=None):
        """
        Parameters
        ----------
        rf_importance_dict : dict {feature_name: importance}
        rob_df : DataFrame with columns [feature, icc]
        strategy : "rrf" | "borda" | "weighted"
        top_k : 保留特征数
        layer : 1 or 2 — 决定 ICC 惩罚严格度
        w_f, w_rf : Weighted 策略权重
        icc_min : Layer 2 ICC 最低阈值
        clinical_priority_map : 可选的临床先验字典
        channel_diversity : 是否启用通道多样性惩罚 (None=Layer2自动启用)
        max_per_channel : 每通道组最大特征数 (None=auto: max(2, top_k//10))
        """
        self.rf_importance_dict = rf_importance_dict or {}
        self.rob_df = rob_df
        self.strategy = strategy
        self.top_k = top_k
        self.layer = layer
        self.w_f = w_f
        self.w_rf = w_rf
        self.icc_min = icc_min
        self.clinical_priority_map = clinical_priority_map or {}
        self.channel_diversity = channel_diversity
        self.max_per_channel = max_per_channel
        self.selected_features_ = None
        self.screening_report_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            feat_names = X.columns.tolist()
            X_arr = X.values
        else:
            feat_names = [f"f{i}" for i in range(X.shape[1])]
            X_arr = X

        n_features = len(feat_names)

        # 1) Scoring: F-test+RF / mRMR / elastic_net
        rank_mrmr = None
        if self.strategy == "mrmr":
            # mRMR: MI relevance + |r| redundancy, 自身已处理冗余
            if y is not None and len(np.unique(y)) >= 2:
                fusion_raw, rank_mrmr = _mrmr_scores(
                    X_arr, y, feat_names, top_k=self.top_k,
                )
            else:
                fusion_raw = np.ones(n_features)
                rank_mrmr = np.arange(1, n_features + 1)
            rank_F = rank_mrmr  # for report
            rank_RF = np.full(n_features, n_features, dtype=int)  # not used
            fusion_raw = fusion_raw * (n_features / self.top_k)

        elif self.strategy == "elastic_net":
            # Elastic Net: L1+L2 嵌入式选择, 自动处理多重共线性
            #   l1_ratio sweep: 偏 L1 侧 (0.7-1.0) 以鼓励稀疏性
            #   C sweep: 对数空间 20 个点, 内层 CV 自动选最优
            from sklearn.linear_model import LogisticRegressionCV
            from sklearn.preprocessing import StandardScaler

            if y is not None and len(np.unique(y)) >= 2:
                X_scaled = StandardScaler().fit_transform(X_arr)
                enet = LogisticRegressionCV(
                    penalty='elasticnet', solver='saga',
                    l1_ratios=[0.7, 0.8, 0.9, 0.95, 1.0],
                    Cs=20, cv=min(5, max(2, len(y) // 50)),
                    scoring='balanced_accuracy',
                    class_weight='balanced',
                    max_iter=5000, tol=1e-3,
                    random_state=42, n_jobs=-1,
                )
                enet.fit(X_scaled, y)
                # 系数绝对值作为特征得分 (sparse: 很多系数为 0)
                coef = np.abs(enet.coef_[0])
                # 映射为 fusion-like score: 非零系数得高分, 零系数得 -inf
                # 先用 rank 再映射, 保证与 borda 兼容后续 ICC/clinical
                max_coef = coef.max()
                if max_coef > 0:
                    fusion_raw = np.where(coef > 0, coef / max_coef * n_features, -np.inf)
                else:
                    fusion_raw = np.ones(n_features)
                # 构建排名报告
                order = np.argsort(-np.where(coef > 0, coef, -1))
                rank_F = np.zeros(n_features, dtype=int)
                for r, idx in enumerate(order):
                    rank_F[idx] = r + 1
                rank_RF = np.full(n_features, n_features, dtype=int)
            else:
                fusion_raw = np.ones(n_features)
                rank_F = np.arange(1, n_features + 1)
                rank_RF = np.full(n_features, n_features, dtype=int)

        else:
            if y is not None and len(np.unique(y)) >= 2:
                f_vals, _ = f_classif(X_arr, y)
            else:
                f_vals = np.ones(n_features)

            # 2) 双排序融合
            fusion_raw, rank_F, rank_RF = _normalize_ranking(
                f_vals, self.rf_importance_dict, feat_names,
                strategy=self.strategy, w_f=self.w_f, w_rf=self.w_rf,
            )

        # 3) ICC 惩罚
        icc_dict = {}
        if self.rob_df is not None:
            rob_sub = self.rob_df.set_index("feature")
            for name in feat_names:
                if name in rob_sub.index:
                    icc_dict[name] = rob_sub.loc[name, "icc"]
                else:
                    icc_dict[name] = np.nan
        else:
            icc_dict = {name: 0.75 for name in feat_names}

        icc_vals = np.array([icc_dict.get(name, np.nan) for name in feat_names])
        penalties = _icc_penalty_multiplier(icc_vals, layer=self.layer)

        # 4) 严格 ICC 过滤 (仅 Layer 2)
        icc_mask = np.ones(n_features, dtype=bool)
        if self.layer == 2:
            icc_mask = np.array([
                not np.isnan(icc_dict[name]) and icc_dict[name] >= self.icc_min
                for name in feat_names
            ])
            if icc_mask.sum() < self.top_k:
                # 放宽限制
                icc_mask = np.ones(n_features, dtype=bool)

        # 5) 临床先验加分: penalty∈[0,1], 0=最高优先
        #    映射到乘数 [0.7, 1.3], 默认 1.0
        clinical_bonus = np.ones(n_features, dtype=np.float64)
        if self.clinical_priority_map:
            for i, name in enumerate(feat_names):
                for prefix, penalty in self.clinical_priority_map.items():
                    if name.startswith(prefix):
                        clinical_bonus[i] = 1.0 + 0.6 * (0.5 - penalty)
                        break

        # 6) 最终得分
        final_score = fusion_raw * penalties * clinical_bonus
        final_score[~icc_mask] = -np.inf

        # 7) 通道多样性选择 (防止单一颜色通道垄断 top-k)
        use_diversity = self.channel_diversity
        if use_diversity is None:
            use_diversity = (self.layer == 2)  # 默认 Layer 2 启用
        max_per_ch = self.max_per_channel
        if max_per_ch is None:
            max_per_ch = max(2, self.top_k // 10)

        if use_diversity:
            # 计算每个特征的通道分组
            channel_groups = [_extract_channel_group(name) for name in feat_names]

            # 按 final_score 降序排列的索引
            sorted_idx = np.argsort(-final_score)

            selected = []
            channel_counts = {}
            for idx in sorted_idx:
                if final_score[idx] <= -np.inf:
                    continue
                grp = channel_groups[idx]
                cnt = channel_counts.get(grp, 0)
                if cnt < max_per_ch:
                    selected.append(idx)
                    channel_counts[grp] = cnt + 1
                if len(selected) >= self.top_k:
                    break

            # 如果通道限制太严格导致无法填满 top-k, 放宽填充
            if len(selected) < self.top_k:
                for idx in sorted_idx:
                    if idx in selected or final_score[idx] <= -np.inf:
                        continue
                    selected.append(idx)
                    if len(selected) >= self.top_k:
                        break

            top_indices = np.array(selected)
        else:
            top_indices = np.argsort(-final_score)[:self.top_k]

        self.selected_features_ = [feat_names[i] for i in top_indices]

        # 构建报告
        self.screening_report_ = pd.DataFrame({
            "feature": feat_names,
            "rank_F": rank_F,
            "rank_RF": rank_RF,
            "icc": icc_vals,
            "icc_penalty": penalties,
            "fusion_raw": fusion_raw,
            "final_score": final_score,
            "selected": [name in self.selected_features_ for name in feat_names],
            "category": [_assign_category(name) for name in feat_names],
            "channel_group": [_extract_channel_group(name) for name in feat_names],
        }).sort_values("final_score", ascending=False)

        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("Must call fit() before transform()")
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            raise ValueError("X must be a pandas DataFrame")

    def get_report(self, n=30):
        if self.screening_report_ is None:
            raise RuntimeError("Must call fit() first")
        return self.screening_report_.head(n)


def screen_features_layer(X, y_binary, rf_importance_dict=None,
                           rob_df=None, strategy="rrf", top_k=50,
                           layer=2, w_f=0.5, w_rf=0.5, icc_min=0.75,
                           clinical_priority_map=None, logger=None,
                           channel_diversity=None, max_per_channel=None):
    """
    对单个层执行特征筛选的一键式函数。

    Parameters
    ----------
    X : pd.DataFrame, 特征矩阵
    y_binary : array-like, 该层的二分类标签 (e.g. "vasc" vs "non-vasc" → 0/1)
    rf_importance_dict : dict {feature: importance}
    rob_df : ICC DataFrame
    strategy : "rrf" | "borda" | "weighted"
    top_k : 保留特征数
    layer : 1 or 2
    clinical_priority_map : 临床先验字典
    channel_diversity : 是否启用通道多样性 (None=Layer2自动启用)
    max_per_channel : 每通道最大特征数 (None=auto)

    Returns
    -------
    X_screened : pd.DataFrame
    report : pd.DataFrame
    selected_names : list of str
    """
    screener = FeatureScreener(
        rf_importance_dict=rf_importance_dict,
        rob_df=rob_df,
        strategy=strategy,
        top_k=top_k,
        layer=layer,
        w_f=w_f,
        w_rf=w_rf,
        icc_min=icc_min,
        clinical_priority_map=clinical_priority_map,
        channel_diversity=channel_diversity,
        max_per_channel=max_per_channel,
    )
    screener.fit(X, y_binary)

    if logger:
        report_head = screener.get_report(10)
        logger.info(f"  Layer {layer} screening: top-{top_k} features selected")
        logger.info(f"  Top-5: {screener.selected_features_[:5]}")
        cats = report_head["category"].value_counts()
        logger.info(f"  Category distribution (top-{len(report_head)}): {cats.to_dict()}")

    X_screened = screener.transform(X)
    return X_screened, screener.screening_report_, screener.selected_features_


def compute_layer_specific_f_ranking(X, y_binary, feature_names, save_path=None):
    """
    在给定 binary label 上计算 F-test 排名，保存至 CSV。
    用于替代全局 3 类 F-test 排名 —— 按层独立计算。
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    f_vals, p_vals = f_classif(X_arr, y_binary)
    df = pd.DataFrame({
        "feature": feature_names,
        "F": f_vals,
        "p": p_vals,
    }).sort_values("F", ascending=False)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def compute_layer_specific_rf_importance(X, y_binary, feature_names,
                                          random_state=42, save_path=None):
    """
    在给定 binary label 上训练 RF，输出特征重要性排名。
    返回 {feature_name: importance} dict 和排序后的 DataFrame。
    """
    from sklearn.ensemble import RandomForestClassifier
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    rf = RandomForestClassifier(
        n_estimators=400, n_jobs=-1, class_weight="balanced",
        random_state=random_state,
    )
    rf.fit(X_arr, y_binary)
    imp = pd.Series(rf.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False)
    if save_path:
        imp.reset_index().to_csv(save_path, index=False,
                                 header=["feature", "importance"])
    return dict(imp), imp.reset_index().rename(columns={"index": "feature", 0: "importance"})
