"""
Layer-2 专属跨族交互特征工程
===============================
仅在 mel vs nv 样本上生成结构族 (Structural) × 颜色族 (Color) 的
乘积/比值交互特征，用于捕获单族特征无法表达的联合诊断模式。

文献依据：
  - Moldovanu 2023: hybrid features (color+texture+morphological) 优于纯颜色
  - Liopyris 2023: 高一致性 dermoscopic features 全为结构性
  - 当前 SHAP 交互分析已检测到 bord_blur_L × int_Lab_b_p75 等有意义的交互对

算法：
  1. 在 mel/nv 子集上分别计算 F-test (mel vs nv binary) 和 RF importance
  2. delta = RF_norm - F_norm: 正 delta → Color 族, 负 delta → Structural 族
  3. top-M Structural × top-N Color → 乘积/比值特征
  4. 互信息 (mutual_info_classif) 后筛选至 max_keep 维
"""
import numpy as np
import pandas as pd
from pathlib import Path


def _normalize_to_01(series_or_array):
    """线性归一化到 [0, 1]"""
    arr = np.asarray(series_or_array, dtype=np.float64)
    rng = arr.max() - arr.min()
    if rng < 1e-12:
        return np.ones_like(arr) * 0.5
    return (arr - arr.min()) / rng


def identify_complementary_families(feat_names, f_vals, rf_importance_dict):
    """
    通过 delta = RF_norm - F_norm 识别两个互补家族。

    Parameters
    ----------
    feat_names : list of str
    f_vals : F-test 值数组 (binary mel vs nv)
    rf_importance_dict : {feature_name: importance}

    Returns
    -------
    structural_feats : list of str — delta 最负的特征 (F 占优, 结构/纹理)
    color_feats : list of str — delta 最正的特征 (RF 占优, 颜色)
    delta_df : pd.DataFrame with columns [feature, F_norm, RF_norm, delta, family]
    """
    n = len(feat_names)
    f_norm = _normalize_to_01(f_vals)

    rf_vals = np.array([rf_importance_dict.get(name, 0.0) for name in feat_names])
    rf_norm = _normalize_to_01(rf_vals)

    delta = rf_norm - f_norm
    delta_df = pd.DataFrame({
        "feature": feat_names,
        "F_norm": f_norm,
        "RF_norm": rf_norm,
        "delta": delta,
    })

    # 正 delta → Color 族, 负 delta → Structural 族
    delta_df["family"] = np.where(delta > 0, "Color", "Structural")

    return delta_df


def generate_layer2_interactions(X_full, y_mel_nv,
                                  f_ranking_path=None,
                                  rf_importance_path=None,
                                  n_structural=15, n_color=15,
                                  interaction_types=("product", "ratio"),
                                  max_keep=40,
                                  random_state=42,
                                  logger=None):
    """
    为 Layer 2 (mel vs nv) 生成跨族交互特征。

    Parameters
    ----------
    X_full : pd.DataFrame — mel+nv 样本的完整特征矩阵
    y_mel_nv : array-like — mel vs nv 的二分类标签 (0=nv, 1=mel)
    f_ranking_path : str or None — 预计算的 Layer 2 F-test ranking CSV
    rf_importance_path : str or None — 预计算的 Layer 2 RF importance CSV
    n_structural : top-M Structural family features
    n_color : top-N Color family features
    interaction_types : ("product", "ratio", "squared")
    max_keep : post-selection max interaction features
    random_state : int

    Returns
    -------
    X_interact : pd.DataFrame — 筛选后的交互特征矩阵
    report : dict — 含 family_assignment, selected_pairs, post_mi_report
    """
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier

    feature_names = X_full.columns.tolist()
    X_arr = X_full.values

    if logger:
        logger.info(f"  Identifying complementary families "
                    f"(M={n_structural}, N={n_color})")

    # Step 1: Compute per-feature F-test (binary mel/nv)
    f_vals, _ = f_classif(X_arr, y_mel_nv)

    # Load or compute RF importance
    if rf_importance_path and Path(rf_importance_path).exists():
        rf_df = pd.read_csv(rf_importance_path)
        if "importance" in rf_df.columns and "feature" in rf_df.columns:
            rf_importance_dict = dict(zip(rf_df["feature"], rf_df["importance"]))
        else:
            # Old format: unnamed index column + importance
            rf_importance_dict = {}
            for _, row in rf_df.iterrows():
                feat = row.get("feature", row.get(row.index[0], ""))
                imp = row.get("importance", 0)
                if feat and feat != "importance":
                    rf_importance_dict[str(feat)] = float(imp)
    else:
        rf = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced",
            random_state=random_state,
        )
        rf.fit(X_arr, y_mel_nv)
        rf_importance_dict = dict(zip(feature_names, rf.feature_importances_))

    # Step 2: Identify complementary families
    delta_df = identify_complementary_families(feature_names, f_vals, rf_importance_dict)

    structural_candidates = (delta_df
                             .sort_values("delta")
                             .head(n_structural)["feature"]
                             .tolist())
    color_candidates = (delta_df
                        .sort_values("delta", ascending=False)
                        .head(n_color)["feature"]
                        .tolist())

    if logger:
        logger.info(f"  Structural family (delta most negative): {structural_candidates[:5]}...")
        logger.info(f"  Color family (delta most positive): {color_candidates[:5]}...")

    # Step 3: Generate cross-family interactions
    rows = X_full.shape[0]
    interaction_cols = []
    interaction_pairs = []
    interaction_data = {}

    eps = 1e-8
    for s_feat in structural_candidates:
        s_vals = X_full[s_feat].values.astype(np.float64)
        s_std = np.std(s_vals)
        for c_feat in color_candidates:
            c_vals = X_full[c_feat].values.astype(np.float64)
            c_std = np.std(c_vals)

            if "product" in interaction_types:
                name = f"inter_{s_feat}_x_{c_feat}"
                vals = s_vals * c_vals
                if np.std(vals) > 1e-12:
                    interaction_data[name] = vals
                    interaction_cols.append(name)
                    interaction_pairs.append((s_feat, c_feat, "product"))

            if "ratio" in interaction_types and c_std > 1e-8:
                name = f"inter_{s_feat}_div_{c_feat}"
                vals = s_vals / (c_vals + eps * np.mean(np.abs(c_vals)))
                if np.std(vals) > 1e-12:
                    interaction_data[name] = vals
                    interaction_cols.append(name)
                    interaction_pairs.append((s_feat, c_feat, "ratio"))

            if "difference" in interaction_types:
                name = f"inter_{s_feat}_sub_{c_feat}"
                vals = s_vals - c_vals
                if np.std(vals) > 1e-12:
                    interaction_data[name] = vals
                    interaction_cols.append(name)
                    interaction_pairs.append((s_feat, c_feat, "difference"))

    if logger:
        logger.info(f"  Generated {len(interaction_cols)} raw interaction features")

    if len(interaction_cols) == 0:
        return pd.DataFrame(index=X_full.index), {
            "family_assignment": delta_df,
            "selected_pairs": [],
            "n_raw": 0,
            "n_selected": 0,
        }

    X_interact_raw = pd.DataFrame(interaction_data, index=X_full.index)

    # Step 4: Remove constant/near-constant
    stds = X_interact_raw.std()
    valid = stds[stds > 1e-8].index.tolist()
    X_interact_raw = X_interact_raw[valid]

    # Step 5: Remove highly correlated (>0.95)
    corr = X_interact_raw.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X_interact_dedup = X_interact_raw.drop(columns=to_drop)

    if logger:
        logger.info(f"  After dedup: {X_interact_dedup.shape[1]} (dropped {len(to_drop)})")

    # Step 6: Mutual information post-selection
    n_keep = min(max_keep, X_interact_dedup.shape[1])
    mi_scores = mutual_info_classif(
        X_interact_dedup.values, y_mel_nv,
        random_state=random_state,
    )
    mi_df = pd.DataFrame({
        "feature": X_interact_dedup.columns,
        "mutual_info": mi_scores,
    }).sort_values("mutual_info", ascending=False)

    selected = mi_df.head(n_keep)["feature"].tolist()
    X_interact = X_interact_dedup[selected]

    # 统计跨族对占比
    cross_family_count = sum(
        1 for f in selected if f.startswith("inter_")
    )
    cross_ratio = cross_family_count / len(selected) if selected else 0

    if logger:
        logger.info(f"  Final: {len(selected)} interaction features selected")
        logger.info(f"  Cross-family ratio: {cross_ratio:.1%}")
        top3 = mi_df.head(3)
        for _, r in top3.iterrows():
            logger.info(f"    {r['feature'][:80]}: MI={r['mutual_info']:.4f}")

    report = {
        "family_assignment": delta_df,
        "selected_pairs": interaction_pairs,
        "n_raw": len(interaction_cols),
        "n_dedup": X_interact_dedup.shape[1],
        "n_selected": len(selected),
        "cross_family_ratio": cross_ratio,
        "mi_ranking": mi_df,
        "structural_family": structural_candidates,
        "color_family": color_candidates,
    }

    return X_interact, report


def apply_interaction_pairs(X, selected_pairs):
    """
    将预选的交互特征对应用到新数据 (用于 CV test fold)。

    Parameters
    ----------
    X : pd.DataFrame — 特征矩阵
    selected_pairs : list of (s_feat, c_feat, op_type) tuples
        op_type ∈ {"product", "ratio", "difference"}

    Returns
    -------
    X_interact : pd.DataFrame — 交互特征矩阵, 列名与训练时一致
    """
    eps = 1e-8
    interaction_data = {}
    for s_feat, c_feat, op_type in selected_pairs:
        s_vals = X[s_feat].values.astype(np.float64)
        c_vals = X[c_feat].values.astype(np.float64)

        if op_type == "product":
            name = f"inter_{s_feat}_x_{c_feat}"
            interaction_data[name] = s_vals * c_vals
        elif op_type == "ratio":
            name = f"inter_{s_feat}_div_{c_feat}"
            interaction_data[name] = s_vals / (c_vals + eps * np.mean(np.abs(c_vals)))
        elif op_type == "difference":
            name = f"inter_{s_feat}_sub_{c_feat}"
            interaction_data[name] = s_vals - c_vals

    return pd.DataFrame(interaction_data, index=X.index)
