"""
临床错误诊断与可视化
=====================
找出 Layer 2 中高置信度恶性漏检案例 (true=mel, pred=nv, P(nv)>0.7)，
生成诊断组合图并输出汇总。

文献依据：
  - 课程要求: 临床错误分析
  - Dolianitis 2005: mel/nv 鉴别是临床公认最难任务
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from .viz_features import setup_style


def find_misclassified_mel(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           proba: np.ndarray = None,
                           class_names=None,
                           confidence_threshold: float = 0.7) -> pd.DataFrame:
    """
    找出高置信度恶性漏检案例。

    Parameters
    ----------
    y_true : 真实标签
    y_pred : 预测标签
    proba : 预测概率矩阵 [n_samples, n_classes], 或 None
    class_names : 类别名列表
    confidence_threshold : 置信度阈值

    Returns
    -------
    DataFrame with columns: index, true_label, pred_label, confidence, probability
    """
    if class_names is None:
        class_names = sorted(set(y_true) | set(y_pred))

    rows = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == "mel" and p == "nv":
            if proba is not None:
                # 找 nv 的列索引
                nv_idx = list(class_names).index("nv") if "nv" in class_names else 1
                nv_conf = proba[i, nv_idx]
            else:
                nv_conf = 0.5

            if nv_conf >= confidence_threshold:
                rows.append({
                    "global_idx": i,
                    "true_label": t,
                    "pred_label": p,
                    "confidence_nv": nv_conf,
                    "probability": proba[i] if proba is not None else None,
                })

    return pd.DataFrame(rows)


def create_error_diagnostic_image(image_path: str,
                                   mask_path: str,
                                   sample_features: dict,
                                   feature_stats: dict,
                                   top_features: list,
                                   save_path: str,
                                   sample_id: str = ""):
    """
    为单个漏检案例创建诊断组合图。

    左侧: 原图 + 掩膜叠加
    右侧: 特征数值对比表格，高亮偏离 mel 分布的特征
    """
    setup_style()

    fig = plt.figure(figsize=(16, 10))

    # ---- 左侧：原图 + 掩膜叠加 ----
    ax_img = fig.add_subplot(1, 2, 1)

    # 处理可能的空路径或无效路径
    img = None
    if image_path and Path(image_path).exists():
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None

    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_img.imshow(img_rgb)
    else:
        # 占位图
        ax_img.text(0.5, 0.5, f"Image not available\n{sample_id}",
                   transform=ax_img.transAxes, ha="center", va="center",
                   fontsize=10, color="gray")
        ax_img.set_xlim(0, 1); ax_img.set_ylim(0, 1)

    mask = None
    if mask_path and Path(mask_path).exists():
        try:
            mask_data = np.fromfile(mask_path, dtype=np.uint8)
            mask = cv2.imdecode(mask_data, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        except Exception:
            mask = None

    if img is None and mask is None:
        pass  # already handled
    elif mask is not None and img is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            ax_img.plot(cnt[:, 0, 0], cnt[:, 0, 1], 'r-', linewidth=1.5)

    ax_img.set_title(f"Sample {sample_id}\nTrue: mel -> Predicted: nv",
                     fontsize=12, fontweight="bold", color="red")
    ax_img.axis("off")

    # ---- 右侧：特征值对比表 ----
    ax_table = fig.add_subplot(1, 2, 2)
    ax_table.axis("off")

    # 选取 top_features 中有数据的特征
    available = [f for f in top_features if f in sample_features
                 and f in feature_stats.get("mel", {})]

    if len(available) > 15:
        available = available[:15]

    # 构建表格数据
    table_data = []
    cell_colors = []
    anomaly_features = []

    for feat in available:
        val = sample_features.get(feat, np.nan)
        mel_mean = feature_stats.get("mel", {}).get(feat, {}).get("mean", np.nan)
        mel_std = feature_stats.get("mel", {}).get(feat, {}).get("std", np.nan)
        nv_mean = feature_stats.get("nv", {}).get(feat, {}).get("mean", np.nan)
        nv_std = feature_stats.get("nv", {}).get(feat, {}).get("std", np.nan)

        # 判断是否偏离 mel 分布
        is_anomaly = False
        if not np.isnan(val) and not np.isnan(mel_mean) and not np.isnan(mel_std) and mel_std > 1e-8:
            z_score = abs(val - mel_mean) / mel_std
            if z_score > 1.5:
                is_anomaly = True
                anomaly_features.append(feat)

        table_data.append([
            f"{feat[:40]}",
            f"{val:.3f}" if not np.isnan(val) else "N/A",
            f"{mel_mean:.3f}±{mel_std:.3f}" if not np.isnan(mel_mean) else "N/A",
            f"{nv_mean:.3f}±{nv_std:.3f}" if not np.isnan(nv_mean) else "N/A",
            "⚠" if is_anomaly else "✓",
        ])
        cell_colors.append(
            ["#FFE4E1" if is_anomaly else "#F5F5F5",
             "#FFE4E1" if is_anomaly else "#FFFFFF",
             "#E8F4FD", "#E8FDE8", "#FFE4E1" if is_anomaly else "#F5F5F5"]
        )

    # 画表格
    if len(table_data) > 0 and len(table_data) == len(cell_colors):
        col_labels = ["Feature", "Value", "mel (mean+/-std)", "nv (mean+/-std)", "Flag"]
        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellColours=cell_colors,
            loc="center",
            cellLoc="left",
        )
    else:
        ax_table.text(0.5, 0.5, "No matching features found",
                     transform=ax_table.transAxes, ha="center", va="center")
        table = None
    if table is not None:
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1.0, 1.15)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_fontsize(7)
                cell.set_text_props(weight="bold")

    ax_table.set_title(
        f"Feature Comparison\n"
        f"Anomalous features (|z|>1.5 from mel distribution): {len(anomaly_features)}",
        fontsize=11, fontweight="bold",
    )

    plt.suptitle("Error Diagnosis: High-Confidence Melanoma Miss",
                 fontsize=14, fontweight="bold", color="darkred")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return anomaly_features


def run_error_analysis(y_true, y_pred, proba, X_features,
                       dataset_df, top_features,
                       output_dir, class_names=None,
                       feature_stats=None,
                       max_cases=10,
                       logger=None) -> dict:
    """
    运行完整的错误分析流程。

    Parameters
    ----------
    y_true, y_pred : 真实/预测标签
    proba : 预测概率矩阵
    X_features : 特征 DataFrame
    dataset_df : 原始数据集索引 (含 image_path, mask_path, image_id)
    top_features : SHAP Top-10 特征名列表
    output_dir : 输出目录
    class_names : 类别名
    feature_stats : {class_name: {feat: {"mean": ..., "std": ...}}}
    max_cases : 最多显示多少案例

    Returns
    -------
    dict with keys: n_misclassified, anomaly_summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = np.unique(np.concatenate([y_true, y_pred]))

    # 找全部 mel→nv 错误, 按 nv 置信度降序排列 (最确信的错误排前面)
    misclassified = find_misclassified_mel(
        y_true, y_pred, proba, class_names,
        confidence_threshold=0.0,  # 不限置信度, 纳入所有 mel→nv 错误
    )

    n_total = len(misclassified)
    if logger:
        logger.info(f"Found {n_total} total mel->nv errors")
        logger.info(f"  y_true length: {len(y_true)}, dataset_df length: {len(dataset_df)}")

    # 按置信度降序排列, 最确信的错误排前面 (优先诊断)
    misclassified = misclassified.sort_values("confidence_nv", ascending=False).reset_index(drop=True)

    if n_total == 0:
        return {"n_misclassified": 0, "anomaly_summary": {}, "misclassified_df": misclassified}

    # 计算各类别特征统计
    if feature_stats is None:
        feature_stats = {}
        for cls in class_names:
            mask = np.array(y_true) == cls
            if mask.sum() > 0:
                cls_X = X_features.iloc[mask] if isinstance(X_features, pd.DataFrame) else X_features[mask]
                feature_stats[cls] = {}
                for feat in top_features:
                    if feat in cls_X.columns:
                        vals = cls_X[feat].values
                        feature_stats[cls][feat] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                        }

    # 生成每个案例的诊断图
    all_anomalies = []
    n_cases = min(len(misclassified), max_cases)

    for i in range(n_cases):
        row = misclassified.iloc[i]
        global_idx = int(row["global_idx"])

        # 获取图像路径（带边界检查）
        if global_idx < len(dataset_df):
            ds_row = dataset_df.iloc[global_idx]
        else:
            ds_row = {}
        image_path = ds_row.get("image_path", "")
        mask_path = ds_row.get("mask_path", "")
        sample_id = ds_row.get("image_id", str(global_idx))
        aug_tag = ds_row.get("aug_tag", "")

        # 获取该样本的特征值
        if isinstance(X_features, pd.DataFrame):
            sample_feats = X_features.iloc[global_idx].to_dict()
        else:
            sample_feats = {f: X_features[global_idx, j]
                           for j, f in enumerate(top_features)}

        save_path = output_dir / f"error_case_{i+1:02d}_{sample_id}_{aug_tag}.png"
        anomalies = create_error_diagnostic_image(
            image_path, mask_path, sample_feats,
            feature_stats, top_features, str(save_path),
            sample_id=f"{sample_id}_{aug_tag}",
        )
        all_anomalies.extend(anomalies)

    # 汇总
    from collections import Counter
    anomaly_counts = Counter(all_anomalies)
    anomaly_summary = anomaly_counts.most_common(10)

    if logger:
        logger.info(f"\n{'='*60}")
        logger.info(f"Error Analysis Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total high-confidence mel→nv: {n_total}")
        logger.info(f"Analyzed {n_cases} cases")
        logger.info(f"\nMost common anomalous features:")
        for feat, count in anomaly_summary:
            logger.info(f"  {feat}: {count} cases")
        logger.info(f"\nDiagnostic images saved to: {output_dir}")
        logger.info(f"{'='*60}")

    return {
        "n_misclassified": n_total,
        "anomaly_summary": dict(anomaly_summary),
        "misclassified_df": misclassified,
        "feature_stats": feature_stats,
    }
