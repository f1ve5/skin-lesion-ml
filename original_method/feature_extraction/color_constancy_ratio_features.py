"""
颜色恒常性比率特征 (Color Constancy Ratios)
=============================================
提取病灶各子区域相对于周围皮肤的通道比值，增强对光照和肤色的不变性。

文献依据：
  - Stanley 2007: relative color approach
  - Barata 2019: 颜色恒常性在皮肤镜分析中的重要性
  - 课程要求: 新增 ccr_* 比率特征

算法：
  对 RGB / HSV-V / Lab-L 的 inner50 和 border 区域，
  计算 mean(region) / mean(outer)，得到 6 维比率特征。
"""
import numpy as np
import cv2
from typing import Dict


def _safe_ratio(num: float, denom: float, eps: float = 1e-6) -> float:
    if abs(denom) < eps:
        return 0.0
    return float(num / denom)


def extract_color_constancy_ratio_features(
        img_bgr: np.ndarray,
        regions: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    提取颜色恒常性比率特征。

    对每个色彩通道:
      - inner50 / outer 比率
      - border / outer 比率

    色彩空间覆盖:
      - RGB: R, G, B 三通道
      - HSV: 仅 V (亮度)
      - Lab: 仅 L (亮度)
    总计 3 + 1 + 1 = 5 通道 × 2 区域 = 10 维 → 实际取 6 维
    (RGB 三通道的 inner50 和 border 各 3 + HSV/Lab 的 V/L 各 2 = 10 维)
    """
    feats = {}
    inner50 = regions.get("inner50")
    border = regions.get("border")
    outer = regions.get("outer")

    if inner50 is None or border is None or outer is None:
        # 退化处理
        for key in ["ccr_inner50_R", "ccr_inner50_G", "ccr_inner50_B",
                     "ccr_border_R", "ccr_border_G", "ccr_border_B",
                     "ccr_inner50_HSV_V", "ccr_border_HSV_V",
                     "ccr_inner50_Lab_L", "ccr_border_Lab_L"]:
            feats[key] = 0.0
        return feats

    has_outer = outer.sum() > 0
    has_inner = inner50.sum() > 0
    has_border = border.sum() > 0

    # --- RGB 三通道 ---
    rgb = img_bgr.astype(np.float64)  # BGR order
    outer_mean = rgb[outer].mean(axis=0) if has_outer else np.zeros(3)

    rgb_ch_names = [("ccr_inner50_B", "ccr_border_B"),
                    ("ccr_inner50_G", "ccr_border_G"),
                    ("ccr_inner50_R", "ccr_border_R")]
    for c in range(3):
        if has_inner:
            inner_mean = rgb[inner50][:, c].mean()
            feats[rgb_ch_names[c][0]] = _safe_ratio(inner_mean, outer_mean[c])
        else:
            feats[rgb_ch_names[c][0]] = 0.0
        if has_border:
            border_mean = rgb[border][:, c].mean()
            feats[rgb_ch_names[c][1]] = _safe_ratio(border_mean, outer_mean[c])
        else:
            feats[rgb_ch_names[c][1]] = 0.0

    # --- HSV V 通道 ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    v_ch = hsv[:, :, 2]
    outer_v_mean = v_ch[outer].mean() if has_outer else 0.0
    if has_inner:
        feats["ccr_inner50_HSV_V"] = _safe_ratio(v_ch[inner50].mean(), outer_v_mean)
    else:
        feats["ccr_inner50_HSV_V"] = 0.0
    if has_border:
        feats["ccr_border_HSV_V"] = _safe_ratio(v_ch[border].mean(), outer_v_mean)
    else:
        feats["ccr_border_HSV_V"] = 0.0

    # --- Lab L 通道 ---
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    l_ch = lab[:, :, 0]
    outer_l_mean = l_ch[outer].mean() if has_outer else 0.0
    if has_inner:
        feats["ccr_inner50_Lab_L"] = _safe_ratio(l_ch[inner50].mean(), outer_l_mean)
    else:
        feats["ccr_inner50_Lab_L"] = 0.0
    if has_border:
        feats["ccr_border_Lab_L"] = _safe_ratio(l_ch[border].mean(), outer_l_mean)
    else:
        feats["ccr_border_Lab_L"] = 0.0

    return feats
