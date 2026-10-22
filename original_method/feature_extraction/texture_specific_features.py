"""
纹理特异性增强特征
=====================
三个针对性检测器，弥补通用纹理特征 (GLCM/LBP/Gabor/Wavelet/Spectral)
对特定皮肤镜结构表征不足的缺陷。

1. blue_whitish_veil_ratio  — 蓝白纱 (Blue-Whitish Veil) 检测
2. dark_pigment_network_density — 色素网络 (Pigment Network) 密度
3. streak_orientation_coherence — 放射状条纹 (Radial Streaks) 方向一致性

文献依据：
  - Argenziano et al. 2003: 皮肤镜诊断结构的定义
  - Celebi 2007: 蓝白纱在 Lab 空间 b 通道的表现
  - Barata 2019 综述 §III-D: dermoscopic structures 的检测方法
"""
import numpy as np
import cv2
from scipy import stats as sp_stats


def blue_whitish_veil_ratio(img_bgr: np.ndarray,
                            mask: np.ndarray,
                            inner50_mask: np.ndarray = None) -> dict:
    """
    蓝白纱检测：在病灶核心区 (inner50) 中，Lab b 通道 < -10 的像素占比。

    原理：
      - 蓝白纱 (blue-whitish veil) 是黑色素瘤的关键诊断特征，表现为
        病灶深部的蓝灰色调，在 CIE Lab 空间中 b < -10 对应蓝色调。
      - 只统计核心区 (inner50) 是因为蓝白纱出现在病灶最深部。

    特征名: specific_blue_veil_ratio
    """
    if inner50_mask is not None and inner50_mask.sum() > 0:
        roi_mask = inner50_mask & (mask > 0)
    else:
        roi_mask = mask > 0

    if roi_mask.sum() == 0:
        return {"specific_blue_veil_ratio": 0.0}

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    b_channel = lab[:, :, 1]  # Lab 的 b 通道 (黄蓝轴)
    b_roi = b_channel[roi_mask]
    ratio = float((b_roi < -10).mean())
    return {"specific_blue_veil_ratio": ratio}


def dark_pigment_network_density(img_bgr: np.ndarray,
                                 mask: np.ndarray) -> dict:
    """
    色素网络密度检测：
      灰度图 → 黑帽运算 (kernel 17) → Otsu 二值化 → 计算前景像素占比。

    原理：
      - 色素网络 (pigment network) 表现为较周围皮肤略深的网状结构，
        黑帽 (blackhat) 形态学运算正好提取"比周围暗的细结构"。
      - 核大小 17 对应 256×256 图上约 0.4mm 的典型网络网格尺寸。

    特征名: specific_pigment_network_density
    """
    if mask.sum() == 0:
        return {"specific_pigment_network_density": 0.0}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_roi = gray.copy()
    gray_roi[mask == 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray_roi, cv2.MORPH_BLACKHAT, kernel)

    # Otsu 二值化黑帽结果
    bh_roi = blackhat[mask > 0]
    if bh_roi.size == 0:
        return {"specific_pigment_network_density": 0.0}

    _, binary = cv2.threshold(blackhat, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground_ratio = float((binary[mask > 0] > 0).mean())
    return {"specific_pigment_network_density": foreground_ratio}


def streak_orientation_coherence(img_bgr: np.ndarray,
                                  mask: np.ndarray,
                                  frequencies=(0.1, 0.3, 0.5),
                                  thetas_deg=(0, 45, 90, 135)) -> dict:
    """
    放射状条纹方向一致性检测：
      1. 在多个 Gabor 滤波器响应中，找到每个像素响应最强的方向
      2. 对强响应区域做阈值分割，提取连通域
      3. 计算各连通域主轴方向直方图（12 bins）的熵
      4. 低熵 → 方向高度一致（典型条纹特征）

    特征名: specific_streak_coherence
    """
    if mask.sum() == 0:
        return {"specific_streak_coherence": 1.0}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    from skimage.filters import gabor

    responses = []
    for f in frequencies:
        for theta in thetas_deg:
            real, _ = gabor(gray, frequency=f, theta=np.deg2rad(theta))
            responses.append(np.abs(real))

    if not responses:
        return {"specific_streak_coherence": 1.0}

    # 逐像素找最强响应方向
    resp_stack = np.stack(responses, axis=0)  # [n_filters, H, W]
    best_dir = np.argmax(resp_stack, axis=0)  # [H, W]
    best_magnitude = np.max(resp_stack, axis=0)

    # 阈值：取 top 30% 的强响应区域
    roi_mag = best_magnitude[mask > 0]
    if roi_mag.size < 10:
        return {"specific_streak_coherence": 1.0}
    thresh = np.percentile(roi_mag, 70)

    strong_mask = (best_magnitude >= thresh) & (mask > 0)
    if strong_mask.sum() < 20:
        return {"specific_streak_coherence": 1.0}

    # 连通域分析
    num_labels, labels = cv2.connectedComponents(
        strong_mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return {"specific_streak_coherence": 1.0}

    # 对每个连通域（除背景外）计算主轴方向
    orientations = []
    for lbl in range(1, num_labels):
        comp_mask = (labels == lbl)
        if comp_mask.sum() < 10:
            continue
        ys, xs = np.where(comp_mask)
        if len(xs) < 3:
            continue
        # 用 PCA 求主轴方向
        cx, cy = xs.mean(), ys.mean()
        dx = xs - cx
        dy = ys - cy
        cov_xx = (dx * dx).sum()
        cov_yy = (dy * dy).sum()
        cov_xy = (dx * dy).sum()
        if abs(cov_xx - cov_yy) < 1e-8 and abs(cov_xy) < 1e-8:
            continue
        angle = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)
        orientations.append(angle)

    if len(orientations) < 3:
        return {"specific_streak_coherence": 1.0}

    # 12 bin 角度直方图，计算熵
    hist, _ = np.histogram(orientations, bins=12, range=(-np.pi/2, np.pi/2))
    prob = hist.astype(np.float64) / (hist.sum() + 1e-12)
    entropy = float(-(prob[prob > 0] * np.log2(prob[prob > 0])).sum())
    max_entropy = np.log2(12.0)  # 均匀分布时熵最大 ≈ 3.58

    # 归一化：低熵=高一致性，映射到 [0, 1]，1 表示完全一致
    coherence = 1.0 - min(entropy / (max_entropy + 1e-12), 1.0)
    return {"specific_streak_coherence": coherence}


def extract_texture_specific_features(img_bgr: np.ndarray,
                                       mask: np.ndarray,
                                       regions: dict = None) -> dict:
    """
    一站式提取全部纹理特异性特征。
    若提供 regions (含 inner50)，则 blue_veil 使用核心区。
    """
    feats = {}
    inner50 = regions.get("inner50") if regions else None

    feats.update(blue_whitish_veil_ratio(img_bgr, mask, inner50))
    feats.update(dark_pigment_network_density(img_bgr, mask))
    feats.update(streak_orientation_coherence(img_bgr, mask))
    return feats
