"""
一阶/直方图特征模块 - 完全覆盖课件 First Order Feature 章节

课件涉及：
  - Energy
  - Mean, Median
  - Max, Min, Range
  - Standard deviation, Variance
  - Skewness, Kurtosis
  - Uniformity, Entropy
  - Smoothness R = 1 - 1/(1 + σ²)

我额外加入：
  - Percentiles (10/25/50/75/90)
  - Root Mean Squared
  - 多色彩空间（RGB, HSV, Lab, YCrCb 共 12 通道）
"""
import numpy as np
from scipy import stats as sp_stats


def _safe_stats_on_roi(channel: np.ndarray, mask: np.ndarray,
                       percentiles=(10, 25, 50, 75, 90)) -> dict:
    """对单通道单ROI计算完整一阶特征"""
    pixels = channel[mask > 0].astype(np.float64)
    feat = {}
    if pixels.size == 0:
        keys = ["energy", "mean", "median", "min", "max", "range",
                "std", "var", "skew", "kurt", "uniformity", "entropy",
                "smoothness", "rms", "mad"]
        for k in keys:
            feat[k] = 0.0
        for p in percentiles:
            feat[f"p{p}"] = 0.0
        return feat

    # 基础统计
    feat["energy"]    = float(np.sum(pixels ** 2))
    feat["mean"]      = float(np.mean(pixels))
    feat["median"]    = float(np.median(pixels))
    feat["min"]       = float(np.min(pixels))
    feat["max"]       = float(np.max(pixels))
    feat["range"]     = feat["max"] - feat["min"]
    feat["std"]       = float(np.std(pixels))
    feat["var"]       = float(np.var(pixels))
    # 偏度峰度
    feat["skew"]      = float(sp_stats.skew(pixels)) if pixels.size > 2 else 0.0
    feat["kurt"]      = float(sp_stats.kurtosis(pixels)) if pixels.size > 3 else 0.0
    # 均匀性与熵 - 基于归一化直方图
    hist, _ = np.histogram(pixels, bins=64, range=(0, 256))
    p = hist / (hist.sum() + 1e-12)
    feat["uniformity"] = float(np.sum(p ** 2))
    p_nz = p[p > 0]
    feat["entropy"]   = float(-(p_nz * np.log2(p_nz)).sum()) if p_nz.size else 0.0
    # 平滑度 R
    var_norm = feat["var"] / (255.0 ** 2)
    feat["smoothness"] = float(1 - 1 / (1 + var_norm))
    # RMS
    feat["rms"]       = float(np.sqrt(np.mean(pixels ** 2)))
    # 平均绝对偏差
    feat["mad"]       = float(np.mean(np.abs(pixels - feat["mean"])))
    # 百分位数
    for p_val in percentiles:
        feat[f"p{p_val}"] = float(np.percentile(pixels, p_val))
    return feat


def extract_intensity_features(color_spaces: dict,
                               mask: np.ndarray,
                               channels_per_space: dict = None,
                               percentiles=(10, 25, 50, 75, 90)) -> dict:
    """
    在 ROI 内对多色彩空间的每个通道提取一阶特征
    
    参数:
        color_spaces: {'RGB': arr, 'HSV': arr, ...}
        mask: 二值mask
        channels_per_space: 指定每个色彩空间的通道命名
    """
    if channels_per_space is None:
        channels_per_space = {
            "RGB":   ["R", "G", "B"],
            "HSV":   ["H", "S", "V"],
            "Lab":   ["L", "a", "b"],
            "YCrCb": ["Y", "Cr", "Cb"],
        }
    out = {}
    for space, img in color_spaces.items():
        names = channels_per_space.get(space, [f"c{i}" for i in range(img.shape[2])])
        for ci, name in enumerate(names):
            ch = img[:, :, ci]
            stat = _safe_stats_on_roi(ch, mask, percentiles)
            for k, v in stat.items():
                out[f"int_{space}_{name}_{k}"] = v
    return out


# ============================================================
# 灰度图的一阶特征（也单独提取一组，方便后续与GLCM对照）
# ============================================================
def extract_gray_intensity_features(img_bgr: np.ndarray,
                                     mask: np.ndarray,
                                     percentiles=(10, 25, 50, 75, 90)) -> dict:
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    stat = _safe_stats_on_roi(gray, mask, percentiles)
    return {f"int_gray_{k}": v for k, v in stat.items()}
