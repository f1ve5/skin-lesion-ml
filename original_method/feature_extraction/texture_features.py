"""
纹理特征模块 - 完全覆盖课件 Texture Feature 章节

课件涉及：
  - GLCM (Co-occurrence matrix)
      * Maximum probability
      * Correlation
      * Contrast
      * Uniformity (Energy / ASM)
      * Homogeneity
      * Entropy

我额外加入（让结果更有亮点）：
  - 多距离 [1,2,3] 多方向 [0/45/90/135] 的GLCM
  - LBP (Local Binary Patterns) - 旋转不变
  - Gabor 滤波响应统计
  - 小波分解能量（Daubechies-4，2级）
  - 频谱纹理 S(r), S(θ) （课件 slide 43-45）
"""
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
import pywt
from scipy import stats as sp_stats


# ============================================================
# GLCM特征（课件 slide 37-42）
# ============================================================
def quantize_gray(gray: np.ndarray, levels: int = 32) -> np.ndarray:
    """灰度量化到指定级数，减小GLCM尺寸"""
    g = gray.astype(np.float64)
    g = (g / 256.0) * levels
    g = np.clip(g.astype(np.int32), 0, levels - 1)
    return g.astype(np.uint8)


def extract_glcm_features(img_bgr: np.ndarray,
                          mask: np.ndarray,
                          levels: int = 32,
                          distances=(1, 2, 3),
                          angles_deg=(0, 45, 90, 135),
                          props=('contrast', 'correlation', 'energy',
                                 'homogeneity', 'dissimilarity', 'ASM')) -> dict:
    """
    在ROI内提取多距离多方向GLCM特征
    每个 (distance, angle, prop) 是一个独立特征；额外计算各方向均值/极差以增加旋转鲁棒性
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    qgray = quantize_gray(gray, levels)
    # ROI外置0：使用 mask 把外面像素置为0级，但同时只在ROI内统计GLCM
    qgray_roi = qgray.copy()
    qgray_roi[mask == 0] = 0  # 边缘也算0级
    # 转弧度
    angles_rad = [np.deg2rad(a) for a in angles_deg]
    # skimage的GLCM一次性算出所有距离x角度
    glcm = graycomatrix(qgray_roi, distances=list(distances),
                         angles=angles_rad,
                         levels=levels, symmetric=True, normed=True)
    # 形状: [levels, levels, n_dist, n_angle]
    feat = {}
    for prop in props:
        # 兼容 skimage 不同版本: 0.19+ 支持 ASM, 老版本叫 'ASM'
        if prop == 'ASM':
            try:
                v = graycoprops(glcm, prop='ASM')
            except (ValueError, KeyError):
                v = graycoprops(glcm, prop='energy') ** 2
        else:
            v = graycoprops(glcm, prop=prop)  # shape: [n_dist, n_angle]
        for di, d in enumerate(distances):
            for ai, a in enumerate(angles_deg):
                feat[f"glcm_{prop}_d{d}_a{a}"] = float(v[di, ai])
            # 每个距离取多方向均值/极差 - 旋转鲁棒
            feat[f"glcm_{prop}_d{d}_mean"]  = float(v[di].mean())
            feat[f"glcm_{prop}_d{d}_range"] = float(v[di].max() - v[di].min())

    # 额外: 自己算 Maximum probability 和 Entropy（skimage的graycoprops 不直接给）
    # glcm shape [L, L, nd, na]
    L = glcm.shape[0]
    nd, na = glcm.shape[2], glcm.shape[3]
    for di, d in enumerate(distances):
        max_p = []
        ent = []
        for ai in range(na):
            P = glcm[:, :, di, ai]
            max_p.append(P.max())
            P_nz = P[P > 0]
            ent.append(-(P_nz * np.log2(P_nz)).sum())
        max_p = np.array(max_p); ent = np.array(ent)
        feat[f"glcm_maxprob_d{d}_mean"] = float(max_p.mean())
        feat[f"glcm_entropy_d{d}_mean"] = float(ent.mean())
    return feat


# ============================================================
# LBP特征（额外加入：旋转不变纹理）
# ============================================================
def extract_lbp_features(img_bgr: np.ndarray,
                         mask: np.ndarray,
                         radius: int = 3,
                         n_points: int = 24,
                         method: str = 'uniform') -> dict:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    # uniform LBP的bin数为 n_points + 2
    n_bins = int(lbp.max() + 1) if method != 'uniform' else (n_points + 2)
    lbp_roi = lbp[mask > 0]
    hist, _ = np.histogram(lbp_roi, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-12)
    feat = {}
    for i, v in enumerate(hist):
        feat[f"lbp_bin_{i}"] = float(v)
    # 直方图自身的统计量
    feat["lbp_entropy"] = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
    feat["lbp_energy"]  = float(np.sum(hist ** 2))
    return feat


# ============================================================
# Gabor滤波器响应（额外）
# ============================================================
def extract_gabor_features(img_bgr: np.ndarray,
                           mask: np.ndarray,
                           frequencies=(0.1, 0.3, 0.5),
                           thetas_deg=(0, 45, 90, 135)) -> dict:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    feat = {}
    for f in frequencies:
        for theta in thetas_deg:
            real, _ = gabor(gray, frequency=f, theta=np.deg2rad(theta))
            roi = real[mask > 0]
            if roi.size > 0:
                feat[f"gabor_f{f}_t{theta}_mean"] = float(roi.mean())
                feat[f"gabor_f{f}_t{theta}_std"]  = float(roi.std())
                feat[f"gabor_f{f}_t{theta}_eng"]  = float(np.mean(roi ** 2))
            else:
                feat[f"gabor_f{f}_t{theta}_mean"] = 0.0
                feat[f"gabor_f{f}_t{theta}_std"]  = 0.0
                feat[f"gabor_f{f}_t{theta}_eng"]  = 0.0
    return feat


# ============================================================
# 小波分解（额外）
# ============================================================
def extract_wavelet_features(img_bgr: np.ndarray,
                             mask: np.ndarray,
                             wavelet: str = 'db4',
                             level: int = 2) -> dict:
    """对 ROI 内灰度图做 2D 离散小波分解，提取各子带能量与熵"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # mask外置0，避免泄漏
    gray_roi = gray * (mask > 0)
    coeffs = pywt.wavedec2(gray_roi, wavelet, level=level)
    feat = {}
    # coeffs[0] = cA_n （近似系数）
    cA = coeffs[0]
    feat[f"wave_cA_eng"] = float(np.mean(cA ** 2))
    feat[f"wave_cA_std"] = float(np.std(cA))
    # 其它各级 (cH, cV, cD)
    for lvl, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
        for sub_name, sub in [("cH", cH), ("cV", cV), ("cD", cD)]:
            feat[f"wave_L{lvl}_{sub_name}_eng"] = float(np.mean(sub ** 2))
            feat[f"wave_L{lvl}_{sub_name}_std"] = float(np.std(sub))
    return feat


# ============================================================
# 频谱纹理 S(r), S(θ) - 课件 slide 43-45
# ============================================================
def extract_spectral_texture(img_bgr: np.ndarray,
                              mask: np.ndarray,
                              r_bins: int = 30,
                              theta_bins: int = 36) -> dict:
    """
    对ROI内灰度图做FFT，转极坐标后统计 S(r), S(θ) 的描述子
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # 把ROI外置为该ROI的均值以减少边界泄漏
    if mask.sum() > 0:
        mean_val = gray[mask > 0].mean()
        gray_proc = np.where(mask > 0, gray, mean_val)
    else:
        gray_proc = gray
    F = np.fft.fftshift(np.fft.fft2(gray_proc))
    S = np.abs(F) ** 2  # 功率谱
    H, W = S.shape
    cy, cx = H // 2, W // 2
    y, x = np.indices(S.shape)
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)
    theta_mod = (theta + np.pi) % np.pi  # 0..π
    R_max = min(cx, cy)
    # S(r)
    r_edges = np.linspace(0, R_max, r_bins + 1)
    sr = np.zeros(r_bins)
    for i in range(r_bins):
        sel = (r >= r_edges[i]) & (r < r_edges[i+1])
        if sel.any():
            sr[i] = S[sel].sum()
    # S(θ)
    t_edges = np.linspace(0, np.pi, theta_bins + 1)
    st = np.zeros(theta_bins)
    for i in range(theta_bins):
        sel = (theta_mod >= t_edges[i]) & (theta_mod < t_edges[i+1]) & (r < R_max)
        if sel.any():
            st[i] = S[sel].sum()

    # 归一化 + 统计描述
    feat = {}
    for name, vec in [("Sr", sr), ("Stheta", st)]:
        p = vec / (vec.sum() + 1e-12)
        feat[f"spec_{name}_mean"]    = float(vec.mean())
        feat[f"spec_{name}_std"]     = float(vec.std())
        feat[f"spec_{name}_skew"]    = float(sp_stats.skew(vec)) if vec.size > 2 else 0.0
        feat[f"spec_{name}_kurt"]    = float(sp_stats.kurtosis(vec)) if vec.size > 3 else 0.0
        feat[f"spec_{name}_entropy"] = float(-(p[p>0] * np.log2(p[p>0])).sum())
        feat[f"spec_{name}_peak"]    = float(np.argmax(vec))
    return feat


# ============================================================
# 主入口
# ============================================================
def extract_texture_features(img_bgr: np.ndarray,
                              mask: np.ndarray,
                              cfg) -> dict:
    feat = {}
    feat.update(extract_glcm_features(
        img_bgr, mask,
        levels=cfg.GLCM_LEVELS,
        distances=cfg.GLCM_DISTANCES,
        angles_deg=cfg.GLCM_ANGLES,
        props=cfg.GLCM_PROPS,
    ))
    feat.update(extract_lbp_features(
        img_bgr, mask,
        radius=cfg.LBP_RADIUS, n_points=cfg.LBP_N_POINTS,
        method=cfg.LBP_METHOD,
    ))
    feat.update(extract_gabor_features(
        img_bgr, mask,
        frequencies=cfg.GABOR_FREQUENCIES,
        thetas_deg=cfg.GABOR_THETAS,
    ))
    feat.update(extract_wavelet_features(
        img_bgr, mask,
        wavelet=cfg.WAVELET_NAME,
        level=cfg.WAVELET_LEVEL,
    ))
    feat.update(extract_spectral_texture(
        img_bgr, mask,
        r_bins=cfg.SPECTRAL_R_BINS,
        theta_bins=cfg.SPECTRAL_THETA_BINS,
    ))
    return feat
