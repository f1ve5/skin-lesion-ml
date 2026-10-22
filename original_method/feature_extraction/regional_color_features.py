"""
区域划分与相对色彩特征
============================
文献依据：
  - Celebi ME et al. A methodological approach to the classification of
    dermoscopy images. Comput Med Imaging Graph 2007; 31(6):362-373.
    用 Euclidean distance transform 把 lesion 分成内/外区域。
  - Stanley RJ, Stoecker WV, Moss RH. A relative color approach to color
    discrimination for malignant melanoma detection in dermoscopy images.
    Skin Res Technol 2007; 13: 62-72.
  - Barata 2019 综述 §III-C: 计算 relative color 可处理不同肤色和
    Tyndall effect.

提供两件事：
  A) 把 lesion mask 切成 inner_50% / inner_75% / border 三个子区域
  B) 用 lesion 周围皮肤的平均颜色作 reference，计算 lesion ROI 减去
     surrounding skin 的"相对颜色"，再提取一阶统计
"""
import numpy as np
import cv2
from typing import Dict


# ============================================================
# A. 用欧氏距离变换划分 inner / border
# ============================================================
def split_lesion_regions(mask: np.ndarray,
                          inner_ratio_50: float = 0.5,
                          inner_ratio_75: float = 0.75) -> Dict[str, np.ndarray]:
    """
    Celebi 2007 风格的区域划分。

    Distance Transform 给出每个 lesion 像素到 boundary 的距离，
    然后用相对距离阈值切分：
      inner50: 距离 ≥ 50% 最大距离的区域（"病变核心"）
      inner75: 距离 ≥ 25% 最大距离的区域（即最内 75% 半径）
      border : lesion 内但距离 < 25% 最大距离（"边界过渡区"）
      lesion : 整个病变（与原 mask 相同）
      outer  : 病变外、且在 lesion 一倍最大距离半径内的"周围皮肤"

    返回 dict[name] = bool mask (HxW)
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        H, W = mask.shape
        empty = np.zeros((H, W), dtype=bool)
        return {"lesion": empty, "inner50": empty, "inner75": empty,
                "border": empty, "outer": empty}

    # 1) 内部距离
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    max_d_in = dist_in.max()
    if max_d_in <= 0:
        max_d_in = 1.0

    inner50 = (dist_in >= inner_ratio_50 * max_d_in)
    inner75 = (dist_in >= (1 - inner_ratio_75) * max_d_in)
    # ↑ inner_ratio_75=0.75 表示"最内 75%"，所以阈值是 25% 最大距离
    border  = (m > 0) & (~inner75)

    # 2) 外部距离（病变周围一圈皮肤）
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    # outer band 取与最大内距相当的一圈，避免太远的皮肤
    outer = (dist_out > 0) & (dist_out <= max_d_in)

    return {
        "lesion":  m.astype(bool),
        "inner50": inner50,
        "inner75": inner75,
        "border":  border,
        "outer":   outer,
    }


# ============================================================
# B. 相对色彩特征
# ============================================================
def relative_color_image(img_bgr: np.ndarray,
                          outer_mask: np.ndarray) -> np.ndarray:
    """
    用 outer_mask（周围正常皮肤）的均值作 reference，
    返回 (img - reference) 的浮点图。
    Stanley 2007 风格：处理不同肤色 + Tyndall effect。
    """
    if outer_mask.sum() == 0:
        # fallback：用整张图减自己均值
        ref = img_bgr.reshape(-1, 3).mean(axis=0)
    else:
        ref = img_bgr[outer_mask].mean(axis=0)
    return img_bgr.astype(np.float64) - ref[np.newaxis, np.newaxis, :]


def _region_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """对一组数值算 8 个一阶统计"""
    feats = {}
    if values.size == 0:
        for k in ["mean", "std", "min", "max", "median", "skew", "kurt", "range"]:
            feats[f"{prefix}_{k}"] = 0.0
        return feats
    v = values.astype(np.float64)
    mean = v.mean()
    std = v.std()
    feats[f"{prefix}_mean"]   = float(mean)
    feats[f"{prefix}_std"]    = float(std)
    feats[f"{prefix}_min"]    = float(v.min())
    feats[f"{prefix}_max"]    = float(v.max())
    feats[f"{prefix}_median"] = float(np.median(v))
    feats[f"{prefix}_range"]  = float(v.max() - v.min())
    if std > 1e-8:
        feats[f"{prefix}_skew"] = float(((v - mean) ** 3).mean() / (std ** 3))
        feats[f"{prefix}_kurt"] = float(((v - mean) ** 4).mean() / (std ** 4) - 3)
    else:
        feats[f"{prefix}_skew"] = 0.0
        feats[f"{prefix}_kurt"] = 0.0
    return feats


def extract_regional_color_features(color_spaces: Dict[str, np.ndarray],
                                      regions: Dict[str, np.ndarray],
                                      space_names=("RGB", "HSV", "Lab", "YCrCb"),
                                      channel_idx=(0, 1, 2),
                                      region_names=("inner50", "border", "outer"),
                                      ) -> Dict[str, float]:
    """
    对每个色彩空间、每个通道、每个区域计算 8 项一阶统计。
    Celebi 2007 强调 inner / border 分别提取信息很关键。

    维度估算: 4(spaces) × 3(channels) × 3(regions) × 8(stats) = 288
    为避免维度爆炸，默认 region_names 只取 3 个；可减到 2 个。
    """
    feats = {}
    space_short = {"RGB": "RGB", "HSV": "HSV", "Lab": "Lab", "YCrCb": "YCrCb"}
    channel_names = {
        "RGB":   ["R", "G", "B"],
        "HSV":   ["H", "S", "V"],
        "Lab":   ["L", "a", "b"],
        "YCrCb": ["Y", "Cr", "Cb"],
    }
    for sp in space_names:
        img = color_spaces[sp]
        for c in channel_idx:
            ch = img[:, :, c]
            ch_name = channel_names[sp][c]
            for rg in region_names:
                rmask = regions[rg]
                if rmask.sum() == 0:
                    pixels = np.array([], dtype=np.float64)
                else:
                    pixels = ch[rmask]
                prefix = f"reg_{rg}_{space_short[sp]}_{ch_name}"
                feats.update(_region_stats(pixels, prefix))
    return feats


def extract_relative_color_features(img_bgr: np.ndarray,
                                      regions: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Stanley 2007 相对色彩：用 outer 皮肤作 ref，
    在 lesion ROI 上计算 (R-ref, G-ref, B-ref) 的统计。
    另外加 Lab、HSV 的相对值，更稳健。
    """
    feats = {}
    outer = regions["outer"]
    lesion = regions["lesion"]
    if lesion.sum() == 0:
        # 退化情形
        for sp in ["RGB", "Lab", "HSV"]:
            for ch in ["c0", "c1", "c2"]:
                for stat in ["mean", "std", "min", "max", "median",
                              "skew", "kurt", "range"]:
                    feats[f"rel_{sp}_{ch}_{stat}"] = 0.0
        return feats

    # RGB 相对图
    rel_rgb = relative_color_image(img_bgr, outer)
    for c, name in enumerate(["B", "G", "R"]):
        feats.update(_region_stats(rel_rgb[:, :, c][lesion], f"rel_RGB_{name}"))

    # Lab 相对图
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    rel_lab = relative_color_image(lab, outer)
    for c, name in enumerate(["L", "a", "b"]):
        feats.update(_region_stats(rel_lab[:, :, c][lesion], f"rel_Lab_{name}"))

    # HSV 相对图（仅 V 通道有意义；H 是角度，不适合直接减）
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    rel_v = relative_color_image(hsv[:, :, 2:3], outer)
    feats.update(_region_stats(rel_v[:, :, 0][lesion], "rel_HSV_V"))

    return feats


# ============================================================
# C. 区域边界对比度（Celebi 2007 风格 "border sharpness"）
# ============================================================
def extract_border_contrast_features(img_bgr: np.ndarray,
                                       regions: Dict[str, np.ndarray]
                                       ) -> Dict[str, float]:
    """
    边界 sharpness：lesion 边界过渡区(border) 和外圈皮肤(outer)
    之间的颜色对比度。临床上 melanoma 边界往往更模糊或更锐利。
    """
    feats = {}
    border = regions["border"]
    outer  = regions["outer"]
    if border.sum() == 0 or outer.sum() == 0:
        for sp in ["L", "a", "b"]:
            feats[f"bord_contrast_{sp}"] = 0.0
            feats[f"bord_blur_{sp}"] = 0.0
        return feats

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    for c, name in enumerate(["L", "a", "b"]):
        ch = lab[:, :, c]
        m_in  = ch[border].mean()
        m_out = ch[outer].mean()
        s_in  = ch[border].std() + 1e-6
        s_out = ch[outer].std() + 1e-6
        # 对比度: |均值差| / 联合 std
        feats[f"bord_contrast_{name}"] = float(abs(m_in - m_out) /
                                                  np.sqrt((s_in**2 + s_out**2) / 2))
        # blur: border 内部的 std 越大说明越模糊
        feats[f"bord_blur_{name}"] = float(s_in)
    return feats
