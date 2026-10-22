"""
图像预处理（升级版 v2）
========================
原 pipeline 顺序:
  1. resize  -> 2. DullRazor 去毛发 -> 3. 转 4 个色彩空间

升级后顺序:
  1. resize -> 2. DullRazor -> 3. 色彩归一化(Shades-of-Gray) -> 4. 转色彩空间

新增依据：
  Barata C, Celebi ME, Marques JS. Improving Dermoscopy Image
  Classification Using Color Constancy. IEEE J Biomed Health Inform 2015;
  19(3):1146-1152.  实验证明 color constancy 能提升分类一致性。
"""
import cv2
import numpy as np
from typing import Optional, Dict

from .color_normalization import normalize_color


# ============================================================
# DullRazor 毛发去除（Lee 1997）
# ============================================================
def dullrazor_hair_removal(img_bgr: np.ndarray,
                            kernel_size: int = 17,
                            inpaint_radius: int = 3) -> np.ndarray:
    """blackhat 形态学检测深色长线条 → inpaint 修补"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                        (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    clean = cv2.inpaint(img_bgr, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return clean


# ============================================================
# Otsu fallback mask（当没有官方 mask 时）
# ============================================================
def fallback_mask_from_otsu(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    # 只保留最大连通块
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    # 跳过 label 0 (background)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.where(labels == largest, 255, 0).astype(np.uint8)
    return out


# ============================================================
# 转 4 个色彩空间
# ============================================================
def convert_color_spaces(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "RGB":   cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "HSV":   cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV),
        "Lab":   cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB),
        "YCrCb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb),
    }


# ============================================================
# 主入口
# ============================================================
def preprocess_image(img_bgr: np.ndarray,
                      mask: Optional[np.ndarray],
                      target_size: tuple = (256, 256),
                      do_hair_removal: bool = True,
                      hair_kernel: int = 17,
                      inpaint_radius: int = 3,
                      color_normalize: str = "shades_of_gray",
                      color_norm_p: int = 6) -> Dict:
    """
    返回:
      image_bgr_raw  : resize 后的原图
      image_bgr      : 经过去毛发 + 色彩归一化的图
      mask           : 处理后的二值 mask
      color_spaces   : 4 个色彩空间的字典
      hair_removed   : 是否做了去毛发
      cn_method      : 色彩归一化方法名
    """
    # 1) Resize
    img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        _, mask_resized = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
    else:
        mask_resized = None

    img_raw = img_resized.copy()

    # 2) DullRazor 去毛发
    if do_hair_removal:
        img_proc = dullrazor_hair_removal(img_resized, hair_kernel, inpaint_radius)
    else:
        img_proc = img_resized.copy()

    # 3) Color Normalization（新加：在 mask 还未生成前对原图做）
    img_proc = normalize_color(img_proc, method=color_normalize, p=color_norm_p)

    # 4) 如果还没有 mask，用 Otsu 自动生成
    if mask_resized is None:
        mask_resized = fallback_mask_from_otsu(img_proc)

    # 5) 转色彩空间
    cs = convert_color_spaces(img_proc)

    return {
        "image_bgr_raw": img_raw,
        "image_bgr":     img_proc,
        "mask":          mask_resized,
        "color_spaces":  cs,
        "hair_removed":  do_hair_removal,
        "cn_method":     color_normalize,
    }
