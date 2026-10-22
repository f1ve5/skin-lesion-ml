"""
色彩归一化（Color Normalization）
==================================
文献依据：
  - Barata C, Celebi ME, Marques JS. Improving Dermoscopy Image
    Classification Using Color Constancy. IEEE J Biomed Health Inform
    2015; 19(3): 1146-1152.
  - Finlayson GD, Trezzi E. Shades of Gray and Colour Constancy.
    Color Imaging Conf 2004.

主要方法：
  - Gray-World 假设: 整张图像 R/G/B 三通道平均值应相等
  - Shades-of-Gray (Minkowski p-norm): Gray-World的推广，p=6效果最佳
"""
import numpy as np
import cv2


def shades_of_gray(img_bgr: np.ndarray, p: int = 6, eps: float = 1e-6) -> np.ndarray:
    """
    Shades-of-Gray 色彩归一化（Finlayson & Trezzi 2004）

    每通道 estimate illuminant 用 Minkowski p-范数：
        e_c = (mean(I_c^p))^(1/p)
    然后线性缩放使三通道的illuminant都等于 gray.

    p=1 退化为 Gray-World；p=∞ 退化为 White-Patch（Max-RGB）。
    实验显示 p=6 在皮肤镜上效果最佳（Barata 2015）。
    """
    img = img_bgr.astype(np.float64)
    # cv2 是 BGR 顺序
    illum = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p)
    illum = np.maximum(illum, eps)
    # 用三通道 illuminant 的几何均值作为目标灰度
    gray = np.power(np.prod(illum), 1.0 / 3.0)
    scale = gray / illum  # shape (3,)
    img_norm = img * scale[np.newaxis, np.newaxis, :]
    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
    return img_norm


def gray_world(img_bgr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Gray-World assumption: p=1 的 Shades-of-Gray 特例"""
    return shades_of_gray(img_bgr, p=1, eps=eps)


def max_rgb(img_bgr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """White-Patch (Max-RGB)：用每通道最大值作为 illuminant"""
    img = img_bgr.astype(np.float64)
    illum = np.maximum(img.max(axis=(0, 1)), eps)
    gray = np.power(np.prod(illum), 1.0 / 3.0)
    scale = gray / illum
    img_norm = np.clip(img * scale[np.newaxis, np.newaxis, :], 0, 255).astype(np.uint8)
    return img_norm


def normalize_color(img_bgr: np.ndarray,
                     method: str = "shades_of_gray",
                     p: int = 6) -> np.ndarray:
    """
    统一接口：
      method ∈ {"none", "gray_world", "shades_of_gray", "max_rgb"}
    """
    if method == "none" or method is None:
        return img_bgr
    if method == "gray_world":
        return gray_world(img_bgr)
    if method == "shades_of_gray":
        return shades_of_gray(img_bgr, p=p)
    if method == "max_rgb":
        return max_rgb(img_bgr)
    raise ValueError(f"未知色彩归一化方法: {method}")
