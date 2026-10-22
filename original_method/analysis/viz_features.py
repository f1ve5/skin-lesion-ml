"""
可视化模块（第1部分，升级版 v2）
================================
预处理流程 + 形状特征 + 【新增】色彩归一化对比 + 区域分割可视化
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_extraction.preprocessing import (
    dullrazor_hair_removal, fallback_mask_from_otsu, convert_color_spaces
)
from feature_extraction.color_normalization import (
    shades_of_gray, gray_world, max_rgb
)
from feature_extraction.regional_color_features import split_lesion_regions
from feature_extraction.shape_features import _get_largest_contour


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ============================================================
# 1. 预处理流程展示（升级版：加色彩归一化对比）
# ============================================================
def viz_preprocessing_pipeline(img_bgr, mask, save_path,
                                hair_kernel=17, inpaint_radius=3):
    """
    完整流程可视化：原图 -> 毛发mask -> 去毛发 -> 色彩归一化 -> 病变mask
    + 4 色彩空间
    """
    setup_style()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 毛发检测
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                        (hair_kernel, hair_kernel))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    clean = dullrazor_hair_removal(img_bgr, hair_kernel, inpaint_radius)

    # 色彩归一化 (Shades-of-Gray p=6, Barata 2015)
    clean_cn = shades_of_gray(clean, p=6)
    clean_cn_rgb = cv2.cvtColor(clean_cn, cv2.COLOR_BGR2RGB)

    # 病变 mask
    if mask is None:
        mask = fallback_mask_from_otsu(clean_cn)

    # 4 色彩空间
    cs = convert_color_spaces(clean_cn)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("(a) Original")
    axes[0, 1].imshow(hair_mask, cmap="gray")
    axes[0, 1].set_title("(b) Hair Mask\n(Blackhat morphology)")
    axes[0, 2].imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("(c) DullRazor Inpaint")
    axes[0, 3].imshow(clean_cn_rgb)
    axes[0, 3].set_title("(d) +Shades-of-Gray\n(Barata 2015)")
    axes[0, 4].imshow(mask, cmap="gray")
    axes[0, 4].set_title("(e) Lesion Mask")

    axes[1, 0].imshow(cs["RGB"]); axes[1, 0].set_title("RGB")
    axes[1, 1].imshow(cs["HSV"]); axes[1, 1].set_title("HSV")
    axes[1, 2].imshow(cs["Lab"]); axes[1, 2].set_title("Lab")
    axes[1, 3].imshow(cs["YCrCb"]); axes[1, 3].set_title("YCrCb")
    axes[1, 4].axis("off")

    for ax in axes.flatten():
        if ax.has_data():
            ax.axis("off")
    fig.suptitle("Preprocessing Pipeline (v2 - Color Constancy added)",
                 fontsize=13, fontweight="bold")
    plt.savefig(save_path)
    plt.close(fig)


# ============================================================
# 1B. 色彩归一化对比图（新）
# ============================================================
def viz_color_normalization_comparison(img_bgr, save_path):
    """
    并排展示 4 种色彩归一化方法的效果差异。
    """
    setup_style()
    methods = {
        "Original":          img_bgr,
        "Gray-World":        gray_world(img_bgr),
        "Shades-of-Gray\n(p=6)": shades_of_gray(img_bgr, p=6),
        "Max-RGB\n(White-Patch)": max_rgb(img_bgr),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 上排：图像
    for col, (name, img) in enumerate(methods.items()):
        axes[0, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(name)
        axes[0, col].axis("off")

    # 下排：RGB 直方图（用线条紧凑展示）
    colors = ("b", "g", "r")
    ch_names = ("B", "G", "R")
    for col, (name, img) in enumerate(methods.items()):
        ax = axes[1, col]
        for c, color, ch_name in zip(range(3), colors, ch_names):
            hist = cv2.calcHist([img], [c], None, [64], [0, 256])
            ax.plot(np.linspace(0, 256, 64), hist.flatten(),
                    color=color, lw=1.0, label=ch_name)
        ax.set_xlim(0, 256)
        ax.set_xlabel("Pixel value"); ax.set_ylabel("Count")
        ax.set_title(f"RGB Histogram"); ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Color Normalization Methods Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ============================================================
# 1C. 区域分割可视化（新）
# ============================================================
def viz_region_segmentation(img_bgr, mask, save_path):
    """
    展示 Celebi 2007 风格的 inner50 / inner75 / border / outer 分区。
    """
    setup_style()
    regions = split_lesion_regions(mask)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    axes[0, 0].imshow(img_rgb); axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
    axes[0, 1].imshow(regions["lesion"], cmap="gray")
    axes[0, 1].set_title(f"Lesion (full)\n{regions['lesion'].sum()} px"); axes[0, 1].axis("off")
    axes[0, 2].imshow(regions["inner50"], cmap="Reds")
    axes[0, 2].set_title(f"Inner 50%\n{regions['inner50'].sum()} px"); axes[0, 2].axis("off")

    axes[1, 0].imshow(regions["inner75"], cmap="Oranges")
    axes[1, 0].set_title(f"Inner 75%\n{regions['inner75'].sum()} px"); axes[1, 0].axis("off")
    axes[1, 1].imshow(regions["border"], cmap="Purples")
    axes[1, 1].set_title(f"Border (transition)\n{regions['border'].sum()} px"); axes[1, 1].axis("off")
    axes[1, 2].imshow(regions["outer"], cmap="Greens")
    axes[1, 2].set_title(f"Outer (surrounding skin)\n{regions['outer'].sum()} px"); axes[1, 2].axis("off")

    fig.suptitle("Region Segmentation (Celebi 2007 - Euclidean Distance Transform)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ============================================================
# 2. 形状特征可视化（保持不变）
# ============================================================
def viz_shape_features(img_bgr, mask, save_path, n_fd_list=(4, 10, 20, 40)):
    setup_style()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    contour = _get_largest_contour(mask)
    if contour is None:
        return

    pts = contour.reshape(-1, 2).astype(np.float64)
    cx, cy = pts.mean(axis=0)

    sig_bins = 72
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)
    bins = np.linspace(0, 2*np.pi, sig_bins + 1)
    sig = np.zeros(sig_bins)
    for i in range(sig_bins):
        sel = (theta >= bins[i]) & (theta < bins[i+1])
        sig[i] = r[sel].mean() if sel.any() else 0
    if (sig == 0).any():
        idx_nz = np.where(sig > 0)[0]
        if len(idx_nz) > 0:
            mean_val = sig[idx_nz].mean()
            sig[sig == 0] = mean_val

    complex_pts = pts[:, 0] + 1j * pts[:, 1]
    coeffs = np.fft.fft(complex_pts)
    K = len(coeffs)

    def reconstruct(P):
        c2 = np.zeros_like(coeffs)
        half = P // 2
        c2[:half] = coeffs[:half]
        c2[-half:] = coeffs[-half:]
        rec = np.fft.ifft(c2)
        return rec.real, rec.imag

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_rgb)
    ax.plot(pts[:, 0], pts[:, 1], "y-", lw=1.2)
    ax.plot(cx, cy, "r+", markersize=12)
    ax.set_title("Original + Boundary"); ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(mask, cmap="gray")
    ax.plot(pts[:, 0], pts[:, 1], "r-", lw=1.0)
    ax.set_title("Mask + Boundary"); ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(img_rgb)
    if len(contour) >= 5:
        (ex, ey), (MA, ma), ang = cv2.fitEllipse(contour)
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(xy=(ex, ey), width=MA, height=ma, angle=ang,
                          edgecolor="cyan", lw=2, fill=False)
        ax.add_patch(ellipse)
    ax.set_title("Fitted Ellipse"); ax.axis("off")

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(img_rgb)
    hull = cv2.convexHull(contour).reshape(-1, 2)
    hull_closed = np.vstack([hull, hull[:1]])
    ax.plot(pts[:, 0], pts[:, 1], "y-", lw=1, label="boundary")
    ax.plot(hull_closed[:, 0], hull_closed[:, 1], "g--", lw=1.5, label="convex hull")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Convex Hull (Solidity)"); ax.axis("off")

    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(np.degrees(bins[:-1]), sig, "b-", lw=1.5)
    ax.fill_between(np.degrees(bins[:-1]), 0, sig, alpha=0.2)
    ax.set_xlabel("θ (degrees)"); ax.set_ylabel("r(θ)")
    ax.set_title("Signature r(θ)")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot(pts[:, 0], pts[:, 1], "k-", lw=1.5, alpha=0.5, label=f"original (K={K})")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(n_fd_list)))
    for P, c in zip(n_fd_list, colors):
        rx, ry = reconstruct(P)
        rx_closed = np.append(rx, rx[0]); ry_closed = np.append(ry, ry[0])
        ax.plot(rx_closed, ry_closed, "-", color=c, lw=1.2, label=f"P={P}")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Fourier Descriptor Reconstruction")
    ax.grid(alpha=0.3)

    fig.suptitle("Shape Feature Visualization", fontsize=13, fontweight="bold")
    plt.savefig(save_path)
    plt.close(fig)
