"""
可视化模块（第2部分）：直方图特征、GLCM、LBP、Gabor、Wavelet、Spectral
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pywt
from skimage.feature import graycomatrix, local_binary_pattern
from skimage.filters import gabor

from .viz_features import setup_style
from feature_extraction.texture_features import quantize_gray


# ============================================================
# 3. 多色彩空间直方图
# ============================================================
def viz_color_histograms(img_bgr, mask, save_path):
    setup_style()
    from feature_extraction.preprocessing import convert_color_spaces
    cs = convert_color_spaces(img_bgr)
    channel_names = {
        "RGB":   ["R", "G", "B"],
        "HSV":   ["H", "S", "V"],
        "Lab":   ["L", "a", "b"],
        "YCrCb": ["Y", "Cr", "Cb"],
    }
    colors_for_ch = {
        "R": "red", "G": "green", "B": "blue",
        "H": "magenta", "S": "orange", "V": "purple",
        "L": "gray", "a": "olive", "b": "brown",
        "Y": "navy", "Cr": "tomato", "Cb": "teal",
    }

    fig, axes = plt.subplots(4, 3, figsize=(13, 12))
    for row, (space, img) in enumerate(cs.items()):
        names = channel_names[space]
        for col, name in enumerate(names):
            ax = axes[row, col]
            ch = img[:, :, col]
            roi_pixels = ch[mask > 0]
            bg_pixels = ch[mask == 0]
            if roi_pixels.size > 0:
                ax.hist(roi_pixels, bins=40, range=(0, 256),
                        color=colors_for_ch.get(name, "blue"),
                        alpha=0.7, density=True,
                        label=f"Lesion ROI (n={roi_pixels.size})")
            if bg_pixels.size > 0:
                ax.hist(bg_pixels, bins=40, range=(0, 256),
                        color="lightgray", alpha=0.5, density=True,
                        label=f"Background")
            ax.set_title(f"{space} — {name}")
            ax.legend(fontsize=7, loc="best")
            ax.set_xlim(0, 256)
            ax.grid(alpha=0.2)
    fig.suptitle("Per-Channel Intensity Histograms (Lesion vs Background)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ============================================================
# 4. GLCM可视化
# ============================================================
def viz_glcm(img_bgr, mask, save_path, levels=32):
    setup_style()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    qgray = quantize_gray(gray, levels)
    qgray_roi = qgray.copy(); qgray_roi[mask == 0] = 0

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    angle_names = ["0°", "45°", "90°", "135°"]
    distances = [1, 3]

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes[0, 0].imshow(qgray_roi, cmap="gray")
    axes[0, 0].set_title(f"Quantized Gray\n(levels={levels})")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Original"); axes[1, 0].axis("off")

    for row_i, d in enumerate(distances):
        for col_i, (a, an) in enumerate(zip(angles, angle_names)):
            glcm = graycomatrix(qgray_roi, distances=[d], angles=[a],
                                 levels=levels, symmetric=True, normed=True)
            P = glcm[:, :, 0, 0].copy()
            # 屏蔽掉(0,0)处的统计（背景配对），凸显ROI内部纹理
            P[0, 0] = 0
            # 重新归一化方便对比
            s = P.sum()
            if s > 0: P = P / s
            ax = axes[row_i, col_i + 1]
            im = ax.imshow(np.log1p(P * 1000), cmap="hot", aspect="auto")
            ax.set_title(f"d={d}, θ={an}")
            ax.set_xlabel("j (gray)"); ax.set_ylabel("i (gray)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("GLCM (Gray-Level Co-occurrence Matrix) - log scale",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ============================================================
# 5. LBP可视化
# ============================================================
def viz_lbp(img_bgr, mask, save_path, radius=3, n_points=24):
    setup_style()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    lbp_roi = lbp[mask > 0]
    hist, _ = np.histogram(lbp_roi, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(float) / (hist.sum() + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(lbp, cmap="viridis")
    axes[1].set_title(f"LBP map (R={radius}, P={n_points})")
    axes[1].axis("off")
    axes[2].bar(range(n_bins), hist, color="steelblue", edgecolor="black", lw=0.5)
    axes[2].set_title("LBP histogram in lesion ROI")
    axes[2].set_xlabel("Uniform LBP bin"); axes[2].set_ylabel("Probability")
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ============================================================
# 6. Gabor响应
# ============================================================
def viz_gabor(img_bgr, save_path,
              frequencies=(0.1, 0.3, 0.5),
              thetas_deg=(0, 45, 90, 135)):
    setup_style()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    nf, nt = len(frequencies), len(thetas_deg)
    fig, axes = plt.subplots(nf, nt, figsize=(3.2*nt, 3*nf))
    if nf == 1: axes = np.array([axes])
    if nt == 1: axes = axes.reshape(-1, 1)
    for i, f in enumerate(frequencies):
        for j, theta in enumerate(thetas_deg):
            real, _ = gabor(gray, frequency=f, theta=np.deg2rad(theta))
            ax = axes[i, j]
            ax.imshow(real, cmap="seismic")
            ax.set_title(f"f={f}, θ={theta}°")
            ax.axis("off")
    fig.suptitle("Gabor Filter Responses", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ============================================================
# 7. 小波分解
# ============================================================
def viz_wavelet(img_bgr, save_path, wavelet="db4", level=2):
    setup_style()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    coeffs = pywt.wavedec2(gray, wavelet, level=level)

    # 组合成可视化的"图谱"
    arr, slices = pywt.coeffs_to_array(coeffs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original Grayscale"); axes[0].axis("off")
    im = axes[1].imshow(np.log1p(np.abs(arr)), cmap="viridis")
    axes[1].set_title(f"Wavelet ({wavelet}, level={level}) — log scale")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ============================================================
# 8. 频谱纹理：FFT + S(r), S(θ)
# ============================================================
def viz_spectral(img_bgr, mask, save_path, r_bins=30, theta_bins=36):
    setup_style()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    if mask.sum() > 0:
        mean_v = gray[mask > 0].mean()
        gray_proc = np.where(mask > 0, gray, mean_v)
    else:
        gray_proc = gray

    F = np.fft.fftshift(np.fft.fft2(gray_proc))
    S = np.abs(F) ** 2
    H, W = S.shape; cy, cx = H//2, W//2
    y, x = np.indices(S.shape)
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    theta = np.arctan2(y-cy, x-cx)
    theta_mod = (theta + np.pi) % np.pi
    R_max = min(cx, cy)

    r_edges = np.linspace(0, R_max, r_bins + 1)
    sr = np.zeros(r_bins)
    for i in range(r_bins):
        sel = (r >= r_edges[i]) & (r < r_edges[i+1])
        if sel.any(): sr[i] = S[sel].sum()
    t_edges = np.linspace(0, np.pi, theta_bins + 1)
    st = np.zeros(theta_bins)
    for i in range(theta_bins):
        sel = (theta_mod >= t_edges[i]) & (theta_mod < t_edges[i+1]) & (r < R_max)
        if sel.any(): st[i] = S[sel].sum()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(np.log1p(S), cmap="magma")
    axes[0].set_title("Log Power Spectrum")
    axes[0].axis("off")

    axes[1].plot(r_edges[:-1], np.log1p(sr), "b-", lw=1.5)
    axes[1].fill_between(r_edges[:-1], 0, np.log1p(sr), alpha=0.2)
    axes[1].set_xlabel("radius r (pixels)")
    axes[1].set_ylabel("log(1+S(r))")
    axes[1].set_title("Radial spectrum S(r)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(np.degrees(t_edges[:-1]), np.log1p(st), "g-", lw=1.5)
    axes[2].fill_between(np.degrees(t_edges[:-1]), 0, np.log1p(st), alpha=0.2)
    axes[2].set_xlabel("angle θ (degrees)")
    axes[2].set_ylabel("log(1+S(θ))")
    axes[2].set_title("Angular spectrum S(θ)")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Spectral Texture Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
