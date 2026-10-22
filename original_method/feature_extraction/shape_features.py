"""
形状特征模块 - 完全覆盖课件 Shape Feature 章节

课件涉及：
  Boundary:
    - length (perimeter)
    - diameter, major/minor axis, eccentricity
    - signature (centroid-distance vs angle)
    - Fourier Descriptors
    - tortuosity
  Region:
    - perimeter, area
    - compactness, circularity
    - eccentricity (covariance-based)
  Topological:
    - Euler number (connected components - holes)

我额外加入：
  - solidity, extent, equivalent diameter, aspect ratio
  - 非对称指数（皮肤病变 ABCD 规则的 A）
  - convex hull defects
"""
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from skimage import measure


def _get_largest_contour(mask: np.ndarray) -> np.ndarray:
    """从二值mask中提取最大轮廓"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# ============================================================
# Boundary Descriptors（课件 Slide 14-19, 21）
# ============================================================
def boundary_basic_descriptors(contour: np.ndarray) -> dict:
    """
    基础边界描述子: length, diameter, major/minor axis, eccentricity
    """
    feat = {}
    if contour is None or len(contour) < 5:
        return {k: 0.0 for k in [
            "bnd_length", "bnd_diameter", "bnd_major_axis", "bnd_minor_axis",
            "bnd_eccentricity_ellipse", "bnd_orientation"
        ]}

    pts = contour.reshape(-1, 2).astype(np.float32)

    # 长度（边界像素数 = arcLength近似）
    feat["bnd_length"] = float(cv2.arcLength(contour, True))

    # 直径：两点最大距离（高效近似用凸包）
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    if len(hull) >= 2:
        d = cdist(hull, hull)
        feat["bnd_diameter"] = float(d.max())
    else:
        feat["bnd_diameter"] = 0.0

    # 椭圆拟合 -> major/minor axis, eccentricity, orientation
    if len(contour) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour)
        major = max(MA, ma)
        minor = min(MA, ma)
        feat["bnd_major_axis"] = float(major)
        feat["bnd_minor_axis"] = float(minor)
        if major > 0:
            ratio = minor / major
            feat["bnd_eccentricity_ellipse"] = float(np.sqrt(max(0.0, 1 - ratio**2)))
        else:
            feat["bnd_eccentricity_ellipse"] = 0.0
        feat["bnd_orientation"] = float(angle)
    else:
        feat["bnd_major_axis"] = feat["bnd_minor_axis"] = 0.0
        feat["bnd_eccentricity_ellipse"] = feat["bnd_orientation"] = 0.0
    return feat


# ============================================================
# Signature（课件 Slide 16）
# ============================================================
def boundary_signature_features(contour: np.ndarray, n_bins: int = 36) -> dict:
    """
    Signature = r(θ) : 质心到边界的距离 vs 角度
    输出 signature 的统计特征
    """
    feat = {}
    if contour is None or len(contour) < 3:
        return {k: 0.0 for k in [
            "sig_mean", "sig_std", "sig_min", "sig_max",
            "sig_range", "sig_peaks", "sig_entropy"
        ]}
    pts = contour.reshape(-1, 2).astype(np.float64)
    cx, cy = pts.mean(axis=0)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # 把θ映射到[0, 2π), 然后用n_bins分箱取均值
    theta_mod = (theta + 2*np.pi) % (2*np.pi)
    bins = np.linspace(0, 2*np.pi, n_bins + 1)
    sig = np.zeros(n_bins)
    for i in range(n_bins):
        sel = (theta_mod >= bins[i]) & (theta_mod < bins[i+1])
        sig[i] = r[sel].mean() if sel.any() else 0
    # 填充缺失
    if (sig == 0).any():
        sig = _interp_zeros(sig)

    feat["sig_mean"] = float(sig.mean())
    feat["sig_std"] = float(sig.std())
    feat["sig_min"] = float(sig.min())
    feat["sig_max"] = float(sig.max())
    feat["sig_range"] = float(sig.max() - sig.min())
    # 峰值计数
    peaks = _count_peaks(sig)
    feat["sig_peaks"] = float(peaks)
    # 简单熵
    p = sig / (sig.sum() + 1e-12)
    feat["sig_entropy"] = float(-(p * np.log2(p + 1e-12)).sum())
    return feat


def _interp_zeros(arr):
    """循环序列中的0填充：用最近非零值的线性插值"""
    arr = arr.copy()
    n = len(arr)
    if (arr == 0).all():
        return arr
    nonzero_idx = np.where(arr > 0)[0]
    for i in range(n):
        if arr[i] == 0:
            # 找前后最近非零
            left = nonzero_idx[nonzero_idx < i]
            right = nonzero_idx[nonzero_idx > i]
            lv = arr[left[-1]] if len(left) else arr[nonzero_idx[-1]]
            rv = arr[right[0]] if len(right) else arr[nonzero_idx[0]]
            arr[i] = 0.5 * (lv + rv)
    return arr


def _count_peaks(sig):
    """简单峰值计数（局部最大且高于均值）"""
    n = len(sig)
    mean_v = sig.mean()
    count = 0
    for i in range(n):
        l = sig[(i-1) % n]; r = sig[(i+1) % n]
        if sig[i] > l and sig[i] > r and sig[i] > mean_v:
            count += 1
    return count


# ============================================================
# Fourier Descriptors（课件 Slide 17-19）
# ============================================================
def fourier_descriptors(contour: np.ndarray, n_descriptors: int = 10) -> dict:
    """
    Fourier 描述子：
        s(k) = x(k) + j*y(k)
        a(u) = DFT{s(k)}
    取前n个|a(u)|，并对a(1)做归一化以获得尺度不变性。
    """
    feat = {}
    if contour is None or len(contour) < n_descriptors + 1:
        return {f"fd_{i}": 0.0 for i in range(n_descriptors)}

    pts = contour.reshape(-1, 2).astype(np.float64)
    complex_pts = pts[:, 0] + 1j * pts[:, 1]
    # FFT
    coeffs = np.fft.fft(complex_pts)
    # 取幅值
    mags = np.abs(coeffs)
    # 归一化：除以|a(1)|得到尺度不变；丢弃a(0)（平移信息）
    if mags[1] > 1e-12:
        mags_norm = mags / mags[1]
    else:
        mags_norm = mags
    # 取前 n_descriptors个（跳过a(0)和a(1)）
    for i in range(n_descriptors):
        idx = i + 2
        feat[f"fd_{i}"] = float(mags_norm[idx]) if idx < len(mags_norm) else 0.0
    return feat


# ============================================================
# Region descriptors（课件 Slide 20-22）
# ============================================================
def region_basic_descriptors(mask: np.ndarray, contour: np.ndarray) -> dict:
    """
    面积、周长、紧致度、圆形度、矩特征、凸包相关
    """
    feat = {}
    area = float(np.count_nonzero(mask))
    feat["reg_area"] = area

    if contour is None or area == 0:
        return {k: 0.0 for k in [
            "reg_area", "reg_perimeter", "reg_compactness", "reg_circularity",
            "reg_solidity", "reg_extent", "reg_equiv_diameter", "reg_aspect_ratio",
            "reg_eccentricity_cov"
        ]}

    perim = float(cv2.arcLength(contour, True))
    feat["reg_perimeter"] = perim
    feat["reg_compactness"] = perim**2 / (area + 1e-12)   # p^2/A
    feat["reg_circularity"] = 4 * np.pi * area / (perim**2 + 1e-12)

    # 凸包
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    feat["reg_solidity"] = area / (hull_area + 1e-12)
    feat["reg_convex_hull_area"] = hull_area
    feat["reg_convex_defect_ratio"] = (hull_area - area) / (hull_area + 1e-12)

    # 包围矩形
    x, y, w, h = cv2.boundingRect(contour)
    feat["reg_extent"] = area / (w * h + 1e-12)
    feat["reg_aspect_ratio"] = w / (h + 1e-12)
    feat["reg_equiv_diameter"] = float(np.sqrt(4 * area / np.pi))

    # 协方差矩阵特征（课件 slide 21 的 covariance eccentricity）
    ys, xs = np.where(mask > 0)
    if len(xs) >= 2:
        coords = np.stack([xs, ys], axis=1).astype(np.float64)
        coords -= coords.mean(axis=0)
        cov = np.cov(coords, rowvar=False)
        eigs = np.linalg.eigvalsh(cov)
        eigs = np.sort(eigs)[::-1]  # 大到小
        l1, l2 = max(eigs[0], 1e-12), max(eigs[1], 1e-12)
        feat["reg_eccentricity_cov"] = float(np.sqrt(max(0.0, 1 - l2 / l1)))
    else:
        feat["reg_eccentricity_cov"] = 0.0

    return feat


# ============================================================
# Hu Moments（不变矩 - 平移/旋转/尺度不变）
# ============================================================
def hu_moment_features(mask: np.ndarray) -> dict:
    """
    Hu's 7个不变矩，对数变换数值稳定
    """
    M = cv2.moments(mask)
    hu = cv2.HuMoments(M).flatten()
    # 对数变换
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return {f"hu_{i}": float(hu_log[i]) for i in range(7)}


# ============================================================
# 拓扑特征（课件 Slide 24-26）
# ============================================================
def topological_features(mask: np.ndarray) -> dict:
    """
    Euler number = C - H (connected components - holes)
    """
    # skimage 的 measure.euler_number
    binary = (mask > 0).astype(np.uint8)
    feat = {
        "topo_euler_number": float(measure.euler_number(binary, connectivity=2)),
    }
    # 连通分量数
    n_cc, _ = cv2.connectedComponents(binary, connectivity=8)
    feat["topo_n_components"] = float(n_cc - 1)  # 减去背景
    # 孔洞数：用反向连通分量计算
    inv = 1 - binary
    # 移除背景孔
    inv_padded = np.pad(inv, 1, constant_values=1)
    n_inv, _ = cv2.connectedComponents(inv_padded.astype(np.uint8), 8)
    feat["topo_n_holes"] = float(max(0, n_inv - 2))  # -1背景 -1外部
    return feat


# ============================================================
# ABCD规则中的 Asymmetry（皮肤镜临床特征 - 亮点）
# ============================================================
def asymmetry_features(mask: np.ndarray, contour: np.ndarray) -> dict:
    """
    皮肤镜诊断 ABCD 规则中的 A（Asymmetry）:
        把病变绕质心旋转到主轴方向后，比较与镜像的差异面积
    """
    feat = {"asym_horizontal": 0.0, "asym_vertical": 0.0, "asym_total": 0.0}
    if contour is None or np.count_nonzero(mask) == 0:
        return feat

    # 主轴方向
    if len(contour) < 5:
        return feat
    (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour)
    # 旋转到主轴水平
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                             flags=cv2.INTER_NEAREST)
    # 水平对称
    flipped_h = cv2.flip(rotated, 1)
    flipped_v = cv2.flip(rotated, 0)
    area = float(np.count_nonzero(rotated)) + 1e-12
    diff_h = np.count_nonzero(cv2.bitwise_xor(rotated, flipped_h))
    diff_v = np.count_nonzero(cv2.bitwise_xor(rotated, flipped_v))
    feat["asym_horizontal"] = diff_h / area
    feat["asym_vertical"] = diff_v / area
    feat["asym_total"] = 0.5 * (feat["asym_horizontal"] + feat["asym_vertical"])
    return feat


# ============================================================
# 主入口：提取全部形状特征
# ============================================================
def extract_shape_features(mask: np.ndarray, n_fd: int = 10,
                           sig_bins: int = 36) -> dict:
    """
    一次性提取全部形状特征
    """
    feat = {}
    contour = _get_largest_contour(mask)
    feat.update(boundary_basic_descriptors(contour))
    feat.update(boundary_signature_features(contour, sig_bins))
    feat.update(fourier_descriptors(contour, n_fd))
    feat.update(region_basic_descriptors(mask, contour))
    feat.update(hu_moment_features(mask))
    feat.update(topological_features(mask))
    feat.update(asymmetry_features(mask, contour))
    return feat
