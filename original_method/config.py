"""
配置文件 (v2 升级版)
========================
新增可调参数：
  - COLOR_NORMALIZE : 色彩归一化方法
  - COLOR_NORM_P    : Shades-of-Gray 的 p 参数（Barata 2015 建议 p=6）
  - USE_REGIONAL_COLOR / USE_RELATIVE_COLOR / USE_BORDER_CONTRAST : 新特征开关
"""
from pathlib import Path

# ============================================================
# 数据路径
# ============================================================
# 修改为你本地的数据路径
import os
DATA_ROOT = os.environ.get("SKIN_LESION_DATA_ROOT", "data/private/Data_Proj2")
# DATA_ROOT = "/home/claude/synthetic_data"  # demo时可改为这个

LABEL_CSV = Path(DATA_ROOT) / "label.csv"
IMAGE_DIR = Path(DATA_ROOT) / "image"
MASK_DIR  = Path(DATA_ROOT) / "mask"

# ============================================================
# 输出路径
# ============================================================
OUTPUT_ROOT = Path("./outputs")
FIGURE_DIR  = OUTPUT_ROOT / "figures"
RESULT_DIR  = OUTPUT_ROOT / "results"
LOG_DIR     = OUTPUT_ROOT / "logs"
for d in (FIGURE_DIR, RESULT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 测试与模型路径
TEST_DIR   = Path("./test")
TEST_IMAGE = TEST_DIR / "image"
TEST_MASK  = TEST_DIR / "mask"
MODEL_DIR  = Path("./models")

# ============================================================
# 预处理
# ============================================================
RESIZE_SHAPE       = (256, 256)
HAIR_KERNEL_SIZE   = 17
HAIR_INPAINT_RADIUS = 3

# 【对照实验开关】是否启用 DullRazor 去毛发
#   True  = 标准 v2 流程
#   False = 不去毛发对照组（保留病灶内部细节，验证去毛发是否损伤诊断特征）
# 一般通过命令行 --no-hair-removal 切换，不需要手动改这里
DO_HAIR_REMOVAL = True

# 【新增 v2】色彩归一化（Barata 2015）
COLOR_NORMALIZE = "shades_of_gray"   # {"none","gray_world","shades_of_gray","max_rgb"}
COLOR_NORM_P    = 6                  # Shades-of-Gray 的 Minkowski p（p=6 最佳）

# ============================================================
# 新特征开关（v2）
# ============================================================
USE_REGIONAL_COLOR  = True  # inner50/border/outer × 4 色彩空间 × 3 通道 × 8 stat
USE_RELATIVE_COLOR  = True  # lesion 减去 surrounding skin 的相对色
USE_BORDER_CONTRAST = True  # border vs outer 在 Lab 上的对比度

# ============================================================
# 特征提取参数
# ============================================================
FOURIER_DESCRIPTOR_N = 10
SIGNATURE_BINS       = 72
PERCENTILES          = (10, 25, 50, 75, 90)

GLCM_LEVELS    = 32
GLCM_DISTANCES = (1, 3, 5)
GLCM_ANGLES    = (0, 45, 90, 135)   # 度
GLCM_PROPS     = ("contrast", "homogeneity", "energy",
                  "correlation", "dissimilarity", "ASM")

LBP_RADIUS   = 3
LBP_N_POINTS = 24
LBP_METHOD   = "uniform"

GABOR_FREQUENCIES = (0.1, 0.3, 0.5)
GABOR_THETAS      = (0, 45, 90, 135)   # 度

WAVELET_NAME  = "db4"
WAVELET_LEVEL = 2

SPECTRAL_R_BINS     = 30
SPECTRAL_THETA_BINS = 36

# ============================================================
# 联合特征选择的临床优先级映射 (0~1, 越小越优先)
# ============================================================
# 文献依据：ABCD 法则 (Stolz 1994), Celebi 2007, Barata 2019
# 使用前缀/子串匹配，按顺序匹配第一个命中
CLINICAL_PRIORITY = {
    # 高优先 (0.0-0.2): ABCD 直接对应的关键特征
    "bord_":               0.05,   # 边界对比度 — B (Border sharpness)
    "asym_":               0.10,   # 不对称指数 — A (Asymmetry)
    "reg_convex_defect":   0.10,   # 凸缺陷比 — B (Border irregularity)
    "reg_solidity":        0.15,   # 凸性 — B
    "hu_":                 0.20,   # Hu 不变矩 — A (Asymmetry)
    "fd_":                 0.20,   # 傅里叶描述子 — B
    # 中优先 (0.25-0.45): 有判别力的一阶/区域颜色特征
    "reg_inner50_HSV_H":   0.25,   # 核心区色调 — C (Color)
    "reg_border_HSV_H":    0.25,   # 边界区色调
    "int_HSV_H_p":         0.25,   # HSV-H 分位数
    "int_HSV_H_mean":      0.25,
    "int_HSV_H_median":    0.25,
    "reg_inner50_Lab_a":   0.30,   # 核心区红绿轴
    "reg_border_Lab_a":    0.30,
    "int_Lab_a":           0.30,
    "reg_inner50_Lab_b":   0.30,   # 核心区黄蓝轴
    "reg_border_Lab_b":    0.30,
    "int_Lab_b":           0.30,
    "int_YCrCb_Cb":        0.30,
    "int_YCrCb_Cr":        0.30,
    "rel_":                0.35,   # 相对色彩 — C (肤色归一化)
    "ccr_":                0.35,   # 颜色恒常性比率
    "specific_":           0.35,   # 特异性纹理检测器
    "reg_inner50_":        0.35,   # 核心区通用
    "reg_border_":         0.35,   # 边界区通用
    "reg_outer_":          0.80,   # 外圈皮肤 — 几乎无判别力，必须放在 reg_ 前面
    "reg_":                0.40,   # 区域形状 (Shape-Region)
    "bnd_":                0.40,   # 边界形状
    "sig_":                0.40,   # 签名
    "int_":                0.40,   # 一阶强度通用
    "glcm_":               0.40,   # GLCM 纹理
    "wave_":               0.45,   # 小波
    "spec_":               0.45,   # 频谱
    "gabor_":              0.50,   # Gabor — 对旋转敏感
    "lbp_":                0.50,   # LBP — 对旋转敏感
    # 低优先 (>0.50): 次要或稳定性差的特征
    "topo_":               0.80,   # 拓扑 — 常数特征
}

# 联合特征选择权重
JOINT_FS_ALPHA = 0.5   # F-test 排名权重
JOINT_FS_BETA  = 0.3   # ICC 鲁棒性排名权重
JOINT_FS_GAMMA = 0.2   # 临床先验权重
JOINT_FS_K     = 40    # 默认选择特征数

# ============================================================
# 分类
# ============================================================
RANDOM_SEED    = 42
CV_FOLDS       = 5
TOP_K_FEATURES = 30

# ============================================================
# v5.0: 非对称 PCA 特征筛选
# ============================================================
# 核心创新: Layer 1 和 Layer 2 使用不同的特征空间
#   Layer 1 (vasc vs non-vasc): PCA 去噪 → 可用更多特征 (k=30)
#   Layer 2 (mel vs nv): 原始特征 → 保留全部判别信号 (k=30)
# 效果: F1m=0.789, BalAcc=0.815, mel_rec=0.724 (+3.4pp vs 全局PCA)
SCREENING_STRATEGY = "mrmr"            # mrmr (v5.0 最优: F1m=0.789 BalAcc=0.815)
SCREENING_LAYER1_TOP_K   = 30          # Layer 1: 30 (非对称 PCA 下 30>20, PCA 去噪后更多特征不引入噪声)
SCREENING_LAYER2_TOP_K   = 30          # Layer 2: 30 (原始特征, 30>40>50)
SCREENING_LAYER1_ICC_MIN = 0.50        # Layer 1 ICC 最低阈值 (宽松)
SCREENING_LAYER2_ICC_MIN = 0.75        # Layer 2 ICC 最低阈值 (严格)

# ============================================================
# v4: 交互特征工程 (Layer 2 专属)
# ============================================================
# 文献依据: Moldovanu 2023 — hybrid features 优于纯颜色特征
#           Liopyris 2023 — 高一致性 dermoscopic 特征全为结构性
INTERACTION_STRUCTURAL_M = 5           # Structural family top-M (消融实验: 5 优于 15)
INTERACTION_COLOR_N      = 5           # Color family top-N (消融实验: 5 优于 15)
INTERACTION_MAX_KEEP     = 10          # 后筛选保留交互特征数 (自动校准: min(cfg, n_mel_nv//50))
INTERACTION_TYPES        = ["product", "ratio", "difference"]  # product / ratio / difference
SCREENING_MAX_PER_CHANNEL = 999           # 通道多样性约束 (mRMR 内置冗余控制, 关闭硬桶)

# v5.0: 非对称 PCA 降维 — Layer 1 和 Layer 2 独立控制
# 原理: Layer 1 (vasc vs non-vasc) 受益于 PCA 去噪→vasc_rec↑
#       Layer 2 (mel vs nv) PCA 丢弃低方差但具判别力信号→mel_rec↓
#       分离 PCA 控制使两层各取所需
SCREENING_L1_PCA          = True       # Layer 1: PCA 去噪, 提升 vasc 识别
SCREENING_L1_PCA_VARIANCE = 0.99       # Layer 1 PCA 保留方差
SCREENING_L2_PCA          = False      # Layer 2: 保留原始特征, 保护 mel/nv 判别信号
SCREENING_L2_PCA_VARIANCE = 0.99       # Layer 2 PCA 保留方差 (L2 关闭 PCA 时无效)

# ============================================================
# v4: 集成分类器 (Layer 2 专属)
# ============================================================
# 文献依据: Sharafudeen 2023 — 手工特征 + Soft Voting, Sens=90.49%
#           Alfi 2022 — Stacking + SHAP, Diagnostics
ENSEMBLE_TOP_N_BASE     = 3            # Voting 基分类器数
ENSEMBLE_STACKING_META  = "LogReg"     # Stacking 元学习器
USE_XGBOOST             = True         # XGBoost (需 pip install xgboost)
USE_LIGHTGBM            = False        # LightGBM (默认关闭)
