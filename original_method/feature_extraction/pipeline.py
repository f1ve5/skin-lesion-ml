"""
完整的特征提取流水线（升级版 v2）
========================================
对每张图：
  preprocessing (含色彩归一化)
    -> 计算 inner/border/outer 区域 (Celebi 2007 风格)
    -> 形状特征(46)
    -> 一阶特征(260)
    -> 纹理特征(202)
    -> 【新增】区域分割一阶特征 (~144)
    -> 【新增】相对色彩特征 (~56)
    -> 【新增】边界对比度特征 (6)
最终输出 features.csv

总特征维度从原来的 ~510 提升到约 ~720。
"""
import time
from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocessing import preprocess_image
from .shape_features import extract_shape_features
from .intensity_features import (
    extract_intensity_features, extract_gray_intensity_features
)
from .texture_features import extract_texture_features
from .regional_color_features import (
    split_lesion_regions,
    extract_regional_color_features,
    extract_relative_color_features,
    extract_border_contrast_features,
)
from .texture_specific_features import extract_texture_specific_features
from .color_constancy_ratio_features import extract_color_constancy_ratio_features
from utils.io_utils import read_image_bgr, read_mask


def extract_all_features_one_image(image_path: str,
                                    mask_path: str,
                                    cfg) -> Dict[str, float]:
    """对单张图像提取全部特征（升级版）"""
    img = read_image_bgr(image_path)
    mask = read_mask(mask_path) if mask_path else None

    # 1) 预处理（已含色彩归一化）
    pre = preprocess_image(
        img, mask,
        target_size=cfg.RESIZE_SHAPE,
        do_hair_removal=getattr(cfg, "DO_HAIR_REMOVAL", True),
        hair_kernel=cfg.HAIR_KERNEL_SIZE,
        inpaint_radius=cfg.HAIR_INPAINT_RADIUS,
        color_normalize=getattr(cfg, "COLOR_NORMALIZE", "shades_of_gray"),
        color_norm_p=getattr(cfg, "COLOR_NORM_P", 6),
    )
    bgr   = pre["image_bgr"]
    mask2 = pre["mask"]
    cs    = pre["color_spaces"]

    # 2) 区域划分（Celebi 2007 风格）
    regions = split_lesion_regions(mask2)

    feats = {}

    # 3) 形状
    feats.update(extract_shape_features(
        mask2, n_fd=cfg.FOURIER_DESCRIPTOR_N,
        sig_bins=cfg.SIGNATURE_BINS,
    ))

    # 4) 一阶（整 lesion 区域上，多色彩空间）
    feats.update(extract_intensity_features(cs, mask2,
                                              percentiles=cfg.PERCENTILES))
    feats.update(extract_gray_intensity_features(bgr, mask2,
                                                   percentiles=cfg.PERCENTILES))

    # 5) 纹理
    feats.update(extract_texture_features(bgr, mask2, cfg))

    # 6) 【新增】区域分割一阶 (inner50/border/outer × 4 色彩空间 × 3 通道 × 8 stat)
    if getattr(cfg, "USE_REGIONAL_COLOR", True):
        feats.update(extract_regional_color_features(
            cs, regions,
            space_names=("RGB", "HSV", "Lab", "YCrCb"),
            channel_idx=(0, 1, 2),
            region_names=("inner50", "border", "outer"),
        ))

    # 7) 【新增】相对色彩 (lesion - surrounding skin) — Stanley 2007
    if getattr(cfg, "USE_RELATIVE_COLOR", True):
        feats.update(extract_relative_color_features(bgr, regions))

    # 8) 【新增】边界对比度 — border vs outer
    if getattr(cfg, "USE_BORDER_CONTRAST", True):
        feats.update(extract_border_contrast_features(bgr, regions))

    # 9) 【新增 v3】纹理特异性特征 (蓝白纱/色素网络/条纹)
    feats.update(extract_texture_specific_features(bgr, mask2, regions))

    # 10) 【新增 v3】颜色恒常性比率特征
    feats.update(extract_color_constancy_ratio_features(bgr, regions))

    return feats


def run_feature_extraction(dataset_df: pd.DataFrame,
                            cfg,
                            logger=None,
                            save_csv: bool = True) -> pd.DataFrame:
    """批量特征提取"""
    records = []
    t0 = time.time()
    n_total = len(dataset_df)
    if logger:
        logger.info(f"开始批量特征提取 (升级版 v2)，共 {n_total} 张图")
        logger.info(f"  去毛发 (DullRazor): {getattr(cfg, 'DO_HAIR_REMOVAL', True)}")
        logger.info(f"  色彩归一化: {getattr(cfg, 'COLOR_NORMALIZE', 'shades_of_gray')}")
        logger.info(f"  启用区域分割特征: {getattr(cfg, 'USE_REGIONAL_COLOR', True)}")
        logger.info(f"  启用相对色彩特征: {getattr(cfg, 'USE_RELATIVE_COLOR', True)}")
        logger.info(f"  启用边界对比度特征: {getattr(cfg, 'USE_BORDER_CONTRAST', True)}")
    fail = 0
    for idx, row in tqdm(dataset_df.iterrows(), total=n_total, ncols=80):
        try:
            f = extract_all_features_one_image(
                row["image_path"], row.get("mask_path"), cfg
            )
            f["image_id"] = row["image_id"]
            f["aug_tag"]  = row["aug_tag"]
            f["dx"]       = row["dx"]
            f["filename"] = row["filename"]
            records.append(f)
        except Exception as e:
            fail += 1
            if logger:
                logger.warning(f"图像 {row.get('filename')} 处理失败: {e}")

    df = pd.DataFrame(records)
    meta_cols = ["image_id", "aug_tag", "dx", "filename"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]

    dt = time.time() - t0
    if logger:
        logger.info(f"完成 {len(df)}/{n_total} 张图，失败 {fail} 张")
        logger.info(f"耗时 {dt:.1f}s，平均 {dt/max(1,n_total):.2f}s/张")
        logger.info(f"特征维度 = {len(feat_cols)}")

    if save_csv:
        out = cfg.RESULT_DIR / "features.csv"
        df.to_csv(out, index=False)
        if logger:
            logger.info(f"已保存: {out}")
    return df
