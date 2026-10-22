"""
============================================================
皮肤病变特征提取主程序 — Project 2 进阶版 v3
============================================================
对比 v1 的改进（文献依据见各模块）:
  [P1] 预处理新增 Shades-of-Gray 色彩归一化   (Barata 2015)
  [P2] 区域分割: inner50 / border / outer    (Celebi 2007)
  [P3] 相对色彩特征 (lesion - skin)          (Stanley 2007)
  [P4] 边界对比度特征                          (Celebi 2007)
  [P5] 鲁棒特征筛选消融实验 (ICC 阈值扫描)     (Xue 2021)
  [P6] 三维联合特征选择                        (v3 新增)
  [P7] 纹理特异性检测器                        (v3 新增)
  [P8] 颜色恒常性比率特征                      (v3 新增)
  [P9] 层次分类器 + 嵌套 GroupKFold            (v3 新增)
  [P10] SHAP 可解释性分析                      (v3 新增)
  [P11] 临床错误诊断                           (v3 新增)

完整流水线:
  1. 读取数据 + 索引
  2. 单张样本可视化
  3. 批量特征提取 (含新特征)
  4. 数据集级别可视化
  5. 鲁棒性分析
  6. 多分类器对比
  7. 特征重要性 + 鲁棒特征筛选消融
  【进阶模式 --advanced】
  8. 联合特征选择验证
  9. 层次分类器评估
  10. SHAP 可解释性分析
  11. 错误诊断
============================================================
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import config as cfg
from utils import (setup_logger, load_label_csv, build_dataset_index,
                    read_image_bgr, read_mask, list_image_files,
                    parse_image_id, find_mask_path)
from feature_extraction.pipeline import run_feature_extraction
from feature_extraction.preprocessing import preprocess_image

from analysis import (
    viz_preprocessing_pipeline, viz_shape_features,
    viz_color_normalization_comparison, viz_region_segmentation,
    viz_color_histograms, viz_glcm, viz_lbp, viz_gabor, viz_wavelet, viz_spectral,
    split_meta_features, clean_features,
    viz_class_distribution, viz_feature_category_count,
    viz_correlation_heatmap, viz_pca_tsne, viz_pca_variance,
    viz_top_discriminative_features,
    compute_robustness_per_feature, viz_robustness_overview,
    viz_robustness_scatter_examples,
    JointFeatureSelector, get_clinical_penalty,
    build_icc_ranks, print_top_features,
    run_shap_analysis, print_shap_summary,
    run_error_analysis,
    FeatureScreener, screen_features_layer,
    compute_layer_specific_f_ranking,
    compute_layer_specific_rf_importance,
    generate_layer2_interactions,
)
from classification import (
    evaluate_classifiers, viz_classifier_comparison,
    viz_confusion_matrices, viz_feature_importance,
    HierarchicalClassifier, evaluate_hierarchical,
    get_extended_classifiers, evaluate_layer2_ensembles,
    viz_ensemble_comparison,
)
from analysis.robustness_filter import (
    robustness_filter_experiment, viz_robustness_filter_experiment,
)


def _apply_saved_pca(X_arr, feat_names_list, pca_info):
    """使用训练时保存的 PCA 对象变换测试特征 (仅 L1 相关组)."""
    import pandas as pd

    groups = {}
    from analysis.feature_screening import _extract_channel_group
    for i, name in enumerate(feat_names_list):
        grp = _extract_channel_group(name)
        groups.setdefault(grp, []).append(i)

    pca_objects = pca_info.get("pca_objects", {})
    scaler_objects = pca_info.get("scaler_objects", {})
    parts = []
    for grp_name, indices in sorted(groups.items()):
        X_grp = X_arr[:, indices]
        if grp_name in pca_objects:
            # 使用训练时保存的 scaler + PCA 变换
            scaler = scaler_objects[grp_name]
            X_grp_scaled = scaler.transform(X_grp)
            pca = pca_objects[grp_name]
            n_keep = pca_info["group_components"].get(grp_name, 1)
            X_grp_pca = pca.transform(X_grp_scaled)[:, :n_keep]
            col_names = [f"PCA_{grp_name}_{j}" for j in range(n_keep)]
            parts.append(pd.DataFrame(X_grp_pca, columns=col_names))
        else:
            # 该组太小未做 PCA — 保留原始特征
            parts.append(pd.DataFrame(
                X_grp,
                columns=[feat_names_list[i] for i in indices],
            ))
    return pd.concat(parts, axis=1)


def run_test_mode():
    """--test 模式：加载已保存模型，对 test/image/ 下图像做预测"""
    import joblib
    import json
    import pickle
    import sys

    # 确保 test 目录存在
    cfg.TEST_IMAGE.mkdir(parents=True, exist_ok=True)
    cfg.TEST_MASK.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Test Mode — 皮肤病灶分类预测")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n[1/5] 加载模型: {cfg.MODEL_DIR}")
    if not (cfg.MODEL_DIR / "classifier.joblib").exists():
        print(f"  ERROR: 模型文件不存在！请先运行:")
        print(f"    python main.py --no-hair-removal --advanced --save-model")
        sys.exit(1)

    clf = joblib.load(cfg.MODEL_DIR / "classifier.joblib")
    with open(cfg.MODEL_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(cfg.MODEL_DIR / "config.json") as f:
        model_config = json.load(f)

    interact_pairs = []
    if (cfg.MODEL_DIR / "interaction_pairs.pkl").exists():
        with open(cfg.MODEL_DIR / "interaction_pairs.pkl", "rb") as f:
            interact_pairs = pickle.load(f)

    # v5.0: 加载 L1 PCA 对象
    pca_l1_info = None
    if (cfg.MODEL_DIR / "pca_l1.pkl").exists():
        with open(cfg.MODEL_DIR / "pca_l1.pkl", "rb") as f:
            pca_l1_info = pickle.load(f)

    print(f"  Strategy: {model_config.get('strategy')}, "
          f"L1_k={model_config.get('l1_k')}, L2_k={model_config.get('l2_k')}")
    print(f"  L2 type: {model_config.get('l2_type')}, "
          f"L1 thresh={model_config.get('l1_threshold')}, "
          f"L2 thresh={model_config.get('l2_threshold')}")
    print(f"  Feature dims: {len(feature_names)}")
    print(f"  Interaction pairs: {len(interact_pairs)}")
    if pca_l1_info:
        print(f"  L1 PCA: {pca_l1_info['total_before']}→{pca_l1_info['total_after']} "
              f"({pca_l1_info['total_after']/pca_l1_info['total_before']*100:.0f}%)")
    print(f"  Hair removal: {model_config.get('do_hair_removal')}")

    # 设置预处理参数与训练一致
    cfg.DO_HAIR_REMOVAL = model_config.get("do_hair_removal", True)
    cfg.COLOR_NORMALIZE = model_config.get("color_normalize", "shades_of_gray")

    # 2. 扫描测试图像
    print(f"\n[2/5] 扫描测试图像: {cfg.TEST_IMAGE}")
    image_files = list_image_files(str(cfg.TEST_IMAGE))
    if not image_files:
        print(f"  ERROR: {cfg.TEST_IMAGE}/ 下没有图像文件！")
        print(f"  请将测试图像放入此文件夹后重新运行。")
        sys.exit(1)

    rows = []
    for fn in image_files:
        base_id, aug_tag = parse_image_id(fn)
        rows.append({
            "image_id": base_id,
            "aug_tag": aug_tag,
            "filename": fn,
            "dx": "unknown",
            "image_path": str(cfg.TEST_IMAGE / fn),
            "mask_path": find_mask_path(str(cfg.TEST_MASK), fn),
        })
    test_ds = pd.DataFrame(rows)
    # 修复: mask_path 全为 None 时 pandas 会转为 NaN float, 导致 read_mask 报错
    test_ds["mask_path"] = test_ds["mask_path"].apply(
        lambda x: x if isinstance(x, str) else ''
    )
    n_masks = (test_ds["mask_path"] != '').sum()
    print(f"  找到 {len(test_ds)} 张图像 (含 mask: {n_masks}/{len(test_ds)})")

    # 3. 特征提取
    print(f"\n[3/5] 特征提取...")
    feat_df = run_feature_extraction(test_ds, cfg, save_csv=False)
    X, _, _ = split_meta_features(feat_df)
    X_clean = clean_features(X)

    # 对齐特征列 (训练时可能比测试多列，因特征常量被移除)
    missing = set(feature_names) - set(X_clean.columns)
    extra = set(X_clean.columns) - set(feature_names)
    if missing:
        print(f"  填充缺失特征 ({len(missing)}): {sorted(list(missing))[:5]}...")
        for c in missing:
            X_clean[c] = 0.0
    if extra:
        print(f"  丢弃多余特征 ({len(extra)})")
    X_raw = X_clean[feature_names]
    print(f"  原始特征维度: {X_raw.shape}")

    # v5.0: 非对称 PCA — 构建 hybrid features (L1: PCA, L2: raw)
    if pca_l1_info is not None:
        from analysis.feature_screening import apply_within_group_pca
        # 使用保存的 PCA 对象变换测试特征
        X_arr = X_raw.values
        feat_names_list = X_raw.columns.tolist()
        X_pca_l1 = _apply_saved_pca(X_arr, feat_names_list, pca_l1_info)
        print(f"  PCA 特征维度: {X_pca_l1.shape}")
        # Hybrid: PCA columns (L1) + raw columns (L2)
        X_final = pd.concat([
            X_pca_l1.reset_index(drop=True),
            X_raw.reset_index(drop=True),
        ], axis=1)
    else:
        X_final = X_raw

    # 4. 生成交互特征
    if interact_pairs:
        print(f"\n[4/5] 生成交互特征 ({len(interact_pairs)} pairs)...")
        from analysis.feature_interaction import apply_interaction_pairs
        # 交互特征从原始特征生成 (L2 用原始特征)
        X_interact_test = apply_interaction_pairs(X_raw, interact_pairs)
        X_final = pd.concat([X_final.reset_index(drop=True),
                             X_interact_test.reset_index(drop=True)], axis=1)
        print(f"  最终维度: {X_final.shape}")
    elif pca_l1_info is not None:
        print(f"  最终维度 (hybrid): {X_final.shape}")
    else:
        pass  # X_final already = X_raw

    # 填充交互特征中可能出现的 NaN (division by near-zero)
    if X_final.isna().any().any():
        n_nan = X_final.isna().sum().sum()
        X_final = X_final.fillna(0.0)
        print(f"  填充 NaN 值: {n_nan}")

    # 5. 预测
    print(f"\n[5/5] 预测...")
    preds = clf.predict(X_final)
    try:
        proba = clf.predict_proba(X_final)
        has_proba = True
    except Exception:
        has_proba = False

    # 输出结果
    results = test_ds[["image_id", "filename"]].copy()
    results["dx_predicted"] = preds
    if has_proba:
        results["prob_mel"] = proba[:, 0]
        results["prob_nv"] = proba[:, 1]
        results["prob_vasc"] = proba[:, 2]

    out_path = cfg.TEST_DIR / "predictions.csv"
    results.to_csv(out_path, index=False)
    print(f"\n  预测完成！结果保存到: {out_path}")
    print(f"  预测类别分布:\n{results['dx_predicted'].value_counts().to_string()}")
    print("=" * 60)


def main(args):
    # ============================================================
    # --test 模式：加载模型预测 test/image/ 下图像 (独立于训练流程)
    # ============================================================
    if args.test:
        run_test_mode()
        return

    # ============================================================
    # 应用命令行覆盖 (在创建 logger 前完成，确保日志也落到对应目录)
    # ============================================================
    if args.no_hair_removal:
        cfg.DO_HAIR_REMOVAL = False
        # 把所有输出重定向到独立子目录，避免覆盖标准 v2 的结果
        cfg.OUTPUT_ROOT = cfg.OUTPUT_ROOT / "no_hair_removal"
        cfg.FIGURE_DIR  = cfg.OUTPUT_ROOT / "figures"
        cfg.RESULT_DIR  = cfg.OUTPUT_ROOT / "results"
        cfg.LOG_DIR     = cfg.OUTPUT_ROOT / "logs"
        for d in (cfg.FIGURE_DIR, cfg.RESULT_DIR, cfg.LOG_DIR):
            d.mkdir(parents=True, exist_ok=True)
        logger_name = "skin_lesion_v2_no_hair"
    else:
        logger_name = "skin_lesion_v2"

    logger = setup_logger(logger_name, cfg.LOG_DIR)
    logger.info("=" * 60)
    logger.info("Skin Lesion Feature Extraction Pipeline (v2)")
    logger.info(f"  实验组: {'对照组 (不去毛发)' if args.no_hair_removal else '标准 v2 (去毛发)'}")
    logger.info(f"  输出目录: {cfg.OUTPUT_ROOT}")
    logger.info("=" * 60)

    # 1. 数据索引
    logger.info(f"读取标签: {cfg.LABEL_CSV}")
    label_df = load_label_csv(cfg.LABEL_CSV)
    logger.info(f"  共 {len(label_df)} 个标签, dx 分布:\n{label_df['dx'].value_counts()}")

    logger.info(f"扫描图像目录: {cfg.IMAGE_DIR}")
    ds = build_dataset_index(cfg.IMAGE_DIR, cfg.MASK_DIR, label_df)
    logger.info(f"  共找到 {len(ds)} 张图像")
    logger.info(f"  aug_tag 分布: {ds['aug_tag'].value_counts().to_dict()}")

    if args.limit:
        ids = ds["image_id"].unique()[:args.limit]
        ds = ds[ds["image_id"].isin(ids)].reset_index(drop=True)
        logger.info(f"⚠  限制为前 {args.limit} 个 image_id (共 {len(ds)} 张图)")

    n_with_mask = ds["mask_path"].notna().sum()
    logger.info(f"  含官方 mask: {n_with_mask}/{len(ds)}")

    # ============================================================
    # 2. 单张样本可视化
    # ============================================================
    if not args.skip_demo and len(ds) > 0:
        logger.info("\n[STEP 2] 单张样本的可视化演示")
        demo_dir = cfg.FIGURE_DIR / "demo_single_sample"
        demo_dir.mkdir(parents=True, exist_ok=True)
        demo_row = ds.iloc[0]
        img = read_image_bgr(demo_row["image_path"])
        mask = read_mask(demo_row["mask_path"]) if demo_row["mask_path"] else None
        pre = preprocess_image(
            img, mask, target_size=cfg.RESIZE_SHAPE,
            do_hair_removal=getattr(cfg, "DO_HAIR_REMOVAL", True),
            hair_kernel=cfg.HAIR_KERNEL_SIZE,
            inpaint_radius=cfg.HAIR_INPAINT_RADIUS,
            color_normalize=getattr(cfg, "COLOR_NORMALIZE", "shades_of_gray"),
            color_norm_p=getattr(cfg, "COLOR_NORM_P", 6),
        )
        bgr_clean = pre["image_bgr"]; mask2 = pre["mask"]

        logger.info("  画 预处理流程")
        viz_preprocessing_pipeline(img, mask, demo_dir / "01_preprocessing.png",
                                     hair_kernel=cfg.HAIR_KERNEL_SIZE,
                                     inpaint_radius=cfg.HAIR_INPAINT_RADIUS)
        logger.info("  画 色彩归一化方法对比 (v2新增)")
        viz_color_normalization_comparison(
            img, demo_dir / "01b_color_normalization_comparison.png")
        logger.info("  画 区域分割 (v2新增, Celebi 2007)")
        viz_region_segmentation(
            bgr_clean, mask2, demo_dir / "01c_region_segmentation.png")
        logger.info("  画 形状特征")
        viz_shape_features(bgr_clean, mask2, demo_dir / "02_shape.png")
        logger.info("  画 色彩直方图")
        viz_color_histograms(bgr_clean, mask2, demo_dir / "03_color_hist.png")
        logger.info("  画 GLCM")
        viz_glcm(bgr_clean, mask2, demo_dir / "04_glcm.png",
                 levels=cfg.GLCM_LEVELS)
        logger.info("  画 LBP")
        viz_lbp(bgr_clean, mask2, demo_dir / "05_lbp.png",
                radius=cfg.LBP_RADIUS, n_points=cfg.LBP_N_POINTS)
        logger.info("  画 Gabor")
        viz_gabor(bgr_clean, demo_dir / "06_gabor.png",
                   frequencies=cfg.GABOR_FREQUENCIES, thetas_deg=cfg.GABOR_THETAS)
        logger.info("  画 Wavelet")
        viz_wavelet(bgr_clean, demo_dir / "07_wavelet.png",
                     wavelet=cfg.WAVELET_NAME, level=cfg.WAVELET_LEVEL)
        logger.info("  画 频谱纹理")
        viz_spectral(bgr_clean, mask2, demo_dir / "08_spectral.png",
                      r_bins=cfg.SPECTRAL_R_BINS, theta_bins=cfg.SPECTRAL_THETA_BINS)
        logger.info(f"  demo 图已保存到 {demo_dir}")

    # ============================================================
    # 3. 批量特征提取
    # ============================================================
    logger.info("\n[STEP 3] 批量特征提取（v2: 含区域+相对色彩+边界对比度）")
    feat_csv = cfg.RESULT_DIR / "features.csv"
    if feat_csv.exists() and not args.recompute:
        logger.info(f"  发现已存在的特征文件，直接读取: {feat_csv}")
        feat_df = pd.read_csv(feat_csv)
    else:
        feat_df = run_feature_extraction(ds, cfg, logger=logger, save_csv=True)

    X, y, feat_cols = split_meta_features(feat_df)
    X_clean = clean_features(X, logger=logger)
    feat_cols_clean = X_clean.columns.tolist()
    logger.info(f"  最终特征维度 = {len(feat_cols_clean)}")

    cleaned = feat_df[["image_id", "aug_tag", "dx", "filename"] + feat_cols_clean]
    cleaned.to_csv(cfg.RESULT_DIR / "features_cleaned.csv", index=False)

    # ============================================================
    # 4. 数据集级可视化
    # ============================================================
    logger.info("\n[STEP 4] 数据集级别的特征分析")
    ds_fig = cfg.FIGURE_DIR / "dataset_analysis"
    ds_fig.mkdir(parents=True, exist_ok=True)

    viz_class_distribution(feat_df, ds_fig / "01_class_distribution.png")
    cat_counts = viz_feature_category_count(feat_cols_clean,
                                              ds_fig / "02_feature_categories.png")
    logger.info(f"  特征分类: {cat_counts}")
    viz_correlation_heatmap(X_clean, ds_fig / "03_correlation_heatmap.png",
                             top_k=min(50, len(feat_cols_clean)))
    viz_pca_variance(X_clean, ds_fig / "04_pca_variance.png", max_pc=30)
    viz_pca_tsne(X_clean, y, ds_fig / "05_pca_tsne.png",
                  random_state=cfg.RANDOM_SEED)
    rank_df = viz_top_discriminative_features(
        X_clean, y, ds_fig / "06_top_discriminative.png", top_k=15)
    if rank_df is not None:
        rank_df.to_csv(cfg.RESULT_DIR / "feature_ranking_F.csv", index=False)

    # ============================================================
    # 5. 鲁棒性分析
    # ============================================================
    logger.info("\n[STEP 5] 鲁棒性分析（原图 vs 增强）")
    rob_fig = cfg.FIGURE_DIR / "robustness"
    rob_fig.mkdir(parents=True, exist_ok=True)

    rob_df = compute_robustness_per_feature(feat_df, feat_cols_clean)
    rob_df.to_csv(cfg.RESULT_DIR / "robustness_per_feature.csv", index=False)
    valid = rob_df["icc"].notna().sum()
    if valid > 0:
        logger.info(f"  ICC 中位数 = {rob_df['icc'].median():.3f}")
        logger.info(f"  ICC ≥ 0.75 占比 = {(rob_df['icc']>=0.75).mean():.1%}")
        logger.info(f"  ICC ≥ 0.90 占比 = {(rob_df['icc']>=0.90).mean():.1%}")

    viz_robustness_overview(rob_df, rob_fig / "01_robustness_overview.png")
    viz_robustness_scatter_examples(feat_df, rob_df,
                                      rob_fig / "02_robustness_scatter.png",
                                      n_top=3, n_bot=3)

    # ============================================================
    # 6. 分类对比
    # ============================================================
    if not args.skip_classify:
        logger.info("\n[STEP 6] 多分类器对比")
        clf_fig = cfg.FIGURE_DIR / "classification"
        clf_fig.mkdir(parents=True, exist_ok=True)

        vc = y.value_counts()
        keep_cls = vc[vc >= cfg.CV_FOLDS].index.tolist()
        keep_mask = y.isin(keep_cls).values
        Xk = X_clean.loc[keep_mask].reset_index(drop=True)
        yk = y.loc[keep_mask].reset_index(drop=True)
        groups = feat_df.loc[keep_mask, "image_id"].reset_index(drop=True)

        results, X_sel = evaluate_classifiers(
            Xk, yk, groups,
            top_k_features=cfg.TOP_K_FEATURES,
            n_splits=cfg.CV_FOLDS,
            random_state=cfg.RANDOM_SEED,
            logger=logger)

        if results:
            cmp_df = viz_classifier_comparison(
                results, clf_fig / "01_classifier_comparison.png")
            cmp_df.to_csv(cfg.RESULT_DIR / "classifier_comparison.csv")
            viz_confusion_matrices(results,
                                     clf_fig / "02_confusion_matrices.png")
            imp = viz_feature_importance(X_sel, yk,
                                            clf_fig / "03_feature_importance.png",
                                            top_k=25,
                                            random_state=cfg.RANDOM_SEED)
            imp.to_csv(cfg.RESULT_DIR / "feature_importance_rf.csv",
                       header=["importance"])

        # ============================================================
        # 7.【新增】鲁棒特征筛选消融实验 (Xue 2021)
        # ============================================================
        logger.info("\n[STEP 7] 鲁棒特征筛选消融 (ICC threshold ablation)")
        rfilt_fig = cfg.FIGURE_DIR / "robustness_filter"
        rfilt_fig.mkdir(parents=True, exist_ok=True)

        rf_res = robustness_filter_experiment(
            Xk, yk, groups, rob_df,
            thresholds=(0.0, 0.5, 0.75, 0.90),
            classifier_name="SVM-RBF",
            n_splits=cfg.CV_FOLDS,
            random_state=cfg.RANDOM_SEED,
            logger=logger,
        )
        rf_res.to_csv(cfg.RESULT_DIR / "robustness_filter_results.csv", index=False)
        viz_robustness_filter_experiment(
            rf_res, rfilt_fig / "01_icc_threshold_ablation.png")
        logger.info(f"  保存到 {rfilt_fig}")

    # ============================================================
    # 【进阶模式 v4】分层特征筛选 + 交互特征 + 集成分类器 + SHAP + 错误诊断
    # ============================================================
    if not args.skip_classify and args.advanced:
        logger.info("\n" + "=" * 60)
        logger.info("进阶模式 v4: 分层筛选 + 交互特征 + 集成 + SHAP + 错误诊断")
        logger.info("=" * 60)

        # 确保 feat_df 只包含 ds 中的 image_id
        ds_ids = set(ds["image_id"].unique())
        feat_mask = feat_df["image_id"].astype(str).isin(ds_ids).values
        if not feat_mask.all() and feat_mask.sum() > 0:
            logger.info(f"  filtering feat_df to match ds: "
                        f"{feat_mask.sum()}/{len(feat_df)} rows kept")
            feat_df = feat_df.loc[feat_mask].reset_index(drop=True)
            X, y, feat_cols = split_meta_features(feat_df)
            X_clean = clean_features(X, logger=logger)
            feat_cols_clean = X_clean.columns.tolist()

        # 准备分类数据
        vc = y.value_counts()
        keep_cls = vc[vc >= cfg.CV_FOLDS].index.tolist()
        keep_mask = y.isin(keep_cls).values
        Xk = X_clean.loc[keep_mask].reset_index(drop=True)
        yk = y.loc[keep_mask].reset_index(drop=True)
        groups = feat_df.loc[keep_mask, "image_id"].reset_index(drop=True)

        # 设置进阶输出目录
        adv_root = Path("outputs") / "advanced"
        if args.no_hair_removal:
            adv_root = cfg.OUTPUT_ROOT / "advanced"
        adv_fig = adv_root / "figures"
        adv_res = adv_root / "results"
        adv_log = adv_root / "logs"
        err_dir = adv_fig / "error_analysis"
        for d in (adv_fig, adv_res, adv_log, err_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 读取鲁棒性数据
        rob_csv = cfg.RESULT_DIR / "robustness_per_feature.csv"
        rob_df = pd.read_csv(rob_csv) if rob_csv.exists() else None

        # RF importance dict (global, from STEP 6)
        rf_importance_path = cfg.RESULT_DIR / "feature_importance_rf.csv"
        rf_importance_dict = {}
        if rf_importance_path.exists():
            rf_df = pd.read_csv(rf_importance_path)
            if "feature" in rf_df.columns and "importance" in rf_df.columns:
                rf_importance_dict = dict(zip(rf_df["feature"], rf_df["importance"]))
            elif "importance" in rf_df.columns:
                for _, row in rf_df.iterrows():
                    feat = row.get("feature", row.get(rf_df.columns[0], ""))
                    imp = row["importance"]
                    if feat:
                        rf_importance_dict[str(feat)] = float(imp)

        # 分层标签
        y_vasc_bin = np.where(yk.values == "vasc", 1, 0)
        mel_nv_mask = yk.isin(["mel", "nv"]).values
        y_mel_bin = np.where(yk.values == "mel", 1, 0)[mel_nv_mask]

        # 筛选策略参数
        screen_strategy = getattr(cfg, "SCREENING_STRATEGY", "rrf")
        l1_k = getattr(cfg, "SCREENING_LAYER1_TOP_K", 30)
        l2_k = getattr(cfg, "SCREENING_LAYER2_TOP_K", 50)
        l1_icc = getattr(cfg, "SCREENING_LAYER1_ICC_MIN", 0.50)
        l2_icc = getattr(cfg, "SCREENING_LAYER2_ICC_MIN", 0.75)
        interact_m = getattr(cfg, "INTERACTION_STRUCTURAL_M", 15)
        interact_n = getattr(cfg, "INTERACTION_COLOR_N", 15)
        interact_max_keep = getattr(cfg, "INTERACTION_MAX_KEEP", 40)
        interact_types = getattr(cfg, "INTERACTION_TYPES", ["product", "ratio"])

        # 自动校准：交互特征数不应超过 mel/nv 样本数的 1/50
        n_mel_nv = mel_nv_mask.sum()
        interact_max_keep = min(interact_max_keep, max(5, n_mel_nv // 50))

        # 预先定义 mel/nv 子集，供后续所有步骤使用
        X_mn_full = Xk.loc[mel_nv_mask]
        groups_mn = groups[mel_nv_mask].values

        # ============================================================
        # STEP 8a: 分层 F-test 排名计算
        # ============================================================
        logger.info("\n[ADV STEP 8a] 分层 F-test + RF 排名计算")
        if not getattr(args, "no_screen", False):
            layer1_f_ranking = compute_layer_specific_f_ranking(
                Xk, y_vasc_bin, Xk.columns.tolist(),
                save_path=adv_res / "feature_ranking_F_layer1.csv",
            )
            layer2_f_ranking = compute_layer_specific_f_ranking(
                Xk.loc[mel_nv_mask], y_mel_bin, Xk.columns.tolist(),
                save_path=adv_res / "feature_ranking_F_layer2.csv",
            )
            layer1_rf_imp, _ = compute_layer_specific_rf_importance(
                Xk, y_vasc_bin, Xk.columns.tolist(),
                random_state=cfg.RANDOM_SEED,
                save_path=adv_res / "feature_importance_rf_layer1.csv",
            )
            layer2_rf_imp, _ = compute_layer_specific_rf_importance(
                Xk.loc[mel_nv_mask], y_mel_bin, Xk.columns.tolist(),
                random_state=cfg.RANDOM_SEED,
                save_path=adv_res / "feature_importance_rf_layer2.csv",
            )
            logger.info(f"  Layer 1 F-ranking top-5: "
                        f"{layer1_f_ranking['feature'].head(5).tolist()}")
            logger.info(f"  Layer 2 F-ranking top-5: "
                        f"{layer2_f_ranking['feature'].head(5).tolist()}")
        else:
            # 使用全局排名作为回退
            layer1_rf_imp = rf_importance_dict.copy()
            layer2_rf_imp = rf_importance_dict.copy()

        # ============================================================
        # STEP 8b: 分层特征筛选 (全量数据, 诊断参考用; CV 评估在 fold 内独立做)
        # ============================================================
        logger.info(f"\n[ADV STEP 8b] 分层双排序融合筛选 (strategy={screen_strategy})")
        if not getattr(args, "no_screen", False):
            # Layer 1 筛选: vasc vs non-vasc (诊断参考)
            X_screened_l1, report_l1, selected_l1 = screen_features_layer(
                Xk, y_vasc_bin,
                rf_importance_dict=layer1_rf_imp,
                rob_df=rob_df, strategy=screen_strategy,
                top_k=l1_k, layer=1, icc_min=l1_icc,
                clinical_priority_map=cfg.CLINICAL_PRIORITY,
                logger=logger,
            )
            report_l1.to_csv(adv_res / "screening_report_layer1.csv", index=False)

            # Layer 2 筛选: mel vs nv (诊断参考)
            X_screened_l2, report_l2, selected_l2 = screen_features_layer(
                Xk.loc[mel_nv_mask], y_mel_bin,
                rf_importance_dict=layer2_rf_imp,
                rob_df=rob_df, strategy=screen_strategy,
                top_k=l2_k, layer=2, icc_min=l2_icc,
                clinical_priority_map=cfg.CLINICAL_PRIORITY,
                channel_diversity=True,
                max_per_channel=cfg.SCREENING_MAX_PER_CHANNEL,
                logger=logger,
            )
            report_l2.to_csv(adv_res / "screening_report_layer2.csv", index=False)

            logger.info(f"  Layer 1 selected: {len(selected_l1)} features")
            logger.info(f"  Layer 2 selected: {len(selected_l2)} features")
        else:
            selected_l1 = Xk.columns.tolist()
            selected_l2 = Xk.columns.tolist()

        # ============================================================
        # STEP 8c: Layer-2 交互特征工程 (全量数据, 诊断参考用; CV 评估在 fold 内独立做)
        # ============================================================
        logger.info("\n[ADV STEP 8c] Layer-2 跨族交互特征工程")
        X_interact_full = None
        interact_report = {}
        if not getattr(args, "no_interact", False):
            X_interact_full, interact_report = generate_layer2_interactions(
                X_mn_full, y_mel_bin,
                f_ranking_path=adv_res / "feature_ranking_F_layer2.csv",
                rf_importance_path=adv_res / "feature_importance_rf_layer2.csv",
                n_structural=interact_m, n_color=interact_n,
                interaction_types=interact_types, max_keep=interact_max_keep,
                random_state=cfg.RANDOM_SEED, logger=logger,
            )
            if interact_report.get("mi_ranking") is not None:
                interact_report["mi_ranking"].to_csv(
                    adv_res / "interaction_mi_ranking.csv", index=False)
            if interact_report.get("family_assignment") is not None:
                interact_report["family_assignment"].to_csv(
                    adv_res / "interaction_family_assignment.csv", index=False)

            logger.info(f"  Interaction features: {X_interact_full.shape[1]} selected")

        # ============================================================
        # STEP 9: 层次分类器评估 (CV-leak-free: 每折独立做筛选+交互)
        # ============================================================
        logger.info("\n[ADV STEP 9] 层次分类器评估 (per-fold 筛选 + 交互)")

        force_feats = [c for c in Xk.columns
                       if c.startswith("specific_") or c.startswith("ccr_")]

        hier_results = evaluate_hierarchical(
            Xk, yk, groups,
            rob_df=rob_df,
            clinical_priority_map=cfg.CLINICAL_PRIORITY,
            force_include_features=force_feats,
            n_splits=cfg.CV_FOLDS,
            random_state=cfg.RANDOM_SEED,
            logger=logger,
            screen_strategy=screen_strategy,
            l1_k=l1_k, l2_k=l2_k, l1_icc=l1_icc, l2_icc=l2_icc,
            interact_m=interact_m, interact_n=interact_n,
            interact_types=interact_types, interact_max_keep=interact_max_keep,
            max_per_channel=cfg.SCREENING_MAX_PER_CHANNEL,
            l1_pca=cfg.SCREENING_L1_PCA,
            l1_pca_var=cfg.SCREENING_L1_PCA_VARIANCE,
            l2_pca=cfg.SCREENING_L2_PCA,
            l2_pca_var=cfg.SCREENING_L2_PCA_VARIANCE,
        )

        # 打印对比
        logger.info("\n" + "-" * 50)
        logger.info("v4 层次分类器 vs v3 基线对比")
        logger.info("-" * 50)
        logger.info(f"  v4 Hierarchical: Acc={hier_results['accuracy']:.3f}, "
                    f"BalAcc={hier_results['balanced_acc']:.3f}, "
                    f"F1m={hier_results['f1_macro']:.3f}")
        logger.info(f"  Per-class precision: {hier_results['per_class_precision']}")
        logger.info(f"  Per-class recall: {hier_results['per_class_recall']}")

        if results and "SVM-RBF" in results:
            svm = results["SVM-RBF"]
            logger.info(f"  SVM-RBF baseline: Acc={svm['accuracy']:.3f}, "
                        f"BalAcc={svm['balanced_acc']:.3f}, "
                        f"F1m={svm['f1_macro']:.3f}")
            delta_f1 = hier_results["f1_macro"] - svm["f1_macro"]
            logger.info(f"  ΔF1-macro (v4-SVM) = {delta_f1:+.3f}")

        pd.DataFrame([{
            "accuracy": hier_results["accuracy"],
            "balanced_acc": hier_results["balanced_acc"],
            "f1_macro": hier_results["f1_macro"],
            "per_class_precision": str(hier_results["per_class_precision"]),
            "per_class_recall": str(hier_results["per_class_recall"]),
        }]).to_csv(adv_res / "hierarchical_results.csv", index=False)

        if "fold_details" in hier_results:
            hier_results["fold_details"].to_csv(
                adv_res / "hierarchical_fold_details.csv", index=False)

        # ============================================================
        # STEP 9b: Layer-2 集成分类器评估
        # ============================================================
        if not getattr(args, "no_ensemble", False):
            logger.info("\n[ADV STEP 9b] Layer-2 集成分类器评估")

            groups_mn_series = pd.Series(groups_mn)

            if not getattr(args, "no_screen", False) and selected_l2:
                X_l2_combined = Xk.loc[mel_nv_mask, selected_l2].reset_index(drop=True)
            else:
                X_l2_combined = X_mn_full

            if X_interact_full is not None and X_interact_full.shape[1] > 0:
                X_l2_combined = pd.concat(
                    [X_l2_combined, X_interact_full.reset_index(drop=True)], axis=1)

            logger.info(f"  Layer-2 combined features: {X_l2_combined.shape[1]}")

            ensemble_results = evaluate_layer2_ensembles(
                X_l2_combined, y_mel_bin, groups_mn,
                n_splits=cfg.CV_FOLDS,
                random_state=cfg.RANDOM_SEED,
                logger=logger,
            )
            ensemble_df = viz_ensemble_comparison(
                ensemble_results, adv_fig / "ensemble_comparison.png")
            ensemble_df.to_csv(adv_res / "ensemble_results.csv", index=False)
            logger.info(f"  Ensemble results saved to {adv_res}")

        # ============================================================
        # STEP 10: SHAP 可解释性分析
        # ============================================================
        logger.info("\n[ADV STEP 10] SHAP 可解释性分析")

        shap_result = {}
        if len(X_mn_full) > 10:
            # 使用 Layer-2 筛选+交互特征
            if not getattr(args, "no_screen", False):
                X_shap = Xk.loc[mel_nv_mask, selected_l2].reset_index(drop=True)
            else:
                X_shap = Xk.loc[mel_nv_mask].reset_index(drop=True)
            if X_interact_full is not None and X_interact_full.shape[1] > 0:
                X_shap = pd.concat(
                    [X_shap, X_interact_full.reset_index(drop=True)], axis=1)

            import sklearn.preprocessing
            import sklearn.pipeline
            import sklearn.svm
            model_type = "svm"
            try:
                import xgboost as xgb
                scale_weight = (y_mel_bin == 0).sum() / max((y_mel_bin == 1).sum(), 1)
                model = sklearn.pipeline.Pipeline([
                    ("sc", sklearn.preprocessing.StandardScaler()),
                    ("clf", xgb.XGBClassifier(
                        objective="binary:logistic",
                        scale_pos_weight=scale_weight,
                        max_depth=5, n_estimators=200,
                        random_state=cfg.RANDOM_SEED,
                        use_label_encoder=False, eval_metric="logloss",
                        verbosity=0)),
                ])
                le = sklearn.preprocessing.LabelEncoder()
                y_mn_enc = le.fit_transform(y_mel_bin)
                model.fit(X_shap.values, y_mn_enc)
                model_type = "xgboost"
                logger.info("  Using XGBoost for SHAP (TreeExplainer)")
            except (ImportError, Exception) as e:
                logger.info(f"  XGBoost unavailable ({e}), using SVM")
                le = sklearn.preprocessing.LabelEncoder()
                y_mn_enc = le.fit_transform(y_mel_bin)
                model = sklearn.pipeline.Pipeline([
                    ("sc", sklearn.preprocessing.StandardScaler()),
                    ("clf", sklearn.svm.SVC(kernel="rbf", C=10, gamma="scale",
                                           probability=True, class_weight="balanced",
                                           random_state=cfg.RANDOM_SEED)),
                ])
                model.fit(X_shap.values, y_mn_enc)

            train_n = min(150, len(X_shap))
            test_n = min(200, len(X_shap))
            shap_result = run_shap_analysis(
                model, X_shap.iloc[:train_n], X_shap.iloc[:test_n],
                output_dir=adv_fig, model_type=model_type,
                logger=logger, n_background=100, n_test_samples=200,
            )
            print_shap_summary(shap_result, logger=logger)

        # ============================================================
        # STEP 11: 临床错误诊断
        # ============================================================
        logger.info("\n[ADV STEP 11] 临床错误诊断")

        y_true_arr = np.array(hier_results["y_true"])
        y_pred_arr = np.array(hier_results["y_pred"])

        # 将 y_true/y_pred 从 GroupKFold 累积顺序重排为 DataFrame 原始顺序
        # (test_indices[j] = 原始 DataFrame 中第 j 个 test 样本的位置)
        test_indices = hier_results.get("test_indices", list(range(len(y_true_arr))))
        y_true_df_order = np.empty_like(y_true_arr)
        y_pred_df_order = np.empty_like(y_pred_arr)
        for fold_pos, df_pos in enumerate(test_indices):
            y_true_df_order[df_pos] = y_true_arr[fold_pos]
            y_pred_df_order[df_pos] = y_pred_arr[fold_pos]
        y_true_arr = y_true_df_order
        y_pred_arr = y_pred_df_order

        proba = None
        try:
            # 全量数据上构建 Screeners 用于最终诊断模型
            from analysis.feature_screening import FeatureScreener, apply_within_group_pca

            # v5.0: 非对称 PCA — L1 做 PCA, L2 用原始特征
            Xk_l1 = Xk.copy()
            final_pca_info = None
            if cfg.SCREENING_L1_PCA:
                Xk_arr = Xk.values
                feat_names = Xk.columns.tolist()
                Xk_l1, _, final_pca_info = apply_within_group_pca(
                    Xk_arr, Xk_arr, feat_names,  # train=test (全量 fit)
                    variance_threshold=cfg.SCREENING_L1_PCA_VARIANCE,
                    min_group_size=3,
                )

            final_l1 = FeatureScreener(
                rf_importance_dict=layer1_rf_imp, rob_df=rob_df,
                strategy=screen_strategy, top_k=l1_k, layer=1, icc_min=l1_icc,
                clinical_priority_map=cfg.CLINICAL_PRIORITY,
            )
            final_l1.fit(Xk_l1, y_vasc_bin)  # L1: PCA 特征

            final_l2 = FeatureScreener(
                rf_importance_dict=layer2_rf_imp, rob_df=rob_df,
                strategy=screen_strategy, top_k=l2_k, layer=2, icc_min=l2_icc,
                clinical_priority_map=cfg.CLINICAL_PRIORITY,
            )
            final_l2.fit(Xk.loc[mel_nv_mask], y_mel_bin)  # L2: 原始特征

            # 构建 hybrid X: PCA 列 (L1) + 原始列 (L2)
            Xk_hybrid = pd.concat([
                Xk_l1.reset_index(drop=True),
                Xk.reset_index(drop=True),
            ], axis=1)

            final_clf = HierarchicalClassifier(
                random_state=cfg.RANDOM_SEED, logger=None,
                force_include_features=force_feats,
            )
            final_clf.fit(Xk_hybrid, yk.values, groups=groups.values,
                          layer1_selector=final_l1,
                          layer2_selector=final_l2,
                          X2_interact=X_interact_full)
            # 将交互特征 join 到 hybrid X 中供 predict_proba 使用
            Xk_for_proba = Xk_hybrid
            if X_interact_full is not None and X_interact_full.shape[1] > 0:
                Xk_for_proba = Xk_hybrid.join(X_interact_full, how="left")
                for c in X_interact_full.columns:
                    Xk_for_proba[c] = Xk_for_proba[c].fillna(0.0)
            proba = final_clf.predict_proba(Xk_for_proba)
        except Exception as e:
            logger.warning(f"  predict_proba failed: {e}")

        # SHAP top features
        shap_top = []
        if shap_result.get("top_features"):
            shap_top = [f for f, _ in shap_result["top_features"][:10]]
        else:
            from sklearn.feature_selection import f_classif
            le = LabelEncoder()
            y_enc = le.fit_transform(yk.values)
            f_vals, _ = f_classif(Xk.values, y_enc)
            top_idx = np.argsort(-f_vals)[:10]
            shap_top = [Xk.columns[i] for i in top_idx]

        # path map
        path_map = {}
        for _, r in ds.iterrows():
            key = (str(r["image_id"]), str(r["filename"]))
            path_map[key] = (str(r.get("image_path", "")), str(r.get("mask_path", "")))
        feat_rows = feat_df.loc[Xk.index].copy().reset_index(drop=True)
        feat_rows["image_path"] = feat_rows.apply(
            lambda r: path_map.get(
                (str(r["image_id"]), str(r["filename"])), ("", ""))[0], axis=1)
        feat_rows["mask_path"] = feat_rows.apply(
            lambda r: path_map.get(
                (str(r["image_id"]), str(r["filename"])), ("", ""))[1], axis=1)

        err_result = run_error_analysis(
            y_true_arr, y_pred_arr, proba, Xk,
            feat_rows,
            top_features=shap_top,
            output_dir=err_dir,
            class_names=sorted(yk.unique()),
            max_cases=10,
            logger=logger,
        )
        err_result.get("misclassified_df", pd.DataFrame()).to_csv(
            adv_res / "error_misclassified.csv", index=False)

        logger.info("\n" + "=" * 60)
        logger.info("v4 Advanced Pipeline Complete.")
        logger.info(f"  Screening: {adv_res / 'screening_report_layer*.csv'}")
        logger.info(f"  Interaction: {adv_res / 'interaction_mi_ranking.csv'}")
        logger.info(f"  Hierarchical: {adv_res / 'hierarchical_results.csv'}")
        logger.info(f"  Ensemble: {adv_res / 'ensemble_results.csv'}")
        logger.info("=" * 60)

        # ============================================================
        # 保存最终模型 (--save-model)
        # ============================================================
        if args.save_model:
            import joblib
            import json

            cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"\n[SAVE] 保存最终模型到 {cfg.MODEL_DIR}")

            # 保存 HierarchicalClassifier
            joblib.dump(final_clf, cfg.MODEL_DIR / "classifier.joblib")
            logger.info("  classifier.joblib")

            # 保存 feature names
            with open(cfg.MODEL_DIR / "feature_names.json", "w") as f:
                json.dump(feat_cols_clean, f)
            logger.info(f"  feature_names.json ({len(feat_cols_clean)} features)")

            # 保存交互特征对
            if X_interact_full is not None and interact_report:
                import pickle
                pairs = interact_report.get("selected_pairs", [])
                with open(cfg.MODEL_DIR / "interaction_pairs.pkl", "wb") as f:
                    pickle.dump(pairs, f)
                logger.info(f"  interaction_pairs.pkl ({len(pairs)} pairs)")

            # 保存 PCA 对象 (v5.0: 非对称 PCA, 仅 L1)
            if final_pca_info is not None:
                import pickle
                with open(cfg.MODEL_DIR / "pca_l1.pkl", "wb") as f:
                    pickle.dump(final_pca_info, f)
                logger.info(f"  pca_l1.pkl "
                            f"({final_pca_info['total_before']}→{final_pca_info['total_after']})")

            # 保存配置摘要
            config_summary = {
                "strategy": screen_strategy,
                "l1_k": l1_k, "l2_k": l2_k,
                "l1_pca": getattr(cfg, "SCREENING_L1_PCA", True),
                "l2_pca": getattr(cfg, "SCREENING_L2_PCA", False),
                "do_hair_removal": getattr(cfg, "DO_HAIR_REMOVAL", True),
                "color_normalize": getattr(cfg, "COLOR_NORMALIZE", "shades_of_gray"),
                "l1_threshold": getattr(final_clf, "layer1_threshold_", 0.5),
                "l2_threshold": getattr(final_clf, "layer2_threshold_", 0.5),
                "l2_type": getattr(final_clf, "layer2_type_", "unknown"),
            }
            with open(cfg.MODEL_DIR / "config.json", "w") as f:
                json.dump(config_summary, f, indent=2)
            logger.info("  config.json")

    logger.info("\n" + "=" * 60)
    logger.info("All done.")
    logger.info(f"图表: {cfg.FIGURE_DIR}")
    logger.info(f"结果数据: {cfg.RESULT_DIR}")
    if args.advanced:
        logger.info(f"进阶输出: {Path('outputs')/'advanced'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="只处理前 N 个 image_id（调试用）")
    ap.add_argument("--recompute", action="store_true",
                    help="强制重新计算特征 (默认: 若 features.csv 存在则复用)")
    ap.add_argument("--skip-demo", action="store_true",
                    help="跳过单图 demo 可视化")
    ap.add_argument("--skip-classify", action="store_true",
                    help="跳过分类评估")
    ap.add_argument("--no-hair-removal", action="store_true",
                    help="对照实验：关闭 DullRazor 去毛发。"
                         "所有输出会落到 outputs/no_hair_removal/ 子目录，"
                         "不会覆盖标准 v2 的结果。")
    ap.add_argument("--advanced", action="store_true",
                    help="进阶模式：联合特征选择 + 层次分类器 + SHAP + 错误诊断")
    ap.add_argument("--no-screen", action="store_true",
                    help="跳过 v4 分层特征筛选，回退到 v3 JointFeatureSelector")
    ap.add_argument("--no-interact", action="store_true",
                    help="跳过 v4 Layer-2 交互特征工程")
    ap.add_argument("--no-ensemble", action="store_true",
                    help="跳过 v4 Layer-2 集成分类器评估")
    ap.add_argument("--save-model", action="store_true",
                    help="训练完成后保存最终模型到 models/ (供 --test 使用)")
    ap.add_argument("--test", action="store_true",
                    help="测试模式：加载 models/ 中模型，预测 test/image/ 下图像")
    args = ap.parse_args()
    main(args)
