"""Reassess the original course implementation without replacing its models.

Reads the original code/data; all generated artifacts stay under --output.
The source instrumentation for paired predictions is recorded verbatim.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(y, pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                f1_macro=f1_score(y, pred, average="macro"),
                **dict(zip(["mel_recall", "nv_recall", "vasc_recall"],
                           recall_score(y, pred, labels=["mel", "nv", "vasc"], average=None))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["extract", "evaluate", "baselines"], required=True)
    parser.add_argument("--features", help="Optional feature table, for a separately labelled cached-data check")
    parser.add_argument("--cached", help="Optional original feature cache to compare after fresh extraction")
    parser.add_argument("--variants", nargs="+", default=["original", "fold-icc", "no-pca", "no-interactions"])
    args = parser.parse_args()
    feature_path = Path(args.features).resolve() if args.features else None
    cached_path = Path(args.cached).resolve() if args.cached else None
    source, dataset, output = [Path(p).resolve() for p in (args.source, args.data, args.output)]
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    sys.path.insert(0, str(source))
    import config as cfg
    cfg.DO_HAIR_REMOVAL = False
    cfg.RESULT_DIR = output
    cfg.DATA_ROOT = str(dataset)
    cfg.LABEL_CSV, cfg.IMAGE_DIR, cfg.MASK_DIR = dataset / "label.csv", dataset / "image", dataset / "mask"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(output / f"{args.stage}.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger("reassessment")
    from utils.io_utils import load_label_csv, build_dataset_index, read_image_bgr
    from analysis.dataset_analysis import split_meta_features, clean_features
    from analysis.robustness import compute_robustness_per_feature
    versions = {p: importlib.metadata.version(p) for p in
                ["numpy", "pandas", "scikit-learn", "scipy", "scikit-image", "opencv-python", "PyWavelets", "xgboost"]}
    source_hashes = {str(p.relative_to(source)): digest(p) for p in source.rglob("*.py")}
    (output / "environment.json").write_text(json.dumps(dict(versions=versions, source_sha256=source_hashes), indent=2), encoding="utf-8")
    parameters = {k: getattr(cfg, k) for k in dir(cfg) if k.startswith(("SCREENING_", "INTERACTION_"))
                  or k in {"RANDOM_SEED", "CV_FOLDS", "TOP_K_FEATURES", "CLINICAL_PRIORITY", "DO_HAIR_REMOVAL", "COLOR_NORMALIZE"}}
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    labels = load_label_csv(cfg.LABEL_CSV)
    index = build_dataset_index(cfg.IMAGE_DIR, cfg.MASK_DIR, labels)
    if args.stage == "extract":
        from feature_extraction.pipeline import run_feature_extraction
        from utils.io_utils import list_image_files
        index.to_csv(output / "dataset_index_private.csv", index=False)
        checks = []
        for row in index.itertuples():
            image = read_image_bgr(row.image_path)
            checks.append(dict(image_id=row.image_id, filename=row.filename,
                               pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                               file_sha256=digest(row.image_path), height=image.shape[0], width=image.shape[1]))
        manifest = pd.DataFrame(checks)
        manifest.to_csv(output / "image_manifest_private.csv", index=False)
        duplicates = manifest.groupby("pixel_sha256").image_id.nunique()
        audit = dict(label_rows=len(labels), indexed_images=len(index), independent_image_ids=index.image_id.nunique(),
                     images_on_disk=len(list_image_files(cfg.IMAGE_DIR)), masks_matched=int(index.mask_path.notna().sum()),
                     labels=index.dx.value_counts().to_dict(), views=index.aug_tag.value_counts().to_dict(),
                     duplicate_pixel_sets_across_groups=int((duplicates > 1).sum()),
                     label_file_sha256=digest(cfg.LABEL_CSV))
        logger.info("Dataset audit: %s", audit)
        (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        features = run_feature_extraction(index, cfg, logger=logger)
        if len(features) != len(index):
            raise RuntimeError("Some images failed feature extraction; evaluation aborted.")
        original_cache = cached_path or source / "outputs/no_hair_removal/results/features.csv"
        if not original_cache.exists():
            logger.info("No optional cache supplied; fresh features saved for evaluation.")
            return
        cached = pd.read_csv(original_cache, dtype={"image_id": str})
        fresh = pd.read_csv(output / "features.csv", dtype={"image_id": str})
        fresh.image_id = fresh.image_id.astype(str)
        keys = ["image_id", "aug_tag", "filename", "dx"]
        cached, fresh = [f.set_index(keys).sort_index() for f in (cached, fresh)]
        if not cached.index.equals(fresh.index) or set(cached.columns) != set(fresh.columns):
            raise RuntimeError("Dataset keys or feature schema differ from cached table.")
        fresh = fresh[cached.columns]
        a, b = cached.to_numpy(float), fresh.to_numpy(float)
        close = np.isclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True)
        comparison = dict(shape=list(a.shape), matching_cells=int(close.sum()), total_cells=int(close.size),
                          all_close=bool(close.all()), max_absolute_difference=float(np.nanmax(np.abs(a-b))),
                          changed_columns=[c for c, good in zip(cached.columns, close.all(axis=0)) if not good],
                          cached_sha256=digest(original_cache),
                          recomputed_sha256=digest(output / "features.csv"),
                          byte_identical=digest(original_cache) == digest(output / "features.csv"))
        (output / "feature_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        logger.info("Fresh/cached features: %s", comparison)
        return

    features = pd.read_csv(feature_path or output / "features.csv")
    X, y, _ = split_meta_features(features)
    X = clean_features(X, logger)
    groups = features.image_id
    if args.stage == "baselines":
        from classification.classify import get_classifiers
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.model_selection import cross_val_predict, GroupKFold
        from sklearn.pipeline import Pipeline
        rows = []
        cv = GroupKFold(cfg.CV_FOLDS)
        for name, model in get_classifiers(cfg.RANDOM_SEED).items():
            logger.info("Fold-local baseline %s", name)
            pipe = Pipeline([("select", SelectKBest(f_classif, k=cfg.TOP_K_FEATURES)), ("model", model)])
            pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv, n_jobs=1)
            rows.append(dict(model=name, **scores(y, pred)))
        pd.DataFrame(rows).to_csv(output / "fold_local_baselines.csv", index=False)
        logger.info("Baselines complete: %s", rows)
        return
    robust = compute_robustness_per_feature(features, X.columns.tolist())
    robust.to_csv(output / "robustness_recomputed.csv", index=False)
    original_robust_path = source / "outputs/no_hair_removal/results/robustness_per_feature.csv"
    if original_robust_path.exists():
        original_robust = pd.read_csv(original_robust_path)
        joined = robust.merge(original_robust, on="feature", suffixes=("_fresh", "_cached"))
        logger.info("ICC max difference: %s", (joined.icc_fresh - joined.icc_cached).abs().max())

    from classification import classify
    from analysis.feature_interaction import apply_interaction_pairs
    original_function = inspect.getsource(classify.evaluate_hierarchical)
    # Only instrumentation: the original prediction is retained, then the same
    # fitted model predicts again with label-independent interaction generation.
    needle = "        y_pred = clf.predict(X_test_eval)"
    replacement = '''        y_pred = clf.predict(X_test_eval)
        X_label_free = X_test_eval.copy()
        if interact_cols:
            all_interactions = apply_interaction_pairs(X_test_orig, sel_pairs)
            for col in interact_cols:
                X_label_free[col] = all_interactions[col].values
        corrected = clf.predict(X_label_free)
        _paired.append((test_idx.copy(), y_test.values.copy(), y_pred.copy(), corrected.copy(),
                        clf.layer2_type_, getattr(clf, "layer1_threshold_", 0.5),
                        getattr(clf, "layer2_threshold_", 0.5)))'''
    assert original_function.count(needle) == 1
    source_paired = original_function.replace(needle, replacement)
    (output / "instrumented_evaluator.py").write_text(source_paired, encoding="utf-8")
    summaries = []
    for variant in args.variants:
        logger.info("START variant %s", variant)
        scope = dict(classify.__dict__)
        scope["_paired"] = []
        instrumented = source_paired
        if variant != "original":
            # Unsupervised ICC must also exclude outer test groups. No changes
            # to mRMR, model competition, grids, threshold search or prediction.
            instrumented = instrumented.replace(
                "        # ---- Per-fold Layer 1 Feature Screener",
                "        rob_df = _fold_icc(X_train_orig, groups_train, _metadata)\n\n        # ---- Per-fold Layer 1 Feature Screener", 1)
        def fold_icc(train_X, train_groups, metadata):
            allowed = set(train_groups)
            sub = metadata[metadata.image_id.isin(allowed)]
            return compute_robustness_per_feature(sub, train_X.columns.tolist())
        scope.update(_fold_icc=fold_icc, _metadata=features)
        (output / f"instrumented_{variant}.py").write_text(instrumented, encoding="utf-8")
        exec(compile(instrumented, str(output / "instrumented_evaluator.py"), "exec"), scope)
        started = time.monotonic()
        result = scope["evaluate_hierarchical"](
            X, y, groups, rob_df=robust, clinical_priority_map=cfg.CLINICAL_PRIORITY,
            force_include_features=[c for c in X if c.startswith(("specific_", "ccr_"))],
            n_splits=cfg.CV_FOLDS, random_state=cfg.RANDOM_SEED, logger=logger,
            screen_strategy=cfg.SCREENING_STRATEGY, l1_k=cfg.SCREENING_LAYER1_TOP_K,
            l2_k=cfg.SCREENING_LAYER2_TOP_K, l1_icc=cfg.SCREENING_LAYER1_ICC_MIN,
            l2_icc=cfg.SCREENING_LAYER2_ICC_MIN, interact_m=0 if variant == "no-interactions" else cfg.INTERACTION_STRUCTURAL_M,
            interact_n=0 if variant == "no-interactions" else cfg.INTERACTION_COLOR_N,
            interact_types=cfg.INTERACTION_TYPES,
            interact_max_keep=min(cfg.INTERACTION_MAX_KEEP, max(5, int((y != "vasc").sum()) // 50)),
            max_per_channel=cfg.SCREENING_MAX_PER_CHANNEL, l1_pca=variant != "no-pca",
            l1_pca_var=cfg.SCREENING_L1_PCA_VARIANCE, l2_pca=cfg.SCREENING_L2_PCA,
            l2_pca_var=cfg.SCREENING_L2_PCA_VARIANCE)
        paired = pd.concat([pd.DataFrame(dict(row_index=idx, y_true=truth, original_pred=pred,
                                              label_free_pred=fixed, layer2_type=model,
                                              l1_threshold=t1, l2_threshold=t2, fold=fold))
                            for fold, (idx, truth, pred, fixed, model, t1, t2) in enumerate(scope["_paired"], 1)])
        paired.sort_values("row_index").to_csv(output / f"{variant}_paired_oof_private.csv", index=False)
        for prediction in ("original_pred", "label_free_pred"):
            summaries.append(dict(variant=variant, prediction=prediction, **scores(paired.y_true, paired[prediction]),
                                  elapsed_seconds=time.monotonic()-started))
        result["fold_details"].to_csv(output / f"{variant}_fold_metrics.csv", index=False)
        pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
        logger.info("RESULT %s", summaries[-2:])
    logger.info("Completed: %s", output / "summary.csv")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
