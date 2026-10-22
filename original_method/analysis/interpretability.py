"""
SHAP 可解释性分析 (含交互效应)
===============================
在 Layer 2 (mel vs nv) 模型上计算 SHAP 值，生成：
  1. shap_summary_plot.png — 最重要的 15 个特征及影响方向
  2. shap_interaction_*.png — 交互强度最高的 3 对特征的 dependence plot
  3. 控制台输出交互对的临床解读

文献依据：
  - Lundberg & Lee 2017: SHAP unified framework
  - 课程要求: 模型可解释性与临床错误分析
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .viz_features import setup_style


def _interpret_interaction(feat1: str, feat2: str) -> str:
    """根据特征名给出交互对的临床解读"""
    interpretations = []

    # 蓝白纱相关
    if "bord_blur_L" in (feat1, feat2) and any("Lab_b" in f for f in (feat1, feat2)):
        interpretations.append(
            "边界模糊度与Lab-b的交互提示模型识别了蓝白纱(blue-whitish veil)特征: "
            "边界模糊且b通道极值为负的病灶更可能是黑色素瘤")

    # 色素网络相关
    if "specific_pigment" in (feat1, feat2):
        other = feat2 if "specific_pigment" in feat1 else feat1
        interpretations.append(
            f"色素网络密度与{other}的交互反映了色素网络在特定色彩/纹理条件下的判别力增强")

    # 条纹一致性
    if "specific_streak" in (feat1, feat2):
        other = feat2 if "specific_streak" in feat1 else feat1
        interpretations.append(
            f"条纹方向一致性与{other}的交互提示放射状条纹在特定结构背景下的诊断价值")

    # 颜色-边界交互
    if any("HSV_H" in f for f in (feat1, feat2)) and any("bord_" in f for f in (feat1, feat2)):
        interpretations.append(
            "HSV色调与边界特征的强交互表明: 边界不规则性+特定色调组合是黑色素瘤的关键信号")

    # 区域对比度交互
    if any("inner50" in f for f in (feat1, feat2)) and any("border" in f for f in (feat1, feat2)):
        interpretations.append(
            "核心区与边界区的特征交互捕捉了病灶从核心到边缘的颜色渐变模式")

    # 默认解读
    if not interpretations:
        interpretations.append(
            f"{feat1} 与 {feat2} 的强交互提示两者在模型决策中存在非线性的联合效应, "
            "建议在临床评估中同时关注这两个维度")

    return " | ".join(interpretations)


def run_shap_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: np.ndarray = None,
                      output_dir=None,
                      model_type="xgboost",
                      logger=None,
                      n_background=100,
                      n_test_samples=200) -> dict:
    """
    执行 SHAP 分析并生成可视化。

    Parameters
    ----------
    model : 已训练的 sklearn 模型
    X_train : 训练特征 (用于背景样本)
    X_test : 测试特征 (用于解释)
    model_type : "xgboost", "svm", 或 "linear"
    output_dir : Path, 输出目录
    n_background : KernelExplainer 背景样本数
    n_test_samples : KernelExplainer 测试样本数

    Returns
    -------
    dict with keys: shap_values, top_features, interactions, interaction_pairs
    """
    setup_style()
    result = {}

    if output_dir is None:
        return result

    output_dir = output_dir if isinstance(output_dir, type(output_dir)) else output_dir
    import pathlib
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap
    except ImportError:
        if logger:
            logger.warning("shap 未安装，跳过 SHAP 分析。pip install shap")
        return result

    if logger:
        logger.info(f"Computing SHAP values (model_type={model_type})...")

    # 数据准备
    feat_names = X_train.columns.tolist()
    X_train_arr = X_train.values
    X_test_arr = X_test.values[:n_test_samples] if len(X_test) > n_test_samples else X_test.values

    try:
        if model_type == "xgboost":
            # TreeExplainer: 快速精确
            explainer = shap.TreeExplainer(
                model.named_steps["clf"] if hasattr(model, "named_steps") else model
            )
            # 获取 SHAP 值
            if hasattr(model, "named_steps"):
                X_proc = model.named_steps["sc"].transform(X_test_arr)
                shap_values = explainer.shap_values(X_proc)
            else:
                shap_values = explainer.shap_values(X_test_arr)

            # TreeExplainer 对二分类返回 list of 2 arrays
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 取正类 (mel)

        else:
            # KernelExplainer: 适用于 SVM/线性模型
            n_bg = min(n_background, len(X_train_arr))
            background = X_train_arr[:n_bg]
            if hasattr(model, "named_steps"):
                background = model.named_steps["sc"].transform(background)
                X_test_proc = model.named_steps["sc"].transform(X_test_arr)

                def predict_fn(x):
                    return model.named_steps["clf"].predict_proba(x)
            else:
                X_test_proc = X_test_arr

                def predict_fn(x):
                    return model.predict_proba(x)

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_test_proc, nsamples=100)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        result["shap_values"] = shap_values
        result["feature_names"] = feat_names

        # ---- Summary Plot ----
        if logger:
            logger.info("Generating SHAP summary plot...")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_test_arr,
            feature_names=feat_names,
            max_display=15, show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_plot.png", dpi=150,
                    bbox_inches="tight")
        plt.close("all")

        # ---- Feature Importance from SHAP ----
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(-mean_abs_shap)[:20]
        result["top_features"] = [(feat_names[i], mean_abs_shap[i]) for i in top_idx]

        # ---- SHAP Interaction Values ----
        if model_type == "xgboost":
            if logger:
                logger.info("Computing SHAP interaction values...")
            try:
                if hasattr(model, "named_steps"):
                    X_proc = model.named_steps["sc"].transform(X_test_arr)
                    shap_interaction = explainer.shap_interaction_values(X_proc)
                else:
                    shap_interaction = explainer.shap_interaction_values(X_test_arr)

                # 计算交互强度矩阵
                n_features = shap_interaction.shape[1]
                interaction_strength = np.zeros((n_features, n_features))
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        # 交互强度 = 非对角元素的平均绝对值
                        strength = np.abs(shap_interaction[:, i, j]).mean()
                        interaction_strength[i, j] = strength
                        interaction_strength[j, i] = strength

                # 找交互最强的 3 对特征
                pairs = []
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        pairs.append((i, j, interaction_strength[i, j]))
                pairs.sort(key=lambda x: -x[2])
                top_pairs = pairs[:3]

                result["interaction_pairs"] = []
                for rank, (i, j, strength) in enumerate(top_pairs):
                    f1, f2 = feat_names[i], feat_names[j]
                    result["interaction_pairs"].append((f1, f2, strength))

                    # Dependence plot
                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.dependence_plot(
                        i, shap_interaction, X_test_arr,
                        feature_names=feat_names,
                        interaction_index=j,
                        show=False, ax=ax,
                    )
                    ax.set_title(f"SHAP Interaction: {f1} ↔ {f2}\n"
                                 f"(interaction strength={strength:.4f})",
                                 fontsize=11)
                    plt.tight_layout()
                    plt.savefig(
                        output_dir / f"shap_interaction_{rank+1}.png",
                        dpi=150, bbox_inches="tight",
                    )
                    plt.close("all")

                    # 临床解读
                    interpretation = _interpret_interaction(f1, f2)
                    if logger:
                        logger.info(f"  Interaction #{rank+1}: {f1} ↔ {f2}")
                        logger.info(f"    Strength: {strength:.4f}")
                        logger.info(f"    Interpretation: {interpretation}")

                result["interaction_matrix"] = interaction_strength

            except Exception as e:
                if logger:
                    logger.warning(f"SHAP interaction computation failed: {e}")

        result["status"] = "ok"

    except Exception as e:
        if logger:
            logger.warning(f"SHAP analysis failed: {e}")
        result["status"] = "error"
        result["error"] = str(e)

    return result


def print_shap_summary(shap_result: dict, logger=None):
    """打印 SHAP 分析摘要"""
    if shap_result.get("status") != "ok":
        return

    log = print if logger is None else logger.info

    log("\n" + "=" * 60)
    log("SHAP Analysis Summary")
    log("=" * 60)

    if shap_result.get("top_features"):
        log("\nTop-10 most important features (by mean |SHAP|):")
        for rank, (feat, importance) in enumerate(shap_result["top_features"][:10]):
            log(f"  {rank+1:2d}. {feat:<50s} {importance:.4f}")

    if shap_result.get("interaction_pairs"):
        log("\nTop-3 feature interactions:")
        for rank, (f1, f2, strength) in enumerate(shap_result["interaction_pairs"]):
            log(f"\n  #{rank+1}: {f1} ↔ {f2}")
            log(f"       Strength: {strength:.4f}")
            log(f"       Clinical: {_interpret_interaction(f1, f2)}")

    log("=" * 60)
