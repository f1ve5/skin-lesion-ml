# 原课程方法：完整代码

此目录保留原 `proj_v2_ablation` 的主程序、分类、分析、特征提取和工具模块。除 `config.py` 将本机数据路径改为环境变量外，29 个 Python 文件的其余内容保持原样。校验记录见 `source_manifest.json`。

分类工作是个人贡献；特征提取与图像预处理模块作为团队上游依赖保留，不归为个人独立成果。

## 推荐复核入口

在仓库根目录安装包含图像处理及 XGBoost 的依赖：

```bash
python -m pip install -e ".[original]"
```

将自己的原数据集放在私有目录（包含 `image/`、`mask/`、`label.csv`），然后执行：

```bash
python reassessment/assess_original.py --source original_method --data data/private/Data_Proj2 --output outputs/original-reassessment --stage extract
python reassessment/assess_original.py --source original_method --data data/private/Data_Proj2 --output outputs/original-reassessment --stage evaluate
python reassessment/assess_original.py --source original_method --data data/private/Data_Proj2 --output outputs/original-reassessment --stage baselines
```

原始图像重提使用不去毛发、Shades-of-Gray 归一化与原提取参数。评估保留原 mRMR、ICC、临床先验、两层分类、XGBoost/SVM 择优、SVM 参数网格和阈值策略；不是 `src/skin_lesion_ml/` 中简化后的工程示例。

`evaluate` 默认执行原配置、训练折 ICC 配置及两项消融。每项都对同一批训练出的模型生成两套预测：原测试交互特征逻辑，以及对所有测试行计算交互特征的无标签逻辑。输出 `summary.csv` 中应以 `label_free_pred` 查看无标签推理的指标。逐样本文件和特征缓存保留在被忽略的 `outputs/` 中。

原代码的 `l2_pca` 参数当前未被评估函数实际用于变换二层数据；本次固定为原最优配置的 False，不用它制造“全局 PCA”对照。复核记录会列明其余实现限制。

## 原主程序入口

如需原来的图表、分析和训练流程，在此目录运行：

```powershell
$env:SKIN_LESION_DATA_ROOT = '你的数据集绝对路径'
python main.py --no-hair-removal --advanced --skip-demo --no-ensemble
```

主程序入口保留历史行为；本次结果复核使用上面的专用入口，以避免图表和可选 SHAP 等步骤干扰实验记录。没有随仓库分发原始数据或预训练模型。
