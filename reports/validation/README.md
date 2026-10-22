> 此处记录此前的简化工程示例。原方法的数据复核、原配置复现和对照以 [复核报告](../reassessment/README.md) 为准。

# 整理后验证记录

## 软件验证

- 7 项回归测试通过：独立组隔离与每行一次 OOF 覆盖、元数据排除、输入错误拒绝、缺失/常数/单列特征处理、无标签层次推理、推理批次一致性、序列化与列顺序对齐，以及内层搜索逐折重新拟合特征处理。
- 合成数据默认 mRMR 路径：8 个模型/消融配置，外层 3 折、内层 2 折，全部完成。
- 合成数据训练 → 保存 → 加载 → CSV 预测命令通过。
- Python wheel 构建及从 wheel 导入包通过。
- GitHub Actions 配置已提供；尚未在远端触发执行。

## 真实特征表本地验证

来源：原项目 `outputs/no_hair_removal/results/features.csv`。600 行、200 个 image_id、874 个数值特征。使用已有特征表，不重新提取图像特征。文件 SHA-256 与依赖版本记录在 `real-feature-table/protocol.json`。

复现命令（将原始特征表自行放入指定位置）：

```bash
python -m skin_lesion_ml evaluate --data data/private/features.csv --models logreg svm random-forest gradient-boosting knn hierarchical hierarchical-no-pca hierarchical-no-interactions --selection f --folds 5 --inner-folds 4 --k 30 --output outputs/local-validation
```

所有配置共用外层划分，内层目标为 Macro-F1。一次运行完成，不根据这些外层成绩继续修改配置。`real-feature-table/` 仅保存不含逐样本 ID 的汇总、混淆矩阵、分类报告、参数与图表；逐行 OOF 与划分明细只保存在本地被忽略的 `outputs/` 中。

本次 F-test 配置下 SVM Macro-F1=0.715，层次模型=0.706，去掉 L1 PCA 后=0.693，去掉交互后=0.700。不能把这组消融推导为默认 mRMR 配置的效果，也不能与原方法的不同配置作直接提升比较；原方法已另用原始数据完成复核，见 ../../README.md。

交互候选的 F-test 在部分训练子集中遇到常数列，会出现 sklearn 警告；实现将非有限评分归零，所有训练与输出均正常完成。原始图像及特征生成流程未在本次重新审计。

## 合成数据复现

```bash
python -m skin_lesion_ml evaluate --data data/demo_features.csv --models logreg svm random-forest gradient-boosting knn hierarchical hierarchical-no-pca hierarchical-no-interactions --folds 3 --inner-folds 2 --k 10 --output outputs/demo-validation
```

合成指标位于 `synthetic-demo/`，仅用于检查代码执行。合成数据包含人为构造的区分信号，成绩不具有医学含义。
