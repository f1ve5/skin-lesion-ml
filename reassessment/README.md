# 原方法复核入口

`assess_original.py` 读取原数据与原代码，不修改二者，所有生成文件进入显式指定的 `--output`。

- `--stage extract`：核对图像、掩膜和分组，重新提取全部特征；如提供 `--cached`，同时核对原特征缓存。源目录存在原缓存时也会自动比较。
- `--stage evaluate`：保留原分类器与参数，运行原配置、训练折 ICC 配置及两项消融。对同一个训练出的模型计算原预测和无标签交互预测，输出成对记录。
- `--stage baselines`：原五类固定参数模型，F-test 筛选仅在外层训练折拟合。

`--source` 指向 `original_method/`；`--data` 指向包含 image、mask 和 label.csv 的目录。运行命令见 [原方法说明](../original_method/README.md)，完成结果见 [复核报告](../reports/reassessment/README.md)。

可选 `--features` 仅用于显式读取已存在的特征表；从原图复核时先执行 extract，再省略此参数。`--variants` 可限定 evaluate 的配置，如 `--variants original fold-icc`。评估中的类名和标签映射沿用原代码，不借助测试标签选择模型。

脚本保留原实现的行为和已知限制，不代表对原代码所有环节的重构。函数插桩是为了量化单项改动影响，其源码会保存在本地输出目录。原始特征与逐行输出不进入公开材料。
