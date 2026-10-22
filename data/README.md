# 特征表数据契约

每行表示一张原图或一个增强视图。默认格式：

```csv
image_id,aug_tag,dx,filename,int_Lab_a_mean,glcm_contrast_d1_mean,asym_index
example_001,orig,nv,example_001.jpg,12.0,0.35,0.18
example_001,aug1,nv,example_001_aug1.jpg,12.2,0.34,0.19
```

上面仅展示列格式；实际训练需要三类与足够多独立组。

- `image_id`：必需，原图及其增强共用 ID。显式指定 `--group-column` 时，该列替代默认分组列。
- `dx`：必需，标签必须为 `mel`、`nv`、`vasc`；同一组只能有一个标签。
- `aug_tag`、`filename`、`patient_id`、`lesion_id`：可选元数据，永不自动作为特征。
- 其余列必须是数值特征；自定义标签列和分组列同样会从特征中剔除。额外字符串元数据应先删除，避免误入模型。
- NaN 允许，由每个训练折的中位数填补；训练折全缺失列填 0，再过滤常数列。无穷值、重复列名、缺失标签/分组和组内标签冲突会被拒绝。
- 如果存在同患者/同病灶多图，优先按患者/病灶分组；不要仅修改字符串以制造更多“独立”组。
- 推理不需要标签；允许列顺序不同，模型会按训练时列名对齐。缺失必需特征列会报错，额外列不参与预测。

`demo_features.csv` 由 `python -m skin_lesion_ml demo` 生成，属于合成数据。真实图像、特征表与患者信息不包含在发布包中；本地输入放在 `data/private/`。
