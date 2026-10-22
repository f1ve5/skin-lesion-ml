# 原课程核心代码快照

此目录用于溯源，四个 Python 文件按原样复制自 `proj_v2_ablation/`：

| 文件 | 原位置 |
| --- | --- |
| `classify.py` | `classification/classify.py` |
| `ensemble.py` | `classification/ensemble.py` |
| `feature_screening.py` | `analysis/feature_screening.py` |
| `feature_interaction.py` | `analysis/feature_interaction.py` |

这些文件保留原始依赖与已知评估问题，不构成独立可运行的包。运行请使用 `src/skin_lesion_ml/` 中的新入口。变更原因见 [评估说明](../docs/EVALUATION.md)，文件校验信息见 [provenance.json](../reports/historical/provenance.json)。

保留原课程实现以便核对设计来源，不表示其上游或团队代码全部由个人独立完成。团队与个人贡献边界见仓库首页。
