"""
三维联合特征选择器
=====================
算法：临床先验 + 统计显著性 (ANOVA F) + 鲁棒性 (ICC)
  Joint_Rank = α·rank_F + β·rank_ICC + γ·clinical_penalty

文献依据：
  - Barata 2019: 皮肤镜特征综述，强调临床知识引导
  - Xue 2021: ICC 过滤提升可泛化性
  - 课程要求: 多维度联合评估特征质量
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif


def get_clinical_penalty(feature_name: str, priority_map: dict) -> float:
    """
    根据特征名前缀匹配临床优先级映射表，返回惩罚值 [0, 1]。
    按映射表中 key 的顺序匹配，首个命中则返回对应值。
    若未命中任何规则，返回默认值 0.5（中性）。
    """
    for prefix, penalty in priority_map.items():
        if feature_name.startswith(prefix):
            return float(penalty)
    return 0.5


class JointFeatureSelector(BaseEstimator, TransformerMixin):
    """
    sklearn 兼容的联合特征选择 Transformer。

    在 fit() 阶段：
      1. 在训练集 X, y 上计算 ANOVA F 值 → rank_F（每折重新计算）
      2. 结合全局 rank_ICC（从 robustness_per_feature.csv 加载）
      3. 结合全局 clinical_penalty（从 CLINICAL_PRIORITY 字典）
      4. 按 alpha:beta:gamma 权重求和得到 Joint_Rank
      5. 选取 Joint_Rank 最小的前 k 个特征

    transform() 返回选中的特征子集。
    """

    def __init__(self, icc_ranks: dict, clinical_penalty_map: dict,
                 alpha=0.5, beta=0.3, gamma=0.2, k=40):
        """
        Parameters
        ----------
        icc_ranks : dict
            {feature_name: ICC_rank}, rank=1 为最优（ICC 最高）。
            也可直接传 ICC 值，内部会转换为排名。
        clinical_penalty_map : dict
            前缀→惩罚值 映射表，值域 [0,1]，越小越优先。
        alpha, beta, gamma : float
            三项权重，建议和为 1.0。
        k : int
            最终保留的特征数。
        """
        self.icc_ranks = icc_ranks
        self.clinical_penalty_map = clinical_penalty_map
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.selected_features_ = None
        self.joint_scores_ = None

    def fit(self, X, y=None):
        """
        计算联合排名并选出 Top-k 特征。
        X: pd.DataFrame or np.ndarray, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)
        """
        if isinstance(X, pd.DataFrame):
            feat_names = X.columns.tolist()
            X_arr = X.values
        else:
            feat_names = [f"f{i}" for i in range(X.shape[1])]
            X_arr = X

        n_features = len(feat_names)

        # 1) rank_F: 在训练集上计算 ANOVA F 值，rank=1 为 F 最大
        if y is not None and len(np.unique(y)) >= 2:
            f_vals, _ = f_classif(X_arr, y)
            # 从大到小排名: 最大 F → rank 1
            rank_F = np.zeros(n_features)
            order_F = np.argsort(-f_vals)  # 降序
            for r, idx in enumerate(order_F):
                rank_F[idx] = r + 1
        else:
            rank_F = np.ones(n_features) * (n_features / 2.0)

        # 2) rank_ICC: 全局固定，rank=1 为 ICC 最高
        rank_ICC = np.zeros(n_features)
        for i, name in enumerate(feat_names):
            rank_ICC[i] = self.icc_ranks.get(name, n_features)

        # 3) clinical_penalty: 全局固定，0~1，越小越优先
        clinical_pen = np.zeros(n_features)
        for i, name in enumerate(feat_names):
            clinical_pen[i] = get_clinical_penalty(name, self.clinical_penalty_map)

        # 归一化 clinical_penalty 到与 rank_F/rank_ICC 相近的尺度
        #   rank_F 和 rank_ICC 范围约为 1~n_features
        #   将 clinical_penalty 线性映射到 [1, n_features]
        clinical_pen_norm = 1.0 + clinical_pen * (n_features - 1)

        # 4) 联合得分 (越小越优)
        joint_score = (self.alpha * rank_F +
                       self.beta * rank_ICC +
                       self.gamma * clinical_pen_norm)

        # 5) 选出 Top-k
        top_indices = np.argsort(joint_score)[:self.k]
        self.selected_features_ = [feat_names[i] for i in top_indices]

        # 保存得分供外部查看
        self.joint_scores_ = pd.DataFrame({
            "feature": feat_names,
            "rank_F": rank_F.astype(int),
            "rank_ICC": rank_ICC.astype(int),
            "clinical_penalty": clinical_pen,
            "joint_score": joint_score,
        }).sort_values("joint_score")

        return self

    def transform(self, X):
        """返回选中特征的数据"""
        if self.selected_features_ is None:
            raise RuntimeError("Must call fit() before transform()")
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            # 需要 feature name 映射
            raise ValueError("X must be a pandas DataFrame with named columns")

    def get_top_features(self, n=20):
        """返回 Top-n 特征名称、各项排名和得分"""
        if self.joint_scores_ is None:
            raise RuntimeError("Must call fit() first")
        return self.joint_scores_.head(n)


def build_icc_ranks(rob_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    从 robustness_per_feature.csv 构建 ICC 排名字典。
    rank=1 为 ICC 最高（最鲁棒）。
    对 ICC 为 NaN 的特征赋予最低排名。
    """
    # 只保留 feature_cols 中的特征
    rob_sub = rob_df[rob_df["feature"].isin(feature_cols)].copy()
    icc_vals = rob_sub.set_index("feature")["icc"]

    # 对存在的特征按 ICC 降序排名
    valid = icc_vals.dropna().sort_values(ascending=False)
    n_total = len(feature_cols)
    ranks = {}
    for r, (feat, _) in enumerate(valid.items()):
        ranks[feat] = r + 1
    # NaN 特征给最低排名
    for f in feature_cols:
        if f not in ranks:
            ranks[f] = n_total
    return ranks


def print_top_features(selector: JointFeatureSelector, n=20):
    """打印 Top-n 特征及其评分明细"""
    top = selector.get_top_features(n)
    print(f"\n{'='*90}")
    print(f"Top-{n} Joint Feature Selection Results")
    print(f"{'='*90}")
    print(f"{'#':<4} {'Feature':<45} {'F-rank':<8} {'ICC-rank':<9} {'Clinical':<9} {'Joint':<8}")
    print(f"{'-'*90}")
    for i, (_, row) in enumerate(top.iterrows()):
        print(f"{i+1:<4} {row['feature']:<45} {row['rank_F']:<8} "
              f"{row['rank_ICC']:<9} {row['clinical_penalty']:<9.2f} {row['joint_score']:<8.2f}")
    print(f"{'='*90}\n")
    return top
