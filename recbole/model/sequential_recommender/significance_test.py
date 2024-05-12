import scipy.stats as stats
import numpy as np
# 1wei sas
# 三组数据，每组数据包括NDCG@10的值
NDCG1 = [2,2,2,2,3,2,2,2,2,2]
NDCG2 = [3,3,3,4,4,1,1,3,3,3]

# NDCG1 = [0.1075, 0.1123, 0.1129,0.055,0.02]
# NDCG2 = [0.1155, 0.1186, 0.1138,0.1211,0.03]

Recall1 = [0.2001, 0.1924,0.2098, 0.2002,0.22,0.19]
Recall2 = [0.2002, 0.1991,0.2099,0.2066,0.212,0.23]


MRR1 = [0.0794, 0.0879,0.0841]
MRR2 = [0.0898, 0.941, 0.0926]


# 执行方差分析（ANOVA）
# f_statistic, p_value = stats.f_oneway(group1, group2)
# f_statistic, p_value = stats.kruskal(group1, group2)
f_statistic, p_value = stats.wilcoxon(NDCG1, NDCG2)
f_statistic1, p_value1 = stats.wilcoxon(Recall1, Recall2)
f_statistic2, p_value2 = stats.wilcoxon(MRR1, MRR2)
# 输出方差分析的结果
print("F-statistic:", f_statistic)
print("p-value:", p_value)
print("p-value1:", p_value1)
print("p-value2:", p_value2)

# 判断是否存在显著性差异
alpha = 0.05
if p_value < alpha:
    print("ndcg存在显著性差异，拒绝零假设")
else:
    print("ndcg没有足够的证据拒绝零假设")

if p_value1 < alpha:
    print("RECALL存在显著性差异，拒绝零假设")
else:
    print("RECALL没有足够的证据拒绝零假设")

if p_value2 < alpha:
    print("mrr存在显著性差异，拒绝零假设")
else:
    print("mrr没有足够的证据拒绝零假设")

# 如果ANOVA显示显著性差异，可以进行事后比较
