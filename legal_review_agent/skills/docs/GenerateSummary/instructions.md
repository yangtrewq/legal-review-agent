# GenerateSummary 使用指南（L2 / 结论文档格式规范）

## 调用前置条件
- 全部风险点已经 IdentifyRisk 识别并经人工确认；
- 每个待整改风险点已有对应的 GenerateOpinion 意见；
- contract_metadata 使用 QueryRequirement 的原始输出。

## overall_assessment 判定规则
- **通过**：无高风险且中风险均有可执行意见；
- **有条件通过**：存在高风险但均有明确修改意见与妥协方案；
- **不通过**：存在无法通过修改化解的红线风险。

## review_conclusion 文档格式规范（Markdown）
```
# 合同评审结论
## 整体评估      ← 一段话给出结论与依据
## 重点风险      ← 按等级降序列出，每条含条款位置
## 综合建议      ← 可执行的整改清单，与 opinions 一一对应
```
