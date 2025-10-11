## 金融时间序列数据生成的最新先进技术综述

> This repository displays a paper collection of the survey of recent financial data-generation technologies.

### 综述大纲
```mermaid
graph LR
    A(金融数据生成) --> B([时序数据预测（Forecasting）])
    B --> B1(基于LLM（通用）) ==> C1(TimeGPT、Chronos)
    B --> B2(基于DNN（专用）) ==> C2(N-BEATS、N-HiTS、PatchTST、iTransformer、KAN)
    A --> C([时序数据插补（Imputation）])
    A --> D([时序数据增强（Augmentation）])
```

| 章节 | 标题 | 主要内容（暂定） |
| --- | --- | --- |
| 1 | 引言 | |
| 2 | 时间序列数据预测 | 问题定义、方法分类、评价指标、主要挑战 |
| 3 | 时间序列数据补插 | 问题定义、方法分类、评价指标、主要挑战 |
| 4 | 时间序列数据增强 | 问题定义、方法分类、评价指标、主要挑战 |
| 5 | 数据集整理 | 为方便研究工作的开展，列表总结金融时序数据生成领域的数据集，维度包括开放性、下游任务、数据量等 |
| 6* | 未来方向* | 总结金融时序数据生成领域（非时序预测/合成/增强）未来的研究方向 |
| 7 | 总结 | |

### 文献梳理

Please refer to `papers_summary.md`.
