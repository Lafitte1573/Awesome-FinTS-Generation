# Forecasting stock prices changes using long‑short term memory neural network with symbolic genetic programming

本文提出的方法是一种融合符号遗传编程（Symbolic Genetic Programming, SGP）与长短期记忆神经网络（Long Short-Term Memory, LSTM）的混合深度神经网络架构（SGP-LSTM），旨在提升对中国股市横截面股票收益的预测精度与风险调整后超额收益（alpha）。

该方法包含四个核心阶段：

1. **数据预处理（Phase 1）**  
   使用S&P Global Market Intelligence的Alpha Factor Library数据集，包含中国A股市场约4500只股票的304个基本面指标（来自资产负债表、利润表和现金流量表）与技术指标（基于价格与成交量）。预处理步骤包括：缺失值处理、异常值剔除、噪声过滤，并采用z-score标准化方法对所有原始特征进行归一化，为后续SGP演化提供稳定输入。

2. **数据增强：符号遗传编程（Phase 2）**  
   引入改进的符号遗传编程（SGP）作为核心特征工程工具，替代传统人工特征选择或PCA降维。SGP以原始标准化特征与一组预定义的启发式算子（如`ts_corr`, `ts_rank`, `ts_stddev`, `delta`, `rank_add`, `rank_div`, `ts_zscore`, `winsorize`, `delay`, `ts_return`等）为基因库，通过进化机制自动生成非线性组合特征。  
   - **种群初始化**：随机生成由数学运算符与原始特征构成的符号表达式树（symbolic tree），每个表达式代表一个潜在特征。  
   - **遗传操作**：采用40%交叉概率、40%替换概率和极低变异概率（避免噪声过度引入），确保演化稳定。  
   - **滚动窗口增强**：为所有启发式算子引入3–20天的滚动窗口，动态生成时序依赖的符号公式，提升对市场动态的捕捉能力。  
   - **适应度函数设计**：综合三个目标优化特征质量：  
     (i) 原始Rank IC（预测值与未来收益的秩相关系数）；  
     (ii) 最大化底部组累计收益与整体均值的偏离（Top_R）；  
     (iii) 维持按预测值排序的分组收益单调性（Monotonicity）。  
     最终适应度公式为：  
     $$
     \text{Fitness} = \text{Top}_R + \lambda_1 \times \text{Monotonicity} + \lambda_2 \times \text{Rank IC}
     $$  
     其中默认权重 $\lambda_1 = 0.4$, $\lambda_2 = 2$。  
   - **特征筛选**：对生成的符号公式，基于Rank IC成功率（正确预测正负收益的比例）与IC-PNL比率（IC均值/标准差）进行二次过滤，保留高稳定性与高信息量的特征。

3. **特征转换与序列构建（Phase 3）**  
   将SGP筛选出的最优特征按15日滞后序列组织，形成适合LSTM处理的时间序列输入格式，确保模型能捕捉跨期依赖关系。

4. **特征提取：LSTM模型（Phase 4）**  
   将经过SGP增强与序列化处理的特征输入一个双层LSTM网络，使用最终隐藏层输出作为股票未来5日收益的预测值。为验证LSTM的优越性，同时与多层感知机（MLP）进行对比实验，但最终选择LSTM因其天然适合处理时序依赖，优于MLP对历史状态的建模能力。

该方法的创新点在于：  
- **首次将SGP引入股票收益预测的特征工程环节**，实现从人工设计到自动演化非线性因子的转变；  
- **设计多目标适应度函数**，不仅优化Rank IC，还强化组合收益的单调性与底部收益表现，更贴合投资实践；  
- **构建端到端的SGP-LSTM框架**，将符号演化与深度时序建模深度融合，突破传统DNN在特征表达能力上的瓶颈。

最终，该模型在2014–2022年中国股市数据上验证，显著优于Fischer与Ghosh的基准LSTM模型、PCA-LSTM、单层LSTM等，实现Rank IC与ICIR的大幅提升，并在回测中年化超额收益超越CSI 300（31.00%）、CSI 500（24.48%）与市场平均（16.38%），信息比率高达2.49，证明其在实战中的有效性。