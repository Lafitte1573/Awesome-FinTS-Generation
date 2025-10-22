# LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics

本文提出的方法名为 LLM-PS（Large Language Models for Time Series Forecasting with Patterns and Semantics），旨在通过挖掘时间序列数据固有的**多尺度时序模式**与**语义信息**，显著提升大语言模型（LLM）在时间序列预测（TSF）任务中的性能。传统LLM基于文本预训练，难以有效建模时间序列特有的结构特性，而LLM-PS通过两个核心模块——多尺度卷积神经网络（MSCNN）和时间到文本语义提取器（T2T）——显式提取并融合这些关键信息，从而增强LLM对时间序列的深层理解。

### 1. 多尺度卷积神经网络（MSCNN）：捕捉多尺度时序模式

MSCNN 的核心目标是从输入时间序列中高效提取同时包含**短期波动**（如周期性变化）和**长期趋势**（如缓慢变化的全局趋势）的多尺度特征。与传统CNN固定感受野的局限不同，MSCNN采用分层并行分支结构，每个模块包含 B 个并行分支，每个分支通过递归连接的 3×3 卷积层逐步扩大感受野。

具体流程如下：
- 输入特征 $\mathbf{F}_{\mathrm{in}} \in \mathbb{R}^{C \times V}$ 首先经过 $1 \times 1$ 卷积降维，然后被均分为 B 个子特征 $\{\mathbf{F}_1, \dots, \mathbf{F}_B\}$，每个子特征维度为 $\mathbb{R}^{C/B \times V}$。
- 每个分支 $\mathbf{F}_i$ 依次输入其对应的 $3 \times 3$ 卷积层，且从第2个分支开始，输入为当前分支特征与前一分支输出的叠加（$\mathbf{F}_i + \bar{\mathbf{F}}_{i-1}$），从而形成递增的感受野：
  $$
  \bar{\mathbf{F}}_i = 
  \begin{cases} 
  \mathrm{Conv}_i(\mathbf{F}_i), & i=1 \\
  \mathrm{Conv}_i(\mathbf{F}_i + \bar{\mathbf{F}}_{i-1}), & 1 < i \leq B 
  \end{cases}
  $$
- 所有分支输出 $\{\bar{\mathbf{F}}_1, \dots, \bar{\mathbf{F}}_B\}$ 被拼接后，通过一个 $1 \times 1$ 卷积融合，并与原始输入 $\mathbf{F}_{\mathrm{in}}$ 残差连接，得到最终输出 $\mathbf{F}_{\mathrm{out}}$。
- 多个MSCNN块堆叠，生成最终的多尺度特征集合 $\mathbf{F}_{\mathrm{MS}}$，用于后续语义融合与LLM输入。

### 2. 时序模式解耦与重组（Temporal Patterns Decoupling and Assembling）

为更精确地分离和强化短期与长期模式，LLM-PS引入基于**小波变换**（Wavelet Transform, WT）的解耦机制，替代传统的平均池化或傅里叶变换方法。

具体步骤如下：
- 对MSCNN输出的每个分支特征 $\bar{\mathbf{F}}_b$ 应用小波变换，分解为低频分量（代表长期趋势）$\mathbf{W}_{\mathrm{low}}^b$ 和多个高频分量（代表短期波动）$\{\mathbf{W}_{\mathrm{high},i}^b\}_{i=1}^w$。
- 通过**逆小波变换**（IWT）重建两个独立模式：
  - **短期模式** $\mathbf{P}_S^b = \mathrm{IWT}(\mathrm{Zero}(\mathbf{W}_{\mathrm{low}}^b), \{\mathbf{W}_{\mathrm{high},i}^b\})$：保留高频分量，置零低频分量。
  - **长期模式** $\mathbf{P}_L^b = \mathrm{IWT}(\mathbf{W}_{\mathrm{low}}^b, \{\mathrm{Zero}(\mathbf{W}_{\mathrm{high},i}^b)\})$：保留低频分量，置零高频分量。
- 为增强模式的上下文一致性，进行**局部到全局**与**全局到局部**的重组：
  - 短期模式：从第2块到第B块，逐级累加前一模式（$\mathbf{P}_S^b = \mathbf{P}_S^b + \mathbf{P}_S^{b-1}$），使局部波动信息向全局传播。
  - 长期趋势：从第B-1块到第1块，逐级累加后一趋势（$\mathbf{P}_L^b = \mathbf{P}_L^b + \mathbf{P}_L^{b+1}$），使全局趋势信息向局部细化。
- 最终，每个分支特征通过短期与长期模式叠加重构：$\bar{\mathbf{F}}_b = \mathbf{P}_S^b + \mathbf{P}_L^b$，从而实现对多尺度模式的精细化建模。

### 3. 时间到文本语义提取器（T2T）：从稀疏时序中提取语义

时间序列数据语义稀疏，单个点无明确含义，需整段序列表达语义（如“骤升”、“平稳”）。T2T模块借鉴自监督学习思想（如HuBERT），通过掩码重建与语义标签预测，从时序中提取与LLM文本嵌入对齐的语义信息。

具体流程如下：
- 输入时间序列 $\mathbf{X} \in \mathbb{R}^{H \times V}$ 被划分为 P 个长度为 L 的时序块 $\{\mathbf{X}_i\}_{i=1}^P$。
- 以约75%的掩码率随机遮蔽部分块，T2T作为编码器-解码器结构，目标为：
  - 重建被掩码块 $\hat{\mathbf{X}}_i$；
  - 预测每个块（包括未掩码块）的语义标签 $l_i$。
- 语义标签 $l_i$ 的定义：将每个时序块 $\mathbf{X}_i$ 通过线性投影 $\mathrm{Proj}(\cdot)$ 映射到LLM文本嵌入空间，计算其与词汇表中所有词嵌入 $\mathbf{E}$ 的相似度 $\mathbf{S}_i = \mathrm{Proj}(\mathbf{X}_i) \cdot \mathbf{E}^\top$，选择相似度最高的词作为标签 $l_i$。
- T2T的损失函数为重建误差与语义预测交叉熵的加权和：
  $$
  \mathcal{L}_{\mathrm{T2T}} = \frac{1}{P} \sum_{i=1}^P \left( \mathbb{1}_{[\mathbf{M}(i)=1]} \|\mathbf{X}_i - \hat{\mathbf{X}}_i\|_2 + l_i \log \frac{l_i}{\hat{l}_i} \right)
  $$
  其中 $\mathbb{1}_{[\mathbf{M}(i)=1]}$ 为掩码指示函数，$\hat{l}_i$ 为预测的语义标签概率。
- T2T输出的语义特征 $\mathbf{F}_{\mathrm{T2T}}$ 与MSCNN输出的多尺度特征 $\mathbf{F}_{\mathrm{MS}}$ 通过特征对齐（$\mathcal{L}_{\mathrm{FEAT}} = \|\mathbf{F}_{\mathrm{MS}} - \mathbf{F}_{\mathrm{T2T}}\|_2$）进行融合，共同输入LLM。

### 4. LLM微调与联合优化

LLM-PS以预训练的GPT-2为骨干，采用**低秩自适应**（LoRA）技术进行高效微调，避免全参数更新的高成本。整体训练目标为时序预测损失与语义对齐损失的加权和：
$$
\mathcal{L}_{\mathrm{OBJ}} = \mathcal{L}_{\mathrm{TIME}} + \lambda \mathcal{L}_{\mathrm{FEAT}}
$$
其中：
- $\mathcal{L}_{\mathrm{TIME}} = \frac{1}{T} \sum_{i=1}^T \|\mathbf{Y}_i - \hat{\mathbf{Y}}_i\|_2$：预测值与真实值的均方误差；
- $\mathcal{L}_{\mathrm{FEAT}} = \frac{1}{C} \sum_{j=1}^C \|\mathbf{F}_{\mathrm{MS}}^j - \mathbf{F}_{\mathrm{T2T}}^j\|_2$：多尺度特征与语义特征的L2距离；
- $\lambda = 0.01$ 为平衡系数。

最终，MSCNN与T2T提取的融合特征作为LLM的输入，LLM直接生成未来时间序列 $\hat{\mathbf{Y}}$，实现端到端的时序预测。

综上，LLM-PS通过MSCNN精准建模多尺度时序模式、T2T从稀疏序列中提取语义信息、LoRA高效微调LLM，三者协同，首次系统性地将时间序列的内在结构特性融入LLM框架，显著提升其在各类TSF任务中的泛化能力与预测精度。