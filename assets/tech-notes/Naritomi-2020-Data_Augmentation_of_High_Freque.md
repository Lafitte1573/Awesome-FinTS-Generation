# Data Augmentation of High Frequency Financial Data Using Generative Adversarial Network

本文提出了一种基于生成对抗网络（GAN）的高频金融数据增强方法，旨在通过合成真实市场行为数据来提升金融时间序列预测模型的性能。该方法的核心是利用Wasserstein GAN（WGAN）模拟东京证券交易所（TSE）的高频交易事件，并基于合成事件构建人工市场，进而生成合成执行价格，用于训练预测模型。

方法的具体步骤如下：

1. **数据定义：订单事件与市场事件**  
   - **订单事件（Order Event）**：定义为四元组 $(s, d, \Delta, v)$，其中 $s \in \{\text{MKT}, \text{LMT}, \text{CAN}\}$ 表示订单类型（市价单、限价单、撤单），$d \in \{\text{S}, \text{B}\}$ 表示交易方向（卖/买），$\Delta \in \{-10, \dots, 10\}$ 表示相对于最优买卖价的价差位置（以tick为单位），$v \in \{100, 200, \dots, 2000\}$ 表示订单数量（单位：股）。  
   - **市场事件（Market Event）**：定义为五元组 $\mathbf{x}_i = (t_i, o_i, a_i, b_i, w_i)$，其中 $t_i$ 是订单事件发生的时间，$o_i$ 是对应的订单事件，$a_i$ 和 $b_i$ 分别是订单事件发生后限价簿（LOB）中的最优卖价和最优买价，$w_i$ 是当前tick宽度。  
   - **限价（Limit Price）**：根据订单方向计算：若为卖单（S），则 $p_i = b_i + w_i \Delta_i$；若为买单（B），则 $p_i = a_i - w_i \Delta_i$。  
   - **买卖价差（Bid-Ask Spread）**：定义为 $sp_i = (a_i - b_i) / w_i$（以tick为单位）。  
   - **时间间隔**：$\Delta t_i = t_i - t_{i-1}$，表示连续订单事件之间的时间差。  
   - **历史市场事件序列**：为每个市场事件 $i$ 构建一个长度为 $k=20$ 的历史序列 $H_{i-k:i} = \{\mathbf{x}_j \mid i-k \leq j < i\}$，作为条件输入。

2. **GAN架构设计**  
   - **生成器（Generator）**：  
     输入包括：  
     （1）噪声向量 $\mathbf{z} \in \mathbb{R}^{100}$，从均匀分布 $U[-1,1]^{100}$ 采样；  
     （2）历史市场事件序列 $H_{i-k:i}$。  
     架构流程：  
     （a）订单事件 $o_i$ 通过嵌入层（Embedding）转换为实值向量 $\mathbf{e}_i$；  
     （b）历史序列 $H_{i-k:i}$ 的嵌入向量输入至LSTM层，提取时序特征；  
     （c）LSTM输出与噪声向量 $\mathbf{z}$ 拼接；  
     （d）拼接后的向量输入全连接层，再通过一维卷积神经网络（1D-CNN）和全连接层，输出合成市场事件 $\hat{\mathbf{x}}_i = g_\theta(\mathbf{z}, H_{i-k:i})$。  
     生成器的目标是学习条件概率分布 $\mathbb{P}_r(\mathbf{x}_i | H_{i-k:i})$，生成与真实数据分布一致的市场事件。

   - **判别器（Discriminator）**：  
     输入包括：  
     （1）真实市场事件 $\mathbf{x}_i$ 或合成市场事件 $\hat{\mathbf{x}}_i$；  
     （2）对应的历史序列 $H_{i-k:i}$。  
     架构流程：  
     （a）历史序列 $H_{i-k:i}$ 经嵌入层和LSTM层提取特征；  
     （b）市场事件 $\mathbf{x}_i$ 或 $\hat{\mathbf{x}}_i$ 与LSTM输出拼接；  
     （c）拼接向量经一维卷积层和全连接层，输出一个标量值 $y_i = f_w(\mathbf{x}_i, H_{i-k:i})$，表示输入为真实数据的概率。  
     判别器的目标是区分真实数据与合成数据，其输出用于计算Wasserstein距离。

3. **Wasserstein GAN训练机制**  
   - 采用WGAN-GP（Wasserstein GAN with Gradient Penalty）算法，以解决传统GAN的训练不稳定问题。  
   - 目标是最小化真实分布 $\mathbf{r}$ 与生成分布 $\mathbf{g}_\theta$ 之间的Wasserstein距离：  
     $$
     L = \inf_\theta \sup_{f_w \in \mathcal{L}_1} \left\{ \mathbb{E}[f_w(\mathbf{r})] - \mathbb{E}[f_w(\mathbf{g}_\theta)] \right\}
     $$
     其中 $\mathcal{L}_1$ 是所有1-Lipschitz函数的集合。  
   - 判别器 $f_w$ 由神经网络实现，通过梯度惩罚（Gradient Penalty）强制其满足1-Lipschitz约束：  
     $$
     \text{Loss} = \mathbb{E}[f_w(\tilde{\mathbf{x}})] - \mathbb{E}[f_w(\mathbf{x})] + \lambda \cdot \left( \|\nabla_{\hat{\mathbf{x}}} f_w(\hat{\mathbf{x}})\|_2 - 1 \right)^2
     $$
     其中 $\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1-\epsilon)\tilde{\mathbf{x}}$，$\epsilon \sim U[0,1]$，$\lambda = 0.1$ 为惩罚系数。  
   - 生成器通过最大化判别器对合成数据的输出进行优化：  
     $$
     \theta \leftarrow \text{Adam} \left( \nabla_\theta \frac{1}{m} \sum_{b=1}^m -f_w(g_\theta(\mathbf{z}^{(b)}, H_{i-k:i}^{(b)})) \right)
     $$
   - 训练采用交替优化：每轮生成器更新前，判别器更新5次（$n_{dis} = 5$），使用Adam优化器（$\alpha = 0.001, \beta_1 = 0.5, \beta_2 = 0.9$），批量大小 $m = 32$。

4. **数据增强与人工市场模拟**  
   - 使用东京证券交易所的FLEX Full数据（2020年1月为训练期，2020年2月至5月为测试期），聚焦TOPIX CORE 30成分股。  
   - 将原始订单流转换为订单事件序列，构建限价簿，进而生成市场事件序列。  
   - 训练好的生成器用于生成大量合成市场事件 $\hat{\mathbf{x}}_i$，并据此重构人工限价簿，模拟订单匹配过程，生成合成执行价格。  
   - 合成数据与真实数据共同作为训练集，输入到LSTM预测模型中，用于预测未来执行价格的涨跌方向。

5. **评估与验证**  
   - 通过比较真实订单事件与合成订单事件的概率分布（如订单类型、数量、价差位置、时间间隔等），验证合成数据的统计真实性。  
   - 实验结果表明，合成数据的分布与真实数据高度接近，且在使用合成数据增强训练集后，LSTM模型对执行价格涨跌的预测准确率显著优于未使用数据增强的基线模型。

综上，本文方法通过WGAN生成高保真度的高频订单事件序列，构建人工市场以合成执行价格，实现对真实金融数据的高质量增强，从而提升预测模型的泛化能力与稳定性。