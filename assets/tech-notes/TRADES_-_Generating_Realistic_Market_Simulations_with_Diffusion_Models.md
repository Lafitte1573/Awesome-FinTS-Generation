# TRADES: Generating Realistic Market Simulations with Diffusion Models

本文提出的方法名为TRADES（TRAnsformer-based Denoising Diffusion Probabilistic Engine for LOB Simulations），是一种基于Transformer架构的条件去噪扩散概率模型，专门用于生成高保真、高响应性的限价订单簿（Limit Order Book, LOB）市场模拟数据。其核心目标是解决传统市场模拟方法（如基于代理的仿真IABS和Wasserstein GAN）在真实性、有用性和响应性方面的不足。

TRADES的生成过程是一个条件化的时间序列生成任务，其输入包括两部分：  
1. **条件部分**（$\mathbf{x}^c$）：包含过去$N-1$个订单序列（每个订单由价格、数量、方向、深度、时间偏移和订单类型六维特征构成）以及最近$N$个LOB快照（每个快照包含前$L=10$个买卖盘层级的买卖价格与数量）；  
2. **生成部分**（$\mathbf{x}^g$）：即待生成的下一个订单（单样本，$S=1$），其维度与订单特征一致（$K=6$）。

模型的前向扩散过程仅对生成部分$\mathbf{x}_0^g$施加高斯噪声，而保持条件部分$\mathbf{x}_0^c$不变。噪声逐步添加至$\mathbf{x}_T^g$，使其变为纯高斯噪声。在反向去噪过程中，模型学习一个条件概率分布$p_\theta(\mathbf{x}_{t-1}^g | \mathbf{x}_t^g, \mathbf{x}_0^c)$，通过迭代从$\mathbf{x}_T^g$逐步恢复出真实的$\mathbf{x}_0^g$。

模型架构基于Transformer编码器，其输入为条件部分与生成部分的拼接张量。为增强表示能力，模型首先通过两个独立的多层感知机（MLP）对订单序列和LOB快照进行特征升维（augmentation），然后将升维后的特征在特征维度上拼接，并叠加扩散时间步嵌入（diffusion embedding）与位置编码（positional embedding），输入至多个Transformer编码层。每个Transformer层包含多头自注意力机制和前馈网络，用于建模订单间及订单与LOB状态间的长程时空依赖关系。最终，Transformer输出两个参数：预测的噪声$\epsilon_\theta$和可学习的方差$\Sigma_\theta$，二者经反向MLP映射回原始空间，用于计算去噪后的样本$\mathbf{x}_{t-1}^g$。

训练采用自监督方式：对真实订单序列，仅对最后一个订单施加噪声，其余作为条件，训练模型预测噪声项。损失函数由两部分组成：  
1. $\mathcal{L}_\epsilon$：预测噪声与真实噪声的L2损失，用于引导去噪方向；  
2. $\mathcal{L}_\Sigma$：基于扩散过程KL散度的负对数似然损失，用于优化方差参数，提升生成质量。  
总损失为$\mathcal{L} = \mathcal{L}_\epsilon + \lambda \mathcal{L}_\Sigma$。

在推理阶段，TRADES以自回归方式运行：每次生成一个订单，将其追加至条件序列尾部，滑动窗口向前推进，持续生成订单序列，直至模拟结束。模型通过融合历史订单与高维LOB状态，实现了对市场供需动态（如买卖盘不平衡、价格发现机制）的精确建模，从而生成在统计特性（如波动率聚集、成交量-波动率正相关、收益分布形态）上高度贴近真实市场的订单流。

TRADES的核心创新在于：  
- 首次将条件化扩散模型应用于LOB生成，克服了GAN的模式坍缩与训练不稳定性；  
- 引入LOB快照作为关键条件输入，显著提升市场状态建模能力；  
- 采用Transformer架构，有效捕捉长序列依赖与多变量交互；  
- 实现了对市场冲击的响应性，允许外部交易代理介入并观察价格永久性影响；  
- 通过自回归机制支持连续、动态、可交互的市场仿真。

该方法不依赖于任何预设的市场微观结构假设，而是从数据中直接学习真实订单流的复杂分布，为算法交易策略回测、市场影响实验和金融监管政策模拟提供了前所未有的高保真仿真平台。