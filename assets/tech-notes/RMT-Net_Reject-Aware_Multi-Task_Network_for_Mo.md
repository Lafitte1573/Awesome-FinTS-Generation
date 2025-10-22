# RMT-Net: Reject-Aware Multi-Task Network for Modeling Missing-Not-At-Random Data in Financial Credit Scoring

本文提出了一种名为**Reject-aware Multi-Task Network（RMT-Net）**的新型多任务学习框架，用于建模金融信用评分中因“非随机缺失”（Missing-Not-At-Random, MNAR）导致的偏差数据问题。其核心思想是：**默认/非默认分类任务与拒绝/批准分类任务高度相关，因此可通过多任务学习机制，利用拒绝/批准任务的信息来增强默认预测的准确性，尤其在未观测标签的拒绝样本上**。

### 1. 问题背景与动机
- 在信用评分中，仅对**获批申请**（approved samples）可观察到真实的违约（default）或非违约（non-default）标签，而**被拒申请**（rejected samples）无任何违约标签，导致数据缺失机制为MNAR。
- 传统方法（如重加权、半监督学习）未能有效利用拒绝/批准任务与默认任务之间的内在相关性。
- 通过真实数据（Lending Club）分析和理论证明（定理1）发现：**被拒客户的违约率显著高于获批客户**，且两个任务存在正相关关系，因此可利用拒绝/批准任务的丰富、无偏数据来辅助默认预测。

### 2. RMT-Net 架构（单策略场景）
RMT-Net 包含四个核心组件：

#### （1）嵌入层（Embedding Layer）
- 将原始高维特征向量 $ \mathbf{x}_i \in \mathbb{R}^d $ 转换为稠密嵌入表示 $ \mathbf{e}_i \in \mathbb{R}^{dk} $，其中 $ k $ 为嵌入维度。
- 对数值型特征进行离散化处理以提升嵌入效率。

#### （2）拒绝/批准预测网络（R/A-Net）
- 一个深度神经网络，用于预测样本被拒绝的概率。
- 结构为多层全连接网络，每层使用 ReLU 激活函数，最后一层使用 Sigmoid 函数输出拒绝概率：
  $$
  p_i^{(t)} = \sigma\left( p_i^{(t-1)} \mathbf{w}_R^{(t)} + \mathbf{b}_R^{(t)} \right)
  $$
- 其中 $ p_i^{(t)} \in \mathbb{R} $ 表示样本 $ i $ 的拒绝概率，$ t $ 为网络层数。

#### （3）默认/非默认预测网络（D/N-Net）
- 与 R/A-Net 结构相同（层数 $ t $），但其每一层的隐藏表示**动态融合**来自 R/A-Net 的信息。
- 关键创新：引入**门控网络**（gating network），根据 R/A-Net 输出的拒绝概率自适应控制信息共享比例。
- 在第 $ j $ 层，门控权重计算为：
  $$
  g_i^{(j)} = \sigma\left( \alpha^{(j)} p_i^{(t)} + \beta^{(j)} \right)
  $$
  其中 $ \alpha^{(j)}, \beta^{(j)} $ 为可学习参数，$ g_i^{(j)} \in \mathbb{R} $ 表示从 R/A-Net 向 D/N-Net 传递信息的比重。
- D/N-Net 第 $ j $ 层的隐藏表示为：
  $$
  \mathbf{q}_i^{(j)} = \text{ReLU}\left( \mathbf{q}_i^{(j-1)} \mathbf{w}_D^{(j)} + \mathbf{b}_D^{(j)} \right) + g_i^{(j)} \cdot \mathbf{p}_i^{(j)}
  $$
  - **核心机制**：当拒绝概率 $ p_i^{(t)} $ 越高 → $ g_i^{(j)} $ 越大 → D/N-Net 越多依赖 R/A-Net 的信息，因为高拒绝概率样本的默认标签不可靠，需借助拒绝/批准任务的判别能力。
- 最终输出为默认概率：
  $$
  q_i^{(t)} = \sigma\left( q_i^{(t-1)} \mathbf{w}_D^{(t)} + \mathbf{b}_D^{(t)} \right)
  $$

#### （4）损失函数
- **拒绝/批准任务损失**（L1）：对所有样本（包括被拒）计算二元交叉熵：
  $$
  \mathcal{L}_1 = -\sum_{i=1}^N \left[ p_i^{(t)} \log(r_i) + (1 - p_i^{(t)}) \log(1 - r_i) \right]
  $$
- **默认/非默认任务损失**（L2）：**仅对获批样本（$ r_i = 0 $）计算**，被拒样本损失被屏蔽：
  $$
  \mathcal{L}_2 = -\sum_{i=1}^N \left[ q_i^{(t)} \log(y_i) + (1 - q_i^{(t)}) \log(1 - y_i) \right] \cdot (1 - r_i)
  $$
- **总损失**：加权组合，$ \eta $ 为超参数：
  $$
  \mathcal{L} = (1 - \eta) \cdot \mathcal{L}_1 + \eta \cdot \mathcal{L}_2
  $$

### 3. RMT-Net++（多策略场景扩展）
为应对现实中多个动态变化的审批策略（如不同时间段、不同风控规则），RMT-Net++ 扩展了 RMT-Net：

#### （1）多个拒绝/批准网络（R/A-Nets++）
- 设有 $ M $ 个独立的 R/A-Net，每个网络 $ m $ 专门学习由策略 $ f_m $ 产生的样本的拒绝概率：
  $$
  p_{i,[m]}^{(t)} = \sigma\left( p_{i,[m]}^{(t-1)} \mathbf{w}_{R,[m]}^{(t)} + \mathbf{b}_{R,[m]}^{(t)} \right)
  $$

#### （2）扩展的门控机制
- 为每个策略 $ m $ 的 R/A-Net++ 设计独立的门控权重：
  $$
  g_{i,[m]}^{(j)} = \sigma\left( \alpha_{[m]}^{(j)} p_{i,[m]}^{(t)} + \beta_{[m]}^{(j)} \right)
  $$

#### （3）扩展的默认预测网络（D/N-Net++）
- 每一层的隐藏表示融合所有策略的贡献：
  $$
  \mathbf{q}_i^{(j)} = \text{ReLU}\left( \mathbf{q}_i^{(j-1)} \mathbf{w}_D^{(j)} + \mathbf{b}_D^{(j)} \right) + \sum_{m=1}^M g_{i,[m]}^{(j)} \cdot \mathbf{p}_{i,[m]}^{(j)}
  $$

#### （4）多策略损失函数
- 拒绝/批准任务损失按策略分组计算：
  $$
  \mathcal{L}_1 = -\sum_{m=1}^M \sum_{\substack{i=1 \\ f_{s_i} = f_m}}^N \left[ p_{i,[m]}^{(t)} \log(r_i) + (1 - p_{i,[m]}^{(t)}) \log(1 - r_i) \right]
  $$
- 默认预测损失仍仅在获批样本上计算，与 RMT-Net 一致。

### 4. 核心创新总结
- **首次将多任务学习应用于信用评分中的 MNAR 问题建模**，突破传统重加权或半监督方法的局限。
- **提出基于拒绝概率的门控机制**：拒绝概率越高，D/N-Net 越依赖 R/A-Net 的信息，实现**样本自适应的信息共享**。
- **理论与实证结合**：通过真实数据和定理证明两个任务的正相关性，为信息共享提供依据。
- **扩展至多策略场景（RMT-Net++）**：支持动态审批策略，增强模型在真实金融环境中的适用性。
- **损失函数设计**：仅对获批样本计算默认预测损失，避免对无标签样本的错误监督，同时利用全部样本训练拒绝/批准任务。

RMT-Net 和 RMT-Net++ 通过显式建模任务间相关性与动态门控机制，有效缓解了 MNAR 偏差，显著提升了对获批和被拒样本的违约预测性能。