# TimeHF: Billion-Scale Time Series Models Guided by Human Feedback

本文提出的时间序列大模型框架TimeHF，是一个面向工业级时间序列预测任务的、基于人类反馈强化学习的百亿参数级模型构建与优化体系，其核心方法由三个紧密衔接的阶段构成：**基础模型训练（Patch Convolutional Large Time Series Model, PCLTM）**、**监督微调（Supervised Fine-Tuning, SFT）** 和 **时序策略优化（Time-Series Policy Optimization, TPO）**。以下为各部分的详细说明：

---

### **1. 基础模型：Patch Convolutional Large Time Series Model (PCLTM)**

PCLTM 是一个纯时间序列大模型（pure LTM），不依赖语言模型或视觉模型的预训练权重，而是从零开始设计，专为捕捉长时序依赖和跨通道复杂模式而优化。其架构包含三个核心组件：

#### **1.1 跨块投影层（Cross-Patch Projection Layers）**
- **动机**：传统分块编码方法（如线性投影）仅将每个时间块（patch）独立编码，忽略了块与块之间的长程依赖。
- **方法**：引入**卷积网络模块（patchConv）**，在将原始时间序列划分为 $ s $ 个长度为 $ d_p $ 的块后，通过卷积操作在**跨块通道**上聚合信息，将每个块的表示从 $ \mathbb{R}^{d_p} $ 映射到更高维空间 $ \mathbb{R}^{d_e} $，从而在嵌入阶段即融合了超出单块范围的时序上下文。
- **公式**：  
  $$
  y_{\text{patch.embed}} = \text{patchConv}(y_{\text{patch}}) \in \mathbb{R}^{s \times d_e}
  $$
  其中 $ y_{\text{patch}} \in \mathbb{R}^{s \times d_p} $ 为分块后的输入序列。

#### **1.2 带时序位置编码的分组注意力机制（Grouped Query Attention with Temporal Position Encoding）**
- **动机**：标准Transformer的自注意力计算复杂度高，且缺乏对时间顺序的精细建模。
- **方法**：
  - 使用**分组查询注意力（GQA）**：将查询（Query）分组共享键（Key）和值（Value），大幅降低参数量和计算开销，同时保留注意力的表达能力。
  - 引入**旋转位置编码（Rotary Position Embedding, ROPE）**：将时间位置信息 $ y_{\text{time.index}} \in \mathbb{R}^{s \times 1} $ 编码为旋转矩阵 $ R_{i-j} $，直接作用于查询和键向量，实现位置感知的注意力计算。
- **公式**：
  $$
  \begin{aligned}
  q_i &= W_q y_{\text{patch.embed},i}, \quad k_i = W_k y_{\text{patch.embed},i}, \quad v_i = W_v y_{\text{patch.embed},i} \\
  \text{Atten}_{i,j} &= \text{softmax}(q_i^T R_{i-j} k_j), \quad R_{i-j} = \text{ROPE}(y_{\text{time-index}})_{i-j} \\
  y_{\text{atten.ffn}} &= \text{FFN}(\text{Atten} * v) \in \mathbb{R}^{s \times d_e}
  \end{aligned}
  $$
  其中 FFN 为标准前馈网络。

#### **1.3 输出层（Output Layers）**
- 将Transformer输出的 $ \mathbb{R}^{s \times d_e} $ 特征进行**展平（flatten）**，再通过一个**多层感知机（MLP）** 映射到目标预测长度 $ H $，输出未来 $ H $ 个时间步的预测值：
  $$
  \hat{y}_{L+1:L+H} = \text{MLP}(\text{flatten}(y_{\text{atten.ffn}})) \in \mathbb{R}^H
  $$
- 训练时使用**均方误差（MSE）** 作为损失函数，直接优化点预测精度。

---

### **2. 监督微调（Supervised Fine-Tuning, SFT）**

- **目的**：将预训练的PCLTM适配到特定下游场景（如长尾商品、季节性商品等），利用领域专家标注或高置信度的标注数据提升预测性能。
- **方法**：
  - 构建**场景专属的微调数据集**，包含JD.com的各类产品销售序列（标准品、新品、长尾品、爆款、季节品、间歇品）。
  - 对PCLTM进行端到端微调，目标函数为：
    $$
    \hat{y}_{L+1:L+H}^{\text{SFT}} = f_{\theta}^{\text{SFT}}(y_{1:L}^{\text{SFT}})
    $$
- **效果**：SFT显著提升模型在特定场景下的泛化能力，为后续RLHF提供高质量的初始策略（即 $ \pi^{\text{SFT}} $）。

---

### **3. 时序策略优化（Time-Series Policy Optimization, TPO）**

TPO 是本文的核心创新，是**首个专为纯时间序列模型设计的RLHF框架**，解决了传统RLHF（如PPO、DPO）无法直接应用于确定性时间序列模型的根本性障碍。

#### **3.1 核心挑战**
- 传统时间序列模型输出为**确定性数值**，无概率分布，无法计算KL散度、策略概率或TD误差。
- 无法直接使用PPO（需策略、价值、奖励三模型）或RLOO（需在线采样）等方法。

#### **3.2 关键设计**

##### **3.2.1 反馈对比对（Feedback Contrast Pair）**
- **构造方式**：由JD.com资深数据分析师构建多个**小规模专家模型**（如XGBoost、统计模型），针对同一输入序列 $ y_{1:L}^{\text{RL}} $，生成两个预测：
  - **“优选”预测** $ \hat{y}_{L+1:L+H}^{\text{chosen}} $：专家认为更准确、更符合业务逻辑的预测。
  - **“拒绝”预测** $ \hat{y}_{L+1:L+H}^{\text{rejected}} $：专家认为存在偏差、不合理或易产生幻觉的预测。
- **作用**：为TPO提供**人类专家的隐性知识**，使大模型学习“什么预测是好的，什么是坏的”，而非仅依赖数值误差。

##### **3.2.2 概率化预测（Probabilistic Prediction）**
- **方法**：为所有预测值（包括模型输出、优选预测、拒绝预测）**假设服从正态分布** $ \mathcal{N}(\mu, 1) $，其中 $ \mu $ 即为预测值本身，方差固定为1。
- **作用**：由此可计算任意预测的**概率密度**：
  $$
  \pi_{\text{RL}}(f_\phi^{\text{RL}}(y_{1:L}^{\text{RL}}); \mu_\phi^{\text{RL}}, \sigma_{\text{RL}}) = \text{PDF}(\hat{y}_{L+1:L+H}^{\text{RL}}; \mu_\phi^{\text{RL}}, 1)
  $$
  同理可得 $ \pi_{\text{chosen}} $ 和 $ \pi_{\text{rejected}} $，从而实现**概率化策略比较**，使KL项和策略比值在TPO目标函数中可计算。

##### **3.2.3 优势函数（Advantage Function）**
- **动机**：避免使用TD误差（不适用于多步预测），采用**REINFORCE风格**的优势估计。
- **定义**：
  $$
  \hat{A}(y_{1:L}^{\text{RL}}, \hat{y}_{L+1:L+H}^{\text{RL}}) = R(y_{1:L}^{\text{RL}}, \hat{y}_{L+1:L+H}^{\text{RL}}) - b_{ts}
  $$
- **奖励函数 $ R $** 包含两部分：
  $$
  \begin{aligned}
  R &= \underbrace{\log(\pi_{\text{chosen}})}_{\text{对齐专家偏好}} - \beta \underbrace{\log\left( \frac{\pi_{\text{RL}}}{\pi_{\text{SFT}}} \right)}_{\text{KL惩罚，防止偏离SFT基线}}
  \end{aligned}
  $$
- **基线 $ b_{ts} $**：使用“拒绝预测”的概率密度作为基线：
  $$
  b_{ts} = \log(\pi_{\text{rejected}})
  $$
  该设计避免了RLOO中昂贵的在线采样，直接利用专家标注的“坏预测”作为对比基准，降低方差，提升学习稳定性。

#### **3.3 TPO目标函数**
最终优化目标为：
$$
\begin{aligned}
\text{objective}(\phi) = & \quad \hat{A}(y_{1:L}^{\text{RL}}, \hat{y}_{L+1:L+H}^{\text{RL}}) \cdot \frac{\pi_{\text{RL}}(f_\phi^{\text{RL}}(y_{1:L}^{\text{RL}}); \mu_\phi^{\text{RL}}, \sigma_{\text{RL}})}{\pi_{\text{RL}}(f_\phi^{\text{RL}_{\text{old}}}(y_{1:L}^{\text{RL}}); \mu_\phi^{\text{RL}_{\text{old}}}, \sigma_{\text{RL}})} \\
& + \gamma \left( \alpha \cdot \text{MSE}(f_\phi^{\text{RL}}(y_{1:L}^{\text{RL}}), \hat{y}_{L+1:L+H}^{\text{chosen}}) + \omega \cdot \text{MSE}(f_\phi^{\text{RL}}(y_{1:L}^{\text{RL}}), \hat{y}_{L+1:L+H}^{\text{rejected}}) \right)
\end{aligned}
$$
- **第一项**：策略比值（类似PPO的裁剪目标），引导模型向“优选预测”靠近。
- **第二项**：**双MSE损失**：鼓励模型输出接近“优选预测”，同时远离“拒绝预测”，显式利用反馈对比对的监督信号。
- **参数**：
  - $ \gamma $：控制时序损失权重；
  - $ \alpha, \omega $：分别控制“优选”与“拒绝”MSE的权重；
  - $ \beta $：控制KL惩罚强度，防止策略过度偏离SFT基线。

#### **3.4 TPO优势总结**
- **仅训练一个模型**（策略模型），无需训练价值函数或奖励模型，计算成本远低于PPO/RLOO。
- **无需在线采样**，完全基于离线反馈对比对，训练稳定高效。
- **兼容确定性模型**，通过概率化假设解决RLHF在时间序列中的适用性难题。
- **显式引入专家知识**，通过“好/坏”预测对，使模型学习到业务语义层面的判断标准，而非仅数值误差。

---

### **总结：TimeHF方法体系**
TimeHF通过三阶段协同构建了**首个百亿参数级、可工业部署的纯时间序列大模型**：
1. **PCLTM**：以卷积跨块编码 + GQA + ROPE 构建高效、长程依赖建模的基础模型；
2. **SFT**：利用场景数据微调，提升任务适应性；
3. **TPO**：首创基于反馈对比对和概率化假设的RLHF框架，首次实现人类专家隐性知识向大模型的高效迁移。

该框架不仅在JD.com实现33.21%的预测精度提升，更在公开数据集上达到SOTA水平，为时间序列大模型的发展提供了可复用、可扩展、工业落地的完整范式。