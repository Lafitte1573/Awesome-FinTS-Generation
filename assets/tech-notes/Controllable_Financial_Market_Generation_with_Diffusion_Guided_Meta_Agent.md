# Controllable Financial Market Generation with Diffusion Guided Meta Agent

本文提出的方法为**扩散引导元智能体（Diffusion Guided meta Agent, DiGA）**，旨在实现**可控制的金融市场价格流生成**，其核心思想是通过**两阶段架构**将宏观市场情景控制与微观订单生成解耦，结合扩散模型的动态建模能力与金融经济先验的生成机制，实现高保真、强可控的市场仿真。

---

### **1. 问题形式化：可控制金融市场的生成**

传统市场仿真仅生成无条件的订单流，而DiGA将问题建模为**条件生成任务**：给定一个目标市场情景（由宏观指标如日收益率、日内波动率、价格振幅等定义），生成与该情景高度一致的订单流。

- **控制目标**：设 $ a = \mathcal{F}(O) $ 为从真实订单流 $ O $ 中计算出的市场指标（如日收益率），目标是使生成的订单流 $ \tilde{O} \sim p_{\mathcal{M}}(O|a) $ 满足 $ \tilde{a} = \mathcal{F}(\tilde{O}) \approx a $，即最小化：
  $$
  \min_{\mathcal{M}} \mathbb{E}_{a, \tilde{O} \sim p_{\mathcal{M}}(O|a)} \left[ \| \mathcal{F}(\tilde{O}) - a \|^{2} \right]
  $$

- **保真度目标**：生成的订单流需在“风格化事实”（stylized facts）上与真实市场分布一致，如收益率自相关性、波动率聚集性、订单不平衡比等，最小化真实分布与生成分布之间的KL散度：
  $$
  \min_{\mathcal{M}} \mathbb{E}_{O \sim q(O), \tilde{O} \sim p_{\mathcal{M}}(O|\cdot)} \mathcal{D} \left( p(\mathcal{F}'(\tilde{O})) \parallel p(\mathcal{F}'(O)) \right)
  $$

---

### **2. DiGA模型架构：两阶段设计**

为克服直接在原始订单流（高维、不规则长度、高噪声）上应用扩散模型的困难，DiGA采用**两阶段分层建模**：

#### **阶段一：元控制器（Meta Controller）——扩散模型建模市场状态动态**

- **市场状态定义**：将每个交易日的分钟级市场状态表示为 $ \mathbf{x}_t = \{ \Delta r_t, \lambda_t \} $，其中：
  - $ \Delta r_t $：第 $ t $ 分钟的中价回报率（mid-price return rate）
  - $ \lambda_t $：第 $ t $ 分钟的订单到达率（order arrival rate）

- **扩散模型建模**：将每个交易日的分钟级状态序列 $ \mathbf{x} = \{ \mathbf{x}_1, \dots, \mathbf{x}_{T} \} $ 视为一个样本，使用**条件扩散模型**（Conditional Diffusion Model）学习其分布 $ q(\mathbf{x}|c) $，其中 $ c $ 为控制目标（如目标日收益率）。

- **扩散过程**：
  - 前向过程：对真实状态 $ \mathbf{x}_0 $ 加入高斯噪声，逐步生成噪声样本 $ \mathbf{x}_n $：
    $$
    \mathbf{x}_n = \sqrt{\bar{\alpha}_n} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_n} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
    $$
  - 反向过程：训练一个U-Net网络 $ \boldsymbol{\epsilon}_\theta(\mathbf{x}_n, n, \phi(c)) $ 预测噪声 $ \boldsymbol{\epsilon} $，训练目标为：
    $$
    L_C = \mathbb{E}_{\mathcal{H}_e(O), c, \boldsymbol{\epsilon}, n} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_n, n, \phi(c)) \|^{2} \right]
    $$
    其中 $ \phi(c) $ 为控制目标编码器（见下文）。

- **控制目标编码器 $ \phi(c) $**：
  - **离散编码**：将指标值按分位数划分为5个离散区间（Lower, Low, Medium, High, Higher），每个区间映射为类别标签，使用嵌入矩阵学习潜在表示。
  - **连续编码**：使用全连接网络直接将归一化后的指标值映射为低维向量。
  - **无分类器引导（Classifier-Free Guidance）**：训练时以0.5概率随机丢弃条件，实现条件与无条件采样联合训练；采样时使用线性插值：
    $$
    \tilde{\boldsymbol{\epsilon}}_{\theta, \phi}(\mathbf{x}_n, n, c) = (1 - s) \boldsymbol{\epsilon}_\theta(\mathbf{x}_n, n) + s \boldsymbol{\epsilon}_\theta(\mathbf{x}_n, n, \phi(c))
    $$
    其中 $ s $ 为引导强度超参数。

- **采样**：采用DDIM采样加速，从 $ \mathbf{x}_N \sim \mathcal{N}(0, \mathbf{I}) $ 逐步去噪，得到生成的市场状态序列 $ \tilde{\mathbf{x}}_0 $。

#### **阶段二：订单生成器（Order Generator）——基于金融经济先验的元智能体**

- **模拟交易所**：复现真实的双拍卖市场机制，用于接收订单、撮合成交、更新订单簿和价格序列。

- **元智能体（Meta Agent）**：作为市场中所有交易者的代表，其行为由**金融经济先验**驱动，而非纯数据驱动。其核心是基于**常数相对风险厌恶（CARA）效用函数**的理性决策框架。

- **订单生成流程（每分钟）**：
  1. **唤醒机制**：根据元控制器输出的 $ \lambda_t $，按指数分布采样间隔 $ \delta_i \sim \text{Exp}(\lambda_t) $，决定何时“唤醒”一次交易决策。
  2. **创建智能体（Actor Agent）**：每次唤醒生成一个异质智能体，其行为由三个成分加权决定：
     - **基本面（Fundamental）**：$ r_t $，由元控制器输出的中价回报率。
     - **图表分析（Chartist）**：历史平均回报 $ \bar{r} $，来自模拟交易所。
     - **噪声（Noise）**：小高斯扰动 $ r_\sigma \sim \mathcal{N}(0, \sigma^2) $。
     - 综合预期回报：$ \hat{r} = g_f r_t + g_c \bar{r} + g_n r_\sigma $，权重 $ g_f, g_c, g_n \sim \text{Exp}(\cdot) $，且 $ \mathbb{E}[g_f] > \mathbb{E}[g_c] > \mathbb{E}[g_n] $，体现基本面主导。
  3. **价格预测与需求函数**：
     - 预测未来价格：$ \hat{p}_t = p_t \exp(\hat{r}) $
     - 基于CARA效用推导需求函数：$ u(p) = \frac{\ln(\hat{p}_t / p)}{a V} $，其中 $ a $ 为风险厌恶系数，$ V $ 为历史波动率。
     - 计算最低可接受价格 $ p_l $：满足 $ p_l (u(p_l) - S) = C $，其中 $ S $ 为持仓，$ C $ 为现金。
  4. **订单采样**：
     - 价格：$ p_i \sim \mathcal{U}(p_l, \hat{p}_t) $
     - 数量：$ q_i = u(p_i) - S $
     - 类型：$ o_i = \text{sign}(q_i) $（1为买入，0为卖出）
     - 时间戳：$ t_i = \sum_{j=1}^i \delta_j $

- **输出**：生成的订单流为 $ \tilde{\mathcal{O}} = \{ o_1, \dots, o_{\text{max}} \} \sim p(\mathcal{O} | \tilde{\mathbf{x}}, \gamma) $，其中 $ \gamma $ 为元智能体的固定参数集合（如风险厌恶系数、权重分布参数等）。

---

### **3. 核心创新与机制总结**

| 创新点 | 说明 |
|--------|------|
| **双阶段解耦** | 将宏观控制（扩散模型建模市场状态）与微观生成（经济先验驱动订单）分离，解决原始订单流噪声大、长度不规则的建模难题。 |
| **扩散模型用于金融状态建模** | 首次将扩散模型应用于金融市场的状态序列（中价回报率+订单到达率）建模，而非直接建模订单流。 |
| **金融经济先验驱动** | 元智能体基于CARA效用最大化、异质行为成分（基本面/图表/噪声）等经典金融理论生成订单，确保生成行为具有经济合理性，而非黑箱拟合。 |
| **无分类器引导控制** | 通过条件/无条件联合训练与线性插值引导，实现对日收益率、波动率、振幅等关键市场指标的精确控制，支持生成特定极端情景（如暴跌、暴涨）。 |
| **端到端可控制生成** | 从输入“目标市场情景”到输出“符合该情景的订单流”形成完整闭环，为下游任务（如RL训练）提供可控、多样、逼真的仿真环境。 |

---

DiGA通过将**扩散模型的动态建模能力**与**金融经济理论的生成约束**相结合，首次实现了**可控制、高保真、经济合理**的金融订单流生成，为金融人工智能研究提供了全新的仿真范式。