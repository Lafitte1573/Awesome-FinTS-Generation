# Article Enhancing Portfolio Performance through Financial Time-Series Decomposition-Based Variational Encoder-Decoder Data Augmentation

本文提出的方法主要包括两个核心组成部分：**金融时间序列分解的变分编码器-解码器（FED）** 和 **基于FED的两类资产组合分散化方法（FED2Port）**。以下是对这两个方法的详细介绍：

---

### **1. FED：金融时间序列分解的变分编码器-解码器**

FED 是一种面向金融时间序列的数据增强方法，旨在解决历史金融数据中“不确定性缺失”和“训练数据不足”的问题。其核心思想是：**将金融时间序列的潜在表示（latent representation）分解为三个具有明确金融意义的组件——趋势（trend）、离散度（dispersion）和残差（residual）**，并在潜在空间中对这些组件进行建模和生成，从而生成更具真实性和多样性的合成时间序列数据。

#### **1.1 数据分解结构**
FED 将金融时间序列的潜在变量 $\mathbf{h}_t$ 分解为三个独立的概率组件，每个组件对应一个金融动态特征：
- **趋势组件** $\boldsymbol{\nu}_t \sim \mathcal{N}(\boldsymbol{\mu}_{\nu t}, \boldsymbol{\Sigma}_{\nu t})$：表示时间序列的长期方向，即平均收益率，反映市场整体上升或下降趋势。
- **离散度组件** $\boldsymbol{\tau}_t \sim \mathcal{N}(\boldsymbol{\mu}_{\tau t}, \boldsymbol{\Sigma}_{\tau t})$：表示收益率的波动性，即标准差，反映市场不确定性或风险水平。
- **残差组件** $\boldsymbol{\xi}_t \sim \mathcal{N}(\boldsymbol{\mu}_{\xi t}, \boldsymbol{\Sigma}_{\xi t})$：表示无法由趋势和离散度解释的随机噪声或非结构化波动。

这三个组件通过乘积形式组合成完整的潜在变量：
$$
\mathbf{h}_t = \boldsymbol{\nu}_t \times \boldsymbol{\tau}_t \times \boldsymbol{\xi}_t
$$

由于三个组件均为多元正态分布，其乘积仍为多元正态分布（根据多元正态分布的乘积性质），因此可计算出组合后潜在变量 $\mathbf{h}_t \sim \mathcal{N}(\boldsymbol{\mu}_{ht}, \boldsymbol{\Sigma}_{ht})$ 的均值和协方差：
$$
\begin{aligned}
\boldsymbol{\Sigma}_{1t} &= (\boldsymbol{\Sigma}_{\nu t}^{-1} + \boldsymbol{\Sigma}_{\tau t}^{-1})^{-1} \\
\boldsymbol{\mu}_{1t} &= \boldsymbol{\Sigma}_{1t} \boldsymbol{\Sigma}_{\nu t}^{-1} \boldsymbol{\mu}_{\nu t} + \boldsymbol{\Sigma}_{1t} \boldsymbol{\Sigma}_{\tau t}^{-1} \boldsymbol{\mu}_{\tau t} \\
\boldsymbol{\Sigma}_{ht} &= (\boldsymbol{\Sigma}_{1t}^{-1} + \boldsymbol{\Sigma}_{\xi t}^{-1})^{-1} = (\boldsymbol{\Sigma}_{\nu t}^{-1} + \boldsymbol{\Sigma}_{\tau t}^{-1} + \boldsymbol{\Sigma}_{\xi t}^{-1})^{-1} \\
\boldsymbol{\mu}_{ht} &= \boldsymbol{\Sigma}_{ht} \boldsymbol{\Sigma}_{\nu t}^{-1} \boldsymbol{\mu}_{\nu t} + \boldsymbol{\Sigma}_{ht} \boldsymbol{\Sigma}_{\tau t}^{-1} \boldsymbol{\mu}_{\tau t} + \boldsymbol{\Sigma}_{ht} \boldsymbol{\Sigma}_{\xi t}^{-1} \boldsymbol{\mu}_{\xi t}
\end{aligned}
$$

#### **1.2 模型架构**
FED 采用**三重编码器-解码器结构**：
- **三个独立编码器**：分别编码趋势、离散度和残差组件，从输入的 $d$-日对数收益率向量 $\mathbf{x}_t$ 中推断出各自潜在变量的后验分布 $q_{\phi_\nu}(\nu_t | x_t)$、$q_{\phi_\tau}(\tau_t | x_t)$ 和 $q_{\phi}(\xi_t | x_t)$。
- **一个解码器**：以组合后的潜在变量 $\mathbf{h}_t$ 为输入，重构原始时间序列 $\tilde{\mathbf{x}}_t = D(\mathbf{h}_t)$。
- **重参数化技巧**：为实现端到端训练，使用重参数化技巧（reparameterization trick）将随机采样过程可微化，使梯度可反向传播。

#### **1.3 训练目标：联合最大化证据下界（ELBO）**
FED 的训练目标是最大化三个组件及其整体重构的证据下界（ELBO）的加权组合：
$$
\begin{aligned}
L_{FED} := &\ \alpha \left( -D_{KL} \left[ q_{\phi_\nu}(\nu_t | x_t) \parallel p(\nu_t) \right] + \mathbb{E}_{q_{\phi_\nu}(\nu_t | x_t)} \left[ \log p_{\theta_\nu}(m_t | \nu_t) \right] \right) \\
&+ \beta \left( -D_{KL} \left[ q_{\phi_\tau}(\tau_t | x_t) \parallel p(\tau_t) \right] + \mathbb{E}_{q_{\phi_\tau}(\tau_t | x_t)} \left[ \log p_{\theta_\tau}(s_t | \tau_t) \right] \right) \\
&+ \gamma \left( -D_{KL} \left[ q_{\phi}(h_t | x_t) \parallel p(h_t) \right] + \mathbb{E}_{q_{\phi}(h_t | x_t)} \left[ \log p_{\theta}(x_t | h_t) \right] \right)
\end{aligned}
$$
其中：
- $p_{\theta_\nu}(m_t | \nu_t)$、$p_{\theta_\tau}(s_t | \tau_t)$ 和 $p_{\theta}(x_t | h_t)$ 分别为趋势、离散度和原始序列的重构似然；
- $q_{\phi_\nu}(\nu_t | x_t)$、$q_{\phi_\tau}(\tau_t | x_t)$、$q_{\phi}(h_t | x_t)$ 为对应的近似后验；
- $p(\nu_t)$、$p(\tau_t)$、$p(h_t)$ 为先验分布（通常设为标准正态分布）；
- $\alpha, \beta, \gamma$ 为超参数，用于平衡三个组件的训练权重。

通过该目标，FED 不仅学习了原始数据的分布，还显式建模了金融时间序列中**趋势的演化、波动性的变化和随机噪声的结构**，从而在潜在空间中“复活”了历史数据中已消失的不确定性，生成具有真实金融动态特性的合成数据。

---

### **2. FED2Port：基于FED的两类资产组合分散化方法**

FED2Port 是一个强化学习（RL）驱动的资产配置框架，其核心创新在于：**将FED生成的合成金融时间序列作为环境输入，使RL代理在包含更丰富市场不确定性的环境中学习最优投资策略**。

#### **2.1 环境定义**
FED2Port 的强化学习环境由以下三个要素构成：
- **状态（State）**：当前投资组合的回报 $\mathbf{s}_t = a_{t-1,hr} \mathbf{x}_{t,hr} + a_{t-1,lr} \mathbf{x}_{t,lr}$，其中 $\mathbf{x}_{t,hr}$ 和 $\mathbf{x}_{t,lr}$ 分别为高风险与低风险资产的 $d$-日对数收益率向量，$a_{t-1,hr}$ 和 $a_{t-1,lr}$ 为上一时刻的资产权重。
- **动作（Action）**：当前时刻的资产权重向量 $\mathbf{a}_t = [a_{t,hr}, a_{t,lr}]^\top$，满足 $a_{t,hr} + a_{t,lr} = 1$，$a_{t,hr}, a_{t,lr} \geq 0$。
- **奖励（Reward）**：采用**市场自适应比率**（Market-Adaptive Ratio, MAR）：
  $$
  r_t(\tilde{\mathbf{x}}_{t+d,hr}, \tilde{\mathbf{x}}_{t+d,lr}, \mathbf{a}_t) = \frac{(\bar{R}_p - R_f)^{\rho_{hr}}}{\sigma_p^{1/\rho_{hr}}}
  $$
  其中：
  - $\bar{R}_p$ 为投资组合预期回报，$\sigma_p$ 为组合标准差，$R_f = 0$（无风险利率为零）；
  - $\rho_{hr} = \frac{2}{1 + e^{-R_{hr}}}$，为高风险资产回报 $R_{hr}$ 的函数，用于动态调整风险偏好：
    - 当 $R_{hr} > 0$（牛市）时，$\rho_{hr} \to 2$，奖励更重视超额收益；
    - 当 $R_{hr} < 0$（熊市）时，$\rho_{hr} \to 0$，奖励更重视降低波动性；
  - $\tilde{\mathbf{x}}_{t+d,hr}$ 和 $\tilde{\mathbf{x}}_{t+d,lr}$ 为FED生成的未来高风险与低风险资产的对数收益率序列。

#### **2.2 学习目标**
FED2Port 的目标是学习一个策略函数 $\pi_\omega(\mathbf{s}_t)$，使得在FED生成的合成市场环境中，**长期期望奖励最大化**：
$$
\max_\omega \mathbb{E}_{\tilde{\mathbf{x}}_{t+d,hr}, \tilde{\mathbf{x}}_{t+d,lr}} \left[ r_t(\tilde{\mathbf{x}}_{t+d,hr}, \tilde{\mathbf{x}}_{t+d,lr}, \mathbf{a}_t) \right]
$$

#### **2.3 创新性**
- **不确定性注入**：传统RL模型仅在历史数据上训练，导致模型“过拟合”于已发生的单一路径（不确定性缺失）。FED2Port 利用FED生成的**多条合成路径**，使RL代理在训练中暴露于**历史中存在的、但已消失的多种市场情景**（如高波动牛市、低波动熊市等），从而提升策略的鲁棒性。
- **动态风险偏好**：MAR奖励函数根据市场状态自适应调整风险偏好，使策略在不同市场周期中自动切换“进取”或“防御”模式。
- **端到端增强框架**：FED生成的合成数据直接作为RL环境的输入，形成“数据增强→环境模拟→策略学习”的闭环，实现对金融不确定性的系统性建模。

---

### **总结**
FED 通过**分解潜在空间中的趋势、离散度与残差**，并在变分框架下联合建模，实现了对金融时间序列深层结构的解析与再生，有效缓解了数据不足与不确定性缺失问题。FED2Port 则将FED生成的多样化合成数据作为强化学习环境，结合**市场自适应奖励函数**，训练出能够适应未来不确定性的稳健投资策略。二者共同构成一个**数据驱动、结构化、动态适应**的金融决策增强框架，显著提升了组合管理的性能与鲁棒性。