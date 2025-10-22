# Article Enhancing Portfolio Performance through Financial Time-Series Decomposition-Based Variational Encoder-Decoder Data Augmentation

本文提出的方法包括两个核心组成部分：**金融时间序列分解的变分编码器-解码器（FED）** 和 **基于FED的两分类投资组合分散化方法（FED2Port）**。以下为对这两个方法的详细介绍：

---

### **1. FED：金融时间序列分解的变分编码器-解码器**

FED 是一种**模式中心的数据增强方法**，旨在解决金融时间序列数据中**训练样本不足**和**不确定性缺失**（uncertainty deficiency）两大核心问题。其核心思想是：**将金融时间序列的潜在表示（latent variable）分解为三个具有明确金融意义的成分——趋势（trend）、离散度（dispersion）和残差（residual）——并在潜在空间中对这些成分进行建模与重构，从而生成更真实、更具多样性的合成时间序列数据。**

#### **（1）时间序列成分分解**
FED 将金融时间序列的潜在表示 $\mathbf{h}_t$ 分解为三个独立的、概率化的成分：
- **趋势成分** $\boldsymbol{\nu}_t \sim \mathcal{N}(\boldsymbol{\mu}_{\nu t}, \boldsymbol{\Sigma}_{\nu t})$：代表金融时间序列在时间 $t$ 的平均回报方向（即均值漂移），反映长期趋势。
- **离散度成分** $\boldsymbol{\tau}_t \sim \mathcal{N}(\boldsymbol{\mu}_{\tau t}, \boldsymbol{\Sigma}_{\tau t})$：代表金融时间序列在时间 $t$ 的波动性（即标准差），反映市场不确定性或风险水平。
- **残差成分** $\boldsymbol{\xi}_t \sim \mathcal{N}(\boldsymbol{\mu}_{\xi t}, \boldsymbol{\Sigma}_{\xi t})$：代表无法由趋势和离散度解释的随机噪声或非结构化波动。

这三个成分通过**乘积形式**组合成完整的潜在变量：
$$
\mathbf{h}_t = \boldsymbol{\nu}_t \times \boldsymbol{\tau}_t \times \boldsymbol{\xi}_t
$$

由于三个成分均为多元正态分布，其乘积仍为多元正态分布（基于多元正态分布乘积的闭合性质），因此可精确计算组合后潜在变量 $\mathbf{h}_t \sim \mathcal{N}(\boldsymbol{\mu}_{ht}, \boldsymbol{\Sigma}_{ht})$ 的均值和协方差：
$$
\begin{aligned}
\boldsymbol{\Sigma}_{1t} &= (\boldsymbol{\Sigma}_{\nu t}^{-1} + \boldsymbol{\Sigma}_{\tau t}^{-1})^{-1} \\
\boldsymbol{\mu}_{1t} &= \boldsymbol{\Sigma}_{1t} \boldsymbol{\Sigma}_{\nu t}^{-1} \boldsymbol{\mu}_{\nu t} + \boldsymbol{\Sigma}_{1t} \boldsymbol{\Sigma}_{\tau t}^{-1} \boldsymbol{\mu}_{\tau t} \\
\boldsymbol{\Sigma}_{ht} &= (\boldsymbol{\Sigma}_{1t}^{-1} + \boldsymbol{\Sigma}_{\xi t}^{-1})^{-1} = (\boldsymbol{\Sigma}_{\nu t}^{-1} + \boldsymbol{\Sigma}_{\tau t}^{-1} + \boldsymbol{\Sigma}_{\xi t}^{-1})^{-1} \\
\boldsymbol{\mu}_{ht} &= \boldsymbol{\Sigma}_{ht} \boldsymbol{\Sigma}_{\nu t}^{-1} \boldsymbol{\mu}_{\nu t} + \boldsymbol{\Sigma}_{ht} \boldsymbol{\Sigma}_{\tau t}^{-1} \boldsymbol{\mu}_{\tau t} + \boldsymbol{\Sigma}_{ht} \boldsymbol{\Sigma}_{\xi t}^{-1} \boldsymbol{\mu}_{\xi t}
\end{aligned}
$$

#### **（2）变分编码器-解码器架构**
FED 采用**三重编码器-解码器结构**，分别建模三个成分：
- **趋势编码器** $q_{\phi_\nu}(\nu_t | \mathbf{x}_t)$：从输入的 $d$-日对数收益率向量 $\mathbf{x}_t$ 推断趋势成分的后验分布。
- **离散度编码器** $q_{\phi_\tau}(\tau_t | \mathbf{x}_t)$：推断离散度成分的后验分布。
- **残差编码器** $q_{\phi}(\mathbf{h}_t | \mathbf{x}_t)$：推断完整潜在变量 $\mathbf{h}_t$ 的后验分布。

每个编码器均使用**重参数化技巧**（reparameterization trick）实现可微分采样，以支持端到端训练。

解码器 $p_{\theta_\nu}(m_t | \nu_t)$、$p_{\theta_\tau}(s_t | \tau_t)$ 和 $p_{\theta}(\mathbf{x}_t | \mathbf{h}_t)$ 分别从采样出的成分重建：
- 趋势 $m_t$（均值）
- 离散度 $s_t$（标准差）
- 完整的 $d$-日对数收益率向量 $\tilde{\mathbf{x}}_t$

#### **（3）损失函数：三重证据下界（ELBO）联合优化**
FED 的训练目标是最大化三个成分的证据下界（ELBO）的加权组合：
$$
\begin{aligned}
L_{FED} := & \ \alpha \left( -D_{KL}\left[ q_{\phi_\nu}(\nu_t | \mathbf{x}_t) \ || \ p(\nu_t) \right] + \mathbb{E}_{q_{\phi_\nu}(\nu_t | \mathbf{x}_t)} \left[ \log p_{\theta_\nu}(m_t | \nu_t) \right] \right) \\
+ & \ \beta \left( -D_{KL}\left[ q_{\phi_\tau}(\tau_t | \mathbf{x}_t) \ || \ p(\tau_t) \right] + \mathbb{E}_{q_{\phi_\tau}(\tau_t | \mathbf{x}_t)} \left[ \log p_{\theta_\tau}(s_t | \tau_t) \right] \right) \\
+ & \ \gamma \left( -D_{KL}\left[ q_{\phi}(\mathbf{h}_t | \mathbf{x}_t) \ || \ p(\mathbf{h}_t) \right] + \mathbb{E}_{q_{\phi}(\mathbf{h}_t | \mathbf{x}_t)} \left[ \log p_{\theta}(\mathbf{x}_t | \mathbf{h}_t) \right] \right)
\end{aligned}
$$
其中 $\alpha, \beta, \gamma$ 为超参数，用于平衡三个任务的重要性。

该损失函数确保：
- 潜在成分的后验分布接近先验（正则化）；
- 解码器能准确重构原始数据；
- 趋势和离散度成分被显式建模，从而恢复历史中“消失”的市场不确定性。

通过 FED，模型能够**生成具有真实统计特性的合成金融时间序列**，其趋势、波动性和噪声模式均与历史数据一致，但具有多样性，从而**弥补了历史数据中因事件确定化导致的“不确定性缺失”问题**。

---

### **2. FED2Port：基于FED的两分类投资组合分散化方法**

FED2Port 是一个**强化学习（RL）驱动的投资组合决策框架**，其核心是**将 FED 生成的合成数据作为市场环境输入**，使 RL 算法能够学习在**更全面、更真实的市场不确定性**下进行资产配置。

#### **（1）环境定义**
- **状态（State）**：当前投资组合的回报向量  
  $$
  \mathbf{s}_t = a_{t-1, hr} \cdot \mathbf{x}_{t, hr} + a_{t-1, lr} \cdot \mathbf{x}_{t, lr}
  $$
  其中 $\mathbf{x}_{t, hr}$ 和 $\mathbf{x}_{t, lr}$ 分别为高风险与低风险资产的 $d$-日对数收益率向量，$a_{t-1, hr}$ 和 $a_{t-1, lr}$ 为上一时刻的资产权重（满足 $a_{t-1, hr} + a_{t-1, lr} = 1$）。

- **动作（Action）**：当前时刻的资产权重向量  
  $$
  \mathbf{a}_t = \begin{bmatrix} a_{t, hr} \\ a_{t, lr} \end{bmatrix}, \quad a_{t, hr}, a_{t, lr} \geq 0, \quad a_{t, hr} + a_{t, lr} = 1
  $$

- **奖励（Reward）**：采用**市场自适应比率**（Market-Adaptive Ratio, MAR）  
  $$
  r_t(\tilde{\mathbf{x}}_{t+d, hr}, \tilde{\mathbf{x}}_{t+d, lr}, \mathbf{a}_t) = \frac{(\bar{R}_p - R_f)^{\rho_{hr}}}{\sigma_p^{1 / \rho_{hr}}}
  $$
  其中：
  - $\bar{R}_p$：投资组合的预期回报；
  - $\sigma_p$：投资组合回报的标准差；
  - $R_f = 0$：无风险利率（本文设为0）；
  - $\rho_{hr} = \frac{2}{1 + e^{-R_{hr}}}$：**市场状态自适应参数**，由高风险资产的回报 $R_{hr}$ 动态决定；
    - 当 $R_{hr} > 0$（牛市）：$\rho_{hr} \to 2$，奖励更关注超额收益；
    - 当 $R_{hr} < 0$（熊市）：$\rho_{hr} \to 0$，奖励更关注风险控制（分母权重增大）。

  该奖励函数**自动适应市场周期**，使策略在牛市中追求收益、在熊市中注重保本，克服了传统 Sharpe 比率对上下波动一视同仁的缺陷。

#### **（2）FED 的集成**
FED2Port 的关键创新在于：
- **使用 FED 生成的合成数据** $\tilde{\mathbf{x}}_{t+d, hr}$ 和 $\tilde{\mathbf{x}}_{t+d, lr}$ 作为未来收益的输入，而非仅依赖历史观测；
- 每次决策时，FED 从学习到的分布中**随机采样多个可能的未来路径**，使 RL 算法在训练中暴露于**一个完整的市场不确定性谱**（包括历史中未发生的、但概率上可能的市场情景）；
- 因此，RL 策略 $\pi_\omega(\mathbf{s}_t) = \mathbf{a}_t$ 不仅学习历史模式，更学习**如何在未知但合理的市场波动中做出稳健决策**。

#### **（3）优化目标**
FED2Port 的目标是最大化预期奖励：
$$
\max_\omega \mathbb{E}_{\tilde{\mathbf{x}}_{t+d, hr}, \tilde{\mathbf{x}}_{t+d, lr}} \left[ r_t(\tilde{\mathbf{x}}_{t+d, hr}, \tilde{\mathbf{x}}_{t+d, lr}, \mathbf{a}_t) \right]
$$
其中期望在 FED 生成的合成数据分布上计算，而非仅在历史数据上。

---

### **总结：FED 与 FED2Port 的协同机制**
- **FED**：通过分解并重建潜在成分，**复活历史中“消失”的市场不确定性**，生成多样化的合成数据，解决**数据不足**与**不确定性缺失**问题；
- **FED2Port**：将 FED 生成的合成数据作为强化学习的**动态市场环境**，结合**市场自适应奖励函数**，使 RL 策略在训练阶段即学习应对**真实世界中可能的极端与未知情景**，从而提升投资组合在真实市场中的**鲁棒性与适应性**。

二者共同构成一个**从数据增强到决策优化的端到端框架**，首次将**金融时间序列的结构分解**与**变分生成建模**结合，并应用于**强化学习投资组合管理**，显著提升模型对不确定性的建模能力与最终投资绩效。