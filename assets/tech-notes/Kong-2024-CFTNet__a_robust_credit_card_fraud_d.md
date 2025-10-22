# CFTNet: a robust credit card fraud detection model enhanced by counterfactual data augmentation

本文提出了一种基于反事实数据增强的三元组网络模型（CFTNet），用于提升信用卡欺诈检测的准确性和鲁棒性。该方法的核心思想是通过反事实样本（Counterfactual Examples, CFs）增强模型对特征与标签之间因果关系的学习能力，从而克服传统模型因依赖虚假相关性而导致的泛化能力差、鲁棒性低的问题。方法分为两个主要阶段：反事实样本生成和基于三元组的因果表示学习。

### 1. 反事实样本生成：基于P-DQN的强化学习优化

CFTNet首先将最优反事实样本的生成建模为一个马尔可夫决策过程（MDP），并采用P-DQN（Policy-based Deep Q-Network）强化学习算法求解最优反事实生成器 $ g^* $。

- **状态（State）**：在时间步 $ t $，状态 $ s_t = (x_t, f_t) $ 包含两个部分：当前扰动后的样本 $ x_t $ 和一个二进制向量 $ f_t \in \{0,1\}^{|\mathcal{F}|} $，表示到当前步为止哪些特征已被修改。
- **动作（Action）**：动作空间为离散-连续混合空间。在每一步，智能体首先从未修改的特征集合 $ \mathcal{F}_t \subset \mathcal{F} $ 中选择一个离散动作 $ k_t $（即选择一个特征），然后选择一个连续动作 $ \nu_{k_t} \in \mathbb{R} $，表示对该特征的修改幅度。因此，完整动作为 $ a_t = (k_t, \nu_{k_t}) $。
- **转移函数（Transition）**：执行动作 $ a_t $ 后，新状态 $ s_{t+1} = (x_{t+1}, f_{t+1}) $ 由以下规则确定：$ x_{t+1}[k_t] = x_t[k_t] + \nu_{k_t} $，且 $ f_{t+1}[k_t] = 1 $，其余特征保持不变。
- **奖励（Reward）**：奖励函数设计为鼓励智能体以最小扰动翻转预测标签：
  $$
  r(s_t, a_t) = 
  \begin{cases} 
  1 - \lambda (\ell_{\text{reg}}^t - \ell_{\text{reg}}^{t-1}) & \text{if } h_\omega(x_t) \neq h_\omega(x) \\
  -\lambda (\ell_{\text{reg}}^t - \ell_{\text{reg}}^{t-1}) & \text{otherwise}
  \end{cases}
  $$
  其中 $ \ell_{\text{reg}} $ 为 $ L^1 $ 范数惩罚项，控制扰动幅度；$ \lambda $ 为权衡参数。当样本被成功翻转时给予正奖励，同时惩罚过大的扰动。
- **策略（Policy）**：使用P-DQN网络优化策略 $ \pi_{\theta} $。P-DQN包含两个神经网络：
  - $ Q^{\theta_1}(s_t, k_t, \nu_{k_t}) $：评估动作值函数，输入为状态、离散动作和连续动作。
  - $ \nu_{k_t}^{\theta_2}(s_t) $：确定性策略网络，直接输出给定离散动作 $ k_t $ 下的最优连续动作。
  通过最小化以下损失函数进行训练：
  $$
  \mathcal{L}_Q(\theta_1) = \left[ Q^{\theta_1}(s_t, k_t, \nu_{k_t}) - y_t \right]^2, \quad \mathcal{L}_\pi(\theta_2) = -\sum_{k_t \in \mathcal{F}} Q(s_t, k_t, \nu_{k_t}^{\theta_2}(s_t))
  $$
  其中 $ y_t $ 为 $ n $-step TD目标。训练过程采用经验回放和随机梯度下降（SGD）进行参数更新。

最终，该强化学习框架输出最优反事实生成器 $ g^* $，对每个欺诈样本 $ x $ 生成其对应的反事实样本 $ \tilde{x}^* = g^*(x) $，该样本具有与原样本高度相似的特征分布，但预测标签被翻转为“正常”。

### 2. 三元组因果表示学习：CFTNet架构

在获得反事实样本后，CFTNet构建一个三元组学习框架，利用正样本、其反事实样本和负样本之间的结构关系，实现特征表示的解耦（disentanglement），以学习真正与欺诈标签 $ Y $ 相关的因果特征（$ X $），而非被混杂变量 $ Z $ 干扰的虚假相关性。

- **输入三元组**：每个训练样本为三元组 $ (x_i, \tilde{x}_i^*, x_j) $，其中：
  - $ x_i $：欺诈正样本（标签 $ y_i = 1 $），
  - $ \tilde{x}_i^* $：由 $ g^* $ 生成的反事实样本（标签 $ \tilde{y}_i^* = 0 $），
  - $ x_j $：随机选取的非欺诈负样本（标签 $ y_j = 0 $）。

- **特征编码器**：三者输入共享的三层DNN特征编码器 $ f_\omega $，使用ReLU激活函数，输出 $ d $ 维嵌入表示：
  $$
  f_\omega(x_i) = f_{\omega,3}(\omega_3 \cdot f_{\omega,2}(\omega_2 \cdot f_{\omega,1}(\omega_1 \cdot x_i + b_1) + b_2) + b_3)
  $$

- **表示解耦模块（Sep）**：为分离风险表示 $ X $（与欺诈因果相关）和混杂表示 $ Z $（与欺诈无关但与 $ X $ 相关），引入一个全连接层参数化模块 $ \mathrm{Sep}_\varpi(\cdot) $：
  $$
  r_i, z_i = \mathrm{Sep}_\varpi(f_\omega(x_i)), \quad r_i = \sigma(\varpi \cdot f_\omega(x_i)) \odot f_\omega(x_i), \quad z_i = (1 - \sigma(\varpi \cdot f_\omega(x_i))) \odot f_\omega(x_i)
  $$
  其中 $ \sigma $ 为Sigmoid函数，$ \odot $ 为逐元素乘法。$ r_i $ 为风险表示（对应 $ X $），$ z_i $ 为混杂表示（对应 $ Z $）。对于反事实样本 $ \tilde{x}_i^* $，同样得到 $ \tilde{r}_i^* $ 和 $ \tilde{z}_i^* $。

- **分类器**：仅使用风险表示 $ r_i $、$ \tilde{r}_i^* $ 和负样本的原始嵌入 $ f_\omega(x_j) $ 输入三层全连接分类器 $ \phi(\cdot) $，预测概率：
  $$
  \hat{y}_i = \phi(r_i), \quad \hat{\tilde{y}}_i^* = \phi(\tilde{r}_i^*), \quad \hat{y}_j = \phi(f_\omega(x_j))
  $$

### 3. 基于对比学习的表示解耦正则化

为强化三类样本在表示空间中的结构关系，CFTNet引入基于对比学习（Contrastive Learning, CL）的正则项 $ \ell_{\text{sim}} $：

- **语义相似性约束**：
  - 正样本与反事实样本：$ z_i \approx z_i^* $（混杂表示相似），但 $ r_i \neq r_i^* $（风险表示不同）。
  - 反事实样本与负样本：$ r_i^* \approx r_j $（风险表示相似，均属非欺诈）。
  - 正样本与负样本：$ r_i \neq r_j $（风险表示不同）。

- **对比损失函数**：
  $$
  \ell_{\text{sim}} = -\sum_{i=1}^{\lfloor B \rfloor} \log \frac{ \exp\left( \frac{s(z_i, z_i^*)}{\tau} \right) + \exp\left( \frac{s(z_i^*, f_\omega(x_j))}{\tau} \right) }{ \exp\left( \frac{s(r_i, r_i^*)}{\tau} \right) + \exp\left( \frac{s(r_i, z_i)}{\tau} \right) + \exp\left( \frac{s(r_i, f_\omega(x_j))}{\tau} \right) }
  $$
  其中 $ s(\cdot, \cdot) $ 为余弦相似度，$ \tau $ 为温度系数。该损失鼓励模型拉近混杂表示对 $ (z_i, z_i^*) $ 和 $ (z_i^*, f_\omega(x_j)) $，同时推远风险表示对 $ (r_i, r_i^*) $、$ (r_i, z_i) $ 和 $ (r_i, f_\omega(x_j)) $。

### 4. 实例加权的因果分类损失

为进一步提升模型对因果特征的敏感性，引入基于混杂表示相似度的实例加权机制：

- **实例权重**：对每个正样本 $ i $，其权重 $ w_i = s(z_i, z_i^*) $，即其反事实样本与原样本在混杂空间的相似度。相似度越高，说明该样本的“风险表示”被有效解耦，其预测更可信，应赋予更高权重。
- **加权分类损失**：
  $$
  \ell_{\text{wcls}} = \sum_{i=1}^{|B|} \sum_{k \in \{i, i^*, j\}} w_k \left[ y_k \log(\hat{y}_k) + (1 - y_k) \log(1 - \hat{y}_k) \right], \quad w_k = \begin{cases} s(z_i, z_i^*) & \text{if } k = i \\ 1 & \text{otherwise} \end{cases}
  $$
  该损失函数在形式上与因果推断中的后门调整公式（Eq. 2）一致，即对风险表示 $ X $ 进行干预，消除混杂变量 $ Z $ 对 $ Y $ 的间接影响。

### 5. 总体损失函数

CFTNet的最终损失函数为三部分之和：
$$
\ell_{\text{total}} = \ell_{\text{wcls}} + \alpha \ell_{\text{sim}} + \Omega
$$
其中 $ \alpha $ 为超参数，控制对比正则项的强度，$ \Omega $ 为模型参数正则项（如L2正则化）。

综上，CFTNet通过强化学习生成高质量反事实样本，再结合三元组对比学习与实例加权机制，在特征表示空间中显式解耦因果特征与混杂特征，从而构建一个能捕捉真实因果关系、鲁棒性强、泛化能力优的信用卡欺诈检测模型。