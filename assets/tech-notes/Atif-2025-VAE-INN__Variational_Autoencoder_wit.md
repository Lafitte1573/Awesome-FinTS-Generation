# VAE-INN: Variational Autoencoder with Integrated Neural Network Classifier for Imbalanced Credit Scoring, Utilizing Weighted Loss for Improved Accuracy

本文提出的方法称为 **VAE-INN（Variational Autoencoder with Integrated Neural Network Classifier）**，是一种用于处理不平衡信用评分问题的统一深度学习框架，其核心创新在于将**特征提取、类不平衡处理与分类预测**三者整合于单一端到端模型中，通过**加权损失函数**在潜在空间中直接优化 Minority 类（违约）的识别能力，从而显著降低 Type II 错误（漏判违约），减少银行损失。

以下是该方法的详细说明：

---

### **1. 模型整体架构：VAE 与神经网络分类器的深度集成**

VAE-INN 的架构由三个核心组件构成，形成一个**联合优化的单阶段系统**：

- **编码器（Encoder）**：将原始高维输入特征 $\mathbf{x}$ 映射到低维潜在空间，输出潜在变量的均值 $\mu$ 和对数方差 $\log \sigma^2$，二者共同定义一个高斯分布 $q_{\phi}(\mathbf{z}|\mathbf{x})$。
- **重参数化采样（Reparameterization Trick）**：从该高斯分布中采样潜在变量 $\mathbf{z}$：
  $$
  \mathbf{z} = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
  $$
  该操作使模型可微，支持反向传播。
- **解码器（Decoder）**：从 $\mathbf{z}$ 重构原始输入 $\hat{\mathbf{x}}$，目标是还原数据结构。
- **集成分类器（Integrated Neural Network Classifier）**：**直接以潜在变量 $\mathbf{z}$ 作为输入**，输出违约概率 $\hat{\mathbf{y}}$，其结构为一个全连接神经网络，参数记为 $\psi$。

该架构的关键在于：**分类器不作用于原始特征，而是作用于 VAE 学习出的潜在表示 $\mathbf{z}$**，从而实现特征提取与分类任务的**端到端联合优化**，避免传统多阶段流水线（如先 SMOTE 再特征选择再分类）带来的偏差累积。

---

### **2. 损失函数设计：三重目标 + 类别加权**

VAE-INN 的总损失函数由三部分组成，通过一个可调参数 $\alpha$ 控制分类任务的相对重要性：

$$
\mathcal{L}_{\text{loss}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}} + \alpha \mathcal{L}_{\text{class}}
$$

#### **(1) 重构损失（Reconstruction Loss）$\mathcal{L}_{\text{recon}}$**
衡量输入 $\mathbf{x}$ 与重建输出 $\hat{\mathbf{x}}$ 的差异，采用均方误差（MSE）：
$$
\mathcal{L}_{\text{recon}} = \sum_{i=1}^{N} (\mathbf{x}_i - \hat{\mathbf{x}}_i)^2
$$
确保潜在空间保留原始数据的关键信息，避免信息丢失。

#### **(2) KL 散度正则项（KL Divergence）$\mathcal{L}_{\text{KL}}$**
约束潜在分布 $q_{\phi}(\mathbf{z}|\mathbf{x})$ 接近标准正态先验 $p(\mathbf{z}) = \mathcal{N}(0, I)$，防止过拟合并促进潜在空间的连续性与泛化性：
$$
\mathcal{L}_{\text{KL}} = -0.5 \sum_{j=1}^{d} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)
$$
该正则项使潜在空间结构平滑、稠密，为分类器提供稳定输入。

#### **(3) 加权分类损失（Weighted Classification Loss）$\mathcal{L}_{\text{class}}$**
这是本方法的核心创新。分类器使用**类别加权的二元交叉熵损失**，对少数类（违约）赋予更高权重，以补偿数据不平衡：

$$
\mathcal{L}_{\text{class}} = - \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] \cdot w_i
$$

其中，样本 $i$ 的类别权重 $w_i$ 根据其所属类别 $l$ 计算：
$$
w_i = w_l = \frac{N}{2 n_l}
$$
- $N$：总样本数
- $n_l$：类别 $l$ 的样本数（$l \in \{0, 1\}$，1 为违约类）

该权重设计确保：**少数类的每个样本在损失中贡献更大**，迫使模型更关注违约客户的特征学习。

此外，该分类损失被乘以一个**可调超参数 $\alpha$**，用于平衡分类任务与重构/正则化任务的重要性：
- $\alpha$ 值越大 → 分类任务权重越高 → 模型更侧重于提升违约检测能力；
- $\alpha$ 值过小 → 模型可能偏向重构，忽略分类目标。

$\alpha$ 通过**交叉验证**在验证集上优化，以在保持良好重建质量的前提下最大化 Minority 类的分类性能（如 F1-score 或 AUC）。

---

### **3. 潜在空间的联合优化机制**

VAE-INN 的核心优势在于：**分类器的梯度直接反向传播至编码器**，从而在训练过程中**动态塑造潜在空间**，使其同时满足：

- **重构保真度**（解码器驱动）
- **分布正则性**（KL 散度驱动）
- **类别可分性**（加权分类器驱动）

具体而言：

- 在反向传播中，分类损失 $\mathcal{L}_{\text{class}}$ 的梯度 $\frac{\partial \mathcal{L}_{\text{class}}}{\partial \phi}$ 会更新编码器参数 $\phi$，促使编码器学习**对违约与非违约样本更具判别力的潜在表示**。
- 由于 $\mathcal{L}_{\text{class}}$ 对少数类样本赋予更高权重，编码器会**主动增强少数类特征的表达能力**，避免传统特征选择方法因追求整体方差而丢弃少数类关键特征的问题。
- 同时，KL 正则项防止潜在空间坍缩，确保类间边界清晰但不扭曲数据分布，从而**避免生成式方法（如 GAN/SMOTE）引入的虚假样本或噪声放大问题**。

因此，**潜在空间不再是被动的特征压缩结果，而是被主动优化为“对两类样本公平且可分”的结构**，特别强化了对违约客户的识别能力。

---

### **4. 模型参数更新机制**

模型的三个部分（编码器 $\phi$、解码器 $\theta$、分类器 $\psi$）通过梯度下降联合优化：

#### **(1) 编码器参数 $\phi$ 更新：**
$$
\frac{\partial \mathcal{L}_{\text{loss}}}{\partial \phi} = \frac{\partial \mathcal{L}_{\text{recon}}}{\partial \phi} + \frac{\partial \mathcal{L}_{\text{KL}}}{\partial \phi} + \alpha \frac{\partial \mathcal{L}_{\text{class}}}{\partial \phi}
$$
- 编码器同时接收来自重构、正则化和分类三个任务的梯度信号；
- 分类梯度的引入使编码器**学习到对违约更敏感的特征映射**。

#### **(2) 解码器参数 $\theta$ 更新：**
$$
\frac{\partial \mathcal{L}_{\text{loss}}}{\partial \theta} = \frac{\partial \mathcal{L}_{\text{recon}}}{\partial \theta}
$$
- 解码器仅受重构损失影响，确保其专注于重建原始数据，不干扰分类目标。

#### **(3) 分类器参数 $\psi$ 更新：**
$$
\frac{\partial \mathcal{L}_{\text{loss}}}{\partial \psi} = \alpha \frac{\partial \mathcal{L}_{\text{class}}}{\partial \psi}
$$
- 分类器仅基于加权 BCE 损失更新，专注于在潜在空间中学习最优决策边界。

---

### **5. 核心优势总结**

| 特性 | VAE-INN 的优势 |
|------|----------------|
| **一体化设计** | 将特征提取、不平衡处理、分类预测整合为单一模型，避免多阶段流水线的偏差累积 |
| **潜在空间优化** | 通过分类器梯度直接塑造潜在空间，使少数类特征被主动增强，而非被动保留 |
| **无数据扰动** | 不依赖 SMOTE、GAN 等生成式过采样，避免引入虚假样本或扭曲原始分布 |
| **类别加权机制** | 使用基于类频的动态权重 $w_i = N/(2n_l)$，科学分配样本重要性 |
| **可调平衡因子 $\alpha$** | 通过交叉验证灵活控制分类与重构的优先级，适应不同不平衡程度 |
| **专注 Type II 错误** | 模型目标明确为降低漏判违约（FN），直接服务于银行风险控制需求 |
| **可解释性与实用性** | 潜在空间为连续低维向量，分类器输出为概率，便于金融场景部署与决策支持 |

---

### **6. 与现有方法的本质区别**

| 方法 | 问题 | VAE-INN 的改进 |
|------|------|----------------|
| SMOTE/ADASYN | 人工合成样本，扭曲分布，引入噪声 | 无数据生成，保留真实分布 |
| GANs/VAEs 生成模型 | 生成不合理样本，放大噪声，边界模糊 | 仅用 VAE 做特征提取，不生成样本 |
| 多阶段流水线 | 每阶段独立优化，偏差累积 | 单阶段联合优化，端到端梯度传播 |
| 成本敏感分类（如 Xiao et al.） | 仅在输出层加权，未改变特征表示 | 在潜在空间加权，重塑特征结构 |
| 特征选择 | 倾向于多数类，丢弃少数类关键特征 | 潜在空间由分类器引导，公平保留两类信息 |

---

### **结论**

VAE-INN 方法通过**在 VAE 的潜在空间中集成一个加权神经网络分类器**，实现了**无数据扰动、端到端、类别公平的不平衡信用评分**。它不依赖外部预处理，不引入人工样本，而是通过**反向传播中分类损失对编码器的直接作用**，使模型自动学习到对违约客户最敏感、最具判别力的潜在特征表示。其损失函数中的 $\alpha$ 参数与类别权重共同确保模型在保持数据重建能力的同时，最大化对少数类的识别能力，从而**系统性降低 Type II 错误**，为银行提供更稳健、更可信赖的信用风险评估工具。