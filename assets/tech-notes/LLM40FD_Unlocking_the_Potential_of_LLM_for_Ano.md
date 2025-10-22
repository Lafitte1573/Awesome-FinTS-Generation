# LLM40FD: Unlocking the Potential of LLM for Anonymous Zero-Shot Fraud Detection

本文提出的LLM40FD是一种面向匿名零样本信用卡欺诈检测的新型框架，其核心目标是利用大语言模型（LLM）在无需微调、无需标签、无需领域适配的前提下，实现对匿名交易数据的高效欺诈检测。该方法通过三个关键组件协同工作：**行走嵌入（Walking Embedding）**、**基于分布的一类函数（Distribution-based One-Class, DOC）** 和 **双重增强策略（Dual-Augmentation）**，构建了一个不依赖传统监督信号、可泛化至未知数据分布的欺诈检测范式。

### 1. 行走嵌入（Walking Embedding）——统一匿名特征表示

由于信用卡交易数据通常因隐私保护而被匿名化（如PCA降维后的V1–V28特征），原始语义信息丢失，传统模型无法理解特征含义。LLM40FD提出**行走嵌入**，将高维匿名特征序列转化为LLM可处理的统一文本式嵌入序列。

- **核心思想**：将原始特征向量 $x_i \in \mathbb{R}^d$ 按固定步长（stride）滑动分组，每组作为“token”输入LLM。例如，若步长为2，则将特征对 $(x_1, x_2), (x_3, x_4), \dots$ 视为一个token；若特征数为奇数，最后一维补零构成完整token。
- **多步长遍历**：为保留特征间长程依赖关系，使用 $n$ 个不同步长（如1, 2, 3, …, n）分别生成嵌入序列 $\mathcal{E}_1, \mathcal{E}_2, \dots, \mathcal{E}_n$，每个序列长度为 $L_i$。
- **嵌入拼接**：将所有步长生成的嵌入在上下文维度拼接，形成最终统一嵌入表示：
  $$
  \mathcal{E} = \text{Concat}(\mathcal{E}_1, \mathcal{E}_2, \dots, \mathcal{E}_n, \text{dim}=1) \in \mathbb{R}^{B \times L \times V}
  $$
  其中 $B$ 为批量大小，$V$ 为LLM隐藏维度，$L = \sum_{i=1}^n L_i$ 为拼接后总上下文长度。
- **LLM输入**：将 $\mathcal{E}$ 输入预训练LLM，通过池化操作（如取最后一时刻输出）生成交易的统一表示：
  $$
  U' = \text{LLM}(\mathcal{E}), \quad U = \text{Pooling}(U') \in \mathbb{R}^{B \times V}
  $$
- **降维投影**：使用MLP将 $U$ 投影至低维空间 $\mathbb{R}^{B \times d_{\text{model}}}$，以适配后续对比学习。

该方法不依赖特征语义，仅通过结构化序列化将匿名数值特征转化为LLM可处理的“类文本”序列，突破了传统LLM无法处理无文本结构化数据的限制。

### 2. 基于分布的一类函数（DOC）——知识蒸馏目标

LLM40FD不微调LLM参数，而是通过**知识蒸馏**将LLM的通用知识引导至欺诈检测任务。为此，提出**DOC函数**作为蒸馏目标，其本质是学习一个“正常交易”的分布中心。

- **知识中心定义**：设 $C \in \mathbb{R}^{d_{\text{model}}}$ 为正常交易行为的分布中心，代表LLM所“理解”的正常模式。
- **DOC损失函数**：通过最小化统一表示 $U$ 与知识中心 $C$ 的KL散度，迫使正常样本的嵌入向中心聚集：
  $$
  \mathcal{L}_{\text{DOC}} = \frac{1}{B} \sum \text{Softmax}(U) \log \text{Softmax}(C)
  $$
  此处Softmax将向量归一化为概率分布，使模型学习“哪些嵌入更接近正常模式”。
- **检测机制**：训练完成后，任何远离 $C$ 的嵌入被视为异常（欺诈）。在推理阶段，使用DOC得分评估异常：
  $$
  \text{AnomScore}_{\text{DOC}} = \frac{1}{V} \sum_{j=1}^V \text{Softmax}(U)_j \log \text{Softmax}(C)_j
  $$
  得分越高，越可能为欺诈。

DOC函数使LLM无需标签即可“记住”正常行为的分布，实现无监督的一类分类，同时避免了对LLM参数的任何修改，保留其原始推理能力。

### 3. 双重增强策略（Dual-Augmentation）——平衡样本与强化边界

由于欺诈样本极少（极端不平衡），且正样本过多易导致过拟合，LLM40FD引入**双重增强**，在隐式对比学习框架下生成正负样本，强化模型对正常与异常边界的区分能力。

- **正样本生成**（增强正常行为）：
  $$
  X_{\text{pos}} = \mu + \gamma_{\text{pos}} \cdot \sigma \cdot \mathcal{N}, \quad \mu = \text{mean}(X), \; \sigma = \text{std}(X)
  $$
  在批次均值 $\mu$ 附近添加高斯噪声，生成“更典型”的正常样本，使 $U^p$ 更靠近知识中心 $C$，提升中心凝聚力。
  
- **负样本生成**（模拟欺诈行为）：
  $$
  X_{\text{neg}} = X + \gamma_{\text{neg}} \cdot \sigma \cdot \mathcal{N}
  $$
  在原始样本 $X$ 基础上添加噪声，模拟偏离正常模式的异常行为（因无欺诈先验，假设偏离即异常），使 $U^n$ 远离 $C$。

- **双重DOC损失**：
  $$
  \mathcal{L}_{\text{DOC}}^p = \frac{1}{N} \sum \text{Softmax}(U^p) \log \text{Softmax}(C) \quad \text{(拉近正样本)} \\
  \mathcal{L}_{\text{DOC}}^n = -\frac{1}{N} \sum \text{Softmax}(U^n) \log \text{Softmax}(C) \quad \text{(推远负样本)}
  $$
  正样本损失鼓励靠近中心，负样本损失为负号，强制远离中心，形成隐式对比学习。

- **最终目标函数**：
  $$
  \mathcal{L} = \mathcal{L}_{\text{DOC}} + \mathcal{L}_{\text{DOC}}^p + \mathcal{L}_{\text{DOC}}^n
  $$

双重增强策略在无标签条件下，动态构建“正常-异常”对比对，使模型在训练中自适应学习清晰的决策边界，显著提升对稀有欺诈模式的敏感性。

### 总结：LLM40FD的完整工作流程

1. **输入**：匿名化交易数据 $X \in \mathbb{R}^{N \times d}$（无标签，无语义）。
2. **行走嵌入**：用多步长滑动窗口将特征序列转化为LLM可处理的嵌入序列 $\mathcal{E}$。
3. **LLM编码**：通过LLM生成交易统一表示 $U$，并投影至低维空间。
4. **知识中心学习**：通过DOC损失，使正常样本嵌入向中心 $C$ 聚集。
5. **双重增强**：在每批次中动态生成正负样本，通过 $\mathcal{L}_{\text{DOC}}^p$ 和 $\mathcal{L}_{\text{DOC}}^n$ 强化边界。
6. **推理**：计算新样本的DOC得分，高于阈值则判定为欺诈。

**关键创新**：  
- 首次将LLM用于**匿名**、**零样本**、**无微调**的金融欺诈检测；  
- 行走嵌入解决**无语义特征**的LLM适配问题；  
- DOC函数实现**无标签知识蒸馏**；  
- 双重增强实现**隐式对比学习**，无需真实标签即可构建判别边界。

整个框架完全避免对LLM的参数更新，仅训练轻量级行走嵌入和MLP投影头，实现高效、可迁移、零样本的欺诈检测。