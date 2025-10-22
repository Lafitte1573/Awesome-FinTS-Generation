# Management Analysis Method of Multivariate Time Series Anomaly Detection in Financial Risk Assessment

本文提出了一种基于生成对抗网络（GAN）框架、融合对比学习与几何分布掩码数据增强的多变量时间序列（MTS）异常检测方法，旨在解决金融风险评估中模型过拟合、泛化能力弱和标注数据稀缺等核心问题。该方法由四个协同模块构成：数据增强、生成器、判别器和对比学习模块，整体架构以Transformer为骨干，实现对正常模式的精准建模与异常的高效识别。

**1. 数据增强模块（Data Augmentation）**  
为提升模型对多样化输入的适应性与泛化能力，本文引入一种基于几何分布的随机掩码（Geometric Distribution Masking）策略。该方法通过构建一个双状态（“掩码”与“未掩码”）的马尔可夫链，对多变量时间序列进行局部扰动，模拟数据缺失或不确定性。掩码序列的长度服从几何分布，其状态转移概率由两个参数控制：掩码段平均长度 $ l_m $ 和掩码与未掩码状态间的切换比率 $ r $。具体地，状态转移矩阵为：

$$
P = \begin{bmatrix}
p_{m:m} & p_{m:u} \\
p_{u:m} & p_{u:u}
\end{bmatrix}
= \begin{bmatrix}
1 - \frac{1}{l_m} & \frac{1}{l_m} \\
\frac{1}{l_m} \cdot \frac{r}{1 - r} & 1 - \frac{1}{l_m} \cdot \frac{r}{1 - r}
\end{bmatrix}
$$

其中 $ p_{m:m} $ 表示保持掩码状态的概率，$ p_{m:u} $ 为从掩码切换至未掩码的概率，其余同理。该机制在不破坏序列整体结构的前提下，人为引入多样化的局部缺失模式，显著扩展了训练样本的分布空间，增强模型对真实金融数据中噪声与缺失的鲁棒性。

**2. 生成器模块（Generator）**  
生成器采用基于Transformer的自编码器结构，其核心任务是学习并重建正常多变量时间序列的潜在分布。编码器由多头自注意力（Multi-Head Self-Attention）机制构成，其计算过程如下：

$$
Q_i, K_i, V_i = \hat{\mathcal{X}} W_i^Q, \hat{\mathcal{X}} W_i^K, \hat{\mathcal{X}} W_i^V
$$
$$
\mathcal{Z}_i = \mathrm{softmax}\left( \frac{Q_i K_i^T}{\sqrt{d_k}} \right) V_i
$$
$$
Z = \mathrm{Concat}(\mathcal{Z}_1, ..., \mathcal{Z}_h) W^o
$$

其中 $ \hat{\mathcal{X}} $ 为输入序列，$ W_i^Q, W_i^K, W_i^V $ 为查询、键、值的投影矩阵，$ h $ 为注意力头数，$ d_k $ 为键向量维度。编码器输出的高维特征通过一个两层多层感知机（MLP）解码器重建原始序列：

$$
\hat{\mathcal{X}} = \sigma(W_{g2} \cdot \sigma(W_{g1} \cdot Z + b_{g1}) + b_{g2})
$$

其中 $ W_{g1}, W_{g2} $ 为权重矩阵，$ b_{g1}, b_{g2} $ 为偏置，$ \sigma $ 为Sigmoid激活函数。生成器接收原始序列及其增强版本作为输入，目标是尽可能精确地重建正常模式，从而学习其内在统计特性。

**3. 判别器模块（Discriminator）**  
判别器作为GAN框架中的对抗组件，用于区分真实序列（原始或增强后的输入）与生成器重建的序列。其结构为三层MLP，实现从输入序列中提取深层特征并输出判别概率：

$$
\mathcal{H}_{d1} = \sigma(W_{d1} \cdot \mathcal{X}_{dis} + \mathbf{b}_{d1})
$$
$$
\mathcal{H}_{d2} = \sigma(W_{d2} \cdot \mathcal{H}_{d1} + \mathbf{b}_{d2})
$$
$$
P(Y|\mathcal{X}_{dis}) = \mathrm{softmax}(W_{d3} \cdot \mathcal{H}_{d2} + \mathbf{b}_{d3})
$$

其中 $ \mathcal{X}_{dis} $ 为输入序列（真实或重建），$ Y $ 为标签（真实/伪造），$ \sigma $ 为Sigmoid函数。判别器的损失函数采用标准GAN的二元交叉熵：

$$
\mathcal{L}_{dis} = -\frac{1}{N} \left[ \log D(\mathcal{X}_{dis}) + \log(1 - D(G(\mathcal{X}_{dis}))) \right]
$$

通过对抗训练，判别器迫使生成器不断改进重建质量，使重建序列在统计分布上无限逼近真实正常序列，从而有效抑制过拟合。

**4. 对比学习模块（Contrastive Learning）**  
为进一步提升模型的泛化能力与判别器对正常模式分布的鲁棒性，本文在判别器的特征空间中引入对比学习约束。在每个训练批次中，仅将同一原始序列及其增强版本对应的重建表示视为正样本对（positive pair），其余所有其他重建表示均视为负样本对（negative pair）。对比损失函数定义为：

$$
\mathcal{L}_{contrastive} = -\frac{1}{N} \sum_{i=1}^{N} \log \left( \frac{e^{s \cdot \mathrm{Sim}(z_i, z_{i+1})}}{\sum_{j=1}^{2N} e^{s \cdot \mathrm{Sim}(z_i, z_j)}} \right)
$$

其中 $ z_i $ 与 $ z_{i+1} $ 为正样本对的潜在表示，$ \mathrm{Sim}(\cdot, \cdot) $ 为余弦相似度函数，$ s $ 为温度系数，用于调节相似度分布的锐度。该损失函数通过“拉近”正样本对、“推开”负样本对，迫使模型学习到更具判别性的、与数据增强方式无关的正常模式表征，从而增强模型在未见数据上的泛化能力。

**5. 整体训练机制**  
整个模型通过端到端联合优化训练：生成器与判别器在GAN框架下进行对抗训练，同时对比学习损失作为正则化项被加总至总损失函数中，共同引导模型学习鲁棒的正常模式表示。训练过程中，输入数据首先通过几何分布掩码增强，随后送入生成器重建，判别器同时接收真实数据与重建数据进行对抗判断，而对比损失则作用于判别器输出的潜在特征空间，形成“增强—重建—对抗—对比”四重协同机制。该机制有效缓解了传统重建方法因过度拟合正常模式而导致的低灵敏度问题，同时克服了监督方法对稀缺异常标签的依赖，为金融场景下的无监督异常检测提供了高效、稳健的新范式。