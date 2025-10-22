# Hybrid boosted attention-based LightGBM framework for enhanced credit risk assessment in digital finance

本文提出了一种名为**混合增强注意力机制的LightGBM框架（Hybrid Boosted Attention-based LightGBM, HBA-LGBM）**，用于提升数字金融场景下的信用风险评估性能。该框架通过四个核心技术创新，系统性地解决了传统信用评分模型在高维数据、类别不平衡、特征交互复杂性和模型可解释性方面的关键挑战。以下是该方法的详细说明：

---

### **1. 多阶段特征选择机制（Multi-stage Feature Selection Mechanism）**

为提升模型效率并消除冗余特征，HBA-LGBM引入了**动态多阶段特征筛选机制**，区别于传统静态特征选择方法（如方差阈值或相关系数过滤）。

- **核心思想**：基于LightGBM的直方图算法（Histogram Algorithm），在树节点分裂过程中，仅遍历离散化的特征分桶（buckets），而非原始样本，大幅降低计算复杂度（图1）。
- **动态采样策略**：在计算信息增益时，优先关注具有**高梯度值**的样本（即对损失函数贡献大的样本），构建一个高梯度样本子集 $ A $，并结合低梯度样本子集 $ B $ 进行加权估计，公式为：
  $$
  V_{j|d} = \frac{1}{n} \left( \sum_{x_i \in A} g_i + \frac{1 - a}{b} \sum_{x_i \in B} g_i \right)^2 \frac{1}{n_{l|j}^d} + \left( \sum_{x_i \in A_r} g_i + \frac{1 - a}{b} \sum_{x_i \in B_r} g_i \right)^2 \frac{1}{n_{r|j}^d}
  $$
  其中 $ A_l, A_r $ 为分裂前后高梯度样本子集，$ B_l, B_r $ 为低梯度样本子集，$ a, b $ 为调节参数。
- **优势**：减少无效计算，聚焦于对预测影响最大的样本，提高特征选择的效率与准确性。

---

### **2. 基于注意力机制的特征增强层（Attention-based Feature Enhancement Layer）**

为捕捉金融特征间动态、上下文相关的依赖关系，HBA-LGBM引入**注意力机制**，实现特征重要性的自适应重加权，突破传统静态权重分配的局限。

- **特征变换**：输入特征矩阵 $ X $ 首先通过非线性变换：
  $$
  H = \tanh(W_f X + b_f)
  $$
  其中 $ W_f $ 和 $ b_f $ 为可学习参数，生成语义增强的特征表示 $ H $。
- **注意力分数计算**：基于查询（Query）、键（Key）和值（Value）变换：
  $$
  Q = W_Q X,\quad K = W_K X,\quad V = W_V X
  $$
  计算每个特征的注意力权重：
  $$
  \alpha_j = \frac{\exp(Q_j K_j^T / \sqrt{d_k})}{\sum_k \exp(Q_k K_k^T / \sqrt{d_k})}
  $$
  其中 $ d_k $ 为键向量维度，用于缩放点积，避免梯度消失。
- **特征重加权**：最终增强特征矩阵为：
  $$
  X' = \sum_j \alpha_j V_j
  $$
  该过程在**每次迭代中动态调整**特征权重，使模型在不同样本和经济环境下自动聚焦关键风险因子（如收入波动、负债率、历史违约行为等）。
- **集成到LightGBM**：增强后的特征矩阵 $ X' $ 作为LightGBM的输入，替代原始特征，更新预测函数为：
  $$
  \hat{y} = \sum_{m=1}^M f_m(X')
  $$
  显著提升模型对非线性、时变风险模式的捕捉能力。

---

### **3. 混合提升机制（Hybrid Boosting Mechanism）**

为弥补LightGBM在捕捉高阶非线性交互关系上的不足，HBA-LGBM构建了**LightGBM与自适应神经网络的混合提升架构**。

- **神经网络组件**：引入一个**多层感知机（MLP）**，用于建模特征间的复杂交互：
  $$
  h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})
  $$
  其中 $ \sigma $ 为ReLU激活函数，$ L $ 为层数，最终输出为：
  $$
  \hat{y}_N = \mathrm{softmax}(W_h h^{(L)} + b_h)
  $$
- **自适应融合**：将LightGBM的预测 $ \hat{y}_B $ 与MLP的预测 $ \hat{y}_N $ 通过**可学习权重 $ \lambda $** 动态融合：
  $$
  \hat{y} = \lambda \hat{y}_B + (1 - \lambda) \hat{y}_N
  $$
  权重 $ \lambda $ 由输入特征 $ X $ 通过sigmoid函数自适应计算：
  $$
  \lambda = \frac{1}{1 + e^{-W_g X}}
  $$
  该机制使模型在不同样本上自动平衡树模型的可解释性与神经网络的表达能力，实现“何时依赖树、何时依赖神经网络”的智能决策。
- **优势**：兼顾LightGBM的计算效率与深度网络的非线性拟合能力，显著提升对复杂借款人行为模式的建模精度。

---

### **4. 高级不平衡学习策略（Advanced Imbalanced Learning Strategy）**

针对信用数据中违约样本（少数类）远少于正常还款样本（多数类）的严重类别不平衡问题，HBA-LGBM融合**代价敏感学习**与**增强型SMOTE**。

- **代价敏感损失函数**：在交叉熵损失中引入类别权重：
  $$
  L = \sum_{i=1}^n w_{y_i} \cdot \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]
  $$
  其中类别权重为：
  $$
  w_{y_i} = \frac{1}{\sqrt{P(y_i)}}
  $$
  该公式对少数类（违约）赋予更高权重，抑制模型偏向多数类。
- **增强型SMOTE（Synthetic Minority Over-sampling Technique）**：在少数类样本周围生成合成样本，提升数据多样性：
  $$
  X_{new} = X_i + \delta (X_i^{NN} - X_i) + \epsilon \cdot \mathcal{N}(0, \sigma^2)
  $$
  其中：
  - $ X_i $ 为少数类样本；
  - $ X_i^{NN} $ 为其最近邻样本；
  - $ \delta, \epsilon \sim U(0,1) $ 为随机扰动系数；
  - $ \mathcal{N}(0, \sigma^2) $ 为高斯噪声，用于增加合成样本的多样性。
- **正则化约束**：为防止合成样本过度偏离原始分布，引入正则项：
  $$
  \mathcal{L}_{\mathrm{final}} = L + \gamma \sum_{i=1}^m \| X_{new}^{(i)} - X_i^{NN} \|^2
  $$
  其中 $ \gamma $ 控制合成样本对原始结构的偏离程度，确保生成样本保持统计一致性。
- **效果**：显著提升模型对违约客户的识别能力，避免“虚假高准确率”陷阱，增强风险评估的公平性与实用性。

---

### **5. 模型参数优化与最终框架**

HBA-LGBM在上述四个模块基础上，对LightGBM核心参数进行系统调优，关键参数如下：

| 参数 | 描述 | 取值范围/默认值 |
|------|------|----------------|
| `learning_rate` | 学习率，控制每棵树的贡献 | 0.05–0.1 |
| `n_estimators` | 弱学习器（树）数量 | 100–1000 |
| `max_depth` | 树的最大深度 | 3–10（默认8） |
| `num_leaves` | 每棵树的叶子节点数 | ≤ $ 2^{\text{max\_depth}} - 1 $（默认25） |
| `min_data_in_leaf` | 叶子节点最小样本数 | ≥200（防止过拟合） |
| `feature_fraction` | 每次分裂使用的特征比例 | 1（全特征） |
| `bagging_fraction` | 每次迭代使用的样本比例 | 0.5 |

最终模型命名为 **HBA-LGBM**，整合了动态特征选择、注意力增强、混合提升与不平衡学习四大模块，形成一个**高效、精准、可解释且鲁棒**的信用风险评估系统。

---

该框架通过在LendingClub大规模真实数据集（104万笔贷款）上的实证验证，实现了**RMSE=11.53、MAPE=4.44%、R²=0.998**的卓越性能，显著优于传统逻辑回归、随机森林、XGBoost、CatBoost及纯深度学习模型，为数字金融平台提供了可落地的智能信用评估解决方案。