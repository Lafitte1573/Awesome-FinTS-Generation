# Stripping the Swiss discount curve using kernel ridge regression

本文提出的方法是基于核岭回归（Kernel Ridge Regression, KR）的无风险贴现曲线估计方法，其核心思想是在再生核希尔伯特空间（Reproducing Kernel Hilbert Space, RKHS）中通过正则化最小化定价误差与曲线平滑性之间的权衡，从而获得稳健、灵活且数据驱动的贴现曲线估计。

### 1. 问题设定
在任一交易日，观测到 $M$ 个固定收益证券的市场价格向量 $P = (P_1, \dots, P_M)^\top$，这些证券的现金流发生在 $N$ 个不同的到期时间点 $0 < x_1 < \cdots < x_N$。现金流结构由矩阵 $C$ 表示，其中 $C_{ij}$ 表示第 $i$ 个证券在时间 $x_j$ 的现金流。贴现曲线 $g: [0, \infty) \to \mathbb{R}$ 是一个未观测的函数，表示到期时间为 $x$ 的零息债券的现值，即 $g(x) = e^{-y(x) \cdot x}$，其中 $y(x)$ 为对应的零息收益率。

根据“一价定律”，证券的理论价格应为：
$$
P^g = C g(\mathbf{x}), \quad \text{其中} \quad \mathbf{x} = (x_1, \dots, x_N)^\top, \quad g(\mathbf{x}) = (g(x_1), \dots, g(x_N))^\top.
$$
由于市场流动性不足、报价噪声或数据错误，实际观测价格与理论价格存在偏差：
$$
P = P^g + \epsilon, \quad \epsilon \in \mathbb{R}^M.
$$

### 2. 核岭回归（KR）优化框架
KR 方法通过在特定的 RKHS 空间 $\mathcal{G}_{\alpha, \delta}$ 中求解以下正则化优化问题来估计贴现曲线 $g$：
$$
\min_{g \in \mathcal{G}_{\alpha, \delta}} \left\{ \sum_{i=1}^M \omega_i (P_i - P_i^g)^2 + \lambda \|g\|_{\alpha, \delta}^2 \right\},
$$
其中：
- 第一项为**定价误差**，$\omega_i > 0$ 为权重，用于调整不同证券的误差贡献；
- 第二项为**平滑性惩罚项**，$\lambda > 0$ 为正则化参数，控制拟合精度与平滑性之间的权衡；
- 空间 $\mathcal{G}_{\alpha, \delta}$ 由所有满足 $g(0) = 1$ 且具有有限范数的二阶可微函数组成，其范数定义为：
  $$
  \|g\|_{\alpha, \delta}^2 := \int_0^\infty \left( \delta [g'(x)]^2 + (1 - \delta) [g''(x)]^2 \right) e^{\alpha x} dx,
  $$
  其中：
  - $\delta \in [0,1]$ 为形状参数，决定是惩罚一阶导数（斜率变化，张力）还是二阶导数（曲率变化）；
  - $\alpha \geq 0$ 为到期时间权重，用于控制函数在长期端的衰减速度（防止过度外推）。

该范数结构结合了张力（tension）和曲率（curvature）两种平滑性度量，且通过指数权重 $e^{\alpha x}$ 控制长期端的平滑程度。

### 3. 再生核与闭式解
该问题的解由再生核 $k: [0,\infty) \times [0,\infty) \to \mathbb{R}$ 完全刻画。对于 $\alpha > 0$ 且 $\delta = 0$（默认设置），核函数为：
$$
k(x, y) = -\frac{\min\{x, y\}}{\alpha^2} e^{-\alpha \min\{x, y\}} + \frac{2}{\alpha^3} \left(1 - e^{-\alpha \min\{x, y\}} \right) - \frac{\min\{x, y\}}{\alpha^2} e^{-\alpha \max\{x, y\}}.
$$
根据再生核理论和表示定理（Representer Theorem），最优解 $\hat{g}$ 具有如下闭式表达：
$$
\hat{g}(x) = 1 + \sum_{j=1}^N k(x, x_j) \beta_j,
$$
其中系数向量 $\beta = (\beta_1, \dots, \beta_N)^\top$ 由下式计算：
$$
\beta = C^\top (C K C^\top + \Lambda)^{-1} (P - C \mathbf{1}),
$$
其中：
- $K$ 是 $N \times N$ 的核矩阵，$K_{ij} = k(x_i, x_j)$；
- $\Lambda = \mathrm{diag}(\lambda / \omega_1, \dots, \lambda / \omega_M)$，当 $\omega_i = \infty$ 时，$\lambda / \omega_i = 0$，表示对该证券强制精确定价；
- $\mathbf{1} = (1, \dots, 1)^\top$ 是常数向量，用于满足 $g(0) = 1$ 的约束。

该解仅需对 $M \times M$ 矩阵求逆，计算高效。

### 4. 超参数选择与权重设定
KR 方法完全数据驱动，超参数 $\lambda, \alpha, \delta$ 通过**交叉验证**（out-of-sample cross-validation）选择，以最小化加权定价误差。权重 $\omega_i$ 采用金融文献中标准的**修正久期加权**：
$$
\omega_i = \frac{1}{M} \frac{1}{(D_i P_i)^2},
$$
其中 $D_i$ 为第 $i$ 个证券的修正久期。该权重设计使得价格误差在第一阶近似下等价于收益率误差：
$$
\omega_i (P_i - P_i^g)^2 \approx \frac{1}{M} (Y_i - Y_i^g)^2,
$$
从而直接优化收益率拟合精度。

此外，KR 支持**外部观点的集成**：若希望曲线在某一点 $x_j$ 精确匹配一个外部给定的收益率（如央行短期基准利率 SARON），可将该点设为一个零息债券，令其 $\omega_i = \infty$，从而强制 $\hat{g}(x_j)$ 精确等于该收益率对应的现值。

### 5. 方法优势
- **非参数性**：不预设函数形式（如 Nelson-Siegel、Smith-Wilson），可灵活捕捉局部和全局收益率曲线形态；
- **数据驱动**：超参数通过交叉验证自动选择，无需人为设定；
- **可解释性与透明性**：解为核函数的线性组合，结构清晰，可复现；
- **鲁棒性**：正则化机制有效抑制噪声和异常值影响；
- **计算高效**：闭式解仅需矩阵求逆，适合高频更新；
- **理论完备**：建立在 RKHS 理论之上，保证解的存在唯一性与最优性。

### 6. 与 Smith-Wilson 方法的统一性
本文进一步证明 Smith-Wilson（SW）方法是 KR 的一个特例：当 $\alpha = 0$，$\delta \in (0,1)$，且所有债券权重 $\omega_i = \infty$（强制精确拟合至最后流动性点 LLP），SW 的解可由 KR 框架完全复现。SW 的“收敛速度”$\rho = \sqrt{\delta/(1-\delta)}$ 和“终极远期利率”$y_\infty$ 分别对应 KR 中的 $\delta$ 和通过指数倾斜 $e^{-y_\infty x}$ 引入的参数。因此，KR 是 SW 的广义化，且其解空间更大，能容纳更多样化的曲线形态，理论上优于 SW。

综上，本文提出的 KR 方法是一种基于再生核希尔伯特空间、通过正则化最小化定价误差与平滑性权衡、完全数据驱动的非参数贴现曲线估计框架，具有理论严谨性、计算可行性、实践灵活性与卓越的拟合性能。