# Regression estimation for continuous-time functional data processes with missing at random response

本文提出了一种针对具有“随机缺失”（Missing At Random, MAR）响应的连续时间函数型数据过程的非参数核回归估计方法。其核心思想是，在响应变量 $Y_t$ 可能因观测机制缺失（由指示变量 $\zeta_t \in \{0,1\}$ 标记）的情况下，基于部分观测的三元组数据 $(X_t, Y_t, \zeta_t)_{t \in [0,T]}$，估计一个广义回归函数 $m_\psi(x, y) = \mathbb{E}[\psi(Y, y) \mid X = x]$，其中 $X_t$ 是取值于无限维空间 $\mathcal{E}$ 的函数型预测变量，$Y_t$ 是实值响应变量，$\psi(\cdot, y)$ 是一个可测函数，可对应条件均值、条件分布函数或条件分位数等不同目标。

### 方法框架与核心思想

1. **缺失机制建模**：  
   响应变量 $Y_t$ 的缺失由 Bernoulli 过程 $\zeta_t$ 控制，满足 MAR 条件：  
   $$
   \mathbb{P}(\zeta_t = 1 \mid X_t = x, Y_t = y) = \mathbb{P}(\zeta_t = 1 \mid X_t = x) := p(x)
   $$  
   即缺失概率仅依赖于预测变量 $X_t$，而非未观测的 $Y_t$。这一假设允许在估计中通过条件期望进行修正，避免因缺失导致的偏差。

2. **广义回归函数的表达**：  
   通过在原回归模型 $\psi(Y, y) = m_\psi(X, y) + \varepsilon$ 两边乘以 $\zeta$，并取条件期望，得到：  
   $$
   m_\psi(x, y) = \frac{\mathbb{E}[\zeta \psi(Y, y) \mid X = x]}{\mathbb{E}[\zeta \mid X = x]}
   $$  
   该表达式表明，即使响应缺失，只要知道缺失机制仅依赖于 $X$，即可通过加权方式重建回归函数。

3. **核型估计器构造**：  
   基于连续时间观测数据 $(X_t, Y_t, \zeta_t)_{t \in [0,T]}$，定义核估计器：  
   $$
   \widehat{m}_{\psi, T}(x, y) = 
   \begin{cases}
   \displaystyle \frac{\int_0^T \zeta_t \psi(Y_t, y) \Delta_t(x) \, dt}{\int_0^T \zeta_t \Delta_t(x) \, dt}, & \text{if } \int_0^T \zeta_t \Delta_t(x) \, dt \neq 0 \\
   \displaystyle \frac{1}{T} \int_0^T \zeta_t \psi(Y_t, y) \, dt, & \text{otherwise}
   \end{cases}
   $$  
   其中 $\Delta_t(x) = K\left( \frac{d(x, X_t)}{h_T} \right)$ 是核函数，$K(\cdot)$ 是定义在 $[0,1]$ 上的非负、单调递减、$\mathcal{C}^1$ 核函数（因 $d(x, X_t)$ 为非负距离），$h_T \to 0$ 为带宽参数。  
   该估计器仅使用 $\zeta_t = 1$ 的观测数据进行核加权平均，从而自然地处理了缺失数据，无需插补。

4. **连续时间过程的理论框架**：  
   为处理连续时间函数型数据的依赖结构，作者引入了以下关键机制：  
   - **过滤器（filtration）**：定义 $\mathcal{F}_{t-\delta} = \sigma\{(X_s, Y_s) : 0 \le s < t - \delta\}$，表示 $t$ 时刻前 $\delta$ 时间内的历史信息。  
   - **小球概率函数**：定义 $F_x(u) = \mathbb{P}(d(x, X_t) \le u)$，并假设其渐近行为为 $F_x(u) = \phi(u) f(x) + o(\phi(u))$，其中 $\phi(u)$ 是刻画小球概率衰减速率的函数（如幂函数或对数函数），$f(x)$ 是“密度”函数。  
   - **条件小球概率**：假设 $\mathbb{P}(d(x, X_t) \le u \mid \mathcal{F}_{t-\delta}) = \phi(u) f_{t,t-\delta}(x) + g_{t,t-\delta,x}(u)$，其中 $g_{t,t-\delta,x}(u) = o_{a.s.}(\phi(u))$，确保条件分布与无条件分布渐近一致，体现“遍历性”（ergodicity）。

5. **遍历性假设（而非强混合）**：  
   与以往依赖 $\alpha$-混合条件的文献不同，本文仅要求过程为**平稳遍历**（stationary and ergodic），不假设任何混合系数或协方差结构。这使得方法适用于更广泛的过程，包括：  
   - 长记忆过程（如分数布朗运动驱动的 Ornstein-Uhlenbeck 过程）  
   - Bernoulli 移位过程（弱相关但非混合）  
   - 非混合的 AR(1) 过程（如由 Bernoulli 噪声驱动）  
   这些过程在金融、气象、电力负荷等领域常见，而传统混合方法无法处理。

6. **渐近性质推导工具**：  
   为证明估计器的收敛性，作者采用**鞅差序列**（martingale difference）和**投影序列**（projection onto $\sigma$-fields）作为核心数学工具，而非依赖混合条件下的相关不等式。这使得推导能在无混合假设下完成，并能处理连续时间积分形式的依赖结构。

7. **收敛速率与偏差-方差分解**：  
   在假设 (A1)–(A3) 下，作者证明了点态收敛速率：  
   $$
   \widehat{m}_{\psi, T}(x, y) - m_\psi(x, y) = \mathcal{O}_{a.s.}(h_T^\beta) + \mathcal{O}_{a.s.}\left( \sqrt{\frac{\log T}{T \phi(h_T)}} \right)
   $$  
   其中：  
   - $h_T^\beta$ 项来自回归函数的光滑性（Hölder 条件）引起的**偏差**；  
   - $\sqrt{\frac{\log T}{T \phi(h_T)}}$ 项来自数据稀疏性（小球概率 $\phi(h_T)$ 小）和样本量 $T$ 有限引起的**方差**。  
   为达到最优收敛，需平衡 $h_T$：当 $\phi(h_T) \sim (\log T / T)^{2\beta/(2\beta+1)}$ 时，总误差为 $\mathcal{O}_{a.s.}((\log T / T)^{\beta/(2\beta+1)})$。

8. **渐近分布与置信区间构建**：  
   作者进一步推导了估计器的渐近正态性：  
   $$
   \sqrt{T \phi(h_T)} \left( \widehat{m}_{\psi, T}(x, y) - m_\psi(x, y) - B_T(x, y) \right) \xrightarrow{d} \mathcal{N}(0, \sigma^2(x, y))
   $$  
   其中 $B_T(x, y)$ 为条件偏差项，$\sigma^2(x, y)$ 为渐近方差。这使得可构建点处的**渐近置信区间**，用于不确定性量化。

9. **离散采样下的适用性**：  
   尽管理论建立在连续时间框架下，但作者明确指出：若数据以固定采样间隔 $\delta = T/n$ 离散采集，即 $(X_{t_k}, Y_{t_k}, \zeta_{t_k})_{k=1}^n$，则该估计器仍适用，且结果在 $n \to \infty, \delta \to 0$ 下保持一致。这为实际应用（如高频金融数据、智能电表数据）提供了理论支撑。

综上，本文方法的核心贡献在于：**在无混合假设、响应随机缺失、连续时间函数型数据的复杂背景下，构造了一个基于核加权的非参数估计器，并严格推导了其几乎必然一致收敛性、收敛速率、渐近分布与置信区间构建方法，拓展了函数型数据分析在缺失数据场景下的理论边界。**