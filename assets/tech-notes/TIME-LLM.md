# TIME-LLM: TIME SERIES FORECASTING BY REPROGRAMMING LARGE LANGUAGE MODELS

本文提出的方法名为TIME-LLM，是一种用于时间序列预测的重编程（reprogramming）框架，其核心思想是在不修改预训练大语言模型（LLM）参数的前提下，通过将时间序列数据重新编码为自然语言可理解的文本原型表示，并结合提示（prompt）引导模型推理，从而激活LLM在时间序列预测任务中的强大泛化与推理能力。方法整体架构包含三个核心组件：输入变换、冻结的预训练LLM、输出投影，整个过程仅更新轻量级的输入与输出模块参数，而LLM主干完全冻结。

### 1. 输入变换（Input Transformation）

输入变换阶段将原始多变量时间序列 $\mathbf{X} \in \mathbb{R}^{N \times T}$（包含 $N$ 个变量、$T$ 个时间步）分解为 $N$ 个独立的单变量时间序列 $\mathbf{X}^{(i)} \in \mathbb{R}^{1 \times T}$，并依次处理每个通道。

- **归一化**：对每个单变量序列应用可逆实例归一化（RevIN），使其均值为0、标准差为1，以缓解时间序列分布偏移问题。
- **分块（Patching）**：将归一化后的序列划分为多个重叠或非重叠的时序块（patches），每个块长度为 $L_p$，滑动步长为 $S$，得到 $P = \lfloor (T - L_p) / S \rfloor + 2$ 个块，形成 $\mathbf{X}_P^{(i)} \in \mathbb{R}^{P \times L_p}$。分块旨在保留局部语义信息，并将连续序列离散化为紧凑的“token”序列，降低计算负担。
- **嵌入（Embedding）**：通过一个简单的线性层将每个块 $\mathbf{X}_P^{(i)}$ 映射为高维嵌入表示 $\hat{\mathbf{X}}_P^{(i)} \in \mathbb{R}^{P \times d_m}$，其中 $d_m$ 为嵌入维度。

### 2. 块重编程（Patch Reprogramming）

这是TIME-LLM的核心创新，旨在弥合时间序列（连续数值）与自然语言（离散符号）之间的模态鸿沟，使LLM能够理解时序模式。

- **文本原型（Text Prototypes）**：利用LLM主干中已有的词嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{V \times D}$（$V$ 为词汇表大小，$D$ 为隐藏维度），从中线性探针（linear probing）出一个极小的、可学习的文本原型集合 $\mathbf{E}' \in \mathbb{R}^{V' \times D}$，其中 $V' \ll V$（如100或1000个原型）。这些原型是人工设计的语义短语（如“short up”、“steady down”），代表时间序列中常见的局部模式。
- **多头交叉注意力（Multi-head Cross-Attention）**：将时序块嵌入 $\hat{\mathbf{X}}_P^{(i)}$ 作为查询（Query），文本原型嵌入 $\mathbf{E}'$ 作为键（Key）和值（Value），通过多头交叉注意力机制进行重编程：
  $$
  \mathbf{Q}_k^{(i)} = \hat{\mathbf{X}}_P^{(i)} \mathbf{W}_k^Q, \quad \mathbf{K}_k^{(i)} = \mathbf{E}' \mathbf{W}_k^K, \quad \mathbf{V}_k^{(i)} = \mathbf{E}' \mathbf{W}_k^V
  $$
  其中 $\mathbf{W}_k^Q \in \mathbb{R}^{d_m \times d}$，$\mathbf{W}_k^K, \mathbf{W}_k^V \in \mathbb{R}^{D \times d}$，$d = \lfloor d_m / K \rfloor$，$K$ 为头数。
  注意力输出为：
  $$
  \mathbf{Z}_k^{(i)} = \mathrm{Softmax}\left( \frac{\mathbf{Q}_k^{(i)} \mathbf{K}_k^{(i)\top}}{\sqrt{d}} \right) \mathbf{V}_k^{(i)}
  $$
  所有头的输出拼接后得到 $\mathbf{Z}^{(i)} \in \mathbb{R}^{P \times d_m}$，再通过线性投影对齐至LLM隐藏维度 $D$，得到重编程后的表示 $\mathbf{O}^{(i)} \in \mathbb{R}^{P \times D}$。此过程将时序块“翻译”为语言模型可处理的语义表示，而无需修改LLM参数。

### 3. 提示作为前缀（Prompt-as-Prefix, PaP）

为增强LLM对时间序列任务的推理能力，本文提出“提示作为前缀”（Prompt-as-Prefix, PaP），在输入序列前添加自然语言提示，提供上下文、任务指令和统计信息，引导模型理解并执行时序变换。

- **提示结构**：每个提示包含三部分：
  1. **数据集上下文**：描述数据领域特性（如“Electricity Transformer Temperature (ETT) indicates the electric power long-term deployment.”）；
  2. **任务指令**：明确预测目标（如“Forecast the next 96 time steps of oil temperature.”）；
  3. **输入统计信息**：提供计算的时序特征（如最小值、最大值、中位数、前5个自相关滞后值），帮助模型识别趋势与周期性。
- **格式示例**：
  ```
  <BOS> The Electricity Transformer Temperature (ETT) indicates the electric power long-term deployment. Forecast the next 96 time steps of oil temperature. The input has a minimum of <min value>, a maximum of <max value>, and a median of <median value>. The top five lags are <lag values>. <EOS>
  ```
  这些提示以自然语言token形式与重编程后的时序块嵌入 $\mathbf{O}^{(i)}$ 拼接，形成最终输入序列，送入冻结的LLM。

### 4. 输出投影（Output Projection）

- 将LLM的输出表示（包含提示前缀和时序块响应）中除去提示部分，仅保留对应时序块的输出 $\tilde{\mathbf{O}}^{(i)} \in \mathbb{R}^{P \times D}$。
- 将其展平为一维向量 $\tilde{\mathbf{O}}^{(i)} \in \mathbb{R}^{P \times D}$，再通过一个线性投影层映射为预测结果 $\hat{\mathbf{Y}}^{(i)} \in \mathbb{R}^{H}$，其中 $H$ 为预测步长。
- 最终预测目标是最小化均方误差（MSE）：$\frac{1}{H} \sum_{h=1}^{H} \|\hat{\mathbf{Y}}_h - \mathbf{Y}_h\|_F^2$。

### 5. 模型训练与效率

- **参数更新**：仅训练输入变换模块（线性嵌入层、文本原型矩阵 $\mathbf{W}$、多头交叉注意力权重、输出投影层），LLM主干参数完全冻结。
- **高效性**：整个可训练参数量极小（如Llama-7B中仅约0.2%），训练仅需少量epoch和时间序列样本，无需大规模微调。
- **可扩展性**：可结合量化等技术进一步压缩模型，适用于资源受限场景。

TIME-LLM通过“文本原型重编程 + 提示引导”实现了对冻结LLM的跨模态适配，将时间序列预测转化为一种“语言任务”，使LLM无需微调即可在零样本、少样本和全样本场景下超越专用模型，展现出强大的泛化能力与数据效率。