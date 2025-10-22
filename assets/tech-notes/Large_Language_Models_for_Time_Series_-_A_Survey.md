# Large Language Models for Time Series: A Survey

本文提出了一种系统性的方法论框架，用于将大型语言模型（LLMs）应用于时间序列分析，并通过五类核心技术方法来弥合文本语言模型与数值时间序列之间的模态鸿沟。每类方法均针对标准LLM处理文本的典型流水线中的一个特定阶段进行改造与适配，具体如下：

### 1. 直接提示（Prompting）
该方法将时间序列数据直接作为原始文本输入，无需任何结构化转换，利用预训练LLM的零样本推理能力完成任务。分为两类：

- **数字无关提示（Number-Agnostic Prompting）**：  
  将数值时间序列以字符串形式直接拼接为自然语言提示，例如将温度序列 `[20.1, 21.3, 22.0]` 转换为文本：“从t1到tobs，区域Um每天的平均温度分别为20.1、21.3、22.0度。tobs时的温度是多少？”  
  代表性工作包括 PromptCast（用于温度预测）、AuxMobLCast（POI人流预测）、LLM-Mob（用户下一位置预测）、TabLLM（表格数据分类）和 Xie et al.（股票价格特征提示ChatGPT）。  
  优点：无需训练、零样本可用；缺点：丢失数值语义、上下文效率低、不适用于高精度或多变量序列。

- **数字相关提示（Number-Specific Prompting）**：  
  为解决BPE分词对数字的破坏性分割问题，LLMTime 提出在数字间插入空格（如“0.123” → “0 . 1 2 3”），并用逗号分隔时间步（如“0 . 1 2 3 , 1 . 2 3 , 1 2 . 3”），同时统一保留固定小数位数（如两位）以优化上下文长度。BloomberGPT 采用类似策略，将每个数字作为独立token处理，提升数值运算的可解释性。  
  优点：保留数字结构、支持算术推理；缺点：仍受限于LLM的上下文长度，且未利用训练数据优化。

### 2. 时间序列量化（Quantization）
将连续数值时间序列离散化为有限集合的符号（token），作为LLM的输入。分为三类：

- **基于VQ-VAE的离散索引**：  
  使用向量量化变分自编码器（VQ-VAE）学习一个码本 $\mathcal{C} = \{ \mathbf{c}_i \}_{i=1}^K$，对时间序列编码表示 $g_\phi(\mathbf{x}_s)$ 中每个时间步找到最近码字，以索引 $k_i = \arg\min_j \| g_\phi(\mathbf{x}_s)_i - \mathbf{c}_j \|_2$ 作为离散token。  
  代表性工作：Auto-TTE（ECG信号生成）、DeWave（EEG-to-text翻译）、TOTEM（多任务预测与翻译）、UniAudio（多模态音频生成）、VioLA（语音-文本统一编码）、AudioGen（文本引导音频生成）。

- **基于K-Means的离散索引**：  
  使用K-Means聚类对神经音频编解码器输出的激活向量进行聚类，以聚类中心索引作为离散token。  
  代表性工作：SpeechGPT（多模态感知与生成）、AudioLM（音频生成与结构建模）、AudioPaLM（融合PaLM-2与AudioLM，共享语音与文本离散词汇表）。

- **基于文本类别的量化**：  
  将数值变化映射为预定义文本标签，常用于金融领域。如TDML将价格周波动分为12个类别，用“Ui”表示上涨（i=1,2,3,4,5,5+），“Di”表示下跌，形成如“U3,D2,U1”等文本序列。  
  优点：直接适配LLM的文本输入格式；缺点：信息损失严重，依赖人工定义类别。

- **基于频域的量化**：  
  FreqTST 将时间序列通过傅里叶变换转换为频率分量，以频率单位及其权重作为token，构建频域“词典”用于预测。  
  Chronos 将实值序列映射到离散的数值区间（bin），并使用交叉熵损失在预训练语言模型上微调。

### 3. 对齐（Aligning）
训练独立的时间序列编码器，将时间序列嵌入空间与语言模型的语义空间对齐。分为两类：

- **通过对比损失实现相似性匹配**：  
  使用对比学习最小化时间序列嵌入 $g_\phi(\mathbf{x}_s)$ 与文本嵌入 $f_\theta(\mathbf{x}_t)$ 之间的距离，目标函数为：  
  $$
  \mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \log \frac{ \exp\left( \text{sim}(g_\phi(\mathbf{x}_{si}), f_\theta(\mathbf{x}_{ti}))^{1/\gamma} \right) }{ \sum_{k=1}^B \exp\left( \text{sim}(g_\phi(\mathbf{x}_{si}), f_\theta(\mathbf{x}_{tk}))^{1/\gamma} \right) }
  $$  
  其中 $\text{sim}(\cdot,\cdot)$ 为内积相似度，$\gamma$ 为温度系数。  
  代表性工作：ETP（ECG与临床报告对齐）、TEST（实例级/特征级/文本原型对齐）、TENT（IoT传感器与文本对齐）、JoLT（使用Q-Former对齐）、ECG-LLM（使用最优传输损失对齐）、MTAM（使用典型相关分析和Wasserstein距离对齐）。

- **以LLM为骨干的对齐**：  
  将时间序列编码后直接输入冻结的预训练LLM（如GPT-2、LLaMA），仅微调编码器或嵌入层。  
  代表性工作：EEG-to-Text（EEG嵌入输入BART）、GPT4TS（使用patching嵌入输入GPT-2）、TEMPO（季节趋势分解后输入）、LLM4TS（两阶段微调）、UniTime（引入领域描述）、GATGPT（图注意力机制）、ST-LLM（时空嵌入模块）、Time-LLM（将时间序列重编程为文本原型输入LLaMA-7B）、Lag-Llama（基于LLaMA架构的单变量概率预测模型）。  
  在音频领域：WavPrompt、SpeechLLaMA、MU-LLaMA、LTU、SALMONN 等均采用类似架构，将语音/音乐编码后输入LLM。

### 4. 视觉作为桥梁（Vision as Bridge）
利用视觉模态作为中间桥梁，将时间序列与视觉表示关联，再通过视觉语言模型（VLM）连接文本。

- **配对数据对齐**：  
  ImageBind 学习将图像、文本、音频、深度、热力和IMU时间序列统一映射到共享嵌入空间。IMU2CLIP 将IMU序列投影至CLIP的视觉-文本联合空间。AnyMAL 在此基础上训练轻量适配器，将IMU嵌入映射至LLaMA-2的文本token空间，实现跨模态转换。

- **物理关系生成**：  
  IMUGPT 利用文本描述生成3D人体运动（使用T2M-GPT），再根据运动动力学物理模型反推IMU数据，实现文本→3D运动→IMU的生成链。

- **时间序列图作为图像**：  
  CLIP-LSTM 将股票价格序列转换为K线图图像，与文本描述一起输入CLIP模型提取视觉-语言联合特征。Insight Miner 将时间序列窗口绘制成折线图，输入LLaVA模型生成趋势描述。

### 5. 工具集成（Tool Integration）
不直接用LLM处理时间序列，而是让LLM生成外部工具（如代码或API调用）间接辅助分析。

- **代码生成**：  
  CTG++ 使用GPT-4根据自然语言描述生成可微分损失函数代码，指导扩散模型生成交通轨迹，实现“语言→代码→扩散模型”的协同。

- **API调用**：  
  ToolLLM 构建通用工具使用框架，使LLM能调用天气、股票预测等API，实现外部工具的动态集成。

- **领域知识增强**：  
  SHARE 使用GPT-4增强人体活动标签名称的语义结构；GG-LLM 使用LLaMA2编码人类行为常识进行动作预测；SCRL-LG 使用LLaMA-7B从新闻标题中提取股票特征，用于强化学习中的特征对齐。

---

以上五类方法构成完整的LLM赋能时间序列分析的技术谱系，分别覆盖输入、分词、嵌入、模型推理与输出五个阶段，形成从“直接提示”到“工具协同”的渐进式知识迁移路径。