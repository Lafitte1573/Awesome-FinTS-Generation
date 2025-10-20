## 金融时间序列数据生成的最新先进技术综述

> This repository displays a paper collection of the survey of recent financial data-generation technologies.

### 写作大纲

| 章节 | 标题 | 主要内容 |
| --- | --- | --- |
| 1 | 引言 | |
| 2 | 时间序列数据预测 | 问题定义、方法分类、评价指标、主要挑战 |
| 3 | 时间序列数据补插 | 问题定义、方法分类、评价指标、主要挑战 |
| 4 | 时间序列数据增强 | 问题定义、方法分类、评价指标、主要挑战 |
| 5 | 数据集整理 | 为方便研究工作的开展，列表总结金融时序数据生成领域的数据集，维度包括开放性、下游任务、数据量等 |
| 6* | 未来方向* | 总结金融时序数据生成领域（非时序预测/合成/增强）未来的研究方向 |
| 7 | 总结 | |

### 基本思路

1. **确定任务**：金融时序预测（基于数据生成方法）、金融时序数据补插、金融时序数据增强
2. **确定技术路线**：
- 金融时序预测：基于机器学习、基于深度学习（基于GAN、DM）、基于大模型（Single Agent、Multi-Agent）、融合方法
- 金融时序数据补插：
- 金融时序数据增强：

### 分类学大纲
```mermaid
graph LR
    A(金融数据生成（Synthesis）) --> B([金融时序预测（Forecasting）])
    B --> B1(GAN_based) --> C1(Wavelet-GAN, TG-3DCNN, ORGAN-FVT, AssetGANs, CT-GAN-NAR-NN, TSS-CGAN)
    B --> B2(LLM_based) --> C2(TimeLLM, Chronos, Time-LLM, LLM4FTS, TimesFM_Financial_Finetuning, StockTime, TimeS, FINMEM)
    B --> B3(Machine_Learning) --> C3(SGP-LSTM, LVQ-CBR, Hierarchical-LSTM, MVO-BiGRU, ALN, 优化不变正则化器（L_OB^e）, PEC-W)
    A --> C([金融时序数据插补（Imputation）])
    C --> D1(GAN_based) --> E1(Market-GAN, RSQGAN, WGAN-based Market Simulation, , DAT-CGAN, CTS-GAN, Tail-GAN, TCN-GAN, WGAN-BiLSTM, CoMeTS-GAN, , Jinkou, MC-TE-GAN, Generative-CNN, QWGAN-GP)
    C --> D2(Machine_Learning) --> E2(SGP-LSTM, FED2Port, VAE-GRU-MCMC, CFTNet)
    C --> D3(Diffusion_based) --> E3(CoFinDiff, DiGA, Wavelet-Diffusion, TRADES)
    A --> D([金融时序数据增强（Augmentation）])
    D --> E(Machine_Learning) --> F1(PPCA-TVP-FAVAR, Kernel Ridge Regression for Swiss Discount Curve, NMTucker)
```

----

## 文献列表

### 金融时序预测

#### StockTime: A Time Series Specialized Large Language Model Architecture for Stock Price Prediction
- **Authors**: Shengkun Wang, Taoran Ji, Linhan Wang, Yanshen Sun, Shang-Ching Liu, Amit Kumar, Chang-Tien Lu
- **Year**: 2024
- **Abstract**: 为解决传统金融大语言模型（FinLLMs）在股票价格预测中忽视时序特征、依赖冗余文本信息且效率低下的问题，本文提出StockTime，一种专门针对股票价格时序数据的LLM架构，通过将股票价格分块为令牌、提取其相关性与统计趋势等文本信息，并与自回归编码器提取的时序特征在嵌入空间融合，利用冻结的LLM进行下一令牌预测，从而在不微调LLM的情况下实现更高精度、更低资源消耗的多周期股票价格预测。StockTime架构包含四个核心组件：(1) 分块输入：对标准化后的股票价格序列进行非重叠分块，每块作为基本令牌，降低计算负担；(2) 自回归编码器：使用LSTM层捕捉分块序列的时序依赖关系，并通过全连接层将特征投影至LLM嵌入维度，生成价格嵌入；(3) 多模态融合：从同一价格数据中提取股票相关性、行业分类、统计指标（极值、均值、变化率）和时间戳，构建文本模板，经冻结LLM编码为文本嵌入，并与价格嵌入在潜在空间相加融合，实现时序与文本信息的无缝集成；(4) 令牌级预测：将融合后的嵌入序列输入冻结的LLM，利用其自然语言的下一令牌预测能力，输出预测的下一价格块，通过线性投影层还原为预测价格序列，使用均方误差（MSE）损失函数仅训练嵌入与投影层，实现高效微调。整个过程完全基于股票价格时序数据，无需外部文本或FinLLM微调。

#### Forecasting stock prices changes using long‑short term memory neural network with symbolic genetic programming
- **Authors**: Qi Li, Norshaliza Kamaruddin, Siti Sophiayati Yuhaniz, Hamdan Amer Ali Al-Jaif
- **Year**: 2023
- **Abstract**: 为解决中国股票市场跨截面收益预测中特征工程薄弱和传统深度学习模型精度不足的问题，本文提出一种结合符号遗传编程（SGP）与长短期记忆网络（LSTM）的混合模型，通过SGP自动生成并优化融合基本面与技术面指标的非线性特征，再输入LSTM进行时序模式学习，显著提升了预测准确率和风险调整收益，Rank IC和ICIR分别提升1128%和5360%（基本面）及20%和2752%（技术面），年化超额收益超越CSI 300达31.00%该方法包含四个阶段：（1）数据预处理：从S&P Alpha Factor Library获取中国A股4500只股票的基本面与技术面指标（共304个），进行缺失值处理、异常值剔除和Z-score标准化；（2）符号遗传编程（SGP）增强：以原始指标为基因，结合40种启发式算子（如ts_corr、ts_rank、delta等），通过遗传算法进行选择、交叉（40%概率）和突变（低概率），构建符号表达式树，使用改进的适应度函数（融合Rank IC、收益单调性、Top/Bottom组超额收益）评估特征质量，并通过成功率（IC_success_ratio）和盈亏比（IC_PNL）筛选最优特征；（3）特征序列化：将SGP生成的最优特征按15日滞后序列化，形成时序输入；（4）LSTM特征提取：将序列化特征输入双层LSTM网络，使用最终隐藏层输出预测个股未来5日收益是否高于市场中位数，通过Rank IC与ICIR评估模型性能，并构建多头投资组合（前10%股票等权持有），在2019-2022年滚动窗口（训练1020天、验证160天、测试20天，滚动周期20天）下进行回测，最终实现显著超越CSI 300、CSI 500及市场平均的超额收益。

#### A hybrid carbon price forecasting model combining time series clustering and data augmentation
- **Authors**: Yue Wang, Zhong Wang, Yuyan Luo
- **Year**: 2023
- **Abstract**: 为解决碳价格序列高波动、非平稳和数据量小导致的预测精度低与过拟合问题，本文提出一种融合时间序列聚类、数据增强与TVFEMD分解的混合预测模型Informer-DA-BOHB-TVFEMD-CL，通过TVFEMD分解获得高细节IMF分量，利用时间序列聚类保留高频分量并重构低频分量，采用VAE生成数据增强训练集，并结合BOHB优化的Informer模型进行预测，显著提升了多碳市场的预测精度与交易盈利能力。首先，使用时间变滤波经验模态分解（TVFEMD）对碳价格序列进行分解，获得多个本征模态函数（IMF）；其次，采用基于动态时间规整（DTW）距离的k-means时间序列聚类方法，将IMF分为高频与低频组，保留高频IMF，合并低频IMF进行重构；接着，利用变分自编码器（VAE）对每个重构分量的训练集进行数据增强，以缓解数据稀缺与过拟合问题；然后，采用贝叶斯优化与超带（BOHB）算法自动优化Informer模型的超参数（如编码器层数、注意力头数、学习率等）；随后，将增强后的训练数据输入优化后的Informer模型，分别预测各分量；最后，将各分量的预测结果加总，得到最终碳价格预测值。实验在EU-ETS、广东、湖北和全国碳市场验证，结果表明该模型在MAE、RMSE、MAPE和R²等指标上显著优于基准模型，且构建的交易策略具有更高盈利能力。

#### Large Scale Financial Time Series Forecasting with Multi-faceted Model
- **Authors**: Defu Cao, Yixiang Zheng, Parisa Hassanzadeh, Simran Lamba, Xiaomo Liu, Yan Liu
- **Year**: 2023
- **Abstract**: 为解决金融时序预测中因分布偏移导致的模型泛化能力差的问题，本文提出一种基于松弛不变风险最小化的多面统一模型，通过引入优化驱动的正则化项放宽传统IRM的严格约束，在S&P 500多行业数据上联合训练线性/非线性模型，显著提升对已见和零样本行业的预测准确性，EBITDA预测误差平均降低27.87%。本文提出一种优化驱动的不变性正则化方法，用于提升金融时序预测模型在分布偏移场景下的泛化能力。首先，收集S&P 500公司10个行业的季度财务数据（收入、COGS、SG&A、RD_EXP等），构建多变量时间序列预测任务，目标为预测未来一年的Revenue和EBITDA。其次，采用多层感知机（MLP）作为基础预测模型，结合滑动窗口（24季度窗口，前20季度输入，后4季度预测）构建训练样本。然后，提出一种松弛的不变性正则化项：$\mathrm{L}_{\mathrm{OB}}^{e} = \| (w \circ \nabla R^{e}(w, f))^{\circ c} \|_{2}^{2}$，其中$w$为预测器权重，$\nabla R^{e}(w, f)$为环境e的损失梯度，$\circ$为Hadamard积，$c > 1$为可调超参数，用于增强对大梯度维度的惩罚，同时通过乘以权重$w$抑制非因果特征。该正则项替代传统IRM的严格不变性约束，通过梯度幅值动态调节正则强度，使模型在保持跨环境一致性的同时，允许更灵活的表示学习。最后，总损失为$\mathrm{L}^{e} = \mathrm{L}_{\mathrm{MSE}}^{e} + \gamma \mathrm{L}_{\mathrm{OB}}^{e}$，通过梯度下降联合优化模型参数与正则化项，在7个已见行业训练，在3个零样本行业（Consumer Defensive, Consumer Cyclical, Industrials）评估。实验表明，该方法在EBITDA预测上相比最佳基线平均降低27.87%的SMAPE误差，且模型参数量仅为Transformer的1/856，具备高效与强泛化双重优势。

#### A Novel Wavelet based Generative Model for Time Series Prediction
- **Authors**: Chaofan Dai, Xiaoguang Yuan, Zongkai Tian, Xinyue Hu, Zhen Luan, Youchen Wang
- **Year**: 2024
- **Abstract**: 为解决股票市场非线性、非平稳时间序列预测精度低的问题，本文提出一种基于小波变换的生成对抗网络（Wavelet-GAN），通过小波分解将原始价格序列分解为多尺度频率分量，分别用ARMA模型预测小波系数，再将预测系数输入Wasserstein GAN框架生成合成价格趋势，最终实现比GRU、LSTM和传统GAN更精准的预测效果（R²达0.99）。首先，对原始股票价格序列进行离散小波变换（DWT），采用DB2小波基函数分解为多尺度低频和高频系数；其次，对每一层小波系数建立ARMA模型进行独立预测；然后，将预测得到的小波系数作为输入，构建Wasserstein GAN（WGAN-GP）生成器与判别器：生成器由两层GRU（256和128单元）组成，用于生成合成价格序列，判别器由三层一维卷积层（32、64、128通道）构成，通过Wasserstein距离与梯度惩罚项优化对抗训练，确保稳定收敛；最后，将预测并生成的小波系数进行小波重构，恢复为完整的价格序列预测结果。整个流程在TensorFlow 2.13.0上实现，使用Adam优化器训练100轮，对比GRU、LSTM和传统GAN，显著降低MSE、RMSE和MAE，提升R²至0.99。

#### Enhancing Recurrent Neural Networks For Stock Market Forecasts through PEC-W Framework
- **Authors**: Bus¸ra C¸ alıs¸kan
- **Year**: 2024
- **Abstract**: 为解决股票市场短期预测中LSTM和GRU模型存在的过拟合与训练时间长问题，本文提出一种基于PEC-W预处理框架的方法，通过滚动窗口均值聚合、均值减法归一化和离散小波变换（DWT）增强时序特征并降低数据维度，同时结合SHAP可解释性分析验证模型有效性，显著提升了预测精度并大幅缩短训练时间。该方法首先选取与调整收盘价高度相关的开盘价作为辅助特征；接着应用PEC-W预处理框架：(1) 使用60天滚动窗口对原始数据进行分割；(2) 在每个窗口内计算每周（5天）均值，生成聚合序列；(3) 对聚合序列进行均值减法归一化以消除趋势影响；(4) 应用离散小波变换（DWT）提取低频趋势与高频波动成分，压缩数据维度；(5) 将处理后的开盘价和调整收盘价特征拼接，输入至LSTM或GRU模型进行训练；(6) 模型结构为：64单元RNN层（返回序列）→ 20% Dropout → 32单元LSTM/GRU层 → 单输出Dense层，使用Adam优化器与MSE损失函数，训练200轮并采用EarlyStopping防止过拟合；(7) 使用SHAP方法分析特征重要性，验证预处理提升的鲁棒性。实验在AAPL、AMZN、MSFT三只股票数据上验证，结果显示PEC-W框架使RMSE降低约40%-70%，训练时间缩短60%-90%，R²提升至0.97以上。

#### Adversarial Learning Networks for FinTech Applications Using Heterogeneous Data Sources
- **Authors**: Parus Khuwaja, Sunder Ali Khowaja, Kapal Dev
- **Year**: 2023
- **Abstract**: 为解决金融市场上由于数据异构性、缺失值和市场崩盘期间预测性能下降的问题，本文提出一种基于对抗学习网络（ALN）的股票价格预测框架，通过融合股票价格、推文和全球宏观指标构建异构知识库，采用改进的牛顿插值多项式（NDDP）进行缺失值插补，利用LSTM提取时序特征，并设计HDFM Q-learning（批评者）与对抗性Q-learning（参与者）网络进行对抗训练，显著提升了在市场波动和崩盘场景下的预测准确率，相比现有方法在准确率上提升5.58%以上。1) 构建异构知识库：整合OHLC数据、调整后收盘价、交易量、社交媒体推文情感得分和全球宏观指标（如GDP、失业率等）；2) 数据预处理：使用Haar小波变换去噪股票数据，用TextBlob提取推文情感得分并取日均值，对月度宏观指标采用高斯分布随机插值以匹配每日采样率；3) 缺失值插补：提出改进的牛顿插值多项式（NDDP）方法，基于局部数据点和采样密度动态插补缺失情感得分，显著优于均值、EM等传统方法；4) 市场崩盘检测：采用拓扑数据分析（TDA）中的持久同调方法，通过滑动窗口构建点云几何结构，计算欧氏距离生成崩盘触发阈值（20%-35%），提取不稳定市场时段数据；5) 特征提取：使用双层LSTM（每层100单元）分别处理融合数据（HDF）和崩盘数据（HDFM），通过遗忘门、输入门和输出门机制学习长期依赖；6) 对抗强化学习框架：设计两个对抗网络——HDFM Q-learning（批评者）仅用崩盘数据训练，预测不稳定市场行为；对抗性Q-learning（参与者）使用全量数据训练，通过策略网络（Actor）选择动作，批评者网络提供Q值反馈；7) 训练机制：采用经验回放和Adam优化器，批评者网络最小化Q值均方误差，参与者网络通过策略梯度更新最大化预期奖励，奖励函数采用夏普比率；8) 最终预测：通过Actor-Critic对抗训练，使模型在稳定与崩盘市场中均能自适应优化，实现71.42%的准确率和0.238的MCC，显著优于现有方法。

#### Distributed Generative Adversarial Networks for Fuzzy Portfolio Optimization
- **Authors**: Xueying Yang, Chen Li, Zidong Han, Zhonghua Lu
- **Year**: 2023
- **Abstract**: 为解决金融时间序列多步预测精度低、训练效率差及模糊投资组合优化计算耗时的问题，本文提出基于WGAN-GP的分布式生成对抗网络AssetGANs，通过卷积神经网络生成器与判别器联合训练模拟未来多日资产收益，并结合模糊模拟与MPI并行化遗传算法优化模糊Mean-CVaR投资组合模型，实现了比LSTM更低的RMSE（0.4615 vs 0.6638）和8 GPU下573倍的训练加速，同时模糊组合优化并行效率达96.3%。首先，构建基于WGAN-GP的AssetGANs模型，生成器采用4层卷积网络提取过去k天资产收益模式，结合随机噪声向量，通过反卷积生成未来Δl天的收益序列；判别器采用5层卷积网络区分真实与生成序列，并引入梯度惩罚稳定训练。模型采用PyTorch的DistributedDataParallel在多GPU上并行训练，使用Ring All-reduce算法聚合梯度，Gloo通信库实现节点间通信。其次，利用训练好的生成器对资产未来收益进行模糊模拟，通过蒙特卡洛采样生成多组收益路径，估计每个资产的三角模糊变量参数（a_i, b_i, c_i），构建隶属度函数。然后，使用SARPROP神经网络近似模糊期望收益与模糊CVaR，加速目标函数计算。最后，采用MPI并行化的遗传算法进行全局优化，通过轮盘赌选择、交叉变异操作求解满足约束的最优资产权重，实现模糊Mean-CVaR模型的高效求解。

#### Large Language Models for Financial Time Series Forecasting
- **Authors**: Miguel Noguer i Alonso, Rodolfo Pereira Franklin
- **Year**: 2025
- **Abstract**: 为解决传统时间序列模型在金融数据中泛化能力差和依赖大量标注数据的问题，本文评估了包括Time-LLM在内的多种大语言模型（LLM）在股票价格预测中的表现，其基本思路是通过文本重编程（patch reprogramming）和提示引导（Prompt-as-Prefix）将连续时间序列转化为语言模型可处理的离散文本表示，无需微调基础LLM即可实现零样本或小样本预测，实验表明Time-LLM、PatchTST和KAN在稳定与波动市场中均能超越传统模型如NBEATS和NHITS。本文方法核心为Time-LLM框架，其步骤包括：1) 将多变量时间序列分解为多个单变量序列；2) 对每个序列应用可逆实例归一化（RevIN）以处理非平稳性；3) 将时间序列划分为长度为L_p的重叠或非重叠补丁（patches）；4) 使用学习的文本原型将补丁嵌入映射到预训练语言模型（如Llama、GPT-2）的词嵌入空间，实现跨模态重编程；5) 引入提示前缀（Prompt-as-Prefix, PaP）将任务指令、统计特征（如趋势、周期、极值）以自然语言形式作为输入前缀，引导LLM理解时序语义；6) 利用多头交叉注意力机制对齐时间序列补丁与语言嵌入，捕捉局部与全局依赖；7) 通过线性投影层将LLM输出映射回时间序列域，生成未来H步预测；8) 仅训练轻量级的输入转换与输出投影模块，保持基础LLM参数冻结，实现高效微调。同时，论文对比评估了NBEATS（基于神经基底展开）、NHITS（多尺度分层插值）、PatchTST（Transformer补丁编码）和Chronos（量化token化）等模型，采用MAE、MSE、RMSE指标在Google、Apple、Amazon、JPMorgan、Meta等股票数据上验证性能。

#### Large Language Models for Financial Aid in Financial Time-series Forecasting
- **Authors**: Md Khairul Islam, Ayush Karmacharya, Timothy Sue, Judy Fox
- **Year**: 2024
- **Abstract**: 为解决金融援助领域因数据稀缺导致传统深度学习模型效果不佳的问题，本文提出利用预训练大语言模型（LLM）作为基础模型，在仅使用少量训练数据（少样本）或完全不微调（零样本）的情况下，对州级财政援助资金进行年度预测，实验表明TimeLLM和PatchTST在少样本场景下表现最优，而GPT4TS在零样本场景下表现相对较好，但整体零样本效果仍有限。本文首先收集了来自四个金融领域（股票、商品、货币、机构）的8个时间序列数据集，其中金融援助数据为17年（2004–2020）的年度数据，其他为日频数据。对于金融援助任务，使用过去10年的数据预测下一年的援助金额；其他任务使用过去96天预测未来24天。模型方面，选取了5个传统时间序列模型（DLinear、iTransformer、TimesNet、PatchTST、TimeMixer）和3个基于预训练GPT-2的LLM基础模型（TimeLLM、CALF、GPT4TS）。训练时，传统模型从零开始训练，LLM模型仅微调输出层，其余参数冻结。在少样本实验中，仅使用10%的训练数据进行微调；在零样本实验中，直接使用预训练模型在测试集上评估，不进行任何微调。评估指标为MSE和MAE，实验结果表明LLM在少样本学习中显著优于传统模型，但在零样本场景下预测性能仍不理想，表明其尚未完全适应金融时序数据的分布特性。

#### An Efcient GAN‑Based Multi‑classifcation Approach for Financial Time Series Volatility Trend Prediction
- **Authors**: Lei Liu, Zheng Pei, Peng Chen, Hang Luo, Zhisheng Gao, Kang Feng, Zhihao Gan
- **Year**: 2023
- **Abstract**: 为解决金融时间序列中短期、中性、长期波动趋势的多分类预测问题，本文提出了一种基于序回归生成对抗网络（ORGAN-FVT）的方法，通过ConvLSTM生成器学习时序数据分布，结合引入序回归惩罚机制的MLP判别器，优化预测结果对误判方向（如将短期误判为长期）的敏感性，显著提升了在MSFT、TSLA和PAICC三个股票数据集上的AUC和F1分数，最高提升达20.81%本文提出的ORGAN-FVT方法包含生成器与判别器两部分：生成器采用ConvLSTM结构，以包含11个技术指标（如Close、RSI、ADX等）的30天滑动窗口时序数据为输入，通过卷积层提取局部特征，再经LSTM层捕获长期时序依赖，最后通过全连接层与Softmax输出三类趋势（短、中、长）的概率分布；判别器采用MLP结构，输入包括真实标签的one-hot向量和生成器输出的预测概率向量，通过交叉熵损失函数进行训练，并引入序回归约束：当真实标签为短期时，惩罚预测为长期（增大β而非γ）；当真实标签为长期时，惩罚预测为短期（增大β而非α），从而在损失函数中对跨类误判施加更高代价。整个模型通过对抗训练最小化生成器损失和最大化判别器损失，最终达到纳什均衡，训练完成后使用生成器进行端到端的三分类预测。实验在三个真实股票数据集上进行，对比LSTM、GRU、CNN、ConvLSTM和GAN-FVT等基线方法，采用宏观平均和加权平均的F1-score、AUC等指标评估，结果表明ORGAN-FVT在多数指标上显著优于其他方法。

#### A semi-heterogeneous ensemble forecasting method for stock returns based on sentiment analysis
- **Authors**: Xiao Zhang, Peide Liu, Jing Feng
- **Year**: 2023
- **Abstract**: 为解决股票收益预测中传统模型忽视投资者情绪与特征多样性的问题，本文提出一种基于情感分析的半异构集成预测方法，通过监督数据增强构建注意力-PCA情感指数，结合变量扰动生成多样化的基模型（MLR与BPNN），并采用加权集成策略融合异构与同构优势，显著提升了S&P 500收益预测的准确性与泛化能力。1. 构建情感词典：从《纽约时报》和《华尔街日报》金融新闻中提取词项，经预处理（去停用词、词干化、二元搭配）后，使用TF-IDF筛选前100个高频词，形成N维词频矩阵。2. 构建监督情感指数：采用OLS、Ridge、LASSO、Elastic Net回归计算每个词项与S&P 500收益的相关系数ci，将词项分为pos、neg、pos+neg、all四类；对每类词项应用soft-max函数计算注意力权重ωi，加权构造新特征矩阵X*。3. 应用注意力-PCA：对加权后的X*进行主成分分析，提取第一主成分作为监督情感指数。4. 生成基模型池：结合四种回归方法（OLS/Ridge/LASSO/Elastic Net）与两种预测模型（MLR、BPNN），在不同情感指数输入（pos/neg/pos+neg/all）和变量扰动下，生成多组基模型。5. 构建半异构集成框架：集成策略融合同构（相同算法不同输入）与异构（不同算法）模型，采用基于预测性能的加权平均（如ABC算法优化权重）整合预测结果。6. 评估与优化：使用RMSE、MAE、R_OOS^2、Dstat评估性能，通过偏差-方差-协方差分解与双错误率度量评估模型多样性，实验表明该方法在S&P 500月度收益预测中显著优于基准模型。

#### Price Forecast of Treasury Bond Market Yield: Optimize Method Based on Deep Learning Model
- **Authors**: WEIYING PING, YUWEN HU, LIANGQING LUO
- **Year**: 2023
- **Abstract**: 为解决国债收益率预测中多变量时间序列高噪声、非线性和多重共线性导致的传统模型精度不足的问题，本文提出一种基于LASSO-SMLR-PCA降维与贝叶斯优化LSTM的深度学习模型框架，通过逐步筛选和降维处理输入变量，再利用贝叶斯优化调整LSTM超参数，实现了对国债收益率的高精度滚动预测，显著提升了模型拟合效果与实际应用稳定性。首先，选取中国债券市场十年期国债收益率作为目标变量，结合经济基本面、宏观货币政策、货币市场价格、大类资产配置和债券交易情绪五个维度的64个高频指标作为原始输入；其次，对数据进行缺失值插补（Pchip方法）和标准化处理；然后，采用LASSO回归剔除多重共线性变量和不显著变量，保留54个有效变量；接着，使用逐步回归（SMLR）进一步筛选与目标变量显著相关的变量；随后，应用主成分分析（PCA）对筛选后的变量进行降维，提取累计贡献率超过85%的主成分作为新输入；之后，将主成分与滞后一期的国债收益率作为LSTM模型的输入，使用贝叶斯优化算法自动搜索并确定LSTM的最优超参数（如学习率、神经元数量、层数等）；最后，在训练集上训练优化后的LSTM模型，并在测试集上进行滚动预测，使用RMSE、MAE、MAPE和R²等指标评估模型性能，验证其在国债收益率预测中的高精度与稳定性。

#### Article
- **Authors**: Farhat Iqbal, Dimitrios Koutmos, Eman A. Ahmed, Lulwah M. Al-Essa
- **Year**: 2024
- **Abstract**: 为解决低频金融时间序列数据中过拟合和预测精度低的问题，本文提出了一种名为MVO-BiGRU的混合深度学习模型，该方法通过变分模态分解（VMD）将汇率序列分解为多个子序列，利用防过拟合模块（Prevention module）随机组合子序列进行数据增强，结合预测模块（Prediction module）使用全部子序列进行建模，并通过Optuna优化超参数，最终融合两模块输出通过全连接网络预测汇率，实现了显著优于基准模型的预测精度和泛化能力。本文提出的MVO-BiGRU方法包含以下步骤：首先，使用变分模态分解（VMD）将原始外汇汇率时间序列分解为K个具有不同频率的变分模态函数（VMFs）；其次，构建两个双向门控循环单元（BiGRU）模块：Prevention模块每10个训练周期随机选择n_K（1 < n_K < K）个VMFs作为输入，通过动态组合增强数据多样性以防止过拟合；Prediction模块则使用全部K个VMFs作为输入进行预测；接着，利用Optuna算法自动优化BiGRU的超参数（包括窗口长度、神经元数量、层数、学习率、dropout概率）以及Prevention模块的n_K值；然后，两个模块的输出被融合为新特征，输入至一个全连接神经网络进行最终汇率预测；训练过程中采用L2正则化和Dropout进一步抑制过拟合，使用Adam优化器最小化均方误差（MSE）；最后，模型在EUR/SAR和EUR/CNY两个汇率数据集上进行验证，结果表明其在R²、MAE、MSE、MAPE和方向准确率（DA）等指标上均显著优于传统ML/DL基准模型。

#### Financial Fine-tuning a Large Time Series Model
- **Authors**: Xinghong Fu, Masanori Hirano, Kentaro Imajo
- **Year**: 2024
- **Abstract**: 为解决金融价格数据的非平稳性和极端波动导致基础时间序列模型TimesFM预测性能差的问题，本文提出对TimesFM进行金融数据持续预训练，通过对价格数据进行对数变换稳定损失函数并优化掩码策略，使模型在多种金融市场中显著提升预测准确率，并在模拟交易中实现优于基准模型的收益、夏普比率和最大回撤表现。本文基于TimesFM（2亿参数的解码器型时间序列基础模型）进行金融数据持续预训练。首先，收集包含100M时间点的金融数据集（涵盖股票、外汇、加密货币，粒度为日频和小时频），并保留2023年后数据作为测试集。其次，对原始价格序列进行对数变换，将MSE损失应用于对数域，以缓解大值偏差和市场崩盘导致的数值不稳定问题；同时，采用动态掩码策略，随机采样128~512长度的上下文片段进行训练，增强模型对不同长度序列的泛化能力。训练采用SGD优化器，线性热身25轮后余弦衰减，峰值学习率5e-4，批量大小1024，共训练100轮。训练完成后，通过多步预测（预测步长h=2,4,...,128）评估价格涨跌分类准确率和Macro F1-score，并设计两种模拟交易策略：基本策略（根据预测方向做多/做空）和市场中性策略（减去日均仓位以消除市场整体趋势影响）。实验表明，微调后的TimesFM在预测准确率和F1-score上显著优于原始模型和随机基准，在S&P500等市场中实现最高3.6%的年化收益和1.68的年化夏普比率，且最大回撤控制在1%以内。

#### Retrieval-augmented Large Language Models for Financial Time Series Forecasting
- **Authors**: Mengxi Xiao, Zhengyu Chen, Lingfei Qian, Zihao Jiang, Yueru He, Yijing Xu, Yuechen Jiang, Dong Li, Ruey-Ling Weng, Jimin Huang, Min Peng, Sophia Ananiadou, Jian-Yun Nie, Qianqian Xie
- **Year**: 2024
- **Abstract**: 为解决金融时序数据中传统检索方法难以捕捉复杂时序依赖与隐含市场信号的问题，本文提出FinSrag框架，通过引入基于LLM反馈训练的领域专用检索器FinSeer，从包含28个金融指标的增强数据集中检索最具预测价值的历史序列，并将其注入微调的StockLLM中进行股票涨跌预测，显著提升了预测准确率并超越了现有文本和距离基检索方法。FinSrag框架包含三个核心步骤：(1) 构建增强型检索候选池：基于Yahoo Finance数据，整合6个基础价格指标、10个技术指标和18个通过互信息筛选的Alpha因子，形成共34个金融特征的候选序列，每个候选为过去5日的单一指标值序列，并采用JSON格式序列化；(2) 训练FinSeer检索器：利用StockLLM-1B-Instruct作为教师模型，对每个查询-候选对生成预测logits，将其转化为概率作为相关性分数，选取Top-1为正样本、Bottom-15为负样本，通过对比学习最大化查询与正样本嵌入相似度、最小化与负样本相似度，并使用KL散度进行知识蒸馏，使FinSeer学习LLM的隐式偏好；(3) 推理阶段：FinSeer根据训练好的嵌入空间检索最相关的候选序列，将其结构化后作为上下文输入StockLLM，由LLM综合历史信号预测未来股价涨跌。该方法首次实现金融时序数据的领域自适应检索增强，突破了传统基于价格或距离（如DTW）的检索局限。

#### Nonlinear Regression with Hierarchical Recurrent Neural Networks Under Missing Data
- **Authors**: S. Onur Sahin, Suleyman S. Kozat
- **Year**: 2023
- **Abstract**: 为解决序列数据中存在缺失值导致传统神经网络性能下降的问题，本文提出了一种层次化LSTM架构，通过将输入空间根据历史输入的‘存在模式’划分为多个区域，并为每个模式分配独立的LSTM专家网络，仅利用实际存在的输入进行预测，避免数据插补带来的误差累积，从而在金融和真实世界数据集上显著提升了预测精度且计算复杂度与传统LSTM相当。本文提出层次化LSTM（Hierarchical-LSTM）架构，其核心思想是：1) 定义一个滑动窗口（长度为L），记录最近L个输入的存在/缺失模式（presence-pattern），每个模式是一个长度为L的二进制向量；2) 为每个可能的presence-pattern（共2^L - 1种非全零模式）分配一个独立的LSTM网络（称为叶网络），并设置一个主LSTM网络处理窗口外的历史信息；3) 当新输入到达时，根据当前窗口的presence-pattern确定激活的LSTM网络集合（包括所有子模式对应的网络）；4) 主LSTM网络将其状态传递给所有激活的叶LSTM网络作为初始状态；5) 每个激活的叶LSTM网络仅使用其对应模式中存在的输入进行前向计算，生成输出；6) 使用softmax加权融合所有激活LSTM网络的输出，权重由当前输入模式、网络模式和输出向量共同决定；7) 最终预测值由融合后的状态向量通过线性变换得到；8) 该架构不进行任何数据插补，完全基于实际观测值和缺失模式进行建模，且计算复杂度随缺失率升高而降低，在高缺失场景下优于传统插补方法。

#### A Financial Time Series Denoiser Based on Diffusion Models
- **Authors**: Zhuohan Wang, Carmine Ventre
- **Year**: 2024
- **Abstract**: 为解决金融时间序列信噪比低导致预测不准和交易效率低的问题，本文提出一种基于条件扩散模型的去噪方法，通过前向加噪与反向去噪过程结合分类器无引导和总变差/傅里叶损失辅助优化，重构更平滑且趋势保留的时序数据，显著提升未来收益分类准确率与交易收益并降低交易频率。本文提出一种基于条件扩散模型的金融时间序列去噪框架。首先，使用带条件的变分爆炸（VE-SDE）或方差保持（VP-SDE）扩散模型，将原始时间序列（如EMA）作为输入，通过前向过程在不同噪声水平t下添加噪声；训练阶段采用分数匹配目标函数，通过Transformer架构的神经网络s_θ(x,t,c)估计噪声数据的分数函数，其中条件c设为原始输入x₀以稳定去噪方向。推理阶段采用无引导分类器机制（ω=1）和两个辅助损失函数：总变差（TV）损失促进局部平滑，傅里叶损失（FFT滤波）保留主要频域特征。去噪过程从中间噪声水平T'（如T'=0.5）开始，而非随机高斯噪声，通过预测-校正采样器（Predictor-Corrector）和多随机种子平均减少采样随机性，最终输出稳定去噪序列。该去噪序列用于下游分类任务（预测未来1/5/10步收益）和交易策略（MACD、布林带），显著提升F1分数、MCC指标和交易利润，同时减少交易次数。

#### Probabilistic simulation of electricity price scenarios using Conditional Generative Adversarial Networks
- **Authors**: Viktor Walter, Andreas Wagner
- **Year**: 2023
- **Abstract**: 为解决日前电力市场价格分布建模中传统统计模型依赖复杂假设和难以捕捉市场特异性规律的问题，本文提出一种基于条件生成对抗网络（TSS-CGAN）的方法，通过融合电价预测、可再生能源出力、负荷、小时信息等多变量时序特征，利用一维卷积网络提取空间特征、双向LSTM捕捉时序依赖，并通过生成器与判别器对抗训练生成24小时电价场景，实现了比DeepAR等基准模型降低50%连续排序概率得分（CRPS）的显著效果，能准确反映可再生能源波动对电价的非线性影响。本文提出的时间序列模拟条件生成对抗网络（TSS-CGAN）包含生成器与判别器两个部分。生成器输入包括：1）随机噪声向量z（来自标准正态分布）；2）未来24小时的多变量特征矩阵X（包含电价预测、风电/光伏出力预测、负荷预测、剩余负荷预测、燃气/煤电容量预测及小时编码）；3）过去24小时的真实电价序列y。噪声向量z经两层一维卷积（32和64个滤波器，核大小2）和最大池化后，与特征矩阵X进行逐元素相乘，再与过去24小时真实电价序列拼接，形成(24, n+1)维度输入，经双向LSTM层（24个单元）提取时序特征，最后通过全连接层输出24维电价场景。判别器输入为生成或真实的电价场景、相同特征矩阵X和过去24小时真实电价y，结构与生成器类似，但输入为真实或生成的电价序列，同样经一维卷积、逐元素乘法、双向LSTM和全连接层，最终通过Sigmoid激活输出真实概率。训练采用二元交叉熵损失，生成器预训练10000步（MSE损失），判别器每轮更新2次，生成器更新1次，优化器为Adam（学习率0.0001）。模型在德国EPEX SPOT市场3年数据（2020–2023）上训练，测试期生成10000组24小时电价场景，通过CRPS、分位数覆盖、敏感性分析等评估其对市场规律（如午间电价谷、周末低价、可再生能源影响）的建模能力。

#### Article
- **Authors**: Farhat Iqbal, Dimitrios Koutmos, Eman A. Ahmed, Lulwah M. Al-Essa
- **Year**: 2024
- **Abstract**: 为解决低频金融时间序列数据中过拟合和预测精度低的问题，本文提出了一种名为MVO-BiGRU的混合深度学习模型，通过变分模态分解（VMD）将汇率序列分解为多个子序列，利用预防模块随机组合子序列进行数据增强以提升泛化能力，再结合预测模块使用全部子序列进行建模，并通过Optuna优化超参数，最终在EUR/SAR和EUR/CNY汇率预测中实现了显著优于基准模型的预测精度和稳定性。首先，使用变分模态分解（VMD）将原始外汇汇率时间序列分解为K个具有不同频率的变分模态函数（VMFs）；接着，构建双模块架构：预防模块（Prevention Module）每十轮训练随机选取n_K个VMFs（1 < n_K < K）作为输入，通过动态组合生成大量数据变体，增强模型泛化能力并防止过拟合；预测模块（Prediction Module）则使用全部K个VMFs作为输入进行标准序列建模；两个模块均采用双向门控循环单元（BiGRU）网络处理时序依赖；使用Optuna算法自动优化关键超参数（如窗口长度、神经元数量、层数、学习率、丢弃率和n_K值）以获得最优配置；最后，将预防模块和预测模块的输出特征拼接，输入一个全连接神经网络进行最终汇率预测；模型训练使用Adam优化器和L2正则化，评估指标包括R²、MAE、MSE、MAPE和方向准确率（DA），并通过模型置信集（MCS）和Pesaran-Timmermann（PT）检验验证统计显著性。

#### LLMs for Time Series: an Application for Single Stocks and Statistical Arbitrage
- **Authors**: Sebastien Valeyre, Sofiane Aboura
- **Year**: 2024
- **Abstract**: 为解决金融时间序列预测中传统模型难以识别微弱市场无效性的问题，本文提出使用预训练和微调的LLM模型Chronos对美国个股残差收益进行日度预测，通过零样本和在线微调方式构建多空投资组合，实证表明Chronos能在不依赖金融数据预训练的情况下识别出可盈利的交易信号，实现高达4.21的夏普比率，虽仍低于专用模型但证明了LLM在噪声金融数据中提取Alpha的潜力。本文使用Amazon开发的Chronos-T5-Tiny模型（1100万参数），该模型预训练于非金融时间序列数据（如温度、能源等）。实验使用Guijarro-Ordonnez等（2022）发布的美国前550大股票的残差日收益数据（基于IPCA、PCA、Fama-French三因子模型剔除系统性风险）。方法分三部分：（1）零样本预测：输入过去100天残差收益（或其指数移动平均EMA，α=0.3为最优），由Chronos输出次日收益预测，按预测值排序构建50%多头、50%空头的等权投资组合，权重按排名与中位数距离调整并可按波动率逆向缩放；（2）在线微调：每日使用前100天数据对Chronos进行15步AdamW优化（τ=15），更新模型权重，持续微调并保持预测能力；（3）基准对比：与CNN-Transformer（169参数）、AutoARIMA和短期反转策略（STR，β=0.2/0.3）比较。结果显示，零样本Chronos（α=0.3）在PCA数据上夏普比率达3.17，微调后（α=0.3, τ=15）提升至3.97，加波动率调整后达4.21，虽仍低于CNN-Transformer（5.01），但显著优于AutoARIMA和STR，证明LLM可从噪声数据中提取非线性无效性，实现Alpha生成。

#### Natural Language Processing and Deep Learning for Bankruptcy Prediction: An End-to-End Architecture
- **Authors**: GIANFRANCO LOMBARDO, ANDREA BERTOGALLI, SERGIO CONSOLI, DIEGO REFORGIATO RECUPERO
- **Year**: 2023
- **Abstract**: 为解决企业破产预测中传统方法无法有效利用财务报告文本信息且难以适应概念漂移的问题，本文提出一种端到端的Transformer与多头LSTM融合架构，通过文本摘要模块提取SEC年报中的关键语义信息并结合财务时序数据进行联合建模，实现了87.5%的预测准确率和0.84的违约召回率，显著优于单一数据源模型。本文提出一种端到端的破产预测架构，包含两个核心模块：1) NLP模块：首先对SEC 10-K年报进行提取式摘要，包括三步操作——(a) 主题选择：基于词袋模型（BoW）和Loughran-McDonald金融词典训练逻辑回归分类器，筛选出对破产预测最有效的报告章节（Item 1、5、7）；(b) 文档减法：计算连续两年报告的文本相似度，剔除重复内容，保留新增或修改部分；(c) 情感分析：使用FinBERT模型对文本块进行情感分类，过滤中性语句，仅保留积极或消极内容；随后将处理后的文本分块输入预训练的DistilBERT模型生成文档嵌入向量。2) 财务时序模块：采用多头LSTM架构，为18个财务变量分别构建独立的LSTM子网络，处理过去三年的时序数据。两个模块分别预训练后，将其输出向量拼接，通过全连接层进行端到端微调，完成二分类破产预测任务。该架构避免了人工特征工程，实现了文本与财务数据的联合自适应学习，提升了模型对概念漂移的鲁棒性。

#### CAMEF: Causal-Augmented Multi-Modality Event-Driven Financial Forecasting by Integrating Time Series Patterns and Salient Macroeconomic Announcements
- **Authors**: Yang Zhang, Wenbo Yang, Jun Wang, Qiang Ma, Jie Xiong
- **Year**: 2025
- **Abstract**: 为解决现有金融预测方法忽视宏观事件与市场反应之间的因果关系及多模态信息融合不足的问题，本文提出CAMEF方法，通过整合高频率时间序列数据与宏观事件文本，利用LLM生成反事实事件增强因果学习，并设计多模态编码-融合-解码架构，显著提升对市场反应的预测准确性与鲁棒性。CAMEF方法包含四个核心组件：(1) 文本编码器使用RoBERTa提取宏观事件文本的语义特征，并通过三层线性网络投影为768维向量；(2) 时间序列编码器采用预训练的MOMENT模型提取5分钟高频金融数据特征，并通过多残差网络进一步优化为1024维向量；(3) 特征融合模块将文本与时间序列向量拼接后输入两层GELU激活的前馈网络，生成联合嵌入，再由GPT-2解码器进行自回归建模；(4) 引入基于LLM的反事实事件增强机制，通过结构化提示生成语义一致但情感极性不同的反事实事件脚本，并设计多样采样策略（同类型与异类型反事实事件），结合因果对比学习目标，使模型能区分真实事件与反事实事件，从而强化因果推理能力；最终通过MSE与MAE联合损失优化预测结果，实现对金融资产未来价格的高精度预测。

#### A Novel Wavelet based Generative Model for Time Series Prediction
- **Authors**: Chaofan Dai, Xiaoguang Yuan, Zongkai Tian, Xinyue Hu, Zhen Luan, Youchen Wang
- **Year**: 2024
- **Abstract**: 为解决股票市场非线性、非平稳时间序列预测精度低的问题，本文提出一种基于小波变换的生成对抗网络（Wavelet-GAN），通过小波分解将原始股价序列分解为多尺度频域分量，对各分量分别建立ARMA模型预测其系数，再将预测系数输入Wasserstein GAN框架进行生成对抗训练以重建高精度股价序列，实验表明该方法在预测准确率上显著优于GRU、LSTM和传统GAN模型。首先，对原始股票价格序列进行离散小波变换（DWT），采用DB2小波基函数将序列分解为不同尺度的低频和高频子序列；其次，对每个尺度的子序列分别建立ARMA模型，预测其小波系数；然后，将预测得到的小波系数作为输入，构建Wasserstein GAN（WGAN-GP）生成模型，其中生成器由两层GRU组成，判别器由三层一维卷积层构成，使用Wasserstein距离配合梯度惩罚项优化训练过程，以增强生成序列的真实性与稳定性；最后，将预测的小波系数通过小波重构算法还原为最终的股价预测序列。该方法融合了小波变换的多尺度特征提取能力与GAN的生成对抗机制，有效捕捉了金融时间序列的局部与全局动态特性。

#### Text2TimeSeries: Enhancing Financial Forecasting through Time Series Prediction Updates with Event-Driven Insights from Large Language Models
- **Authors**: Litton Jose Kurisinkel, Pruthwik Mishra, Yue Zhang
- **Year**: 2023
- **Abstract**: 为解决金融时序预测中传统模型忽略事件驱动非数值因素导致预测不准确的问题，本文提出Text2TimeSeries方法，利用大语言模型（LLM）预测事件对股票价格的多步变化趋势（离散标签），并通过门控循环单元计算股票状态，生成价格放大/衰减值，动态更新时间序列模型的预测结果，从而在小盘、中盘和大盘股票上显著降低预测误差（RMSE和MAE）。本文方法包含三个核心步骤：(1) 为每只股票训练独立的多变量时间序列模型（如D-Linear或PatchTST），输入历史价格、纳斯达克指数和美元汇率，预测未来n天的价格趋势；(2) 使用微调的T5模型（Base/Large/3B）作为LLM，输入股票代号和新闻事件文本，输出未来n天的离散价格变化标签（INC/DEC及其幅度）；(3) 将LLM输出的标签输入GRU网络，结合股票嵌入向量计算每个时间步的股票状态，再通过线性层预测价格变化方向的概率分布（增加、减少、不变），计算期望放大/衰减值（As），最后将该值与时间序列模型预测结果进行线性融合，得到最终更新后的价格预测。整个系统以MSE为损失函数进行端到端训练，其中时间序列模型在微调阶段被冻结，仅更新融合模块。

#### LLM4FTS: Enhancing Large Language Models for Financial Time Series Prediction
- **Authors**: Renjun Jia, Zian Liu, Peng Zhu, Dawei Cheng, Yuqi Liang
- **Year**: 2024
- **Abstract**: 为解决金融时间序列中低信噪比与多尺度模式难以建模的问题，本文提出LLM4FTS框架，通过基于DTW的K-means++聚类识别尺度不变模式、自适应分段策略保留模式完整性、动态小波卷积模块实现多尺度时频特征提取，结合两阶段预训练与微调，在四个真实金融市场数据集上实现了超越现有SOTA方法的股票收益预测精度与风险调整收益，并成功部署于实盘交易系统获得持续超额收益。LLM4FTS框架包含三个核心模块：(1) 离线尺度不变模式识别：使用DTW距离度量对金融时间序列进行不同长度的分段，通过K-means++聚类（初始质心采用远距离选择策略）识别具有相似形状但长度各异的模式，生成最优分段点集；(2) 可学习分段策略：基于聚类结果，在预训练阶段采用动态分段机制，将历史序列按最优分段点与滑动窗口结合生成可变长度的patch序列，作为LLM（如GPT-2）的输入，进行next-patch预测的自监督预训练；(3) 动态小波卷积模块：在微调阶段，使用Daubechies小波的低通和高通滤波器初始化卷积层权重，并在训练中动态正交化滤波器以保持小波变换的正交性，实现多分辨率时频特征提取；模型采用两阶段训练：第一阶段在多市场数据上预训练，第二阶段在目标市场微调；最终预测任务建模为排序问题，损失函数结合点对点回归损失与成对排序损失（ReLU修正的排序误差项），优化股票收益排序与绝对值预测精度。

### 金融时序数据插补

#### Handling missing data in Burundian sovereign bond market
- **Authors**: Irene Irakoze, Rédempteur Ntawiratsa, David Niyukuri
- **Year**: 2024
- **Abstract**: 为解决布隆迪主权债券市场因数据缺失导致收益率曲线构建困难的问题，本文提出使用线性回归和前值插补方法对缺失的债券价格、收益率和票息率进行填补，通过模拟缺失数据并评估各方法的平均绝对误差及正态性检验，发现线性回归方法在误差分布接近正态分布和预测能力方面表现最优，显著提升了收益率曲线的准确性与市场透明度。本文首先基于布隆迪央行数据（65%完整率）构建包含35%缺失值的模拟数据集，涵盖债券价格、收益率和票息率；随后采用六种插补方法：前值填充、后值填充、K近邻（KNN）、MICE、随机森林插补（missForest）和线性回归（利用历史同期限数据）；对每种方法，计算插补后的平均绝对误差（MAE），并通过Shapiro-Wilk正态性检验评估误差分布的正态性（p>0.05为符合正态）；最终比较各方法在三个变量上的MAE分布和p值，发现线性回归和前值填充在误差正态性和准确性上表现最佳，其中线性回归因兼具预测能力和正态分布特性被推荐为最优方法。

#### Missing value imputation and the effect of feature normalisation on financial distress prediction
- **Authors**: Kuen-Liang Sue, Chih-Fong Tsai, Hau-Min Tsau
- **Year**: 2024
- **Abstract**: 为解决财务困境预测中缺失值插补和特征归一化对模型性能的影响问题，本文比较了KNN、随机森林、MICE和深度神经网络等多种插补方法，并评估了最小-最大归一化对不同分类器（SVM、RF、DNN）预测效果的影响，发现随机森林插补效果最优，且归一化显著提升SVM和DNN性能但对RF无显著增益。本文首先在9个财务困境数据集（含破产预测和信用评分数据）上模拟10%至50%的缺失值（MCAR机制），并使用SMOTE平衡类别；随后对比四种插补方法：均值/众数（基线）、KNN、MICE、随机森林（RF）和深度神经网络（DNN）；插补后，对数据进行最小-最大归一化（[0,1]），并分别训练SVM、RF和DNN三类分类器；使用AUC和II类错误率评估预测性能；实验发现：在高缺失率（≥40%）下，RF插补表现最佳；归一化对DNN和SVM显著提升性能（p<0.05），但对RF分类器无显著影响，故在使用RF分类器时可省略归一化步骤。

#### A Fast Non-Linear Coupled Tensor Completion Algorithm for Financial Data Integration and Imputation
- **Authors**: Dan Zhou, Ajim Uddin, Zuofeng Shang, Cheickna Sylla, Xinyuan Tao, Dantong Yu
- **Year**: 2023
- **Abstract**: 为解决金融数据中高稀疏张量的缺失值插补问题，本文提出了一种名为RegTensor的正则化非线性耦合张量完成算法，通过引入多层感知机（MLP）建模嵌入向量间的非线性交互，并结合正交正则化抑制过拟合与特征冗余，同时耦合辅助张量增强嵌入学习，实现在债券特征和分析师盈利预测等金融数据集上显著优于线性和现有非线性模型的插补精度（提升2%-52%）。本文提出RegTensor方法，其核心包括四个步骤：（1）为每个张量模式（如时间、公司、债券）学习低维嵌入矩阵（U, V, W）；（2）使用多层感知机（MLP）替代传统CP分解中的线性点积，对嵌入向量的拼接进行非线性重建，以捕捉复杂交互关系；（3）引入耦合张量分解机制，将目标张量（如EPS预测）与辅助张量（如公司基本面）共享部分模式（如季度、公司），通过联合优化两个张量的重建误差增强嵌入质量；（4）在目标函数中加入正交正则化项（L_{2,2}范数约束嵌入矩阵的列向量正交），以消除特征冗余、降低协方差、抑制过拟合；最终采用随机梯度下降（SGD）仅在观测值索引上优化，实现高效可扩展的训练。该方法在多个金融数据集上验证，显著提升了插补准确率并保持较低计算开销。

#### Обработка пропусков в рыночных данных на примере задачи оценки кривой доходностей облигаций
- **Authors**: M. S. Makushkin, V. A. Lapshin
- **Year**: 2023
- **Abstract**: 为解决新兴市场债券市场中收益曲线估计因市场数据缺失而导致的偏差问题，本文比较了删除缺失值与三种插补方法（最后观察值前推、卡尔曼滤波、EM算法）对模型估计质量的影响，发现对于敏感模型（如bootstrap）插补能显著提升估计精度，而对于低敏感模型（如Nelson-Siegel）效果微弱，且不同插补方法间效果相近，从而提出根据模型敏感性选择是否插补的实践建议。本文基于俄罗斯联邦债券（ОФЗ）2012–2015年的市场数据，首先将收盘价转换为票息收益率以处理pull-to-par效应并满足正态性假设；随后对收益率序列中的缺失值（平均缺失率10%）采用四种策略处理：（1）直接删除缺失值（基准）；（2）最后观察值前推（LOCF）；（3）卡尔曼滤波（利用历史动态）；（4）EM算法（利用收益率序列的协方差结构，通过平均收益率简化高维协方差估计）。插补后，将收益率转换回债券价格，再分别用Nelson-Siegel参数化模型和bootstrap非参数方法估计零息收益曲线。采用交叉验证计算模型在真实观测值上的平均绝对误差，比较不同插补策略下误差降低的天数比例。结果表明：对于bootstrap方法，插补使65%以上交易日的估计精度显著提升（95%置信水平），而Nelson-Siegel模型几乎无改善；且三种插补方法效果无显著差异，表明简单方法（如LOCF）即可达到复杂方法的性能。

#### A two-stage case-based reasoning driven classification paradigm for financial distress prediction with missing and imbalanced data
- **Authors**: Lean Yu, Mengxin Li, Xiaojun Liu
- **Year**: 2023
- **Abstract**: 为解决金融困境预测中缺失数据和样本不平衡带来的预测性能下降问题，本文提出一种两阶段基于案例推理（CBR）的分类范式，第一阶段采用混合加权CBR方法插补缺失值，第二阶段构建LVQ-CBR分类器通过聚类增强少数类样本学习，显著提升了缺失与不平衡数据下的整体预测准确率和少数类识别能力。本文提出一种两阶段CBR驱动的分类范式。第一阶段为CBR驱动的缺失数据插补：首先对数据进行Min-Max归一化；对每个缺失特征，以其余特征构建目标案例和基础案例库；使用曼哈顿距离和欧氏距离双指标检索最相似的三个基础案例；根据距离倒数计算权重，加权平均各相似案例的特征值，分别得到两种距离下的插补值，最终取其平均作为插补结果；插补后的案例被保留至案例库以实现增量学习。第二阶段为LVQ-CBR分类预测：先对插补后的数据进行PCA降维以消除多重共线性并减少特征维度；将训练集样本通过LVQ算法聚类，利用类标签引导原型向量迭代更新，形成多个类别子案例库；对测试样本，先计算其与各原型向量的距离，定位最近的子案例库，再在该子库内用欧氏距离检索七个最相似案例，通过多数投票决定其分类标签；预测后的测试案例被保留至对应子库中以增强模型学习能力。该范式在七组中国上市公司数据集上验证，显著优于传统插补与不平衡处理方法的组合，在高缺失率（50%）和高不平衡率（4:1）下仍保持最优整体预测性能。

#### NMTucker: Non-linear Matryoshka Tucker Decomposition for Financial Time Series Imputation
- **Authors**: Uras Varolgunes, Dan Zhou, Dantong Yu, Ajim Uddin
- **Year**: 2023
- **Abstract**: 为解决金融时间序列中高稀疏性数据的缺失值插补问题，本文提出NMTucker方法，通过递归分解Tucker核心张量并引入多层非线性激活函数模拟复杂非线性交互，显著降低过拟合并提升插补精度，在多个真实金融数据集上比现有模型降低最高53.91%的RMSE。NMTucker方法通过构建多层神经网络架构实现非线性Tucker张量分解：首先，将金融数据建模为三阶张量（时间、公司、属性）；其次，在每一层中，使用嵌入矩阵（U, V, W）和核心张量（G）进行元素级Tucker分解，并在每次张量-矩阵乘法后引入非线性激活函数σ，形成非线性Tucker操作；接着，采用递归Matryoshka式结构，将上层核心张量G^(l)通过下层的非线性Tucker操作分解为更小的核心张量G^(l+1)，从而减少可训练参数并隐式正则化模型复杂度；所有层共享相同网络结构，仅第一层的核心张量和所有嵌入矩阵为可训练参数，其余核心张量在前向传播中动态计算；训练时采用随机梯度下降（SGD）优化均方误差（MSE）损失，仅对观测值进行反向传播；最终通过多层递归分解实现高维稀疏张量的高效、非线性插补，显著提升泛化能力。

#### A Fast Non-Linear Coupled Tensor Completion Algorithm for Financial Data Integration and Imputation
- **Authors**: Dan Zhou, Ajim Uddin, Zuofeng Shang, Cheickna Sylla, Xinyuan Tao, Dantong Yu
- **Year**: 2023
- **Abstract**: 为解决金融数据中高稀疏性张量的缺失值插补问题，本文提出了一种名为RegTensor的快速非线性耦合张量补全算法，通过引入多层感知机（MLP）建模嵌入向量间的非线性交互、正交正则化抑制嵌入冗余与过拟合，并联合多个相关张量进行协同因子分解，显著提升了插补精度，在债券特征和分析师盈利预测等数据集上比线性模型提升40%-74%，比现有非线性模型提升2%-52%。RegTensor方法包含四个核心步骤：(1) 构建嵌入学习模块，为张量每个模式（如时间、公司、债券）学习低维潜在嵌入矩阵U、V、W；(2) 使用多层感知机（MLP）替代传统CP分解中的线性点积，对嵌入向量的拼接结果进行非线性重建，预测缺失值；(3) 引入耦合张量分解机制，将目标张量（如EPS预测）与辅助张量（如公司基本面）共享部分模式（如季度、公司），通过联合优化两者的重建误差实现信息互补；(4) 添加正交正则化项，强制嵌入矩阵的列向量相互正交，消除特征冗余与共线性，降低过拟合风险，最终采用随机梯度下降（SGD）对所有参数进行端到端优化，仅在观测值上计算损失，实现高效可扩展训练。

#### Miguel C. Herculano\* and Punnoose Jacob
- **Authors**: Miguel C. Herculano, Punnoose Jacob
- **Year**: 2023
- **Abstract**: 为解决高频率金融数据中大量缺失值导致金融状况指数（FCI）构建不准确的问题，本文提出一种结合概率主成分分析（PPCA）与贝叶斯因子增强VAR模型的两步估计方法，通过在不完整数据面板上初始化因子并使用卡尔曼滤波与平滑技术动态估计时变参数，显著提升了FCI在样本内拟合和样本外预测中的准确性与稳定性，尤其在高频（周度）数据下优于传统方法。本文方法包含三个核心步骤：第一步，使用概率主成分分析（PPCA）或变分贝叶斯PCA（VBPCA）对包含缺失值的22个金融变量组成的非平衡面板进行因子初始化，避免传统PCA因设缺失值为零导致的过拟合问题；第二步，将初始化的因子作为观测变量，结合4个宏观变量，构建时变参数因子增强VAR（TVP-FAVAR）模型，利用卡尔曼滤波估计时变载荷系数与状态方程参数，缺失观测对应卡尔曼增益设为零以保持状态不变；第三步，基于第二步估计的参数，使用卡尔曼平滑器对因子进行重新估计，最终输出平滑后的金融状况指数（FCI）。该方法通过概率建模处理缺失值，引入时变波动率与参数动态性，并利用贝叶斯框架实现稳健估计，在高频数据缺失率高达62%的情况下仍能生成低噪声、高解释力的FCI。

### 金融时序数据增强

#### CFTNet: a robust credit card fraud detection model enhanced by counterfactual data augmentation
- **Authors**: Menglin Kong, Ruichen Li, Jia Wang, Xingquan Li, Shengzhong Jin, Wanying Xie, Muzhou Hou, Cong Cao
- **Year**: 2024
- **Abstract**: 为解决信用卡欺诈检测中因数据极度不平衡和模型依赖虚假相关性导致的鲁棒性差与召回率低问题，本文提出CFTNet模型，通过强化学习生成反事实样本（CFs）并构建三元组网络，利用风险表征与混淆表征的解耦学习增强因果建模能力，显著提升了检测准确率与模型鲁棒性。CFTNet方法包含两个核心阶段：第一阶段，将最优反事实样本（CFs）的生成建模为马尔可夫决策过程（MDP），采用P-DQN强化学习算法，在离散-连续混合动作空间中优化策略，通过最小化扰动使欺诈样本（正样本）的预测标签翻转，生成高质量的反事实样本；第二阶段，构建三元组网络结构，输入包括正样本、其对应的CFs和随机负样本，通过特征编码器提取嵌入表示，引入分离模块（Sep）将表征分解为风险表征（X）和混淆表征（Z），并基于对比学习设计相似性正则项ℓ_sim，约束正样本与CFs在Z上相似、在X上相异，同时引入实例加权分类损失ℓ_wcls，根据Z的相似性动态加权，强化对因果特征的学习。最终联合优化总损失ℓ_total = ℓ_wcls + αℓ_sim + Ω，使模型在推理阶段仅依赖X进行预测，有效消除Z引入的虚假相关性，提升模型在不平衡数据下的泛化能力与因果解释性。

#### RESEARCH ARTICLE
- **Authors**: Aya Salama Abdelhady, Nadia Dahmani, Lobna M. AbouEl-Magd, Ashraf Darwish, Aboul Ella Hassanien
- **Year**: 2024
- **Abstract**: 为解决全球绿色金融数据稀缺和非平稳性导致的预测精度不足问题，本文提出一种基于条件生成对抗网络（CT-GAN）数据增强与非线性自回归神经网络（NAR-NN）预测的混合模型，首先通过ADF检验确认数据非平稳性，继而使用CT-GAN生成高质量合成数据以扩充训练集，最后用NAR-NN进行时序预测，实现在欧洲、亚洲及其他地区分别达到98.8%、96.6%和99%的R²预测准确率，显著优于未增强的基线模型。首先，收集来自40个国家的绿色金融时间序列数据，并按大洲（欧洲、亚洲、其他地区）分组；其次，对各组数据执行Augmented Dickey-Fuller（ADF）检验，确认其非平稳性，从而选择适合非平稳序列的NAR-NN模型；接着，采用条件生成对抗网络（CT-GAN）对每组数据进行增强：生成器学习真实数据分布并生成合成样本，判别器区分真实与合成数据，通过最小化对抗损失函数迭代优化，直至生成数据逼真；增强后的数据用于训练非线性自回归神经网络（NAR-NN），该网络以过去20期绿色金融值为输入，通过含20个神经元的单隐藏层和tanh/logsig激活函数，学习非线性时序映射关系；最后，使用MSE、RMSE和R²评估预测性能，并与未增强的NAR-NN和NARX-NN模型进行对比，验证CT-GAN增强对预测精度的显著提升。

#### Article Enhancing Financial Time Series Prediction with Quantum-Enhanced Synthetic Data Generation: A Case Study on the S&P 500 Using a Quantum Wasserstein Generative Adversarial Network Approach with a Gradient Penalty
- **Authors**: Filippo Orlandi, Enrico Barbierato, Alice Gatti
- **Year**: 2024
- **Abstract**: 为解决金融时间序列中极端事件样本稀缺导致预测模型性能不足的问题，本文提出一种量子增强的Wasserstein生成对抗网络（QWGAN-GP）方法，通过量子生成器与经典判别器协同生成与S&P 500对数收益率统计特性高度一致的合成数据，并结合LSTM模型验证其在提升预测准确性（尤其是极端事件预测）方面的有效性，实验表明融合合成数据的模型显著优于仅使用真实数据的基准模型。本文提出一种量子Wasserstein生成对抗网络带梯度惩罚（QWGAN-GP）模型，用于生成S&P 500指数的合成金融时间序列。首先，采集2000–2008年S&P 500日收盘价，计算对数收益率，并通过Z-score标准化和Lambert-W逆变换使其分布趋近高斯分布；接着，采用滚动窗口（窗口长度10，步长2）将序列划分为子序列作为训练样本。量子生成器由5个量子比特构成，包含Hadamard初始化、三层Rx/Ry旋转门与CNOT纠缠门结构，输入为随机噪声（0–2π角度），通过Pauli-X和Pauli-Z测量输出量子态并转换为经典数据；经典判别器为CNN结构，包含三层卷积层（64、128、128滤波器）、LeakyReLU激活、Flatten、32神经元全连接层、20% Dropout和单神经元输出层，采用Wasserstein距离作为损失函数并引入梯度惩罚以稳定训练。模型训练2000轮，批量大小为20。生成的合成数据经Wasserstein距离、DTW、熵、QQ图、PDF/CDF、ACF和散点图等多维度评估，结果显示其与真实数据分布高度相似（Wasserstein距离≈0.00086，DTW距离=1.954），随后将合成数据与真实数据混合训练LSTM预测模型，实验表明该混合训练策略显著提升了对S&P 500趋势与极端事件的预测准确率。

#### Market-GAN: Adding Control to Financial Market Data Generation with Semantic Context
- **Authors**: Haochong Xia, Shuo Sun, Xinrun Wang, Bo An
- **Year**: 2023
- **Abstract**: 为解决金融数据缺乏语义上下文控制、生成数据保真度低且难以用于下游任务的问题，本文提出Market-GAN，一种融合上下文建模、自编码器与对抗训练的生成模型，通过市场动态建模提取股票 ticker、历史状态和市场动态三类语义上下文，并采用两阶段训练（预训练+对抗训练）结合C-TimesBlock架构，实现高保真、强对齐、符合市场事实的金融时序数据生成，显著提升下游预测任务性能。Market-GAN 由三部分构成：(1) 构建Contextual Market Dataset，通过市场动态建模算法（MDM）对价格序列进行去噪、分段、线性回归斜率分类与聚类，提取三类语义上下文：长期股票代号（l）、中期市场动态（d，分熊市、盘整、牛市三类）、短期历史波动（H）；(2) 设计混合架构，包括数据变换层（将OHLC重参数化为非负偏差形式以保留市场约束）、C-TimesBlock（融合RNN与Inception模块以捕获多尺度时序依赖并缓解模式崩溃）、自编码器（e和r）、上下文监督器（s_d, s_l, es_d, es_l）和判别器dis；(3) 采用两阶段训练：预训练阶段利用自编码器重构损失和上下文分类损失初始化生成器，使生成器学习上下文对齐的潜在表示；对抗训练阶段联合优化生成器（最小化对抗损失、上下文监督损失、重构损失）与判别器（最大化真实与生成数据区分能力），最终通过逆变换层输出符合OHLC约束的市场特征。评估采用四维度指标：上下文对齐（CE损失）、保真度（判别器准确率偏离50%的程度）、市场事实（OHLC约束违反率）和下游任务可用性（SMAPE预测误差），在DJI 29只股票2000–2023年数据上显著优于TimeGAN、SigCWGAN等基线模型。

#### Article
- **Authors**: Francesco Bruni Prenestino, Enrico Barbierato, Alice Gatti
- **Year**: 2025
- **Abstract**: 为解决金融时序数据稀缺与隐私保护下合成数据保真度低的问题，本文提出一种融合变分自编码器（VAE）与马尔可夫链蒙特卡洛（MCMC）采样的混合架构，通过GRU捕获长期时序依赖，并利用MCMC在潜在空间中生成相关样本序列，显著提升了合成数据在统计特性、时序模式和缺失数据鲁棒性方面的保真度。该方法首先使用GRU作为编码器和解码器构建VAE框架，编码器将输入金融时序数据（如Google、Tesla、Nestlé股票的开盘价、最高价、最低价、收盘价、成交量和调整收盘价）压缩为潜在空间的均值与对数方差，通过重参数化采样生成初始潜在向量；解码器则通过GRU层和TimeDistributed层重建原始序列。随后，引入MCMC采样机制替代传统单次随机采样：利用GRUCell建模潜在空间中连续样本间的马尔可夫转移依赖，以当前潜在状态为隐藏状态，逐步生成一系列相关联的潜在向量，从而更全面地探索潜在分布的多样性；每个生成的潜在向量输入解码器得到合成时序数据。训练时采用重构损失（均方误差）与KL散度损失的加权组合，并引入掩码机制处理缺失值（将NaN替换为0并屏蔽其损失计算）。实验验证表明，该方法在判别分数和预测分数上优于VAE-Conv、VAE-GRU及TimeGAN等基线模型，能有效保留原始数据的统计分布与时间依赖性，且在数据缺失场景下仍保持高保真度。

#### Can GANs Learn the Stylized Facts of Financial Time Series?
- **Authors**: Sohyeon Kwon, Yongjae Lee
- **Year**: 2024
- **Abstract**: 为解决传统金融时序模拟方法难以捕捉随机游走、均值回归、跳跃和时变波动率等stylized facts的问题，本文通过实验验证了多种GAN生成器架构（MLP、MLP-CNN、LSTM、GRU、TCN）在模拟五种随机过程（BM、GBM、OU、JD、HT）中的表现，发现GAN能较好学习简单随机过程的分布，但对复杂时序结构（如均值回归和多变量依赖）建模能力有限，且性能高度依赖生成器架构选择，表明需谨慎设计和验证GAN模型以有效增强金融时序数据。本文采用生成对抗网络（GAN）框架，通过对抗训练生成金融时序数据。首先，构建五种代表性随机过程（布朗运动、几何布朗运动、Ornstein-Uhlenbeck过程、跳跃扩散模型、Heston模型）的合成数据集，涵盖单变量和双变量场景。其次，测试五种生成器架构（MLP、MLP-CNN、LSTM、GRU、TCN），固定判别器为MLP-CNN。为优化超参数，设计基于Jensen-Shannon散度的复合目标函数，结合对数收益率分布和最终值分布的散度，对单变量数据；对多变量数据，进一步加入变量间联合分布的KL散度和边际分布的JS散度。通过对比生成数据与真实数据的统计特征（如均值、标准差、跳跃次数、波动率分布、相关系数等），评估各架构对stylized facts的捕捉能力。实验发现，MLP和MLP-CNN在多数任务中表现最佳，但所有架构均难以精确复现均值回归速度和多变量依赖结构，尤其在长序列和高波动场景下性能下降显著。

#### Generation of synthetic financial time series by diffusion models
- **Authors**: Tomonori Takahashi, Takayuki Mizuno
- **Year**: 2024
- **Abstract**: 为解决现有生成模型难以同时复现金融时间序列多重统计特征（如厚尾、波动聚类、日内季节性和跨序列相关性）的问题，本文提出一种结合小波变换与去噪扩散概率模型（DDPM）的方法，通过将股票价格、买卖价差和交易量三类时间序列转换为RGB彩色图像，利用DDPM学习图像中的多尺度结构并生成合成图像，再经逆小波变换还原为时间序列，成功实现了对所有关键stylized facts的高精度复现。首先，对原始金融时间序列（股票价格对数收益、买卖价差、交易量）进行预处理：通过镜像扩展使序列长度为2^n，计算对数收益，对交易量应用arcsinh变换，再进行幂变换和标准化，并通过winsorization处理异常值；其次，对每类预处理后的时间序列应用离散小波变换（使用Haar小波），将各级小波系数按层级排列为灰度图像的像素行，其中第k级系数填充图像第k行，形成三个独立的灰度图像；接着，将这三个灰度图像分别作为RGB三通道合成一张彩色图像，每张图像代表一天内三类时间序列的联合结构；然后，使用UNet架构的DDPM模型在这些彩色图像上进行训练，通过逐步加噪与去噪过程学习金融时间序列的复杂分布；训练完成后，DDPM生成新的合成彩色图像，再通过逆小波变换将图像还原为三类时间序列数据；最终，通过对比真实数据，验证生成序列在厚尾分布、自相关慢衰减（波动聚类）、日内U型季节性模式以及三序列间交叉相关性等方面均与真实市场数据高度一致，显著优于TimeGAN、QuantGAN及无小波的DDPM方法。

#### Article Enhancing Portfolio Performance through Financial Time-Series Decomposition-Based Variational Encoder-Decoder Data Augmentation
- **Authors**: Bayartsetseg Kalina, Ju-Hong Lee, Kwang-Tek Na
- **Year**: 2024
- **Abstract**: 为解决金融时间序列数据不足和历史数据中不确定性缺失导致的投资组合模型性能不佳的问题，本文提出了一种基于金融时间序列分解的变分编码器-解码器（FED）数据增强方法，通过将时间序列分解为趋势、离散度和残差三个潜在组件并重建具有历史不确定性的合成数据，进而构建FED2Port强化学习投资组合模型，显著提升了投资组合的收益风险比和鲁棒性。首先，FED方法通过三个独立的编码器分别建模金融时间序列的潜在趋势（ν_t）、离散度（τ_t）和残差（ξ_t）分量，每个分量均服从多元正态分布；利用多元正态分布的乘积性质，将三者融合为联合潜在变量h_t = ν_t × τ_t × ξ_t，并通过解码器重构原始序列；通过最大化包含KL散度项和重构似然项的组合ELBO损失函数训练FED模型，从而生成具有真实统计特性和历史不确定性的合成金融时间序列。其次，基于FED生成的数据构建FED2Port强化学习环境：状态为上一时刻的资产组合收益，动作是高风险与低风险资产的权重分配（满足和为1），奖励函数采用市场自适应比率（考虑牛市/熊市特征），通过策略网络π_ω优化长期期望奖励，最终实现对投资组合权重的动态调整，提升风险调整后收益。

#### Generative Adversarial Networks: A Systematic Review of Characteristics, Applications, and Challenges in Financial Data Generation and Market Modeling: 2019-2024
- **Authors**: D. Wilson, A. Azmani
- **Year**: 2025
- **Abstract**: 为解决金融数据隐私受限、稀缺及传统模型难以捕捉复杂市场动态的问题，本文通过系统综述2019–2024年30篇文献，分析各类GAN架构（如CTGAN、WGAN、TGAN、TTGAN等）在生成高保真合成金融数据中的应用，发现其能有效增强数据隐私性并提升股票预测、风险评估、组合优化等任务性能，但仍面临模式坍塌、训练不稳定和缺乏统一评估标准等挑战。本文采用系统性文献综述方法，遵循PRISMA框架，从IEEE Xplore、Web of Science、Scopus和arXiv四大数据库中筛选2019–2024年间30篇关于GAN在金融数据生成与市场建模中应用的高质量论文。首先，通过关键词组合（"Generative Adversarial Networks" OR "GAN" AND "financial data" OR "synthetic financial data" AND "data generation"）进行文献检索；其次，依据标题、摘要和内容进行筛选，排除无关或低质量文献；接着，对入选论文进行分类分析，系统梳理16种主流GAN变体（如Vanilla GAN、cGAN、WGAN、TGAN、CTGAN、StyleGAN、PATE-GAN、SigWGAN等）在金融场景中的适用性与性能表现；然后，归纳其在股票价格预测、算法交易、组合优化、风险评估、欺诈检测等六大金融任务中的具体应用与效果；最后，识别当前研究中的主要挑战，包括训练不稳定性、模式坍塌、市场动态建模不足、评估指标缺失、计算资源需求高及缺乏宏观经济学整合，并提出未来研究方向，如结合强化学习、引入极端事件建模、开发在线学习机制、融合市场博弈理论等，以推动GAN在金融领域的稳健应用。

#### Article
- **Authors**: César Vaca, Jesús-Ángel Román-Gallego, Verónica Barroso-García, Fernando Tejerina, Benjamín Sahelices
- **Year**: 2025
- **Abstract**: 为解决金融领域非结构化文本（如公司治理报告中的董事履历）数据稀缺导致深度学习模型性能受限的问题，本文提出了一种名为连接增强（Concatenation Augmentation, CA）的新数据增强方法，通过将原始文本样本串联并基于逻辑激活函数的逆变换对标签进行凸加性融合，生成语义连贯的新样本，显著提升了模型在低数据场景下的准确率（92.4%–99.7%）和鲁棒性。Concatenation Augmentation (CA) 方法通过以下步骤实现：1) 从训练集中随机选取两个样本 (x_i, y_i) 和 (x_j, y_j)，其中 x 为董事履历文本，y 为六维专业背景标签（每维为 [0,1] 区间内 0.1 间隔的数值）；2) 对文本进行串联操作：\tilde{x} = x_i \oplus x_j（仅在缺少句末标点时添加句号，无其他预处理）；3) 对标签进行非线性凸组合：\tilde{y} = \Phi(y_i, y_j) = \sigma(\sigma^{-1}(y_i) + \sigma^{-1}(y_j))，其中 \sigma 为逻辑函数，\sigma^{-1}(y) = \ln(y/(1-y)) 为其逆函数，该操作等价于在对数几率空间中进行加法后映射回概率空间；4) 将生成的新样本 (\tilde{x}, \tilde{y}) 加入训练集，用于训练 AWD-LSTM 或 Transformer 集成模型；5) 该方法无需外部模型、不依赖超参数调优、计算开销极低，且可直接集成于数据预处理流水线中，适用于任何满足凸加性假设的回归型 NLP 任务。

#### Generative Adversarial Networks applied to synthetic financial scenarios generation
- **Authors**: Matteo Rizzato, Julien Wallart, Christophe Geissler, Nicolas Morizet, Noureddine Boumlaik
- **Year**: 2023
- **Abstract**: 为解决金融领域中多变量时序数据在宏观情景约束下难以生成符合现实统计特性合成场景的问题，本文提出Jinkou算法，基于双向GAN（BiGAN）和条件GAN（cGAN）的耦合架构，先通过BiGAN生成宏观状态变量变化，再以这些变化为条件驱动cGAN生成金融工具特征的时序变化，最终通过MCMC采样实现情景条件生成，成功复现了金融市场的经典统计特征并实现了对能源和金融组合的高保真情景模拟。Jinkou算法由两个耦合的生成对抗网络组成：1）双向GAN（BiGAN）用于建模宏观状态变量（如通胀、油价、利率）的时序变化分布，其包含生成器和编码器，通过将高斯噪声映射为状态变量变化，并逆映射回潜在空间，学习状态变量的隐式分布；2）条件GAN（cGAN）用于在给定当前金融工具特征和未来状态变量变化的条件下，生成工具特定特征（如价格、市值、ESG得分）的增量变化，其输入为当前状态向量和状态变量变化向量，输出为特征变化量；训练时，BiGAN和cGAN分别使用历史数据中的状态变量变化和工具特征变化进行独立训练；在推理阶段，用户通过设定宏观情景（如通胀区间）作为超矩形约束，利用MCMC在BiGAN的潜在空间中采样满足该约束的状态变量变化，再将这些变化输入cGAN，结合初始金融工具状态，递归生成多步时序轨迹；最终通过笛卡尔积组合状态变量和工具特征的采样，构建完整投资组合的多轨迹模拟，实现对金融风险的条件化评估。

#### Bankruptcy Prediction: Data Augmentation, LLMs and the Need for Auditor's Opinion
- **Authors**: Andreas Sideras, Konstantinos Bougiatiotis, Elias Zavitsanos, Georgios Paliouras, George Vouros
- **Year**: 2024
- **Abstract**: 为解决企业破产预测中样本极度不平衡及单一数据源信息不足的问题，本文提出一种融合管理层讨论与分析（MD&A）和审计意见（AO）文本的多源数据增强方法，利用变分自编码器（VAE）生成合成破产样本，并采用晚期融合策略整合双源预测结果，显著提升了破产预测的F1分数和召回率，同时评估了大语言模型（LLM）在零样本预测和数据增强中的表现与局限性。首先，扩展ECL数据集，加入审计意见（AO）文本，构建多源数据集ECL+AO；其次，对MD&A和AO文本分别提取30K维tf-idf特征向量；接着，使用变分自编码器（VAE）对破产样本（正类）进行建模，生成7500个MD&A合成样本和20000个AO合成样本以缓解类别不平衡；然后，训练两个独立的逻辑回归（LR）分类器，分别以MD&A和AO的原始或增强文本为输入，预测破产概率；在测试阶段，采用晚期融合策略，对两个模型的预测概率取平均作为最终预测结果；此外，实验还探索了使用LLM（如Llama-3）进行零样本破产预测和基于关键词的文本生成数据增强，但发现LLM受提示词敏感、存在前瞻偏差，且生成质量不稳定，效果不如VAE；最终，通过对比实验验证了多源融合与VAE数据增强的组合方法在F1-score和R@100等指标上优于单源模型和早期融合方法。

#### Improving Anti-money Laundering via Fourier-Based Contrastive Learning
- **Authors**: Meihan Tong, Shuai Wang, Xinyu Chen, Jinsong Bei
- **Year**: 2024
- **Abstract**: 为解决现有深度学习反洗钱模型对数据扰动鲁棒性不足的问题，本文提出一种基于傅里叶变换的对比学习模型（FCLM），通过将交易数据从时域映射到频域生成高差异性增强视图，并利用对比学习使模型对原始交易及其增强视图保持预测一致性，从而显著提升检测鲁棒性与泛化能力，在合成与真实数据集上均超越七种先进基线方法。FCLM模型包含四个模块：1）数据增强模块：对原始交易数据进行嵌入编码（分类特征随机初始化嵌入，数值特征Z-score归一化后分桶嵌入），再通过快速傅里叶变换（FFT）将整个交易序列从时域转换到频域，生成高差异性的傅里叶增强视图；2）特征编码模块：使用8层Transformer编码器分别对原始交易和其傅里叶增强视图进行编码，输出对应的隐藏表示hi和hi^ω；3）对比预训练模块：通过对比损失L1最小化原始表示与增强表示之间的距离，同时最大化其与负样本表示的距离，迫使模型学习对扰动不变的鲁棒表征；4）反洗钱检测模块：在优化后的Transformer编码器基础上，构建多层感知机（MLP）分类器，输入为原始交易与增强视图的隐藏表示之和（hi + hi^ω），通过Softmax输出分类概率，并使用交叉熵损失L2进行监督训练，最终实现对洗钱交易的精准识别。

#### Tail-GAN:Learning to Simulate Tail Risk Scenarios∗
- **Authors**: Rama Cont, Mihai Cucuringu, Renyuan Xu, Chao Zhang
- **Year**: 2025
- **Abstract**: 为解决传统生成模型在金融场景模拟中无法准确捕捉尾部风险的问题，本文提出Tail-GAN方法，通过联合可 elicibility 性质设计基于VaR和ES的尾部敏感评分函数作为生成对抗网络的训练目标，使生成的多资产价格场景能精确保留基准交易策略的尾部风险特征，并在合成与真实市场数据上验证了其在尾部风险估计和泛化能力上的优越性。Tail-GAN方法的核心步骤包括：(1) 定义一组用户指定的基准交易策略，每个策略将高维价格轨迹映射为一维PnL分布，从而将多维分布学习问题降维为K个一维分布学习问题；(2) 利用VaR与ES的联合可 elicibility 性质，采用Acerbi-Szekely形式的严格一致评分函数S_α(v,e,x)作为损失函数，该函数能精确衡量生成分布与真实分布在尾部风险（VaR和ES）上的差异；(3) 构建生成器G与判别器D的对抗框架，生成器G学习生成价格场景，判别器D通过最小化评分函数S_α的期望值来评估生成场景与真实场景在尾部风险上的匹配程度，训练目标为min_G max_D E[S_α(v,e,X_real)] - E[S_α(v,e,X_fake)]；(4) 为提升可扩展性，结合主成分分析（PCA）对高维输入数据进行降维，仅在主要成分空间中训练生成器；(5) 在训练中优化H_1(v) = -W_α/2 v^2和H_2(e) = α/2 e^2，确保评分函数在尾部区域具有良好的凸性与收敛性，最终生成的场景能准确复现真实市场数据的重尾性、自相关性和跨资产依赖性，并在动态交易策略下显著优于传统GAN方法。

#### Stock Price Prediction with Heavy‑Tailed Distribution Time‑Series Generation Based on WGAN‑BiLSTM
- **Authors**: Ming Kang
- **Year**: 2024
- **Abstract**: 为解决新上市公司股票数据稀缺导致预测精度低的问题，本文提出WGAN-BiLSTM模型，利用WGAN生成符合真实数据重尾分布的增强样本，并结合BiLSTM双向提取时序特征进行预测，显著提升了小样本场景下的预测准确性。首先，使用WGAN生成与真实股票价格数据具有相似重尾分布的增强样本，WGAN的生成器以200维高斯噪声为输入，通过全连接层（FC）构建，采用RMSProp优化器，判别器通过FC层学习真实数据分布，并通过Kolmogorov-Smirnov (KS)检验和对数-对数图面积评估生成数据的重尾特性；其次，将生成的增强数据（扩展比例为30%、60%、100%）与原始数据合并，输入BiLSTM模型进行预测，BiLSTM包含三层隐藏层（100、150、300个神经元），使用Dropout防止过拟合，采用Adam优化器和均方误差（MSE）损失函数，双向结构同时捕捉历史与未来时间依赖信息；最后，通过MAE、RMSE和R²评估预测性能，实验表明在数据增强后，BiLSTM模型在两个小样本数据集上预测精度显著优于无增强模型及其他基线模型（如LSTM、GRU、SVR等），R²最高提升至0.973。

#### Controllable Financial Market Generation with Diffusion Guided Meta Agent
- **Authors**: Yu-Hao Huang, Chang Xu, Yang Liu, Weiqing Liu, Wu-Jun Li, Jiang Bian
- **Year**: 2023
- **Abstract**: 为解决金融市场上订单流生成缺乏可控性与高保真度的问题，本文提出Diffusion Guided meta Agent (DiGA)模型，通过条件扩散模型建模市场状态（如中价回报率和订单到达率）的时变分布，并结合具有金融经济先验的元代理按分布采样订单，实现了对市场场景（如收益、波动率）的精准控制与高保真订单流生成。DiGA模型由两个模块组成：1）元控制器（Meta Controller）：使用条件扩散模型（DDPM）学习市场状态（分钟级中价回报率和订单到达率）的时变分布，通过引入控制目标（如日收益、振幅、波动率）的编码器（离散或连续）和无分类器引导（classifier-free guidance）实现对生成过程的控制；2）订单生成器（Order Generator）：包含模拟交易所和元代理，元代理根据扩散模型输出的市场状态参数，遵循CARA效用函数优化决策，通过指数分布采样唤醒时间，结合基本面、技术面和噪声成分估算未来收益，推导需求函数并均匀采样订单价格与数量，最终生成订单流。训练时联合优化扩散模型的去噪损失与控制目标的均方误差，采样时通过引导尺度调整控制强度，最终生成的订单流在控制精度和金融统计特性（stylized facts）上均达到SOTA水平。

#### SimMix: Local similarity-aware data augmentation for time series
- **Authors**: Pin Liu, Yuxuan Guo, Pengpeng Chen, Zhijun Chen, Rui Wang, Yuzhu Wang, Bin Shi
- **Year**: 2023
- **Abstract**: 为解决时间序列分类任务中数据增强强度控制不当导致模型性能下降的问题，本文提出SimMix方法，通过动态时间规整（DTW）对同类样本进行局部相似性对齐，基于距离与长度复合指标动态选择最优对齐段，并采用无插值的点对点替换策略进行切片混合，从而精确控制增强强度，在10个真实数据集上显著超越现有方法，平均准确率提升3.5%以上。SimMix方法包含三个核心步骤：(1) 局部相似性匹配：使用DTW对同一类别的两个时间序列进行对齐，生成最优路径P，从中提取具有唯一映射关系的对齐段（即路径端点仅属于当前段的连续子序列）；(2) 动态选择对齐段：为每个对齐段计算距离指标ξ^d = |2/π·1/tan(∑σ_e(a_i,b_j)) - λ_th|（将距离映射至[0,1]并贴近预设阈值λ_th）和长度指标ξ^l = len(s_A) + len(s_B)，通过复合函数ζ(p,q)综合比较多个对齐段，优先选择距离适中且长度较大的段；(3) 非等长切片混合：对选中的对齐段，将其路径元素拆解为一对一或一对多映射关系，一对一映射直接替换对应点值，一对多映射取多个源点的均值替换目标点值，避免上下采样引入噪声或丢失语义信息，最终生成增强样本。该方法基于PAC理论证明了增强强度需适中，过强或过弱均损害泛化能力，从而实现可控、高质量的数据增强。

#### Generative-CNN for Pattern Recognition in Finance
- **Authors**: Jeevesh Natarajan, Wayne Wang, Yaqiao Jiang, Zeqi Zhang, Huanhui Ye, Lingxi Kuang
- **Year**: 2024
- **Abstract**: 为解决金融领域中K线模式识别因标注图像数据稀缺导致的卷积神经网络（CNN）性能受限问题，本文提出Generative-CNN方法，通过使用深度卷积生成对抗网络（DCGAN）基于少量真实K线图像生成大量合成图像，并将合成图像与真实图像结合训练Inception V3 CNN模型，从而显著提升K线模式分类准确率至80%以上，有效缓解了数据稀缺瓶颈。首先，手动收集并标注400张真实K线图像，分为两类：看涨反转（Inverse Head and Shoulders）和看跌反转（Head and Shoulders）；其次，使用其中70张图像训练一个DCGAN生成器，生成840张高质量合成K线图像，模拟真实模式的三峰/三谷结构；接着，将70张真实图像与840张合成图像合并，构建一个增强的混合数据集；然后，采用预训练的Inception V3模型作为分类器，冻结底层参数，替换顶层分类层以适配二分类任务，并使用二元交叉熵损失函数进行训练；训练过程中先固定基础层进行微调，再解冻部分层进行精细优化；最后，通过对比仅使用70张真实图像的基准模型（准确率约58%）与使用增强数据集的Generative-CNN模型（准确率约80%），验证了该方法在提升分类性能上的显著效果。

#### Decision-Aware Conditional GANs for Time Series Data
- **Authors**: He Sun, Zhun Deng, Hui Chen, David C. Parkes
- **Year**: 2023
- **Abstract**: 为解决金融时序数据稀缺导致决策相关量（如投资组合权重、协方差）估计不可靠的问题，本文提出决策感知条件生成对抗网络（DAT-CGAN），通过在Wasserstein GAN损失函数中引入多步决策相关量的多Wasserstein距离项，并结合重叠块采样和条件对齐机制，使生成数据不仅逼近原始时序数据，更精准模拟决策关键量，从而提升投资组合优化的仿真质量与训练稳定性。DAT-CGAN通过以下步骤实现决策感知的时序数据生成：1) 定义多步决策相关量（如资产收益的移动平均估计均值、协方差矩阵、投资组合权重），并将其作为生成目标；2) 构建多Wasserstein损失函数，同时最小化原始时序数据和每个决策相关量在不同前瞻步长k下的条件分布差异，权重按指数衰减（ω_k = λ_{j,k} = 0.8^k）；3) 采用重叠块采样机制，提升小样本下的数据利用率；4) 为生成器和判别器提供相同的条件信息x_t，避免判别器过强；5) 通过Kantorovich-Rubinstein对偶形式构建可优化的代理损失，判别器最大化真实与生成数据的Wasserstein差距，生成器最小化该差距；6) 训练中交替更新判别器（s_D=1次）和生成器（s_G=5次），使用梯度裁剪和小学习率（α=1e-5）确保稳定收敛；7) 在投资组合应用中，决策相关量包括资产收益、估计的精度矩阵和最终的投资组合权重，生成器需同步模拟这些量，从而支持高保真风险评估与策略回测。

#### Time Series Generation with GANs for Momentum Effect Simulation on Moscow Stock Exchange
- **Authors**: Maksim Kazadaev, Vitaliy Pozdnyakov, Ilya Makarov
- **Year**: 2023
- **Abstract**: 为解决金融时间序列数据稀缺导致的交易策略过拟合问题，本文提出基于时间卷积网络（TCN）的生成对抗网络（GAN）方法，通过生成具有真实统计特性的多维股票对数收益率序列来增强训练数据，从而支持动量效应策略的回测与超参数调优，实验表明该方法能有效模拟股票间相关性但未能充分捕捉动量效应的复杂依赖关系。本文提出一种基于时间卷积网络（TCN）的生成对抗网络（GAN）架构，用于生成莫斯科证券交易所五只股票的多维日对数收益率序列。生成器与判别器均采用TCN结构，包含四层膨胀卷积（膨胀率分别为1、2、4、8），输入为126天窗口长度的随机噪声向量（维度为15），输出为与真实数据相同维度的收益率序列。训练过程中采用权重裁剪（[-0.01, 0.01]）和交替训练策略（判别器每批训练，生成器隔批训练）以提升稳定性，使用RMSprop优化器，学习率0.0002，训练800轮。生成的序列通过归一化使其均值与标准差与真实数据匹配，以保留统计特性。生成的数据用于动量策略的回测与超参数调优，动量指标定义为过去n_start到n_finish天的累计收益率，策略按该指标对股票排序并按排名比例分配资金，评估指标为夏普比率。实验对比真实数据与生成数据在策略表现上的差异，发现GAN能准确模拟股票间相关性，但在捕捉动量效应的弱依赖关系上表现不足，导致超参数调优结果与真实数据不一致。

#### Regime-Specific Quant Generative Adversarial Network: A Conditional Generative Adversarial Network for Regime-Specific Deepfakes of Financial Time Series
- **Authors**: Andrew Huang, Matloob Khushi, Basem Suleiman
- **Year**: 2023
- **Abstract**: 为解决金融时间序列在市场危机等罕见 regimes 下数据稀缺和非平稳性导致的风险评估困难问题，本文提出了一种名为RSQGAN的条件生成对抗网络，通过结构断点算法（贪婪高斯分割）识别市场 regimes 并将其作为条件标签，利用时序卷积网络（TCN）生成符合特定 regimes 特征的合成资产回报数据，并引入Z-裁剪超参数控制合成数据保真度与多样性，实验证明其在危机 regimes 下的合成数据质量显著优于无条件GAN模型。1. 使用贪婪高斯分割（GGS）算法对金融时间序列进行结构断点检测，自动划分出非重叠的市场 regimes（如危机与非危机），并为每个时间片段生成 one-hot 编码的 regime 类别标签；2. 构建条件生成对抗网络（RSQGAN），其生成器和判别器均基于时序卷积网络（TCN）架构，TCN 采用膨胀因果卷积、残差连接和跳过连接以捕获长期依赖和多尺度时序特征；3. 将 regime 类别标签作为条件输入，与噪声向量 z 一起输入生成器，使生成器学习生成特定 regime 下的资产回报序列，同时判别器判断输入序列的真实性并识别其所属 regime；4. 引入 Z-裁剪（Z-clipping）机制，对噪声向量 z 的每个维度进行截断（|z| ≥ z_clip 时重采样），以控制生成数据的多样性与真实数据的保真度之间的权衡；5. 采用 WGAN-GP 损失函数（带梯度惩罚）提升训练稳定性，避免模式坍塌；6. 使用四个专为危机环境设计的路径依赖评估指标（如最大回撤、波动率聚类、尾部风险等）对生成数据质量进行量化评估，验证 RSQGAN 在模拟危机 regimes 行为上的优越性；7. 通过参数共享机制，利用非危机 regimes 的大量数据辅助学习通用时序特征，提升对稀有危机 regimes 的生成能力。

#### Graph-Based Inductive Learning for Credit Risk Prediction with Imbalance Mitigation
- **Authors**: Sogand Pourkhoshgoftar, Asadollah Shahbahrami, Nima Esmi
- **Year**: 2025
- **Abstract**: 为解决信用风险预测中极端类别不平衡和非线性借款人关系建模不足的问题，本文提出一种结合条件表格生成对抗网络（CTGAN）与图采样与聚合图神经网络（GraphSAGE）的混合方法，先通过CTGAN生成合成违约样本以平衡数据分布，再构建借款人相似性图并利用GraphSAGE进行归纳式关系学习，最终在GMSC和GC数据集上显著提升了准确率、F1分数和AUC指标，同时通过SHAP增强模型可解释性。首先，使用条件表格生成对抗网络（CTGAN）对少数类（违约）样本进行合成增强，通过模式感知归一化和条件向量引导生成与真实数据统计特性一致的合成违约实例；其次，基于K近邻（KNN）构建借款人图结构，其中每个借款人作为节点，边表示特征空间中的相似性；然后，采用两层GraphSAGE模型对图进行归纳式学习，通过邻居采样和均值聚合机制迭代更新节点嵌入，捕捉高阶关系；最后，将学习到的节点嵌入输入多层感知机（MLP）分类器进行违约预测，并结合SHAP方法对特征重要性进行量化解释，提升模型透明度。整个流程在GMSC和GC两个信用数据集上验证，显著优于传统机器学习和现有平衡技术。

#### On Correlated Stock Market Time Series Generation
- **Authors**: Giuseppe Masi, Matteo Prata, Michele Conti, Novella Bartolini, Svitlana Vyetrenko
- **Year**: 2023
- **Abstract**: 为解决多股票市场中合成时序数据难以准确捕捉资产间相关性动态的问题，本文提出CoMeTS-GAN框架，基于条件Wasserstein生成对抗网络（C-WGAN），通过在判别器中引入交叉相关性评分项，联合优化价格与成交量序列的统计真实性与资产间相关性，实现了在保持金融时序经典统计特征（如尖峰厚尾、波动聚集）的同时，精准复现多资产间正负相关关系，并支持自回归生成任意长度序列，训练效率显著优于现有模型。CoMeTS-GAN是一种基于条件Wasserstein生成对抗网络（C-WGAN）的多变量金融时间序列生成框架。其核心包括：（1）生成器采用7个扩张卷积块（dilated temporal convolutional blocks）与线性输出层，输入为过去P个时间步的多股票价格/成交量序列与随机噪声向量，输出为未来F个时间步的合成序列；（2）判别器（critic）由两部分组成：第一部分为卷积与线性层组成的序列真实度评分器o₁，第二部分为接收所有股票对（共C(n,2)对）相关系数的线性层，输出相关性评分o₂；（3）判别器最终得分o = o₁ + α·o₂，其中α为超参数，用于平衡序列真实度与相关性匹配度；（4）训练时，使用Wasserstein损失函数优化生成器与判别器，通过谱归一化（spectral normalization）保证Lipschitz连续性；（5）生成过程采用自回归方式：将生成的F步序列的后P步作为下一时刻的输入，迭代生成任意长度序列；（6）为评估相关性捕捉能力，提出交叉相关距离（cross-correlation distance）指标，即真实与生成序列间Pearson相关系数的均方误差。该方法在Sines、高斯模型和真实股票数据（如KO、PEP、NVDA、KSU）上验证，能同时复现金融时序的六大经典统计特征（尖峰厚尾、聚合正态性、无自相关、波动聚集、量-波正相关）与复杂资产间动态相关性，训练时间仅为TimeGAN的约1/10，且支持扩展至30只股票的DJIA指数。

#### Macroeconomic Conditioned Synthetic Financial Markets
- **Authors**: Alexander Michael Rusnak, Stéphane Daul
- **Year**: 2024
- **Abstract**: 为解决金融领域因历史数据稀缺和极端事件稀少导致的深度学习模型训练困难问题，本文提出了一种名为MC-TE-GAN的宏观条件Transformer编码器生成对抗网络，通过将宏观经济学变量作为条件输入，结合Transformer编码器结构和谱归一化对抗训练，生成具有真实统计特性（如波动聚集、杠杆效应、多资产相关性）且能模拟市场崩盘或牛市等极端场景的多资产金融时序数据，显著优于基准模型CoMeTS-GAN，尤其在出样本场景下展现出更强的条件响应能力和场景复现能力。MC-TE-GAN由生成器和判别器组成，生成器输入包括：(1) n_a个资产的宏观经济学特征（如利率、失业率、CPI等，形状为n_a×n_f）；(2) 每个资产过去64天的标准化对数回报序列（形状为n_a×P）；(3) 从标准正态分布采样的噪声向量（形状为n_a×Z）。三者拼接后输入由多个Transformer编码器块组成的生成器，经线性层映射至隐藏层，再经两层线性层输出未来64天的合成回报序列（形状为n_a×F）。生成的序列与原始条件和历史序列拼接，形成n_a×(n_f+P+F)的样本矩阵，与真实数据一同输入判别器。判别器采用相同结构的Transformer编码器，输出二分类标签。训练采用二元交叉熵损失（BCE）与谱归一化防止模式坍塌，并引入自适应学习率平衡器：每10个epoch比较生成器与判别器损失，若一方持续占优（如L_g ≤ 0.9×L_d或L_g ≥ 10×L_d）达50个epoch，则降低胜方学习率、提升败方学习率。为评估条件生成效果，提出聚类匹配法：将真实与生成的64天窗口序列合并，用k-means或Wasserstein-k-means聚类，计算真实与生成序列聚类标签的匹配准确率，验证模型对宏观条件的响应能力。

#### Prediction of index futures movement using TimeGAN and 3D-CNN: Empirical evidence from Korea and the United States
- **Authors**: Woojung Kim, Jiyoung Jeon, Sanghoe Kim, Minwoo Jang, Heesoo Lee, Sanghyuk Yoo, Kyong Joo Oh
- **Year**: 2023
- **Abstract**: 为解决指数期货市场中时序数据稀缺与高波动性导致预测性能不佳的问题，本文提出一种结合TimeGAN数据增强与3D-CNN的混合模型，通过TimeGAN生成具有时间动态特性的合成数据并将其扩展为三维张量，再利用3D-CNN捕捉多市场、多特征、多时间步的时空关联，从而在韩国与美国期货市场实现风险调整后收益提升1.35倍、训练效率提升6390倍的显著效果。首先，使用TimeGAN对历史期货数据（如KOSPI200、S&P500、NASDAQ100期货的37个技术指标）进行时序数据增强，TimeGAN通过嵌入网络、生成网络、判别网络和恢复网络四部分，结合重建损失、无监督对抗损失和监督时序损失，生成与真实数据时间结构一致的合成序列；其次，将原始数据与生成的合成数据按相似度（MSE）筛选并堆叠成三维张量（时间步×特征×样本数），形成包含多场景模拟的时空结构；然后，将该三维张量输入3D-CNN模型，通过1×1×k卷积提取每日特征，再通过3×3×k卷积捕捉跨市场与跨时间的动态模式，最终输出次日价格涨跌预测；最后，模型在滑动窗口框架下进行5年回测，结合保证金机制与交易成本，评估其在收益、波动率、夏普比率和累积回报等指标上的表现，结果表明10TG-3DCNN（使用10组增强样本）在风险调整收益和计算效率上均显著优于基线模型。

#### Article Enhancing Portfolio Performance through Financial Time-Series Decomposition-Based Variational Encoder-Decoder Data Augmentation
- **Authors**: Bayartsetseg Kalina, Ju-Hong Lee, Kwang-Tek Na
- **Year**: 2024
- **Abstract**: 为解决金融时间序列数据不足和历史数据不确定性缺失导致的投资组合模型性能受限问题，本文提出了一种基于金融时间序列分解的变分编码器-解码器（FED）数据增强方法，通过将时序数据分解为趋势、离散度和残差三个潜在成分并分别建模，生成更具真实性和多样性的合成数据，进而构建FED2Port强化学习投资组合模型，使算法能在更全面的市场不确定性环境中学习，显著提升投资组合绩效。本文提出的方法包括两个核心部分：FED数据增强和FED2Port投资组合决策模型。首先，FED方法通过三个独立的编码器分别建模金融时间序列的潜在趋势（ν_t）、离散度（τ_t）和残差（ξ_t）成分，每个成分均假设为多元正态分布，并通过乘积运算合成完整潜在变量h_t = ν_t × τ_t × ξ_t，利用变分自编码器框架最大化三个成分的证据下界（ELBO）联合目标函数，从而生成具有真实统计特性的合成金融时序数据。其次，FED2Port框架基于强化学习构建，状态为当前投资组合收益，动作是高风险与低风险资产的权重分配（满足和为1），奖励函数采用市场自适应比率（Market-Adaptive Ratio），该比率根据市场涨跌状态动态调整风险偏好，训练智能体在FED生成的合成市场环境中最大化长期期望奖励，最终实现更稳健和高性能的投资组合配置。

#### CoFinDiff: Controllable Financial Diffusion Model for Time Series Generation
- **Authors**: Yuki Tanaka, Ryuji Hashimoto, Takehiro Takayanagi, Zhe Piao, Yuri Murayama, Kiyoshi Izumi
- **Year**: 2023
- **Abstract**: 为解决金融领域因真实数据稀缺导致的极端事件模拟不足与合成数据可控性差的问题，本文提出CoFinDiff，一种基于条件扩散模型的金融时间序列生成方法，通过将对数收益率序列转换为Haar小波图像，并将趋势与已实现波动率作为条件通过交叉注意力机制注入扩散模型，从而生成符合金融stylized facts（如肥尾、波动聚集）且精准满足指定趋势与波动率条件的多样化合成数据，显著提升了深度对冲任务的模型性能。CoFinDiff方法包含三个核心步骤：(1) 数据预处理：将原始股票价格序列转换为对数收益率序列，标准化后应用Haar小波变换生成二维图像表示，同时计算趋势（对数收益率之和）与已实现波动率（对数收益率平方和）作为条件变量；(2) 训练阶段：采用条件扩散模型（基于DDPM架构），将小波图像作为输入，趋势与波动率经仿射变换与卷积处理后作为Key和Value输入交叉注意力模块，使模型学习条件与数据间的复杂非线性关系；(3) 推理阶段：给定任意目标趋势与波动率条件，通过扩散模型生成对应的图像，再经逆Haar小波变换重构为对数收益率序列，从而输出符合指定条件的合成金融时间序列。为提升极端事件生成能力，训练数据中对高绝对趋势样本进行五倍上采样。实验验证该方法能同时满足金融统计特性、条件精度、数据多样性，并提升下游深度对冲任务的鲁棒性。

#### Data Augmentation Using BERT-Based Models for Aspect-Based Sentiment Analysis
- **Authors**: Bron Hollander, Flavius Frasincar, Finn van der Knaap
- **Year**: 2023
- **Abstract**: 为解决方面情感分析（ABSA）中训练数据稀缺导致模型性能受限的问题，本文提出在HAABSA++模型中引入多种BERT-based数据增强方法，通过掩码语言建模（MLM）生成语义一致的增强样本，并结合标签感知的BERTprepend和BERTexpand策略保留情感标签信息，显著提升了模型在SemEval 2015和2016数据集上的测试准确率，最高提升达1.85个百分点。本文基于HAABSA++模型，该模型由领域情感本体和LCR-Rot-hop++神经网络组成。为增强训练数据，作者比较了五种数据增强方法：(1) EDA-adjusted：基于词典和词性标注的同义词替换、随机插入、交换和删除，并引入词义消歧与目标词跨句交换；(2) BERT：使用预训练BERT的MLM任务，以15%概率掩码每个词，用最高概率词替换（排除原词）；(3) C-BERT：将BERT的段嵌入替换为标签嵌入，训练时注入情感标签信息以指导掩码词预测；(4) BERTprepend：在输入序列前拼接情感标签（不加入词汇表），使BERT在生成时感知标签但不破坏通用性；(5) BERTexpand：与BERTprepend类似，但将情感标签作为单个token加入BERT词汇表。所有BERT模型均在SemEval训练集上微调10个epoch，使用默认掩码参数。最终，将增强后的数据输入HAABSA++的LCR-Rot-hop++模块进行情感分类。实验表明，BERTprepend和BERTexpand在SemEval 2016上表现最佳，将测试准确率从82.62%提升至84.47%。

#### Desensitized Financial Data Generation Based on Generative Adversarial Network and Differential Privacy
- **Authors**: Fan Zhang, Luyao Wang, Xinhong Zhang
- **Year**: 2024
- **Abstract**: 为解决金融数据敏感性强、可用样本少导致深度学习模型训练困难的问题，本文提出NVF-DPGAN模型，通过在生成对抗网络（GAN）的判别器训练过程中引入高斯噪声实现差分隐私保护，并结合噪声可见性函数（NVF）自适应调整噪声强度以保留数据关键特征，从而生成与真实金融数据统计特性高度一致的合成数据，实现数据增强与隐私保护的双重目标。本文提出NVF-DPGAN模型，其核心流程包括：(1) 从CSMAR数据库选取8个关键金融时序变量（如总资产、现金等价物等）构建原始数据集；(2) 构建生成器G与判别器D的深度神经网络结构，生成器输入100维高斯噪声，通过多层反卷积输出22×8维金融数据，采用批量归一化和ReLU激活函数提升训练稳定性；(3) 在判别器训练阶段引入高斯噪声以实现差分隐私，噪声方差根据全局敏感度S_f计算，确保满足(ε, δ)-差分隐私；(4) 引入噪声可见性函数（NVF）动态调整噪声注入强度，NVF基于局部图像纹理能量（由像素方差和权重函数w(i,j)计算）评估各数据点对噪声的敏感度，高纹理区域允许更大噪声，平坦区域限制噪声以保留关键结构；(5) 生成器损失函数在传统GAN基础上增加lnf_loss正则项，最小化生成样本与真实样本梯度幅度差异，提升生成质量；(6) 通过对抗训练使生成器逐步生成满足差分隐私、与真实数据分布一致的合成数据，实验表明生成数据在均值、概率密度、相关性及时间序列预测性能上与真实数据高度相似，实现隐私保护与数据增强的协同优化。

#### Simulating Asset Prices using Conditional Time-Series GAN
- **Authors**: Riasat Ali Istiaque, Chi Seng Pun, Yuli Song
- **Year**: 2024
- **Abstract**: 为解决金融资产价格生成中传统模型无法有效模拟多变量时序依赖且缺乏条件控制的难题，本文提出条件时间序列生成对抗网络（CTS-GAN），通过五网络架构结合移动窗口机制，在给定历史条件序列下生成出样本多变量资产价格序列，并利用stylized facts验证其统计特性，成功实现了对真实市场动态的高保真模拟，应用于投资组合风险价值（VaR）估算，显著提升风险评估的准确性与鲁棒性。CTS-GAN采用五网络架构：嵌入器（E）、恢复器（R）、监督器（S）、生成器（G）和判别器（D）。训练分三阶段：第一阶段，E和R联合训练以最小化重构损失，将原始价格序列映射到潜在空间并重建；第二阶段，S训练以学习潜在空间中的时序动态，最小化时序损失；第三阶段，G和D在条件生成对抗框架下联合训练，引入无监督损失（使生成序列在潜在空间中难以被区分）和监督损失（使生成序列经监督器后仍具真实时序特性），并加入正则化项约束生成序列的一阶与二阶矩。模型采用移动窗口（窗口长度257，序列长度12，条件长度7）处理数据，每次训练后偏移条件序列1步以生成下一时间步的出样本数据。生成时，从均匀分布采样噪声，与条件序列拼接后输入G，经S和R恢复为价格序列，再反归一化得到最终价格。模型在S&P500五只股票数据上验证，通过t-SNE、PCA可视化、stylized facts（无线性自相关、厚尾分布、波动聚集、粗细波动预测、盈亏不对称、Marchenko-Pastur分布、Perron-Frobenius性质）全面评估，最终用于计算投资组合的VaR，结果优于传统正态假设方法。

#### TRADES: Generating Realistic Market Simulations with Diffusion Models
- **Authors**: Leonardo Berti, Bardh Prenkaj, Paola Velardi
- **Year**: 2025
- **Abstract**: 为解决金融市场上真实限价订单簿（LOB）数据稀缺且现有生成模型缺乏 realism、responsiveness 和 usefulness 的问题，本文提出 TRADES，一种基于 Transformer 的去噪扩散概率模型，通过条件化历史订单和 LOB 快照生成高保真、可响应的订单流时间序列，显著超越现有方法，在预测得分上提升 3.27–3.48 倍，并能复现金融市场的典型统计特征（stylized facts）。TRADES 是一种条件化去噪扩散概率模型，用于生成限价订单簿（LOB）的订单流时间序列。其核心流程包括：（1）前向过程：仅对目标生成部分（即下一个订单）添加高斯噪声，保留历史条件数据（前 N−1 个订单和最近 N 个 LOB 快照）不变；（2）模型架构：采用 Transformer 编码器作为主干网络，输入通过两个 MLP 分别对订单序列和 LOB 快照进行特征增强，再拼接后输入 Transformer，同时融合时间步嵌入和位置嵌入；（3）去噪过程：通过神经网络 ε_θ 预测噪声，并学习可变方差 Σ_θ，使用重参数化公式逐步去噪，重构原始订单；（4）训练目标：最小化预测噪声的均方误差（L_ε）与扩散过程负对数似然（L_Σ）的加权和；（5）推理过程：以滑动窗口方式自回归生成订单，每生成一个订单后将其加入条件序列，持续生成直至模拟结束；（6）条件输入：融合前 256 个订单（含价格、数量、方向、深度、时间偏移、订单类型）和前 256 个 LOB 快照（前 10 层买卖盘价格与成交量），以捕捉市场供需动态；（7）评估指标：提出预测得分（Predictive Score），即在合成数据上训练价格预测模型，在真实数据上测试 MAE，用以量化生成数据的实用性；（8）系统实现：构建开源框架 DeepMarket，集成 TRADES 模型、合成数据集 TRADES-LOB 和扩展的 ABIDES 仿真环境，支持引入实验性交易代理进行市场冲击实验。

#### Integrating Symbolic Genetic Programming With Lstm for Forecasting Cross-Sectional Price Returns: A Comparative Analysis of Chinese And Japanese Stock Market
- **Authors**: Li Qi, Norshaliza Kamaruddin, Xun Gong, Chen Peng
- **Year**: 2024
- **Abstract**: 为解决传统深度学习模型在跨市场股票收益排序预测中因特征数量有限和过拟合导致的精度不足问题，本文提出一种融合符号遗传编程（SGP）与长短期记忆网络（LSTM）的混合模型，通过SGP自动生成高质量金融因子并进行数据增强，再输入LSTM进行跨截面股票收益排序预测，在中国和日本市场分别实现Rank IC提升588.03%和194.27%，并获得显著超额收益。该方法包含四个阶段：（1）数据准备：收集中日市场4600+只股票的244个基本面因子，进行清洗、标准化和7:2:1划分；（2）数据增强：使用符号遗传编程（SGP）以基本面因子、数学运算符和滚动窗口为基因，通过交叉、变异和选择生成新因子，采用自定义适应度函数（结合Top R、单调性与信息系数）评估因子质量，并设置过滤条件（Rank IC > 7%、Success Ratio > 75%、IC PNL > 1.8）筛选高质量因子；（3）特征选择与建模：将增强后的因子输入LSTM或MLP模型，通过Rank IC和ICIR指标选择最优网络结构（LSTM表现更优），模型配置为两层结构（100-100节点）、学习率0.01、Dropout 0.6；（4）投资策略与回测：在2019-2022年期间使用20日滚动窗口进行训练、验证和测试，中国采用多头策略（买入前k%股票），日本采用多空策略，以CSI 300/500、N225/TPX和等权组合为基准，最终实现中国年化收益22.35%（超CSI 300）、日本超N225 4.56%和TPX 5.10%。

### 其他（通用时序 \& Survey）

#### An LLM-Based Framework for Synthetic Data Generation
- **Authors**: Mandeep Goyal, Qusay H. Mahmoud
- **Year**: 2024
- **Abstract**: 为解决医疗、金融等敏感领域数据稀缺与隐私保护的难题，本文提出一种基于微调大语言模型（LLM）与差分隐私技术的合成数据生成框架，通过微调LLM学习真实数据分布，结合IBM diffprivlib施加差分隐私保护，实现对结构化与非结构化数据的高保真合成，显著提升数据质量与隐私安全性，且在机器学习任务中保持接近真实数据的模型性能。该框架首先通过微调大语言模型（如OpenAI API）使其学习特定领域（如医疗、金融）的真实数据分布，利用提示工程（prompt engineering）引导模型生成符合领域特征的合成数据；当用户提供原始数据时，使用IBM diffprivlib对数据施加差分隐私（ε参数控制隐私强度），确保原始数据不被泄露；随后，LLM基于隐私化后的数据进行模式学习，通过迭代查询机制持续生成合成记录，直至达到用户指定的数据规模；对于无原始数据的用户，框架提供五大预定义领域（医疗、金融、零售、物流、网络安全），用户输入简要描述后，LLM基于领域微调生成针对性合成数据；同时，框架集成数据预处理模块，支持缺失值填补（均值/中位数/众数）、异常值处理与类别平衡；数据生成后，系统可自动生成可视化代码（如直方图、散点图）辅助分析；最终，通过实验验证，在ε=10.0时合成数据与原始数据统计分布高度一致，机器学习模型性能仅轻微下降（准确率91.1% vs 95.5%），且在与ChatGPT、Gemini等通用LLM对比中显著优于其生成质量，同时计算效率优于GAN和Copula方法。

#### TIME-LLM: TIME SERIES FORECASTING BY REPROGRAMMING LARGE LANGUAGE MODELS
- **Authors**: Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuanfang Li, Shirui Pan, Qingsong Wen
- **Year**: 2023
- **Abstract**: 为解决传统时间序列预测模型缺乏通用性、数据效率低和推理能力弱的问题，本文提出TIME-LLM框架，通过将时间序列数据重编程为文本原型并结合Prompt-as-Prefix提示机制，使冻结的大语言模型能够直接理解时序模式并生成预测，实现在长周期、短周期、少样本和零样本场景下超越现有专用模型的预测性能。TIME-LLM框架包含三个核心步骤：(1) 输入变换：将多变量时间序列按通道分离，通过可逆实例归一化（RevIN）处理后，划分为重叠或非重叠的时序块（patches），并使用线性层嵌入为低维向量；(2) 模态对齐与增强：使用多头交叉注意力机制，将时序块嵌入与一组预训练的语言文本原型（text prototypes）进行对齐，将时序模式转化为语言模型可理解的语义表示；同时引入Prompt-as-Prefix（PaP）机制，在输入前缀中注入数据集上下文、任务指令和统计特征（如趋势、前五阶自相关滞后值），引导模型推理；(3) 输出投影：将经过冻结LLM处理后的输出表示去除前缀部分，展平后通过线性投影层映射为未来H步的预测值。整个过程中仅优化轻量级的嵌入层、文本原型和投影层参数，LLM主干完全冻结，实现高效、无微调的跨模态迁移学习。

#### A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges
- **Authors**: Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M. Mulvey, H. Vincent Poor, Qingsong Wen, Stefan Zohren
- **Year**: 2024
- **Abstract**: 为系统梳理大语言模型（LLM）在金融领域的应用进展、技术优势与挑战，本文综述了LLM在金融文本分析、情感分析、时间序列预测、金融推理和智能体建模等六大类任务中的研究现状，通过整合模型架构、数据集、基准测试与实践案例，揭示了LLM在提升金融决策效率与智能化水平方面的潜力，并指出数据、建模、伦理等关键挑战以指导未来研究。本文并未提出新的具体方法，而是一篇系统性综述论文。其方法论基于对现有文献的全面收集、分类与分析：首先，梳理了金融领域专用LLM模型（如FinBERT、BloombergGPT、FinGPT等）的架构与训练策略；其次，将金融应用划分为六大核心方向（语言任务、情感分析、时间序列分析、金融推理、智能体建模及其他），并深入分析各方向下的关键技术（如文本摘要、知识图谱构建、零样本推理、多智能体仿真等）；再次，整理并发布了主流金融LLM相关的数据集、代码库与基准测试资源；最后，系统归纳了当前应用中存在的挑战，包括数据偏差、 lookahead bias、可解释性缺失、伦理风险等，并提出未来研究方向，旨在为学术界与工业界提供全面、实用的参考框架。

#### LLM40FD: Unlocking the Potential of LLM for Anonymous Zero-Shot Fraud Detection
- **Authors**: Kaixiang Yang, Zhijie Zhong, Song Sun, Zhiwen Yu, C. L. Philip Chen, Tong Zhang
- **Year**: 2024
- **Abstract**: 为解决信用卡欺诈检测中标签稀缺、数据匿名化和跨系统泛化能力不足的问题，本文提出LLM40FD框架，通过行走嵌入将匿名交易特征统一转化为LLM可处理的序列，结合基于分布的一类函数（DOC）进行知识蒸馏，并利用隐式对比学习的双重数据增强策略生成正负样本以强化决策边界，无需微调LLM即可在零样本和全样本设置下实现SOTA性能。LLM40FD框架包含三个核心组件：（1）行走嵌入（Walking Embedding）：将匿名交易特征按固定步长分割为多个token，通过线性层映射为LLM的隐藏维度向量，并以不同步长多次采样后拼接，形成统一上下文表示，解决匿名特征语义缺失和维度不一致问题；（2）知识蒸馏（DOC函数）：定义一个正常交易的知识中心C，通过最小化统一表示U与C之间的Softmax交叉熵损失（L_DOC），引导LLM将正常行为聚类至该中心，从而隐式学习正常模式分布；（3）双重数据增强（Dual-Augmentation）：在每个mini-batch中，基于均值μ和标准差σ生成正样本（X_pos = μ + γ_pos·σ·N）以强化正常行为聚类，生成负样本（X_neg = X + γ_neg·σ·N）以模拟异常行为并远离知识中心，分别计算正负样本的DOC损失（L_DOC^p 和 L_DOC^n），最终联合优化总损失L = L_DOC + L_DOC^p + L_DOC^n。推理阶段，计算每个样本的DOC异常得分，超过阈值则判定为欺诈。整个流程无需微调LLM参数，仅训练行走嵌入和投影头，实现跨数据集零样本迁移。

#### Auto-Generating Earnings Report Analysis via an Augmented LLM
- **Authors**: Van-Duc Le
- **Year**: 2023
- **Abstract**: 为解决金融分析师人工撰写财报分析耗时耗力的问题，本文提出一种基于检索增强指令微调的增强型LLM方法，通过结合金融领域文档上下文与教师模型自动生成指令数据，微调Llama-2-7b模型，使其在财报分析任务中性能超越开源模型Llama-2-7b并接近GPT-3.5的商业水平。首先，从半导体公司（如NVIDIA、AMD）的季度财报中提取文本块，利用教师模型GPT-3.5-turbo生成通用金融指令数据（如财务指标问答对）；其次，设计六类种子指令（公司核心信息、关键财务指标、对比分析、未来展望、总结、深度分析），结合具体财报上下文生成领域专用的指令数据；接着，构建向量数据库（ChromaDB）实现检索增强，使教师模型在生成问答对时能依据上下文精准定位信息；然后，采用Llama-2-7b作为基础模型，使用LoRA进行低秩适配，并结合QLoRA的4-bit量化技术实现高效微调；最后，在 Broadcom 的财报上进行评估，使用GPT-4作为评分器，通过正确性（1-10分）和语义相似度（低分更优）指标验证性能，实验表明该方法显著优于Llama-2-7b，接近GPT-3.5。

#### TimeHF: Billion-Scale Time Series Models Guided by Human Feedback
- **Authors**: Yongzhi Qi, Hao Hu, Dazhou Lei, Jianshen Zhang, Zhengxin Shi, Yulin Huang, Zhengyu Chen, Xiaoming Lin, Zuo-Jun Max Shen
- **Year**: 2024
- **Abstract**: 为解决大规模时间序列模型在 scalability、泛化能力和零样本预测性能上的挑战，本文提出TimeHF框架，通过构建210B规模的高质量时间序列数据集、设计基于补丁卷积的60亿参数纯时间序列模型（PCLTM），并首次引入面向时间序列的强化学习人类反馈优化方法（TPO），利用专家构建的优劣预测对比对引导模型学习隐性专家知识，最终在京东供应链中实现预测精度提升33.21%。TimeHF框架包含三个核心步骤：(1) 基础模型训练：提出Patch Convolutional Large Time Series Model (PCLTM)，采用补丁卷积层（PatchConv）捕捉跨补丁长期依赖，结合旋转位置编码（ROPE）和分组查询注意力（GQA）提升建模能力，模型输入为时间序列分块，输出为多步预测；(2) 监督微调（SFT）：在领域特定数据集（如京东销售数据）上对PCLTM进行微调，提升特定场景预测精度；(3) 时序策略优化（TPO）：首次将RLHF引入纯时间序列模型，构建由专家模型生成的‘优预测-劣预测’对比对，采用REINFORCE风格的奖励函数，通过概率化预测（假设预测服从正态分布）计算策略比值与优势函数，联合优化模型以逼近专家偏好，同时引入MSE损失约束优劣预测的差距，仅训练单一策略模型，避免传统PPO/RLOO的多模型开销，显著提升效率与效果。最终模型在京东20,000个SKU的自动补货系统中部署，实现33.21%的精度提升。

#### Combining Financial Data and News Articles for Stock Price Movement Prediction Using Large Language Models
- **Authors**: Ali Elahi, Fatemeh Taghvaei
- **Year**: 2024
- **Abstract**: 为解决金融市场上股票价格走势预测中多源异构数据（结构化财务数据与非结构化新闻文本）融合建模的难题，本文提出一种基于大语言模型（LLM）的零样本至四样本提示方法，通过检索增强生成（RAG）技术提取与公司相关的新闻摘要，并将财务指标与新闻摘要拼接为长文本提示输入LLM进行二分类预测，实现了3个月和6个月预测窗口下加权F1分数分别达59.2%和59.1%的效果。首先，收集20家高交易量公司从2021年至2024年的财务数据（来自10-K报告，包括总收入、净利润、自由现金流等）和新闻文章（通过关键词和公司名爬取，共5000篇）。其次，对新闻文章进行提取式摘要处理：使用OpenAI嵌入模型计算新闻片段（每段三句）与用户查询（如'Should I invest in Apple in July 2022?'）的语义相似度，选取最相关的6个新闻片段作为上下文。接着，将公司行业与业务描述、6个新闻摘要、最近四个季度的财务指标拼接为结构化提示（平均2500个token），并以二分类标签（[UP]/[DOWN]）表示未来3或6个月的股价涨跌。然后，使用GPT-3.5、GPT-4、LLaMA2和LLaMA3等预训练LLM在零样本、两样本和四样本设置下进行推理，不进行微调。最后，采用加权F1分数（WF1）和马修斯相关系数（MCC）评估模型性能，发现GPT-3.5（零样本）在3个月预测中表现最佳（WF1=59.2%），GPT-4（两样本）在6个月预测中表现最佳（WF1=59.1%），且增加示例数量并未显著提升性能，可能因提示过长导致模型混淆。

#### LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics
- **Authors**: Jialiang Tang, Shuo Chen, Chen Gong, Jing Zhang, Dacheng Tao
- **Year**: 2024
- **Abstract**: 为解决大型语言模型（LLM）在时间序列预测中忽视时序数据固有特性（如多尺度时间模式和语义稀疏性）导致性能不佳的问题，本文提出LLM-PS方法，通过多尺度卷积神经网络（MSCNN）提取短长期时间模式，并通过时间到文本模块（T2T）从时间序列片段中提取语义信息，再将二者融合输入LLM进行预测，在多种数据集上实现了最先进的预测精度，尤其在少样本和零样本场景下表现优异。LLM-PS方法包含两个核心模块：1）多尺度卷积神经网络（MSCNN）：通过堆叠瓶颈块，每个块并行使用多个3×3卷积分支，逐步扩大感受野以捕捉不同尺度的时间模式；每个分支的输出通过残差连接和1×1卷积融合，生成多尺度特征；2）时间到文本语义提取模块（T2T）：将输入时间序列分割为多个时间片段，随机掩码约75%的片段，通过编码器-解码器结构重建被掩码片段并预测其语义标签（使用LLM文本嵌入相似性匹配），以学习时间序列中的语义信息；3）模式解耦与组装：对MSCNN输出的多尺度特征应用小波变换，分离低频（长期趋势）和高频（短期波动）成分，通过局部到全局和全局到局部的累加方式增强模式表达，再重构特征；4）LLM微调：将MSCNN提取的多尺度特征与T2T提取的语义特征通过特征对齐融合，输入预训练GPT-2模型，使用LoRA参数高效微调，联合优化预测误差（L_TIME）和特征对齐损失（L_FEAT），实现端到端训练。该方法在多个真实数据集上显著优于现有LLM和传统深度学习方法，尤其在长周期、少样本和零样本预测中表现卓越。

#### Lending an Ear: How LLMs Hear Your Banking Intentions
- **Authors**: Varad Srivastava
- **Year**: 2024
- **Abstract**: 为解决银行领域数据稀缺条件下客户意图识别的低性能与高成本问题，本文提出基于小样本学习和检索增强生成（RAG）的开源大语言模型（LLM）方法，通过在Banking77数据集上对比开源与闭源LLM的性能与成本，发现小规模开源模型Mistral-7B结合RAG和参数高效微调（PEFT）可超越大型闭源模型，实现94.58%的微F1分数且成本降低540倍。本文首先在Banking77数据集上评估了Llama-3、Mistral-7B、Gemma-2和Falcon等开源指令微调LLM在零样本和小样本（1-shot、3-shot）场景下的意图分类性能，比较随机选取示例与专家精选示例的效果；其次，提出使用RAG方法，通过MPNet模型计算测试句与训练句的语义相似性，仅选取最相似的5、10或20个示例作为上下文，显著减少输入规模并提升性能；接着，探索使用Mistral-7B生成人工标注样本以增强小样本数据，发现其增益有限；随后，通过成本分析证明Mistral-7B在RAG设置下性能优于GPT-4且成本仅为1/540；为验证RAG的偏差影响，进行去偏差实验（强制每个类仅选一个示例），发现性能显著下降，表明语义相似示例的类别偏差对性能提升至关重要；最后，采用QLoRA、rsLoRA和DoRA三种参数高效微调方法对Mistral-7B进行微调，最终提出BAI-Fintent模型，在仅使用4-bit量化和部分注意力模块微调下，达到94.58%的微F1分数，成为当前最优方法。

#### MARS: A FINANCIAL MARKET SIMULATION ENGINE POWERED BY GENERATIVE FOUNDATION MODEL
- **Authors**: Junjie Li, Yang Liu, Weiqing Liu, Shikai Fang, Lewen Wang, Chang Xu, Jiang Bian
- **Year**: 2024
- **Abstract**: 为解决传统金融市场模拟器缺乏订单级细粒度、可控性与交互性的问题，本文提出基于生成基础模型LMM的MarS仿真引擎，通过订单序列与订单批次的双尺度建模，结合条件生成机制和模拟撮合引擎，实现高保真、可控制、可交互的市场行为仿真，显著提升预测、风险检测、市场影响分析和智能体训练等金融任务的性能与实用性。本文提出Large Market Model (LMM)，作为金融市场的生成基础模型，其核心包括两个互补模块：(1) Order Model：使用因果Transformer对个体订单序列和限价簿(LOB)信息进行编码，通过嵌入订单类型、价格、成交量及LOB的10档买卖量和中间价，实现订单级的自回归生成；(2) Order-Batch Model：将时间窗口内的订单聚合为图像格式（order image），使用VQ-VAE进行离散表示与生成，捕捉宏观市场行为模式。LMM通过集成这两个模型，构建统一的生成框架，并引入条件生成机制，支持四种输入条件：用户描述的市场情景（DES.TEXT）、用户注入的交互订单（Interactive Orders）、初始历史订单序列（Starting Sequence）和市场撮合规则（MTCH.R）。MarS引擎基于LMM构建，通过模拟撮合引擎实时匹配生成订单与用户订单，动态更新LOB状态，形成闭环仿真。生成过程遵循两大原则：'基于已实现现实塑造未来'和'从所有可能未来中选择最优匹配'，确保仿真既符合市场微观结构，又能响应用户干预。通过大规模训练（32B订单token）和扩展性验证，MarS能精确复现金融市场的14项经典统计特征（stylized facts），并支持高精度市场预测、异常检测、'假设分析'（如发现新市场影响因子resiliency、LOB pressure）和强化学习智能体训练，实现金融仿真从统计建模到生成式交互环境的范式转变。

#### Black-Scholes Meet Imitation Learning: Evidence From Deep Hedging in China
- **Authors**: Fuwei Jiang, Jie Kang, Ruzheng Tian, Qingdong Xu
- **Year**: 2025
- **Abstract**: 为解决中国股票指数期权市场中深度对冲算法因数据稀缺和尾部风险难以管理而导致的性能不足问题，本文提出了一种融合Black-Scholes-Merton模型示范与深度强化学习的模仿学习深度对冲（ILDH）算法，通过将BSM模型生成的对冲动作作为专家示范与智能体自身探索数据结合进行训练，并引入循环神经网络记忆历史信息，显著提升了对冲收益、降低了风险与交易成本。本文提出模仿学习深度对冲（ILDH）算法，其核心包括三部分：（1）采用TD3（Twin Delayed Deep Deterministic Policy Gradient）作为基础强化学习算法，处理连续状态与动作空间的期权对冲问题，利用裁剪双Q学习和延迟策略更新提升稳定性；（2）引入模仿学习模块，采用DAgger算法，将Black-Scholes-Merton模型生成的对冲动作作为专家示范，与智能体在真实交易数据中探索得到的动作样本共同存储于经验回放缓冲区，进行混合训练，实现数据增强并提升对尾部风险的应对能力；（3）在TD3的Actor与Critic网络中嵌入循环神经网络（RNN）结构，将历史观测序列（如价格、波动率、持仓等）作为记忆输入，使智能体能够建模部分可观测马尔可夫决策过程（POMDP），更准确推断隐藏状态；训练过程中，每个对冲周期的奖励函数结合当期盈亏与风险厌恶系数，通过最大化期望收益减去风险惩罚来引导策略优化；最终，算法在CSI 300期权数据集上实证表明，其收益高于传统Delta对冲、纯数据驱动对冲（EDH）和模型驱动对冲（MDDH），且在不同交易成本和风险厌恶水平下均保持稳健性能。

#### Data Augmentation using Large Language Models: Data Perspectives, Learning Paradigms and Challenges
- **Authors**: Bosheng Ding, Chengwei Qin, Ruochen Zhao, Tianze Luo, Xinze Li, Guizhen Chen, Wenhan Xia, Junjie Hu, Anh Tuan Luu, Shafiq Joty
- **Year**: 2024
- **Abstract**: 为解决大规模语言模型（LLM）时代训练数据稀缺与标注成本高昂的问题，本文系统综述了基于LLM的数据增强方法，从数据生成、标注、重构和人机协同四个视角出发，结合生成式与判别式学习范式，利用LLM生成高质量合成数据以提升模型性能，实现了在低资源任务中接近或超越人工标注数据的效果，并揭示了数据污染、可控性、文化适配、多模态与隐私等关键挑战。本文并非提出一种新方法，而是一篇系统性综述，全面梳理了基于大语言模型（LLM）的数据增强（DA）技术。首先，从数据视角将现有方法分为四类：（1）数据创建：利用LLM的少样本学习能力生成合成数据，如通过提示工程生成对话、指令、推理链、多语言数据等；（2）数据标注：使用LLM对大规模无标签数据（如跨语言文本、社交媒体内容、视觉问答图像）进行自动标注，其准确率可媲美甚至超越人工标注；（3）数据重构：通过LLM改写、生成反事实样本、重述句子或生成图像描述，以增加数据多样性并提升模型鲁棒性；（4）协同标注：构建人机协作流程，利用LLM的不确定性评估分配标注任务，或结合人类反馈迭代优化生成数据。其次，从学习范式视角，将LLM驱动的数据增强分为生成式学习（包括监督指令学习、上下文学习、对齐学习）和判别式学习（包括伪标签分类与伪评分回归），并详述了如Self-Instruct、DPO、Eureka等代表性方法。最后，本文系统分析了该领域面临的五大挑战：数据污染、可控性不足、文化偏见、多模态融合困难与隐私泄露风险，并为未来研究指明方向。

#### Pattern recognition with limited data: an AI model inspired by BCR-Net and SwitchNet
- **Authors**: Ali Sever
- **Year**: 2024
- **Abstract**: 为解决小样本数据下深度学习模型性能受限的问题，本文提出了一种受BCR-Net和SwitchNet启发的模式识别系统（PRS），通过将逆问题数学框架与神经网络结合，利用积分算子的低秩结构和小波分解实现数据驱动的参数选择与增强，在多个合成与真实生物图像数据集上显著提升了模式识别的准确率与鲁棒性。本文提出的模式识别系统（PRS）通过以下步骤实现：首先，将模式识别任务建模为第一类Fredholm积分方程的逆问题，形式为Af(x)=g(x)，其中核函数K(x,y)为Riesz型；其次，引入SwitchNet模块近似正向算子A及其伴随算子A*，并使用BCR-Net模块近似正则化逆算子(A*A + αI)⁻¹，构建端到端的神经网络架构g(x) → SwitchNet → BCR-Net → f(x)；BCR-Net基于非标准小波形式，将积分算子的低秩非对角块分解为三个矩阵乘积，其中中间矩阵为对角占优结构，通过堆叠多层并引入ReLU激活函数实现非线性扩展；训练时，利用合成数据和BBBC022真实显微图像数据集（80%/20%划分）生成输入-输出对，通过梯度下降优化网络参数，正则化参数α和网格点数经调优以平衡稳定性与精度；最终，PRS在小样本条件下实现了优于SVM、LSTM、原型网络等基线模型的识别准确率，在多流形分类任务中达到91%的准确率，同时保持较低的测试计算成本和良好的泛化能力。

#### DATA-CENTRIC FINANCIAL LARGE LANGUAGE MODELS
- **Authors**: Zhixuan Chu, Huaiyu Guo, Xinyuan Zhou, Yijia Wang, Fei Yu, Hong Chen, Wanqing Xu, Xin Lu, Qing Cui, Longfei Li, Jun Zhou, Sheng Li
- **Year**: 2023
- **Abstract**: 为解决金融领域中大型语言模型（LLM）难以有效整合多源复杂金融信息进行深度分析的问题，本文提出一种数据中心化的金融大语言模型（FLLM）结合抽象增强推理（AAR）的方法，通过多任务提示微调对金融文本进行预处理与语义理解，并利用AAR自动增强高质量训练数据，显著提升了金融分析与解读任务的性能，达到当前最优水平。本文方法包括两个核心组件：1）金融大语言模型（FLLM）：采用多任务提示微调框架，对原始金融文本进行三项子任务处理——事件匹配与类比（匹配政策与相关报告）、观点质量评估（筛选高价值观点句）、关键点提取（抽取行业、情感、维度等结构化信息），从而构建一个经过深度预处理的金融知识表示；2）抽象增强推理（AAR）：为解决标注数据稀缺问题，AAR利用FLLM生成的伪标签，通过三个模块迭代优化：FAP（动态知识提问）自动生成针对伪标签缺陷的分析问题，FAE（一致知识回答）基于领域专家知识给出高质量答案，FADOM（知识融合修正）将问答结果融合回原始输出，生成更准确的标注数据用于FLLM再训练。最终，预处理后的结构化数据被输入到冻结的LLM（如ChatGPT）中生成高质量金融分析报告。实验表明，该方法在金融分析基准上显著优于直接使用原始文本的LangChain和纯LLM标注方法，且AAR中每个模块对性能提升均有贡献。

#### FINMEM: A PERFORMANCE-ENHANCED LLM TRADING AGENT WITH LAYERED MEMORY AND CHARACTER DESIGN
- **Authors**: Yangyang Vu, Haohang Li, Zhi Chen, Yuechen Jiang, Yang Li, Denghui Zhang, Rong Liu, Jordan W. Suchow, Khaldoun Khashanah
- **Year**: 2024
- **Abstract**: 为解决传统金融交易代理在处理多源异构金融数据时缺乏可解释性、记忆能力不足和无法自适应市场变化的问题，本文提出FINMEM，一种基于大语言模型（LLM）的自主交易代理框架，通过分层记忆模块模拟人类工作记忆与长期记忆结构，结合动态角色配置（如风险偏好自适应）和多层信息处理机制，实现对新闻、财报等时序金融数据的高效整合与优先级排序，显著提升交易收益与决策鲁棒性。FINMEM包含三个核心模块：(1) Profiling模块：为代理配置专业背景知识（如行业、公司历史财务表现）和三种风险偏好（风险偏好型、风险规避型、自适应风险型），支持根据累计收益动态切换风险策略；(2) Memory模块：采用分层长期记忆结构（浅层、中层、深层），分别对应日度新闻（Q_s=14天）、季度报告（Q_i=90天）和年度报告（Q_d=365天），每层记忆事件通过综合评分γ_l^E = S_Recency + S_Relevancy + S_Importance进行排序，其中：① S_Recency基于指数衰减模型 e^(-δ^E/Q_l) 计算时间衰减；② S_Relevancy使用OpenAI文本嵌入模型计算查询与记忆的余弦相似度；③ S_Importance通过分段函数v_l^E（浅层：80%概率40分，15%概率60分，5%概率80分；深层反之）乘以时间衰减因子θ_l = (α_l)^δ^E（α_shallow=0.9, α_intermediate=0.967, α_deep=0.988）计算重要性；同时引入访问计数器，将高影响力事件自动升级至深层记忆；工作记忆执行摘要、观察（训练时用未来价格标签，测试时用M日累计收益）和反思（即时反思生成交易决策与理由，扩展反思汇总M日表现并存入深层记忆）；(3) Decision-making模块：在测试阶段，综合当前市场趋势、扩展反思结果和各层Top-K记忆事件，通过LLM生成Buy/Sell/Hold决策。整个系统基于FAISS向量数据库实现高效检索，利用LLM进行文本摘要与推理，实现端到端的自主交易决策。

#### Variational autoencoder-based anomaly detection in time series data for inventory record inaccuracy
- **Authors**: Halil ARGUN, S. Emre ALPTEKİN
- **Year**: 2023
- **Abstract**: 为解决零售业库存记录不准确（IRI）问题，本文提出一种基于变分自编码器（VAE）的无监督异常检测方法，通过构建单变量和多变量时间序列模型，利用编码-解码结构学习库存数据的潜在分布，并通过重建误差识别异常点，实现了在不依赖人工标注的情况下有效检测高低库存异常，显著减少误报并支持动态阈值调整。首先，从土耳其一家大型超市的大数据平台收集18个乳制品子类别的每日库存数据，构建单变量（单子类）和多变量（多子类）时间序列；其次，进行特征工程，包括计算过去7天的平均值、标准差、变异系数、日变化率，并使用正弦-余弦编码表示日期、周、月的周期性特征；接着，对数据进行预处理，将负值和缺失值替换为0，使用3倍IQR识别并插值极端值，再对所有特征进行最小-最大归一化；然后，构建VAE模型，编码器由11-20-3（单变量）或18-25-5（多变量）神经元组成，解码器对称，使用均方误差作为损失函数，通过优化ELBO（证据下界）学习潜在变量的高斯分布；训练后，计算每个时间点的重建误差，将子类别的重建误差超过Q3+1.5×IQR的点标记为异常；最后，通过阈值分析和业务验证，实现对异常库存的自动识别与报警，支持动态调整阈值和新增特征。

#### Enterprise violation risk deduction combining generative Al and event evolution graph
- **Authors**: Chao Zhong, Pengjun Li, Jinlong Wang, Xiaoyun Xiong, Zhihan Lv, Xiaochen Zhou, Qixin Zhao
- **Year**: 2023
- **Abstract**: 为解决上市公司违规事件因果逻辑缺失、可解释性低和训练数据不足的问题，本文提出一种融合生成式AI与事件演化图的违规风险推断框架，通过ChatGLM2生成违规文本摘要、基于UIE模型提取事件实体、设计CDDP-GAT模型提取因果关系、构建加权事件演化图，最终实现对违规风险路径与后果的精准识别与可视化，显著提升风险推断的准确性和可解释性。首先，利用ChatGLM2大语言模型对冗长的上市公司违规公告生成简洁因果摘要，去除敏感信息并保留核心违规逻辑；其次，通过Prompt工程对ChatGLM2进行微调，实现自动化数据增强，生成带有结构化标签（违规事件、影响事件、处罚事件）的训练样本，扩充数据集；接着，采用UIE（统一结构生成模型）对摘要进行事件实体抽取，将文本映射为结构化事件三元组；然后，提出CDDP-GAT模型，结合中文词典预训练的WoBERT进行词向量编码，利用依存句法分析构建句子依赖图，通过图注意力网络（GAT）动态学习事件实体间的因果权重，区分‘违规→影响’和‘影响→处罚’两类因果关系；随后，对相似事件进行合并，计算事件间因果关联权重，并使用Neo4j图数据库构建企业违规事件演化图；最后，基于演化图进行风险推断，可视化违规事件的因果链与潜在后果，形成可解释的金融违规专家系统，有效识别传统方法难以挖掘的深层违规路径与连锁风险。

#### Management Analysis Method of Multivariate Time Series Anomaly Detection in Financial Risk Assessment
- **Authors**: Yongshan Zhang, Weifang University of Science and Technology, China, Zhiyun Jiang, Weifang University of Science and Technology, China, Cong Peng, Guizhou University of Commerce, China, Xiumei Zhu, Weifang University of Science and Technology, China, Gang Wang, Imperial College London, UK
- **Year**: 2023
- **Abstract**: 为解决金融多变量时间序列异常检测中模型过拟合与泛化能力不足的问题，本文提出一种结合对比学习与生成对抗网络（GAN）的创新方法，通过几何分布掩码进行数据增强，利用Transformer自编码器学习正常模式分布，并在判别器中引入对比损失以增强对正常模式的判别能力，实验表明该方法在四个真实金融数据集上显著优于现有主流方法，有效提升了异常检测的准确性和鲁棒性。本文提出的方法包含四个核心模块：（1）数据增强模块：采用基于几何分布的随机掩码策略，通过马尔可夫链生成二值掩码矩阵，对多变量时间序列进行局部数据扰动，以扩大输入空间并提升模型泛化能力；（2）生成器模块：构建基于Transformer的自编码器，编码器通过多头自注意力机制提取时间序列的长程依赖特征，解码器为两层MLP，用于重构原始序列，目标是学习正常模式的潜在分布；（3）判别器模块：采用三层MLP结构，以原始数据为真实样本、重构数据为假样本，通过对抗训练约束生成器输出，使其逼近真实分布，判别损失采用标准GAN的二元交叉熵形式；（4）对比学习模块：在每个批次中，将两组重构序列的潜在表示视为正样本对，其余为负样本对，通过对比损失函数拉近正样本对距离、推远负样本对距离，增强判别器对正常模式的泛化能力。整个模型联合优化四个模块，以无监督方式训练，最终通过重构误差与对比表示的联合判别实现异常检测。

#### A Comprehensive Survey of Time Series Forecasting: Architectural Diversity and Open Challenges
- **Authors**: Jongseon Kim, Hyungjoon Kim, HyunGi Kim, Dongjun Lee, Sungroh Yoon
- **Year**: 2024
- **Abstract**: 为解决时间序列预测领域中模型架构单一和开放性挑战（如通道依赖、分布偏移、因果性等）的问题，本文系统综述了从传统统计方法到深度学习架构（MLP、CNN、RNN、GNN、Transformer）以及新兴模型（扩散模型、Mamba、基础模型）的发展脉络，通过横向比较各类架构的优劣与演进趋势，揭示了架构多样化是当前研究的核心方向，并系统梳理了应对关键挑战的最新方法，为领域研究者提供了全面的理论框架与实践指南。本文并非提出一种新的预测模型，而是一篇全面的综述性研究。其方法包括：（1）系统梳理时间序列预测的历史演进，从统计模型（ARIMA、指数平滑）到传统深度学习模型（MLP、CNN、RNN、GNN），再到Transformer的兴起与局限；（2）深入分析新兴架构的崛起，包括Transformer的改进（如分块注意力、跨维注意力）、传统模型的复兴（MLP/CNN/RNN在新设计下超越Transformer）、扩散模型（通过条件扩散过程建模时间序列分布）、Mamba模型（基于状态空间模型SSM的高效长序列建模）以及基础模型（预训练大模型在TSF中的迁移应用）；（3）系统归纳时间序列预测面临的五大开放性挑战：通道依赖（多变量间相关性建模）、分布偏移（训练与测试数据分布不一致）、因果性（区分相关性与因果关系）、特征提取（有效捕捉趋势、季节性、周期性等）以及评估指标的局限性，并综述针对每个挑战的最新解决方案；（4）通过大量图表和表格（如模型分类、数据集对比、评估指标汇总）对各类方法进行结构化对比与分析，构建了当前TSF研究的全景图，旨在降低新研究者入门门槛并为资深研究者指明未来方向。

#### Large Language Models for Time Series: A Survey
- **Authors**: Xiyuan Zhang, Ranak Roy Chowdhury, Rajesh K. Gupta, Jingbo Shang
- **Year**: 2024
- **Abstract**: 为解决大型语言模型（LLM）在处理数值型时间序列数据时面临的模态鸿沟问题，本文系统综述了五类方法（直接提示、时间序列量化、对齐、视觉作为桥梁、工具集成），通过将时间序列转换为文本、离散令牌、对齐嵌入、视觉表示或调用外部工具，实现LLM在气候、医疗、金融等领域的时序分析任务，显著提升了零样本和少样本场景下的性能与泛化能力。本文提出了一种系统性分类框架，将LLM应用于时间序列分析的方法分为五类：（1）直接提示（Prompting）：将时间序列数值直接作为文本输入LLM，如PromptCast使用模板将时间序列转为自然语言问题；（2）时间序列量化（Quantization）：通过VQ-VAE、K-Means或频率分析将连续时间序列离散化为令牌，如Chronos将数值分箱为离散标签，TDML将价格变化编码为文本类别；（3）对齐（Aligning）：训练时间序列编码器，使其嵌入与语言模型语义空间对齐，如ETP使用对比学习对齐ECG信号与临床文本，GPT4TS将时间序列分块嵌入后输入冻结的GPT-2；（4）视觉作为桥梁（Vision as Bridge）：将时间序列绘制为图像，利用视觉-语言模型（如CLIP、LLaVA）作为中介，如CLIP-LSTM将股价图输入CLIP生成特征；（5）工具集成（Tool Integration）：使用LLM生成代码或API调用（如CTG++生成扩散模型损失函数，ToolLLM调用金融API），间接增强时序分析能力。本文还整合了多模态数据集（如PTB-XL、Ego4D、PIXIU）并分析了各方法在数据需求、模型规模、效率和优化难度上的权衡，为未来研究提供全面的技术路线图。

#### Anomaly Detection In Time Series Data Using Reinforcement Learning, Variational Autoencoder, and Active Learning
- **Authors**: Bahareh Golchin, Banafsheh Rekabdar
- **Year**: 2023
- **Abstract**: 为解决时间序列数据中异常检测依赖大量标注数据、难以识别新型异常的问题，本文提出一种结合深度强化学习（DRL）、变分自编码器（VAE）和主动学习的RLVAL方法，通过LSTM建模时序依赖，利用VAE生成异常评分作为内在奖励，结合标注数据的外在奖励与主动学习的边际采样策略，实现对未知异常类别的高效探索与检测，显著提升检测性能并减少对标注数据的依赖。本文提出RLVAL方法，其核心流程如下：1）使用LSTM网络建模时间序列的长期依赖关系，作为DRL智能体的状态表示；2）构建变分自编码器（VAE），仅用正常数据训练，通过重构误差衡量样本异常程度，将该误差标准化后作为内在奖励（r₂）；3）设计深度Q网络（DQN）作为强化学习智能体，其奖励函数由外在奖励（r₁）和内在奖励（r₂）组成：r₁在智能体对标注异常数据采取'异常'动作时给予+1，对未标注数据采取'正常'动作时给予0，其他情况为-1；r₂由VAE的重构误差计算得出，激励智能体探索高重构误差的未标注样本；4）引入主动学习模块，采用边际采样策略（Margin Sampling）从未标注数据池中选择模型预测最不确定的样本（即两类Q值差异最小的样本），交由人工标注后反馈至训练集，以提升模型判别能力；5）通过经验回放机制缓解样本相关性，使用目标网络稳定训练，最终实现对已知和未知异常类别的联合检测，显著优于传统方法。

#### Hybrid boosted attention-based LightGBM framework for enhanced credit risk assessment in digital finance
- **Authors**: Chengwei Ying, Anlu Shi, Xiongyi Li
- **Year**: 2024
- **Abstract**: 为解决数字金融中信贷风险评估面临的高维数据、类别不平衡和模型可解释性差等问题，本文提出了一种混合增强注意力机制的LightGBM框架（HBA-LGBM），通过多阶段特征选择、注意力特征增强、混合提升机制和合成数据增强与代价敏感学习相结合的不平衡学习策略，显著提升了违约预测的准确性和模型稳定性，在LendingClub数据集上实现了RMSE=11.53、MAPE=4.44%和R²=0.998的优异性能。本文提出的HBA-LGBM框架包含四个核心模块：(1) 多阶段特征选择机制：基于梯度大小动态筛选高影响力特征，采用加权采样减少计算开销，提升特征效率；(2) 注意力特征增强层：通过可学习的Query、Key、Value矩阵对输入特征进行非线性变换，计算特征注意力权重，动态加权重组特征表示，增强关键风险因子的贡献；(3) 混合提升机制：将LightGBM与多层感知机（MLP）结合，通过可学习权重λ线性融合两者的预测结果，λ由输入特征通过Sigmoid函数自适应计算，兼顾树模型的效率与神经网络的非线性建模能力；(4) 不平衡学习策略：采用基于类频率倒数平方根的代价敏感损失函数，并结合改进的SMOTE算法（引入高斯噪声和邻近扰动）生成多样化的少数类样本，同时添加正则化项约束合成样本与原始样本的相似性，防止数据失真。最终模型在LendingClub大规模数据集上训练，通过五折交叉验证和参数调优（如学习率0.1、最大深度8、叶子节点数25等）实现最优性能。

#### A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges
- **Authors**: Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M. Mulvey, H. Vincent Poor, Qingsong Wen, Stefan Zohren
- **Year**: 2024
- **Abstract**: 为系统梳理大语言模型（LLM）在金融领域的应用进展、技术优势与关键挑战，本文通过分类综述 linguistic tasks、sentiment analysis、financial time series、financial reasoning 和 agent-based modeling 等六大核心方向，整合了主流模型、数据集与基准，并深入分析了数据偏差、伦理风险与可解释性等现实障碍，为金融行业智能化转型提供了全面的理论与实践参考。本文并未提出新的模型或算法，而是一篇系统性综述论文。其方法包括：（1）系统收集并分类整理近五年来LLM在金融领域的研究文献，按六大应用方向（语言任务、情感分析、金融时序、金融推理、基于智能体的建模、其他应用）进行结构化梳理；（2）详细分析代表性金融专用LLM（如FinBERT、BloombergGPT、FinGPT、XuanYuan 2.0等）的架构、预训练策略与微调方法，对比zero-shot与fine-tuning的应用场景；（3）汇总公开可用的金融数据集、代码库与基准测试平台，为研究者提供资源指南；（4）深入探讨金融LLM部署中的独特挑战，如前瞻偏差、数据污染、法律合规、可解释性与隐私问题；（5）通过对比现有相关综述（如Lee et al.、Li et al.等），突出本综述在应用广度、挑战深度与实践导向上的创新性，最终构建一个连接学术研究与工业实践的全景式框架，推动LLM在金融领域的负责任创新与落地。

#### RMT-Net: Reject-Aware Multi-Task Network for Modeling Missing-Not-At-Random Data in Financial Credit Scoring
- **Authors**: Qiang Liu, Yingtao Luo, Shu Wu, Zhen Zhang, Xiangnan Yue, Hong Jin, Liang Wang
- **Year**: 2023
- **Abstract**: 为解决金融信用评分中因拒绝样本无标签导致的缺失不随机偏倚问题，本文提出RMT-Net，通过多任务学习框架联合建模拒绝/批准任务与违约/非违约任务，利用拒绝概率动态控制信息共享权重，使模型能有效利用拒绝样本信息提升对批准与拒绝样本的违约预测准确性，实验表明其相比传统方法平均提升47.9%，相比最优基线平均提升11.9%。RMT-Net由四部分组成：(1)嵌入层将原始特征转换为稠密向量；(2)拒绝/批准预测网络（R/A-Net）输出每个样本的拒绝概率；(3)违约/非违约预测网络（D/N-Net）用于预测违约概率，其每一层的隐层表示通过门控网络融合R/A-Net的输出，门控权重由拒绝概率通过可学习参数σ(α·p_i^(t) + β)动态计算，拒绝概率越高，从R/A-Net共享的信息越多；(4)损失函数由两部分组成：R/A-Net的二元交叉熵损失（对所有样本）和D/N-Net的二元交叉熵损失（仅对批准样本，通过(1−r_i)掩码），二者加权求和。RMT-Net++进一步扩展为支持多个拒绝/批准策略，为每个策略独立训练一个R/A-Net，并在D/N-Net中对所有策略的门控输出进行加权求和，实现多策略下的联合建模。

#### Advanced Progress in Optimized Generative Adversarial Network Applications Across Domains: A Comprehensive Survey
- **Authors**: Sudha Senthilkumar, P. Kumaresan, K. Brindha, Yu-Chen Hu
- **Year**: 2025
- **Abstract**: 为解决生成对抗网络（GAN）在电力需求预测、供应链库存管理、医学图像合成、农业数据增强和投资组合优化等跨领域应用中的稳定性、模式坍塌和数据稀缺等问题，本文通过系统性综述分析了多种优化GAN架构（如cGAN、WGAN、CycleGAN、StyleGAN等）及其与LSTM、CNN、DenseNet121、AlexNet等网络的结合方法，通过改进损失函数（如Wasserstein损失、二元交叉熵）、引入数据增强与混合优化算法，显著提升了生成数据的真实性与模型收敛性，实现了在各领域更准确的预测与更高效的决策支持。本文通过系统性文献综述方法，基于PRISMA框架，从IEEE Xplore、Springer、Elsevier等数据库中筛选2018–2023年间51篇高质量全文文献，涵盖GAN架构优化与跨领域应用。首先，全面梳理了条件GAN（cGAN）、Wasserstein GAN（WGAN）、CycleGAN、StyleGAN、Progressive GAN等主流变体的结构与优化机制；其次，分析了生成器与判别器的网络设计（如LSTM、CNN、MLP、DenseNet121、AlexNet）及损失函数（如最小最大损失、Wasserstein距离、梯度惩罚）对训练稳定性的影响；接着，针对五大核心应用领域展开深入分析：（1）电力需求预测中，采用cDCGAN、TimeGAN、CWGAN-GP等模型结合气象与时间特征，降低MAE与RMSE；（2）供应链库存管理中，利用E-commerce GAN与V-GAN生成真实订单分布，优化库存策略；（3）医学图像合成中，结合WGAN、DCGAN与Whale Optimization Algorithm生成高质量CT/MRI图像，提升疾病检测准确率；（4）农业数据增强中，使用Dual GAN与SAM-GAN生成高分辨率病害图像，训练DenseNet121实现高精度分类；（5）投资组合优化中，采用LSTM-GAN生成多因子金融时序数据，通过MLP判别器实现股票趋势预测。最后，本文总结了模式坍塌、收敛不稳定、评估指标缺失等挑战，并提出未来研究需融合物理模型、多模态数据与自适应优化算法以提升泛化能力。

#### Stripping the Swiss discount curve using kernel ridge regression
- **Authors**: Nicolas Camenzind, Damir Filipović
- **Year**: 2024
- **Abstract**: 为解决瑞士国债市场中无风险贴现曲线估计的鲁棒性与灵活性不足问题，本文提出基于核岭回归（KR）的方法，通过在再生核希尔伯特空间中最小化定价误差与曲线平滑性的加权和，实现数据驱动、可解释且优于传统方法（如Smith–Wilson、SST和SNB）的曲线拟合与外推效果。本文提出的方法为核岭回归（KR），其核心是将贴现曲线g(x)建模为再生核希尔伯特空间G_{α,δ}中的函数，该空间由[0,∞)上二次可微且g(0)=1的函数构成，其范数由平滑性惩罚项||g||_{α,δ}^2 = ∫₀^∞ [δ(g'(x))² + (1−δ)(g''(x))²] e^{αx} dx定义。目标函数为最小化加权定价误差∑ω_i(P_i − P_i^g)²与平滑性惩罚项λ||g||_{α,δ}^2之和。通过再生核k(x,y)和Representer定理，该无限维优化问题可转化为有限维线性系统：解为ĝ(x) = 1 + ∑_{j=1}^N k(x,x_j)β_j，其中β = C^⊤(CKC^⊤ + Λ)^{-1}(P − C1)，Λ为对角权重矩阵。超参数λ、α、δ通过交叉验证选择，权重ω_i采用修正久期加权以近似YTM误差。KR方法可严格包含Smith–Wilson方法作为特例（当λ=0且α=0时），并支持通过设置无穷权重ω_i=∞强制拟合外部收益率观点，且能生成置信带以反映数据稀疏区域的不确定性。

#### Combining Financial Data and News Articles for Stock Price Movement Prediction Using Large Language Models
- **Authors**: Ali Elahi, Fatemeh Taghvaei
- **Year**: 2024
- **Abstract**: 为解决金融市场上股票价格走势预测中多源异构数据（结构化财务数据与非结构化新闻文本）融合分析的难题，本文提出一种基于大语言模型（LLM）的零样本至四样本提示方法，通过检索增强生成技术从新闻文章中提取相关片段并结合财务指标构建提示输入，利用GPT和LLaMA系列模型进行二分类预测，实现了3个月和6个月预测周期下加权F1分数分别达59.2%和59.1%的效果。首先，收集20家高交易量公司从2021年至2024年的财务报表数据（10-K文件）和新闻文章，提取关键财务指标（如总收入、净利润、自由现金流等）和历史股价动量（过去6、12个月变化）。其次，使用基于OpenAI嵌入的提取式摘要方法，对新闻文章按三句块进行相关性检索与筛选，确保与公司及日期匹配。接着，构造包含公司行业与产品描述、最多6个精选新闻摘要、最近四个季度财务数据和二分类问题的提示模板（如‘公司股价在未来3个月是上涨还是下跌？’），并以未来3或6个月的股价涨跌（1/0）作为标签。然后，在零样本、两样本和四样本设置下，将提示输入至GPT-3.5、GPT-4、LLaMA2和LLaMA3等预训练大语言模型，使其直接输出[UP]或[DOWN]预测结果。最后，采用加权F1分数和马修斯相关系数评估模型性能，发现GPT-3.5（零样本）和GPT-4（两样本）在3个月和6个月预测中表现最优，且增加样本数未显著提升性能，可能因提示过长（平均2500令牌）导致模型混淆。

#### Regression estimation for continuous-time functional data processes with missing at random response
- **Authors**: Mohamed Chaouch, Naâmane Laïb
- **Year**: 2024
- **Abstract**: 为解决连续时间函数型数据中响应变量随机缺失（MAR）下的非参数回归估计问题，本文提出了一种基于核平滑的广义回归估计器，通过利用观测数据构建加权积分估计量，并结合连续时间遍历过程的渐近性质，实现了点态与一致几乎必然收敛率的理论保证，同时提供了置信区间构建方法，并成功应用于金融对数收益预测与家庭用电需求插补。本文提出了一种适用于连续时间函数型数据且响应变量缺失为MAR机制的非参数核回归估计器。首先，定义广义回归函数 m_ψ(x,y) = E[ψ(Y,y)|X=x]，其中ψ可为均值、分布函数或分位数函数；其次，针对缺失机制，利用指示变量ζ_t构建核加权估计量：当∫ζ_t Δ_t(x) dt ≠ 0时，估计器为 ∫ζ_t ψ(Y_t,y) Δ_t(x) dt / ∫ζ_t Δ_t(x) dt，其中Δ_t(x) = K(d(X_t,x)/h_T)为核函数，h_T为带宽；为处理连续时间数据，引入σ-代数序列和遍历过程假设（A1–A3），并利用鞅差分工具推导渐近性质；进一步，推导出估计量的点态与一致几乎必然收敛速率，以及渐近均方误差和分布，用于构建置信区间；最后，提出插补方法：用估计器预测缺失值，构造插补响应变量~ψ(Y_t,y) = ζ_t ψ(Y_t,y) + (1−ζ_t) m̂_ψ,T(X_t,y)，并基于插补数据重新估计回归函数。该方法在理论上保证了在无混合条件、适用于长记忆与Bernoulli移位过程的广泛场景下的收敛性，并通过仿真与金融、电力应用验证了其有效性。

#### Foundation Models for Time Series Analysis: A Tutorial and Survey
- **Authors**: Yuxuan Liang, Haomin Wen, Yuqi Nie, Yushan Jiang, Ming Jin, Dongjin Song, Shirui Pan, Qingsong Wen
- **Year**: 2024
- **Abstract**: 为解决时间序列分析中模型泛化能力不足和任务适配效率低的问题，本文系统综述了时间序列基础模型（TSFMs）的最新进展，提出了一种以方法论为核心的分类体系，涵盖模型架构、预训练技术、适配方法和数据模态，阐明了TSFMs如何通过大规模预训练与灵活适配实现跨域通用时间序列理解与预测，显著提升了零样本和少样本场景下的性能。本文并未提出新的具体模型，而是构建了一个系统性的方法论分类框架来综述和分析时间序列基础模型（TSFMs）。首先，将时间序列数据分为标准时间序列、时空序列、轨迹与事件四类；其次，从方法论角度将TSFMs划分为四大核心组件：（1）模型架构，包括基于Transformer（编码器-解码器、编码器-only、解码器-only）、非Transformer（MLP、CNN、RNN）和扩散模型；（2）预训练技术，分为全监督、自监督（生成式、对比式、混合式）和跨模态预训练（如使用LLM、VLM）；（3）适配方法，包括零样本直接使用、微调（全模型或部分组件）、提示工程（静态/可学习提示）和时间序列分词（如分块、归一化、分解）；（4）数据模态，涵盖单模态（仅时间序列）和多模态（时间序列+文本、图像、音频等）。通过该框架，本文系统梳理了代表性工作如Lag-Llama、TimeGPT-1、TS2Vec、Time-LLM、ClimaX、DiffTraj等，揭示了TSFMs通过大规模预训练获取通用时序表征，并通过灵活适配策略实现跨任务、跨领域高效迁移的核心机制，为后续研究提供了统一的分析视角与发展方向。

#### VAE-INN: Variational Autoencoder with Integrated Neural Network Classifier for Imbalanced Credit Scoring, Utilizing Weighted Loss for Improved Accuracy
- **Authors**: Dalia ATIF
- **Year**: 2025
- **Abstract**: 为解决信贷评分中类别不平衡导致的Type II错误（漏判违约者）问题，本文提出VAE-INN方法，通过将带加权损失的神经网络分类器集成到变分自编码器（VAE）的潜在空间中，联合优化特征提取与分类，使潜在空间均衡表征多数与少数类，并利用类别权重和缩放因子α强化对违约样本的学习，显著降低误判率并提升金融风险识别能力。本文提出VAE-INN方法，其核心是将一个带加权损失的神经网络分类器直接集成到变分自编码器（VAE）的潜在空间中，实现特征提取与分类的端到端联合优化。具体步骤如下：（1）使用VAE编码器将高维信贷数据映射到低维潜在空间，输出每个样本的均值μ和标准差σ，通过重参数化技巧采样潜在变量z = μ + σ ⊙ ε；（2）将采样得到的潜在变量z同时输入解码器（用于重构原始输入）和集成的神经网络分类器（用于预测违约概率）；（3）构建复合损失函数：L_loss = L_recon + L_KL + α·L_class，其中L_recon为重构误差（均方误差），L_KL为潜在分布与标准正态分布的KL散度（正则化项），L_class为带类别权重的二元交叉熵损失，权重w_i = N / (2·n_l)，N为总样本数，n_l为类别l的样本数；（4）引入超参数α调节分类损失的相对重要性，通过交叉验证优化α值，确保在提升分类性能的同时不损害潜在空间的重构质量；（5）通过反向传播同时优化编码器参数φ、解码器参数θ和分类器参数ψ，使潜在空间在保持数据重构能力的同时，被分类目标主动拉向两类均衡分布，尤其强化对少数类（违约者）的判别能力；（6）最终模型以最小化Type II错误（漏判违约）为目标，在真实不平衡信贷数据集上验证，显著优于传统多阶段方法。

#### Deep learning for time series forecasting: a survey
- **Authors**: Xiangjie Kong, Zhenghao Chen, Weiyao Liu, Kaili Ning, Lechao Zhang, Syauqie Muhammad Marier, Yichen Liu, Yuhao Chen, Feng Xia
- **Year**: 2025
- **Abstract**: 为解决深度学习时间序列预测模型缺乏系统性架构分类、特征提取方法综述和数据集汇总的问题，本文提出了一种动态分类框架，系统梳理了编码器-解码器、Transformer、生成对抗网络等五大模型架构范式，结合时间序列的趋势、季节性和残差成分分析特征提取方法，并整合多领域数据集，全面总结了当前挑战与未来研究方向，为DTSF领域提供了首个结构化、多维度的综述体系。本文首先定义了时间序列预测（TSF）的基本概念与任务分类（单变量/多变量、短期/长期），并回顾了传统统计模型（如ARIMA、指数平滑）的局限性；随后，提出一种动态分类框架，将深度学习模型划分为显式结构（编码器-解码器、Transformer、生成对抗网络）与隐式结构（集成模型、级联模型）两大类，系统梳理了如Informer、Autoformer、FEDformer、TimeGAN等代表性模型的架构设计、优缺点与应用场景；接着，从时间序列的三要素（趋势、季节性、残差）出发，系统总结了特征增强方法，包括维度分解、时频变换、预训练和分块（patch-based）分割；同时，构建了涵盖能源、交通、金融、医疗等领域的TSF数据集汇编；最后，归纳了当前面临的挑战，如长序列建模、外部因素融合、可解释性不足、数据异构性等，并提出未来研究方向，包括模型轻量化、跨域迁移、与大语言模型（LLM）结合、因果推理增强等，形成了一套完整的DTSF研究体系。

#### Data Augmentation Strategies for Improving Time Series Classification Accuracy
- **Authors**: Pongpanod Sankosik, Chotirat Ratanamahatana
- **Year**: 2024
- **Abstract**: 为解决时间序列分类中数据稀缺、类别不平衡和噪声干扰等问题，本文系统评估了多种数据增强技术（如wDBA、SMOTE、窗口扭曲等）对MiniRocket分类器在85个UCR数据集上的性能影响，发现增强效果高度依赖数据集特性，其中wDBA在44个数据集上显著提升准确率，但整体平均性能略低于基线，强调了数据增强需采用数据集特定策略。本研究使用MiniRocket作为基础分类器，在85个UCR时间序列数据集上系统评估了8种数据增强方法：幅度扭曲（Magnitude Warping）、窗口扭曲（Window Warping）、傅里叶变换（Fourier Transform）、短时傅里叶变换（STFT）、SMOTE、判别引导扭曲（DGW）、加权动态时间规整质心平均（wDBA）和双指数平滑（Double Exponential Smoothing）。每种方法在多个参数组合下运行（共165次/数据集），使用预定义的训练/测试划分，以分类准确率为评估指标。通过对比增强后与原始数据的准确率差异，分析不同数据集特征（如类型、长度、大小、噪声、不平衡性）对增强效果的影响。结果发现，wDBA在44个数据集上优于基线，尤其在小规模、有噪声和模拟类数据上表现突出；而ECG数据集几乎无提升，大型数据集增强效果有限。最终结论是：数据增强并非普遍有效，必须根据数据特性选择合适方法，推荐采用数据集特定的增强策略。


