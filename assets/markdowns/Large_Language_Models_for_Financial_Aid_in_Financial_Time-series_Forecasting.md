# Large Language Models for Financial Aid in Financial Time-series Forecasting

Md Khairul Islam 1, Ayush Karmacharya 1, Timothy Sue 1, Judy Fox 1,2 1 Computer Science Department, University of Virginia 2 School of Data Science, University of Virginia Charlottesville, USA Email : $\{ \mathrm { m i } 3 \mathrm { s e }$ , psb7wm, und2yw, cwk9mp}@virginia.edu

Abstract—Considering the difficulty of financial time series forecasting in financial aid, much of the current research focuses on leveraging big data analytics in financial services. One modern approach is to utilize ”predictive analysis”, analogous to forecasting financial trends. However, many of these time series data in Financial Aid (FA) pose unique challenges due to limited historical datasets and high dimensional financial information, which hinder the development of effective predictive models that balance accuracy with efficient runtime and memory usage. Pre-trained foundation models are employed to address these challenging tasks. We use state-of-the-art time series models including pre-trained LLMs (GPT-2 as the backbone), transformers, and linear models to demonstrate their ability to outperform traditional approaches, even with minimal (”few-shot”) or no fine-tuning (”zero-shot”). Our benchmark study, which includes financial aid with seven other time series tasks, shows the potential of using LLMs for scarce financial datasets.

Index Terms—Financial Aid, Time Series Forecast, Deep Learning, Foundation Models, Large Language Models

# I. INTRODUCTION

The advancement of AI has taken over many domains including the field of financial market [1] and big data [2]. In particular, financial time series forecasting has improved significantly from using statistical models to machine learning [3] and then deep learning [4]. These financial forecasting areas include currency exchange rate [5], [6], stock market [2] [5] [4], commodity prices [7] [8] and more. Pre-trained foundation models, such as large language models (LLMs) have driven the progress in Natural Language Processing (NLP) and Computer Vision (CV). Foundation models like GPT [9], and Vision Transformer [10] can perform well on a diverse range of tasks in few-shot (little training) or zero-shot (no training) learning. This enables applications where historical data is limited or mostly missing.

Financial time series forecasting (FTSF) is an important domain that needs more attention among the multivariate time series forecasting tasks. Previous works on FTSF rely heavily on machine learning [3] or traditional deep learning [4] methods. With some recent works on multi-modal FTSF [11] [12]. Financial aid (FA) is crucial to many students’ educational journeys, providing the resources needed to pursue academic dreams while fostering educational equity. However, the process, including tasks like processing the free application for Federal Student Aid (FAFSA), is often manual, time-consuming, and susceptible to errors. Access to historical datasets is limited to yearly intervals and is subject to changes in policy.

We describe and evaluate LLM-based foundation models in the FTSF domain using 8 deep-learning models and compare especially the financial aid with 7 other financial datasets. Our research questions are,

• $Q l$ : Which models are better as few-shot learners? • Q2: Can pre-trained LLMs perform zero-shot learning?

Answering these questions will help us better understand the current advancement of LLMs for financial time series forecasting. In summary, our contributions are,

• Collect eight datasets from four financial domains (Stock, Commodity, Currency, Institution) for over 10 years. • Benchmark five state-of-the-art time series deep learning models, and three LLM-based foundation models on these datasets. • Open source code and datasets at GitHub 1 to facilitate full reproducibility and further research in this domain.

# II. METHODOLOGY

# A. Problem Statement

Given the input dataset, $X \in \mathcal { R } ^ { F \times T }$ , $T$ denotes the total timesteps in days and $F$ input features (including past targets and other features). With a lookback window of $L$ past days, the input at time $t$ is $X _ { t } = X _ { t - ( L - 1 ) : t }$ which contains inputs of the last $L$ days. Given this input $X _ { t }$ , the model $f$ predicts the targets $O$ (e.g. stock prices) for the next $\tau _ { m a x }$ days. The target output $y _ { t }$ at time $t$ can be expressed as,

$$
\begin{array} { r l } & { \hat { y } _ { t } = f ( X _ { t } ) , \mathrm { w h e r e , } } \\ & { X _ { t } = x _ { t - ( L - 1 ) : t } = [ x _ { t - ( L - 1 ) } , x _ { t - ( L - 2 ) } , \cdots , x _ { t } ] } \\ & { \quad ~ = \{ x _ { f , l , t } \} , \ f \in \{ 1 , \cdots , F \} , \ l \in \{ 1 , \cdots , L \} } \end{array}
$$

For the financial aid data, funds allocated to each state are a time series with $T$ years (2004 to 2020). A lookback $L$ of 10 years is used to predict funds $( O )$ for the next year $\tau _ { m a x } = 1$ ). For all other datasets, we have daily inputs for 10 years. With a lookback window of the past 96 days $L = 9 6$ ), we predict the targets for the next 24 days $\tau _ { m a x } = 2 4$ ).

TABLE I: Datasets overview. Time series indicates the number of target time series (i.e., channels). Input features are past observations and dataset size is depicted as training, validation, and test.   

<table><tr><td>Domain</td><td>Dataset</td><td>Date</td><td>Frequency</td><td>Time series</td><td>Lookback</td><td>Horizon</td><td>Size</td></tr><tr><td>Institution</td><td>Financial Aid</td><td>2004-2020</td><td>Yearly</td><td>56</td><td>10</td><td>1</td><td>(137, 92, 92)</td></tr><tr><td rowspan="3">Stock</td><td>S&amp;P 500</td><td>Sep 1,2014- Aug 29,2024</td><td>Daily</td><td>4</td><td>96</td><td>24</td><td>(1903,231, 229)</td></tr><tr><td>Apple</td><td>Sep 2,2014 - Aug 29,2024</td><td>Daily</td><td>5</td><td>96</td><td>24</td><td>(1893,230, 228)</td></tr><tr><td>Microsoft</td><td>Sep 3,2014 - Aug 30,2024</td><td>Daily</td><td>4</td><td>96</td><td>24</td><td>(1893,230,228)</td></tr><tr><td rowspan="3">Commodity</td><td>Crude Oil</td><td>Sep 1,2014 - Aug 29, 2024</td><td>Daily</td><td>4</td><td>96</td><td>24</td><td>(1893,230,228)</td></tr><tr><td>Gold</td><td>Sep 1,2014 - Aug 29,2024</td><td>Daily</td><td>4</td><td>96</td><td>24</td><td>(1893,230, 228)</td></tr><tr><td>Natural Gas</td><td>Sep 1,2014 - Aug 29,2024</td><td>Daily</td><td>4</td><td>96</td><td>24</td><td>(1893,230, 228)</td></tr><tr><td>Currency</td><td>Exchange Rate</td><td>Aug 1,2014 - Aug1,2024</td><td>Daily</td><td>7</td><td>96</td><td>24</td><td>(1861,225,224)</td></tr></table>

# B. Dataset

We use the following financial datasets: (1) Financial Aid: Financial aid distributed to each US state by the Government to support student education and collected from years 2004 to 2020 [13]. Details of available features are in Table II and the yearly aggregated aid in Fig 1. (2) Stock Market [5] [12] [4] [14]: Includes the daily stock prices (Close, Open, High, Low) and volumes for each of the following stocks up to 10 years from the NASDAQ database: S&P 500 (SPX), Microsoft Corporation (MSFT), and Apple Inc (AAPL); (3) Commodities [7] [8]: Contains data (Close, Open, Volume, High, Low) on different kinds of raw materials such as Natural Gas, Crude Oil and Gold. (4) Currency Exchange Rate [5] [6]: The currency units per U.S. dollar reported daily by the issuing central bank (rates are not recorded on the weekends and certain holidays). This data covers the following currencies: Australian Dollar (AUD), Canadian Dollar (CAD), Chinese yuan (CNY), Euro (EUR), Indian rupee (INR), Japanese yen (JPY), and the U.K. pound (GBP);

![](images/148d2c1c15d9f69346aecc07ff34f22f5bb00b728283e1d87f8a38c7cddbfe69.jpg)  
Fig. 1: Financial Aid data aggregated at the state level from 2004 to 2020 (17 years), in billions of US dollars. Access to historical datasets is limited to yearly intervals.

The statistics are in Table I. The train, validation, and test split follows the 8:1:1 ratio, the validation set follows the train set, then the test set. The data is standard normalized before passing to the model. The few missing values $( < 1 \% )$ are imputed using last-seen valid values.

# C. Models

We use the following time series models in our work. The models are chosen based on their popularity and recently

TABLE II: List of available features in financial aid [13]. Aid is given based on financial needs, academic merit, or both. The sub-categories are simplified and describe multiple features.

<table><tr><td>Category</td><td>Sub-category</td><td>Description</td></tr><tr><td rowspan="6">Need, Merit, both</td><td>Identifier</td><td>State id and name abbreviation.</td></tr><tr><td>Number</td><td>Total students receiving the award.</td></tr><tr><td>Public/Private</td><td>Whether the funds can be used for public or private sectors and how long (2 or 4 years).</td></tr><tr><td>Flags</td><td>O or1based on whether the aid falls in a particular category.</td></tr><tr><td>Program</td><td>Aid program with the most generous eligibility criteria.</td></tr><tr><td>Notes</td><td>Related text.</td></tr><tr><td>Threshold</td><td>GPA,SAT,income,and other academic or financial limits to qualify for the aid.</td></tr><tr><td>Time</td><td>Year</td><td>Fiscal or academic year.</td></tr><tr><td>Target</td><td>Amount</td><td>Aid amount received by the students.</td></tr></table>

published work. We focus on point forecasting in our work. A high-level overview of how pre-trained LLMs are fine-tuned for custom datasets is illustrated in Figure 2.

1) LLM Foundation Models: The LLM-based foundation models are selected based on their versatility in time series forecasting. We use the configurations from [15]. The pre-trained foundation models are frozen except for the last layer when fine-tuning. These models use a pre-trained GPT-2 [9] as the LLM backbone. We select the following recent models: (1) TimeLLM [16] (2) CALF [17] (3) GPT4TS (One Fits All, [18]).

2) Traditional Models: We choose the following recent non-pre-trained models: (1) DLinear [19] (2) iTransformer [20] (3) TimesNet [6] (4) PatchTST [21] (5) TimeMixer [22]. Most of these models are Transformer-based and have shown great performance in capturing temporal patterns.

# $D$ . Implementation Details

We use the PyTorch framework and follow [6] [15] to implement our experiments. Each experiment runs three times with different random seeds (648, 506, 608), and the average results are presented. Following [18] [17] we use the pre-trained GPT2 as the backbone for the LLMs and only fine-tune the output layer during training shown in Fig. 2. The traditional models are trained from scratch. We use the Adam optimizer with a learning rate 1e-3 and a dropout rate 0.1. The GPT4TS [18] model uses the L1 loss. The CALF [17] model uses the weighted average of task, feature, and logit loss using the L1 loss. The other models use the Mean Squared Error loss. Models are trained for 10 epochs max, with batch size 32. The experiments were run on an NVIDIA 2080Ti GPU with $^ { 1 1 \mathrm { G B } + }$ memory with 32GB RAM. Following [22] [17] [14], we use Mean Square Error (MSE) and Mean Absolute Error (MAE) as evaluation metrics. Lower is better for these metrics.

TABLE III: Q1. Few-shot learning performance with $10 \%$ training data. TimeLLM and PatchTST outperform the other models. The best and the second best results are in bold and underlined.   

<table><tr><td>Method</td><td colspan="2">DLinear[19]</td><td colspan="2">PatchTST [21]</td><td colspan="2">TimesNet [6]</td><td colspan="2">TimeMixer [22]</td><td colspan="2">iTransformer [20]</td><td colspan="2">TimeLLM [23]</td><td colspan="2">CALF[17]</td><td colspan="2">GPT4TS[18]</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSEMAE</td><td></td><td>MSE</td><td>MAE</td></tr><tr><td>Financial Aid</td><td>2.94</td><td>1.36</td><td>2.23</td><td>0.95</td><td>2.17</td><td>1.04</td><td>3.04</td><td>1.36</td><td>2.22</td><td>1.14</td><td>2.06</td><td>1.08</td><td>1.64</td><td>0.89</td><td>2.36</td><td>1.20</td></tr><tr><td>S&amp;P 500</td><td>2.08</td><td>1.14</td><td>2.03</td><td>1.14</td><td>2.43</td><td>1.18</td><td>2.49</td><td>1.25</td><td>2.19</td><td>1.19</td><td>1.94</td><td>1.14</td><td>2.37</td><td>1.19</td><td>3.07</td><td>1.40</td></tr><tr><td>Apple</td><td>2.78</td><td>1.30</td><td>2.36</td><td>1.21</td><td>3.22</td><td>1.41</td><td>3.44</td><td>1.46</td><td>3.05</td><td>1.39</td><td>3.00</td><td>1.36</td><td>2.33</td><td>1.21</td><td>2.80</td><td>1.29</td></tr><tr><td>Microsoft</td><td>2.51</td><td>1.10</td><td>2.13</td><td>1.06</td><td>3.55</td><td>1.43</td><td>2.85</td><td>1.18</td><td>2.49</td><td>1.15</td><td>2.40</td><td>1.10</td><td>2.97</td><td>1.27</td><td>3.40</td><td>1.37</td></tr><tr><td>Crude Oil</td><td>1.66</td><td>101</td><td>1.96</td><td>1.05</td><td>2.75</td><td>1.29</td><td>2.06</td><td>1.14</td><td>1.92</td><td>1.07</td><td>1.75</td><td>1.01</td><td>2.13</td><td>1.14</td><td>2.42</td><td>1.15</td></tr><tr><td>Gold</td><td>2.78</td><td>1.14</td><td>2.68</td><td>1.15</td><td>2.66</td><td>1.17</td><td>2.57</td><td>1.11</td><td>3.00</td><td>1.22</td><td>2.71</td><td>1.17</td><td>3.06</td><td>1.24</td><td>3.38</td><td>1.32</td></tr><tr><td>Natural Gas</td><td>2.20</td><td>1.16</td><td>2.48</td><td>1.24</td><td>2.55</td><td>1.23</td><td>2.91</td><td>1.34</td><td>2.12</td><td>1.12</td><td>2.36</td><td>1.20</td><td>2.30</td><td>1.17</td><td>2.17</td><td>1.13</td></tr><tr><td>Exchange</td><td>1.48</td><td>0.92</td><td>1.28</td><td>0.85</td><td>2.87</td><td>1.34</td><td>1.29</td><td>0.84</td><td>1.65</td><td>0.96</td><td>1.20</td><td>0.81</td><td>1.48</td><td>0.92</td><td>1.22</td><td>0.81</td></tr></table>

![](images/c4ae1cb8a0641d598f963c14ab4c62f1ba6d146c4e1c25c619d743ff35dcc9a4.jpg)  
Fig. 2: A high-level overview of pre-training an LLM and fine-tuning on a custom dataset (e.g. the Financial Aid dataset) for downstream tasks.

# III. EXPERIMENTS AND RESULTS

In this section, we investigate the research questions, the setup, and the results.

# A. Q1. Which models are better as few-shot learners?

Pre-trained models are preferred largely due to their generalizability and good performance in few-shot or zero-shot learning settings [24]. Since these LLMs are already trained on many datasets, they often outperform the other models when few training data are available [17] or without training [14].

Following [17], we select the last $10 \%$ training data to train the models in a few-shot learning setting.

Results. The few-shot learning results are shown in Table III. All model performance drops significantly after reducing the train data size. This is due to the models’ inability to learn enough temporal patterns from the limited input. TimeLLM performs the best overall (three best and four 2nd best cases). While PatchTST performs the 2nd best with a close margin (three best and two 2nd best cases). DLinears performance drops significantly as it is a simple linear model. However, overall the LLMs performed better in the few-shot learning.

# B. Q2. Can LLMs perform zero-shot learning in FTSF?

Real-world scenarios can often have no available past observations (i.e. new company stock in the market, newly launched product). Having zero-shot learning ability is crucial in forecasting those cases since traditional deep learning time series models are unable to train and forecast those cases. We investigate whether LLMs can effectively assist in those cases. We load the pre-trained LLMs and evaluate them on the test set without fine-tuning. Since no training is done in this part, we exclude the traditional time series models from this analysis.

TABLE IV: $Q 2$ . Zero shot performance. GPT4TS performs the best. The best and the second best results are in bold and underlined. The traditional models are excluded here since they are not pre-trained.

<table><tr><td>Method</td><td colspan="2">TimeLLM [23]</td><td>CALF [17]</td><td>GPT4TS[18]</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE MAE</td></tr><tr><td>Financial Aid</td><td>2.99</td><td>1.29</td><td>3.82 1.59</td><td>3.17 1.41</td></tr><tr><td>S&amp;P 500</td><td>5.04</td><td>1.91</td><td>3.98 1.74</td><td>3.89 1.76</td></tr><tr><td>Apple</td><td>4.17</td><td>1.61</td><td>3.36 1.44</td><td>3.05 1.36</td></tr><tr><td>Microsoft</td><td>5.13</td><td>1.81</td><td>4.12 1.62</td><td>3.96 1.59</td></tr><tr><td>Crude Oil</td><td>3.05</td><td>1.39 2.21</td><td>1.18</td><td>1.89 1.08</td></tr><tr><td>Gold</td><td>6.15</td><td>1.95 5.12</td><td>1.76</td><td>5.00 1.77</td></tr><tr><td>Natural Gas</td><td>4.09</td><td>1.61 3.27</td><td>1.43</td><td>2.96 1.35</td></tr><tr><td>Exchange</td><td>3.25</td><td>1.47</td><td>2.41 1.29</td><td>2.10 1.23</td></tr></table>

Results. Table IV shows the zero-shot results of the LLMs. Compared to $Q I$ , the results achieved here are significantly worse. Since each financial data may have distinct temporal patterns, without fine-tuning the LLMs fail to forecast them effectively. We conclude, LLMs are yet not quite effective for zero-shot learning for financial time series.

# IV. RELATED WORKS

Deep learning for time series has significantly outperformed machine learning approaches [22] [20], also in finance [4]. [5] used RNN models to forecast stock market prices, and currency exchange rates. Many recent deep learning models have been used to forecast the stock market [2] [5] [4], commodity prices [7] [8]. Foundation models in time series have recently gained significant attention [24]. Pre-trained LLMs and vision models have been enhanced for time series. [14] [25] showed the ability of LLMs to perform in zero-shot and few-shot settings in time series tasks. GPT4TS [18] leverages pre-trained language models without altering important layers. TimeLLM [16] reprograms LLM’s ability to reason with time series data by proposing a prompt-as-prefix technique. CALF [17] proposed a novel fine-tuning framework to reduce the distribution discrepancy between textual and temporal data. Chronos [26] performed significantly in probabilistic forecasting.

# V. CONCLUSION AND FUTURE WORKS

In this paper, we benchmark financial datasets from multiple domains using state-of-the-art time series models and LLM-based foundation models. Our results show that LLMs are more effective for few-shot and zero-shot learning. Especially, the few-shot and zero-shot capabilities of LLMs can be effective for financial Aid practitioners who are currently unable to apply deep learning methods due to limited data availability. We focus on point forecasting with a single modality in this work. Incorporating data from different financial modalities into time series models will be future work, and probabilistic forecasting can help the financial domain by outputting a probabilistic distribution. Our research highlights the potential of foundation LLMs in financial aid and the overall finance time series forecasting domain.

# ACKNOWLEDGMENT

This work is partly supported by NSF grant CCF-1918626 Expeditions: Collaborative Research: Global Pervasive Computational Epidemiology, and NSF Grant 2200409 for CyberTraining:CIC: CyberTraining for Students and Technologies from Generation Z.

# REFERENCES

[1] L. Cao, “Ai in finance: challenges, techniques, and opportunities,” ACM Computing Surveys (CSUR), vol. 55, no. 3, pp. 1–38, 2022.   
[2] Y. Zu, J. Mi, L. Song, S. Lu, and J. He, “Finformer: A static-dynamic spatiotemporal framework for stock trend prediction,” in 2023 IEEE International Conference on Big Data (BigData). IEEE, 2023, pp. 1460–1469.   
[3] T. Leung and T. Zhao, “Financial time series analysis and forecasting with hilbert–huang transform feature generation and machine learning,” Applied Stochastic Models in Business and Industry, vol. 37, no. 6, pp. 993–1016, 2021.   
[4] X. Li, X. Shen, Y. Zeng, X. Xing, and J. Xu, “Finreport: Explainable stock earnings forecasting via news factor analyzing model,” in Companion Proceedings of the ACM on Web Conference 2024, 2024, pp. 319–327.   
[5] K. Sako, B. N. Mpinda, and P. C. Rodrigues, “Neural networks for financial time series forecasting,” Entropy, vol. 24, no. 5, p. 657, 2022.   
[6] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “Timesnet: Temporal 2d-variation modeling for general time series analysis,” in International Conference on Learning Representations, 2023.   
[7] X. Xu and Y. Zhang, “Commodity price forecasting via neural networks for coffee, corn, cotton, oats, soybeans, soybean oil, sugar, and wheat,” Intelligent Systems in Accounting, Finance and Management, vol. 29, no. 3, pp. 169–181, 2022.   
[8] S. Zhang, J. Luo, S. Wang, and F. Liu, “Oil price forecasting: A hybrid gru neural network based on decomposition–reconstruction methods,” Expert Systems with Applications, vol. 218, p. 119617, 2023.   
[9] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.   
[10] A. Dosovitskiy, “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.   
[11] Z. Chen, L. N. Zheng, C. Lu, J. Yuan, and D. Zhu, “Chatgpt informed graph neural network for stock movement prediction,” arXiv preprint arXiv:2306.03763, 2023.   
[12] X. Yu, Z. Chen, Y. Ling, S. Dong, Z. Liu, and Y. Lu, “Temporal data meets llm–explainable financial time series forecasting,” arXiv preprint arXiv:2306.11025, 2023.   
[13] R. K., B. D., J. Ortagus, K. R., E. A., L. M., and C. J., “State financial aid dataset,” 2023. [Online]. Available: https: //informedstates.org/state-financial-aid-dataset-download   
[14] N. Gruver, M. Finzi, S. Qiu, and A. G. Wilson, “Large language models are zero-shot time series forecasters,” Advances in Neural Information Processing Systems, vol. 36, 2024.   
[15] M. Tan, M. A. Merrill, V. Gupta, T. Althoff, and T. Hartvigsen, “Are language models actually useful for time series forecasting?” arXiv preprint arXiv:2406.16964, 2024.   
[16] M. Jin, S. Wang, L. Ma, Z. Chu, J. Y. Zhang, X. Shi, P.-Y. Chen, Y. Liang, Y.-F. Li, S. Pan et al., “Time-llm: Time series forecasting by reprogramming large language models,” arXiv preprint arXiv:2310.01728, 2023.   
[17] P. Liu, H. Guo, T. Dai, N. Li, J. Bao, X. Ren, Y. Jiang, and S.-T. Xia, “Taming pre-trained llms for generalised time series forecasting via cross-modal knowledge distillation,” arXiv preprint arXiv:2403.07300, 2024.   
[18] T. Zhou, P. Niu, L. Sun, R. Jin et al., “One fits all: Power general time series analysis by pretrained lm,” Advances in neural information processing systems, vol. 36, pp. 43 322–43 355, 2023.   
[19] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?” in Proceedings of the AAAI Conference on Artificial Intelligence, 2023.   
[20] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “itransformer: Inverted transformers are effective for time series forecasting,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https: //openreview.net/forum?id=JePfAI8fah   
[21] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” arXiv preprint arXiv:2211.14730, 2022.   
[22] S. Wang, H. Wu, X. Shi, T. Hu, H. Luo, L. Ma, J. Y. Zhang, and J. Zhou, “Timemixer: Decomposable multiscale mixing for time series forecasting,” arXiv preprint arXiv:2405.14616, 2024.   
[23] M. Jin, H. Tang, C. Zhang, Q. Yu, C. Liu, S. Zhu, Y. Zhang, and M. Du, “Time series forecasting with llms: Understanding and enhancing model capabilities,” arXiv preprint arXiv:2402.10835, 2024.   
[24] Y. Liang, H. Wen, Y. Nie, Y. Jiang, M. Jin, D. Song, S. Pan, and Q. Wen, “Foundation models for time series analysis: A tutorial and survey,” in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 6555–6565.   
[25] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” arXiv preprint arXiv:2310.10688, 2023.   
[26] A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. P. Arango, S. Kapoor et al., “Chronos: Learning the language of time series,” arXiv preprint arXiv:2403.07815, 2024.