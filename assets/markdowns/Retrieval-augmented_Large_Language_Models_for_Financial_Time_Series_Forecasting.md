# Retrieval-augmented Large Language Models for Financial Time Series Forecasting

Mengxi $\mathbf { X i a o } ^ { a }$ , Zhengyu $\mathbf { C h e n } ^ { b }$ , Lingfei Qianc, Zihao $\mathbf { J i a n g } ^ { b }$ , Yueru $\mathbf { H } \mathbf { e } ^ { c }$ , Yijing $\mathbf { X } \mathbf { u } ^ { d }$ , Yuechen Jiange, Dong $\mathbf { L i } ^ { b }$ , Ruey-Ling Wengc, Jimin Huangf , Min Penga, Sophia Ananiadouf , Jian-Yun $\mathbf { N i e } ^ { g }$ , Qianqian ${ \bf X _ { i e } } ^ { a _ { * } }$

a School of Artificial Intelligence, Wuhan University, b School of Computer Science, Wuhan University, c The Fin AI, d Columbia University, e Stevens Institute of Technology, $f$ University of Manchester, g University of Montreal

# Abstract

Accurately forecasting stock price movements is critical for informed financial decision-making, supporting applications ranging from algorithmic trading to risk management. However, this task remains challenging due to the difficulty of retrieving subtle yet high-impact patterns from noisy financial time-series data, where conventional retrieval methods, whether based on generic language models or simplistic numeric similarity, often fail to capture the intricate temporal dependencies and context-specific signals essential for precise market prediction. To bridge this gap, we introduce $\bar { \mathbf { F i n S r a g } ^ { 2 } }$ , the first retrieval-augmented generation (RAG) framework with a novel domain-specific retriever FinSeer for financial time-series forecasting. FinSeer leverages a candidate selection mechanism refined by LLM feedback and a similarity-driven training objective to align queries with historically influential sequences while filtering out financial noise. Such training enables FinSeer to identify the most relevant time-series data segments for downstream forecasting tasks, unlike embedding or distance-based retrieval methods used in existing RAG frameworks. The retrieved patterns are then fed into StockLLM, a 1B-parameter LLM fine-tuned for stock movement prediction, which serves as the generative backbone. Beyond the retrieval method, we enrich the retrieval corpus by curating new datasets that integrate a broader set of financial indicators, capturing previously overlooked market dynamics. Experiments demonstrate that FinSeer outperforms existing textual retrievers and traditional distance-based retrieval approaches in enhancing the prediction accuracy of StockLLM, underscoring the importance of domain-specific retrieval frameworks in handling the complexity of financial time-series data.

# 1 Introduction

Financial time-series forecasting is crucial for maintaining market stability and efficiency, with profound implications for investment decisions, risk assessment, and economic policy formulation [Fama and French, 2000]. However, the extreme volatility and nonlinear dynamics of financial markets pose significant analytical challenges, requiring sophisticated approaches to distill actionable signals from complex, noise-laden data streams. Stock movement prediction, which involves determining future price direction (up or down), represents a particularly critical yet demanding task that has attracted substantial research interest [Xie et al., 2023a, 2024a,b]. Although recent LLM-based approaches have shown promise in stock prediction, they predominantly rely on textual data like news and social media [Wang et al., 2024], using only the past several days’ closing prices as reference while largely ignoring the wealth of information contained in historical time-series patterns [Bustos and PomaresQuimbaya, 2020]. This oversight creates a critical gap in effectively harnessing temporal financial data for prediction. The challenge is further compounded by the enormous scale and complexity of such data, spanning multiple influencing factors and extended historical contexts. Addressing this requires an intelligent retrieval mechanism capable of efficiently navigating vast time-series datasets to extract and deliver the most relevant patterns, thereby empowering LLMs to generate more accurate and reliable stock movement forecasts.

![](images/c2265fc7334c6f166224e6c6fce048deab1ad521280425e291c0e54ed5b56760.jpg)  
Figure 1: Overview of our time-series RAG framework.

Despite the critical importance of retrieving relevant historical patterns for stock movement prediction, current embedding-based and distance-based retrieval-augmented generation (RAG) approaches [Joshi et al., 2024, He et al., 2024, Cui et al., 2024] face fundamental limitations when applied to financial time-series data. Conventional embedding-based methods struggle because numeric financial sequences often exhibit superficially similar patterns that lack explicit semantic meaning, making it difficult for text-trained retrievers to distinguish meaningful signals from spurious correlations. Meanwhile, distance-based techniques like Dynamic Time Warping (DTW) [Yang et al., 2024] prove inadequate as they typically rely on single-variable comparisons (e.g., adjusted closing prices), ignoring the rich contextual information from other financial indicators. This approach limits retrieval to simplistic trend matching and fails to capture deeper, context-dependent relationships where sequences with divergent or opposing trends may actually contain complementary predictive signals. The inherent complexity of financial markets demands retrieval methods that can move beyond surface-level pattern matching to identify truly informative historical sequences that may exhibit complex, non-obvious relationships to current market conditions.

To address these challenges, we introduce FinSrag, the first Financial time-Series RAG framework for stock movement prediction, using FinSeer, a Financial Time-Series Retriever to effectively retrieve the most beneficial historical sequences beyond surface-level pattern matching for downstream forecasting. FinSeer is trained to identify sequences that are embedded with essential or complementary information with limited similarity compared with the query that are beneficial for prediction. A key challenge in training FinSeer lies in the absence of explicit retrieval ground truth, that for any given query, it is inherently ambiguous which historical patterns will most effectively aid forecasting given the vast amount of data. To address this, we propose an LLM-guided relevance estimation, where the language model itself is used to assess and refine the selection of candidate sequences during training. Building on the approach of Zhang et al. [2023], we assess candidate relevance by feeding query-candidate pairs into the LLM and using the generation logits corresponding to the correct answer as a proxy for relevance. Intuitively, candidates yielding higher logits are considered more informative for the forecasting task.

We then train FinSeer to effectively distinguish high-value sequences (positives) from irrelevant sequences (negatives) by learning a retrieval embedding space where relevant candidates are closely aligned with their corresponding queries. Specifically, for each query, FinSeer is optimized to maximize the similarity score between the query embedding and the embedding of top-ranked candidates (i.e., those associated with higher generation logits from the LLM), while simultaneously pushing apart embeddings of low-ranked, uninformative candidates. This contrastive learning objective encourages FinSeer to internalize the LLM’s implicit relevance signals, thereby enhancing its ability to retrieve sequences that are truly beneficial for downstream forecasting.

Beyond improving relevant sequence retrieval, our framework introduces a crucial advancement in what to retrieve by expanding the candidate pool beyond traditional price-based segments. Specifically, we segment the financial time series into candidates, where each candidate corresponds to a time-series segment of a single feature, such as the adjusted close price or a specific technical indicator. Unlike prior approaches that retrieve only from price sequences, for the first time, our method incorporates 28 additional financial indicators, allowing retrieval over a richer set of candidate sequences that capture diverse aspects of market behavior3. To support this, we construct a new dataset and enriched existing two where time-series segments from both price data and a broad set of financial indicators are included as retrieval candidates, enabling FinSeer to discover deeper predictive patterns beyond surface-level price trends.

Comprehensive evaluations across multiple financial time-series datasets demonstrate that FinSeer, our specialized retriever, is the only approach that consistently enhances LLM forecasting accuracy, outperforming five state-of-the-art retrievers from the MTEB leaderboard when integrated into the FinSrag framework. This superior performance not only validates our domain-specific design for time-series data, but also crucially reveals FinSeer’s unique capability to identify and retrieve the most predictive financial indicator segments. Notably, the distance-based method which cannot leverage FinSrag’s architecture, shows weaker performance, further confirming the advantages of our framework’s structured approach to sequence selection, by incorporating diverse financial indicators as candidates.

More than just improving accuracy, FinSeer’s retrieval patterns provide valuable financial insights: By automatically discovering and leveraging latent market factors that conventional approaches miss, our method offers an empirically grounded alternative to theoretical assumptions in economic models. This data-driven approach enables systematic identification of predictive market features, opening new avenues to study market inefficiencies and dynamic dependencies through the lens of actual predictive relevance rather than ex-ante hypotheses.

In conclusion, our contributions are summarized as follows:

1. We introduce FinSrag, the first RAG framework with domain-specific retriever for financial time-series forecasting. By improving the capability of identifying and retrieving the most relevant time-series data segments, FinSrag significantly enhances LLMs’ forecasting capabilities in financial markets.   
2. We propose FinSeer, a novel domain-specific retriever trained via LLM feedback to uncover overlooked market signals. We complement this with an enriched dataset featuring 28 expert-selected technical indicators that go beyond conventional price data.   
3. Experimental results demonstrate the effectiveness of our framework, which surpasseses state-of-the-art retrievers and distance-based methods.

# 2 Prelimilaries

Retrieval-augmented financial time-series forecasting [Sezer et al., 2020] involves predicting future values or trends $( G )$ based on a given query sequence $( q )$ and a set of retrieved historical sequences (c). These sequences are collected over time at regular intervals. In the retrieval process, the goal of the retrieval model $( R )$ is to efficiently identify and extract the most useful historical sequences from a large pool of candidates. By providing relevant context, the retrieval model enhances the forecasting model’s ability to make accurate and reliable predictions. In the specific task of stock movement prediction $\mathrm { [ X u }$ and Cohen, 2018], the problem is framed as a binary classification task: predicting whether a stock’s price will rise or fall on the next trading day. Given a query sequence $q$ which represents the stock’s price over the previous $t$ days, the model retrieves relevant sequences as context and predicts the stock’s movement $M _ { q , d }$ for the next trading day $d$ . Details about threshold settings are illustrated in Appendix C.

# 3 Methodology

# 3.1 Retrieval Candidate Pool Design and Dataset Construction

To support our retrieval framework, we introduce a new dataset comprising high-volume stocks from 2022-2023 and enhance two existing datasets with 28 carefully selected financial indicators. Complete dataset statistics are presented in Table 1, while partitioning details and additional specifications appear in Appendix E.1. The following parts detail our methodology for: (1) stock price data collection and indicator selection (in Collection of Indicators), and (2) temporal scope determination and content specification for queries and candidate sequences (in Query and Candidate Construction).

Table 1: Dataset statistics of query and candidate sequence amounts4.   

<table><tr><td>Dataset</td><td colspan="3">License</td><td>Train</td><td>Valid</td><td>Test</td></tr><tr><td rowspan="2">ACL18 [Xu and Cohen, 2018]</td><td rowspan="2">MIT License</td><td>query</td><td>2015.06.03-2015.12.31</td><td>3,312</td><td>440</td><td>2.912</td></tr><tr><td>candidate</td><td>2014.06.02-2015.12.31</td><td>441,694</td><td>67,320</td><td>444,312</td></tr><tr><td rowspan="2">BIGDATA22 [Soun et al., 2022]</td><td rowspan="2">Public</td><td>query</td><td>2020.04.09-2020.12.31</td><td>3,229</td><td>434</td><td>2.868</td></tr><tr><td>candidate</td><td>2019.04.01-2020.12.31</td><td>328,372</td><td>44,778</td><td>328,372</td></tr><tr><td rowspan="2">STOCK23</td><td rowspan="2">MIT License</td><td>query</td><td>2023.01.03-2023.12.31</td><td>4,268</td><td>570</td><td>4,128</td></tr><tr><td>candidate</td><td>2022.01.03-2023.12.31</td><td>404,736</td><td>50.592</td><td>404,736</td></tr></table>

Collection of Indicators. Our financial data foundation comes from the Yahoo Finance API [Xu and Berkely, 2014], which provides 6 daily trading metrics, including opening, highest, lowest, close prices, adjusted close prices, along with trading volume. While these basic price data offer essential market snapshots, they fail to capture deeper market dynamics crucial for forecasting. For example, if the adjusted close price increases on continuous trading dates, it only shows upward momentum but cannot determine when a trend is exhausted. The KDJ indicator, however, combines price momentum and range position to signal overbought $\mathrm { ( > 8 0 ) }$ or oversold $( < 2 0 )$ conditions Wu and Diao [2015].

These limitations motivate our multi-layer dataset design to extract deeper and latent market signals beyond raw price movements. This multi-layer feature design is inspired by established practices in quantitative finance and machine learning Heaton et al. [2017], Jansen [2020], Chen et al. [2024], which emphasize the importance of hierarchical signal extraction from raw market data to enhance model predictive power. The dataset consists of three hierarchical components: (1) Core Price Metrics: We begin with aforementioned 6 fundamental indicators, including raw OHLCV (Open, High, Low, Close, Volume) and Adjusted Close prices. (2) Primary Technical Indicators: To capture mid-level market dynamics (e.g., trend strength, volatility, and price-volume interactions) Lo et al. [2000], Park and Irwin [2007], Wu and Diao [2015], which are not directly observable in raw OHLCV, we incorporate 10 widely-used technical indicators such as log returns, price momentum, and VWAP, whose practice is supported by both empirical studies and industry applications Chan [2013], Fischer and Krauss [2018], Gu et al. [2020a], Ma and Yan [2022]. (3) Alpha Factors: To uncover higher-order predictive signals, we identify 18 Alpha Factors via a Mutual Information (MI)-based selection process Guyon and Elisseeff [2003], Jansen [2020]. This approach quantifies the nonlinear dependency between candidate features and future returns, allowing us to retain only those indicators with strong predictive relationships. The effectiveness of alpha factors in enhancing financial model performance has been extensively supported in the literature Harvey et al. [2016], Tulchinsky [2019], Gu et al. [2020b], Kakushadze [2016].

The integrated dataset, exemplified in Appendix E.3, combines these three layers (6 price metrics $^ +$ 10 technical indicators $+ 1 8$ alpha factors) to provide a multidimensional view of market conditions. Complete specifications for all 34 indicators appear in Table 4.

Query and Candidate Construction. The query construction process involves temporal range determination and query content specification. For temporal boundaries, we establish that each query must have sufficient candidate sequences available for retrieval by requiring that the query date occurs at least one year after the start date of the corresponding dataset split. This temporal buffer ensures adequate historical context for meaningful retrieval. We implement a one-day sliding window across trading days to generate consecutive queries. For the example shown in Figure 2(a), in the ACL18 dataset spanning from 2014-06-02 to 2015-12-31, we define queries as those occurring between 2015-06-03 and 2015-12-31. This configuration guarantees that even the earliest query (2015-06-03) can access a full year of preceding candidate sequences. Each query consists of the stock identifier, query date, and recent market data for query-candidate matching. The market data includes adjusted closing prices from the five most recent trading days. The five-day window aligns with standard financial practices, where multiples of five are commonly used for price change calculations.

The candidate construction follows similar temporal and content specifications as queries. Candidates are drawn from all available historical data preceding each query date, spanning from the dataset’s start date to the trading day before the query. We apply a one-day sliding window to generate sequential candidates, with all stocks in the dataset eligible as candidates regardless of query stock matching. Each candidate consists of the stock ticker, price movement direction on a specific trading date, a relevant financial indicator, and corresponding indicator values calculated from the preceding five consecutive trading days. Figure 2(b) demonstrates this construction through examples of both rising and falling candidates retrieved from the pool.

Our approach enables seamless market monitoring through incremental updates to the candidate pool. When advancing the query date (e.g., from 2015-12-30 to 2015-12-31), we simply append the new trading day’s data (2015-12-30) to the existing pool (2014-06-02 through 2015-12-29), maintaining all historical candidates without requiring model retraining. This efficient update mechanism ensures continuous operation while preserving the complete candidate history.

![](images/f63628b66c164be2ed9ec87ee3debe5b3771633ad46f0fb3a6a11595394b9065.jpg)  
Figure 2: Illustration of query and candidate construction. (a) illustrates how to construct all queries for a given dataset. (b) illustrates the corresponding candidate pool for a given query, and the update of candidate pool with trading date updates.

# 3.1.1 Sequence Serialization

Since stock movement prediction depends on the changes in related features rather than their exact values, we serialize stock prices and financial indicators into a time-series format. We use JSON to represent these sequences, as it has been demonstrated to effectively support LLMs in interpreting time-series data [Fang et al., 2024, Singha et al., 2023, Yin et al., 2023].

The following are two examples of a query and a candidate sequence. When inquerying about stock MO on 2015-06-02, the query sequence contains the stock identifier (MO), query date (2015-06-02), and the adjusted close prices of last five trading dates. The serialized sequence is shown below:

A potential candidate in the candidate pool represent stock MO on date 2014-07-02, with the highest price as its indicator. The sequence includes the stock ticker (MO), price movement direction (freeze) on a specific trading date (2014-07-02), a key financial indicator (the highest price), and corresponding indicator values calculated from the preceding five consecutive trading days. The serialized sequence is shown below:

{ " candidate_stock ": "MO", " candidate_date ": " 2014 -07 -02 ", " candidate_movement ": " freeze ", " recent_date_list ": [" 2014 -06 -25 ", " 2014 -06 -26 ", " 2014 -06 -27 ", " 2014 -06 -30 ", " 2014 -07 -01 "] , " high_list ": [42.2 , 42.0 , 41.86 , 42.28 , 42.0]}

# 3.2 Retriever Training

We then train FinSeer to effectively distinguish high-value sequences (positives) from irrelevant sequences (negatives) by learning a retrieval embedding space where relevant candidates are closely aligned with their corresponding queries. To achieve this, we score and select positive and negative candidates as the retriever training corpus (in Candidate Scoring and Selection), then specify the training objective (in Training Objective) and conduct knowledge distillation (in Knowledge Distillation).

Candidate Scoring and Selection. To determine whether a candidate sequence assists in predicting the movement of the query, we use LLM feedback to score each candidate. The details of our LLM backbone, StockLLM-1B-Instruct (hereafter referred to as StockLLM), are shown in Appendix F.1. Specifically, for a given query $q$ , we integrate the query sequence and each candidate sequence $c _ { i }$ from the candidate pool as concurrent inputs to the StockLLM. Then StockLLM outputs logits, which are unnormalized scores representing the model’s confidence for each possible class (e.g., rise or fall). These logits are transformed into probabilities $P ( c )$ using the softmax function:

$$
P ( c ) = { \frac { e ^ { z _ { c } } } { \sum _ { j } e ^ { z _ { j } } } } ,
$$

where $z _ { c }$ is the logit for the correct class (e.g., rise if the true movement is upward) and $z _ { j }$ represents the logits for all possible classes. The resulting probability $P ( c )$ serves as the score for the candidate $c _ { i }$ with respect to the query $q$ .

We rank the candidate sequences in descending order based on their scores $P ( c )$ . The top-1 sequence is selected as a positive candidate, while the bottom 15 sequences are chosen as negative candidates. The sets of selected positive and negative sequences are denoted as $\mathbb { C } _ { \mathbb { P } }$ and $\mathbb { C } _ { \mathbb { N } }$ , respectively.

Training Objective. Our retriever $R ( q )$ is designed to intelligently distinguish between historically significant sequences $\mathbb { C } _ { \mathbb { P } }$ and noisy sequences $\mathbb { C } _ { \mathbb { N } }$ . The training objective is to ensure that $R ( q )$ prioritizes sequences from $\mathbb { C } _ { \mathbb { P } }$ while minimizing attention to those from $\mathbb { C } _ { \mathbb { N } }$ . This is achieved by maximizing a similarity measure $s u p ( q , s )$ between the query sequence $q$ and candidate sequences $s$ Mathematically, the retriever’s objective is formulated as:

$$
R ( q ) = \arg \operatorname* { m a x } _ { s \in \mathbb { C } _ { \mathbb { P } } \cup \mathbb { C } _ { \mathbb { N } } } s u p ( q , s ) .
$$

By focusing on sequences that maximize $s u p ( q , s )$ , the retriever ensures that the most informative and contextually relevant historical sequences are identified.

Knowledge Distillation. To leverage the scoring derived from StockLLM, we employ knowledge distillation, which transfers knowledge from the teacher model (StockLLM) to the student model (retriever) by mimicking the teacher’s output distribution. This approach effectively captures nuanced patterns and predictions from StockLLM. Specifically, we minimize the Kullback-Leibler (KL) divergence between the candidate distributions computed using the LLM’s rewards and those predicted by the embedding model. For each query $q$ and its candidate list $\{ \mathbb { C } _ { \mathbb { P } } , \mathbb { C } _ { \mathbb { N } } \}$ , we derive StockLLM’s rewards for the candidates, denoted as $\{ \bar { P } ( c _ { i } ) , i = 1 , . . . , n \}$ . To make these rewards suitable for distillation, we normalize them using a softmax function with temperature $\alpha$ :

$$
w _ { i } = \operatorname { s o f t m a x } _ { R } \left( { \frac { P ( c _ { i } ) } { \alpha } } \right) .
$$

The KL divergence is then computed as follows:

$$
\operatorname* { m i n } \sum _ { c } - w _ { i } \times \log \left( \frac { \exp \left( \left. e _ { q } , e _ { c _ { i } } \right. / \tau \right) } { \sum _ { c ^ { \prime } \in \mathbb { C } } \exp \left( \left. e _ { q } , e _ { c ^ { \prime } } \right. / \tau \right) } \right) ,
$$

where $e _ { q }$ and $e _ { c _ { i } }$ are the embeddings of the query $q$ and candidate $c _ { i }$ , respectively, and $\tau$ is a temperature parameter. This loss function optimizes the similarity between the query embedding and the embeddings of the top-ranked candidates, enhancing the retriever’s ability to accurately predict stock price movements.

# 3.3 Inference

During inference, the key innovation of our FinSrag framework lies in how FinSeer’s retrieval directly enhances StockLLM’s forecasting capability. Given a query, FinSeer first identifies the most relevant historical sequences by evaluating both temporal patterns and predictive relationships learned during training, filtering out noisy but numerically similar candidates that typically mislead traditional retrievers. These selected sequences are then structured and injected into StockLLM’s context window. Crucially, unlike standard RAG that simply concatenates retrieved documents, this end-to-end alignment between retrieval and generation is what enables FinSrag to outperform conventional forecasting pipelines where retrieval and prediction models are optimized separately.

# 4 Experiment

# 4.1 Experimental Settings

Datasets. We evaluate the effectiveness of our RAG framework on the test sets of the three datasets described in Table 1, with ACL18 containing 2,876 query sequences, BIGDATA22 containing 2,868 queries, and STOCK23 containing 4,128 queries. These thousand-scale queries help mitigate random bias, ensuring a robust and reliable evaluation of model performance.

Candidate Pool Settings. To ensure a comprehensive evaluation, for each query sequence, we include all sequences containing financial indicators across all stocks in the test set (not limited to the same stock), with data available up to the query date, as potential candidates. No additional restrictions are imposed, enabling a robust assessment of the models’ performance in real-world financial forecasting scenarios.

Baselines. To evaluate the efficiency of the FinSrag framework, the bare StockLLM-1B-Instruct without retrieval serves as our baseline to figure out whether the retrieval step enhances StockLLM’s prediction ability. To evaluate our retriever FinSeer, we tested other retrieval methods, including random retrieval, DTW distance, and five competitive retrieving models from the top of the MTEB English Leaderboard as baselines, containing: (1) Instructor-large [Su et al., 2023], a 335M instructionfinetuned text embedder that encodes sequences into 768-dimensional tensors. (2) UAE-large-v1 [Li and Li, 2023], a 335M ANGLE-optimized text embedding model that encodes sequences into 1024- dimensional tensors. (3) E5-mistral-7b-instruct [Wang et al., 2023], a 7111M embedder initialized from Mistral-7B-v0.1 [Jiang et al., 2023] and fine-tuned on multilingual datasets, encoding sequences into 4096-di-mensional tensors. (4) BGE-large-en-v1.5 [Xiao et al., 2023], a 335M general embedder pre-trained with RetroMAE [Xiao et al., 2022], encoding sequences into 1024-dimensional tensors. (5) LLM-Embedder [Zhang et al., 2023], a 109M embedder fine-tuned from BGE-large-en-v1.5, designed as a unified embedding model to support diverse retrieval augmentation needs for LLMs. It encodes sequences into 768-dimensional tensors. More details are shown in Appendix F.2.

Evaluation Metrics. We employ Accuracy (ACC) and Matthews Correlation Coefficient (MCC) [Matthews, 1975] to assess the performance of FinSeer and the baseline models on the stock movement prediction task. These metrics evaluate the performance of stock movement prediction based on the distribution of positive and negative samples.

# 4.2 Main Results

As shown in Table 2, experimental results demonstrate our framework’s effectiveness by outperforming all baseline retrieval methods in assisting LLM-based stock movement prediction.

Table 2: Results of stock movement predictions using StockLLM-1B-Instruct and retrieval models.   

<table><tr><td rowspan="2">Retrieving Methods (+ StockLLM-1B-Instruct)</td><td colspan="2">ACL18</td><td colspan="2">BIGDATA22</td><td colspan="2">STOCK23</td></tr><tr><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td></tr><tr><td>w/o Retrieval</td><td>0.498</td><td>-0.006</td><td>0.493</td><td>-0.017</td><td>0.509</td><td>0.021</td></tr><tr><td>Random Retrieval</td><td>0.485</td><td>-0.028</td><td>0.495</td><td>-0.007</td><td>0.496</td><td>-0.004</td></tr><tr><td>DTW</td><td>0.516</td><td>0.041</td><td>0.500</td><td>0.010</td><td>0.492</td><td>-0.007</td></tr><tr><td>Instructor [Su et al., 2023]</td><td>0.498</td><td>-0.005</td><td>0.493</td><td>-0.015</td><td>0.505</td><td>0.010</td></tr><tr><td>UAE [Li and Li,2023]</td><td>0.486</td><td>-0.029</td><td>0.493</td><td>-0.011</td><td>0.494</td><td>-0.009</td></tr><tr><td>E5 [Wang et al., 2023]</td><td>0.510</td><td>0.027</td><td>0.498</td><td>-0.002</td><td>0.499</td><td>-0.001</td></tr><tr><td>BGE [Xiao et al., 2023]</td><td>0.492</td><td>-0.012</td><td>0.501</td><td>0.002</td><td>0.488</td><td>-0.014</td></tr><tr><td>LLM Embedder [Zhang et al., 2023]</td><td>0.503</td><td>0.007</td><td>0.459</td><td>-0.083</td><td>0.503</td><td>0.007</td></tr><tr><td>FinSeer</td><td>0.517</td><td>0.035</td><td>0.510</td><td>0.023</td><td>0.542</td><td>0.085</td></tr></table>

First, compared with bare StockLLM and random retrieval, FinSeer demonstrates the critical importance of retrieving truly valuable information. While trained on a comprehensive stock movement prediction corpus, bare StockLLM’s limited input with only the recent five-day adjusted close price results in unstable performance that fluctuates around random guessing levels. Similarly, when randomly retrieved sequences are provided as supposedly relevant context, they introduce instability by arbitrarily confusing or occasionally coincidentally benefiting StockLLM’s decision-making.

Second, our comparison with the five top-ranked retrievers from the MTEB English leaderboard reveals FinSeer’s superiority in time-series retrieval. The negligible performance gap between instruction-finetuned retriever (Instructor) and no-retrieval baselines underscores the fundamental challenges of time-series retrieval. Unlike text retrieval, this task cannot rely solely on task understanding since candidate sequences often exhibit visual similarity while differing in predictive value. Other retrievers demonstrate inconsistent cross-dataset performance because their similarity differentiation fails to align with the LLM’s perception of importance. Even LLM Embedder, our backbone model trained with LLM feedback, shows limited generalization to time-series retrieval, further emphasizing the problem’s complexity. Among those retrievers, FinSeer consistently outperforms other retrievers across all datasets, proving its superior ability to learn LLM preferences and effectively enhance time-series forecasting through retrieval.

Third, our framework’s advantages over distance-based retrieval methods highlight the value of incorporating diverse feature types. While DTW achieves comparable performance to FinSeer during specific market periods (ACL18 [2014-2015] and BIGDATA22 [2019-2020]), its retrieval capability proves insufficient during the volatile 2022-2023 period (STOCK23) marked by significant market surges and fluctuations. During this challenging phase, DTW significantly degrades StockLLM’s performance, while FinSeer maintains its enhancement capability.

# 4.3 Ablation Study

In this section, we explore two aspects of our framework based on retrieval results: which indicators are most retrieved by all retrievers (in Indicator Occurrences), and whether StockLLM-1B-Instruct makes predictions by analyzing candidates or just based on the candidates’ movements (in Appendix G.1). We also visualize indicator sequence embeddings in Appendix G.2. Moreover, we explore the performance of these retrieval methods with a larger size of StockLLM (in Appendix G.3).

In this section, we analyze the retrieved indicators of all RAG models. We calculate indicator occurrences on the ACL18 test set, and the results are shown in Figure 3. As shown in the figure, FinSeer is the only model that successfully extracts a diverse and comprehensive set of indicators while achieving superior performance. This clearly demonstrates its advanced temporal retrieval capabilities. Specifically, while other models like LLM Embedder and Instructor predominantly focus on basic indicators such as close price and adjusted close price, FinSeer effectively identifies and retrieves a wide range of technical indicators, including kdj crossover, MACD Histogram, Bollinger Bands, and various alpha factors. This richer set of retrieved indicators provides FinSeer with more comprehensive auxiliary information, enabling more accurate and reliable predictions.

![](images/fb46bf885f917ecb8e83897ca1e3099bf391784a7f808f81d837423548875681.jpg)  
Figure 3: Indicator occurrences of different RAG models on ACL18 dataset.

# 4.4 Case Study

This case study illustrates the critical importance of alignment between the retriever and the LLM’s forecasting preferences in financial time-series analysis. We examine the stock XOM on 2015-06-25 from the ACL18 dataset, where the adjusted close price exhibited a pronounced downward trend. The query sequence is as follows:

" query_stock ": " XOM ",   
" query_date ": " 2015 -06 -25 ",   
" recent_date_list ": [" 2015 -06 -18 ", " 2015 -06 -19 ", " 2015 -06 -22 ", " 2015 -06 -23 ", " 2015 -06 -24 "] ,   
" adjusted_close_list ": [58.0813 , 57.8979 , 57.8707 , 57.8027 , 57.5377]}

While multiple retrievers were evaluated, only FinSeer successfully enabled StockLLM to predict the correct movement as a fall. Specifically, FinSeer retrieved five diverse indicators: close price, adjusted close price, alpha021, alpha054, and the highest price, providing a comprehensive view of the stock’s behavior. Alpha021 identifies trends based on short- and long-term price averages and volume conditions, while alpha054 combines price and volume rankings to assess performance within a specific time window. These indicators allowed StockLLM to accurately assess whether the downward trend would persist or reverse, demonstrating the value of retrieving contextually relevant and diverse features.

In contrast, other retrievers, such as Instructor, BGE, LLM Embedder, and E5, extracted sequences dominated by close or adjusted close prices, all reflecting similar downward trends. While these sequences aligned with the current trend, they failed to provide actionable insights for forecasting future movements, leading StockLLM to misinterpret them as noise and incorrectly predict a rise. Similarly, UAE retrieved sequences indicating overbought and oversold conditions, including three rise and two freeze trends. Although overbought signals often suggest a potential downturn, the retrieved sequences themselves exhibited rising or frozen trends, confusing StockLLM and resulting in an erroneous prediction. This case study underscores the superiority of FinSeer in retrieving meaningful and diverse indicators that align with the LLM’s forecasting logic, enabling more accurate and reliable predictions.

# 5 Conclusion

In this paper, we present FinSrag, the first retrieval-augmented generation framework tailored for financial time-series forecasting. At its core, FinSeer, a novel retriever refined by LLM feedback, effectively identifies historically influential market sequences while filtering out financial noise. Combined with StockLLM, a fine-tuned LLM with 1B parameters, our framework leverages enriched financial datasets to capture previously overlooked market dynamics. Empirical results confirm that FinSeer surpasses both textual and distance-based retrievers in improving StockLLM’s prediction accuracy, highlighting the necessity of domain-specific retrieval in financial forecasting. Beyond stock markets, FinSrag establishes a blueprint for integrating RAG in time-sensitive decision-making domains.

References   
Eugene F. Fama and Kenneth R. French. Forecasting profitability and earnings. The Journal of Business, 73(2):161–175, 2000.   
Qianqian Xie, Weiguang Han, Yanzhao Lai, Min Peng, and Jimin Huang. The wall street neophyte: A zero-shot analysis of chatgpt over multimodal stock movement prediction challenges. arXiv preprint arXiv:2304.05351, 2023a.   
Qianqian Xie, Dong Li, Mengxi Xiao, Zihao Jiang, Ruoyu Xiang, Xiao Zhang, Zhengyu Chen, Yueru He, Weiguang Han, Yuzhe Yang, et al. Open-finllms: Open multimodal large language models for financial applications. arXiv preprint arXiv:2408.11878, 2024a.   
Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, et al. The finben: An holistic financial benchmark for large language models. arXiv preprint arXiv:2402.12659, 2024b.   
Meiyun Wang, Kiyoshi Izumi, and Hiroki Sakaji. Llmfactor: Extracting profitable factors through prompts for explainable stock movement prediction. In Findings of the Association for Computational Linguistics ACL 2024, pages 3120–3131, 2024.   
Oscar Bustos and Alexandra Pomares-Quimbaya. Stock market movement forecast: A systematic review. Expert Systems with Applications, 156:113464, 2020.   
Pankaj Joshi, Aditya Gupta, Pankaj Kumar, and Manas Sisodia. Robust multi model rag pipeline for documents containing text, table & images. In 2024 3rd International Conference on Applied Artificial Intelligence and Computing (ICAAIC), pages 993–999. IEEE, 2024.   
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. arXiv preprint arXiv:2402.07630, 2024.   
Lingxi Cui, Huan Li, Ke Chen, Lidan Shou, and Gang Chen. Tabular data augmentation for machine learning: Progress and prospects of embracing generative ai. arXiv preprint arXiv:2407.21523, 2024.   
Silin Yang, Dong Wang, Haoqi Zheng, and Ruochun Jin. Timerag: Boosting llm time series forecasting via retrieval-augmented generation. arXiv preprint arXiv:2412.16643, 2024.   
Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. Retrieve anything to augment large language models. arXiv preprint arXiv:2310.07554, 2023.   
Mingyuan Wu and Xiaotian Diao. Technical analysis of three stock oscillators testing macd, rsi and kdj rules in sh & sz stock markets. In 2015 4th International Conference on Computer Science and Network Technology (ICCSNT), volume 1, pages 320–323. IEEE, 2015.   
Omer Berat Sezer, Mehmet Ugur Gudelek, and Ahmet Murat Ozbayoglu. Financial time series forecasting with deep learning: A systematic literature review: 2005–2019. Applied soft computing, 90:106181, 2020.   
Yumo Xu and Shay B Cohen. Stock movement prediction from tweets and historical prices. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1970–1979, 2018.   
Yejun Soun, Jaemin Yoo, Minyong Cho, Jihyeong Jeon, and U Kang. Accurate stock movement prediction with self-supervised learning from sparse noisy tweets. In 2022 IEEE International Conference on Big Data (Big Data), pages 1691–1700. IEEE, 2022.   
Selene Yue Xu and CU Berkely. Stock price forecasting using information from yahoo finance and google trend. UC Brekley, pages 1–22, 2014.   
James B Heaton, Nick G Polson, and Jan Hendrik Witte. Deep learning for finance: deep portfolios. Applied Stochastic Models in Business and Industry, 33(1):3–12, 2017.   
Stefan Jansen. Machine Learning for Algorithmic Trading: Predictive models to extract signals from market and alternative data for systematic trading strategies with Python. Packt Publishing Ltd, 2020.   
Luyang Chen, Markus Pelger, and Jason Zhu. Deep learning in asset pricing. Management Science, 70(2):714–750, 2024.   
Andrew W. Lo, Harry Mamaysky, and Jiang Wang. Foundations of technical analysis: Computational algorithms, statistical inference, and empirical implementation. Journal of Finance, 55(4):1705– 1765, 2000.   
Cheol-Ho Park and Scott H. Irwin. What do we know about the profitability of technical analysis? Journal of Economic Surveys, 21(4):786–826, 2007.   
Ernie Chan. Algorithmic trading: winning strategies and their rationale. John Wiley & Sons, 2013.   
Thomas Fischer and Christopher Krauss. Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2):654–669, 2018.   
Shihao Gu, Bryan Kelly, and Dacheng Xiu. Empirical asset pricing via machine learning. Review of Financial Studies, 33(5):2223–2273, 2020a.   
Chenyao Ma and Sheng Yan. Deep learning in the chinese stock market: The role of technical indicators. Finance Research Letters, 49:103025, 2022.   
Isabelle Guyon and André Elisseeff. An introduction to variable and feature selection. Journal of machine learning research, 3(Mar):1157–1182, 2003.   
Campbell R Harvey, Yan Liu, and Heqing Zhu. . . . and the cross-section of expected returns. The Review of Financial Studies, 29(1):5–68, 2016.   
Igor Tulchinsky. Finding Alphas: A quantitative approach to building trading strategies. John Wiley & Sons, 2019.   
Shihao Gu, Bryan Kelly, and Dacheng Xiu. Empirical asset pricing via machine learning. The Review of Financial Studies, 33(5):2223–2273, 2020b.   
Zura Kakushadze. 101 formulaic alphas. Wilmott, 2016(84):72–81, 2016.   
Xi Fang, Weijie Xu, Fiona Anting Tan, Jiani Zhang, Ziqing Hu, Yanjun Qi, Scott Nickleach, Diego Socolinsky, Srinivasan Sengamedu, and Christos Faloutsos. Large language models on tabular data–a survey. arXiv preprint arXiv:2402.17944, 2024.   
Ananya Singha, José Cambronero, Sumit Gulwani, Vu Le, and Chris Parnin. Tabular representation, noisy operators, and impacts on table structure understanding tasks in llms. arXiv preprint arXiv:2310.10358, 2023.   
Yuwei Yin, Yazheng Yang, Jian Yang, and Qi Liu. Finpt: Financial risk prediction with profile tuning on pretrained foundation models. arXiv preprint arXiv:2308.00065, 2023.   
Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. One embedder, any task: Instruction-finetuned text embeddings. In ACL (Findings), pages 1102–1121. Association for Computational Linguistics, 2023.   
Xianming Li and Jing Li. Angle-optimized text embeddings. arXiv preprint arXiv:2309.12871, 2023.   
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368, 2023.   
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.   
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-pack: Packaged resources to advance general chinese embedding, 2023.   
Shitao Xiao, Zheng Liu, Yingxia Shao, and Zhao Cao. Retromae: Pre-training retrieval-oriented language models via masked auto-encoder. In EMNLP, 2022. URL https://arxiv.org/abs/ 2205.12035.   
Brian W Matthews. Comparison of the predicted and observed secondary structure of t4 phage lysozyme. Biochimica et Biophysica Acta (BBA)-Protein Structure, 405(2):442–451, 1975.   
Jaemin Yoo, Yejun Soun, Yong-chan Park, and U Kang. Accurate multivariate stock movement prediction via data-axis transformer with multi-level contexts. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 2037–2045, 2021.   
Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. Pixiu: A large language model, instruction data and evaluation benchmark for finance. arXiv preprint arXiv:2306.05443, 2023b.   
Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, and Garrison Cottrell. A dual-stage attention-based recurrent neural network for time series prediction. arXiv preprint arXiv:1704.02971, 2017.   
Fuli Feng, Huimin Chen, Xiangnan He, Ji Ding, Maosong Sun, and Tat-Seng Chua. Enhancing stock movement prediction with adversarial training. arXiv preprint arXiv:1810.09936, 2018.   
Xiao Ding, Yue Zhang, Ting Liu, and Junwen Duan. Deep learning for event-driven stock prediction. In Twenty-fourth international joint conference on artificial intelligence, 2015.   
Qikai Liu, Xiang Cheng, Sen Su, and Shuguang Zhu. Hierarchical complementary attention network for predicting stock price movements with news. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pages 1603–1606, 2018.   
Huizhe Wu, Wei Zhang, Weiwei Shen, and Jun Wang. Hybrid deep sequential modeling for social textdriven stock prediction. In Proceedings of the 27th ACM international conference on information and knowledge management, pages 1627–1630, 2018.   
Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, et al. Time-llm: Time series forecasting by reprogramming large language models. arXiv preprint arXiv:2310.01728, 2023.   
Xinli Yu, Zheng Chen, Yuan Ling, Shujing Dong, Zongyi Liu, and Yanbin Lu. Temporal data meets llm–explainable financial time series forecasting. arXiv preprint arXiv:2306.11025, 2023.   
Chenxi Liu, Qianxiong Xu, Hao Miao, Sun Yang, Lingzheng Zhang, Cheng Long, Ziyue Li, and Rui Zhao. Timecma: Towards llm-empowered time series forecasting via cross-modality alignment. arXiv preprint arXiv:2406.01638, 2024.   
CoinAPI. Understanding ohlcv in crypto market data analysis. https://www.coinapi.io/blog/ understanding-ohlcv-in-market-data-analysis, 2023. Accessed: 2025-05-15.   
Akhilesh Ganti. Adjusted closing price: How it works, types, pros & cons. https://www. investopedia.com/terms/a/adjusted_closing_price.asp, 2020. Accessed: 2025-05- 15.   
Carla Tardi. Price change: Definition, types, causes & effects. https://www.investopedia.com/ terms/p/price-change.asp, 2022. Accessed: 2025-05-15.   
Narasimhan Jegadeesh. Evidence of predictable behavior of security returns. Journal of Finance, 45 (3):881–898, 1990.   
Narasimhan Jegadeesh and Sheridan Titman. Returns to buying winners and selling losers: Implications for stock market efficiency. Journal of Finance, 48(1):65–91, 1993.   
Daniel Mitchell, Jedrzej Bialkowski, and Stathis Tompaidis. Volume-weighted average price tracking: A theoretical and empirical study. IISE Transactions, 52(8):864–889, 2020.   
Gabriel Dan Ioan Anghel. Stock market efficiency and the macd: Evidence from countries around the world. Procedia Economics and Finance, 32:1414–1431, 2015.   
John Bollinger. Bollinger on Bollinger Bands. McGraw-Hill, New York, 2001.   
J. Welles Wilder. New Concepts in Technical Trading Systems. Trend Research, Greensboro, NC, 1978.   
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.   
Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of $1 0 0 +$ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.   
Jian Tang, Jingzhou Liu, Ming Zhang, and Qiaozhu Mei. Visualizing large-scale and high-dimensional data. In Proceedings of the 25th international conference on world wide web, pages 287–297, 2016.

# A Limitation

While our FinSrag framework demonstrates significant improvements in LLM-based time-series forecasting, several limitations suggest important directions for future research. First, although our current evaluation focuses specifically on stock movement prediction, the core methodology and architectural principles of our framework are inherently generalizable to other time-series prediction tasks. Future work will systematically validate this generalizability across domains, including but not limited to economic indicators, energy demand forecasting, and epidemiological trend prediction.

Second, the present study deliberately focuses on unimodal time-series data to provide clear, controlled evidence for the effectiveness of our retrieval mechanisms. This focused approach has allowed us to rigorously quantify the contribution of temporal pattern retrieval independent of other factors. However, we recognize that many real-world forecasting scenarios could benefit from multimodal integration, where time-series data interacts with textual reports, visual data, or other information modalities. Future work will incorporate cross-modal corpora to enable more comprehensive information retrieval.

# B Ethical Statement

Our framework and its contents are provided for academic and educational purposes only. None of the material constitutes financial, legal, or investment advice. No warranties, express or implied, are offered regarding the accuracy, completeness, or utility of the content. The authors and contributors are not responsible for any errors, omissions, or any consequences arising from the use of the information herein. Users should exercise their own judgment and consult professionals before making any financial, legal, or investment decisions. The use of the software and information contained in this repository is entirely at the user’s own risk.

FinSrag and all its components are licensed under MIT.

# C Task Definition Details

Rise/Fall Threshold Settings. To classify daily movements as rise or fall, we first calculate returns $R _ { t }$ , which represents the percentage change in the closing price over consecutive days.

$$
R _ { t } = { \frac { \mathrm { a d j } _ { - } \mathrm { c l o s e } _ { d } - \mathrm { a d j } _ { - } \mathrm { c l o s e } _ { d - 1 } } { \mathrm { a d j } _ { - } \mathrm { c l o s e } _ { d - 1 } } } * 1 0 0
$$

Following Yoo et al. [2021] and Soun et al. [2022], we classify the movement as rise if return $R _ { t }$ exceeds 0.55, fall if it is below -0.5. In line with previous experimental settings [Xie et al., 2023b, 2024b], we do not evaluate freeze cases as queries. However, we include sequences with $R _ { t } \in [ - 0 . 5 , 0 . 5 5 ]$ ] as freeze candidates in the candidate pool, ensuring a diverse and comprehensive set of historical data for context.

$$
M _ { q , d } = \left\{ \begin{array} { l l } { { \mathrm { r i s e } , } } & { { R _ { t } > 0 . 5 5 } } \\ { { \mathrm { f a l l } , } } & { { R _ { t } < - 0 . 5 } } \\ { { \mathrm { f r e e z e } , } } & { { - 0 . 5 \leq R _ { t } \leq 0 . 5 5 } } \end{array} \right.
$$

Rationale for Unbalanced Thresholds. The asymmetrical thresholds for classifying daily movements (0.55 for rises and -0.5 for falls) reflect the inherent dynamics of stock market behavior. Stock prices typically rise gradually due to sustained investor optimism but fall sharply during panic selling or profit-taking. The stricter rise threshold prevents minor upward fluctuations from being misclassified as significant increases, while the more lenient fall threshold ensures meaningful downward trends are captured. This approach aligns with market realities, improving the reliability of movement classifications.

# D Related Work

# D.1 Stock Movement Prediction

Non-LLM Methods. Traditional approaches to stock movement prediction have focused on various aspects of financial data. One prominent category of methods analyzes stock price sequences and their corresponding technical indicators [Qin et al., 2017, Feng et al., 2018] to identify patterns in historical data for predicting future movements. However, due to the complexity of factors influencing stock prices, subsequent methods have incorporated additional contextual information, such as news articles [Ding et al., 2015, Liu et al., 2018] or social media posts $\mathrm { [ X u }$ and Cohen, 2018, Wu et al., 2018]. Despite efforts, these methods are highly susceptible to noise and struggle to analyze the vast and diverse nature of financial information effectively.

LLM-based Methods. Recent studies have explored using LLMs for financial prediction tasks, either by fine-tuning open-source models or prompting advanced models like GPT-4. However, even state-of-the-art models, including GPT-4, have achieved only random-guessing-level accuracy in stock movement prediction [Xie et al., 2023b, 2024b, 2023a]. This highlights the inherent challenges in identifying and analyzing meaningful patterns in a domain as volatile and multifaceted as stock market prediction.

# D.2 Time-series forecasting with LLMs

Non-retrieval Methods. To enhance the performance of LLMs in time-series forecasting, existing methods have primarily focused on aligning temporal and textual data, either by transforming timeseries into textual formats or by encoding both modalities into a unified vector space. For instance, a study [Jin et al., 2023] reprograms time-series data into textual representations suitable for LLMs, enhancing prediction accuracy via declarative prompts. Similarly, some other studies [Yu et al., 2023, Liu et al., 2024] explore cross-modal alignment: the former applies LLMs to financial forecasting by integrating stock prices with news data, while the latter introduces a cross-modality framework to align time-series with text for improved predictive performance. While these advancements improve alignment, existing methods often struggle to process large-scale time-series data comprehensively due to LLM input limitations. As a result, effectively leveraging extensive historical data remains a challenge. This highlights the need for retrieval-augmented methods, a gap our approach directly addresses.

RAG Methods. With the rapid advancement of RAG techniques in various applications of LLMs, recent research has begun exploring RAG for time-series forecasting. For instance, TimeRAG [Yang et al., 2024] integrates RAG into time-series forecasting by combining Dynamic Time Warping (DTW) with LLMs to improve prediction accuracy. However, relying solely on numeric similarity is insufficient for financial time-series forecasting, as it fails to capture deeper semantic relationships. This underscores the need for a more targeted retrieval framework tailored to the complexities of financial data, a challenge our framework effectively addresses.

# E Dataset Construction Details

# E.1 Dataset Partition

To construct our datasets, we select high-trade-volume U.S. stocks across three periods: 2014-2015, 2017-2018, and 2022-2023. The first two periods align with two benchmark datasets: for 2014-2015, we use the same stocks as the ACL18 dataset [Xu and Cohen, 2018], and for 2017-2018, we use the same stocks as the BIGDATA22 dataset [Soun et al., 2022]. To incorporate more recent data, we manually retrieve high-trade-volume stocks from 2022 to 2023 to create the STOCK23 dataset. This ensures our training and evaluation reflect recent market conditions and provide a robust benchmark for stock movement prediction.

To maintain temporal integrity and prevent information leakage, we implement stock-wise partitioning rather than temporal splitting of the dataset. This approach addresses a critical characteristic of our data where many high-trade-volume stocks (e.g., Google (GOOG)) appear across multiple years (2014-2015 and 2019-2020).

Table 3 details the complete stock lists and their corresponding sectors for all subsets, demonstrating two key properties of our partitioning: first, the complete absence of overlap between training and test sets, and second, comprehensive sector coverage in both sets. The alphabetical partitioning serves as a neutral, systematic approach that is uncorrelated with any financial or temporal characteristics while achieving our core objectives of clean separation and representative sampling. This method guarantees that the model encounters truly novel stocks during testing without any temporal leakage, while maintaining balanced sector representation across all partitions.

Table 3: Stock information in datasets.   

<table><tr><td>Dataset</td><td>Split (amount)</td><td>Stock List</td><td>Sector Count</td></tr><tr><td rowspan="3">ACL18</td><td>train (33)</td><td>ABBV,AEP,AMGN,AMZN, BA,BABA,BAC,BCH, BHP, BP, BRK-A, BSAC,BUD,C, CAT.CHTRx.D.CDA.,CODL, DUK,EXC,GD,GE,GOOG,</td><td>basic-materials (1),communication-services (4),</td></tr><tr><td>valid (5)</td><td>HD,HON,HSBC,IEP, INTC JNJ, JPM, KO,LMT, MA</td><td></td></tr><tr><td>test (33)</td><td>MCD,MDT,MMM,MO,MRK, MSFT,NEE,NGG,NVS,ORCL, PCG,PEP,PFE,PG,PM,PPL, REX, SLB,SNY, SO,SPLP, SRE,T,TM,TSM,UL,UNH, UPS,V,VZ,WFC,WMT, XOM</td><td>basic-materials (1),communication-services (2), consumer-cyclical (2),consumer-defensive (6), energy (2),financial-services (2), healthcare (6), industrials (3),technology (3),utilities (6)</td></tr><tr><td rowspan="2">BIGDATA22</td><td>train (22)</td><td>AEP,AMGN,AMZN,BA, BAC,C,CAT, CODI, CSCO, CVX,D,DIS,DUK,EXC, GD,GE, GMRE,GOOG, HD, HON, INTC, JNJ</td><td>communication-services (2),consumer-cyclical (2), energy (1),financial-services (2), healthcare (2) industrials (6),real-estate (1),technology (2),utilities (4)</td></tr><tr><td>valid (3)</td><td>JPM, KO, LMT</td><td></td></tr><tr><td rowspan="2"></td><td>test (22)</td><td>MA,MCD,MDT, MMM,MO, WMT, XOM</td><td>basic-materials (1),communication-services (2), industrials (2),technology (2),utilities (6)</td></tr><tr><td>train (24)</td><td>ADBE,ADSK,AMD,AVGO, COIN, COST, DELL,DXCM, ELV, ETSY,FOX,HOOD,HPQ,IBM, INTU, JBL,JNJ,KKR,KMX, LDOS,LLY,LOW,LULU, LUV</td><td>communication-services (1),consumer-cyclical (4), consumer-defensive (1),financial-services (3), healthcare (4),industrials (1),technology (10)</td></tr><tr><td rowspan="2">STOCK23</td><td>valid (3)</td><td>MARA,META,MNST</td><td>financial-services (1),communication-services (1), consumer-defensive (1)</td></tr><tr><td>test (24)</td><td>MRVL,MS,MSCI, NDAQ, NKE,NVDA,ORLY,PANW,PEAK, PLTR,PNC,PNW,RY, SMCI, SNOW, SNPS,T, TGT,TSLA, UBER,ULTA,V,WBA,WDAY</td><td>communication-services (1),consumer-cyclical (4), consumer-defensive (1),financial-services (6), healthcare (2),technology(9),utilities (1)</td></tr></table>

# E.2 Financial Indicators

Financial Indicators Selection We compute the MI scores for each indicator using nonparametric estimation methods provided by mutual_info_regression from the Scikit-learn library. For each query with retrieved candidates, we convert the movements of retrieved sequences and LLM’s prediction to numerical representations.

$$
M _ { c _ { j } } , M _ { q } = \left\{ \begin{array} { l l } { { 1 , } } & { { \mathrm { r i s e } } } \\ { { 0 , } } & { { \mathrm { f r e e z e } } } \\ { { - 1 , } } & { { \mathrm { f a l l } } } \end{array} \right.
$$

Then, we calculate the average value of the $M _ { c _ { i } }$ from the five retrieved sequences to represent the RAG-provided result, and the final prediction of the query is recorded as another numerical value:

$$
\begin{array} { l } { { \displaystyle x _ { i } = \frac { 1 } { 5 } \sum _ { j = 0 } ^ { 4 } M _ { c _ { j } } } , } \\ { { \displaystyle y _ { i } = M _ { q _ { i } } . } } \end{array}
$$

Lastly, we adopt the Pearson correlation coefficient to compute the correlation:

$$
r = { \frac { \sum ( x _ { i } - { \bar { x } } ) ( y _ { i } - { \bar { y } } ) } { \sqrt { \sum ( x _ { i } - { \bar { x } } ) ^ { 2 } \sum ( y _ { i } - { \bar { y } } ) ^ { 2 } } } } .
$$

Then we normalize them into range $( 0 , 1 )$ . These scores provide insight into the strength of the dependency between each financial indicator and the forward return. Higher MI scores indicate stronger relationships, highlighting which indicators are most predictive of future price movements. We select the top-18 alpha indicators with the highest MI scores in our candidates, as is shown in Figure 4.

![](images/c4240fb21b80c88e7b2e1cfb35abbb4613720d7926ec76c65bdca68df86430d7.jpg)  
Figure 4: Mutual Information(MI) scores between indicators and forward return.

Indicators Descriptions We provided a brief description of each indicator in Table 4.

Table 4: Brief descriptions of indicators.   

<table><tr><td>Group</td><td>Indicator</td><td>Description</td></tr><tr><td colspan="3"></td></tr><tr><td rowspan="5">1</td><td>open (opening price) CoinAPI [2023]</td><td>reflecting the stock&#x27;s value at the start of the trading session.</td></tr><tr><td>high,low (highest/lowest price) CoinAPI [2023]</td><td>capture the price range during the day, offering insights into volatility.</td></tr><tr><td>close (close price) CoinAPI [2023]</td><td>often the most important for technical analysis,represents the final trading price for the day.</td></tr><tr><td>adj_close (adjusted close price) Ganti [2020]</td><td>accounts for corporate actions like dividends or stock splits,providing a more accurate long-term price history.</td></tr><tr><td>volume CoinAPI [2023]</td><td>inii</td></tr><tr><td colspan="3">devivedfinancial indicators</td></tr><tr><td rowspan="3">2</td><td></td><td>represents the change in stock price between consecutive trading days.</td></tr><tr><td>returns Tardi [2022],Jegadeesh [1990],Je- gadeesh and Titman [1993]</td><td>represents the percentage change in the closing price over consecutive days.</td></tr><tr><td>VWAP (Volume Weighted Average Price) Mitchell et al. [2020]</td><td>calculates the average price of a stock weighted by trading volume, providing a benchmark for institutional investors.</td></tr><tr><td rowspan="2">3</td><td>macd_histogram (values) Anghel [2015]</td><td>e</td></tr><tr><td>macd_crossover (signals) Anghel [2015]</td><td>indicates a buy or sell signal when the MACD line crosses above or below the signal line.</td></tr><tr><td rowspan="2">4</td><td>bollinger_bands (signals) Bollinger [2001]</td><td></td></tr><tr><td>exceeding_upper/lower(values）Bollinger [2001]</td><td>the value of the adjusted close price exceeds the upper or lower price limits around a moving average.</td></tr><tr><td rowspan="2">5</td><td>overbought_and_oversold_conditions Wilder [1978]</td><td>classify the market state into &quot;overbought&quot;or &quot;oversold&quot; areas based on predefined thresholds for K,D,and J.</td></tr><tr><td>kdj_crossover (signals) Wu and Diao [2015]</td><td>identify &quot;bullish&quot;or &quot;bearish&quot; signals based on the relationship between K and D.</td></tr><tr><td rowspan="9"></td><td>alpha_smr Kakushadze [2016]</td><td>measuresshort-term rersalpatrns by capturigstcs tendency torertafterabrupt price</td></tr><tr><td>alpha_mom Kakushadze [2016]</td><td></td></tr><tr><td>alpha_009 Kakushadze [2016]</td><td>uses recent price changes to clasify trends based on their range within a 5-day window.</td></tr><tr><td>alpha_012 Kakushadze [2016]</td><td>relates volume change direction to the negative change in price.</td></tr><tr><td>alpha_041 Kakushadze [2016]</td><td>measures the difference between price range midpoint and VWAP.</td></tr><tr><td>alpha_049 Kakushadze [2016]</td><td>detects strong downtrends or reversals in lagged price movements.</td></tr><tr><td>alpha_051 Kakushadze [2016]</td><td>e</td></tr><tr><td>alpha_054 Kakushadze [2016]</td><td>comesastsip</td></tr><tr><td>alpha_101 Kakushadze [2016]</td><td>normalizes price change by daily price range.</td></tr><tr><td rowspan="7">7</td><td>alpha_002 Kakushadze [2016]</td><td>measpr</td></tr><tr><td>alpha_006 Kakushadze [2016]</td><td>computes the negative correlation between opening price and volume over a 10-day window.</td></tr><tr><td>alpha_024 Kakushadze [2016]</td><td>compares long-term and short-term price movements to identify breakouts or retracements.</td></tr><tr><td>alpha_028 Kakushadze [2016]</td><td>scales the relationship between average price,low price correlation,and close price.</td></tr><tr><td>alpha_032 Kakushadze [2016]</td><td>cobnesdev</td></tr><tr><td></td><td>identifies stocks exhibiting strong recent performance (typically over 3-6 months) while con- trolling for volatility,aiming to capture persistent trends while mitigating risk from erratic price</td></tr><tr><td>alpha_053 Kakushadze [2016]</td><td>swings.</td></tr><tr><td rowspan="3">8</td><td>alpha_021 Kakushadze [2016]</td><td>identifies trends based on short- and long-term price averages and volume conditions.</td></tr><tr><td>alpha_023 Kakushadze [2016]</td><td>tracks changes in high prices if they exceed their 2O-day average.</td></tr><tr><td>alpha_046 Kakushadze [2016]</td><td>analyzes trends in lagged and current 0-day price changes.</td></tr></table>

# E.3 Sequences and Serialization

An example in our datastore is shown below.

" stock_name ":" ABBV ",   
" query_date ":" 2014 -06 -05 ",   
" movement ":" rise ",   
" open " :54.549999 ,   
" high " :55.32 ,   
" low " :54.360001 ,   
" close " :55.299999 ,   
" adj_close " :36.980961 ,   
" volume " :4847300 ,   
" macd_histogram " :0.0621141602 ,   
" macd_crossover ":null ,   
" bollinger_bands ":null ,   
" exceeding_upper ":null ,   
" exceeding_lower ":null ,   
" overbought_and_oversold_conditions ":null ,   
" kdj_crossover ":null ,   
" returns " :0.0131913094 ,   
" VWAP " :54.9933333333 ,   
" alpha_smr " :0.000549858 ,   
" alpha_mom " :0.0121661224 ,   
" alpha_002 ":null ,   
" alpha_006 ":null ,   
" alpha_009 " : -0.481476 ,   
" alpha_012 " : -0.481476 ,   
" alpha_021 ":1 ,   
" alpha_023 ":0.0 ,   
" alpha_024 " : -0.769035 ,   
" alpha_028 ":null ,   
" alpha_032 ":null ,   
" alpha_041 " : -0.1554335256 ,   
" alpha_046 " : -0.481476 ,   
" alpha_049 " : -0.481476 ,   
" alpha_051 " : -0.481476 ,   
" alpha_053 ":null   
" alpha_054 " :126.4264005641 ,   
" alpha_101 " : -18.282056485

# F Experimental Settings

# F.1 StockLLM-1B-Instruct Backbone

To activate the LLM’s inherent knowledge and ensure instruction-following capabilities, we fine-tune a 1B parameter LLM (LLaMA 3.2-1B-Instruct) [Grattafiori et al., 2024] using the LoRA technique for efficient low-rank adaptation, resulting in StockLLM-1B-Instruct. By intentionally using a smaller backbone model, we establish a more challenging experimental setup, ensuring that performance improvements are attributable to FinSeer’s retrieval capabilities rather than the LLM’s capacity. More details of fine-tuning are illustrated in Appendix F.1.

The fine-tuning process is implemented using the LlamaFactory framework [Zheng et al., 2024], with the following configuration: a learning rate of 5e-5, a cosine scheduler, gradient accumulation over 8 steps, mixed-precision (fp16) training, and 5 epochs of training with regular evaluation to log metrics and select the best-performing checkpoint. This setup demonstrates the robustness of our framework in extracting meaningful insights even under constrained model size, highlighting the effectiveness of FinSeer in enhancing financial forecasting tasks. The fine-tuning prompt is shown in Figure 5, with candidate sequences randomly retrieved from the dataset. The fine-tuning step is to train the StockLLM-1B-Instruct to follow instructions, without additional steps for retrieving certain candidates.

# F.2 Experimental Settings

Baselines. The introduction of retrieval baselines is as follows:

• Random retrieval: The candidate range is aligned with the "Candidate Pool Settings" part. We use the random.sample() function in Python to retrieve random candidates from the candidate pool.   
• DTW (Dynamic Time Warping) distance: The formula measures the similarity between two temporal sequences by aligning them non-linearly in time, minimizing the cumulative warping cost. $\begin{array} { r } { D _ { \mathrm { D T W } } ( X , Y ) = \operatorname* { m i n } _ { \pi \in \mathcal { P } } \sqrt { \sum _ { ( i , j ) \in \pi } ( x _ { i } - y _ { j } ) ^ { 2 } } } \end{array}$ , where $X , Y$ are sequences, $\pi$ is a warping path, and $\mathcal { P }$ is the set of all possible paths.   
• Instructor-large [Su et al., 2023]: A 335M instruction-finetuned text embedder that encodes sequences into 768-dimensional tensors.   
• UAE-large-v1 [Li and Li, 2023]: A 335M ANGLE-optimized text embedding model that encodes sequences into 1024-dimensional tensors.   
• E5-mistral-7b-instruct [Wang et al., 2023]: A 7111M embedder initialized from Mistral-7Bv0.1 [Jiang et al., 2023] and fine-tuned on multilingual datasets, encoding sequences into 4096-di-mensional tensors.   
• BGE-large-en-v1.5 [Xiao et al., 2023]: A 335M general embedder pre-trained with RetroMAE [Xiao et al., 2022], encoding sequences into 1024-dimensional tensors.

• LLM-Embedder [Zhang et al., 2023]: A 109M embedder fine-tuned from BGE-large-env1.5, designed as a unified embedding model to support diverse retrieval augmentation needs for LLMs. It encodes sequences into 768-dimensional tensors.

Prompts. The prompt for bare StockLLM-1B-Instruct is shown in Figure 6, and the prompt for retrieval methods is shown in Figure 5.

![](images/207a35b544ddc66cd3cfe4123d29dbf34313cb8678fac0296997168bdc5b984e.jpg)  
Figure 5: The prompt template for StockLLM-1B-Instruct fine-tuning and RAG methods evaluation.

![](images/bfbcfdb3e497ddab33bac867fe1db271a487f1ab874a0ee43b8fed47d1ba259b.jpg)  
Figure 6: The prompt template for bare StockLLM-1B-Instruct evaluation.

Hyperparameters. For each model, we employ default parameter settings, utilizing official models for open-source LLMs obtained from Hugging Face. These testing procedures take place on a computational infrastructure consisting of one NVIDIA A800 Tensor Core GPU, equipped with 80GB of memory.

Error Bars. To account for potential variability, we conducted three randomized retrieval trials and report averaged performance metrics. For similarity-based retrieval methods, since both queries and candidates are encoded as fixed vector representations, the retrieved candidates remain deterministic across runs. To verify this stability, we performed three identical inference passes through StockLLM, observing minimal variance in prediction outcomes.

# G Addtional Experimental Results

# G.1 Candidate Movement Correlation

In this section, we investigate whether StockLLM-1B-Instruct relies on the movement trends of retrieved sequences to make predictions. To this end, we compute the correlation between the movements of retrieved sequences and the LLM’s generated results.

The results in Table 5 reveal no significant correlation between the movement direction (rise or fall) of the retrieved candidate sequences and the final predictions of StockLLM. This finding indicates that StockLLM-1B-Instruct does not simply mirror the movement trends of the retrieved sequences but instead analyzes their specific content to infer the query’s movement direction. This ability highlights the LLM’s capacity to extract meaningful insights from complex time-series data.

Table 5: Correlation between the movements of retrieved sequences and LLM’s generated results.   

<table><tr><td>Dataset</td><td>RAG Model</td><td>Correlation</td><td>P-Value</td></tr><tr><td rowspan="6">ACL18</td><td>Instructor</td><td>-0.098</td><td>0.000</td></tr><tr><td>UAE</td><td>-0.068</td><td>0.000</td></tr><tr><td>E5</td><td>-0.108</td><td>0.000</td></tr><tr><td>BGE</td><td>0.175</td><td>0.000</td></tr><tr><td>LLM Embedder</td><td>0.014</td><td>0.457</td></tr><tr><td>FinSeer</td><td>0.054</td><td>0.004</td></tr><tr><td rowspan="6">BigData22</td><td>Instructor</td><td>-0.072</td><td>0.000</td></tr><tr><td>UAE</td><td>0.009</td><td>0.633</td></tr><tr><td>E5</td><td>-0.040</td><td>0.032</td></tr><tr><td>BGE</td><td>0.093</td><td>0.000</td></tr><tr><td>LLMEmbedder</td><td>0.060</td><td>0.001</td></tr><tr><td>FinSeer</td><td>0.077</td><td>0.000</td></tr><tr><td rowspan="6">STOCK23</td><td>Instructor</td><td>-0.114</td><td></td></tr><tr><td>UAE</td><td></td><td>0.000</td></tr><tr><td>E5</td><td>-0.093 -0.064</td><td>0.000 0.000</td></tr><tr><td>BGE</td><td></td><td>0.000</td></tr><tr><td>LLM Embedder</td><td>0.131</td><td></td></tr><tr><td>FinSeer</td><td>0.015 0.032</td><td>0.350 0.041</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

This observation underscores the critical role of retrieval quality in RAG models, as the relevance and informativeness of retrieved sequences directly influence prediction performance. Notably, StockLLM-1B-Instruct is a relatively small 1B parameter LLM, making this a particularly challenging setting. Despite this, our time-series-tailored RAG model, FinSeer, successfully enhances StockLLM’s performance by retrieving rich and relevant time-sensitive information. This demonstrates the effectiveness of our approach in improving prediction accuracy and showcases its potential for financial time-series forecasting.

# G.2 Indicator Sequence Visualization

To intuitively examine how FinSeer embeds indicator sequences, we use LargeVis [Tang et al., 2016] to reduce the embeddings to two dimensions. In Figure 7, we visualize5 the vector space of nine indicators, including adjusted close price, open price, high price, low price, close price, volume, movement, returns, and WAP. The results show that the green points, representing movement information, are well-clustered, indicating that FinSeer effectively captures meaningful patterns. However, this also suggests that deeper relationships between stock movements and indicators remain to be explored, highlighting the potential for further research in this area.

# G.3 Results of StockLLM-8B-Instruct

We additionally trained StockLLM-8B-Instruct based on LLaMA-3.1-8B-Instruct to evaluate FinSeer’s performance on larger LLMs. As shown in Table 6, the results demonstrate that FinSeer can consistently enhance LLM performance regardless of the backbone model’s inherent capability for stock movement prediction. Compared to the 1B version, we observe greater performance variability in the bare LLM, while different retrieval methods maintain similar relative impacts on StockLLM’s performance. Notably, random retrieval and other retrieval methods continue to show inconsistent effects, sometimes improving but other times degrading performance.. FinSeer not only outperforms all retrieval-based approaches but also delivers stable performance enhancements to StockLLM, validating the effectiveness of our RAG framework.

![](images/512ba2979942a30bf7cc7656c6fe903007309d6dd64f81422bde47a991064021.jpg)  
Figure 7: Visualization of indicator sequence embeddings on ACL18 dataset.

Table 6: Results of stock movement predictions using StockLLM-8B-Instruct and retrieval models.   

<table><tr><td rowspan="2">Retrieving Methods (+ StockLLM-8B-Instruct)</td><td colspan="2">ACL18</td><td rowspan="2"></td><td colspan="2">BIGDATA22</td><td colspan="2">STOCK23</td></tr><tr><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td><td>ACC</td><td>MCC</td></tr><tr><td>w/o Retrieval</td><td>0.479</td><td>-0.041</td><td>0.446</td><td>-0.108</td><td></td><td>0.510</td><td>0.020</td></tr><tr><td>Random Retrieval</td><td>0.509</td><td>0.028</td><td>0.450</td><td>-0.149</td><td></td><td>0.495</td><td>-0.012</td></tr><tr><td>DTW</td><td>0.509</td><td>0.020</td><td>0.457</td><td></td><td>-0.087</td><td>0.496</td><td>-0.008</td></tr><tr><td>Instructor</td><td>0.497</td><td>-0.005</td><td>0.457</td><td></td><td>-0.087</td><td>0.506</td><td>0.011</td></tr><tr><td>UAE</td><td>0.509</td><td>0.019</td><td>0.465</td><td></td><td>-0.081</td><td>0.509</td><td>0.018</td></tr><tr><td>E5</td><td>0.484</td><td>-0.031</td><td>0.453</td><td></td><td>-0.099</td><td>0.498</td><td>-0.005</td></tr><tr><td>BGE</td><td>0.522</td><td>0.058</td><td>0.460</td><td></td><td>-0.104</td><td>0.507</td><td>-0.001</td></tr><tr><td>LLMEmbedder</td><td>0.486</td><td>-0.034</td><td>0.454</td><td></td><td>-0.123</td><td>0.499</td><td>-0.002</td></tr><tr><td>FinSeer</td><td>0.554</td><td>0.122</td><td>0.469</td><td></td><td>-0.065</td><td>0.511</td><td>0.023</td></tr></table>