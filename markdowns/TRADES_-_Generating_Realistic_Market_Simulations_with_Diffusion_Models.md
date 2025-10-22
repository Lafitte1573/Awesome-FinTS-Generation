# TRADES: Generating Realistic Market Simulations with Diffusion Models

Leonardo Berti∗ leonardo.berti@tum.de Technical University of Munich Germany

Bardh Prenkaj bardh.prenkaj@tum.de Technical University of Munich Germany

Paola Velardi velardi@uniroma1.it Sapienza University of Rome, Italy

# Abstract

Financial markets are complex systems characterized by high statistical noise, nonlinearity, volatility, and constant evolution. Thus, modeling them is extremely hard. Here, we address the task of generating realistic and responsive Limit Order Book (LOB) market simulations, which are fundamental for calibrating and testing trading strategies, performing market impact experiments, and generating synthetic market data. Previous works lack realism, usefulness, and responsiveness of the generated simulations. To bridge this gap, we propose a novel TRAnsformer-based Denoising Diffusion Probabilistic Engine for LOB Simulations (TRADES). TRADES generates realistic order flows as time series conditioned on the state of the market, leveraging a transformer-based architecture that captures the temporal and spatial characteristics of high-frequency market data. There is a notable absence of quantitative metrics for evaluating generative market simulation models in the literature. To tackle this problem, we adapt the predictive score, a metric measured as an MAE, by training a stock price predictive model on synthetic data and testing it on real data. We compare TRADES with previous works on two stocks, reporting a $\times 3 . 2 7$ and $\times 3 . 4 8$ improvement over SoTA according to the predictive score, demonstrating that we generate useful synthetic market data for financial downstream tasks. Furthermore, we assess TRADES’s market simulation realism and responsiveness, showing that it effectively learns the conditional data distribution and successfully reacts to an experimental agent, giving sprout to possible calibrations and evaluations of trading strategies and market impact experiments. We developed DeepMarket, the first open-source Python framework for market simulation with deep learning. In our repository, we include a synthetic LOB dataset composed of TRADES’s generated simulations. We release the code at https://github.com/LeonardoBerti00/DeepMarket.

# CCS Concepts

• Computing methodologies Neural networks.

# Keywords

Diffusion Model, Market Simulation, Limit Order Book

# ACM Reference Format:

Leonardo Berti, Bardh Prenkaj, and Paola Velardi. 2025. TRADES: Generating Realistic Market Simulations with Diffusion Models. In Proceedings of Make sure to enter the correct conference title from your rights confirmation emai (Conference acronym ’XX). ACM, New York, NY, USA, 14 pages. https://doi.org/XXXXXXX.XXXXXXX

# 1 Introduction

A realistic and responsive market simulation1 has always been a dream in the finance world [31, 35, 40, 42, 51, 52]. Recent years have witnessed a surge in interest towards deep learning-based market simulations [15, 18, 36, 44]. An ideal market simulation should fulfill four key objectives: (1) enable the calibration and evaluation of algorithmic trading strategies including reinforcement learning models [75]; (2) facilitate counterfactual experiments to analyze (2.1) the impact of orders [70], and (2.2) the consequences of changing financial regulations and trading rules, such as price variation limits; (3) analyze market statistical behavior and stylized facts in a controlled environment; (4) generate useful granular market data to facilitate research on finance and foster collaboration.

For these objectives, two key elements are paramount: the realism and the responsiveness of the simulation. Realism [43] refers to the similarity between the generated probability distribution and the actual data distribution. Responsiveness captures how the simulated market reacts to an experimental agent’s actions. Furthermore, the generated data usefulness is crucial for achieving the last objective (4). Usefulness refers to the degree to which the generated market data can contribute to other related financial tasks, such as predicting the trends of stock prices [50, 57, 67].

Backtesting [24] and Interactive Agent-Based Simulations (IABS) [9] are two of the most used traditional market simulation methods. Backtesting assesses the effectiveness of trading strategies on historical market data. It is inherently non-responsive since there is no way to measure the market impact of the considered trading strategies, making the analysis partial. Minimizing market impact has been the focus of extensive research efforts over many years [1], resulting in the development of sophisticated algorithms designed to mitigate the price effects of executing large orders, through timing, order, size, and venue selection. IABS, on the other hand, enables the creation of heterogeneous pools of simplified traders with different strategies, aiming to approximate the diversity of the real market. However, obtaining realistic multi-agent simulations is challenging, as individual-level historical data of market agents is private, which impedes the calibration of the trader agents, resulting in an oversimplification of real market dynamics.

Recent advances in generative models, particularly Wasserstein GANs [13–15, 36], have shown promise in generating plausible market orders for simulations. However, GANs are susceptible to mode collapse [64] and training instability [12], leading to a lack of realism and usefulness in the generated data. These limitations hinder their ability to satisfy the market simulation objectives and their real-world applicability.

To address these shortcomings, we present our novel TRAnsformer based Denoising Diffusion Probabilistic Engine for LOB market Simulations (TRADES). TRADES generates realistic high-frequency market data,2 which are time series, conditioned on past observations. We demonstrate that TRADES surpasses state-of-the-art (SoTA) methods in generating realistic and responsive market simulations. Importantly, due to its ability to handle multivariate time series generation, TRADES is adaptable to other domains requiring conditioned sequence data generation. Furthermore, TRADES readily adapts to an experimental agent introduced into the simulation, facilitating counterfactual market impact experiments. In summary, our contributions are:

(1) Realistic and responsive market simulation method: We develop a Denoising Diffusion Probabilistic Engine for LOB Simulations (TRADES), exploiting a transformer-based neural network architecture. We remark that TRADES is easily adaptable to any multivariate time-series generation domain.   
(2) Plug-and-play framework: We release DeepMarket, the first open-source Python framework for market simulation with deep learning. We also publish TRADES’s implementation and checkpoints to promote reproducibility and facilitate comparisons and further research.   
(3) Synthetic LOB dataset: We release a synthetic LOB dataset composed of the TRADES’s generated simulations. We show in the results (section 7.1) how the synthetic market data can be useful to train a deep learning model to predict stock prices.   
(4) New “world” agent for market simulations: We extend ABIDES [9], an agent-based simulation environment, in our framework by introducing a new world agent class accompanied by a simulation configuration, which, given in input a trained generative model, creates limit order book market simulations. Our experimental framework does not limit the simulation to a single-world agent but enables the introduction of other trading agents, which interact among themselves and with the world agent. This defines a hybrid approach between deep learning-based and agent-based simulations.

(5) First quantitative metric for market simulations: The literature shows a notable absence of quantitative metrics to evaluate the generated market simulations. Typically, the evaluation relies on plots and figures. We posit that a robust and quantitative assessment is essential for the comparative analysis of various methodologies. To this end, we adapt the predictive score introduced in [73] to objectively and quantitatively evaluate the usefulness of the generated market data.

(6) Extensive experiments assessing usefulness, realism, and responsiveness: We perform a suite of experiments to demonstrate that TRADES-generated market simulations abide by these three principles. We show how TRADES outperforms SoTA methods [9, 14, 15, 69] according to the adopted predictive score and illustrate how TRADES follows the established stylized facts in financial markets [69].

# 2 Background

Here, we provide background information on multivariate time series generation and limit order book markets. Furthermore, since TRADES is an extension of the Denoising Diffusion Probabilistic Model (DDPM), we summarize it in the Appendix A.

# 2.1 Multivariate time series generation

Generating realistic market order streams can be formalized as a multivariate time series generation problem. Let ${ \mathbf { X } } = \left\{ { \mathbf { x } } _ { 1 : N , 1 : K } \right\} \in$ $\mathbb { R } ^ { N \times K }$ , be a multivariate time series, where $N$ is the time dimension (i.e., length) and $K$ is the number of features. The goal of generating multivariate time series is to consider a specific time frame of previous observations, i.e., $\left\{ \mathbf { x } _ { 1 : N , 1 : K } \right\}$ , and to produce the next sample $\mathbf { X } _ { N + 1 }$ . This task can easily be formulated as a self-supervised learning problem, where we leverage the past generated samples as the conditioning variable for an autoregressive model. In light of this, we can define the joint probability of the time series as in Eq. (1).

$$
q ( \mathbf { x } ) = \prod _ { n = 1 } ^ { N } q ( \mathbf { x } _ { n } \mid \mathbf { x } _ { 1 } , . . . , \mathbf { x } _ { n - 1 } )
$$

We leverage this concept at inference time using a sliding window approach. Hence, for every generation step,3 we generate a single sample ${ \bf x } _ { N } \in \mathbb { R } ^ { K }$ . In the next step, we append the generated $\mathbf { x } _ { N }$ to the end of the conditional part and shift the entire time series one step forward (see Section 4 for details with TRADES). Because we aim to generate a multivariate time series starting from observed values, we model the conditioned data distribution $q \big ( \mathbf { x } _ { N } | \mathbf { x } _ { 1 : N - 1 } \big )$ with a learned distribution $p _ { \theta } ( \mathbf { x } _ { N } | \mathbf { x } _ { 1 : N - 1 } )$ , to sample from it. Hereafter, we denote the conditional part4 with $\mathbf { x } ^ { c }$ , and the upcoming generation part with ${ \bf x } ^ { g }$ .

# 2.2 Limit Order Book

In a Limit Order Book (LOB) market, traders can submit orders to buy or sell a certain quantity of an asset at a particular price. There are three main types of orders in a limit order market. (1) A market order is filled instantly at the best available price. (2) A limit order allows traders to decide the maximum (in the case of a buy) or the minimum (in the case of a sell) price at which they want to complete the transaction. A quantity is always associated with the price for both types of orders. (3) A cancel order5 removes an active limit order. The Limit Order Book (LOB) is a data structure that stores and matches the active limit orders according to a set of rules. The LOB is accessible to all the market agents and is updated with each event, such as order insertion, modification, cancellation, and execution. The most used mechanism for matching orders is the Continuous Double Auction (CDA) [6]. In a CDA, orders are executed whenever a price overlaps between the best bid (the highest price a buyer is willing to pay) and the best ask (the lowest price a seller is willing to accept). This mechanism allows traders to trade continuously and competitively. The evolution over time of a LOB represents a multivariate temporal problem. We can classify the research on LOB data into four main types of studies, namely empirical studies analyzing the LOB dynamics [8, 16], price and volatility forecasting [57, 74], modeling the LOB dynamics [17, 23] and LOB market simulation [9, 15, 36].

# 3 Related Works

Diffusion models for time series generation. Diffusion models have been successfully applied to generate images [3, 33], video [41], and text [76]. Recently, they have been exploited also for time series forecasting [37, 53], imputation [5, 63], and generation [38]. To the best of our knowledge, only Lim et al. [38] tackle time series generation using diffusion models. They present TSGM, which relies on an RNN-based encoder-decoder architecture with a conditional score-matching network. Differently, our model is a conditional denoising diffusion probabilistic model which relies on a transformer-based architecture. Other diffusion-based approaches for time series [37, 53, 63] address slightly different problems, such as forecasting and imputation.

Market simulation with deep learning. Generating realistic market simulations using deep learning is a new paradigm. Traditional computational statistical approaches [16, 29] and IABS [9, 47] rely on strong assumptions, such as constant order size, failing to produce realistic simulations. These methods are mainly used to study how the interactions of autonomous agents give rise to aggregate statistics and emergent phenomena in a system [34]. Limit order book simulations are increasingly relying on deep learning. Li et al. [36] were the first to leverage a Wasserstein GAN (WGAN) [2] for generating order flows based on historical market data. Similarly, Coletta et al. [14, 15] employ WGANs in their stock market simulations, addressing the issue of responsiveness to experimental agents for the first time. Differently from [14, 15, 36], we condition with both the last orders and LOB snapshots, pushing the generation process toward a more realistic market simulation. Hultin et al. [30] extend [21] and model individual features with separate conditional probabilities using RNNs. Instead of relying on GANs, which are prone to model collapse [72] and instability [12], and RNNs, often hampered by the vanishing gradient phenomenon, we exploit diffusion-based models with an underlying transformer architecture. Nagy et al. [44] rely on simplified state-space models [58] to learn long-range dependencies, tackling the problem via a masked language modeling approach [22]. Shi and Cartlidge [55] introduce NS-LOB, a novel hybrid approach that combines a pre-trained neural Hawkes [54] process with a multi-agent trader simulation. We refer the reader to [32] for a comprehensive review of limit order book simulations.

# 4 Transformer-based Denoising Diffusion Probabilistic Engine for LOB Simulations

We introduce TRADES, a Transformer-based Denoising Diffusion Probabilistic Engine for LOB Simulations. Conditional diffusion models are better suited than standard diffusion models in generative sequential tasks because they can incorporate information from past observations that guide the generation process towards more specific and desired outputs. We formalize the reverse process for TRADES and the self-supervised training procedure. In Section 5, we specialize our architecture for market simulations.

# 4.1 Generation with TRADES

Here, we focus on an abstract time series generation task with TRADES. The goal of probabilistic generation is to approximate the true conditional data distribution $q ( \mathbf { x } _ { 0 } ^ { g } \mid \mathbf { x } _ { 0 } ^ { c } )$ with a model distribution $p _ { \theta } ( \mathbf { x } _ { 0 } ^ { g } \mid \mathbf { x } _ { 0 } ^ { c } )$ . During the forward process, we apply noise only to the “future” – i.e., the part of the input we want to generate – while keeping the observed values unchanged. Therefore, the forward process is defined as in the unconditional case in Eq. (11) (Appendix). For the reverse process, we extend the unconditional one $p _ { \theta } ( \mathbf { x } _ { 0 : T } )$ , defined in Eq. (12) (Appendix), to the conditional case:

$$
p _ { \boldsymbol { \theta } } ( \mathbf { x } _ { 0 : T } ^ { g } ) : = p ( \mathbf { x } _ { T } ^ { g } ) \prod _ { t = 1 } ^ { T } p _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t - 1 } ^ { g } \mid \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { c } )
$$

$$
p _ { \theta } ( \mathbf { x } _ { t - 1 } ^ { g } \mid \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { c } ) : = N ( \mathbf { x } _ { t - 1 } ^ { g } ; \mu _ { \theta } ( \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { c } , t ) , \Sigma _ { \theta } ( \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { c } , t ) )
$$

We define the conditional denoising learnable function as in Eq. (4)

$$
\boldsymbol { \epsilon } _ { \boldsymbol { \theta } } : ( \mathbf { x } _ { t } ^ { g } \in \mathbb { R } ^ { S \times K } , \ \mathbf { x } _ { 0 } ^ { c } \in \mathbb { R } ^ { M \times K } , \ t \in \mathbb { R } ^ { K } )  \boldsymbol { \epsilon } _ { t } \in \mathbb { R } ^ { S \times K }
$$

where $M + S = N$ . We set $M = N - 1$ and $S = 1$ for our experiments. Using $S > 1$ increases the efficiency, but also the task complexity because, in a single generative step, we generate $S$ samples. We exploit the parametrization proposed in [25] described in Eq. (5) to estimate the mean term.

$$
\begin{array} { r l r } & { } & { \mu _ { \theta } ( \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { c } , t ) = \displaystyle \frac { 1 } { \sqrt { \alpha _ { t } } } \left( \mathbf { x } _ { t } ^ { g } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { c } , t ) \right) } \\ & { } & { \mathrm { w h e r e \quad } \mathbf { x } _ { t } ^ { g } = \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } ^ { g } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon \quad \epsilon \sim N ( \mathbf { 0 } , \mathbf { I } ) . } \end{array}
$$

We do not rely on a fixed schedule as in [25] regarding the variance term. Inspired by [45], we learn it as in Eq. (6).

$$
\begin{array} { r } { \Sigma _ { \theta } ( \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { c } , t ) = \exp ( v \log \beta _ { t } + ( 1 - v ) \log \tilde { \beta } _ { t } ) , } \end{array}
$$

where $v$ is the neural network output, together with $\epsilon _ { \theta }$ . Nichol and Dhariwal [45] found that this choice improves the negative log-likelihood, which we try to minimize. After computing $\epsilon _ { t }$ and $\sigma _ { t }$ , we denoise $\mathbf { x } _ { t - 1 } ^ { g }$ as in Eq (7).

$$
\mathbf { x } _ { t - 1 } ^ { g } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( \mathbf { x } _ { t } ^ { g } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { c } , t ) \right) + \sigma _ { t } \mathbf { z }
$$

where $\mathbf { z } ~ \sim ~ { \cal N } ( \mathbf { 0 } , \mathbf { I } )$ . After the $T$ steps, the denoising process is finished, and $\mathbf { x } _ { 0 } ^ { g }$ is reconstructed.

![](images/30d100bd8eb2981be5fe6ab686abfa9583c48d0c4c6f3aa3babd4fce74cf976f.jpg)  
Figure 1: The training procedure and architecture of TRADES. We apply noise to the time series’s last element until obtaining $x _ { T } ^ { g }$ . We condition $x _ { T } ^ { g }$ on the previous $N - 1$ orders by concatenating it to them on the sequence axis and the last $N$ LOB snapshots. We feed these new tensors to two separate MLPs, each composed of two fully connected layers, to augment them into a higher dimensional space. We concatenate the augmented output vectors on the features axis and sum the diffusion embedding step $t$ and the positional embedding; we detail how they are represented in the Appendix C. We feed the result to the TRADES modules, composed of multi-head self-attention and feedforward layers, producing the noise $\varepsilon _ { \theta }$ and the standard deviation $\Sigma _ { \theta }$ . $\varepsilon _ { \theta }$ and $\Sigma _ { \theta }$ go through a de-augmentation phase via MLPs to map them back to the input space and are used to reconstruct?? ${ \bf x } _ { T - 1 } ^ { g }$ . We repeat this procedure $T$ times until we recover the original $\mathbf { x } _ { 0 } ^ { g }$ .

# 4.2 Self-supervised Training of TRADES

Given generation target $\mathbf { x } _ { 0 } ^ { g }$ and conditional observations $\mathbf { x } _ { 0 } ^ { c }$ , we sample $\mathbf { x } _ { t } ^ { g } = \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } ^ { g } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$ , where $\epsilon \sim { \cal N } ( 0 , { \bf I } )$ , and train $\epsilon _ { \theta }$ by minimizing Eq. (8).

$$
\mathcal { L } _ { \epsilon } ( \theta ) : = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \epsilon } \bigg [ | | \epsilon - \epsilon _ { \theta } ( \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { c } , t ) | | ^ { 2 } \bigg ] .
$$

Inspired by [45], we also learn6 $\Sigma _ { \theta }$ optimizing it according to Eq. (9).

$$
\begin{array} { r l } & { \mathcal { L } _ { \Sigma } ( \boldsymbol { \theta } ) : = \mathbb { E } _ { q } \Bigg [ \underbrace { - p _ { \boldsymbol { \theta } } ( \mathbf { x } _ { 0 } ^ { g } \mid \mathbf { x } _ { 1 } ^ { g } , \mathbf { x } _ { 0 } ^ { c } ) } _ { L _ { 0 } } + \underbrace { D _ { K L } ( q ( \mathbf { x } _ { T } ^ { g } \mid \mathbf { x } _ { 0 } ^ { g } ) \mid \mid p _ { \boldsymbol { \theta } } ( \mathbf { x } _ { T } ^ { g } ) } _ { L _ { T } } } \\ & { \quad \quad + \displaystyle \sum _ { t = 2 } ^ { T } \underbrace { D _ { K L } ( q ( \mathbf { x } _ { t - 1 } ^ { g } \mid \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { g } ) \mid \mid p _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t - 1 } ^ { g } \mid \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { c } ) ) } _ { L _ { t - 1 } } \Bigg ] . } \end{array}
$$

We optimize Eq. (9) to reduce the negative log-likelihood, especially during the first diffusion steps where $\mathcal { L } _ { \Sigma }$ is high [45]. The final loss function is a linear combination of the two as in Eq: (10).

$$
\mathcal { L } = \mathcal { L } _ { \epsilon } + \lambda \mathcal { L } _ { \Sigma } .
$$

We perform training by relying on a self-supervised approach as follows. Given a time series $\mathbf { \bar { x } } _ { 0 } \in \mathbb { R } ^ { N \times K }$ , we apply noise only to its last element – i.e., $\mathbf { x } _ { 0 } ^ { N }$ – through the forward pass. Then, we denoise it via the reverse process and learn $p _ { \theta } ( \mathbf { x } _ { t - 1 } ^ { g } \mid \mathbf { x } _ { t } ^ { g } , \mathbf { x } _ { 0 } ^ { c } )$ which aims to generate a new sample from the last observations. Therefore, during training, the conditioning has only observed values. A sampling time TRADES generates new samples autoregressively, conditioned on its previous outputs, until the simulation ends.

# 5 TRADES for Market Simulation

To create realistic and responsive market simulations, we implement TRADES to generate orders conditioned on the market state. TRADES’s objective is to learn to model the distribution of orders. Fig. 1 presents an overview of the diffusion process and the architecture. The network that produces $\varepsilon _ { \theta }$ and $\Sigma _ { \theta }$ contains several transformer encoder layers to model the temporal and spatial relationships of the financial time series [57]. Since transformers perform better with large dimension tensors, we project the orders tensor and the LOB snapshots to a higher dimensional space, using two fully connected layers. Hence, TRADES operates on the augmeand vector space. After the reverse process, we de-augmeprojecting them back to the input space to reconstruct $\varepsilon _ { \boldsymbol { \theta } }$ $\Sigma _ { \theta }$ $\mathbf { x } _ { t - 1 } ^ { g }$ and compute the loss.

Conditioning. The conditioned diffusion probabilistic model learns the conditioned probability distribution $p _ { \theta } ( o | s )$ , where $s$ is the market state, and ?? is the newly generated order, represented as $( p , q , d , b , \delta , c )$ , where $\mathcal { P }$ is the price, $q$ is the quantity, $d$ is the direction either sell or buy, $^ { b }$ is the depth, i.e., the difference between $\mathcal { P }$ and the best available price, $\delta$ is the time offset representing the temporal distance from the previously generated order and $c$ is the order type – either market, limit or cancel order. The asymmetries between the buying and selling side of the book indicate shifts in the supply and demand curves caused by exogenous and unobservable factors that influence the price [11]. Therefore, as depicted in Fig. 1, the model’s conditioning extends beyond the last $N - 1$ orders. To effectively capture market supply and demand dynamics encoded within the LOB, we incorporate the last $N$ LOB snapshots of the first $L$ LOB levels as input, where each level has a bid price, bid size, ask price, and ask size. We set $N = 2 5 6$ as in [15], and $L = 1 0$ . We argue that this choice of $L$ is a reasonable trade-off between conditioning complexity and feature informativeness. Several works [10, 11, 28, 48, 65] have shown that the orders behind the best bid and ask prices play a significant role in price discovery and reveal information about short-term future price movements. In Sec. 7.3, we delve deeper into the conditioning choice and method, performing an ablation and a sensitivity study.

# 6 DeepMarket framework with synthetic dataset for deep learning market simulations

We present DeepMarket, an open-source Python framework developed for LOB market simulation with deep learning. DeepMarket offers the following features: (1) pre-processing for high-frequency market data; (2) a training environment implemented with PyTorch Lightning; (3) hyperparameter search facilitated with WANDB [4]; (4) TRADES and CGAN implementations and checkpoints to directly generate a market simulation without training; (5) a comprehensive qualitative (via the plots in this paper) and quantitative (via the predictive score) evaluation. To perform the simulation with our world agent and historical data, we extend ABIDES [9], an open-source agent-based interactive Python tool.

# 6.1 TRADES-LOB: a new synthetic LOB dataset

In LOB research one major problem is the unavailability of a large LOB dataset. In fact, if you want to access a large LOB dataset you need to pay large fees to some data provider. The only two freely available LOB datasets are [27] and [46] which have a lot of limitations. The first one is composed of only Chinese stocks, which have totally different rules and therefore resulting behaviors with respect to NASDAQ or LSE stocks. The high cost and low availability of lOB data restrict the application and development of deep learning algorithms in the LOB research community. In order to foster collaboration and help the research community we release a synthetic LOB dataset: TRADES-LOB. TRADES-LOB comprises simulated TRADES market data for Tesla and Intel, for two days. Specifically, the dataset is structured into four CSV files, each containing 50 columns. The initial six columns delineate the order features, followed by 40 columns that represent a snapshot of the LOB across the top 10 levels. The concluding four columns provide key financial metrics: mid-price, spread, order volume imbalance, and Volume-Weighted Average Price (VWAP), which can be useful for downstream financial tasks, such as stock price prediction. In total the dataset is composed of 265,986 rows and 13,299,300 cells, which is similar in size to the benchmark FI-2010 dataset [46]. The dataset will be released with the code in the GitHub repository. We show in the results (section 7.1) how the synthetic market data can be useful to train a deep learning model to predict stock prices.

# 7 Experiments

Dataset and reproducibility. In almost all SoTA papers in this subfield, the authors use one, two, or three stocks [15, 30, 36, 44, 54, 55], most of which are tech. Following this practice, we create a LOB dataset from two NASDAQ stocks7 – i.e., Tesla and Intel – from January 2nd to the 30th of 2015. We argue that stylized facts and market microstructure behaviors, which are the main learning objective of TRADES, are independent of single-stock behaviors (see [7, 8, 20, 23]8), so the particular stock characteristics, such as volatility, market cap, and $\mathrm { p / e }$ ratio, are not fundamental. Each stock has 20 order books and 20 message files, one for each trading day per stock, totaling ${ \sim } 2 4$ million samples. The message files contain a comprehensive log of events from which we select market orders, limit orders, and cancel orders.9. Each row of the order book file is a tuple $\left( P ^ { a s k } ( t ) , V ^ { a s k } ( t ) , P ^ { b i d } ( t ) , V ^ { b i d } ( t ) \right)$ where $P ^ { a s k } ( t )$ and $P ^ { b i d } ( t ) \in \mathbb { R } ^ { L }$ are the prices of levels 1 through $L$ , and $V ^ { a s k } ( t )$ and $V ^ { b i d } ( t ) \in \mathbb { R } ^ { L }$ are the corresponding volumes. We use the first 17 days for training, the $1 8 \mathrm { t h }$ day for validation, and the last 2 for market simulations. We are aware of the widely used FI-2010 benchmark LOB dataset [46] for stock price prediction. However, the absence of message files in this dataset hinders simulating the market since the orders cannot be reconstructed. In Appendix B, we provide an overview of FI-2010 and its limitations.

Experimental setting. After training the model for 70, 000 steps until convergence, we freeze the layers and start the market simulation. A simulation is composed of (1) the electronic market exchange that handles incoming orders and transactions; (2) the TRADESbased “world” agent, which generates new orders conditioned on the market state; and (3) one or more optional experimental agents, that follow a user-customizable trading strategy, enabling counterfactual and market impact experiments. So, the experimental framework is a hybrid approach between a deep learning model and an interactive agent-based simulation.

We conduct the simulations with the first 15 minutes of real orders to compare the generated ones with the market replay.10 Afterward, the diffusion model takes full control and generates new orders autoregressively, conditioned on its previous outputs, until the simulation ends. After the world agent generates a new order, there is a post-processing phase in which the output is transformed into a valid order. We begin the simulation at 10:00 and terminate it at 12:00. This choice ensures that the generated orders are sufficient for a thorough evaluation while maintaining manageable processing times. On average, 50,000 orders are produced during this two-hour time frame. The output CSV file of the simulation contains the full list of orders and LOB snapshots of the simulation. All experiments are performed with an RTX 3090 and a portion of an A100. In Appendix C, we detail the data pre- and post-processing and model hyperparameter choice.

Baselines. We compare TRADES with the Market Replay – i.e., ground truth (market replay) – a IABS configuration, and the Wasserstein GAN – i.e., CGAN – under the setting of [14], similar to the same of those proposed in [13, 15]. We implemented CGAN from scratch given that none of the implementations in [13–15] are available. We report details in the Appendix C. Regarding IABS configuration, we used the Reference Market Simulation Configuration, introduced in [9], which is widely used as comparison [14, 15, 69]. The configuration includes 5000 noise, 100 value,

![](images/83292a939d3b4f51da639223575ca1321450026b06b6019cc06a41290e528a30.jpg)  
Figure 2: PCA analysis on TSLA 29/01. TRADES covers ${ \bf 6 7 . 0 4 \% }$ of the real distribution, better than the other two methods $( 5 2 . 9 2 \%$ for IABS and $5 7 . 4 9 \%$ for CGAN).

25 momentum agents, and 1 market maker.11 We do not compare with other SoTA methods due to the unavailability of open-source implementations and insufficient details to reproduce the results. In some cases, the code is provided but the results are not reproducible due to computational constraints.

# 7.1 Results

Here, we evaluate the usefulness, realism, and the responsiveness of the generated market simulations for Tesla and Intel. We train two TRADES Models, one for each stock. After training, we freeze the models and use them to generate orders during the simulation phase. A major disadvantage of market simulation is the misalignment in the literature for a single evaluation protocol. All the other works [14, 30, 44, 71] analyze performances with plots, hindering an objective comparison between the models. To fill this gap we adapt the predictive score [73] for market simulation. Predictive score is measured as an MAE, by training a stock mid-price predictive model on synthetic data and testing it on real data, that evaluates the usefulness of the generated simulations. Appendix D details the computation of the predictive score.

Table 1: Average predictive score (MAE) over two days on Tesla and Intel stocks. Bold values show the best MAE.   

<table><tr><td></td><td colspan="2">Predictive Score↓</td></tr><tr><td>Method</td><td>Tesla</td><td>Intel</td></tr><tr><td>Market Replay</td><td>0.923</td><td>0.149</td></tr><tr><td>IABS</td><td>1.870</td><td>1.866</td></tr><tr><td>CGAN</td><td>3.453</td><td>0.699</td></tr><tr><td>TRADES</td><td>1.213</td><td>0.307</td></tr></table>

Usefulness: TRADES outperforms the second-best by a factor of $\times 3 . 2 7$ and $\times 3 . 4 8$ on both stocks.12 We report both stocks’ average predictive scores in Table 1. We report the predictive score on the market replay as ground truth (market replay). Notice that the market replay scores represent the desired MAE of each model – i.e., the lower the difference in MAE with the market replay, the better. The table reveals that TRADES exhibits performances approaching that of the real market replay, with an absolute difference of 0.29 and 0.158 from market replay, respectively, for Tesla and Intel, suggesting a diminishing gap between synthetic and real-data training efficacy. Interestingly, although IABS cannot capture the complexity of real trader agents, it outperforms CGAN on Tesla, while it remains the worst-performing strategy on Intel. Note that the mid-price for Intel is, on average, $1 / 2 0 \mathrm { { t h } }$ of the mid-price for Tesla. This discrepancy explains the difference in the scale of the predictive score. In conclusion, we demonstrated how a predictive model trained with TRADES’s generated market data can effectively forecast mid-price.

Realism: TRADES covers the real data distribution and emulates many stylized facts. To evaluate the realism of the generated time series, we compare the real data distributions with those of the generated one. We employ a combination of Principal Component Analysis (PCA) [56], alongside specialized financial methodologies that include the comparison of stylized facts and important features. In Fig. 2, we show PCA plots to illustrate how well the synthetic distribution covers the original data. TRADES (blue points) cover $6 7 . 0 4 \%$ of the real data distribution (red points) according to a Convex Hull intersection compared to $5 2 . 9 2 \%$ and $5 7 . 4 9 \%$ of CGAN and IABS, respectively. Following [69], we evaluate the simulation realism by ensuring that the generated market data mimics stylized facts derived from real market scenarios. Fig. 3 (1) illustrates that TRADES, similarly to the market replay and differently to IABS and CGAN, obey the expected absence of autocorrelation, indicating that the linear autocorrelation of asset returns decays rapidly, becoming statistically insignificant after 15 minutes. Fig. 3 (2) highlights TRADES’s resemblance with the positive volume-volatility correlation. TRADES’s generated orders show a positive volume-volatility correlation. Interestingly, TRADES captures this phenomenon better than that particular market replay day. Recall that TRADES is trained on 17 days of the market, which shows this characteristic correctly capturing this correlation. We acknowledge that the real market might not always respect all the stylized facts due to their inherent non-deterministic and non-stationary nature. Also, CGAN resembles this phenomenon but, differently from TRADES, with an unrealistic intensity.

![](images/0298a68a96243e0a8c3c7e7d66db1b5816185a149486068c01b347d09fcc8af9.jpg)  
Figure 3: Stylized facts on Tesla 29/01. (1) Log returns autocorrelation. (2) The correlation between volume and volatility, and (3) between returns and volatility. (4) Comparison of the minute Log Returns distribution and (5) autocorrelation. (6) Mid-price traces of five different TRADES simulations.

Fig 3 (3) shows that TRADES exhibits asset returns and volatility negative correlation emulating the market replay distribution in contrast with IABS. Similar to the previous case, also, CGAN resembles this phenomenon but, differently from TRADES, with an unrealistic intensity. Fig. 3 (4) illustrates that TRADES almost perfectly resembles the real distribution in terms of log returns. We leave IABS out since it disrupts the plot’s scale. In Fig. 3 (5), we show the autocorrelation function of the squared returns, commonly used as an analytical tool to measure the degree of volatility clustering: i.e., high-volatility episodes exhibit a propensity to occur in close temporal proximity. Note that TRADES emulates the real distribution better than the other two methods. In Fig. 3 (6), we illustrate the mid-price time series of five different TRADES13 simulations on the same day of Tesla (29/01) and the market replay of that day. The mid-price traces generated show diversity and realism. Lastly, to consolidate our claims about TRADES’s realism, in Fig. 5 we also analyze the volume distribution of the first LOB level. Notice how the scale and the overall behavior of the volume time series in the market replay strongly correlate with that of TRADES’s simulation, while it is completely different w.r.t. SoTA approaches.

Responsiveness: TRADES is responsive to external agent. The responsiveness of a LOB market simulation generative model is crucial, especially if the objective of the market simulation is to verify the profitability of trading strategies or perform market impact experiments. Generally [14, 15, 55], the responsiveness of a generator is assessed through a market impact experiment (A/B test) [19]. Therefore, we conducted an experiment running some simulations w/ and w/o a Percentage-Of-Volume (POV) agent, which wakes up every minute and places a bunch of buy orders, until either $\phi$ shares have been transacted or the temporal window ends. We refer the reader to Appendix F for the details of the settings of this experiment. Fig. 4 depicts the normalized mid-price difference between the simulations w/ and w/o the POV agent for the market replay and TRADES. Results are averaged over 5 runs. As expected, the historical market simulation exhibits only instantaneous impact [23], that is the direct effect of the agent’s orders, which rapidly vanishes. Contrarily, the diffusion-based simulations demonstrate substantial deviation from the baseline simulation without the POV agent, altering the price permanently. Quantifying the permanent price impact in real markets poses a significant challenge, as it requires comparing price differences between scenarios where a specific action took place and those where it did not. Such scenario analysis is not feasible with empirical data. However, by using TRADESgenerated realistic simulations, this analysis becomes both feasible and measurable. In fact, the simulations allow us to run identical scenarios both with and without additional trader agents which strategy can be fully defined by the user. These types of counterfactual or "what if" experiments can also be used to have an initial analysis of the consequences of changing financial regulations and trading rules, such as price variation limits, short selling regulation, tick size, and usage rate of dark pools. In conclusion, the observed market impact in the TRADES simulations aligns with real market observations [7, 23], enabling the evaluation of trading strategies14 and counterfactual experiments.

# 7.2 DDIM sampling

One of the known limitations of diffusion models is the sampling time. Indeed, the generation of a single sample necessitates hundreds of iterative passes through the neural network. In this work, each model was trained using a diffusion process comprising 100 steps. Recently, Denoising Diffusion Implicit Model (DDIM) sampling method was proposed in [60] to speed up the generative process. Given that each hour of market simulation required six hours of computation on an RTX 3090, accelerating the simulation process was a relevant improvement. Consequently, we conducted simulations employing DDIM sampling $( \eta = 0 )$ , which is deterministic, utilizing a single step for each order. We use the same trained model. The results, presented in Table 2, demonstrate that the performance degradation is significant but not disastrous despite a remarkable 100-fold increase in computational efficiency.

# 7.3 Ablation and sensitivity studies

Table 3 shows two ablations (i.e., LOB conditioning and augmentation) and two sensitivity analyses (i.e., backbone choice and conditioning method) that highlight the effectiveness of TRADES design choice.

Ablation analyses. We verify two of the hypotheses made in the method design: (1) how much the LOB conditioning part is necessary for the task and (2) how augmenting the feature vectors influences the performance. When we include LOB in the conditioning w.r.t. last orders only, TRADES has an average gain of 2.473. When we augment the features through the MLPs in Fig. 1, we gain an average of 1.980 MAE absolute points.

![](images/56671e37f37898641815660090af29c678b281a165cedd1d839e9458274329d7.jpg)  
Figure 4: Average mid-price difference of market replay simulations and TRADES simulations with (shaded part) and without a POV agent (unshaded part), on 5 different seeds.

![](images/a1aedcd7bf781c53c3cb980f87639debc99db839c95e39d3d6a41dad829eb88a.jpg)  
Figure 5: Volume at the first level of LOB on TSLA 29/01.

Table 2: Average predictive score (MAE) over two days on Tesla and Intel stocks. DDIM sampling is done with a single step, while DDPM with 100.   

<table><tr><td rowspan="2">Method</td><td colspan="2">Predictive Score↓</td></tr><tr><td>Tesla</td><td>Intel</td></tr><tr><td>DDIM</td><td>3.146</td><td>0.486</td></tr><tr><td>DDPM (orig.)</td><td>1.213</td><td>0.307</td></tr></table>

Sensitivity analyses. We aim to verify whether the complexity of the backbone – i.e., the transformer in TRADES – is needed after all. Therefore, we replace the transformer backbone with an LSTM, leaving the augmentation and conditioning invariant. The table shows that the performances degrade by an average of

Table 3: Predictive score for the ablation (A) and sensitivity (S) analyses for two days of Tesla simulations.   

<table><tr><td rowspan="2">Method</td><td colspan="2">Predictive Score↓</td></tr><tr><td>29/01</td><td>30/01</td></tr><tr><td>A</td><td>TRADES w/0 LOB</td><td>2.642 4.728</td></tr><tr><td> TRADES w/o Aug.</td><td>1.442</td><td>4.942</td></tr><tr><td rowspan="2">LSTM backbone S TRADES w/ CA</td><td>8.391</td><td>6.153</td></tr><tr><td>11.90</td><td>4.891</td></tr><tr><td>TRADES (orig.)</td><td>1.336</td><td>1.089</td></tr></table>

6.06 absolute points. This is expected since transformers directly access all other steps in the sequence via self-attention, which theoretically leaves no room for information loss that occurs in LSTMs. Recall that we concatenate the past orders and the LOB snapshots, after the augmentation, into a single tensor and use it to condition the diffusion model – i.e., TRADES (Orig.). We also tried a cross-attention (CA) conditioning strategy – see TRADES w/ CA – between the past orders and the LOB snapshots. TRADES w/ CA reports an average performance loss of 7.814 absolute points w.r.t. the original architecture. Note that cross-attention limits the model’s capability because the orders cannot attend to each other but only to LOB and vice-versa. Instead, with concatenation and self-attention, every sequence part can attend to every other vector.

# 8 Conclusion

We proposed the Transformer-based Denoising Diffusion Probabilistic Engine for LOB Simulations (TRADES) to generate realistic order flows conditioned on the current market state. We evaluated TRADES’s realism and responsiveness. We also adapted the predictive score to verify the usefulness of the generated market data by training a prediction model on them and testing it on real ones. This shows that TRADES can cover the real data distribution by $6 7 \%$ on average and outperforms SoTA by $\times 3 . 2 7$ and $\times 3 . 4 8$ on Tesla and Intel. Furthermore, our analyses reflect that TRADES correctly abides by many stylized facts used to evaluate the goodness of financial market simulation approaches. We release DeepMarket, a Python framework for market simulation with deep learning and TRADES-LOB, a synthetic LOB dataset composed of TRADES’s generated market simulations. We argue that TRADES-LOB and DeepMarket will have a positive impact on the research community, as almost no LOB data is freely available. We believe that TRADES is a viable market simulation strategy in controlled environments, and further tests must be performed to have a mature evaluation trading strategy protocol.

# References

[1] Robert F Almgren. 2003. Optimal execution with nonlinear impact functions and trading-enhanced risk. Applied mathematical finance 10, 1 (2003), 1–18.   
[2] Martin Arjovsky, Soumith Chintala, and Léon Bottou. 2017. Wasserstein generative adversarial networks. In International conference on machine learning. PMLR, 214–223.   
[3] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. 2021. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems 34 (2021), 17981–17993.   
[4] Lukas Biewald. 2020. Experiment Tracking with Weights and Biases. https: //www.wandb.com/ Software available from wandb.com.   
[5] Marin Biloš, Kashif Rasul, Anderson Schneider, Yuriy Nevmyvaka, and Stephan Günnemann. 2023. Modeling temporal data as continuous functions with stochastic process diffusion. In International Conference on Machine Learning. PMLR, 2452–2470.   
[6] J.P. Bouchaud, J. Bonart, J. Donier, and M. Gould. 2018. Trades, Quotes and Prices: Financial Markets Under the Microscope. Cambridge University Press. https://books.google.it/books?id=u45LDwAAQBAJ   
[7] Jean-Philippe Bouchaud, J Doyne Farmer, and Fabrizio Lillo. 2009. How markets slowly digest changes in supply and demand. In Handbook of financial markets: dynamics and evolution. Elsevier, 57–160.   
[8] Jean-Philippe Bouchaud, Marc Mézard, and Marc Potters. 2002. Statistical properties of stock order books: empirical results and models. Quantitative finance 2, 4 (2002), 251.   
[9] David Byrd, Maria Hybinette, and Tucker Hybinette Balch. 2020. ABIDES: Towards high-fidelity multi-agent market simulation. In Proceedings of the 2020 ACM SIGSIM Conference on Principles of Advanced Discrete Simulation. 11–22.   
[10] Charles Cao, Oliver Hansch, and Xiaoxin Wang. 2008. Order placement strategies in a pure limit order book market. Journal of Financial Research 31, 2 (2008), 113–140.   
[11] Charles Cao, Oliver Hansch, and Xiaoxin Wang. 2009. The information content of an open limit-order book. Journal of Futures Markets: Futures, Options, and Other Derivative Products 29, 1 (2009), 16–41.   
[12] Casey Chu, Kentaro Minami, and Kenji Fukumizu. 2020. Smoothness and stability in gans. arXiv preprint arXiv:2002.04185 (2020).   
[13] Andrea Coletta, Joseph Jerome, Rahul Savani, and Svitlana Vyetrenko. 2023. Conditional generators for limit order book environments: Explainability, challenges, and robustness. In Proceedings of the Fourth ACM International Conference on AI in Finance. 27–35.   
[14] Andrea Coletta, Aymeric Moulin, Svitlana Vyetrenko, and Tucker Balch. 2022. Learning to simulate realistic limit order book markets from data as a World Agent. In Proceedings of the Third ACM International Conference on AI in Finance. 428–436.   
[15] Andrea Coletta, Matteo Prata, Michele Conti, Emanuele Mercanti, Novella Bartolini, Aymeric Moulin, Svitlana Vyetrenko, and Tucker Balch. 2021. Towards realistic market simulations: a generative adversarial networks approach. In Proceedings of the Second ACM International Conference on AI in Finance. 1–9.   
[16] Rama Cont. 2001. Empirical properties of asset returns: stylized facts and statistical issues. Quantitative finance 1, 2 (2001), 223.   
[17] Rama Cont. 2011. Statistical modeling of high-frequency financial data. IEEE Signal Processing Magazine 28, 5 (2011), 16–25.   
[18] Rama Cont, Mihai Cucuringu, Jonathan Kochems, and Felix Prenzel. 2023. Limit Order Book Simulation with Generative Adversarial Networks. Available at SSRN 4512356 (2023).   
[19] Rama Cont, Arseniy Kukanov, and Sasha Stoikov. 2013. The Price Impact of Order Book Events. Journal of Financial Econometrics 12, 1 (06 2013), 47–88. https: //doi.org/10.1093/jjfinec/nbt003 arXiv:https://academic.oup.com/jfec/articlepdf/12/1/47/2439285/nbt003.pdf   
[20] Rama Cont, Arseniy Kukanov, and Sasha Stoikov. 2014. The price impact of order book events. Journal of financial econometrics 12, 1 (2014), 47–88.   
[21] Rama Cont, Sasha Stoikov, and Rishi Talreja. 2010. A stochastic model for order book dynamics. Operations research 58, 3 (2010), 549–563.   
[22] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).   
[23] Martin D Gould, Mason A Porter, Stacy Williams, Mark McDonald, Daniel J Fenn, and Sam D Howison. 2013. Limit order books. Quantitative Finance 13, 11 (2013), 1709–1742.   
[24] Campbell R Harvey and Yan Liu. 2015. Backtesting. The Journal of Portfolio Management 42, 1 (2015), 13–28.   
[25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.   
[26] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation 9, 8 (1997), 1735–1780.   
[27] Charles Huang, Weifeng Ge, Hongsong Chou, and Xin Du. 2021. Benchmark dataset for short-term market prediction of limit order book in china markets. The Journal of Financial Data Science 3, 4 (2021), 171–183.   
[28] Roger D Huang and Hans R Stoll. 1994. Market microstructure and stock return predictions. The Review of Financial Studies 7, 1 (1994), 179–213.   
[29] Weibing Huang, Charles-Albert Lehalle, and Mathieu Rosenbaum. 2015. Simulating and analyzing order book data: The queue-reactive model. J. Amer. Statist. Assoc. 110, 509 (2015), 107–122.   
[30] Hanna Hultin, Henrik Hult, Alexandre Proutiere, Samuel Samama, and Ala Tarighati. 2023. A generative model of a limit order book using recurrent neural networks. Quantitative Finance (2023), 1–28.   
[31] Bruce I Jacobs, Kenneth N Levy, and Harry M Markowitz. 2004. Financial market simulation. The Journal of Portfolio Management 30, 5 (2004), 142–152.   
[32] Konark Jain, Nick Firoozye, Jonathan Kochems, and Philip Treleaven. 2024. Limit Order Book Simulations: A Review. arXiv preprint arXiv:2402.17359 (2024).   
[33] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. 2022. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems 35 (2022), 26565–26577.   
[34] Jerzy Korczak and Marcin Hemes. 2017. Deep learning for financial time series forecasting in a-trader system. In 2017 Federated Conference on Computer Science and Information Systems (FedCSIS). IEEE, 905–912.   
[35] Haim Levy, Moshe Levy, and Sorin Solomon. 2000. Microscopic simulation of financial markets: from investor behavior to market phenomena. Elsevier.   
[36] Junyi Li, Xintong Wang, Yaoyang Lin, Arunesh Sinha, and Michael Wellman. 2020. Generating realistic stock market order streams. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34. 727–734.   
[37] Yan Li, Xinjiang Lu, Yaqing Wang, and Dejing Dou. 2022. Generative time series forecasting with diffusion, denoise, and disentanglement. Advances in Neural Information Processing Systems 35 (2022), 23009–23022.   
[38] Haksoo Lim, Minjung Kim, Sewon Park, and Noseong Park. 2023. Regular timeseries generation using sgm. arXiv preprint arXiv:2301.08518 (2023).   
[39] Benjamin Lindemann, Timo Müller, Hannes Vietz, Nasser Jazdi, and Michael Weyrich. 2021. A survey on long short-term memory networks for time series prediction. Procedia Cirp 99 (2021), 650–655.   
[40] Iwao Maeda, David DeGraw, Michiharu Kitano, Hiroyasu Matsushima, Hiroki Sakaji, Kiyoshi Izumi, and Atsuo Kato. 2020. Deep reinforcement learning in agent based financial market simulation. Journal of Risk and Financial Management 13, 4 (2020), 71.   
[41] Kangfu Mei and Vishal Patel. 2023. Vidm: Video implicit diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37. 9117–9125.   
[42] Takanobu Mizuta. 2016. A brief review of recent artificial market simulation (agent-based model) studies for financial market regulations and/or rules. Available at SSRN 2710495 (2016).   
[43] Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjey Choi, and Jaejun Yoo. 2020. Reliable fidelity and diversity metrics for generative models. In International Conference on Machine Learning. PMLR, 7176–7185.   
[44] Peer Nagy, Sascha Frey, Silvia Sapora, Kang Li, Anisoara Calinescu, Stefan Zohren, and Jakob Foerster. 2023. Generative AI for End-to-End Limit Order Book Modelling: A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network. arXiv preprint arXiv:2309.00638 (2023).   
[45] Alexander Quinn Nichol and Prafulla Dhariwal. 2021. Improved denoising diffusion probabilistic models. In International conference on machine learning. PMLR, 8162–8171.   
[46] Adamantios Ntakaris, Martin Magris, Juho Kanniainen, Moncef Gabbouj, and Alexandros Iosifidis. [n. d.]. Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods. http://urn.fi/urn:nbn:fi: csc-kata20170601153214969115. N/A.   
[47] Mark Paddrik, Roy Hayes, Andrew Todd, Steve Yang, Peter Beling, and William Scherer. 2012. An agent based model of the E-Mini S&P 500 applied to Flash Crash analysis. In 2012 IEEE Conference on Computational Intelligence for Financial Engineering & Economics (CIFEr). IEEE, 1–8.   
[48] Roberto Pascual and David Veredas. 2003. What pieces of limit order book information do are informative? an empirical analysis of a pure order-driven market. (2003).   
[49] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. arXiv:1912.01703 [cs.LG]   
[50] Matteo Prata, Giuseppe Masi, Leonardo Berti, Viviana Arrigoni, Andrea Coletta, Irene Cannistraci, Svitlana Vyetrenko, Paola Velardi, and Novella Bartolini. 2023. LOB-Based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study. arXiv:2308.01915 [q-fin.TR]   
[51] Marco Raberto and Silvano Cincotti. 2005. Modeling and simulation of a double auction artificial financial market. Physica A: Statistical Mechanics and its applications 355, 1 (2005), 34–45.   
[52] Marco Raberto, Silvano Cincotti, Sergio M. Focardi, and Michele Marchesi. 2001. Agent-based simulation of a financial market. Physica A: Statistical Mechanics and its Applications 299, 1 (2001), 319–327. https://doi.org/10.1016/S0378-4371(01) 00312-0 Application of Physics in Economic Modelling.   
[53] Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. 2021. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. In International Conference on Machine Learning. PMLR, 8857–8868.   
[54] Zijian Shi and John Cartlidge. 2022. State dependent parallel neural hawkes process for limit order book event stream prediction and simulation. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1607–1615.   
[55] Zijian Shi and John Cartlidge. 2023. Neural Stochastic Agent-Based Limit Order Book Simulation: A Hybrid Methodology. In Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2023, London, United Kingdom, 29 May 2023 - 2 June 2023, Noa Agmon, Bo An, Alessandro Ricci, and William Yeoh (Eds.). ACM, 2481–2483. https://doi.org/10.5555/3545946. 3598974   
[56] Jonathon Shlens. 2014. A tutorial on principal component analysis. arXiv preprint arXiv:1404.1100 (2014).   
[57] Justin A Sirignano. 2019. Deep learning for limit order books. Quantitative Finance 19, 4 (2019), 549–570.   
[58] Jimmy T. H. Smith, Andrew Warrington, and Scott W. Linderman. 2023. Simplified State Space Layers for Sequence Modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. https://openreview.net/pdf?id=Ai8Hw3AXqks   
[59] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning. PMLR, 2256–2265.   
[60] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020).   
[61] Yang Song and Stefano Ermon. 2019. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems 32 (2019).   
[62] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020).   
[63] Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. 2021. Csdi: Conditional score-based diffusion models for probabilistic time series imputation. Advances in Neural Information Processing Systems 34 (2021), 24804–24816.   
[64] Hoang Thanh-Tung and Truyen Tran. 2020. Catastrophic forgetting and mode collapse in GANs. In 2020 international joint conference on neural networks (ijcnn). IEEE, 1–10.   
[65] Dat Thanh Tran, Juho Kanniainen, and Alexandros Iosifidis. 2022. How informative is the order book beyond the best levels? Machine learning perspective. arXiv preprint arXiv:2203.07922 (2022).   
[66] Avraam Tsantekidis, Nikolaos Passalis, Anastasios Tefas, Juho Kanniainen, Moncef Gabbouj, and Alexandros Iosifidis. 2017. Forecasting stock prices from the limit order book using convolutional neural networks. In 2017 IEEE 19th conference on business informatics (CBI), Vol. 1. IEEE, 7–12.   
[67] Avraam Tsantekidis, Nikolaos Passalis, Anastasios Tefas, Juho Kanniainen, Moncef Gabbouj, and Alexandros Iosifidis. 2017. Using deep learning to detect price change indications in financial markets. In 2017 25th European Signal Processing Conference (EUSIPCO). IEEE, 2511–2515.   
[68] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).   
[69] Svitlana Vyetrenko, David Byrd, Nick Petosa, Mahmoud Mahfouz, Danial Dervovic, Manuela Veloso, and Tucker Balch. 2020. Get real: Realism metrics for robust limit order book market simulations. In Proceedings of the First ACM International Conference on AI in Finance. 1–8.   
[70] Kevin T Webster. 2023. Handbook of Price Impact Modeling. Chapman and Hall/CRC.   
[71] Haochong Xia, Shuo Sun, Xinrun Wang, and Bo An. 2024. Market-GAN: Adding Control to Financial Market Data Generation with Semantic Context. In ThirtyEighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (Eds.). AAAI Press, 15996–16004. https://doi.org/10.1609/AAAI. V38I14.29531   
[72] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. 2021. Tackling the generative learning trilemma with denoising diffusion gans. arXiv preprint arXiv:2112.07804 (2021).   
[73] Jinsung Yoon, Daniel Jarrett, and Mihaela Van der Schaar. 2019. Time-series generative adversarial networks. Advances in neural information processing systems 32 (2019).   
[74] Zihao Zhang, Stefan Zohren, and Stephen Roberts. 2019. Deeplob: Deep convolutional neural networks for limit order books. IEEE Transactions on Signal Processing 67, 11 (2019), 3001–3012.   
[75] Zihao Zhang, Stefan Zohren, and Stephen Roberts. 2020. Deep reinforcement learning for trading. The Journal of Financial Data Science 2, 2 (2020), 25–40.   
[76] Yuanzhi Zhu, Zhaohai Li, Tianwei Wang, Mengchao He, and Cong Yao. 2023. Conditional text image generation with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14235–14245.

# A Denoising diffusion probabilistic model

Diffusion models are latent variable models of the form $p _ { \theta } ( \mathbf { x } _ { 0 } ) : =$ $\int p _ { \theta } ( \mathbf { x } _ { 0 : T } ) d \mathbf { x } _ { 1 : T }$ , where $\mathbf { x } _ { 1 } , . . . , \mathbf { x } _ { T }$ are latents of the same dimensionality as the original data sample $x _ { 0 } \approx q ( x _ { 0 } )$ . The objective of diffusion models is to learn a model distribution $p _ { \theta } ( x _ { 0 } )$ that approximates the data distribution $q ( x _ { 0 } )$ . Diffusion probabilistic models [59] are latent variable models composed of two Markov chain processes, i.e., the forward and reverse processes. The forward process is defined as in Eq. (11).

$$
q ( \mathbf { x } _ { 0 : T } ) : = q ( \mathbf { x } _ { 0 } ) \prod _ { t = 1 } ^ { T } q ( \mathbf { x } _ { t } | \mathbf { x } _ { t - 1 } ) ,
$$

where $q ( \mathbf { x } _ { t } | \mathbf { x } _ { t - 1 } ) : = N ( \mathbf { x } _ { t } ; \sqrt { 1 - \beta _ { t } } \mathbf { x } _ { t - 1 } , \beta _ { t } \mathbf { I } )$ . We start from an initial sample $x _ { 0 }$ and add a small amount of Gaussian noise with every step for $T$ steps, according to a variance schedule $\beta _ { 1 } , . . . , \beta _ { T }$ . The schedule is deterministic and defined, so $\mathbf { x } _ { T }$ is pure Gaussian noise. Sampling of $\mathbf { x } _ { t }$ can be define in a closed form $q ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) : =$ $N ( \mathbf { x } _ { t } ; \sqrt { \bar { \alpha } _ { t } } \mathbf { \bar { x } } _ { t - 1 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } )$ , where $\alpha _ { t } : = 1 - \beta _ { t }$ and $\textstyle { \bar { \alpha } } _ { t } : = \prod _ { i = 1 } ^ { t } \alpha _ { i }$ . During the reverse process, $\mathbf { X } T$ is denoised to recover $\mathbf { x } _ { 0 }$ following the Markov chain process in Eq. (12).

$$
\begin{array} { r l } & { ~ p _ { \theta } ( \mathbf { x } _ { 0 : T } ) : = p ( \mathbf { x } _ { T } ) \prod _ { t = 1 } ^ { T } \ L { p _ { \theta } ( \mathbf { x } _ { t - 1 } | \mathbf { x } _ { t } ) } , } \\ & { ~ p _ { \theta } ( \mathbf { x } _ { t - 1 } | \mathbf { x } _ { t } ) : = N ( \mathbf { x } _ { t - 1 } ; \mu _ { \theta } ( \mathbf { x } _ { t } , t ) , \Sigma _ { \theta } ( \mathbf { x } _ { t } , t ) ) } \end{array}
$$

The reverse process model is trained with the variational lower bound of the likelihood of $\mathbf { x } _ { 0 }$ as in Eq. (13).

$$
\begin{array} { c } { \displaystyle \mathcal { L } _ { v l b } : = \mathbb { B } _ { q } \Bigg [ \underbrace { - \{ \rho _ { \theta } ( \mathbf { x } _ { 0 } | \mathbf { x } _ { 1 } ) } _ { L _ { 0 } } } _ { L _ { 0 } }  \\ { + \displaystyle \sum _ { t = 2 } ^ { T } \underbrace { D _ { K L } ( q ( \mathbf { x } _ { t - 1 } | \mathbf { x } _ { t } , \mathbf { x } _ { 0 } ) | | \{ p _ { \theta } ( \mathbf { x } _ { t - 1 } | \mathbf { x } _ { t } ) \} ) } _ { L _ { t - 1 } } } \\ { + \underbrace { D _ { K L } ( q ( \mathbf { x } _ { T } | \mathbf { x } _ { 0 } ) | | \{ p _ { \theta } ( \mathbf { x } _ { T } ) } \Bigg ] . } _ { L _ { T } }  \end{array}
$$

Since both $q$ and $\hbar \theta$ are Gaussian, $D _ { K L }$ (the Kullback–Leibler divergence) can be evaluated in a closed form with only the mean and covariance of the two distributions. Ho et al. [25] propose the following reparametrization of $p _ { \theta } ( \mathbf { x } _ { t - 1 } | \mathbf { x } _ { t } )$ :

$$
{ \pmb \mu } _ { \theta } ( { \bf x } _ { t } , t ) = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( { \bf x } _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } { \pmb \epsilon } _ { \theta } ( { \bf x } _ { t } , t ) \right) ,
$$

where $\mathbf { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$ s.t. $\epsilon \sim { \cal N } ( 0 , \mathrm { I } )$ , and $\Sigma _ { \theta } ( \mathbf { x } _ { t } , t ) = \sigma _ { t } ^ { 2 } \mathbf { I }$ where

$$
\sigma _ { t } ^ { 2 } = \left\{ \beta _ { 1 } \quad t = 1 \atop { \tilde { \beta } _ { t } \quad 1 < t \leq T } \quad \mathrm { a n d } \tilde { \beta } _ { t } = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \beta _ { t } , \right.
$$

where $\epsilon _ { \theta }$ is the trainable function approximated by the neural network, intended to predict $\epsilon$ from $\mathbf { x } _ { t }$ . As shown in [61], the denoising function given by Eq. (14) is equivalent to a score model rescaled for score-based generative models. Using this parameterization, Ho et al. [25] demonstrated that the inverse process can be learned by minimizing the simplified objective function in Eq. (16).

$$
\mathcal { L } _ { \mathrm { s i m p l e } } ( \theta ) : = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \epsilon } \bigg [ | | \epsilon - \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) | | ^ { 2 } \bigg ]
$$

The denoising function $\epsilon _ { \theta }$ aims to recover the noise vector $\epsilon$ that corrupted its input $\mathbf { x } _ { t }$ . This training objective can be interpreted as a weighted variant of denoising score matching, a method for training score-based generative models [61, 62].

# B Overview of FI-2010 dataset

The FI-2010 dataset [46] is the most used LOB dataset in the field of the deep learning application to limit order book [66, 67, 74, 75], especially for forecasting tasks. It contains LOB data from five Finnish companies listed on the NASDAQ Nordic stock market: i.e., Kesko Oyj, Outokumpu Oyj, Sampo, Rautaruukki, and Wärtsilä Oyj. The data covers 10 trading days from June 1st to June 14th, 2010. It records about 4 million limit order snapshots for 10 levels of the LOB. The authors sample LOB observations every 10 events, totaling 394,337 events. The label of the data, representing the midprice movement, depends on the percentage change between the actual price ${ \bf \nabla } { p } _ { t }$ and the average of the subsequent $h$ (chosen horizon) mid-prices:

$$
l _ { t } = \frac { m _ { + } ( t ) - p _ { t } } { p _ { t } }
$$

The labels are then decided based on a threshold $\mathbf { \eta } ^ { ( \theta ) }$ for the percentage change $( l _ { t } )$ . If $l _ { t } > \theta$ or $l _ { t } < - \theta$ , the label is an $u p$ in the former case, and a down in the latter. When $- \theta < l _ { t } < \theta$ , the label is stationary. The dataset provides the time series and the classes for five horizons $h \in H = \{ 1 , 2 , 3 , 5 , 1 0 \}$ . The authors of the dataset employed a single threshold $\theta = 2 \times 1 0 ^ { - 3 }$ for all horizons, balancing only the case for $h = 5$ .

Besides the absence of message files, FI-2010 comes already preprocessed such that the original LOB cannot be reconstructed, impeding comprehensive experimentation. Finally, as shown in [74], this method of labeling the data is susceptible to instability.

# C Implementation Details

All the experiments are run with an RTX 3090 and an A100. We implemented early stopping with a patience of 5 epochs and then used the checkpoint with the best validation loss for simulation. On average we trained each model for 70,000 steps until convergence. Lastly, we halve the learning rate each time the validation loss exceeds the previous one.

Pre-processing. To properly train the neural network, we preprocess the data. The data contains discrete and continuous features, so we rely on different preprocessing methods for each feature type. We replace the timestamp with the time distance from the previous order, which we normalize according to the $\mathbf { Z }$ -score. We also perform a z-score normalization of message and order book file volume and prices. We encode the event type with an embedding layer. Finally, we remove the order ID and add the depth15 to the orders feature vector.

Post-processing. The diffusion output is post-processed before being handled through the exchange simulation. First, we report the continuous features (offset, size, depth) to the original scale using the mean and standard deviation computed on the training set; the direction is discretized by simply checking if it is $> ~ 0$ (buy) or $< 0$ (sell). Lastly, the order type is discretized between 0 (limit order), 1 (cancel order), and 2 (market order) based on the index of the nearest16 embedding layer row. If the order is a limit order, the depth is utilized as the price. For instance, if the depth is 10 and the direction is “buy”, the order will be positioned at a 10-cent difference from the best available bid price. Occasionally, it happens that the size is negative or the depth is over the first 10 levels; in that case, the order is discarded, and a new one is generated. Approximately $2 5 \%$ of the time, the generated cancel order does not directly correspond to an active limit order with the same depth. We identify the limit order with the closest depth and size in such instances.

Hyperparameters search. To find the best hyperparameters, we employ a grid search exploring different values as shown in Table 4. Furthermore, we set the number of diffusion steps to 100. Lastly, We set $\lambda = 0 . 0 1$ to prevent $\mathcal { L } _ { \Sigma }$ from overwhelming $\mathcal { L } _ { \epsilon }$ . We implement this mixed training by relying on the stop gradient functionality [49], in such a way that $\mathcal { L } _ { \epsilon }$ optimizes the error prediction and $\mathcal { L } _ { \Sigma }$ the standard deviation.

Table 4: The hyperparameter search spaces and best choice.   

<table><tr><td>Hyperparameter</td><td> Search Space</td><td>Best Choice</td></tr><tr><td>Optimizer</td><td>{Adam, Lion}</td><td>Adam</td></tr><tr><td>Sequence size</td><td>{64,128,256, 512}</td><td>256</td></tr><tr><td>Learning Rate</td><td>{10-3,10-4}</td><td>10-3</td></tr><tr><td>TRADES Layers</td><td>{4,6,8,16}</td><td>8</td></tr><tr><td>Dropout</td><td>{0,0.1}</td><td>0.1</td></tr><tr><td>Attention Heads</td><td>{1,2,4}</td><td>2</td></tr><tr><td>Augmentation Dim.</td><td>{32,64,128,256}</td><td>64</td></tr><tr><td>入</td><td>{0.1, 0.01, 0.001}</td><td>0.01</td></tr></table>

Noise scheduler. As shown in [45], too much noise at the end of the forward noising process lowers the sample’s quality. Hence, we rely on a non-linear noise scheduler as described in Eq. (17).

$$
\overline { { a _ { t } } } = \frac { f ( t ) } { f ( 0 ) } , \ f ( t ) = \cos \left( \frac { t / T + s } { 1 + s } \cdot \frac { \pi } { 2 } \right) ^ { 2 }
$$

Importance Sampling. Because some diffusion steps contribute to most of the loss, we exploit importance sampling [45] to focus on these steps as in Eq. (18).

$$
\mathcal { L } _ { \Sigma } = \mathbf { E } _ { t \sim p _ { t } } \left[ \frac { \mathcal { L } _ { t } } { \rho _ { t } } \right] , ~ \mathrm { w h e r e } ~ p _ { t } \propto \sqrt { \mathbf { E } [ \mathcal { L } _ { t } ^ { 2 } ] } ~ \mathrm { a n d } ~ \sum _ { t } p _ { t } = 1 .
$$

Diffusion step and positional embedding. We embed the diffusion step $t$ and each vector’s position in the sequence using sinusoidal embedding [68]. Obviously, the diffusion step $t$ embedding is one for the whole sequence, while the position embedding is different for each element.

CGAN implementation. We implemented CGAN from scratch given that none of the implementations in [13–15] are available. Furthermore, we performed a hyperparameters search because the majority of the hyperparameters are not specified. The generator comprises an LSTM, a fully connected, and four 1D convolution layers with a final tanh activation function. Each convolution layer is interleaved with batch normalization and ReLUs. The kernel size of each convolution is 4, and the stride is 2. The optimizer, the learning rate, and the sequence size are the same as TRADES.

An important detail is how the discrete features are post-processed after the tanh function during the generation. We set the binomial feature (direction, quantity type) to -1 if the value is less than 0 and vice versa. While, for the order type, we suppose17 that the authors search for a threshold that resembles the real order type distribution. Our final strategy for Tesla is: if the value is lower than 0.1, we assign $^ { - 1 }$ (limit order); if the value is between 0.1 and 0.25, we set it to 0 (cancel order) and with 1 (market order) otherwise. For Intel, we do smaller changes: if the value is lower than 0.15, we assign -1 (limit order); if the value is between 0.15 and 0.95, we set it to 0 (cancel order) and with 1 (market order) otherwise. The distribution of the generated data is similar to the real one when exploiting this heuristic. We did our best to implement CGAN most competitively. The full implementation is available in our framework.

# D Predictive Score Calculation

We rely on the predictive score [73] to measure how much the synthetic data is effective for the stock mid-price forecasting task. It is computed by training a predictive model $\Phi$ on synthetic data and measures the MAE on a real test set. The task considered is forecasting the mid-price with a horizon of 10, given in input the last 100 market observations. A market observation contains the last order and the first level of the LOB. We choose a 2-layered LSTM [26], standard architecture for time series forecasting [39], for $\Phi$ . We train a $\Phi$ on the generated market simulation for each generative method and each simulated day (29/1 and 30/01 for both stocks) for 100 epochs at maximum. We used early stopping with 5 patience. Next, we evaluate each $\Phi$ on the real test set extracted from the market replay.18 In addition, a comparative $\Phi$ model was trained and tested exclusively on real market data to benchmark performance differences.

# E Additional Results

For completeness, we show the volume (see Fig. 6) and stylized facts (see Fig. 7) on 30/01. Notice how, as shown in Fig. 5, the volume for $3 0 / 0 1$ on Tesla follows realistic trends. Meanwhile, SoTA approaches cannot seem to cope with the GT. Similar reasoning applies to the stylized facts discussed in the main paper (see Fig. 3). Notice how all of them are satisfied, except returns and volatility negative correlation. Interestingly, the correlation between the volume and volatility (2) is slightly negative for the GT. Nevertheless, because TRADES observes a positive correlation during training, this is also reflected in the generated orders.

![](images/aa85a76f36d37768d2e06c060aa6a7ea5225bfeeb03139a178687bd59ddec610.jpg)  
Figure 6: Volume at 1st LOB level extracted from the simulation of TSLA on 30/01

# F Responsiveness Experiment Settings

A $\delta$ -POV strategy is characterized by a percentage level $\delta \in ( 0 , 1 ]$ , a wake-up frequency $\Delta t$ , a direction (buy or sell), and a target quantity of shares $\phi$ . The agent wakes up every $\Delta t$ time unit and places buy or sell orders for several shares equal to $\delta V _ { t }$ . This process continues until either $\phi$ shares have been transacted or the temporal window ends. In our experiments, we use $\delta = 0 . 1$ and a wake-up frequency $\Delta t = 1$ min. The temporal window is set from 09:45 to 10:30, and we set $\phi = 1 0 ^ { 5 }$ .

![](images/847bbd32537c7b0730c6fe3a93961ee54561b9460a62d0bdfd6e29b974cad1ab.jpg)  
Figure 7: Stylized facts of Tesla on 30/01. (1) Log returns autocorrelation. (2) The correlation between volume and volatility, and (3) between returns and volatility. (4-5) Comparison of the minute Log Returns distribution and autocorrelation.