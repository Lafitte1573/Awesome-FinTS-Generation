# Controllable Financial Market Generation with Diffusion Guided Meta Agent

Yu-Hao Huang1, Chang $\mathbf { X } \mathbf { u } ^ { 2 }$ , Yang $\mathbf { L i u ^ { 2 } }$ , Weiqing $\mathbf { L i u ^ { 2 } }$ , ${ \bf W } { \bf u { - } } { \bf J u n L i } ^ { 1 }$ , Jiang Bian2

1National Key Laboratory for Novel Software Technology, Department of Computer Science and Technology, Nanjing University 2Microsoft Research Asia huangyh@smail.nju.edu.cn, {chanx, yangliu2, weiqing.liu}@microsoft.com, liwujun@nju.edu.cn, jiang.bian@microsoft.com

# Abstract

Order flow modeling stands as the most fundamental and essential financial task, as orders embody the minimal unit within a financial market. However, current approaches often result in unsatisfactory fidelity in generating order flow, and their generation lacks controllability, thereby limiting their application scenario. In this paper, we advocate incorporating controllability into the market generation process, and propose a Diffusion Guided meta Agent (DiGA) model to address the problem. Specifically, we utilize a diffusion model to capture dynamics of market state represented by time-evolving distribution parameters about mid-price return rate and order arrival rate, and define a meta agent with financial economic priors to generate orders from the corresponding distributions. Extensive experimental results demonstrate that our method exhibits outstanding controllability and fidelity in generation. Furthermore, we validate DiGA’s effectiveness as generative environment for downstream financial applications.

# 1 Introduction

Generative modeling has impacted a spectrum of machine learning applications, from natural language processing [1], media synthesis [2] to medical applications [3]. Advanced generative methods, such as diffusion models, have demonstrated potentials in producing realistic generation contents [4; 5; 6]. While financial technology is an application sector naturally involved with data of high intricacy, the literature has seldom explored modeling financial data generatively. Most current financial technology (FinTech) efforts take market replay [7] as environment, facing challenges in learning genuine market dynamics interactively. In contrast, market generation provides a more interactive and realistic environment for FinTech tasks, holding potentials for various downstream financial tasks.

Among categories of financial data, order represents the minimal unit of events within a financial market. Since investigating the intrinsic interactive logic and micro-structure of financial market is crucial among researchers [8], investors [9], and policy makers [10], it is important to constitute generated financial worlds that realistically simulate financial markets at order-level with interactiveness [9; 11].

Recent works have attempted to simulate order-level financial market with agent-based methods, either using rule-based agents [12; 13; 14] or learned agents [15; 16]. They aim at replicating “stylized facts” observed in real markets such as volatility clustering, but have resulted in limited fidelity. Ruled-based agents rely on over-simplified assumptions of the market without being trained on real market data, which hurts the simulation fidelity. Learned agents are trained with limited fraction of real world samples due to the intricacy of order-level data, resulting in lower coverage of financial market scenarios. More importantly, the focus of these works is primarily on narrowing the gap between the real and simulated market. The controllability to the generated market is absent from the literature, which is crucial for downstream tasks such as discovering counterfactual cases [11].

Different from existing works, we propose the incorporation of controllability in financial market modeling and formulating the problem as a conditional generation task. The objective involves constructing specific scenarios with varying levels of asset return, intraday volatility, and rare occurrences such as sharp drops or extreme amplitudes. To achieve this, a critical challenge lies in establishing a connection between control targets, such as desired market scenarios, and the generated orders. Inspired by recent advances in diffusion-based generative models with guidance techniques [5; 17; 2], the problem can be addressed by a conditional diffusion model. However, exercising control at the order level is quite challenging, as (1) the long and irregular length of real-world order flow samples block the feasibility of applying diffusion models directly on the raw order-level data, (2) linking the “macro” control target with the every “micro” order separately may not make sense due to the low signal-to-noise ratio in order flow.

In this paper, we present the problem formulation of controllable financial market generation, and propose a Diffusion Guided meta Agent (DiGA) model to address the problem. Specifically, we utilize a conditional diffusion model to capture dynamics of market state represented by time-evolving distribution parameters about mid-price return rate and order arrivial rate, and define a meta agent with financial economic priors to samples orders from the distributions defined by the aforementioned parameters. With DiGA, we are able to control the generation to simulate order flow given target scenario with high fidelity. Our contributions can be summarized as follows:

• We formulate controllable financial market generation problem as a novel challenge for machine learning application in finance.   
• To the best of our knowledge, DiGA stands as the pioneering model that integrates advanced diffusion-based generative models into the domain of financial market modeling.   
• Our experiments, using extensive real stock market data, show DiGA excelling in controlling financial market generation and achieving the highest fidelity to stylized facts. Additionally, its effectiveness is validated with high-frequency trading task, indicating significant potential for assisting classical finance downstream.

# 2 Preliminaries and related work

In this section, we briefly describe the preliminaries of limit order book, and review the related research work regarding financial market simulation and diffusion models.

# 2.1 Limit order book

The majority of financial markets around the world operate on a double-auction system, where orders are the minimal units of events. An order in the market consists of four basic elements: timestamp $t$ , price $p$ , quantity $q$ , and order type $o$ . There are a variety of order types in real markets, such as limit orders, market orders, cancel orders, conditional order, etc. In the literature, it is sufficient to represent trade decisions with limit orders and cancel orders [18].

# The output of market simulation model is a se

![](images/65d11ca153fd7c8dda42938932c5318746f5dc35fa79a611ea4f6df2a3061193.jpg)  
Figure 1: Limit order book and order flow

ries of orders $O = \{ ( t _ { 1 } , p _ { 1 } , q _ { 1 } , o _ { 1 } ) , ( t _ { 2 } , p _ { 2 } , q _ { 2 } , o _ { 2 } ) , . . . \}$ , also known as the order flow, which constitutes the order book and price series. An order book is the collection of outstanding orders that have not been executed. Limit orders can be further categorized into buy limit orders and sell limit orders, with their prices known as bids and asks, respectively. The order book specifically consisting of limit orders are called the limit order book (LOB). The price at each timestamp is commonly determined by the price of the most recently executed order. Prices sampled at a certain frequency form the price series. An illustrative example of limit order book can be found in Figure 1.

# 2.2 Financial market simulation

Early works on market simulation has followed a multi-agent approach [8; 19; 20], using rule-based agents under simplified trading protocols to replicate “stylized facts” [21] such as volatility clustering. Chiarella and Iori [22]; Chiarella et al. [18] extended the simulation analysis to order-driven markets, which more closely resemble the settings of most of current active stock markets. Subsequent works further customized agents’ behaviors based on this protocol to better support research on decisionmaking [12; 13; 23]. With the recent success of machine learning, researcher has also employed neural networks as world agent to directly predict orders given history [9; 15; 16; 24]. In contrast, our model employs conditional diffusion model for controlling agent based models to generate orders.

# 2.3 Diffusion models

The diffusion probabilistic models, also known as the diffusion models, fit sequential small perturbations from a diffusion process to convert between known and target distributions [4; 5; 6]. Diffusion models have been built to generate data in different modalities, such as image [2; 3; 25; 26], audios [27; 28], videos [29; 30] and general time series [31; 32; 33]. Different from existing works, we are the first to apply diffusion model for generating financial market. Diffusion Model

# 3 Method

In this section, we first present the problem formulation of controllable financial market generation.   
Then, we present the detailed architecture design of our Diffusion Guided meta Agent (DiGA) model.

# 3.1 Problem formulation of controllable financial market generation

Different from common market simulation that simulates unconditionally, controllable market generation aims to simulate order flows that correspond to certain scenario with generative models. Possible scenarios can be characterized by indicators that represent the aggregative statistics of an order flow, such as daily return, daily amplitude and intraday volatility. Formally, let $\mathcal { F }$ denote the function for calculating any of the indicators, given real order flow sample $O \sim q ( O )$ , the value of indicator $a$ can be calculated with $a = { \mathcal { F } } ( O )$ .

In controllable financial market generation, a market generator $\mathcal { M }$ provides a conditional sampler $p _ { \mathcal { M } } ( \pmb { O } | \cdot )$ . Given control target as a required market scenario specified by indicator of value $a$ , we denote the conditional sampling output as $\tilde { O } \sim p _ { \mathcal { M } } ( O | a )$ . The controllability objective is to minimize the distance between the conditional input $a$ and the indicator representing the same aggregative statistics of generated order flow $\tilde { a } = \mathcal { F } ( \tilde { O } )$ :

$$
\displaystyle \operatorname* { m i n } _ { \mathcal { M } } \mathbb { E } _ { a , \tilde { a } } \left[ \| \tilde { a } - a \| ^ { 2 } \right] = \operatorname* { m i n } \mathbb { E } _ { a , \tilde { O } \sim p _ { \mathcal { M } } ( O | a ) } \left[ \| \mathcal { F } ( \tilde { O } ) - a \| ^ { 2 } \right] .
$$

In addition, it is also important for the model to generate order flow with high fidelity in the context of controllable market. The fidelity objective is aligned with general market simulation problems, defined as minimizing the distribution divergence measure $\mathcal { D }$ on “stylized facts” between the real and generated order flow. Here, the “stylized facts” can also be expressed as a set of aggregative statistics to be calculated with their corresponding formula. Let ${ \mathcal { F } } ^ { \prime }$ denotes an arbitrary function of stylized fact and $p ( \cdot )$ denotes probability density, the fidelity objective writes:

$$
\operatorname* { m i n } _ { \mathcal { M } } \mathbb { E } _ { O \sim q ( O ) , \tilde { O } \sim p _ { \mathcal { M } } ( O | \cdot ) } \mathcal { D } \left( p ( \mathcal { F } ^ { \prime } ( \tilde { O } ) ) \parallel p ( \mathcal { F } ^ { \prime } ( O ) ) \right) .
$$

# 3.2 Diffusion guided meta agent model

Order flow data is highly intricate and noisy, with a huge amount typically counting for tens of thousand per stock daily which is computational challenging. Thus it is non-trivial modeling the distribution of order flow that covers diverse scenarios. Moreover, linking a “macro” control target with every “micro” order separately may not make sense due to the low signal-to-noise ratio in order flow. Instead of fitting the distribution of raw order flow directly with a diffusion model, we design a two-stage model that exhibits greater efficiency.

![](images/58be785dfce87c3ed0a94b7a80793e866833f3f593ddf323bfd4636d43835007.jpg)  
(a) Meta controller and order generator in DiGA. (b) Data flow of DiGA.   
Figure 2: Overview of DiGA model. Raw order flow are proccessed into market states for the meta controller to fit. The meta agent is guided by the meta controller to generate simulated order flow.

Our model comprises of two modules. The first module is a meta controller $\mathcal { C }$ that learns the intraday dynamics of market states $_ { \textbf { \em x } }$ regarding a scenario $c$ , as the distribution $q ( { \pmb x } | c )$ , using a conditional diffusion model. The second module is an order generator $\mathcal { G }$ that contains a simulated exchange and a meta agent. The meta agent is incorporated with financial economics prior, as well as guided by the meta controller, to generate order through a stochastic process. Figure 2 provides the overview of DiGA model and the model can be expressed as $\mathcal { M } = \{ \mathcal { C } , \mathcal { G } \}$ .

# 3.2.1 Meta controller

For market states $_ { \textbf { \em x } }$ to represent intraday dynamics regarding given scenario $c$ , they should be evolving with time and be of close causal connection with $c$ . We choose extracting the minutely mid-price return rate $\mathbfit { \Delta } \mathbf { r }$ as well as order arrival rate $\boldsymbol { \lambda }$ form the real order flow data as market states $\bar { \boldsymbol { x } } = \{ \bar { r } , \lambda \}$ .

While the number of trading minutes are fixed across trading days, it is natural to treat the stacked market states of each trading day as a sample. Consequently, the training objective for fitting the distribution of the market states can be expressed as:

$$
\begin{array} { r } { \operatorname* { m i n } \mathbb { E } _ { \pmb { x } } \mathcal { D } \left( p _ { \mathcal { G } } ( \pmb { x } ) \parallel \boldsymbol { q } ( \pmb { x } ) \right) . } \end{array}
$$

With the training set of market states √ √ $\{ \pmb { x } \sim q ( \pmb { x } ) \}$ , we first generate the diffusion latent variable series with $\pmb { x } _ { n } \overset { - } { = } \sqrt { \bar { \alpha } _ { n } } \pmb { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { n } } \epsilon$ , where $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ , $n$ is the maximum diffusion step and $\hat { \alpha } _ { n }$ is a transformation of diffusion variance schedule $\{ \beta _ { n } \in ( 0 , 1 ) \} _ { n = 1 , \dots , N }$ . Then we adopt the $\epsilon$ -parameterized denoising diffusion probabilistic model (DDPM) [5] to predict noise from the noisy series ${ \pmb x } _ { n }$ for a given denoising time step $n$ , $\hat { \pmb { \epsilon } } = \pmb { \epsilon } _ { \theta } ( { \pmb x } _ { n } , n )$ , where $\theta$ denotes model parameters. The corresponding training objective can be written as:

$$
L _ { M } : = \mathbb { E } _ { \mathcal { H } _ { e } ( O ) , \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) , n } [ \| \epsilon - \epsilon _ { \theta } ( x _ { n } , n ) \| ^ { 2 } ] .
$$

With sampling of $\tilde { \mathbf { x } } _ { 0 }$ done iteratively using:

$$
\tilde { \pmb { x } } _ { n - 1 } = \frac { 1 } { \sqrt { \alpha _ { n } } } ( \tilde { \pmb { x } } _ { n } - \frac { 1 - \alpha _ { n } } { \sqrt { 1 - \bar { \alpha } _ { n } } } \pmb { \epsilon } _ { \theta } ( \tilde { \pmb { x } } _ { n } , n ) ) + \sigma _ { n } \mathbf { z } ,
$$

from ${ \pmb x } _ { N } \sim \mathcal { N } ( { \bf 0 , I } )$ at time step $N$ to $\scriptstyle { \pmb x } _ { 0 }$ at time step 0, where $\boldsymbol { z } \sim \mathcal { N } ( 0 , \mathbf { I } )$ and $\sigma _ { n } = \sqrt { \beta _ { n } }$ . With $\tilde { \mathbf { x } } _ { 0 }$ , the meta controller is able to guide the meta agent in the order generator to generate orders.

To further exert order generation with control targets, we apply conditioning on the output of diffusion model and thus control the generated order flow. Following common practice [2], we implement conditional $\epsilon$ -parameterized noise predictor $\boldsymbol { \epsilon } _ { \theta } ( \boldsymbol { x } _ { n } , n , c )$ for sampling with control target $c$ .

Specifically, we adopt indicators commonly used to describe the state of financial markets as control targets. These indicators include daily return, amplitude, and volatility. All of these indicators can be numerically derived from return rates of the price series. Controlling these indicators allows the generated order flow to satisfy the need for analyzing markets under a wide range of specific scenarios that can be characterized by these indicators.

To align the control targets with the model, we introduce a target-specific feature extractor $\phi$ that projects target indicators into latent representations $\phi ( c )$ . We propose two types of condition encoders for exerting control. The first one is discrete control encoder, in which conditioning are mapped into a predefined number of discrete bins, with their indexes treated as class labels, and an embedding matrix is learned to extract the latent representation. The second type is continuous control encoder, where a fully connected network is employed to directly map conditions into latent representations. We train $\phi$ concurrently with the conditional sampler and the training objective can then be written as follows:

$$
L _ { C } : = \mathbb { E } _ { \mathcal { H } _ { e } ( O ) , c , \epsilon \sim \mathcal { N } ( \mathbf { 0 } , I ) , n } [ \| \epsilon - \epsilon _ { \theta } ( x _ { n } , n , \phi ( c ) ) \| ^ { 2 } ] .
$$

For both methods, we incorporate classifier-free guidance [17] to perform control. During the training phase, we jointly train unconditional and conditional samplers by randomly dropping out conditions. During sampling, a linear combination of conditional and unconditional score estimates is performed:

$$
\tilde { \epsilon } _ { \theta , \phi } ( { \bf x } _ { n } , n , c ) = ( 1 - s ) \epsilon _ { \theta } ( { \bf x } _ { n } , n ) + s \epsilon _ { \theta } ( { \bf x } _ { n } , n , \phi ( c ) ) ,
$$

where $s$ is the conditioning scale that controls the strength of guidance. Sampling is done iteratively similar to Equation 5, where:

$$
{ \pmb x } _ { n - 1 } = \frac { 1 } { \sqrt { \alpha _ { n } } } ( { \pmb x } _ { n } - \frac { 1 - \alpha _ { n } } { \sqrt { 1 - \bar { \alpha } _ { n } } } { \tilde { \epsilon } } _ { \theta , \phi } ( { \pmb x } _ { n } , n , { \pmb x } ) ) + \sigma _ { n } { \bf z } .
$$

In practice, we adopt DDIM sampling [34] for better efficiency. As for the model backbone, we adopt a U-Net that is primarily built from 1D convolution layers, while sharing parameters across diffusion time steps. We refer the reader to Appendix B for the details of DDPM model.

# 3.2.2 Order generator

The order generator consists of a simulated exchange, and a meta agent. The simulated exchange replicates the double-action market protocol on which the majority of financial markets are operating. It facilitates the agent-market interaction and providing the basis of producing realistic financial market generations. The meta agent is the representative of all traders in the generated market, serving as a world agent. Different from existing works that has used learned agent as world agent, our meta agent is grounded with financial economics prior and is guided by the meta controller.

Specifically, the meta agent generated orders following a stochastic process, whose key parameters are determined by the meta controller. For every trading minute $t$ in the trading day, the meta agent “wake up” by a time interval $\delta _ { i }$ sampled from a exponential distribution $f ( \delta _ { i } ; \bar { \lambda _ { t } } ) = \dot { \lambda _ { t } } e ^ { - \lambda _ { t } \delta _ { i } }$ , where $i$ is the total number of executed wake-ups for this trading day and $\lambda _ { t }$ is given by the meta controller.

Upon each wake-up, the meta agent generate an actor agent $A _ { i }$ within the family of heterogeneous agents [18], who makes decisions following the optimization of CARA utility function given the market observations. The order generation procedures are in Algorithm 2 and described as follows:

• The actor agent is initialized with random holding positions $S$ and the corresponding amount of cash $C$ , as well as the random weights $g _ { f } , g _ { c } , g _ { n }$ of its three heterogeneous components, namely fundamental, chartist and noise. These random variables are samples from independent exponential distribution, with configuration that the expectation of fundamental weight is highest among the three components. • The actor agent then produce an estimation of objective future return $\hat { r }$ as the average of fundamental, chartist and noise with weights above. Fundamental is the $r _ { t }$ determined by the meta controller. Chartist is the historical average return $\bar { r }$ obtained from simulated exchange. Noise is a small Gaussian perturbation rσ. Consequently, rˆ = gf rt+gcr¯+gnrσg +g +g . • With the estimate return, the actor estimate future price $\hat { p } _ { t } = p _ { t } \exp { ( \hat { r } ) }$ , the actor agent is able to obtain its specific demand function $\begin{array} { r } { u ( p ) = \frac { \ln { \left( \hat { p } _ { t } / p \right) } } { a V p } } \end{array}$ ln (ˆpt/p) , where a is risk averse coefficient and $V$ is history price volatility, by deriving from CARA utility on future wealth [18], as well as the lowest order price $p _ { l }$ at which the demand function is satisfied $p _ { l } ( u ( p _ { l } ) - S ) = C$ . • Finally, the actor agent samples its order price uniformly between its lowest price and estimated price $p _ { i } \sim \mathcal { U } ( p _ { l } , \hat { p } )$ , as well as obtain order volume $q _ { i } = u ( p _ { i } ) - S$ and order type $o _ { i } = \mathrm { s i g n } ( q _ { i } )$ where $o _ { i } = 1$ indicates buy order and $o _ { i } = 0$ indicates sell order.

Table 1: MSE between the targeted indicator and the generated aggregative statistics of generated order flow. Best results are highlighted with bold face.   

<table><tr><td></td><td></td><td colspan="5">A-Main</td><td colspan="5">ChiNext</td></tr><tr><td>Target</td><td>Method</td><td>Lower</td><td>Low</td><td>Medium</td><td>High</td><td>Higher</td><td>Lower</td><td>Low</td><td>Medium</td><td>High</td><td>Higher</td></tr><tr><td rowspan="3">Return</td><td>No Control</td><td>1.443</td><td>0.583</td><td>0.529</td><td>0.813</td><td>2.337</td><td>0.979</td><td>0.684</td><td>0.992</td><td>1.718</td><td>3.923</td></tr><tr><td>Discrete</td><td>1.055</td><td>0.494</td><td>0.228</td><td>0.429</td><td>0.664</td><td>1.285</td><td>0.807</td><td>0.243</td><td>0.413</td><td>0.869</td></tr><tr><td>Continuous</td><td>0.206</td><td>0.178</td><td>0.161</td><td>0.184</td><td>0.212</td><td>0.584</td><td>0.539</td><td>0.342</td><td>0.449</td><td>0.840</td></tr><tr><td rowspan="3">Amplitude</td><td>No Control</td><td>0.521</td><td>0.268</td><td>0.268</td><td>0.699</td><td>3.298</td><td>1.130</td><td>0.638</td><td>0.427</td><td>0.608</td><td>2.763</td></tr><tr><td>Discrete</td><td>0.049</td><td>0.088</td><td>0.309</td><td>0.502</td><td>0.930</td><td>0.057</td><td>0.134</td><td>0.346</td><td>0.523</td><td>0.963</td></tr><tr><td>Continuous</td><td>0.054</td><td>0.076</td><td>0.149</td><td>0.247</td><td>0.348</td><td>0.110</td><td>0.116</td><td>0.255</td><td>0.437</td><td>0.973</td></tr><tr><td rowspan="3">Volatility</td><td>No Control</td><td>0.021</td><td>0.115</td><td>0.431</td><td>1.209</td><td>4.288</td><td>0.029</td><td>0.246</td><td>0.713</td><td>1.737</td><td>5.221</td></tr><tr><td>Discrete</td><td>0.016</td><td>0.123</td><td>0.383</td><td>0.890</td><td>2.393</td><td>0.029</td><td>0.188</td><td>0.481</td><td>0.948</td><td>2.257</td></tr><tr><td>Continuous</td><td>0.011</td><td>0.104</td><td>0.318</td><td>0.774</td><td>2.389</td><td>0.028</td><td>0.178</td><td>0.473</td><td>1.016</td><td>2.631</td></tr></table>

The generated order is then recorded as $o _ { i } = ( t _ { i } , p _ { i } , q _ { i } , o _ { i } ) \sim p ( o | r _ { t } , \lambda _ { t } , \gamma )$ , where $\begin{array} { r } { t _ { i } = \sum _ { j = 1 } ^ { i } \delta _ { i } } \end{array}$ and $\gamma$ is the set of fixed parameters for meta agent.

Generation ends at $t _ { m a x }$ when the next $t _ { i }$ will be greater than the total time lengths of trading hours of a trading day. The generated order flow is recorded as:

$$
\tilde { \cal O } = \{ o _ { 1 } , \ldots , o _ { m a x } \} \sim p ( { \cal O } | \tilde { \bf x } , \gamma ) ,
$$

where $\tilde { \mathbf { x } }$ is generated by the meta controller.

# 4 Experiments

In this section, we present settings and results for experiments on real-world datasets to evaluate controllability and fidelity for DiGA. Additionaly, we show results on case study regarding the helpfulness of DiGA in a high-frequency trading reinforcement learning task.

# 4.1 Dataset and model configurations

We conduct the experiments on two tick-by-tick order datasets over China A-share market: A-Main and ChiNext. After preprocessing, 316,287 date-stock pair samples in A-Main and 122,574 in ChiNext are used. For each dataset, we take 5,000 samples each for validation and test, with all of the rest samples for training. For more dataset preprocessing details, please refer to Appendix A

We train the diffusion model on each dataset for 10 epochs, with 200 diffusion steps with AdamW optimizer. There are 256 samples per mini-batch and the learning rate is $1 e ^ { - 5 }$ . For both the discrete and continuous control models, we use a probability of 0.5 to randomly drop conditions during training. We set a pseudo initial stock price at 10 for every generation run. For more parameter settings details, please refer to Appendix B.3.

# 4.2 Evaluation on controlling financial market generation

In this experiment, we evaluate DiGA’s capability for controlling market generation. For enabling DiGA to simulate with control targets as input, we train DiGA with four indicators respectively: return, amplitude and volatility, all of which are indicators that represent a vast range of market scenarios. For each indicator, we first retrieve its empirical distribution from real-world order flow dataset. Then we partition the values into five bins according to uniform percentiles, each representing lower, low, mid, high and higher cases for the scenario characterized by corresponding indicator. We train DiGA with discrete control encoder using these case types as class labels, and train DiGA with continuous control encoder using the exact value of indicators after normalization. On testing, the class labels are directly used as control target for DiGA with discrete control encoder, and we extract the median value of real samples from each bin to represent corresponding scenario as the control target for DiGA with continuous control encoder.

Table 1 shows the mean squared error (MSE) between the indicators computed from simulated order flow and the control target, i.e. the median value of each of the five bins, for DiGA with both discrete and continuous control encoder. The results are averaged from 3 runs with different random seeds. The row “No Control” presents the results from a variant of DiGA by removing the condition mechanism in meta controller, whose generations are independent to the control target. From Table 1, DiGA successfully achieves the effect of control by keeping a relatively small error between realized indicator value and control target, while the method without control tend to be either random or repeating some particular scenarios.

![](images/43ab2cefba8e838903c359947f34e3b2eeb5e5e57723c6c9da34ea446692fd1b.jpg)  
Figure 3: Aggregated price curves demos (first row) and the distribution of targeted indicators (second row). For the first row, each curve represents the order flow of one day. For the second row, each colored density represents the distribution of targeted indicator computed from generation results.

Figure 3 illustrates the mid-price series of controlled generation samples for each scenario, as well as the distribution of indicators in 200 independent generations runs for each scenario. Results show that the sampled price curves successfully represent the desired scenario, and the distribution of the four indicators are shifted correctly following the control target. This demonstrates capability of DiGA to control financial market generation.

# 4.3 Evaluation on generation fidelity

We evaluate generation fidelity of DiGA and compare the results with market simulation method baselines with both rule-based agents and learning-based agents:

• RFD [12] is a multi-agent based market simulation configuration with random fundamental and diverse agent types, consisting of 1 market maker, 25 momentum agents, 100 value agents and 5,000 noise agents.   
• RMSC [14] is the reference market simulation configuration introduced with the ABIDESgym, which contains all RFD agents and 1 extra percentage-of-volume (POV) agent who provide extra liquidity to the simulated market.   
• LOBGAN [16] trains a Conditional Generative Adversarial Network with to generate next order conditioned on market history.

We focus on the distribution discrepancy of several “stylized facts", which are statistics regarding asset returns and order book, between the real and simulated markets. The selected statistics, as listed below, are among the most representative features in the domain of the financial market micro-structure:

• Minutely Log Returns (MinR) are the log difference between two consecutive prices sampled by minutes.   
• Return Auto-correlation (RetAC) is the value of linear auto-correlation function calculated between the return array and its lagged array. Empirical studies on real market data have discovered absence of auto-correlation while the lag is not big enough.   
• Volatility Clustering (VolC) is the value of linear auto-correlation function of the squared returns and their lag. It shows the empirical fact that volatile events tend to appear in cluster with time.

![](images/4a9ada16abaee58c8cc110042c36d624017a507cec68f86795443831734e94c2.jpg)  
Figure 4: Comparison of stylized facts distribution across baselines. The $\mathbf { X }$ -axis is the stylized fact values and the y-axis is the density.

Table 2: K-L divergence of stylized facts distribution between real and simulated order flow.   

<table><tr><td></td><td colspan="4">A-Main</td><td colspan="4">ChiNext</td></tr><tr><td>Model</td><td>MinR</td><td>RetAC</td><td>VolC</td><td>OIR</td><td>MinR</td><td>RetAC</td><td>VolC</td><td>OIR</td></tr><tr><td>RFD</td><td>1.198</td><td>5.010</td><td>0.839</td><td>0.015</td><td>0.272</td><td>2.987</td><td>0.691</td><td>0.022</td></tr><tr><td>RMSC</td><td>2.640</td><td>10.170</td><td>1.237</td><td>0.563</td><td>1.371</td><td>7.461</td><td>0.668</td><td>0.588</td></tr><tr><td>LOBGAN</td><td>0.151</td><td>1.903</td><td>1.101</td><td>0.309</td><td>0.135</td><td>1.711</td><td>0.507</td><td>0.282</td></tr><tr><td>DiGA</td><td>0.084</td><td>2.781</td><td>0.273</td><td>0.009</td><td>0.079</td><td>1.997</td><td>0.218</td><td>0.009</td></tr></table>

• Order Imbalance Ratio (OIR) is proportion difference between best bid volume and best ask volume. It represents the tendency for trading of market participants.

We demonstrate results sampled with the unconditional version of our method DiGA in the experiment. Figure 4 shows the results for fidelity comparison, where the distribution of statistics on real market data is displayed as Real in golden solid line. Overall, DiGA stays the closest to Real compared with other methods. We capture the differences between real and simulated market quantitatively using Kullback-Leibler divergence (K-L). Table 2 provides thequantitative metrics, showing that DiGA can reach the best K-L divergence among the most of the statistics. Although LOBGAN obtains the lowest RetAC divergence because of its auto-regressive nature when generating orders, DiGA still outperform LOBGAN on other metrics by a large margin. Overall, the results demonstrate DiGA’s superiority on generating realistic market dynamic, achieving the state-of-the-art performance regarding fidelity in market simulation.

# 4.4 Evaluation for downstream task: high-frequency trading with reinforcement learning

We evaluate the helpfulness of DiGA as the environment for training reinforcement learning (RL) algorithms to perform high-frequency trading.

# 4.4.1 Settings

We train a trading agent with simulated market as training environment using the A2C algorithm, to optimize for a high-frequency trading task. Agents take an discrete action every 10 seconds. Possible actions include either buy or sell at any one of the best 5 level with integer volume ranging from 1 to 10 units, as well as an option not to submit any order. The observation space is configured as the price changes of last 20 seconds, 10-level bid-ask price-volume pairs and the account status including the current amount of capital, position and cash of agents. Each agent is trained on simulated stock market produced by either history replay, RFU, DiGA and a variant of DiGA that removes the conditioning mechanism of meta controller (DiGA-c). Each train lasts for 200 episodes, where each episode represents a full trading day with four trading hours. Afterwards, we test the agents with an environment replaying out-of-sample real market data for 50 episodes. Tests are repeated three times using three non-overlap periods of out-of-sample real market data. All trains share the same set of RL hyper-parameters for fairness.

We evaluate the performance of RL trading task with four metrics. Daily return (Ret) is the mean return rate across episodes, for assessing profitability. Daily volatility (Vol) is the standard deviation of daily return for assessing risk management, daily Sharpe ratio (SR) is the division of daily return by volatility for assessing return-risk trade-off ability. Maximum drawdown (MDD) is the largest intraday profit drop happens during testing which assesses the performance under extreme cases.

Table 3: Full out-of-sample test results (in percentage). Best results are highlighted with bold face.   

<table><tr><td>Environment</td><td>Period</td><td>Ret(%)(↑)</td><td>Vol(↓)</td><td>SR(↑)</td><td>MDD(%)(↑)</td></tr><tr><td rowspan="4">Replay</td><td>1</td><td>0.031</td><td>0.502</td><td>0.012</td><td>-1.198</td></tr><tr><td></td><td></td><td>0.323</td><td></td><td></td></tr><tr><td>２3</td><td>0.046</td><td></td><td>0.023</td><td>-0.936</td></tr><tr><td>Average</td><td>0.009±0.043</td><td>0.413±0.090</td><td>0.014±0.008</td><td>−1.133±0.173</td></tr><tr><td rowspan="4">RFD</td><td>1</td><td>-0.004</td><td>0.265</td><td>-0.012</td><td>-1.004</td></tr><tr><td>2</td><td>0.009</td><td>0.084</td><td>0.044</td><td>-0.426</td></tr><tr><td>3</td><td>-0.005</td><td>0.128</td><td>0.001</td><td>-0.980</td></tr><tr><td>Average</td><td>0.000±0.008</td><td>0.159±0.094</td><td>0.011±0.029</td><td>-0.803±0.327</td></tr><tr><td rowspan="4">DiGA-c</td><td>1</td><td>0.037</td><td>0.320</td><td>0.025</td><td>-1.250</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>２3</td><td>0.0179</td><td>0.074</td><td>-0.068</td><td>-0.472</td></tr><tr><td>Average</td><td>0.015±0.023</td><td>0.147±0.151</td><td>0.006±0.066</td><td>-0.715±0.464</td></tr><tr><td rowspan="4">DiGA</td><td>1</td><td>0.033</td><td></td><td></td><td>-1.488</td></tr><tr><td>2</td><td>0.046</td><td>0.550 0.330</td><td>0.023 0.084</td><td>-1.188</td></tr><tr><td>3</td><td>0.009</td><td>0.352</td><td>0.040</td><td>-1.264</td></tr><tr><td>Average</td><td>0.029±0.019</td><td>0.411±121</td><td>0.049±0.031</td><td>−1.313±0.156</td></tr></table>

# 4.4.2 Results

Table 3 displays the average numerical results for the high-frequency trading task. The trading agent trained with DiGA generated environment has demonstrated the highest daily return, daily Sharpe ratio and maximum drawdown against all baselines, and the near best performance on daily volatility. These results shows that the DiGA generated environment has help the trading agent learn a better policy. Moreover, the inferior results obtained with DiGA-c indicates the importance of performing control on the training environment for a more sufficient exploration of trading agent.

We further conduct a case study on the behavior of trading agents trained on different simulated environments to investigate possible reason for the outperforming results of agents trained with DiGA. The case shows a trading day representing an extreme volatile scenario with sharp price surge and drop of up to five percents happens in minutes. The return rate of different agents in the same trading day is shown in Figure 5. It is shown that the agent trained with market replay environment learns a passive investment strategy that follows the changes of asset price very closely. The agent trained with RFU environment tries to catch the trend, but suffers from a sudden drop that reverts the trend. The agent trained with DiGA-c fails to take action in this extreme market. In contrast, the agent trained with DiGA has avoided the extreme event and

![](images/9f70cb810e849c2ecd9ea06a5c78549ba5c53f65d79fda2945d560d7894bcb93.jpg)  
Figure 5: The return rate of different agents in the same trading day. The $\mathbf { X }$ -axis denotes time frame in minutes. The y-axis denotes the cumulative return rate obtained by the trading agent.

trades to take profits afterwards. The result showcases the potential that the diversed generated scenario from DiGA has provided experience for agent to learn making better action in these cases.

# 5 Conclusion and future work

In this paper, we present the problem formulation of controllable financial market generation, and propose a Diffusion Guided meta Agent (DiGA) model to address the problem. Specifically, we utilize a diffusion model to capture dynamics of market state represented by time-evolving distribution parameters about mid-price return rate and order arrival rate, and define a meta agent with financial economic priors to generate orders from the corresponding distributions. Extensive experimental results demonstrate that our method exhibits outstanding controllability and fidelity in simulation. While we focus on generating order flow of one individual stock for each time in this work, one future work is to consider the correlation among multiple assets for generating more realistic markets.

# Disclaimer

The DiGA model is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. The DiGA model is aimed to facilitate research and development process in the financial industry and not ready-to-use for any financial investment or advice. Users shall independently assess and test the risks of the DiGA model in a specific use scenario, ensure the responsible use of AI technology, including but not limited to developing and integrating risk mitigation measures, and comply with all applicable laws and regulations in all applicable jurisdictions. The DiGA model does not provide financial opinions or reflect the opinions of Microsoft, nor is it designed to replace the role of qualified financial professionals in formulating, assessing, and approving finance products. The inputs and outputs of the DiGA model belong to the users and users shall assume all liability under any theory of liability, whether in contract, torts, regulatory, negligence, products liability, or otherwise, associated with use of the DiGA model and any inputs and outputs thereof.

# References

[1] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, pages 1877–1901, 2020.   
[2] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.   
[3] Marco Aversa, Gabriel Nobis, Miriam Hägele, Kai Standvoss, Mihaela Chirica, Roderick Murray-Smith, Ahmed M. Alaa, Lukas Ruff, Daniela Ivanova, Wojciech Samek, Frederick Klauschen, Bruno Sanguinetti, and Luis Oala. Diffinfinite: Large mask-image synthesis via parallel random patch diffusion in histopathology. In Advances in Neural Information Processing Systems, pages 78126–78141, 2023.   
[4] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the International Conference on Machine Learning, 2015.   
[5] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Proceedings of Advances in Neural Information Processing Systems, 2020.   
[6] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In Proceedings of the International Conference on Learning Representations, 2021.   
[7] Tucker Hybinette Balch, Mahmoud Mahfouz, Joshua Lockhart, Maria Hybinette, and David Byrd. How to evaluate trading strategies: Single agent market replay or multiple agent interactive simulation? CoRR, abs/1906.12010, 2019.   
[8] Marco Raberto, Silvano Cincotti, Sergio M. Focardi, and Michele Marchesi. Agent-based simulation of a financial market. Physica A: Statistical Mechanics and its Applications, 299(1): 319–327, 2001.

[9] Andrea Coletta, Matteo Prata, Michele Conti, Emanuele Mercanti, Novella Bartolini, Aymeric Moulin, Svitlana Vyetrenko, and Tucker Balch. Towards realistic market simulations: a generative adversarial networks approach. In Proceedings of the ACM International Conference on AI in Finance, 2021.

[10] Takanobu Mizuta. An agent-based model for designing a financial market that works well. In Proceedings of the IEEE Symposium Series on Computational Intelligence, 2020.   
[11] Jian Guo, Saizhuo Wang, Lionel M. Ni, and Heung-Yeung Shum. Quant 4.0: Engineering quantitative investment with automated, explainable and knowledge-driven artificial intelligence. CoRR, abs/2301.04020, 2023.   
[12] Svitlana Vyetrenko, David Byrd, Nick Petosa, Mahmoud Mahfouz, Danial Dervovic, Manuela Veloso, and Tucker Balch. Get real: realism metrics for robust limit order book market simulations. In Proceedings of the ACM International Conference on AI in Finance, 2020.   
[13] David Byrd, Maria Hybinette, and Tucker Hybinette Balch. Abides: Towards high-fidelity multiagent market simulation. In Proceedings of the 2020 ACM SIGSIM Conference on Principles of Advanced Discrete Simulation, 2020.   
[14] Selim Amrouni, Aymeric Moulin, Jared Vann, Svitlana Vyetrenko, Tucker Balch, and Manuela Veloso. Abides-gym: gym environments for multi-agent discrete event simulation and application to financial markets. In Proceedings of ACM International Conference on AI in Finance, 2021.   
[15] Junyi Li, Xintong Wang, Yaoyang Lin, Arunesh Sinha, and Michael P. Wellman. Generating realistic stock market order streams. In Proceedings of the AAAI Conference on Artificial Intelligence, AAAI, 2020.   
[16] Andrea Coletta, Joseph Jerome, Rahul Savani, and Svitlana Vyetrenko. Conditional generators for limit order book environments: Explainability, challenges, and robustness. CoRR, abs/2306.12806, 2023.   
[17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. CoRR, abs/2207.12598, 2022.   
[18] Carl Chiarella, Giulia Iori, and Josep Perello. The Impact of Heterogeneous Trading Rules on the Limit Order Book and Order Flows. Journal of Economic Dynamics and Control, 33(3): 525–537, 2009.   
[19] R.G. Palmer, W. Brian Arthur, John H. Holland, Blake LeBaron, and Paul Tayler. Artificial economic life: a simple model of a stockmarket. Physica D: Nonlinear Phenomena, 75(1): 264–274, 1994.   
[20] Thomas Lux and Michele Marchesi. Scaling and criticality in a stochastic multi-agent model of a financial market. Nature, 397(6719):498–500, 1999.   
[21] Rama Cont. Empirical properties of asset returns: stylized facts and statistical issues. Quantitative finance, 1(2):223, 2001.   
[22] Carl Chiarella and Giulia Iori. A simulation analysis of the microstructure of double auction markets. Quantitative finance, 2(5):346, 2002.   
[23] Xintong Wang and Michael P. Wellman. Spoofing the limit order book: An agent-based model. In Proceedings of the Conference on Autonomous Agents and MultiAgent Systems, 2017.   
[24] Andrea Coletta, Aymeric Moulin, Svitlana Vyetrenko, and Tucker Balch. Learning to simulate realistic limit order book markets from data as a world agent. In Proceedings of the ACM International Conference on AI in Finance, 2022.   
[25] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat gans on image synthesis. In Proceedings of Advances in Neural Information Processing Systems, 2021.   
[26] Jiaming Song, Qinsheng Zhang, Hongxu Yin, Morteza Mardani, Ming-Yu Liu, Jan Kautz, Yongxin Chen, and Arash Vahdat. Loss-guided diffusion models for plug-and-play controllable generation. In Proceedings of the International Conference on Machine Learning, 2023.   
[27] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In Proceedings of the International Conference on Learning Representations, 2021.   
[28] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. AudioLDM: Text-to-audio generation with latent diffusion models. In Proceedings of the International Conference on Machine Learning, 2023.   
[29] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.   
[30] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. International Conference on Learning Representations, 2024.   
[31] Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, and Yuyang (Bernie) Wang. Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting. In Advances in Neural Information Processing Systems, pages 28341–28364, 2023.   
[32] Andrea Coletta, Sriram Gopalakrishnan, Daniel Borrajo, and Svitlana Vyetrenko. On the constrained time-series generation problem. In Advances in Neural Information Processing Systems, pages 61048–61059, 2023.   
[33] Xinyu Yuan and Yan Qiao. Diffusion-ts: Interpretable diffusion for general time series generation. In International Conference on Learning Representations, 2024.   
[34] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Proceedings of the International Conference on Learning Representations1, 2021.

# A Detailed description of dataset and preprocessing

We conduct the experiments on two tick-by-tick order datasets over China A-share market: A-Main and ChiNext. Both datasets are collected from Wind1, retrieving Shenzhen Stock Exchange (SZSE) of year 2020. Table 4 summarizes the statistics of the dataset.

Table 4: Dataset statistics for DiGA.   

<table><tr><td></td><td>A-Main</td><td>ChiNext</td></tr><tr><td>Number of date-stock pairs</td><td>316,287</td><td>122,574</td></tr><tr><td>Number of unique stocks</td><td>1452</td><td>854</td></tr><tr><td>Number of unique trading days</td><td>237</td><td>231</td></tr></table>

For each dataset, we randomly draw 5,000 samples each for validation and test, with all of the rest samples for training.

Preprocessing includes filtering and transformation.

• Filtering We filtered out samples that contains incomplete records (i.e. trading suspended) or invalid orders (i.e. invalid order price).   
• Transformation We transform tick-by-tick data into market states represented with midprice return and order arrival rate. For mid-price return, we first extract minutely mid-price series from order flow samples of each trading day, where mid-price $p _ { t } = \dot { ( } a _ { t , 1 } + \dot { b } _ { t , 1 } ) / 2$ is defined as the average of best ask price $b _ { t , 1 }$ and best bid price ${ a } _ { t , 1 }$ at the end of the $t$ -th trading minute. Then we calculate the log differences between consecutive minutes as the mid-price return rate $r _ { t } = \log ( p _ { t } ) - \bar { \log } ( p _ { t - 1 } )$ . Finally, $\pmb { r } = [ r _ { 1 } , r _ { 2 } , . . . , r _ { T } ]$ . The effective $T$ for mid-price return is 236, excluding the call auction phase at the end of each trading day. For order arrival rate, we first calculate the number or orders within each minutes as $N _ { t }$ . With the assumption that the arrival of order follows a Poisson process, we take $N _ { t }$ as the expected number of orders and the order arrival rate can be approximate by $\lambda _ { t } = N _ { t }$ . Finally, $\lambda = [ \lambda _ { 1 } , \lambda _ { 2 } , . . . , \lambda _ { T } ]$ . During training, the inputs are transformed by $\mathbf { Z }$ -score normalization method with mean and standard deviation calculated from the training split of data. When sampling, model outputs are inverse transformed accordingly.

# B Detailed description of DDPM

# B.1 Brief review of DDPM model

A diffusion probabilistic model [4] learns to reverse the transitions of a Markov chain which is known as the diffusion process that gradually adds noise to data, ultimately destroying the signal.

Let $\mathbf { x } _ { 0 } \in \mathbb { R } ^ { d } \sim q ( \mathbf { x } _ { 0 } )$ be real data of dimension $d$ from space $\mathcal { X }$ . The diffusion process generates $\mathbf { x } _ { 1 } , . . . , \mathbf { x } _ { N }$ from the same space with the same shape as $\mathbf { x } _ { \mathrm { 0 } }$ , using a Markov chain that adds Gaussian noise over $N$ time steps: $\begin{array} { r } { q ( \mathbf { x } _ { 1 } , . . . , \mathbf { x } _ { N } | \mathbf { x } _ { 0 } ) : = \prod _ { n = 1 } ^ { \bar { N } } q ( \mathbf { x } _ { n } | \mathbf { x } _ { n - 1 } ) } \end{array}$ . The transition kernel is commonly defined as:

$$
q ( \mathbf { x } _ { n } | \mathbf { x } _ { n - 1 } ) : = \mathcal { N } ( \mathbf { x } _ { n } ; \sqrt { 1 - \beta _ { n } } \mathbf { x } _ { n } , \beta _ { n } \mathbf { I } ) ,
$$

where $\{ \beta _ { n } \in ( 0 , 1 ) \} _ { n = 1 , \dots , N }$ defines the variance schedule. Note that √ ${ \bf x } _ { n }$ at any arbitrary time step $n$ can be derived in a closed form $q ( \mathbf { x } _ { n } | \mathbf { x } _ { 0 } ) = \mathcal { N } ( \mathbf { x } _ { n } ; \sqrt { \bar { \alpha } _ { n } } \mathbf { x } _ { 0 } , ( 1 - \bar { \alpha } _ { n } ) \mathbf { I } )$ , where $\alpha _ { n } : = 1 - \beta _ { n }$ and $\textstyle { \bar { \alpha } } _ { n } : = \prod _ { s = 1 } ^ { n } \alpha _ { s }$ . For the reverse process, the diffusion model, parameterized by $\theta$ , yields:

$$
p _ { \boldsymbol \theta } ( \mathbf x _ { 0 } , \mathbf x _ { 1 } , . . . , \mathbf x _ { N } ) : = p ( \mathbf x _ { N } ) \prod _ { n = 1 } ^ { N } p _ { \boldsymbol \theta } ( \mathbf x _ { n - 1 } | \mathbf x _ { n } ) ,
$$

where $p _ { \theta } ( \mathbf { x } _ { n - 1 } | \mathbf { x } _ { n } ) \ : = \ N ( \mathbf { x } _ { n - 1 } ; \mu _ { \theta } ( \mathbf { x } _ { n } , n ) , { \boldsymbol \Sigma } _ { \theta } ( \mathbf { x } _ { n } , n ) )$ and the transitions start at $p ( \mathbf { x } _ { N } ) \ =$ $\mathcal { N } (  { \mathbf { x } } _ { n } ;  { \mathbf { 0 } } , \dot { \mathbf { I } } )$ .

While the usual optimization objective can be written as:

$$
L : = \mathbb { E } [ - \log p _ { \theta } ( \mathbf { x } _ { 0 } ) ] \leq \mathbb { E } _ { q } [ - \log \frac { p _ { \theta } ( \mathbf { x } _ { 0 } , . . . , \mathbf { x } _ { N } ) } { q ( \mathbf { x } _ { 1 } , . . . , \mathbf { x } _ { N } | \mathbf { x } _ { 0 } ) } ] ,
$$

a widely adopted parameterization writes:

$$
\pmb { \mu } _ { \theta } ( \mathbf { x } _ { n } , n ) = \frac { 1 } { \sqrt { \alpha _ { n } } } ( \mathbf { x } _ { n } - \frac { \beta _ { n } } { \sqrt { 1 - \bar { \alpha } _ { n } } } \pmb { \epsilon } _ { \theta } ( \mathbf { x } _ { n } , n ) ) ,
$$

which simplifies the objective to:

$$
L _ { s i m p l e } : = \mathbb { E } _ { \mathbf { x } _ { 0 } , \epsilon , n } [ \| \epsilon - \epsilon _ { \theta } ( \sqrt { \bar { \alpha } _ { n } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { n } } \epsilon , n ) \| ^ { 2 } ] .
$$

On sampling, $\begin{array} { r } { { \bf x } _ { n - 1 } = \frac { 1 } { \sqrt { \alpha _ { n } } } ( { \bf x } _ { n } - \frac { 1 - \alpha _ { n } } { \sqrt { 1 - { { \bar { \alpha } } _ { n } } } } \epsilon _ { \theta } ( { \bf x } _ { n } , n ) ) + \sigma _ { n } { \bf z } } \end{array}$ , where $\sigma _ { n } = \sqrt { \beta _ { n } }$ and $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$

To obtain a conditional DDPM model using classifier-free guidance [17], we modify the noise estimator to receive condition embedding $\phi ( c )$ as input, forming $\epsilon _ { \theta } ( \mathbf { x } _ { n } , n , \phi ( \pmb { c } ) )$ . During training, condition $c$ is replaced by an unconditional identifier $c _ { 0 }$ by a probability $p _ { \mathrm { u n c o n d } }$ to obtain unconditional prediction $\epsilon _ { \theta } ( \mathbf { x } _ { n } , n ) = \epsilon _ { \theta } ( \mathbf { x } _ { n } , n , \pmb { c } _ { 0 } )$ . During sampling, a conditioning scale $s$ is set to control the strength of guidance, replacing the noise prediction with

$$
\tilde { \epsilon } _ { \theta , \phi } ( \mathbf { x } _ { n } , n , c ) ) = ( 1 - s ) \epsilon _ { \theta } ( \mathbf { x } _ { n } , n ) + s \epsilon _ { \theta } ( \mathbf { x } _ { n } , n , \phi ( c ) ) .
$$

Both conditional and unconditional sampling can be accelerated by DDIM [34].

# B.2 Algorithmic procedure for training meta controller

<table><tr><td>Algorithm 1: Training meta controller of DiGA</td></tr><tr><td>Data: Order flow O processed into market states x; Result: Network parameters θ for meta controller</td></tr><tr><td></td></tr><tr><td>repeat</td></tr><tr><td>Sample xo~q(x); Calculate target indicator c = F(x);</td></tr><tr><td>Randomly set c as unconditional identifier Cu;</td></tr><tr><td>Randomly sample time step n ~ U(1, N);</td></tr><tr><td>Randomly sample noise ∈ ~ N~ (0,I);</td></tr><tr><td>Corrupt data xn = √αnx+ √1-αn∈;</td></tr><tr><td>Take gradient descent step on: Vθ,ρ|l∈ - ∈θ,(xn,n,c)l;</td></tr><tr><td></td></tr><tr><td>until reach max epochs;</td></tr></table>

# B.3 Detailed parameters of training meta controller

The denoising model $\epsilon _ { \theta }$ used in meta controller is adapted from [5]: The input of denoising model is shaped as $( B , C , T )$ , where $B$ is the batch size, $C$ is the number of channels and $T$ is the number of trading minutes in a day. In our case, $C = 2$ and $T = 2 3 6$ .

The denoising model is structured as a U-Net mainly with 3 down-sampling blocks, 1 middle blocks, 3 up-sampling blocks and 1 output block. Each up/down-sampling block contains 2 ResNet blocks, 1 self-attention layer and 1 up/down sample layer. Each ResNet block contains 2 convolution layers of size 15, with SiLU activation, with residual connection and layer normalization. Each up/down sampling layer use a factor of 2. The number of channels after each up/down sampling starts from 64 and is then scaled by a factor of 4. The middle block contains 2 ResNet block with a self-attention layer in between. The output block is another ResNet block followed by an $1 ^ { * } 1$ convolution layer. In addition, the conditioning embedding is extracted by a fully connected network with 2 layers of dimension 64. Table 5 lists the required parameters along with the above description.

The model is trained on 1 NVIDIA Tesla V100 GPU. Training 10 epochs takes approximately 2 hours for the A-Main dataset and 1 hour for the ChiNext dataset.

Table 5: Detailed parameters used for meta controller training.   

<table><tr><td>Parameter name</td><td>Parameter value</td></tr><tr><td>Denoising shape</td><td>(2,236)</td></tr><tr><td>Diffusion steps</td><td>200</td></tr><tr><td>Residual blocks</td><td>2</td></tr><tr><td>First layer hidden dimension</td><td>64</td></tr><tr><td>U-Net dimension multipliers</td><td>(1,4,16)</td></tr><tr><td>Embedding dimensions</td><td>256</td></tr><tr><td>Convolution kernel size</td><td>15</td></tr><tr><td>Convolution padding length</td><td>7</td></tr><tr><td>Unconditional probability Puncond</td><td>0.5</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Learning rate</td><td>1e-5</td></tr></table>

# C Details of generating with DiGA

C.1 Algorithmic procedure for generating financial market with DiGA.

Algorithm 2: Generating market order flow with DiGA

Input: Meta controller parameter $\theta$ , control target $c$ , conditioning scale $s$ , max trading time $T$ , initial price $p _ { 0 }$ , meta agent parameters $\lambda _ { f } , \lambda _ { c } , \lambda _ { n } , \tau _ { 0 } , \alpha _ { 0 } , \sigma _ { n }$ , simulated exchange   
Output: Order flow $^ o$   
(Phase 1: sampling market states with meta controller)   
Random sample $\bar { \pmb { x } _ { n } } \sim \mathcal { N } ( \mathbf { 0 } , I )$ ;   
for $n = N$ to 1 do Sample $\boldsymbol { z } \sim \mathcal { N } ( \mathbf { 0 } , I )$ ; $\begin{array} { r l } & { \hat { \mathbf { \epsilon } } = \overset { \cdot } { \tilde { \epsilon } } _ { \theta , \phi } ( \mathbf { x } _ { n } , n , \cdot \mathbf { \phi } ) ) = ( 1 - s ) \epsilon _ { \theta } ( \mathbf { x } _ { n } , n ) + s \epsilon _ { \theta } ( \mathbf { x } _ { n } , n , \phi ( c ) ) ) ; } \\ & { \mathbf { x } _ { n - 1 } = \frac { 1 } { \sqrt { \alpha _ { n } } } ( \mathbf { x } _ { n } - \frac { 1 - \alpha _ { n } } { \sqrt { 1 - \bar { \alpha } _ { n } } } \hat { \mathbf { \epsilon } } ) + \sigma _ { n } z ; } \end{array}$   
end   
(Phase 2: generate order flow with meta agent)   
Initilize $t = 0$ $0 , O = \varnothing , p _ { t } = p _ { 0 }$ ;   
repeat Extract $r _ { t }$ and $\lambda _ { t }$ from $_ { \pmb { x } }$ ; Initialize asset for actor agent $\mathbf { \mathcal { A } } _ { i }$ : $S _ { i } \sim$ exponential $( S _ { 0 } )$ , $C _ { i } \sim \mathrm { e x p o n e n t i a l } ( C _ { 0 } )$ , gf ∼ Laplace $\left( \lambda _ { f } \right)$ , gc ∼ Laplace $\left( \lambda _ { c } \right)$ , $g _ { n } \sim$ Laplace $\left( \lambda _ { n } \right)$ , $\begin{array} { r } { \tau _ { i } = \tau _ { 0 } \frac { 1 + g _ { f } } { 1 + g _ { c } } } \end{array}$ αi = α0 1+gc ; , 1+gf Sample time interval $\delta _ { i } \sim \mathrm { e x p o n e n t i a l } ( \lambda )$ , set $t _ { i } = t + \delta _ { i }$ ; Observe $\bar { r }$ from exchange and sample $r _ { \sigma } \sim \mathcal { N } ( 0 , \sigma _ { n } )$ ; Estimate future return rˆ = gf rt+gcr¯+gnrσ ; Estimate future price $\hat { p } _ { t } = p _ { t } \exp { ( \hat { r } ) }$ ; Solve $p _ { l }$ from $p _ { l } ( u ( p _ { l } ) - S _ { i } ) = C _ { i }$ , where $\begin{array} { r } { u ( p ) = \frac { \ln { \left( \hat { p } _ { t } / p \right) } } { \alpha _ { i } V p } } \end{array}$ ; Sample price $p _ { i } \sim \mathcal { U } ( p _ { l } , \hat { p } )$ ; Calculate volume $q _ { i } = u ( p _ { i } ) - S _ { i }$ ; Obtain order type $o _ { i } = \mathrm { s i g n } ( q _ { i } )$ ; Compose order $\pmb { o } _ { i } = ( t _ { i } , p _ { i } , q _ { i } , o _ { i } )$ ; Simulated exchange update current price $p _ { t } = { \mathrm { E x c h a n g e } } ( o _ { i } )$ ; Update $t = t _ { i } , O = O \cup \{ o _ { i } \}$ ;   
until $t > T$ ;

We discuss the input parameters in C.2 and C.3.

# C.2 Discussion on meta controller sampling parameters

When sampling with the meta controller, we apply DDIM [34] to reduce sampling steps to 20. For obtaining best generation result, the classifier guidance scale $s$ should be properly selected. In our experiments, we select the best $s$ from the range of $1 , 2 , 4 , 6 , 8$ for each target scenario. The selection is based on the discrepancy between control target and the generated scenario during training, and we take the $s$ that shows the lowest discrepancy.

# C.3 Discussion on meta agent parameters

Following the heterogeneous agent settings [18], the meta agent employs several probabilistic parameters to ensure heterogeneity. The parameters and their usage are as follows: $\lambda _ { f }$ for fundamental weight, $\lambda _ { c }$ for chartist weight, $\lambda _ { n }$ for noise weight, $\tau _ { 0 }$ for estimation horizon, $\alpha _ { 0 }$ for risk aversion, $\sigma _ { n }$ for noisy return. All the parameters are for mimicking human preference in real stock market.

Throughout our experiments, we fix $\lambda _ { f } ~ = ~ 1 0 , \lambda _ { c } ~ = ~ 1 . 5 , \lambda _ { n } ~ = ~ 1 , \tau _ { 0 } ~ = ~ 3 0 , \alpha _ { 0 } ~ = ~ 0 . 1 , \sigma _ { n } ~ = ~$ $1 e ^ { - 4 } , p _ { 0 } = 1 0$ for all runs to avoid overfitting these parameters. Nevertheless, they can be calibrated to further improve fidelity.

# D Additional results

We provide the full result table with both mean and std across 3 independent run with different random seeds for main tables as below. These results could confirm that the advantage of DiGA is significant.

Table 6: MSE between the targeted indicator and the generated aggregative statistics of generated order flow. Best results are highlighted with bold face.   

<table><tr><td></td><td></td><td></td><td colspan="5">A-Main</td><td colspan="5">ChiNext</td></tr><tr><td>Target</td><td>Method</td><td></td><td>Lower</td><td>Low</td><td>Medium</td><td>High</td><td>Higher</td><td>Lower</td><td>Low</td><td>Medium</td><td>High</td><td>Higher</td></tr><tr><td rowspan="6">Return</td><td rowspan="3">No Control</td><td></td><td>1.443</td><td>0.583</td><td>0.59</td><td>0.813</td><td>2.337</td><td>0.979</td><td>0.684</td><td>0.9023</td><td>1.718</td><td>3.923</td></tr><tr><td>mean</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mean</td><td></td><td>0.44</td><td>0.228</td><td></td><td>0.634</td><td>1.285</td><td>0.807</td><td>0.243</td><td></td><td></td></tr><tr><td rowspan="2">Discrete</td><td></td><td>1.055</td><td></td><td></td><td>0.429</td><td></td><td></td><td></td><td></td><td>0.013</td><td>0.169</td></tr><tr><td>mean</td><td>0.206</td><td>0.178</td><td>0.161</td><td>0.184</td><td>0.21</td><td></td><td></td><td></td><td>0449</td><td>0.522</td></tr><tr><td rowspan="2">Continuous</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.584</td><td>0.539</td><td>0.39</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6">Amplitude</td><td rowspan="3">No Control Discrete</td><td>mean</td><td>8.521</td><td>0.248</td><td>0.268</td><td>0.699</td><td>3.208</td><td>1.134</td><td>0.638</td><td>0.427</td><td>0.608</td><td>2.763</td></tr><tr><td></td><td>0.09</td><td>0.088</td><td>0.309</td><td>0.502</td><td>0.933</td><td></td><td></td><td>0.346</td><td>0.533</td><td></td></tr><tr><td>mean</td><td></td><td></td><td></td><td></td><td></td><td>0.057</td><td>0.134</td><td></td><td></td><td>0.063</td></tr><tr><td rowspan="2">Continuous</td><td></td><td></td><td></td><td></td><td>0.47</td><td>0348</td><td></td><td></td><td>0.045</td><td>0.37</td><td></td></tr><tr><td>mean</td><td>0.014</td><td>8076</td><td>0.149</td><td></td><td></td><td>0.119</td><td>0.116</td><td></td><td></td><td>0.973</td></tr><tr><td rowspan="4"></td><td rowspan="2">No Control</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.713</td><td>1.737</td><td>5.221</td></tr><tr><td>mean std</td><td>0.021 0.003</td><td>0.115 0.018</td><td>0.431 0.038</td><td>1.209 0.065</td><td>4.288 0.123</td><td>0.029 0.006</td><td>0.246 0.015</td><td>0.034</td><td>0.061</td><td>0.116</td></tr><tr><td rowspan="2">Discrete</td><td></td><td></td><td></td><td></td><td>0.890</td><td>2.308</td><td>0.029</td><td>0.18</td><td>0.484</td><td>0.048</td><td>2.057</td></tr><tr><td>mean</td><td>0.016</td><td>0.13</td><td>0.383</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2"></td><td rowspan="2">Continuous</td><td></td><td></td><td></td><td></td><td>8.774</td><td>2.388</td><td>0.028</td><td>0.78</td><td>0.473</td><td>1016</td><td>2.631</td></tr><tr><td>mean</td><td>0.011</td><td>0.104</td><td>0.318</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 7: K-L divergence of stylized facts distribution between real and simulated order flow.   

<table><tr><td></td><td></td><td colspan="4">A-Main</td><td colspan="4">ChiNext</td></tr><tr><td>Model</td><td></td><td>MinR</td><td>RetAC</td><td>VolC</td><td>OIR</td><td>MinR</td><td>RetAC</td><td>VolC</td><td>OIR</td></tr><tr><td rowspan="2">RFD</td><td></td><td></td><td></td><td></td><td></td><td></td><td>2.987</td><td>0.391</td><td></td></tr><tr><td>mean</td><td>1.198</td><td>5.010</td><td>0.39</td><td>0.015</td><td>0.272</td><td></td><td></td><td>0.02</td></tr><tr><td rowspan="2">RMSC</td><td></td><td></td><td></td><td></td><td></td><td>1.371</td><td>7.461</td><td>0.668</td><td></td></tr><tr><td>mean</td><td>2.640</td><td>10.170</td><td>1.237</td><td>0.56</td><td></td><td></td><td></td><td>0.088</td></tr><tr><td rowspan="2">LOBGAN</td><td>mean</td><td>0.151</td><td>1.903</td><td>1.101</td><td>0.309</td><td>0.135</td><td>1.711</td><td>0.507</td><td>0.282</td></tr><tr><td>std</td><td>0.007</td><td>1.045</td><td>0.639</td><td>0.011</td><td>0.002</td><td>1.043</td><td>0.108</td><td>0.010</td></tr><tr><td rowspan="2">DiGA</td><td></td><td></td><td></td><td></td><td>0.009</td><td>0.079</td><td>1.997</td><td>0.218</td><td>0.00</td></tr><tr><td>mean</td><td>0.084</td><td>2.781</td><td>0.273</td><td></td><td></td><td></td><td></td><td></td></tr></table>