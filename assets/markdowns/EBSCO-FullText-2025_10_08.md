# Article Enhancing Portfolio Performance through Financial Time-Series Decomposition-Based Variational Encoder-Decoder Data Augmentation

Bayartsetseg Kalina $\textcircled{1}$ , Ju-Hong Lee \* and Kwang-Tek Na

Citation: Kalina, B.; Lee, J.-H.; Na, K.-T. Enhancing Portfolio Performance through Financial Time-Series Decomposition-Based Variational Encoder-Decoder Data Augmentation. Symmetry 2024, 16, 283. https://doi.org/10.3390/sym16030283

Academic Editors: Diyin Tang and Danyang Han

Received: 1 February 2024   
Revised: 25 February 2024   
Accepted: 27 February 2024   
Published: 29 February 2024

Department of Electrical and Computer Engineering, Inha University, Incheon 22212, Republic of Korea; 22192320@inha.edu (B.K.); kwangtek.na@inha.edu (K.-T.N.)   
\* Correspondence: juhong@inha.ac.kr

Abstract: The objective of portfolio diversification is to reduce risk and potentially enhance returns by spreading investments across different asset classes. Existing portfolio diversification models have traditionally been trained on historical financial time series data. However, several issues arise with historical financial time series data, making it challenging to train models effectively to achieve the portfolio diversification objective: an insufficient amount of training data and the uncertainty deficiency problem, wherein the uncertainty that existed in the past is not visible in the present. Insufficient datasets, characterized by small data size, result in information asymmetry and compromise portfolio performance. This limitation underscores the importance of adopting a pattern-centric data augmentation approach, capable of unveiling hidden patterns and structures within the financial time series data. To address these challenges, this paper introduces the financial time series decomposition-based variational encoder-decoder (FED) method to augment financial time series data, overcoming the limitations of insufficient training data and providing a more realistic and dynamic simulation of the financial market environment. By decomposing the data into distinct components, such as trend, dispersion, and residual, FED leverages pattern-centric data augmentation within the financial time series data. In the environment generated using the FED method, this paper proposes a two-class portfolio diversification, called FED2Port. It integrates stochastic elements into the reward function, enabling a reinforcement learning algorithm to learn from a comprehensive spectrum of financial market uncertainties. The experimental results demonstrate that the proposed model significantly enhances portfolio performance.

Keywords: portfolio diversification; data augmentation; financial time-series decomposition; variational encoder-decoder

# 1. Introduction

Financial investments involve a trade-off between risk and return. Higher potential returns usually come with higher risks. A diversified portfolio is an investment strategy that involves spreading investments across different asset classes. Large-scale funds, such as national pensions worldwide, invest in a diverse range of assets. In most countries, equities and bills and bonds were the two main asset classes in which pension capital was invested in 2020, accounting for more than half of the investment in 35 out of 38 OECD countries and four reporting non-OECD G20 jurisdictions [1]. The Melbourne Mercer Global Pension Index (MMGPI) considers a split between growth and defensive assets [2]. Growth assets typically include high-risk assets, such as equities, property, and some alternative assets. On the other hand, defensive assets include low-risk assets, such as bills and bonds, as well as cash and deposits.

The present study classifies financial assets into two broad categories based on their inherent characteristics and the level of associated risk: high-risk and low-risk assets. This classification is similar to the MMGPI categorization, with growth representing high-risk and defensive representing low-risk. Such categorization assists investors in making wellinformed portfolio decisions, balancing risk tolerance and investment goals. Investments in high-risk assets can offer significantly large returns, making them attractive to investors seeking aggressive growth, but they also come with a higher likelihood of losses. On the other hand, investments in low-risk assets are often considered safer for preserving capital and generating modest, consistent returns. Two-class portfolio diversification involves spreading investments between these two classes of assets to reduce the overall portfolio risk.

A buy-and-hold strategy is a long-term investment approach where an investor buys assets and holds onto them for an extended period, regardless of short-term market fluctuations. Portfolio rebalancing is the process of periodically adjusting the weights of assets in a portfolio. The tangency portfolio among Markowitz optimization [3], risk budgeting [4,5], recurrent reinforcement learning (RRL) [6,7], and deep deterministic policy gradient (DDPG) [8,9] aims to find the optimal proportion of assets within a given period. Traditional portfolio diversification models [3–5] aim to optimize the allocation of assets in a portfolio to balance risk and return. Markowitz optimization [3] provides a mathematical approach for constructing an investment portfolio that maximizes the expected return for a given level of risk or minimizes the risk for an expected return. Risk budgeting [4,5] involves allocating risk across different assets or asset classes based on predefined risk constraints. This strategy aims to control and manage the portfolio risk effectively. Reinforcement learning (RL) portfolio diversification models [6–11] make decisions by interacting with an environment to maximize a cumulative reward signal. While RRL [6,7,10] aims to learn the optimal policy by maximizing the reward functions, DDPG [8,9] achieves this goal by adjusting the parameters of the actor and critic networks iteratively using optimization techniques.

Existing portfolio diversification models have a common deficit; they are trained using only historical financial time series data. On the other hand, historical financial time series data have the following problems.

Uncertainty deficiency. Both the financial market and its empirical time series data contain inherent uncertainty. At some point, probabilities were assigned to different events or market scenarios, including rises, falls, and magnitudes of changes, with nonzero probabilities. On the other hand, as time elapses, all past events collapse into a single outcome. Consequently, only one event is assigned a $1 0 0 \%$ probability, and the probabilities of all other events are set to $0 \%$ . This phenomenon, termed uncertainty deficiency, suggests that historical financial time series data only represent a sequence of singular events, lacking the diversity of market uncertainties that existed in the past. Ignoring financial market uncertainty can lead to overly confident models that fail to account for unforeseen risks. RL algorithms or traditional models optimized solely based on historical financial time series data may lack robustness and show poor capability when applied to novel or extreme events. Insufficient amount of training data. Historical financial time-series datasets are often not large enough for training due to financial market uncertainty. For example, even with 10 years of daily data for an asset class (250 trading days in a year $\times 1 0$ years $= 2 5 0 0$ ), the amount is relatively small, only $2 . 5 \mathrm { k } .$ . Insufficient datasets, characterized by small data size, result in information asymmetry and compromise portfolio performance.

Good results are not possible in the face of future uncertainty because of these problems. A financial time series decomposition-based variational encoder-decoder (FED) data augmentation is proposed to address the challenges of financial market uncertainty and insufficient training data, providing a more realistic and dynamic simulation of the financial market environment. Under the environment generated by FED, this paper proposes a two-class portfolio diversification (FED2Port), allowing the RL algorithm to learn from a comprehensive spectrum of financial market uncertainties.

The main contributions of this paper are as follows.

. FED for Financial Time Series Data Augmentation. The first contribution introduces an innovative financial time series data augmentation called the FED. Generating nonstationary financial time series data is deemed challenging, and FED addresses this challenge by leveraging decomposition techniques, separating the financial time series into distinct components (trend, dispersion, and residual). Based on the encoderdecoder architecture, the FED method utilizes latent variables further decomposed into components. This pattern-centric approach provides a profound understanding of the underlying structure of financial time series data, unveiling the hidden patterns or structures and offering insights into factors influencing observed trends and fluctuations. FED captures the distributions of latent variable components, generating more realistic financial time series data. In doing so, the FED method revives some of the past uncertainty that had disappeared, compensating for the problems of uncertainty deficiency and an insufficient amount of training data.

. FED2Port for Decision-Making under Financial Market Uncertainty. The second contribution is the proposal of FED2Port as a novel diversification approach to enhance the efficiency of RL algorithms. Specifically tailored for RL portfolio diversification models, FED2Port addresses the uncertainty deficiency problem inherent in historical financial time series data. FED2Port trains the RL algorithm under the financial market environment generated using the FED. This environment simulation incorporates stochastic elements in the reward function, enabling the algorithm to learn from a more comprehensive spectrum of financial market uncertainties. Therefore, FED2Port improves the adaptability of the algorithm significantly, empowering it to make well-informed decisions in the face of future uncertainty, ultimately enhancing portfolio performance.

# 2. Related Work

Financial time series data generation plays a significant role in RL portfolio diversification models by addressing the challenges of financial market uncertainty and enhancing portfolio performance. Simulating the financial market environment with additional scenarios and variations can help improve the robustness of portfolio diversification. This ensures that the RL algorithm is exposed to a broader range of market conditions. The two most prominent types of generative models are the generative adversarial nets (GANs) [12,13] and variational autoencoders (VAEs) [14,15].

GANs usually generate more realistic data but face training stability and sampling diversity challenges. GANs are based on a two-player minimax game with value function $V ( G , D )$ :

$$
\begin{array} { r l } & { \underset { G } { \mathrm { m i n } } \underset { D } { \mathrm { m a x } } V \big ( G , D \big ) : = \mathbb { E } _ { { x } \sim { p _ { d a t a } } ( x ) } [ \log D ( x ) ] } \\ & { \qquad + \mathbb { E } _ { z \sim { p _ { Z } } ( z ) } [ \log ( 1 - D ( G ( z ) ) ) ] } \end{array}
$$

where $z , G ,$ and $D$ are random noise, a generator, and a discriminator, respectively. GANs involve training a generator and a discriminator in a competitive setting, which can sometimes lead to training instabilities, mode collapse, or difficulties in convergence. Time-series GAN (TimeGAN) [16] and real-world time series GAN (RTSGAN) [17] are designed to generate synthetic data that closely resemble real-world time series data. In TimeGAN, the generator produces embeddings, and the recovery produces time series data based on the generated embeddings. RTSGAN shares similarities with TimeGAN but sets itself apart by specializing in generating time series data with variable lengths. They do not address the nonstationary financial time series data generation.

VAEs often exhibit more stable training dynamics compared to GANs. VAEs explicitly model the generative process by assuming a specific form for the latent variable distribution. This can be advantageous in scenarios where understanding the generative process is crucial. VAEs aim to maximize the probability of the generated output with respect to the input and produce an output from a target distribution by compressing the input into a latent space. VAEs can learn via maximum likelihood using a variational approach to maximize the evidence lower bound (ELBO) as follows:

$$
E L B O = - D _ { K L } [ q _ { \phi } ( z | x ) | | p ( z ) ] + \mathbb { E } _ { q _ { \phi } ( z | x ) } [ \log p _ { \theta } ( x | z ) ]
$$

where $q _ { \phi } ( z | x )$ is an approximate posterior distribution for the latent variables, also known as a probabilistic encoder; $p ( z )$ is a prior over the latent variables; $p _ { \theta } ( x | z )$ is a likelihood function, also known as a probabilistic decoder.

Time series decomposition [18–26] aims to decompose a time series into its components structurally and interpretably. These components typically include the trend, seasonal, and residual components. The trend component represents the long-term direction or underlying movement in the financial time series data, capturing the overall trajectory, which can be either linear or nonlinear. It helps identify whether the financial time series data generally increases or decreases over time. The seasonal component captures the periodic patterns and fluctuations within a year or specified period, elucidating regular, predictable movements in the financial time series data. The cyclical component represents longer-term fluctuations tied to economic or business cycles, spanning multiple years and identifying broader economic trends. The residual component, the error term or reminder, accounts for random and unexplained variability in the financial time series data, not attributable to the trend, seasonal, or cyclical components. Time series decomposition can be expressed in additive or multiplicative forms. An additive decomposition [22] would be written as

$$
y _ { t } = T _ { t } + S _ { t } + R _ { t }
$$

A multiplicative decomposition [22] would be written as

$$
y _ { t } = T _ { t } \times S _ { t } \times R _ { t }
$$

where $y _ { t } , S _ { t } , T _ { t } ,$ and $R _ { t }$ are the data, seasonal component, trend component, and residual component, respectively, all at time $t$ . The additive decomposition is appropriate when the magnitude of the seasonal fluctuations or the variation around the trend does not change as the level of the time series increases. The multiplicative decomposition is more appropriate when the variation in the seasonal pattern or the variation around the trend is proportional to the time series level [22].

STD decomposition [26] extracts the components of the seasonality, trend, and dispersion, it is expressed as

$$
y _ { t } = S _ { t } \times D _ { t } + T _ { t }
$$

where $y _ { t } , S _ { t } , D _ { t } ,$ and $T _ { t }$ are the data, seasonal component, dispersion component, and trend component, respectively, all at time $t$ . STD with a reminder component [26], called STDR, is defined as follows:

$$
y _ { t } = S _ { t } ^ { \prime } \times D _ { t } + T _ { t } + R _ { t }
$$

where $S _ { t } ^ { \prime }$ is an averaged seasonal component, and $R _ { t }$ is a reminder component, all at time $t$

Decomposition is a crucial tool in analyzing and simulating nonstationary financial time series data, providing insights into changing patterns and helping simulators better understand and model the complexities of financial markets.

The Markowitz optimization [3], known as modern portfolio theory (MPT), provides a mathematical approach to constructing portfolios to maximize the expected returns while minimizing risk. The tangency portfolio represents the optimal portfolio that maximizes a risk-adjusted return measure, the Sharpe ratio [27]. Risk budgeting [4,5] is a portfolio construction approach that involves allocating risk among different assets based on predefined risk constraints. A risk budgeting model helps to manage and control the overall risk of a portfolio while optimizing returns. Previous studies [3–5] relied on the assumption of a stationary market. The Black–Litterman model [28] is an asset allocation framework that combines market equilibrium assumptions [29] with an investor’s subjective views. Applying the Black–Litterman model requires the availability of expert views or a predictive model that can represent those expert views.

Maximizing future rewards typically involves optimizing a sequence of decisions or actions to achieve the best possible outcomes. It is a fundamental problem in various fields, including reinforcement learning. Ref. [6] introduced an RL model called recurrent reinforcement learning (RRL) for portfolio management. They used the Sharpe ratio as the reward. A previous study [7] used the modified RRL, which optimizes the Sharpe ratio with batch learning. The deep deterministic policy gradient (DDPG) [30] was used for portfolio management [8,9]. RL portfolio diversification models [6–9] can construct optimal portfolios that achieve the best possible rewards, such as the expected return, the Sharpe ratio [27], the Sortino ratio [31], or the market-adaptive ratio [32]. The Sharpe ratio [27] relates the excess returns on a portfolio to its risk, the standard deviation of the excess return. The market-adaptive ratio [32] is a risk-adjusted return based on a market-type measure, rho. This ratio is a general form of the Sharpe ratio, considering the characteristics of the market types. During bull markets, the focus is on seeking high returns and embracing risk. In contrast, it aims to preserve capital and minimize risk during bear markets. The Sortino ratio [31] focuses on the downside risk. RL portfolio diversification models leverage insights gained from data analysis instead of relying on an assumption. On the other hand, they use observed historical environments to estimate the model parameters that lead to the uncertainty deficiency and the shortage of training data problems.

# 3. Proposed Methods

The $d$ -day log-return vector of the high-risk asset (or the low-risk asset) at time $t$ is defined as

$$
\begin{array} { r } { \pmb { x } _ { t } = \left[ \begin{array} { c } { x _ { t - d + 1 } } \\ { x _ { t - d + 2 } } \\ { \cdot \cdot \cdot } \\ { x _ { t } } \end{array} \right] = \left[ \begin{array} { c } { \mathrm { l o g } \frac { p _ { t - d + 1 } } { p _ { t - d } } } \\ { \mathrm { l o g } \frac { p _ { t - d + 2 } } { p _ { t - d + 1 } } } \\ { \cdot \cdot \cdot } \\ { \mathrm { l o g } \frac { p _ { t } } { p _ { t - 1 } } } \end{array} \right] } \end{array}
$$

where $p _ { t }$ is the price of a high-risk asset (or the low-risk asset) at time $t$

# 3.1. FED

The encoder-decoder architecture encourages the latent space to have meaningful representations of the data, which is advantageous for operations like interpolation or feature manipulation. Based on this architecture, the FED method utilizes latent variables further decomposed into components. This approach provides a profound understanding of the underlying structure of financial time series data, unveiling the hidden patterns or structures and offering insights into factors influencing observed trends and fluctuations.

Time series decomposition is a fundamental technique in time series analysis that separates complex time series data into individual components, helping understand the underlying dynamics. Most time series decomposition methods have focused on the trend, seasonal, and residual components. Previous work [26] considers a component related to the dispersion of the time series. The trend and dispersion components are crucial for generating financial time series data due to their nonstationary property. FED incorporates components of the trend, dispersion, and residual. The trend component, $m _ { t . }$ , is the mean return at time $t ,$ , representing the direction of financial time series data. The dispersion component, $s _ { t } ,$ is the standard deviation of the return at time $t ,$ , representing the fluctuation of financial time series data. The residual component accounts for the unexplained variability in the financial time series data. The primary concept of the proposed model is to apply decomposition into the hidden space. By emphasizing these components, FED leverages pattern-centric data augmentation within the financial time series data.

Assume that data $\tilde { { \boldsymbol { x } } } _ { t }$ are generated by a decoder with a probabilistic latent variable, $h _ { t }$

$$
\tilde { { \boldsymbol { x } } } _ { t } = D ( h _ { t } )
$$

The FED method is based on the latent variable decomposition,

$$
\pmb { h } _ { t } = \pmb { \nu } _ { t } \times \pmb { \tau } _ { t } \times \pmb { \xi } _ { t }
$$

where $\nu _ { t } \sim N ( \mu _ { \nu t } , \Sigma _ { \nu t } )$ is a probabilistic trend component of the latent variable, $\tau _ { t } \sim N ( \mu _ { \tau t } , \Sigma _ { \tau t } )$ is a probabilistic dispersion component of the latent variable, and $\pmb { \xi } _ { t } \sim N ( \pmb { \mu } _ { \xi t } , \pmb { \Sigma } _ { \xi t } )$ is a probabilistic residual component of the latent variable, all at time $t$ .

The product of two multivariate normal distributions results in another multivariate normal distribution [33], which is valuable and highly useful in the proposed model. Consequently, the parameters of the probabilistic hidden variable, $\pmb { h } _ { t } \sim N ( \pmb { \mu } _ { h t } , \pmb { \Sigma } _ { h t } )$ were calculated, as follows:

$$
\begin{array} { l } { { N ( \mu _ { \nu t } , \Sigma _ { \nu t } ) \times N ( \mu _ { \tau t } , \Sigma _ { \tau t } ) \times N ( \mu _ { \xi t } , \Sigma _ { \xi t } ) = N ( \mu _ { 1 t } , \Sigma _ { 1 t } ) \times N ( \mu _ { \xi t } , \Sigma _ { \xi t } ) } } \\ { { \qquad = N ( \mu _ { h t } , \Sigma _ { h t } ) } } \end{array}
$$

where

$$
\begin{array} { r l } & { \Sigma _ { 1 t } = ( \Sigma _ { \nu t } ^ { - 1 } + \Sigma _ { \tau t } ^ { - 1 } ) ^ { - 1 } } \\ & { \pmb { \mu } _ { 1 t } = \Sigma _ { 1 t } \Sigma _ { \nu t } ^ { - 1 } \pmb { \mu } _ { \nu t } + \Sigma _ { 1 t } \Sigma _ { \tau t } ^ { - 1 } \pmb { \mu } _ { \tau t } } \end{array}
$$

and

$$
\begin{array} { r l } & { \Sigma _ { h t } = ( \Sigma _ { 1 t } ^ { - 1 } + \Sigma _ { \xi t } ^ { - 1 } ) ^ { - 1 } } \\ & { \quad \quad = \big ( \Sigma _ { \nu t } ^ { - 1 } + \Sigma _ { \tau t } ^ { - 1 } + \Sigma _ { \xi t } ^ { - 1 } \big ) ^ { - 1 } } \\ & { \quad \quad \mu _ { h t } = \Sigma _ { h t } \Sigma _ { 1 t } ^ { - 1 } \mu _ { 1 t } + \Sigma _ { h t } \Sigma _ { \xi t } ^ { - 1 } \mu _ { \xi t } } \\ & { \quad \quad \quad = \Sigma _ { h t } \Sigma _ { \nu t } ^ { - 1 } \mu _ { \nu t } + \Sigma _ { h t } \Sigma _ { \tau t } ^ { - 1 } \mu _ { \tau t } + \Sigma _ { h t } \Sigma _ { \xi t } ^ { - 1 } \mu _ { \xi t } } \end{array}
$$

FED employs three encoders to model the three probabilistic components of the latent variables, including trend (return), dispersion (standard deviation), and residual. Similar to reference [14], the reparameterization trick was used. Figure 1 illustrates the general framework of FED.

![](images/09469bc772920538c10b8a3fb19b96b0cfb03dbe6ba405707d81923ec8c2a100.jpg)  
Figure 1. General framework of financial time series decomposition-based variational encoderdecoder (FED). $\boldsymbol { x } _ { t }$ is the $d \cdot$ -day log-return vector of the high-risk asset (or the low-risk asset) at time $t$ $\tilde { m } _ { t } , \tilde { s } _ { t } ,$ and $\tilde { { \boldsymbol { x } } } _ { t }$ are the generated trend (return), the generated dispersion (standard deviation), and the generated $d$ -day log-return vector of the high-risk asset (or the low-risk asset), respectively, all at time t. $\nu _ { t } \sim N ( \mu _ { \nu t } , \Sigma _ { \nu t } )$ is a probabilistic trend component of the latent variable, $\tau _ { t } \sim N ( \mu _ { \tau t } , \Sigma _ { \tau t } )$ is a probabilistic dispersion component of the latent variable, and $\pmb { \xi } _ { t } \sim N ( \pmb { \mu } _ { \xi t } , \pmb { \Sigma } _ { \xi t } )$ is a probabilistic residual component of the latent variable. $\pmb { h } _ { t } = \pmb { \nu } _ { t } \times \pmb { \tau } _ { t } \times \pmb { \xi } _ { t }$ is the decomposed latent variable.

The marginal log-likelihood of the trend $m _ { t }$

$$
\begin{array} { r l } & { \log p ( m _ { t } ) = \log \int p _ { \theta _ { \nu } } ( m _ { t } | \nu _ { t } ) p ( \nu _ { t } ) d \nu _ { t } } \\ & { \qquad = \log \int \frac { q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } ) } { q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } ) } p _ { \theta _ { \nu } } ( m _ { t } | \nu _ { t } ) p ( \nu _ { t } ) d \nu _ { t } } \\ & { \qquad \geq \int q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } ) \log \left[ \frac { p ( \nu _ { t } ) } { q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } ) } p _ { \theta _ { \nu } } ( m _ { t } | \nu _ { t } ) \right] d \nu _ { t } } \\ & { \qquad = - \int q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } ) \log \left[ \frac { q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } ) } { p ( \nu _ { t } ) } \right] d \nu _ { t } + \int q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } ) \log \left[ p _ { \theta _ { \nu } } ( m _ { t } | \nu _ { t } ) \right] d \nu _ { t } } \\ & { \qquad = - D _ { K L } \Big [ q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } ) | | p ( \nu _ { t } ) \Big ] + \mathbb { E } _ { q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } ) } \Big [ \log p _ { \theta _ { \nu } } ( m _ { t } | \nu _ { t } ) \Big ] } \end{array}
$$

where $p _ { \theta _ { \nu } } ( m _ { t } | \nu _ { t } )$ is the conditional probability distribution of the trend $m _ { t }$ given the latent variable $\nu _ { t } ,$ modeled by a decoder and the sampling of the latent variable, and $q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } )$ is the conditional probability distribution of the latent variable $\nu _ { t }$ given data $\boldsymbol { x } _ { t } ,$ modeled by an encoder and the reparameterization trick. The above bound is the evidence lower bound (ELBO).

Similarly, the marginal log-likelihood of dispersion $s _ { t }$ is expressed as

$$
\begin{array} { r l } & { \log p ( s _ { t } ) = \log \displaystyle \int p _ { \theta _ { \tau } } ( s _ { t } | \tau _ { t } ) p ( \tau _ { t } ) d \tau _ { t } } \\ & { \quad \quad \quad = \log \displaystyle \int \frac { q _ { \phi _ { \tau } } ( \tau _ { t } | x _ { t } ) } { q _ { \phi _ { \tau } } ( \tau _ { t } | x _ { t } ) } p _ { \theta _ { \tau } } ( s _ { t } | \tau _ { t } ) p ( \tau _ { t } ) d \tau _ { t } } \\ & { \quad \quad \quad \geq - D _ { K L } \Big [ q _ { \phi _ { \tau } } ( \tau _ { t } | x _ { t } ) | | p ( \tau _ { t } ) \Big ] + \mathbb { E } _ { q _ { \phi \tau } ( \tau _ { t } | x _ { t } ) } \Big [ \log p _ { \theta _ { \tau } } ( s _ { t } | \tau _ { t } ) \Big ] } \end{array}
$$

where $p _ { \theta _ { \tau } } ( s _ { t } | \tau _ { t } )$ is the conditional probability distribution of the dispersion $s _ { t }$ given the latent variable $\tau _ { t } ,$ modeled by a decoder and the sampling of the latent variable, and $q _ { \phi _ { \tau } } ( \tau _ { t } | \boldsymbol { x } _ { t } )$ is the conditional probability distribution of the latent variable $\tau _ { t }$ given data $\boldsymbol { x } _ { t } ,$ modeled by an encoder and the reparameterization trick.

Similarly, the marginal log-likelihood of data $x _ { t }$ is expressed as

$$
\begin{array} { r l } & { \log p ( x _ { t } ) = \log \int p _ { \theta } ( x _ { t } | h _ { t } ) p ( h _ { t } ) d h _ { t } } \\ & { \qquad = \log \int \displaystyle \frac { q _ { \phi } ( h _ { t } | x _ { t } ) } { q _ { \phi } ( h _ { t } | x _ { t } ) } p _ { \theta } ( x _ { t } | h _ { t } ) p ( h _ { t } ) d h _ { t } } \\ & { \qquad \geq - D _ { K L } \Big [ q _ { \phi } ( h _ { t } | x _ { t } ) | | p ( h _ { t } ) \Big ] + \mathbb { E } _ { q _ { \phi } ( h _ { t } | x _ { t } ) } \Big [ \log p _ { \theta } ( x _ { t } | h _ { t } ) \Big ] } \end{array}
$$

where $p _ { \theta } ( \pmb { x } _ { t } | h _ { t } )$ is the conditional probability distribution of data $x _ { t }$ given the latent variable $h _ { t } ,$ , modeled by a decoder and the sampling of the latent variable, and $q _ { \phi } ( h _ { t } | x _ { t } )$ is the conditional probability distribution of the latent variable $h _ { t }$ given data $\boldsymbol { x } _ { t } ,$ modeled by encoders and the reparameterization trick. The FED method aims to maximize the combination of the above three bounds as follows:

$$
\begin{array} { r l } & { L _ { F E D } : = \alpha \Bigg ( - D _ { K L } \Big [ q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } ) | | p ( \nu _ { t } ) \Big ] + \mathbb { E } _ { q _ { \phi _ { \nu } } ( \nu _ { t } | x _ { t } ) } \Big [ \log p _ { \theta _ { \nu } } ( m _ { t } | \nu _ { t } ) \Big ] \Bigg ) } \\ & { \qquad + \beta \Bigg ( - D _ { K L } \Big [ q _ { \phi _ { \tau } } ( \tau _ { t } | x _ { t } ) | | p ( \tau _ { t } ) \Big ] + \mathbb { E } _ { q _ { \phi \tau } ( \tau _ { t } | x _ { t } ) } \Big [ \log p _ { \theta _ { \tau } } ( s _ { t } | \tau _ { t } ) \Big ] \Bigg ) } \\ & { \qquad + \gamma \Bigg ( - D _ { K L } \Big [ q _ { \phi } ( h _ { t } | x _ { t } ) | | p ( h _ { t } ) \Big ] + \mathbb { E } _ { q _ { \phi } ( h _ { t } | x _ { t } ) } \Big [ \log p _ { \theta } ( x _ { t } | h _ { t } ) \Big ] \Bigg ) } \end{array}
$$

where $\alpha , \beta ,$ and $\gamma$ are hyperparameters that control the importance of each task.

# 3.2. FED2Port

The environment of the FED2Port is defined as follows.

. The action is defined as the weight vector:

$$
\begin{array} { r } { \pmb { a } _ { t } = \left[ \begin{array} { l } { a _ { t , h r } } \\ { a _ { t , l r } } \end{array} \right] } \end{array}
$$

where $a _ { t , h r }$ and $a _ { t , l r } \geq 0$ represent the weights of a high-risk asset and a low-risk asset, respectively, with the constraint that $a _ { t , h r } + a _ { t , l r } = 1$ .

The state is defined as the portfolio return $\pmb { s } _ { t }$

$$
\begin{array} { r } { \pmb { s } _ { t } = \ b { a } _ { t - 1 , h r } \pmb { x } _ { t , h r } + \ b { a } _ { t - 1 , l r } \pmb { x } _ { t , l r } } \end{array}
$$

where $x _ { t , h r }$ and $x _ { t , l r }$ are the $d$ -day log-return vectors of the high-risk and low-risk assets, respectively.

The reward is defined as the market-adaptive ratio [32]:

$$
r _ { t } ( \tilde { x } _ { t + d , h r } , \tilde { x } _ { t + d , l r } , \pmb { a } _ { t } ) = \frac { ( \bar { R } _ { p } - R _ { f } ) ^ { \rho _ { h r } } } { \sigma _ { p } ^ { 1 / \rho _ { h r } } }
$$

where $\begin{array} { r } { \rho _ { h r } = \frac { 2 } { 1 + e ^ { - R _ { h r } } } } \end{array}$ represents the rho of the high-risk asset; $R _ { h r }$ is the return of the high-risk asset, $\tilde { \boldsymbol { x } } _ { t + d , h r }$ and $\tilde { { \boldsymbol { x } } } _ { t + d , l r }$ are the generated log-return vectors of the high-risk and low-risk assets, respectively. FED methods are used for high-risk and low-risk assets. ${ \bar { R } } _ { p }$ and $\sigma _ { p }$ represent the expected return and standard deviation of the total portfolio, respectively, and $R _ { f }$ is the risk-free rate. In this paper, the risk-free rate equals zero. By using the market-adaptive ratio as the reward, FED2Port can take into account market characteristics such as bull and bear markets.

The agent, $\pi _ { \omega . }$ , receives the portfolio return and selects an action.

$$
\pmb { a } _ { t } = \pi _ { \omega } ( \pmb { s } _ { t } )
$$

It controls the policy using an evaluation of the reward. Figure 2 illustrates the general framework of FED2Port.

![](images/6d604877d6805d41206680dd0e822d6c1f0a90080d9576d126a3ce35b6588864.jpg)  
Figure 2. General framework of two-class portfolio diversification (FED2Port). $\pmb { s } _ { t }$ and $\mathbf { \alpha } _ { a _ { t } }$ are the state and the action, respectively, at time t. $\tilde { { \boldsymbol { x } } } _ { t + d , h r }$ and $\tilde { { \boldsymbol { x } } } _ { t + d , l r }$ represent the generated log-return vectors of the high-risk and low-risk assets, respectively, at time $t$ . $r _ { t }$ is the reward at time $t$ .

The objective of FED2Port is to maximize the expected reward,

$$
\operatorname* { m a x } _ { \omega } \mathbb { E } _ { \tilde { { \boldsymbol { x } } } _ { t + d , h r } , \tilde { { \boldsymbol { x } } } _ { t + d , l r } } \left[ r _ { t } \big ( \tilde { { \boldsymbol { x } } } _ { t + d , h r } , \tilde { { \boldsymbol { x } } } _ { t + d , l r } , { \boldsymbol { a } } _ { t } \big ) \right]
$$

# 4. Experiment

4.1. Dataset

FED2Port aims to allocate the total investment into two classes: high-risk and low-risk assets. This paper considers three stock indices and three bond funds (Table 1) in the experiment. The daily data from January 2010 to December 2022 (https://finance.yahoo.

com/ accessed on 1 October 2023) were included. To initialize models, they were trained using the five-year data from January 2010 to December 2014 for each dataset. Then, we tested the model using eight-year data, from January 2015 to December 2022.

Table 1. Assets.   

<table><tr><td>Class</td><td>Symbol</td><td>Explanation</td></tr><tr><td rowspan="3">High-risk assets</td><td>SP500</td><td>S&amp;P500 Index</td></tr><tr><td>DAX</td><td>DAX Index</td></tr><tr><td>KOSPI</td><td>KOSPI Index</td></tr><tr><td rowspan="3">Low-risk assets</td><td>BND</td><td>Vanguard Total Bond Market Index Fund</td></tr><tr><td>BSV</td><td>Vanguard Short-Term Bond Index Fund</td></tr><tr><td>VCIT</td><td> Vadgxard Itermediate-Tem Treasury</td></tr></table>

Figure 3 depicts the price data of the assets, while Table 2 lists the differences between stock market indices and bond funds. While stock market indices carry higher risk, bond funds offer lower risk. Nine two-class portfolios (Table 3) were considered, comprising three stock indices and three bond funds (Table 1), to assess the performance of the proposed model.

![](images/d765f79d62587084707a920ecaf4fabfda07344f68c9527dba5c0521700fe470.jpg)  
Figure 3. Graphs of the price data of the assets.

Table 2. Statistic of funds during test period.   

<table><tr><td></td><td>SP500</td><td>DAX</td><td>KOSPI</td><td>BND</td><td>BSV</td><td>VCIT</td></tr><tr><td>The standard deviation of the portfolio return</td><td>0.5221</td><td>0.6125</td><td>0.5564</td><td>0.1429</td><td>0.0590</td><td>0.1702</td></tr></table>

Table 3. Portfolios.   

<table><tr><td></td><td>Portfolio</td><td>Low-Risk Asset</td><td>High-Risk Asset</td></tr><tr><td>1</td><td>BND&amp;SP500</td><td>Vanguard Total Bond Market Index Fund</td><td>S&amp;P500 Index</td></tr><tr><td>2</td><td>BND&amp;DAX</td><td>Vanguard Total Bond Market Index Fund</td><td>DAX Index</td></tr><tr><td>3</td><td>BND&amp;KOSPI</td><td>Vanguard Total Bond Market Index Fund</td><td>KOSPI Index</td></tr><tr><td>4</td><td>BSV&amp;SP500</td><td>Vanguard Short-Term Bond Index Fund</td><td>S&amp;P500 Index</td></tr><tr><td>5</td><td>BSV&amp;DAX</td><td>Vanguard Short-Term Bond Index Fund</td><td>DAX Index</td></tr><tr><td>6</td><td>BSV&amp;KOSPI</td><td>Vanguard Short-Term Bond Index Fund</td><td>KOSPI Index</td></tr><tr><td>7</td><td>VCIT&amp;SP500</td><td>Vanguard Intermediate-Term Treasury Index Fund</td><td>S&amp;P500 Index</td></tr><tr><td>8</td><td>VCIT&amp;DAX</td><td>Vanguard Intermediate-Term Treasury Index Fund</td><td>DAX Index</td></tr><tr><td>9</td><td>VCIT&amp;KOSPI</td><td>Vanguard Intermediate-Term Treasury Index Fund</td><td>KOSPI Index</td></tr></table>

# 4.2. Benchmarks

For comparison, several benchmarks (Table 4) were considered, including buy-andhold strategies, traditional portfolio diversification models, and RL portfolio diversification models. The buy-and-hold strategy is a long-term investment approach in portfolio management where an investor buys financial assets and holds onto them for an extended period, regardless of short-term market fluctuations. Traditional portfolio diversification models help construct portfolios that align with the investors’ risk tolerance and return objectives. RL portfolio diversification models showcase the adaptability and learning capabilities of reinforcement learning.

Table 4. Comparison benchmarks.   

<table><tr><td>Model</td><td></td><td>Explanation</td></tr><tr><td>1</td><td>100% low-risk asset portfolio</td><td></td></tr><tr><td>２3</td><td>Eou li ghishtedset portfolio</td><td>Buy-and-Hold strategies</td></tr><tr><td>4</td><td>Tangency portfolio</td><td>Traditional portfolio diversification models</td></tr><tr><td>5 6</td><td>Risk Budgeting</td><td></td></tr><tr><td>7</td><td>RRL DDPG</td><td>Historical data-based RL portfolio diversification models</td></tr><tr><td>8</td><td></td><td></td></tr><tr><td>9</td><td>TimeGAN2Port RTSGAN2Port</td><td>Data augmentation-based RL portfolio diversification models</td></tr><tr><td></td><td></td><td></td></tr></table>

# 4.3. Performance Measures

The expected portfolio return, the standard deviation of the portfolio return, and the Sharpe ratio were considered to evaluate the effectiveness of portfolio strategies.

The expected portfolio return (Profit) is expressed as

$$
\mu _ { p } = t \times \bar { R } _ { p }
$$

where $t$ is the length of the test period, and ${ \bar { R } } _ { p }$ is the daily mean return of the portfolio. The expected portfolio return provides insight into the overall portfolio performance, capturing the total change in value over time.

The standard deviation of the portfolio return (Risk) is expressed as follows:

$$
\sigma _ { p } = \sqrt { t \times \frac { \sum _ { i = 1 } ^ { t } ( R _ { p , i } - \bar { R } _ { p } ) ^ { 2 } } { t - 1 } }
$$

where $R _ { p , i }$ is a daily return of the portfolio at time $i$ . The standard deviation of the portfolio return is a key metric in assessing the risk associated with a portfolio. A higher standard deviation indicates greater variability in returns, suggesting higher risk, while a lower standard deviation implies more stability.

The Sharpe ratio is a risk-adjusted return that evaluates the portfolio performance, which was calculated using expected return and risk during the test period, as follows.

$$
S h a r p e r a t i o = \frac { \mu _ { p } } { \sigma _ { p } }
$$

# 4.4. Experimental Results

Network architectures in Figure 4 were used for the encoder and decoder of the FED method. The dimensions of the latent variables were set to 100. The network architecture in Figure 5 was used for the FED2Port agent, $\pi _ { \omega } ,$ which utilizes the Softmax function to generate portfolio weights. A rolling window approach was implemented to retrain the FED2Port model annually from January 2015 to December 2022. The total portfolios were rebalanced for each month (20 trading days). The Profit (Equation (16)), Risk (Equation (17)), and Sharpe ratio (Equation (18)) were considered to evaluate the effectiveness of the portfolio strategies.

![](images/4b49898ad532dca2ed5c633463879972a28fdeb69e0286a9b4ef61cf57b066bd.jpg)  
Figure 4. Network architecture of financial time series decomposition-based variational encoderdecoder (FED). (a) Encoders. (b) Decoders.

![](images/381c788a46d0bd1578af7257a2a49a1baf5ac858dd461dbc7d5df05d821f0f8f.jpg)  
Figure 5. Network architecture of two-class portfolio diversification (FED2Port).

The importance of using FED in FED2Port was demonstrated by comparing the performances of TimeGAN2Port and RTSGAN2Port. Synthetic data were generated using TimeGAN [16] for TimeGAN2Port and RTSGAN [17] for RTSGAN2Port. Ten samples were generated at each time step for each generation.

Tables 5–13 list the experimental results. The empirical evaluation of FED2Port across diverse datasets underscored its robustness and superior performance, consistently outperforming benchmark models, including traditional and reinforcement learning models. The risk–return trade-off is a fundamental trading principle that describes the inverse relationship between investment risk and return. The Sharpe ratio is a helpful measure for quantifying this trade-off. For eight portfolios out of nine, $1 0 0 \%$ low-risk asset portfolios provided the lowest risks, but the profits were not sufficiently strong. In the VCIT&DAX dataset (Table 12), TimeGAN2Port provided the lowest risk, but its profit was also lower. For five portfolios out of nine, $1 0 0 \%$ high-risk asset portfolios offered the highest profits but they also came with the highest risks. In the BND&KOSPI (Table 7) and the BSV&KOSPI (Table 10) datasets, DDPG offered the highest profits, but its risks were higher than those of the proposed model, FED2Port. The Sharpe ratios of FED2Port were the highest among the compared models across all portfolios, indicating that FED2Port delivered the most favorable return per unit of risk undertaken. Other RL portfolio diversification models $( { \mathrm { R R L } } ,$ DDPG, TimeGAN2Port, and RTSGAN2Port) exhibited mixed results in terms of robustness. They sometimes outperformed traditional portfolio models (tangency portfolio and risk budgeting) while yielding poorer results at other times. This variability suggests that the performance of these RL models may be sensitive to specific market conditions or dataset characteristics. The primary concept behind FED2Port is to utilize financial market environment simulation through FED. The importance of using FED was highlighted by comparing the performances of FED2Port, TimeGAN2Port, and RTSGAN2Port. The results demonstrated that employing financial market environment simulation through FED is crucial for enhancing portfolio performance.

Table 5. Results of the BND&SP500 portfolio. Cells with a red background color indicate the best Sharpe ratio in the experiment.   

<table><tr><td>Model</td><td>Profit (Higher the Better)</td><td>Risk (Lower the Better)</td><td>Sharpe Ratio (Higher the Better)</td></tr><tr><td>100% low-risk asset portfolio</td><td>0.0904</td><td>0.1429</td><td>0.6322</td></tr><tr><td>Equally Weighted</td><td>0.3833</td><td>0.2779</td><td>1.3793</td></tr><tr><td>100% high-risk asset portfolio</td><td>0.7459</td><td>0.5221</td><td>1.4286</td></tr><tr><td>Tangency portfolio</td><td>0.5587</td><td>0.4183</td><td>1.3356</td></tr><tr><td>Risk Budgeting</td><td>0.2303</td><td>0.2126</td><td>1.0835</td></tr><tr><td>RRL</td><td>0.2866</td><td>0.2483</td><td>1.1540</td></tr><tr><td>DDPG</td><td>0.0853</td><td>0.2939</td><td>0.2903</td></tr><tr><td>TimeGAN2Port</td><td>0.3956</td><td>0.3027</td><td>1.3072</td></tr><tr><td>RTSGAN2Port</td><td>0.1277</td><td>0.2549</td><td>0.5009</td></tr><tr><td>FED2Port (our)</td><td>0.3755</td><td>0.2101</td><td>1.7869</td></tr></table>

Table 6. Results of the BND&DAX portfolio. Cells with a red background color indicate the best Sharpe ratio in the experiment.   

<table><tr><td>Model</td><td> Prigher th Beter)</td><td> Riswer the Better)</td><td>Shighe Rti Beter)</td></tr><tr><td>100% low-risk asset portfolio</td><td>0.0904</td><td>0.1429</td><td>0.6322</td></tr><tr><td>Equally Weighted</td><td>0.2001</td><td>0.3164</td><td>0.6322</td></tr><tr><td>100% high-risk asset portfolio</td><td>0.4074</td><td>0.6125</td><td>0.6652</td></tr><tr><td>Tangency portfolio</td><td>0.3568</td><td>0.4915</td><td>0.7260</td></tr><tr><td>Risk Budgeting</td><td>0.0806</td><td>0.1685</td><td>0.4783</td></tr><tr><td>RRL</td><td>0.0993</td><td>0.1696</td><td>0.5857</td></tr><tr><td>DDPG</td><td>0.2662</td><td>0.3133</td><td>0.8496</td></tr><tr><td>TimeGAN2Port</td><td>0.1444</td><td>0.3209</td><td>0.4500</td></tr><tr><td>RTSGAN2Port</td><td>0.0940</td><td>0.2750</td><td>0.3417</td></tr><tr><td>FED2Port (our)</td><td>0.2084</td><td>0.1778</td><td>1.1722</td></tr></table>

Table 7. Results of the BND&KOSPI portfolio. Cells with a red background color indicate the best Sharpe ratio in the experiment.   

<table><tr><td>Model</td><td>Profit (Higher the Better)</td><td>Risk (Lower the Better)</td><td>Sharpe Ratio (Higher the Better)</td></tr><tr><td>100% low-risk asset portfolio</td><td>0.0904</td><td>0.1429</td><td>0.6322</td></tr><tr><td>Equally Weighted</td><td>0.0990</td><td>0.2873</td><td>0.3447</td></tr><tr><td>100% high-risk asset portfolio</td><td>0.1903</td><td>0.5564</td><td>0.3420</td></tr><tr><td>Tangency portfolio</td><td>0.2562</td><td>0.4268</td><td>0.6002</td></tr><tr><td>Risk Budgeting</td><td>0.1232</td><td>0.1781</td><td>0.6917</td></tr><tr><td>RRL</td><td>-0.1851</td><td>0.2737</td><td>-0.6765</td></tr><tr><td>DDPG</td><td>0.2909</td><td>0.3223</td><td>0.9026</td></tr><tr><td>TimeGAN2Port</td><td>0.0539</td><td>0.1460</td><td>0.3690</td></tr><tr><td>RTSGAN2Port</td><td>0.0452</td><td>0.1510</td><td>0.2995</td></tr><tr><td>FED2Port (our)</td><td>0.2510</td><td>0.1845</td><td>1.3604</td></tr></table>

Table 8. Results of the BSV&SP500 portfolio. Cells with a red background color indicate the best Sharpe ratio in the experiment.   

<table><tr><td>Model</td><td>Profit (Higher the Better)</td><td>Risk (Lower the Better)</td><td>Sharpe Ratio (Higher the Better)</td></tr><tr><td>100% low-risk asset portfolio</td><td>0.0776</td><td>0.0590</td><td>1.3158</td></tr><tr><td>Equally Weighted</td><td>0.3772</td><td>0.2633</td><td>1.4325</td></tr><tr><td>100% high-risk asset portfolio</td><td>0.7459</td><td>0.5221</td><td>1.4286</td></tr><tr><td>Tangency portfolio</td><td>0.6639</td><td>0.4328</td><td>1.5342</td></tr><tr><td>Risk Budgeting</td><td>0.1548</td><td>0.1545</td><td>1.0019</td></tr><tr><td>RRL</td><td>0.1337</td><td>0.2343</td><td>0.5704</td></tr><tr><td>DDPG</td><td>0.1307</td><td>0.2737</td><td>0.4775</td></tr><tr><td>TimeGAN2Port</td><td>0.0825</td><td>0.0592</td><td>1.3931</td></tr><tr><td>RTSGAN2Port</td><td>0.0780</td><td>0.0602</td><td>1.2958</td></tr><tr><td>FED2Port (our)</td><td>0.3964</td><td>0.1562</td><td>2.5377</td></tr></table>

Table 9. Results of the BSV&DAX portfolio. Cells with a red background color indicate the best Sharpe ratio in the experiment.   

<table><tr><td>Model</td><td>Profit (Higher the Better)</td><td>Risk (Lower the Better)</td><td>Sharpe Ratio (Higher the Better)</td></tr><tr><td>100% low-risk asset portfolio</td><td>0.0776</td><td>0.0590</td><td>1.3158</td></tr><tr><td>Equally Weighted</td><td>0.1948</td><td>0.3070</td><td>0.6346</td></tr><tr><td>100% high-risk asset portfolio</td><td>0.4074</td><td>0.6125</td><td>0.6652</td></tr><tr><td>Tangency portfolio</td><td>0.3822</td><td>0.5053</td><td>0.7564</td></tr><tr><td>Risk Budgeting</td><td>0.0782</td><td>0.0947</td><td>0.8264</td></tr><tr><td>RRL</td><td>0.0421</td><td>0.1884</td><td>0.2235</td></tr><tr><td>DDPG</td><td>0.1343</td><td>0.2874</td><td>0.4675</td></tr><tr><td>TimeGAN2Port</td><td>0.2224</td><td>0.4972</td><td>0.4473</td></tr><tr><td>RTSGAN2Port</td><td>0.1487</td><td>0.4921</td><td>0.3021</td></tr><tr><td>FED2Port (our)</td><td>0.1997</td><td>0.1296</td><td>1.5406</td></tr></table>

Table 10. Results of the BSV&KOSPI portfolio. Cells with a red background color indicate the best Sharpe ratio in the experiment.   

<table><tr><td>Model</td><td>Profit (Higher the Better)</td><td>Risk (Lower the Better)</td><td>Sharpe Ratio (Higher the Better)</td></tr><tr><td>100% low-risk asset portfolio</td><td>0.0776</td><td>0.0590</td><td>1.3158</td></tr><tr><td>Equally Weighted</td><td>0.0956</td><td>0.2827</td><td>0.3381</td></tr><tr><td>100% high-risk asset portfolio</td><td>0.1903</td><td>0.5564</td><td>0.3420</td></tr><tr><td>Tangency portfolio</td><td>0.2746</td><td>0.4441</td><td>0.6183</td></tr><tr><td>Risk Budgeting</td><td>0.0835</td><td>0.0715</td><td>1.1677</td></tr><tr><td>RRL</td><td>0.0961</td><td>0.0696</td><td>1.3822</td></tr><tr><td>DDPG</td><td>0.3463</td><td>0.2891</td><td>1.1978</td></tr><tr><td>TimeGAN2Port</td><td>0.0451</td><td>0.0637</td><td>0.7075</td></tr><tr><td>RTSGAN2Port</td><td>0.0407</td><td>0.0651</td><td>0.6245</td></tr><tr><td>FED2Port (our)</td><td>0.2610</td><td>0.1446</td><td>1.8056</td></tr></table>

Table 11. Results of the VCIT&SP500 portfolio. Cells with a red background color indicate the best Sharpe ratio in the experiment.   

<table><tr><td>Model</td><td>Profit (Higher the Better)</td><td>Risk (Lower the Better)</td><td>Sharpe Ratio (Higher the Better)</td></tr><tr><td>100% low-risk asset portfolio</td><td>0.1660</td><td>0.1702</td><td>0.9750</td></tr><tr><td>Equally Weighted</td><td>0.4231</td><td>0.2922</td><td>1.4480</td></tr><tr><td>100% high-risk asset portfolio</td><td>0.7459</td><td>0.5221</td><td>1.4286</td></tr><tr><td>Tangency portfolio</td><td>0.5802</td><td>0.3769</td><td>1.5396</td></tr><tr><td>Risk Budgeting</td><td>0.3235</td><td>0.2429</td><td>1.3319</td></tr><tr><td>RRL</td><td>0.4325</td><td>0.2765</td><td>1.5642</td></tr><tr><td>DDPG</td><td>0.1164</td><td>0.3092</td><td>0.3766</td></tr><tr><td>TimeGAN2Port</td><td>0.4754</td><td>0.2835</td><td>1.6765</td></tr><tr><td>RTSGAN2Port</td><td>0.3242</td><td>0.3162</td><td>1.0252</td></tr><tr><td>FED2Port (our)</td><td>0.4941</td><td>0.2167</td><td>2.2800</td></tr></table>

Table 12. Results of the VCIT&DAX portfolio. Cells with a red background color indicate the best Sharpe ratio in the experiment.   

<table><tr><td>Model</td><td> Prigher th Beter)</td><td> Riswer the Better)</td><td>Shighe Rti Beter)</td></tr><tr><td>100% low-risk asset portfolio</td><td>0.1660</td><td>0.1702</td><td>0.9750</td></tr><tr><td>Equally Weighted</td><td>0.2389</td><td>0.3265</td><td>0.7317</td></tr><tr><td>100% high-risk asset portfolio</td><td>0.4074</td><td>0.6125</td><td>0.6652</td></tr><tr><td>Tangency portfolio</td><td>0.4429</td><td>0.4696</td><td>0.9431</td></tr><tr><td>Risk Budgeting</td><td>0.1447</td><td>0.2078</td><td>0.6964</td></tr><tr><td>RRL</td><td>0.4450</td><td>0.2473</td><td>1.7990</td></tr><tr><td>DDPG</td><td>0.3058</td><td>0.3202</td><td>0.9551</td></tr><tr><td>TimeGAN2Port</td><td>0.1617</td><td>0.1700</td><td>0.9510</td></tr><tr><td>RTSGAN2Port</td><td>0.1779</td><td>0.2882</td><td>0.6173</td></tr><tr><td>FED2Port (our)</td><td>0.5214</td><td>0.2401</td><td>2.1714</td></tr></table>

Table 13. Results of the VCIT&KOSPI portfolio. Cells with a red background color indicate the best Sharpe ratio in the experiment.   

<table><tr><td>Model</td><td>Profit (Higher the Better)</td><td>Risk (Lower the Better)</td><td>Sharpe Ratio (Higher the Better)</td></tr><tr><td>100% low-risk asset portfolio</td><td>0.1660</td><td>0.1702</td><td>0.9750</td></tr><tr><td>Equally Weighted</td><td>0.1374</td><td>0.2962</td><td>0.4637</td></tr><tr><td>100% high-risk asset portfolio</td><td>0.1903</td><td>0.5564</td><td>0.3420</td></tr><tr><td>Tangency portfolio</td><td>0.3115</td><td>0.4115</td><td>0.7570</td></tr><tr><td>Risk Budgeting</td><td>0.1778</td><td>0.1962</td><td>0.9065</td></tr><tr><td>RRL</td><td>0.0478</td><td>0.1767</td><td>0.2706</td></tr><tr><td>DDPG</td><td>0.1967</td><td>0.3161</td><td>0.6223</td></tr><tr><td>TimeGAN2Port</td><td>0.1305</td><td>0.1729</td><td>0.7545</td></tr><tr><td>RTSGAN2Port</td><td>0.0355</td><td>0.2280</td><td>0.1556</td></tr><tr><td>FED2Port (our)</td><td>0.3683</td><td>0.2044</td><td>1.8021</td></tr></table>

FED was compared with the most recent time series data generation models, namely TimeGAN [16] and RTSGAN [17]. TimeGAN and RTSGAN are designed to generate synthetic data that closely resembles real-world time series data. However, neither of these models addresses the generation of nonstationary financial time series data. FED leverages decomposition techniques to break down financial time series data into distinct components, such as trend, dispersion, and residual. By decomposing the data in this manner, FED can capture the various underlying factors influencing the trends and fluctuations in the market, leading to a more accurate representation of real-world financial time series data. The t-SNE plots of original versus generated data were plotted in Figure 6. The results indicated that FED produces synthetic data that closely match the original distribution of the data, suggesting that FED is more effective in capturing the underlying structure and characteristics of financial time series data compared to other models.

![](images/e5433539736333267bcf3a8126001909a83e525b59bb4a15b717401c0fd2ea4a.jpg)  
Figure 6. Cont.

![](images/04257dd7e6e92131f055f04f837617ac8658e03dd50ee8702ef7825798d40d7e.jpg)  
Figure 6. t-SNE plots for original versus generated data. (a) Financial time series decompositionbased variational encoder-decoder (FED). (b) Time-series generative adversarial net (TimeGAN). (c) Real-world time series GAN (RTSGAN).

# 5. Conclusions

This paper introduced a novel portfolio diversification approach called FED2Port, which effectively addresses the uncertainty deficiency problem inherent in historical financial time series data and insufficient training data. This is achieved by utilizing dynamic financial market environment simulation during reinforcement learning algorithm training. Our experimental results across diverse datasets have demonstrated the robustness and superior performance of FED2Port compared to benchmark models, including traditional and reinforcement learning models. Notably, FED2Port consistently outperformed in terms of the Sharpe ratio, emphasizing its effectiveness in delivering risk-adjusted returns. This superior performance underscores the importance of environment simulation in enhancing portfolio diversification strategies, as it allows for a more accurate representation of real-world conditions.

However, it is important to note that the experimental results for TimeGAN2Port and RTSGAN2Port were not as favorable as those of the other benchmarks. This highlights the limitations of solely relying on synthetic data generation methods that do not specifically address the complexities of financial markets. Our findings suggest the necessity of employing financial pattern-centric data augmentation techniques, such as FED, to enhance portfolio diversification strategies. By providing more accurate insights into market trends and fluctuations, FED2Port enables investors to make informed decisions that can potentially enhance portfolio performance and mitigate risks.

Overall, our findings highlight the practical importance of incorporating sophisticated data augmentation techniques, like FED, into portfolio diversification. Moving forward, further research in this area could explore additional applications of FED and similar methods in portfolio optimization and risk management, ultimately contributing to more robust and effective investment strategies in financial markets.

Author Contributions: Methodology, B.K.; Writing—original draft preparation, B.K.; Supervision, J.-H.L. and K.-T.N. All authors have read and agreed to the published version of the manuscript.

Funding: This research received no external funding.

Data Availability Statement: The data download sites referenced in this article are available within the text.

Conflicts of Interest: The authors declare no conflict of interest.

# References

1. Pensions at a Glance 2021: OECD and G20 Indicators. Available online: https://www.oecd-ilibrary.org/finance-and-investment/ pensions-at-a-glance-2021_ca401ebd-en (accessed on 22 January 2024).   
2. Asset Allocation of Pension Funds. Available online: https://www.monash.edu/__data/assets/pdf_file/0003/2357238 /Research-1-Asset-allocation-of-pension-funds.pdf (accessed on 22 January 2024).   
3. Markowitz, H. Portfolio Selection. J. Financ. 1952, 7, 77–91.   
4. Roncalli, T. Introduction to Risk Parity and Budgeting. arXiv 2014, arXiv:1403.1889.   
5. Richard, J.E.; Roncalli, T. Constrained Risk Budgeting Portfolios: Theory, Algorithms, Applications & Puzzles. arXiv 2019, arXiv:1902.05710.   
6. Moody, J.; Wu, L.; Liao, Y.; Saffell, M. Performance Functions and Reinforcement Learning for Trading Systems and Portfolios. J. Forecast. 1998, 17, 441–470. [CrossRef]   
7. Li, L. Financial Trading with Feature Preprocessing and Recurrent Reinforcement Learning. arXiv 2021, arXiv:2109.05283.   
8. Liu, X.; Xiong, Z.; Zhong, S.; Yang, H.; Walid, A. Practical Deep Reinforcement Learning Approach for Stock Trading. arXiv 2018, arXiv.1811.07522.   
9. Kalina, B.; Lee, J.; Song, J. A Study on Portfolio Asset Allocation Using Actor-Critic Model. In Proceedings of the Korea Information Processing Society Conference, Online, 29–30 May 2020; pp. 439–441.   
10. Almahdi, S.; Yang, S.Y. An adaptive portfolio trading system: A risk-return portfolio optimization using recurrent reinforcement learning with expected maximum drawdown. Expert Syst. Appl. 2017, 87, 267–279. [CrossRef]   
11. Pendharker, P.C.; Cusatis, P. Trading financial indices with reinforcement learning agents. Expert Syst. Appl. 2018, 102, 1–13. [CrossRef]   
12. Mirza, M.; Osindero, S. Conditional Generative Adversarial Nets. arXiv 2014, arXiv:1411.1784.   
13. Goodfellow, I.J.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; Bengio, Y. Generative Adversarial Nets. In Proceedings of the 27th Conference on Neural Information Processing Systems, Montréal, QC, Canada, 8–13 December 2014; pp. 2672–2680.   
14. Kingma, D.P.; Welling, M. Auto-Encoding Variational Bayes. arXiv 2013, arXiv:1312.6114.   
15. Kingma, D.P.; Welling, M. An Introduction to Variational Autoencoders. arXiv 2019, arXiv:1906.02691.   
16. Yoon, J.; Jarrett, D.; Schaar, M.v. Time-series Generative Adversarial Networks. In Proceedings of the 33rd Conference on Neural Information Processing Systems, Vancouver, BC, Canada, 8–14 December 2019; pp. 5508–5518.   
17. Pei, H.; Ren, K.; Yang, Y.; Liu, C.; Qin, T.; Li, D. Towards Generating Real-World Time Series Data. In Proceedings of the 2021 IEEE International Conference on Data Mining (ICDM), Auckland, New Zealand, 7–10 December 2021; pp. 469–478.   
18. West, M. Time Series Decomposition. Biometrika 1997, 84, 489–494. [CrossRef]   
19. Wen, Q.; Gao, J.; Song, X.; Sun, L.; Xu, H.; Zhu, S. RobustSTL: A Robust Seasonal-Trend Decomposition Algorithm for Long Time Series. In Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence, Honolulu, HI, USA, 27 January–1 February 2019; pp. 5409–5416.   
20. Patidar, S.; Jenkins, D.P.; Peacock, A.; McCallum, P. Time Series Decomposition Approach for Simulating Electricity Demand Profile. In Proceedings of the 16th IBPSA Conference, Rome, Italy, 2–4 September 2019; pp. 1388–1395.   
21. Wen, Q.; Zhang, Z.; Li, Y.; Sun, L. Fast RobustSTL: Efficient and Robust Seasonal-Trend Decomposition for Time Series with Complex Patterns. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Online, 6–10 July 2020; pp. 2203–2213.   
22. Hyndman, R.J.; Athanasopoulos, G. Time series decomposition. In Forecasting: Principles and Practice, 3rd ed.; OTexts: Melbourne, Australia, 2021; Chapter 3.   
23. Dokumentov, A.; Hyndman, R.J. STR: Seasonal-Trend Decomposition Using Regression. INFORMS J. Data Sci. 2021, 1, 50–62. [CrossRef]   
24. Mishra, A.; Sriharsha, R.; Zhong, S. OnlineSTL: Scaling Time Series Decomposition by 100x. arXiv 2021, arXiv:2107.09110.   
25. Jiang, S.; Syed, T.; Zhu, X.; Levy, J.; Aronchik, B. Bridging Self-Attention and Time Series Decomposition for Periodic Forecasting. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, Atlanta, GA, USA, 17–21 October 2022; pp. 3202–3211.   
26. Dudek, G. STD: A Seasonal-Trend-Dispersion Decomposition of Time Series. IEEE Trans. Knowl. Data Eng. 2023, 35, 10339–10350. [CrossRef]   
27. Sharpe, W.F. Mutual Fund Performance. J. Bus. 1966, 39, 119–138. [CrossRef]   
28. Black, F.; Litterman, R. Global Portfolio Optimization. Financ. Anal. J. 1992, 48, 28–43. [CrossRef]   
29. Sharpe, W.F. Capital asset prices: A theory of market equilibrium under conditions of risk. J. Financ. 1964, 19, 425–442.   
30. Lillicrap, T.P.; Hunt, J.J.; Pritzel, A.; Heess, N.; Erez, T.; Tassa, Y.; Silver, D.; Wierstra, D. Continuous control with deep reinforcement learning. arXiv 2015, arXiv:1509.02971.   
31. Sortino, F.A.; Price, L.N. Performance measurement in a downside risk framework. J. Investig. 1994, 3, 59–64. [CrossRef]   
32. Lee, J.H.; Kalina, B.; Na, K. Market-Adaptive Ratio for Portfolio Management. arXiv 2023, arXiv:2312.13719.   
33. Peterson, K.B.; Pedersen, M.S. 8.1.8 Product of gaussian densities. In The Matrix Cookbook; Technical University of Denmark: Lyngby, Denmark, 2012.

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.

Copyright of Symmetry (20738994) is the property of MDPI and its content may not be copied or emailed to multiple sites or posted to a listserv without the copyright holder's express written permission. However, users may print, download, or email articles for individual use.