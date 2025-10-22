# Adversarial Learning Networks for FinTech Applications Using Heterogeneous Data Sources

Parus Khuwaja , Sunder Ali Khowaja , and Kapal Dev , Member, IEEE

Abstract—The dynamic property and increasing complexity are the key challenges for modeling financial technology (FinTech)-related applications such as stock markets. Over the years, a lot of inflexible predictive strategies have been proposed for predicting stock price movements that failed to achieve satisfactory results especially when a market crash occurs. To cope with this challenge, we propose a prediction framework based on an adversarial training strategy using reinforcement learning for the said FinTech application. The framework uses a heterogeneous knowledge base, including stock prices, tweets, and global indicators. We propose a modified newton-divided difference polynomial (NDDP) for missing data imputation. The informative patterns representing the intrinsic characteristics of financial markets were extracted using long short-term memory networks (LSTM). The two adversarial networks are heterogeneous data fusion representing market crash (HDFM) $\varrho$ -learning and confrontational $\varrho$ -learning network. Both networks are trained in an adversarial fashion to increase the effectiveness of prediction even when the financial market is volatile. The experimental results show the importance of global indicators and the proposed adversarial learning network (ALN) for improving the predictive performance in comparison with the existing state-of-the-art works.

Index Terms—Actor–critic framework, financial technology (FinTech), heterogeneous data, reinforcement learning, stock price prediction.

# I. INTRODUCTION

N RECENT years, the domain of financial technology I (FinTech) services focusing on predictive applications has garnered a lot of attention from researchers and investors alike [1]. Stock price prediction is one of the FinTech services that faces many challenges due to its changing regularity and high volatility [2]. The recent market crash due to the ongoing pandemic is an evidence of such challenge. The cause of volatility in stock prices depends on multiple factors that include financial markets, society, economics, politics, and global indicators. The stock investors are highly interested in stock trend prediction to seek the maximum profit while reducing the intensity of associated risk. Moreover, the prediction also affects the listed stock companies as the trend is directly related to a company’s reputation, future development expectations, and operating conditions [3].

Traditionally, investors used to predict the stock market trend based on their intuition. This prediction technique along with investors’ unreasonable perceptions has been proven to be ineffective, thus causing major financial losses. With the advancement in information management techniques, the focus has been shifted to quantitative trading analysis that considers historical price data and statistical algorithms to determine the market trend [4]. The quantitative trading showed better results in comparison to the typical trading behaviors; however, the use of statistical techniques with only historical price data failed to realize the maximum profit. The researchers considered to extract short-term movement features from openhigh-low-close (OHLC) values along with shallow machine learning approaches to improve predictive performance. The recorded results were better but still lacked to model the financial market well enough. It was realized that the historical price data are not enough to make a good stock market prediction model; hence, heterogeneous data, such as news, tweets, and sentiment scores, should also be used with the OHLC values [2]. Considering the importance of sentiments reflected through online social media platforms, researchers leveraged sentiment analysis to predict the stock market trend. There are two problems associated with combining the data from heterogeneous sources. The first is the data aggregation that needs the data modalities to be homogeneous and has a similar sampling rate. The second is the imputation of missing values in the data. Existing studies do propose preprocessing methods to deal with the said issues but a generalized and unified framework has not been proposed yet.

As stated, shallow machine learning techniques were used for predicting stock prices previously; however, the said learning method heavily relied on feature engineering techniques. Moreover, the shallow learning approaches have greater time complexity in the testing phase with weak generalization capability [5]. Recently, deep learning approaches prevailed in the field of stock price prediction due to their ability to model nonlinear behavior, handle free text data, such as text and tweets, and to reduce the time complexity at the testing phase. Mainly, two families of deep learning architectures are used for such studies, i.e., convolutional neural networks (CNNs) and recurrent neural networks (RNNs). Due to the temporal characteristics of stock prices, RNNs are well suited and have been used extensively for predicting the market trend. The problem with RNN is that the model stops learning with the passage of time as the gradient decays exponentially, i.e., vanishing gradient. To deal with it, two variants of RNN, i.e., long-shortterm memory (LSTM) and gated recurrent units (GRUs), were proposed. The predictive performance of LSTMs and GRUs yields improved results but still underperforms when there is a market crash. We assume that the reason behind such a phenomenon is twofold. The first is the volume of data that is quite high for the stable market situation in comparison to the state representing market crashes. The second is the nonconsideration of global indicators within the learning framework that can provide a context to a market crash.

Another emergent learning approach is reinforcement learning in which the agent is trained to collect maximum rewards based on the selected action/decision. Similar to the CNN and RNN, reinforcement learning has also variants that are used for diverse applications [6], [7]. Some studies used reinforcement learning for stock price prediction but the parameters are optimized using only OHLC values or the extracted features rather than considering a knowledge-base of heterogeneous sources. Moreover, the problem regarding the high volume of data for the stable market situation persists in reinforcement learning-based approaches as well.

In this study, we propose an adversarial learning network (ALN) framework for stock price prediction that solves the aforementioned problems. We first develop our knowledge base by using heterogeneous modalities, such as OHLC, adjusted close, volume, tweets, and global indicators. To the best of our knowledge, the use of global indicators has not been considered for a unified knowledge base to predict the stock price movement. The ALN framework also uses a generalized way to preprocess the said data and fill out the missing values using a modified Newton’s divided difference polynomial (NDDP) method. Existing studies have proved that deep learning architectures such as LSTMs can be used with reinforcement learning. It has also been suggested in existing studies that LSTMs yield better results than GRUs; therefore, we use LSTMs to extract meaningful patterns from our knowledge base [8]. One of the drawbacks of previous approaches was the consideration of a high volume of data representing the stable market situation; in this regard, we first apply the market crash detection method [9] to extract a close range of values representing unstable market and train our critic model on the extracted data while the actor–critic model will be trained in an adversarial manner to correct the former one. We term the two networks as heterogeneous data fusion representing market crash (HDFM) $Q$ -learning and confrontational $Q$ -learning networks, respectively. To the best of our knowledge, such a predictive strategy involving market crash data within the learning framework has not been proposed before. The contributions of the proposed work are summarized as follows.

1) We propose a unified way to construct a knowledge base including the global indicators for stock movement prediction.   
2) We propose a modified Newton divided difference polynomial method for missing value imputation.

3) We propose HDFM $Q$ -learning and confrontational $Q$ -learning networks to train in an adversarial manner.

The remainder of this article is structured as follows. Section II consolidates a review of the existing works. Section III presents the working mechanism of the proposed ALN framework. Section IV provides experimental results along with a comparative analysis with some existing works. Section V concludes the study along with the prospective future works.

# II. RELATED WORK

Stock prediction has fetched a lot of attention from the research community due to its decisive role in financial investments. The efficient market hypothesis theory explicitly states that outperforming the market is impossible as the market price reflects all information; however, researchers firmly believe that the market price may deviate from a fair share price and that beating the market through predictive modeling is still a possibility [3]. The stock price prediction methods can be categorized as either technical or fundamental analysis. The former considers the historical price data and volume to predict future prices. The study [10] combined neural networks with decision trees to predict stock price movement. Khuwaja et al. [11] used extreme learning machines and phase space reconstruction method to predict stock price movement. The aforementioned studies only consider the features from OHLC values that are linear in nature.

In recent years, an explosion of information has been observed in the form of online content, such as tweets, news, and so forth. The price prediction using such sources can be categorized as a fundamental analysis. The researchers used the aforementioned sources extensively in order to predict the price movement. The study [12] proposed the “semanticssentiment” method, which is a sentiment-weighting mechanism with multilayer perceptrons to forecast intraday stock movement. Li et al. [13] proposed a generic framework for the prediction of price movement using sentiment analysis. Similarly, Zhou et al. [14] acquired millions of tweets from Weibo to predict the movement of the Chinese market.

A new way of dealing with the market prediction problem has emerged with the advent of deep learning and reinforcement learning techniques. The study [15] used neural networks for feature extraction and price prediction of the Japanese stock index. Long et al. [16] proposed a multifilters neural network to extract the features and predict daily prices. Hu et al. [17] proposed hybrid attention networks, which use OHLC values along with sentiment analysis to predict the stock prices. Xu et al. [2] proposed an incorporative attention mechanism for price prediction using OHLC values and tweets. Attention networks mostly construct abstract representations but neglect the semantics that are available in the form of global indicators.

Some researchers focused on developing an autonomous trading strategy using reinforcement learning rather than discriminative learning. Carapuço et al. [18] used a reinforcement learning for short-term speculation of the foreign exchange market. Wu et al. [19] used the reinforcement learning strategy in an adversarial manner considering GRU as a feature extractor for price prediction. The aforementioned studies limit the aforementioned method to train on homogeneous data sources; thus, the use of a knowledge base with heterogeneous data has not been exploited concerning the reinforcement learning technique.

# III. PROPOSED METHODOLOGY

# A. Task Description

This study aims to predict the stock price movement for a preselected index on a trading day $d$ . The knowledge base in this study comprises of the historical prices, tweet comments, and global indicator values. The price prediction problem is mostly regarded as time series due to its temporal dependencies. Therefore, the selection of time lag is quite important, not only to fulfill the predictive dependency but to match the sampling rate of heterogeneous modalities, such as tweets and global indicator values. The lag interval is considered to be of fixed size in this study, i.e., $\Delta d$ ; therefore, the interval can be represented as $[ d - \Delta d , d - 1 ]$ . Furthermore, spontaneous news or tweets regarding financial fraud or trading violations published on the day $d _ { 1 }$ will not instantly affect the stock index but the drop could be observed within the interval $[ d _ { 1 } , d _ { 2 } ]$ . This study estimates the binary movement of the stock market, i.e., 0 for the decline and 1 for the upward trend. The aforementioned phenomenon is formally represented in

$$
y = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { ~ i f ~ } } p r _ { d } ^ { a c } > p r _ { d - 1 } ^ { a c } } \\ { 0 , } & { { \mathrm { ~ o t h e r w i s e } } } \end{array} } \right.
$$

where $p r _ { d } ^ { a c }$ refers to the closing price adjusted with respect to the splits and dividends. In reinforcement learning, the agent interacts with the environment to learn the strategy that is optimal for prediction. Therefore, instead of extracting technical indicators, we input the knowledge base to the LSTMs and extract the features using the reinforcement learning method to make predictions related to price movement.

# B. Data Preparation

To show the effectiveness of the proposed approach as well as perform a fair comparison with existing studies, we used the stocknet-data set [20] for stock price prediction. As some of the previous studies showed their novel advances using the stocknet-data set, therefore, it can be considered as a benchmark for price prediction and is well suited for a general comparison of the predictive performance. The stocknet-data set amalgamates the data from 88 stocks from 01/01/2014 to 01/01/2016. We used the same distribution of training, validation, and testing data as proposed in [2]. The main problem in terms of data gathering was to acquire the global indicator values; therefore, we used BeautifulSoup and selenium Python packages for scraping the values of global indicators from multiple but reliable websites within the period of 01/01/2014 to 01/01/2016, accordingly.

# C. ALN Framework

The proposed ALN framework for price movement prediction is shown in Fig. 1. The figure shows an abstract pipeline where each block comprises subprocesses to perform the desired task. The data considered by the ALN framework include OHLC, adjusted close, volume, tweets, and global indicator values, accordingly. Together, these three modalities make the knowledge base that undergoes a preprocessing stage. The heterogeneous data fusion block in the ALN framework consists of denoising, sentiment analysis, sampling rate adjustment, and interpolation of missing values. The information from heterogeneous modalities is transformed into homogeneous for data aggregation. This work uses an ALN, therefore, a critic network is needed, which could provide feedback to the actor network for optimizing its decision. In this regard, we use market crash detection from the knowledge base to extract the affected days of data and use it as a critic network. We use topological data analysis [9] for detecting the market crash. The aforementioned method was selected due to its superior performance in comparison with existing studies. The aggregated data from both streams will undergo LSTMs for the extraction of informative patterns. It is highly assumed that the use of market crash detection block could also improve the predictive performance in current pandemic situation as well. These patterns will be the input to the reinforcement learning block to predict the stock price movement. We provide the details of each of the blocks in the subsequent sections, accordingly.

![](images/3c0f9a2b18b01943a97345d1ca5931f0a3f0c93d081886b43a4d39dcba4ad9fd.jpg)  
Fig. 1. Proposed ALN for FinTech application such as stock price prediction using software IoT sensors.

# D. Knowledge Base

The knowledge base in the proposed study considers three heterogeneous sources, i.e., stock values, tweets, and global indicators. The reason we call the sources heterogeneous is that these sources differ in terms of their format, such as JSON, CSV, and XML or text, and the sampling rate. For instance, the stock price data can be acquired on an hourly basis, whereas the tweets might be acquired on daily basis and the global indicator data are available on a monthly basis. The data employed in our study provide the values for the stock price and tweets, respectively. This study also considers the global indicators listed in the Bloomberg study [21] suggesting that these indicators affect the global economy the most. The global indicators include the balance of trade, business confidence, consumer confidence, GDP growth rate, unemployment rate, inflation rate, and purchasing manager’s index. To collect the data for the said indicators with consistent frequency, we used multiple APIs and sources to develop our knowledge base. The Web sources used to collect the indicator data include Tradingeconomics.com, IMF.org, TheGlobalEconomy.com, and Worldbank.org. For the APIs, BeautifulSoup package was used, accordingly. The packages scrapped the Web sources in the form of HTML and XML parsers, from which we extracted the numerical values, accordingly.

![](images/ec1f9946e4d0cc06d9f07aadf233e7fc62924bead5ebeed2317e586607de4703.jpg)  
Fig. 2. Illustration of heterogeneous data fusion block in ALN framework.

# E. Heterogeneous Data Fusion

The process employed in the heterogeneous data fusion is shown in Fig. 2. There are specifically four operations: 1) denoising; 2) sentiment analysis; 3) sampling rate adjustment; and 4) interpolation. Stock data are considered noisy in its raw form; therefore, existing studies have suggested to use denoising methods to smooth the trend. This study uses Haar wavelet transform [8] to denoise the stock price data.

In order to make the data homogeneous, we used the natural language processing approach to derive sentiment scores from tweets. We use the TextBlob API for extracting the sentiment scores from tweets. We assume that the direct scores would allow the learning algorithm to learn and extract meaningful patterns. As there are multiple tweets for each day, therefore, we average the sentiment scores and use a single value for further aggregation.

The sampling rate of global indicators data is at least once a month, which makes it difficult to aggregate with the other two data sources. Furthermore, duplicating the same value for all days in a month will not make much sense either. In this regard, we apply the sampling rate adjustment to global indicator values, suggesting that the data will be populated from a single value to match the sampling rate of stock data and the sentiment scores. We populate the data through random numbers assuming that the numbers represent a Gaussian distribution. The data population is performed using the formulation shown in

$$
\mathrm { I v a l } = \mu + \sigma * \mathrm { r a n d } .
$$

In the equation above, Ival represents the populated global indicator value where $\mu$ represents the actual value of the indicator, $\sigma$ represents the difference between the current and subsequent month’s indicator value, i.e., $| \mu _ { \mathrm { j a n } } - \mu _ { \mathrm { f e b } } |$ , and the rand refers to the random number generated between 0.5 and $- 0 . 5$ , accordingly. These preprocessing blocks will transform the heterogeneous data to homogeneous for the data aggregation process. However, to increase the data quality, we need to fill in the missing values. Many of the existing studies replace the missing values with zero, mean, or mode values. Another way of filling the missing values is the interpolation technique. One popular method for data interpolation is NDDP. The NDDP method has been mostly used in physics-based studies but considering the simplicity and reliability of the method, the researchers also considered it for data imputation [22]. We modified it so that it could be used for the desired application. The formulation for the modified NDDP is shown in Lemma 1 (supplementary material) for filling the missing sentiment scores.

# F. Market Crash Detection

The market crash detection component is of vital importance to the ALN framework as it allows the networks to be trained in an adversarial manner. We use the market crash detection method proposed in [9] with some slight modifications to extract the data for some specific ranges where the trend has a kind of a derivative. The pipeline for the market crash detection used in this study involves: 1) acquiring the data from the HDF block; 2) extracting the embeddings from the acquired data and performing point cloud transformation through sliding windows; 3) constructing a geometrical shape of the evolving structure from each window; 4) extracting features using persistence homology; 5) comparing features extracted from different windows in Euclidean space; and 6) constructing topological indicator for crash detection based on the Euclidean distance.

Based on the detected market crashes, we vary the threshold between $20 \%$ and $3 5 \%$ so that the values for the days above the threshold could be used to train the critic network, accordingly.

# G. LSTM Block

The HDF block combines the data related to prices and indicators that affect the stock price movement; however, the raw values do not yield better predictive performance as they cannot model the market’s evolving state. We, in this study, use the LSTMs for the pattern extraction from both the HDF and the data from the market crash detection block as shown in Fig. 3. The LSTMs use forgets and cell gate to control the memory as well as the flow of the information. The forget gate is responsible for determining which information could be ignored related to the stock market whereas the cell gate determines how much information should flow from the previous state to the current state. The formulation for the input gate is shown in

$$
\begin{array} { r } { i p _ { t } = \mathrm { s i g } \big ( W _ { i p } \big [ H D _ { t } , h _ { t - 1 } \big ] + b _ { i p } \big ) . } \end{array}
$$

$H D _ { t }$ is regarded as the input that can form the $\mathrm { H D F } _ { t }$ block or HDF followed by the market crash detection block $H D F M _ { t }$ , respectively. The variable $h _ { t - 1 }$ corresponds to the hidden state

![](images/3f781d0e818607bebf6eea67607ff3d3458b286582af593bbf7317ca83792453.jpg)  
Fig. 3. Illustration of LSTM block as a feature extractor.

value from the previous time step, $W _ { i p }$ refers to the weight, and $b _ { i p }$ represents bias term. sig refers to the sigmoid activation function. The formulation for the forget gate is shown in

$$
f g _ { t } = \mathrm { s i g } \big ( W _ { f g } \big [ H D _ { t } , h _ { t - 1 } \big ] + b _ { f g } \big ) .
$$

The formulations for the previous and candidate cell state along with the output gate are shown in

$$
\begin{array} { l } { \hat { c l } _ { t } = \operatorname { t a n h } \big ( W _ { \hat { c l } } [ H D _ { t } , h _ { t - 1 } ] + b _ { \hat { c l } } \big ) } \\ { c l _ { t } = H D _ { t } . \hat { c l } _ { t } + f g _ { t } . c l _ { t - 1 } } \\ { o p _ { t } = \operatorname { s i g } \big ( W _ { o p } \big [ H D _ { t } , h _ { t - 1 } \big ] + b _ { o p } \big ) . } \end{array}
$$

# H. Reinforcement Learning Block

This block is at the core of the proposed ALN as it opts for an actor–critic training strategy to determine the movement of stock prices. The aim of the agents in the reinforcement learning block is to learn how to optimize the maximum accuracy for stock movement based on the derived policy. In this regard, the selection of rewards is of vital importance. Some of the existing studies use the rate of return and sortino ratio to define the reward function. Both ratios represent the movement of the stock price in terms of percentage by design. We adapt the Sharpe ratio [19] to define our reward, i.e., $r w = [ E ( R ) / D e \nu ( R ) ]$ , where $r w$ represents the reward, $E ( R )$ refers to the expected return for a specific episode, and $D e \nu ( R )$ represents the deviation and can be defined as |closepricet − adjustedclosepricet|. Furthermore, existing studies have also revealed that the sharpe ratio yields better results in comparison to the sortino ratio [18].

# I. HDFM Q-Learning Network

The proposed critic network is trained with a $Q$ -learning strategy as shown in Fig. 4(a). The critic network aims to achieve the correct movement prediction through the selected actions for most of the episodes while interacting with the market crash data extracted using topological data analysis. The critic network generated a series of transactions for each episode $e = \{ 1 , \ldots , E \}$ , along with their corresponding rewards $\{ r w _ { 1 } , r w _ { 2 } , \ldots , r w _ { e } \}$ . We define the expected correctness of movement prediction from market crash data using the critic

network in

$$
M C _ { e } = \sum _ { \Omega = 0 } ^ { E - 1 } r w _ { e + \Omega } . \gamma ^ { \Omega }
$$

where $\gamma$ is the discount factor and $M C _ { e }$ is the cumulative discounted reward after episode $e$ . As the expected cumulative discounted reward only relies on the current state $\Psi _ { e }$ , we can use the law of large numbers to formally estimate the expected cumulative reward with respect to the current state as shown in

$$
V l _ { \pi } ( \psi ) = E [ M C _ { e } | \Psi _ { e } = \psi ]
$$

where $V l$ refers to the value of the state and $E [ . ]$ corresponds to the expectation function. We can derive the Bellman equation by substituting the value of $M C _ { e }$ in (7) as shown in

$$
\begin{array} { l } { { \displaystyle { V l ( \psi ) = E \biggl [ \sum _ { \Omega = 0 } ^ { E - 1 } r w _ { e + \Omega } . \gamma ^ { \Omega } | \Psi _ { e } = \psi \biggr ] } } } \\ { { \displaystyle ~ = E \biggl [ r w _ { e } + \sum _ { \Omega = 1 } ^ { E - 1 } r w _ { e + \Omega } . \gamma ^ { \Omega } | \Psi _ { e } = \psi \biggr ] } } \\ { { \displaystyle ~ = E \bigl [ r w _ { e } + \gamma . V l ( \Psi _ { e + 1 } ) | \Psi _ { e } = \psi \bigr ] } . } \end{array}
$$

The formulation of $V l ( \psi )$ shown above exhibits that the value of the current state depends on the value obtained by performing an action in the next step and its corresponding reward. However, $V l$ symbolizes a single action, and given the nature of our problem, the movement prediction has more than a single action. Therefore, the use of $Q ( \psi , a )$ is more justified in comparison to the value. Where $V l$ was only dependent on the state, $Q$ depends on the expected rewards of the subsequent state when performed a certain action. The critic network is designed to optimize the rewards for $Q ( \psi , a )$ as shown in

$$
\pi ^ { * } = \arg \operatorname* { m a x } _ { \pi } Q ( \psi , a )
$$

where $\pi$ refers to the policy. Based on (9), the $Q$ value based on the state–action pair needs to be updated at each iteration, as shown in

$$
\begin{array} { r l } & { \mathcal { Q } _ { \Psi _ { e + 1 } \to \Psi _ { e } } = Q ( \psi _ { e } , a _ { e } ) } \\ & { \qquad + \left( r w _ { e } + \gamma \operatorname* { m a x } _ { a _ { e + 1 } } Q ( \psi _ { e + 1 } , a _ { e + 1 } ) - Q ( \psi _ { e } , a _ { e } ) \right) . \alpha } \end{array}
$$

Equation (18) will derive a policy based on the state–action pair and will update the $Q$ value by observing the reward, action, and the subsequent state. However, it has been shown in existing studies that using the update process shown in (18) leads to the requirement of infinite memory space. In order to overcome the aforementioned memory issue, we incorporate the experience replay memory in the learning process. The values recorded for each transaction during the training process, i.e., $( r w _ { e } , a _ { e } , \psi _ { e } , \psi _ { e + 1 } )$ would be stored in experience replay memory. Thus, the training process intuitively resembles the supervised learning process. We approximate the value of $Q$ by parameterizing the function with hyperparameter $\vartheta$ that would help to approximate the optimal value of $Q$ , as shown in

$$
Q ( \psi , a , \vartheta ) \approx Q ^ { \pi } ( \psi , a )
$$

![](images/ca9052e62acd2444089c3c8f6e747c2346a5b9e9e0a814fea589cfd9044cfcd1.jpg)  
Fig. 4. Training strategy for HDFM $Q$ -learning and confrontational $Q$ -learning networks. (a) HDFM $Q$ -learning network. (b) Confrontational $Q$ -learning network.

The loss function for the critic network is shown in

$$
{ \mathrm { l o s s } } _ { \mathrm { c r i t i c } } ( \vartheta ) = E \Big [ ( y - Q ( \psi , a , \vartheta ) ) ^ { 2 } \Big ] .
$$

# J. Confrontational $Q$ -Learning Network

The actor–critic network in the ALN framework is shown in 4(b). The network is applied to the features extracted directly from the HDF block. The formulation for selection of an action based on the market state in the actor network is shown in

$$
a = \sigma \big ( \psi _ { e } | \vartheta ^ { \sigma } \big ) + \epsilon
$$

where $\epsilon$ represents the random noise and $\sigma$ refers to the maximum probability for a particular action. The selected action will be confronted by the critic network using

$$
\begin{array} { r } { V l = Q \Big ( \psi _ { e } , a _ { e } | \vartheta ^ { Q } \Big ) . } \end{array}
$$

Based on the confrontation, the value of the selected action in episode $e$ is corrected as shown in

$$
\nabla _ { e } = Q ( \psi _ { e + 1 } , a _ { e + 1 } ) - Q ( \psi _ { e } , a _ { e } ) + r w _ { e }
$$

where $\nabla$ is the correction term in the $Q$ -value for the subsequent step. Based on the correction (suggestions from the critic network), the policy distribution of a certain action is updated accordingly. We use the mean square error as the loss function for minimizing the difference between the original and updated $Q$ value as shown in

$$
\begin{array} { r l r } { \mathrm { l o s s } _ { \mathrm { a c t o r - c r i t i c } } = \displaystyle \frac { 1 } { E } \sum _ { \Omega = 1 } ^ { E } \Big ( y _ { \Omega } - Q \Big ( \psi _ { e } , a _ { e } | \vartheta ^ { Q } \Big ) \Big ) ^ { 2 } } & { } & \\ { y _ { \Omega } = Q \Big ( \psi _ { e + 1 } , \sigma \big ( \psi _ { e + 1 } , \vartheta ^ { \sigma } \big ) | \vartheta ^ { Q } \Big ) . r w _ { \Omega } . \gamma } & { } & \end{array}
$$

The $Q$ value for the next state is computed using (16) and (17), respectively. The objective function for maximizing the policy to predict accurate movement is shown in

$$
J ( \vartheta ) = E \big [ Q ( \psi , a ) | \psi _ { e } , \gamma ( \psi _ { e } ) \big ]
$$

# IV. EXPERIMENTS AND RESULTS

In this section, we provide the information regarding baselines, metrics, network parameters, and results to show the efficacy of the proposed ALN framework for stock movement prediction. For the handling of missing data values, we evaluate our modified NDDP method on sentiment scores in terms of mean predicted error rate and compare it with well-known imputation methods, such as complete case analysis, mean imputation, and expectation–maximization imputation, respectively. In general, we opt for two evaluation metrics, i.e., accuracy and Matthews correlation coefficient (MCC). The MCC metric has been used extensively in recent years for evaluating the predictive performance as it avoids the bias from skewed data [2]. Furthermore, we opt for the baselines studies [2], [11], [16], [17], [20] in order to perform a comparative analysis of the stocknet-data set.

We also conduct an ablation study to justify the use of different blocks in the ALN framework. The variations of the ALN framework are briefly defined as follows.

ALN-Critic: It denotes the ALN framework using only the critic network, i.e., trained on market-crash detection values.

ALN-Actor: It denotes the ALN framework using only the actor network, i.e., trained on the data without using market crash detection.

ALN-Actor-wo-GF: It denotes the ALN framework using only the actor network but without using global indicators.

ALN: It denotes the ALN framework with the proposed actor–critic learning strategy.

# A. Experimental Setup and Parameter Setting

The experiments for stock price movement using the ALN framework and ablation results are carried out with Intel Corei5 clocked at $3 . 4 ~ \mathrm { G H z }$ , 32-GB RAM, and GPU GeForce GTX 1080Ti. We use the default parameters for topological data analysis to detect the market crash. We opted for two layers of LSTM each with 100 hidden units. The LSTMs were trained for 25 epochs with a learning rate of 0.001.

For the reinforcement block, we have some hyperparameters, which need to be selected empirically. Both networks, i.e., (actor and critic), had three hidden layers with 64 units each. We evaluated different activation functions, such as ReLU, SeLU, and sigmoid, and selected the ReLU function as it yields the best performance. We restricted the buffer size for experience replay to 150. The learning rate and discount rate were set to 0.001 and 0.0025, respectively. The dropout ratio was chosen to be 0.3 and the exploration rate is set to be 0.35. We also evaluated various optimization algorithms and selected ADAM due to its superior performance. The network was trained for 2000 epochs, respectively.

TABLE I COMPARATIVE ANALYSIS OF EXISTING WORKS AND MODIFIED NDDP FOR THE HANDLING OF MISSING VALUES   

<table><tr><td>Method</td><td>Mean Predicted Error</td></tr><tr><td>Complete case analysis</td><td>0.344</td></tr><tr><td>Mean Imputation</td><td>0.781</td></tr><tr><td>Expectation-Maximization Imputation</td><td>0.426</td></tr><tr><td>Modified NDDP</td><td>0.027</td></tr></table>

# B. Handling of Missing Data

The first set of experiments we conducted is to check the efficacy of the proposed NDDP method for handling the missing data. We experimented on Twitter sentiment scores and compared the complete case analysis, mean imputation, and expectation–maximization imputation, along with the proposed modified NDDP in terms of mean predicted error rate. Table I shows the results for predicting the missing value of sentiment scores in terms of mean predicted error rate using the aforementioned techniques. Complete case analysis performs better than the expectation–maximization imputation; however, the least mean predicted error was achieved using the modified NDDP, which proves the efficacy of the method at least for the opted data/application. Considering the results and the margin attained by the modified NDDP, it can be assumed that the proposed method can be considered as a missing value predictor, in general. We assume that the reason behind the margin is that the modified NDDP not only considers the data itself but also the number of values obtained for each day, which kind of integrates a weighting mechanism without explicitly assigning one.

# C. Variations for ALN-Framework

The key components and contributions of this study are the inclusion of global indicators, the HDF block, and the use of an adversarial learning strategy to improve the stock movement predictive performance. In the following set of experiments, we want to observe whether the adversarial learning strategy helps in improving the predictive accuracy and whether the global indicators have any impact on the predictive performance of the stock movement. In this regard, we evaluate ALN-critic, ALN-actor, ALN-actor-wo-GF, and ALN framework on the stocknet data set. For the sake of generality, we use the ReLU activation function along with ADAM optimizer for all variants of the ALN framework as they achieve superior performance in comparison to their contemporaries, respectively.

The results for the ALN variants for each field as well as the aggregated one are presented in Table II. The lowest accuracy out of all the variants was achieved using ALNcritic, which makes sense due to two reasons. The first is the volume of data used for ALN-critic as it only depends on the market crash data set and mostly considers the market instability. The learning in ALN-critic models unstable data without any confrontation or adversarial effect. On the other hand, ALN-actor achieves the second-best performance, which is because the network considers a larger volume. However, the best results were not achieved, which is intuitively due to the nonconsideration of the unstable market. The actor network trained without global indicators achieves better accuracy than the critic network for the same particular reason, i.e., the volume of data. Apparently, the ALN-actor-wo-GF supports our intuition as the accuracy is less than the ALN-critic. Our assumption that global indicators can help in modeling the market condition, especially the unstable market condition, stands still with the obtained results. Finally, the best performance was recorded using ALN, which incorporates the market instability along with the adversarial learning effect, which is supported by the improvement in accuracy by $5 . 5 8 \%$ and $5 . 4 \%$ margin with respect to aggregate and average accuracy, respectively. The MCC scores for all variations for the ALN-framework are given in the supplementary material.

TABLE IIPERFORMANCE COMPARISON OF ALN VARIANTS ON STOCKNET DATA  

<table><tr><td>Fields</td><td>ALN-critic</td><td>ALN-actor</td><td>ALN-actor-wo-GF</td><td>ALN</td></tr><tr><td>Utilities</td><td>62.97</td><td>74.36</td><td>71.43</td><td>78.97</td></tr><tr><td>Services</td><td>58.89</td><td>67.44</td><td>58.74</td><td>71.24</td></tr><tr><td>Healthcare</td><td>59.56</td><td>67.59</td><td>61.07</td><td>72.36</td></tr><tr><td>Consumer goods</td><td>64.36</td><td>69.82</td><td>63.14</td><td>74.48</td></tr><tr><td>Basic Materials</td><td>61.24</td><td>70.58</td><td>58.49</td><td>76.77</td></tr><tr><td>Finance</td><td>62.78</td><td>68.29</td><td>61.76</td><td>74.29</td></tr><tr><td>Industrial goods</td><td>64.12</td><td>72.43</td><td>64.29</td><td>78.92</td></tr><tr><td>Technology</td><td>59.54</td><td>65.46</td><td>59.36</td><td>72.13</td></tr><tr><td>Average</td><td>61.68</td><td>69.50</td><td>62.29</td><td>74.90</td></tr><tr><td>Aggregate</td><td>58.55</td><td>65.84</td><td>59.79</td><td>71.42</td></tr></table>

TABLE III COMPARATIVE ANALYSIS OF THE PROPOSED ALN FRAMEWORK WITH EXISTING STUDIES   

<table><tr><td>Method</td><td>MCC</td><td>Acc</td></tr><tr><td>[17]</td><td>0.052</td><td>57.68</td></tr><tr><td>[16]</td><td>0.067</td><td>57.90</td></tr><tr><td>[20]</td><td>0.081</td><td>58.23</td></tr><tr><td>[]</td><td>0.085</td><td>58.36</td></tr><tr><td>[2]</td><td>0.130</td><td>59.74</td></tr><tr><td>ALN</td><td>0.238</td><td>71.42</td></tr></table>

# D. Comparison With State-of-the-Art Methods

The motivation behind using the stocknet-data set is to perform a fair comparative analysis by adapting the same protocol for the train and test data set. For some studies, we have reproduced their work based on the information or code they shared; however, for others, we report the results as it is due to the same protocol and data set being followed. The comparison with the existing works is presented in Table III, accordingly. The comparative analysis depicts that the proposed ALN framework outperforms the existing works with a reasonable margin of $0 . 1 0 8 ~ \mathrm { M C C }$ and $1 1 . 6 8 \%$ accuracy, respectively.

The probable reasons for such gain are threefold: 1) the construction of a consistent knowledge base (HDF in this study) helps generalizing the training process; 2) the use of global indicators has shown to be effective when modeling the instability of the market; and 3) the adversarial learning using reinforcement learning strategy improves the predictive performance, in general. The accuracy of ALN-critic is very close to the results obtained from [2]; however, the MCC of ALN-critic is recorded to be better than the aforementioned one.

# V. CONCLUSION

To achieve improved predictive performance for the stock market movement, we propose the ALN framework. The proposed framework is composed of knowledge base, HDF, market crash detection, LSTM, and reinforcement learning blocks. The knowledge base acquires data from heterogeneous modalities including the global indicators, which have never been employed for such purposes in existing studies. The reinforcement learning module used an actor–critic training strategy to predict the stock movement. The results revealed the importance of global indicators as well as the adversarial training approach for stock movement tasks. Furthermore, we show that the best results were achieved using the proposed ALN framework in comparison to the existing state-of-the-art approaches.

One of the probable limitations to this approach is the selection of hyperparameters that requires extensive experiments to be conducted in prior to achieve the best predictive performance. The future direction in which we want to extend the methodology is portfolio management where the portfolios for multiple stocks are built along with the relevant crowdsourced data available on the Internet to improve the stock trading strategy. We firmly believe that the ALN framework trained on global indicators and market crash data would help to predict the market trend efficiently even in a pandemic situation just as the one we are facing currently.

# REFERENCES

[1] R. Xiaodong, A. G. Singh, J. Anish, B. R. Singh, and Z. Peiying, “Adaptive recovery mechanism for SDN controllers in edge-cloud supported fintech applications,” IEEE Internet Things J., early access, Mar. 8, 2021, doi: 10.1109/JIOT.2021.3064468.   
[2] H. Xu, L. Chai, Z. Luo, and S. Li, “Stock movement predictive network via incorporative attention mechanisms based on tweet and historical prices,” Neurocomputing, vol. 418, pp. 326–339, Dec. 2020. [Online]. Available: https://doi.org/10.1016/j.neucom.2020.07.108   
[3] W. Huang, Y. Nakamori, and S.-Y. Wang, “Forecasting stock market movement direction with support vector machine,” Comput. Oper. Res., vol. 32, no. 10, pp. 2513–2522, Oct. 2005. [Online]. Available: https://doi.org/10.1016/j.cor.2004.03.016   
[4] B. Huang, Y. Huan, L. D. Xu, L. Zheng, and Z. Zou, “Automated trading systems statistical and machine learning methods and hardware implementation: A survey,” Enterprise Inf. Syst., vol. 13, no. 1, pp. 132–144, Jul. 2018. [Online]. Available: https://doi.org/10.1080/17517575.2018.1493145   
[5] R. P. Schumaker and H. Chen, “Textual analysis of stock market prediction using breaking financial news,” ACM Trans. Inf. Syst., vol. 27, no. 2, pp. 1–19, Feb. 2009. [Online]. Available: https://doi.org/10.1145/1462198.1462204   
[6] H. Wang, Y. Wu, G. Min, J. Xu, and P. Tang, “Data-driven dynamic resource scheduling for network slicing: A deep reinforcement learning approach,” Inf. Sci., vol. 498, pp. 106–116, Sep. 2019. [Online]. Available: https://doi.org/10.1016/j.ins.2019.05.012   
[7] Y. Yuan et al., “A novel multi-step $Q$ -learning method to improve data efficiency for deep reinforcement learning,” Knowl. Based Syst., vol. 175, pp. 107–117, Jul. 2019. [Online]. Available: https://doi.org/10.1016/j.knosys.2019.03.018   
[8] X. Liang, Z. Ge, L. Sun, M. He, and H. Chen, “LSTM with wavelet transform based data preprocessing for stock price prediction,” Math. Probl. Eng., vol. 2019, pp. 1–8, Jul. 2019, doi: 10.1155/2019/1340174. [Online]. Available: https://www.hindawi.com/journals/mpe/2019/1340174/   
[9] M. Gidea and Y. Katz, “Topological data analysis of financial time series: Landscapes of crashes,” Physica A Stat. Mech. Appl., vol. 491, pp. 820–834, Feb. 2018. [Online]. Available: https://doi.org/10.1016/j.physa.2017.09.028   
[10] X. Lin, Z. Yang, and Y. Song, “Short-term stock price prediction based on echo state networks,” Exp. Syst. Appl., vol. 36, no. 3, pp. 7313–7317, Apr. 2009. [Online]. Available: https://doi.org/10.1016/j.eswa.2008.09.049   
[11] P. Khuwaja, S. A. Khowaja, I. Khoso, and I. A. Lashari, “Prediction of stock movement using phase space reconstruction and extreme learning machines,” J. Exp. Theor. Artif. Intell., vol. 32, no. 1, pp. 59–79, May 2019. [Online]. Available: https://doi.org/10.1080/0952813x.2019.1620870   
[12] A. K. Nassirtoussi, S. Aghabozorgi, T. Y. Wah, and D. C. L. Ngo, “Text mining of news-headlines for FOREX market prediction: A multilayer dimension reduction algorithm with semantics and sentiment,” Exp. Syst. Appl., vol. 42, no. 1, pp. 306–324, Jan. 2015. [Online]. Available: https://doi.org/10.1016/j.eswa.2014.08.004   
[13] X. Li, H. Xie, L. Chen, J. Wang, and X. Deng, “News impact on stock price return via sentiment analysis,” Knowl. Based Syst., vol. 69, pp. 14–23, Oct. 2014. [Online]. Available: https://doi.org/10.1016/j.knosys.2014.04.022   
[14] Z. Zhou, J. Zhao, and K. Xu, “Can online emotions predict the stock market in China?” in Web Information Systems Engineering—WISE. Beijing, China: Springer Int., 2016, pp. 328–342. [Online]. Available: https://doi.org/10.1007/978-3-319-48740-3_24   
[15] X. Zhong and D. Enke, “Forecasting daily stock market return using dimensionality reduction,” Exp. Syst. Appl., vol. 67, pp. 126–139, Jan. 2017. [Online]. Available: https://doi.org/10.1016/j.eswa.2016.09.027   
[16] W. Long, Z. Lu, and L. Cui, “Deep learning-based feature engineering for stock price movement prediction,” Knowl. Based Syst., vol. 164, pp. 163–173, Jan. 2019. [Online]. Available: https://doi.org/10.1016/j.knosys.2018.10.034   
[17] Z. Hu, W. Liu, J. Bian, X. Liu, and T.-Y. Liu, “Listening to chaotic whispers,” in Proc. 11th ACM Int. Conf. Web Search Data Min., Feb. 2018. [Online]. Available: https://doi.org/10.1145/3159652.3159690   
[18] J. Carapuço, R. Neves, and N. Horta, “Reinforcement learning applied to forex trading,” Appl. Soft Comput., vol. 73, pp. 783–794, Dec. 2018. [Online]. Available: https://doi.org/10.1016/j.asoc.2018.09.017   
[19] X. Wu, H. Chen, J. Wang, L. Troiano, V. Loia, and H. Fujita, “Adaptive stock trading strategies with deep reinforcement learning methods,” Inf. Sci., vol. 538, pp. 142–158, Oct. 2020. [Online]. Available: https://doi.org/10.1016/j.ins.2020.05.066   
[20] Y. Xu and S. B. Cohen, “Stock movement prediction from tweets and historical prices,” in Proc. 56th Annu. Meeting Assoc. Comput. Linguist. Vol. 1 Long Papers, 2018, pp. 1–9. [Online]. Available: https://doi.org/10.18653/v1/p18-1183   
[21] S. Flanders and S. Kennedy. The 12 Global Economic Indicators to Watch. Accessed: Nov. 10, 2020. [Online]. Available: https://www. bloomberg.com/graphics/world-economic-indicators-dashboard/   
[22] S. Yan, T. Yiyuan, D. Shuxue, L. Shipin, and C. Yifen, “Diagnose the mild cognitive impairment by constructing bayesian network with missing data,” Exp. Syst. Appl., vol. 38, no. 1, pp. 442–449, 2011. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0957417410005889