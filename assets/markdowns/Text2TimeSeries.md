# Text2TimeSeries: Enhancing Financial Forecasting through Time Series Prediction Updates with Event-Driven Insights from Large Language Models

Litton Jose Kurisinkel1 Pruthwik Mishra2 Yue Zhang3 Institute for Infocomm Research, A\*STAR, Singapore1 IIIT Hyderabad2, Westlake University3 litton_kurisinkel@i2r.a-star.edu.sg, pruthwik.mishra@research.iiit.ac.in, yue.zhang $@$ wias.org.cn

# Abstract

Time series models, typically trained on numerical data, are designed to forecast future values. These models often rely on weighted averaging techniques over time intervals. However, real-world time series data is seldom isolated and is frequently influenced by non-numeric factors. For instance, stock price fluctuations are impacted by daily random events in the broader world, with each event exerting a unique influence on price signals. Previously, forecasts in financial markets have been approached in two main ways: either as time-series problems over price sequence or sentiment analysis tasks. The sentiment analysis tasks aim to determine whether news events will have a positive or negative impact on stock prices, often categorizing them into discrete labels. Recognizing the need for a more comprehensive approach to accurately model time series prediction, we propose a collaborative modeling framework that incorporates textual information about relevant events for predictions. Specifically, we leverage the intuition of large language models about future changes to update real number time series predictions. We evaluated the effectiveness of our approach on financial market data.

# 1 Introduction

In the rapidly evolving field of global finance, Artificial Intelligence (AI) plays a pivotal role. In an interconnected world characterized by cross-border trade and expanding economies, marked by intricate relationships and interdependencies, AI is essential for navigating these complexities Cao [2022]. Predicting stock price movements has been a long-standing focus for the AI community, as the stock market is highly sensitive to macroeconomic events, making accurate forecasting a significant challenge. Historically, research has primarily concentrated on forecasting financial markets using univariate time series prediction methods Wah and Qian [2002]. Some studies have addressed this issue by employing multivariate time series prediction or by considering the interdependence of price series from different companies to forecast price movements Wu et al. [2013], Xiang et al. [2022]. While time series models are effective at predicting cyclical trends and overall market growth Zhou et al. [2022], Woo et al. [2022], they often fail to capture the impact of sequential financial events. Predictions that do not consider such events tend to be less precise. The current work explores time series prediction of stock prices in a multi-modal setting that incorporates both text and time series data, where the textual description of an event is considered for short-term price prediction.

Event-driven stock sentiment prediction primarily focuses on anticipating how an event will affect stock prices, typically classifying the impact into discrete labels such as increase, decrease, or no noticeable change Ding et al. [2014, 2016]. Some approaches incorporate historical price sequences to forecast whether prices will rise or fall Sawhney et al. [2020]. However, the effects of an event may span several days, with varying rates of price changes. A simple sentiment label may not be sufficient to capture this complexity. Therefore, instead of assigning a limited number of sentiment polarities to an event, we model the effects of the event in terms of change directions with associated real values. Our current work investigates methods to convert market excitement related to events into real-valued stock prices over the subsequent $n$ days. We are motivated by the fact that forecasting an event’s influence on stock prices over an extended period is beneficial for devising effective intervention strategies Pricope [2021].

![](images/a9de4b12fcb31d991ffdf47ea29eec28b720a3fd2f76b20c98cb3e5dffa15c59.jpg)  
Figure 1: Stock Price Dynamics: Event Induced Changes in Time Series

Leveraging the capability of short-term market excitement, large language models (LLMs) could excel in intuitively predicting future changes based on specific events Lopez-Lira and Tang [2023]. LLMs like ChatGPT are particularly adept at capturing the finer nuances in stock-specific news texts and accurately predicting daily stock market returns due to their superior language understanding capabilities. Lopez-Lira and Tang [2023] also highlight the limitations of basic models like BERT in natural language understanding. Our objective is to explore the ability of LLMs to anticipate changes across multiple time points and represent these as distinct labels corresponding to different future time spans. By "time span," we refer to the period of short-term excitement in the market. Additionally, we aim to examine how these insights can inform adjustments in predictions within time-series models.

In our current research, we integrate multivariate time series data with textual information from stock specific news events to forecast how events either enhance or diminish signals in stock prices relative to the overall trend. This particular scenario we are trying to address is depicted in the Figure 1. Initially, we train multivariate time series models to predict individual stock prices. Drawing inspiration from state change models, we conceptualize market excitement following an event as shifts in the stock state Bosselut et al. [2017]. To accomplish this, we leverage the event-based insights generated as discrete labels by a Large Language Model regarding the price changes for the next $n$ time points following an event occurrence. We utilize these stock state changes to anticipate the increase or decrease in a stock’s price beyond what is projected by the time series model. Following this, we combine the time series model’s predictions with the event-induced changes predicted by the state change model to refine our forecasts. To the best of our knowledge, we are the first to develop a scheme for predicting short-term excitement in stock price time series. We are introducing a novel scheme for short-term excitement prediction in stock price time series, utilizing a Large Language Model to forecast sequences of discrete labels representing event-induced price changes over time.

# 2 Related Work

Methods for Time Series Analysis. Recent advancements in deep learning architectures, such as Long Short-Term Memory (LSTM) networks Hochreiter and Schmidhuber [1997], Gated Recurrent

Units (GRU) Chung et al. [2014], and transformers Vaswani et al. [2017], have demonstrated significant capabilities in capturing complex temporal relationships within time series data. Various transformer models have been proposed Li et al. [2019], Zhou et al. [2021a], Wu et al. [2021], Zhou et al. [2022], Liu et al. [2021] for forecasting time series, often designing novel attention mechanisms to handle longer sequences and using point-wise attention, which can overlook the importance of patches. Although Triformer Cirstea et al. [2022] introduces patch attention, it does not use patch inputs. Patch Time Series Transformer (Patch TST) Nie et al. [2022] was the first transformer model to use patches as inputs, capturing the semantic coherence among neighboring patches. However, these techniques cannot be directly adapted to a multimodal setting involving textual information. Our current work investigates time series prediction in a multimodal setting, comprising both time series and textual information.

# Time Series Analysis for Stock Prediction.

Several time series analysis methods and machine learning techniques can be applied for stock prediction. These include ARIMA models, Exponential Smoothing State Space models (ETS) Brown [1956], and machine learning techniques such as linear regression, decision trees, random forest, SVM, gradient boosting, Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models Tse and Tsui [2002], Engle [2002], and ensemble methods involving multiple models. Hu et al. [2018] developed a hybrid attention mechanism to predict stock market movements using news articles, while BERT representations have been used to encode texts for the FEARS index Da et al. [2011] in predicting movements in the S&P 500 index Yang et al. [2019]. However, these techniques are typically adapted to handle information derived from a sequence of financial events, which can result in inaccurate predictions during unforeseen events that impact financial decisions. Our approach models time series prediction in a multimodal setting, where predictions are evaluated in the context of specific events.

NLP for Finance. Financial services have always been tightly regulated by governments due to their pervasive impact on the masses. However, following liberalization and the easing of regulations, financial technology (FinTech) has emerged as one of the top business avenues in the last decade. Chen et al. [2020] highlights the application areas of NLP in the finance domain. Financial institutions use end-to-end transformer models to scan and extract financial events from various news articles and financial announcements Zheng et al. [2019], evaluating the debt-paying ability of corporate customers. Online forums, blogs, and social media posts are monitored to extract sentiment, which is then used to predict company sales using model-agnostic meta-learning methods Lin et al. [2019], Finn et al. [2017]. Similarly, insurance companies track daily posts from customers to detect and initiate early treatment of diseases Losada et al. [2019], Burdisso et al. [2019], mitigating the chances of hazards. Social media posts also serve as indicators for stock recommendations Tsai et al. [2019]. Most of these works are formulated as simple sentiment label predictions, which may not fully capture the complexity of financial events. Therefore, instead of assigning a limited number of sentiment polarities to an event, we model the effects of the event in terms of change directions with associated real values. Our current work investigates methods to convert market excitement related to events into real-valued stock prices over the subsequent $n$ days.

# 3 TimeS: Overall Method

Our objective is to forecast the impact of an event on the price signal of a stock for the next $n$ time units and adjust the prediction of our time series model accordingly. Let’s break down the task into three steps.

$$
\begin{array} { r l } & { P _ { s } [ t : t + n ] \longleftarrow T _ { s } ( P _ { s } [ t : t - h ] ; \theta _ { 1 } ) } \\ & { \Delta P _ { s } [ t : t + n ] \longleftarrow F ( E , s ; \theta _ { 2 } ) } \\ & { P _ { s } ^ { \prime } [ t : t + n ] \longleftarrow U ( \Delta P _ { s } [ t : t + n ] , P _ { s } [ t : t + n ] ; \theta _ { 3 } ) } \end{array}
$$

Where, $T _ { s }$ represents the time series function which takes the historic price of a specific stock $s$ for the previous $h$ time points as an argument and forecasts its future values for $n$ time units. $F$ denotes a function predicting the impact of an event $E$ on the price of stock $s$ for the subsequent $n$ time units from the point of occurrence of the event. Finally, $U$ signifies an update function that takes outputs from $T _ { s }$ and $F$ , adjusting the time signal for the upcoming $n$ time- steps by amplifying or attenuating it. we commence by training a dedicated time-series model, denoted as $T _ { s }$ , for each individual stock $s$ . This model is designed to project the trajectory and expansion of the stock over the subsequent $n$ days, leveraging prices derived from the preceding $h$ days as its input. Central to our approach is the utilization of function $F$ within the problem formulation, tasked with assessing the influence of specific event, represented as $E$ , on the market sentiment surrounding stock $s$ . We conceptualize this process as a state transition problem, aimed at depicting the stock’s behavior over the ensuing $n$ days following the occurrence of an event. Within this framework, we quantify the extent of amplification or attenuation in the stock price for each future day, predicated on its corresponding stock state. The state transition and prediction are guided by the intuition of an LLM regarding the patterns of future price changes of the stock within the context of the event. Following this assessment, we implement an update mechanism denoted as $U$ to refine the predictions generated by the time-series model, integrating insights into amplification or attenuation derived from the preceding analysis. Notably, while each stock is assigned its own $T _ { s }$ , the other components remain consistent across all stocks. The rationale behind this strategic design choice will be explained in subsequent discussions.

# 3.1 $T _ { s }$ :Time Series Model

Time series models are trained to predict the values for next $n$ time points by taking previous $h$ time point values. Our time series model can be represented as follows.

$$
P _ { s } [ t : t + n ] = T _ { s } ( H _ { s } [ t : t - h ] )
$$

Where $P _ { s } [ t : t + n ]$ is the price of the stock $s$ for next $n$ time points from the current time $t$ $H _ { s } [ t : t - \mathbf { \bar { \boldsymbol { h } } } ]$ is a multivariate sequence of historic data of previous $h$ time points. The multivariate sequence contains a parallel sequence such as stock prices of the stock, different index values, or exchange rates which can play a role in modeling general market tendencies and its effect on price of $s$ .

# 3.1.1 $F$ :Stock state computation using Indicators Predicted by large Language Models

LLMs trained on text data could intuitively grasp stock price movements across various future time spans, albeit without predicting exact values. For the purpose we fine-tune large language models to predict stock predict stock price trend as discrete labels containing the intuition of large models regarding price change of stock for next $n$ days as follows,

$$
l _ { s , 1 } , l _ { s , 2 } , \ldots , l _ { s , n } = \mathbf { L L M _ { s t o c k } } ( E , S )
$$

The process of fine-tuning to produce these price change labels is explained in the Appendix D. We calculate the stock state transition using a Gated Recurrent Unit initialized with the embedding $E m b ( s )$ of the stock $s$ , which takes the corresponding LLM-predicted label $l _ { s , t }$ at each time-step $t$ to produce temporal state $S _ { t }$ of the stock $s$ .

Amplification Prediction using Temporal stock State $S _ { t }$ The time series can be viewed as a random walk in the 2D grid as shown in Figure 3. At any point of time, it takes any of the three directions namely increase, decrease, or stay steady which could be represented by direction indicator values 1, -1, 0 respectively. We use the stock states compute the probability for time series to take each of the directions, increase, steady, or decrease. The expected value direction indicator is computed using these probabilities represent the amplification/attenuation value which can be subsequently used to update the time series. With this view in mind, we compute the price amplification/attenuation from stock state $S _ { t }$ at time step $t$ as follows.

$$
\begin{array} { r l r } { \mathrm { P r o b } \mathrm { D } _ { t } = } & { \boldsymbol { W } _ { a } \cdot \boldsymbol { S } _ { t } } & \\ { \boldsymbol { A } \boldsymbol { s } _ { t } = ( 1 ) * \cdot \mathrm { P r o b } \mathrm { D } _ { t } [ 0 ] + ( - 1 ) * \cdot \mathrm { P r o b } \mathrm { D } _ { t } [ 2 ] + ( 0 ) * \cdot \mathrm { P r o b } \mathrm { D } _ { t } [ 1 ] } & \end{array}
$$

Where $W _ { a }$ is a parameter matrix and $\mathrm { P r o b } \mathrm { D } _ { t }$ belongs to $R ^ { 3 }$ which contains the probablity for increase, decrease, and neutral. $A s _ { t }$ is the amplification or attenuation value. We concatenate the $A s _ { 1 }$ to $A s _ { n }$ o form the amplification vector $A _ { s } [ 1 : n ] \epsilon R ^ { n }$ .

# 3.2 U: Updating time Series Price Predictions

Once we compute $A _ { s } [ 1 : n ]$ , we use it to update the values predicted by time series model $T _ { s }$ . We take a simple linear transformation of the concatenated vector $[ A _ { s } [ 1 : n ] , P _ { s } [ t : t + n ] ]$ to predict the

![](images/a4cf0548f24524de105ffd02bfe3e8adf9b73c3c2b3579038f7c270d163fb39f.jpg)  
Figure 2: TimeS: In the lower portion of the diagram, the LLM utilizes stock and event data as inputs to forecast price change indicators for the subsequent $n$ time intervals, which are then employed to determine stock states. In the upper portion of the diagram, time series are updated using price amplification values derived from these stock states.

![](images/e2f10610f5a4bc1969dcc2b9b3a59c4295d74a42beead62874283d89920bc460.jpg)  
Figure 3: Time series depicted as a Random Walk on a 2D Grid, where at every time point, it may either increase, decrease, or remain neutral, denoted by 1, -1, and 0, respectively.

update price of stock $S$ in the context of the event $E$

$$
P _ { s } ^ { \prime } [ t : t + n ] = W _ { a } \cdot [ \alpha * A _ { s } [ 1 : n ] , P _ { s } [ t : t + n ] ]
$$

$P _ { s } [ t : t + n ]$ is the price predictions by the time series model as represented by the Equation 1 and $\alpha$ is a hyper-parameter.

Loss: We opt for Mean Squared Error (MSE) loss to quantify the disparity between the prediction and the actual values. The loss is computed as the MSE loss between updated price $P _ { s } ^ { \prime } [ t ]$ and expected price $P ^ { a } { } _ { s } [ t ]$ .

# 4 Experiments

Our primary objective is to enhance time series predictions in response to events using a large language model (LLM). As illustrated in Figure 2, our method integrates several key components: a time series model, an LLM trained to predict stock price changes over various future time spans as discrete labels, and mechanisms for updating the time series based on the LLM’s predictions. This section details the data, settings, and results for the following tasks: 1) Sub Task1: Training the time series models, 2) Sub Task2: Fine-tuning the LLM for price change prediction, and 3) Main Task1: Overall approach for updating the time series using the LLM’s predicted labels, as depicted in Figure 2.

# 4.1 Datasets

ExtEDT: Extended EDT Dataset with News Events and Time Series Data Our experimentation utilized the EDT Dataset, serving as the foundational resource Zhou et al. [2021b]. This dataset comprises stock tickers, with each entry corresponding to a specific company’s stock, accompanied by a textual description of a company-related news event and the event’s date of occurrence. To enable a detailed evaluation, we partitioned the dataset into small-cap, mid-cap, and large-cap stocks. In order to tailor the dataset to our task, we retrieved the closing price of each stock for the subsequent $n$ days following the event using the Yahoo Finance API 1. Additionally, we automatically annotated the price change labels for future $n$ days, for each event within every record, adhering to the methodology outlined in Appendix D. The EDT dataset is divided into training, validation, and test sets, containing 46397, 5210, and 5263 samples. To create these partitions, we allocate ticker-wise samples in an 80:10:10 ratio.

Dataset: Training Time Series Models The focus of the present paper is on updating Time series models trained on long-term stock price sequences. As previously stated, we chose to train separate time series models for each stock available in the EDT dataset. To achieve this, we gathered time series data of closing prices for each stock over the past 30 years, along with the corresponding values for the dollar exchange index and NASDAQ exchange index using yahoo Finance $\mathrm { A P I } ^ { 2 }$ . For every stock, we amalgamated these sequences to form a multivariate time series. This multivariate sequence is then divided into different source and target sequences with fixed source length, target length, and stride values. The input comprises the NASDAQ index, dollar exchange rate, and stock price sequence, while the output is a univariate sequence of stock prices. More details of training individual time series models can be found in Appendix A.1

# 4.2 Fine tuning LLM for Price Change Label Prediction

This task is modeled as a sequence-to-sequence prediction task where the input is a news event about a stock prepended with the ticker’s name and the output is a sequence of price change labels. Each price change label is discrete in nature where we capture the type of the change with its actual value. The type of change can belong to any of two categories: increase (INC) and decrease (DEC). The actual change value is represented in terms of integers instead of real values. For cases where there is no change in the values, we consider that as an increment (INC) with a zero change value. One example from our dataset is shown in Table 7.

# 4.2.1 Settings:

We leverage three variants of T5 (Text-To-Text-Transfer-Transformer) Raffel et al. [2020] models for the price change predictions. T5’s unified framework excels at transferring knowledge from various tasks via pre-training on a massive dataset. We restrict ourselves from using newer LLMs Touvron et al. [2023a,b], Jiang et al. [2023, 2024], Le Scao et al. [2023], Li et al. [2023], Zhang et al. [2022] to avoid the potential effects of data contamination as these newer models might report overestimated performance in the test sets. We fine tune 3 variants of T5: T5-Base, T5-Large, and T5-3B. For the T5-Base model, we fine tune all its parameters whereas for larger models we fine tune on reduced sets of parameters. We freeze all the encoders layers of the T5-Large model whereas 8-bit low rank adaptation Hu et al. [2021] is applied to the T5-3B model.

# 4.2.2 Evaluation and Results

We evaluate the predictions at two levels. The first one deals with the performance of predicting the change type accurately whereas the second level evaluates the prediction of values. Instead of exactly matching the values, we employ a mechanism of window of values matching for this. We label a prediction correct if the value lies with in a window around the exact value. We use a windows of length 5 for the evaluation of values. For a value $\nu$ , the window of length 5 is represented as the range $\nu { - } 5 . . \nu { + } 5 $ . The change type is evaluated using micro F1 score and the details of the performance of different T5 variants are presented in Table 1.

<table><tr><td>Model</td><td> Validation</td><td>Test</td></tr><tr><td>T5-Base</td><td>0.68</td><td>0.65</td></tr><tr><td>T5-Large</td><td>0.63</td><td>0.61</td></tr><tr><td>T5-3B</td><td>0.64</td><td>0.61</td></tr></table>

Table 1: Results for different T5 variants of Change Type Predictions using Micro-F1 Scores

The F1-scores of predicting the actual change values with different window sizes is reported in Table 2.   

<table><tr><td>Model</td><td>Validation</td><td>Test</td></tr><tr><td>T5-Base</td><td>0.55</td><td>0.56</td></tr><tr><td>T5-Large</td><td>0.55</td><td>0.56</td></tr><tr><td>T5-3B</td><td>0.55</td><td>0.55</td></tr></table>

Table 2: Results of Change Values Using T5 Variants in a Window Length of 5 using Micro-F1 Scores

# 4.3 Main Task: Updating Time Series Prediction with Insights from LLM

# 4.3.1 Baseline Settings

We compared our approach with several state-of-the-art time series models, including variants of Patch-TST and D-Linear, to assess their effectiveness in updating time series predictions. Specifically, we adapted the Patch- $\mathrm { T S T } { + } \mathrm { W }$ and D-Linear $+ \mathbf { W }$ variants for multi-channel input to single-channel output prediction (see Appendix A.1 for more details). Additionally, we explored a class of models based on lightweight natural language processing techniques used for stock sentiment predictions. To facilitate a fair comparison, we modified these models to create a time series-specific version that predicts future time-step values instead of sentiment labels. For more information on these settings, please refer to Appendix B.

# 4.3.2 Model Variants

We combined our approach T imeS depicted in Figure 2, for stock state computation and amplification prediction with different finetuned variants of T5 model mentioned in Section 4.2. For T imeS, we set the learning rate to $1 0 ^ { - } 4$ , using the Adam optimization algorithm Kingma and Ba [2014]. During the training of T imeS, the pretrained time series component $D L i n e a r + W$ was frozen. In time series, simpler models made surprising models as in the case of D-Linear. Inspired by this scheme created simpler model TimeL, without including stock change computation. This approach re-approximates the original percentage change from discrete labels predicted by the LLM component and this sequence of values are used for updating time series predictions. Details of this setting can be seen in Appendix 3. This approach was also tested with different variants of T5.

Table 3: The table presents results for 9 days following the event. The $S m a l l - C a p$ test set includes 1,067 stocks and 2,623 events, the $M i d - C a p$ test set comprises 386 stocks and 887 events, and the $L a r g e - C a p$ test set contains 488 stocks and 1,724 events. Lower values indicate improved performance. The best results are highlighted in bold.   

<table><tr><td rowspan="2">Setting</td><td colspan="2">Small-Cap</td><td colspan="2">Mid-Cap</td><td colspan="2">Large-Cap</td></tr><tr><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td></tr><tr><td>DLinear</td><td>0.13</td><td>0.30</td><td>0.141</td><td>0.270</td><td>0.122</td><td>0.261</td></tr><tr><td>PatchTST/5</td><td>0.190</td><td>0.35</td><td>0.190</td><td>0.280</td><td>0.162</td><td>0.271</td></tr><tr><td>SentiEvent</td><td>0.180</td><td>0.37</td><td>0.171</td><td>0.370</td><td>0.172</td><td>0.392</td></tr><tr><td>T5-base+ TimeS</td><td>0.120</td><td>0.205</td><td>0.101</td><td>0.206</td><td>0.108</td><td>0.190</td></tr><tr><td>T5-Large+TimeS</td><td>0.120</td><td>0.225</td><td>0.110</td><td>0.230</td><td>0.106</td><td>0.210</td></tr><tr><td>T5-3b+TimeS</td><td>0.121</td><td>0.227</td><td>0.113</td><td>0.231</td><td>0.124</td><td>0.216</td></tr><tr><td>T5-base+ TimeL</td><td>0.123</td><td>0.270</td><td>0.135</td><td>0.25</td><td>0.127</td><td>0.25</td></tr><tr><td>T5-Large+ TimeL</td><td>0.127</td><td>0.290</td><td>0.136</td><td>0.28</td><td>0.120</td><td>0.270</td></tr><tr><td>T5-3b+ TimeL</td><td>0.126</td><td>0.293</td><td>0.137</td><td>0.27</td><td>0.123</td><td>0.270</td></tr></table>

# 5 Results

We assessed the primary task of updating time series using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) as metrics. RMSE measures the square root of the average squared differences between predicted and actual values, while MAE represents the average of the absolute differences between predicted and actual values. The results of the updated price prediction, in the context of an event, are presented in the Table 3. Clearly, updates based on LLM-predicted indicators have improved the accuracy of the time-series predictions. In contrast, SentiEvent performed poorly compared to the LLM-based models. This disparity is likely due to the sophisticated background understanding and enhanced text comprehension capabilities of LLMs in the financial domain. The TimeS settings outperformed the TimeL settings. TimeS computes amplification in a probabilistic space, whereas TimeL approximates actual values of amplification from LLM-predicted labels. This approximation limits TimeL’s ability to detect errors in LLM predictions and make the necessary adjustments in amplification computation.

# 6 Ablation Study

6.1 Ablation Study: Performance T5 During Increment and Decrement   

<table><tr><td rowspan="2">Model</td><td colspan="4">DEC ValidatioOverall</td><td colspan="2">INest Overall</td></tr><tr><td></td><td></td><td></td><td>DEC</td><td></td><td></td></tr><tr><td>T5-Base</td><td>0.56</td><td>0.75</td><td>0.68</td><td>0.53</td><td>0.73</td><td>0.65</td></tr><tr><td>T5-Large</td><td>0.42</td><td>0.72</td><td>0.63</td><td>0.39</td><td>0.71</td><td>0.61</td></tr><tr><td>T5-3B</td><td>0.5</td><td>0.71</td><td>0.64</td><td>0.47</td><td>0.7</td><td>0.61</td></tr></table>

Table 4: Label Wise Results for different T5 variants of Change Type Predictions using Micro-F1 Scores

From Table 4, it is evident that all the models perform better in predicting the INC label while DEC label prediction task is challenging for them. Table 5 depicts a picture of the performance in terms of different magnitude ranges of change values for change type predictions. We denote change values in the range of 0..15 as Low, 16..31 as Medium, and rest as Large. We can observe that the performance of all the models to predict the $D E C$ tag increase as we move from the Low to Large range of change values while that of INC. This may result from the low sensitivity of T5 models towards events which leads to minimal changes in the decrement direction. For a detailed analysis, please refer to Appendix E.

<table><tr><td rowspan="2">Data</td><td rowspan="2">Model</td><td colspan="3">#Sampw INcDEC</td><td colspan="3">#SMepiumNCanDEC</td><td colspan="2">#Samprge NcngDEC</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Test</td><td>T5-Base T5-Large</td><td rowspan="2">10811</td><td>0.71 0.7</td><td>0.49 0.34</td><td rowspan="2">3195</td><td>0.75</td><td rowspan="2">0.59 0.44</td><td rowspan="2">1780</td><td>0.78 0.65 0.74</td></tr><tr><td>T5-3B</td><td>0.68</td><td>0.72</td><td></td><td>0.55 0.58</td></tr><tr><td rowspan="2">Val</td><td rowspan="2">T5-Base T5-Large</td><td rowspan="2"></td><td>0.73</td><td>0.44 0.5</td><td rowspan="2">3334</td><td>0.72 0.79</td><td rowspan="2">0.52 0.65</td><td rowspan="2">1843</td><td>0.73 0.81 0.69</td></tr><tr><td>10453</td><td>0.34</td><td></td><td></td></tr><tr><td rowspan="2"></td><td rowspan="2">T5-3B</td><td rowspan="2"></td><td>0.7 0.7</td><td></td><td rowspan="2"></td><td>0.74</td><td rowspan="2">0.5</td><td rowspan="2">0.75</td><td>0.58</td></tr><tr><td></td><td>0.46</td><td>0.73 0.56</td><td>0.75 0.62</td></tr></table>

Table 5: Micro-F1 Scores Comparison between different Ranges of Change Magnitudes for Change Type Predictions.

6.2 Ablation Study : Performance During Different Range of Price Variations   

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">Low Change</td><td colspan="3">Medium Change</td><td colspan="3">Large Change</td></tr><tr><td colspan="3">Window Size</td><td colspan="3">Window Size</td><td colspan="3">Window Size</td></tr><tr><td rowspan="3">Test</td><td></td><td>5</td><td>10</td><td>15</td><td>5</td><td>10</td><td>15</td><td>5</td><td>10</td><td>15</td></tr><tr><td>T5-Base</td><td>0.71</td><td>0.92</td><td>0.96</td><td>0.28</td><td>0.58</td><td>0.85</td><td>0.1</td><td>0.18</td><td>0.28</td></tr><tr><td>T5-Large</td><td>0.77</td><td>0.97</td><td>0.99</td><td>0.17</td><td>0.44</td><td>0.78</td><td>0.03</td><td>0.06</td><td>0.13</td></tr><tr><td rowspan="3">Validation</td><td>T5-3B</td><td>0.72</td><td>0.92</td><td>0.96</td><td>0.26</td><td>0.55</td><td>0.82</td><td>0.09</td><td>0.15</td><td>0.26</td></tr><tr><td>T5-Base</td><td>0.72</td><td>0.91</td><td>0.96</td><td>0.26</td><td>0.57</td><td>0.83</td><td>0.1</td><td>0.18</td><td>0.27</td></tr><tr><td>T5-Large T5-3B</td><td>0.77 0.72</td><td>0.96 0.92</td><td>0.99 0.96</td><td>0.17 0.26</td><td>0.44 0.54</td><td>0.78 0.82</td><td>0.05 0.08</td><td>0.08 0.16</td><td>0.14 0.26</td></tr></table>

Table 6: Micro-F1 Scores Comparison between different Ranges of Change Magnitudes for Change Value Predictions With Different Window Lengths.

Table 6 represents the prediction accuracies for change values belonging to different categories as mentioned above. It is challenging for all the models to accurately predict the change values when change values are large while smaller change values are predicted with high precision. However, T5 models appear to struggle with anticipating price fluctuations during extreme shifts. For case studies on the prediction of price changes and subsequent updates to time series data, please see Appendix E.

# 7 Limitations

To avoid data contamination, we restrict ourselves from using newer LLMs. This results in suboptimal predictions for change types and actual change values. The test data and the validation data contains news articles focusing on trading events from PRNewswire and Businesswire websites in the financial year of 2020-21. As T5 models were released before this duration, we could safely assume that training data of T5 did not overlap with the data considered in this research work. However, capabilities have improved tremendously in the recent past.

# 8 Conclusion

The paper introduces a multi-modal framework for modeling stock price time-series within the context of financial events. This framework integrates insights from large language models (LLMs), using predicted price changes as discrete labels to update the time series. This approach improves the accuracy of stock price forecasts during financial events. The paper also presents various experimental results demonstrating the ability of LLMs to anticipate price changes.

References   
Antoine Bosselut, Omer Levy, Ari Holtzman, Corin Ennis, Dieter Fox, and Yejin Choi. Simulating action dynamics with neural process networks, 2017.   
Robert G Brown. Exponential smoothing for predicting demand. Little, 1956.   
Sergio G Burdisso, Marcelo Errecalde, and Manuel Montes-y Gómez. A text classification framework for simple and effective early depression detection over social media streams. Expert Systems with Applications, 133:182–197, 2019.   
Longbing Cao. Ai in finance: challenges, techniques, and opportunities. ACM Computing Surveys (CSUR), 55(3):1–38, 2022.   
Chung-Chi Chen, Hen-Hsen Huang, and Hsin-Hsi Chen. Nlp in fintech applications: past, present and future. arXiv preprint arXiv:2005.01320, 2020.   
Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.   
Razvan-Gabriel Cirstea, Chenjuan Guo, Bin Yang, Tung Kieu, Xuanyi Dong, and Shirui Pan. Triformer: Triangular, variable-specific attentions for long sequence multivariate time series forecasting–full version. arXiv preprint arXiv:2204.13767, 2022.   
Zhi Da, Joseph Engelberg, and Pengjie Gao. In search of attention. The journal of finance, 66(5): 1461–1499, 2011.   
Xiao Ding, Yue Zhang, Ting Liu, and Junwen Duan. Using structured events to predict stock price movement: An empirical investigation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 1415–1425, 2014.   
Xiao Ding, Yue Zhang, Ting Liu, and Junwen Duan. Knowledge-driven event embedding for stock prediction. In Proceedings of coling 2016, the 26th international conference on computational linguistics: Technical papers, pages 2133–2142, 2016.   
Robert Engle. Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models. Journal of Business & Economic Statistics, 20(3): 339–350, 2002.   
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pages 1126–1135. PMLR, 2017.   
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735–1780, 1997.   
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.   
Ziniu Hu, Weiqing Liu, Jiang Bian, Xuanzhe Liu, and Tie-Yan Liu. Listening to chaotic whispers: A deep learning framework for news-oriented stock trend prediction. In Proceedings of the eleventh ACM international conference on web search and data mining, pages 261–269, 2018.   
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.   
Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.   
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.   
Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman ´ Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176bparameter open-access multilingual language model. 2023.   
Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. Advances in neural information processing systems, 32, 2019.   
Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023.   
Zhaojiang Lin, Andrea Madotto, Genta Indra Winata, Zihan Liu, Yan Xu, Cong Gao, and Pascale Fung. Learning to learn sales prediction with social media sentiment. In Proceedings of the First Workshop on Financial Technology and Natural Language Processing, pages 47–53, 2019.   
Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In International conference on learning representations, 2021.   
Alejandro Lopez-Lira and Yuehua Tang. Can chatgpt forecast stock price movements? return predictability and large language models. arXiv preprint arXiv:2304.07619, 2023.   
David E Losada, Fabio Crestani, and Javier Parapar. Overview of erisk at clef 2019: Early risk prediction on the internet (extended overview). CLEF (Working Notes), 2019.   
Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In The Eleventh International Conference on Learning Representations, 2022.   
Tidor-Vlad Pricope. Deep reinforcement learning in quantitative algorithmic trading: A review, 2021.   
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.   
Ramit Sawhney, Shivam Agarwal, Arnav Wadhwa, and Rajiv Shah. Deep attentive learning for stock movement prediction from social media text and company correlations. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8415–8426, 2020.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.   
Yu-Che Tsai, Chih-Yao Chen, Shao-Lun Ma, Pei-Chi Wang, You-Jia Chen, Yu-Chieh Chang, and Cheng-Te Li. Finenet: a joint convolutional and recurrent neural network model to forecast and recommend anomalous financial items. In Proceedings of the 13th ACM conference on recommender systems, pages 536–537, 2019.   
Yiu Kuen Tse and Albert K C Tsui. A multivariate generalized autoregressive conditional heteroscedasticity model with time-varying correlations. Journal of Business & Economic Statistics, 20(3):351–362, 2002.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
Benjamin W Wah and Minglun Qian. Constrained formulations and algorithms for stock-price predictions using recurrent fir neural networks. In AAAI/IAAI, pages 211–216, 2002.   
Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, and Steven Hoi. Etsformer: Exponential smoothing transformers for time-series forecasting, 2022.   
Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in neural information processing systems, 34:22419–22430, 2021.   
Yue Wu, José Miguel Hernández-Lobato, and Ghahramani Zoubin. Dynamic covariance models for multivariate financial time series. In International Conference on Machine Learning, pages 558–566. PMLR, 2013.   
Sheng Xiang, Dawei Cheng, Chencheng Shang, Ying Zhang, and Yuqi Liang. Temporal and heterogeneous graph neural network for financial time series prediction. In Proceedings of the 31st ACM international conference on information & knowledge management, pages 3584–3593, 2022.   
Linyi Yang, Ruihai Dong, Tin Lok James Ng, and Yang Xu. Leveraging bert to improve the fears index for stock forecasting. In Proceedings of the First Workshop on Financial Technology and Natural Language Processing, pages 54–60, 2019.   
Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 11121–11128, 2023.   
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models, 2022.   
Shun Zheng, Wei Cao, Wei Xu, and Jiang Bian. Doc2edag: An end-to-end document-level framework for chinese financial event extraction. arXiv preprint arXiv:1904.07535, 2019.   
Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 11106–11115, 2021a.   
Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International conference on machine learning, pages 27268–27286. PMLR, 2022.   
Zhihan Zhou, Liqian Ma, and Han Liu. Trade the event: Corporate events detection for news-based event-driven trading. arXiv preprint arXiv:2105.12825, 2021b.

newpage

# A Appendix

# A.1 Time Series Model

In this section, we describe our adaptations of the PatchTST Nie et al. [2022] and D-LinearZeng et al.   
[2023] time series models for handling multi-channel input to single-channel output.

# A.1.1 PatchTST+W

The proposed Transformer-based model for multivariate time series forecasting and self-supervised representation learning utilizes two main methodological components: firstly, the segmentation of time series into subseries-level patches, serving as input tokens for the Transformer model. Secondly, the model adopts a channel-independent approach, where each channel represents a single univariate time series, sharing embedding and Transformer weights across all series. This methodological framework offers advantages such as retaining local semantic information in the embedding, reducing computation and memory usage quadratically, and enabling the model to attend to longer historical contexts. Outputs layers of individual channels are flattened and concatenated to project using a transformation matrix W. We utilized a patch window of 5 and set the learning rate to $1 0 ^ { - } 4$ , employing the Adam optimization algorithm Kingma and Ba [2014].

# A.1.2 DLinear+W

In this study, the authors challenge the effectiveness of Transformer-based solutions for long-term time series forecasting (LTSF), arguing that while Transformers excel in capturing semantic correlations, their permutation-invariant self-attention mechanism leads to temporal information loss in time series modeling. They propose a simple one-layer linear model, LTSF-Linear, which surprisingly outperforms existing Transformer-based LTSF models across nine real-life datasets, highlighting the importance of preserving temporal relations. The findings suggest a need to reconsider the suitability of Transformer-based approaches for LTSF and other time series analysis tasks, potentially opening up new research directions in the field. Outputs layers of individual channels are flattened and concatenated to project using a transformation matrix W. We set the learning rate to $1 0 ^ { - } 4$ , employing the Adam optimization algorithm Kingma and Ba [2014].

Individual Time series models are trained on look back window 30 and prediction length 20.

# A.1.3 Why we use Different time series models for different stocks?

Different stocks exhibit unique behaviors and patterns over time, requiring the use of different time series models. This diversity arises from several factors. Firstly, volatility levels vary, with some stocks experiencing frequent and significant price fluctuations, while others remain stable. Secondly, stocks may follow distinct trends, whether upward, downward, or sideways. Additionally, seasonal patterns or cyclical trends, influenced by factors such as weather, holidays, or economic cycles, contribute to the diversity of stock behavior. Moreover, the degree of randomness or noise in stock prices varies among stocks. Furthermore, the liquidity of stocks plays a crucial role, with different levels impacting market behavior. Therefore, selecting appropriate time series models tailored to these factors is essential for effective stock analysis and forecasting.

![](images/adfa6d93f9dbb1bc5de3d57ca2ef928344b51e294cc04f5093217c4d3725b082.jpg)  
Figure 4: SentiEvent: Base Model Setting for Price Amplification Prediction Using Bert

# B SentiEvent: Base Model Settings

In the current section we explain our method $F _ { 1 }$ serves to calculate the event-induced price amplification levels for stock $S$ over the subsequent $n$ time steps using a BERT approach. The entire method is depicted in the Figure 2

# B.1 $F _ { 1 }$ :Price Amplification Computation Using Temporal Event Embeddings and Stock States

The impact of an event on a stock’s price tends to fade gradually. This fading effect differs across various stocks and event categories. Hence, in our approach denoted as $F _ { 1 }$ , we calculate the changes in stock states by considering the temporal representation of the event over the subsequent $n$ time units. Rest of the methods explain $F _ { 1 }$ in detail.

$E _ { s }$ :Computing Stock Specific event representation Each events impacts different stocks differently and the event details relevant for a different stocks are different. For this reason our method computes stock specific event representation encompassing the relevant information. We encode the event details using Bert model.

$$
\boxed { E _ { b e r t } = \mathsf { b e r t } ( E ) }
$$

To compute the stock specific representation of the event, we use muti- head attention of stock in event bert encodings follows.

$$
E _ { s } = \mathrm { M u l t i H e a d } ( E _ { b e r t } , E m b ( S ) )
$$

Where $E m b ( S )$ is the embedding of stock ticker of stock $S$ from a look up table.

Updating Event Representation for Temporal Information The effect of an event on a stock changes over time. For this reason, we have to incorporate temporal changes of an event. We compute the temporal representations for $E _ { s }$ for next $n$ time units as $[ E _ { s , 1 } , E _ { s , 2 } , E _ { s , 3 } , . . . . . . , E _ { s , n } ]$ by adding positional embedding of the corresponding time unit to $E _ { s }$ .

Stock state transition computation and Price fluctuation Predition We compute the stock state transition using a Gated Recurrent Unit initialized with $E m b ( S )$ and takes corresponding temporal event representation $E _ { s , t }$ at each time- step $t$ . Each state is used for price amplification computation and updated prices using Equations 4 and 5. We set the learning rate to $1 0 ^ { - } 3$ , employing the Adam optimization algorithm Kingma and Ba [2014].

# C TimeL: A Simpler Approach without Stock States

There are time series models which yielded state of art results with embarrassingly simple one-layer linear models. Inspired by this idea we also include an simple model with temporal stock states computation for computing updated price based on the price change indicator labels predicted by $L L M _ { s t o c k }$ . For this purpose, we use reverse computation of Equations 7 and 8 using the LLM predicted labels $[ l _ { s , 1 } , l _ { s , 2 } , \ldots , l _ { s , n } ]$ to approximate the fractional change $\left( \frac { P _ { s , t } - P _ { s , t - 1 } } { P _ { s , t } } \right)$ in the Equation 7. Such values for the entire label sequence is combined for forming the price amplification sequence. We set the learning rate to $1 0 ^ { - } 4$ , employing the Adam optimization algorithm Kingma and Ba [2014].

# D How we train LLM?Converting Price Change Values to Discrete Labels

For each stock-event pairs in our training set we compute discrete labels of their price change using the available price time series data for the stock, for $n$ time steps after the event. At any time step $t$ label $l _ { s , t }$ is computed as follows,

$$
c _ { s , t } = \left\lfloor \frac { \left( \frac { P _ { s , t } - P _ { s , 1 } } { P _ { s , 1 } } \times 1 0 0 \right) } { I } \right\rfloor
$$

$$
l _ { s , t } = \left\{ \begin{array} { l l } { \mathrm { I N C _ { - } } + | c _ { s , t } | } & { \mathrm { i f } \ c _ { s , t } > 0 } \\ { N e u t r a l } & { \mathrm { i f } \ c _ { s , t } = 0 } \\ { \mathrm { D E C _ { - } } + | c _ { s , t } | } & { \mathrm { i f } \ c _ { s , t } < 0 } \end{array} \right.
$$

In Equation 7, $P _ { s , t }$ is the price of the stock at time-step $t$ . The Equation 7 computes the percentage of change in price of the stock $s$ between time steps $t$ and 1 divided by a fractional value $I$ and $\scriptstyle { c _ { s , t } }$ is computed as the floor of the subsequent value. $c _ { s , t }$ can take negative values as absolute values of price change is not considered during computation. Equation 8 is used assign price change label $l _ { s , t }$ for the time step $t$ . clearly, each percentage of price change in between a fraction value of $I$ is project to a single discrete label. For our experiments we set $I { = } 0 . 3$ . ’INC’ and ’DEC’ prefixes indicates whether percentage of change is in increasing or decreasing direction. Using the auto-computed price change labels for all time- steps, an LLM is trained to predict the price change labels for $n$ time-steps for stock $S$ after the event $E$ . To improve predictability, we divide the $n$ time steps into three windows, and the maximum change value within each window is taken as $P _ { s , t }$ for any timestep within the window. For this reason, every time step within a given window receives the same label. Table 7 provides an example of the records used to train the LLM.

<table><tr><td>Ticker</td><td>FNB</td></tr><tr><td>Event</td><td>F.N.B. Corporation Schedules Fourth Quarter 2020 Earnings Report and Conference Call. PITTSBURGH, Jan. 6, 2021 /PRNewswire/ -F.N.B. Corporation (NYSE: FNB) announced today that it plans to issue financial results for the fourth quarter of 202O at 6:O0 PM ET Tuesday, January 19,2021. Chairman, President and Chief Executive Officer, Vincent J. Delie, Jr., Chief Financial Officer, Vincent J. Calabr- ese, Jr.,and Chief Credit Officer, Gary L. Guerrieri, plan to host a</td></tr><tr><td></td><td>conference call to discuss the Company&#x27;s financial results on Wednes- day, January 20,2021 at 8:15 AM ET.</td></tr><tr><td>Label Sequence Input for TimeS</td><td>INC_6 INC_15 INC_10 INC_6INC_6INC_6INC_15INC_15INC_15INC_10INC_10INC_10</td></tr></table>

Table 7: Example of event text with ticker value and change label sequence

# E CASE STUDIES

Case Study 1, depicted in Figure 5, illustrates a scenario of moderate upward price movement. The accompanying news highlights the company’s victory in a competition, which carries clear positive sentiments. Moreover, the time series updates are nearly accurate. In Case Study 2, also in Figure 6, a pharmaceutical company’s success in a clinical trial is showcased. The market’s high level of excitement can be easily inferred by a Language and Logic Model (LLM). The time series updates in this case closely approximate the trajectory of upward movement. Both Case Studies 3 and (Figures 7)represent instances of partially accurate market predictions. These involve highly volatile stocks, for which the LLM lacks information on volatility during training or inference. Towards the end of the predicted sequence, the updated time series $\mathrm { T 5 + T i m e S }$ tends to be biased towards DLinear $+ \mathbf { W } .$ . Moving on to Case Study 5 in Figure 9, the stock under consideration is a low-valued, highly volatile one. The challenge for the LLM lies in accurately identifying the magnitude of price movement due to its ignorance of the stock’s volatility. In Case Study 6, the event concerns operational changes within the company, signaling a potentially risky situation. Consequently, the LLM may predict a negative momentum, and the computed updated time series is nearly accurate. In Case Study 7 (Figure 11), the event revolves around a lawsuit against the company. With enough instances in the training set, the LLM can readily anticipate the magnitude of the negative trend. Finally, in Case Study 8 (Figure 12), the news relates to the quarterly results of a company. Initially appearing positive, the LLM predicts positive labels. However, the company’s performance falls short in comparison to previous quarters. The LLM’s limitations become apparent here, as it lacks the necessary context and capability for such numerical comparisons.

![](images/bf284dd8578e7c55c5f9f4c954c8e1ed00178a16d607e06fd8ee2c0b752f9041.jpg)  
Figure 5: CASE STUDY1:Accurate Prediction During Moderate Upward Price Movement,Stock:FICO, ,Event:"FICO Recognized by Chartis as Category Winner in Innovation, AI Applications, and Financial Crime-Enterprise Fraud; Ranked Sixth Overall in the 2021 Chartis RiskTech 100 Report Position Reflects FICO’s Analytic Innovation Strategy and Ability to Help Organizations Manage the Complexity of Their Analytic Assets. SAN JOSE, Calif., Nov. 30, 2020 /PRNewswire/ – Highlights: FICO ranked sixth in this year’s RiskTech 100 a comprehensive study of the world’s major solution providers in risk and compliance technology FICO was recognized as category winner in Innovation for the fourth consecutive year FICO also won category awards for AI Applications and Financial Crime - Enterprise Fraud Global analytics software provider FICO, today announced that it has ranked sixth in Chartis Research’s annual RiskTech100 report of world’s leading risk technology providers. FICO also won category awards for Innovation, AI Applications, and Financial Crime Enterprise Fraud. ""FICO’s top-ten ranking reflects its innovation strategy"", said Sid Dash, research director at Chartis Research." Expected Labels:INC_5 INC_16 INC_17, Predicted Labels:INC_6 INC_11 INC_16 "

# Price over Time

![](images/a6e9e8e86b98d0d0859dfc91a94c7c074337fed44fdb2ec987c100e832f1c87e.jpg)  
Figure 6: CASE STUDY2: Accurate Prediction During High Updward Price Movement, Stock:SYNBX Event:"Synlogic Initiates Phase 1 Study of SYNB8802 for the Treatment of Enteric Hyperoxaluria. CAMBRIDGE, Mass., Nov.4,2020 /PRNewswire/ – Synlogic, Inc. (Nasdaq: SYBX), a clinical stage companybringing the transformative potential of synthetic biology to medicine, today announced it has treated the first healthy volunteer in its Phase 1 study of theinvestigational Synthetic Biotic medicine SYNB8802 for the treatment of Enteric Hyperoxaluria(HOX). ""We are thrilled to be moving SYNB8802 into the clinic ahead of schedule,"" said Aoife Brennan, M.B. Ch.B., Synlogic’s President and Chief Executive Officer. Expected Labels:INC_20 INC_27 INC_24 Predicted Labels:INC_17 INC_21 INC_22 "

![](images/41fac67f42f02385f89ce5412da23d4abe93b18ae93fa057e0596f7a7476c0b7.jpg)  
Figure 7: CASE STUDY3:Partially Accurate Predictions During High Upward Price Movements,Stock: SHO, Event:"Sunstone Hotel Investors Reports Results For Third Quarter 2020. IRVINE, Calif., Nov. 5, 2020 /PRNewswire/ – Sunstone Hotel Investors, Inc. (the ""Company"" or ""Sunstone"") (NYSE: SHO), the owner of Long-Term Relevant Real Estate in the hospitality sector, today announced results for the third quarter ended September 30, 2020. Third Quarter 2020 Operational Results (as compared to Third Quarter 2019): Resumption of Hotel Operations: Six of the Company’s 19 hotels were in operation for the entirety of the third quarter of 2020. Six additional hotels opened during the third quarter of 2020, largely in July and August." Expected labels: INC_44 INC_43 INC_61 Predicted labels:INC_31 INC_31 INC_31

![](images/c9a7ff160e75b192b7f927c20a2f5e1fc1f429c292219fe335c60f298362522c.jpg)  
Figure 8: CASE STUDY4:Partially Accurate Predictions During High Upward Price Movements,Stock: BIG EVENT: Big Lots Provides Business Update. COLUMBUS, Ohio, Jan. 13, 2021 /PRNewswire/ –Big Lots, Inc. (NYSE: BIG) today provided an update on results for the fourth quarter of fiscal 2020. On a quarter-to-date basis, the company has achieved a comparable sales increase of approximately $7 . 5 \%$ , reflecting double-digit comps in all merchandise categories other than Seasonal, which is down by a mid-teen percentage due to low levels of Christmas inventory in December, and Food, which is up low single digits. Ecommerce demand quarter-to-date is up approximately $13 5 \%$ . Expected labels: INC_17 INC_11 INC_70 Predicted Labels:INC_11 INC_16 INC_16

![](images/ff523e09a44984cbe13f382a38902cc7e36bfdb32261503956f1659e4696ac1f.jpg)  
Figure 9: CASE STUDY5: Incorrect Prediction During High Upward Price Movement Stock: stock (CYH) Event: Community Health Systems to Participate in Barclays Global Healthcare Conference. FRANKLIN, Tenn.–(BUSINESS WIRE)–Community Health Systems, Inc. (NYSE:CYH) today announced that management will participate virtually in the Barclays Global Healthcare Conference to be held March 9-11, 2021. The investor presentation will begin at $1 { : } 1 5 \ \mathrm { p . m }$ . Eastern time, 12:15 p.m. Central time, on Thursday, March 11, 2021, and will be available to investors via a live audio webcast. A link to the broadcast can be found at the investor relations section of the Companys website, www.chs.net, and a replay will be available using that same link. Expected Labels:INC_18 INC_57 INC_90 Predicted Labels:INC_8 INC_8 INC_8

![](images/ffd678d60b831b3506d4f554232643c0e1a9fa1f436a8242d038202ace88aceb.jpg)  
Price over Time   
Figure 10: CASE STUDY 6: Accurate Prediction During Moderate Downward Price Movement Stock: CGC, Event:"Canopy Growth Announces Changes to Canadian Operations. SMITHS FALLS, ON, Dec. 9, 2020 /PRNewswire/ -Canopy Growth Corporation (""Canopy Growth"" or the ""Company"") (TSX: WEED) (NASDAQ: CGC) today announced a series of Canadian operational changes designed to streamline its operations and further improve margins. Canopy Growth will cease operations at the following sites: St. John’s, Newfoundland and Labrador; Fredericton, New Brunswick; Edmonton, Alberta; Bowmanville, Ontario; as well as its outdoor cannabis grow operations in Saskatchewan. Approximately 220 employees have been impacted as a result of these closures." Expected Labels:DEC_15 DEC_8 DEC_13, Predicted Labels: DEC_9 DEC_10 DEC_10

![](images/f659143ad38d35f65b556cef09686f9dc0a7028cc1f1eef93460e81f9a7e3aae.jpg)  
Figure 11: CASE STUDY 7: Partialy Accurate Prediction During Downward Movement Stock: UAVS Event:36690 VXRT UAVS "Lead Plaintiff Deadline Approaching: Kessler Topaz Meltzer & Check, LLP Announces Deadline in Securities Fraud Class Action Lawsuit Filed Against AgEagle Aerial Systems, Inc.. RADNOR, Pa., April 7, 2021 /PRNewswire/ – The law firm of Kessler Topaz Meltzer & Check, LLP reminds AgEagle Aerial Systems, Inc. (NYSE: UAVS) (""AgEagle"") investors that a securities fraud class action lawsuit has been filed against on behalf of those who purchased or acquired AgEagle securities between September 3, 2019 and February 18, 2021, inclusive (the ""Class Period""). Investor Deadline Reminder: Investors who purchased or acquired AgEagle securities during the Class Period may, no later than April 27, 2021, seek to be appointed as a lead plaintiff representative of the class. For additional information or to learn how to participate in this litigation please contact Kessler Topaz Meltzer & Check, LLP: James Maro, Esq." Expected Labels DEC_27 DEC_51 DEC_55 Predicted labels:DEC_17 DEC_23 DEC_41

![](images/0363af0c51657f6224355019085a72c99a57567b177b93915437c63370365469.jpg)  
Figure 12: CASE STUDY 8: Incorrect Decrement Movement Prediction in Incorrect Direction Stock: NDSN Nordson Corporation Reports Fiscal Year 2020 Third Quarter Results Sales were $\$ 538$ million, a $4 \%$ year-over-year decrease Operating profit was $\$ 112$ million, or $21 \%$ of sales EBITDA was $\$ 148$ million, or $28 \%$ of sales Earnings were $\$ 1.49$ per diluted share Adjusted earnings were $\$ 1.42$ per diluted share, a $12 \%$ decrease from prior year. WESTLAKE, Ohio–(BUSINESS WIRE)–Nordson Corporation (Nasdaq: NDSN) today reported results for the third quarter of fiscal year 2020. For the quarter ended July 31, 2020, sales were $\$ 538$ million, a $4 \%$ decrease compared to the prior years third quarter sales of $\$ 560$ million. The diversity of our end market exposure and broad global customer base contributed to the sales performance in the quarter. Expected Labels:DEC_16 DEC_15 DEC_17 Predicted Labels: INC_7 INC_9 INC_10