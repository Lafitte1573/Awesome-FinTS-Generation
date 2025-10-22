# Enhancing Recurrent Neural Networks For Stock Market Forecasts through PEC-W Framework

Bus¸ra C¸ alıs¸kan ¨   
Computer Engineering   
Istanbul Technical University   
Istanbul, Turkiye ¨   
caliskanb21@itu.edu.tr

Abstract—It is difficult to predict stock prices and many indicators are used for this purpose. This difficulty is even greater in short-term transactions. Regardless of the term, it should definitely be used for buying and selling timing. In this study, we propose an approach to predict this timing. This paper addresses challenges in time series forecasting using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks, commonly employed in stock market forecasting, such as overfitting and extended learning times. To mitigate these issues, the PEC-W preprocessing framework, generally adapted from Convolutional Neural Networks (CNN), is applied to enhance forecasting accuracy without altering model parameters. This approach incorporates normalization and data augmentation to prevent overfitting, and utilizes Discrete Wavelet Transform (DWT) to reduce learning times while preserving temporal characteristics. Aggregation through averaging and mean subtraction further improves data visibility and model accuracy. The effectiveness of the PEC-W technique is validated using Explainable Artificial Intelligence (XAI) method, such as SHAP, which confirm the robustness of the enhanced forecasting approach.

Index Terms—Discrete Wavelet Transform(DWT), Trend Analysis, Seasonality, Time series, Long Short-Term Memory(LSTM), Gated Recurrent Unit(GRU), Data Augmentation, Rolling Window, Normalization,Explainable Artificial Intelligence, Recursive Feature Elimination, SHapley Additive exPlanations(SHAP)

# I. INTRODUCTION

In stock market forecasting, it is difficult to predict price values or movements. For this, technical analysis is used in the financial world, which plays an important role in predicting future price movements based on historical data. While fundamental analysis is safer in stock selection, it is important to refer to technical analysis in trading. In classical technical analysis, chart patterns such as head and shoulders, double tops and triangles are used a lot. While there are many methods used to predict trading time, with advances in machine learning, we can reinterpret it in a more complex and data-driven way.

In this study, we use LSTM and GRU networks machine learning techniques with and without PEC-W preprocessing framework on real financial data obtained from Yahoo Finance to potentially create a trading strategy by focusing only on stock closing price movements. For our analysis, we used data for three different stocks (Amazon.com, Inc.(AMZN), Microsoft Corporation(MSFT) and Apple Inc.(AAPL)) until the end of July 2024. The studies show that there is a strong correlation between the closing price and our predictions, and that the prediction accuracy is higher and the time to result is shorter in scenarios using the PEC-W preprocessing framework.

This paper is organized as follows: the next section gives background information about the methods used and their challenges. Section III details the methodology used. Section IV presents and discusses the results obtained. The paper concludes with Section V.

# II. BACKGROUND INFORMATION

# A. Challenges in Long Short-Term Memory and Gated Recurrent Unit for Time Series Forecasting

Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are two distinct recurrent neural network architectures widely used in the field of deep learning. Both are successfully applied in tasks such as time series analysis and natural language processing. Despite their similar functionalities, there are noteworthy differences between them. LSTM is designed specifically to address long-term dependencies in data. Its core feature is a specialized cell structure that includes three main gates: the forget gate, input gate, and output gate. These gates are employed to update and control the information within the cell. This enables LSTM to effectively capture and retain long-term dependencies. However, due to these features, LSTM often requires more parameters and computational power, potentially resulting in a slower training process. While GRU shares a similar structure with LSTM, it has a simpler cell design. GRU has only two gates: the reset gate and the update gate. This simplified architecture allows GRU to have fewer parameters and generally facilitates a faster training process. However, this simplicity may make GRU less effective in capturing long-term dependencies compared to LSTM. While LSTM is adept at handling long-term dependencies, GRU offers a simpler structure and is typically faster to train.Despite their success in this domain, these architectures face challenges when applied to time series data. One prominent issue is the difficulty in accurately capturing the underlying trends and seasonal characteristics of the data. In the realm of predictions, the capability of both LSTM and GRU to utilize information from previous time steps, while powerful, can sometimes lead to overfitting [4]. This occurs when the models excessively rely on older data by shifting it, hindering their adaptability to the evolving nature of the underlying patterns in the data. This understanding gap highlights the need for preprocessing steps to address these challenges and enhance the overall effectiveness of time series forecasting models based on LSTM and GRU.

# B. Challenges and Benefits of Time Series Data Augmentation

In time series analysis, the temporal aspect often plays a critical role. Data augmentation processes have the potential to distort the inherent temporal properties and weaken the model’s relationship with real-world data. Time series data typically exhibits specific patterns, and data augmentation may disrupt these patterns, influencing the model’s ability to learn accurately. Furthermore, future information is generally unavailable in time series data, and data augmentation introducing knowledge of future data may hinder the model’s real-world performance evaluation. However, augmenting time series data and taking the average during windowing, when added back to the original data, can contribute to the learning process and overall model performance in certain cases. Averaging data during windowing can enhance data diversity when added to the original dataset, assisting the model in better adapting to different patterns [5]. Taking the average of the data can reduce noise within a specific time frame, enabling a smoother training process. This is particularly beneficial for models correcting fluctuations within a specific period. The averaging process can be computed by dividing the sum of the data within a specific window, enhancing numerical stability and moderating the influence of large values.

# C. Time Series Forecasting through Rolling Window

In the realm of forecasting,rolling window refers to the process of selecting data within a specific window size and sliding this window over time to segment the time series into smaller parts. Focusing on data within a defined window size can contribute to the clearer emergence and analysis of seasonal patterns [6]. Seasonality embodies repeated patterns at specific intervals within a time series. On the other hand, trend signifies the long-term upward or downward trajectory observed in a time series.

# D. Normalization through Mean Subtraction in Time Series

Normalization through mean subtraction is a crucial preprocessing step in time series analysis, designed to eliminate the overall trend of the data and improve the learning process of models. This normalization method is commonly referred to as ”mean subtraction” or ”mean intervention.” One of the key advantages of this method is its ability to enable models to more precisely focus on the general trend of time series data [7].

# E. Discrete Wavelet Transform (DWT) in Time Series Analysis

DWT emerges as a powerful tool in time series analysis, offering the capability to discern varying patterns and features by isolating specific frequency components within time series data. In the realm of time series, the decomposition achieved by DWT provides insights into the signal’s characteristics. Lower-frequency components typically capture the overall trend and essential features of the signal. In contrast, higherfrequency components delve into more intricate and abrupt changes within the data. Utilizing fewer bits, we can efficiently represent the low-frequency components of the signal, contributing to a more concise representation that preserves key characteristics [9].

In our contribution to the literature, we analyzed the PECW preprocessing framework of Kulaglic et al. [1] without the Error Compensation part and integrated it with Recurrent Neural Networks (RNNs). The software we developed can be accessed through the citation provided [10]. Our work addresses the following key aspects while maintaining RNNs’ existing model parameters:

1) We demonstrate its ability to manage overfitting, normalization, and data augmentation techniques during preprocessing while preserving critical temporal features.

2) We prove that Discrete Wavelet Transform (DWT) to reduce extended learning times, ensuring that temporal characteristics are preserved.   
3) We substantiate that aggregating windows through averaging improves the visibility of seasonal and trend features, thereby enhancing the accuracy of model training.   
4) We utilize Explainable Artificial Intelligence (XAI) methods, specifically SHAP, to validate and demonstrate the effectiveness of the PEC-W preprocessing techniques.

# III. METHODOLOGY

We have applied the predictive performance of the previously used LSTM and GRU models without altering any of the existing parameters.

Our experiment is divided into two parts:

1) Training and predicting with standard Recurrent Neural Networks (RNNs) without preprocessing, using a 60-day rolling window of stock market data.   
2) Training and predicting with RNNs that have been enhanced with the PEC-W preprocessing framework [1].

The steps involved in the PEC-W preprocessing framework for RNNs, aimed at predicting the adjusted close price, are as follows:

(a) Feature Selection: Identify an additional feature with a high correlation to the adjusted close price.   
(b) Application of PEC-W Preprocessing: The PEC-W preprocessing was applied to the two selected features, involving several steps. First, a backward rolling window was implemented for temporal data segmentation, which allowed for effective handling of sequential information. To enhance model training, the amount of data was increased. Subsequently, the sequences were normalized by subtracting the mean to ensure consistent scaling. Finally, Discrete Wavelet Transform (DWT) was applied for feature extraction, capturing important patterns from the sequences.   
(c) Combine the processed features before feeding them into the learning model.

# A. Dataset

The dataset utilized in this study includes stock market data from three companies: Amazon.com, Inc.(AMZN), Microsoft Corporation(MSFT) and Apple Inc.(AAPL), which were downloaded online from Yahoo Finance [2]. The adjusted close prices of these companies were employed for analysis. The choice of adjusted close prices is due to their ability to account for corporate actions such as stock splits and dividends, providing a more accurate reflection of the stock’s value over time [8]. Feature highly correlated with the adjusted close prices were selected for the analysis. Prior to training, both features underwent PEC-W preprocessing and were concatenated side by side. For the training phase, features from January 1, 2010, to December 31, 2021, were used; for validation, features from January 1, 2022, to December 31, 2023, were selected; and for testing, features from January 1, 2024, to July 29, 2024, were utilized.

# B. Feature Selection

In addition to the ’Adj Close’ price feature, it is necessary to perform a correlation analysis to add new features. Recursive Feature Elimination (RFE) is commonly employed to identify features with the highest correlation with the ’Adjusted

Close’ price in stock market data. Typically, the ’Open’ price exhibits the highest correlation with the ’Adjusted Close’ price.Normally, the Volume feature would exhibit a high correlation with the adjusted close price. However, since the selected data involves stocks prone to manipulation, a higher correlation has been observed with the Open price instead.Due to the public availability of common datas, we were compelled to use the Open prices in our analysis.

# C. Data Pre-processing

As illustrated in Fig. 1, during the rolling window process labeled ”a,” we calculated the averages of the windows labeled ”b” and concanated them based on the inherent characteristics of the data, all while preserving the temporal relationships. Subsequently, we focused on the data’s trend and seasonality features by subtracting them from the averages of the windowed data.where the parameter a is mandated to be consistently power of 2, as the Discrete Wavelet Transform (DWT) operates based on Fourier logic. In order to expedite the learning process without sacrificing temporal relationships, we specifically employed the Discrete Wavelet Transform. The given statement can be reformulated into a methodology form as follows: The methodology of Fig. 1 for predicting data at time $t + 1$ is depicted in:

$$
\begin{array} { r } { x _ { p } ( t + 1 ) = \left[ \begin{array} { c } { x ( t ) , x ( t - 1 ) , . . . , x ( t - a + 2 ) , x ( t - a + 1 ) , } \\ { \arg ( x ( t ) , . . . , x ( t - b + 1 ) ) , } \\ { \arg ( x ( t - b ) , . . . , x ( t - 2 b + 1 ) ) , } \\ { . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . } \\ { . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . } \\ { \arg ( x ( t - a b + 2 a ) , x ( t - a b + a + 1 ) ) , } \\ { \arg ( x ( t - a b + a ) , x ( t - a b + 1 ) ) } \end{array} \right] } \end{array}
$$

In order to utilize our data more effectively and extract additional information, we will apply windowing and weekly averaging processes. We planned to use $a = 4$ windows for windowing. This means that we will divide our data set into four different windows.To calculate weekly averages, we will take the weekly average of the daily data within each window. Since there are typically 5 trading days in a week in stock market data, we will set $b = 5 .$ .For the prediction process, we will set the initial time point as $t = 1 9$ and predict the next time point as $t + 1 = 2 0$ . In this prediction process, the $x _ { p } ( 2 0 )$ vector will be calculated using the following formula:

$$
\begin{array} { r } { x _ { p } ( 2 0 ) = \left[ \begin{array} { c } { x ( 1 6 ) , x ( 1 7 ) , x ( 1 8 ) , x ( 1 9 ) , } \\ { \arg ( x ( 1 5 ) , x ( 1 6 ) , x ( 1 7 ) , x ( 1 8 ) , x ( 1 9 ) ) , } \\ { \arg ( x ( 1 4 ) , x ( 1 3 ) , x ( 1 2 ) , x ( 1 1 ) , x ( 1 0 ) ) , } \\ { \arg ( x ( 9 ) , x ( 8 ) , x ( 7 ) , x ( 6 ) , x ( 5 ) ) , } \\ { \arg ( x ( 4 ) , x ( 3 ) , x ( 2 ) , x ( 1 ) , x ( 0 ) ) , } \end{array} \right] } \end{array}
$$

Finally, the two preprocessed features, ’Adj Close’ and ’Open Price,’ are combined for training purposes.

# D. Model Selection

Based on result of Varadharaja et al. [3], the model parameters that yielded the best result were selected, and a sequential model was constructed using Keras. The model comprises an initial Recurrent Neural Networks’ layer with 64 units and return sequences is True, which processes sequences of input data. This layer is followed by a dropout layer with a dropout rate of 20 percent to mitigate overfitting. A subsequent LSTM and GRU layer with 32 units are included to capture more complex patterns within the data. The model is trained over 200 epochs with a batch size of 64. To monitor performance and prevent overfitting, an EarlyStopping callback is employed, which monitors the validation loss and restores the best weights if no improvement is observed over 10 consecutive epochs. Training data is used along with validation data to evaluate the model’s performance during training. The output layer consists of a dense layer with a single unit to produce the final prediction. The model is compiled using the Adam optimizer and Mean Squared Error (MSE) as the loss function.

# E. Workspace

This project was conducted using Google Colab. This is a cloud-based development environment designed for programmers working with the Python programming language.

# IV. EXPERIMENT RESULTS

This section explores the effectiveness of our models in predicting stock prices. The tables (Table I, II, III) in this section compare various model performance metrics for each stock and show the results of four different models (PECW LSTM, LSTM, PEC-W GRU and GRU. The performance metrics include Root Mean Square Error (RMSE), Mean Percentage Error (MAPE), Mean Square Error (MSE), Mean Absolute Error (MAE), and R-Square $( \mathbb { R } ^ { 2 } )$ .

TABLE I: Apple Stock’ Adj Close Prediction Comparison   

<table><tr><td>Model</td><td>RMSE</td><td>MAPE (%)</td><td>MSE</td><td>MAE</td><td>R²</td></tr><tr><td>PEC-WLSTM</td><td>3.294</td><td>1.174</td><td>10.854</td><td>2.369</td><td>0.975</td></tr><tr><td>LSTM</td><td>5.550</td><td>2.053</td><td>30.895</td><td>3.895</td><td>0.911</td></tr><tr><td>PEC-WGRU</td><td>3.375</td><td>1.194</td><td>11.391</td><td>2.417</td><td>0.974</td></tr><tr><td>GRU</td><td>12.09</td><td>2.042</td><td>146.173</td><td>8.364</td><td>0.672</td></tr></table>

TABLE II: Amazon Stock’ Adj Close Prediction Comparison   

<table><tr><td>Model</td><td>RMSE</td><td>MAPE (%)</td><td>MSE</td><td>MAE</td><td>R²</td></tr><tr><td>PEC-WLSTM</td><td>3.091</td><td>1.275</td><td>9.556</td><td>2.307</td><td>0.854</td></tr><tr><td>LSTM</td><td>5.465</td><td>2.074</td><td>29.871</td><td>3.584</td><td>0.803</td></tr><tr><td>PEC-W GRU</td><td>3.178</td><td>1.314</td><td>10.104</td><td>2.39</td><td>0.848</td></tr><tr><td>GRU</td><td>6.447</td><td>2.523</td><td>41.566</td><td>4.372</td><td>0.72</td></tr></table>

TABLE III: Microsoft Stock’ Adj Close Prediction Comparison   

<table><tr><td>Model</td><td>RMSE</td><td>MAPE (%)</td><td>MSE</td><td>MAE</td><td>R²</td></tr><tr><td>PEC-WLSTM</td><td>5.830</td><td>1.05</td><td>33.99</td><td>4.61</td><td>0.901</td></tr><tr><td>LSTM</td><td>10.901</td><td>1.868</td><td>118.851</td><td>7.638</td><td>0.733</td></tr><tr><td>PEC-WGRU</td><td>6.40</td><td>1.17</td><td>41.03</td><td>5.05</td><td>0.871</td></tr><tr><td>GRU</td><td>12.090</td><td>2.042</td><td>146.173</td><td>8.364</td><td>0.672</td></tr></table>

In general, the PEC-W LSTM and PEC-W GRU models outperform the other models with lower error rates and higher explanatory values. The LSTM and GRU models, on the other hand, underperform with higher error rates and lower R² values. According to the RMSE value, for all stocks, PECW LSTM and PEC-W GRU models have lower error rates and better forecasting performance than other models. When we examine the MAPE values, it shows that the percentage error rate of the forecasts of these models is lower than the other models and provides higher accuracy. MSE values indicate that the mean squared error of the predictions of these models is lower than the other models and contains less error. The MAE value indicates that the mean absolute error of the predictions of these models is lower than the other models. The R² value indicates that these models explain the variance of the data points better and provide a higher explanatory power.

![](images/adce4503e9cf46732bb74d368d7c61c14dc76a8e4baf3ebae4abfa2c5cc97a5e.jpg)  
Fig. 1: Architecture of Model

TABLE IV: Execution Times for LSTM and GRU Models   

<table><tr><td>Model</td><td>Dataset</td><td>Normal</td><td>PEC-WFramework</td></tr><tr><td>LSTM</td><td>AAPL</td><td>32 min and 34 sec</td><td>3 min and 40 sec</td></tr><tr><td>GRU</td><td>AAPL</td><td>3 min and 16 sec</td><td>2 min and 47 sec</td></tr><tr><td>LSTM</td><td>AMZN</td><td>26 min and 29 sec</td><td>3 min and 20 sec</td></tr><tr><td>GRU</td><td>AMZN</td><td>3 min and 27 sec</td><td>1 min and 54 sec</td></tr><tr><td>LSTM</td><td>MSFT</td><td>22 min and 20 sec</td><td>3 min and 8 sec</td></tr><tr><td>GRU</td><td>MSFT</td><td>3 min and 48 sec</td><td>3 min and 30 sec</td></tr></table>

Table IV compares the execution times of the LSTM and GRU models on different data sets (AAPL, AMZN, MSFT) under the normal and PEC-W framework. The table compares the execution times of each model and data set and examines the impact of the PEC-W framework on execution times. According to these values, the PEC-W framework tends to reduce processing times in general. Especially for the LSTM models, a significant reduction in processing time is achieved for the stock data sets. Improvements in processing time are also observed for the GRU models, but the improvement is generally smaller. The AMZN dataset shows the largest trading time improvements under the PEC-W framework for both the LSTM and GRU models. In the MSFT dataset, the improvement in processing times is more limited. In conclusion, the PEC-W framework improves performance by significantly reducing processing times, especially on large datasets, but the rate of improvement varies depending on the model and dataset.

In the context of APPLE stock, an examination of Fig. 2 reveals that the PEC-W pre-processed model accurately captures actual stock price movements, with predicted values closely following actual values, particularly during upward and downward trends. The model effectively captures overall trends, including the major upward trend from March 2024 to June 2024, with minimal overfitting observed. In contrast, the standard model (Fig. 2.b and 2.d) shows some deviations, particularly in capturing rapid changes. While it captures the general trend, it fails to accurately predict certain fluctuations, such as the sharp rise in February 2024. Some evidence of overfitting is visible, with erratic deviations and a less smooth trend following compared to the preprocessed model.

For AMAZON stock, Fig. 3 reveal that the PEC-W preprocessed model exhibits a high degree of conformity between actual and predicted values. The model accurately captures overall trends and price fluctuations, particularly during periods of sudden changes from March 2024 to June 2024. Notably, there are no signs of overfitting, suggesting that PEC-W pre-processing has improved the model’s performance and reduced overfitting risk. In contrast, the standard model’s predictions (Fig. 3.b and 3.d) are less accurate, with significant deviations observed from January 2024 to February 2024. Although the model’s performance improves later, it still struggles to capture some fluctuations. Mild signs of overfitting are visible, indicating potential overfitting to training data. However, the model partially adapts to general trends in April and May.

Regarding MICROSOFT stock, Fig. 4 reveal that the PECW pre-processed model has successfully captured the significant features of the data. Notably, it has accurately predicted both increasing and decreasing trends after March 2024. The difference between actual and predicted values is minimal, indicating the model’s good performance. In contrast, the standard models (Fig. 3.b and 3.d) exhibit deviations in predictions during significant fluctuations, failing to accurately predict the large drop in January 2024. The difference between actual and predicted values is greater, indicating slightly lower performance. The standard LSTM model exhibits more overfitting compared to the PEC-W pre-processed model.

# A. Discussion

The PEC-W pre-processing technique significantly enhances the performance of both LSTM and GRU models across different stocks.Overall, the PEC-W framework yielded better results with both GRU and LSTM models. In LSTM, it reduced the computation time by a factor of 10, while in GRU, it showed a notable improvement in accuracy by nearly $100 \%$ . It reduces errors, increases R-squared values, and improves trend capturing while minimizing overfitting. The execution times for models are also substantially reduced, making them more efficient. Overall, PEC-W pre-processing proves to be a valuable enhancement for predictive accuracy and computational efficiency in stock price prediction models.

As shown in Fig. 5, the PEC-W Preprocessing with LSTM model has a more evenly distributed feature importance, suggesting a more balanced model that relies on multiple features for its predictions. In contrast, the Normal LSTM model has higher top SHAP values, indicating stronger contributions from its top features but also suggesting potential over-reliance on a few features.

![](images/eb4a8494eb59df43c953b7aa7e443b0ffd32ccf58abf46553672f4ab9e427b8e.jpg)  
Fig. 2: APPLE stock prediction

![](images/8e300d6944b5a39cb443cc5543af502cacfec37b75e0adfa59c18a6b2f5073dc.jpg)  
Fig. 3: AMAZON stock prediction

# V. CONCLUSION

Forecasting stock prices remains a complex challenge, especially in the context of short-term trading where the timing of buying and selling is critical. Predicting the timing of a stock’s bid-ask spread is also crucial in medium- and long-term trades. This paper addresses these challenges by proposing a novel approach using LSTM and GRU networks developed by the PEC-W preprocessing framework.

![](images/81618080a8bb682fe967496a6686e879517fdbe91992fbd4b084757fe9e1ce46.jpg)  
Fig. 4: MICROSOFT stock prediction

The empirical validation performed in this study verifies the robustness and effectiveness of the PEC-W advanced forecasting approach. The study uses real financial data from Yahoo Finance, focusing on the closing prices of Apple, Amazon and Microsoft until July 2024. The results show that the incorporation of the PEC-W pre-processing framework leads to higher forecast accuracy and reduced trading times, highlighting its potential in developing effective trading strategies. This research highlights the value of advanced preprocessing techniques in improving stock market predictions and supports the integration of machine learning methods for more accurate and efficient trading decisions.

![](images/b37d915e95469c1772f9404c93963f3b127999b4a874153f2330d9766f7db2db.jpg)  
Fig. 5: SHAP mean values of PEC-W LSTM (a) and standard LSTM (b) in APPLE stock

In future research, stocks with lower susceptibility to manipulation will be selected to achieve more accurate results. Additionally, the analysis will be expanded by incorporating variables such as volume, statistical features like the standard deviation of data, and applying various feature transformations.

# REFERENCES

[1] A. Kulaglic and B. B. Ustundag, ”Stock Price Prediction Using Predictive Error Compensation Wavelet Neural Networks,” Computers, Materials & Continua, vol. 68, no. 3, pp. 1-15, Sep. 2021.   
[2] Yahoo Finance, “Apple Inc. (AAPL), Amazon.com, Inc. (AMZN) and Microsoft Corporation (MSFT) ” https://finance.yahoo.com/.   
[3] V. Varadharajan, N. Smith, D. Kalla, G. R. Kumar, F. Samaah, and K. Polimetla, ”Stock Closing Price and Trend Prediction with LSTMRNN,” Journal of Artificial Intelligence and Big Data, vol. 1, no. 1, pp. 1-10, 2024, doi: 10.31586/jaibd.2024.877.   
[4] Z. Chen, M. Ma, T. Li, H. Wang, and C. Li, ”Long sequence time-series forecasting with deep learning: A survey,” Information Fusion, vol. 97, 2023, Art. no. 101819, doi: 10.1016/j.inffus.2023.101819.   
[5] G. Iglesias, E. Talavera, A. Gonz ´ alez-Prieto, A. Mozo, and S. G ´ omez- ´ Canaval, ”Data Augmentation techniques in time series domain: a survey and taxonomy,” Neural Computing and Applications, vol. 35, pp. 10123- 10145, Mar. 2023.   
[6] H. Li and C. Wang, ”Combining first prediction time identification and time-series feature window for remaining useful life prediction of rolling bearings with limited data,” Proceedings of the Institution of Mechanical Engineers, Part C: Journal of Mechanical Engineering Science, vol. 238, no. 2, Jan. 2023, doi: 10.1177/1748006X221147441.   
[7] Y. Chen, S. Liu, J. Yang, H. Jing, W. Zhao, and G. Yang, ”A Joint Time-Frequency Domain Transformer for Multivariate Time Series Forecasting,” Neural Networks, vol. 106, 2024, doi: 10.1016/j.neunet.2024.106334.   
[8] ”Understanding Adjusted Closing Price: A Key Metric in Financial Markets,” FasterCapital, 9 Jun 2024. [Online]. Available: https://fastercapital.com/content/Understanding-Adjusted-ClosingPrice–A-Key-Metric-in-Financial-Markets.html.   
[9] A. H. Amshi and R. Prasad, ”Time series analysis and forecasting of cholera disease using discrete wavelet transform and seasonal autoregressive integrated moving average model,” Scientific African, vol. 20, e01652, Jul. 2023. [Online]. Available: https://doi.org/10.1016/j.sciaf.2023.e01652.   
[10] [1] B. Caliskan, ”Enhancing Recurrent Neural Networks for Stock Market Forecasts through PEC-W Framework,” Zenodo, Sep. 22, 2024. [Online]. Available: https://doi.org/10.5281/zenodo.15079527