# A Novel Wavelet based Generative Model for Time Series Prediction

1st Chaofan Dai   
National Key Laboratory of   
Information Systems Engineering   
National University of Defense   
Technology   
Changsha, China   
cfdai@nudt.edu.cn   
$2 ^ { \mathrm { n d } }$ Xiaoguang Yuan\*   
National Key Laboratory of   
Information Systems Engineering   
National University of Defense   
Technology   
Changsha, China   
yxg_1008@163.com   
$3 ^ { \mathrm { r d } }$ Zongkai Tian\*   
Beijing Institute of Computer   
Technology and Application   
Beijing, China   
tzk_bit@163.com   
4th Xinyue Hu   
Beijing Institute of Computer   
Technology and Application   
Beijing, China   
429820303@qq.com   
$5 ^ { \mathrm { t h } }$ Zhen Luan   
Beijing Institute of Computer   
Technology and Application   
Beijing, China   
luanz917@126.com 6th Youchen Wang   
Beijing Institute of Computer   
Technology and Application Beijing, China   
Youchen_Wang@hotmail.com

Abstract—Generative models have become an exciting area of research in recent years. Generative Adversarial Networks (GANs) and Denoising Diffusion Probabilistic Models (DDPMs) have been utilized in various data augmentation applications. However, generative models can learn high-dimensional features of data through adversarial learning, making them suitable for nonlinear and nonstationary time series analysis, such as stock market prediction, high-frequency trading, and ocean current forecasting. In this paper, the researchers focus on using a wavelet-based GAN to predict stock market prices by generating synthetic stock market price trends. Historical stock price data from 2014 to 2024 is used for our experiments, and the results show that the wavelet-based GAN outperforms deep learning baseline models..

# Keywords—Stock Market Prediction, Data Augmentation, Generative Adversarial Network, Wavelet Transform

# I. INTRODUCTION

The stock price market is a complex time series system with nonlinear and non-stationary characteristics. These factors make it difficult to capture its linear and nonlinear behaviors, making accurate predictions of stock price trends challenging. Traditional regression algorithms, such as ARIMA and ARX, can capture the linear characteristics of the sequence more accurately but struggle to capture the nonlinear characteristics of complex financial time series, resulting in low prediction accuracy. Nonlinear sequence models, such as NARX, NARMAX, and wavelet-related methods, can capture some low-dimensional characteristics of the sequence and effectively improve the prediction accuracy of nonlinear sequences. Although these models are interpretable, their improvement over regression algorithms is limited, and they cannot capture long-range dependencies. With the emergence and popularization of deep learning techniques, such as RNN, LSTM, and GRU models, these algorithms can effectively extract high-dimensional characteristics from sequences based on extensive training, providing more accurate predictions. However, they have high training requirements and lack interpretability. In recent years, the advent of generative models has introduced a new approach to sequence prediction. The learning mechanism based on adversarial training between two deep learning models can further capture the high-dimensional characteristics of nonlinear non-stationary sequences and significantly improve the accuracy of sequence predictions.

# II. RELATED WORK

# A. Deep Learning Methods

Deep learning methods have become crucial in stock market prediction due to their ability to handle complex, large-scale datasets and enhance prediction accuracy by capturing intricate patterns. Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks, are effective for processing sequential data and retaining long-term dependencies. For example, Soni et al. [1] demonstrated that LSTM models, with careful data preprocessing and feature selection, significantly improved prediction performance. Similarly, Dara et al. [2] highlighted the strengths of combining RNNs with Convolutional Neural Networks (CNNs) to handle large datasets and capture both temporal and spatial features.

Convolutional Neural Networks (CNNs) are also prominent in stock market prediction, particularly for spatial feature extraction. Vargas et al. [3] explored the use of CNNs and RNNs to process financial news and technical indicators, showing that CNNs are effective in capturing semantic information while RNNs excel in modeling temporal characteristics. Additionally, hybrid models that integrate various deep learning techniques have shown promise in improving accuracy. For instance, Liu and Long [4] proposed a model combining LSTM with empirical wavelet transforms and extreme learning machines, achieving superior prediction accuracy. Incorporating sentiment analysis, as Li et al. [5] did, further refines predictions by considering investor sentiment from social media. Lastly, Jiang [6] reviewed the application of large-scale pre-trained models like BERT and T5, emphasizing their ability to handle complex financial datasets and produce reliable forecasts.

# B. Generative Adversarial Networks

Generative Adversarial Networks (GANs) have become a valuable tool in stock market prediction by generating synthetic data that mimics real-world scenarios, thereby improving prediction accuracy. For example, Li et al. [7] developed a GAN model that integrates social media text mining, effectively capturing the impact of investor sentiment on stock prices, which led to superior predictive performance. He and Kita [8] optimized GAN-LSTM models using Genetic Algorithms, demonstrating enhanced accuracy in stock price forecasting. Zhang et al. [9] introduced a GAN with an LSTM generator and an MLP discriminator to predict S&P 500 stock prices, achieving better accuracy than traditional models. Similarly, Wang and Huang [10] showed that GANs outperform LSTM and EGARCH models in predicting stock price volatility.

Hybrid models have also shown promise, as Polamuri et al. [11] combined GANs with reinforcement learning and Bayesian optimization to improve prediction accuracy. Lin et al. [12] used a GAN model with GRU and CNN, incorporating FinBERT for sentiment analysis, which resulted in better multi-step predictions. Zhang et al. [13] further enhanced predictions by integrating sentiment data into a Conditional GAN (CGAN) model, outperforming traditional LSTM models. Staffini [14] proposed a Deep Convolutional GAN (DCGAN) that excelled in forecasting stock prices on the FTSE MIB index. Labiad et al. [15] and [16] demonstrated the effectiveness of GANs in improving intraday stock predictions and predicting extreme market events, respectively, using advanced GAN frameworks.

# III. METHODOLOGY

# A. Wavelet Transform

The primary objective in determining the stock market system for our proposed method involves wavelet transforms, which decompose nonlinear time series into a sequence of channels to simulate local and global features using polynomial and kernel functions. In this paper, the researchers employ the DB2 wavelet function, which is applied to approximate specific instances to represent the frequency information of the stock market system. Wavelet transform is used to resolve and decompose the different levels of frequencies of a series. Stock market series are usually treated as discrete series, and the mother wavelet of Mallat ’ s formulation that defines the discrete wavelet transform (DWT) is shown below:

$$
\begin{array} { r } { \varphi _ { m , n } ( t ) = \frac { 1 } { \sqrt { s _ { 0 } ^ { m } } } \varphi \left( \frac { t - n u _ { 0 } s _ { 0 } ^ { m } } { s _ { 0 } ^ { m } } \right) } \end{array}
$$

In this equation, $\varphi ( t )$ represents the mother wavelet function, $s _ { 0 }$ is the scale factor that determines the dilation of the wavelet function, and $u _ { 0 }$ is the translation factor that controls the position of the wavelet function along the time axis. The parameters $m$ and $n$ correspond to scale and translation, respectively.

In many practical applications, a simplified form of the wavelet function is used in DWT, assuming the scale factor $s _ { 0 } = 2$ and the translation factor $u _ { 0 } { = } 1$ . The wavelet function can then be expressed as:

$$
\varphi _ { m , n } ( t ) = 2 ^ { \frac { - m } { 2 } } \varphi ( 2 ^ { - m } t - n )
$$

This form is more concise and is well-suited for computational implementation. It applies the wavelet function to the signal by scaling (determined by $2 ^ { - m }$ ) and translating (determined by $n$ ) along the time axis at different scales and positions.

The core of the Discrete Wavelet Transform lies in calculating the wavelet coefficients, which represent the characteristics of the signal at different scales and translations. The formula for computing the DWT coefficients is:

$$
W _ { f , m , n } = 2 ^ { \frac { - m } { 2 } } { \sum } _ { 0 } ^ { N - 1 } \varphi ( 2 ^ { - m } i - n ) f _ { i }
$$

In this formula, $W _ { f , m , n }$ denotes the wavelet coefficient of the signal $f ( t )$ at scale $m$ and translation $n$ . The term $f _ { i }$ represents the discrete sampled values of the signal, and $\varphi ( 2 ^ { - m } i - n )$ is the value of the wavelet function at scale $\spadesuit$ m and translation $n$ .

After performing wavelet decomposition on the data sequence, the result of each layer of decomposition is that the low-frequency signal obtained from the previous decomposition is separated into low-frequency and highfrequency series. Next, we perform wavelet reconstruction to predict the source signal. Firstly, we decompose the original sequence to obtain the wavelet coefficients for each layer. Secondly, we establish ARMA models for the wavelet coefficients of each layer and predict those coefficients. Lastly, we reconstruct the data using the predicted wavelet coefficients.

# B. Network Architecture

We use the Wasserstein GAN with a Gradient Penalty (GP) framework to simulate the sequence features. By using Wasserstein distance to optimize the traditional GAN loss, it overcomes the instability issues associated with traditional GAN training. More specifically, the discriminator of Deep Convolutional GANs (DCGANs) cannot provide the generator with efficient feedback for its training, and the Euclidean distance significantly increases the training computational complexity. The Wasserstein distance addresses these problems.

$$
\begin{array} { r } { L = \underset { \tilde { x } \sim \mathbb { P } _ { g } } { \mathbb { E } } \big [ D ( \widetilde { x } ) \big ] - \underset { x \sim \mathbb { P } _ { r } } { \mathbb { E } } \big [ D ( x ) \big ] } \\ { + \lambda \underset { \hat { x } \sim \mathbb { P } _ { \hat { x } } } { \mathbb { E } } \Big [ \big ( \| \nabla _ { \hat { x } } D ( \widehat { x } ) \| _ { 2 } - 1 \big ) ^ { 2 } \Big ] } \end{array}
$$

The loss function combines the original Wasserstein loss (which encourages the model to differentiate between real and fake data) with a gradient penalty term (which regularizes the model to improve stability). The gradient penalty enforces the Lipschitz constraint on the critic, a key requirement for the framework. This model architecture creates a stable balance between the training of the generator and the discriminator. Additionally, the Wasserstein loss creates continuous and differentiable linear gradients that are effective in addressing the vanishing gradient problem.

Additionally, weight clipping has its drawbacks in training. As the clipping parameter increases, the training process becomes slower, preventing the discriminator from reaching Nash equilibrium. In contrast, a small clipping parameter can lead to vanishing gradients

# IV. EXPERIMENTS

In this section, we evaluate the performance of different models on the task of stock price prediction. The models assessed include GRU, LSTM, GAN, and Wavelet-GAN. Each model is trained and tested on the same dataset to ensure a fair comparison. The dataset used consists of historical stock prices, and the evaluation metrics include Mean Squared Error (MSE) and Root Mean Squared Error (RMSE), which are chosen for their relevance in measuring the accuracy of continuous predictions, with RMSE providing a more interpretable error metric in the same units as the stock prices. To provide a more comprehensive assessment of model performance, we also include Mean Absolute Error (MAE) and the Coefficient of Determination $( \mathbb { R } ^ { 2 } )$ as additional metrics. MAE reflects the average absolute difference between predicted and actual values, offering insight into overall accuracy, while $\mathrm { R } ^ { 2 }$ indicates the proportion of variance in the dependent variable that is predictable from the independent variables, highlighting the model's explanatory power.

# A. Dataset

The experiments were conducted on historical stock price data, which included daily closing prices over a period of several years. The data was split into training and testing sets, with the model trained on data up to the end of 2023 and tested on data from 2024 onward. The data preprocessing steps included normalization using MinMaxScaler to ensure the model inputs were scaled to a range of [0, 1].

# B. Experimental Settings

Experiments were conducted using TensorFlow version 2.13.0 and Keras version 2.13.1, with a system equipped with an NVIDIA 4090 GPU and $3 2 \mathrm { \ G B }$ of memory. In the experimental setup, both the GRU and LSTM models were designed with two recurrent layers, consisting of 128 and 64 units each, and incorporated two dense layers. These models were trained using the Adam optimizer with a learning rate of 0.0001, a batch size of 128, and for 100 epochs. The GAN model featured a generator composed of three GRU layers with 1024, 512, and 256 units, and a discriminator comprising three 1D convolutional layers with 32, 64, and 128 channels. This model was trained using the Adam optimizer with a learning rate of 0.00016 over 100 epochs. The Wavelet-GAN employed a generator with two GRU layers (256 and 128 units) and a discriminator structure similar to that of the GAN, also optimized with Adam at a learning rate of 0.0001 for 100 epochs. The configurations of each model were carefully selected to effectively capture the temporal dependencies in the data, while maintaining consistent training parameters to ensure comparability across all models.

# C. Baseline Methods

To evaluate the performance of our models, we compare them against several baseline methods. A detailed description of each method is provided below:

a) GRU (Gated Recurrent Unit): GRU is a simplified RNN variant that combines the forget and input gates, allowing it to efficiently capture long-term dependencies in sequential data. In stock price prediction, GRUs are particularly useful for modeling temporal patterns, enabling the prediction of future price movements based on historical data.

b) LSTM (Long Short-Term Memory): LSTM addresses the vanishing gradient problem in traditional RNNs by using memory cells and gates, making it highly effective for processing long sequences. In stock prediction, LSTMs excel at retaining important information over time, which is crucial for forecasting trends and price movements over extended periods.

c) GAN (Generative Adversarial Network): GANs consist of a generator and a discriminator trained in opposition. In stock prediction, GANs can be used to generate synthetic stock price data that mimics real market conditions, aiding in the development of robust predictive models by enhancing data diversity and quality.

d) Wavelet-GAN: Wavelet-GAN incorporates wavelet transforms to capture both time and frequency information, enhancing data representation in stock price prediction. This approach allows for more accurate modeling of complex market dynamics, improving the prediction of stock prices by better capturing intricate temporal patterns.

# D. Experimental Results

In this section, we present the results obtained from our experiments using GRU, LSTM, GAN, and Wavelet-GAN models on the stock price prediction task. The performance of each model was evaluated using four metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the Coefficient of Determination (R²). The predictions were compared against real stock price data, and the results are summarized in both quantitative metrics and visual representations.

TABLE I. PERFORMANCE METRICS FOR STOCK PRICE PREDICTIONUSING DIFFERENT MODELS  

<table><tr><td rowspan="2">Model</td><td colspan="4">Performance</td></tr><tr><td>MSE</td><td>RMSE</td><td>MAE</td><td>R²</td></tr><tr><td>GRU</td><td>73.23</td><td>8.56</td><td>6.24</td><td>0.85</td></tr><tr><td>LSTM</td><td>23.56</td><td>4.85</td><td>3.83</td><td>0.93</td></tr><tr><td>GAN</td><td>9.07</td><td>3.01</td><td>2.98</td><td>0.97</td></tr><tr><td>WaveNet-</td><td>2.97</td><td>1.72</td><td>1.59</td><td>0.99</td></tr></table>

The performance comparison of different models for stock price prediction, as shown in Table 1, reveals significant variations in their accuracy. The GRU model, known for its simplicity, exhibited the highest MSE of 73.23, RMSE of 8.56, and an MAE of 6.24, with an $\mathrm { R } ^ { 2 }$ of 0.85, suggesting limited effectiveness in capturing the complex temporal dependencies inherent in stock price data. The LSTM model, specifically designed to handle long-term dependencies, demonstrated marked improvement with a lower MSE of 23.56, RMSE of 4.85, MAE of 3.83, and a higher $\mathrm { R } ^ { 2 }$ of 0.93, indicating enhanced predictive accuracy.

The GAN model further improved the results with an MSE of 9.07, RMSE of 3.01, MAE of 2.98, and $\mathrm { R } ^ { 2 }$ of 0.97, showing its capability to generate more precise predictions through adversarial learning. The Wavelet-GAN model outperformed all other models, achieving the lowest MSE of 2.97, RMSE of 1.72, MAE of 1.59, and the highest $\mathrm { R } ^ { 2 }$ of 0.99, highlighting its superior ability to capture intricate patterns and complex temporal dependencies in stock price data.

![](images/d57284ef46fd196bed0744839ecd0abf271074852bd63aa715539c1e319e5674.jpg)  
Fig. 1. GRU Model Prediction Performance.

![](images/09c006e96810866cb973b34990a68b89b20700865f31bd766aee42529d353ecf.jpg)  
Fig. 2. LSTM Model Prediction Performance.

![](images/a13fade2d32f893072f77596163126deed617033251275084c02e463dfd0fc81.jpg)  
Fig. 3. GAN Model Prediction Performance.

![](images/02a4db9330a8db980e6bd968055a502842325eac47efedeeb913f45ecc4ee362.jpg)  
Fig. 4. Wavelet-GAN Model Prediction Performance.

Figures 1-4 illustrate the prediction performance of the GRU, LSTM, GAN, and Wavelet-GAN models in stock price forecasting. The GRU model exhibited lower alignment with actual price trends, particularly during the early to mid2022 period, where it failed to accurately capture significant market fluctuations, resulting in substantial prediction errors. However, the model performed relatively well during more stable market periods, such as from late 2023 to late 2021, where it closely followed the overall price trend. During the market downturn from late 2022 to early 2023, the GRU model struggled to reflect the downward trends, highlighting its limitations in handling complex market dynamics. These findings are consistent with the model's higher MSE and RMSE values, indicating its inadequacy in capturing intricate temporal dependencies.

In contrast, the LSTM model demonstrated better predictive accuracy across most time periods. Notably, during the mid-2022 to early 2023 period, characterized by significant market volatility, the LSTM model effectively captured the decline in prices, showcasing its strength in managing complex temporal dependencies. Although there were still some deviations during the late 2021 to early 2022 period, the LSTM model generally outperformed the GRU model. Additionally, the LSTM model exhibited strong performance during the market rebound from late 2023 to early 2024, accurately predicting the upward trend. These results suggest that the LSTM model offers greater stability and accuracy for long-sequence forecasting, particularly in volatile market conditions.

The GAN model, leveraging adversarial learning, showed a marked improvement over both the GRU and LSTM models. In periods of sharp market movements, such as mid2022 and late 2023, the GAN model effectively captured the downward and rebound trends. While the GAN model exhibited some prediction errors during extreme fluctuations, particularly in late 2022 and early 2024, the overall prediction errors were significantly reduced. This reflects the GAN model's potential in generating more accurate predictions, particularly in handling complex time series data, demonstrating stronger adaptability to intricate market patterns.

Among all models, the Wavelet-GAN demonstrated the most outstanding performance. By incorporating wavelet transforms into the GAN architecture, the Wavelet-GAN model showed exceptional predictive accuracy across the entire time span. During the market correction in 2022, the Wavelet-GAN not only accurately tracked the downward trend but also provided predictions closely aligned with actual prices during the market rebound. The model continued to excel during the extreme volatility in early 2024, underscoring its superior capability in processing complex time series data. Compared to other models, the WaveletGAN achieved the lowest MSE and RMSE values, further validating its effectiveness and accuracy in stock price prediction tasks.

Overall, the performance of the models varies significantly across different time periods. The GRU model shows notable limitations in capturing complex temporal dependencies, particularly during periods of high market volatility. The LSTM model generally performs better, especially in managing long-term dependencies. The GAN model improves prediction accuracy through adversarial learning, demonstrating strong adaptability in volatile markets. The Wavelet-GAN model, by incorporating wavelet transforms, further enhances the ability to process complex time series data, emerging as the most effective predictive model. These analyses not only highlight the strengths and weaknesses of each model but also provide valuable insights for future research.

# REFERENCES

[1] A. Soni, V. Pandey, and S. Yadav, “Stock Market Prediction Using Deep Learning,” International Journal for Research in Applied Science and Engineering Technology, vol. 11, no. 2, pp. 45–52, 2023.   
[2] S. Dara, A. Gayathri, K. Deepika, L. Anitha, and K. Indu, “Stock Market Prediction Using Deep Learning Approach,” International Journal of Innovative Research in Engineering and Management, vol. 10, no. 3, pp. 34–41, 2023.   
[3] M. R. Vargas, B. S. Lima, and A. Evsukoff, “Deep Learning for Stock Market Prediction from Financial News Articles,” in 2017 IEEE International Conference on Computational Intelligence and Virtual Environments for Measurement Systems and Applications (CIVEMSA), London: IEEE, 2017, pp. 60–65.   
[4] H. Liu and Z. Long, “An Improved Deep Learning Model for Predicting Stock Market Price Time Series,” Digital Signal Processing, vol. 102, pp. 102741, 2020.   
[5] J. Li, H. Bu, and J. Wu, “Sentiment-aware Stock Market Prediction: A Deep Learning Method,” in 2017 International Conference on Service Systems and Service Management, New York: IEEE, 2017, pp. 1–6.   
[6] W. Jiang, “Applications of Deep Learning in Stock Market Prediction: Recent Progress,” Expert Systems with Applications, vol. 184, pp. 115537, 2020.   
[7] Y. Li, D. Cheng, X. Huang, and C. Li, “Stock price prediction Based on Generative Adversarial Network,” 2022 International Conference on Big Data, Information and Computer Network (BDICN), 2022, pp. 637-641.   
[8] B. He and E. Kita, “GA-Based Optimization of Generative Adversarial Networks on Stock Price Prediction,” 2021 International Conference on Computational Science and Computational Intelligence (CSCI), 2021, pp. 199-202.   
[9] K. Zhang, G. Zhong, J. Dong, S. Wang, and Y. Wang, “Stock Market Prediction Based on Generative Adversarial Network,” Procedia Computer Science, vol. 147, 2019, pp. 400-406.   
[10] L. Wang and Z. Huang, “Research on Stock Price Volatility Prediction Based on Generative Adversarial Network,” 2021 International Conference on Networking, Communications and Information Technology (NetCIT), 2021, pp. 332-335.   
[11] S. R. Polamuri, K. Srinivas, and A. K. Mohan, “Multi-Model Generative Adversarial Network Hybrid Prediction Algorithm (MMGAN-HPA) for stock market prices prediction,” Journal of King Saud University - Computer and Information Sciences, vol. 34, 2021, pp. 7433-7444.   
[12] H.-Y. Lin, C. Chen, G. Huang, and A. Jafari, “Stock price prediction using Generative Adversarial Networks,” Journal of Computer Science, vol. 17, 2021, pp. 188-196.   
[13] Y. Zhang, J.-Y. Li, H. Wang, and S.-C. T. Choi, “Sentiment-Guided Adversarial Learning for Stock Price Prediction,” Frontiers in Applied Mathematics and Statistics, vol. 7, 2021.   
[14] A. Staffini, “Stock Price Forecasting by a Deep Convolutional Generative Adversarial Network,” Frontiers in Artificial Intelligence, vol. 5, 2022.   
[15] B. Labiad, L. Benabbou, and A. Berrado, “Improving Stock Market Intraday Prediction by Generative Adversarial Neural Networks,” Proceedings of the International Conference on Industrial Engineering and Operations Management, 2022.   
[16] B. Labiad, A. Berrado, and L. Benabbou, “Predicting extreme events in the stock market using generative adversarial networks,” International Journal of Advances in Intelligent Informatics, vol. 9, no. 2, 2023.