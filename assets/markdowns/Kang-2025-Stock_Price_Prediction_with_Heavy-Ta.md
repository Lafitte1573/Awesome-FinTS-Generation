# Stock Price Prediction with Heavy‑Tailed Distribution Time‑Series Generation Based on WGAN‑BiLSTM

Ming Kang1

Accepted: 22 May 2024 / Published online: 12 June 2024   
$\circledcirc$ The Author(s), under exclusive licence to Springer Science+Business Media, LLC, part of Springer Nature   
2024

# Abstract

Accurate stock price prediction is essential for investing in the fnancial market. Aiming at the problem’s accuracy, the prediction model is restricted due to the lack of sufcient data samples for the stock price of newly listed companies. This paper proposes a novel approach to improve the generalization of Bidirectional Long Short-Term Memory Network (BiLSTM) with Wasserstein Generative Adversarial Network (WGAN) on stock time-series data augmentation. WGAN is used to learn the distribution rules of the real data, and the two-sample KS test and the area between the log–log plots are adopted to evaluate the extent of heavy-tailed distribution of the generated samples similar to that of real-world data. BiLSTM is employed to extract and predict data’s forward and reverse time information. With the stock price data, the prediction results of diferent algorithm models without and after augmented data are compared. Experimental results show that the accuracy of BiLSTM enhanced by WGAN is signifcantly improved, and the proposed WGANBiLSTM model is suitable for stock price prediction in few-shot settings.

Keywords Financial time-series forecast $\cdot$ Data augmentation $\cdot$ Fat-tailed distribution $\cdot$ Generation adversarial network (GAN) $\cdot$ Long short-term memory (LSTM)

# 1 Introduction

Stock price prediction is a challenging problem with critical signifcance in investing in fnancial markets. In recent years, deep neural networks for time-series forecast, which boast good parameter learning ability and nonlinear data ftting ability, have become the focus of fnancial prediction, including feed-forward deep neural network or deep multilayer perceptron (Yong et al., 2017), deep Long ShortTerm Memory (LSTM) network (Chen et al., 2015; Fischer & Krauss, 2017), deep Convolutional Neural Network (CNN) (Gunduz et  al., 2017; Persio & Honchar, 2016), and combination of CNN and LSTM (Lin et al., 2017).

However, the intelligence system powered by deep learning is a vulnerable and lucrative target to attackers. With the advent of adversarial attacks to computer vision models in 2014–2015 (Goodfellow et  al., 2015; Szegedy et  al., 2014), the highly accurate modern deep learning models in diverse types of data are susceptible to adversarial samples that are derived from the real data by adding small perturbations. It was found that adversarial samples could also attack time series forecasting (Dang-Nhu et al., 2020) and classifcation (Karim et al., 2021) models. Moreover, adversarial training became a meaningful way to improve the robustness and generalization of deep neural networks by adapting to changes in adversarial examples mixed with minor disturbances for various data, including time-series data. At the same time, time-series generation through adversarial training of deep learning, as one of the time-series data augmentation methods, enhances the size and quality of training data for diferent tasks, for example, time-series forecast, classifcation, and anomaly detection (Wen et al., 2021).

For stock price prediction in the few-shot settings, like newly listed companies, traditional methods are typically unavailable due to a lack of samples, dramatically decreasing prediction models’ accuracy. Given the problem, this paper draws on the technical experience of adversarial time-series generation for stock price prediction. This work puts forward an approach of stock data augmentation based on the Wasserstein Generative Adversarial Network (WGAN) (Arjovsky et al., 2017) and stock price prediction based on Bidirectional Long Short-Term Memory (BiLSTM) network (Graves & Schmidhuber, 2005a, 2005b).

One of the most common properties of the distribution of fnancial time-series data is the peak and thick tail, called heavy-tailed or fat-tailed distribution, which makes the error of ftting the data with the normal distribution extensive.

This study uses WGAN-BiLSTM to predict stock prices with adversarial generation on a small dataset. The model architecture is specifcally presented after related work. Comparison experiments are designed and developed after the model architecture explanation. The approach draws on a GAN variant to improve the accuracy of the stock prediction model with adversarial learning and frstly establishes the WGAN-BiLSTM model to enhance the performance of stock price prediction.

# 2 Related Work

# 2.1 Stock Time Series Prediction Using LSTM Network

By highly improving stock time series prediction accuracy, the LSTM network has become the major baseline model in the state of the art. In recent works, developing hybrid LSTM networks has been a central concern in stock market prediction. Su et  al. (2021) restructured LSTM with a rectifed forgetting gate to learn the characteristics and rules of the stock market more objectively. Liu et  al. (2022) enhanced the generalization ability of the LSTM network with the meta-learning algorithm. Other works engaged in optimization algorithms (Lin et al., 2021, 2022), hierarchical attention (Teng et al., 2022), and multilayer sequential LSTM (Quadir et al., 2023). Nevertheless, stock time series predicting with time-series generation is rarely found in the literature.

# 2.2 Time Series Data Generation for Prediction

Applying adversarial training on time series data efectively improves forecast performance since adversarial learning is realistic regularization for supervised generators. Generally, GAN architectures are used to enrich the understanding of time series samples, such as data augmentation and data transformation, and after obtaining high-quality virtual data, the problem of low prediction accuracy caused by small sample data is signifcantly improved. Diferent kinds of GANs exist to realize time-series data augmentation for specifc domains. Time-series scenarios of renewable resources (Chen et al., 2018), power plant (Choi et al., 2020), wind power (Yin et al., 2021), web trafc (Zhou et al., 2020), trafc fow (Zhang et al., 2021), trafc accident (Chen et al., 2021) and smart agriculture (Cheng et al., 2022) all used to be generated by using vanilla GANs (Goodfellow et al., 2015). Feng et al. (2019) employed an adversarial attentive LSTM to classify the stock movement in the near future into two categories: up or down. Multivariate Anomaly Detection with GAN (MAD-GAN) (Li et  al., 2019) considered the entire variable set concurrently to capture the latent interactions amongst the variables. TimeGAN (Yoon et al., 2019) combined the unsupervised GAN with supervised autoregressive models. Li et al. (2020) generated stock market order streams with a Stock-GAN. DoppelGANger (Lin et  al., 2020) improved the fdelity of traditional GAN on time series datasets with metadata.

For heavy-tailed distribution data generation, FIN-GAN (Takahashi et al., 2019) demonstrated the capacity of vanilla GAN with the Multilayer Perceptron (MLP) and MLP-CNN network architectures to reproduce the heavy-tail distributed and long-range dependent time-series. HTGAN (Zhang & Zhou, 2021) employed student t-distribution to improve the applicability of vanilla GANs for industrial heavy tail data generation. Pareto GAN (Huster et  al., 2021) can learn a more compact representation of heavy-tailed datasets than the other GANs with uniform, normal, and lognormal distributions.

# 2.3 WGAN

Vanilla GANs are optimized alternately to achieve the stable state of Nash equilibrium. However, since the generation of the adversarial network, the problem of the model training difculties of the checker and generator has persisted, and the virtual sample generated by the generator lacks diversity. Another phenomenon is that the better the checker model, the more serious the gradient disappearance of the generator model during training.

In an effort to solve the above GAN problems, many scholars are trying to solve them, but the effect is not apparent. Until WGAN, the problem of the instability of the original GAN network model was fixed. WGAN changed the measurement of the two probability distributions in GAN from the f divergence to Wasserstein distance, making the training of generators and checkers more stable in subsequent experiments and generating more diverse virtual samples closer to the data distribution of the original pieces.

The advantage of Wasserstein distance over relative entropy and JensenShannon (JS) divergence is that even if the two distributions do not overlap, Wasserstein distance can still reflect their proximity. Relative entropy and JS divergence are abrupt and extreme when Wasserstein distance is smooth and can be optimized with gradient descent, which is impossible with relative entropy and JS divergence. Thus, joining Wasserstein distance dramatically improves the training process of GAN, increases the stability of the model, and reduces the phenomenon of gradient explosion or gradient disappearance.

![](images/75966400d86e5374d1581c8d1efbb45fa7070270f7c149b14b746b7511352db9.jpg)  
Fig. 1 WGAN-BiLSTM prediction framework

Table 1 FC Layers and parameters of the discriminator   

<table><tr><td>Number of layers</td><td>Internal parameter</td><td>Value</td></tr><tr><td>FC layer 1</td><td>Number of neurons</td><td>512</td></tr><tr><td></td><td>Activation function</td><td>ReLu</td></tr><tr><td></td><td>Parameters</td><td>74,240</td></tr><tr><td>FC layer 2</td><td>Number of neurons</td><td>256</td></tr><tr><td></td><td>Activation function</td><td>ReLu</td></tr><tr><td></td><td>Parameters</td><td>131,328</td></tr><tr><td>FC layer 3</td><td>Number of neurons</td><td>128</td></tr><tr><td></td><td>Activation function</td><td>ReLu</td></tr><tr><td></td><td>Parameters</td><td>32,896</td></tr><tr><td>FC layer 4</td><td>Number of neurons</td><td>1</td></tr><tr><td></td><td>Activation function</td><td>ReLu</td></tr><tr><td></td><td>Parameters</td><td>129</td></tr></table>

Table 2 Layers and parameters of the generator   

<table><tr><td>Number of layers</td><td>Internal parameter</td><td>Value</td></tr><tr><td rowspan="2">FC layer 5</td><td>Number of neurons</td><td>256</td></tr><tr><td>Activation function</td><td>ReLu</td></tr><tr><td rowspan="3">FC layer 6</td><td>Parameters</td><td>25,856</td></tr><tr><td>Number of neurons</td><td>512</td></tr><tr><td>Activation function</td><td>ReLu</td></tr><tr><td rowspan="3">FC layer 7</td><td>parameters</td><td>131,584</td></tr><tr><td>Number of neurons</td><td>1024</td></tr><tr><td>Activation function</td><td>ReLu</td></tr><tr><td rowspan="3">FC layer 8</td><td>Parameters</td><td>525,312</td></tr><tr><td>Number of neurons</td><td>144</td></tr><tr><td>Activation function</td><td>ReLu</td></tr><tr><td></td><td>Parameters</td><td>576</td></tr></table>

Table 3 Structure and parameters of BiLSTM   

<table><tr><td>Structure</td><td>Parameter</td></tr><tr><td>Loss function</td><td>Mean Square Error (MSE)</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>1stLayer</td><td>Neurons=100,input_dim=5,return_ seq=true, input_timesteps=54</td></tr><tr><td>Dropout layer</td><td>Dropout rate=0.5</td></tr><tr><td>2nd Layer</td><td>Neurons=150,return_seq=true</td></tr><tr><td>3rd Layer</td><td>Neurons =300,return_seq=false</td></tr><tr><td>Dropout layer</td><td>Dropout rate= 0.5</td></tr><tr><td>Dense layer</td><td>Neurons =1,activate function=sigmoid</td></tr></table>

# 2.4 BiLSTM

The typical LSTM network architecture (Gref et  al., 2017) in one-way training mode cannot fully use the data global time information; in the training process, the status update of the implicit layer is achieved through the one-way timing sample data input. Like time-dependent data such as stock prices, the stock price of a moment is related not only to that of the previous moment but also to that of the following moment.

BiLSTM network (Graves & Schmidhuber, 2005a, 2005b) connects two hidden layers in opposite directions to the same output layer, giving the output layer information about past and future states, meaning that BiLSTM network can learn information from two diferent data directions for more accurate predictions. The essence of BiLSTM is to split the regular LSTM neurons into positive state (positive time direction) and a reverse state (negative time direction). The output of each step is composed of positive and negative bidirectional LSTM, with the ability to train historical data in both directions and learn more efective information from historical data. The stock price prediction studied in this paper is a typical timing problem. The price of a particular moment is infuenced by the previous moment and historical multi-moment, so the BiLSTM model chosen for the stock price prediction is a particular type of recurrent neural network that learns to rely on information for a long time.

# 3 Model

In this work, WGAN is built using fully connected (FC) layers to learn the data distribution of each single feature in the real timing sample. The model uses the noise of $2 0 0 \times 1$ in line with the Gaussian distribution as input to the generator, and the expansion ratio is $3 0 \%$ , $6 0 \%$ and $1 0 0 \%$ , respectively. After that, the generator could generate diferent amounts of virtual data for experimental comparison. The WGAN model’s purpose is to stabilize the model training process, better capture the inherent characteristics of historical data, and generate new data with diferent features and similar distributions to the real data. The basic framework of the model is shown in Fig. 1.

The FC layers and parameters of the discriminator and generator in the WGAN network are shown in Tables 1 and 2. The model uses RMSProp (Hinton et al., 2012) optimizers. The generator and discriminator play games with each other during training, complete the training after Nash equilibrium, and then generate the same distributed data as the real data according to diferent noise amounts to reduce the training difculty of the algorithm model and improve the prediction accuracy.

Table 4 Number of samples of datasets (a) and (b)   

<table><tr><td>Dataset</td><td>Training set</td><td>Testing set</td><td>Total</td></tr><tr><td>(a)</td><td>1065</td><td>266</td><td>1331</td></tr><tr><td>(b)</td><td>947</td><td>236</td><td>1183</td></tr></table>

Table 5 Training settings of WGAN   

<table><tr><td>Training parameter</td><td>Value</td></tr><tr><td>Learning rate (α)</td><td>0.0005</td></tr><tr><td>Gradient parameters (c)</td><td>0.01</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Number of iterations (n)</td><td>5</td></tr></table>

Table 6 Training settings of BiLSTM   

<table><tr><td>Training parameter</td><td>Value</td></tr><tr><td>Sequence length</td><td>55</td></tr><tr><td>Epochs</td><td>2</td></tr><tr><td>Batch size</td><td>32</td></tr></table>

The neurons of three layers in the BiLSTM separately are 100, 150 and 300. To avoid overftting, two dropout layers are added. The optimizer Adam (Kingma & Ba, 2015) is employed. The specifc structure and parameters of the BiLSTM are illustrated in Table 3.

# 4 Experiments

# 4.1 Datasets

Two datasets of stock prices with a few-shot setting from diferent industries are used in this study. The dataset (a) contains 1331 days of daily stock market trading of Ping An Bank Co., Ltd. (stock code: 000001.SZ, or 000001.Shenzhen) from 1 January 2017 to 31 May 2017. The dataset (b) consists of 1184 samples of Haier Smart Home Co., Ltd. (stock code: 600690.SH, or 60069.Shanghai) from 1 August 2013 to 18 May 2018. The two datasets are open access from any stock market price quote, for example, Yahoo, Reuters, or Bloomberg. The closing price is the prediction target, and the sampling point is 10 per day. Taking into account the correlation between the closing price of the stock and other characteristic factors and the time series law of the stock price, the closing price of the previous sample point is named previous close so that each sample point has six characteristics, including open, high, low, close, turnover and volume.

The training and testing sets are divided by 8:2 in the real data. The training sets [from 1 January 2017 to 31 March 2017 of dataset (a) and 1 August 2013 to 6 June 2017 of dataset (b)] are trained to generate new samples by the WGAN-BiLSTM model. The testing sets [from 1 April 2017 to 31 May 2017 of dataset (a) and 7 June

2017 to 18 May 2018 of dataset (b)] are used to verify the validity of the model. Table 4 presents the number of samples of both datasets.

# 4.2 Data Processing

Because of the signifcant diference in the range of values of several features entered by the model, to avoid the negative efect of numerical range diferences between elements on data augmentation and prediction, it is necessary to normalize the real data and to narrow the range of data to between $^ { - 1 }$ and 1, the formula for data normalization is as follows:

$$
y = { \frac { x - M i n V a l u e } { M a x V a l u e - M i n V a l u e } }
$$

# 4.3 Training Settings

The control variation method is used to select the appropriate value range for diferent parameters, and the parameters are constantly adjusted in the training process until the model’s prediction efect is the best. The values of training settings in the experiment are separately listed in Tables 5 and 6 for WGAN and BiLSTM.

![](images/4f543ed920b665fd9ff33fe72d28392e8590248bd390a1832d681208a38d5d37.jpg)  
Fig. 2 Comparison of diferent columns of datasets between the real and generated stock data

![](images/3ff0700610ca56e81d999e265ffacde81df8385019a278a7a56c1c409f5c66f6.jpg)  
Fig. 2 (continued)

# 4.4 Evaluation Metrics

# 4.4.1 Data Generation Metrics

The two-sample Kolmogorov–Smirnov (KS) test is used to determine diferences between the real and generated samples. KS statistic is sensitive to check whether the two data samples come from the same distribution. Also, the area between the log–log plots of the complementary cumulative distribution functions of the real and generated samples provides a reliable indication of the extent to which the generated heavy tails align with the characteristics of actual samples. KS statistic and the Area are computed as in Pareto GAN (Huster et al., 2021).

Table 7 KS statistic value comparison of diferent GANs in diferent noise dimensions   

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">Noise dimension</td></tr><tr><td>50</td><td>100</td><td>200</td></tr><tr><td rowspan="3">(a)</td><td>GAN</td><td>0.1409</td><td>0.1290</td><td>0.1131</td></tr><tr><td>DCGAN</td><td>0.1030</td><td>0.0968</td><td>0.0921</td></tr><tr><td>WGAN</td><td>0.1030</td><td>0.0968</td><td>0.0921</td></tr><tr><td rowspan="3">(b)</td><td>GAN</td><td>0.1350</td><td>0.1286</td><td>0.1122</td></tr><tr><td>DCGAN</td><td>0.1028</td><td>0.0962</td><td>0.0918</td></tr><tr><td>WGAN</td><td>0.0728</td><td>0.0692</td><td>0.0641</td></tr></table>

![](images/12d25a10e2a5bea946b929006659d8262ffff6b093744f4838d4a812984027cb.jpg)  
Fig. 3 Comparison of KS statistic values of diferent GANs in diferent noise dimensions

# 4.4.2 Prediction Performance Metrics

Three error methods—Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and R-Squared $( \mathbb { R } ^ { 2 } )$ —are selected to evaluate the prediction efect of all models.

$$
M A E = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left| \widehat { y } _ { i } ( t ) - y _ { i } ( t ) \right|
$$

Table 8 Area value comparison of diferent GANs in diferent noise dimensions   

<table><tr><td>Dataset</td><td>Model</td><td colspan="3">Noise dimension</td></tr><tr><td></td><td></td><td>50</td><td>100</td><td>200</td></tr><tr><td>(a)</td><td>GAN</td><td>15.987</td><td>15.943</td><td>15.212</td></tr><tr><td></td><td>DCGAN</td><td>9.745</td><td>9.488</td><td>8.408</td></tr><tr><td></td><td>WGAN</td><td>4.256</td><td>4.033</td><td>3.074</td></tr><tr><td>(b)</td><td>GAN</td><td>19.887</td><td>19.403</td><td>17.456</td></tr><tr><td></td><td>DCGAN</td><td>14.701</td><td>13.009</td><td>11.509</td></tr><tr><td></td><td>WGAN</td><td>7.094</td><td>6.149</td><td>6.124</td></tr></table>

$$
R M S E = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ \widehat { y } _ { i } ( t ) - y _ { i } ( t ) \right] ^ { 2 } }
$$

$$
R ^ { 2 } = 1 - \frac { \sum _ { i } \big [ \widehat { y } _ { i } ( t ) - y _ { i } ( t ) \big ] ^ { 2 } } { \sum _ { i } \big [ \overline { { y } } - y _ { i } ( t ) \big ] ^ { 2 } }
$$

where $\widehat { y } _ { i } ( t )$ represents the predicted closing price at the moment t; $y _ { i } ( t )$ represents the real closing price at the moment t; n is the length size of the data. In the above indicators, the smaller the value of MAE and RMSE, the better the model performance; while the closer to 1 the value of $\mathbb { R } ^ { 2 }$ is, the higher the model’s prediction accuracy.

# 4.5 Efectiveness of Data Generation

In the experiments of data generation, 50, 100, or 200 diferent noise dimensions extracted from the predefned Gaussian distribution are input to the trained generator.

The comparison of diferent columns of datasets between the real and generated stock data with the enhanced ratio of $1 0 0 \%$ by WGAN is shown in Fig. 2. The graph visually shows how the virtual sample compares to the original training set. The resulting data is very similar to the data from the validation set, and the WGAN model does not use the data in training. Take.

To further test the efectiveness of the generated data, this paper verifes the efective distribution of the data generated by the WGAN network by using two methods: KS statistic, the Area, and correlation coefcient matrix color map. Also, in a bid to highlight the experimental efect, the experiment used the vanilla GAN network deep convolutional GAN (DCGAN) (Radford et  al., 2016) to make a comparison. All the closing price data are taken as an example.

As can be seen from Table 7 and Fig. 3, the KS statistic value of WGAN is higher than the other two GANs, which shows that the matching degree of WGAN data and real data is better than other comparison models, and is closest to 1 when the input noise dimension is 200.

![](images/0b1072e6b8e4775906d10f4567b003b10a6e13de7840741939b81c741ffae71f.jpg)  
Fig. 4 Comparison of the area values of diferent GANs in diferent noise dimensions

It can be observed from Table  8 and Fig.  4 that in the experiments of the two datasets, compared with the GAN and DCGAN generation models, the area of the log–log plot between the WGAN-generated samples and the real samples has a signifcant downward trend, which shows that the tail of the WGAN generated data is better consistent with real data.

To verify the spatial correlation between the generated data of diferent GAN networks and the real data, the same noise is injected into the model generator under other models with an enhanced ratio of $1 0 0 \%$ , and $2 4 \times 2 4$ data points of the generated sample are randomly selected in the generated data and in the real data for the same time as the continuous generation of the samples for correlation comparison. The coefcient matrix and the results of color matching are shown in Fig. 5.

![](images/1e284ff2a5c17af2e77c9ddc4962b8ffb5600239788012fdc8867adcb2811e9b.jpg)  
Fig. 5 Comparison of matrix color maps of the correlation coefcient generated by diferent GANs

It can be seen intuitively from Fig. 5 that the contrast among subgraphs (a2)/(b2), (a3)/ (b3) and (a1)/(b1) does not have a highly similar color scheme between subgraphs (a1)/ (b1) and (a4)/(b4), which indicates that the data generated by WGAN has a highly similar time and spatial correlation with the real data.

# 4.6 Empirical Results

From Table 9 and Fig. 6, it can be concluded that:

![](images/6e1eec75ed1ae6c47559c99a3ce2c4abc336fd91f81c8da696fd8350580cde13.jpg)  
Fig. 5 (continued)

(i) At any augmentation scale, each data augmentation-based machine learning model outperforms the WGAN network without data augmentation. For example, in the case of the augmentation of $3 0 \%$ , the MSE and RMSE of the WGAN-Support Vector Regression (SVR) model are lower than the single SVR model and reduced by $2 . 7 5 \%$ and $2 . 0 4 \%$ , respectively. Its $\mathbb { R } ^ { 2 }$ is $2 . 6 4 \%$ higher than other models with the dataset (a). The performance of diferent models is the same as SVR.

(ii) In the case of the same augmentation ratio, the BiLSTM model is higher than others in the predictive performance. Take the $1 0 0 \%$ augmentation ratio as an instance; the MAE and RMSE of BiLSTM with dataset (a) are reduced by $9 . 0 2 1 \%$ , $7 . 8 9 8 \%$ , and $8 . 3 5 3 \%$ , $6 . 5 3 7 \%$ , respectively than those of LSTM and Gated Recurrent Unit (GRU). However, $\mathbb { R } ^ { 2 }$ of BiLSTM with the dataset (a) improved by $1 0 . 2 0 3 \%$ and $7 . 7 5 2 \%$ , respectively. This indicates that BiLSTM can extract data in both directions and improve the model’s predictive performance.

Table 9 Comparison of the performance of each model with diferent augmentation ratios in data   

<table><tr><td>Ratio</td><td>Model</td><td>MAE</td><td>RMSE</td><td>R²</td></tr><tr><td colspan="5">(a)</td></tr><tr><td>0</td><td>Random forest</td><td>9.423</td><td>8.784</td><td>0.683</td></tr><tr><td></td><td>SVR</td><td>9.514</td><td>8.991</td><td>0.719</td></tr><tr><td></td><td>LSTM</td><td>8.531</td><td>8.295</td><td>0.844</td></tr><tr><td></td><td>GRU</td><td>8.011</td><td>7.541</td><td>0.848</td></tr><tr><td>30%</td><td>BiLSTM</td><td>7.814</td><td>6.761</td><td>0.921</td></tr><tr><td></td><td>Random forest</td><td>9.225</td><td>8.496</td><td>0.719</td></tr><tr><td></td><td>SVR</td><td>9.324</td><td>8.743</td><td>0.738</td></tr><tr><td></td><td>LSTM</td><td>8.297</td><td>8.175</td><td>0.846</td></tr><tr><td></td><td>GRU</td><td>7.943</td><td>7.541</td><td>0.850</td></tr><tr><td>60%</td><td>BiLSTM</td><td>7.139</td><td>6.702</td><td>0.947</td></tr><tr><td></td><td>Random forest</td><td>9.011</td><td>8.144</td><td>0.725</td></tr><tr><td></td><td>SVR</td><td>9.316</td><td>8.732</td><td>0.746</td></tr><tr><td></td><td>LSTM</td><td>8.024</td><td>7.875</td><td>0.861</td></tr><tr><td></td><td>GRU</td><td>7.453</td><td>6.993</td><td>0.884</td></tr><tr><td></td><td>BiLSTM</td><td>6.824</td><td>6.692</td><td>0.951</td></tr><tr><td>100%</td><td>Random forest</td><td>8.745</td><td>8.164</td><td>0.749</td></tr><tr><td></td><td>SVR</td><td>9.218</td><td>8.652</td><td>0.776</td></tr><tr><td></td><td>LSTM</td><td>7.424</td><td>7.275</td><td>0.866</td></tr><tr><td></td><td>GRU</td><td>7.103</td><td>6.843</td><td>0.903</td></tr><tr><td></td><td>BiLSTM</td><td>6.034</td><td>6.174</td><td>0.973</td></tr><tr><td colspan="5">(b)</td></tr><tr><td>0</td><td>Random forest</td><td>9.554</td><td>8.967</td><td>0.653</td></tr><tr><td></td><td>SVR</td><td>9.348</td><td>9.004</td><td>0.726</td></tr><tr><td></td><td>LSTM</td><td>8.934</td><td>8.438</td><td>0.816</td></tr><tr><td></td><td>GRU</td><td>8.773</td><td>7.841</td><td>0.899</td></tr><tr><td>30%</td><td>BiLSTM</td><td>7.969</td><td>6.463</td><td>0.921</td></tr><tr><td></td><td>Random forest</td><td>9.435</td><td>8.816</td><td>0.683</td></tr><tr><td></td><td>SVR</td><td>9.501</td><td>8.873</td><td>0.738</td></tr><tr><td></td><td>LSTM</td><td>8.756</td><td>8.342</td><td>0.856</td></tr><tr><td></td><td>GRU</td><td>8.238</td><td>7.859</td><td>0.909</td></tr><tr><td>60%</td><td>BiLSTM</td><td>7.348</td><td>6.713</td><td>0.921</td></tr><tr><td></td><td>Random forest</td><td>9.399</td><td>8.743</td><td>0.601</td></tr><tr><td></td><td>SVR</td><td>9.647</td><td>8.688</td><td>0.694</td></tr><tr><td></td><td>LSTM</td><td>8.564</td><td>7.907</td><td>0.853</td></tr><tr><td></td><td>GRU</td><td>7.901</td><td>7.013</td><td>0.891</td></tr><tr><td></td><td>BiLSTM</td><td>6.538</td><td>6.850</td><td>0.946</td></tr><tr><td>100%</td><td>Random forest</td><td>9.745</td><td>8.235</td><td>0.777</td></tr><tr><td></td><td>SVR</td><td>9.998</td><td>8.641</td><td>0.735</td></tr><tr><td></td><td>LSTM</td><td>7.847</td><td>7.431</td><td>0.841</td></tr><tr><td></td><td>GRU</td><td>7.397</td><td>6.903</td><td>0.933</td></tr><tr><td></td><td>BiLSTM</td><td>6.144</td><td>6.349</td><td>0.971</td></tr></table>

![](images/d58a1249db25afaf995f76f9207e12d16d041ca803c66b33846883b4ef2c49c2.jpg)  
Fig. 6 Comparison of model prediction charts under diferent augmentation ratios

(iii) Using WGAN networks to practice data augmentation, the higher the augmentation ratio, the higher the predictive performance of each model. Take BiLSTM with the dataset (b) as an example; the MAE and RMSE with an augmentation ratio of $1 0 0 \%$ are reduced by $1 6 . 3 8 5 \%$ , $6 . 0 2 6 \%$ , and $5 . 4 2 2 \%$ , $7 . 3 1 4 \%$ , respectively than those of $6 0 \%$ and $3 0 \%$ . $\mathbb { R } ^ { 2 }$ increased by $5 . 4 2 9 \%$ and $2 . 6 4 3 \%$ , respectively.

Additionally, it can be seen from Fig.  6 that the model prediction curve on the testing sets is closer to the real data with the increase in the ratio of augmentation.

# 5 Conclusion

Given the low accuracy of model prediction caused by the lack of data on stock prices of newly listed companies, this paper proposes a short-term stock price prediction model based on WGAN and BiLSTM. Using one year’s data from a particular company, the experiment simulation concludes the following conclusions:

![](images/a2c12aa0df2f5cb2db27de106267b2a3ee15e84ea2d2d62b63366bc4f5297a18.jpg)  
Fig. 6 (continued)

(i) WGAN can efectively learn the data distribution rules of the real data and generate new, higher-quality data highly similar to the real data, making up for the lack of data responsible for the low prediction accuracy.   
(ii) Adopting the BiLSTM as the prediction model, I can extract the data time correlation forward and reverse, break through the limitation of vanilla LSTM to remove time information one-way, and make full use of the time information of the data, to sharpen the predictive performance of the model.

There are some limitations in this study. First, the generative adversarial model is insignifcant when the dataset exceeds a certain magnitude, for example, tens of thousands or even hundreds of thousands. Second, the case of small samples is only ft for the newly listed companies.

While many problems currently exist, people can believe the future of an adversarial generation is very bright, and its use in stock price prediction will continue to grow. To improve the generation performance of the generative adversarial model could enhance the accuracy of the prediction model. For further research, to test generalization performance with stock market time series.

Acknowledgements I thank all the anonymous reviewers for their constructive remarks and helpful suggestions for improving the article.

Funding The author has not disclosed any funding.

Data Availability Available upon request. All datasets and codes on which the conclusions of the manuscript rely to be can be either deposited in publicly available repositories (where available and appropriate) or presented in the main paper or additional supporting fles, in machine-readable format (such as spreadsheets rather than PDFs) whenever possible.

# Declarations

Confict of interest The author has no confict of interest related to the article.

Ethical Approval No particular ethical approval was required for this study because it does not entail human participation or personal data.

Informed Consent No consent was required for this study because it does not entail human participation or personal data. The analysis was based on open-access stock market data without fees.

# References

Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein generative adversarial networks. Proceedings of Machine Learning Research, 70, 214–223.   
Chen, Y., Wang, Y., Kirschen, D., & Zhang, B. (2018). Model-free renewable scenario generation using generative adversarial networks. IEEE Transaction on Power Systems, 33(3), 3265–3275.   
Chen, Z., Zhang, J., Zhang, Y., & Huang, Z. (2021). Trafc accident data generation based on improved generative adversarial networks. Sensors, 21(17), 5767.   
Chen, K., Zhou, Y., & Dai, F. (2015). A LSTM-based method for stock returns prediction: A case study of China stock market. In H. Ho, B. C. Qoi, M. J. Zaki, X. Hu, L. Hass, V. Kumar, S. Rachuri, S. Yu, M. H.-I. Hsiao, J. Li, F. Luo, S. Pyne, & K. Ogan (Eds.), Proceedings of the 2015 IEEE international conference on big data (pp. 149–153). IEEE.   
Cheng, W., Ma, T., Wang, X., & Wang, G. (2022). Anomaly detection for internet of things time series data using generative adversarial networks with attention mechanism in smart agriculture. Frontiers in Plant Science, 13, 890563.   
Choi, Y., Lim, H., Choi, H., & Kim, I.-J. (2020). GAN-based anomaly detection and localization of multivariate time series data for power plant. In Proceedings of the 2020 IEEE international conference on big data and smart computing (pp. 71–74). IEEE.   
Dang-Nhu, R., Singh, G., Bielik, P., & Vechev, M. (2020). Adversarial attacks on probabilistic autoregressive forecasting models. Proceedings of Machine Learning Research, 119, 2356–2365.   
Feng, F., Chen, H., He, X., Ding, J., Sun, M., & Chua, T.-S. (2019). Enhancing stock movement prediction with adversarial training. In S. Kraus (Ed.), Proceedings of the twenty-eighth international joint conference on artifcial intelligence (pp. 5543–5849). IJCAI.   
Fischer, T., & Krauss, C. (2017). Deep learning with long short-term memory networks for fnancial market predictions. European Journal of Operational Research, 270(2), 654–669.   
Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. In Proceedings of the 3rd international conference on learning representations 2015. ICLR.   
Graves, A., & Schmidhuber, J. (2005b). Framewise phoneme classifcation with bidirectional LSTM and other neural network architectures. Neural Networks, 18(5–6), 602–610.   
Graves, A., & Schmidhuber, J. (2005a). Framewise phoneme classifcation with bidirectional LSTM networks. In Proceedings of the 2005 international joint conference on neural networks, (pp. 2047– 2052). IEEE.   
Gref, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A search space Odyssey. IEEE Transactions on Neural Networks and Learning Systems, 28(10), 2222–2232.   
Gunduz, H., Yaslan, Y., & Cataltepe, Z. (2017). Intraday prediction of Borsa Istanbul using convolutional neural networks and feature correlations. Knowledge-Based Systems, 137, 138–148.   
Hinton, G., Srivastava, Y., & Swersky, K. (2012). Rmsprop: Divide the gradient by a running average of its recent magnitudes. In Neural networks and machine learning: Lecture, Vol. 6 (pp. 26–31). Coursera.   
Huster, T., Cohen, J., Lin, Z., Chan, K., Kamhoua, C., Leslie, N. O., Chiang, C.-Y. J., & Sekar, V. (2021). Pareto GAN: Extending the representational power of GANs to heavy-tailed distributions. Proceedings of Machine Learning Research, 139, 4523–4532.   
Karim, F., Majumdar, S., & Darabi, H. (2021). Adversarial attacks on time series. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(10), 3309–3320.   
Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In Proceedings of the 3rd international conference on learning representations 2015. ICLR.   
Li, J., Wang, X., Lin, Y., Sinha, A., & Wellman, M. P. (2020). Generating realistic stock market order streams. Proceedings of the AAAI Conference on Artifcial Intelligence, 34(01), 727–734.   
Li, D., Chen, D., Jin, B., Shi, L., Goh, J., & Ng, S.-K. (2019). MAD-GAN: Multivariate anomaly detection for time series data with generative adversarial networks. In I. V. Tetko, V. Kůrková, P. Karpov, & F. Theis (Eds.), Artifcial neural networks and machine learning – ICANN 2019: Text and time series, 28th international conference on artifcial neural networks, Munich, Germany, September 17–19, 2019, Proceedings, Part $I V$ (pp. 703–716). Springer.   
Lin, Y., Lin, Z., Liao, Y., Li, Y., Xu, J., & Yan, Y. (2022). Forecasting the realized volatility of stock price index: A hybrid model integrating CEEMDAN and LSTM. Expert Systems with Applications, 206, 117736.   
Lin, Y., Yan, Y., Xu, J., Liao, Y., & Ma, F. (2021). Forecasting stock index price using the CEEMDANLSTM model. North American Journal of Economics and Finance, 57, 101421.   
Lin, T., Guo, T., & Aberer, K. (2017). Hybrid neural networks for learning the trend in time series. In C. Sierra, & IIIA-CSIC (Eds.), Proceedings of the twenty-sixth international joint conference on artifcial intelligence (pp. 2273–2279). IJCAI.   
Lin, Z., Jain, A., Wang, C., Fanti, G., & Sekar, V. (2020). Using GANs for sharing networked time series data: Challenges, initial promise, and open questions. In IMC ’20: Proceedings of the 2020 ACM internet measurement conference (pp. 464–483). ACM.   
Liu, T., Ma, X., Li, S., Li, X., & Zhang, C. (2022). A stock price prediction method based on meta-learning and variational mode decomposition. Knowledge-Based Systems, 252, 109324.   
Persio, L. D., & Honchar, O. (2016). Artifcial neural networks architectures for stock price prediction: Comparisons and applications. International Journal of Circuits, Systems and Signal Processing, 10, 403–413.   
Quadir, M. A., Kapoor, S., Junni, A. V. C., Sivaraman, A. K., Tee, K. F., Sabireen, H., & Janakiraman, N. (2023). Novel optimization approach for stock price forecasting using multi-layered sequential LSTM. Applied Soft Computing, 134, 109830.   
Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised representation learning with deep convolutional generative adversarial networks. In Proceedings of the 4th international conference on learning representations 2016. ICLR.   
Su, Z., Xie, H., & Han, L. (2021). Multi-factor RFG-LSTM algorithm for stock sequence predicting. Computational Economics, 57(4), 1041–1058.   
Szegedy, C., Zaremba, W., Sutskever, I, Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). Intriguing properties of neural networks. In Proceedings of the 2nd international conference on learning representations 2014. ICLR.   
Takahashi, S., Chen, Y., & Tanaka-Ishii, K. (2019). Modeling fnancial time-series with generative adversarial networks. Physica A: Statistical Mechanics and its Applications, 527, 121261.   
Teng, X., Zhang, X., & Luo, Z. (2022). Multi-scale local cues and hierarchical attention-based LSTM for stock price trend prediction. Neurocomputing, 505, 92–100.   
Wen, Q., Sun, L., Yang, F., Song, X., Gao, J., Wang, X., & Xu, H. (2021). Time series data augmentation for deep learning: A survey. In Z.-H. Zhou (Ed.), Proceedings of the thirtieth international joint conference on artifcial intelligence (pp. 4653–4660). IJCAI.   
Yin, H., Ou, Z., Zhu, Z., Xu, X., Fan, J., & Meng, A. (2021). A novel asexual-reproduction evolutionary neural network for wind power prediction based on generative adversarial networks. Energy Conversion and Management, 247, 114714.   
Yong, B. X., Abdul Rahim, M. R., & Abdullah, A. S. (2017). A stock market trading system using deep neural network. In M. Mohamed Ali, H. Wahid, N. Mohd Subha, S. Sahlan, M. Md. Yunus, & A. Wahap (Eds.), Modeling, design and simulation of systems: 17th Asia simulation conference, AsiaSim 2017, Melaka, Malaysia, August 27–29, 2017, Proceedings, Part I (pp. 356–364). Springer.   
Yoon, J., Jarrett, D., & van der Schaar, M. (2019). Time-series generative adversarial networks. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. Fox, & R. Garnett (Eds.), Advances in Neural Information Processing Systems 32 (NeurIPS 2019). Curran Associates.   
Zhang, X., Wang, S., Chen, B., Cao, J., & Huang, Z. (2021). TrafcGAN: Network-scale deep trafc prediction with generative adversarial nets. IEEE Transaction on Intelligent Transportation Systems, 22(1), 219–230.   
Zhang, Y., & Zhou, J. (2021). A heavy-tailed distribution data generation method based on generative adversarial network. In Proceedings of the 2021 IEEE 10th data driven control and learning systems conference (pp. 535–540). IEEE.   
Zhou, K., Wang, W., Hu, T., & Deng, K. (2020). Time series forecasting and classifcation models based on recurrent with attention mechanism and generative adversarial networks. Sensors, 20(24), 7211.

Publisher’s Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional afliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.