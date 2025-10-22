# An Efcient GAN‑Based Multi‑classifcation Approach for Financial Time Series Volatility Trend Prediction

Lei Liu1 $\cdot$ Zheng Pei1 $\cdot$ Peng Chen1 $\textcircled{1}$ · Hang Luo2  · Zhisheng Gao1 $\cdot$ Kang Feng1 $\cdot$ Zhihao Gan1

Received: 18 August 2021 / Accepted: 28 February 2023   
$\circledcirc$ The Author(s) 2023

# Abstract

Deep learning has achieved tremendous success in various applications owing to its robust feature representations of complex high-dimensional nonlinear data. Financial time-series prediction is no exception. Hence, the volatility trend prediction in fnancial time series (FTS) has been an active topic for several decades. Inspired by generative adversarial networks (GAN), which have been studied extensively in image processing and have achieved excellent results, we present the ordinal regression GAN for fnancial volatility trends (ORGAN-FVT) method for the end-to-end multi-classifcation task of FTS. An improved generative model based on convolutional long short-term memory (ConvLSTM) and multilayer perceptron (MLP) is proposed to capture temporal features efectively and mine the data distribution of volatility trends (short, neutral, and long) from given FTS data. Meanwhile, ordinal regression is leveraged for the discriminator to improve the multi-classifcation performance, making the model more practical. Finally, we empirically compare ORGAN-FVT with several state-of-the-art approaches on three real-world stock datasets: MICROSOFT(MSFT), Tesla(TSLA), and The People’s Insurance Company of China(PAICC). ORGAN-FVT demonstrated signifcantly better AUC and F1 scores, at most $2 0 . 8 1 \%$ higher than its competitors.

Keywords Financial time series $\cdot$ Generative adversarial nets $\cdot$ Convolutional LSTM $\cdot$ Classifcation

# Abbreviations

FTS Financial time series GAN Generative adversarial nets GAN-FVT GAN for fnancial volatility trends \* Peng Chen chenpeng@mail.xhu.edu.cn Lei Liu 1163454848@qq.com Zheng Pei pqyz@263.net Hang Luo robin.h.luo@gmail.com Zhisheng Gao 46235604@qq.com Kang Feng 1831147112@qq.com Zhihao Gan 444624141@qq.com

ORGAN-FVT GAN with ordinary regression for fnancial volatility trends   
ConvLSTM Convolutional long short-term memory   
MLP Multilayer perceptron   
TSC Time-series classifcation   
DL Deep learning   
CNN Convolutional neural networks   
LSTM-FCN Long short-term memory fully convolutional network   
AUC Area under curve   
ROC Receiver operating characteristic

# 1 Introduction

In the past 2 decades, people have become increasingly interested in the classifcation of time series, and an increasing number of scholars have joined the research. Moreover, with the advent of the 5G era, big data are closely related to our lives. Time-series data are found everywhere, especially in the medical, industrial and meteorology felds [1–4]. In the fnancial feld, because accurate and efective fnancial time series analysis methods can avoid risks and provide proftable investment strategies for investors, more and more researchers have joined fnancial time series analysis and research. Financial time series analysis refers to the ability to mine the volatility trend law of fnancial products through historical data to guide investors’ rational investment. However, because the fnancial market is afected by many different factors, the data mining of fnancial products is quite challenging. More and more fnancial time series analysis methods have also been proposed in recent years. This type of research is usually based on three methods: traditional methods based on traditional statistical analysis methods, prediction-based analysis methods, and reinforcement learning-based analysis methods. The research on future price trends based on historical data of fnancial products has become an increasingly popular direction, especially the research on fnancial time series classifcation.

The time-series classifcation (TSC) is a critical issue in time-series data mining research. TSC accurately classifes a series of unknown time series according to the known “category” labels in the time series. TSC can be regarded as a “supervised” learning mode in the time series category,. unlike the traditional classification method that only considers numerical attributes, the TSC must consider the order relationship between adjacent samples in the time series. It is the same as fnancial time series classifcation problems. Meanwhile, compared with other types of time series, the fnancial time series has complex, highly noisy, dynamic, nonlinear, and nonlinear characteristics. Therefore, it is more challenging than traditional classifcation methods [5]. When facing these challenges, using appropriate methods and better models to learn the data characteristics of financial time series to have better performance and higher accuracy in classifcation performance will be very challenging. Since 2015, hundreds of TSC algorithms have been proposed [6]. Traditional TSC methods based on sequence distance have been proven to achieve the best classifcation performance in most felds, except for distancebased methods. Additionally, feature-based classifcation methods have excellent classifcation performance based on existing good features. However, it is challenging to design good features when faced with a FTS to capture some inherent properties in the series. Although methods based on distance or features are used in many studies, these two methods have resulted in far too many calculations for many practical applications [7]. As many researchers have applied deep learning (DL) methods to TSC, an increasing number of TSC methods have been proposed, especially with the emergence of new deep structures such as residual neural networks and convolutional neural networks (CNN). These methods are applied to image, text, and audio areas, and can also be used for processing time-series data and related analysis. A multivariate long short-term memory fully convolutional network (LSTM-FCN) was proposed for TSC, which further improved the model’s classifcation accuracy by improving the structure of the full convolution block [8]. JIANG solved the question of imbalanced time series in industrial applications. Jiang proposes a novel anomaly detection approach based on generative adversarial networks (GAN) to overcome this problem [9]. Deng proposed Imputation Balanced GAN (IB-GAN), a novel method that joins data augmentation and classifcation in a one-step process via an imputation-balancing approach. Empirical experiments show signifcant performance gains against state-of-the-art GAN baselines [10, 11]. Li proposed an unsupervised multivariate anomaly detection method based on Generative Adversarial Networks (GANs), using the Long-Short-Term-Memory Recurrent Neural Networks (LSTM-RNN) as the base models (namely, the generator and discriminator) in the GAN framework to capture the temporal correlation of time series distributions. The experimental results showed that the proposed MAD-GAN is efective [12].

Inspired by the classifcation application of DL in the image feld, such as generative adversarial networks (GAN), which have achieved remarkable success in generating high-quality images in computer vision, we explore a DL framework for multivariate FTS classifcation. The model uses convolutional long short-term memory (ConvLSTM) as the generator to learn the distribution characteristics of the data and multilayer perceptron (MLP) as the discriminator to discriminate whether the output data of the generator are true or false. The model is similar to a tiny two-person game. We can roughly divide stock prices into three categories (short, neutral, and long), similar to the following example. For instance, a customer’s rating on a movie might be a do-not-bother, only-if-you-must, good, very-good, and runto-see. The ratings have a natural order, which distinguishes ordinal regression from general multiclass classifcation [13]. The idea of ordinal regression has been added to machine learning in many studies. For example, Crammer and Singer [14] used the multi-thresholding of the online perceptron algorithm to perform ordinal regression, which has a similar meaning to our multi-classifcation of stock price trends [13]. The two situations we wanted to see in the model’s prediction results are as follows: frst, predicting a short as long will cause a signifcant loss in actual trading, and second, forecasting the long as a short, which will cause us to lose the opportunity to proft in this transaction. Therefore, we made improvements to the discriminator in the GAN model for the above two cases. The detailed steps are presented in Sect. 3.3. We evaluated the performance of our model on three publicly available stock datasets and selected several classic comparison methods. The experimental results show that the classifcation performance of the ordinal regression GAN for fnancial volatility trends (ORGAN-FVT) model proposed in this study on the three

FTS datasets of TSLA, PAICC, and MSFT is signifcantly improved compared to the competitors, especially the model optimised by adding the ordinal regression idea in the classifcation results. Moreover, it has practical signifcance because the model has a better practical reference value and less preprocessing.

We summarise our contributions as follows:

We propose an effective GAN-based volatility trend multi-classifcation model for multivariate FTS based on stock data with multiple technical indicators. To the best of our knowledge, such generative multi-classifcation model hasn’t been studied much in existing research. We improve the generator of GAN by adopting ConvLSTM to efciently capture temporal dependencies and exploit ordinal regression in the discriminator to achieve better multi-classifcation performance. Based on experimental comparison on three real-world stock datasets with state-of-the-art methods, the proposed ORGAN-FVT model outperforms its competitors and leads to a practical performance in fnancial predictions.

This paper is organised as follows: Sect.  2 reviews the relevant research work. Section 3 introduces the architecture of the proposed ORGAN-FVT model in detail. Section 4 presents the experimental settings and results. Finally, we present our conclusions in Sect. 5.

# 2 Related Works

The task of time series classifcation (TSC) is to obtain a classifier through training on the datasets, which can obtain the probability distribution of the input data variable mapped to the label. Time series classifcation is an important and challenging problem in data mining. Unlike traditional classification methods, time series classifcation methods have time series characteristics, so the numerical relationship between diferent attributes needs to be considered when classifying and the order relationship between time series points. From the perspective of time development, we can divide time series classifcation into early time series classifcation and deep learning-based classifcation methods.

Traditional TSC methods have performed well in many scenarios, such as the model-based classifcation method, and the ftted regression model proposed by Zhang et al. [15–17]. These methods frst generate a specifc model for each time series and then look for the diferences between the models. The similarity is used to realise TSC, and it has an excellent classifcation advantage for time series with solid model adaptability.The early time series classifcation method based on prefx was frst proposed in the paper [18]. The idea is to obtain the MPL (Minimum Prefx Length, MPL) of the time series through a large amount of data training and then classify the test set. Classifcation methods based on local features, such as the shapelet classifcation method proposed by Ye and Keogh in 2009 [19], the classifcation results of the entire sequence are obtained by calculating the similarity between the features of diferent sub-sequence. TSC algorithms based on global features treat the entire time series as a full feature and classify it by calculating the similarity between the entire time series, such as the distance-based K-nearest neighbour classifcation method, have proven to have good performance. Moreover, an increasing number of studies have demonstrated that dynamic time warping is the best method for sequence distance measurement in most felds [20–22]. The key to the classifcation methods based on local features is to fnd local data features with clear classifcation features [23]. The classifer is more efcient in performing corresponding classifcation operations because the sub-sequence length that can refect the local features is much shorter than the entire time series.

The fnancial time series classifcation method based on deep learning have been extensively studied. Since DNN (Deep Neural Network, DNN) changed computer vision, the classifcation method based on deep learning has gradually been applied in time series classifcation. For example, Michael et al. took the lead in applying RNN to time series classifcation [24, 25]. Due to its robust feature extraction capabilities, deep convolutional neural networks (DCNN) have been added to the time series classifcation and combined with other models to achieve good results. Yi et al. explored and transformed CNN (Convolutional Neural Network, CNN) and proposed the MC-DCNN model to improve the classifcation performance [26, 27]. At the same time, in order to better classify the time series, before training the model, stacked denoising autoencoding (SDAE) are used in pre-training, which can better learn the potential features of the time series model.In the multi-classifcation problem of price fuctuation trends in fnancial time series, fnancial time-series classifcation (FTC) has signifcant value for investment managers. Therefore, it has attracted much attention in the past few decades. Kim and Han [28] have proposed feature selection methods based on Genetic Algorithm (GA) combined with a NN model to select useful features to predict the trend of stock price. Teixeira et al. [29] have used the technical indicators, which often are used in technical analysis, as the representation of fnancial data and feed them into the classifcation model for FTC. Durn-Rosal et al. [30] have used piecewise linear regression-based turning points to segment the target sequence, and then use a NN to predict these points.

Many DL methods have been applied in the classifcation of time series. With the continuous development of DL in various classifcations, the DCNN proposed by Krizhevsky has achieved great success in the feld of computer vision [31], in particular, graphics recognition tasks, such as GAN, have achieved remarkable success in computer vision high-quality image generation. The application scenarios of GAN have been rapidly developed, covering images, texts, and time series. GAN have been increasingly researched for data generation, anomaly detection, timeseries prediction, and classifcation as researchers continue to invest in them. Goodfellow et al. frst proposed GAN to generate high-quality pictures [32]. Later, Zhan used improved GAN and LSTM to predict satellite images [33], thus obtaining important resources for weather forecasting. The network can efectively capture the evolutionary rules of the weather system, which guides people to accurately forecast the weather. Recently, an increasing number of studies have used generative adversarial nets (GAN) in FTS, and the research on price trend fuctuation prediction is of great practical value. Zhang et al. applied GAN to stock price prediction [34], used GAN to capture the distribution of actual stock data, and achieved good results compared with existing DL methods. Feng proposed a method based on adversarial training to improve neural network prediction models [35]. Their idea is to add disturbances to simulate the randomness of price variables to enhance the model’s generalisation ability and fnally predict whether the stock price is rising or falling. The results show that their model performs better than the existing methods. Given the complex, highly noisy, dynamic, nonlinear, non-parametric, and chaotic characteristics of FTS, whether GAN can successfully learn the data distribution characteristics of FTS price trends is a signifcant challenge.

Previously, some researchers have used GAN to enhance data to optimise classifcation performance or used GAN to study FTS price forecasts. Others have used adversarial learning to predict price declines and rises. However, few studies have been done on price trend prediction for fnancial time series. Moreover, there is even less research for deep-learning-based price trend multi-classifcation of fnancial time series. Feng explored a model for adversarial training, but it was aimed at binary classifcation research. Considering the unbalanced distribution of data trends in the multi-classifcation problem, especially if there are only few sudden drop and rise samples, whether the model can solve these problems will be a considerable challenge. Inspired by previous research, we use GAN to conduct a multi-classifcation study of price movements in FTS. According to the characteristics of FTS, we know that the challenge of this research is how to allow GAN to learn the price data trend distribution of the original data to achieve a better performance in end-to-end classifcation. Meanwhile, the three-classifcation research on the FTS price trend is more challenging than binary classifcation. However, it has an outstanding reference value for stock trading.

# 3 Methodology

# 3.1 Proposed ORGAN‑FVT Method

The characteristics of FTS determine the difculty of its research, but the research has considerable market value. Therefore, learning the trend distribution of data and predicting price fuctuations is worth studying. Many researchers have proposed algorithms to solve these problems. This study proposes a generative adversarial networks from another perspective. GAN is a framework that trains two models, such as a zero-sum game. In the adversarial process, the generator can be seen as a cheater to generate data similar to the actual data. Simultaneously, the discriminator plays the role of a judge to distinguish between the actual and generated data. They can reach an ideal point named the Nash equilibrium state, where the discriminator cannot distinguish between these two types of data. At this point, the generator can obtain the data distribution of the original input. We propose a new GAN architecture for end-to-end three-classifcation of stockclosing price trends based on this principle. Undoubtedly, price and transaction volume is signifcant for closing price’s three-category prediction. In addition, technical indicators calculated from price and transaction volume are widely used as input variables in previous studies. In papers [36–38] show that many fund managers and investors recognise these 11 technical indicators based on price and trading volume, and these technical indicators are commonly used in the stock market as signals of future market trends. Kim frst used these indicators in support vector machines for financial time series forecasting problems in 2003, and then Yakup Kara [36] proposed these indicators in his article. We know that a variety of technical indicators are available. Some technical indicators are efective under trending markets and others perform better under no trending or cyclical markets [39]. We selected these 11 technical indicators for model training based on previous research.

The closing price is an indicator commonly recognised by market participants, and it contains useful information. We selected the daily data of multiple stocks in recent decades, combined with 11 fnancial factors to classify short, neutral, and long stocks. The 11 technical indicators of stock data in a day are indicators $=$ {‘Close’, ‘High’, ‘Low’, ‘Open’, ‘RSI’, ‘ADX’, ‘CCI’, ‘FASTD’, ‘SLOWD’, ‘WILLER’, and ‘SMA’} [40]. These 11 indicators are valuable in precious research, such as technical analysis and mean regression. Therefore, these technical indicators can be used as the input characteristics of stock data for a three-category study of price fuctuation trends. Our input is $X = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { t } \}$ , which is composed of daily stock data for $t$ days. Each input $X$ is a vector composed of the 11 indicators, and we input $X$ into the generator to obtain $\hat { C } _ { t + 1 }$ as false data and record it as $X _ { f a k e }$ . Simultaneously, we record $C _ { t + 1 }$ as real data recorded as $X _ { t r u e }$ . Based on the generator, we extract the output of ConvLSTM and put it into a fully connected layer to generate three types of probability matrices of short, neutral, and long through the softmax activation function, which is defned as follows:

![](images/03e48ca4c21e0bcf00388218ad003cf44e912d8ef77f2ca14395e27f9c958c0d.jpg)  
Fig. 1 The architecture of ORGAN-FVT

$$
C _ { t + 1 } = [ \alpha , \beta , \gamma ] , ( \alpha + \beta + \gamma = 1 ) .
$$

A detailed structure description is shown in Fig. 1. In the ORGAN-FVT model, both the generator and discriminator try to optimise a value function until they reach an equilibrium point, called the Nash equilibrium. Therefore, we can defne our value function $V ( G , D )$ as follows:

$$
\begin{array} { l } { \underset { G } { \mathrm { m i n } } \underset { D } { \mathrm { m a x } } = E [ \log D ( X _ { r e a l } ) ] \ } \\ { ~ + E [ \log ( 1 - D ( X _ { f a k e } ) ) ] . } \end{array}
$$

where $X _ { r e a l }$ denotes the actual input of discriminator, and $X _ { f a k e }$ denotes the fake input of generator. $D ( X _ { r e a l } )$ represents the discriminator’s actual output, and $D ( X _ { f a k e } )$ represents the fake output. Thus, a detailed description is provided in Sect. 3.3. When calculating the error of the probability matrix one-hot encoding, we use the cross-entropy loss function. Given two probability distributions $p$ and $q$ , the cross-entropy of $q$ expressed by $\mathbf { q }$ is defned as follows:

$$
H ( p , q ) = - \sum _ { i = 1 } ^ { n } p ( x ) \log q ( x ) .
$$

where $p$ represents the actual label, $n$ represents the category numbers, $i$ corresponds to the category order and $q$ represents the predicted label. We obtain the probability rate matrix $\hat { C } _ { t }$ and calculate the cross-entropy loss with the actual probability matrix $C _ { t }$ at that moment. We present the loss function defnitions of the generator and discriminator in the model in Sects. 3.2 and 3.3, respectively. Thus, we can obtain the losses of the generator and discriminator.

$$
D _ { l o s s } = \frac { 1 } { m } \sum ^ { m } H ( D ( X _ { r e a l } ) , D ( X _ { f a k e } ) ) .
$$

![](images/8449031d4308a53420f82308ef98fb87c47fbea13995d7e8b7da2eeea555825a.jpg)  
Fig. 2 The generator designed with an ConvLSTM

$$
G _ { l o s s } = \frac { 1 } { m } \sum _ { t = 1 } ^ { m } H ( C _ { t } , \hat { C } _ { t } ) .
$$

Where the $D _ { l o s s }$ denotes the training loss of discriminator, the $G _ { l o s s }$ denotes the training loss of generator, and the $m$ denotes the length of FTS.

In Fig. 1, the input of the model is an FTS, where $G _ { i n p u t }$ denotes the generator’s input, $D _ { f a k e \_ i n p u t }$ is the output of ConvLSTM with the softmax function in the last layer, and this output is used as the input of the discriminator. Simultaneously, take the real price trend one-hot matrix $D _ { r e a l \_ i n p u t }$ from the label of the original data as the input of the discriminator, and the discriminator distinguishes between the true and false outputs of the generator. Moreover, we have added the concept of ordinal regression to the discriminator, and the penalty for predicting long as short and short as long when discrimination increases, thus optimising our classifcation results. As shown in the legend of Fig.  1, ordinal regression works in the discriminator. It is an optimization method in the discriminator. The classifcation result shown by the blue arrow is allowed, and it is not allowed by the red arrow in the discriminator. Therefore, the discriminator will increase the penalty for the above two prediction errors. When the result of the generator is correct, the generator keeps the generated results and continues training. If it is wrong, it will return to the discriminator, which will increase the overall error of the GAN model. The generator and discriminator optimise the objective function in Eq. 8 and fnally obtain a generator that has learnt the data distribution trend. Eventually, we can use the trained generator to predict the results. We verifed its performance using the test sets. A specifc experimental description is provided in Sect. 4. We continue to provide a detailed description of the generator and discriminator.

# 3.2 The Generator

The generator in ORGAN-FVT is designed with ConvLSTM, which has stronger time-series data processing capabilities. The structure of the generator is shown in Fig. 2. It is composed of ConvLSTM, with 11 technical indicators as inputs. The goal is to let $\hat { C } _ { t + 1 }$ approach $C _ { t + 1 }$ The output of the generator $G ( X )$ is defned as follows:

$$
h _ { t } = g ( x ) .
$$

$$
G ( x ) = \hat { C } _ { t + 1 } = \delta ( W _ { h } ^ { T } h _ { t } + b _ { h } ) .
$$

where $g ( \cdot )$ denotes the output of ConvLSTM, and $h _ { t }$ is the output of the ConvLSTM with $X$ as the input. $\delta$ denotes the softmax activation function. $W _ { h }$ and $b _ { h }$ denote the weight and bias in the fully connected layer, respectively. We also used dropout as a regularisation method to avoid overftting. Additionally, we can use the concept of a sliding window to predict $\hat { C } _ { t + 1 }$ by $\hat { C } _ { t }$ and $X$ .

# 3.3 The Discriminator

The role of the discriminator is to construct a diferentiable function D to classify the input data. The discriminator distinguishes the authenticity of the generator’s data by discriminating between the actual input data and the false input data. We chose the MLP as the generator model, where $h _ { 1 } , h _ { 2 } , h _ { 3 }$ , and $h _ { 4 }$ are fully connected layers. The Relu activation function was used between the hidden layers, and the softmax function was used for the output layer. Regarding the input and output of the discriminator, we provide the following description. The output of the discriminator is defned as follows:

$$
D ( X _ { f a k e } ) = \rho ( d ( X _ { f a k e } ) .
$$

$$
D ( X _ { r e a l } ) = \rho ( d ( X _ { r e a l } ) .
$$

where $d ( \cdot )$ denotes the output of MLP and $\rho$ denotes the softmax activation function, and $X _ { f a k e }$ and $X _ { r e a l }$ are probability matrices with one row and three columns, representing the probability of the short, neutral, and long at that moment. In Fig. 3, we show the structure of the discriminator. We optimise the prediction results accordingly in the following two situations, which are described as follows:

(a) The true label is short (represented by a one-hot matrix as [1,0,0]). We make $\beta$ in the prediction result $\hat { C } _ { t + 1 } = [ \alpha , \beta , \gamma ] , ( \alpha + \beta + \gamma = 1 )$ as large as possible, instead of $\gamma$ , to avoid forecasting short as long;   
(b) The true label is long ([0,0,1]). We make $\beta$ in the prediction result $\hat { C } _ { t + 1 }$ as large as possible instead of $\alpha$ , so we can try to avoid forecasting for long as short.

This method provides constraints with practical trading signifcance for the model. The objective function of the discriminator is as follows:

![](images/d53fc96cfbd0552a7308449c57b4f2ceb14c20ea5e1bb8ab70b8a1c2f6f6e1cc.jpg)  
Fig. 3 Discriminator designed using an MLP with $X _ { r e a l }$ and $X _ { f a k e }$ as the inputs

$$
D _ { l o s s } = L o s s _ { r e a l } + L o s s _ { f a k e } .
$$

where $L o s s _ { f a k e }$ denotes the cross-entropy loss between the generator’s input and the negative sample data, $L o s s _ { f a k e }$ is the model’s loss function after the discriminator discriminates the generator’s input from the real data, and $L o s s _ { r e a l }$ is the discriminator’s training loss. The concept of ordinal regression is embodied in $L o s s _ { f a k e }$ . The abovementioned two situations are constrained by $L o s s _ { f a k e }$ to optimise the classifcation results of our model. For convenience, let $\kappa = [ 1 , 0 , 0 ]$ , $\nu = [ 0 , 1 , 0 ]$ , and $\mathbf { 0 } = [ 0 , 0 , 1 ]$ . The defnitions of $L o s s _ { r e a l }$ and $L o s s _ { f a k e }$ are given as follows:

$$
L o s s _ { r e a l } = H ( { \cal D } ( X _ { r e a l } ) , C _ { t + 1 } ) .
$$

$$
L o s s _ { f a k e } = \left\{ \begin{array} { l l } { H ( 0 , \hat { C } _ { t + 1 } ) , I f \hat { C } _ { t + 1 } = \kappa } \\ { H ( \nu , \hat { C } _ { t + 1 } ) , I f \hat { C } _ { t + 1 } = \nu } \\ { H ( \kappa , \hat { C } _ { t + 1 } ) , I f \hat { C } _ { t + 1 } = 0 } \end{array} \right.
$$

The frst loss $L o s s _ { r e a l }$ of the discriminator can be obtained by calculating the cross-entropy loss between the discriminator and the actual data label $C _ { t + 1 }$ . In Eq. 10, the second loss $L o s s _ { f a k e }$ of the discriminator can be obtained by calculating the cross-entropy loss between the predicted value of the generator and the negative sample label. The structure of the model is illustrated in Fig. 3 below.

# 4 Evaluation

# 4.1 Datasets

We selected actual stock trading data from the1 Yahoo Finance to evaluate our model and selected several classic

DL methods as baseline methods. These stock data include three data sets: Tesla Motors (TSLA) stock price, PAICC, and Microsoft Corporation (MSFT) of the National Securities Exchange Negotiation (NASDAQ), which can be downloaded from the Yahoo website. A detailed description of the dataset is given in Table 1. Each stock contains several information indicators such as the opening price (Open), highest price (High), lowest price (Low), closing price (Close), and trading volume (Volume). We construct our label data using the closing price (Close) and defne $x _ { i + 1 } - x _ { i } > \mu$ as short, $x _ { i + 1 } - x _ { i } < \theta$ as long, and $x _ { i + 1 } - x _ { i } = \lambda$ as neutral $\left( 0 < i < n \right)$ , where $\mu , \theta , \lambda \geq 0$ is the parameter set according to the corresponding stock. Additionally, previous studies have also widely used technical indicators calculated from prices and transaction volumes as input variables [40]. In addition to the fundamental market indicators of the input variables in this study, 11 other technical indicators are selected, provided by Eq. 1. According to the above technical indicators as input variables, we frst normalise the data with $\mathbf { Z }$ -scores to eliminate the infuence of the dimensions between diferent variables. Our goal is to predict the trend of the stocks closing price on the next day and obtain the trend of the closing price on the $t + 1$ day through the input $X _ { t }$ of the past t days. Through repeated experiments in this study, we set $t$ to 30. Our data are divided into training and testing. We select the frst $8 5 { - } 9 0 \%$ of the data on each stock as the training set and the rest $( 1 0 - 1 5 \% )$ as the test set. We present the trend chart of the three datasets in Fig. 4.

Figure 4 shows the trend chart of the closing prices of the three stock data over time. We can intuitively see that the price trends in the three datasets are diferent. The closing prices in the MSFT dataset fuctuated from the beginning. When it reached 2000, it began to decline in an oscillating trend before remaining in a long-term turbulence “stable” until it began to rise in 2012. The closing price in the PAICC dataset fuctuated upward and downward as a whole. In contrast, the closing price in the TSLA data set has been stable from 2010 to 2020 without signifcant fuctuations and then rapidly rises to the absolute shock dropped. Note that the three datasets represent diferent data trends, cover most of the natural scenes, and better refect the robustness of diferent models. Table 1 shows the detailed data description and dataset division. The rows in the table indicate the date range of the dataset, the length of the dataset, the length of the training set, the length of the validation set, and the length of the test set in the divided data set.

# 4.2 Experimental Settings

In our experiments, the ConvLSTM module of the generator and the MLP module of the discriminator remained unchanged. The optimal number of ConvLSTM cells was found to be in the range of 8–256 cell units through a hyperparameter search. The number of flters in the convolutional layer was set to 256 and 128, the size of the convolution kernel was two, and the activation function was ELU. After the convolutional layer, we add a pooling layer of size two, the convolutional layer is connected to the LSTM layer, and the number of cells is 100, 100. Then, a fully connected layer is output with the softmax activation function. For the fairness of the experiment, we also used the generator parameter settings in the ConvLSTM benchmark method.

Table 1 The details description of our datasets   

<table><tr><td>Name</td><td>MSFT</td><td>PAICC</td><td>TSLA</td></tr><tr><td>Date range</td><td>1999/1/4</td><td>2004/6/24</td><td>2010/6/29</td></tr><tr><td></td><td>- 2018/12/31</td><td>- 2018/12/28</td><td>- 2021/5/27</td></tr><tr><td>Length</td><td>5031</td><td>3594</td><td>2748</td></tr><tr><td>Training length</td><td>4276</td><td>3055</td><td>2336</td></tr><tr><td>Validation length</td><td>252</td><td>180</td><td>137</td></tr><tr><td>Test length</td><td>503</td><td>359</td><td>275</td></tr></table>

![](images/bff0ea195da3b60a54899b87ac3399028375351cbc3139ecf29854d7246fc169.jpg)  
Fig. 4 The trend images of three datasets

The number of cells in the four layers of the discriminator is 256, 128, 100, and 3, and the softmax activation function is used in the last fully connected layer to output the probability matrix of the three classifcations. The training epochs are kept at 1000, and we set the initial batch size to 60. To prevent overftting, we add a dropout layer with a value of 0.2 after the CNN layer and the LSTM layer. The method used in this study and the existing comparative experimental methods are all trained using the Adam optimiser [41]. The initial learning rate of the generator is 1e-3, and the fnal learning rate was 1e-4, and the learning rate of the discriminator is set to the generator 1.2 times a learning rate. For every 50 epochs, if the recall index on the validation set did not improve, the learning rate decreased by 2e-5 until the fnal learning rate is reached. All model training was performed using the Keras version 2.3.1 library with TensorFlow version 2.0 background. The experimental operating system was Ubuntu 16.04, and an NVIDIA GeForce GTX 1080Ti GPU. Third-party libraries, such as Talib, were used to calculate technical indicators.

# 4.3 Evaluation Metrics

In this section, we provide a detailed description of the multi-classifcation indicators used in this study. The specifc evaluation indicators are true positive (TP), false negative (FN), false positive (FP), true negative (TN), accuracy, precision, recall, and harmonic mean f1-score based on accuracy and recall. Considering the complex characteristics of FTS, especially the uneven data distribution, we selected the weighted-average indicator. Meanwhile, considering that the small samples in the actual application scenarios of trend prediction are also worthy of attention (such as sudden skyrocketing and falling), we also selected the macro-indicator to better refect the robustness of the model in the experiment. We provide a detailed description of these indicators. Our classifcation strategy is short, neutral, and long, recorded as 1, 2, and 3, respectively. The following descriptions are provided.

According to the confusion matrix, the following classifcation performance indicators can be obtained:

# 1 Weighted average

The indicator assigns weights according to diferent categories, and each category is multiplied by its weight and then added. This method considers the imbalance of categories, and its value is more likely to be afected by common categories. The weight ratio of the number of diferent categories is $W _ { 1 } : W _ { 2 } : W _ { 3 } = N _ { 1 } : N _ { 2 } : N _ { 3 }$ , where $W$ and $N$ denotes the weight and actual number of samples in this category, respectively.

Table 2 Confusion matrix   

<table><tr><td>Predicted True</td><td>1 (short)</td><td>2 (neutral)</td><td>3 (long)</td></tr><tr><td>1 (short)</td><td>TP1</td><td>FP2</td><td>FP</td></tr><tr><td>2 (neutral)</td><td>FP1</td><td>TP2</td><td>F2P</td></tr><tr><td>1 (short)</td><td>FP1</td><td>FP</td><td>TP</td></tr></table>

$$
W e i g h t e d - p r e c i s i o n = \sum _ { i = 1 } ^ { 3 } P _ { i } W _ { i } .
$$

$$
W e i g h t e d - R e c a l l = \sum _ { i = 1 } ^ { 3 } R _ { i } W _ { i } .
$$

$$
\cdot f 1 - s c o r e = \sum _ { i = 1 } ^ { 3 } F S _ { i } W _ { i } .
$$

where $\begin{array} { r } { P = \frac { T P } { T P + F P } } \end{array}$ ( $P$ denotes Precision), $\begin{array} { r } { R = \frac { T P } { T P + F N } } \end{array}$ ( $R$ denotes Recall), $\begin{array} { r } { F S = \frac { T P } { \frac { 1 } { P } + \frac { 1 } { R } } } \end{array}$ ( $F S$ denotes f1-score), and the subscript $i$ corresponds to the category order in Table 2.

# 2 Macro-average

This indicator directly adds up the evaluation indicators of diferent categories (Precision/Recall/f1-score) to the average. The feature of this method is to treat each category equally but will be afected by classes with fewer numbers.

$$
M a c r o ^ { - } p r e c i s i o n = ( P _ { 1 } + P _ { 2 } + P _ { 3 } ) / 3 .
$$

where $\begin{array} { r } { P = \frac { T P } { T P + F P } ( P } \end{array}$ denotes Precision),

$$
M a c r o - r e c a l l = ( R _ { 1 } + R _ { 2 } + R _ { 3 } ) / 3 .
$$

where $\begin{array} { r } { R = \frac { T P } { T P + F N } \left( R \right. } \end{array}$ denotes Recall),

$$
f 1 - s c o r e = ( F S _ { 1 } + F S _ { 2 } + F S _ { 3 } ) / 3 .
$$

where $\begin{array} { r } { F S = \frac { T P } { \frac { 1 } { P } + \frac { 1 } { R } } } \end{array}$ (FS denotes f1-score).

# 3 Area under curve (AUC)

AUC is the receiver operating characteristic (ROC) curve area for each category. We refer to the defnition used to calculate the AUC indicator [42]. To describe the AUC, we frst provide the true positive rate (TPR) and false positive rate (FPR) defnition.

$$
T P R = T P / ( T P + F N ) .
$$

$$
F P R = F P / ( F P + T N ) .
$$

We can obtain the ROC for each category, and the AUC can be calculated from the ROC curve area under each category.

# 4.4 Experimental Results

Table 3 The experiment result on TSLA   

<table><tr><td>Indicator</td><td>LSTM</td><td>GRU</td><td>CNN</td><td>ConvLSTM</td><td>GAN-FVT</td><td>ORGAN-FVT</td></tr><tr><td>Class0 AUC</td><td>0.5094</td><td>0.5395</td><td>0.5037</td><td>0.4831</td><td>0.5482</td><td>0.5212</td></tr><tr><td>Class1 AUC</td><td>0.6201</td><td>0.6126</td><td>0.5955</td><td>0.5482</td><td>0.6372</td><td>0.7698</td></tr><tr><td>Class2 AUC</td><td>0.5391</td><td>0.5124</td><td>0.4956</td><td>0.5014</td><td>0.5364</td><td>0.5218</td></tr><tr><td>Macro-precision</td><td>0.3451</td><td>0.3529</td><td>0.3803</td><td>0.3294</td><td>0.3695</td><td>0.3816</td></tr><tr><td>Macro-recall</td><td>0.3473</td><td>0.3518</td><td>0.4058</td><td>0.3822</td><td>0.4529</td><td>0.4707</td></tr><tr><td>Macro-f1-score</td><td>0.3087</td><td>0.3051</td><td>0.3761</td><td>0.2922</td><td>0.3495</td><td>0.3858</td></tr><tr><td>Weighted precision</td><td>0.5199</td><td>0.5324</td><td>0.5182</td><td>0.4862</td><td>0.5332</td><td>0.5378</td></tr><tr><td>Weighted recall</td><td>0.4814</td><td>0.4839</td><td>0.4888</td><td>0.3722</td><td>0.4615</td><td>0.5012</td></tr><tr><td>Weighted-f1-score</td><td>0.4439</td><td>0.4363</td><td>0.4843</td><td>0.4159</td><td>0.4713</td><td>0.5133</td></tr></table>

We conducted a detailed experimental analysis on the three datasets of TSLA, PAICC, and MSFT based on several different comparison methods. We selected macro, weighted, and AUC based on the multi-classifcation indicators given in Sect.4.2. Among them, the indicators of AUC include the classifcation performance description of each category of short, neutral, and long. Macro and weighted include the corresponding precision, recall, and f1-score indicators. The detailed results are listed in Tables 3, 4, and 5. Our method corresponds to GAN-FVT and ORGAN-FVT in the table. GAN-FVT model was optimised using ordinal regression, whereas ORGAN-FVT was optimised by adding ordinal regression. For ease of description, the bold value in our table represents the best value in the comparison, and the underlined value indicates the second best. Simultaneously, the Macro-f1-score and Weighted-f1-score indicators of different methods on the three datasets are shown in Figs. 5, 6, and 7.

Table 4 The experiment result on PAICC   

<table><tr><td>Indicator</td><td>LSTM</td><td>GRU</td><td>CNN</td><td>ConvLSTM</td><td>GAN-FVT</td><td>ORGAN-FVT</td></tr><tr><td>Class0 AUC</td><td>0.5603</td><td>0.4890</td><td>0.5663</td><td>0.4389</td><td>0.5427</td><td>0.5759</td></tr><tr><td>Class1 AUC</td><td>0.6621</td><td>0.5867</td><td>0.6611</td><td>0.3216</td><td>0.5035</td><td>0.6647</td></tr><tr><td>Class2 AUC</td><td>0.5427</td><td>0.4950</td><td>0.5028</td><td>0.4904</td><td>0.4998</td><td>0.5228</td></tr><tr><td>Macro-precision</td><td>0.2940</td><td>0.3634</td><td>0.3148</td><td>0.2887</td><td>0.3842</td><td>0.4010</td></tr><tr><td>Macro-recall</td><td>0.3351</td><td>0.3362</td><td>0.3365</td><td>0.3251</td><td>0.3368</td><td>0.3653</td></tr><tr><td>Macro-f1-score</td><td>0.2229</td><td>0.3314</td><td>0.2257</td><td>0.2781</td><td>0.2560</td><td>0.3586</td></tr><tr><td>Weighted precision</td><td>0.3850</td><td>0.4031</td><td>0.4120</td><td>0.3780</td><td>0.4148</td><td>0.4257</td></tr><tr><td>Weighted recall</td><td>0.4442</td><td>0.4212</td><td>0.4462</td><td>0.4288</td><td>0.4269</td><td>0.4385</td></tr><tr><td>Weighted-f1-score</td><td>0.2951</td><td>0.4041</td><td>0.2987</td><td>0.3657</td><td>0.3158</td><td>0.4114</td></tr></table>

Table 5 The experiment result on MSFT   

<table><tr><td>Indicator</td><td>LSTM</td><td>GRU</td><td>CNN</td><td>ConvLSTM</td><td>GAN-FVT</td><td>ORGAN-FVT</td></tr><tr><td>Class0 AUC</td><td>0.5129</td><td>0.5358</td><td>0.5221</td><td>0.5220</td><td>0.5335</td><td>0.5170</td></tr><tr><td>Class1 AUC</td><td>0.5440</td><td>0.4930</td><td>0.5322</td><td>0.4856</td><td>0.5027</td><td>0.5703</td></tr><tr><td>Class2 AUC</td><td>0.5273</td><td>0.5632</td><td>0.5256</td><td>0.5310</td><td>0.5033</td><td>0.5724</td></tr><tr><td>Macro-precision</td><td>0.3506</td><td>0.3179</td><td>0.3575</td><td>0.3438</td><td>0.3609</td><td>0.3639</td></tr><tr><td>Macro-recall</td><td>0.3528</td><td>0.3450</td><td>0.3585</td><td>0.3425</td><td>0.3536</td><td>0.3669</td></tr><tr><td>Macro-f1-score</td><td>0.3134</td><td>0.3011</td><td>0.3519</td><td>0.3400</td><td>0.3463</td><td>0.3584</td></tr><tr><td>Weighted precision</td><td>0.3670</td><td>0.3407</td><td>0.3690</td><td>0.3588</td><td>0.3732</td><td>0.3784</td></tr><tr><td>Weighted recall</td><td>0.3664</td><td>0.4040</td><td>0.3597</td><td>0.3450</td><td>0.3705</td><td>0.3664</td></tr><tr><td>Weighted-f1-score</td><td>0.3299</td><td>0.3414</td><td>0.3575</td><td>0.3490</td><td>0.3607</td><td>0.3664</td></tr></table>

![](images/1f187254c09eda023ed82dcd13c23f13ba14b9e0ee09fa2485d48398ff25110c.jpg)  
Fig. 5 The f1-scores of TSLA

# 4.4.1 Results On Three Datasets

From Table 3, note that on the TSLA dataset, the ORGANFVT model performed better than the contrasted DL methods on seven indicators, primarily the indicator on Class1 AUC reached 0.7698. Compared with the highest value of 0.6372 in the comparison method, an increase of 0.1326. In the other evaluation indicators, the macro-average and weighted average are better than the contrasted DL methods.

![](images/00835667f75c5518dbdb5ad60ecf7e72be063281bb296b54ef6398549a354649.jpg)  
Fig. 6 The f1-scores of PAICC

As can be observed in Fig. 5, compared to GAN-FVT, the ORGAN-FVT has improved by 0.0363 and 0.042 in the indicators of Macro-f1-score and Weighted-f1-score.

From Table  4, note that on the PAICC dataset, the ORGAN-FVT model performed better than the contrasted DL methods on seven indicators, especially when the indicators on Macro-f1-score reached 0.3586. Compared with the highest value of 0.3314 in the comparison method, it is increased by 0.0272. As can be observed in Fig. 6, compared to the GAN-FVT, the ORGAN-FVT model has improved on both Macro-f1-score and Weighted-f1-score, increasing by 0.1026 and 0.0956, respectively.

![](images/7f9c7d390abffc27ef010db6135fb6c868429545830408187d879835b4509d2f.jpg)  
Fig. 7 The f1-scores of MSFT

From Table  5, note that on the MSFT dataset, the ORGAN-FVT model performed better than the contrasted DL methods on seven indicators, primarily the Class1 AUC indicator reached 0.5703. Compared with the highest value of 0.5440 in the comparison method, it is improved by 0.0263. As can be observed in Fig. 7, compared to the GAN-FVT model, the optimised ORGANFVT model slightly improved in Macro-f1-score and Weighted-f1-score.

From Tables 3, 4 and 5, we know that the ORGAN-FVT model outperforms the existing DL methods for most indicators. Note that we selected the best performance among the methods for comparison with our method. Based on a separate indicator for each dataset, our method will have a more remarkable improvement. The generator in our ORGAN-FVT was the ConvLSTM. Tables 3, 4 and 5 show that GAN-FVT and ORGAN-FVT improved on several of the nine indicators.

# 4.4.2 Results Based on Ordinary Regression

Note that ConvLSTM is added as a generator to GCN, and the classifcation performance is improved compared to the end-to-end ConvLSTM, and the ORGAN-FVT model optimised by adding ordinal regression improved classifcation results compared with the GAN-FVT model. We present the confusion matrix results for the experimental data set in Table 6. In Table 6, the frst type of error is used to predict long as short and neutral. Next, we provide the proportion of short in short and neutral. The smaller ones are marked with bold numbers in parentheses, and the larger ones are marked with the underline and perform the same operation in the second type of error predicting short as long and neutral. In the TSLA and PAICC datasets, our model ORGANFVT outperforms GAN-FVT in the above two cases, with the short ratio decreasing in the frst type of error and the long ratio decreasing in the second type of error. However, on the MSFT dataset with a relatively uniform data distribution in the test set, ORGAN-FVT performs slightly worse than GAN-FVT in avoiding the situation of predicting short as long. We assume that the data distribution of this dataset is more suitable for the GAN-FVT model; therefore, in this case, it outperforms ORGAN-FVT. However, after ordinal regression optimisation, the ORGAN-FVT model performs better than GAN-FVT in the above two cases, and the overall performance is improved.

Table 6 The results of confusion matrix in the datasets   

<table><tr><td></td><td colspan="3">GAN-FVT</td><td colspan="4">ORGAN-FVT</td></tr><tr><td></td><td colspan="3">Predicted</td><td></td><td colspan="3">Predicted</td></tr><tr><td>True</td><td>Long</td><td>Neutral</td><td>Short</td><td>True</td><td>Long</td><td>Neutral</td><td>Short</td></tr><tr><td>TSLA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Long</td><td>104</td><td>7</td><td>63 (0.90)</td><td>Long</td><td>93</td><td>10</td><td>71 (0.87)</td></tr><tr><td>Neutral</td><td>3</td><td>2</td><td>0</td><td>Neutral</td><td>3</td><td>2</td><td>0</td></tr><tr><td>Short</td><td>135 (0.94)</td><td>8</td><td>81</td><td>Short</td><td>102 (0.87)</td><td>15</td><td>107</td></tr><tr><td>PAICC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Long</td><td>Neutral</td><td>Short</td><td></td><td>Long</td><td>Neutral</td><td>Short</td></tr><tr><td>Long</td><td>148</td><td>4</td><td>78 (0.95)</td><td>Long</td><td>109</td><td>9</td><td>112(0.92)</td></tr><tr><td>Neutral</td><td>22</td><td>2</td><td>42</td><td>Neutral</td><td>34</td><td>11</td><td>21</td></tr><tr><td>Short</td><td>128 (0.96)</td><td>5</td><td>91</td><td>Short</td><td>86 (0.80)</td><td>21</td><td>117</td></tr><tr><td>MSFT</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Long</td><td>53</td><td>74</td><td>119 (0.62)</td><td>Long</td><td>63</td><td>99</td><td>84 (0.45)</td></tr><tr><td>Neutral</td><td>26</td><td>61</td><td>102</td><td>Neutral</td><td>43</td><td>81</td><td>65</td></tr><tr><td>Short</td><td>57 (0.38)</td><td>91</td><td>162</td><td>Short</td><td>79 (0.43)</td><td>102</td><td>129</td></tr></table>

# 5 Discussion and Conclusion

Our improved GAN model significantly improves the existing DL methods in researching the movement trend classification of financial time-series prices. We add ConvLSTM to our model as the generator, which has excellent time-series processing capabilities. The experimental results show that it is better than CNN and ConvLSTM alone in end-to-end classifcation. Furthermore, the addition of the concept of ordinal regression improves the performance of our model in the classifcation results, particularly when predicting short to long and long to short, which is not conducive to actual trading in reality. Compared with GAN-FVT, ORGAN-FVT has been further optimised under the above circumstances so that our model improves the overall classifcation performance and guides actual transactions. The experimental results also show that our model has improved classifcation performance compared to the benchmark method ORGAN-FVT on datasets with diferent distribution characteristics. However, the proposed ORGAN-FVT model still has the following limitations: (1) The 11 technical indicators selected in this experiment may not be the best, which requires further research to optimise diferent indicator combinations that may have different effects on model performance and (2) In this study, we investigated FTS; however, whether the model can be applied in other time series is worth studying. A comparative analysis of the above factors is also a follow-up work arrangement.

Author Contributions LL proposed the theoretical analysis and methodology, completed the investigation and conceptualization, conducted the experiments and software, wrote original draft; ZP provided the methodology. PC finished conceptualization, data curation, writing-review & editing, supervision and funding acquisition. HL fnished review-writing & editing and provided funding acquisition. ZG completed validation and provided resources. KF and ZG completed validation and visualisation works.

Funding This research was funded by China Scholarship Council, National Natural Science Foundation of China, grant number 71971174, Department of Science and Technology of Sichuan Province, grant number 2020YFG0326 and Department of Science and Technology of Sichuan Province, grant number 2021JDR0222.

Data Availability The data that support the fnding of this study are available in the public domain: https://fnance.yahoo.com/.

# Declarations

Conflict of Interest The authors declare that they have no competing interests.

Ethical Approval Not applicable.

Consent to Participate Not applicable.

Content for Publication Not applicable.

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons. org/licenses/by/4.0/.

# References

1. Maleki, Mohsen, Mahmoudi, Mohammad Reza, Wraith, Darren, Pho, Kim-Hung.: Time series modelling to forecast the confrmed and recovered cases of Covid-19. Travel Med Infect. Dis. 37, 101742 (2020)   
2. Sezer, O.B., Gudelek, M.U., Ozbayoglu, A.M.: Financial time series forecasting with deep learning: a systematic literature review: 2005–2019. Appl. Soft Comput. 90, 106181 (2020)   
3. Gao, Zhong-Ke., Small, Michael, Kurths, Juergen: Complex network analysis of time series. Europhys. Lett. 116(5), 50001 (2017)   
4. Liu, L., Pei, Z., Chen, P., Gao, Z., Gan, Z., Feng, K.: An improved quantile-point-based evolutionary segmentation representation method of fnancial time series. Int. Arab J. Inf. Technol. 19(6), 873–883 (2022)   
5. Yang, Qiang, Xindong, Wu.: 10 challenging problems in data mining research. Int. J. Info. Technol. Decis. Making 5(04), 597–604 (2006)   
6. Bagnall, Anthony, Lines, Jason, Bostrom, Aaron, Large, James, Keogh, Eamonn: The great time series classifcation bake of: a review and experimental evaluation of recent algorithmic advances. Data Mining Knowl. Discov. 31(3), 606–660 (2017)   
7. Xi, X., Keogh, E., Shelton, C., Wei, L., Ratanamahatana, C.A.: Fast time series classifcation using numerosity reduction. In Proceedings of the 23rd international conference on Machine learning, pp. 1033–1040 (2006)   
8. Karim, Fazle, Majumdar, Somshubra, Darabi, Houshang, Harford, Samuel: Multivariate lstm-fcns for time series classifcation. Neural Netw. 116, 237–245 (2019)   
9. Jiang, W., Hong, Y., Zhou, B., He, X., Cheng, C.: A gan-based anomaly detection approach for imbalanced industrial time series. IEEE Access. 7, 143608–143619 (2019)   
10. Deng, G., Han, C., Dreossi, T., Lee, C., Matteson, D.S.: Ib-gan: A unifed approach for multivariate time series classifcation under class imbalance, arXiv preprint arXiv:2110.07460 (2021)   
11. Chen, P., Liu, H., Xin, R., Carval, T., Zhao, J., Xia, Y., Zhao, Z.: Efectively detecting operational anomalies in large-scale IoT data infrastructures by using a GAN-Based predictive model. Comput. J. 65(11), 2909–2925 (2022)   
12. Li, D., Chen, D., Jin, B., Shi, L., Goh, J., Ng, S.-K.: Mad-gan: Multivariate anomaly detection for time series data with generative adversarial networks. In: International Conference on Artifcial Neural Networks. Springer, pp. 703–716 (2019)   
13. Fathony, R., Bashiri, M.A., Ziebart, B.D.: Adversarial surrogate losses for ordinal regression. In NIPS, pp. 563–573 (2017)   
14. Lee, Daniel D., Pham, P., Largman, Y., Ng, A.: Advances in neural information processing systems 22. Technical report (2009)   
15. Zhang, Yong, Liu, Bo., Ji, Xiaomin, Huang, Dan: Classifcation of eeg signals based on autoregressive model and wavelet packet decomposition. Neural Process. Lett. 45(2), 365–378 (2017)   
16. Bagnall, Anthony, Janacek, Gareth: A run length transformation for discriminating between auto regressive time series. J. Classifcation 31(2), 154–178 (2014)   
17. Jeong, Young-Seon., Jayaraman, Raja: Support vector-based algorithms with weighted dynamic time warping kernel function for time series classifcation. Knowl-based Syst. 75, 184–191 (2015)   
18. Xing, Z., Pei, J., Dong, G., Yu, P.S.: Mining sequence classifers for early prediction. In: Proceedings of the 2008 SIAM international conference on data mining. SIAM, pp. 644–655 (2008)   
19. Ye, L., Keogh, E.: Time series shapelets: a new primitive for data mining. In: Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 947– 956 (2009)   
20. Batista, G.E.A.P.A., Wang, X., Keogh, E.J.: A complexityinvariant distance measure for time series. In: Proceedings of the 2011 SIAM international conference on data mining, SIAM, pp. 699–710 (2011)   
21. Ding, Hui, Trajcevski, Goce, Scheuermann, Peter, Wang, Xiaoyue, Keogh, Eamonn: Querying and mining of time series data: experimental comparison of representations and distance measures. Proc. VLDB Endowment 1(2), 1542–1552 (2008)   
22. Rakthanmanon, T., Campana, B., Mueen, A., Batista, G., Westover, B., Zhu, Q., Zakaria, J., Keogh, E.: Searching and mining trillions of time series subsequences under dynamic time warping. In: Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 262–270 (2012)   
23. Xing, Zhengzheng, Pei, Jian, Keogh, Eamonn: A brief survey on sequence classifcation. ACM Sigkdd Explorations Newsletter 12(1), 40–48 (2010)   
24. Hüsken, M., Stagge, P.: Recurrent neural networks for time series classifcation. Neurocomputing 50, 223–235 (2003)   
25. Song, Y., Xin, R., Chen, P., Zhang, R., Chen, J., Zhao, Z.: Identifying performance anomalies in fuctuating cloud environments: a robust correlative-GNN-based explainable approach. Future Generat. Comput. Syst. (2023). https://doi.org/10.1016/j.future. 2023.03.020   
26. Yi, D., Lei, Z., Li, S.Z.: Age estimation by multi-scale convolutional network. In: Asian conference on computer vision. Springer, pp. 144–158 (2014)   
27. Liu, Z., Luo, H., Chen, P., Xia, Q., Gan, Z., Shan, W.: An efcient isomorphic CNN-based prediction and decision framework for fnancial time series. Intell. Data Anal. 26(4), 893–909 (2022)   
28. Kim, K.-J., Han, I.: Genetic algorithms approach to feature discretization in artifcial neural networks for the prediction of stock price index. Expert Syst. Appl. 19(2), 125–132 (2000)   
29. Teixeira, L.A., De Oliveira, A.L.I.: A method for automatic stock trading combining technical analysis and nearest neighbor classifcation. Expert Syst. Appl. 37(10), 6885–6890 (2010)   
30. Song, Y., Lee, J.W., Lee, J.: A study on novel fltering and relationship between input-features and target-vectors in a deep learning model for stock price prediction. Appl. Intell. 49(3), 897–911 (2019)   
31. Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classifcation with deep convolutional neural networks. Commun. ACM 60(6), 84–90 (2017)   
32. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., WardeFarley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems 27 (2014)   
33. Xu, Z., Du, J., Wang, J., Jiang, C., Ren, Y.: Satellite image prediction relying on gan and lstm neural networks. In: ICC 2019-2019 IEEE International Conference on Communications (ICC), IEEE, pp. 1–6 (2019)   
34. Zhang, K., Zhong, G., Dong, J., Wang, S., Wang, Y.: Stock market prediction based on generative adversarial network. Procedia Comput. Sci. 147, 400–406 (2019)   
35. Feng, F., Chen, H., He, X., Ding, J., Sun, M., Chua, T-S.: Enhancing stock movement prediction with adversarial training. arXiv preprint arXiv:1810.09936, (2018)   
36. Kara, Y., Boyacioglu, M.A., Baykan, Ö.K.: Predicting direction of stock price index movement using artifcial neural networks and support vector machines: The sample of the istanbul stock exchange. Expert Syst. Appl. 38(5), 5311–5319 (2011)   
37. Armano, G., Marchesi, M., Murru, A.: A hybrid genetic-neural architecture for stock indexes forecasting. Info. Sci. 170(1), 3–33 (2005)   
38. Masoud, N.: Predicting direction of stock prices index movement using artifcial neural networks: The case of Libyan fnancial market. J Econ Manage Trade 4(4), 597–619 (2014)   
39. Takahashi, T., Tamada, R., Nagasaka, K.: Multiple line-segments regression for stock prices and long-range forecasting system by neural network. In: Proceedings of the 37th SICE Annual Conference. International Session Papers. IEEE, pp. 1127–1132 (1998)   
40. Patel, Jigar, Shah, Sahil, Thakkar, Priyank, Kotecha, Ketan: Predicting stock and stock price index movement using trend deterministic data preparation and machine learning techniques. Expert Syst. Appl. 42(1), 259–268 (2015)   
41. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, (2014)   
42. Fawcett, Tom: An introduction to roc analysis. Pattern Recogn. Lett. 27(8), 861–874 (2006)   
43. Zheng, Y., Liu, Q., Chen, E., Ge, Y., Zhao, J.L.: Time series classifcation using multi-channels deep convolutional neural networks. In: International conference on web-age information management, Springer, pp. 298–310 (2014)

Publisher's Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional afliations.