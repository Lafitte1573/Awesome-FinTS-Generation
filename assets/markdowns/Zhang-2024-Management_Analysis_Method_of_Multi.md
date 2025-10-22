# Management Analysis Method of Multivariate Time Series Anomaly Detection in Financial Risk Assessment

Yongshan Zhang, Weifang University of Science and Technology, China Zhiyun Jiang, Weifang University of Science and Technology, China Cong Peng, Guizhou University of Commerce, China\* Xiumei Zhu, Weifang University of Science and Technology, China Gang Wang, Imperial College London, UK

# ABSTRACT

The significance of financial risk lies in its impact on economic stability and individual/institutional financial security. Effective risk management is crucial for market confidence and crisis prevention. Current methods for multivariate time series anomaly detection have limitations in adaptability and generalization. To address this, we propose an innovative approach integrating contrastive learning and Generative Adversarial Networks (GANs). We use geometric distribution masking for data augmentation to enhance dataset diversity. Within the GAN framework, we train a Transformer-based autoencoder to capture normal point distributions. We include contrastive loss in the discriminator to ensure robust generalization. Rigorous experiments on four real-world datasets show that our method effectively mitigates overfitting and outperforms state-of-the-art approaches. This enhances anomaly identification in risk management, paving the way for deep learning in finance, and offering insights for future research and practical use.

# Ke ywords

Financial Risk Management, Multivariate Time Series, Anomaly Detection, Generative adversarial networks, Contrastive learning, Transformer

# INTRODUCTION

Financial risk, as a core issue in the field of finance, has always garnered widespread attention. The instability and risks in financial markets have significant impacts on the stability and sustainable development of the global economic system (Cao et al., 2022). Therefore, studying the nature of financial risks, and methods to address these risks, becomes crucial. The uncertainty and volatility in financial markets makes it difficult for investors and decision-makers to predict and manage potential risks, further increasing the complexity of financial markets (Ahmed et al., 2022). Instability and risks often manifest in the form of sharp price fluctuations, increasing credit risks, and outbreaks of liquidity issues in the market. Market instability can be triggered by various factors, such as macroeconomic fluctuations, political events, natural disasters, and more. Credit risk involves the inability of corporations and governments to repay debt on time, potentially causing losses for financial institutions and investors. Additionally, liquidity issues can lead to market trading inefficiencies, further adding to market uncertainty (Taghian et al., 2022). To better address the instability and risks in financial markets, traditional financial models and methods have proven to be less flexible and accurate. The rise of deep learning technology offers a new approach to solving this problem (Swathi et al., 2022). Deep learning can handle high-dimensional, nonlinear, and dynamic financial data, effectively uncovering patterns and correlations hidden within the data. This capability provides financial professionals with more accurate predictive tools, promising to improve the efficiency and effectiveness of financial risk management.

Multivariate Time Series (MTS) data is an important data type in the financial field, comprising collections of multiple time series, such as the prices, interest rates, and trading volumes of different financial assets (Chauhan & Lee, 2022). The uniqueness of MTS data lies in its ability to capture the spatiotemporal correlations among different financial variables, reflecting the complexity and diversity of financial markets. It is this multidimensional information that makes MTS data a crucial source for financial risk management and forecasting (Fu et al., 2022). The analysis of such data can assist in making informed decisions and predictions within information management systems. Specifically, anomaly detection in MTS involves identifying and revealing uncommon patterns or events, which is crucial for areas such as fraud prevention and predictive maintenance in finance. For instance, detecting anomalies in financial time series data can help identify fraudulent activities and mitigate financial risks, which is particularly important for financial institutions and investors. Furthermore, timely identification and handling of anomalies in the financial market can reduce potential losses and enhance the efficiency of capital (Jo & Kim, 2023).

Past research has predominantly focused on statistical methods such as Autoregressive Integrated Moving Average (ARIMA) and Principal Component Analysis (PCA) (Siwach & Mann, 2022). While these traditional approaches have laid the foundation for the analysis of multivariate time series data, to some extent, they also exhibit notable limitations. ARIMA models often rely on the assumption of linearity, making it challenging to capture complex nonlinear correlations and anomalous patterns; thereby, limiting their applicability in the financial domain. On the other hand, although PCA can reduce the dimensionality of data, it assumes orthogonality among principal components, potentially overlooking underlying correlations in multivariate time series data and resulting in information loss. Alternatively, some studies have explored the use of clustering analysis to handle multivariate time series data, aiming to discover similarities and differences among different assets or time periods (J. Yao et al., 2022). The advantage of clustering analysis lies in its ability to identify potential market patterns and behaviors, offering better decision support for investors. However, traditional clustering methods often require pre-determining the number of clusters and sample distance measures, introducing subjectivity and limitations to the results.

Recently, significant progress has been made in the field of deep learning techniques aimed at detecting anomalies in multivariate time series (MTS) data. These techniques include models such as Long Short-Term Memory (LSTM) (Vos et al., 2022), Gated Recurrent Unit (GRU) (Tang et al., 2023), and Transformer (L. Chen et al., 2022). These models excel in capturing long-term dependencies and temporal features in multivariate time series data by establishing complex nonlinear relationships and dynamic memory mechanisms. However, these models also face some challenges. Firstly, they often require a large amount of labeled anomalous data for supervised training, which is scarce and hard to obtain in the financial domain. Secondly, these models may be prone to overfitting, especially when dealing with high-dimensional data, requiring careful hyperparameter tuning and data preprocessing (Lin et al., 2022). To overcome these challenges, reconstruction-based deep learning methods have gained widespread attention, especially in scenarios where anomaly labels are unavailable.

Reconstruction-based methods aim to obtain the underlying representation of normal MTS data, then measure the deviation between actual MTS data and its reconstructed representation, known as reconstruction error. A higher reconstruction error indicates a higher likelihood of classifying MTS data as anomalous. The advantage of this approach is that it does not require anomaly labels, automatically learning representations of normal data, making it more suitable for scenarios like finance that lack abundant anomaly labels. However, existing reconstruction-based methods also have some limitations. Some methods may overly rely on assumptions about the distribution of data, leading to insufficient robustness against anomalous data. Additionally, excessive complexity and deep models may lead to overfitting issues, particularly in cases with limited samples. Therefore, despite some progress, there are still many challenges and opportunities in the field of multivariate time series anomaly detection that require further research and improvement (Lee & Kang, 2022).

To mitigate these challenges, we propose an innovative approach grounded in the Autoencoder (AE) framework within the context of Generative Adversarial Networks (GANs) for anomaly detection in multivariate time series (MTS). Addressing concerns related to overfitting and aiming to boost model performance, we employ two key strategies. Firstly, we implement data augmentation techniques to augment the dataset’s diversity. Secondly, we employ an autoencoder built on the Transformer architecture to capture temporal dependencies within time series data. While autoencoders are susceptible to overfitting, we strategically integrate them into the Generative Adversarial Network framework, introducing additional constraints to bolster the model’s robustness. Specifically, we incorporate contrastive constraints to enhance the discriminator’s capacity to discern the latent distribution of normal patterns.

The main contributions of this study can be succinctly summarized as follows:

This study introduces a novel approach for anomaly detection in multivariate time series (MTS) using an autoencoder (AE) within the framework of Generative Adversarial Network (GAN). This method combines the advantages of deep learning and generative adversarial networks to more accurately identify anomalies in MTS, providing an innovative tool for financial risk management.   
To address the common issue of overfitting in deep learning models, two strategies are proposed in this study, including data augmentation techniques and contrastive constraints. These strategies effectively enhance the robustness of the model, reduce false positive rates, and make the results of anomaly detection more reliable.   
• Extensive experiments validate the performance of the proposed method and demonstrate its potential application in the field of financial risk management. These experiments highlight the effectiveness of the new approach in MTS anomaly detection, providing financial practitioners with a powerful tool to better understand and manage risks in financial markets.

The subsequent sections of this manuscript are organized as follows: Section 2 provides an in-depth discussion of related work. Section 3 outlines the methodology proposed in this paper. Experimental results and corresponding analyses are presented in Section 4. Finally, Section 5 summarizes the conclusions of this article.

# RELATED WORK

In this section, we will explore the latest research methods involving Transformer-based anomaly detection, GAN-based anomaly detection, and contrastive learning methods in multivariate time series (MTS) data. The aim is to provide a comprehensive understanding of the practical applications of multivariate time series anomaly detection in the financial domain. This exploration is intended to furnish financial professionals with more accurate and reliable risk assessment tools, enabling them to better navigate the complex and dynamic environment of financial markets.

# Transformer-Based Time Series Anomaly Detection

The Transformer model, originally applied in natural language processing, has now found widespread application in multivariate time series anomaly detection (Raza et al., 2023). Its core concept involves introducing self-attention mechanisms, enabling the model to effectively capture dependencies at different positions within a sequence. This feature is particularly crucial for financial time series data, where market fluctuations are often influenced by various factors with complex temporal and spatial correlations. By allowing each time point to focus on other points in the sequence, the Transformer model excels in capturing long-term dependencies and temporal features, providing a powerful tool for identifying financial risks (Ma et al., 2022).

Recently, the TranAD model proposed by Tuli, Casale, and Jennings has been introduced as a deep learning model specifically designed for multivariate time series anomaly detection (Pang et al., 2022). Drawing inspiration from the Transformer model, TranAD utilizes self-attention mechanisms to effectively capture abnormal patterns within time series data. In the financial domain, TranAD can be applied to time series data of various financial indicators, such as stock prices and trading volumes, contributing to the timely detection of potential risks and anomalies. Its superior sequence modeling capability positions TranAD as a standout performer in financial risk identification (Guipeng Ding, 2023).

Additionally, the Anomaly Transformer algorithm is a Transformer-based anomaly detection method. This algorithm employs self-attention mechanisms, comprehensively considering the correlations between different points in time series data, making it well-suited for dynamic and complex anomaly patterns in financial markets (Xu et al., 2023). In the financial sector, Anomaly Transformer can be utilized to monitor transaction anomalies, market fluctuations, providing more accurate early warnings for risk identification.

Notwithstanding their efficacy, some challenges still hinder the practical application of these methods. Notably, there is a risk of overfitting to anomalous patterns during reconstruction, as well as non-robustness while training on contaminated data.

# GAN-Based Time Series Anomaly Detection

Generative Adversarial Network (GAN) is a deep learning model initially proposed by Goodfellow et al. in 2014. Its fundamental idea involves adversarial training of two networks: a generator and a discriminator. The generator aims to produce realistic data, while the discriminator works to distinguish between generated and real data (Xia et al., 2022). This adversarial training process continually enhances the realism of the generated data by improving the generator’s ability, while the discriminator becomes increasingly adept at differentiation. In the context of time series anomaly detection, the generative capability of GANs positions them as a promising tool for learning and generating the distribution of normal time series data to detect anomalous patterns (Zhang et al., 2023).

In recent times, a series of GAN-based time series anomaly detection models has emerged, including GANomaly and TadGAN (Mitra et al., 2022). These models, through cooperative training of the generator and discriminator, successfully learn the distribution within time series data and effectively capture anomalous patterns. The GANomaly model reconstructs normal time series to detect anomalies through reconstruction error, while TadGAN incorporates LSTM structures for handling time series data, giving it an advantage in capturing temporal features (Cabreza et al., 2022).

However, GAN-based time series anomaly detection methods also have some limitations. Firstly, they typically require a substantial amount of labeled anomalous data for supervised training, and in practical scenarios such as the financial domain, anomalous data is often scarce (Zhu et al., 2022). Secondly, these models are susceptible to overfitting, especially when dealing with high-dimensional data, requiring careful hyperparameter tuning and data preprocessing.

# Contrastive Learning in the Application of Financial Risk Identification

Contrastive learning is a method that learns useful representations by comparing the similarity and dissimilarity between data points (B. Chen et al., 2022). In the context of financial risk identification, this approach leverages the differences between normal and abnormal patterns, providing the model with richer information and enhancing its awareness of potential risks (Luo et al., 2022).

In recent years, the application of contrastive learning in the financial domain has gained increasing attention. These methods not only contribute to improving the distinguishability between normal and abnormal patterns, but also reveal potential market trends and behaviors (Zheng et al., 2022). Contrastive learning techniques have found wide-ranging applications in financial risk management, including transaction anomaly detection, credit assessment, and fraud detection, offering decision-makers a more comprehensive view of the financial landscape (Guipeng Ding, 2023). However, contrastive learning methods come with their challenges. These approaches often require predefined clustering quantities and sample distance metrics, potentially introducing subjectivity and limitations. Particularly when dealing with large-scale, high-dimensional financial time series data, further research and refinement are needed to optimize the performance of contrastive learning methods (Kim et al., 2022).

The application of contrastive learning in financial risk identification provides a robust tool for enhancing model sensitivity and discovering potential market patterns and behaviors (Wang et al., 2022). Nevertheless, to better adapt to the practical challenges in the financial domain, such as data uncertainty and dynamics, ongoing research is essential for refining and improving contrastive learning methods. The advancement of this field holds the promise of delivering more precise and reliable tools for financial practitioners to better understand and manage risks in the financial market.

# METHOD

# Overview of Proposed Model

The model network architecture we propose is depicted in Figure 1, comprising four modules: data augmentation, generator, discriminator, and contrastive learning. Each module plays a critical role in the system, collaborating to enhance the performance of multivariate time series (MTS) anomaly detection.

The data augmentation module leverages the characteristics of MTS, employing a novel random masking approach to expand the unexplored input space following a geometric distribution. The objective of the data augmentation module is to improve the model’s adaptability to diverse data, enhancing its generalization.

The generator module learns the underlying distribution of normal patterns in MTS and accurately reconstructs it using a Transformer-based autoencoder. This module aids the model in learning and capturing the key features of normal patterns; thereby, improving anomaly detection performance.

The discriminator module imposes constraints on reconstruction within the GAN framework to better capture normal patterns in MTS. Introducing additional constraints enhances the model’s sensitivity to normal patterns, reducing false positive rates.

The contrastive learning module imposes contrastive constraints on the representation of MTS, enhancing the model’s generalization ability and promoting joint training of the discriminator. Through contrastive learning, the aim is to further improve the model’s performance, adapting to different MTS data scenarios.

These four modules collaborate to comprehensively capture normal and abnormal patterns in MTS, enhancing the accuracy and robustness of anomaly detection. The following sections will detail the functionality and interactions of each module.

![](images/387660e2ac8b294f1da40de0fc04fd996ba6f82dfced5392514b8332ef8e145d.jpg)  
Figure 1. Overview of the proposed model

# Data Augmentation

In terms of data augmentation, we have implemented a simple yet impactful technique known as random masking, following a geometric distribution (F. Liu et al., 2022). The essence of random masking involves introducing variability into multivariate time series (MTS) data; thereby, expanding the input space not fully explored by the system. This method incorporates masks at specific time points in the sequence, mimicking local data loss or uncertainty. The choice of the geometric distribution is deliberate, leveraging its natural discreteness and adaptability, which allows for variability while preserving the overall sequence structure. The objective of employing random masking is to bolster the model’s adaptability to diverse data, consequently enhancing its generalization performance. Through the introduction of this randomness, our aim is to fortify the model, enabling it to adeptly handle varied features and potential anomalous patterns with increased efficacy (Ning et al., 2023).

As shown in Figure 2, the binary noise mask matrix construction process, denoted as $M$ , utilizes a Markov chain consisting of two states: ‘‘Mask’’ and ‘‘Unmasked.’’ The state transition matrix in the Markov chain can be expressed as follows:

$$
P = \left[ { \begin{array} { l l l l l l } { p _ { 0 0 } } & { } & { p _ { 0 1 } } \\ { p _ { 1 0 } } & { } & { p _ { 1 1 } } \end{array} } \right] = \left[ { \begin{array} { l l l l l } { p _ { m : m } } & { } & { p _ { m : u } } \\ { p _ { u : m } } & { } & { p _ { u : u } } \end{array} } \right]
$$

The complete transition probability matrix can be formulated as follows:

$$
\begin{array} { c } { { P ( s _ { t + 1 } = 0 \mid s _ { t } = 0 ) = p _ { m : m } = 1 - \displaystyle \frac { 1 } { l _ { m } } } } \\ { { P ( s _ { t + 1 } = 1 \mid s _ { t } = 0 ) = p _ { m : d } = \displaystyle \frac { 1 } { l _ { m } } } } \\ { { P ( s _ { t + 1 } = 0 \mid s _ { t } = 1 ) = p _ { u : m } = \displaystyle \frac { 1 } { l _ { m } } * \displaystyle \frac { r } { 1 - r } } } \\ { { P ( s _ { t + 1 } = 1 \mid s _ { t } = 1 ) = p _ { u : u } = 1 - \displaystyle \frac { 1 } { l _ { m } } * \displaystyle \frac { r } { 1 - r } } } \end{array}
$$

![](images/c4d1697ef4f056bbd332fa2fad2f0ffa31a622732433238a6c1072efd81e2916.jpg)  
Figure 2. Data augmentation model

# Generator

Our model is embedded within the framework of a Generative Adversarial Network (GAN), where the task of the generator (G) is to reconstruct the original multivariate time series (MTS), including those that have undergone data augmentation (Aftabi et al., 2023). The discriminator (D), on the other hand, aims to distinguish between the original MTS and the reconstructed MTS generated by the generator.

Within this framework, the primary role of the generator is to meticulously reconstruct the underlying distribution of normal patterns in the multivariate time series. Employing a Transformerbased autoencoder, the generator adeptly captures the temporal features of the original data, producing reconstructed sequences endowed with comparable statistical characteristics.

To effectively capture long-term dependencies and intricate patterns in the normal multivariate time series (MTS), the encoder ???????? is built upon the Transformer architecture, illustrated in Fig. 3. This encoder can be formulated as follows:

$$
\begin{array} { c } { { Q _ { _ i } , K _ { _ i } , V _ { _ i } = \hat { \mathcal { X } } W _ { _ i } ^ { Q } , \hat { \mathcal { X } } W _ { _ i } ^ { K } , \hat { \mathcal { X } } W _ { _ i } ^ { V } } } \\ { { { } } } \\ { { \mathcal { Z } _ { _ i } = s o f t m a x { \displaystyle \left[ \frac { Q _ { _ i } K _ { _ i } ^ { T } } { \sqrt { d _ { _ k } } } \right] } V _ { _ i } } } \\ { { { } } } \\ { { Z = C o n c a t { \displaystyle \left( \mathcal { Z } _ { _ 1 } , . . . , \mathcal { Z } _ { _ h } \right) } W ^ { o } } } \end{array}
$$

where $W _ { i } ^ { \mathcal { Q } } , W _ { i } ^ { K } , W _ { i } ^ { V }$ represent the parameter matrices for query, key, value of Multi-Head attention, $h$ is the number of the head. Our decode is a two-layer MLP which is formulated as follows:

$$
\begin{array} { r } { \mathcal { Z } _ { i } = s o f t m a x \Bigg ( \frac { Q _ { i } K _ { i } ^ { T } } { \sqrt { d _ { k } } } \Bigg ) V _ { i } } \\ { Z = C o n c a t \Big ( \mathcal { Z } _ { 1 } , . . . , \mathcal { Z } _ { h } \Big ) W ^ { o } } \end{array}
$$

![](images/2d4572fc95ec70e76d882bc03faaae33695c787e2f29993cb7945fda025b5c90.jpg)  
Figure 3. Structure of the transformer encoder

where $W _ { g 1 }$ and $W _ { g 2 }$ are the weight matrices for the first and second layer respectively, $b _ { g 1 }$ and $b _ { g 2 }$ are the bias vectors, and $\sigma$ is the Sigmoid activation function.

# Discriminator

The discriminator (D) imposes constraints on the generator (G) to regulate the reconstruction error. Specifically, we guide the training of the discriminator by representing original multivariate time series (MTS) data as real samples and the generator-reconstructed MTS data as fake samples (Liu et al., 2023).

The discriminator’s role is to effectively distinguish between real MTS data and the fake MTS data generated by the generator. By contrasting the differences between the two, the discriminator can push the generator to enhance its generated sequences, making them closer to the statistical characteristics of the original data. This adversarial training mechanism contributes to improving the performance of the generator and ultimately narrowing the distribution gap between real and fake samples.

The discriminator module plays a crucial role throughout the entire training process by reinforcing the accuracy requirements for the generator’s output, thereby enhancing the model’s performance in multivariate time series anomaly detection.

In our implementation, $f _ { d i s }$ is constructed as a three-layer MLP. The first two layers are dedicated to feature extraction, while the final layer serves the purpose of classification, which can be formulated as follows:

$$
\begin{array} { c } { \mathcal { H } _ { _ { d 1 } } = \sigma \left( W _ { _ { d 1 } } \cdot \mathcal { X } _ { _ { d i s } } + \mathbf { b } _ { _ { d 1 } } \right) } \\ { \mathcal { H } _ { _ { d 2 } } = \sigma \left( W _ { _ { d 2 } } \cdot \mathcal { H } _ { _ { d 1 } } + \mathbf { b } _ { _ { d 2 } } \right) } \\ { P ( Y \mid \mathcal { X } _ { _ { d i s } } ) = s o f t m a x \left( W _ { _ { d 3 } } \cdot \mathcal { H } _ { _ { d 2 } } + \mathbf { b } _ { _ { d 3 } } \right) } \end{array}
$$

where $W _ { d 1 } , \ W _ { d 2 }$ and $W _ { d 3 }$ are the weight matrices for the first, second and last layer respectively, $\mathbf { b } _ { d 1 } , \mathbf { b } _ { d 2 }$ and $\mathbf { b } _ { d 3 }$ are the bias vectors, $\mathcal { H } _ { d 1 }$ and $\mathcal { H } _ { d 2 }$ are hidden latent representations for MTS data, $Y$ is the label for real or fake and $\sigma$ is the Sigmoid activation function.

Within the GAN framework, the Discriminator loss $L$ can be formulated as follows:

$$
L _ { d i s } = - \frac { 1 } { N } \Bigl [ \log D \left( \mathcal { X } _ { d i s } \right) + \log \left( 1 - D \left( \mathrm { G } \left( \mathcal { X } _ { d i s } \right) \right) \right) \Bigr ] 
$$

# Contrastive Learning

Contrastive learning is a method that involves learning useful representations by comparing the similarities and differences between data points. In the context of our model, contrastive learning plays a crucial role in enhancing the generalization ability for multivariate time series (MTS) anomaly detection (Li & Jung, 2023).

In each batch, we employ a strategy where only two reconstructed MTS data representations are considered positive pairs, while all other instances are treated as negative pairs. The main idea is to encourage the model to bring the representations of positive pairs closer (“attraction”), while simultaneously pushing apart the representations of negative pairs (“separation”), as illustrated in Figure 1.

The contrastive loss ( contrastive ) is defined as follows:

$$
\mathcal { L } _ { \mathrm { c o n t r a s t i v e } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \frac { e ^ { s \cdot \mathrm { S i m } \left( z _ { i } , z _ { i + 1 } \right) } } { { \sum _ { j = 1 } ^ { 2 N } e ^ { s \cdot \mathrm { S i m } \left( z _ { i } , z _ { j } \right) } } } \right]
$$

In this equation, $N$ represents the number of positive pairs in the batch, $z _ { i }$ and $z _ { i + 1 }$ are the latent representations of two positive pairs, $S i m \left( \cdot , \cdot \right)$ denotes the similarity function, and $s$ is the temperature parameter. The contrastive loss encourages the model to maximize the similarity of positive pairs, promoting the learning of more generalized latent representations.

# Model Training

Our proposed model training involves the collaborative optimization of the data augmentation, generator, discriminator, and contrastive learning modules within the framework of Generative Adversarial Networks (GAN). The objective is to enhance the model’s capability for accurately detecting anomalies in multivariate time series (MTS) data.

# EXPERIMENTS

In this section, extensive experiments were conducted on four real-world datasets to assess the effectiveness of our proposed method. Additionally, various ablation studies were performed to investigate the impact of different components in the model.

# Experimental Setup

# Public Datasets

In this experiment, we utilized four publicly available datasets aimed at identifying anomalous behavior in multivariate time series and further exploring their potential applications in financial risk assessment.

The Yahoo Finance dataset (YAHOO) is sourced from the Yahoo Finance platform, providing researchers with extensive financial market data globally (Y. Yao et al., 2022). This dataset includes key indicators such as stock prices, trading volumes, etc., allowing users to select companies or indices of interest to obtain detailed historical time series data. The diversity of this dataset makes it an ideal choice for analyzing trends and trading activities in the stock market.   
• The Quandl WIKI dataset (WIKI) is a valuable resource containing a large number of stocks in the U.S. stock market (Oswal et al., 2023). It offers daily and weekly data for these stocks, including key information such as stock prices and trading volumes. Researchers can leverage this dataset to delve into the performance of different industries and companies, gaining a better understanding of market dynamics and changes.   
• The Federal Reserve Economic Data (FRED) comes from the Federal Reserve’s FRED dataset, covering economic time series data from the United States and the globe (Dannels, 2023). It includes key macroeconomic indicators such as GDP and inflation rates. This dataset is not only suitable for macroeconomic research but also provides researchers with an opportunity to gain insights into economic trends and policy impacts.   
• The S&P 500 dataset (S&P) tracks the stock index of 500 large companies in the United States, serving as a crucial indicator of U.S. stock market performance (Gan, 2023). This dataset includes information spanning multiple economic cycles, offering important clues for analyzing stock market trends and overall economic conditions. Researchers can use this dataset to assess market risk and formulate corresponding investment strategies.

The detailed statistics of the four datasets are presented in Table 1.

# Experimental Environment

To conduct experiments for this study, we utilized a computer equipped with an Intel Core i9-9900K CPU (clocked at 3.60GHz), NVIDIA RTX 3090 GPU, and 32GB of RAM. In terms of software, we employed PyTorch 1.8.0 as the deep learning framework, Python 3.8 as the programming language, and configured CUDA 11.3. This powerful experimental environment provided ample computational resources and high-performance hardware support, enabling us to fully leverage the advantages of GPU acceleration to expedite model training. This facilitated a more effective evaluation of the performance of our proposed method for multivariate time series anomaly detection. Additionally, the choice of PyTorch framework version and CUDA configuration ensured that we could utilize the latest tools and libraries in deep learning during the experiments, ensuring the reliability and reproducibility of the experimental results.

Table 1. Dataset statistics   

<table><tr><td>Dataset</td><td> Train</td><td>Test</td><td>Dimensions</td><td> Anomalies (%)</td><td>Dataset</td></tr><tr><td>YAH00</td><td>56,797</td><td>72,209</td><td>45</td><td>11.78</td><td>YAHO0</td></tr><tr><td>WIKI</td><td>133,663</td><td>426.097</td><td>25</td><td>14.18</td><td>WIKI</td></tr><tr><td>FRED</td><td>104,464</td><td>86,321</td><td>35</td><td>23.85</td><td>FRED</td></tr><tr><td>S&amp;P</td><td>495,280</td><td>448,399</td><td>41</td><td>21.98</td><td>S&amp;P</td></tr></table>

# Baselines

We considered nine baseline anomaly detection methods, which are as follows:

Principal Component Analysis (PCA): PCA is a commonly used dimensionality reduction technique that identifies anomalies in reduced-dimensional space by transforming the original data into uncorrelated principal components (Soleimani-Babakamali et al., 2023).   
. MERLIN (Meta-Learning Robust and Interpretable Neural Networks): MERLIN is based on meta-learning principles and aims to construct robust and interpretable neural networks. By leveraging shared knowledge obtained from different tasks, MERLIN adapts to time series data from various domains, enhancing its anomaly detection performance (Isgut et al., 2022).   
• MAD-GAN (Multiple Anomaly Detection Generative Adversarial Network): MAD-GAN is a variant of Generative Adversarial Networks (GAN) focused on multiple anomaly detection. By generating the distribution of normal data and identifying data points significantly deviating from it, MAD-GAN effectively identifies anomalous behavior in time series data (Fu et al., 2023).   
. GDN (Generative Dropout Networks): GDN is a generative model that incorporates a generative dropout mechanism to enhance robustness against anomalies. By introducing dropout within the network, GDN can better capture irregular patterns in time series data (Guan et al., 2022).   
• Unsupervised Anomaly Detection (USAD): USAD is an unsupervised anomaly detection method that relies on modeling normal data. By learning the distribution of normal data, USAD can identify abnormal behavior in time series that does not conform to expected patterns (Wang et al., 2023).   
. TranAD (Transformation-based Anomaly Detection): TranAD is a transformation-based anomaly detection method that maps anomalous data to a more easily detectable space by applying transformations to time series data. This approach improves sensitivity to potential anomalies in time series (Rewicki et al., 2023).   
• BeatGAN: BeatGAN is a novel anomaly detection method that utilizes beat-based generative models. By capturing temporal dependencies in time series data, BeatGAN aims to effectively identify anomalous patterns (S. Liu et al., 2022).   
• FGANomaly: FGANomaly is an anomaly detection method based on the principles of Feature Generation Networks (FGN). This method focuses on generating features that highlight anomalous patterns in time series data to improve detection performance (Qiao et al., 2022).   
. Anomaly Transformer (AT): AT is an anomaly detection method that utilizes a transformer architecture. By capturing long-range dependencies and patterns in time series data, AT aims to enhance anomaly detection performance in various applications (Lee & Kang, 2022).

These baseline methods cover a variety of anomaly detection strategies and aim to advance our understanding and capabilities in identifying anomalies in multivariate time series data.

# Evaluation Metrics

In our research, we employed multiple evaluation metrics to quantify the performance of the considered model anomaly detection methods. These metrics provide a comprehensive assessment of the effectiveness of the models in identifying anomalous behavior in multivariate time series data. Below, we introduce the evaluation metrics used in this experiment:

Precision measures the accuracy of the model’s anomaly judgments, i.e., the proportion of actual anomalies among all samples that the model classified as anomalies.

$$
P = { \frac { T P } { T P + F P } }
$$

where $T P$ stands for True Positives, representing the number of samples correctly identified as anomalies by the model. $F P$ stands for False Positives, indicating the number of normal samples mistakenly identified asanomalies by the model.

Recall assesses the model’s ability to capture real anomalies, i.e., the proportion of actual anomalies among all samples that were successfully identified as anomalies by the model.

$$
R = { \frac { T P } { T P + F N } }
$$

where: $T P$ stands for True Positives. FN stands for False Negatives, representing the number of actual anomaly samples that the model failed to correctly identify.

The F1 Score provides a comprehensive assessment of the model’s overall performance by combining precision and recall.

$$
F _ { _ 1 } = { \frac { 2 \cdot P \cdot R } { P + R } }
$$

ROC AUC quantifies the model’s performance at different anomaly detection thresholds, reflecting the balance between True Positive Rate (TPR) and False Positive Rate (FPR).

$$
\mathrm { R O C A U C } = \int _ { \mathrm { 0 } } ^ { 1 } \mathrm { T P R } d \mathrm { F P R }
$$

where: TPR is the True Positive Rate, calculated as $\frac { T P } { T P + F N }$ FPR is the False Positive Rate, calculated as $\frac { F P } { F P + T N } . T N$ stands for True Negatives, representing the number of samples correctly identified as normal by the model.

These evaluation metrics help provide a comprehensive understanding of the performance of each anomaly detection method in multivariate time series data and serve as a scientific basis for relative comparisons.

Table 2. Overview performance of all methods (%)   

<table><tr><td> Dataset</td><td colspan="3"> YAHOO</td><td colspan="3">WIKI</td><td colspan="3"> FRED</td><td colspan="3"> S&amp;P</td></tr><tr><td>Metrics</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>PCA</td><td>92.13</td><td>82.48</td><td>87.04</td><td>89.16</td><td>56.07</td><td>68.92</td><td>76.95</td><td>97.71</td><td>86.12</td><td>31.36</td><td>34.01</td><td>34.01</td></tr><tr><td>MERLIN</td><td>50.68</td><td>46.92</td><td>48.73</td><td>24.19</td><td>97.9</td><td>39.34</td><td>53.38</td><td>97.39</td><td>69.09</td><td>23.94</td><td>36.16</td><td>35.16</td></tr><tr><td>MAD-GAN</td><td>83.63</td><td>37.77</td><td>52.66</td><td>79.04</td><td>90.63</td><td>84.01</td><td>87.1</td><td>94.98</td><td>90.87</td><td>94.4</td><td>79.12</td><td>79.12</td></tr><tr><td>GDN</td><td>90.55</td><td>97.39</td><td>93.49</td><td>73.27</td><td>97.38</td><td>83.65</td><td>88.61</td><td>92.16</td><td>90.35</td><td>68.04</td><td>79.48</td><td>79.48</td></tr><tr><td>USAD</td><td>87.87</td><td>86.09</td><td>87.97</td><td>82.4</td><td>86.52</td><td>84.41</td><td>89.07</td><td>93.42</td><td>96.99</td><td>75.25</td><td>84.39</td><td>84.39</td></tr><tr><td>TranAD</td><td>88.85</td><td>98.47</td><td>93.41</td><td>78.9</td><td>98.47</td><td>87.62</td><td>91.09</td><td>98.21</td><td>94.52</td><td>76.24</td><td>79.9</td><td>79.9</td></tr><tr><td>BeatGAN</td><td>87.59</td><td>84.83</td><td>86.18</td><td>77.42</td><td>92.16</td><td>84.03</td><td>88.3</td><td>92.07</td><td>90.07</td><td>71.41</td><td>80.61</td><td>80.61</td></tr><tr><td>FGANomaly</td><td>88.52</td><td>92.07</td><td>90.26</td><td>74.56</td><td>98.4</td><td>84.87</td><td>93.39</td><td>97.25</td><td>96.98</td><td>77.01</td><td>86.48</td><td>86.48</td></tr><tr><td>AT</td><td>89.75</td><td>84.36</td><td>86.97</td><td>91.06</td><td>96.97</td><td>93.45</td><td>95.29</td><td>96.43</td><td>95.86</td><td>93.71</td><td>91.79</td><td>91.79</td></tr><tr><td>Ours</td><td>91.03</td><td>97.73</td><td>93.9</td><td>86.65</td><td>97.86</td><td>91.92</td><td>95.79</td><td>96.26</td><td>96.03</td><td>90.63</td><td>94.99</td><td>92.76</td></tr></table>

# Overview Performance

We will demonstrate the performance of our proposed anomaly detection method for multivariate time series (MTS) data. To comprehensively evaluate our approach, we selected four representative datasets, namely YAHOO, WIKI, FRED, and S&P. We employed multiple evaluation metrics, including Precision (P), Recall (R), and F1-score (F1), to holistically assess the comprehensive performance of each method.

Table 2 presents the comparative results of our method and other classical methods on each dataset. On the YAHOO dataset, our method achieved a Precision of $9 1 . 0 3 \%$ , Recall of $9 7 . 7 3 \%$ , and an F1-score of $9 3 . 9 0 \%$ , outperforming other methods in overall performance. Notably, compared to traditional methods like PCA, our approach significantly improved the recognition performance of abnormal patterns, highlighting the superiority of deep learning methods in handling high-dimensional and nonlinear data. On the WIKI dataset, our method also performed exceptionally well, reaching a Precision of $8 6 . 6 5 \%$ , Recall of $9 7 . 8 6 \%$ , and an F1-score of $9 1 . 9 2 \%$ . In comparison to other methods, our approach demonstrated a clear advantage in overall performance, especially relative to methods such as MAD-GAN and GDN, showcasing the efficient modeling capability of our method for time series data. The experimental results on the FRED dataset revealed that our method surpassed other methods in Precision $( 9 5 . 7 9 \% )$ , Recall $( 9 6 . 2 6 \% )$ , and F1-score $( 9 6 . 0 3 \% )$ , particularly when compared to traditional methods like PCA and MERLIN. This further validates the potential advantages of our method in the field of financial risk management. Finally, on the S&P dataset, our method exhibited a Precision of $9 0 . 6 3 \%$ , Recall of $9 4 . 9 9 \%$ , and an F1-score of $9 2 . 7 6 \%$ , obtaining significant advantages over other methods. Compared to approaches like BeatGAN and AT, our method demonstrated superior overall performance, affirming the generality and robustness of our method across multiple datasets.

Our method excelled in the task of multivariate time series anomaly detection, exhibiting higher accuracy and comprehensive coverage compared to traditional and existing deep learning methods. This innovation provides a valuable tool for financial risk management.

Considering efficiency, we directly compared the training times for each epoch in Table 3, contrasting the baseline with our model. While our model may not be optimal in terms of performance, a comparative analysis with other methods reveals that our approach still achieves relatively satisfactory results across multiple datasets. Compared to the baseline PCA method, our model exhibits performance improvements of $1 1 . 9 \%$ and $2 6 . 1 \%$ on the MSL and SWaT datasets, respectively. Although there are performance gaps on certain datasets compared to methods like

Table 3. Comparison of training times in seconds per epoch   

<table><tr><td>Method</td><td>YAHOO</td><td>WIKI</td><td>FRED</td><td>S&amp;P</td></tr><tr><td>PCA</td><td>9.91</td><td>12.07</td><td>7.81</td><td>17.41</td></tr><tr><td>MERLIN</td><td>7.65</td><td>9.42</td><td>9.42</td><td>12.65</td></tr><tr><td>USAD</td><td>23.75</td><td>26.16</td><td>21.51</td><td>25.25</td></tr><tr><td>MAD-GAN</td><td>28.80</td><td>32.02</td><td>22.83</td><td>30.32</td></tr><tr><td>GDN</td><td>99.24</td><td>64.86</td><td>32.74</td><td>61.93</td></tr><tr><td>TranAD</td><td>7.80</td><td>6.08</td><td>5.71</td><td>3.40</td></tr><tr><td>BeatGAN</td><td>26.16</td><td>31.13</td><td>22.22</td><td>27.60</td></tr><tr><td>FGAnomaly</td><td>68.90</td><td>86.07</td><td>32.80</td><td>54.54</td></tr><tr><td>AT</td><td>9.36</td><td>10.07</td><td>11.37</td><td>8.92</td></tr><tr><td>Ours</td><td>11.06</td><td>15.12</td><td>10.01</td><td>12.85</td></tr></table>

GDN, we still demonstrate competitive effectiveness relative to traditional approaches. Regarding training time, our method maintains relative efficiency. While not achieving optimal performance on all datasets, our model strikes a balance between overall performance and training efficiency, providing a comprehensive and efficient solution for practical applications.

In Figure 4, we provide a comprehensive comparison between model performance and computational efficiency. Regarding our model, the research results are compelling. Notably, in most instances, the effectiveness of our model consistently outperforms faster models. These results further affirm the consistent and exceptional performance of our model.

# Ablation Study

As shown in Table 4, the multivariate time series (MTS) anomaly detection method based on geometric masks achieved the highest F1 scores and AUC values across most datasets. Specifically, this method achieved an F1 score of 93.90 on the YAHOO dataset, 91.92 on the WIKI dataset, 96.03 on FRED, and 92.76 on the S&P dataset. Simultaneously, it demonstrated AUC values of 97.14, 98.35, 98.38, and 97.29 on these four datasets, respectively. This performance outperforms other methods such as window warping, STFT augmentation, and Bernoulli masks, indicating that geometric masks can introduce more beneficial perturbations to MTS data. These perturbations may aid the model in capturing anomalous patterns in the data, enabling the model to generalize better and achieve superior detection performance across diverse datasets.

![](images/f5af0a7d27e69ea2c6382d20347fe0dcf8a22073d4d83f26b780e2127c7a8fc7.jpg)  
Figure 4. Performance comparison of computational efficiency and F1 scores

Table 4. Performance of different data augmentation methods (%)   

<table><tr><td rowspan="2"> Method </td><td colspan="2"> YAHOO</td><td colspan="2">WIKI</td><td colspan="2"> FRED</td><td colspan="2">S&amp;P</td></tr><tr><td>F1</td><td>AUC</td><td>F1</td><td>AUC</td><td>F1</td><td>AUC</td><td>F1</td><td>AUC</td></tr><tr><td> Window warping</td><td>90.45</td><td>93.64</td><td>90.32</td><td>95.63</td><td>91.79</td><td>97.42</td><td>92.28</td><td>94.26</td></tr><tr><td>STFT augmented</td><td>92.52</td><td>94.96</td><td>91.72</td><td>97.47</td><td>92.39</td><td>96.99</td><td>92.44</td><td>95.74</td></tr><tr><td>Bernoulli mask</td><td>93.84</td><td>95.89</td><td>91.15</td><td>97.19</td><td>93.34</td><td>98.02</td><td>91.88</td><td>95.01</td></tr><tr><td>Geometric mask</td><td>93.9</td><td>97.14</td><td>91.92</td><td>98.35</td><td>96.03</td><td>98.38</td><td>92.76</td><td>97.29</td></tr></table>

In the end, we conducted an in-depth exploration of the pivotal models in our proposed approach. The first model is named Transformer-AE, trained solely using a standard transformer-based autoencoder with the mean square error (MSE) loss. The second model, referred to as “No-Contrastive Learning,” is a transformer-based generative adversarial network (GAN) trained without incorporating contrastive learning. The third model, named “w/o Adversarial Training,” is a transformer-based autoencoder specially trained without adversarial training. We conducted experiments on all five datasets, and the results are depicted in Figure 5. Clearly, all three models are crucial, as removing any one of them leads to a decrease in both F1 scores and AUC.

# CONCLUSION

In this article, we introduce a novel approach based on the Autoencoder (AE) and Generative Adversarial Network (GAN) frameworks for anomaly detection in multivariate time series (MTS). Our method leverages the advantages of deep learning and generative adversarial networks to more accurately identify anomalous data in MTS, providing an innovative tool for financial risk management.

Nevertheless, our method does exhibit certain limitations that warrant acknowledgment. Firstly, further research is essential to elevate the model’s performance, particularly when confronted with high-dimensional data and sparsely labeled anomalous data. In scenarios characterized by limited data within the financial domain, it becomes imperative to explore supplementary methods that can fortify the model’s robustness and enhance its generalization capabilities. Secondly, despite the incorporation of contrastive constraints to bolster the discriminator’s ability, the model may still be subject to certain assumptions about data distribution, necessitating additional research for refinement. We remain optimistic about the continued growth of applying deep learning in the field of financial risk management. Future research endeavors can strategically focus on the following areas: firstly, a meticulous refinement of our approach to address existing limitations, with a specific emphasis on enhancing robustness and generalization performance, especially in data-scarce scenarios. Secondly, an exploration of additional data augmentation techniques and contrastive constraint methods to further fine-tune the model’s performance and reliability. Lastly, the integration of deep learning techniques with traditional financial risk management methods to create a synergistic approach that adapts more effectively to the dynamic landscape of markets and financial environments.

![](images/026df68d883a25ff3e460c33753858a7bd417f446f60a48030ecfd82063f2748.jpg)  
Figure 5. Ablation studies

While our research provides an innovative approach for financial risk management, there are still many opportunities and challenges that need further exploration and resolution. We look forward to future research driving further applications of deep learning in the financial domain, providing financial professionals with more powerful tools to better understand and manage risks in financial markets.

# CONFLICTS OF INTEREST

We wish to confirm that there are no known conflicts of interest associated with this publication and there has been no significant financial support for this work that could have influenced its outcome.

# FUNDING STATEMENT

Shandong Province 2023 Humanities and Social science project (Project Approval Number 2023- XWKZ-043) titled ‘Research on high-quality development of financial development serving the real economy in Shandong Province.’ The project duration is from October 2023 to October 2025.

REFERENCES   
Aftabi, S. Z., Ahmadi, A., & Farzi, S. (2023). Fraud detection in financial statements using data mining and GAN models. Expert Systems with Applications, 227, 120144. doi:10.1016/j.eswa.2023.120144   
Ahmed, S., Alshater, M. M., El Ammari, A., & Hammami, H. (2022). Artificial intelligence and machine learning in finance: A bibliometric review. Research in International Business and Finance, 61, 101646. doi:10.1016/j. ribaf.2022.101646   
Cao, Y., Shao, Y., & Zhang, H. (2022). Study on early warning of E-commerce enterprise financial risk based on deep learning algorithm. Electronic Commerce Research, 22(1), 21–36.   
Chauhan, S., & Lee, S. (2022). Machine Learning-Based Anomaly Detection for Multivariate Time Series with Correlation Dependency. IEEE Access : Practical Innovations, Open Solutions, 10, 132062–132070. doi:10.1109/ ACCESS.2022.3230352   
Chen, B., Zhang, J., Zhang, X., Dong, Y., Song, J., Zhang, P., Xu, K., Kharlamov, E., & Tang, J. (2022). Gccad: Graph contrastive learning for anomaly detection. IEEE Transactions on Knowledge and Data Engineering, 1–14. doi:10.1109/TKDE.2022.3200459   
Chen, L., You, Z., Zhang, N., Xi, J., & Le, X. (2022). UTRAD: Anomaly detection and localization with U-transformer. Neural Networks, 147, 53–62. doi:10.1016/j.neunet.2021.12.008 PMID:34973607   
Dannels, S. (2023). Creating disasters: Recession forecasting with GAN-generated synthetic time series data. arXiv preprint arXiv:2302.10490.   
Guipeng Ding, G. T., Chunqiao Pang, Xiaofeng Wang, Guiru Duan. (2023). Lightweight Siamese network target tracking algorithm based on ananchor free. Journal of Jilin University (Science Edition)/Jilin Daxue Xuebao (Lixue Ban), 61(4).   
Fu, E., Zhang, Y., Yang, F., & Wang, S. (2022). Temporal self-attention-based Conv-LSTM network for multivariate time series prediction. Neurocomputing, 501, 162–173. doi:10.1016/j.neucom.2022.06.014 Fu, J., Wang, L., Ke, J., Yang, K., & Yu, R. (2023). GANAD: A GAN-based method for network anomaly detection. World Wide Web (Bussum), 26(5), 1–22. doi:10.1007/s11280-023-01160-4   
Gan, R. (2023). Stock Price Simulation under Jump-Diffusion Dynamics: A WGAN-Based Framework with Anomaly Detection.   
Guan, S., Zhao, B., Dong, Z., Gao, M., & He, Z. (2022). GTAD: Graph and temporal neural network for multivariate time series anomaly detection. Entropy (Basel, Switzerland), 24(6), 759. doi:10.3390/e24060759 PMID:35741480   
Isgut, M., Gloster, L., Choi, K., Venugopalan, J., & Wang, M. D. (2022). Systematic review of advanced AI methods for improving healthcare data quality in post COVID-19 era. IEEE Reviews in Biomedical Engineering. PMID:36269930   
Jo, W., & Kim, D. (2023). Neural additive time-series models: Explainable deep learning for multivariate timeseries prediction. Expert Systems with Applications, 228, 120307. doi:10.1016/j.eswa.2023.120307   
Kim, D., Jeong, D., Kim, H., Chong, K., Kim, S., & Cho, H. (2022). Spatial contrastive learning for anomaly detection and localization. IEEE Access : Practical Innovations, Open Solutions, 10, 17366–17376. doi:10.1109/ ACCESS.2022.3149130   
Lee, Y., & Kang, P. (2022). AnoViT: Unsupervised anomaly detection and localization with vision transformerbased encoder-decoder. IEEE Access : Practical Innovations, Open Solutions, 10, 46717–46724. doi:10.1109/ ACCESS.2022.3171559   
Li, G., & Jung, J. J. (2023). Deep learning for anomaly detection in multivariate time series: Approaches, applications, and challenges. Information Fusion, 91, 93–102. doi:10.1016/j.inffus.2022.10.008   
Lin, Z., Wang, H., & Li, S. (2022). Pavement anomaly detection based on transformer and self-supervised learning. Automation in Construction, 143, 104544. doi:10.1016/j.autcon.2022.104544 Liu, F., Ma, X., Wu, J., Yang, J., Xue, S., Beheshti, A., Zhou, C., Peng, H., Sheng, Q. Z., & Aggarwal, C. C. (2022). Dagad: Data augmentation for graph anomaly detection. 2022 IEEE International Conference on Data Mining (ICDM), Liu, R., Liu, W., Zheng, Z., Wang, L., Mao, L., Qiu, Q., & Ling, G. (2023). Anomaly-GAN: A data augmentation method for train surface anomaly detection. Expert Systems with Applications, 228, 120284. doi:10.1016/j.eswa.2023.120284   
Liu, S., Zhou, B., Ding, Q., Hooi, B., Zhang, Z., Shen, H., & Cheng, X. (2022). Time series anomaly detection with adversarial reconstruction networks. IEEE Transactions on Knowledge and Data Engineering, 35(4), 4293–4306. doi:10.1109/TKDE.2021.3140058   
Luo, X., Wu, J., Yang, J., Xue, S., Peng, H., Zhou, C., Chen, H., Li, Z., & Sheng, Q. Z. (2022). Deep graph level anomaly detection with contrastive learning. Scientific Reports, 12(1), 19867. doi:10.1038/s41598-022- 22086-3 PMID:36400802   
Ma, S., Nie, J., Kang, J., Lyu, L., Liu, R. W., Zhao, R., Liu, Z., & Niyato, D. (2022). Privacy-preserving anomaly detection in cloud manufacturing via federated transformer. IEEE Transactions on Industrial Informatics, 18(12), 8977–8987. doi:10.1109/TII.2022.3167478   
Mitra, J., Qiu, J., MacDonald, M., Venugopal, P., Wallace, K., Abdou, H., Richmond, M., Elansary, N., Edwards, J., Patel, N., Morrison, J., & Marinelli, L. (2022). Automatic hemorrhage detection from color doppler ultrasound using a generative adversarial network (GAN)-based anomaly detection method. IEEE Journal of Translational Engineering in Health and Medicine, 10, 1–9. doi:10.1109/JTEHM.2022.3199987 PMID:36051823 Ning, E., Wang, C., Zhang, H., Ning, X., & Tiwari, P. (2023). Occluded person re-identification with deep learning: A survey and perspectives. Expert Systems with Applications, 122419.   
Oswal, S., Shinde, S. K., & Vijayalakshmi, M. (2023). Deep learning-based anomaly detection in cyber-physical system. In Big Data Analytics in Intelligent IoT and Cyber-Physical Systems (pp. 59–71). Springer.   
Pang, W., He, Q., & Li, Y. (2022). Predicting skeleton trajectories using a Skeleton-Transformer for video anomaly detection. Multimedia Systems, 28(4), 1481–1494. doi:10.1007/s00530-022-00915-9   
Qiao, Y., Zhang, B., & Zhang, Z. (2022). Unsupervised Anomaly Detection for IoT Data based on Robust Adversarial Learning. 2022 IEEE 24th Int Conf on High Performance Computing & Communications; 8th Int Conf on Data Science & Systems; 20th Int Conf on Smart City; 8th Int Conf on Dependability in Sensor, Cloud & Big Data Systems & Application (HPCC/DSS/SmartCity/DependSys). IEEE.   
Raza, A., Tran, K. P., Koehl, L., & Li, S. (2023). AnoFed: Adaptive anomaly detection for digital health using transformer-based federated learning and support vector data description. Engineering Applications of Artificial Intelligence, 121, 106051. doi:10.1016/j.engappai.2023.106051   
Rewicki, F., Denzler, J., & Niebling, J. (2023). Is it worth it? Comparing six deep and classical methods for unsupervised anomaly detection in time series. Applied Sciences (Basel, Switzerland), 13(3), 1778. doi:10.3390/ app13031778   
Siwach, M., & Mann, S. (2022). Anomaly detection for weblog data analysis using weighted PCA technique. Journal of Information and Optimization Sciences, 43(1), 131–141. doi:10.1080/02522667.2022.2037283 Soleimani-Babakamali, M. H., Soleimani-Babakamali, R., Sarlo, R., Farghally, M. F., & Lourentzou, I. (2023). On the effectiveness of dimensionality reduction for unsupervised structural health monitoring anomaly detection. Mechanical Systems and Signal Processing, 187, 109910. doi:10.1016/j.ymssp.2022.109910   
Swathi, T., Kasiviswanath, N., & Rao, A. A. (2022). An optimal deep learning-based LSTM for stock price prediction using twitter sentiment analysis. Applied Intelligence, 52(12), 13675–13688. doi:10.1007/s10489- 022-03175-2   
Taghian, M., Asadi, A., & Safabakhsh, R. (2022). Learning financial asset-specific trading rules via deep reinforcement learning. Expert Systems with Applications, 195, 116523. doi:10.1016/j.eswa.2022.116523 Tang, C., Xu, L., Yang, B., Tang, Y., & Zhao, D. (2023). GRU-based interpretable multivariate time series anomaly detection in industrial control system. Computers & Security, 127, 103094. doi:10.1016/j.cose.2023.103094 Vos, K., Peng, Z., Jenkins, C., Shahriar, M. R., Borghesani, P., & Wang, W. (2022). Vibration-based anomaly detection using LSTM/SVM approaches. Mechanical Systems and Signal Processing, 169, 108752. doi:10.1016/j. ymssp.2021.108752   
Wang, C., Ning, X., Sun, L., Zhang, L., Li, W., & Bai, X. (2022). Learning discriminative features by covering local geometric space for point cloud analysis. IEEE Transactions on Geoscience and Remote Sensing, 60, 1–15. doi:10.1109/TGRS.2022.3170493   
Wang, X., Yao, Z., & Papaefthymiou, M. (2023). A real-time electrical load forecasting and unsupervised anomaly detection framework. Applied Energy, 330, 120279. doi:10.1016/j.apenergy.2022.120279   
Xia, X., Pan, X., Li, N., He, X., Ma, L., Zhang, X., & Ding, N. (2022). GAN-based anomaly detection: A review. Neurocomputing, 493, 497–535. doi:10.1016/j.neucom.2021.12.093   
Xu, L., Ding, X., Zhao, D., Liu, A. X., & Zhang, Z. (2023). A three-dimensional ResNet and transformer-based approach to anomaly detection in multivariate temporal–spatial data. Entropy (Basel, Switzerland), 25(2), 180. doi:10.3390/e25020180 PMID:36832547   
Yao, J., Wu, W., & Li, S. (2022). Anomaly detection model of mooring system based on LSTM PCA method. Ocean Engineering, 254, 111350. doi:10.1016/j.oceaneng.2022.111350   
Yao, Y., Ma, J., & Ye, Y. (2022). KfreqGAN: Unsupervised detection of sequence anomaly with adversarial learning and frequency domain information. Knowledge-Based Systems, 236, 107757. doi:10.1016/j. knosys.2021.107757   
Zhang, Z., Li, W., Ding, W., Zhang, L., Lu, Q., Hu, P., Gui, T., & Lu, S. (2023). STAD-GAN: Unsupervised anomaly detection on multivariate time series with self-training generative adversarial networks. ACM Transactions on Knowledge Discovery from Data, 17(5), 1–18. doi:10.1145/3572780   
Zheng, Y., Wang, X., Deng, R., Bao, T., Zhao, R., & Wu, L. (2022). Focus your distribution: Coarse-to-fine noncontrastive learning for anomaly detection and localization. 2022 IEEE International Conference on Multimedia and Expo (ICME). IEEE. doi:10.1109/ICME52920.2022.9859925   
Zhu, H., Kang, Y., Zhao, Y., Yan, X., & Zhang, J. (2022). Anomaly detection for surface of laptop computer based on PatchCore GAN algorithm. 2022 41st Chinese Control Conference (CCC). IEEE.

Yongshan Zhang, male, born on March 11, 1972, Han nationality, Shouguang City, Shandong Province, PhD in Economics, Southwest University of Finance and Economics, majoring in international trade in September 2020, associate professor, specializing in trade and investment, research direction: Trade and Investment.

Zhiyun Jiang, female, born on September 23, 1974, Han nationality, Weifang City, Shandong Province, bachelor degree, graduated from Shandong Normal University in 1995, major in Chinese language and literature, associate professor, research direction traditional culture.

Cong Peng, male, April 18, 1976, Han nationality, Wuhan, Hubei, PhD in Management, Southwest University of Finance and Economics, June 2019, Associate Professor of Industrial Economics, Industrial Organization, Development Economics

Xiumei Zhu, female, March 12, 1979, Han nationality, Shouguang, Shandong, PhD in Management, Lyceum Of The Philippines University, October 2023, Associate Professor of Industrial Economics, Financial Management.

Gang Wang, a distinguished researcher, holds a Ph.D. in Bioengineering from Imperial College London. His expertise spans bioengineering and data engineering, and he has made significant contributions to both fields. Currently affiliated with Imperial College London and NingboTech University, he has secured research funding, including the Ningbo Key R&D Program, Zhejiang Province Postdoctoral Research Funding, and Ningbo Natural Science Foundation.