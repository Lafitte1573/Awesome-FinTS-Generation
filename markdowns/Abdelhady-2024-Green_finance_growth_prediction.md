RESEARCH ARTICLE

# Green finance growth prediction model based on time-series conditional generative adversarial networks

Aya Salama Abdelhady1,2, Nadia Dahmani $\mathbb { o } ^ { 3 , 4 * }$ , Lobna M. AbouEl-Magd2,5, Ashraf Darwish2,6,7, Aboul Ella HassanienID2,8,9

1 Faculty of Mathematical and Computational Sciences, University of Prince Edward Island, Charlottetown, Canada, 2 Scientific Research School of Egypt (SRSEG), Cairo, Egypt, 3 College of Technological Innovation, Zayed University, Dubai, UAE, 4 LARODEC, Institut Supe´rieur de Gestion de Tunis, Universite´ de Tunis, Tunis, Tunisia, 5 Computer Science Department, Misr Higher Institute, Mansoura, Egypt, 6 Faculty of Science, Helwan University, Helwan, Egypt, 7 Egyptian Chinese University, Cairo, Egypt, 8 Faculty of Computers and AI, Cairo University, Cairo, Egypt, 9 College of Business Administration (CBA), Kuwait University, Kuwait City, Kuwait

\* nadia.dahmani@isg.rnu.tn, nadia.dahmani@zu.ac.ae

Citation: Abdelhady AS, Dahmani N, AbouEl-Magd LM, Darwish A, Hassanien AE (2024) Green finance growth prediction model based on timeseries conditional generative adversarial networks. PLoS ONE 19(7): e0306874. https://doi.org/ 10.1371/journal.pone.0306874

Editor: Mahmud Iwan Solihin, UCSI University Kuala Lumpur Campus: UCSI University, MALAYSIA

Received: February 14, 2024

Accepted: June 21, 2024

Published: July 24, 2024

Copyright: $\circledcirc$ 2024 Abdelhady et al. This is an open access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.

Data Availability Statement: The used dataset on green finance is publicly available and accessible through the International Monetary Fund’s Climate Data platform: https://climatedata.imf.org/pages/fiindicators#fr3.

Funding: The author(s) received no specific funding for this work.

Competing interests: The authors have declared that no competing interests exist.

# Abstract

Climate change mitigation necessitates increased investment in green sectors. This study proposes a methodology to predict green finance growth across various countries, aiming to encourage such investments. Our approach leverages time-series Conditional Generative Adversarial Networks (CT-GANs) for data augmentation and Nonlinear Autoregressive Neural Networks (NARNNs) for prediction. The green finance growth predicting model was applied to datasets collected from forty countries across five continents. The Augmented Dickey-Fuller (ADF) test confirmed the non-stationary nature of the data, supporting the use of Nonlinear Autoregressive Neural Networks (NARNNs). CT-GANs were then employed to augment the data for improved prediction accuracy. Results demonstrate the effectiveness of the proposed model. NARNNs trained with CT-GAN augmented data achieved superior performance across all regions, with R-squared $( { \mathsf { R } } ^ { 2 } )$ values of $9 8 . 8 \%$ , $9 6 . 6 \%$ , and $9 9 \%$ for Europe, Asia, and other countries respectively. While the RMSE for Europe, Asia, and other countries are 1. $2 6 0 + 2$ , $2 . 1 6 \mathsf e { + } 2$ , and $1 . 1 6 \mathsf e + 2$ respectively. Compared to a baseline NARNN model without augmentation, CT-GAN augmentation significantly improved both ${ \mathsf { R } } ^ { 2 }$ and RMSE. The ${ \mathsf { R } } ^ { 2 }$ values for the Europe, Asia, and other countries models are $96 \%$ , $73 \%$ , and $9 7 . 2 \%$ , respectively. The RMSE values for the Europe, Asia, and various countries models are $2 . 2 4 \mathsf { e } + 2$ , $7 6 + 2$ , and $2 . 0 7 6 \mathrm { + } 2$ , respectively. The Nonlinear Autoregressive Exogenous Neural Network (NARX-NN) exhibited significantly lower performance across Europe, Asia, and other countries with R2 values of $74 \%$ , $52 \%$ , and $86 \%$ , and RMSE values of $1 . 1 1 6 + 2$ , $3 . 6 3 \mathsf e + 2$ , and $1 . 8 6 + 2$ , respectively.

# Introduction

Global warming and climate change are considered as the biggest economic failures and challenging situations. Earth’s atmosphere is witnessing a huge concentration of carbon dioxide, almost more than 420 parts per million (ppm) as per NASA’s data [1]. Accordingly, tackling the challenges of global warming and reducing air pollution falls not only on these nations but is a collective responsibility shared by all of humanity [2, 3].

Since the 1960s, national and international policymakers, economists, and environmental activists have been more conscious of the damaging effects of environmental degradation on climate change. Subsequently, to promote economic development, numerous nations have put forth laws and policies to combat environmental deterioration. To guarantee a clean, safe, healthy, and productive environment, for example, Malaysia implemented the Environmental Quality Act in 1974 [4]. Increasing economic growth is linked to increasing levels of environmental pollution to increase growth engines that depend on consumer and manufacturing activities to meet societal requirements, which in turn causes wasteful pollution and strains environmental resources [4].

There are some commitments made by many countries, including China, against climate change, and some of these pledges include developing the renewable energy industry and modernizing the energy system. Policymakers and authorities have made extensive efforts to make this a reality [5].

Accordingly, the need for green financing has developed to achieve long-term growth and sustainable development, as green financing is defined as financial investments aimed at sustainable development projects that protect the environment. Green finance has many types such as climate finance, industrial pollution control, water sanitation, and biodiversity protection. The main goal of green finance is to protect the environment by reducing or avoiding emissions of greenhouse gases (GHGs).

For all the above, green finance is one of the most important areas of research. This concept has been widely addressed in Western countries that have had the greatest impact on the environment, such as China [6].

While artificial intelligence (AI) facilitates greater efficiency in marketing, creativity has been emphasized as the future of business. Existing theories and frameworks in the literature have failed to adequately explore the impact of AI on investment innovation [7].

This study aims to introduce a model for forecasting the expansion of green finance using time-series conditional generative adversarial networks. The proposed model employs artificial intelligence algorithms on publicly available data to construct a green finance recommendation system capable of forecasting the overall volume of green investments globally. Below are the principal contributions of this research:

• Forecasting the growth of green finance worldwide were proposed   
• Incorporating CT-GAN to address data scarcity issues.   
• Utilizing a straightforward neural network (NAR-NN) suitable for the dataset’s characteristics.

# Related work

According to the studied literature, there are few research investigations that measure the impact of pollution on investments and capital flow [8]. In these studies, researchers have usually relied on mathematical tools such as stochastic calculus [9], random processes, ARIMA time series regression [10], and GARCH volatility models [11] to detect the various time-series patterns. However, the value of financial assets is influenced by an array of factors spanning both financial and non-financial domains. Accordingly, this complexity renders traditional models inadequate.

Authors in [12], discussed the importance of analyzing and forecasting carbon emissions, energy consumption, and the outputs for transitioning to a clean energy economy, especially in rapidly growing markets like China. The paper utilized a nonlinear grey Bernoulli model (NGBM) to predict these indicators and proposed a method to optimize its parameters. The results indicated that the forecasting ability of NGBM with optimized parameters (NGBM-OP) outperforms traditional models like GM and ARIMA, with Mean Absolute Percentage Errors (MAPEs) ranging from 1.10 to 6.26 for out-of-sample data (2004–2009).

The predictions also suggested that between 2011 and 2020, China’s compound annual emissions are expected to grow by $4 . 4 7 \%$ , while energy consumption was forecasted to decrease slightly $\left( - 0 . 0 6 \% \right)$ , and real GDP is expected to increase by $6 . 6 7 \%$ . Moreover, authors in [5] highlighted the strategic importance of developing renewable energy. Through a timeseries analysis, this research revealed that financial development contributed significantly, explaining $4 2 . 4 2 \%$ of the variation in renewable energy growth. Capital market development emerges as the most crucial factor, followed by foreign investment. A comparison with the EU and the US cases suggested that the EU’s approach is more relevant and warrants careful study by Chinese policymakers.

Furthermore, in [13], authors were developing the renewable energy sector and upgrading China’s energy structure play pivotal roles in addressing climate change commitments. Financial issues emerge as a critical constraint, directly tied to the country’s financial development. The study proved that financial development contributes significantly, with capital market development being the most crucial factor, followed by foreign investment, advocating for a closer examination of the EU’s approach by Chinese policymakers.

Three significant contributions have been presented in [14]. First, the authors started by talking about the evolution of the financial well-being domain. Second, they put forth a theoretical framework that delineates the antecedents-based interventions that can be implemented in a particular socioeconomic context to achieve economic well-being. Third, a list of methodological and topical propositions was provided for future researchers and academics to review. They also developed ten future research agendas (FRAs) concerning financial well-being, addressing the need to examine diverse nations with diverse market structures.

On the other side, Machine learning techniques enable investors to enhance financial assets prediction and forecast market strength more accurately than conventional methods. The advent of advanced computer technology such as deep neural networks, and Long Short-Term Memory (LSTM) networks, have prompted a shift toward capturing complex information impacting financial assets [15]. LSTM networks excel at retaining long-term information, unlike traditional models. In addition, Convolutional Neural Networks (CNNs), were adopted to extract features and recognize local dependencies [16].

Combining CNN and LSTM, a model known as ConvLSTM2D was proposed in [17], this research proposed a regression and neural network technique to model stock prices alongside environmental factors, aiming to offer a more precise time series model for stock prices. The model incorporated the ConvLSTM2D network, which extracted all necessary information from air pollution data from major industrialized Chinese cities including Beijing, Taiyuan, Changchun, and Shijiazhuang. Furthermore, Bidirectional LSTM was used in [18] to in investigate how air pollutants indirectly influence investor sentiment and endeavors to establish a more comprehensive and effective stock price prediction framework. The study focused on the SSE Shanghai Enterprises (SSESHE) index and introduced six distinct air pollutants as crucial input parameters. The predictive model developed both, Bidirectional and Long Short-Term Memory (BiLSTM) to project stock closing prices. Additionally, the study compared the proposed model against Support Vector Regression (SVR), Long Short-Term Memory (LSTM), and Gate Recurrent Unit (GRU) models. The experiments concluded that the BiLSTM model that integrated air pollutant data in stock forecasting, achieved the highest prediction accuracy of $9 4 . 1 \%$ .

According to the conducted literature review conducted, the impact of pollution on investments in green finance specifically has never been addressed, despite its importance in measuring the evolution of green finance over the years. Consequently, this research focuses on studying neural time series techniques that can evaluate the success of green finance across various time periods. To aid in analysis and forecasting issues within the tested dataset of investments in green finance across continents over the years, the nonlinear autoregressive neural network (NAR-NN) and NAR-NN have been explored.

# Economic growth of the studied countries

In this paper, we analyze green finance from 40 different countries across 5 continents. Table 1 summarizes the financial status of these countries. The Gross Domestic Product (GDP) represents the total monetary value of all goods and services produced and sold within a country for one year. The global GDP is estimated to be $\$ 100,562,000,000,000$ . Among the countries studied, Tunisia stood out as the sole representative of Africa. Classified as an upper-middleincome country, Tunisia’s Gross Domestic Product (GDP) grew at an annual rate of $3 . 5 \%$ in the pre-revolution period, from 2008 to 2010 [19]. A research program outlined in [20] proposes a multilevel and multidisciplinary approach to financial system policy, aiming for environmental, social, and economic sustainability. The program leverages social sciences to teach students how financial tools can address economic, social, and environmental challenges. A key focus is on achieving the European Union’s "Europe $2 0 3 0 "$ goals, which require an estimated annual investment of EUR 180 billion for the next 20 years, particularly in Central and Eastern Europe, to improve energy efficiency and reduce transport emissions. According to [21], the green bond market, a specific segment focused on climate-friendly projects, was launched in 2007–2008 with the help of the first offerings from Multilateral Development Banks. This market has seen a surge in participation from sub-national agencies, local development funds, and institutions like the World Bank, International Monetary Fund, and the European Investment Bank, particularly between 2007 and 2012.

In Europe, Turkey stands out as an upper-middle-income country with a mixed-market emerging economy, reflecting its ongoing economic development and growth [22]. Shifting to North America, Costa Rica, a Central American nation, is another upper-middle-income country that has witnessed steady economic expansion over the past 25 years [23]. Canada, also in North America, boasts the world’s ninth-largest economy and maintains strong trade partnerships with the United States, China, and the United Kingdom [24]. Finally, in Asia, Japan reigns supreme as the world’s third-largest economy. Moreover, Japan’s position as the world’s leading creditor nation grants it significant global influence with far-reaching economic implications [25].

# Preliminaries

# GAN for data augmentation

Machine learning algorithms often struggle with imbalanced datasets, where one class has significantly more samples than others. To address this challenge, we can leverage two techniques: Generative Adversarial Networks (GANs) and Synthetic Minority Over-sampling Technique (SMOTE). While SMOTE is a useful tool, it can create new samples too similar to the majority class, leading to overfitting and poor model performance. In contrast, GANs excel at learning the distribution of the minority class, generating more representative samples. Additionally, GANs offer a robust way to enrich existing data. These networks consist of two key components: a generator and a discriminator. The generator synthesizes new data points, while the discriminator attempts to distinguish real data from the generated samples. Through this adversarial process, the generator learns to create increasingly realistic synthetic data that fools the discriminator [26].

Table 1. Financial analysis of the studied countries.   

<table><tr><td>Index</td><td>Country</td><td>GDP (nominal, 2022)</td><td>GDP (Abbrev.)</td><td>GDP growth</td><td>Population (2022）</td><td>GDP per capita</td></tr><tr><td>1</td><td>Brunei</td><td>$16,681,531,646</td><td>$16.68 billion</td><td>-1.63%</td><td>449,002</td><td>$37,152</td></tr><tr><td>2</td><td>Malta</td><td>$17,765,270,015</td><td>$17.77 ilion</td><td>6.85%</td><td>533,286</td><td>$33,313</td></tr><tr><td>3</td><td>Iceland</td><td>$27,841,648,044</td><td>$27.84 billion</td><td>6.44%</td><td>372,899</td><td>$74,663</td></tr><tr><td>4</td><td>Cyprus</td><td>$28,439,052,741</td><td>$28.44 billion</td><td>5.63%</td><td>1,251,488</td><td>$22,724</td></tr><tr><td>5</td><td>Estonia</td><td>$38,100,812,959</td><td>$38.10 billion</td><td>-1.29%</td><td>1,326,062</td><td>$28,732</td></tr><tr><td>6</td><td>Latvia</td><td>$41,153,912,663</td><td>$41.15 billion</td><td>1.98%</td><td>1,850,651</td><td>$22,238</td></tr><tr><td>7</td><td>Tunisia</td><td>$46,664,948,952</td><td>$46.66 billion</td><td>2.52%</td><td>12,356,117</td><td>$3,777</td></tr><tr><td>8</td><td>Slovenia</td><td>$62,117,768,015</td><td>$62.12 billion</td><td>5.37%</td><td>2,119, 844</td><td>$29,303</td></tr><tr><td>9</td><td>Costa Rica</td><td>$68,380,838,316</td><td>$68.38 billion</td><td>4.31%</td><td>5,180,829</td><td>$13,199</td></tr><tr><td>10</td><td>Lithuania</td><td>$70,334,299,008</td><td>$70.33billion</td><td>1.89%</td><td>2,750,055</td><td>$25,576</td></tr><tr><td>11</td><td>Croatia</td><td>$70,964,606,465</td><td>$70.96 billion</td><td>6.33%</td><td>4,030,358</td><td>$17,608</td></tr><tr><td>12</td><td>Bulgaria</td><td>$89,040,398,406</td><td>$89.04 billion</td><td>3.36%</td><td>6,781,953</td><td>$13,129</td></tr><tr><td>13</td><td>Slovakia</td><td>$115,469,000,000</td><td>$115 billion</td><td>1.67%</td><td>5,643,453</td><td>$20,461</td></tr><tr><td>14</td><td>Greece</td><td>$219,066,000,000</td><td>$219 billion</td><td>5.91%</td><td>10,384,971</td><td>$21,095</td></tr><tr><td>15</td><td>Kazakhstan</td><td>$220,623,000,000</td><td>$221 billion</td><td>3.20%</td><td>19,397,998</td><td>$11,373</td></tr><tr><td>16</td><td>Peru</td><td>$242,632,000,000</td><td>$243 bilion</td><td>2.68%</td><td>34,049,588</td><td>$7,126</td></tr><tr><td>17</td><td>Portugal</td><td>$251,945,000,000</td><td>$252 bilion</td><td>6.69%</td><td>10,270,865</td><td>$24,530</td></tr><tr><td>18</td><td>Finland</td><td>$280,826,000,000</td><td>$281 billion</td><td>2.08%</td><td>5,540,745</td><td>$50,684</td></tr><tr><td>19</td><td>Czech Republic</td><td>$290,924,000,000</td><td>$291 billion</td><td>2.46%</td><td>10,493,986</td><td>$27,723</td></tr><tr><td>20</td><td>Colombia</td><td>$343,939,000,000</td><td>s344bilin</td><td>7.50%</td><td>51,874,024</td><td>$6,630</td></tr><tr><td>21</td><td>HongKong</td><td>$359,839,000,000</td><td>$360billion</td><td>-3.48%</td><td>7,488,865</td><td>$48,050</td></tr><tr><td>22</td><td>Philppines</td><td>$404,284,000,000</td><td>$404 billion</td><td>7.57%</td><td>115,559,009</td><td>$3,499</td></tr><tr><td>23</td><td>Malaysia</td><td>$406,306,000,000</td><td>$406 billion</td><td>8.69%</td><td>33,938,221</td><td>$11,972</td></tr><tr><td>24</td><td>Ireland</td><td>$529,245,000,000</td><td>$529billion</td><td>11.97%</td><td>5,023,109</td><td>$105,362</td></tr><tr><td>25</td><td>Belgium</td><td>$578,604,000,000</td><td>$579 billion</td><td>3.25%</td><td>11,655,930</td><td>$49,640</td></tr><tr><td>26</td><td>Norway</td><td>$579,267,000,000</td><td>$579 billion</td><td>3.28%</td><td>5,434,319</td><td>$106,594</td></tr><tr><td>27</td><td>Argentina</td><td>$632,770,000,000</td><td>$633 billion</td><td>5.24%</td><td>45,510,318</td><td>$13,904</td></tr><tr><td>28</td><td>Switzerland</td><td>$807,706,000,000</td><td>$808 billion</td><td>2.06%</td><td>8,740,472</td><td>$92,410</td></tr><tr><td>29</td><td>Turkey</td><td>$905,988,000,000</td><td>$906billion</td><td>5.57%</td><td>85,341,241</td><td>$10,616</td></tr><tr><td>30</td><td>Netherlands</td><td>$991,115,000,000</td><td>$991 billion</td><td>4.48%</td><td>17,564,014</td><td>$56,429</td></tr><tr><td>31</td><td>Indonesia</td><td>$1,319,100,000,000</td><td>$1.319trillion</td><td>5.31%</td><td>275,501,339</td><td>$4,788</td></tr><tr><td>32</td><td>Denmark</td><td>$395,404,000,000</td><td>$395billion</td><td>3.82%</td><td>5,882,261</td><td>$67,220</td></tr><tr><td>33</td><td>Spain</td><td>$1,397,510,000,000</td><td>$1.398 trillion</td><td>5.45%</td><td>47,558,630</td><td>$29,385</td></tr><tr><td>34</td><td>South Korea</td><td>$1,665,250,000,000</td><td>$1.665 trilion</td><td>2.56%</td><td>51,815,810</td><td>$32,138</td></tr><tr><td>35</td><td>Brazil</td><td>$1,920,100,000,000</td><td>$1.920 trillion</td><td>2.90%</td><td>215,313,498</td><td>$8,918</td></tr><tr><td>36</td><td>Italy</td><td>$2,010,430,000,000</td><td>$2.010 trillion</td><td>3.67%</td><td>59,037,474</td><td>$34,053</td></tr><tr><td>37</td><td>Canada</td><td>$2,139,840,000,000</td><td>$2.140 trillion</td><td>3.40%</td><td>38,454,327</td><td>$55,646</td></tr><tr><td>38</td><td>Russia</td><td>$2,240,420,000,000</td><td>$2.240 trillion</td><td>-2.07%</td><td>144,713,314</td><td>$15,482</td></tr><tr><td>39</td><td>France</td><td>$2,782,910,000,000</td><td>$2.783 trillion</td><td>2.56%</td><td>64,626,628</td><td>$43,061</td></tr><tr><td>40</td><td>Germany</td><td>$4,072,190,000,000</td><td>$4.072 trillion</td><td>1.79%</td><td>83,369,843</td><td>$48,845</td></tr><tr><td>41</td><td>Japan</td><td>$4,231,140,000,000</td><td>$4.231 trillion</td><td>1.03%</td><td>123,951,692</td><td>$34,135</td></tr><tr><td>42</td><td>China</td><td>$17,963,200,000,000</td><td>$17.963 trillion</td><td>2.99%</td><td>1,425, 887,337</td><td>$12,598</td></tr></table>

https://doi.org/10.1371/journal.pone.0306874.t001

Generative Adversarial Networks (GANs) offer an alternative to conventional augmentation techniques by generating synthetic samples resembling the minority class. GANs excel in learning the distribution of minority classes, resulting in the creation of diverse and realistic synthetic samples, surpassing the interpolation of existing data. Unlike traditional augmentation methods, which may lead to overfitting due to the replication of existing samples, GANs produce samples that deviate from the majority class. This enhances the model’s ability to generalize effectively and accommodate new data instances [27]. GAN training uses iterative optimization. The generator and discriminator are alternately updated using gradient descent to minimize loss functions. This makes the generator and discriminator compete throughout training. The game theory-inspired minimax loss function is the most frequent GAN loss function. Eq (1) calculates mini-max loss for a GAN with generator $\pmb { G }$ and discriminator $\pmb { D }$ [28].

$$
L _ { _ { G A N } } ( G . D ) = E _ { x \sim p _ { d a t a } } ( x ) [ l o g D ( \pmb { x } ) ] + E _ { z \sim p _ { z } } ( z ) [ l o g ( 1 - D ( G ( z ) ) ) ]
$$

Wherex represents real data samples drawn from the true data distribution pdata $( x )$ , $\pmb { z }$ represents random noise (latent vector) drawn from a prior distribution $\pmb { p } z ( z )$ (often a uniform or normal distribution), $G ( z )$ is the output of the generator given the noise z generating synthetic samples, and $D ( x )$ is the discriminator’s output, representing the probability thatx is representing.

The generator minimizes this loss, while the discriminator maximizes it. After training, the generator produces more realistic data that confuses the discriminator, while the discriminator becomes better at distinguishing real from fake data. Conditional GANs for synthetic data generation, also known as CT-GAN, is a synthetic tabular data generator that was developed to solve several problems that were present in the classic GAN. CT-GAN exceeds every method that has been developed to this day and is at least $8 7 . 5 \%$ more effective than Bayesian networks [29].

# Time series neural network

This study employs two distinct types of Time Series Neural Networks which are the Nonlinear Autoregressive Exogenous Neural Network and the Nonlinear Autoregressive Neural Network. Subsequent sections will delve into detailed discussions of these networks.

# 1) Nonlinear Autoregressive Exogenous Neural Network (NARX-NN)

This Network predicts how things change over time [30]. In this case, we use a method called NARX-NN, which is good at giving accurate guesses. Here’s what NARX-NN time series means in this context:

$$
\begin{array} { r l } & { y ( t ) = h ( x ( t - 1 ) . x ( t - 2 ) . \dots . . . . . x ( t - k ) . y ( t - k ) . y ( t - 1 ) . y ( t - 2 ) . . . . . . . y ( t - p ) ) } \\ & { \qquad + \epsilon ( t ) } \end{array}
$$

The anticipated time series $s ( t )$ , is determined by the past value p and is influenced by an additional external time series, $x ( t )$ . The external time series $\mathbf { \eta } ( t )$ , might either have a single dimension or be multi-dimensional. The NARX-NN prediction model utilizes the previous output values along with exogenous input to estimate future values [31]. In this paper, the use of green finance is considered as the input time series at time $t { - } 1$ , denoted as $\left( t - l \right)$ , while the nation variable is regarded as the exogenous input at time $t - l$ , denoted as $x ( t { - } l )$ . The sole resultant is denoted as y(t). The NARX-NN and NAR-NN exhibit significant similarities. The country variable serves as an exogenous input in the NARX model.

# 2) Nonlinear Autoregressive Neural Network (NAR-NN)

Linear mathematical models struggle to capture the complexities of real-world economic scenarios, particularly in forecasting the growth of green finance. These complexities often involve numerous challenges and random fluctuations. To address this limitation, a nonlinear model, as represented by Eq (3), is necessary to predict the magnitude of these fluctuations in green finance growth. One such powerful tool for nonlinear time series forecasting is the Nonlinear AutoRegressive Neural Network (NARNN) described in [32].

$$
y ( t ) = f ( y ( t - 1 ) , \ y ( t - 2 ) , \ y ( t - 3 ) _ { \perp } , y ( t - n ) ) + \in ( t )
$$

In this case, $\boldsymbol { y }$ is the green finance data series at a time $t , n$ is the green finance data series input delay, and $f$ is a transfer function. The neural network is trained to learn the underlying function. This is achieved by adjusting the weights of connections between neurons and the biases of individual neurons to minimize the difference between the network’s predictions and the actual function’s outputs. The $\gamma -$ series of green finance was found by getting close to the term $( \mathrm { t } ) , \in$ which stands for “error tolerance.”

The following is a way to describe NARNN’s endogenous input.

$$
\hat { y } ( t ) = f ( y ( t - 1 ) , \ y ( t - 2 ) , \ y ( t - 3 ) , \ldots y ( t - 2 0 ) ) + \in ( t )
$$

where delay of input ${ \bf n } = 2 0$ . NAR-NN consists of one input layer, one or more hidden layer(s), and one output layer.

NARNN is recurrent and dynamic due to the connection of feedback. In this study, we used the narnet() built-in function for NAR-NN to implement the hyperbolic tangent (tansig, (5)) and sigmoid (logsig, (6)) functions to compare the network accuracies in the context of green finance forecasting.

$$
O _ { t a \mathrm { n s i g } } = \frac { e ^ { u } - e ^ { - u } } { e ^ { u } + e ^ { - u } }
$$

$$
\mathrm { O _ { l o g s i g } = \frac { 1 } { 1 + e ^ { - u } } }
$$

# The Augmented Dickey-Fuller test (ADF)

The Augmented Dickey-Fuller test (ADF) falls under the category of statistical tests known as unit root tests. Certain stochastic processes, like random walks, possess unit roots, which can complicate statistical inference when utilizing time series models. A unit root indicates nonstationarity and doesn’t always exhibit a trend [33]. The ADF test is an ‘augmented’ version of the Dickey Fuller test, it allows for higher-order autoregressive processes by including $\Delta \gamma _ { t - p }$

$$
y _ { t } = c + \beta t + \alpha y _ { t - 1 } + \varnothing _ { 1 } \Delta \gamma _ { t - 1 } + \varnothing _ { 2 } \Delta \gamma _ { t - 2 } \ldots \ldots + \varnothing _ { p } \Delta \gamma _ { t - p } + e _ { t }
$$

ADF tests yield statistics and p-values. At $1 \%$ , $5 \%$ , and $1 0 \%$ significance levels, the test statistic is compared to important values. Decide whether to reject the null hypothesis and declare the time series stationary if the test statistic is less than a predetermined number. As a result, you cannot rule out the null hypothesis, which suggests that there is a unit root in the time series if the test statistic is less negative than this crucial value. The p-value indicates the probability that a test statistic will be obtained that is equally or more extreme than the null hypothesis that was observed. Reject the null hypothesis and, if the p-value is less than the predetermined significance level, conclude that the time series exhibits stationarity. On the contrary, the null

![](images/c17c884609cbdad5deafc256bc2e40dfda7f08bee1536364aa7591b7931ef1f4.jpg)  
Fig 1. The general architecture of the proposed prediction model.

https://doi.org/10.1371/journal.pone.0306874.g001

hypothesis cannot be rejected if the p-value surpasses the predetermined significance level; this would suggest the existence of a unit root in the time series [33].

# The proposed prediction model architecture

A generic preview of the proposed model architecture is presented in Fig 1. It consists of three main phases: data preparation, data augmentation using CT-GAN, and prediction phase using time series network NAR. Algorithm 1 presents the prediction model algorithm, and the next sections present these phases in detail.

Algorithm 1: green finance prediction model 1. Read the dataset. 2. Data aggregation group by continent 3. Perform ADF test to select the appropriate prediction model 4. For each continent’s countries (3 continent) • Generate a fake data from real data using a generator and discriminator models that calculates minimax loss for a GAN with generator G and discriminator D $L _ { _ { G M N } } ( G . D ) = E _ { x \sim p _ { d a t a } } ( x ) [ l o g D ( \pmb { x } ) ] + E _ { z \sim p _ { z } } ( z ) [ l o g ( 1 - D ( G ( z ) ) ) ]$ • Update the dataset • Construct NARNN using narnet() • Train the narnet to predict the green finance amount. • Evaluation the performance based MSE, R2 5. Compare the performance with other time series approaches

# Data preparation

This phase is crucial in readying the data for analysis. It involves two key processes: selecting and aggregating data and conducting statistical analyses to identify the most appropriate prediction model.

Data selection and aggregation. The studied data set includes data green finance data from 40 countries across 5 continents spanning several years, obtained from [34]. Table 2 provides a sample of the data for Denmark, a European nation. While the data included entries from various continents, Europe and Asia had the most comprehensive coverage.

Table 2. Sample of the dataset.   

<table><tr><td>year</td><td>Country</td><td> green finance</td></tr><tr><td>2013</td><td>Denmark</td><td>491.068993</td></tr><tr><td>2014</td><td>Denmark</td><td>287.234455</td></tr><tr><td>2015</td><td>Denmark</td><td>333.685167</td></tr><tr><td>2016</td><td>Denmark</td><td>324.362851</td></tr><tr><td>2017</td><td>Denmark</td><td>300.12899</td></tr><tr><td>2018</td><td>Denmark</td><td>287.217543</td></tr></table>

https://doi.org/10.1371/journal.pone.0306874.t002

Preprocessing was necessary due to the presence of categorical data (shown in Table 3). Additionally, data for different countries were scattered throughout the dataset. To address this, we implemented a two-step organization process. The first step consists of identifying the continent for each country and grouped them into separate files. This analysis revealed that only Europe and Asia had sufficient data for further analysis. The second step is related to data transformation where categorical data is transformed into numerical values.

Preliminary experiments indicated the need for data augmentation. Consequently, we employed CT-GAN (likely referring to Conditional Generative Adversarial Network) to augment the dataset as the final preprocessing step. The visualizations of green finance growth in Europe and Asia are presented in Figs 2–4.

Statistical analysis. Time-series data is valuable for analysis and prediction because it captures trends and patterns that change over time. However, stationary data, which exhibits little change over time, lacks these patterns and isn’t ideal for forecasting. Therefore, it’s crucial to assess data stationarity before proceeding.

To analyze stationarity in our green finance growth data for Europe, Asia, and various countries, we first visualized it. Fig 5(A)–5(C) display the plots for each region, respectively. These visualizations suggest that the data might be non-stationary. To confirm our suspicions, we will employ the Augmented Dickey-Fuller statistical test, a robust method for detecting stationarity [33]

The Augmented Dickey-Fuller (ADF) test results, presented in Table 4, reveal that the green finance growth data across all regions (Europe, Asia, and Other Countries) exhibits non-stationary characteristics. This implies that the data lacks consistent trends or patterns over time.

For each category, the test statistic is higher than the critical values at various significance levels, and the corresponding p-values all exceed the chosen significance level of 0.05. In statistical terms, these results fail to reject the null hypothesis of non-stationarity. Consequently, the green finance growth data cannot be directly used for traditional forecasting methods that rely on stationary data.

Hence, to effectively predict future green finance growth patterns, this study proposes using a Nonlinear AutoRegressive Neural Network (NARNN) model. This type of model is wellsuited for analyzing and predicting non-stationary time-series data.

Table 3. Categories of the dataset.   

<table><tr><td colspan="2">Europe</td><td colspan="2">Asia</td><td colspan="2">Various countries</td></tr><tr><td># courtiers</td><td>Dataset size</td><td># courtiers</td><td>Dataset size</td><td># courtiers</td><td>Dataset size</td></tr><tr><td>24</td><td>249</td><td>9</td><td>102</td><td>7</td><td>66</td></tr></table>

![](images/22c408c82539b99ba8ccb5beca69a3705d56608f9c0acacd27b0165275edfca5.jpg)  
Fig 2. European green finance growth.

https://doi.org/10.1371/journal.pone.0306874.g002

# Experimental results and analysis

To optimize the network’s performance, we employed an iterative approach, evaluating different configurations through multiple tests. The most accurate results were achieved with a single hidden layer containing 20 neurons. We opted for the Levenberg-Marquardt Backpropagation (LMBP) algorithm for training due to its efficiency [35].

Since our goal was one-step-ahead forecasting, a simpler architecture was chosen compared to the typical closed-loop structure used for multi-step predictions. The effectiveness of the final three network configurations was assessed using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-Squared $( \mathbb { R } ^ { 2 } )$ .

![](images/727a6a21848f2e35f7289c3f2801cbba06a291a36a4456cf75f684ab639d08d0.jpg)  
Fig 3. Asian green finance growth.

https://doi.org/10.1371/journal.pone.0306874.g003 https://doi.org/10.1371/journal.pone.0306874.g004

![](images/3df84e6f9b8cfcec7d2388536b603cbe03146b36f9f0ef60c59203ac14ca1142.jpg)  
Fig 4. Various countries’ green finance growth.

![](images/f36a385a9a9571b5c7c22ddf61812d32ab698afcfd73bc2da4f82345be729b5a.jpg)  
Fig 5. (a) Growth in green finance in Europe (b) Growth in green finance in Asia (c) Growth in green finance in Various Countries.

Table 4. The results of the augmented dickey-fuller test.   

<table><tr><td rowspan="7">Europe</td><td>Test Statistic&#x27;</td><td>-1.797584</td></tr><tr><td>&#x27;p-value&#x27;</td><td>0.381662</td></tr><tr><td>Critical Value (1%)</td><td>-4.068854</td></tr><tr><td>Critical Value (5%))</td><td>-3.127149</td></tr><tr><td>Critical Value (10%)</td><td>-2.701730</td></tr><tr><td>Test Statistic</td><td>0.310327</td></tr><tr><td>&#x27;p-value</td><td>0.977813</td></tr><tr><td rowspan="6"></td><td>Critical Value (1%)’</td><td>-4.665186</td></tr><tr><td>&#x27;Critical Value (5%)’</td><td>-3.367187</td></tr><tr><td>&#x27;Critical Value (10%)</td><td>-2.802961</td></tr><tr><td>Test Statistic&#x27;</td><td>-1.111513</td></tr><tr><td>&#x27;p-value</td><td>0.710459</td></tr><tr><td>&#x27;Critical Value (1%)&#x27;</td><td>-4.068854</td></tr><tr><td rowspan="4"></td><td>&#x27;Critical Value (5%)</td><td>-3.127149</td></tr><tr><td>&#x27;Critical Value (10%)</td><td></td></tr><tr><td></td><td>-2.701730</td></tr></table>

# https://doi.org/10.1371/journal.pone.0306874.t004

MSE is a common metric in regression tasks. It measures the average squared difference between predicted values and actual targets (Eq 8). It’s important to note that MSE tends to inflate the impact of small errors due to the squaring, potentially overstating the model’s shortcomings [11].

$$
\mathrm { M S E } = \left( \frac { 1 } { \mathrm { N } } \right) \sum _ { \mathrm { i = 1 } } ^ { \mathrm { N } } \left( \hat { \mathbf { y } } _ { \mathrm { i } } - \mathbf { y } _ { \mathrm { i } } \right) ^ { 2 }
$$

To assess the prediction accuracy, $N$ represents the total number of test samples, where $\mathrm { y _ { i } }$ denotes the ith test sample, and $\hat { \mathsf { y } }$ stands for the predicted value of $\mathrm { y _ { i } }$ . MSE serves as an indicator of the precision of the forecasting results, with a smaller MSE indicating a more accurate forecast.

As shown in Eq 9, the Root Mean Squared Error (RMSE) is utilized to compute the discrepancy between the actual and observed values.

$$
\mathit { R M S E } = \sqrt { \left( \frac { 1 } { N } \right) \sum _ { i = 1 } ^ { N } \left( \hat { y } _ { i } - y _ { i } \right) ^ { 2 } }
$$

Where $N$ is the number of test samples that subscribe to the $\mathrm { i } ^ { \mathrm { t h } }$ test sample, and $\hat { \boldsymbol { y } } _ { i }$ is the predicted value of $\mathrm { y _ { i } }$ .

Because RMSE uses the average error, it is susceptible to aberrant points. The RMSE value is greatly affected if the regression value of a point is not credible, since this will result in a relatively large error. The more accurate the predicted results, the smaller the RMSE. Moving on to R-square (R2), its primary objective is to measure the degree of correlation between predicted and observed data. Consider a dataset comprising $n$ values labeled $\mathrm { { y _ { d 2 } , y _ { d 3 } , . . . , y _ { n } } }$ (often denoted as $\mathrm { y _ { i } }$ or represented as a vector $\mathbf { y } = ( \mathrm { y } 1 , \mathrm { y } 2 , . . . , \mathrm { y } n ) ^ { \textrm { T } }$ , each corresponding to a predicted value f $\mathbf { \tau } _ { 1 } , \mathbf { f } _ { 2 } , . . . . , \mathbf { f } _ { \mathrm { n } }$ . To compute both the total sum of squares and the sum of squares remaining, employ Eq (10) and Eq (11) as follows:

The total of all squares:

$$
\mathrm { \hat { S } _ { t } = \sum _ { i = 1 } ^ { n } \left( y _ { i } - \hat { y } _ { i } \right) ^ { 2 } }
$$

The sum of residual squares is another name occasionally used to refer to the sum of squares.

The definition of it is as follows.

$$
\mathrm { { S _ { r e } } = \sum _ { i = 1 } ( \mathrm { { y _ { i } - f _ { i } } ) ^ { 2 } } }
$$

The most common expression of the coefficient of determination is given in Eq (12).

$$
\mathrm { R } ^ { 2 } = 1 - \frac { \mathrm { S _ { t } } } { \mathrm { S _ { r e } } }
$$

The low values of MSE are indicative of optimal performance. On the other hand, a high R2 signified a considerable degree of accuracy [11]. All experiments will be listed as follows: The low values of MSE indicate the best result. In contrast, the high value of $\mathrm { R } ^ { 2 }$ indicated a high accuracy.

This paper presents four implemented experiments. They are as follows:

• Experiment (1) Data augmentation using CT-GAN   
• Experiment (2) Prediction using NARX- NN   
• Experiment (3) Predicting the green finance using NAR- NN for the original data. • Experiment (4) Predicting the green finance using NAR- NN for the original data.

# Experiment implementation

Experiments are presented in the following sub-sections; the first experiment is carried out for data preparation, and the remaining experiments are carried out to obtain the most accurate findings possible for the prediction process. All these experiments are discussed as follows.

# Experiment (1): Data augmentation using CT-GAN

The experiments were conducted using TensorFlow and Keras on a TPU through Google Colab. To address the issue of limited data, this experiment was initiated. The CT-GAN architecture employs a conditional generator to produce rows conditioned on a single discrete column. It samples training data based on the log-frequency of each category, rather than randomly sampling, thereby ensuring a more balanced representation of minor categories within highly imbalanced categorical columns. This approach aids the GAN model in exploring discrete values evenly.

Additionally, for tabular data, unlike pixel data in images, continuous variables may exhibit non-Gaussian or complex multi-modal distributions. CT-GAN addresses this by normalizing continuous columns based on their mode. In this experiment, 1000 data points were generated for Europe, Asia, and various distributed countries. To evaluate the generated fake data and its similarity to the read data, it shows the evaluation results for European countries in Fig 6(A), Asian countries in Fig 6(B), and other various countries in Fig 6(C), they all show the clear similarities between real and fake generated data.

# Experiment (2): Prediction using NARX- NN

The experiment and subsequent ones were conducted using MATLAB (version R2023a) as the implementation platform. In this experiment, NARX-NN was employed to forecast the effects of carbon dioxide emissions and pollution on investment and capital flows. Once the data preparation phase was completed, the primary dataset was used for this experiment. Predictions were made for various countries, encompassing those from Europe, Asia, and other regions. Fig 7 depicts the performance of the model for each of the three categories. This

![](images/c038d9e5f38a40a8575de8dd2d1ee55bb526e26a085f1feecddd789c19306f38.jpg)  
Fig 6. Experiment (1): Data augmentation using CT-GAN.

https://doi.org/10.1371/journal.pone.0306874.g006

experiment’s $\mathrm { R } ^ { 2 }$ test yielded findings of $7 4 \%$ , $5 2 \%$ , and $8 6 \%$ , respectively, for Europe, Asia, and other regions.

# Experiment (3): Predicting the green finance using NAR- NN for the original data

As a result of the disappointing outcome of the prior experiment, it has been determined to make use of the NAR-NN model because it will be appropriate for the characteristics of the data. NAR-NN was utilized to make predictions on the impact that carbon dioxide emissions and pollution have on investment and capital flows. The experiment was carried out on the primary dataset once the preparation has been completed without data augmentation. The experiment on making predictions was carried out for several countries, including those in Europe, Asia, and other regions. Fig 8 depicts the performance of the model for each of the three categories. The $\mathrm { R } ^ { 2 }$

![](images/b47792d5c9343a5428d3b5580464b4a0c81fad11b088caf443fc82b49f2036d2.jpg)  
Fig 7. Experiment (2) Prediction using NARX-NN.

https://doi.org/10.1371/journal.pone.0306874.g007

test was performed in this experiment, and the findings showed that it was acceptable; the results were $9 6 \%$ , $7 3 \%$ , and $9 7 \%$ , respectively, for Europe, Asia, and other regions.

# Experiment (4) Predicting the green finance using NAR- NN for the original data

The results of the previous experiment were acceptable. However, they were not satisfactory enough; this could be due to the limited amount of data that was trained on. In this experiment, we use data augmented with CT-GAN, and the NAR-NN model will be applied for prediction. The experiment on making predictions was conducted out for a variety of countries, including those in Europe, Asia, and other regions of the world. The performance of the model is depicted in Fig 9 for each of the three different categories. The $\mathrm { R } ^ { 2 }$ test was carried out, and the results were successful with the values $9 8 . 8 \%$ , $9 6 . 6 \%$ , and $9 9 \%$ for Europe, Asia, and other regions respectively.

![](images/a05e39f99ad75f4290b34760ff7749493a73879cd7d9a340a65d69cd24958a10.jpg)  
(c) NAR-NN model performance for various countries   
Fig 8. Experiment (3): prediction using NAR-NN without data augmentation.

https://doi.org/10.1371/journal.pone.0306874.g008 https://doi.org/10.1371/journal.pone.0306874.g009

![](images/9375311fcb105ad033ed1695916bba1e688d073af06ae3e4a6b53dd288345533.jpg)  
(b) NAR-NN model performance for Asia   
  
Fig 9. Experiment (4): Prediction using NAR-NN based CT-GAN data augmentation.

# Results analysis

The findings of all of the experiments are presented in Table 5. While the training $\mathrm { R } ^ { 2 }$ result for the NAR-NN model without data augmentation is superior to the $\mathrm { R } ^ { 2 }$ result of the NARX-NN model without data augmentation, the training results after the data augmentation are lower than the NAR-NN model without data augmentation. This is the case for the model used in European countries. On the other hand, the test and validation $\mathrm { R } ^ { 2 }$ results for the NAR-NN model with CT-GAN data augmentation yield the greatest results $9 8 . 7 \%$ and $9 8 . 8 \%$ respectively.

Table 5. The results the of the all experiments.   

<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">NARX-NN model without data augmentation</td><td colspan="3">NAR-NN model without data augmentation</td><td colspan="3">NAR-NN model with CT-GAN data augmentation</td></tr><tr><td>MSE</td><td>RMSE</td><td>R2%</td><td>MSE</td><td>RMSE</td><td>R2%</td><td>MSE</td><td>RMSE</td><td>R2%</td></tr><tr><td rowspan="3">Europe</td><td>training</td><td>1.48E+04</td><td>1.22E+02</td><td>86</td><td>445.2</td><td>2.11E+01</td><td>99.9</td><td>1.07E+04</td><td>1.03E+02</td><td>99.2</td></tr><tr><td>validation</td><td>2.92E+04</td><td>1.71E+02</td><td>71.3</td><td>1.60E+04</td><td>1.26E+02</td><td>98</td><td>1.80E+04</td><td>1.34E+02</td><td>98.7</td></tr><tr><td>test</td><td>1.23E+04</td><td>1.11E+02</td><td>74</td><td>5.02E+04</td><td>2.24E+02</td><td>96</td><td>1.60E+04</td><td>1.26E+02</td><td>98.8</td></tr><tr><td rowspan="3">Asian</td><td>training</td><td>1.05E+04</td><td>1.02E+02</td><td>97</td><td>2.87E+04</td><td>1.69E+02</td><td>98</td><td>4.80E+04</td><td>2.19E+02</td><td>96.6</td></tr><tr><td>validation</td><td>1.35E+05</td><td>3.67E+02</td><td>61</td><td>2.23E+05</td><td>4.72E+02</td><td>87.6</td><td>4.89E+04</td><td>2.21E+02</td><td>96.5</td></tr><tr><td>test</td><td>1.32E+05</td><td>3.63E+02</td><td>52</td><td>4.90E+05</td><td>7.00E+02</td><td>73</td><td>4.66E+04</td><td>2.16E+02</td><td>96.6</td></tr><tr><td rowspan="3"> various countries</td><td>training</td><td>6.80E+03</td><td>8.25E+01</td><td>98</td><td>3.46E+04</td><td>1.86E+02</td><td>97.7</td><td>1.03E+04</td><td>1.01E+02</td><td>99.2</td></tr><tr><td>validation</td><td>2.05E+05</td><td>4.53E+02</td><td>11</td><td>4.30E+04</td><td>2.07E+02</td><td>96.8</td><td>1.44E+04</td><td>1.20E+02</td><td>98.9</td></tr><tr><td>test</td><td>3.43E+04</td><td>1.85E+02</td><td>86</td><td>4.30E+04</td><td>2.07E+02</td><td>97.2</td><td>1.35E+04</td><td>1.16E+02</td><td>99</td></tr></table>

https://doi.org/10.1371/journal.pone.0306874.t005

The model used for the Asian countries is comparable to the approach used with the European countries. The $\mathrm { R } ^ { 2 }$ value of the NAR-NN model without data augmentation is higher than the $\mathrm { R } ^ { 2 }$ value of the NARX-NN model without data augmentation. Nevertheless, the implementation of data augmentation has led to reduced training results for the NAR-NN model compared to when data augmentation was not used. However, the NAR-NN model, when combined with CT-GAN data augmentation, achieves the highest $\mathrm { R } ^ { 2 }$ result during both the test and validation phases.

Regarding $\mathrm { R } ^ { 2 }$ final results for training, validation, and testing, the NAR-NN model augmented with CT-GAN data yields the highest values for the models of different countries are $9 9 . 2 \%$ , $9 8 . 9 \%$ , and $9 9 \%$ , respectively.

Based on the analysis of the results, the NAR-NN model outperforms other models in all three continents. This is consistent with our previous statistical analysis, which recommended that the NAR-NN model is the most suitable for prediction. The proposed CT-GAN also had a significant positive impact on enhancing the results.

The proposed model enhances market confidence by providing reliable forecasts of green finance growth, reducing uncertainty, and attracting more investment. This research contributes to economic resilience by diversifying economic portfolios, creating new job opportunities, and stimulating technological advancements. The insights can inform evidence-based policies to accelerate the transition to a sustainable economy, such as targeted incentives and subsidies. Green finance also yields societal benefits, such as mitigating environmental degradation and improving public health. By aligning financial interests with environmental objectives, this research contributes to sustainable development and a prosperous future.

# Conclusion and future work

Climate change, driven by rising atmospheric carbon dioxide levels, poses a significant environmental threat. In response, environmentally responsible finance, or "green finance," has emerged as a critical tool. While research on weather and stock prices remains limited, the link between air pollution and financial markets is gaining recognition. Machine learning techniques, particularly time series neural networks, offer more promising forecasting abilities compared to traditional models in financial analysis.

This study aims to predict the future trajectory of green finance and encourage investments in green projects. Notably, the relationship between pollution levels and green finance investments has not been extensively explored. To the best of our knowledge, the relationship between pollution and investments in green finance, particularly, has not been addressed in the literature.

Our research leverages neural time series methods to assess green finance effectiveness across different periods. To address the challenges of analyzing and forecasting investment data from various continents over time, we employed a Nonlinear AutoRegressive Neural Network (NARNN) and similar techniques. We utilized machine learning to predict green finance investments in key continents–Asia and Europe. The data was augmented using a Generative Adversarial Network (GAN) before applying neural time series prediction. The achieved Rsquared $( \mathrm { R } ^ { 2 } )$ values of $9 9 \%$ for both Europe and Asia demonstrate the feasibility and accuracy of our proposed approach.

Future work can involve expanding the study by collecting more data from additional continents like Africa and America. Additionally, incorporating more country-specific features can provide deeper insights into factors influencing green finance.

# Author Contributions

Conceptualization: Ashraf Darwish, Aboul Ella Hassanien.

Data curation: Nadia Dahmani.

Formal analysis: Aya Salama Abdelhady.

Investigation: Lobna M. AbouEl-Magd.

Resources: Aya Salama Abdelhady.

Supervision: Ashraf Darwish, Aboul Ella Hassanien.

Validation: Aboul Ella Hassanien.

Visualization: Lobna M. AbouEl-Magd.

Writing – original draft: Aya Salama Abdelhady.

Writing – review & editing: Nadia Dahmani, Lobna M. AbouEl-Magd, Ashraf Darwish, Aboul Ella Hassanien.

# References

1. Bodansky D. The Paris Climate Change Agreement: A New Hope? American Journal of International Law. 2016; 110(2):288–319. https://doi.org/10.5305/amerjintelaw.110.2.0288   
2. Anwar MN, Shabbir M, et al. Emerging challenges of air pollution and particulate matter in China, India, and Pakistan and mitigating solutions. J Hazard Mater. 2021 Aug 15; 416:125851. https://doi.org/10. 1016/j.jhazmat.2021.125851 PMID: 34492802   
3. Lelieveld J., Evans J., Fnais M. et al. The contribution of outdoor air pollution sources to premature mortality on a global scale. Nature 525, 367–371 (2015). https://doi.org/10.1038/nature15371 PMID: 26381985   
4. Shahbaz Muhammad, et al. "Does financial development reduce CO2 emissions in Malaysian economy? A time series analysis." Economic modelling 35 (2013): 145–152.   
5. Ji Qiang, and Zhang Dayong. "How much does financial development contribute to renewable energy growth and upgrading of energy structure in China?." Energy Policy 128 (2019): 114–124.   
6. Zhou Xiaoguang, Tang Xinmeng, and Zhang Rui. "Impact of green finance on economic development and environmental quality: a study based on provincial panel data from China." Environmental Science and Pollution Research 27 (2020): 19915–19932. https://doi.org/10.1007/s11356-020-08383-2 PMID: 32232752   
7. Ameen Nisreen, et al. "Toward advancing theory on creativity in marketing and artificial intelligence." Psychology & marketing 39.9 (2022): 1802–1825.   
8. Farooq Umar, Ashfaq Khurram, Rustamova Dilbar Rustamovna Ahmad A. Al-Naimi, Impact of air pollution on corporate investment: New empirical evidence from BRICS, Borsa Istanbul Review, 2023, https://doi.org/10.1016/j.bir.2023.03.004.   
9. Grigoriu Mircea. Stochastic calculus: applications in science and engineering. Springer Science & Business Media, 2013.   
10. Adebiyi Ayodele Ariyo, Aderemi Oluyinka Adewumi, and Charles Korede Ayo. "Comparison of ARIMA and artificial neural networks models for stock price prediction." Journal of Applied Mathematics, Volume 2014 | Article ID 614342 | https://doi.org/10.1155/2014/614342.   
11. Alberg D., Shalit H. and Yosef R. (2008) Estimating Stock Market Volatility Using Asymmetric GARCH Models. Applied Financial Economics, 18.15, 1201–1208. https://doi.org/10.1080/ 09603100701604225   
12. Pao Hsiao-Tien, Fu Hsin-Chia, and Tseng Cheng-Lung. "Forecasting of CO2 emissions, energy consumption and economic growth in China using an improved grey model." Energy 40. 1 (2012): 400– 409.   
13. Shahzad Umer, Gupta Mansi, Gagan Deep Sharma Amar Rao, Chopra Ritika, Resolving energy poverty for social change: Research directions and agenda, Technological Forecasting and Social Change, 2022,https://doi.org/10.1016/j.techfore.2022.121777.   
14. Mahendru Mandeep, Gagan Deep Sharma, et al., Is it all about money honey? Analyzing and mapping financial well-being research and identifying future research agenda, Journal of Business Research,2022,https://doi.org/10.1016/j.jbusres.2022.06.034.   
15. Pang X., Zhou Y., Wang P., Lin W., & Chang V. (2018). An innovative neural network approach for stock market prediction. Journal of Supercomputing. https://doi.org/10.1007/s11227-017-2228-y   
16. Albawi S., Mohammed T. A. and Al-Zawi S., "Understanding of a convolutional neural network," 2017 International Conference on Engineering and Technology (ICET), Antalya, Turkey, 2017, pp. 1–6, https://doi.org/10.1109/ICEngTechnol.2017.8308186   
17. Fang Zheng et al. “Climate Finance: Mapping Air Pollution and Finance Market in Time Series”. In: Econometrics 9.4 (2021), p. 43.   
18. Liu Bingchun, et al. "Prediction of SSE Shanghai Enterprises index based on bidirectional LSTM model of air pollutants." Expert Systems with Applications 204 (2022): 117600.   
19. Roio Lavinia Clara Del, et al. "Work-related asthma consequences on socioeconomic, asthma control, quality of life, and psychological status compared with non-work-related asthma: A cross-sectional study in an upper-middle-income country." American journal of industrial medicine 66.6 (2023): 529– 539.   
20. Santos Ana C., Serra Nuno, and Teles Nuno. "Finance and housing provision in Portugal." FESSUD Working Paper Series 79 (2015): 1–58.   
21. Aleksandrova-Zlatanska Svetlana and Desislava Zheleva Kalcheva. “Alternatives for Financing of Municipal Investments—Green Bonds.” Review of Economic and Business Studies 12 (2019): 57–77.   
22. Tasar Izzet, Gultekin Esma, Acci Yunus. "IS Turkey in a middle income TRAP?", Journal of Applied Research in Finance and Economics, (2024), Vol. 1, No. 1, 36–41.   
23. https://www.worldbank.org/en/country/costarica/overview   
24. https://www.investopedia.com/articles/investing/042315/fundamentals-how-canada-makes-its-money. asp.   
25. Thakrar Harsh. "The Japanese Economy–Then, Now and Road Ahead." Now and Road Ahead (December 9, 2022) (2022).   
26. Salehi Pegah, Chalechale Abdolah and Taghizadeh Maryam. “Generative Adversarial Networks (GANs): An Overview of Theoretical Model, Evaluation Metrics, and Recent Developments.” ArXiv abs/ 2005.13178 (2020)   
27. Navidan H. et al., “Generative Adversarial Networks (GANs) in networking: A comprehensive survey and evaluation,” Computer Networks, vol. 194, Jul. 2021, https://doi.org/10.1016/j.comnet.2021. 108149   
28. Dash A., Ye J., and Wang G., “A review of Generative Adversarial Networks (GANs) and its applications in a wide variety of disciplines—From Medical to Remote Sensing,” Oct. 2021, [Online]. Available: http://arxiv.org/abs/2110.01442.   
29. Xu Lei, Skoularidou Maria, Alfredo Cuesta-Infante, Kalyan Veeramachaneni, Modeling Tabular data using Conditional GAN, 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada. https://doi.org/10.48550/arXiv.1907.00503   
30. Lin T., Horne B. G., Tiˇno P., and Giles C. L., “Learning longterm dependencies in NARX recurrent neural networks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 7, no. 6, pp. 1329– 1338, 1996. https://doi.org/10.1109/72.548162 PMID: 18263528   
31. Sarkar R., Julai S., Hossain S., Chong W.T. and Rahman M. (2019) A Comparative Study of Activation Functions of NAR and NARX Neural Network for Long-Term Wind Speed Forecasting in Malaysia. Mathematical Problems in Engineering, 2019, Article ID: 6403081. https://doi.org/10.1155/2019/ 6403081   
32. Nyanteh Y. D., Srivastava S. K., Edrington C. S., and Cartes D. A., “Application of artificial intelligence to stator winding fault diagnosis in Permanent Magnet Synchronous Machines,” Electric Power Systems Research, vol. 103, pp. 201–213, 2013.   
33. Guo Zhichao, Research on the Augmented Dickey-Fuller Test for Predicting Stock Prices and Returns, Proceedings of the 7th International Conference on Economic Management and Green Development,2023. https://doi.org/10.54254/2754-1169/44/20232198   
34. https://climatedata.imf.org/pages/fi-indicators#fr3.   
35. Lv C. et al., "Levenberg–Marquardt Backpropagation Training of Multilayer Neural Networks for State Estimation of a Safety-Critical Cyber-Physical System," in IEEE Transactions on Industrial Informatics, vol. 14, no. 8, pp. 3436–3446, Aug. 2018, https://doi.org/10.1109/TII.2017.2777460