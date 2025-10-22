# Distributed Generative Adversarial Networks for Fuzzy Portfolio Optimization

Xueying Yang $^ { 1 , 2 ( \boxtimes ) }$ , Chen $\mathrm { L i ^ { 1 } }$ , Zidong Han $^ { 1 , 2 }$ , and Zhonghua Lu $^ { 1 , 2 }$ $^ { 1 }$ Computer Network Information Center, Chinese Academy of Sciences, Beijing 100190, China {yangxueying,zdhan,zhlu}@cnic.cn, lichen@sccas.cn 2 University of Chinese Academy of Sciences, Beijing 100049, China

Abstract. Financial time series is one of the most important data in the field of economics and finance, and it is important to forecast and simulate such data effectively based on historical patterns and trends. Existing forecasting models mainly forecasting one-step ahead, and cannot retain the complex characteristics of financial time series data such as serial correlation and the long-term time-dependent relationship. On the other hand, the large-scale data makes the training of the deep learning models a time-consuming process. Therefore, how to forecast financial time series multi-step ahead efficiently has become a key point to improve the asset management capability. At the same time, constructing a fuzzy portfolio optimization for different distributions is also an important direction to improve the robustness of a portfolio model. This paper proposes a distributed financial time series simulating model AssetGANs that simulating multi-step ahead based on GANs, and apply GANs as a parameter simulation method to fuzzy portfolio optimization to provide users with better strategy choices. The paper carries on numerical experiments on real market stock data, compares the results with LSTM and achieves a training speedup of over 573 with 8 GPUs compared to the CPU version.

Keywords: Generative Adversarial Networks  Fuzzy Simulation Portfolio Optimization · Parallel computing

# 1 Introduction

Financial time series forecasting and simulation are vital for investment management decisions, and has become one of the hot spots of scientific research today. In the early research, scholars used Autoregressive Model (AR), Moving Average Model (MA) for financial time series forecasting. However, the above models only reflect linear patterns and are not capable of tackling nonlinear data. In addition, these methods mostly require that the time-series data are stable, which is difficult to achieve in real financial markets. With the increasing complexity, efficiency and volatility of financial markets, the pattern of correlations among data becomes more complex, and accurate simulation of financial time series becomes more difficult. The rapid development of deep learning techniques has provided new solutions for this problem and have become a popular research direction today.

In 2014, Goodfellow et al. [1] proposed Generative Adversarial Networks (GANs), which consist of a generator and a discriminator to generate samples that cannot be discriminated from the real data. GANs are proposed to meet the research and industrial needs of many fields [2–4]. Currently, the fields of image generation are one of the most widely studied and applied fields for GANs, while there are few applications in the field of financial time series analysis [5, 6].

Ricardo and Carrillo predicted whether the price would increase one day ahead by using LSTM as the generator and Convolutional Neural Network as the discriminator [7]. In this case, the accuracy of GANs is slightly lower than LSTM. However, Zhang et al. forecasted the closing price of the stock one day ahead with the LSTM as the generator and Multi-Layer Perceptron as the discriminator [8]. The GANs they proposed performs better than LSTM. Lin et al. simulated future stock price three days ahead with Gated Recurrent Units as the generator and Convolutional Neural Network as the discriminator. It is proved that the GANs got a lower RSME than LSTM when apply to one single stock [9].

GANs can fully exploit the important information in financial time series data and improve the cognitive ability of financial markets [10,11], but their generators and discriminators can oscillate significantly during the iterative process, resulting in longer training time [12]. In order to meet the requirements of practical applications, distributed training is usually used to train models [13].

This paper proposes a financial time series simulating model that can fully capture the complex features of financial time series data, and can effectively reflect the nonlinear dynamic interactions among data, based on the WGANGP [14], and achieves the distributed training of the model based on the highperformance heterogeneous computing system. In addition, this paper conducts comparison experiments on stock data sets to verify the effectiveness of the model and the distributed efficiency. We construct a fuzzy investment portfolio model with the simulated sampling data, and optimizing and solving the portfolio model.

# 2 Generative Adversarial Networks Architecture

# 2.1 Theoretical Background

Traditional GANs consist of a generator and a discriminator. The generator captures the potential distribution of real data samples and generates new data samples; the discriminator discriminates whether the input is real data or generated samples. Traditional GANs are difficult for training and are prone to pattern collapse problem. This problem can be avoided by improving the loss function. Gulrajani et al. proposed a new loss function using the Wasserstein distance. The Wasserstein distance is used to calculate the distance for two data distributions $x \sim P _ { r }$ and $y \sim P _ { g }$ , which is mathematically defined as the minr gimum cost to transform from the data distribution $P _ { r }$ to the data distribution $P _ { g }$ :

$$
W ( P _ { r } , P _ { g } ) = \operatorname* { i n f } _ { \gamma \in \Pi ( P _ { r } , P _ { g } ) } E _ { ( x , y ) \sim \gamma } [ \| x - y \| ] .
$$

The objective of the generator is to minimize the Wasserstein distance. The output of the discriminator is to estimate the Wasserstein distance between the simulated distribution and the real distribution, and the objective function to be maximized is:

$$
L \left( D \right) = - E _ { x \sim P _ { r } } \left[ D \left( x \right) \right] + E _ { x \sim P _ { g } } \left[ D \left( x \right) \right]
$$

The WGAN loss function (the opposite of Eq. 2) can reflect the quality of the generated distribution. The smaller it is, the better the generator’s ability to generate sequences. Therefore, it can be judged from the loss function curve whether the model training has converged. WGAN is particularly sensitive to parameters, and WGAN-GP uses a gradient penalty to avoid this problem. The objective function of the WGAN-GP discriminator is:

$$
L \left( D \right) = - E _ { x \sim P _ { r } } \left[ D \left( x \right) \right] + E _ { x \sim P _ { g } } \left[ D \left( x \right) \right] + \lambda E _ { x \sim P _ { x } } \sim \left[ \left. \nabla D \left( x \right) \right. - 1 \right] ^ { 2 }
$$

# 2.2 Architecture

The generator G is constructed with convolutional neural networks. The input parameter of the convolutional neural network is the normalized $M _ { k }$ , which is used to learn the patten of asset returns over the past $k$ kdays. The input of the simulator is the patten and the random noise vector, which is used to generate the asset return trend for the next $\it { \Delta } l$ days. The training purpose is to minimize the Wasserstein distance between the synthetic data and the real data $M _ { l }$ based on historical data. The architecture is shown in Fig. 1.

![](images/d21ea7889633bdd86e919bd1da60ab5a8551d1e7851c84e90889dd09f99ecf22.jpg)  
Fig. 1. AssetGANs architecture.

# 2.3 Generator

We set a 4-layer convolutional neural networks for Generator at first. The kernel size is 5, the stride is 2, the padding is 2, and the fifth layer is the dense layer. Then, we concatenate the noise vector to the output of the return feature. A dense layer and 2-layer deconvolutional neural networks (the kernel size is 4, the stride is 2, and the padding is 1) are used to generate the simulation returns. The details can be seen in Fig. 2, bs refers to batch size, while sn means number of stocks.

![](images/899ea7025f3d094c539ad2f6b0a302634920056316de8b689c003903f89fe0ba.jpg)  
Fig. 2. Generator architecture.

# 2.4 Discriminator

Discriminator D is based on a 5-layer convolutional neural networks. The kernal size is 5, the stride is 2, the padding is 2. The input is the concatenation of the past real data to future real data or future synthetic data. The details can be seen in Fig. 3.

![](images/c03a2c55df5395d5572f88b50167736f8166a08dfc442a1f05053b5bd861d596.jpg)  
Fig. 3. Discriminator architecture.

# 2.5 Adversarial Training

As the network adversarial training continues, the difference between the real data and the generated data becomes smaller and smaller, and finally D cannot distinguish the authenticity of the input data. At this time, the network reaches a balanced state and the training is completed.

# 3 Generative Adversarial Networks Architecture

In deep learning, the weight matrix is updated by backpropagation, which is conducted by a gradient-based optimization algorithm. All workers have a copy of the model. The global data batch is divided into smaller batches and are processed by different workers. Each worker computes the corresponding loss and gradient for the data it owns. Our AssetGANs are implemented by Pytorch. With the help of the GPUs, we use DistributedDataParallel method to train the GANs. We use init_process_group() to initialize the process group. DistributedSampler() is used to partition data sets, make sure data in every batch is distributed equally to each process, and each process can obtain different data. We use DistributedDataParallel() to package the model, and perform all reduce for the gradients obtained on different GPUs. The losses and gradients are averaged across all GPUs by the Ring All-reduce algorithm before the parameters are updated in each period. Communication library Gloo is used to communicate among multiple nodes.

![](images/bae9005e89891f78ec43a8770d34fa4a526e91dac1940dc007522e7ce8856a5e.jpg)  
Fig. 4. Distributed training for AssetGANs.

# 4 Experimental Results

# 4.1 Experimental Setting

The dataset is the daily prices of stocks selected from Shenzhen Stock Exchange stocks from 2007-12-01 to 2021-12-30. Normalization method is Standard Scale. The parameters are estimated using the Adam optimizer, the learning rate adopts the LR attenuation method, and the number of epochs is set to 32. All experiments run on the computing platform with the following parameters shown in Table 1.

Table 1. The parameters.   

<table><tr><td>Parameter</td><td>ConfRev</td></tr><tr><td>Each Node</td><td>32-core CPU, 4 GPUs</td></tr><tr><td>Operating System</td><td>Centos7.6</td></tr><tr><td>MPI</td><td>hpcx-2.4.1</td></tr><tr><td>Network</td><td>HDR Infiniband (200 Gb)</td></tr></table>

# 4.2 Experimental Results

The loss function curve using the AssetGANs is visualized as Fig. 5. According to the mathematics meaning of the loss function of the WGAN network introduced above, it proves that the simulated data generated by the generator are already getting closer in distribution as the loss function of the discriminator has been declining. The simulations for asset 000002.SZ are shown in Fig. 5.

Strong Scalability and Weak Scalability. We analyze Strong scalability and Weak scalability by training the GANs on 1 Node 1 GPU, 1 Node 4 GPUs, 2 Node 8 GPUs and compute their speedup over the computing time on CPU. We also test when the number increased, the speedup changed.

# 4.3 Experimental Evaluation

The simulation results of GANs are compared to LSTM(Long short-term memory network), the traditional network in Fig. 7. We set a 5-layer network for LSTM model. The first three layers are the LSTM layer (dimension: 64,64,32), the fourth layer is the dropout layer ( $\mathrm { d r o p o u t } = 0 . 0 5 4$ , to prevent over fitting), and the fifth layer is the full connection layer (the number of neurons is 20, to predict the future 20 prices). The parameters are estimated using the Adam optimizer, the learning rate adopts the LR attenuation method, and the maximum number of generations is set to 70. Loss curves for the training set and the validation set are as follows. After 100 epochs, the convergence is achieved. The same data frame as the GANs experiment is shown in Fig. 8. We evaluate the performance for the models through Root Mean Square Error (RMSE), which is defined as:

$$
R M S E = \sqrt { \frac { \displaystyle \sum _ { i = 1 } ^ { N } \left( \mathbf { x } _ { i } - \hat { \boldsymbol { x } } \right) ^ { 2 } } { N } }
$$

![](images/737bd76d9111dfaa3e8afe9c9e594399bd29d39c49aadebe879ca3da09f4ab25.jpg)  
Fig. 5. AssetGANs loss and simulation.

![](images/5c0ebf026a17d5bd4f05fca7894a328e0b4df7f85ef6ccef04fa45eb532632f8.jpg)  
Fig. 6. Strong and weak scalability.

![](images/4c0316202dcfb87a66627bee16710d38cb628d74944ff821731d81c2ddfaf8f1.jpg)  
Fig. 7. LSTM loss and simulation.

Table 2. The parameters.   

<table><tr><td></td><td>LSTM</td><td>AssetGANs</td></tr><tr><td></td><td>RMSE|0.663826|0.461514</td><td></td></tr><tr><td>MSE</td><td>0.440666|0.212995</td><td></td></tr></table>

Table 2 shows the RMSE and MSE of LSTM and AssetGANs. It turns out that our AssetGANs has a lower RMSE than LSTM, and therefore performs better than LSTM in the terms of simulating multiple days in the future.

# 5 Parallel Fuzzy Portfolio Optimization

In recent years, fuzzy portfolio selection theory has been fully developed while the returns of assets were described as fuzzy variables [15]. This paper applies AssetGANs to predict asset returns for fuzzy portfolio optimization. Fuzzy simulation is used to obtain the training data and testing data for the Simulated Annealing Resilient Back Propagation (SARPROP) neural network [16]. Then, Genetic Algorithms are used for global optimization search to solve the optimal solution of the fuzzy Mean-CVaR model. To further improve the model solving efficiency and shorten the model solving time, the MPI-based algorithm is parallelized.

# 5.1 Fuzzy Mean-CVaR Portfolio Model

Take the fuzzy CVaR (Conditional value at risk) as the risk metric, the fuzzy Mean-CVaR portfolio model is:

$$
\begin{array} { l } { \operatorname* { m i n } { \xi _ { C V a R } ( \alpha ) } } \\ { \mathit { s . t . E } [ \sum _ { i = 1 } ^ { n } x _ { i } \xi _ { i } ] \geq r } \\ { \mathit { \Pi } _ { i = 1 } ^ { n } x _ { i } = 1 } \\ { 0 \leq x _ { i } \leq 1 , i = 1 , 2 , . . . , n } \end{array}
$$

$x _ { i }$ is the weight of asset $_ i$ in the portfolio, $\xi _ { i }$ is the rate of return of asset $i$ , which i ican be set as a triangular fuzzy variable $\xi _ { i } \sim ( a _ { i } , b _ { i } , c _ { i } )$ , then the expected return of the portfolio is $E [ \textstyle \sum _ { i = 1 } ^ { n } x _ { i } \xi _ { i } ]$ . $\xi _ { C V a R } ( \alpha )$ i i iis the CVaR risk metric

$$
\xi _ { C V a R } ( \alpha ) = ( \int _ { \alpha } ^ { 1 } \xi _ { V a R } ( \beta ) d \beta ) / ( 1 - \alpha )
$$

$C _ { r }$ is credibility measure, while $\xi _ { V a R } ( \alpha ) = \operatorname* { i n f } \{ x | C r \{ \xi \leq x \} \geq \alpha \}$ .

# 5.2 Optimization Algorithms

Fuzzy Simulation. At the end of the training, the posterior probability distribution of future asset returns learned from the adversarial training process is sampled to generate simulations of future asset returns:

$$
\begin{array} { l } { s _ { 1 } = ( s _ { 1 , 1 } , . . . , s _ { i , 1 } , . . . , s _ { i = n , 1 } ) . . . } \\ { s _ { m } = ( s _ { 1 , m } , . . . , s _ { i , m } , . . . , s _ { i = n , m } ) } \end{array}
$$

The value interval of future return of asset $i$ is $[ b _ { i } , c _ { i } ]$ ,then $b _ { i } = \operatorname* { m i n } _ { 1 \le j \le m } s _ { i , j }$ , $c _ { i } =$ j mmax s . In addition, calculate the mean value of future return of asset $i$ , $a _ { i } =$ $a v a e r a g e s _ { i , j }$ . Then, the triangular fuzzy return of asset $i$ is $a _ { i } = a v a e r a g e s _ { i , j }$ , $1 { \le } j { \le } m$ $1 { \le } j { \le } m$ j mthe triangle-shape grade of membership function is:

$$
u _ { \xi _ { \mathrm { i } } } = \left\{ \begin{array} { l l } { 1 - \frac { a _ { i } - x } { a _ { i } - b _ { i } } , b _ { i } \le x \le a _ { i } } \\ { 1 - \frac { x - a _ { i } } { c _ { i } - a _ { i } } , a _ { i } \le x \le c _ { i } } \\ { 0 } \end{array} \right.
$$

Fuzzy simulation is an application of Monte-Carlo methods, and the details was described in the book [17]. The minimum $r$ that satisfies $C r \{ \sum _ { i = 1 } ^ { n } x _ { i } \xi _ { i } \geq r \} \geq \alpha$ iis fuzzy VaR risk metric. Fuzzy CVaR risk metric can be calculated as:

$$
L ( r ) = \frac { 1 } { 2 } ( \operatorname* { m a x } _ { 0 \leq k \leq N } \{ u _ { k } | \sum _ { i = 1 } ^ { n } x _ { i } \xi _ { i } \geq r \} + 1 - \operatorname* { m a x } _ { 0 \leq k \leq N } \{ u _ { k } | \sum _ { i = 1 } ^ { n } x _ { i } \xi _ { i } < r \} )
$$

Fuzzy simulation is used to generate training data sets for the neural network.

SARPROP Neural Network. The trained SARPROP neural network is used to approximate the fuzzy return expectation and fuzzy CVaR of the portfolio to improve the model solving speed. The output side is the objective function of the model $E [ \textstyle \sum _ { i = 1 } ^ { n } x _ { i } \xi _ { i } ]$ and the fuzzy CVaR risk measure $\xi _ { C V a R } ( \alpha )$ .

Genetic Algorithms. Genetic algorithms are heuristic algorithms that search for optimal solutions by simulating natural evolutionary processes and are suitable for solving global optimization problems. A general genetic algorithm usually consists of chromosome representation, constraint processing, population initialization, individual selection, crossover and variation.

The genetic algorithm is the final step to solve the fuzzy Mean-CVaR model. The next generation population is selected by a rotating roulette wheel method, and the population is changed with a certain probability (crossover and variation). The best individual is selected as the optimal solution of the model.

# 5.3 Parallelization and Results

The algorithm is parallelized by MPI, and the schematic diagram is shown in Fig. 8. The assets used to perform the portfolio are 000963.SZ, 300146.SZ, 600535.SH and 601318.SH. The experiment runs on the same machine and has the same settings as in Chapter 4. The weights for different assets are (0.2491255, 0.3388552, 0.2820555,0.1299638), and the return of the portfolio in 20 days under the model is 4.940837533, which is higher than the return of 4.785934489 when divided the capital equally.

![](images/ffe866abdd88dcb808797f6eda220e2c3834d8f917d2fa4547accb208501b5f8.jpg)  
Fig. 8. The schematic diagram of parallel algorithm.

![](images/aa4baebcc45861db54bcabbb65cdb10b42db47a1eede62060b23903fc672a19f.jpg)  
Fig. 9. Speedup for fuzzy simulation and portfolio optimization.

Figure 9 shows the speedup under different processor scores, and the maximum parallel efficiency is $9 6 . 3 \%$ when the number of processors is 2.

# 6 Conclusion

The development of the financial market provides fund managers with a wealth of investment choices. Simulating financial time series is crucial for constructing efficient investment portfolios. Traditional statistical simulations to capture the complexity of the market are not comprehensive enough. This paper implements distributed AssetGANs based on WGAN-GP to simulate the expected return in the future, and applies the multiple simulated distributions to the fuzzy portfolio. The structure and effect of the neural network and portfolio model are introduced and evaluated by empirical analysis, and the effectiveness of the model is proved by comparing it with the LSTM model. The correlations among time series stock data are preserved when generating data. We solve the problem of long training time for AssetGANs and long simulating time in the process of fuzzy investment portfolio optimization. Our future research plan is to implement to apply AssetGANs to high-frequency financial time series.

Acknowledgements. This work is partially supported by the China Postdoctoral Science Foundation (Grant No. 2021M693226) and Beijing Natural Science Foundation (Grant No. 4232039).

# References

1. Goodfellow, I., et al.: Generative adversarial nets. In: Advances in Neural Information Processing Systems, vol. 27 (2014)   
2. Polamuri, S.R., Srinivas, K., Mohan, A.K.: Multi-model generative adversarial network hybrid prediction algorithm (MMGAN-HPA) for stock market prices prediction. J. King Saud Univ.-Comput. Inf. Sci. 34(9), 7433–7444 (2022) 3. Staffini, A.: Stock price forecasting by a deep convolutional generative adversarial network. Front. Artif. Intell. 5, 837596 (2022)   
4. Mariani, G., et al.: Pagan: portfolio analysis with generative adversarial networks. arXiv preprint arXiv:1909.10578 (2019)   
5. Faraz, M., Khaloozadeh, H.: Multi-step-ahead stock market prediction based on least squares generative adversarial network. In: 2020 28th Iranian Conference on Electrical Engineering (ICEE), pp. 1–6. IEEE (2020)   
6. Jiang, J.: Stock market prediction based on SF-GAN network. In: 6th International Symposium on Computer and Information Processing Technology (ISCIPT), pp. 97–101. IEEE (2021) 7. Romero, R.A.C.: Generative adversarial network for stock market price prediction. CD230: Deep Learning, p. 5. Stanford University (2018)   
8. Zhang, K., Zhong, G., Dong, J., Wang, S., Wang, Y.: Stock market prediction based on generative adversarial network. Procedia Comput. Sci. 147, 400–406 (2019)   
9. Lin, H., Chen, C., Huang, G., Jafari, A.: Stock price prediction using generative adversarial networks. J. Comput. Sci. 17–188 (2021)   
10. Zhou, X., Pan, Z., Hu, G., Tang, S., Zhao, C.: Stock market prediction on highfrequency data using generative adversarial nets. Math. Probl. Eng. (2018)   
11. Sonkiya, P., Bajpai, V., Bansal, A.: Stock price prediction using BERT and GAN. arXiv preprint arXiv:2107.09055 (2021)   
12. Kumar, D., Sarangi, P.K., Verma, R.: A systematic review of stock market prediction using machine learning and statistical techniques. Mater. Today Proc. 49, 3187–3191 (2022)   
13. Using the latest advancements in AI to predict stock market movements. https:// github.com/borisbanushev/stockpredictionai. Accessed 4 Oct 2023   
14. Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., Courville, A.C.: Improved training of Wasserstein GANs. In: Advances in Neural Information Processing Systems, vol. 30 (2017)   
15. Zadeh, L.A.: Fuzzy sets. Inf. Control 8(3), 338–353 (1965)   
16. Treadgold, N., Gedeon, T.: The Sarprop algorithm, a simulated annealing enhancement to resilient back propagation. In: Proceedings International Panel Conference on Soft and Intelligent Computing, pp. 293–298 (1996)   
17. Liu, B., Liu, Y.-K.: Expected value of fuzzy variable and fuzzy expected value models. IEEE Trans. Fuzzy Syst. 10(4), 445–450 (2002)