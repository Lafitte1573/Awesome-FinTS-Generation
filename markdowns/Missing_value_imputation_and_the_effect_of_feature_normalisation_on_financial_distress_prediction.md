# Missing value imputation and the effect of feature normalisation on financial distress prediction

Kuen-Liang Sue, Chih-Fong Tsai & Hau-Min Tsau

To cite this article: Kuen-Liang Sue, Chih-Fong Tsai & Hau-Min Tsau (2024) Missing value imputation and the effect of feature normalisation on financial distress prediction, Journal of Experimental & Theoretical Artificial Intelligence, 36:8, 1467-1483, DOI: 10.1080/0952813X.2022.2153278

To link to this article: https://doi.org/10.1080/0952813X.2022.2153278

# ARTICLE

# Missing value imputation and the effect of feature normalisation on financial distress prediction

Kuen-Liang Sue, Chih-Fong Tsai and Hau-Min Tsau

Department of Information Management, National Central University, Taoyuan, Taiwan

# ABSTRACT

In this paper, we focus on comparing the imputation performance of different deep and machine learning techniques on nine related datasets containing different missing rates ranging from $10 \%$ to $5 0 \%$ . Moreover, since each feature value ranges differently, such as total liability/equity ratio and earnings per share, the effect of feature normalisation on the imputation results is also examined to see whether normalising the feature values after missing value imputation can improve the prediction model performance. The experimental results show that the deep neural network technique does not necessarily perform better than the traditional machine learning technique for missing value imputation. In particular, the random forest imputation model performs the best, whereas the $\mathsf { k }$ -nearest neighbour method is the second best imputation model in terms of the AUC rates and type II errors. The performance in improvement of prediction models after performing feature normalisation is heavily dependent on the chosen classification technique. There is thus no need to consider the normalisation step when the random forest classifier is used. It is found that the deep neural network and support vector machine classifiers can significantly outperform those without feature normalisation.

# ARTICLE HISTORY

Received 6 April 2022   
Accepted 25 November 2022

# KEYWORDS

Machine learning; deep learning; missing value imputation; feature normalisation; financial distress prediction

# Introduction

Financial distress prediction has long been regarded as an important problem in data mining and machine learning. It focuses on developing models which can identify whether a customer or company is likely to go bankrupt or not, for classification into the good or bad credit class. Financial institutions can take advantage of the prediction results to help them make loan decisions (Lin et al., 2011; Louzada et al., 2016; Qu et al., 2019).

In the related literature, the major aim of most studies has been to develop more effective models to produce more accurate predictions than the baseline models. For example, deep learning techniques have recently been employed to construct bankruptcy prediction models (Feng et al., 2019; Smiti & Soui, 2020) and credit scoring models (Herasymovych et al., 2019; Shen et al., 2021), demonstrating the superior performance of deep learning techniques over many traditional machine learning models. It has also been demonstrated that better prediction accuracies can be obtained by combining multiple single models based on ensemble learning techniques, such as bagging, boosting, and stacking (Liang et al., 2018; Plawiak et al., 2019; Sivasankar et al., 2020).

On the other hand, data preprocessing techniques, such as feature selection to filter out noisy or unrepresentative features, have also been studied and are regarded as an important factor affecting the performance of prediction models (Liang et al., 2020; Tsai et al., 2021). Similarly, data sampling techniques including under- and over-sampling methods to balance the numbers of data samples in the majority (e.g., non-bankrupt) and minority (e.g., bankrupt) classes have been applied to solve the class imbalance problem in bankruptcy prediction and credit scoring datasets, to avoid the prediction models producing biased results (Shen et al., 2020; Veganzones & Severin, 2018).

However, practical financial distress datasets usually contain some missing attribute values, a problem usually dealt with in previous studies by directly removing the data with missing values from the collected datasets in order to construct the prediction models. In the literature very few works have focused on this problem in financial distress prediction. For example, Zhou and Lai (2017) investigated several well-known missing value imputation methods, including the k-nearest neighbour and attribute mean methods, in combination with different prediction models by using the AdaBoost algorithm to identify the best combination for bankruptcy prediction. Based on American and Japanese bankruptcy datasets, they showed that the combination of the $\mathsf { k }$ -nearest neighbour technique with the AdaBoost based decision tree model can provide better performance. Cheng et al. (2019) proposed a purity-based k-nearest neighbour algorithm to improve the missing value imputation result. In particular, they simulated different missing rates, including $5 \%$ , $1 0 \%$ , $1 5 \%$ , $2 0 \%$ , and $2 5 \% ,$ over the Taiwanese bankruptcy prediction dataset and showed that their proposed algorithm could outperform several well-known imputation methods, such as multiple imputation and k-nearest neighbour. Lan et al. (2020) proposed a novel method based on the Bayesian network to impute missing values in three credit scoring datasets, in which a $1 0 \%$ missing rate is simulated. The results show that it outperformed the mean/mode and expectation maximisation imputation methods.

There are several limitations in these related works focusing on the missing value problem. First, deep learning related techniques have not been employed for missing value imputation of financial distress datasets. It is unknown whether deep learning techniques can provide better imputation results than traditional machine learning techniques, such as the k-nearest neighbour and random forest methods. Second, since the collected features representing each data sample are based on related financial ratios (Liang et al., 2020), they are all continuous feature values but their ranges are usually significantly different, for example total liability/equity ratios ranging from 1.48 to 10,967.38, earnings per share from $- 2 2 . 6 8 $ to 83.09, and operating funds to liability from $- 6 . 7 6$ to 13.09.1 In such cases, feature normalisation can be performed to normalise the range of all features to [0, 1] or [−1, 1] (Han et al., 2011; Singh & Singh, 2020). However, there has been no study focusing on using a combination of feature normalisation and missing value imputation, to impute the original ranges of feature values and then normalise these feature values.

Therefore, there are two research objectives pursued in this study, corresponding to the limitations in past work discussed above. The first is to compare the performance of different deep and machine learning techniques, in particular, the k-nearest neighbour, random forest, MICE (multivariate imputation by chained equations) and deep neural network methods, which are employed over nine financial distress datasets, including four bankruptcy prediction and five credit scoring datasets. Moreover, different missing value percentages are simulated, from $1 0 \%$ to $5 0 \% ,$ at $1 0 \%$ intervals, in order to fully understand the performance of different imputation models. Second, feature normalisation is performed over the imputation results produced by different imputation models to determine whether the combination of missing value imputation and feature normalisation can further improve the prediction performances.

The rest of this paper is organised as follows. Section 2 review missing value imputation and feature normalisation. The research methodology including the experimental procedure and setup are described in section 3. Section 4 presents the experimental results and Section 5 concludes the paper.

# Literature review

# Missing value imputation

According to Little and Rubin (2002), there are three types of missing data, which are missing completely at random (MCAR), missing at random (MAR), and missing not at random (MNAR). When data are MCAR, it means that the missingness is unrelated to any observed or unobserved values from the dataset. In MAR, the cause of the missing data is related to observed values from the dataset. In MNAR, the missing data may be related to the same value and/or with other unknown data.

Missing value imputation can be divided into statistical-based and machine learning-based methods, with most methods, such as linear regression, k-nearest neighbour, random forests, etc., based on developing a model to estimate a suitable value to replace the missing value (GarciaLaencina et al., 2010; Lin & Tsai, 2020; Pereira et al., 2020).

One representative statistical-based method is multivariate imputation by chain equations (MICE), which uses a series of regression models to create multiple imputations, i.e., filling in the missing values multiple times (Azur et al., 2011).

The k-nearest neighbour (KNN) technique is one of the most widely used machine learning-based methods. It searches for the most similar neighbours, i.e., observed data, based on some distance functions. Once the nearest neighbour(s) have found, a replacement value to substitute the missing attribute value can be estimated (Lin & Tsai, 2020).

On the other hand, the random forest (RF) method is based on an ensemble of decision trees, in which a decision tree is trained to impute single or multiple missing values within the data. Consequently, the imputation result, i.e., the final prediction, is obtained by merging the predictions produced by multiple decision trees (Stekhoven & Buhlmann, 2012).

# Feature normalization

In feature normalisation the raw data are either re-scaled or transformed such that each feature has a uniform contribution. Specifically, it aims at normalising the range of independent variables or features of data, as a data pre-processing step in data mining (Han et al., 2011). Min-max normalisation and z-score normalisation are the most commonly used methods. In min-max normalisation, which is the simplest method, the range of features is re-scaled so that they end up in the range of [0, 1] or [−1, 1]. For example, the min-max normalisation of [0, 1] is based on

$$
x ^ { \prime } = \frac { x - m i n ( x ) } { M a x ( x ) - m i n ( x ) } ,
$$

where $x$ and $x ^ { \prime }$ are the original value and normalised value of a specific feature, respectively.

Z-score normalisation, also called standardisation, ensures that the value of each feature in the data will have a zero mean (by subtracting the mean in the numerator) and unit variance. Specifically, z-score normalisation determines the distribution mean and standard deviation for each feature, and then the mean from each feature is subtracted. Next, the value (mean is already subtracted) of each feature is divided by its standard deviation. The general formula for z-score normalisation is

$$
x ^ { \prime } = \frac { x - a v e r a g e ( x ) } { s d } ,
$$

where $x$ is the original feature value; average $( x )$ is the mean of the feature; and $s d$ is its standard deviation.

# Research methodology

# Missing value imputation

The process of missing value imputation is shown in Figure 1. Given a dataset, the 5-fold cross validation method is applied to divide the dataset into $80 \%$ training and $2 0 \%$ testing subsets, respectively. Then, the Synthetic Minority Oversampling Technique (SMOTE) algorithm is performed over the training subset to generate synthetic data samples for the minority class, i.e., the bankrupt or bad credit class in this paper. As a result, the original class imbalanced training subset becomes balanced, containing the same number of data samples in both majority and minority classes.

![](images/86f53a30afdb9157f1a3dbb53442bfa6d5b624b0a4e9bbef0a6ce00d87adbcd6.jpg)  
Figure 1. The process of missing value imputation.

Next, the missing value simulation is performed over the balanced training subset based on the missing completely at random (MCAR) mechanism. As a result, the balanced training subset becomes incomplete, which contains a number of missing data. Then, different imputation algorithms are used to individually impute the incomplete training subset. At the same time, the testing subset is also simulated with some missing values, and then the constructed imputation models are used to impute the missing values of the testing subset.

After imputation of the incomplete training subset by each imputation algorithm, a specific classification technique is employed to construct a class as the bankruptcy prediction or credit scoring model. Finally, in order to examine the classifier performance, the imputed testing subset is used.

# Combination of missing value imputation and feature normalization

Figure 2 shows the process of combining missing value imputation and feature normalisation. The over-sampling step and missing value simulation as well as imputation process are the same as described in the previous process. After imputation of the incomplete training and testing subsets, minMax normalisation is performed over the imputed training subset for feature normalisation. After identifying the maximum and minimum feature values in the training subset, they are applied to normalise the feature values in the testing subset.

When both training and testing subsets are normalised, they are respectively used to train and test the classifiers. Examining the performance of the classifiers which have been trained and tested by the two different processes can allow us to better understand the effect of feature normalisation on missing value imputation for financial distress prediction.

# Experiments

# Experimental setup

# Datasets

In this study, 9 related datasets including four bankruptcy prediction and five credit scoring datasets, which contain various numbers of features and instances, are chosen for the experiments. Table 1 lists the basic information for these 9 datasets. Note that for the later experiments, the SMOTE algorithm is used to balance all of the datasets except for JPNBDS and USABDS to ensure that their class imbalance ratios become 1. Moreover, for missing value simulation, five different missing rates, $10 \%$ , $20 \%$ , $30 \%$ , $4 0 \% ,$ and $5 0 \%$ , are compared and the performance trend examined. The missing rate simulation is based on the missing completely at random (MCAR) mechanism.

![](images/b8300f7fb06f7d99ce5986164a82c2f0f5169dd539af5568df7888f1c5583c9c.jpg)  
Figure 2. The process of combining missing value imputation and feature normalization.

# Related algorithms and their parameters

In this paper, the four imputation algorithms used for performance comparison are KNN, MICE, RF, and the deep neural network (DNN). Feature normalisation is based on the application of min-max normalisation to transform each feature into the range [0, 1]. After each dataset is normalised and imputed, three different classifiers are constructed individually; they are support vector machine (SVM), RF, and DNN. The algorithms are implemented using the Scikit-Learn Library and GitHub Packages in Python. The related parameters are listed in Table 2. 8

# Evaluation metrics

In this study, the area under the Receiver Operating Characteristics (ROC) curve (AUC) rates (Fawcett, 2006) and type II errors of the constructed classifiers are examined, based on the result of a confusion matrix. An example of bankruptcy prediction is shown in Table 3.

The ROC is represented by an x-y axis graph, where the x and y axes show the TP and FP rates with different thresholds. In particular, the TP rate means the rate of correct classification of bankruptcy cases whereas the FP rate indicates the rate of incorrect classification of the non-bankrupt cases. The TP and FP rates are calculated by

$$
T P _ { r a t e } = \frac { T P } { ( T P + F N ) } ;
$$

Table 1. Basic information for the nine datasets.   

<table><tr><td colspan="5"></td></tr><tr><td>Dataset</td><td>No.of features</td><td>No. of instances</td><td>No.of bankruptcy/bad credit cases</td><td>Imbalance ratios</td></tr><tr><td>Bankruptcy prediction</td><td></td><td>1321</td><td>697</td><td>1.1</td></tr><tr><td>Bankruptcy (Olson et al., 2012)</td><td>16</td><td></td><td></td><td></td></tr><tr><td>JPNBDS (Zhou &amp; Lai, 2017)</td><td>11</td><td>152</td><td>76</td><td>1</td></tr><tr><td>TEJ-Taiwan2 USABDS</td><td>95 11</td><td>6819 2336</td><td>220 1168</td><td>30 1</td></tr><tr><td>(Zhou &amp; Lai,2017)</td><td></td><td></td><td></td><td></td></tr><tr><td>Credit scoring</td><td></td><td></td><td></td><td></td></tr><tr><td>Australian3</td><td>14</td><td>690</td><td>383</td><td>1.2</td></tr><tr><td>German4</td><td>20</td><td>1000</td><td>300</td><td>2.3</td></tr><tr><td>Japanese5</td><td>15</td><td>690</td><td>383</td><td>1.2</td></tr><tr><td>Kaggle6</td><td>10</td><td>150000</td><td>10026</td><td>13.4</td></tr><tr><td>PAKDD7</td><td>37</td><td>50000</td><td>13041</td><td>2.8</td></tr></table>

Table 2. Related parameters of the implemented algorithms.   

<table><tr><td>Algorithms</td><td>Parameters</td></tr><tr><td>SVM</td><td>Kernel = RBF</td></tr><tr><td>RF</td><td>n_estimators = 100 criterion = gini</td></tr><tr><td>DNN</td><td>hidden_layer= 2 layer_size= (100,100) learning_rate= 0.01 activation = relu epochs = 100 early_stopping = true optimizer= Adam</td></tr></table>

Table 3. The confusion matrix.   

<table><tr><td rowspan="2" colspan="2"></td><td colspan="2">Actual classification</td></tr><tr><td>Bankrupt</td><td>Non-bankrupt</td></tr><tr><td rowspan="2">Predicted classification</td><td>Bankrupt</td><td>True positive (TP)</td><td>False positive (FP)</td></tr><tr><td>Non-bankrupt</td><td>False negative (FN)</td><td>True negative (TN)</td></tr></table>

$$
F P _ { r a t e } = \frac { F P } { ( T N + F P ) } .
$$

On the other hand, the type II error measures the rate of classification of bankruptcy cases into the non-bankrupt class. This information is very critical for financial institutions since high type II error rates are likely to increase bad debts. The type II error rate is obtained by

$$
T y p e I I = \frac { F N } { ( T P + F N ) } .
$$

# Results on missing value imputation

Figure 3 shows the AUC rates of the RF classifier for different imputation algorithms, whereas Figure 4 shows the type II error results. Note that the baseline is for the mean and mode methods for the continuous and discrete feature values, respectively.

These results show that when the missing rates increase, there is a gradual degradation in the performance of the RF classifier. On average, these imputation algorithms perform similarly when the missing rates are less than $3 0 \% ,$ in which case there is significant difference in their level of performance. However, when the missing rates are larger than $3 0 \%$ i.e., $40 \%$ or $5 0 \% ,$ KNN and RF are the top two imputation methods. In contrast, there is a sharp degradation in the performance of the DNN method when the missing rates increase. More specifically, the imputation results cause the RF classifier to provide the lowest AUC and highest type II error rates. This indicates that deep learning techniques, such as DNN, may not be a good choice for missing value imputation for bankruptcy prediction and credit scoring datasets.

Tables 4 and 5 show the average performance obtained with the three different classifiers, DNN, RF, and SVM, based on the imputation results. The best combination of imputation method and classifier for each missing rate is underlined.

As we can see, the RF classifier outperforms the DNN and SVM classifiers for $10 \%$ to $5 0 \%$ missing rates, in terms of the AUC rates and type II errors. The level of performance difference between them is significant $( p < 0 . 0 5 )$ .

The RF imputation method is the better choice for combination with the best imputation method and KNN is second best. The exception is when the missing rate is $1 0 \%$ , when the DNN imputation results allow the RF classifier to provide the highest AUC rates. On the other hand, for the type II error, the baseline imputation method combined with the RF classifier performs reasonably well. On average, the imputation results provided by the mean/mode method are better than the DNN and MICE results.

![](images/1cb8794e7e1b3d66998d581beb6ffb9453d57f12bab559d50c0efdbaf5c55c19.jpg)  
Figure 3. AUC rates for the RF classifier by different imputation algorithms.

![](images/ded50b874ea4fc13779cd87906f4ff432f407fa1905fe3fe9810e4ab7dc19a3f.jpg)  
Figure 4. Type II errors for the RF classifier by different imputation algorithms.

Table 4. AUC rates for the DNN, RF, and SVM classifiers.   

<table><tr><td colspan="2">Imputation</td><td colspan="5">Missing rates</td><td rowspan="2">Average</td></tr><tr><td>methods</td><td>Classifiers</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td rowspan="3">Baseline</td><td>DNN</td><td>0.73</td><td>0.72</td><td>0.69</td><td>0.66</td><td>0.63</td><td>0.69</td></tr><tr><td>RF</td><td>0.84</td><td>0.83</td><td>0.81</td><td>0.79</td><td>0.78</td><td>0.81</td></tr><tr><td>SVM</td><td>0.66</td><td>0.65</td><td>0.61</td><td>0.59</td><td>0.58</td><td>0.62</td></tr><tr><td rowspan="3">DNN</td><td>DNN</td><td>0.73</td><td>0.71</td><td>0.69</td><td>0.68</td><td>0.65</td><td>0.69</td></tr><tr><td>RF</td><td>0.85</td><td>0.83</td><td>0.81</td><td>0.79</td><td>0.77</td><td>0.81</td></tr><tr><td>SVM</td><td>0.69</td><td>0.69</td><td>0.66</td><td>0.63</td><td>0.60</td><td>0.65</td></tr><tr><td rowspan="3">KNN</td><td>DNN</td><td>0.74</td><td>0.73</td><td>0.72</td><td>0.70</td><td>0.68</td><td>0.71</td></tr><tr><td>RF</td><td>0.84</td><td>0.84</td><td>0.82</td><td>0.80</td><td>0.78</td><td>0.82</td></tr><tr><td>SVM</td><td>0.70</td><td>0.69</td><td>0.68</td><td>0.67</td><td>0.64</td><td>0.68</td></tr><tr><td rowspan="3">MICE</td><td>DNN</td><td>0.73</td><td>0.71</td><td>0.69</td><td>0.66</td><td>0.64</td><td>0.68</td></tr><tr><td>RF</td><td>0.84</td><td>0.83</td><td>0.81</td><td>0.79</td><td>0.77</td><td>0.81</td></tr><tr><td>SVM</td><td>0.68</td><td>0.66</td><td>0.64</td><td>0.63</td><td>0.61</td><td>0.64</td></tr><tr><td rowspan="3">RF</td><td>DNN</td><td>0.74</td><td>0.73</td><td>0.73</td><td>0.71</td><td>0.69</td><td>0.72</td></tr><tr><td>RF</td><td>0.84</td><td>0.84</td><td>0.82</td><td>0.81</td><td>0.79</td><td>0.82</td></tr><tr><td>SVM</td><td>0.71</td><td>0.70</td><td>0.69</td><td>0.68</td><td>0.67</td><td>0.69</td></tr></table>

Table 5. Type II errors for the DNN, RF, and SVM classifiers.   

<table><tr><td rowspan="2">Imputation methods</td><td></td><td colspan="5">Missing rates</td><td rowspan="2">Average</td></tr><tr><td>Classifiers</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td rowspan="3">Baseline</td><td>DNN</td><td>0.37</td><td>0.39</td><td>0.42</td><td>0.46</td><td>0.49</td><td>0.43</td></tr><tr><td>RF</td><td>0.22</td><td>0.24</td><td>0.25</td><td>0.26</td><td>0.28</td><td>0.25</td></tr><tr><td>SVM</td><td>0.40</td><td>0.42</td><td>0.44</td><td>0.49</td><td>0.53</td><td>0.46</td></tr><tr><td rowspan="3">DNN</td><td>DNN</td><td>0.36</td><td>0.38</td><td>0.41</td><td>0.45</td><td>0.49</td><td>0.42</td></tr><tr><td>RF</td><td>0.22</td><td>0.24</td><td>0.25</td><td>0.27</td><td>0.30</td><td>0.26</td></tr><tr><td>SVM</td><td>0.39</td><td>0.40</td><td>0.40</td><td>0.42</td><td>0.45</td><td>0.41</td></tr><tr><td rowspan="3">KNN</td><td>DNN</td><td>0.36</td><td>0.37</td><td>0.39</td><td>0.41</td><td>0.44</td><td>0.39</td></tr><tr><td>RF</td><td>0.22</td><td>0.23</td><td>0.24</td><td>0.26</td><td>0.28</td><td>0.25</td></tr><tr><td>SVM</td><td>0.36</td><td>0.37</td><td>0.38</td><td>0.40</td><td>0.44</td><td>0.39</td></tr><tr><td rowspan="3">MICE</td><td>DNN</td><td>0.37</td><td>0.39</td><td>0.41</td><td>0.44</td><td>0.49</td><td>0.42</td></tr><tr><td>RF</td><td>0.23</td><td>0.24</td><td>0.26</td><td>0.27</td><td>0.30</td><td>0.26</td></tr><tr><td>SVM</td><td>0.37</td><td>0.39</td><td>0.41</td><td>0.44</td><td>0.51</td><td>0.42</td></tr><tr><td rowspan="3">RF</td><td>DNN</td><td>0.35</td><td>0.36</td><td>0.39</td><td>0.41</td><td>0.43</td><td>0.38</td></tr><tr><td>RF</td><td>0.22</td><td>0.23</td><td>0.24</td><td>0.26</td><td>0.28</td><td>0.25</td></tr><tr><td>SVM</td><td>0.36</td><td>0.37</td><td>0.38</td><td>0.40</td><td>0.42</td><td>0.39</td></tr></table>

# Results from a combination of missing value imputation and feature normalization

The AUC rates and type II errors of the RF classifier obtained by combining feature normalisation with different imputation methods are shown in Figures 5 and 6, respectively. On average, the different imputation methods demonstrate similar performance in terms of the AUC rates, with no significant level of difference in performance. That is, there is only about a 0.01 (i.e., $1 \%$ ) performance difference between these imputation methods no matter what the missing rates are, except for DNN with a $5 0 \%$ missing rate. Specifically, DNN shows the highest level of performance degradation for missing rates from $1 0 \%$ to $5 0 \%$ i.e., 0.85 to 0.76. The AUC rate results indicate that performing feature normalisation after missing value imputation can reduce the performance differences between different imputation methods.

![](images/0d5c25b0fe32a9bc18f16519383b9507344072dfaa4f8c6ad77f0810ae2b232c.jpg)  
Figure 5. AUC rates for the RF classifier obtained by combining feature normalization with different imputation algorithms.

![](images/71048c14ea206b85ab9003ec966b784cb73a46d828331c79712d6a0fcebf8096.jpg)  
Figure 6. Type II error for the RF classifier obtained by combining feature normalization with different imputation algorithms.

Table 6. AUC rates for the DNN, RF, and SVM classifiers.   

<table><tr><td rowspan="2">Imputation methods</td><td rowspan="2">Classifiers</td><td colspan="5">Missing rates</td><td rowspan="2">Average</td></tr><tr><td>10%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td rowspan="3">Baseline</td><td>DNN</td><td>0.82</td><td>0.81</td><td>0.80</td><td>0.78</td><td>0.76</td><td>0.79</td></tr><tr><td>RF</td><td>0.84</td><td>0.82</td><td>0.81</td><td>0.80</td><td>0.77</td><td>0.81</td></tr><tr><td>SVM</td><td>0.74</td><td>0.73</td><td>0.72</td><td>0.70</td><td>0.68</td><td>0.71</td></tr><tr><td rowspan="3">DNN</td><td>DNN</td><td>0.84</td><td>0.82</td><td>0.80</td><td>0.78</td><td>0.75</td><td>0.80</td></tr><tr><td>RF</td><td>0.85</td><td>0.83</td><td>0.82</td><td>0.80</td><td>0.76</td><td>0.81</td></tr><tr><td>SVM</td><td>0.78</td><td>0.77</td><td>0.75</td><td>0.73</td><td>0.70</td><td>0.74</td></tr><tr><td rowspan="3">KNN</td><td>DNN</td><td>0.83</td><td>0.82</td><td>0.81</td><td>0.80</td><td>0.78</td><td>0.81</td></tr><tr><td>RF</td><td>0.84</td><td>0.83</td><td>0.81</td><td>0.80</td><td>0.78</td><td>0.81</td></tr><tr><td>SVM</td><td>0.76</td><td>0.76</td><td>0.74</td><td>0.73</td><td>0.71</td><td>0.74</td></tr><tr><td rowspan="3">MICE</td><td>DNN</td><td>0.83</td><td>0.82</td><td>0.80</td><td>0.77</td><td>0.74</td><td>0.79</td></tr><tr><td>RF</td><td>0.85</td><td>0.84</td><td>0.82</td><td>0.80</td><td>0.77</td><td>0.81</td></tr><tr><td>SVM</td><td>0.77</td><td>0.76</td><td>0.74</td><td>0.73</td><td>0.70</td><td>0.74</td></tr><tr><td rowspan="3">RF</td><td>DNN</td><td>0.84</td><td>0.83</td><td>0.82</td><td>0.81</td><td>0.79</td><td>0.82</td></tr><tr><td>RF</td><td>0.84</td><td>0.83</td><td>0.82</td><td>0.80</td><td>0.78</td><td>0.81</td></tr><tr><td>SVM</td><td>0.77</td><td>0.76</td><td>0.75</td><td>0.74</td><td>0.72</td><td>0.75</td></tr></table>

Table 7. Type II errors for the DNN, RF, and SVM classifiers.   

<table><tr><td colspan="2">Imputation</td><td colspan="5">Missing rates</td><td rowspan="2">Average</td></tr><tr><td>methods</td><td>Classifiers</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td rowspan="3">Baseline</td><td>DNN</td><td>0.24</td><td>0.25</td><td>0.27</td><td>0.32</td><td>0.36</td><td>0.29</td></tr><tr><td>RF</td><td>0.22</td><td>0.24</td><td>0.25</td><td>0.27</td><td>0.29</td><td>0.25</td></tr><tr><td>SVM</td><td>0.27</td><td>0.28</td><td>0.32</td><td>0.34</td><td>0.40</td><td>0.32</td></tr><tr><td rowspan="3">DNN</td><td>DNN</td><td>0.24</td><td>0.25</td><td>0.30</td><td>0.33</td><td>0.37</td><td>0.30</td></tr><tr><td>RF</td><td>0.22</td><td>0.24</td><td>0.26</td><td>0.28</td><td>0.30</td><td>0.26</td></tr><tr><td>SVM</td><td>0.25</td><td>0.27</td><td>0.31</td><td>0.33</td><td>0.37</td><td>0.31</td></tr><tr><td rowspan="3">KNN</td><td>DNN</td><td>0.22</td><td>0.24</td><td>0.25</td><td>0.28</td><td>0.31</td><td>0.26</td></tr><tr><td>RF</td><td>0.22</td><td>0.23</td><td>0.25</td><td>0.27</td><td>0.29</td><td>0.25</td></tr><tr><td>SVM</td><td>0.27</td><td>0.28</td><td>0.30</td><td>0.33</td><td>0.37</td><td>0.31</td></tr><tr><td rowspan="3">MICE</td><td>DNN</td><td>0.25</td><td>0.28</td><td>0.31</td><td>0.34</td><td>0.42</td><td>0.32</td></tr><tr><td>RF</td><td>0.23</td><td>0.25</td><td>0.27</td><td>0.30</td><td>0.35</td><td>0.28</td></tr><tr><td>SVM</td><td>0.28</td><td>0.30</td><td>0.32</td><td>0.34</td><td>0.42</td><td>0.33</td></tr><tr><td rowspan="3">RF</td><td>DNN</td><td>0.22</td><td>0.24</td><td>0.25</td><td>0.28</td><td>0.31</td><td>0.26</td></tr><tr><td>RF</td><td>0.22</td><td>0.23</td><td>0.25</td><td>0.27</td><td>0.29</td><td>0.25</td></tr><tr><td>SVM</td><td>0.26</td><td>0.28</td><td>0.30</td><td>0.32</td><td>0.36</td><td>0.30</td></tr></table>

On the other hand, for the type II errors, the baseline, KNN, and RF imputation methods perform similarly for missing rates from $10 \%$ to $5 0 \%$ . However, the RF classifier does not provide good performance with DNN and MICE when the missing rates increase. Overall, these results indicate that the baseline, KNN, and RF methods all produce relatively stable imputation results for different missing rates. On the other hand, DNN and MICE are not recommended for missing value imputation in bankruptcy prediction and credit scoring datasets.

![](images/cf7f65824dde79b41d4947ff8e37a5e40ed83e4ff2672826b9f66f61d5ba48bd.jpg)

Figure 7. Average AUC rates obtained by imputation alone and by combining imputation and normalization.

![](images/b384e0d40d55efe6180c2bb5b0079ceab6bd7559a250909fa32d893c65e68ada.jpg)  
(a) DNN classifier

![](images/54674a9f526c33471aeb7895a135cd49f272c8e7108cc727b0e35d17d1ed8116.jpg)  
(b) RF classifier

![](images/b06c8fbd16d48bf5aa819f327da3d2eb4f05750e1bb9466532db224cf76d9930.jpg)  
Figure 8. Average type II errors from imputation alone and by combining imputation and normalization.

For comparison, Tables 6 and 7 show the average performance for three different classifiers, DNN, RF, and SVM, based on the combination of imputation and normalisation results. When the missing rates are relatively low, i.e., $1 0 \%$ to $3 0 \%$ , the RF classifier provides the highest AUCE rates when MICE is combined with feature normalisation, whereas normalising the RF imputation results allows the DNN classifier to perform the best when the missing rates are $40 \%$ to $50 \%$ . On average, missing value imputation by RF in association with the DNN classifier performs the best.

On the other hand, the type II error results obtained with the RF classifier after normalising the imputation result are similar for each imputation algorithm for missing rates from $1 0 \%$ to $5 0 \%$ missing, except with the MICE imputation algorithm.

# Further comparisons

In order to find out whether performing feature normalisation can affect the imputation result, the performance differences between imputation alone and combining imputation and normalisation are examined. The average AUC rates and type II errors obtained with different imputation algorithms and classifiers are shown in Figures 7 and 8, respectively. Note that ‘FN’ means feature normalisation.

As can be seen, for both AUC rates and type II errors, a significant improvement in improvement can be attained by performing feature normalisation after imputation when the DNN and SVM classifiers are used $( p < 0 . 0 5 )$ . On the other hand, the combination of imputation and normalisation does not necessarily make the RF classifier perform better than employing imputation alone. This indicates that the classifier performance is heavily dependent on the input feature representation. In particular, the best performance can be obtained by RF-FN in combination with the DNN classifier and by KNN and RF with the RF classifier for the AUC rates (i.e., 0.82), and the baseline/baseline-FN, KNN/KNN-FN, and RF/RF-FN with the RF classifier (i.e., 0.25). In short, if the classification technique is chosen carefully, there is no need to consider the feature normalisation step when missing value imputation is completed.

# Conclusion

In this paper, we focus on missing value imputation for financial distress prediction datasets containing different proportions of missing attribute values. More specifically, comparison of the performance of five well-known missing value imputation algorithms, in terms of the AUC rates and type II errors, is made, including the mean/mode, DNN, KNN, MICE, and RF algorithms. The experimental results show that when the missing rates are below $3 0 \%$ , these imputation algorithms perform similarly. However, when the missing rates increase to $40 \%$ and $5 0 \%$ RF performs the best. Moreover, another key factor affecting the final prediction performance is the choice of classification technique. We found that the RF classifier significantly outperforms the DNN and SVM classifiers.

We further examined the effects of normalising the feature values, including the imputed features, on the final prediction performance of the different classifiers. The results show that performing feature normalisation after missing value imputation can reduce the differences in performance between the different imputation methods in terms of the AUC rates. However, for type II errors, the mean/model, KNN, and RF perform better than DNN or MICE.

There is a significant improvement in prediction performance when the DNN and SVM classifiers are constructed, when feature normalisation is employed after missing value imputation. However, performing feature normalisation does not have a positive effect on the RF classifier. In short, the highest AUC rates and lowest type II errors are provided by missing value imputation by RF and the constructed RF classifier.

There are several data level issues that can be considered in the future. First, since many of the input feature values are continuous, data discretisation, aimed at transforming continuous values into discrete ones, can be employed, and combined with the missing value imputation step. Second, as many of the financial distress prediction datasets are class imbalanced, related under- and over-sampling techniques can be performed to balance the datasets. It is worth investigating the order of operations needed when combining the data re-sampling step and missing value imputation. Third, the usefulness of feature selection, an important data preprocessing step in data mining, in improving prediction model performance has been demonstrated. It is important to examine the effect of feature selection on missing value imputation for financial distress prediction.

# Notes

1. https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction.   
2. https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction.   
3. https://archive.ics.uci.edu/ml/datasets/Statlog+(Australian+Credit+Approval.   
4. https://archive.ics.uci.edu/ml/datasets/Statlog+(German $^ +$ Credit+Data.   
5. https://archive.ics.uci.edu/ml/datasets/Credit+Approval.   
6. http://www.kaggle.com/c/GiveMeSomeCredit.   
7. http://sede.neurotech.com.br/PAKDD2010/.   
8. The computing environment is based on AMD Ryzen™ 5 2600 CPU@3.40 GHz with 16GB of memory.

# Disclosure statement

No potential conflict of interest was reported by the author(s).

# References

Azur, M. J., Stuart, E. A., Frangakis, C., & Leaf, P. J. (2011). Multiple imputation by chained equations: What is it and how does it work? International Journal of Methods in Psychiatric Research, 20(1), 40–49. https://doi.org/10.1002/mpr.329   
Cheng, C. -H., Chan, C. -P., & Sheu, Y. -J. (2019). A novel purity-based k nearest neighbors imputation method and its application in financial distress prediction. Engineering Applications of Artificial Intelligence, 81, 283–299. https://doi. org/10.1016/j.engappai.2019.03.003   
Fawcett, T. (2006). An introduction to ROC analysis. Pattern recognition letters, 27(8), 861–874. https://doi.org/10.1016/j. patrec.2005.10.010   
Feng, M., Shaonan, T., Chihoon, L., & Ling, M. (2019). Deep learning models for bankruptcy prediction using textual disclosures. European Journal of Operational Research, 274(2), 743–758. https://doi.org/10.1016/j.ejor.2018.10.024   
Garcia Laencina, P. J., Sancho-Gomez, J. -L., & Figueiras-Vidal, A. R. (2010). Pattern classification with missing data: A review. Neural Computing & Applications, 19, 263–282. https://doi.org/10.1007/s00521-009-0295-6   
Han, J., Kamber, M., & Pei, J. (2011). Data mining: concepts and techniques $( 3 ^ { r d }$ ed.). Morgan Kaufmann.   
Herasymovych, M., Marka, K., & Lukason, O. (2019). Using reinforcement learning to optimize the acceptance threshold of a credit scoring model. Applied Soft Computing, 84, 105697. https://doi.org/10.1016/j.asoc.2019.105697   
Lan, Q., Xu, X., Ma, H., & Li, G. (2020). Multivariable data imputation for the analysis of incomplete credit data. Expert Systems with Applications, 141, 112926. https://doi.org/10.1016/j.eswa.2019.112926   
Liang, D., Tsai, C. -F., Dai, A. -J., & Eberle, W. (2018). A novel classifier ensemble approach for financial distress prediction. Knowledge and Information Systems, 54(2), 437–462. https://doi.org/10.1007/s10115-017-1061-1   
Liang, D., Tsai, C. -F., Lu, H. -Y., & Chang, L. -S. (2020). Combining corporate governance indicators with stacking ensembles for financial distress prediction. Journal of Business Research, 120, 137–146. https://doi.org/10.1016/j. jbusres.2020.07.052   
Lin, W. -Y., Hu, Y. -H., & Tsai, C. -F. (2011). Machine learning in financial crisis prediction: A survey. IEEE Transactions on Systems, Man, and Cybernetics, Part C: Applications and Reviews, 42(4), 421–436. https://doi.org/10.1109/TSMCC.2011. 2170420   
Lin, W. -C., & Tsai, C. -F. (2020). Missing value imputation: A review and analysis of the literature (2006–2017). Artificial Intelligence Review, 53(2), 1487–1509. https://doi.org/10.1007/s10462-019-09709-4   
Little, R. J., & Rubin, D. B. (2002). Statistical analysis with missing data. $( 2 ^ { \mathsf { n d } }$ ed.). John Wiley & Sons, Inc.   
Louzada, F., Ara, A., & Fernandes, G. B. (2016). Classification methods applied to credit scoring: Systematic review and overall comparison. Surveys in Operations Research and Management Science, 21(2), 117–134. https://doi.org/10.1016/ j.sorms.2016.10.001   
Olson, D. L., Delen, D., & Meng, Y. (2012). Comparative analysis of data mining methods for bankruptcy prediction. Decision Support Systems, 52(2), 464–473. https://doi.org/10.1016/j.dss.2011.10.007   
Pereira, R. C., Santos, M. S., Rodrigues, P. P., & Abreu, P. H. (2020). Reviewing autoencoders for missing data imputation technical trends, applications and outcomes. The Journal of Artificial Intelligence Research, 69, 1255–1285. https://doi. org/10.1613/jair.1.12312   
Plawiak, P., Abdar, M., & Acharya, U. R. (2019). Application of new deep genetic cascade ensemble of SVM classifiers to predict the Australian credit scoring. Applied Soft Computing, 84, 105740. https://doi.org/10.1016/j.asoc.2019. 105740   
Qu, Y., Quan, P., Lei, M., & Shi, Y. (2019). Review of bankruptcy prediction using machine learning and deep learning techniques. Procedia computer science, 162, 895–899. https://doi.org/10.1016/j.procs.2019.12.065   
Shen, F., Liu, Y., Wang, R., & Zhou, W. (2020). A dynamic financial distress forecast model with multiple forecast results under unbalanced data environment. Knowledge-Based Systems, 192, 105365. https://doi.org/10.1016/j.knosys.2019. 105365   
Shen, F., Zhao, X., Kou, G., & Alsaadi, F. E. (2021). A new deep learning ensemble credit risk evaluation model with an improved synthetic minority oversampling technique. Applied Soft Computing, 98, 106852. https://doi.org/10.1016/j. asoc.2020.106852   
Singh, D., & Singh, B. (2020). Investigating the impact of data normalization on classification performance. Applied Soft Computing, 97(part B), 105524. https://doi.org/10.1016/j.asoc.2019.105524   
Sivasankar, E., Selvi, C., & Mahalakshmi, S. (2020). Rough set-based feature selection for credit risk prediction using weight-adjusted boosting ensemble method. Soft Computing, 24, 3975–3988. https://doi.org/10.1007/s00500-019- 04167-0   
Smiti, S., & Soui, M. (2020). Bankruptcy prediction using deep learning approach based on borderline SMOTE. Information Systems Frontier, 22(5), 1067–1083. https://doi.org/10.1007/s10796-020-10031-6   
Stekhoven, D. J., & Buhlmann, P. (2012). MissForest – non-parametric missing value imputation for mixed-type data. Bioinformatics, 28(1), 112–118. https://doi.org/10.1093/bioinformatics/btr597   
Tsai, C. -F., Sue, K. -L., Hu, Y. -H., & Chiu, A. (2021). Combining feature selection, instance selection, and classification techniques for improved financial distress prediction. Journal of Business Research, 130, 200–209. https://doi.org/10. 1016/j.jbusres.2021.03.018   
Veganzones, D., & Severin, E. (2018). An investigation of bankruptcy prediction in imbalanced datasets. Decision Support Systems, 112, 111–124. https://doi.org/10.1016/j.dss.2018.06.011   
Zhou, L., & Lai, K. K. (2017). AdaBoost models for corporate bankruptcy prediction with missing data. Computational Economics, 50, 69–94. https://doi.org/10.1007/s10614-016-9581-4