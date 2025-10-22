# A Fast Non-Linear Coupled Tensor Completion Algorithm for Financial Data Integration and Imputation

Dan Zhou dz239@njit.edu New Jersey Institute of technology USA

Ajim Uddin au76@njit.edu New Jersey Institute of technology USA

Zuofeng Shang zuofeng.shang@njit.edu New Jersey Institute of technology USA

Cheickna Sylla cheickna.sylla@njit.edu New Jersey Institute of technology USA

Xinyuan Tao xinyuan.tao@njit.edu New Jersey Institute of technology USA

Dantong Yu dtyu@njit.edu New Jersey Institute of technology USA

# ABSTRACT

Missing data imputation is crucial in finance to ensure accurate fnancial analysis,risk management, investment strategies,and other financial applications.Recently,tensor factorization and completion have gained momentum in many finance data imputation applications, primarily due to recent breakthroughs in applying deep neural networks for nonlinear tensor analysis.However,one limitation of these approaches is that they are prone to overfitting sparse tensors that contain only a small number of observations.This paper focuses on learning highly reliable embedding for the tensor imputation problem and applies orthogonal regularizations for tensor factorization, reconstruction,and completion.The proposed neural network architecture for sparse tensors,called “RegTensor", includes multiple components: an embedding learning module for each tensor order, MLP (multilayer perception) to model nonlinear interactions among embeddings,and a regularization module to minimize overfitting problems due to the large tensor rank. Our algorithm is efficient in factorizing both single and multiple tensors (coupled tensor factorization) without incurring high training and optimization costs.We have applied this algorithm in a variety of practical scenarios,including the imputation of bond characteristics and financial analyst EPS forecast data.Experimental results demonstrate its superiority with significant performance improvements: $4 0 \% - 7 4 \%$ better than linear tensor completion models and $2 \% - 5 2 \%$ better than the state-of-the-art nonlinear models.

# CCS CONCEPTS

. Computing methodologies Factorization methods; Regularization; Learning latent representations; Neural networks;· Information systems Spatial-temporal systems.

# KEYWORDS

Sparse Tensor Completion, Coupled Tensor Decomposition, Nonlinear Tensor Factorization,FinTech.

# ACMReference Format:

Dan Zhou,Ajim Uddin, Zuofeng Shang, Cheickna Sylla, Xinyuan Tao, and Dantong Yu.2023.A Fast Non-Linear Coupled Tensor Completion Algorithm for Financial Data Integration and Imputation.In 4th ACM International Conference onAI inFinance (ICAIF'23),November27-29,2023, Brooklyn,NY,USA.ACM,New York,NY,USA,9 pages.https://doi.org/10. 1145/3604237.3626899

# 1INTRODUCTION

The issue of missing data is widespread in finance. Examples of this include cross-sectional asset pricing characteristics, company fundamentals,financial analyst earnings forecasts,and bond pricing. Historically, most studies exclude data instances with missing values or use some form of mean imputation [6,12,17]. These two approaches might introduce bias into the data.For example, excluding missing data may result in removing important instances from the data or reducing the portfolio size in the asset pricing study [8,34]. Mean imputation or class mean imputation assumes that the data are randomly missing and follow a normal distribution.However, financial data is not missing at random [8] and often violates the above-mentioned assumptions [4].Several contemporaneous studies focused on dealing with missing financial data [7-9,11,34,39] and demonstrate that the return distribution of fully observed panel differ significantly from only non-missing firms,and may change the findings of previous asset pricing studies. However,most of these approaches use traditional econometrics methods such as principal component component analysis [8, 9,39] or generalized method of moments [15] and require the data to follow either a multivariate normal distribution [9]or a fully observed pre-selected set of characteristics [15].

In this paper, we proposed a machine learning based generalized missing value imputation approach that can be applied in multiple financial data.We applied a novel nonlinear tensor completion framework to model both spatial and temporal dependencies from observed entries to impute the missing values.Tensor completion algorithms are mainly based on two representative low-rank tensor factorization models:CANDECOMP/PARAFAC(CP)[18] and Tucker [32].These approaches attempt to identify low-rank factor matrices using the observed entries and then reconstruct the target tensor based on these factors matrices.The low-rank tensor factorization essentially boils down to a two-step problem with two objectives:representation learning and subsequent multi-way relationship prediction. Over the years,a number of low-rank tensor completion methods have been proposed, including convex and scalable algorithms for tensor completion [1,16],and non-negative tensor completion [3o],and also neural network based nonlinear tensor completion [24,25].These existing algorithms suffer from two mutually exclusive problems in achieving two objectives.

First, linear algorithms of low-rank matrix factorization (e.g., [1,16])attain low-rank embeddings but fail to capture the nonlinear relationships that are common in real-world tensor applications.The lack of non-linear relationship modeling results in suboptimal performance in tensor completion and downstream machine learning predictions [14]. Second,algorithms with nonlinear reconstruction (e.g.,[24,25]) focus on nonlinear relationship learning by encoding a large fraction of information in kernel functions or neural network layers.When multi-way relationship learning becomes dominant,it absorbs the majority of the information signal and might leave little information but noises in embedding [20]. Figure 1 shows that the latent temporal factors learned from a financial dataset by CoSTCo [25] in Column (1b) appear to be sporadic compared to the other two factorizations that focus their learning attention on embeddings and simplify the second step of multiway relationship learning.When we apply the learned temporal embeddings to a forecast problem,the noise in embeddings is unavoidably passed to forecast results.To avoid this problem,we must introduce structural regularization into the learned embedding.In addition,regularization also helps to mitigate the complexity of learned embedding and prevent overfitting in the final objective of tensor algorithms.

In this paper, we design a regularized coupled tensor reconstruction (RegTensor) that combines embedding learning and neural network based prediction.RegTensor learns the distributed representations effectively for prediction and multi-way relationship tasks.Unlike existing nonlinear tensor completion methods,it avoids the difficult trade-off between the two tensor objectives and introduces structure and constraints in the embedding learning step.The resulted architecture has less reconstruction error and generates high-quality embedding matrix.

Another major problem with existing low-rank tensor completion algorithms is that when the percentage of unobserved elements increases, the accuracy of low-rank tensor factorization worsens [27]. Fortunately, we are no longer constrained by just one data source in the era of big data.Now we can easily augment our target tensor with related heterogeneous datasets.Recently,a series of works [3,23] have shown that using auxiliary information from the secondary tensors significantly enhances tensor completion.

Existing techniques for introducing auxiliary information involve coupled matrix and tensor factorization (CMTF)[13, 27] or using regularization function [5].We extend our RegTensor to facilitate the factorization of two or more sparse tensors.In RegTensor, we propose an innovative coupled tensor completion algorithm usinga modified objective function for element-wise reconstruction and SGD optimization.We evaluate our proposed model in four financial data with missing ovservations: bond characteristics, bond-fund holding data,analyst EPS forecast,and firm fundamentals.

The experiment results reveal the consistency and reliability of our model. The main contributions of our paper are as follows.

·We propose a novel neural networks based tensor factorization method for financial data imputation. The model can efficiently learn embedding matrix and nonlinear interaction between the embedded vectors to sucessfully impute missing financail data.   
· The SGD based nonlinear coupled tensor completion algorithm thatis fast and scalable.   
· We introduce regularization to reduce the redundancy in learned embeddings,mitigate overfitting,and learn highquality embedding for downstream machine-learning tasks.   
· We apply our proposed algorithm to impute missing data in four financial data sets and demonstrate that our model is superior to the current state-of-the-art techniques.

# 2RELATED WORK

# 2.1 Imputing Missing Financial Data

Over the years,cross-sectional means of imputing missing asset pricing data were prominent [6,12,17].However, empirical studies show that the mean or the cross-sectional mean may produce a biased data representation for model estimation[4].Using advanced methods to impute missing financial data has recently gained some traction from the academic community.For example,in [8, 39], the authors use latent factors to recover missing characteristics in the cross-sectional asset pricing characteristics.A generalized method of moment-based imputation is proposed in [15].These traditional econometrics approaches require fully observed preselected characteristics [15] or the data to followa multivariate normal distribution [11].

Only a handful of papers used neural networks and machine learning to impute missing financial data.In [7],the authors used a popular temporal attention mechanism in natural language processing (NLP) to impute cross-sectional asset pricing characteristics. The machine learning-based model is also used to impute missing values in the financial analyst's earnings forecast [33-35].In [34] theauthors proposed a coupled matrix factorization to first impute missing values and then used the imputed data to estimate firms' earnings.Later,in [33,35],the authors show that tensor-based factorization is more effective than matrix factorization when it comes to filling in missing data in analyst earning forecasts.Different from these works,our paper proposed a machine learning based generalized framework that can impute the missing financial attributes associated with cross-sectional characteristics or financial analyst earning forecast data.

# 2.2 Tensor Factorization for Imputing Missing Data

In recent years,a number of low-rank tensor completion algorithms [1,16,26,28,30,31] are developed based on classical CP[10,18] and Tucker factorizations [32].The low-rank approach assumes that a (in)complete tensor can be represented and reconstructed using low-rank factor matrices and multi-linear multiplications. However, the reconstruction is not always precise,instead involving an approximate solution with some noise and data loss.In realworld data applications,the imprecise problem aggravates as it fails to capture frequent nonlinear interactions.Nonlinear tensor factorization techniques have recently gained attention, including kernel-based tensor factorization to capture nonlinear relationships in the real world [14,19]. These techniques require full observation of the target tensor and are,therefore,unsuitable for sparse tensor completion without customization.

![](images/296edabf7228a387197521750f2ff49bb0d414251508a0e4970ac63c31dfcb81.jpg)  
Figure 1: The embedding matrix forquarters and firms learned from (a)CPbased factorization[1],(b)CoSTCo[25],and (c) RegTensor(Ours).Time-seriesdatausuallyavetrend,seasoal,andcyclic paters.Compared toCoSTCo,CPandRegesor cancapture these trend in theirfactors.The latent factorslearnedforquartersbyRegTensorcapture trends in factors 1,2,4, cycles in factors 6 and 7,and seasonal patterns in factors 2, 4, 9,and 10.

Neural network-based tensor factorization captures non-linear interactions and can be used for sparse tensor completion.Some recent efforts [24,38] attempt to replace multi-linear operation with multi-layer perceptions (MLP).A convolution neural networkbased architecture is offered by [25] to factorize the sparse tensor. These works above try to learn the low-ranking representation of single tensor and do not consider any auxiliary information to improve the factor matrices.

In fact,using auxiliary information significantly improves the accuracy of tensor factorization and completion [27].Coupled matrix-tensor factorization (CMTF) is the most prominent approach [3,5,13] to integrate multiple tensors for the imputation. These approaches factorize a higher-order tensor with a related matrix in a coupled fashion. Unlike the CMTF method,our approach is a coupled tensor factorization for sparse data in which all datasets come from higher-order tensors.In recent years,various coupled tensor factorization models have been proposed [22,37]. These models,however,are not suitable for sparse data and necessitate a full observation of all input tensors.In contrast,our RegTensor relaxes the requirements for complete observations and captures nonlinearity from both target tensors.

# 3METHODOLOGY

In this section,we first elaborate on the nonlinear tensor factorization model in a single tensor and describe the RegTensor on how to learn the embedding matrix and multi-way interaction among the embedding in the same set of networks (Section 3.1).We extend the factorization of the tensor to handle heterogeneous data sources (Section 3.2) and design a scalable algorithm for coupled tensor learning in Figure 2 (Section 3.3).Then we apply regularization to nonlinear embedding learning (Section 3.4).

# 3.1Non-Linear Model for Tensor Factorization

CANDECOMP/PARAFAC algorithm (CP) is frequently used for tensor decomposition.For simplicity,assume a third-order tensor $\chi \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } \times \bar { d } _ { 3 } }$ . Given a rank $r _ { : }$ CP factorizes it into three-factor matrices $U \in \mathbb { R } ^ { d _ { 1 } \times r } , V \in \mathbb { R } ^ { d _ { 2 } \times r }$ and $W \in \mathbb { R } ^ { d _ { 3 } \times r }$ . The predicted tensor entry is:

$$
\hat { x } _ { i , j , k } = \sum _ { s = 1 } ^ { r } U _ { i s } V _ { j s } W _ { k s }
$$

The dot product among three vectors has limited capacity to model complex relations.The key idea of non-linear RegTensor is replacing the multilinear operations in standard element-wise CP decomposition with multi-layer perceptrons (MLP) [29].The modified RegTensor consists of an embedding module and a non-linear tensor reconstruction module.The embedding module first learns three latent feature matrices $U \in \mathbb { R } ^ { d _ { 1 } \times r }$ $V \in \mathbb { R } ^ { d _ { 2 } \times r }$ ， $W \in \mathbb { R } ^ { d _ { 3 } \times r }$ for each dimension of $\chi$ .Then the embedding model takes the input index of a tensor entry $( i , j , k )$ and generates three embedding vectors $U _ { i , : }$ $, V _ { j , : } , W _ { k , }$ : where $U _ { i , \astrosun }$ : denotes the $i \cdot$ -th row of the factor matrix $U$ and $1 \leq i \leq d _ { 1 }$ .The same notation is applied to $V _ { j , \astrosun }$ : and $W _ { k , : }$ .Here the nonlinear tensor reconstruction module consists of MLP,defined as follows:

$$
\hat { x } _ { i j k } = M L P ( [ U _ { i , : } , V _ { j , : } , W _ { k , : } ] )
$$

where $\hat { x } _ { i j k }$ represents the predicted tensor entry and $[ \cdot , \cdot , \cdot ]$ denotes the concatenation of input vectors.

# 3.2 Coupled Tensor Decomposition for Embedding Learning

When we apply the tensor completion algorithm on some sparse dataset, for example, the EPS dataset, the performance on some firms is not better than that of a simple consensus prediction using the average value of the available analysts’ forecasts.Nearly all tensor completion algorithms,including our proposed RegTensor, often suffer the cold start and the extreme low signal to noise ratio (SNR)[2].The time and firm latent factors learned from the analysts' tensor are less informative because of insufficient signals and even missing critical data in the incomplete analyst forecast dataset.We must introduce the new dataset that is synergistic to the tensor to be imputed.The firm fundamentals share the same time and firm dimensions with EPS and provides complementary information about any firm in the tensor, including key performance indices that are used to minimize uncertainty and noises and enhance the completion result.

![](images/8213e541471778e2714de35daddc4fcd5dfe7c7c409c04cc8106f797c2c5d3a4.jpg)  
Figure 2: Coupled Tensor Decomposition and Completion

The coupled matrix and tensor factorization algorithm (CMF) attempts to enforce the same time factor and firm factor matrices during factorization [2].The information propagates from the auxiliary tensor (firm accounting fundamentals) to the target tensor of the forecast dataset during data imputation by coupling the firm and time factors.Figure 2 shows that two third-order tensors $\chi$ and $_ y$ share two factor matrix $U$ and $V$ .The objective function for coupled tensor is to minimize the mean square error of two tensor factorizations by co-optimizing the following equation for CP decomposition:

$$
\begin{array} { l l l } { { \mathcal { L } } } & { { = } } & { { \displaystyle \left| { \boldsymbol { \mathcal { X } } } - [ [ \Lambda _ { 1 } ; U , V , T ] ] \right| _ { F } ^ { 2 } + \lambda \big | { \boldsymbol { \mathcal { Y } } } - [ [ \Lambda _ { 2 } ; U , V , W ] ] \big | _ { F } ^ { 2 } } } \\ { { } } & { { = } } & { { \displaystyle \left| { \boldsymbol { \mathcal { X } } } - \sum _ { i = 1 } ^ { r } \Lambda _ { 1 i } \cdot U _ { : i } \otimes V _ { : i } \otimes T _ { : i } \right| _ { F } ^ { 2 } } } \\ { { } } & { { } } & { { \displaystyle + \lambda \big | { \boldsymbol { \mathcal { Y } } } - \sum _ { i = 1 } ^ { r } \Lambda _ { 2 i } \cdot U _ { : i } \otimes V _ { : i } \otimes W _ { : i } \big | _ { F } ^ { 2 } } } \end{array}
$$

Here $\lambda$ is the hyper-parameter to adjust the relative importance between the two coupled tensors. $\otimes$ represents vector outer product. We choose Frobenius Norm $\| \cdot \| _ { F } ^ { 2 }$ to calculate the residual sum of square for tensor completion. $\Lambda _ { 1 : 2 }$ are the weights to the columns of $U , V ,$ and $W$ In this paper, we set all weights in $\Lambda _ { 1 : 2 }$ to be one and allow matrix columns to take their length freely.Witha simple modification to replace the linear dot products among $U _ { i ; } V _ { j }$ ,and $W _ { k }$ ：in Eqn (3)and (4) with the MLP on the three vectors in Eqn (2), we can define the objective function of the MLP based non-linear tensor decomposition.During the training process,we minimize the mean square error of both the target tensor (the tensor we aim to impute) and the auxiliary tensor (the tensor offers complementary information that enhances the completion performance of the target tensor),adhering to the objective function defined in Eqn (3). Afterward, when making inferences,we focus solely on evaluating

Input :Target tensor $\boldsymbol { X } \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } \times d _ { 3 } }$ to be completed, auxiliary tensor $y \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } \times d _ { 4 } }$ , rank of tensor decomposition $r$ ,index set of observed entries $\Omega _ { X }$ in the tensor $\chi$ and $\Omega _ { J }$ in the tensor $_ y$ Output:Updated factor matrices $U , V , W , T$   
1 Initialize $U , V , W , T$   
2 repeat   
3 for $\alpha = \forall ( i , j , k ) \in \Omega _ { X } \cup \Omega _ { y }$ do //Forward Propagation   
4 Calculate loss LReg ; // Eqn (6) // Backward Propagation   
5 if $\alpha \in \Omega _ { \chi }$ then   
6 Update $U , V , T$ based on Eqn (5) using Loss Term Eqn (6).   
7 else   
8 Update all U,V,W using Loss Term $\lambda \mathcal { L } _ { R e g }$ in Eqn (6).

9 until the maximum number of epochs or early stop;

the tensor completion result of the target tensor and calculating the loss that is only caused by the target tensor.

# 3.3SGD-based Coupled Tensor Factorization and Integration

The coupled-matrix and tensor factorization (CMTF) algorithm can be simplified to minimize the mean square error in Eqn 3. One constraint is that CMTF requires complete tensors for factorization and suffers the slow convergence and long computation time for large tensors.In this paper, we adopt an element-wise approach to reconstruct tensors from the observation set $\Omega$ while ignoring the remaining missing entries.We revise the objective function and consider minimizing the reconstruction loss $\mathcal { L } _ { R e c }$ only incurred by observed elements:

$$
\begin{array} { c c l } { \mathcal { L } _ { R e c } } & { = } & { \displaystyle \sum _ { i , j , k } \mathbb { 1 } _ { \Omega _ { X } } ( i j k ) \big | X _ { i j k } - \sum _ { s = 1 } ^ { r } U _ { i s } V _ { j s } T _ { k s } \big | _ { F } ^ { 2 } + } \\ & & { \displaystyle \quad \lambda \mathbb { 1 } _ { \Omega _ { y } } ( i j k ) \big | y _ { i j k } - \sum _ { s = 1 } ^ { r } U _ { i s } V _ { j s } W _ { k s } ) \big | _ { F } ^ { 2 } } \end{array}
$$

where $\mathbb { 1 } _ { \Omega _ { X } } ( i j k )$ is an indicator function.It is straightforward to implement Eqn (4) with the Stochastic Gradient Descent (SGD): we first mix the training data from $\Omega _ { X }$ and $\Omega _ { y }$ ,randomly choose one mini-batch of training samples,and calculate the mean square error in Eqn (4).This mixing strategy intelligently employs indicator functions to allow any observation to be treated uniformly, thereby enabling parallel processing of the samples in the same mini-batch.It essentially implements a multi-task learning task and is highly flexible to allow more than one tensor to be factorized simultaneously.

We calculate the gradient of $\mathcal { L } _ { R e c }$ in Eqn (4) concerning the factor matrices $( U , V , T , W )$ for stochastic gradient descent update. Consider that we pick one index among tensor $\chi$ index $( i , j , k ) \in$ $\Omega _ { X }$ ,we calculate the corresponding gradient of $\mathcal { L } _ { R e c }$ to the factors and update the parameters as follows:

$$
U _ { i , : } \longleftarrow U _ { i , : } - \eta ( \sum _ { s = 1 } ^ { r } U _ { i s } V _ { j s } T _ { k s } - \chi _ { i j k } ) ( V _ { j , : } \odot T _ { k , : } )
$$

Here $\odot$ is the Hadamard product of two vectors.The gradients on other factor matrices $V , W , T$ have the identical formula to Eqn (5).The network architecture of RegTensor and the SGD-based optimization are shown in Algorithm 1.

# 3.4Regularization to Enforce Structure on Factor Matrices

Simple concatenation among multiple tensors allows useful information to propagate from one tensor to another.However, the concatenation might also introduce noises,inconsistency,and redundancy with information. Similar to other data-driven models, tensor completion cannot eliminate all noises from the information. Explicit regularization is required to impose penalties on unwanted information (noise),redundancy,and discrepancies.One of the major goals for learning high-quality low-rank representation is that the individual features of the embedding matrix should be as different as possible.The regularization in our proposed objective function imposes the orthogonality constraints on each tensor mode to eliminate the redundancy among learned features and associated undesired artifacts (high sensitivity and high variance) in the downstream learning tasks.When two features are orthogonal, they share no information,reduce covariance,and facilitate the reliable induction of models.The updated objective function with orthogonality constraint is defined as follows:

$$
\mathcal { L } _ { R e g } = \mathcal { L } _ { R e c } + \beta \| U ^ { \top } U \odot ( \mathbf { 1 1 } ^ { \top } - I ) \| _ { 2 , 2 }
$$

where $1 1 ^ { \top }$ is the matrix of all ones. The second regularization term in Eqn (6) is the $L _ { p , q }$ matrix norm. The $\ell _ { 2 , 2 }$ norm enforces the element-wise sparsity in a matrix and orthogonality $( U ; i \ \perp \ U _ { : , j }$ when $i \neq j )$ .The objective penalizes the inter-dependency (correlation or collinearity) among the latent features,i.e.,the column vectors $U _ { ; , i }$ and $U _ { : , j }$ in the time matrix $U$

# 4EXPERIMENT

We conducted two experiments on four datasets to evaluate our algorithm: 1) effciency in tensor completion in both time and accuracy compared to other state-of-the-art tensor completion techniques and 2) ability to factorize a sparse coupled tensor while learning a meaningful factor matrix. To alleviate the overfiting problem and boost the generalization performance, we apply orthogonal regularization to enforce the structure and constraints in learned embedding.

We compare our single tensor complation algorithm (RegTensor), orthogonal regularized algorithm (RegTensor- $\ell _ { 2 , 2 }$ ),and coupled tensor completion algorithm (RegTensor (Coupled)) with CPWOPT [1]-the benchmark low-rank sparse multi-linear tensor completion method and CoSTCo [25] -CNN based state-of-the-art nonlinear tensor completion method.We adopt three metrics:RMSE,MAE, and MAPE.

Table 1: Statistics of data   

<table><tr><td>Datasets</td><td>Shape</td><td>Observations</td></tr><tr><td>Bond Characteristics</td><td>(173,5753,29)</td><td>5,293,254</td></tr><tr><td>Bond-Fund holding</td><td>(58,5753,12608)</td><td>17,570,040</td></tr><tr><td>EPS</td><td>(32,175,173)</td><td>41,969</td></tr><tr><td>Fundamentals</td><td>(32, 175,19)</td><td>106,133</td></tr></table>

Table 2: Hyper-parameter settings   

<table><tr><td>Datasets</td><td>lr</td><td>epochs</td><td>batch size</td></tr><tr><td>Bond Characteristics</td><td>1e-2</td><td>2000</td><td>1024</td></tr><tr><td>EPS datasets</td><td>1e-4</td><td>500</td><td>128</td></tr></table>

# 4.1 Experiment Data

Bond Characteristics [month, bond, characteristics]: We obtain the monthly bond characteristics data from the corporate bond dataset in [21]1 that comprises 29 bond/firm variables.Due to infrequent trading in the corporate bond market, only $1 8 . 3 3 \%$ of bonds, on average,have trading transactions each month,i.e, $\sim 8 1 . 6 7 \%$ missing rate in the monthly trading data.Following the normalization method described in [21],we scale the bond characteristics to fit within the range of [-0.5,0.5].Our data samples span from July 2002 to December 2016,a total of 173 months.

Bond-Fund holding data [quarter, bond, fund]: The quarterly bond-fund holding data from eMAXX includes the end-of-quarter holding position for each bond in each fund during a given quarter. In the coupled tensor completion experiment, we merge quarterly bond-fund holding data with monthly bond characteristics and ensure that the months within a quarter share the same bond-fund investment data.

We also experiment with the financial analyst EPS forecast and firms'fundamental data.EPS forecast and fundamental data are available with paid subscriptions from the respective vendors (IBES, COMPUSTAT,and CRSP).The fundamental data capture the relationship among“quarter",“firm",“fundamental",while the EPS data pertain to the relationship among [quarter, firm,analyst].Table 1 lists the size and shape of the four data sets.

# 4.2Coupled Tensor Completion

In the coupled tensor completion experiments,we leverage two pairs of related data: one comprises bond characteristics and bondfund holding data,while the other pair involves EPS and Fundamentals.To handle the bond data,we designate the bond characteristics as the“target tensor"to be imputed and the bond fund holding data as the auxiliary tensor. These two tensors share common modes (month and bond) and are factorized together using the loss function defined in Eqn(4).Regarding the firm data,we use EPS as the target tensor and Fundamentals as the auxiliary tensor,where the shared modes are quarter and firm.To conduct the experiments, we split the available observations of Bond characteristics and EPS into 80/20 training and testing samples.To avoid overfiting,we set the early stopping patience parameter to ten epochs.For reference,

Table 3: Tensor Completion Results of Bond Characteristic:   

<table><tr><td></td><td>Metric</td><td></td><td colspan="3">RMSE</td><td colspan="4">MAE</td><td colspan="4">MAPE</td></tr><tr><td>Data</td><td>Model/Rank</td><td>10</td><td>20</td><td>50</td><td>100</td><td>10</td><td>20</td><td>50</td><td>100</td><td>10</td><td>20</td><td>50</td><td>100</td></tr><tr><td rowspan="5">Characeities</td><td>CPWOPT</td><td>0.3021</td><td>0.3369</td><td>0.2896</td><td>0.3943</td><td>0.2580</td><td>0.2757</td><td>0.2507</td><td>0.2887</td><td>96.5890</td><td>108.0795</td><td>90.1966</td><td>111.9994</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.7979</td><td>0</td><td>07</td><td>022</td><td></td><td>0.505</td><td>0.46</td><td>02405</td><td>864478</td><td>8804</td><td></td><td></td></tr><tr><td>RegTensor (Coupled)</td><td>0.1620</td><td>0.1530</td><td>0.1196</td><td>0.1005</td><td>0.1159</td><td>0.1079</td><td>0.0779</td><td>0.0632</td><td>53.3481</td><td>50.3601</td><td>38.4141</td><td>32.1510</td></tr></table>

Table 4: Tensor Completion Results of EPS   

<table><tr><td rowspan="2">Metric Data Model/Rank</td><td rowspan="2"></td><td colspan="4">RMSE</td><td colspan="4">MAE</td><td colspan="4">MAPE</td></tr><tr><td>10</td><td>20</td><td>30</td><td>40</td><td>10</td><td>20</td><td>30</td><td>40</td><td>10 20</td><td>30</td><td></td><td>40</td></tr><tr><td rowspan="5">EPS</td><td>CPWOPT</td><td>0.3865</td><td>0.3434</td><td>0.3352</td><td>0.4171</td><td>0.1135</td><td>0.1722</td><td>0.2274</td><td>0.2137</td><td>25.7778</td><td>36.2522</td><td>32.3325</td><td>39.8641</td></tr><tr><td>CoSTCo</td><td>0.2536</td><td>0.2364</td><td>0.2455</td><td>0.2311</td><td>0.1337</td><td>0.1338</td><td>0.1185</td><td>0.1096</td><td>29.7298</td><td>30.4989</td><td>26.4746</td><td>24.0901</td></tr><tr><td>RegTensor</td><td>0.2462</td><td>0.1945</td><td>0.1919</td><td>0.1921</td><td>0.1561</td><td>0.1167</td><td>0.1140</td><td>0.1139</td><td>35.5532</td><td>26.3954</td><td>25.5205</td><td>25.7024</td></tr><tr><td>RegTensor-l2,2</td><td>0.2481</td><td>0.1942</td><td>0.1856</td><td>0.1825</td><td>0.1416</td><td>0.1142</td><td>0.1024</td><td>0.0972</td><td>30.8139</td><td>25.4899</td><td>24.3845</td><td>22.5827</td></tr><tr><td>RegTensor (Coupled)</td><td>0.2432</td><td>0.1932</td><td>0.1623</td><td>0.1508</td><td>0.1559</td><td>0.1128</td><td>0.0924</td><td>0.0897</td><td>27.3265</td><td>21.8437</td><td>19.5397</td><td>20.5283</td></tr></table>

Table 5: Portfolios Performance Sorted on Imputed Bond Excess Returns   

<table><tr><td>model</td><td>Low</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>High</td><td>High-Low</td><td>tstatistics</td></tr><tr><td>CPWOPT</td><td>0.3124</td><td>0.2481</td><td>0.2668</td><td>0.2171</td><td>0.1661</td><td>0.2506</td><td>0.1960</td><td>0.2193</td><td>0.2087</td><td>0.1873</td><td>-0.1251</td><td>-1.33918</td></tr><tr><td>CoSTCO</td><td>0.2554</td><td>0.2000</td><td>0.1924</td><td>0.1747</td><td>0.2081</td><td>0.2376</td><td>0.2516</td><td>0.2840</td><td>0.2200</td><td>0.2456</td><td>-0.0098</td><td>-0.13478</td></tr><tr><td>RegTensor</td><td>-0.0213</td><td>0.0116</td><td>0.0162</td><td>0.2093</td><td>0.2572</td><td>0.2163</td><td>0.2774</td><td>0.3478</td><td>0.3973</td><td>0.2938</td><td>0.3151</td><td>3.24160</td></tr><tr><td>RegTensor-l2.2</td><td>-0.0336</td><td>0.0561</td><td>0.1501</td><td>0.1765</td><td>0.1811</td><td>0.2191</td><td>0.2560</td><td>0.3056</td><td>0.3226</td><td>0.4606</td><td>0.4943</td><td>4.70456</td></tr><tr><td>RegTensor (Coupled)</td><td>-0.1143</td><td>0.0180</td><td>0.1302</td><td>0.1889</td><td>0.1897</td><td>0.2729</td><td>0.2977</td><td>0.3968</td><td>0.4130</td><td>0.6012</td><td>0.7155</td><td>6.47158</td></tr></table>

teAll $5 \%$ level or better.

Table 2 provides an overview of the hyperparameters used in our experiments.

Coupled tensor factorization achieves much higher accuracy than single tensor factorization when imputing missing values.In Table 3,the results demonstrate that RegTensor (Coupled) outperforms two baselines: CPWOPT by a significant margin of $\geq 4 6 \%$ in reducing RMSE and CoSTCo by at least $4 0 \%$ in RMSE.Furthermore, it surpasses our RegTensor for single tensor completion byat least $1 8 \%$ in RMSE.The underwhelming performance of the single tensor completion baselines can be attributed to the limited trading frequency of corporate bonds issued by firms,which often results in incomplete and highly sparse bond characteristics data.As the percentage of unobserved elements increases, the accuracy of lowrank tensor factorization deteriorates.To address this challenge, RegTensor (Coupled) takes a proactive approach by integrating data from the related quarterly bond-fund holding dataset as auxiliary information. This allows RegTensor (Coupled) to capture underlying dependencies among the bonds and significantly enhance the accuracy of tensor completion for bond characteristic.

Table 4 presents the results of the EPS and Fundamentals data. Remarkably,RegTensor (coupled) outperforms CPWOPT by an impressive $5 2 \%$ in RMSE (when rank $= 3 0$ ,CPWOPTachieves its optimal performance) and also surpasses CoSTCo by $3 5 \%$ in RMSE (CoSTCo with rank $= 4 0$ achieves its best performance). This performance enhancement by RegTensor (coupled) reinforces the advantages of coupled tensor completion.Incorporating the second auxiliary tensor (Fundamentals) into our algorithm is crucial to guide the imputation process, especially to generate accurate embedding matrices for dimensions quarter and firm.By better learning these embedding matrices,the RegTensor (coupled) exhibits fewer completion errors than any other single tensor completion method.

![](images/dc072bf937c866d392b282ef937410ed5f172afd7a6fdfcecec7042cff301c89.jpg)  
Figure 3: RMSE of Bond Characteristics tensor completion under two varying hyper parameters: (a) the coefficient $\lambda$ adjusts the relative importance between the two coupled tensors and (b) the coefficient $\beta$ of the orthonormality regularization term.

# 4.3Sparse Single Tensor Completion

RegTensor applies orthogonal regularization to learn structured embedding and uses the non-linear module to capture complex interactions in single tensors and coupled tensors.

Table 3 presents that RegTensor surpasses all linear and nonlinear baseline methods in terms of the performance for the Bond

![](images/b0b7031b501d39f64933b41bae72052b6f83d5982bbbff7c921e8b6097ded177.jpg)  
Figure 4: Running time of different tensor completion algorithms at different ranks on two datasets: (a) Bond Characteristics data and (b) financial analyst EPS forecast data.

Characteristics data.Specifically,RegTensor- $\cdot \ell _ { 2 , 2 }$ shows notable results,achieving $2 7 \% - 4 5 \%$ lower RMSE compared to CPWOPT and $2 6 \% - 3 1 \%$ lower RMSE compared to CoSTCo. The better performance of RegTensor can be attributed to its ability to introduce non-linearity into the model. Bond characteristics data may not conform to linear relationships,and its underlying patterns might be obscured by complexities.RegTensor incorporates multi-layer perceptrons (MLPs),replacing the multilinear operations in standard element-wise CP decomposition.Including MLPs allows RegTensor to effectively capture the true underlying complex nonlinear patterns,enhancing representation learning and enabling better imputation of missing values in the bond characteristics data.

Notably,as the rank increases,we discover that the test performance of CPWOPT, CoSTCo,and RegTensor does not show significant improvement,and in some cases,it may even deteriorate due to the challenges posed by data sparsity and the potential overfitting problem.On the contrary,our RegTensor models with regularization modules $( \ell _ { 2 , 2 } )$ successfully circumvent these issues for all different ranks. Incorporating orthogonal regularization proves to be highly effective in mitigating overfitting by eliminating interdependency among different dimensions and promoting a more compact and meaningful representation. By enforcing orthogonality, the model becomes adept at focusing on the most relevant features while avoiding redundancies.Consequently, the model's sensitivity to minor fluctuations in the training data is reduced, leading to better generalization on unseen data and effectively combating overfiting problems.

Table 4 presents the results of the single tensor completion for the EPS forecast data from the financial analyst. Our RegTensor $\ell _ { 2 , 2 }$ （204 consistently outperforms CPWOPT on RMSE,achieving reductions of $3 6 \%$ $4 3 \%$ $4 5 \%$ and $5 6 \%$ for ranks 10, 20,30,and 40,respectively. Furthermore,RegTensor- $\ell _ { 2 , 2 }$ demonstrates better performance than CoSTCo on RMSE,with improvements of $2 \%$ $1 8 \%$ $2 4 \%$ ,and $2 1 \%$ for ranks 10,20,30,and 40,respectively.

# 4.4 Experiment on Regularization Term

We apply the coupled tensor decomposition algorithm to the bond characteristics data and investigate the influence of the regularization term on algorithm performance.We present the results in Figure 3.In this analysis, $\lambda$ controls the relative importance between the two coupled tensors in the coupled tensor decomposition, while $\beta$ governs the penalty for orthogonal regularization in each tensor model to eliminate redundancy between learned features.

Figure 3(a) illustrates the errors under the varying $\lambda$ while fixing the rank at 10o (where the model achieves the best performance in the test set for coupled tensor completion).Figure 3(b) depicts the RMSE values for varying $\beta$ while fixing the rank at 10 (When we set rank to 10,RegTensor achieves the best performance in the test set for single tensor completion).All two curves demonstrate that the model performance has three regions (overfitting, good-fitting, and underfitting).All models improve first when we increase the regularization strength,indicating that the models are overfitting the observations. Once all models reach the elbow point, they deteriorate as excessive regularization causes the underfitting problem. The coupled tensor coeficient λ guarantees that the propagation of information along the common tensor modes (month,bond) between the two coupled tensors (bond characteristics and bond-fund holding data) and the corresponding learned factor matrices must maintain the interconnected information between these coupled tensors.The orthogonal regularization also improves the model performance and indicates the importance of eliminating colinearity and covariance on the features of the embedding matrix.

# 4.5Portfolio Analysis

To delve deeper into the economic significance of RegTensor, we follow the common practices in finance literature [17,36] and conduct portfolio analysis.The underlying intuition is as follows: if a given model can perform accurate imputation, the bonds with high (low) imputed returns will have realized high (low) returns.Investment portfolio buying bonds with high imputed returns and selling bonds with low impute returns then generates positive returns,i.e., the higher portfolio returns of a model indicate better imputation of the model. One point to be aware is that imputation utilizes all time periods and therefore,this portfolio analysis is mainly to demonstrate the outperformance of our algorithm with economical significance,rather than proposing a practical investment strategy. To construct the portfolio,at the beginning of each month, we sort the bonds into ten groups from low to high based on the imputed returns in test set.The portfolio is buying (selling) bonds with highest (lowest) imputed returns and portfolio returns are the realized returns difference between the highest and lowest group.Results are presented in Table 5. The long-short portfolio returns (Column "High - Low") of three RegTensor algorithm are significantly positive.In contrast, two baselines fail to generate positive portfolio profits. Notably, our proposed RegTensor (Coupled) achieves the highest return of $0 . 7 1 5 5 \%$ with statistical significance at $5 \%$ ，Additionally, our model generates at least $0 . 7 2 5 4 \%$ higher portfolio monthly returns than the other baselines,i.e.,approximately annualized $8 . 7 0 4 8 \%$ higher returns,which is economically significant.

# 4.6 Running Time

RegTensor takes the indices of the observed values as input variables and only optimizes on these observed tensor elements; therefore,itis linear to the number of available observations rather than the tensor size of the target tensor.For the bond characteristics data, we keep the batch size fixed at 1024 and the learning rate at $1 e - 2$ for all models.We then compare the running times of our models and baselines across different ranks.On the EPS data,we maintain a batch size of 128 and a learning rate of $1 e - 4$ for comparison. The orthogonality is expensive in non-convex Lagrange optimization. SGD helps avoid the costly exact optimization and accelerates the process to get an approximate solution. Figure 4 shows the running time of each algorithm in each data set at different ranks.The time reported for each algorithm already incorporates the early stopping criteria and does not necessarily increase monotonically with the tensor rank.Therefore,sometimes with higher ranks,several algorithms converge faster than lowerranks.In the experiments on both datasets,our proposed algorithm consistently matches or outperforms the CoSTCo model in terms of runtime,except for the Coupled RegTensor that attempts to optimize multiple tensors simultaneously.Indeed, the RegTensor (coupled) needs extra computing time for completing the auxiliary tensors to augment the target tensor. However,it is still in the same complexity range as other low-rank tensor factorizations.The time complexity of RegTensor does not increase drastically with higher ranks.

# 5CONCLUSION

This paper looks into several recent non-linear tensor algorithms that excel in tensor factorization and missing data completion. Closely examining non-linear algorithms reveals that the learned embedding vectors contain non-negligible noises and redundancies. We design an innovative approach to improve the quality of representation learning and confirm its central role in a tensor algorithm. With the high-quality embedding space,the CP tensor algorithm or simple Multi-Layer Perceptron (MLP) can suficiently model multiwayrelationships.We implement a fast and scalable coupled tensor reconstruction algorithm (RegTensor) using deep neural networks to learn interactions among factors and regularization to enforce structure in embedding vectors.We show that our algorithm works exceptionally wellfor single tensor or coupled tensor factorizations. Our experiment further confirms that our strategy of incorporating structure via regularization in factor matrices delivers impressive performance in embedding learning and end-to-end tensor models and outperforms other alternative approaches that only introduce nonlinearity in the phase of reconstructing tensors from factor matrices.

# REFERENCES

[1]Evrim Acar,Daniel MDunlavy, TamaraG Kolda,and Morten Morup.2011.Scalable tensor factorizations for incomplete data. Chemometrics and Inteligent Laboratory Systems 106,1(2011),41-56.   
[2]Evrim Acar,Gozde Gurdeniz,Morten ARasmussen,Daniela Rago,LarsOragsted,and Rasmus Bro.2012.Coupled matrix factorization with sparse factors to identify potential biomarkers in metabolomics.In 2012 IEEE 12th International Conference on Data Mining Workshops.IEEE,1-8.   
[3] Evrim Acar, Tamara G Kolda,and Daniel MDunlavy. 2011.All-at-once optimization for coupled matrix and tensor factorizations.arXiv preprint arXiv:1105.3422 (2011).   
[4] Abdelmonem A Afifi and Robert M Elashoff.1966.Missing observations in multivariate statistics I.Review of the literature.J.Amer. Statist.Assoc.61,315 (1966),595-604.   
[5] Sanaz Bahargam and Evangelos E Papalexakis.2018.Constrained coupled matrixtensor factorization and its application in pattern and topic detection.In 2018 IEEE/ACMInternational ConferenceonAdvancesin Social NetworksAnalysisand Mining (ASONAM).IEEE,91-94.   
[6] Turan G Bali, Amit Goyal, Dashan Huang,Fuwei Jiang,and Quan Wen. 2021. Different strokes: Return predictability across stocks and bonds with machine learning and big data.Swiss Finance Institute,Research Paper Series 20-110 (2021).   
[7] Heiner Beckmeyer and Timo Wiedemann. 2023.Recovering missing firm characteristics with attention-based machine learning.Available at SSRN 4003455 (2023).   
[8]SvetlanaBrygalova,SvenLerner,MartinLetauandarkus Pelger2.iss ing financial data.Available at SSRN 4106794 (2022).   
[9]Ercument Cahan,JushanBai,anderenaNg.2023.Factor-basedimputationof missing values and covariances in panel data of large dimensions.Journal of Econometrics 233,1(2023),113-131.   
[10] JDouglas Carrolland Jih-Jie Chang.197o.Analysis of individual differences in multidimensional scaling via an N-way generalization of“Eckart-Young”decomposition.Psychometrika 35,3(1970),283-319.   
[11] Andrew Y Chen and Jack McCoy.2022. Missing values and the dimensionality of expected returns.arXiv preprint arXiv:2207.13071(2022).   
[12] Andrew Y Chen and Tom Zimmermann. 2021. Open source cross-sectional asset pricing.CriticalFinance Review,Forthcoming (2021).   
[13] Dongjin Choi, Jun-Gi Jang,and Uksong Kang. 2017.Fast,accurate,and scalable method for sparse coupled matrix-tensor factorization.arXiv preprint arXiv:1708.08640 (2017).   
[14] Xiaomin Fang,Rong Pan, Guoxiang Cao, Xiuqiang He,and Wenyuan Dai. 2015. Personalized tag recommendation through nonlinear tensor factorization using gaussian kernel.In Twenty-NinthAAAIConference on Artificial Intelligence.   
[15] Joachim Freyberger, Bjorn Hoppner,Andreas Neuhierl,and Michael Weber. 2022. Missing data in asset pricing panels.Technical Report.National Bureau of Economic Research.   
[16] Silvia Gandy, Benjamin Recht,and Isao Yamada.2011. Tensor completion and low-n-rank tensor recovery via convex optimization. Inverse Problems 27,2(2011), 025010.   
[17]ShihaoGu,Bryan Kell,and Dacheng Xiu.2020.Empirical asset pricingvia machine learning. The Review of Financial Studies 33,5 (2020),2223-2273.   
[18] Richard A Harshman et al.1970.Foundations of the PARAFAC procedure: Models and conditions for an" explanatory" multimodal factor analysis.(1970).   
[19]Lifang He,Chun-Ta Lu, Guixiang Ma,Shen Wang,Linlin Shen,Philip SYu,and Ann BRagin.2017. Kernelized support tensor machines.In Proceedingsof the 34th International Conference on Machine Learning-Volume70.JMLR.org,1442-1451.   
[20] Zhenyu Hou, Xiao Liu,Yukuo Cen, YuxiaoDong, Hongxia Yang,Chunjie Wang, and Jie Tang.2022. Graphmae: Self-supervised masked graph autoencoders. In Proceedings of the 28thACM SIGKDD Conference on Knowledge Discovery and Data Mining.594-604.   
[21] Bryan T.Kelly,Diogo Palhares,and Seth Pruitt.2023.Modeling corporate bond returns.Journal of Finance forthcoming (2023).   
[22] Suleiman A Khan,Eemeli Leppäaho,and Samuel Kaski. 2016. Bayesian multitensor factorization. Machine Learning 105,2 (2016),233-253.   
[23] Yejin Kim,RobertEl-Kareh,Jimeng Sun,Hwanjo Yu,and Xiaoqian Jiang.2017. Discriminative and distinct phenotyping by constrained tensor factorization. Scientific reports7,1(2017),1-12.   
[24] Bin Liu,LirongHe,Yingming Li,Shandian Zhe,and Zenglin Xu.2018.Neuralcp: Bayesian multiway data analysis with neural tensor decomposition. Cognitive Computation 10,6 (2018),1051-1061.   
[25] Hanpeng Liu, Yaguang Li, Michael Tsang,and Yan Liu. 2019.CoSTCo:A Neural Tensor Completion Model for Sparse Tensors.In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery&Data Mining (Anchorage,AK,USA) (KDD‘19).Association for Computing Machinery,NewYork, NY,USA,324-334. https://doi.0rg/10.1145/3292500.3330881   
[26] Ji Liu,Przemyslaw Musialski,Peter Wonka,and Jieping Ye.2012. Tensor completionfor estimating missing values in visual data. IEE transactions on pattern analysis and machine intelligence 35,1(2012),208-220.   
[27]Atsuhiro Narita, Kohei Hayashi,Ryota Tomioka,and Hisashi Kashima.2012. Tensorfactorizationusingauxiliaryinformation.Data Miningand Knowledge Discovery 25,2 (2012),298-24.   
[28] Bernardino Romera-Paredes and Massmiliano Pontil. 2013.A new convex relaxation for tensor completion.In Advances in Neural Information Processing Systems. 2967-2975.   
[29] David E Rumelhart, James L McClelland,and CORPORATE PDP Research Group. 1986.Parallel distributed processing:Explorations in the microstructure of cognition, Vol.1: Foundations. MIT press.   
[30] Amnon Shashua and Tamir Hazan. 2oo5.Non-negative tensor factorization with applications to statisticsandcomputer vision.In Proceedings of the 22nd international conference on Machine learning.792-799.   
[31] Qingquan Song, Xiao Huang,Hancheng Ge,James Caverlee,and Xia Hu. 2017. Multi-aspect streaming tensor completion.In Proceedings of the 23rd ACMSIGKDD International Conferenceon Knowledge Discovery and Data Mining.435-443.   
[32] Ledyard R Tucker.1966.Some mathematical notes on three-mode factor analysis. Psychometrika 31,3(1966),279-311.   
[33]Ajim Uddin, Xinyuan Tao,Chia-Ching Chou,and Dantong Yu. 202o. Nonlinear Tensor Completion Using Domain Knowledge: An Application in Analysts' EarningsForecast.In2020 International Conference on Data Mining Workshops (ICDMW). IEEE,377-384.   
[34] Ajim Uddin, Xinyuan Tao,Chia-Ching Chou,and Dantong Yu.2022.Are missing values important for earnings forecasts?A machine learning perspective. Quantitative finance 22,6 (2022),1113-1132.   
[35] Ajim Uddin, Xinyuan Tao,Chia-Ching Chou,and Dantong Yu. 2022.Machine Learning for Earnings Prediction:A Nonlinear Tensor Approach for Data Integration and Completion.In Proceedings of the Third ACMInternational Conference on AI in Finance.282-290.   
[36] Ajim Uddin, Xinyuan Tao,and Dantong Yu. 2021.Attention Based Dynamic Graph Learning Framework forAssetPricing.In Proceedings of the 30thACM International Conference on Information& Knowledge Management.1844-1853.   
[37] Qing Wu,Jie Wang,JinFan,Gang Xu,Jia Wu,Blake Johnson,XingfeiLi, Quan Do, and Ruiquan Ge.2019.Improved coupled tensor factorization with its applications in health data analysis. Complexity 2019 (2019).   
[38] Xian Wu, Baoxu Shi, Yuxiao Dong,Chao Huang,and Nitesh V.Chawla. 2019. Neural Tensor Factorization for Temporal Interaction Learning.In Proceedings ofthe TwelfthACM International Conference on Web Searchand Data Mining (Melbourne VIC,Australia) (WSDM '19).537-545.   
[39] Ruoxuan Xiong and Markus Pelger. 2023. Large dimensional latent factor modeling with missing observations and applications to causal inference. Journal of Econometrics 233,1(2023),271-301.