# RMT-Net: Reject-Aware Multi-Task Network for Modeling Missing-Not-At-Random Data in Financial Credit Scoring

Qiang Liu , Member, IEEE, Yingtao Luo, Shu Wu $\textcircled{1}$ , Senior Member, IEEE, Zhen Zhang , Xiangnan Yue $\textcircled{1}$ , Hong Jin $\textcircled{1}$ , and Liang Wang , Fellow, IEEE

Abstract—In financial credit scoring, loan applications may be approved or rejected. We can only observe default/non-default labels for approved samples but have no observations for rejected samples, which leads to missing-not-at-random selection bias. Machine learning models trained on such biased data are inevitably unreliable. In this work, we find that the default/non-default classification task and the rejection/approval classification task are highly correlated, according to both real-world data study and theoretical analysis. Consequently, the learning of default/non-default can benefit from rejection/approval. Accordingly, we for the first time propose to model the biased credit scoring data with Multi-Task Learning (MTL). Specifically, we propose a novel Reject-aware Multi-Task Network (RMT-Net), which learns the task weights that control the information sharing from the rejection/approval task to the default/non-default task by a gating network based on rejection probabilities. RMT-Net leverages the relation between the two tasks that the larger the rejection probability, the more the default/non-default task needs to learn from the rejection/approval task. Furthermore, we extend RMT-Net to RMT-Net+ $^ { \ast }$ for modeling scenarios with multiple rejection/approval strategies. Extensive experiments are conducted on several datasets, and strongly verifies the effectiveness of RMT-Net on both approved and rejected samples. In addition, RMT-Net $^ { + + }$ further improves RMT-Net’s performances.

Index Terms—Credit scoring, default prediction, missing-not-at-random, multi-task learning, reject inference

# 1 INTRODUCTION

REDIT scoring aims to use machine learning methods Cto measure customers’ default probabilities of credit loans [1], [2], [3], [4], [5]. Based on the evaluated credits, financial institutions such as banks and online lending companies can decide whether to approve or reject credit loan applications.

When a customer applies for credit loan, his or her application may be approved or rejected. If the application is approved, it will become an approved sample, and the customer will get the loan. After a period, if the customer repays the credit loan timely, it will be a non-default sample; if the customer fails to timely repay, it will be a default sample. In contrast, if the application is not approved, it will become a rejected sample, and the customer will not get credit loan. Since a rejected sample gets no loans, we have no way to observe whether it will be default or non-default. Above process is illustrated in Fig. 1. Credit scoring models are usually constructed based on approved samples, as we have no ground-truth default/non-default labels for rejected samples [6], [7], [8], [9]. The rejection/approval strategies are usually machine learning models or expert rules based on the features of customers, thus approved and rejected samples share different feature distributions. This makes us face the missing-not-at-random selection bias in data [9], [10], [11]. However, when serving online, credit scoring models need to infer credits of loan applications in feature distributions of both approved and rejected samples. Training models with such biased data has severe consequences that the model parameters are biased [12], i.e., the predicted relation between input features and default probability is incorrect. Using such models on samples across various data distributions leads to significant economic losses [7], [13], [14]. Therefore, for reliable credit scoring, besides the modeling of approved samples, we also need to take rejected ones into consideration and infer their true credits [15].

In practice, machine learning models like Logistic Regression (LR), Support Vector Machines (SVM), Multi-Layer Perceptron (MLP) and XGBoost (XGB) are widely used for modeling credit scoring data. However, they are affected by the missing-not-at-random bias in data to produce reliable and accurate predictions. To tackle this problem, some existing approaches address the selection bias and conduct reject inference from multiple perspectives. Some approaches apply the self-training algorithm [16], which iteratively adds rejected samples with higher default probability as default samples to retrain the model [17]. This is a semi-supervised approach [18]. Besides, Semi-Supervised SVM (S3VM) [6] and Semi-Supervised Gaussian Mixture Models (SSGMM) [7] are also deployed in credit scoring systems. In another perspective, some approaches attempt to re-weight the training approved samples to approximate unbiased data [14], [19], [20], [21]. These approaches are similar to counterfactual learning [10], [11], [22], [23], which attempts to re-weight observed samples to remove bias in data.

![](images/ee698225837fe054d2794d0f50a39ad16ad6d852593170dcb39f7595f9529caf.jpg)  
Fig. 1. Illustration of data bias in credit scoring.

Though some of the above approaches have achieved relative improvements on some credit scoring datasets [7], [14], they cannot achieve optimal performances due to the lack of consideration of some key factors. Specifically, we find that the default/non-default classification task and the rejection/ approval classification task are highly correlated in real credit scoring applications, according to both real-world data study and theoretical analysis in Section 3. Intuitively speaking, with an effective credit approval system, rejected customers have higher default ratios, while approved customers have lower ones. Consequently, the learning of default/non-default can benefit from the learning of rejection/approval. Accordingly, it might be promising to incorporate Multi-Task Learning (MTL) [24] for modeling biased credit scoring data.

Nowadays, state-of-the-art MTL approaches mainly focus on adaptively learning weights of different tasks in a mixture-of-experts structure [25], [26], [27], [28], [29]. This makes task weights changing in different samples so that tasks can share useful but not conflict information adaptively. Such MTL approaches achieve promising performances in various scenarios. However, when we use state-of-the-art MTL approaches for modeling the default/non-default task and the rejection/approval task, we do not achieve satisfactory performances, and even achieve poor performances in default prediction on rejected samples. This may be because we have no observed default/non-default labels for rejected samples during model training. The task weights, which decide how much information is shared between the two tasks, are not well optimized in the feature distribution of rejected samples. Thus, exiting MTL approaches fail in modeling the biased credit scoring data, and we need a novel and specially-designed MTL approach.

Accordingly, we propose a Reject-aware Multi-Task Network (RMT-Net). RMT-Net learns the weights that contask to the default/non-default task by a gating network based on rejection probabilities. With larger rejection probability, less reliable information can be learned in the default/non-default network and more information is shared from the rejection/approval network. In this way, we can consider the correlation between rejected samples and default samples, as well as personalize the information sharing weights in the feature distribution of rejected samples. Furthermore, we consider cases with multiple rejection/ approval strategies, and extend RMT-Net to RMT- $\mathbf { \nabla \cdot N e t + + }$ which models several rejection/approval classification tasks in the MTL framework.

In all, we verify RMT-Net and RMT-Net $^ { + + }$ on 10 datasets under different settings, in which significant improvements are achieved for default prediction on both accepted and rejected samples. Evaluated by the commonly-used Kolmogorov-Smirnov (KS) metric1 in credit scoring, comparing with conventional classifiers, i.e. LR, DNN, and XGB, RMTNet relatively improves the performances by $4 7 . 9 \%$ on average. Comparing with the most competitive reject inference approaches, RMT-Net relatively improves the performances by $1 1 . 9 \%$ on average. In addition, we show in an extra experiment with multiple rejection/approval strategies that RMT- $\mathrm { N e t } { + + }$ can further relatively improve the performances of RMT-Net by $5 . 8 \%$ on average.

The main contributions of this work are concluded:

We for the first time propose to model biased credit scoring data using an MTL approach, namely RMTNet. Instead of directly using conventional MTL approaches, we present several modifications to improve the poor performances of existing MTL approaches on credit scoring.   
We further consider multiple rejection/approval strategies, and extend RMT-Net to RMT- $\mathrm { N e t } { + + }$ . In this way, our work suits different application scenarios in real applications.   
Extensive experiments are conducted on 10 datasets under different settings. Significant improvements are achieved by our proposed RMT-Net approach on both accepted and rejected samples. In addition, we show that RMT-Net+ $^ { - + }$ with multiple strategies can further improve the performances.

The rest of the paper is organized as follows. In Section 2, we review some related work on reject inference, counterfactual learning and multi-task learning. Then we analyze the correlation between the default/non-default task and the rejection/approval task according to both real-world data study and theoretical analysis in Section 3. Sections 4 and 5 detail our proposed RMT-Net and RMT- $\mathrm { \Delta N e t { + + } }$ under single strategy and multiple strategies respectively. In Section 6, we conduct empirical experiments to verify the effectiveness of RMT-Net and RMT-NET $^ { + + }$ . Section 7 concludes our work.

# 2 RELATED WORK

In this section, we review some works on reject inference, as well as two important related research aspects: counterfactual learning and multi-task learning.

# 2.1 Reject Inference

In the credit scoring task, we have only ground-truth default/non-default labels for approved samples but no ground-truth default/non-default labels for rejected samples. This causes the missing-not-at-random bias in data [9], [10], [11] for machine learning models. Some reject inference approaches are accordingly proposed [8], [14], [15].

Augmentation is a re-weighting approach [19], [20], [21], in which accepted samples are re-weighted to represent the entire distribution. A common way to achieve this is reweighting according to the rejection/approval probability. Moreover, the augmentation approach has been extended in a fuzzy way [14]. Parcelling is also a re-weighting approach, where the re-weighting is determined by the default probability by score-band that is adjusted by the credit modeler [8], [21]. To be noted, these re-weighting methods are similar to the researches on counterfactual learning [10], [11], [22], [23]. Counterfactual learning aims to remove data bias, in which the re-weighting of training samples is widely adopted.

Meanwhile, semi-supervised approaches are also applied to deal with the reject inference task. In [17], the authors use a self-training algorithm to improve the performance of SVM on credit scoring. Self-training, also known as selflabeling or decision-directed learning, is the most simple semi-supervised learning method [16], [30], [31]. This approach trains a model on approved samples, and labels rejected samples with largest default probabilities as default samples according to model predictions. Then, the newly labeled samples are added to retrain the model, and this process continues iteratively. Though the self-training algorithm is only used to promote SVM in [17], it can also promote other classifiers, such as LR, MLP and XGB. Besides, another semi-supervised version of SVM called S3VM [6] is also applied in reject inference. S3VM uses approved and rejected samples to fit an optimal hyperplane with maximum margin, but have problem in fitting large-scale data [7]. Meanwhile, earlier works have used some statistical machine learning methods, such as Expectation-Maximization (EM) algorithm [32], Gaussian Mixture Models (GMM) [33] and survival analysis [34], for reject inference. Based on GMM and inspired by semi-supervised generative models [35], [36], SS-GMM [7] is proposed for modeling biased credit scoring data. The counterfactual re-weighting and semi-supervised learning are the main methods for reject inference, but neither approach considers the correlation between the learning of rejection/approval and the learning of default/non-default.

# 2.2 Counterfactual Learning

Counterfactual learning [23] is a key direction of the research on causal inference [37], [38]. Counterfactual learning aims to simulate counterfactuals to alleviate the missing-not-at-random bias for a less biased model training [39]. In the context of credit scoring, we know that rejected samples are unobserved, but counterfactual learning tries to answer “what if they are observable?”

The traditional counterfactual learning approaches usually re-weight samples based on propensity scores [22], [23], [40], [41], [42], [43]. Propensity scores indicate the probabilities of observation under different environments, e.g., approval and rejection in the credit scoring scheme. These propensity scorebased methods and re-weighting approaches in reject inference [19], [20], [21] are similar, and both try to balance the data distributions of observed and unobserved samples. Stable learning is another perspective of counterfactual learning, in which there is no implicit treatments and the distribution of unobserved samples is unknown [44]. Stable learning is usually done via decorrelation among features of samples, which tries to make the feature distribution closer to independently identically distribution [45], [46], [47]. Meanwhile, Sample Reweighted Decorrelation Operator (SRDO) [46] generates some unobserved samples, and trains a binary classifier to get the probabilities of observation for re-weighting the observed samples. This is somehow similar to the propensity score-based approaches.

Missing-not-at-random selection bias is also frequently discussed in recommender systems, where we can only observe feedback of displayed user-item pairs [10], [48]. In counterfactual recommendation, propensity score is also applied, and the Inverse Propensity Score (IPS) approach [11], [49] that re-weights observed samples with the inverse of displayed probabilities is proposed. Based on IPS, Doubly Robust (DR) [50] and Joint Learning Doubly Robust (DRJL) [51] are proposed to consider doubly robust estimator. After that, some improvements have been presented, such as asymmetrically tri-training [52], considering information theory [53], and proposing better doubly robust estimators [54]. Meanwhile, the Adversarial Counterfactual Learning (ACL) approach [55] incorporates adversarial learning for counterfactual recommendation. Besides, some works on counterfactual recommendation rely on a small amount of random unbiased data [56], [57], [58]. However, random data requires high costs, especially in financial applications.

# 2.3 Multi-Task Learning

MTL learns multiple tasks simultaneously in one model, and has been proven to improve performances through information sharing between tasks [24], [26]. It has succeed in scenarios such as computer vision [29], [59], [60], recommender systems [25], [26], [27], [28], [61], [62], healthcare [63], and other prediction problems [64], [65].

The simplest MTL approach is hard parameter sharing, which shares hidden representations across different tasks, and only the last prediction layers are special for different tasks [24]. However, hard parameter sharing suffers from conflicts among tasks, due to the simple sharing of representations. To deal with this problem, some approaches propose to learn weights of linear combinations to fuse hidden representations in different tasks, such as Cross-Stitch Network [59] and Sluice Network [60]. However, in different samples, the weights of different tasks stay the same, which limits the performances of MTL. This inspires the research on applying gating structures in MTL [25], [26], [27], [66]. Mixture-Of-Experts (MOE) first proposes to share and combine several experts through a gating network [66]. Based on MOE, to make the weights of different tasks varying across different samples and to improve the performances of MTL, Multi-gate MOE (MMOE) [25] proposes to use different gates for different tasks. Progressive Layered Extraction (PLE) further extends MMOE, and incorporates multilevel experts and gating networks [26]. Besides, attention networks are also utilized for assigning weights of tasks according to different feature representations [28], [29].

TABLE 1 The Mean Values of Fico Score, Debt-to-Income Ratio, Loan Amount and Employment Length of Approved and Rejected Customers in the Lending Club Dataset Across Different Years   

<table><tr><td rowspan="2">Year</td><td colspan="2">Fico Score</td><td colspan="2">Debt-To-Income Ratio (%)</td><td colspan="2">Loan Amount</td><td colspan="2">Employment Length (year)</td></tr><tr><td>Approved</td><td>Rejected</td><td>Approved</td><td>Rejected</td><td>Approved</td><td>Rejected</td><td>Approved</td><td>Rejected</td></tr><tr><td>2013</td><td>692</td><td>649</td><td>15.04</td><td>20.22</td><td>13921</td><td>13271</td><td>2.34</td><td>1.68</td></tr><tr><td>2014</td><td>687</td><td>638</td><td>14.76</td><td>19.88</td><td>12386</td><td>12040</td><td>2.14</td><td>1.58</td></tr><tr><td>2015</td><td>687</td><td>640</td><td>14.71</td><td>22.03</td><td>13542</td><td>13663</td><td>2.55</td><td>1.56</td></tr><tr><td>2016</td><td>692</td><td>636</td><td>14.43</td><td>24.14</td><td>13912</td><td>13698</td><td>2.70</td><td>1.37</td></tr><tr><td>2017</td><td>692</td><td>636</td><td>14.20</td><td>23.00</td><td>12644</td><td>12501</td><td>2.40</td><td>1.26</td></tr><tr><td>2018</td><td>702</td><td>632</td><td>14.56</td><td>20.37</td><td>13897</td><td>13385</td><td>3.28</td><td>1.23</td></tr></table>

# 3 ANALYSIS

In this section, we plan to analyze the correlation between the default/non-default task and the rejection/approval task. First, we analyze the correlation between the rejected samples and the approved samples in real-world datasets, in which we can conclude that users whose loan applications are rejected are more likely to default. Then, we theoretically prove that the reject/approve task and the default/ non-default task are correlated, so that we are motivated to model the reject/approve task and the default/non-default task by multi-task learning.

# 3.1 Data Study

With analyses of public real-world datasets, we plan to illustrate the difference between the rejected samples and the approved samples. Specifically, we adopt the Lending Club dataset2 in our analysis. Lending $\mathrm { C l u } \hat { \boldsymbol { \mathrm { b } } } ^ { 3 }$ is one of the largest credit loan companies worldwide. Though there are no ground-truth default/non-default labels associated with the rejected samples, the Lending Club dataset is a valuable dataset since it is a rare publicly available credit scoring dataset that contains rejected samples and their features. With the help of the Lending Club dataset, we can empirically investigate the difference between the approved samples and the rejected samples.

In the Lending Club dataset, there are totally four feature fields: Fico score, debt-to-income ratio, loan amount, and employment length. In order to investigate the difference between the approved samples and the rejected samples, in Table 1, we show Fico score, debt-to-income ratio, loan amount and employment length of the approved customers and the rejected customers in the Lending Club dataset across different years. Meanwhile, we remove samples with missing values in the Lending Club dataset. From results in Table 1, we have following observations.

1) Approved customers have higher Fico scores, while rejected customers have lower Fico scores. Fico score, which is provided by the Fico Company4 , integrates a customer’s credit record. A larger Fico score means better credit history of a customer, which leads to lower default risk.

2) Loan amounts are similar between approved and rejected customers. And approved customers have lower debt-to-income ratios, while rejected customers have higher debt-to-income ratios. Debt-to-income ratio is calculated as $\frac { M o n t h l y I n c o m e } { L o a n A m o u n t }$ , which means debtto-income ratio is a derived variable of loan amount. A larger debt-to-income ratio usually means worse repayment ability, which leads to a larger probability of default.

3) Approved customers have larger employment length than rejected customers. Customers with larger employment length usually have better repayment ability. These observations tell us that, rejected loan applications are more likely to default or overdue.

# 3.2 Motivation

We show by Theorem 1 that the prediction of default/nondefault task and the prediction of rejection/approval task are positively correlated. Therefore, the learning of rejection/approval task can be beneficial for the learning of default/non-default task via multi-task learning methods.

Theorem 1 (Correlation between default/non-default and rejection/approval). Assume that default samples are denoted as $D _ { \iota }$ non-default samples are denoted as $\bar { D } _ { \ell }$ , rejected samples are denoted as $R ,$ and approved samples are denoted as ${ \bar { R } } . A$ loan is either default or non-default, and is either rejected or approved. If the prerequisite rejection strategy is effective, which means that the default rate of rejected samples is indeed larger than the default rate of the approved samples, i.e., $P ( D | R ) >$ $P ( D | \bar { R } )$ , then the correlation coefficient $c o r r ( D , R ) =$ $\begin{array} { r } { \frac { \dot { P } ( D R ) - P ( D ) P ( R ) } { \sqrt { P ( D ) P ( \bar { D } ) P ( R ) P ( \bar { R } ) } } > 0 , } \end{array}$ , i.e., the rejection and the default are posð Þ ð Þ ð Þ ðitively correlated.

Proof. If $P ( D | R ) > P ( D | \bar { R } ) ,$ , we have $P ( D | R ) > P ( D | R +$ $\bar { R } ) = P ( D R + D \bar { R } ) P ( R + \bar { R } ) = P ( D )$ ð j Þsince that $R + \bar { R }$ þis Þ ¼ ð þ Þthe full set and $P ( R + \bar { R } ) = 1$ Þ þ. Therefore, we have $P ( D R ) = P ( D | R ) P ( R ) > P ( D ) P ( R ) .$ , and $c o r r ( D , R ) =$ $\begin{array} { r } { \frac { P ( D R ) - P ( D ) P ( R ) } { \sqrt { P ( D ) P ( \bar { D } ) P ( R ) P ( \bar { R } ) } } > 0 , } \end{array}$ Þ ð Þ ð Þ ð Þ ¼i.e., the learning of rejection and ð Þ ð Þ ð Þ ð Þthe learning of default are positively correlated. □

To be noted, in Theorem 1, we rely on the assumption that the prerequisite rejection strategy is effective, which means the credit approval system is useful. By this assumption, we tenable in real-world applications, otherwise, credit loan institutions will face tremendous economic losses, so that the strategy is unlikely to be under using.

![](images/2f0d0cecfe72411802b0b83318dbc16a28f0f9b623d822142d8ee3cb524a63a0.jpg)  
Fig. 2. Schematic diagram of RMT-Net and RMT- $\mathsf { N e t } { + } +$ . $" G "$ parts denote the gating networks for calculating the weights of the rejection/approval task. The solid lines indicate the feedforward of feature representations. The dotted lines indicate the message passing in gating networks for calculating task weights.

In summary, the learning of default/non-default (with limited and biased data) can benefit from the learning of rejection/approval (with a larger amount of unbiased data).

# 4 RMT-NET UNDER SINGLE REJECTION/APPROVAL STRATEGY

According to both data study and theoretical analysis in Section 3, the default/non-default task and the rejection/ approval task are highly correlated. Thus, it is proper to model biased credit scoring data with multi-task learning. In this section, we detail our proposed RMT-Net model under single strategy, which focuses on scenarios with a constant rejection/approval strategy.

# 4.1 Notations

All applications are denoted as a set of samples ${ \boldsymbol { S } } =$ $\{ s _ { 1 } , s _ { 2 } , . . . . . , s _ { N } \} ,$ where $N$ S ¼is the total number of loan applif gcations. Each sample $s _ { i }$ is associated with a feature vector $\boldsymbol { x } _ { i } \in \mathcal { R } ^ { d }$ , where $d$ is the feature dimensionality. For each 2 Rsample $s _ { i } \in S ,$ we have a $r _ { i } \in \{ 0 , 1 \}$ to indicate its rejection/ 2 Sapproval label, where $r _ { i } = 0$ f gmeans approval and $r _ { i } = 1$ ¼means rejection. For each sample $s _ { i }$ approved, i.e., $r _ { i } = 0$ ¼, we have ground-truth default/non-default label $y _ { i } \in \{ 0 , 1 \} ,$ where $y _ { i } = 1$ means default and $y _ { i } = 0$ 2 f gmeans non-default. ¼For each sample $s _ { i }$ rejected, i.e., $r _ { i } = 1 _ { . }$ , we have no ground¼truth default/non-default label for model training. We need to use the whole set of $s$ to train a model, which should perSform well on both approved and rejected samples for default loans prediction.

# 4.2 Model Architecture

Current state-of-the-art MTL approaches mainly focus on adaptively learning weights of different tasks in a mixtureof-experts structure, such as MMOE [25] and PLE [26]. In such MTL structures, task weights change in different samples, which makes useful but not conflicting information to adaptively share among tasks. Considering these MTL approaches have achieved promising performances in various scenarios, a simple way for reject inference might be directly applying them in training credit scoring models. However, according to experiments in Section 6.3, we obtain very poor performances in default prediction on rejected samples. This could be because some task weights are not well learned, since we have no observed default/ non-default labels for rejected samples during model training. In the feature distribution of rejected samples, there is no supervision for optimizing the task weights to control how much information is shared from the rejection/ approval task to the default/non-default task. Thus, existing MTL approaches fail in biased credit scoring data, and we propose RMT-Net to learn the task weights by a gating network based on rejection probabilities.

Our proposed RMT-Net consists of 1) an embedding layer that learns the dense representation of the feature vectors; 2) a multi-layer rejection/approval prediction network (R/A-Net); 3) a multi-layer default/non-default prediction network (D/N-Net); 4) a gating network that learns the weights of the rejection/approval task for the default/nondefault task. The model architecture is shown in Fig. 2a.

# 4.3 Embedding Layer

The embedding layer transforms each feature into a dense vector to facilitate learning efficiency. For numerical features that are infeasible to embed, feature discretization techniques [67], [68], [69] are conducted, which are proven useful to improve the learning efficiency [70]. The embedding layer converts each feature vector $\boldsymbol { x } _ { i } \in \mathcal { R } ^ { d }$ into an embedded representation $e _ { i } \in \mathcal { R } ^ { d \times k }$ , where $k$ 2 Ris the embed2 Rding dimensionality. Afterward, the embedded representation is flattened to form a one-dimensional embedded vector $e _ { i } \in \mathcal { R } ^ { d k }$ . Therefore, after embedding, the data samples $\mathcal { S } \in \mathcal { R } ^ { N \times d }$ will become $E \in \mathcal { R } ^ { N \times d k }$ .

# 4.4 Rejection/Approval Prediction Network

The rejection/approval prediction network (R/A-Net) has multiple linear layers with an activation function. The first term layer multiplies the embedded vector $w _ { R } ^ { ( 1 ) } \in \mathcal { R } ^ { d k \times h _ { 1 } }$ , adds a bias term $e _ { i } \in \mathcal { R } ^ { d k }$ $b _ { R } ^ { ( 1 ) } \in \mathcal { R } ^ { h _ { 1 } }$ with a weight , and acti2 R 2 Rvates with a nonlinear function relu. The first layer results in the latent representation of the first layer $\check { p } _ { i } ^ { ( 1 ) } \in \mathcal { R } ^ { h _ { 1 } }$ , where $h _ { j }$ means the dimensionality of the $j$ 2 R-th layer. Except the final layer, the other layers further update the latent representations in the same way of the first layer as

$$
p _ { i } ^ { ( j ) } = r e l u \Big ( p _ { i } ^ { ( j - 1 ) } w _ { R } ^ { ( j ) } + b _ { R } ^ { ( j ) } \Big ) ,
$$

where $p _ { i } ^ { ( j ) } \in \mathcal { R } ^ { h _ { j } }$ . For the final layer, if we denote the num2 Rber of layers as $t ,$ we will have $h _ { t } = 1$ . The final output of ¼the R/A-Net will be the rejection probability

$$
p _ { i } ^ { ( t ) } = \sigma \Big ( p _ { i } ^ { ( t - 1 ) } w _ { R } ^ { ( t ) } + b _ { R } ^ { ( t ) } \Big ) ,
$$

where $p _ { i } ^ { ( t ) } \in \mathcal { R } ^ { 1 }$ , and $\sigma ( \cdot )$ is the sigmoid function.

# 4.5 Default/Non-Default Prediction Network

The default/non-default prediction network (D/N-Net) has the same number of layers of the R/A-Net as $t$ and resembles its basic linear computations with activation. The difference is that the latent representation is determined under the proposed multi-task learning framework. We first design a gating network at each layer $j$ that uses the output of R/A-Net as

$$
g _ { i } ^ { ( j ) } = \sigma \Big ( \alpha ^ { ( j ) } p _ { i } ^ { ( t ) } + \beta ^ { ( j ) } \Big ) ,
$$

where $g _ { i } ^ { ( j ) } \in \mathcal { R } ^ { 1 }$ . $\alpha ^ { ( j ) } \in \mathcal { R } ^ { 1 }$ and $\beta ^ { ( j ) } \in \mathcal { R } ^ { 1 }$ are learnable 2 R 2 Rparameters to control the value of $g ,$ 2 R which is designed to indicate the ratio of learning the rejection/approval task for the learning of default/non-default. Apparently, the activated output of the R/A-Net, i.e. the probability of rejection, determines how much information in R/A-Net are used to learn the default/non-default task. Except the final layer, the latent representation $q _ { i } ^ { ( j ) } \in \mathcal { R } ^ { h _ { j } }$ of each layer of the D/NNet can be denoted as

$$
q _ { i } ^ { ( j ) } = r e l u \Bigl ( q _ { i } ^ { ( j - 1 ) } w _ { D } ^ { ( j ) } + b _ { D } ^ { ( j ) } \Bigr ) + g _ { i } ^ { ( j ) } p _ { i } ^ { ( j ) } ,
$$

where $w _ { D } ^ { ( j ) }$ and $b _ { D } ^ { ( j ) }$ denote the weight term and the bias term respectively in the D/N-Net. For the final layer, the final output of the D/N-Net is denoted as

$$
q _ { i } ^ { ( t ) } = \sigma \Big ( q _ { i } ^ { ( t - 1 ) } w _ { D } ^ { ( t ) } + b _ { D } ^ { ( t ) } \Big ) ,
$$

where $\boldsymbol q _ { i } ^ { ( t ) } \in \mathcal { R } ^ { 1 }$ means the default probability.

2 RIn this way, we can adaptively control the weights of the rejection/approval task in different samples by the rejection probability, and overcome the under-fitting problem of conventional MTL approaches in the feature distribution of rejected samples. In our proposed MTL architecture, the model can learn that, with a larger rejection probability, less reliable information can be learned in R/A-Net, and more information should be shared from D/N-Net.

# 4.6 Loss Function

For the rejection/approval task, given the rejection/ approval label $r _ { i }$ and the output of the R/A-Net ${ p } _ { i } ^ { ( t ) }$ , we have

$$
\mathcal { L } _ { 1 } = - \sum _ { i = 1 } ^ { N } p _ { i } ^ { ( t ) } l o g ( r _ { i } ) + \Big ( 1 - p _ { i } ^ { ( t ) } \Big ) l o g ( 1 - r _ { i } ) .
$$

For the default/non-default prediction task, considering we have no observed default/non-default labels for rejected samples, we need to mask the loss with reject/approval labels. Given the default/non-default label $y _ { i } ,$ , the rejection/ approval label ${ \boldsymbol { r } } _ { i } ,$ and the output of the D/N-Net ${ q _ { i } ^ { ( t ) } }$ , we have

$$
\mathcal { L } _ { 2 } = - \sum _ { i = 1 } ^ { N } ( q _ { i } ^ { ( t ) } l o g ( y _ { i } ) + ( 1 - q _ { i } ^ { ( t ) } ) l o g ( 1 - y _ { i } ) ) ( 1 - r _ { i } ) .
$$

If we define $\eta$ as a hyperparameter to balance the two losses, the overall loss function is denoted as

$$
\mathcal { L } = ( 1 - \eta ) \cdot \mathcal { L } _ { 1 } + \eta \cdot \mathcal { L } _ { 2 } .
$$

# 5 RMT-NET $^ { + + }$ UNDER MULTIPLE REJECTION/ APPROVAL STRATEGIES

In Section 4, we have detailed the RMT-Net model under single constant rejection/approval strategy. However, in realworld applications, rejection/approval strategies change frequently, so we usually have multiple strategies in different periods. For example, financial institutions may modify approval ratios, add important factors or new factors in the credit evaluation systems. Thus we encounter a variety of rejection/approval segmentation in the training data of credit scoring. If we simply regard these strategies as one strategy, there will be conflicts in the model when classifying approved and rejected samples. Therefore, we extend RMTNet to ${ \mathrm { R M T - N e t + + } } ,$ , which further incorporates multiple strategies in the multi-task learning framework to improve the prediction accuracy.

# 5.1 Notations

Basic notations still follow the notations in Section 4.1. Besides, we have $M$ different rejection/approval strategies $\left\{ f _ { 1 } , f _ { 2 } , \dots . . . . . , f _ { M } \right\}$ . Each sample $s _ { i } \in S$ is under one specific f grejection/approval strategy $f _ { s _ { i } }$ 2 S, and its rejection/approval label $r _ { i } \in \{ 0 , 1 \}$ is determined by the corresponding strat2 f gegy. Multiple-strategy scenarios can degenerate to singlestrategy scenarios when $M = 1$ .

# 5.2 Model Architecture

The model architecture of RMT- $\mathrm { N e t } { + + }$ is based on that of RMT-Net. RMT-Net $^ { + + }$ has the same embedding layer as the RMT-Net. Because RMT-Net assumes $M = 1 .$ , i.e., it only ¼uses a single rejection/approval strategy for the learning of default/non-default and contains one rejection/approval prediction network (R/A-Net). RMT- $\mathrm { \cdot N e t { + + } }$ , on the contrary, has $M \neq 1$ R/A-Nets that each learns the data samples 6¼whose labels are collected by its unique strategy. Therefore, the gating network and the default/non-default prediction network (D/N-Net) in the RMT- $\mathrm { N e t } { + + }$ will change accordingly. The model architecture is shown in Fig. 2b.

# 5.3 Rejection/Approval Networks++

The rejection/approval networks in RMT-Net $^ { + + }$ (R/A-Nets $+ + )$ include $M \neq 1$ rejection/approval prediction networks 6¼(R/A-Net) that each only learns on data samples of the corresponding rejection/approval strategy. Here, each R/ANet is the same as used in RMT-Net, as described in Section 4.4. To distinguish among different R/A-Nets, we might as well add a square bracket to the lower right corner of the notations of each R/A-Net. In this case, for the $m$ -th R/A-Net, we can denote the latent representations of its layers as

$$
\begin{array} { r } { p _ { i , [ m ] } ^ { ( j ) } = r e l u \Big ( p _ { i , [ m ] } ^ { ( j - 1 ) } w _ { R , [ m ] } ^ { ( j ) } + b _ { R , [ m ] } ^ { ( j ) } \Big ) , } \end{array}
$$

and denote the final outputs as

$$
p _ { i , [ m ] } ^ { ( t ) } = \sigma \Big ( p _ { i , [ m ] } ^ { ( t - 1 ) } w _ { R , [ m ] } ^ { ( t ) } + b _ { R , [ m ] } ^ { ( t ) } \Big ) ,
$$

where pðjÞ $p _ { i , [ m ] } ^ { ( j ) } \in \mathcal { R } ^ { h _ { j } }$ $p _ { i , [ m ] } ^ { ( t ) } \in \mathcal { R } ^ { 1 }$

# 5.4 Default/Non-Default Network++

The default/non-default network $\mathrm { ( D / N  – N e t + + ) }$ ) has the same number of layers of each $\mathrm { R } / \mathrm { A } – \mathrm { N e t } + +$ as $t$ . Its latent representation is calculated under the proposed multi-task learning framework with multiple strategies. We design a gating network at each layer $j$ that uses the output of the $m$ -th R/A-Net++ as

$$
g _ { i , [ m ] } ^ { ( j ) } = \sigma \Big ( \alpha _ { [ m ] } ^ { ( j ) } p _ { i , [ m ] } ^ { ( t ) } + \beta _ { [ m ] } ^ { ( j ) } \Big ) ,
$$

where $g _ { i , [ m ] } ^ { ( j ) } \in \mathcal { R } ^ { 1 }$ . $\alpha _ { [ m ] } ^ { ( j ) } \in \mathcal { R } ^ { 1 }$ and $\beta _ { [ m ] } ^ { ( j ) } \in \mathcal { R } ^ { 1 }$ are learnable ½  ½ parameters that control the value of ${ \mathit { g } } _ { [ m ] } ,$ which are designed ½ to indicate the ratio of learning each rejection/approval task under strategy $f _ { m }$ for the learning of default/non-default. Here, we consider the activated output of each $\mathrm { R / A  – N e t + + } ,$ , i.e. the probability of rejection by each strategy, determines how much parameters of each $\mathrm { R } / \mathrm { A } – \mathrm { N e t } + +$ are used for the default/non-default task. Except the final layer, the latent representation of each layer of the $\mathrm { D } / \mathrm { N } { \mathrm { - N e t + + } }$ can be denoted as

$$
q _ { i } ^ { ( j ) } = r e l u \Bigl ( q _ { i } ^ { ( j - 1 ) } w _ { D } ^ { ( j ) } + b _ { D } ^ { ( j ) } \Bigr ) + \sum _ { m = 1 } ^ { M } g _ { i , [ m ] } ^ { ( j ) } p _ { i , [ m ] } ^ { ( j ) } ,
$$

where $q _ { i } ^ { ( j ) } \in \mathcal { R } ^ { h _ { j } }$ . And the final output is denoted as

$$
q _ { i } ^ { ( t ) } = \sigma \Big ( q _ { i } ^ { ( t - 1 ) } w _ { D } ^ { ( t ) } + b _ { D } ^ { ( t ) } \Big ) ,
$$

where $\boldsymbol q _ { i } ^ { ( t ) } \in \mathcal { R } ^ { 1 }$ means the default probability.

# 5.5 Loss Function

For the rejection/approval task, given the rejection/approval label $r _ { i }$ and the outputs of the R/A-Net++, we have

$$
\mathcal { L } _ { 1 } = - \sum _ { m = 1 } ^ { M } \sum _ { i = 1 , f _ { s _ { i } } = f _ { m } } ^ { N } p _ { i , [ m ] } ^ { ( t ) } l o g ( r _ { i } ) + ( 1 - p _ { i , [ m ] } ^ { ( t ) } ) l o g ( 1 - r _ { i } ) .
$$

For the default/non-default prediction task, given the

and the output of the $\mathrm { D } / \mathrm { N - N e t } + { q } _ { i } ^ { ( t ) }$ , we have

$$
\mathcal { L } _ { 2 } = - \sum _ { i = 1 } ^ { N } ( q _ { i } ^ { ( t ) } l o g ( y _ { i } ) + ( 1 - q _ { i } ^ { ( t ) } ) l o g ( 1 - y _ { i } ) ) ( 1 - r _ { i } ) .
$$

With $\eta$ as a hyperparameter to balance the two losses, we denote the overall loss function as

$$
\mathcal { L } = ( 1 - \eta ) \cdot \mathcal { L } _ { 1 } / M + \eta \cdot \mathcal { L } _ { 2 } .
$$

# 6 EXPERIMENTS

In this section, we conduct empirical experiments to verify the effectiveness of RMT-Net and RMT-NET $^ { + + }$ .

# 6.1 Datasets

In our experiments, we are going to verify our proposed approaches from three aspects: experiments on approved samples, experiments on both approved and rejected samples, experiments under multiple rejection/approval policies.

Approval-only datasets: Firstly, we need to investigate whether our proposed approaches can improve the performances on approved samples. This means, training set and testing set, which are both approved samples, share similar data distributions. We select datasets with real rejected samples, but without ground-truth default/non-default labels for rejected samples. From the Lending Club dataset, we extract samples in 2013, 2014 and 2015 to construct the Lending1 dataset, the Lending2 dataset and the Lengding3 dataset respectively. In our experiments, we randomly use $6 0 \%$ , $2 0 \%$ , and $2 0 \%$ approved samples as training, validation, and testing set respectively. Features of rejected samples can be used during the training of credit scoring models.

Approval-rejection datasets: Secondly, we investigate whether our proposed approaches can stably achieve promising performances in different data distributions. This requires us to conduct experiments on both approved and rejected samples. However, ground-truth default/nondefault labels for real rejected samples are hard to obtain, so that we need to generate synthetic rejected samples from some real-world credit scoring datasets. Here, we incorporate the Home5 and $\mathrm { P P D } ^ { 6 }$ datasets. To be noted, samples in these two datasets are actually all approved samples with ground-truth default/non-default labels. The process of generating synthetic rejected samples is conducted as follows: (1) use random $1 / 3$ samples (denoted initial samples) and $\varepsilon \cdot d$ features to train an LR model as the synthetic rejection/approval policy of the credit scoring system, where $\varepsilon$ is a ratio that $0 \% < \varepsilon \leq 1 0 0 \% ,$ and $d$ is the dimensionality of input features; (2) use the trained LR model to predict the default probabilities of the rest $2 / 3$ samples (denoted as main samples); (3) assign $3 / 4$ of the main samples with largest default probabilities as synthetic rejected samples; (4) assign $1 / 4$ of the main samples with smallest default probabilities as approved samples. This process is very similar to a real-world credit scoring system, which uses some initial loan applications to train a machine learning model for future decisions of rejection/approval. $\varepsilon$ is to control the strength of rejection/approval. With larger $\varepsilon$ values, more features will be used, and data distributions between rejected and approved samples will be more distinguishable. On each of Home and PPD, we run the above synthetic process two times and set $\varepsilon = 1 0 0 \%$ and $\varepsilon = 5 0 \%$ respec¼ ¼tively. This results in four approval-rejection datasets: Home1, Home2, PPD1 and PPD2. On each dataset, we randomly use $6 0 \%$ , $2 0 \%$ , and $2 0 \%$ approved samples as training, validation, and testing set respectively. Meanwhile, the testing set also contains all the rejected samples to conduct performance comparison across different data distributions. Moreover, features of rejected samples are also used during the training of credit scoring models, but the ground-truth default-non-default labels cannot be used during training.

TABLE 2 Details of Datasets With Single rejection/approval Policy   

<table><tr><td></td><td>Rejection Ratio (%)</td><td colspan="2">Default Ratio (%)</td></tr><tr><td>Dataset</td><td></td><td>Approved</td><td>Rejected</td></tr><tr><td>Lending1</td><td>84.00</td><td>15.40</td><td>=</td></tr><tr><td>Lending2</td><td>87.63</td><td>17.36</td><td>=</td></tr><tr><td>Lending3</td><td>54.86</td><td>18.08</td><td>=</td></tr><tr><td>Home1</td><td>75.00</td><td>1.97</td><td>10.06</td></tr><tr><td>Home2</td><td>75.00</td><td>2.85</td><td>9.69</td></tr><tr><td>PPD1</td><td>75.00</td><td>4.40</td><td>15.90</td></tr><tr><td>PPD2</td><td>75.00</td><td>2.20</td><td>9.04</td></tr></table>

Multi-policy datasets: Thirdly, we need to verify the effectiveness of our proposed approaches under multiple policies. In the Lending Club dataset, we regard the policies in 2013, 2014 and 2015 as three different policies and obtain a multipolicy dataset named Lending-M. For Home and PPD, we split each dataset into two equal subsets, and run the synthetic rejected sample generation process on each sub-set with $\varepsilon =$ $5 0 \%$ ¼. This results in two different rejection/approval policies on each dataset. Thus, we obtain two multi-policy datasets named Home-M and PPD-M. The training/validation/testing split of Lending-M is the same as that of previous approval-only datasets. The training/validation/testing split of Home-M and PPD-M is the same as that of previous approval-rejection datasets. That is to say, there are only approved samples in Lending-M, and both approved and synthetic rejected samples in Home-M and PPD-M.

Approval-only datasets and approval-rejection datasets are both datasets with a single rejection/approval policy, and their details are shown in Table 2. And details of multipolicy datasets are illustrated in Table 3. As in real systems, features in above datasets are constructed based on records before the time of each loan application. This makes us available to predict whether a customer will default if he or she gets the loan, for we need to know the default probability before the loan approval. If a customer applies for loan for more than one time, there will exist multiple samples in the data, and each of them corresponds to one application, with features constructed based on records before the corresponding application time.

# 6.2 Settings

We compare four types of approaches: baselines, semisupervised learning approaches, counterfactual learning approaches and multi-task learning approaches.

TABLE 3 Details of Datasets With Multiple rejection/approval Policies   

<table><tr><td colspan="3">(a) The Lending-M dataset.</td></tr><tr><td>Policy</td><td>Rejection Ratio (%)</td><td>Default Ratio (%) Approved Rejected</td></tr><tr><td>1</td><td>84.00</td><td></td></tr><tr><td>2</td><td>87.63</td><td></td></tr><tr><td>3</td><td>54.86</td><td>=</td></tr><tr><td colspan="3">(b) The Home-M dataset.</td></tr><tr><td>Policy</td><td>Rejection Ratio (%)</td><td>Default Ratio (%) Approved Rejected</td></tr><tr><td>1</td><td>75.00</td><td>3.54 9.45 2.60</td></tr><tr><td>2</td><td>75.00</td><td>9.84</td></tr><tr><td colspan="3">(c) The PPD-M dataset.</td></tr><tr><td>Policy</td><td>Rejection Ratio (%)</td><td>Default Ratio (%) Approved Rejected</td></tr><tr><td>1</td><td>75.00</td><td>2.92 8.63 3.56</td></tr><tr><td>2</td><td>75.00</td><td>8.53</td></tr></table>

For single-policy datasets, i.e., approval-only datasets and approval-rejection datasets, the following approaches are compared. Baselines consists of three commonly-used classifiers for credit scoring: LR, MLP and XGB. Among semi-supervised approaches, Self-Training (ST) [17] is incorporated with LR, MLP and XGB. Another typical semisupervised reject inference approach SS-GMM [7] is also compared. For counterfactual approaches, we involve IPS [11], [49], DR [50], DRJL [51], ACL [55] and SRDO [46], and incorporate them with LR and MLP. For multi-task approaches, besides our proposed RMT-Net, we involve Cross-Stitch [59], MMOE [25] and PLE [26].

For multi-policy datasets, we use the same baselines, semi-supervised approaches and counterfactual approaches as above. The only difference is that we consider multiple policies in multi-task approaches. Specifically, we adjust Cross-Stitch, MMOE and PLE under multiple policies, and name them as Cross-Stitch-M, MMOE-M and PLE-M. Moreover, our proposed RMT-Net++ is also compared.

For our proposed RMT-Net and RMT- $\mathrm { \cdot N e t { + + } }$ , we empirically set learning rate as 0.001, embedding dimensionality of each input feature as 4, and dimensionality of hidden layers as 16. Meanwhile, according to the best performances on the validation set, we tune the loss balancing parameter $\lambda$ in the range of [0.1,0.2,0.3,0.4,0.5], and the layer number $t$ in the range of [2,3,4]. For other compared approaches, their hyper-parameters are tuned according to the performances on the validation set. For all compared approaches including RMT-Net and RMT$\mathrm { N e t } { + + }$ , early-stopping is conducted according to the performances on the validation set. We run each approach 10 times and report the median values.

We evaluate the performances on testing sets in terms of two commonly-used metrics for credit scoring and default prediction: AUC and KS. AUC measures the overall ranking performance. KS measures the largest difference between true positive rate and false positive rate on the ROC curve, which can be the threshold for loan approval.

TABLE 4 Performance Comparison on Approval-Only Datasets, Evaluated by AUC $( \% )$ and $K S ( \% )$   

<table><tr><td rowspan="2">Type</td><td rowspan="2">Approach</td><td colspan="2">Lending1</td><td colspan="2">Lending2</td><td colspan="2">Lending3</td><td colspan="2">average</td></tr><tr><td>AUC</td><td>KS</td><td>AUC</td><td>KS</td><td>AUC</td><td>KS</td><td>AUC</td><td>KS</td></tr><tr><td rowspan="3">Baseline</td><td>LR</td><td>59.94</td><td>13.53</td><td>60.52</td><td>15.81</td><td>61.67</td><td>17.43</td><td>60.71</td><td>15.59</td></tr><tr><td>MLP</td><td>59.96</td><td>13.77</td><td>60.50</td><td>15.72</td><td>61.54</td><td>17.53</td><td>60.67</td><td>15.67</td></tr><tr><td>XGB</td><td>59.93</td><td>13.62</td><td>60.54</td><td>15.69</td><td>61.34</td><td>17.18</td><td>60.60</td><td>15.50</td></tr><tr><td rowspan="4">Semi-Supervised Learning</td><td>ST+LR</td><td>60.18</td><td>14.02</td><td>60.49</td><td>15.73</td><td>61.87</td><td>17.86</td><td>60.85</td><td>15.87</td></tr><tr><td>ST+MLP ST+XGB</td><td>60.13</td><td>13.89</td><td>60.54</td><td>15.89</td><td>61.71</td><td>17.78</td><td>60.79</td><td>15.85</td></tr><tr><td>SS-GMM</td><td>60.19</td><td>14.11</td><td>60.51</td><td>15.86</td><td>61.83</td><td>17.89</td><td>60.84</td><td>15.95</td></tr><tr><td></td><td>60.09</td><td>13.75</td><td>60.55</td><td>15.78</td><td>62.06</td><td>18.24</td><td>60.90</td><td>15.92</td></tr><tr><td rowspan="10">Counterfactual Leaning</td><td>IPS+LR</td><td>60.21</td><td>13.45</td><td>60.36</td><td>15.52</td><td>61.53</td><td>17.59</td><td>60.70</td><td>15.52</td></tr><tr><td>IPS+MLP</td><td>60.25</td><td>14.58</td><td>60.44</td><td>16.00</td><td>61.59</td><td>17.63</td><td>60.76</td><td>16.07</td></tr><tr><td>DR+LR</td><td>60.18</td><td>13.69</td><td>60.32</td><td>15.36</td><td>61.37</td><td>17.41</td><td>60.62</td><td>15.49</td></tr><tr><td>DR+MLP</td><td>60.22</td><td>14.34</td><td>60.48</td><td>15.72</td><td>61.68</td><td>17.72</td><td>60.79</td><td>15.93</td></tr><tr><td>DRJL+LR</td><td>60.28</td><td>14.53</td><td>60.41</td><td>15.52</td><td>61.42</td><td>17.58</td><td>60.70</td><td>15.88</td></tr><tr><td>DRJL+MLP</td><td>60.25</td><td>14.59</td><td>60.54</td><td>16.10</td><td>61.73</td><td>17.81</td><td>60.84</td><td>16.17</td></tr><tr><td>ACL+LR</td><td>60.31</td><td>14.46</td><td>60.49</td><td>15.75</td><td>61.79</td><td>17.88</td><td>60.86</td><td>16.03</td></tr><tr><td>ACL+MLP</td><td>60.34</td><td>14.63</td><td>60.59</td><td>15.86</td><td>61.85</td><td>17.94</td><td>60.93</td><td>16.14</td></tr><tr><td>SRDO+LR</td><td>60.24</td><td>14.03</td><td>60.56</td><td>15.74</td><td>61.58</td><td>17.59</td><td>60.79</td><td>15.79</td></tr><tr><td>SRDO+MLP</td><td>60.13</td><td>13.59</td><td>60.39</td><td>15.82</td><td>61.47</td><td>17.47</td><td>60.66</td><td>15.63</td></tr><tr><td rowspan="4">Multi-Task Learning</td><td>Cross-Stitch</td><td>60.33</td><td>14.51</td><td>60.76</td><td>16.54</td><td>62.17</td><td>18.56</td><td>61.09</td><td>16.54</td></tr><tr><td>MMOE</td><td>60.24</td><td>14.63</td><td>60.62</td><td>16.17</td><td>62.09</td><td>18.31</td><td>60.98</td><td>16.37</td></tr><tr><td>PLE</td><td>60.32</td><td>14.41</td><td>60.71</td><td>16.45</td><td>61.97</td><td>18.04</td><td>61.00</td><td>16.30</td></tr><tr><td>RMT-Net</td><td>60.61*</td><td>15.35*</td><td>61.02*</td><td>17.08*</td><td>62.48*</td><td>18.97*</td><td>61.37*</td><td>17.13*</td></tr></table>

Average values are also listed. $^ *$ denotes statistically significant improvement, measured by T-Test with $p$ -Value $< 0 . 0 1$ , over the second-best approach on each dataset.

TABLE 5 Performance Comparison on Approval-Rejection Datasets, Evaluated by AUC $( \% )$ and $K S ( \% )$   

<table><tr><td rowspan="2">Type</td><td rowspan="2">Approach</td><td colspan="2">Home1</td><td colspan="2">Home2</td><td colspan="2">PPD1</td><td colspan="2">PPD2</td><td colspan="2">Average</td></tr><tr><td>AUC</td><td>KS</td><td>AUC</td><td>KS</td><td>AUC</td><td>KS</td><td>AUC</td><td>KS</td><td>AUC</td><td>KS</td></tr><tr><td rowspan="3">Baseline</td><td>LR</td><td>54.87</td><td>7.42</td><td>68.05</td><td>26.46</td><td>62.83</td><td>19.88</td><td>58.80</td><td>13.41</td><td>61.14</td><td>16.79</td></tr><tr><td>MLP</td><td>55.17</td><td>7.63</td><td>67.72</td><td>25.82</td><td>60.80</td><td>16.54</td><td>58.63</td><td>12.75</td><td>60.58</td><td>15.69</td></tr><tr><td>XGB</td><td>55.60</td><td>8.12</td><td>67.81</td><td>26.07</td><td>63.21</td><td>20.93</td><td>59.32</td><td>14.19</td><td>61.49</td><td>17.33</td></tr><tr><td rowspan="4">Semi-Supervised Learning</td><td>ST+LR</td><td>66.12</td><td>23.96</td><td>67.88</td><td>27.71</td><td>66.22</td><td>23.90</td><td>64.57</td><td>21.11</td><td>66.20</td><td>24.17</td></tr><tr><td>ST+MLP</td><td>66.37</td><td>24.40</td><td>67.93</td><td>27.56</td><td>64.35</td><td>21.28</td><td>64.87</td><td>21.40</td><td>65.88</td><td>23.66</td></tr><tr><td>ST+XGB SS-GMM</td><td>66.60 66.21</td><td>24.58 23.61</td><td>67.77 68.59</td><td>27.23</td><td>66.58</td><td>24.71</td><td>64.79</td><td>21.22</td><td>66.44</td><td>24.44</td></tr><tr><td></td><td></td><td></td><td></td><td>27.71</td><td>67.12</td><td>25.93</td><td>64.50</td><td>20.96</td><td>66.61</td><td>24.55</td></tr><tr><td rowspan="9">Counterfactual Leaning</td><td>IPS+LR</td><td>66.26</td><td>24.43</td><td>67.24</td><td>24.99</td><td>69.20</td><td>28.43</td><td>63.73</td><td>19.98</td><td>66.61</td><td>24.46</td></tr><tr><td>IPS+MLP</td><td>66.37</td><td>24.59</td><td>67.93</td><td>25.87</td><td>69.88</td><td>29.32</td><td>63.87</td><td>20.26</td><td>67.01</td><td>25.01</td></tr><tr><td>DR+LR</td><td>65.72</td><td>24.14</td><td>67.66</td><td>25.23</td><td>69.66</td><td>29.64</td><td>63.49</td><td>19.95</td><td>66.63</td><td>24.74</td></tr><tr><td>DR+MLP</td><td>65.87</td><td>24.20</td><td>67.92</td><td>25.93</td><td>69.43</td><td>29.39</td><td>64.26</td><td>21.19</td><td>66.87</td><td>25.18</td></tr><tr><td>DRJL+LR</td><td>66.67</td><td>24.61</td><td>68.42</td><td>27.11</td><td>69.31</td><td>29.10</td><td>64.20</td><td>21.27</td><td>67.15</td><td>25.52</td></tr><tr><td>DRJL+MLP</td><td>66.30</td><td>24.42</td><td>68.68</td><td>27.97</td><td>69.57</td><td>29.71</td><td>64.59</td><td>21.58</td><td>67.29</td><td>25.92</td></tr><tr><td>ACL+LR</td><td>65.97</td><td>23.56</td><td>67.81</td><td>25.09</td><td>68.49</td><td>27.54</td><td>62.86</td><td>19.11</td><td>66.28</td><td>23.83</td></tr><tr><td>ACL+MLP</td><td>66.59 66.14</td><td>24.23 24.53</td><td>67.91</td><td>25.40</td><td>69.30 69.47</td><td>27.69</td><td>63.69</td><td>19.77</td><td>66.87</td><td>24.27</td></tr><tr><td>SRDO+LR SRDO+MLP</td><td>65.98</td><td>24.20</td><td>68.26 67.58</td><td>26.13 25.49</td><td>69.62</td><td>28.60 28.49</td><td>63.41 63.66</td><td>19.74 19.90</td><td>66.82 66.71</td><td>24.75 24.52</td></tr><tr><td rowspan="4">Multi-Task Learning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Cross-Stitch</td><td>64.33 56.31</td><td>21.96 14.39</td><td>66.96</td><td>23.81</td><td>66.71</td><td>26.63</td><td>60.93</td><td>17.06</td><td>64.73</td><td>22.37</td></tr><tr><td>MMOE</td><td>57.63</td><td>15.70</td><td>63.55 63.90</td><td>22.49</td><td>66.09 65.89</td><td>24.53 24.61</td><td>56.17 54.62</td><td>10.54</td><td>60.53</td><td>17.99</td></tr><tr><td>PLE RMT-Net</td><td>71.03*</td><td>30.84*</td><td>71.99*</td><td>21.51 32.40*</td><td>71.00*</td><td>30.30*</td><td>69.44*</td><td>9.16 29.00*</td><td>60.51 70.87*</td><td>17.75 30.64*</td></tr></table>

Average values are also listed. $^ *$ denotes statistically significant improvement, measured by T-Test with $p { \cdot }$ -Value $< 0 . 0 1$ , over the second-best approach on each dataset.

# 6.3 Performance Comparison

In our experiments, the performance comparison is conducted from three perspectives.

Firstly, we need to conduct performance comparison on approved samples. Table 4 illustrates performance comparison on approval-only datasets. Overall speaking, semi-supervised, counterfactual and multi-task approaches achieve relative improvements compared with baselines. This means, these approaches are effective for credit scoring, even when training and testing samples share similar data distributions. Moreover, it is clear that RMT-Net achieves the best performances. On average, RMT-Net relatively improves baselines by $9 . 3 2 \%$ , and improves the second-best compared approach by $3 . 6 1 \%$ , evaluated by KS.

Secondly, we need to investigate whether our proposed approaches can stably achieve promising performances in different data distributions. This requires us to conduct experiments on both approved and rejected samples. This is a very important experiment, for we strongly need credit scoring models that can stably and accurately infer credits of loan applications in feature distributions of both approved and rejected samples in real-world applications. Table 5 illustrates performance comparison on approval-rejection datasets. It is clear that baselines achieve poor performances, due to the distribution shift between training and testing samples. Both semi-supervised and counterfactual approaches significantly outperform baselines, and counterfactual approaches slightly outperform semi-supervised approaches. Meanwhile, multitask learning approaches except RMT-Net perform poorly, and some of them are even worse than baselines. This could be because some task weights are not well learned, since we have no observed default/non-default labels for rejected samples during model training. In the feature distribution of rejected samples, there is no supervision for optimizing the task weights to control how much information is shared from the rejection/approval task to the default/non-default task. Instead, RMT-Net clearly achieves such supervision for optimizing the task weights with the best performances on all datasets. Comparing with Cross-Stitch, MMOE, and PLE, RMT-Net relatively improves KS by $3 6 . 9 7 \%$ , $7 0 . 3 2 \%$ , and $7 2 . 6 2 \%$ on average respectively. Moreover, on average, RMT-Net relatively improves baselines by $7 6 . 8 0 \%$ , and improves the second-best compared approach by $1 8 . 2 1 \%$ , evaluated by KS. These improvements are much larger than those on approval-only datasets. This is because rejected samples are evaluated during testing. With a better learning of the default/non-default task for rejected samples, RMT-Net provides bigger room for improvements.

TABLE 6 Performance Comparison on Multi-Policy Datasets, Evaluated by AUC $( \% )$ and $K S ( \% )$ .   

<table><tr><td rowspan="2">Type</td><td rowspan="2">Approach</td><td colspan="2">Lending-M</td><td colspan="2">Home-M</td><td colspan="2">PPD-M</td><td colspan="2">Average</td></tr><tr><td>AUC</td><td>KS</td><td>AUC</td><td>KS</td><td>AUC</td><td>KS</td><td>AUC</td><td>KS</td></tr><tr><td rowspan="3">Baseline</td><td>LR MLP</td><td>60.66 60.59</td><td>15.43 15.48</td><td>70.31 70.03</td><td>29.47 29.20</td><td>62.66 62.44</td><td>18.83 19.27</td><td>64.54 64.35</td><td>21.24 21.32</td></tr><tr><td>XGB</td><td>60.69</td><td>15.53</td><td>70.11</td><td>29.32</td><td>62.76</td><td>19.54</td><td>64.52</td><td>21.46</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Semi-Supervised Learning</td><td>ST+LR ST+MLP</td><td>60.59</td><td>15.47</td><td>70.53</td><td>30.29</td><td>65.66</td><td>23.89</td><td>65.59</td><td>23.22</td></tr><tr><td></td><td>60.76</td><td>15.51</td><td>70.28</td><td>29.67</td><td>66.26</td><td>24.68</td><td>65.77</td><td>23.29</td></tr><tr><td>ST+XGB SS-GMM</td><td>60.63 60.81</td><td>15.42 15.70</td><td>70.11 70.47</td><td>29.39 29.88</td><td>65.89 66.08</td><td>24.11 24.57</td><td>65.54 65.79</td><td>22.97 23.38</td></tr><tr><td rowspan="9">Counterfactual</td><td>IPS+LR</td><td>60.74</td><td>15.76</td><td>68.36</td><td>27.01</td><td>61.89</td><td>18.28</td><td>63.66</td><td>20.35</td></tr><tr><td>IPS+MLP</td><td>60.82</td><td>15.68</td><td>70.12</td><td>29.65</td><td>62.97</td><td>19.42</td><td>64.64</td><td>21.58</td></tr><tr><td>DR+LR</td><td>60.71</td><td>15.55</td><td>67.81</td><td>26.50</td><td>65.23</td><td>23.59</td><td>64.58</td><td>21.88</td></tr><tr><td>DR+MLP</td><td>60.78</td><td>15.59</td><td>69.72</td><td>28.96</td><td>65.76</td><td>24.13</td><td>65.42</td><td>22.89</td></tr><tr><td>DRJL+LR</td><td>60.76</td><td>15.86</td><td>69.44</td><td>28.51</td><td>67.17</td><td>25.70</td><td>65.79</td><td>23.36</td></tr><tr><td>DRJL+MLP</td><td>60.89</td><td>15.89</td><td>70.52</td><td>30.23</td><td>67.42</td><td>26.32</td><td>66.28</td><td>24.15</td></tr><tr><td>ACL+LR</td><td>60.76</td><td>15.80</td><td>70.50</td><td>30.05</td><td>65.78</td><td>23.49</td><td>65.68</td><td>23.11</td></tr><tr><td>ACL+MLP</td><td>60.71</td><td>16.68</td><td>70.47</td><td>29.95</td><td>66.07</td><td>23.90</td><td>65.75</td><td>23.51</td></tr><tr><td>SRDO+LR</td><td>60.85</td><td>15.90</td><td>70.06</td><td>29.50</td><td>63.22</td><td>20.12</td><td>64.71</td><td>21.84</td></tr><tr><td rowspan="5">Multi-Task Learning</td><td>SRDO+MLP</td><td>60.79</td><td>15.56</td><td>69.56</td><td>29.11</td><td>62.07</td><td>18.80</td><td>64.14</td><td>21.16</td></tr><tr><td>Cross-Stitch-M</td><td>61.10</td><td>16.40</td><td>70.59</td><td>30.33</td><td>66.86</td><td>25.18</td><td>66.18</td><td>23.97</td></tr><tr><td>MMOE-M</td><td>61.07</td><td>16.52</td><td>70.55</td><td>30.29</td><td>64.81</td><td>21.87</td><td>65.48</td><td>22.89</td></tr><tr><td>PLE-M</td><td>61.16</td><td>16.61</td><td>70.62</td><td>30.45</td><td>63.52</td><td>20.94</td><td>65.10</td><td>22.67</td></tr><tr><td>RMT-Net RMT-Net++</td><td>61.28 61.60*</td><td>16.84 17.43*</td><td>71.01 71.96*</td><td>30.84 32.28*</td><td>68.86 70.41*</td><td>28.11 30.50*</td><td>67.05 67.99*</td><td>25.26 26.74*</td></tr></table>

Average Values are Also Listed. $^ *$ denotes statistically significant improvement, measured by T-Test with $p { - } V a l u e < 0 . 0 1$ , over the second-best approach on each dataset.

![](images/9888115e02909615f194151ba20c45594ced041f61d0d453af76442b237e9e0a.jpg)  
Fig. 3. Performances of RMT-Net and RMT-Net $^ { + + }$ on testing set with varying hyper-parameters: (1) the left part shows the impact of the loss balancing parameter $\lambda$ ; (2) the right part shows the impact of the layer number $t$ . To be noted, we report results of RMT-Net on single-policy dataset, and results of RMT- $\mathsf { N e t } { + } +$ on multi-policy datasets.

Thirdly, we also need to conduct performance comparison under multiple policies. Table 6 shows performance comparison on multi-policy datasets. We can clearly observe that RMT- $\mathrm { N e t } { + + }$ can further improve the performances of RMT-Net. On average, RMT-Net $^ { + + }$ relatively improves RMT-Net by $5 . 8 3 \%$ , evaluated by KS.

These experimental results strongly verify the effectiveness of our proposed RMT-Net and RMT-Net $^ { + + }$ .

# 6.4 Hyper-Parameter Study

Furthermore, we are going to investigate the impact of hyper-parameters in our proposed RMT-Net and RMT-Net $+ +$ . In Fig. 3, we illustrate the performances of RMT-Net and RMT- $\mathrm { N e t } { + + }$ on testing set with varying loss balancing parameter $\lambda$ and layer number $t$ .

Firstly, the loss balancing parameter $\lambda$ somehow affects the performances of RMT-Net and RMT-Net++. Thus, it is better for us to tune this hyper-parameter according to validation set for optimal performances. In Section 6.3, we report results on testing set via hyper-parameter tuning on validation set. Moreover, the performances are not very sensitive to $\lambda ,$ and performance on each dataset stays stable in a range of loss balancing parameter $\lambda$ .

![](images/0acd52b05b5b4c41648f3f072d3a27c1a4f16d5dce039f39a993149484edc98b.jpg)  
Fig. 4. Visualization of gating networks in RMT-Net and RMT-Net++, in which we illustrate the relationship between the rejection probability and the weight of the rejection/approval task for the learning of the default/non-default task.

Secondly, the layer number $t$ has very slight effects on the performances of RMT-Net and RMT-Ne $^ { + + }$ . Thus, we do not need to carefully tune this hyper-parameter. To be noted, $t$ includes the final layer in MLP, and the minimum value of $t$ is 2. This means, $t$ layers indicate we have $t - 1$ hidden layers in RMT-Net and RMT- $\mathrm { \cdot N e t { + + } }$ for each task. If we set $t = 1$ , there will be no hidden layers in RMT-Net and RMT- $\mathrm { \cdot N e t { + + } }$ , and our design reject-aware multi-task learning framework will be invalid. Accordingly, for simplicity, we set $t = 2$ in rest of our ¼experiments, and report according results on testing set in Section 6.3.

# 6.5 Visualization

In Fig. 4, we illustrate the relationship between the rejection probability and the weight of the rejection/approval task for the learning of the default/non-default task in RMT-Net and RMT-Net $^ { + + }$ . This demonstrates the status of gating networks. For all datasets, the larger the rejection probability, the larger the weight of the rejection/approval task. This means our proposed approaches learns that, with a larger rejection probability, less reliable information can be learned in the default/non-default network, and more information should be shared from the rejection/approval network. With such a gating network, we can alleviate the under-fitting problem of conventional multi-task approaches in the feature distribution of rejected samples.

# 7 CONCLUSION

In this paper, we focus on modeling biased credit scoring data, in which we have only ground-truth labels for approved samples and no observations for rejected samples. Such bias affects the reliability of default prediction, and we aim to improve the prediction accuracy on both approved and rejected samples. We find that the default/non-default classification task and the rejection/approval classification task are highly correlated in credit scoring applications, according to both real-world data study and theoretical analysis. We for the first time propose to model biased credit scoring data using an MTL framework, and propose a novel RMT-Net approach, which learns the task weights that control the information sharing from the rejection/approval task to the default/non-default task by a gating network based on rejection probabilities. According to empirical experiments on 10 datasets under different settings, RMTNet improves the poor performances of existing MTL approaches, and significantly outperforms several state-ofthe-art approaches from different perspectives. Furthermore, we extend RMT-Net to RMT- $\mathrm { N e t } { + + }$ for modeling scenarios with multiple rejection/approval strategies. According to an extra experiment, RMT- $\mathrm { \Delta N e t { + + } }$ with multiple strategies can further improve the performances of RMT-Net in a more complex multi-policy scenario.

# ACKNOWLEDGMENT

The authors would like to thank the anonymous reviewers for their valuable comments and suggestions allowing them to improve the quality of this paper.

# REFERENCES

[1] D. West, “Neural network credit scoring models,” Comput. Operations Res., vol. 27, no. 11–12, pp. 1131–1152, 2000.   
[2] B. Hu et al., “Loan default analysis with multiplex graph learning,” in Proc. 29th ACM Int. Conf. Inf. Knowl. Manage., 2020, pp. 2525–2532.   
[3] Y. Liu, X. Ao, Q. Zhong, J. Feng, J. Tang, and Q. He, “Alike and unlike: Resolving class imbalance problem in financial credit risk assessment,” in Proc. 29th ACM Int. Conf. Inf. Knowl. Manage., 2020, pp. 2125–2128.   
[4] Q. Liu, Z. Liu, H. Zhang, Y. Chen, and J. Zhu, “Mining cross features for financial credit risk assessment,” in Proc. 29th ACM Int. Conf. Inf. Knowl. Manage., 2021, pp. 1069–1078.   
[5] D. Babaev, M. Savchenko, A. Tuzhilin, and D. Umerenkov, “ETRNN: Applying deep learning to credit loan applications,” in Proc. 25th ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining, 2019, pp. 2183–2190.   
[6] Z. Li, Y. Tian, K. Li, F. Zhou, and W. Yang, “Reject inference in credit scoring using semi-supervised support vector machines,” Expert Syst. Appl., vol. 74, pp. 105–114, 2017.   
[7] R. A. Mancisidor, M. Kampffmeyer, K. Aas, and R. Jenssen, “Deep generative models for reject inference in credit scoring,” Knowl.- Based Syst., vol. 196, 2020, Art. no. 105758.   
[8] A. Ehrhardt, C. Biernacki, V. Vandewalle, P. Heinrich, and S. Beben, “Reject inference methods in credit scoring,” J. Appl. Statist., vol. 48, no. 13-15, pp. 2734–2754, 2021.   
[9] N. Goel, A. Amayuelas, A. Deshpande, and A. Sharma, “The importance of modeling data missingness in algorithmic fairness: A causal perspective,” in Proc. AAAI Conf. Artif. Intell., 2021, pp. 7564–7573.   
[10] B. M. Marlin and R. S. Zemel, “Collaborative prediction and ranking with non-random missing data,” in Proc. 3rd ACM Conf. Recommender Syst., 2009, pp. 5–12.   
[11] T. Schnabel, A. Swaminathan, A. Singh, N. Chandak, and T. Joachims, “Recommendations as treatments: Debiasing learning and evaluation,” in Proc. Int. Conf. Mach. Learn., 2016, pp. 1670–1679.   
[12] M. Bucker, M. van Kampen, and W. Kr € €amer, “Reject inference in consumer credit scoring with nonignorable missing data,” J. Bank. Finance, vol. 37, no. 3, pp. 1040–1045, 2013.   
[13] T. B. Astebro and G. G. Chen, “The economic value of reject inference in credit scoring,” in Proc. Credit Scoring Credit Control Conf. (VII), 2001.   
[14] H.-T. Nguyen et al., “Reject inference in application scorecards: Evidence from france,” Univ. Paris Nanterre, EconomiX, Tech. Rep. 2016-10, 2016.   
[15] D. J. Hand and W. E. Henley, “Can reject inference ever work?,” IMA J. Manage. Math., vol. 5, no. 1, pp. 45–55, 1993.   
[16] A. Agrawala, “Learning with a probabilistic teacher,” IEEE Trans. Inf. Theory, vol. 16, no. 4, pp. 373–379, Jul. 1970.   
[17] S. Maldonado and G. Paredes, “A semi-supervised approach for reject inference in credit scoring using SVMs,” in Proc. Int. Conf. Data Mining, 2010, pp. 558–571.   
[18] X. Zhu and A. B. Goldberg, “Introduction to semi-supervised learning,” Synth. Lectures Artif. Intell. Mach. Learn., vol. 3, no. 1, pp. 1–130, 2009.   
[19] D. C. Hsia, “Credit scoring and the equal credit opportunity act,” Hastings LJ, vol. 30, pp. 371–372, 1978.   
[20] J. Banasik and J. Crook, “Reject inference, augmentation, and sample selection,” Eur. J. Oper. Res., vol. 183, no. 3, pp. 1582–1594, 2007.   
[21] J. Banasik, J. Crook, and L. Thomas, “Sample selection bias in credit scoring models,” J. Oper. Res. Soc., vol. 54, no. 8, pp. 822–832, 2003.   
[22] N. Hassanpour and R. Greiner, “Counterfactual regression with importance sampling weights,” in Proc. Int. Joint Conf. Artif. Intell., 2019, pp. 5880–5887.   
[23] H. Zou et al., “Counterfactual prediction for bundle treatment,” in Proc. Adv. Neural Inf. Process. Syst., 2020, pp. 19 705–19 715. of-experts,” in Proc. 24th ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining, 2018, pp. 1930–1939.   
[26] H. Tang, J. Liu, M. Zhao, and X. Gong, “Progressive layered extraction (PLE): A novel multi-task learning (MTL) model for personalized recommendations,” in Proc. 3rd ACM Conf. Recommender Syst., 2020, pp. 269–278.   
[27] D. Xi et al., “Modeling the sequential dependence among audience multi-step conversions with multi-task learning in targeted display advertising,” in Proc. 27th ACM SIGKDD Conf. Knowl. Discov. Data Mining, 2021, pp. 3745–3755.   
[28] J. Zhao, B. Du, L. Sun, F. Zhuang, W. Lv, and H. Xiong, “Multiple relational attention network for multi-task learning,” in Proc. 25th ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining, 2019, pp. 1123–1131.   
[29] S. Liu, E. Johns, and A. J. Davison, “End-to-end multi-task learning with attention,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2019, pp. 1871–1880.   
[30] M. Culp and G. Michailidis, “An iterative algorithm for extending learners to a semi-supervised setting,” J. Comput. Graphical Statist., vol. 17, no. 3, pp. 545–571, 2008.   
[31] G. R. Haffari and A. Sarkar, “Analysis of semi-supervised learning with the yarowsky algorithm,” in Proc. 23rd Conf. Uncertainty Artif. Intell., 2007, pp. 159–166.   
[32] B. Anderson and J. M. Hardin, “Modified logistic regression using the em algorithm for reject inference,” Int. J. Data Anal. Techn. Strategies, vol. 5, no. 4, pp. 359–373, 2013.   
[33] A. Feelders, “Credit scoring and reject inference with mixture models,” Intell. Syst. Accounting, Finance Manage., vol. 9, no. 1, pp. 1–8, 2000.   
[34] S. Y. Sohn and H. Shin, “Reject inference in credit operations based on survival analysis,” Expert Syst. Appl., vol. 31, no. 1, pp. 26–29, 2006.   
[35] D. J. Rezende, S. Mohamed, and D. Wierstra, “Stochastic backpropagation and approximate inference in deep generative models,” in Proc. Int. Conf. Mach. Learn., 2014, pp. 1278–1286.   
[36] D. P. Kingma, S. Mohamed, D. J. Rezende, and M. Welling, “Semisupervised learning with deep generative models,” in Proc. Neural Inf. Process. Syst., 2014, pp. 3581–3589.   
[37] J. Pearl, “Causal inference in statistics: An overview,” Statist. Surv., vol. 3, pp. 96–146, 2009.   
[38] S. L. Morgan and C. Winship, Counterfactuals and Causal Inference. Cambridge, U.K.: Cambridge Univ. Press, 2015.   
[39] R. J. Little and D. B. Rubin, Statistical Analysis with Missing Data., vol. 793. Hoboken, NJ, USA: Wiley, 2019.   
[40] P. C. Austin, “An introduction to propensity score methods for reducing the effects of confounding in observational studies,” Multivariate Behav. Res., vol. 46, no. 3, pp. 399–424, 2011.   
[41] P. R. Rosenbaum and D. B. Rubin, “The central role of the propensity score in observational studies for causal effects,” Biometrika, vol. 70, no. 1, pp. 41–55, 1983.   
[42] N. Hassanpour and R. Greiner, “Learning disentangled representations for counterfactual regression,” in Proc. Int. Conf. Learn. Representations, 2019.   
[43] M. J. Lopez and R. Gutman, “Estimation of causal effects with multiple treatments: A review and new ideas,” Stat. Sci., vol. 32, no. 3, pp. 432–454, 2017.   
[44] K. Kuang, P. Cui, S. Athey, R. Xiong, and B. Li, “Stable prediction across unknown environments,” in Proc. 24th ACM SIGKDD Int. Conf. Knowl. Discov. Data mining, 2018, pp. 1617–1626.   
[45] K. Kuang, R. Xiong, P. Cui, S. Athey, and B. Li, “Stable prediction with model misspecification and agnostic distribution shift,” in Proc. AAAI Conf. Artif. Intell., 2020, pp. 4485–4492.   
[46] Z. Shen, P. Cui, T. Zhang, and K. Kunag, “Stable learning via sample reweighting,” in Proc. AAAI Conf. Artif. Intell., 2020, pp. 5692–5699.   
[47] X. Zhang, P. Cui, R. Xu, L. Zhou, Y. He, and Z. Shen, “Deep stable learning for out-of-distribution generalization,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2021, pp. 5372–5382.   
[48] M. Sato, S. Takemori, J. Singh, and T. Ohkuma, “Unbiased learning for the causal effect of recommendation,” in Proc. 3rd ACM Conf. Recommender Syst., 2020, pp. 378–387.   
[49] A. Swaminathan and T. Joachims, “The self-normalized estimator for counterfactual learning,” in Proc. Neural Inf. Process. Syst., 2015, pp. 3231–3239.   
[50] N. Jiang and L. Li, “Doubly robust off-policy value evaluation for reinforcement learning,” in Proc. Int. Conf. Mach. Learn., 2016, pp. 652–661.   
[51] X. Wang, R. Zhang, Y. Sun, and J. Qi, “Doubly robust joint learning for recommendation on data missing not at random,” in Proc. Int. Conf. Mach. Learn., 2019, pp. 6638–6647.   
[52] Y. Saito, “Asymmetric tri-training for debiasing missing-not-atrandom explicit feedback,” in Proc. 43rd Int. ACM SIGIR Conf. Res. Develop. Inf. Retrieval, 2020, pp. 309–318.   
[53] Z. Wang, X. Chen, R. Wen, S.-L. Huang, E. Kuruoglu, and Y. Zheng, “Information theoretic counterfactual learning from missing-not-at-random feedback,” in Proc. Neural Inf. Process. Syst., 2020, pp. 1854–1864.   
[54] S. Guo et al., “Enhanced doubly robust learning for debiasing post-click conversion rate estimation,” in Proc. 44th Int. ACM SIGIR Conf. Res. Develop. Inf. Retrieval, 2021, pp. 275–284.   
[55] D. Xu, C. Ruan, E. Korpeoglu, S. Kumar, and K. Achan, “Adversarial counterfactual learning and evaluation for recommender system,” in Proc. Neural Inf. Process. Syst., 2020, pp. 13 515–13 526.   
[56] S. Bonner and F. Vasile, “Causal embeddings for recommendation,” in Proc. 3rd ACM Conf. Recommender Syst., 2018, pp. 104–112.   
[57] B. Yuan et al., “Improving ad click prediction by considering nondisplayed events,” in Proc. 29th ACM Int. Conf. Inf. Knowl. Manage., 2019, pp. 329–338.   
[58] J. Chen et al., “Autodebias: Learning to debias for recommendation,” in Proc. 44th Int. ACM SIGIR Conf. Res. Develop. Inf. Retrieval, 2021, pp. 21–30.   
[59] I. Misra, A. Shrivastava, A. Gupta, and M. Hebert, “Cross-stitch networks for multi-task learning,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 3994–4003.   
[60] S. Ruder, J. Bingel, I. Augenstein, and A. Søgaard, “Sluice networks: Learning what to share between loosely related tasks,” 2017, arXiv:1705.08142.   
[61] Q. Liu, S. Wu, and L. Wang, “Multi-behavioral sequential prediction with recurrent log-bilinear model,” IEEE Trans. Knowl. Data Eng., vol. 29, no. 6, pp. 1254–1267, Jun. 2017.   
[62] H. Wen et al., “Entire space multi-task modeling via post-click behavior decomposition for conversion rate prediction,” in Proc. 43rd Int. ACM SIGIR Conf. Res. Develop. Inf. Retrieval, 2020, pp. 2377–2386.   
[63] B. Liu, Y. Li, S. Ghosh, Z. Sun, K. Ng, and J. Hu, “Complication risk profiling in diabetes care: A Bayesian multi-task and feature relationship learning approach,” IEEE Trans. Knowl. Data Eng., vol. 32, no. 7, pp. 1276–1289, Jul. 2020.   
[64] L. Zhao, Q. Sun, J. Ye, F. Chen, C.-T. Lu, and N. Ramakrishnan, “Feature constrained multi-task learning models for spatiotemporal event forecasting,” IEEE Trans. Knowl. Data Eng., vol. 29, no. 5, pp. 1059–1072, May 2017.   
[65] J. Xu, P.-N. Tan, J. Zhou, and L. Luo, “Online multi-task learning framework for ensemble forecasting,” IEEE Trans. Knowl. Data Eng., vol. 29, no. 6, pp. 1268–1280, Jun. 2017.   
[66] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, “Adaptive mixtures of local experts,” Neural Comput., vol. 3, no. 1, pp. 79–87, 1991.   
[67] H. Liu, F. Hussain, C. L. Tan, and M. Dash, “Discretization: An enabling technique,” Data Mining Knowl. Discov., vol. 6, no. 4, pp. 393–423, 2002.   
[68] V. Franc, O. Fikar, K. Bartos, and M. Sofka, “Learning data discretization via convex optimization,” Mach. Learn., vol. 107, no. 2, pp. 333–355, 2018.   
[69] Q. Liu, Z. Liu, and H. Zhang, “An empirical study on feature discretization,” 2020, arXiv:2004.12602.   
[70] O. Chapelle, E. Manavoglu, and R. Rosales, “Simple and scalable response prediction for display advertising,” ACM Trans. Intell. Syst. Technol., vol. 5, no. 4, pp. 1–34, 2014.

![](images/9e827c25930707087107fd9dcabb289d46f2376f2980cd0ceb986eb7632bb66c.jpg)

Qiang Liu (Member, IEEE) received the PhD degree in pattern recognition from the Chinese Academy of Sciences (CASIA). He is currently an assistant researcher with the Center for Research on Intelligent Perception and Computing (CRIPAC), Institute of Automation, Chinese Academy of Sciences. He has authored or coauthored more than 30 papers in top-tier journals and conferences, such as IEEE TKDE, AAAI, IJCAI, NeurIPS, WWW, SIGIR, CIKM, and ICDM. His current research interests include data mining, recommender systems, text mining, knowledge graph, graph representation learning, and causal inference.

![](images/7b3b51834e9ff0b70705426dfa8a9290d7acffe9cb8f7dc20109064642362f86.jpg)

Yingtao Luo is currently working toward the PhD degree in information system with Carnegie Mellon University. His research interests include data mining, machine learning, and causal inference. He has authored or coauthored papers in conferences, such as WWW and KDD. He is a reviewer of NeurIPS and ICLR.

![](images/2408ec9ab47ecfa07bf0251e417f68a96f66dd206cab13c7ecb431398112ce6a.jpg)

Shu Wu (Senior Member, IEEE) is currently an associate professor with the Center for Research on Intelligent Perception and Computing. He has authored or coauthored more than 50 papers in the areas of data mining and information retrieval at international journals and conferences, such as IEEE TKDE, WWW, AAAI, SIGIR, and ICDM.

Zhen Zhang received the PhD degree in engineering from Zhejiang University in 2014. He is currently a machine learning engineer with Ant Group’s Risk Management Department. His research focuses on applications on machine learning algorithms in risk management problems.

![](images/4ca994e18d5896d79d2876fe0141489e01de41ae1d68411146cffed23126a2ba.jpg)

![](images/8f91a8bea82acf5ae3de53d5e9c70e353d997aedb7eaddd2fb991e33f144face.jpg)

Xiangnan Yue received the master’s degree with Ecole Nationale Sup erieure des Mines de Paris in 2018. He is currently a machine learning engineer with Risk Management Department of Ant Group. Currently he focuses on industrial applications of machine learning algorithms.

![](images/00d8c7e8b3b0c39f6899efc529ecc12c2e078a766b0f20e7c40596424a9f9632.jpg)

Hong Jin received the master’s degree in statistics from McMaster University. He is currently in charge of Modeling Team, Risk Management Department of Ant Group, focusing on using machine learning/ deep learning algorithms for fraud detection. He is one of the key members who lead the team to develop intelligent algorithms of the risk management engine - AlphaRisk.

![](images/06e7660a0b98e48deec3c333c246dcd0ef5353e0252f5824ec50d85c0312b65e.jpg)

Liang Wang (Fellow, IEEE) received the BEng and MEng degrees from Anhui University in 1997 and 2000, respectively, and the Ph.D. degree from the Institute of Automation, Chinese Academy of Sciences (CASIA) in 2004. He is currently a full professor of the Hundred Talents Program with the National Lab of Pattern Recognition, CASIA. His main research interests include machine learning, pattern recognition, and computer vision. He has widely authored or coauthored in highly ranked international journals, such as IEEE TPAMI, IEEE

TKDE, and IEEE TIP, and leading international conferences, such as CVPR, ICCVand ICDM. He is an IAPR Fellow.

$\vartriangleright$ For more information on this or any other computing topic, please visit our Digital Library at www.computer.org/csdl.