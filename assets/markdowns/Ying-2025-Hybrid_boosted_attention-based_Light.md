# Hybrid boosted attention-based LightGBM framework for enhanced credit risk assessment in digital finance

Chengwei Ying1 , Anlu Shi2 & Xiongyi Li3✉

In the rapidly evolving landscape of digital finance, accurate credit risk assessment is critical for mitigating financial risks and improving lending decision-making. Traditional credit scoring methods often struggle with high-dimensional data, class imbalance, and limited interpretability, reducing their effectiveness in predicting borrower default risks. To address these challenges, this study proposes an enhanced Hybrid Boosted Attention-based LightGBM (HBA-LGBM) framework. The HBA-LGBM model introduces four key innovations: (1) a multistage feature selection mechanism that dynamically filters key borrower attributes; (2) an attention-based feature enhancement layer, which prioritizes critical financial risk factors dynamically based on contextual importance; (3) a hybrid boosting strategy, integrating LightGBM with an adaptive neural network, enabling the model to capture complex borrower behavior and non-linear credit risk patterns; and (4) an advanced imbalanced learning strategy, combining synthetic data augmentation and cost-sensitive learning to mitigate class imbalance and enhance minority class predictions. To evaluate the effectiveness of HBALGBM, experiments were conducted using a large-scale LendingClub online loan dataset. The model was compared with five state-of-the-art methods. The results demonstrate that HBALGBM achieves the lowest RMSE (11.53) and MAPE $( 4 . 4 4 \% )$ ), with an ${ \mathsf { R } } ^ { 2 }$ score of 0.998, outperforming deep learning and ensemble-based approaches. The model’s superior performance is attributed to its ability to adaptively refine borrower risk assessment, effectively balance computational efficiency with model interpretability, and provide a robust, scalable solution for digital finance applications. This research contributes to the advancement of hybrid machine learning techniques in financial risk management, offering an effective and interpretable approach to credit risk evaluation in digital lending platforms.

# Introduction

n the context of the rapid development of the Internet, digital finance has always attracted much attention. From a global perspective, governments, financial institutions and Internet enterprises of all countries are involved in the field of digital finance. For instance, the Chinese government has promoted the use of Digital Currency Electronic Payment (DCEP) to advance its digital currency ecosystem, improving transaction efficiency and financial inclusion (Liu et al., 2024). The World Bank emphasizes both the transformative potential and the regulatory challenges of digital finance, urging stakeholders to adopt balanced strategies that maximize benefits while mitigating associated risks.

Digital finance refers to financial services provided via digital channels, including mobile payments, internet banking, and digital lending platforms. It integrates financial services with digital technologies such as big data, artificial intelligence, and blockchain, covering services like mobile wallets, peer-to-peer lending, and central bank digital currencies (Song et al., 2025). With its core advantages of low cost, low threshold and strong coverage, digital finance is deeply affecting the traditional financial system and playing a positive role in invigorating the financial system, improving service efficiency and innovating business types. However, due to the existence of objective factors such as the weak awareness of the digital financial risk, the low level of risk control technology of financial institutions, the imperfect regulatory system of regulatory institutions, and the imperfect market credit system and credit reporting system, digital financial risks have become more and more complex. Such risks include cybersecurity threats, personal data privacy breaches, digital fraud, operational risks from technological failures, and regulatory compliance issues (Song et al., 2025). International regulators first paid attention to the issue of digital financial supervision. The Basel Committee on banking supervision (BCBS) mentioned in the regulatory behavior of institutions related to digital finance that the increasingly active innovation of digital financial products makes banking supervision face new challenges (Shpachuk and Trinh, 2024). Some innovative models of digital finance will increase conventional risks, such as credit risk, operational risk, liquidity risk, etc., and will also give birth to new risks such as customer information leakage and digital divide (Ji et al., 2022). Therefore, it is very important to evaluate and forecast the credit risk of users scientifically and effectively.

With the rapid development of computer technology, digital financial data began to accumulate explosively (Liu et al., 2024). The maturing big data technology and the rapidly rising machine learning algorithm can grade users’ credit from a variety of channels and angles, so they are widely used in the design of digital financial credit risk models. Lowry et al. (2023) utilizes transaction data from the past six months on the specified platform to construct a logistic regression model, concluding that personal information significantly influences a borrower’s likelihood of repayment. In contemporary digital lending environments, personal information—including demographic details, employment status, financial history, and behavioral transaction data—is increasingly integrated into advanced machine learning models. With the help of a logistic regression model and empirical analysis, Ghamry and Shamma (2022) concluded that among the factors affecting the overdue behavior of customers, the interest rate level has the largest impact on whether customers have overdue behavior. Madaan et al. (2021) constructed models based on several methods such as decision tree, logistic regression, neural network, SVM and random forest, and compared them with the credit data. The results showed that random forest had a better fitting effect. Zhang and Yu (2024) uses a combination of weak classifiers to build a credit risk model, but this method reduces the interpretability of the model. Ma et al. (2021) Built a classification model based on random forest to evaluate and control credit risk in social lending. They used random forest algorithm, SVM, KNN and other models to model respectively, and proved that it was better than FICO credit assessment in identifying the default risk of applicants.

However, the above algorithm has many disadvantages: first, we have no way to know the internal structure, and the classification effect can only be determined by adjusting parameters, which has great randomness and contingency; Secondly, we can’t cope with the problem of too much sample noise, and in this case, the model often fails. In this study, the credit database released by Lending-club, the largest online loan platform in the United States, is taken as a data sample. We use lightgbm algorithm to establish a credit risk prediction model, which has achieved better results in online loan risk prediction and also provided a good foundation for default prediction and credit rating.

While existing studies have applied various machine learning techniques to credit risk modeling, many have focused primarily on prediction accuracy without addressing key issues such as feature selection efficiency, interpretability, and real-time adaptability. Moreover, studies relying solely on commonly used models like logistic regression or standard boosting methods fail to capture the complexities of borrower behavior and evolving financial risks. Given the widespread application of boosting models in financial risk assessment, comparing LightGBM with XGBoost and CatBoost provides valuable insights into model selection under different data conditions, addressing a gap in current financial machine learning research. Addressing these gaps, this study proposes an innovative approach that integrates an enhanced Light Gradient Boosting Machine (LightGBM) framework with an optimized model architecture to improve credit risk assessment in digital finance. The main research objectives of this paper are as follows:

(1) Develop an optimized LightGBM-based credit risk assessment model to improve computational efficiency and predictive accuracy.   
(2) Enhance feature selection and model interpretability to ensure better risk assessment and decision-making.   
(3) Address data imbalance challenges using synthetic data augmentation and cost-sensitive learning to improve risk prediction fairness.   
(4) Validate the proposed framework on a large-scale online lending dataset to ensure robustness and practical applicability in digital finance.

The innovations are as follows:

(1) Introduce a multi-stage feature selection mechanism that dynamically filters critical borrower attributes to enhance model interpretability and efficiency.   
(2) Implement an attention-based feature enhancement layer to prioritize key financial risk factors dynamically.   
(3) Develop a hybrid boosting mechanism that integrates LightGBM with an adaptive neural network to better capture complex borrower behaviors.   
(4) Utilize an advanced imbalanced learning strategy combining synthetic data augmentation and cost-sensitive learning to mitigate class imbalance issues.

The remainder of this paper is organized as follows: Section “Related works” reviews related works; Section “Construction of credit risk assessment model” details the proposed model; Section “Experiments and analysis” presents experimental results and analysis; Section “Conclusions” concludes the study and outlines future research directions.

# Related works

The traditional economic statistical evaluation method assesses relevant indicators based on expert judgment, relying on the experience and expertise of senior credit managers in financial institutions. The decision to approve a loan is primarily determined by these professionals, who apply their industry knowledge and professional intuition. This approach is inherently subjective, as it depends on individual judgment rather than objective, datadriven analysis. Several well-known traditional credit evaluation frameworks include the $5 \mathrm { C }$ method (Character, Capacity, Capital, Collateral, and Condition) (Oseni, 2023), the 5 W method (Who—the borrower, Why—the purpose of the loan, When—the repayment term, What—the collateral, and How—the repayment method), and the $5 \mathrm { P }$ method (Personal, Purpose, Payment, Protection, and Perspective) (Xie et al., 2024). However, despite their widespread use, these methods have significant drawbacks. Their reliance on subjective assessment increases the risk of inconsistency and potential bias, making them susceptible to manipulation or errors in judgment. As a result, their effectiveness in modern credit risk analysis has diminished, particularly in an era where advanced data-driven models offer greater accuracy, transparency, and scalability.

With the progress and development of computer technology, typical statistical and operational research evaluation methods in the field of credit scoring, such as linear regression, logical regression and linear programming (Wu and Pan, 2021), have emerged, gradually overcoming the strong subjectivity of traditional economic statistical evaluation methods. Linear regression generally uses the least square method for fitting and modeling, and observes the relationship between each independent variable and dependent variable. This method is also widely used in credit evaluation. For example, Sreesouthry et al. (2021) evaluated the credit quality of credit loan applicants through linear regression. Logistic regression is an algorithm to solve the classification problem. In the credit score, the probability that a sample belongs to good customers or bad customers can be predicted. For example, Ko et al. used logistic regression to predict the default probability of P2P online loan borrower. The empirical results show that the prediction accuracy of this model meets the actual needs, and the probability of making the second type of error is less than $1 \%$ , which can greatly reduce the risk of investors. Maldonado et al. (2017) proposed a method of using linear support vector machines to build classifiers, which was verified on the credit scoring problem of a bank in Chile and found that its method achieved excellent performance in terms of business objectives. Cai and Zhang (2020) combined decision tree and logistic regression algorithm to evaluate the borrower’s credit risk. Nandipati and Boddala (2024) studied the performance of two advanced data mining technologies, random forest and support vector machine, as well as logistic regression algorithm, for detecting credit card fraud based on the real credit card transaction data set.

Recent studies have explored sophisticated methodologies for handling incomplete data in risk modeling and financial forecasting. For instance, Kostecka and Ślepaczuk (2024) provide an in-depth analysis of how XGBoost can be leveraged for realized Loss Given Default (LGD) approximation, demonstrating advanced data preprocessing techniques to mitigate missing values and enhance predictive accuracy. Their work underscores the importance of robust data imputation strategies in credit risk assessment, particularly in scenarios where incomplete datasets may otherwise compromise model reliability.

Furthermore, integrating machine learning with portfolio optimization has gained increasing attention in recent literature. Wysocki and Sakowski (2022), as well as Ślusarczyk and Ślepaczuk (2023), illustrate how ML-driven forecasting models can directly inform Markowitz-style portfolio selection, enhancing asset allocation strategies through improved risk-return tradeoffs. These studies highlight how data-driven approaches can refine traditional financial models, offering a more dynamic and responsive framework for investment decision-making. By incorporating predictive analytics into portfolio optimization, such methodologies not only improve return predictability but also introduce adaptive mechanisms that adjust to evolving market conditions.

In recent years, numerous scholars have applied artificial intelligence technologies to evaluate the creditworthiness of loan applicants, leveraging widely used machine learning algorithms such as RF, Gradient Boosting Decision Trees (GBDT), and XGBoost (Vyshnavi et al., 2024; Varshney et al., 2024). These approaches have significantly improved credit risk evaluation capabilities. The RF consists of multiple classification and regression decision trees. Each tree is trained using randomly selected subsets of the data, and classification is determined through a majority voting mechanism. However, despite its robustness and ability to mitigate overfitting through bootstrap aggregation, RF often struggles with interpretability and computational efficiency when handling high-dimensional financial data. The GBDT algorithm enhances predictive performance by iteratively constructing decision trees, where each successive tree corrects residual errors from previous iterations (Ma et al., 2023). The final prediction is derived from the cumulative sum of all trees. Zhang et al. (2024) proposed a decision tree for credit score promotion based on GBDT algorithm. While GBDT improves model accuracy compared to traditional decision trees, it relies heavily on sequential tree-building, making it computationally expensive for large-scale credit datasets. The XGBoost algorithm extends GBDT by incorporating L1 and L2 regularization to balance model complexity and accuracy, effectively preventing overfitting. Additionally, XGBoost supports column sampling and parallelized computations, improving both generalization ability and efficiency. Due to these advantages, XGBoost has been widely adopted in credit risk modeling. For instance, Li et al. (2019) applied XGBoost and RF to predict P2P default risk, while Zou and Gao (2022) compared XGBoost with RF and GBDT for financial credit scoring, demonstrating its superior predictive performance. Moreover, Wang et al. (2022) utilized XGBoost to develop a customer application scoring model for banking institutions, assessing default probabilities for new clients.

However, while XGBoost has been extensively studied, its dominance in credit risk assessment is increasingly challenged by newer models such as CatBoost and LightGBM. CatBoost, introduced by Alsulamy (2025), is designed to handle categorical data more effectively by implementing an innovative ordered boosting approach, which prevents data leakage and improves model stability. Studies suggest that CatBoost outperforms XGBoost in datasets with a high proportion of categorical variables, a common scenario in credit risk assessment (Hancock and Khoshgoftaar, 2020). Similarly, LightGBM, proposed by Ke et al. (2017), optimizes gradient boosting through a leaf-wise growth strategy, significantly reducing training time while maintaining high accuracy in large-scale datasets. Despite these advancements, comparative studies evaluating these models’ effectiveness in credit risk assessment remain limited. Several recent studies have attempted to bridge this gap. For example, Baser et al. (2023) developed a credit risk assessment model based on XGBoost, yet did not benchmark it against LightGBM or CatBoost, raising concerns about whether the selected model was truly optimal. Similarly, Li et al. (2020) proposed a personal credit scoring method using GBDT but did not evaluate its relative performance against XGBoost or other boosting frameworks. Given the increasing complexity of financial datasets, a more rigorous comparative analysis of LightGBM, XGBoost, and CatBoost is necessary to determine the most suitable model for different credit risk scenarios.

![](images/64ee8252d17872639c7a1b2940a6b7f78d840aba425d49b9cc97f1862f82b0a5.jpg)  
Fig. 1 Schematic diagram of histogram algorithm.   
Fig. 2 Histogram deviation (Mahmood et al., 2022).

Additionally, Babaei et al. (2023) explored explainable fintech lending using tree-based models, while Babaei and Bamdad (2023) applied various credit scoring methods in P2P lending platforms, emphasizing model transparency and decision support. These studies highlight the relevance of tree-based methods, yet differ from our proposed hybrid boosting and attention mechanisms, which further enhance both performance and interpretability. While machine learning techniques have substantially improved credit risk evaluation, the current body of literature lacks a thorough benchmarking of LightGBM against XGBoost and CatBoost in financial applications. Future research should focus on systematic comparisons of these models under various financial conditions, including stressed economic scenarios, imbalanced datasets, and high-dimensional feature spaces, to establish a more definitive understanding of their relative advantages in credit risk assessment.

The prices of financial products, such as options, stock prices, funds and financial derivatives, are subject to fluctuations of many factors (Tripathi et al., 2021). This is easy to build model training through machine learning methods such as supervised learning, including the development of mathematical analytical models using some ensemble approaches, proactive approaches, etc. (Essamlali et al., 2024; Carta et al., 2020) However, the above models often have limitations such as insufficient operation speed and easy over fitting, and most of the studies are simple use of computer algorithms or technologies, and do not fully clean, explore and analyze the data. The LightGBM algorithm can solve the problems encountered by the GBDT algorithm in processing high-dimensional massive data, so that the gradient lifting fitting method is more efficient in actual combat. Finally, the model is tested and verified, and the function of the model is analyzed.

# Construction of credit risk assessment model

Enhanced feature selection for lightGBM algorithm. In general decision tree algorithm, it usually takes a lot of time to find the optimal split point, because it needs to traverse all samples every time, which requires a lot of time and memory. The efficiency is not high when the samples are large and the eigenvalues are complex. In view of this, LightGBM adopts histogram algorithm. The schematic diagram of histogram algorithm is shown in Fig. 1. In this way, when finding the optimal split point in the decision tree, the histogram replaces the original sample and only needs to traverse all histograms. This can greatly reduce the calculation time and make the model more stable.

Another optimization of the lightGBM algorithm using histograms is the acceleration of histogram differences. The histogram difference diagram is shown in Fig. 2. General histogram construction needs to traverse all the data on this leaf node, while lightGBM’s histogram only traverses discrete K buckets to improve the operation speed (Mahmood et al., 2022). Algorithms that utilize decision trees typically adopt a levelwise growth strategy, where node gain is computed by traversing all data points to split leaf nodes within the same layer (Li, 2024). While this approach facilitates parallel processing, it is not always the most efficient. In practice, the gain associated with each leaf node varies, and for some smaller nodes, the gain may be negligible. Performing numerous unnecessary calculations on such nodes can significantly reduce computational efficiency, leading to suboptimal performance in large-scale datasets.

The schematic diagram of two leaf growth modes is shown in Fig. 3. The tree splitting in LightGBM is not traversed in order, but the maximum gain is selected by the leaf wise method to reduce the loss of the model.

In credit risk assessment, feature selection plays a crucial role in eliminating redundant or irrelevant attributes to improve model efficiency and generalizability. Traditional feature selection techniques often rely on simple heuristic measures, which can lead to suboptimal decisions. To address this, we introduce a multi-stage feature selection mechanism that dynamically adjusts attribute importance based on statistical significance and interfeature dependencies.

![](images/c9d3597fc564c97fb73534aa56428d81e684ab28d23925372857f7e18583eba3.jpg)  
Fig. 3 Schematic diagram of two leaf growth modes.

The LightGBM framework selects the optimal feature split by maximizing the information gain of a given feature j at a split point $d _ { \mathrm { { i } } }$ , calculated as:

$$
V _ { j | O } ( d ) = \frac { 1 } { n _ { 0 } } \left( \frac { \left( \sum _ { \left\{ x _ { i } \in O : x _ { i j } \le d \right\} } g _ { i } \right) ^ { 2 } } { n _ { l | O } ^ { j } ( d ) } + \frac { \left( \sum _ { \left\{ x _ { i } \in O : x _ { j } > d \right\} } g _ { i } \right) ^ { 2 } } { n _ { r | O } ^ { j } ( d ) } \right)
$$

Where $\begin{array} { r } { n _ { 0 } = \sum \mathbb { I } \left[ x _ { i } \in O \right] } \end{array}$ represents the total number of samples in node O; $n _ { l | O } ^ { j } ( d )$ denotes the number of samples where $x _ { i j } \leq d$ $n _ { r | O } ^ { j } ( d )$ denotes the number of samples where $x _ { i j } > d$ .

jInstances with larger gradients have a greater influence on the overall information gain, so we adopt a dynamic sampling method where a subset of training instances $A$ with high gradient values is selected. The maximum gain can be estimated by:

$$
V _ { j | d } = \frac { 1 } { n } \left( \sum _ { x _ { i } \in A } g _ { i } + \frac { 1 - a } { b } \sum _ { x _ { i } \in B } g _ { i } \right) ^ { 2 } \frac { 1 } { n _ { l i j } ^ { d } } + \left( \sum _ { x _ { i } \in A _ { r } } g _ { i } + \frac { 1 - a } { b } \sum _ { x _ { i } \in B _ { r } } g _ { i } \right) ^ { 2 } \frac { 1 } { n _ { r | j } ^ { d } }
$$

Where $A _ { l } , A _ { r }$ represent large-gradient instance subsets before and after split $d$ ; $B _ { l } , B _ { r }$ are small-gradient instance subsets. This weighted adjustment reduces computational overhead while improving the model’s ability to focus on informative samples.

According to the gradient and the number of samples of the left subtree, we can get the gradient and the number of samples of the right child node by making a difference with the parent node, and then traverse the gradient and the corresponding number of samples in all histograms. In the implementation process, the optimal segmentation features and segmentation points are found through the optimization methods of feature parallel and data parallel.

Dynamic feature prioritization with attention mechanism. In credit risk assessment, financial attributes often exhibit complex interdependencies. Some features may exert a dominant influence on risk prediction under certain economic conditions, while others may become more relevant as market dynamics evolve. Traditional feature selection techniques, including information gain-based approaches, assign static weights to features, potentially missing crucial time-sensitive variations. To address this limitation, we introduce an attention-based feature enhancement layer that dynamically adjusts feature importance by leveraging contextual information.

The core idea behind the attention mechanism is to allow the model to focus more on influential features while suppressing the contribution of less relevant ones. Given an input feature matrix $X ,$ we first apply a non-linear transformation (Mahmood et al., 2022):

$$
H = \operatorname { t a n h } \Bigl ( W _ { f } X + b _ { f } \Bigr )
$$

where $W _ { f }$ represents a trainable weight matrix, $b _ { f }$ is the bias term, and $H$ captures the transformed feature representation. Next, we compute attention scores to determine the relative importance of each feature:

$$
\alpha _ { j } = \frac { \exp \left( Q _ { j } K _ { j } ^ { T } / \sqrt { d _ { k } } \right) } { \sum _ { k } \exp \left( Q _ { k } K _ { k } ^ { T } / \sqrt { d _ { k } } \right) }
$$

where $Q = W _ { Q } X , K = W _ { K } X ,$ , and $V = W _ { V } X$ are query, key, and value transformations, respectively. These trainable matrices enable the model to learn feature-specific weights adaptively. The final feature representation is then computed as:

$$
X ^ { \prime } = \sum _ { j } \alpha _ { j } V _ { j }
$$

where $X ^ { \prime }$ represents the enhanced feature matrix. This process dynamically recalibrates feature weights at each iteration, allowing the model to focus on the most critical financial indicators.

The transformed feature representation $X ^ { \prime }$ is subsequently passed into the LightGBM framework. The updated decision function is:

$$
\hat { y } = \sum _ { m = 1 } ^ { M } f _ { m } ( X ^ { \prime } )
$$

where $f _ { m }$ denotes the $m$ -th decision tree. By incorporating the attention mechanism, the model can dynamically adjust to changes in borrower risk profiles, improving both predictive accuracy and interpretability.

Hybrid boosting mechanism. Tree-based methods like LightGBM rely on recursive partitioning, which may struggle to capture highly non-linear relationships in borrower behavior. To address this issue, we introduce a hybrid boosting mechanism that integrates LightGBM with an adaptive neural network component.

To complement the LightGBM model, we incorporate a multilayer perceptron (Rashedi et al., 2024) that models higher-order interactions among financial variables. The network consists of $L$ layers, with each layer represented as:

$$
\begin{array} { r } { \begin{array} { r } { h ^ { ( l ) } = \sigma \big ( W ^ { ( l ) } h ^ { ( l - 1 ) } + b ^ { ( l ) } \big ) } \end{array} } \end{array}
$$

where $W ^ { ( l ) }$ and $b ^ { ( l ) }$ are the weight and bias parameters of layer $l _ { : }$ and $\sigma$ is a non-linear activation function (e.g., ReLU). The final output is computed as:

$$
\hat { y } _ { N } = \mathrm { s o f t m a x } \big ( W _ { h } h ^ { ( L ) } + b _ { h } \big )
$$

The final prediction function combines both LightGBM and the neural network using an adaptive weighting mechanism:

$$
\hat { y } = \lambda \hat { y } _ { B } + ( 1 - \lambda ) \hat { y } _ { N }
$$

where $\lambda$ is a learnable parameter that adjusts dynamically based on the confidence level of each model. The adaptive weight is

computed as:

$$
\lambda = \frac { 1 } { 1 + e ^ { - W _ { g } X } }
$$

Ensuring an optimal balance between tree-based and deep learning components. This hybrid approach enhances robustness by leveraging LightGBM’s efficiency and the neural network’s ability to capture complex borrower behaviors.

Imbalanced learning strategy. A major challenge in credit risk assessment is the inherent imbalance in loan repayment data, where the proportion of default cases is significantly lower than non-default cases. Standard classification models tend to be biased toward the majority class, leading to poor predictive performance for high-risk borrowers. To mitigate this issue, we propose an advanced imbalanced learning strategy that combines cost-sensitive learning with synthetic data augmentation.

We introduce a re-weighted loss function:

$$
L = \sum _ { i = 1 } ^ { n } w _ { y _ { i } } \cdot \big [ y _ { i } \log \hat { y } _ { i } + \left( 1 - y _ { i } \right) \log \left( 1 - \hat { y } _ { i } \right) \big ]
$$

where $w _ { y _ { i } }$ is the class weight, computed as:

$$
w _ { y _ { i } } = \frac { 1 } { \sqrt { P ( y _ { i } ) } }
$$

This formulation assigns higher importance to minority class samples, reducing model bias.

To further improve class balance, we apply an enhanced Synthetic Minority Over-sampling Technique (SMOTE) (Zheng et al., 2024) with feature perturbation. Given a minority class instance $X _ { i } ,$ a synthetic sample is generated as:

$$
X _ { n e w } = X _ { i } + \delta \big ( X _ { i } ^ { N N } - X _ { i } \big ) + \epsilon \cdot \mathcal { N } \big ( 0 , \sigma ^ { 2 } \big )
$$

Where $X _ { i } ^ { N N }$ is the nearest neighbor of $X _ { i } ; \delta , \epsilon \sim U ( 0 , 1 )$ introduce stochastic noise for diversity; ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ represents Gaussian noise, ensuring slight variations in generated samples.

This approach enhances the diversity of synthetic instances while preserving the statistical properties of the dataset.

The final training objective is augmented with a regularization term to prevent overfitting:

$$
\mathcal { L } _ { \mathrm { f i n a l } } = L + \gamma \sum _ { i = 1 } ^ { m } \left. X _ { \mathrm { n e w } } ^ { ( i ) } - X _ { i } ^ { N N } \right. ^ { 2 }
$$

where $\gamma$ is a tuning parameter controlling the impact of synthetic data. This ensures that augmented samples contribute meaningfully without distorting underlying patterns.

Parameters of LightGBM algorithm tuning. Finally, we defined the optimized model as HBA-LGBM (Hybrid Boosted Attentionbased LightGBM for Credit Risk Assessment). The main parameters used as tuning parameters by HBA-LGBM algorithm are:

Learning_rate: the node weight is multiplied by the learning rate during calculation. The default value of the model is 0.1, which is generally between 0.05 and 0.1. Generally, choosing a smaller value can get a more stable model performance, but if the value is too small, it may be under fitted.

Maximum iterations of weak learner (n_estimators): the default value of the model is 100. The value is selected from 100 to 1000 according to the specific characteristic value. Generally speaking, the larger the number of training times, the better the accuracy of the model will be, but there may be overfitting, so it still needs to be set reasonably.

Maximum tree depth (max_depth): the most important parameter set by LightGBM algorithm to prevent overfitting, which is generally limited to 3\~5, and the maximum is generally not more than 10. It is the core parameter of the model and plays a decisive role in the performance and generalization ability.

<table><tr><td colspan="3">Table 1 Main parameters and meanings of LightGBM algorithm.</td></tr><tr><td>Variable</td><td>Description</td><td>Value</td></tr><tr><td rowspan="4">learning rate num_leaves max_depth min_data feature_fraction</td><td>Learning rate</td><td>0.1</td></tr><tr><td>Number of node leaves per tree</td><td>25</td></tr><tr><td>Maximum learning depth</td><td>8</td></tr><tr><td>Minimum number of data in leaf node Proportion of selected features in total</td><td>200 1</td></tr><tr><td>bagging_fraction</td><td>features,range[O~1] Proportion of selected data in total data, range [0~1]</td><td>0.5</td></tr></table>

Min_child_weight: the minimum number of samples on a leaf. When splitting, some point samples will stop splitting if they are smaller than this value. If the value is too large, it may cause overfitting.

Number of leaf nodes (num_leaves): refers to the number of leaf nodes in a tree. The default value of the model is 31, which is generally different from max epth cooperates to control the shape of the tree. The setting range is generally the value before $( 0 , \bar { 2 } ^ { \mathrm { m a x } _ { \mathrm { d } } \mathrm { e p t } h } - 1 )$ . The Main parameters and meanings of HBALGBM is shown in Table 1.

# Experiments and analysis

Dataset and preprocessing. The data set is selected from the public data set released by the online lending pioneer Lendingclub. Founded in 2006, the Lending-club is currently the world’s largest P2P lending platform company, headquartered in San Francisco, California. The fields contained in the database include the time of customer loan, annual income, loan interest rate, debt to income ratio, bank card credit line, loan term, loan amount, loan status, maximum credit line, month of opening circular account, etc.

This study uses HBA-LGBM algorithm to establish a credit risk assessment model. The specific steps are as follows: first, we perform data processing on the public data set to eliminate the impact of uncontrollable variables on the evaluation results. Then, we randomly divide the training set into a sample set and a verification set according to a certain proportion. In this study, the sample set was randomly divided into five-fold cross validation to validate the current training results. Next, we divide the floating-point numbers of the data imported into the algorithm model into buckets, process the data in the form of histograms, and generate a cart regression tree model. Then, we use the forward distribution algorithm. The training results of the previous step will affect the next step. Then we carry out iterative training. Finally, we will use the verification set to verify the obtained training model and output the verification results.

In the first training, the default parameters are generally selected first, and the evaluation results of the model trained according to the default parameters are further adjusted. At the end of the training, the model evaluation index can be output; According to the training results of the original parameters, the parameters of the model are optimized until the model with the best fitting degree is finally obtained.

This study uses the global online lending platform lending club to release the lending data from 2007 to 2020, focusing on the settlement and write off stages in the whole project process. The total number of loan data is 1040,398, of which 852,715 are settled normally, accounting for $7 7 . 9 9 \%$ ; There were 187683 write off data, accounting for $2 2 . 0 1 \%$ .

First, data cleaning was performed to reduce the impact of uncontrollable variables on the assessment results while retaining as much survey data as possible for analysis to improve the reliability of the risk assessment. The data cleaning process is as follows:

(1) 31 feature variables with missing values of more than $5 0 \%$ and 22 null feature variables were deleted. Since the focus of this study is on the borrower’s repayment behavior, we have appropriately deleted a series of unnecessary features in the data collation stage.   
(2) Delete the samples with too many missing features, and finally 979,818 samples remain.   
(3) In this study, we selected 200,000 samples as the test set df_test, the training set df_train contains the remaining 779,818 samples.

Additionally, to reduce selection bias and improve data completeness, this study employed Iterative Imputation, an advanced technique for handling missing data. The process follows these key steps:

(1) Initialize missing values: Missing values were initially filled using mean, median, or most frequent values.   
(2) Train predictive models: For each feature with missing values, the other available features were used as input variables to train regression model to predict the missing values.   
(3) Iterative filling: Missing feature values were updated iteratively, using previously filled values as inputs, until the model converged to a stable solution.   
(4) Multiple imputations: The process was repeated multiple times to minimize bias from a single imputation and obtain a more robust estimation of missing values.

In this study, Scikit-learn’s IterativeImputer was applied to leverage the correlation between features, thereby improving data integrity and mitigating the negative effects of missing data.

Evaluation index. The identification results of the model by four evaluation methods are RMSE (root mean square error), MAE (mean absolute error) and $\mathtt { R } ^ { 2 }$ (coefficient of determination), which are common evaluation methods.

The expression for calculating RMSE is:

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { \mathrm { n } } \sum _ { \mathrm { i = 1 } } ^ { \mathrm { n } } \left( \mathrm { y } _ { \mathrm { i } } - \hat { \mathrm { y } _ { \mathrm { i } } } \right) ^ { 2 } }
$$

In this formula, $\hat { \mathsf { y } } _ { \mathrm { i } }$ is the identification value and $\mathrm { y _ { i } }$ is the actual value. The smaller the RMSE obtained through calculation means the higher the degree of algorithm fitting, and the better the effect of prediction. When RMSE is equal to 0, it means that the predicted value and the actual value are completely consistent, and the identification effect is the best. Usually, the RMSE is used to keep the error description at a certain order of magnitude.

The square of RMSE is the MSE, which is usually used as the loss function of linear regression. The expression for calculating MSE is:

$$
\mathrm { { M S E } = \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y _ { i } } ) \right| }
$$

In this equation, $\hat { \mathsf { y } } _ { \mathrm { i } }$ is the identification value and $\mathrm { y _ { i } }$ is the actual value. The smaller the MSE obtained by calculation, the higher the degree of fitting of the algorithm, and the better the predicted result of the algorithm. When MSE is 0, the predicted value of the algorithm completely conforms to the actual value.

When the dimensions of dependent variable and independent variable are different, it may be difficult to judge the quality of model recognition by using RMSE, MSE and MAPE. Using $\mathrm { R } ^ { 2 }$ for evaluation can reduce this effect. The expression for calculating $\mathtt { R } ^ { 2 }$ is:

$$
\mathrm { R } ^ { 2 } = 1 - \frac { \mathrm { S S E } } { \mathrm { S S T } }
$$

Where $\mathrm { S S R } = \sum _ { \mathrm { i = 1 } } ^ { \mathrm { n } } \left( \hat { \mathrm { y _ { i } } } - \mathrm { y ^ { 2 } } \right) ^ { 2 }$ is the sum of regression squares, ${ \mathrm { S S E } } = \sum _ { \mathrm { i = 1 } } ^ { \mathrm { n } } \left( \mathrm { y } _ { \mathrm { i } } - { \hat { \mathrm { y } _ { \mathrm { i } } } } \right) ^ { 2 }$ is the sum of squares of residuals, ${ \mathrm { S S T } } =$ ${ \mathrm { S S R } } + { \mathrm { S S E } } = \sum _ { \mathrm { I } = 1 } ^ { \mathrm { N } } \left( \mathrm { y } _ { \mathrm { i } } - \mathrm { y } ^ { 2 } \right) ^ { 2 }$ is the sum of squares of total deviations.

Model training process. To further analyze the performance and convergence behavior of HBA-LGBM, we visualize the error rate and loss rate for both the training and test sets over multiple epochs. The following Figs. 4 and 5 present these trends for 100 training epochs, where distinct line styles differentiate between training and test sets.

It can be observed that the training error rate drops below $5 \%$ (0.05) at approximately epoch 45, while the test error stabilizes around $4 \%$ (0.04) by epoch 55. This suggests that the model effectively generalizes without excessive overfitting. Similarly, the loss rate for the training set falls below 0.1 by epoch 50, and the test loss follows closely, stabilizing near 0.1 around epoch 60. Beyond this point, further training yields diminishing returns, indicating that HBA-LGBM converges effectively within 60 epochs. This fast convergence demonstrates the efficiency of the model’s hybrid boosting strategy, which accelerates learning through dynamic feature selection and attention-based enhancements, ensuring both computational efficiency and predictive accuracy.

Model comparison. To demonstrate the effectiveness of HBALGBM, we compare it with five state-of-the-art models from recent literature, each representing a different approach to credit risk assessment. The models selected for comparison include treebased ensemble methods, deep learning techniques, and hybrid frameworks. The adopted comparison model and description are given below

(1) XGBoost with Bayesian Hyperparameter Optimization (XGB-Bayesian) (Noviandy et al., 2024): This model applies Bayesian optimization to improve hyperparameter selection in XGBoost, enhancing predictive accuracy and robustness.   
(2) Stacked Autoencoder with Gradient Boosting Decision Trees (SAE-GBDT) (Ginavanee, 2024): This hybrid model leverages stacked autoencoders for feature extraction and then applies GBDT for final prediction, improving feature learning capabilities.   
(3) Transformer-based Credit Risk Model (TCRM) (Korangi et al., 2023): A transformer-based approach that utilizes attention mechanisms to model sequential borrower data and capture temporal dependencies in credit risk profiles.   
(4) Hybrid CNN-LSTM for Credit Risk Forecasting (HCLCredit) (Yao et al., 2024): A deep learning framework combining convolutional neural networks (CNNs) for feature extraction and long short-term memory (LSTM) networks for sequential learning, suited for time-dependent financial data.

![](images/a164c3dbdfcd1756dd592d906cfb60a4817bd88d25939bfca35dbeac6b3ddcdd.jpg)  
Fig. 4 Change of error rate of training set and test set.

![](images/4af2a8c7114ac3327c2183ca2bd4b84aa95c3c1088461b97cdf90945bf619aa3.jpg)  
Fig. 5 Change of loss rate of training set and test set.

<table><tr><td colspan="4">Table 2 Model comparison results.</td></tr><tr><td>Model</td><td>RMSE</td><td>MSE</td><td>R²</td><td>RGA</td></tr><tr><td>HBA-LGBM</td><td>11.53</td><td>133.07</td><td>0.998</td><td>0.962</td></tr><tr><td>XGB-Bayesian</td><td>14.02</td><td>196.56</td><td>0.987</td><td>0.925</td></tr><tr><td>SAE-GBDT</td><td>13.54</td><td>183.39</td><td>0.991</td><td>0.931</td></tr><tr><td>TCRM</td><td>12.78</td><td>163.42</td><td>0.993</td><td>0.940</td></tr><tr><td>HCL-Credit</td><td>12.39</td><td>153.48</td><td>0.994</td><td>0.946</td></tr><tr><td>CSRF-SMOTE</td><td>13.11</td><td>172.65</td><td>0.992</td><td>0.938</td></tr></table>

(5) Cost-Sensitive Random Forest with SMOTE (CSRFSMOTE) (Kavitha and Kasthuri, 2024): A cost-sensitive random forest model that incorporates SMOTE to address imbalanced class distribution in credit datasets.

Each model is trained using five-fold cross-validation, and hyperparameters are optimized using Bayesian optimization for tree-based models and Adam optimizer with adaptive learning rates for deep learning models. The results are shown in Table 2.

Table 2 demonstrates the superior performance of HBA-LGBM compared to five state-of-the-art models in credit risk prediction. The selected baseline models, including XGB-Bayesian, SAEGBDT, TCRM, HCL-Credit, and CSRF-SMOTE, represent recent advancements in ensemble learning, deep feature extraction, attention-based modeling, sequential learning, and class imbalance handling. The results indicate that HBA-LGBM achieves the lowest RMSE (11.53) and MSE (133.07), confirming its superior predictive accuracy and its effectiveness in minimizing relative error. The model’s $\mathtt { R } ^ { 2 }$ score of 0.998 suggests that it captures nearly all variance in borrower default prediction, outperforming deep learning and ensemble-based approaches. The improvements are attributed to HBA-LGBM’s dynamic feature selection mechanism, which refines borrower attributes using an attentionbased enhancement layer, and its hybrid boosting framework, which integrates LightGBM with an adaptive neural network to model complex financial patterns.

Furthermore, the incorporation of cost-sensitive learning with synthetic data augmentation effectively addresses imbalanced class distribution, a challenge that traditional models such as XGB-Bayesian and SAE-GBDT struggle to mitigate. While TCRM and HCL-Credit leverage deep learning for sequential modeling, they exhibit slightly higher errors, highlighting the computational efficiency and generalization capability of HBALGBM. The results confirm that HBA-LGBM offers a robust, interpretable, and computationally efficient solution for credit risk assessment, outperforming existing techniques across multiple evaluation metrics.

To address the classification nature of credit risk assessment, we also employ the Rank Graduation Accuracy (RGA) metric, a generalized form of AUC suitable for both binary and multi-class problems (Giudici, 2024).

Considering that credit risk assessment inherently involves classification tasks—such as distinguishing between different risk levels or default probabilities—we incorporate the Rank

![](images/bd6f1d8a73906b8ad1c48ee2b31ce548305a506eaa341108d1f41bef034f1e03.jpg)  
Fig. 6 Confusion matrices.

Graduation Accuracy (RGA) as an additional evaluation metric. RGA extends the concept of the Area Under the Curve (AUC), making it suitable not only for binary classification but also for multi-class or ordinal classification scenarios. It evaluates a model’ s ability to rank instances according to their true risk levels, providing a more nuanced assessment of model performance in ranking-based tasks. As shown in Table 2, the proposed HBA-LGBM model achieves the highest RGA score of 0.962, further confirming its superior capability to accurately differentiate and rank credit risk levels compared to other benchmark models.

The confusion matrices in Fig. 6 compare HBA-LGBM with five benchmark models in the credit risk classification task. Overall, all models exhibit high accuracy in predicting nondefault cases due to the dataset’s class imbalance, while default cases are more challenging to identify. HBA-LGBM performs well in classifying non-default instances but, like most models, struggles with recall in the default category. XGB-Bayesian and TCRM show similar classification trends, with slightly improved recall for default cases at the cost of increased false positives. CSRF-SMOTE, leveraging synthetic data augmentation, achieves the best recall for default cases, demonstrating the benefits of handling class imbalance more effectively, though it may introduce a higher false positive rate.

The ablation study evaluates the impact of different components of HBA-LGBM by selectively removing key features, as shown in Table 3.

(The ablation study results provide a comprehensive evaluation of the HBA-LGBM model by selectively removing key components and assessing their impact on predictive performance. The findings indicate that removing the attention mechanism results in the most significant degradation, increasing RMSE by $2 3 . 2 \%$ and lowering $\mathtt { R } ^ { 2 }$ to 0.987, confirming that dynamic feature selection is essential for accurate risk prediction. Similarly, excluding hybrid boosting increases RMSE to 13.75, demonstrating that the integration of LightGBM with neural networks significantly enhances learning efficiency.

<table><tr><td colspan="5">Table 3 Performance metrics across different model variants.</td></tr><tr><td>Model Variant</td><td>RMSE</td><td>MSE</td><td>MAPE</td><td>R²</td></tr><tr><td>Full HBA-LGBM Without Attention Mechanism</td><td>11.53</td><td>133.07</td><td>4.44%</td><td>0.998</td></tr><tr><td></td><td>14.21 13.75</td><td>202.35 189.50</td><td>6.10%</td><td>0.987</td></tr><tr><td>Without Hybrid Boosting</td><td>14.03</td><td>195.80</td><td>5.85%</td><td>0.990</td></tr><tr><td>Without Cost-Sensitive Learning</td><td>13.90</td><td>192.40</td><td>5.95%</td><td>0.988</td></tr><tr><td>Without Synthetic Data Augmentation</td><td></td><td></td><td>5.88%</td><td>0.989</td></tr></table>

The absence of cost-sensitive learning and data balancing negatively impacts minority class predictions, leading to biased results. Additionally, excluding LightGBM optimization and neural network integration results in decreased generalization ability, highlighting the contribution of gradient-based learning and deep neural networks to model robustness. These results validate that each component within HBA-LGBM plays a critical role, ensuring that the model effectively captures complex borrower risk patterns while maintaining high predictive accuracy and generalizability.

Credit risk assessment. Figures 7 and 8 illustrate the 3D decision boundaries of HBA-LGBM, showcasing how the model predicts default probability based on different borrower financial attributes.

![](images/e7566c6f794686fff712e585c08d63f182844928b9690e98cc81109e46a0b21a.jpg)  
Fig. 7 3D Decision Boundary of HBA-LGBM Based on Credit Score and Loan Amount.

![](images/ea478dbb9c6ef31877f946b4e692aa998147746e15acc5dfab05c683e7b93f77.jpg)  
Fig. 8 3D Decision Boundary of HBA-LGBM Based on Annual Income and Debt-to-Income Ratio.

Figure 7 presents the default probability as a function of Credit Score (X-axis) and Loan Amount (Y-axis). The decision boundary demonstrates a non-linear decline, where borrowers with high credit scores and lower loan amounts exhibit lower default risks (blue region), while those with lower credit scores and higher loan amounts face significantly higher default probabilities (red region). The smooth gradient transition further confirms the model’s ability to capture risk variations dynamically.

Figure 8 provides an alternative perspective, mapping default probability against Annual Income (X-axis) and Debt-to-Income Ratio (Y-axis). This visualization emphasizes that borrowers with higher annual incomes and lower DTI ratios are deemed low risk, while individuals with high debt-to-income ratios and lower incomes show an increased probability of default. The observed boundary confirms that HBA-LGBM effectively learns risk dependencies across various financial indicators, reinforcing its robust feature selection mechanism.

Figure 9 illustrates the top ten most influential variables in the HBA-LGBM model, ranked by their contribution to the overall predictive performance. The highest-ranking variable exhibits a significantly greater impact compared to the remaining attributes, indicating its dominant role in shaping the model’s credit risk assessment decisions. The next few variables also show relatively high contributions, suggesting their strong influence in predicting default probabilities. The latter variables exhibit more uniform and lower contributions, signifying their secondary importance in the decision-making process.

Among them, the loan occurrence time has the highest contribution to LightGBM cube, followed by loan interest rate, debt to income ratio, bank card credit line, etc. Among the factors affecting risk assessment, loan elements account for the highest proportion. This shows that the product element design of the loan has a great impact on whether the borrower can perform the contract, and the current personal status has the smallest impact. In general, the four types of influencing factors from high to low are loan factors, economic status, credit status and personal status.

![](images/975c2c06607df899ba84240eddd26a3a72475974ff028856f24b4f3b43934ccd.jpg)  
Fig. 9 The top ten contribution variables in the cube and their contribution ranking.

This ranking underscores the effectiveness of the multi-stage feature selection and attention-based enhancement mechanisms within HBA-LGBM, ensuring that the most relevant borrower attributes receive priority during training. By identifying the dominant financial indicators, this analysis further validates the interpretability and reliability of HBALGBM, making it a practical tool for real-world credit risk assessment applications.

Discussion. The experimental results of HBA-LGBM provide significant insights into credit risk assessment, demonstrating both methodological advancements and theoretical contributions. The findings indicate that the attention-based feature enhancement and hybrid boosting mechanisms substantially improve predictive performance, as evidenced by lower RMSE and MSE values and a higher $\mathtt { R } ^ { 2 }$ score. The ablation study further confirms the necessity of each model component, showing that the removal of dynamic feature selection, hybrid boosting, or cost-sensitive learning mechanisms leads to notable declines in model accuracy and generalizability. This highlights the effectiveness of multistage feature selection and adaptive learning strategies in optimizing financial risk models. Moreover, the integration of costsensitive learning and synthetic data augmentation mitigates class imbalance issues, reducing bias in credit risk evaluation, which is a critical challenge in financial machine learning applications. From a theoretical standpoint, this research contributes to the growing body of literature on hybrid machine learning frameworks, demonstrating that combining tree-based gradient boosting models with deep learning components enhances both predictive accuracy and model interpretability. The results also align with recent advancements in explainable AI, emphasizing the importance of feature selection and hierarchical learning representations in financial decision-making. Furthermore, the findings underscore the necessity of fairness-aware optimization strategies, as addressing class imbalance directly influences model fairness and reliability in assessing borrower risk. The study suggests that future research should focus on refining ensemblebased deep learning architectures, integrating real-time risk adaptation mechanisms, and enhancing explainability in highdimensional financial modeling to further advance credit risk assessment methodologies.

# Conclusions

This study proposes HBA-LGBM (Hybrid Boosted Attentionbased LightGBM), a novel framework designed to enhance credit risk assessment by integrating multi-stage feature selection, attention-based feature enhancement, hybrid boosting, and imbalanced learning strategies. The experimental results demonstrate that HBA-LGBM significantly outperforms traditional machine learning and deep learning approaches, achieving the lowest RMSE (11.53) and MAPE $( 4 . 4 4 \% )$ , with an $\mathtt { R } ^ { 2 }$ score of 0.998, highlighting its superior predictive accuracy and robustness in handling borrower risk evaluations. The findings confirm that attention-based feature selection effectively prioritizes key financial risk factors, improving interpretability while maintaining computational efficiency. Furthermore, the hybrid boosting mechanism, which combines LightGBM and neural networks, successfully captures both structured and non-linear borrower behavior patterns, leading to more precise credit risk predictions. Additionally, the cost-sensitive learning and synthetic data augmentation strategies significantly mitigate class imbalance, ensuring fairer predictions for minority-class borrowers, a crucial aspect in digital lending and financial inclusion.

From a theoretical perspective, this research contributes to the advancement of hybrid machine learning models in financial risk assessment, demonstrating that integrating boosting algorithms with deep learning techniques enhances model generalization and interpretability. The proposed approach aligns with the evolving need for explainable AI in financial decision-making, providing a structured methodology for integrating automated feature selection and adaptive learning in high-dimensional financial datasets. From a practical standpoint, the scalability and adaptability of HBA-LGBM make it a viable solution for real-world digital lending applications, where lenders must process large volumes of borrower data while maintaining transparency in decisionmaking. Future research should explore further optimizations in deep feature extraction, real-time risk adaptation, and alternative feature augmentation techniques to refine credit risk modeling for dynamic financial environments. In addition, although this study primarily compares deep learning and ensemble-based methods, more comprehensive techniques like AutoML, which integrate multiple models for performance optimization, are acknowledged as promising directions for future exploration. Beyond predictive performance, model interpretability and robustness are crucial for real-world applications. Future work should evaluate the explainability of HBA-LGBM using techniques like SHAP values and SAFE AI frameworks (Giudici, 2024; Babaei et al., 2023), ensuring that model decisions are transparent and trustworthy.

# Data availability

No datasets were generated or analysed during the current study.

Received: 23 December 2024; Accepted: 5 June 2025; Published online: 07 July 2025

References   
Alsulamy S (2025) Predicting construction delay risks in Saudi Arabian projects: a comparative analysis of CatBoost, XGBoost, and LGBM. Expert Syst Appl 268:126268   
Babaei G, Bamdad S (2023) Application of credit‐scoring methods in a decision support system of investment for peer‐to‐peer lending. Int Trans Operat Res 30(5):2359–2373   
Babaei G, Giudici P, Raffinetti E (2023) Explainable fintech lending. J Econ Bus 125:106126   
Baser F, Koc O, Selcuk-Kestel AS (2023) Credit risk evaluation using clustering based fuzzy classification method. Expert Syst Appl 223:119882   
Cai S, Zhang J (2020) Exploration of credit risk of P2P platform based on data mining technology. J Comput Appl Math 372(1):112718   
Carta S, Ferreira A, Recupero DR, Saia M, Saia R (2020) A combined entropybased approach for a proactive credit scoring. Eng Appl Artif Intell 87:103292   
Essamlali I, Nhaila H, El Khaili M (2024) Supervised machine learning approaches for predicting key pollutants and for the sustainable enhancement of urban air quality: a systematic review. Sustainability 16(3):976   
Ghamry S, Shamma HM (2022) Factors influencing customer switching behavior in Islamic banks: evidence from Kuwait. J Islamic Mark 13(3):688–716   
Ginavanee A (2024) SAGB: a hybrid stacked AutoEncoder-Gradient Boosting model for accurate classification of amyotrophic lateral sclerosis (ALS). In 2024 13th International Conference on System Modeling & Advancement in Research Trends (SMART). IEEE, p. 688-695   
Giudici P (2024) Safe machine learning. Statistics 58(3):473–477   
Hancock JT, Khoshgoftaar TM (2020) CatBoost for big data: an interdisciplinary review. J Big Data 7(1):94   
Ji Y, Shi L, Zhang S (2022) Digital finance and corporate bankruptcy risk: evidence from China. Pac-Basin Financ J 72:101731   
Kavitha M, Kasthuri M (2024) Enhanced cost-sensitive ensemble learning for imbalanced class in medical data. J Electr Syst 20(7s):1043–1053   
Ke G, Meng Q, Finley T, Wang T, Chen W, Ma W, Liu, TY (2017) Lightgbm: a highly efficient gradient boosting decision tree. Adv Neural Inform Process Syst 30   
Korangi K, Mues C, Bravo C (2023) A transformer-based model for default prediction in mid-cap corporate markets. Eur J Operat Res 308(1):306–320   
Kostecka Z, Ślepaczuk R (2024) Improving Realized LGD approximation: a novel framework with XGBoost for handling missing cash-flow data. arXiv preprint arXiv:2406.17308   
Li G, Shi Y, Zhang Z (2019) P2P default risk prediction based on XGBoost, SVM and RF fusion model. 1st Int Conf Bus, Econ, Manag Sci (BEMS 2019) 2019:470–475   
Li H, Cao Y, Li S, Zhao J, Sun Y (2020) XGBoost model and its application to personal credit evaluation. IEEE Intell Syst 35(3):52–61   
Liu M, Li RYM, Deeprasert J (2024) Factors that affect individuals in using digital currency electronic payment in China: SEM and fsQCA approaches. Int Rev Econ Financ 95:103418   
Lowry PB, Xiao J, Yuan J (2023) How lending experience and borrower credit influence rational herding behavior in peer-to-peer microloan platform markets. J Manag Inf Syst 40(3):914–952   
Ma H, Zhao W, Zhao Y, He Y (2023) A data-driven oil production prediction method based on the gradient boosting decision tree regression. CMESComput Model Eng Sci 134(3)   
Ma Z, Hou W, Zhang D (2021) A credit risk assessment model of borrowers in P2P lending based on BP neural network. PloS ONE 16(8):e0255216   
Madaan M, Kumar A, Keshri C, Jain R, Nagrath P (2021) Loan default prediction using decision trees and random forest: a comparative study. In IOP conference series: materials science and engineering. IOP Publishing, p. 012042   
Mahmood J, Mustafa GE, Ali M (2022) Accurate estimation of tool wear levels during milling, drilling and turning operations by designing novel hyperparameter tuned models based on LightGBM and stacking. Measurement 190:110722   
Maldonado S, Bravo C, Lopez J, Pérez J (2017) Integrated framework for profitbased feature selection and SVM classification in credit scoring. Decis Support Syst 104(dec.):113–121   
Nandipati VSS, Boddala LV (2024) Credit card approval prediction: a comparative analysis between logistic regression, KNN, Decision Trees, Random Forest, XGBoost. Bachelor's thesis, Blekinge Institute of Technology. https://www. diva-portal.org/smash/record.jsf?pid=diva2:1883598   
Noviandy TR, Idroes GM, Hardi I (2024) Machine learning approach to predict AXL kinase inhibitor activity for cancer drug discovery using XGBoost and Bayesian optimization. J Soft Comput Data Min 5(1):46–56   
Oseni E (2023) Assessment of the five Cs of credit in the lending requirements of the Nigerian Commercial Banks. Int J Econ Financ Issues 13(4):58–65   
Rashedi KA, Ismail MT, Al Wadi S, Serroukh A, Alshammari TS, Jaber JJ (2024) Multi-layer perceptron-based classification with application to outlier detection in Saudi Arabia stock returns. J Risk Financial Manag 17(2):69   
Shpachuk V, Trinh VQ (2024) Introduction: modern banking and digital transformation. in modern banking and digitalization: the impact of FinTech on the Banking Market. Cham: Springer Nature Switzerland, p. 1-14   
Ślusarczyk D, Ślepaczuk R (2023) Optimal Markowitz portfolio using returns forecasted with time series and machine learning models (No. 2023-17). University of Warsaw, Faculty of Economic Sciences   
Song X, Qin X, Wang W, Li RYM (2025) Exploring the Spatial Spillovers of Digital Finance on Urban Innovation and Its Synergy with Traditional Finance. Emerg Sci J 9(1):433–450   
Sreesouthry S, Ayubkhan A, Rizwan MM, Lokesh D, Raj KP (2021) Loan prediction using logistic regression in machine learning. Ann Rom Soc Cell Biol 25(4):2790–2794   
Tripathi D, Shukla AK, Reddy BR, Bopche GS, Chandramohan D (2021) Credit scoring models using ensemble learning and classification approaches: a comprehensive survey. Wirel Personal Commun 123:785–812   
Varshney V, Goel A, Kumar D, Kaushik D, Sinha A (2024) Utilizing deep learning and machine learning models to predict loan default risk. In 2024 3rd Edition of IEEE Delhi Section Flagship Conference (DELCON). IEEE, p. 1–6   
Vyshnavi SL, Pavan DSV, Teja T, Pujitha SS, Shareefunnisa S (2024) Predicting bank loan eligibility with machine learning: a comparative study of advanced algorithms. In 2024 IEEE 3rd World Conference on Applied Intelligence and Computing (AIC). IEEE, p. 430–435   
Wang K, Li M, Cheng J, Zhou X, Li G (2022) Research on personal credit risk evaluation based on XGBoost. Procedia Comput Sci 199:1128–1135   
Wu Y, Pan Y (2021) Application analysis of credit scoring of financial institutions based on machine learning model. Complexity 2021(1):9222617   
Wysocki M, Sakowski P (2022) Investment portfolio optimization based on modern portfolio theory and deep learning models. University of Warsaw, Faculty of Economic Sciences   
Xie X, Zhang J, Luo Y, Gu J, Li Y (2024) Enterprise credit risk portrait and evaluation from the perspective of the supply chain. Int Trans Operat Res 31(4):2765–2795   
Yao, J, Wang, J, Wang, B Liu, B, & Jiang, M (2024) A Hybrid CNN-LSTM Model for Enhancing Bond Default Risk Prediction. Journal of Computer Technology and Software, 3(6)   
Zhang J, Calabrese R, Dong Y (2024) A novel generalised extreme value gradient boosting decision tree for the class imbalanced problem in credit scoring. J Operat Res Soc 1–18   
Zhang X, Yu L (2024) Consumer credit risk assessment: a review from the state-ofthe-art classification algorithms, data traits, and learning methods. Expert Syst Appl 237:121484   
Zheng S, Zhou K, Chen C (2024) Perturbation-Based SMOTE for multi-class imbalanced classification. In 2024 5th International Conference on Machine Learning and Computer Application (ICMLCA). IEEE. p. 53–56   
Zou Y, Gao C (2022) Extreme learning machine enhanced gradient boosting for credit scoring. Algorithms 15(5):149

# Acknowledgements

The study was jointly supported by National Social Science Foundation (Fund number: 20&ZD102) and Zhejiang Provincial Philosophy and Social Science Planning Project. (Fund number: 20NDJC225YB).

# Author contributions

Conceptualization, CY; Data curation, CY; Formal analysis, AS and AS; Investigation, XL; Resources, XL; Validation, XL and CY; Writing, AS.

# Competing interests

The authors declare no competing interests.

# Ethical statements

This article does not contain any studies with human participants performed by any of the authors.

# Informed consent

This article does not contain any studies with human participants performed by any of the authors.

# Additional information

Correspondence and requests for materials should be addressed to Xiongyi Li.

Reprints and permission information is available at http://www.nature.com/reprints

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/ licenses/by-nc-nd/4.0/.