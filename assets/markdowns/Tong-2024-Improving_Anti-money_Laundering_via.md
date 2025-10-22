# Improving Anti-money Laundering via Fourier-Based Contrastive Learning

Meihan Tong $^ { 1 , 2 ( \boxtimes ) }$ , Shuai Wang $^ { 3 }$ , Xinyu Chen $\bot$ , and Jinsong Bei $\cdot ^ { 1 }$ $^ { 1 }$ China National Clearing Center, Beijing, China $^ 2$ Tsinghua University, Beijing, China tongmeihan@gmail.com tongmeihan@gmail.com3 Hashkey, Beijing, China wangshuai@hashkey .com

Abstract. Anti-money laundering (AML) aims to detect money laundering from daily transactions, which is the key frontier of combating financial crimes. Previous deep-learning AML methods are not robust enough. To address the problem, we propose a novel Fourier-based contrastive learning model (FCLM) to improve AML. With contrastive learning, FCLM can maintain prediction consistency and be more robust in the face of data perturbations. Experiments on both the synthetic benchmark IBM2023 and the real-world benchmark show that FCLM outperforms seven state-of-the-art baselines, demonstrating the effectiveness of the proposed Fourier-based contrastive learning model.

Keywords: Anti-Money laundering  Contrastive Learning  Data Augmentation Fourier Transformation

# 1 Introduction

Anti-money laundering aims to identify money laundering activities that conceal the source of criminal proceeds from massive transactions. Money laundering impairs the stability of the financial market, increases the operational risks of financial institutions, and has caused a great economic loss to the world. According to the report by Cybersecurity Ventures, the economic losses caused by global money laundering crimes will grow at an annual rate of $1 5 \%$ in the next two years, reaching $1 0 . 5 \ S$ trillion per year by $2 0 2 5 ^ { 1 }$ .

Due to the greater dangers of money laundering, anti-money laundering has received great attention from researchers around the world. Early AML methods are mainly rule-based methods, such as [4,26], which suffer from inflexible and high maintenance costs. Later, machine learning AML methods, such as SVM [28] and RF [1], became popular. However, these methods have low accuracy. Recently, deep learning AML methods, such as SkipGNN [33] and HAMLET [29], have been proposed. However, these methods are not robust enough, that is by adding subtle and deliberate perturbations to the transactions, these deeplearning AML models will output erroneous results with a high confidence level.

We propose to leverage data augmentation and contrastive learning to handle the issue. The main idea is to generate the augmented view for each transaction and then leverage contrastive learning to pull the transaction closer to its augmented view to ensure that our model can maintain the consistency of discrimination results in the face of data perturbations, and avoid frequent misjudgments under minor disturbances. We utilize the Fourier data augmentation method to generate the augmented view. Compared to Gaussian noise [3], column sampling2, mask token replacing [20], and window wrapping [25], Fourier data augmentation method maps the entire transaction features from the time domain to the frequency domain, resulting in a more differentiated augmented view and allowing our model capture a wider range of variations and different perspectives of the same sample.

In summary, we propose a Fourier-based contrastive learning model(FCLM) to improve anti-money laundering. FCLM uses the Transformer model as the backbone network. When detecting money laundering, FCLM first employs Fourier transformation to generate the augmented view of each transaction. Then, FCLM uses comparative pre-training to ensure the consistency of its predictions for original transactions and augmented views. Finally, FCLM distinguishes money laundering transactions from normal ones through the MLP classifier. The classifier takes both the original transaction and its augmentation view as input to synthesize both sides of information to improve money laundering detection.

We evaluate FCLM on a synthetic dataset IBM2023 $^ { 3 }$ and a real-world dataset. Experiments show that our proposed method consistently surpasses seven strong baselines on both datasets, demonstrating the superiority of the proposed Fourier-based contrastive learning model.

Our contributions can be summarized as follows:

We propose a Fourier-based contrastive learning model (FCLM) to improve the robustness of money laundering detection.   
We proposed a Fourier-based data augmentation method, which transforms the transaction attributes from the time domain to the frequency domain to ensure the discrepancy of the augmented view, enabling our model to capture wider variations and become more robust in representation learning.   
Experiments on the synthetic dataset IBM2023 and the real-world dataset show that FCLM surpasses seven strong baselines. Detailed studies also show that the proposed Fourier data augmentation method surpasses three commonly used data augmentation algorithms.

The following of the paper is organized as follows. Section 2 introduces commonly used anti-money laundering methods and related work on contrastive learning. Section 3 illustrates the overall architecture of our proposed Fourierbased contrastive learning model (FCLM) and details each module in the FCLM. Section 4 introduces the experimental datasets and hyper-parameters of FCLM, and presents extensive experimental results. Section 5 concludes the paper.

# 2 Related Work

# 2.1 Anti-Money Laundering(AML)

Anti-money laundering is a hot topic. Early AML models were mostly rulesbased approaches [2,4,5]. They commonly use 1) a surge in transaction traffic in a short period, 2) the transaction amount exceeds a specified threshold for multiple consecutive days, etc. to detect money laundering. Rule-based methods are inflexible and can be easily broken by criminals through rule probing.

Other methods adopt machine learning to improve AML [13,19,27,30,35]. For instance, [28] proposes a set of abnormal behavior detection algorithms based on support vector machines (SVM). [1] employs the Light gradient Boosting Algorithm (LGBA) and XGBoost in distinguishing illegal money laundering exercises. These machine-learning AML methods have low accuracy, and their performance is easily affected by the quality of feature engineering.

Deep learning supervised AML methods [6,12,14,16,22,29,32] have gradually attracted the attention of scholars. CS-CNN [16] leverages cost-sensitive CNN and feature matrix for fraud detection. OCGTL [23] combines deep oneclass classification with GNN for graph-level anomaly detection. ComGA [18] leverages a community-aware tailored GCN to handle AML. SkipGNN [33] utilizes a skip connection GNN network and Diga [15] employs the semi-supervised guided diffusion to identify money laundering activities. HAMLET [29] employs a hierarchical transformer to identify complex money laundering at both transaction and sequence levels. However, these methods are not robust enough and often produce inconsistent prediction results when faced with data disturbances. In this paper, we empower AML with contrastive learning to allow the proposed model to maintain prediction consistency under the perturbation of the Fourier augmented view to improve robustness in AML.

# 2.2 Contrastive Learning

Contrastive learning [9] is a self-supervised learning method that has been proven to be effective in a wide range of fields.

In anti-money laundering, we pay more attention to tabular data augmentation. Naive Gaussian [3] is a widely adopted tabular data augmentation method, which generates the augmented data by injecting Gaussian noise. Column sampling (see Footnote 2) adds noise to tabular data by replacing the feature with the value sampled from its overall distribution. Mask token replacement [20] augments the tabular data by masking out the feature and replacing it with a variable-length [MASK] embedding. These methods perturb each feature individually, resulting in a small degree of diversity. Window wrapping [25] is a time-series data augmentation method that augments the time-series feature by speeding it up or slowing it down, which cannot be applied to category features in transactions, such as payee and payer. In this paper, we leverage Fourier to transform the whole features in the transactions from the time domain to the frequency domain to generate augmented views with high diversity.

![](images/55d2f5b016863a2ee107c8e21726c0286d2613ce3d9d05e2bddea43fe10a83c2.jpg)  
Fig. 1. Architecture of the proposed Fourier-based contrastive learning model(FCLM).

# 3 Methdology

# 3.1 Task Definition

Formally, given the corpus $x _ { i } = \{ t _ { i } , t _ { i - 1 } , \dots , t _ { i - m } | _ { i = 1 } ^ { N } \}$ , where $t _ { i }$ is the transaction to be classified, $t _ { i - m }$ is the historical transactions of $t _ { i }$ , $N$ is the total number of transactions, the proposed model FCLM first minimizes $L _ { 1 } =$ log exp(hi,hi - ) Kk=1 exp(hi,hk) to ensure prediction consistency in the face of data perturbations, where $h _ { i }$ and $h _ { i } ^ { \omega }$ are the hidden representations of the original data $x _ { i }$ and its augmented view $\boldsymbol { x } _ { i } ^ { \omega }$ , $K$ is the number of sampled negative examples. Then, FCLM maximizes the classification probability $\begin{array} { r } { p _ { i c } = \frac { e x p ( o _ { i c } ) } { \sum _ { c = 1 } ^ { C } e x p ( o _ { i c } ) } } \end{array}$ to detects money laundering transactions, where $C$ is the category number, $o _ { i c }$ is the predicted probability that $h _ { i }$ belongs to the $c$ th category.

# 3.2 Overall Architecture

Figure 1 illustrates the overall architecture of the proposed Fourier-based contrastive learning model(FCLM), which includes four modules: data augmentation module, feature encoding module, contrastive pre-training module, and money laundering detection module.

The data augmentation module is designed to generate augment views for transactions. The feature encoding module aims to encode the original transaction and its augmented view into hidden representations. The contrastive pretraining module aims to utilize contrastive learning to bring the hidden representation closer between the original transaction and its augmented view, and to push the distance between the original transaction and other transactions farther away. The money laundering detection module aims to distinguish money laundering transactions from normal ones.

# 3.3 Data Augmentation

In this section, we introduce the proposed Fourier data augmentation method.

We start with converting raw transactions into embeddings. We handle categorical and numeric attributes in raw transactions differently. For the categorical attribute, we randomly initialize an embedding for each class in the categorical attribute and then obtain the embedding of the categorical attribute based on the class index. For the numeric attribute, we first employ Z-score normalization for scaling. After that, we discretize the numeric attribute into equal-frequency buckets (each bucket has the same number of elements) and randomly initialize an embedding for each bucket in the numeric attributes to obtain the embedding of the numeric attribute.

After obtaining the embedding $x _ { i }$ of the transaction, we employ the Discrete Fourier Transform (DFT) to generate its augmented data. Formally, for each transaction $x _ { i }$ , the discrete Fourier transform can be formalized as:

$$
F ( \omega ) = \sum _ { i = 0 } ^ { N - 1 } e ^ { - i \frac { 2 \pi } { N } n \omega } x _ { i }
$$

where $\omega \in [ 0 , N - 1 ]$ is the angular frequency, $e$ is the base of natural logarithms and $i$ is the imaginary unit. Due to the large number of transactions, we employ the fast Fourier transform (FFT) algorithm [11] to solve DFT to improve computational efficiency.

In summary, after leveraging DFT, we get a complex number representing magnitude and phase in the frequency domain as the Fourier-augmented view of the transaction data.

# 3.4 Feature Encoding

In this section, we aim to build a feature encoder that can convert both original and Fourier-augmented transactions into hidden representations.

We employ Transformer [31] as our feature encoder to fuse features deeply. Transformer is a multi-layer deep learning architecture that aims to handle longdistance dependencies, which has achieved great success on a wide range of financial fraud tasks, such as credit card fraud detection [34] and insurance fraud detection [10]. The multi-head attention mechanism in each layer of the

Transformer can automatically adjust the attention weights during the training process, enabling the model to discover features that are important to anti-money laundering while ignoring irrelevant features.

We take the output of the last layer of the Transformer as the hidden representation $h _ { i }$ of the original transactions $x _ { i }$ , which is denoted as:

$$
h _ { i } = E n c o d e ( x _ { i } )
$$

In the same way, we obtain the hidden representation $h _ { i } ^ { \omega }$ of the Fourieraugmented transaction $\boldsymbol { x } _ { i } ^ { \omega }$ .

# 3.5 Contrastive Pre-training

In this section, we perform comparative pre-training on the full amount of transaction data to optimize the feature encoder in Sect. 3.4.

Formally, the loss function of contrastive pre-training is defined as:

$$
L _ { 1 } ( h _ { i } , h _ { i } ^ { \omega } , \theta _ { 0 } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } l o g \frac { e x p ( h _ { i } , h _ { i } ^ { \omega } ) } { \sum _ { k = 1 } ^ { K } e x p ( h _ { i } , h _ { k } ) }
$$

where N is the scale of the transaction data, $h _ { i }$ and $h _ { i } ^ { \omega }$ refer to hidden representation of $x _ { i }$ (original transaction) and $\boldsymbol { x } _ { i } ^ { \omega }$ (Fourier augmented transaction) respectively, $h _ { k }$ refers to hidden representation of $x _ { k }$ ( $x _ { k } \neq x _ { i }$ ), and $K$ is the number of sampled examples.

Through comparative pre-training, we can make the hidden representation of the transaction closer to the hidden representation of its own augmented view and farther away from the hidden representation of other transactions, which can make our model more robust in the face of data perturbations.

# 3.6 Money Laundering Detection

The money laundering detection module is designed to distinguish normal transactions from money laundering transactions based on the optimized feature encoder.

Specifically, we first randomly undersample the normal transactions to rebalance the training data, and then, we build a multiple-layer perceptron classifier on the top of the optimized feature encoder to map the hidden representation into the classification space, which can be formally expressed as:

$$
o _ { i } = W ( h _ { i } + h _ { i } ^ { \omega } ) + b
$$

where $h _ { i }$ and $h _ { i } ^ { \omega }$ are the hidden representations of the original transaction and the Fourier augmented transaction respectively. The classifier takes both the original transaction and the Fourier-enhanced transaction as input to combine the time and frequency domain information when determining whether a transaction is a money laundering transaction.

We calculate the classification probability by:

$$
p _ { i c } = \frac { e x p ( o _ { i c } ) } { \sum _ { c = 1 } ^ { C } e x p ( o _ { i c } ) }
$$

where $C$ is the number of categories. Finally, the training loss of the money laundering detection module is defined as:

$$
L _ { 2 } ( h _ { i } , \theta _ { 1 } ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { C } y _ { i c } l o g ( p _ { i c } )
$$

where $p _ { i c }$ is the predicted probability that $h _ { i }$ belongs to the $c$ th category, and $y _ { i c }$ is the category label.

# 4 Experiment

# 4.1 Datasets

IBM2023 (see Footnote 3) is a large synthetic anti-money laundering dataset released by IBM Corporation. IBM2023 creates an entire virtual financial ecosystem by having artificial individuals, companies and banks transact with each other. IBM2023 contains six versions of the dataset, including HI-small, HImedium, HI-large, LI-small, LI-medium, and LI-large. We adopt the HI-small version of the dataset for experimental training, which contains 515K bank accounts and 5M transaction data. The features in the transaction of IBM2023 include account, timestamp, amount, currency type, etc. Following the official instructions, we divide the datasets into 60%/20%/20% for training, validating, and testing respectively.

Unlike previous anti-money laundering models [29] that are only evaluated on synthetic datasets, we further evaluate our model on a real-world dataset. The real-world dataset is built from the China interbank clearing transaction system, which contains a total of 10k money laundering transactions and 10M normal transactions. To maintain data confidentiality, each transaction in this dataset is a summary of the transactions of both parties on the day. The transaction attributes include transaction time, payer, payer’s bank, payer’s account, payee, payee’s bank, payee’s account, the total transaction amount of the day, the number of night transactions, etc., a total of 19 features. We split the real-world dataset into 80%/10%/10% for training, validation, and testing respectively.

Following [7], we evaluate the performance of the proposed model through four metrics, including precision, recall, F1, and AUC. Due to data imbalance in AML scenarios, we use the macro average of these metrics as the final result.

# 4.2 Hyperparameters

Through grid search, we set the learning rate to 1e–4 in the pre-training stage and 2e–5 in the money laundering detection stage. The training step/batch size of the pre-training stage and the money laundering detection stage are 10,000/512 and 1,000/128 respectively. We adopt the 8-layer Transformer as our backbone network, with 5 heads of attention and 128 hidden representation dimensions. We select the checkpoint that performs best on the validation set as the final inference model and report the average result of 10 runs as the final results. We use Adam as the gradient descent optimizer. We use an 8-card A100 server with 80G GPU memory per card for training. The training time is about 10/0.5 h for the pre-training and the money laundering detection stage respectively.

Table 1. Overall Performance of FCLM(%).   

<table><tr><td rowspan="2">Methods</td><td colspan="3">IBM2023</td><td colspan="5">Real-World</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>AUCP</td><td></td><td>R</td><td>F1</td><td>AUC</td></tr><tr><td>VAE</td><td></td><td></td><td></td><td></td><td></td><td>55.254.754.954.153.754.554.1</td><td></td><td>52.9</td></tr><tr><td>SVM</td><td>58.1</td><td>64.6</td><td>61.2</td><td>60.7</td><td>54.5</td><td>52.3</td><td>53.4</td><td>52.7</td></tr><tr><td>RF</td><td>56.8</td><td>60.3</td><td>58.5</td><td>64.7</td><td>54.7</td><td>56.1</td><td>55.4</td><td>58.5</td></tr><tr><td>CS-CNN</td><td></td><td>85.680.3</td><td>82.8</td><td>79.4</td><td>66.2</td><td>67.5</td><td>66.8</td><td>67.8</td></tr><tr><td>SkipGNN</td><td>72.4</td><td>80.6</td><td>76.3</td><td>78.1</td><td>66.3</td><td>71.6</td><td>68.8</td><td>70.2</td></tr><tr><td>Inspection-L</td><td>78.2</td><td>84.5</td><td>81.2</td><td>83.6</td><td>69.6</td><td>74.2</td><td>71.8</td><td>74.7</td></tr><tr><td>HAMLET</td><td></td><td>79.785.1</td><td></td><td>82.386.2</td><td></td><td>71.676.273.8</td><td></td><td>75.9</td></tr><tr><td>FCLM (0urs)82.591.286.691.473.778.275.980.1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

# 4.3 Baselines

We compare our methods with seven AML baselines, including:

VAE [8] uses a sparse autoencoder to learn the distribution of normal transactions and treat transactions that do not fit the distribution as money laundering transactions.   
SVM [21] proposes a support vector machine (SVM) classifier optimized with the random undersampling (RUS) technique to detect financial fraud. RF [24] is a traditional machine learning AML baseline, which leverages rich features and the random forest algorithm for money laundering detection. CS-CNN [16] leverages cost-sensitive CNN and feature matrix for fraud detection. SkipGNN [33] takes transactions as nodes and transaction flow as edges, and leverages skip connections GNN to detect AML at the graph level. Inspection-L [17] is a self-supervised graph neural network (GNN) framework based on Deep Graph Infomax (DGI) and Graph Isomorphism Network (GIN), with Random Forest (RF) to detect illicit transactions for anti-money laundering (AML).   
– HAMLET [29] employs the hierarchical transformer to fuse features at both

transaction and sequence levels to identify money laundering.

Table 1 presents the overall performance of the proposed Fourier-based contrastive learning model on IBM2023 and the real-world dataset. As shown in Table 1, our method outperforms seven state-of-the-art baselines, demonstrating the effectiveness of FCLM and the superiority of the Fourier data augmentation method.

FCLM (ours) outperforms VAE by 37.4% in AUC score on IBM2O23 and $2 7 . 4 \%$ on the real-world dataset. VAE is a semi-supervised AML method, which does not exploit money laundering transactions for training. While FCLM (ours) not only utilizes a large amount of normal transaction data for training, but also utilizes money laundering annotations as supervision signals, taking both sides into account, and thus has better performance.

Compared with non-transformer baselines SVM, RF, CS-CNN, SkipGNN, and Inspection-L, the transformer baselines HAMLET and FCLM(ours) perform better. This shows the strong representation capabilities of the transformer, which can effectively extract and integrate transaction features in anti-money laundering. Among the transformer’s baseline HAMLET and FCLM (ours), FCLM (ours) performs better. This is because we add contrastive learning training to the basic transformer to allow our model to maintain prediction consistency in the face of data perturbations. With more stable prediction results, our model is more robust and therefore generalizes better on the unseen data.

Compared with Inpection-L which also uses contrastive learning, our model achieves better performance (83.6% VS $9 1 . 4 \%$ in IBM2023 and 74.7% VS $8 0 . 1 \%$ in the real-world dataset). One of the reasons is that Inspection-L augments data through graph corruption (randomly adding and deleting edges in the transaction graph), which may fabricate the occurrence of transactions and cannot guarantee that the augmented view and the original data still belong to the same class. As a result, wrong labels are mixed into the training, leading to a decrease in performance. Our method uses Fourier transform to obtain enhanced data, which can ensure the equivalence of the two and there is no risk that the augmented view and the original data do not belong to the same class.

# 4.4 Effectiveness of Data Augmentation

In this section, we hope to observe what happens when the proposed Fourier data augmentation method is replaced with other tabular data augmentation data. In addition to the non-augmentation baseline, we also adopt three baselines, including:

– Mask Replacing [20] augments the transaction data by randomly replacing the features in the transactions with the [MASK] token. Gaussian [3] augments the transaction by adding random Gaussian noise. Column sampling (see Footnote 2) adds noise to the transaction by replacing original features with sampled values from feature distributions.

For all baselines, we uniformly use the transformer as the classifier so that the baselines can be compared fairly with the proposed method.

Table 2. Performance on Different Data Augmentation Methods on IBM2023   

<table><tr><td>Augmentation</td><td>AUC</td></tr><tr><td>Non-Augmentation</td><td>85.6</td></tr><tr><td>Mask Token Replacing</td><td>85.3</td></tr><tr><td>Gaussian</td><td>87.1</td></tr><tr><td>Column Sampling</td><td>87.3</td></tr><tr><td>Fourier(ours)</td><td>91.4</td></tr></table>

As shown in Table 2, Fourier (ours) outperforms the non-augmented baseline and the three data-augmented baselines by $5 . 8 \%$ , $6 . 1 \%$ , $4 . 3 \%$ , and $4 . 1 \%$ in AUC, achieving the best performance. The mask token replacing baseline is not as good as the non-augmentation baseline, which shows that data enhancement does not always improve AML and requires careful design. An in-depth analysis of the failure of the mask token replacement baseline shows that there is serious information lost in the process of replacing the original features with the [MASK] embedding, which makes the augmented view less informative in making money laundering decisions. The poor performance of the Gaussian method and column sampling method is because these two baselines perform data augmentation on each feature independently, resulting in smaller differences between the augmented view and the original data. In contrast, our method maps the entire transaction features from the time domain to the frequency domain. With more differentiated augmented views, our model can capture wider variations and different perspectives of the same sample, thereby becoming more robust in representation learning and better able to generalize to unseen data.

# 4.5 Future Direction

Two promising directions can be considered in future work. First, leverage more data sources for training. FCLM(ours) only incorporates transaction data in the training processes. Future work can consider using additional data information such as account data, portrait data, credit data, corporate financial report data, etc. to improve the monitoring performance of the AML model. Second, utilize the large language models. FCLM (ours) does not take advantage of large language models such as GPT4, PaLM-E, BLOOM, LLaMA, etc. Future work can consider enhancing the AML model with large language models to mine more complex gang-based money laundering behaviors and reduce the false positive rate of money laundering.

# 5 Conclusion

In this paper, we propose a Fourier-based contrastive learning model (FCLM) to improve the robustness of anti-money laundering. FCLM first leverages the

Fourier data augmentation method to transform the transaction attributes from the time to frequency domain to ensure the discrepancy of the augment views, and then utilize contrastive learning to refine the representation of transactions. Through contrastive learning, FCLM can maintain prediction consistency in the face of data perturbations and therefore has stronger robustness and generalization. Experiments on two benchmarks demonstrate the effectiveness of FCLM. Comparative experiments between different data augmentation methods further show the superiority of the proposed Fourier data augmentation method.

# References

1. Ahmed, A.A.A.: Anti-money laundering recognition through the gradient boosting classifier. Acad. Accounting Fin. Stud. J. 25(5), 1–11 (2021)   
2. Ai, L.: Rule-based but risk-oriented approach for combating money laundering in Chinese financial sectors. J. Money Laundering Control 15(2), 198–209 (2012)   
3. Arslan, M., Guzel, M., Demirci, M., Ozdemir, S.: SMOTE and gaussian noise based sensor data augmentation. In: UBMK, pp. 1–5. IEEE (2019)   
4. Bellomarini, L., Laurenza, E., Sallinger, E.: Rule-based anti-money laundering in financial intelligence units: experience and vision. RuleML+ RR 2644(Suppl.), 133–144 (2020)   
5. Butgereit, L.: Anti money laundering: rule-based methods to identify funnel accounts. In: 2021 Conference on Information Communications Technology and Society (ICTAS), pp. 21–26 (2021)   
6. Chai, Z., et al.: Towards learning to discover money laundering sub-network in massive transaction network. In: Proceedings of the AAAI Conference on Artificial Intelligence (2023)   
7. Charitou, C., Garcez, A.D., Dragicevic, S.: Semi-supervised GANs for fraud detection. In: IJCNN, pp. 1–8 (2020)   
8. Chen, J., Shen, Y., Ali, R.: Credit card fraud detection using sparse autoencoder and generative adversarial network. In: IEMCON, pp. 1054–1059. IEEE (2018)   
9. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.E.: A simple framework for contrastive learning of visual representations. CoRR abs/2002.05709 (2020). https:// arxiv.org/abs/2002.05709   
10. Fursov, I., et al.: Sequence embeddings help detect insurance fraud. IEEE Access 10, 32060–32074 (2022)   
11. Heckbert, P.: Fourier transforms and the fast Fourier transform (FFT) algorithm. Comput. Graph. 2(1995), 15–463 (1995)   
12. Hu, B., Zhang, Z., Shi, C., Zhou, J., Li, X., Qi, Y.: Cash-out user detection based on attributed heterogeneous information network with a hierarchical attention mechanism. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, pp. 946–953 (2019)   
13. Kumar, A., Das, S., Tyagi, V., Shaw, R.N., Ghosh, A.: Analysis of classifier algorithms to detect Anti-money laundering. In: Bansal, J.C., Paprzycki, M., Bianchini, M., Das, S. (eds.) Computationally Intelligent Systems and their Applications. SCI, vol. 950, pp. 143–152. Springer, Singapore (2021). https://doi.org/10.1007/ 978-981-16-0407-2_11   
14. Kute, D.V.: Explainable deep learning approach for detecting money laundering transactions in banking system. Ph. D. thesis (2022)   
15. Li, X., Li, Y., Mo, X., Xiao, H., Shen, Y., Chen, L.: Diga: guided diffusion model for graph recovery in anti-money laundering. In: Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 4404–4413 (2023)   
16. Liu, X., Zhang, X., Miao, Q.: A click fraud detection scheme based on cost-sensitive CNN and feature matrix. In: Tian, Y., Ma, T., Khan, M.K. (eds.) ICBDS 2019. CCIS, vol. 1210, pp. 65–79. Springer, Singapore (2020). https://doi.org/10.1007/ 978-981-15-7530-3_6   
17. Lo, W.W., Kulatilleke, G.K., Sarhan, M., Layeghy, S., Portmann, M.: Inspection-l: self-supervised GNN node embeddings for money laundering detection in bitcoin. Appl. Intell. 53, 19406–19417 (2023)   
18. Luo, X., et al.: ComGA: community-aware attributed graph anomaly detection. In: Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, pp. 657–665 (2022)   
19. Misra, S., Thakur, S., Ghosh, M., Saha, S.K.: An autoencoder based model for detecting fraudulent credit card transaction. Procedia Comput. Sci. 167, 254–262 (2020)   
20. Onishi, S., Meguro, S.: Rethinking data augmentation for tabular data in deep learning. arXiv preprint arXiv:2305.10308 (2023)   
21. Pambudi, B.N., Hidayah, I., Fauziati, S.: Improving money laundering detection using optimized support vector machine. In: 2019 International Seminar on Research of Information Technology and Intelligent Systems (ISRITI), pp. 273–278 (2019). https://doi.org/10.1109/ISRITI48646.2019.9034655   
22. Pareja, A., et al.: EvolveGCN: evolving graph convolutional networks for dynamic graphs. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, pp. 5363–5370 (2020)   
23. Qiu, C., Kloft, M., Mandt, S., Rudolph, M.: Raising the bar in graph-level anomaly detection. arXiv preprint arXiv:2205.13845 (2022)   
24. Raiter, O.: Applying supervised machine learning algorithms for fraud detection in anti-money laundering. J. Mod. Issues Bus. Res. 1(1), 14–26 (2021)   
25. Rashid, K.M., Louis, J.: Window-warping: a time series data augmentation of IMU data for construction equipment activity identification. In: ISARC. Proceedings of the International Symposium on Automation and Robotics in Construction, vol. 36, pp. 651–657. IAARC Publications (2019)   
26. Ross, S., Hannan, M.: Money laundering regulation and risk-based decisionmaking. J. Money Laundering Control 10(1), 106–115 (2007)   
27. Sundarkumar, G.G., Ravi, V., Siddeshwar, V.: One-class support vector machine based undersampling: application to churn prediction and insurance fraud detection. In: 2015 IEEE International Conference on Computational Intelligence and Computing Research (ICCIC), pp. 1–7. IEEE (2015)   
28. Tang, J., Yin, J.: Developing an intelligent data discriminating system of antimoney laundering based on SVM. In: 2005 International Conference on Machine Learning and Cybernetics, vol. 6, pp. 3453–3457. IEEE (2005)   
29. Tatulli, M.P., Paladini, T., D’Onghia, M., Carminati, M., Zanero, S.: HAMLET: a transformer based approach for money laundering detection. In: Dolev, S., Gudes, E., Paillier, P. (eds.) International Symposium on Cyber Security, Cryptology, and Machine Learning, vol. 13914, pp. 234–250. Springer, Cham (2023). https://doi. org/10.1007/978-3-031-34671-2_17   
30. Tundis, A., Nemalikanti, S., Mühlhäuser, M.: Fighting organized crime by automatically detecting money laundering-related financial transactions. In: Proceedings of the 16th International Conference on Availability, Reliability and Security, pp. 1–10 (2021)   
31. Vaswani, A., et al.: Attention is all you need. CoRR abs/1706.03762 (2017). https://arxiv.org/abs/1706.03762   
32. Wang, D., et al.: Temporal-aware graph neural network for credit risk prediction. In: Proceedings of the 2021 SIAM International Conference on Data Mining (SDM), pp. 702–710. SIAM (2021)   
33. Weber, M., et al.: Anti-money laundering in bitcoin: experimenting with graph convolutional networks for financial forensics. arXiv preprint arXiv:1908.02591 (2019)   
34. Yuan, M.: A transformer-based model integrated with feature selection for credit card fraud detection. In: 2022 7th International Conference on Machine Learning Technologies (ICMLT), pp. 185–190 (2022)   
35. Zou, J., Zhang, J., Jiang, P.: Credit card fraud detection using autoencoder neural network. arXiv preprint arXiv:1908.11553 (2019)