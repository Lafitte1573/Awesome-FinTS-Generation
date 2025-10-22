# VAE-INN: Variational Autoencoder with Integrated Neural Network Classifier for Imbalanced Credit Scoring, Utilizing Weighted Loss for Improved Accuracy

Dalia ATIF1

Received: 4 August 2024 / Accepted: 18 August 2025   
$\circledcirc$ The Author(s), under exclusive licence to Springer Science+Business Media, LLC, part of Springer Nature   
2025

# Abstract

Assessing credit risk using machine learning tools is critical for minimizing bank losses, which can arise from financial defaults or missed opportunities. Since financial losses are more detrimental than missed opportunities, this paper proposes an integrated approach for imbalanced credit scoring. We leverage a Variational Autoencoder (VAE) with an Integrated Neural Network classifier (VAE-INN) designed to address class imbalance during feature extraction. Unlike traditional methods that separately apply data augmentation and feature selection, potentially altering the original data distribution or introducing biases, our method inherently tackles class imbalance within the latent space, guided by a weighted loss function. While traditional methods attempt to address imbalance through separate steps such as data augmentation or feature selection, these approaches risk distorting the data distribution or discarding features crucial for identifying minority class instances. In particular, feature selection, commonly used to enhance class separability, can inadvertently favor the majority class when applied to imbalanced data. Our method promotes fairer and more robust classification by shaping the latent space to represent both classes equally. We validate our approach on an imbalanced real-world dataset, focusing on reducing Type II errors. Experimental results show that our VAE-INN model with weighted loss significantly outperforms fragmented techniques, achieving substantial Type II error reduction and demonstrating robustness under severe imbalance ratios, thus enhancing the detection of true credit defaults and mitigating financial risk.

Keywords Credit risk assessment $\cdot$ Imbalanced credit scoring $\cdot$ Variational Autoencoder $\cdot$ Neural network classifier $\cdot$ Weighted loss function $\cdot$ Type II error reduction

JEL Classification C45 · G21 · G32

# 1 Introduction

Credit scoring has become a prevalent method to reduce bank losses in the banking sector over the last few decades. The explosion of big data and the 2008 subprime mortgage crisis created a need for more sophisticated credit-scoring methods. This shift coincided with the rise of machine learning tools. Initially, these tools, primarily from supervised learning, such as Logistic Regression (LR), Decision Trees (DT), Support Vector Machines (SVM), Naive Bayes (NB), and MultiLayer Perceptron (MLP), produced satisfactory results (Kruppa et al., 2013; Lessmann et al., 2015; Pandey et al., 2017; Pandey & Bandhu, 2022). Nevertheless, as credit datasets became more complex, issues such as effective feature selection, classifier choice, and, most critically, class imbalance emerged. Early solutions often focused on rebalancing datasets using synthetic oversampling techniques like SMOTE. Although effective in balancing class ratios, these approaches risked altering the data distribution, introducing biases that distorted the natural relationship between features and outcomes, ultimately undermining model reliability. To address these shortcomings, generative models such as Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) and Variational Autoencoders (VAEs) (Kingma & Welling, 2013) have been explored to create synthetic minority samples that better reflect the underlying data distribution. However, generative models are not exempt from bias: they can propagate and even amplify existing minority class noise, generate unrealistic minority examples, or overly smooth class boundaries (Decruyenaere et al., 2024). These distortions can cause classifiers to misinterpret decision boundaries, leading to overfitting to synthetic patterns or underrepresenting rare behaviors in credit risk datasets.

Various misconceptions persist regarding how best to address class imbalance in the context of imbalanced credit scoring. At the data level, rebalancing strategies such as oversampling, undersampling, and synthetic data generation are commonly employed as preprocessing steps to equalize class distributions artificially (Wang et al., 2024; Zhang et al., 2022; Hou et al., 2020; Rao et al., 2023; Tingfei et al., 2020). At the algorithmic level, ensemble methods (e.g., boosting) (Kun et al., 2020; Arora & Kaur, 2020; Yin et al., 2023; Wang et al., 2022; Muslim et al., 2023) and deep learning architectures (Wang et al., 2024; Mancisidor et al., 2021) are often adopted, assuming that their complexity and flexibility inherently mitigate imbalance effects. However, both approaches can introduce distinct forms of bias, each impacting model reliability differently. First, distributional bias arises from data-level methods such as random oversampling, undersampling, or synthetic generation techniques like SMOTE. Often not grounded in the true data-generating process, these approaches may distort the original feature-target relationships, artificially inflate feature correlations, and harm the model’s generalization ability. Second, generative model-based approaches introduce representational bias, including GANs and variational autoencoders. Although these methods aim to approximate the underlying distribution, inaccurate modeling can produce unrealistic or noisy minority class instances, leading to overfitting and poor performance on genuine observations (Decruyenaere et al., 2024; Giusti et al., 2025). Third, cost-sensitive bias emerges in methods that adjust learning based on class-dependent misclassification costs. These strategies heavily rely on the representational quality of the feature space; if feature selection is suboptimal or biased, the assigned costs may be misaligned with the actual structure of the data, further exacerbating imbalance-related challenges.

In addition to the various misconceptions surrounding imbalance handling, whether at the data or algorithmic level, the risk of bias is further amplified by the widespread use of multi-stage pipelines. Addressing credit scoring challenges has traditionally involved pipelines that combine feature selection, oversampling, and classifier training (Hou et al., 2020; Rao et al., 2023; Wang et al., 2024; Zhang et al., 2022). While each component is designed to enhance performance, their sequential application can introduce compounding biases(Rao et al., 2023; Wang et al., 2024; Zhang et al., 2022). Feature selection that ignores class imbalance may discard features critical for detecting minority class instances. Oversampling in a suboptimal feature space can further amplify noise or irrelevant patterns. Even when trained on rebalanced data, classifiers may continue to struggle with generalization. Moreover, cost-sensitive learning models, which aim to internalize imbalance by weighting misclassification penalties, are also susceptible: their effectiveness is tightly linked to the informativeness of the underlying feature space, which earlier stages may have already compromised. These interdependencies highlight how multi-level pipelines can unintentionally exacerbate the challenges of imbalanced credit scoring. Thus, more attention should be devoted to refining these approaches in future research, focusing on developing cohesive frameworks to minimize bias propagation.

Therefore, to overcome these challenges, we propose a unified methodology integrating feature extraction and classification within a single framework. Specifically, we employ a Variational Autoencoder (VAE) as a feature extractor, which learns latent representations designed to serve both the majority and minority classes equally, ensuring no class-specific patterns are overshadowed or lost. The model optimizes feature learning and prediction by embedding the classifier directly into the VAE architecture, eliminating the need for separate oversampling or post hoc adjustments. Moreover, using a weighted loss function during training further reinforces balanced learning by assigning greater importance to the minority class, ensuring that the extracted features and the classification boundaries are jointly shaped to handle class imbalance effectively without introducing bias. This cohesive design preserves the data’s real structure, mitigates the compounding biases of multi-step pipelines, and enhances robustness to economic shifts and uncertainties. Thus, the main contributions of this paper are:

The VAE incorporates a Neural Network Classifier that operates on the latent space to predict default probabilities.   
The Neural Network is weighted to prioritize learning from minority class instances, addressing class imbalance inherent in credit scoring datasets.   
. The VAE learns a lower-dimensional latent representation of both classes equally, capturing essential features while reducing dimensionality, thus improving model efficiency and performance.   
. A balanced Neural Network is trained using the latent representations from VAE, where higher weight is assigned to the classification loss in the Evidence Lower Bound (ELBO), ensuring robust classification performance.

● The methodology emphasizes minimizing Type II classification error (Bank losses).

The remainder of the paper is organized as follows: Section 2 reviews related literature to credit risk assessment, Section 3 presents the methodology operated, Section 4 depicts the experimental design and results, and Section 5 concludes the research while outlining potential directions for future studies.

# 2 Literature Review

In addressing the challenge of imbalance in credit scoring, researchers have traditionally treated imbalance handling as a preprocessing step isolated from subsequent feature selection and classification tasks. This multi-stage approach (Hou et al., 2020; Rao et al., 2023; Wang et al., 2024; Zhang et al., 2022) risks compounding errors and introducing biases at each stage, especially when the imbalance adjustment is decoupled from the predictive modeling itself. A dominant stream of research has focused on oversampling techniques such as SMOTE (Hou et al., 2020; Muslim et al., 2023), its variants (e.g., Borderline-SMOTE, ADASYN) (Rao et al., 2023; Wang et al., 2024), or more recently, generative models that synthesize minority class instances to balance the dataset (Zhang et al., 2022; Tingfei et al., 2020). However, while these methods improve class balance at the data level, they can distort the underlying data distribution, inflate feature correlations, and amplify noise, challenges that persistently undermine model generalizability.

To address these issues more nuancedly, Mou et al. (2024) introduced CERTIFIES, a cost-aware credit scoring framework that enhances imbalanced dataset quality through a novel pre-learning resampling approach. This method employs multiple assistant classifiers to identify and refine critical samples after oversampling, aiming to reduce noise and improve class separability while maintaining logistic regression as the primary classifier. Although CERTIFIES represents progress by incorporating a structured refinement mechanism, it still relies on a sequential pipeline of resampling followed by classification. This separation of imbalanced handling from the core model training can expose the modeling process to bias and limit adaptability when transferred to evolving economic contexts.

Similarly, Han et al. (2024) introduces the Non-parametric Oversampling Technique for Explainable credit scoring (NOTE), which combines a Non-parametric Stacked Autoencoder (NSA) for non-linear feature extraction, a conditional Wasserstein GAN (cWGAN) for minority class oversampling, and an explainability-focused classifier. NOTE demonstrates improved accuracy and stability compared to conventional oversampling techniques. However, it still follows a multi-stage framework, extracting latent features, generating synthetic samples, and performing classification as separate steps. This sequential design risks propagating biases across stages and, despite advances in explainability, does not integrate imbalance handling and classification into a unified learning process. Moreover, reliance on complex generative models introduces additional opacity.

While these generative approaches enhance minority class representation and improve performance in non-linear data contexts, interpretability and alignment with the binary nature of credit scoring remain ongoing concerns. For instance, Mancisidor et al. (2021) proposed training a Variational Autoencoder (VAE) to generate a latent space deliberately structured into multiple distinguishable clusters to enhance feature learning. While this approach offers an innovative perspective on latent space organization, it introduces key limitations in credit-scoring contexts. Given that creditworthiness is inherently binary, the creation of multiple latent clusters complicates interpretability and may hinder the operational clarity needed by financial practitioners. Bank employees, for example, may struggle to map these latent clusters back to actionable credit decisions. In contrast, our model maintains direct alignment with the binary classification objective by training an integrated classifier within the latent space, preserving practical usability and interpretability essential for credit risk assessment.

In a parallel multi-stage spirit, Wang et al. (2024) proposed an elaborate framework combining K-Means-SMOTE for oversampling, ResNet for feature extraction, and a two-layer LSTM for sequential learning, further optimized using the IDWPSO algorithm. Their hybrid ResNet-LSTM model, tested on datasets with extreme imbalance ratios (700:1 and 3:1), outperformed ten competing methods across multiple metrics. However, this approach reinforces the fragmentation of imbalance handling, feature extraction, and classification into distinct stages. While effective in specific datasets, such complex pipelines risk overfitting and complicate deployment, especially given their dependence on external optimization procedures and multiple algorithmic layers that can cover interpretability and reproducibility.

Moving toward more integrated solutions, Xiao et al. (2025) leveraged TabNet as the base classifier, enhanced with a custom loss function designed to account for the cost per individual sample, aiming to improve sensitivity to minority class misclassification. While this cost-sensitive approach offers a more unified response to class imbalance by directly adjusting the decision boundary, it is important to note that the loss function operates at the output prediction level rather than shaping the latent feature representations themselves. As a result, while classification outcomes are weighted, the underlying feature extraction process remains uninfluenced by the imbalance, which may limit the full potential of learning balanced, discriminative representations across both classes.

Most credit-scoring studies continue to adopt multi-stage strategies aimed at improving accuracy (see Table 1). Nevertheless, these fragmented approaches often introduce bias at various stages, increasing the risk of bank losses. This study proposes an integrated process that optimizes feature extraction, addresses imbalance and performs classification within a unified framework. Integrating these components into a single optimization method aims to mitigate biases and enhance overall model effectiveness in credit risk assessment.

Table 1 An overview of related credit risk assessment literature   

<table><tr><td>Works</td><td>Proposed Approaches</td><td>Methodology Type</td><td>Imbalance Handling</td></tr><tr><td>Mou et al. (2024)</td><td>CERTIFIESa</td><td>Fragmented</td><td>pre-learning resampling approach</td></tr><tr><td>Han et al. (2024)</td><td>NOTEb</td><td>Fragmented</td><td>NSA+cWGANd</td></tr><tr><td>Wang et al. (2024)</td><td>ResNete-LSTM</td><td>Fragmented</td><td>K-Means-SMOTE</td></tr><tr><td>Xiao et al. (2025)</td><td>ECS-SDEf</td><td>Integrated</td><td>Cost-Sensitive</td></tr></table>

The table summarizes recent credit risk assessment literature approaches, highlighting methodological frameworks and specific techniques employed to address class imbalance. a CRediT ScorIng Framework Based on ResamplIng and FeaturE Selection b Non-parametric Oversampling Technique for Explainable credit scoring c Non-parametric Stacked Autoencoder d Conditional Wasserstein GANs e Residual neural network f Example-dependent Cost-Sensitive learning based Selective Deep Ensemble

# 3 Methodology

# 3.1 The Variational AutoEncoder (VAE)

It represents a powerful tool for deep learning within the framework of Bayesian inference. Unlike vanilla autoencoders, VAEs incorporate distributional reasoning, offering a more sophisticated approach to data representation. Initially designed for computer vision tasks (Xiao et al., 2024), VAEs have recently gained attention in credit scoring, particularly for their potential to serve as effective data balancers in imbalanced classification problems (Tingfei et al., 2020; Xiao et al., 2024). At its core, a VAE consists of two primary components: the encoder and the decoder (Kingma & Welling, 2013).

The encoder maps the input data x and compresses it into a latent representation of a lower dimension d, represented by a set of probability distributions and described by a mean vector $\mu$ and a standard deviation vector $\sigma$ . These vectors define a Gaussian distribution $q _ { \phi } ( \mathbf { z } / \mathbf { x } )$ from which latent variables z are sampled. The encoder’s parameters $\phi$ represent how we approximate the distribution of latent variables z given the input data x.   
. The decoder’s role in a VAE is to reconstruct the input data $_ \textrm { x }$ from the sampled latent variables z. It is designed to generate data that closely resembles the original input, thereby defining a distribution $p _ { \theta } ( \mathbf { x } / \mathbf { z } )$ that models the likelihood of the data given the latent variables $\mathbf { Z }$ , where $\theta$ represents the parameters of the decoder.

# 3.1.1 Bayesian Foundations and Latent Space

VAEs aim to estimate the posterior distribution of the latent variables given the observed data; the intractability of directly computing the posterior $p ( \mathbf { z } / \mathbf { x } )$ lets the VAEs approximate it using the simpler distribution $q _ { \phi } ( \mathbf { z } / \mathbf { x } )$ , often a Gaussian. The learned latent space is a lower dimensional representation of the input data that captures its essential features (Monshizadeh et al., 2021). The Neural Network of the encoder learns through several layers to extract increasingly abstract features from the data, and this process involves several steps:

1. Parametrization of latent variables: The encoder outputs two components, the mean vector $\mu$ and the standard deviation $\sigma$ ; these vectors parameterize a Gaussian distribution $q _ { \phi } ( \mathbf { z } / \mathbf { x } )$ over the latent variables z.

2. Sampling from the latent distribution $:$ Using the reparameterization trick, we sample the latent variables $\mathbf { Z }$ from the Gaussian distribution defined by the mean $\mu$ and standard deviation $\sigma$ . This trick involves sampling an auxiliary noise $\epsilon$ from a standard normal distribution $N ( 0 , I )$ and transforming the latent variables as follows:

$$
\mathrm { z } = \mu + \sigma \odot \epsilon
$$

Where $\odot$ denotes element-wise multiplication. By expressing z in this way, the stochasticity is isolated in the noise term $\epsilon$ , which is independent of the model parameters. This formulation allows for the gradients to be backpropagated through the deterministic functions $\mu$ and $\sigma$ , facilitating efficient training of the VAE using standard gradient-based optimization techniques.

3. Reconstruction of the input data: The decoder takes the sampled latent variables z and attempts to reconstruct the original input data. This reconstruction is modeled by the distribution $p _ { \theta } ( \mathbf { x } / \mathbf { z } )$ parameterized by the decoder’s weights $\theta$ .

4. Training the VAE: The primary goal in training a VAE is to maximize the reconstruction log-likelihood l $ { \mathcal { I } } g p _ { \theta } (  { \mathbf { x } } /  { \mathbf { z } } )$ , ensuring the generated data closely resembles the original input. To achieve this, the VAE also includes a regularizer in the form of the Kullback-Leibler (KL) divergence. This regularizer encourages the latent variable distribution to be close to a prior distribution (typically a standard normal distribution), promoting diversity and preventing overfitting (Kingma & Welling, 2013). The combination of these two objectives forms the Evidence Lower Bound (ELBO), which the VAE aims to maximize. The ELBO can be expressed as:

$$
\mathcal { L } ( \mathbf { x } , \mathbf { z } ; \theta , \boldsymbol { \phi } ) = \mathbb { E } _ { q _ { \phi } ( \mathbf { z } | \mathbf { x } ) } [ \log p _ { \theta } ( \mathbf { x } | \mathbf { z } ) ] - \mathrm { K L } ( q _ { \phi } ( \mathbf { z } | \mathbf { x } ) \parallel p ( \mathbf { z } ) )
$$

In this expression:

The first term, $\mathbb { E } _ { q _ { \phi } ( \mathbf { z } | \mathbf { x } ) } [ \log p _ { \theta } ( \mathbf { x } | \mathbf { z } ) ]$ represents the expected reconstruction Loss, ensuring that the decoded samples are close to the original input data. The second term, $\mathrm { K L } ( q _ { \phi } ( \mathbf { z } | \mathbf { x } ) \parallel p ( \mathbf { z } ) )$ acts as a regularizer, aligning the learned latent distribution with the prior distribution.

By maximizing the ELBO, VAEs achieve a balance between accurately reconstructing the input data and maintaining a well-structured, diverse latent space. Maximizing the ELBO is equivalent to minimizing the negative ELBO. Thus, the loss function can be written as follows:

$$
\mathcal { L } _ { \mathrm { l o s s } } ( \mathbf { x } , \mathbf { z } ; \theta , \phi ) = - \mathcal { L } ( \mathbf { x } , \mathbf { z } ; \theta , \phi ) = - \mathbb { E } _ { q _ { \phi } ( \mathbf { z } | \mathbf { x } ) } [ \log p _ { \theta } ( \mathbf { x } | \mathbf { z } ) ] + \mathrm { K L } ( q _ { \phi } ( \mathbf { z } | \mathbf { x } ) \parallel p ( \mathbf { z } ) )
$$

That we can rewrite as :

$$
\mathcal { L } _ { \mathrm { l o s s } } = \mathcal { L } _ { \mathrm { r e c o n } } + \mathcal { L } _ { \mathrm { K L } }
$$

Where:

● $\mathcal { L } _ { \mathrm { l o s s } }$ represents the total loss of the model. $\mathcal { L } _ { \mathrm { r e c o n } }$ denotes the reconstruction loss, which measures the difference between the original input and its reconstructed version.   
${ \mathcal { L } } _ { \mathrm { K L } }$ represents the KL divergence, which regularizes the latent space to follow a desired prior distribution, typically a standard normal distribution.

Finally, we use gradient descent to optimize the Loss with respect to the parameters of the encoder $\phi$ and the decoder $\theta$ .

# 3.2 VAEs with Integrated Neural Network for Imbalanced Learning

Due to its intrinsic probabilistic design, the VAE framework is explicitly well-suited to integrating a balanced classifier that shapes the latent space to represent both classes equitably. Unlike non-parametric autoencoders, which focus solely on reconstruction (Han et al., 2024) and may overfit, the VAE imposes a structured probabilistic latent space through its KL divergence regularizer. This regularization ensures that latent representations are not merely accurate reproductions of the input but are also constrained to conform to a Gaussian prior distribution. Such a constraint combats overfitting and enforces continuity and completeness in the latent space (Doersch, 2016).

This smooth, dense latent space is inherently more malleable, providing a foundation where class boundaries can be shaped uniformly across both majority and minority classes. By directly integrating a balanced classification objective into the VAE, the model does not just passively learn representations but actively aligns its latent encoding to reflect class-discriminative structure. The latent space is thus co-shaped by three forces:

1. Reconstruction objective : ensures data fidelity by minimizing the difference between the input and its reconstruction.   
2. KL regularizer : maintains global latent structure by constraining the latent distribution toward a Gaussian prior, preventing collapse into sparse or overconcentrated regions.   
3. Balanced classification loss $:$ actively shapes the latent space by pulling representations of the minority and majority classes toward equal ground.

To operationalize this, we integrated a weighted neural network into our Variational Autoencoder (VAE) framework to tackle the class imbalance issue, leveraging the model’s latent representation for more effective classification. Unlike traditional feature selection methods, which often prioritize features that best describe the majority class (Atif & Salmi, 2022), our VAE-based approach constructs a latent space explicitly structured to capture meaningful variations in both classes, especially the minority class. Standard feature selection techniques may discard or underweight features crucial for minority-class identification (Hussin Adam Khatir & Bee, 2022), as these methods typically optimize for overall variance or predictive power without considering class balance.

Therefore, we introduce class weights into the Binary Cross-Entropy loss function and an additional $\alpha$ parameter to mitigate class imbalance. The $\alpha$ parameter explicitly controls the influence of the classification loss relative to the reconstruction and regularization terms, allowing for a fine-tuned balance between representation learning and classification performance. This ensures the minority class is meaningfully represented in the classifier’s decision boundary.

Additionally, our weighted loss scheme modifies gradient updates during backpropagation, making the latent space more responsive to class distinctions. By integrating VAE as an adaptive feature extractor alongside a weighted classification mechanism, our method (VAE-INN) ensures that the latent space is structured to benefit both classes equitably, overcoming the tendency of conventional feature selection methods to be biased toward the majority class (Longadge & Dongre, 2013); this leads to a more robust and class-sensitive approach to data representation in imbalanced credit scoring. The impact of the weighted loss on shaping the latent space through backpropagation is formally demonstrated in the following section, providing theoretical insight into how the VAE-INN model balances class representations.

Based on the above explanation, the loss can be elaborated as follows:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { \mathrm { l o s s } } = \sum _ { i = 1 } ^ { N } ( \mathbf { x } _ { i } - \hat { \mathbf { x } } _ { i } ) ^ { 2 } + \left( - 0 . 5 \sum _ { j = 1 } ^ { d } ( 1 + \log ( \sigma _ { j } ^ { 2 } ) - \mu _ { j } ^ { 2 } - \sigma _ { j } ^ { 2 } ) \right) } } \\ { { \displaystyle ~ + \alpha \left( - \sum _ { i = 1 } ^ { N } ( \mathbf { y } _ { i } \log ( \hat { \mathbf { y } } _ { i } ) + ( 1 - \mathbf { y } _ { i } ) \log ( 1 - \hat { \mathbf { y } } _ { i } ) ) \cdot w _ { i } \right) } } \end{array}
$$

That we can abstract as:

$$
\mathcal { L } _ { \mathrm { l o s s } } = \mathcal { L } _ { \mathrm { r e c o n } } + \mathcal { L } _ { \mathrm { K L } } + \alpha \mathcal { L } _ { \mathrm { c l a s s } }
$$

Where:

● $\hat { \mathbf { x } } _ { i }$ is the reconstructed value of the input $\mathrm { x } _ { i }$ . $\mu _ { j }$ and $\sigma _ { j }$ are the mean and standard deviation of the latent variable’s approximate posterior distribution. ${ \hat { \mathbf { y } } } _ { i }$ is the predicted probability for the class label $\mathrm { y } _ { i }$ .   
$\bullet$ $\alpha$ is the scaling factor for the binary cross-entropy term. $w _ { i }$ is the class weight for sample $i ,$ , , which can be calculated as:

$$
w _ { i } = w _ { l } = \frac { N } { 2 n _ { l } }
$$

） $N$ is the sample size. ） $n _ { l }$ is the number of samples in class $l$

# 3.2.1 Learning Data Representation for Imbalanced Learning Using Backpropagation

In imbalanced learning, integrating a neural network classifier into the VAE framework offers a promising approach to improve classification performance in minority classes. The VAE comprises an encoder, a decoder, and an integrated classifier. The encoder compresses input data into a latent space characterized by a mean and a log variance. The reparameterization trick is employed to sample latent variables, which are then passed to both the decoder for reconstruction and the classifier for prediction (see Figure 1). Training this model involves minimizing a composite loss function that includes a reconstruction loss, a regularization term (Kullback-Leibler Divergence), and a weighted classification loss (Binary Cross-Entropy). Introducing class weights and an $\alpha$ parameter in the classification loss ensures that the model places greater emphasis on accurately classifying minority classes. During backpropagation, gradients are computed and used to update the model parameters (Goodfellow et al., 2016). This process ensures that the latent space is optimized not only for accurate reconstruction but also for effective classification under class imbalance. This integrated approach enhances the VAE’s capability to generate meaningful representations and improve classification accuracy for imbalanced credit scoring datasets. Outlined below is a detailed breakdown of each component within the model, illustrating its specific function and contribution to the overall methodology:

# ● Variational Autoencoder Setup

Encoder: Compresses the input features into a latent representation that captures the main features of the data while reducing dimensionality. Latent Space Representation: This compressed representation is the foundation for both the reconstruction task (decoding) and the classification task. Decoder: Reconstructs the original input from the latent space to ensure the latent features retain sufficient information to depict the data accurately.

# ● Integrated Classifier

Latent Feature Input: The classifier receives the latent representation generated by the VAE encoder as input.   
Binary Cross-Entropy Loss with Class Weights: The classifier component applies a class-weighted Binary Cross-Entropy (BCE) loss function, which places greater emphasis on the minority class (defaults) by assigning higher weights; this helps the model focus on accurately identifying default cases despite the imbalance in the data.

![](images/6664b286716bd019650a35903e3060cbdb9d129f3a9699e61d63d78ce4790336.jpg)  
Fig. 1 Variational Autoencoder with Integrated Neural Network Classifier. This figure illustrates the architecture of the Variational Autoencoder model integrated with a neural network classifier (VAEINN). The encoder compresses the input data into a latent space characterized by mean $( \mu )$ and standard deviation $( \sigma )$ vectors. The reparameterization trick is employed to sample latent variables $\mathbf { Z }$ from the distribution defined by these vectors. The latent variables are then fed into both the decoder for data reconstruction and the classifier for prediction, enabling the model to effectively handle imbalanced learning tasks

$\alpha$ Weight Parameter for Classification Emphasis: The classifier’s loss is further scaled by an $\alpha$ parameter, which adjusts the importance of classification relative to the VAE’s reconstruction and regularization tasks. A higher $\alpha$ value increases the classifier’s influence, giving more sensitivity to the minority class.

# Optimization and Cross-Validation

Cross-Validation for $\alpha$ Tuning: To ensure optimal performance, cross-validation is used to tune $\alpha$ , finding a balance that maximizes classification accuracy for the minority class without degrading the VAE’s reconstruction quality. Gradient Adjustment via Class Weights: During backpropagation, the gradients are adjusted based on class weights, making the VAE-INN learning process more sensitive to the minority class. This adjustment helps the encoder develop a class-sensitive latent space with features that improve the model’s ability to distinguish between classes effectively.

In the subsequent sections, we will present a detailed derivation of the total loss gradients with respect to the VAE-INN parameters, elucidating how each component of the loss function contributes to the optimization process and enhances the model’s performance in handling imbalanced data.

Gradient of the Loss with Respect to the Encoder Parameters $( \phi )$ : To calculate the gradient with respect to the encoder parameters $\phi$ , let’s denote the decoder function as $f _ { \theta }$ :

$$
\hat { \mathbf { x } } = f _ { \theta } ( \mathbf { z } )
$$

where $\mathbf { Z }$ is the latent variable sampled during the reparameterization step:

$$
\begin{array} { r } { \mathrm { \Delta z } = \mu + \sigma \epsilon \quad \mathrm { w i t h } \quad \epsilon \sim \mathcal { N } ( 0 , I ) } \end{array}
$$

Since $\mu$ and $\sigma$ are functions of the encoder parameters $\phi$ :

$$
\mu = g _ { \phi } ( \mathbf { x } )
$$

$$
\log \sigma ^ { 2 } = h _ { \phi } ( \mathbf { x } ) \quad \mathrm { s o } \quad \sigma = \exp ( 0 . 5 \cdot h _ { \phi } ( \mathbf { x } ) )
$$

Let’s also denote the classifier function as:

$$
\hat { \mathbf { y } } = c _ { \psi } ( \mathbf { z } )
$$

Then, the gradient of the loss with respect to encoder’s parameters $\phi$ is:

$$
\frac { \partial \mathcal { L } _ { \mathrm { l o s s } } } { \partial \phi } = \frac { \partial \mathcal { L } _ { \mathrm { r e c o n } } } { \partial \phi } + \frac { \partial \mathcal { L } _ { \mathrm { K L } } } { \partial \phi } + \alpha \frac { \partial \mathcal { L } _ { \mathrm { c l a s s } } } { \partial \phi }
$$

The update of the encoder’s parameters at time $( t + 1 )$ is:

$$
{ \phi } ^ { ( t + 1 ) } \gets { \phi } ^ { ( t ) } - \eta \frac { \partial \mathcal { L } _ { \mathrm { l o s s } } } { \partial \phi }
$$

Where $\eta$ is the learning rate.

# ● Gradient of the Loss with Respect to the Decoder Parameters $\mathbf { \eta } ^ { ( \theta ) }$ :

$$
\frac { \partial \mathcal { L } _ { \mathrm { l o s s } } } { \partial \theta } = \frac { \partial \mathcal { L } _ { \mathrm { r e c o n } } } { \partial \theta } = \frac { \partial \mathcal { L } _ { \mathrm { r e c o n } } } { \partial \hat { \mathbf { x } } } . \frac { \partial \hat { \mathbf { x } } } { \partial \theta } = \frac { \partial \mathcal { L } _ { \mathrm { r e c o n } } } { \partial \hat { \mathbf { x } } } . \frac { \partial f _ { \theta } ( \mathbf { z } ) } { \partial \theta }
$$

The update of the decoder’s parameters at time $( t + 1 )$ is:

$$
\theta ^ { ( t + 1 ) } \gets \theta ^ { ( t ) } - \eta \frac { \partial \mathcal { L } _ { \mathrm { l o s s } } } { \partial \theta }
$$

# ● Gradient of the Loss with Respect to the classifier Parameters $( \psi )$ :

$$
\frac { \partial \mathcal { L } _ { \mathrm { l o s s } } } { \partial \psi } = \alpha \frac { \partial \mathcal { L } _ { \mathrm { c l a s s } } } { \partial \psi } = \alpha \frac { \partial \mathcal { L } _ { \mathrm { c l a s s } } } { \partial \hat { \mathrm { y } } } . \frac { \partial \hat { \mathrm { y } } } { \partial \psi } = \alpha \frac { \partial \mathcal { L } _ { \mathrm { c l a s s } } } { \partial \hat { \mathrm { y } } } . \frac { \partial c _ { \psi } ( \mathrm { z } ) } { \partial \psi }
$$

The update of the classifier’s parameters at time $( t + 1 )$ is:

$$
\psi ^ { ( t + 1 ) } \gets \psi ^ { ( t ) } - \eta \frac { \partial \mathcal { L } _ { \mathrm { l o s s } } } { \partial \psi }
$$

# 3.3 Classification Performance Evaluation

In bank credit scoring, accurately classifying borrowers is crucial for effectively managing credit risk. Traditional measures derived from confusion matrix components, such as True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN), are commonly used to evaluate classification performance (Başaran et al., 2019; Sertkaya et al., 2019). Banks aim to avoid misclassifying good borrowers as bad borrowers and vice versa, with varying priorities for each type of error. This study focuses on three key metrics: Type I error, Type II error, and AUC. Type I error is the rate of misclassifying a good borrower as a bad borrower, an essential consideration for maintaining business opportunities. Conversely, Type II error is defined as the rate of misclassifying a bad borrower as a good borrower, which is critical for lenders aiming to minimize financial risks. Finally, the AUC (Area Under the Receiver Operating Characteristic Curve) measures the model’s ability to distinguish between positive and negative classes, providing a comprehensive view of classification performance. These metrics influence the lender’s capacity to manage credit risk and make accurate loan approval decisions. The formulas for these metrics are as follows:

$$
\mathrm { e r r } _ { I } = \frac { F P } { F P + T N }
$$

$$
\mathrm { e r r } _ { I I } = \frac { F N } { F N + T P }
$$

$$
\operatorname { A U C } = \int _ { 0 } ^ { 1 } \operatorname { T P R } \left( \operatorname { F P R } ^ { - 1 } ( \nu ) \right) d \nu
$$

Where:

$\mathrm { e r r } _ { I }$ represents the Type I error.   
$\bullet$ $\mathrm { e r r } _ { I I }$ represents the Type II error.   
$\bullet$ TPR is the True Positive Rate (recall).

FPR is the False Positive Rate. $\bullet$ $\nu$ represents a point on the False Positive Rate (FPR) axis.

Our model prioritizes classification performance by integrating the VAE with a neural network classifier, particularly for minority classes; this reduces the critical type II error while maintaining business opportunities and improves overall decision-making in credit scoring.

# 4 Experiments and Results

# 4.1 Dataset

The Taiwan credit card clients dataset, collected by a Taiwanese bank, is a widely used benchmark for credit scoring and classification tasks. It is publicly available at the UCI Machine Learning Repository. The dataset comprises 30000 instances with 23 features, which are part of the dataset’s original structure and reflect the variables deemed relevant by the dataset creators for credit risk assessment. These features include demographic information, payment history, and billing amounts.

The continuous features include the amount of given credit (LIMIT_BAL), age of the client (AGE), bill statement amounts from April to September (BILL_ AMT1 to BILL_AMT6), and amounts paid from April to September (PAY_AMT1 to PAY_AMT6). The categorical features encompass gender (SEX), education level (EDUCATION), marital status (MARRIAGE), and repayment status from April to September (PAY_0 to PAY_6). The target variable, default.payment.next. month, indicates whether the client will default on their next payment $\mathrm { ( 1 = y e s }$ , $0 = \mathfrak { n } \mathfrak { o }$ ). Notably, the dataset presents a significant class imbalance, with 6636 instances of defaults $( 2 2 \% )$ and 23364 instances of non-defaults $( 7 8 \% )$ , resulting in an imbalance ratio of approximately 22:78 (see Table 2). We utilize the entire dataset in our experiments to develop and evaluate methods for addressing imbalanced classification challenges in credit scoring. By leveraging this dataset, our study aligns with established industry standards and existing research in credit scoring methodologies.

This study constructed a combined correlation matrix to evaluate relationships among numerical and categorical features in the dataset, enabling a holistic understanding of interdependencies. Pearson’s correlation coefficient was utilized for numerical-numerical relationships, as it quantifies the strength and direction of linear associations. Cramer’s V was applied for categorical-categorical relationships, measuring the association between categorical variables using the chi-squared statistic, normalized to account for differences in variable dimensions. To assess relationships between numerical and categorical variables, the Point-Biserial correlation coefficient was extended to work with non-binary categorical variables by treating them as scaled values. The goal of calculating the combined correlation matrix was to integrate these diverse measures into a single framework, facilitating the simultaneous analysis of mixed data types. The resulting matrix was visualized as a heatmap (see

Table 2 Description of Taiwan Credit Card Clients Dataset   

<table><tr><td>Feature</td><td>Description</td><td>Mean</td><td>Std. Dev.</td><td>Range (Min, Max)</td></tr><tr><td>LIMIT_BAL</td><td>Amount of credit given to the customer</td><td>167484</td><td>129747</td><td>(10000, 1000000)</td></tr><tr><td>AGE</td><td>Age of the customer in years</td><td>35.5</td><td>9.2</td><td>(21,79)</td></tr><tr><td>BILL_AMT1</td><td>Bill statement amount for September</td><td>51223</td><td>73635</td><td>(-165580, 964511)</td></tr><tr><td>BILL_AMT2</td><td>Bill statement amount for August</td><td>49179</td><td>71173</td><td>(-69777, 983931)</td></tr><tr><td>BILL_AMT3</td><td>Bill statement amount for July</td><td>47013</td><td>69349</td><td>(-157264, 1664089)</td></tr><tr><td>BILL AMT4</td><td>Bill statement amount for June</td><td>43262</td><td>64332</td><td>(-170000, 891586)</td></tr><tr><td>BILL_AMT5</td><td>Bill statement amount for May</td><td>40311</td><td>60797</td><td>(-81334, 927171)</td></tr><tr><td>BILL_AMT6</td><td>Bill statement amount for April</td><td>38871</td><td>59554</td><td>(-339603, 961664)</td></tr><tr><td>PAY_AMT1</td><td>Payment amount in September</td><td>5663</td><td>16563</td><td>(0, 873552)</td></tr><tr><td>PAY_AMT2</td><td>Payment amount in August</td><td>5921</td><td>23040</td><td>(0,1684259)</td></tr><tr><td>PAY_AMT3</td><td>Payment amount in July</td><td>5225</td><td>17606</td><td>(0, 896040)</td></tr><tr><td>PAY_AMT4</td><td>Payment amount in June</td><td>4826</td><td>15666</td><td>(0,621000)</td></tr><tr><td>PAY_AMT5</td><td>Payment amount in May</td><td>4799</td><td>15278</td><td>(0, 426529)</td></tr><tr><td>PAY_AMT6</td><td>Payment amount in April</td><td>5215</td><td>17777</td><td>(0,528666)</td></tr><tr><td>SEX</td><td>Gender of the customer (Male = 60.4%,Female = 39.6%)</td><td>Categorical variable</td><td></td><td></td></tr><tr><td>EDUCATION</td><td>Education level of the customer(Grad School, University,High School, Others)</td><td>Categorical variable</td><td></td><td></td></tr><tr><td>MARRIAGE</td><td>Marital status of the customer (Married = 53%, Single = 46%, Others = 1%)</td><td>Categorical variable</td><td></td><td></td></tr><tr><td>PAY_0 to PAY_6</td><td>Past repayment status (-1= on-time,O= no use, 1-8 = delays)</td><td>Categorical variable</td><td></td><td></td></tr><tr><td>default.payment</td><td>Whether the customer defaulted the next month (No = 78%,Yes=22%)</td><td>Class</td><td></td><td></td></tr></table>

Fig. 2), providing an intuitive representation of the strength and direction of relationships. This visualization aids in identifying significant patterns. From the heatmap, several key insights emerged. A negative relationship was observed between the past payment statuses $( \mathrm { P A Y } \_ { 0 }$ to PAY_6) and the amount of given credit (LIMIT_BAL); this suggests that payment delays are associated with lower credit limits, possibly reflecting stricter credit restrictions for individuals with a history of late payments, reflecting a potential credit risk factor. A positive relationship was evident between consecutive billing amounts (BILL_AMT1 to BILL_AMT6), suggesting that monthly billing amounts are highly correlated, likely reflecting stable spending or repayment patterns. Similarly, a positive correlation was observed between consecutive repayment statuses (PAY_0 to PAY_6), indicating that delays or punctual payments in one month are often consistent with those in subsequent months. These observations highlight meaningful behavioral trends, emphasizing the interconnected nature of the dataset’s features. Such insights are invaluable for understanding customer behavior and assessing creditworthiness.

![](images/500d485245ddb5d3526d5e694ecc491bbd2ac35e70f057a84d59a44ca35985dc.jpg)  
Fig. 2 Combined Correlation Matrix. The colors represent the strength and direction of relationships: blue indicates a strong negative correlation, red indicates a strong positive correlation, and lighter shades near white represent weaker correlations or no correlation

# 4.2 Data Pre-processing

The preprocessing involves several critical steps to prepare the data for analysis. Initially, the dataset is loaded, and continuous features undergo standardization. This process involves transforming each feature to have a mean of zero and a standard deviation of one. This transformation, known as z-score normalization, ensures that all continuous features are on a similar scale, which helps the VAE-INN model learn more effectively and improves convergence during training. Additionally, categorical features are converted into a one-hot encoded format. This transformation allows categorical variables to be represented as binary vectors, making them compatible with the model. The dataset is then split into training and test sets using an 80:20 ratio to evaluate the model’s performance. Moreover, class weights are computed to address the significant class imbalance in the dataset, where non-default instances greatly exceed default instances, as shown in Eq. 7. These weights are incorporated into the loss function to enhance the model’s sensitivity to the minority class, thereby improving its ability to detect defaults. The workflow for the credit scoring model (VAE-INN) involves five primary steps, as depicted in Figure 3. Initially, the dataset undergoes preprocessing, where continuous features are standardized, and categorical features are converted into a one-hot encoded format. Next, the dataset is split into training and testing sets using an 80:20 ratio to ensure the model is evaluated on unseen data. In the third step, class weights are calculated from the training set to address the significant class imbalance, enhancing the model’s sensitivity to the minority class (defaults). The fourth step involves training the VAE-INN model, comprising an encoder, decoder, and classifier. The encoder compresses the input data into a lower-dimensional latent space, the decoder reconstructs the input data from the latent representation, and the classifier predicts the likelihood of default based on the latent representation. Finally, the trained model is tested on the testing set. The encoder of the VAE-INN model transforms the testing set into the latent representation, and then the VAE-INN classifier categorizes it. Performance metrics such as Type I, Type II error, and AUC are calculated to evaluate the model’s effectiveness in predicting defaults.

# 4.3 Hyperparameter Settings

The Variational Autoencoder (VAE) model used in this study is configured with several critical hyperparameters to optimize its performance for credit scoring. The encoder component of the VAE comprises an input layer matching the dimensionality of the training data, a hidden layer with 128 units, and a latent layer with 12 dimensions. The encoder employs the ReLU activation function to introduce nonlinearity. The decoder mirrors the encoder’s structure, includes a hidden layer of 128 units, and uses the ReLU activation function. Its output layer, which reconstructs the input data, is sized to match the original input dimension. The classifier component, integrated into the VAE, consists of a hidden layer with 128 units and an output layer with a single unit, applying the Sigmoid activation function to produce probabilities for binary classification. For training, the model utilizes a batch size of 32, a learning rate of $1 \times 1 0 ^ { - 3 }$ , and is trained for 50 epochs using the Adam optimizer. The loss function combines binary cross-entropy (BCE), mean squared error (MSE), and Kullback-Leibler divergence (KLD) with an additional weight, alpha, to balance these components. The alpha parameter significantly influences the trade-off between reconstruction quality and classification accuracy and was optimized using cross-validation with five folds. The AUC (Area Under the Curve) metric was used as the performance criterion for tuning alpha, ensuring that the model’s sensitivity to the minority class was maximized while maintaining overall predictive performance. Table 3 lists the tuned hyperparameters used above.

![](images/68667ba2526f6c2d833509332b6be2267f59e413186dd5e9c9f40221481caa26.jpg)  
Fig. 3 Workflow diagram. The workflow for the credit scoring model using a Variational Autoencoder (VAE-INN) consists of five steps: 1. Preprocessing the Dataset. 2. Dividing the dataset into training and testing sets. 3. Computing class weights from the training set. 4. Training the model using the training set, which includes an encoder, decoder, and classifier. 5. Testing the trained model on the testing set and calculating performance metrics

# 4.4 VAE-INN Model Analysis

This section aims to evaluate the effectiveness of the proposed integrated model, "VAE-INN", against several strategies commonly employed in imbalanced credit scoring tasks. The goal is to highlight the potential distributional bias introduced when applying data-level augmentation in model settings. We included some of the recent literature’s most widely used classifiers to ensure a comprehensive and relevant comparison (Atif & Salmi, 2022; Muslim et al., 2023; Rao et al., 2023). The

Table 3 Hyperparameter Settings for the VAE-INN Model   

<table><tr><td>Component</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="4">Encoder</td><td>Input Dimension</td><td>Feature dimension</td></tr><tr><td>Hidden Layer Dimension</td><td>128</td></tr><tr><td>Latent Dimension</td><td>12</td></tr><tr><td>Activation Function</td><td>ReLU</td></tr><tr><td rowspan="4">Decoder</td><td>Latent Dimension</td><td>12</td></tr><tr><td>Hidden Layer Dimension</td><td>128</td></tr><tr><td>Output Dimension</td><td>Feature dimension</td></tr><tr><td>Activation Function</td><td>ReLU</td></tr><tr><td rowspan="4">Classifier</td><td>Latent Dimension</td><td>12</td></tr><tr><td>Hidden Layer Dimension</td><td>128</td></tr><tr><td>Output Dimension</td><td>1</td></tr><tr><td>Activation Function</td><td>ReLU</td></tr><tr><td rowspan="4">Training</td><td>Output Activation Function</td><td>Sigmoid</td></tr><tr><td>Batch Size</td><td>32</td></tr><tr><td>Learning Rate</td><td>1×10-3</td></tr><tr><td>Epochs</td><td>50</td></tr><tr><td rowspan="2">Loss Function</td><td>Optimizer</td><td>Adam BCE,MSE,KLD</td></tr><tr><td>Components Alpha</td><td>4.0</td></tr></table>

This table summarizes the hyperparameter settings used for the VAE-INN model. It includes configurations for the encoder, decoder, classifier, training parameters, and loss function components, providing a comprehensive overview of the model’s setup

evaluation used a consistent dataset split with a fixed random seed of 42 to ensure fairness and reproducibility. All models underwent the same preprocessing steps outlined in Section 4.2, ensuring uniformity in the analysis.

Feature selection (FS) was applied uniformly across all models to identify the most informative features. Cohen’s d effect size was utilized for numerical features, as it is well-suited for detecting differences between two classes in imbalanced datasets, emphasizing features with effect sizes exceeding a predefined threshold. Cramer’s V was employed to assess the importance of categorical features. These techniques ensured that the feature selection process focused on features most relevant to class differentiation. This step, denoted as "FS" in Table 4, was consistently integrated across all multi-stage strategies for a fair comparison. Two benchmark oversampling techniques, SMOTE and ADASYN, addressed class imbalance (Hou et al., 2020; Muslim et al., 2023). The classifiers used for comparison included Random Forest (RF), XGBoost, and Support Vector Machine (SVM) (Wang et al., 2022; Muslim et al., 2023; Rao et al., 2023). The multi-stage strategies tested against the cohesive, integrated VAE-INN model included (1) rebalancing the dataset before classifier training, and (2) performing feature selection followed by rebalancing before classifier training. By comparing these strategies with the integrated VAE-INN model, this section demonstrates the advantages of a cohesive approach combining feature extraction, imbalance handling, and classification into a unified framework. The details of the strategies evaluated are outlined below:

. VAE-INN: Variational Autoencoder with Integrated Neural Network. SMOTE+RF: Synthetic Minority Over-sampling Technique (SMOTE) applied before Random Forest classifier. ADASYN+RF: Adaptive Synthetic Sampling (ADASYN) applied before Ran

Table 4 Comparison of VAEINN Model with Multi-Stage Strategies   

<table><tr><td>Method</td><td>Type IError</td><td>Type II Error</td><td>AUC</td></tr><tr><td>VAE-INN</td><td>0.215</td><td>0.390</td><td>0.756</td></tr><tr><td>SMOTE+RF</td><td>0.076</td><td>0.605</td><td>0.742</td></tr><tr><td>ADASYN+RF</td><td>0.075</td><td>0.615</td><td>0.741</td></tr><tr><td>FS+SMOTE+RF</td><td>0.113</td><td>0.547</td><td>0.751</td></tr><tr><td>FS+ADASYN+RF</td><td>0.120</td><td>0.522</td><td>0.747</td></tr><tr><td>SMOTE+XGBoost</td><td>0.064</td><td>0.625</td><td>0.763</td></tr><tr><td>ADASYN+XGBoost</td><td>0.063</td><td>0.632</td><td>0.764</td></tr><tr><td>FS+SMOTE+XGBoost</td><td>0.087</td><td>0.568</td><td>0.768</td></tr><tr><td>FS+ADASYN+XGBoost</td><td>0.088</td><td>0.571</td><td>0.760</td></tr><tr><td>SMOTE+SVM</td><td>0.056</td><td>0.652</td><td>0.710</td></tr><tr><td>ADASYN+SVM</td><td>0.055</td><td>0.651</td><td>0.713</td></tr><tr><td>FS+SMOTE+SVM</td><td>0.167</td><td>0.419</td><td>0.755</td></tr><tr><td>FS+ADASYN+SVM</td><td>0.216</td><td>0.407</td><td>0.753</td></tr></table>

This table compares the integrated model (VAE-INN) with multistage strategies based on Type I error, Type II error, and AUC. It includes combinations of feature selection (FS), oversampling techniques (SMOTE and ADASYN), and classification algorithms (RF, XGBoost, and SVM), highlighting their performance in minimizing errors and maximizing classification accuracy

dom Forest classifier.   
● FS+SMOTE+RF: Feature Selection before SMOTE then Random Forest classifier.   
FS+ADASYN+RF: Feature Selection before ADASYN then Random Forest classifier.   
● SMOTE $+$ XGBoost: SMOTE applied before XGBoost classifier.   
ADASYN $+$ XGBoost: ADASYN applied before XGBoost classifier.   
FS+SMOTE $^ +$ XGBoost: Feature Selection before SMOTE then XGBoost classifier.   
FS+ADASYN $+$ XGBoost: Feature Selection before ADASYN then XGBoost classifier.   
SMOTE $^ +$ SVM: SMOTE applied before Support Vector Machine classifier.   
. ADASYN+SVM: ADASYN applied before Support Vector Machine classifier. FS+SMOTE $^ +$ SVM: Feature Selection before SMOTE then Support Vector Machine classifier. FS+ADASYN $^ +$ SVM: Feature Selection before ADASYN then Support Vector Machine classifier.

In evaluating the performance of the proposed integrated model VAE-INN against various multi-stage strategies, several key insights emerge. As illustrated in Figure 4, models that apply data-level augmentation techniques, such as SMOTE or ADASYN, directly before classification (e.g., SMOTE+RF, ADASYN+RF, SMOTE $^ +$ SVM, ADASYN $^ +$ SVM), tend to introduce distributional bias (Salmi et al., 2024). In this phenomenon, the synthetic samples generated during oversampling alter the original data distribution in a way that can degrade the model’s ability to generalize to the minority class. Specifically, the model may learn from augmented data that does not reflect real conditions, leading to overfitting on synthetic patterns and reduced generalization performance on real test data. This limitation is reflected in the elevated Type II error rates observed for these strategies, such as 0.605 for SMOTE+RF and 0.652 for SMOTE $^ +$ SVM, despite their relatively low Type I error rates. Incorporating feature selection prior to data-level augmentation (e.g., FS+SMOTE $+$ RF, FS+ADASYN+RF, FS+SMOTE $+$ XGBoost, $\mathrm { F S + A D A S Y N + X G B o o s t ) }$ helps mitigate this issue, as feature selection narrows the augmentation process to the most informative features, reducing Type II errors (for example, 0.547 in FS+SMOTE+RF and 0.568 in FS+SMOTE $^ +$ XGBoost) and improving AUC performance.

![](images/bff3718b468753e4060366168cc67dacfac241fe3c304ef330f94609885a0d4d.jpg)  
Fig. 4 Comparison of the Integrated Model (VAE-INN) with Different Multi-Stage Models. The top row illustrates the Type I and Type II error values for various methods, while the bottom row shows the AUC values. These comparisons highlight the integrated model’s effectiveness compared to various multi-stage models in handling imbalanced datasets

Nevertheless, the proposed integrated model VAE-INN outperforms these multistage pipelines, achieving the lowest Type II error of 0.390 and an AUC of 0.756. By addressing feature extraction, rebalancing, and classification within a unified architecture, VAE-INN effectively manages class imbalance without introducing external distributional shifts, resulting in superior overall classification performance. These findings emphasize the advantage of integrated frameworks like VAE-INN in ensuring robust generalization to real-world data.

# 4.5 Ablation Study

This section aims to uncover two critical sources of bias that can undermine the performance of imbalanced learning methods: representational bias and cost-sensitive bias. Representational bias arises in generative model-based approaches, including variational autoencoders. Although these methods aim to approximate the underlying data distribution, inaccuracies in the learned representation can generate unrealistic or noisy minority class examples. This can lead the classifier to overfit synthetic patterns, ultimately damaging its ability to generalize to real, unseen data, thereby worsening performance on the minority class. This analysis reveals representational bias by evaluating the VAE+NN (Oversampling) model, where the VAE is used to generate rebalanced data before classification. Costsensitive bias emerges in methods that incorporate class-dependent misclassification costs. These approaches rely heavily on the quality and representativeness of the feature space; when feature selection is suboptimal or biased, cost adjustments may misalign with the actual data structure, exacerbating imbalance-related challenges instead of mitigating them. Here, cost-sensitive bias is highlighted through evaluating the $\mathrm { V A E + N N }$ (FS) model, where the VAE is used solely for feature extraction before applying a classifier. Therefore, the current section compares the proposed cohesive model, which integrates a classifier within a Variational Autoencoder (VAE-INN), against these two multi-stage strategies. Specifically, we evaluate (1) VAE+NN (FS) and (2) VAE+NN (Oversampling), allowing us to demonstrate how integrating feature representation, imbalance Handling, and classification into a single model can more effectively address these biases. All models employ a consistent classification probability threshold of 0.5, with results summarized in Table 5 and Figure 5.

Table 5 Performance Comparison of Ablation Models Under Varying Random Seeds   

<table><tr><td colspan="4">VAE-INNa</td><td colspan="3">VAE+NN (FS)b</td><td colspan="3">VAE+NN (oversampling)</td></tr><tr><td>Seed</td><td>err I</td><td>errII</td><td>AUC</td><td>err I</td><td>err II</td><td>AUC</td><td>errI</td><td>err II</td><td>AUC</td></tr><tr><td>42</td><td>0.215</td><td>0.390</td><td>0.756</td><td>0.080</td><td>0.663</td><td>0.704</td><td>0.068</td><td>0.621</td><td>0.764</td></tr><tr><td>5</td><td>0.203</td><td>0.401</td><td>0.754</td><td>0.061</td><td>0.641</td><td>0.711</td><td>0.049</td><td>0.630</td><td>0.779</td></tr><tr><td>122</td><td>0.200</td><td>0.422</td><td>0.741</td><td>0.082</td><td>0.662</td><td>0.689</td><td>0.052</td><td>0.635</td><td>0.773</td></tr><tr><td>314</td><td>0.213</td><td>0.411</td><td>0.745</td><td>0.094</td><td>0.685</td><td>0.686</td><td>0.055</td><td>0.635</td><td>0.767</td></tr><tr><td>99</td><td>0.184</td><td>0.458</td><td>0.734</td><td>0.067</td><td>0.634</td><td>0.685</td><td>0.047</td><td>0.656</td><td>0.763</td></tr><tr><td>2024</td><td>0.191</td><td>0.431</td><td>0.732</td><td>0.073</td><td>0.622</td><td>0.692</td><td>0.053</td><td>0.627</td><td>0.765</td></tr><tr><td>56</td><td>0.227</td><td>0.383</td><td>0.752</td><td>0.083</td><td>0.603</td><td>0.705</td><td>0.051</td><td>0.660</td><td>0.779</td></tr><tr><td>867530</td><td>0.179</td><td>0.427</td><td>0.751</td><td>0.058</td><td>0.678</td><td>0.706</td><td>0.048</td><td>0.677</td><td>0.773</td></tr><tr><td>101</td><td>0.195</td><td>0.427</td><td>0.740</td><td>0.050</td><td>0.705</td><td>0.684</td><td>0.055</td><td>0.617</td><td>0.771</td></tr><tr><td>7</td><td>0.204</td><td>0.438</td><td>0.739</td><td>0.062</td><td>0.691</td><td>0.681</td><td>0.077</td><td>0.589</td><td>0.762</td></tr><tr><td>Mean</td><td>0.201</td><td>0.418</td><td>0.744</td><td>0.071</td><td>0.658</td><td>0.694</td><td>0.055</td><td>0.634</td><td>0.769</td></tr></table>

This table presents the Type I error (err I), Type II error (err II), and Area Under the Curve (AUC) for different models across various random seeds. The models compared are VAE with integrated Neural Network (VAE-INN), Neural Network classifier using VAE’s latent space as feature selection (VAE+NN (FS)), and Neural Network classifier using VAE’s latent space for oversampling (VAE+NN (oversampling)).

a VAE model with integrated Neural Network.

b Using the latent space of a VAE model as feature selection for neural network classifier training.

c Using the latent space of a VAE model to oversample the minority class for neural network classifier training

# 4.5.1 Models Description

# 1. VAE+NN (FS) (Neural Network Classifier with Feature Selection)

Architecture: This model uses a Variational Autoencoder (VAE) with an encoder and decoder (same as described in Section 4.3) to create a latent space representation. This latent space is then used for feature selection in training a separate neural network classifier. . Neural Network Details: The classifier has two hidden layers with 128 and 64 neurons, respectively, and an output layer with a sigmoid activation function. It is trained for 50 epochs. Loss Function: A weighted binary cross-entropy loss addresses class imbalance.

# 2. VAE+NN (Oversampling) (Neural Network Classifier with Oversampling)

Architecture: This approach uses a similar architecture, using the VAE’s latent space. However, it uses synthetic samples to balance the dataset rather than feature selection. Neural Network Details: The neural network has two hidden layers with 128 and 64 neurons and a sigmoid output layer trained for 50 epochs.

![](images/cd1e56d2a543206a7d294b56c5ee49dac57fa16c9083aa0799e113b6d98885af.jpg)  
Fig. 5 Ablation study results. This figure presents side-by-side box plots comparing the Type I Error and Type II Error metrics on the first row and the AUC metric on the second row across three different models: VAE-INN, $\mathrm { V A E ^ { + } N N }$ (FS), and $\mathrm { V A E ^ { + } N N }$ (Oversampling)

● Loss Function: The binary cross-entropy loss is not weighted due to dataset balancing through oversampling.

# 4.5.2 Ablation Study Analysis

The primary goal of this ablation study is to uncover and address two critical sources of bias that can undermine the performance of imbalanced learning methods: representational bias and cost-sensitive bias. These biases are revealed through the performance of the VAE+NN (Oversampling) and VAE+NN (FS) models. The integrated VAE-INN model achieves the most balanced performance across all metrics. Despite having the highest Type I error (mean: 0.201), it consistently records the lowest Type II error (mean: 0.418), which is crucial for minimizing missed detections of defaulters and thus reducing financial risk for banks. In contrast, the VAE+NN (FS) model achieves the lowest Type I error (0.071), but this comes at the cost of severely degraded recall, reflected in the highest Type II error (0.658) and the lowest AUC (0.694). This trade-off highlights cost-sensitive bias, as the model relies on class-dependent cost weighting applied to a fixed latent space, where the majority class primarily shapes the representations. As a result, the learned features may be suboptimal and misaligned with the actual structure of the data, and this is why it achieves the lowest Type I error, favoring correct classification of the majority class at the expense of increased misclassification of the minority class.

Meanwhile, the $\mathrm { V A E + N N }$ (Oversampling) model reports the lowest Type I error and the highest AUC (0.769), suggesting strong ranking ability. However, it still suffers from a high Type II error (0.634), exposing a representational bias where the generative process produces synthetic minority samples that are not fully representative of actual default cases, leading to overfitting. This also suggests potential optimism bias, where favorable AUC values mask poor recall. Overall, while the VAE+NN (FS) and $\mathrm { V A E + N N }$ (Oversampling) models each address imbalance through isolated stages, they remain vulnerable to distinct forms of bias. In contrast, the integrated VAE-INN model avoids these pitfalls by learning classification-guided representations, resulting in more robust generalization. Given the study’s objective of minimizing bank losses, particularly under crisis conditions, the VAE-INN model offers the most effective and reliable solution. A statistical analysis will follow to validate these findings and quantify the significance of observed differences.

# 4.5.3 Results of Statistical Tests

The statistical results presented in Tables 6, 7 and 8 offer a comprehensive analysis of the performance of three different models-VAE-INN (an integrated model), $\mathrm { V A E + N N }$ (FS), and VAE+NN (Oversampling) (both multi-stage strategies)-in terms of Type I Error, Type II error and Area Under the Curve (AUC). This discussion will interpret these results, focusing on their implications for minimizing bank losses.

Type I Error Analysis: Type I Error is a less critical metric than Type II Error in credit scoring, reflecting the rate at which non-defaulters are incorrectly classified as defaulters. Although a high Type I Error may result in missed lending opportunities and potential losses for the bank, its impact is generally less severe than failing to identify defaulters (Type II Error). The Friedman test revealed significant differences in Type I Error between the three models $( \chi ^ { 2 } = 1 6 . 2 8$ , $P - v a l u e < 0 . 0 5 )$ . Pairwise Mann-Whitney U tests further highlighted these distinctions, showing that the VAE-NN-OVER model, designed primarily to im

Table 6 Statistical Results for Type I Error   

<table><tr><td>Test</td><td>Comparison</td><td>statistic</td><td>P-value</td></tr><tr><td>Friedman Test</td><td></td><td>16.2800</td><td>0.0002</td></tr><tr><td rowspan="2">Mann-Whitney U Test (Less)</td><td>VAE-NN-OVERa vs.VAE-INNb</td><td>0.0</td><td>0.0000</td></tr><tr><td>VAE-NN-OVER vs.VAE-NN-FSc</td><td>16.0</td><td>0.0056</td></tr><tr><td></td><td>VAE-NN-FS vs.VAE-INN</td><td>0.0</td><td>0.0000</td></tr></table>

This table provides a detailed statistical analysis of the Type I Error across different models. It includes results from the Friedman and pairwise Mann-Whitney U tests to compare the models’ performance. The alternative hypothesis for the Mann-Whitney U tests is one-sided, indicating that one model’s Type I error is less than the other’s.

a Using the latent space of a VAE model to oversample the minority class for neural network classifier training.

b VAE model with integrated Neural Network.

c Using the latent space of a VAE model as feature selection for neural network classifier training

Table 7 Statistical Results for Type II Error   

<table><tr><td>Test</td><td>Comparison</td><td>statistic</td><td>P-value</td></tr><tr><td>Friedman Test</td><td></td><td>18.2000</td><td>0.0001</td></tr><tr><td rowspan="2">Mann-Whitney U Test (Less)</td><td>VAE-INNa vs.VAE-NN-FSb</td><td>0.0</td><td>0.0000</td></tr><tr><td>VAE-INNvs.VAE-NN-OVER</td><td>0.0</td><td>0.0000</td></tr><tr><td></td><td>VAE-NN-OVERvs.VAE-NN-FS</td><td>15.5</td><td>0.0050</td></tr></table>

This table provides a detailed statistical analysis of the Type II Error across different models. It includes results from the Friedman and pairwise Mann-Whitney U tests to compare the models’ performance. The alternative hypothesis for the Mann-Whitney U tests is one-sided, indicating that one model’s Type II error is less than the other’s.

a VAE model with integrated Neural Network.

b Using the latent space of a VAE model as feature selection for neural network classifier training.

c Using the latent space of a VAE model to oversample the minority class for neural network classifier training

Table 8 Statistical Results for AUC   

<table><tr><td>Test</td><td>Comparison</td><td>statistic</td><td>P-value</td></tr><tr><td>Friedman Test</td><td></td><td>20.0000</td><td>0.0000</td></tr><tr><td rowspan="2">Mann-Whitney U Test (Greater)</td><td>VAE-INNa vs.VAE-NN-FSb</td><td>100.0</td><td>0.0001</td></tr><tr><td>VAE-NN-OVERvs.VAE-INN</td><td>100.0</td><td>0.0001</td></tr><tr><td></td><td>VAE-NN-OVERvs.VAE-NN-FS</td><td>100.0</td><td>0.0001</td></tr></table>

This table provides a detailed statistical analysis of the AUC across different models. It includes results from the Friedman and pairwise Mann-Whitney U tests to compare the models’ performance. The alternative hypothesis for the Mann-Whitney U tests is one-sided, indicating that one model’s AUC is greater than the other’s

a VAE model with integrated Neural Network.

b Using the latent space of a VAE model as feature selection for neural network classifier training.

c Using the latent space of a VAE model to oversample the minority class for neural network classifier training.

prove positive class accuracy (defaulters), achieved the lowest mean Type I Error (0.055), albeit at the cost of a higher Type II Error. Conversely, the integrated model VAE-INN not only statistically exhibited the highest Type I Error but also demonstrated the lowest Type II Error, making it more effective in accurately identifying defaulters while maintaining a balanced classification approach.

Type II Error Analysis: Type II Error (err II) is a critical metric in the context of credit scoring, as it represents the rate at which positive cases (defaults) are incorrectly classified as negatives (non-defaults). The Friedman test yielded a statistic of 18.20 with a $P - v a l u e < 0 . 0 5$ , indicating significant differences in the Type II Error between the three models. The pairwise Mann-Whitney U tests further elucidate these differences, and the results underscore that the VAE-INN model consistently demonstrates the lowest Type II Error (mean of 0.418), making it the most effective in correctly identifying defaulters. Consequently, this model is the best choice for minimizing bank losses, as it reduces the likelihood of incorrectly classifying defaulters as non-defaulters.

AUC Analysis: The AUC metric measures the model’s ability to discriminate between positive and negative cases. Although a high AUC is generally desirable, it is essential to consider the context in which the model will be applied. The Friedman test for AUC produced a statistic of 20.0 with a $P - v a l u e < 0 . 0 5$ , indicating significant differences in AUC among models. The pairwise Mann-Whitney U tests revealed the superiority of the $\mathrm { V A E + N N }$ (Oversampling), but this result should be interpreted with caution due to the model’s high Type II Error. The model’s high AUC may be partly attributed to its effectiveness in identifying nondefaults (true negatives), resulting in a low false positive rate. This can lead to an inflated AUC score, as the model may excel at distinguishing non-defaults but still struggle with correctly identifying defaults. Consequently, while the AUC is high, the model’s high Type II Error indicates that it may not be as effective in practical scenarios where accurately identifying defaulters is crucial.

This study aims to minimize bank losses, particularly during financial crises. High Type II Error can result in significant financial repercussions for banks, as undetected defaulters continue to accrue losses. In this context, the VAE-INN model emerges as the most effective option, given its superior performance in correctly identifying positive cases and thus reducing Type II Error. The model also achieves this without causing a substantial loss of opportunities, as its Type I Error remains within an acceptable range. Additionally, the VAE-INN model demonstrates a reasonable AUC, indicating its solid discriminative ability. This combination of low Type II Error, adequate AUC, and minimal opportunity loss makes the model effective at identifying defaulters and distinguishing between defaulters and non-defaulters, aligning well with the study’s objectives to minimize financial losses. The following section will further evaluate the robustness of this model by testing its performance across several hyperparameter and imbalance ratio configurations to ensure its reliability and consistency.

# 4.6 Evaluation of the VAE-INN Model Robustness

To assess the proposed model’s robustness, we systematically analyzed the effects of varying the latent dimension and the imbalance ratio within the dataset on its performance.

The imbalance ratios, represented as Minority: Majority (e.g., 1:30, 1:50, or 1:100), were designed to reflect increasing levels of class imbalance, where a single minority instance corresponds to 30, 50, or 100 majority instances, respectively. These ratios were achieved by resampling the minority class within the training set while keeping the test set unchanged to preserve its original distribution and integrity. Multiple random seeds were used during the resampling process to ensure robust evaluation. Additionally, these experiments were paired with variations in the latent dimensions to examine how the latent representation interacts with escalating class imbalance. The results, illustrating the model’s performance across these conditions, are detailed in Table 9.

The proposed model demonstrated robust performance under extreme imbalance ratios (see Table 9), maintaining consistent AUC values alongside relatively balanced Type I and Type II errors across various configurations. This robustness highlights its capability to effectively handle highly skewed class distributions without significant degradation in performance. Across different latent dimensions, the results revealed only slight variations in the three metrics (Type I error, Type II error, and AUC), further underscoring the stability of the approach. However, larger latent dimensions, such as 16, introduced more significant variability in the results, particularly under severe imbalance conditions. This suggests a trade-off inherent in increasing the model’s representative capacity. While it allows for capturing more nuanced data patterns, it may also heighten the risk of overfitting, as evidenced by the less stable performance metrics. The proposed model outperformed the multi-stage models presented in Table 4, even under extreme imbalances. These multi-stage models, which combine feature selection, rebalancing, and classification in separate stages, struggled to achieve comparable performance levels. This demonstrates the advantage of an integrated, end-to-end approach in addressing complex classification tasks involving extreme class imbalances. Therefore, these results reinforce the importance of adopting innovative, integrated approaches in tackling imbalanced data problems in financial decision-making scenarios.

Table 9 VAE-INN Robustness Results Across Latent Dimensions and Imbalance Ratios Generated Using Different Seeds   

<table><tr><td>Seed</td><td>Latent Dimension</td><td colspan="3">Imbalance Ratio (Minority:Majority)</td></tr><tr><td rowspan="2">42</td><td></td><td>1:30</td><td>1:50</td><td>1:100</td></tr><tr><td>8</td><td>Type I: 0.183 Type II: 0.447 AUC: 0.741</td><td>Type I: 0.204 Type II: 0.416 AUC: 0.749</td><td>Type I: 0.191 Type II: 0.461 AUC: 0.730</td></tr><tr><td rowspan="4">123</td><td>12</td><td>Type I: 0.242 Type II: 0.372 AUC: 0.746</td><td>Type I: 0.207 Type II: 0.442 AUC: 0.736</td><td>Type I: 0.216 Type II: 0.443 AUC: 0.724</td></tr><tr><td>16</td><td>Type I: 0.259 Type II: 0.372 AUC: 0.738</td><td>Type I: 0.226 Type II: 0.402 AUC: 0.738</td><td>Type I: 0.197 Type II: 0.496 AUC: 0.709</td></tr><tr><td>8</td><td>Type I: 0.202 Type II: 0.431 AUC: 0.740</td><td>Type I: 0.216 Type II: 0.428 AUC: 0.734</td><td>Type I: 0.234 Type II: 0.426 AUC: 0.726</td></tr><tr><td>12</td><td>Type I: 0.204 Type II: 0.434 AUC: 0.742</td><td>Type I: 0.220 Type II: 0.423 AUC: 0.741</td><td>Type I: 0.173 Type II: 0.488 AUC: 0.728</td></tr><tr><td rowspan="4">234</td><td>16</td><td>Type I: 0.176 Type II: 0.447 AUC: 0.750</td><td>Type I: 0.216 Type II: 0.429 AUC: 0.732</td><td>Type I: 0.183 Type II: 0.477 AUC: 0.719</td></tr><tr><td>8</td><td>Type I: 0.164 Type II: 0.447 AUC: 0.743</td><td>Type I: 0.186 Type II: 0.450 AUC: 0.736</td><td>Type I: 0.214 Type II: 0.444 AUC: 0.760</td></tr><tr><td>12</td><td>Type I: 0.255 Type II: 0.377 AUC: 0.744</td><td>Type I: 0.227 Type II: 0.412 AUC: 0.731</td><td>Type I: 0.266 Type II: 0.405 AUC: 0.723</td></tr><tr><td>16</td><td>Type I: 0.177 Type II: 0.445 AUC: 0.748</td><td>Type I: 0.260 Type II: 0.373 AUC: 0.738</td><td>Type I: 0.273 Type II: 0.401 AUC: 0.719</td></tr><tr><td rowspan="4">345</td><td>8</td><td>Type I: 0.196 Type II: 0.440 AUC: 0.747</td><td>Type I: 0.217 Type II: 0.408 AUC: 0.745</td><td>Type I: 0.253 Type II: 0.381 AUC: 0.733</td></tr><tr><td>12</td><td>Type I: 0.214 Type II: 0.415 AUC: 0.741</td><td>Type I: 0.190 Type II: 0.439 AUC: 0.741</td><td>Type I: 0.194 Type II: 0.464 AUC: 0.720</td></tr><tr><td>16</td><td>Type I: 0.189 Type II: 0.434 AUC: 0.738</td><td>Type I: 0.178 Type II: 0.460 AUC: 0.738</td><td>Type I: 0.242 Type II: 0.441 AUC: 0.716</td></tr><tr><td>8</td><td>Type I: 0.206 Type II: 0.426 AUC: 0.739</td><td>Type I: 0.194 Type II: 0.450 AUC: 0.733</td><td>Type I: 0.270 Type II: 0.395 AUC: 0.716</td></tr><tr><td rowspan="3"></td><td>12</td><td>Type I: 0.198 Type II: 0.456</td><td>Type I: 0.154 Type II:</td><td>Type I: 0.182 Type II: 0.511 AUC: 0.706</td></tr><tr><td>16</td><td>AUC: 0.726 Type I: 0.234 Type II: 0.399</td><td>0.498 AUC: 0.739 Type I: 0.226 Type II:</td><td>Type I: 0.158 Type</td></tr><tr><td></td><td>AUC: 0.739</td><td>0.423 AUC: 0.739</td><td>II: 0.551 AUC: 0.700</td></tr></table>

This table presents the VAE-INN performance for different imbalance ratios (Minority:Majority) and latent dimensions. The imbalance ratios are generated by resampling the minority class using various random seeds. The metrics shown include Type I error, Type II error, and AUC for each configuration

# 5 Conclusion and Future Works

This study introduced a novel methodology for imbalanced credit scoring: the Variational Autoencoder with Integrated Neural Network and Weighted Loss (VAE-INN). The evaluation of this approach alongside various multi-stage models revealed several key insights and benefits:

Superior Performance: The proposed VAE-INN model demonstrates competitive performance, achieving the lowest Type II error, which indicates that the integrated approach effectively handles class imbalance, leading to improved classification accuracy and reduced error rates. Moreover, it maintains a balanced trade-off concerning Type I errors. This ensures that minimizing Type II errors does not come at the cost of higher false positive rates, resulting in more robust and equitable classification performance.   
. Avoidance of Biases: One of the primary advantages of the VAE-INN methodology is its ability to avoid introducing biases in the analysis. By integrating feature selection, imbalance handling, and classification within a single framework, the model minimizes the risk of overfitting and ensures a more balanced and fair evaluation of the data. The weighted loss function further enhances this by appropriately penalizing misclassifications of the minority class, leading to more reliable and unbiased predictions.   
. Comprehensive Multi-Stage Strategy: VAE-INN’s integrated nature offers a comprehensive and cohesive solution for imbalanced credit scoring. This comprehensive approach simplifies the modeling process and ensures that each stepfeature selection, imbalance handling, and classification-works synergistically to optimize the overall performance, instilling confidence in its effectiveness. Robustness: The proposed model demonstrated robust performance under severe imbalance ratios ranging from 1:30 to 1:100. Despite the challenges posed by such extreme class imbalances, the model maintained competitive results, outperforming or matching the performance of multi-stage methods. This highlights the effectiveness and adaptability of the integrated approach in handling diverse imbalance scenarios without compromising classification quality.   
● Practical Implications and Deployment Considerations: The VAE-INN model provides a robust and unified framework for imbalanced credit scoring, delivering tangible benefits for real-world financial applications. Its integrated structure effectively addresses class imbalance and mitigates analytical biases, making it well-suited for institutions dealing with skewed datasets. However, the model’s reliance on deep architectures and latent representations introduces interpretability challenges, particularly important in high-stakes financial contexts where

transparency is essential. The learned latent features are not inherently intuitive, potentially complicating model validation and regulatory compliance. Deploying the VAE-INN model also presents practical challenges. These include model governance, integration with legacy systems, and the need for careful hyperparameter tuning to maintain performance across diverse credit profiles and market conditions. Addressing these issues requires incorporating interpretability techniques, such as SHAP values, rigorous validation procedures, and a robust postdeployment monitoring strategy.

In conclusion, the VAE-INN methodology is a robust and unbiased solution for imbalanced credit scoring. Its integrated framework ensures enhanced performance and reliability. While the VAE-INN model offers clear advantages, future research could explore the potential of transformer-based architectures for credit scoring tasks, particularly their ability to model complex dependencies in tabular data. These models may offer alternative ways to learn expressive latent representations and improve classification under imbalance.

Acknowledgements The author extends her gratitude to the anonymous reviewers for their valuable feedback and constructive suggestions, which significantly contributed to the enhancement of this manuscript. Their expertise and thorough reviews significantly improved the overall quality of the research.

# Declarations

Conflict of interest/Competing interests The author declares no conflicts of interest regarding the research presented in this paper.

Code availability The code used in this research will be shared upon reasonable request.

# References

Arora, N., & Kaur, P. D. (2020). A bolasso based consistent feature selection enabled random forest classification algorithm: An application to credit risk assessment. Applied Soft Computing, 86, 105936.   
Atif, D., & Salmi, M. (2022). The most effective strategy for incorporating feature selection into credit risk assessment. SN Computer Science, 4(2), 96.   
Başaran, E., Cömert, Z., Şengür, A., Budak, Ü., Çelik, Y., & Toğaçar, M. (2019). Chronic tympanic membrane diagnosis based on deep convolutional neural network. In 2019 4th International Conference on Computer Science and Engineering (ubmk) (pp. 1–4).   
Decruyenaere, A., Dehaene, H., Rabaey, P., Polet, C., Decruyenaere, J., Demeester, T., & Vansteelandt, S. (2024). Debiasing synthetic data generated by deep generative models. arXiv:2411.04216   
Doersch, C. (2016). Tutorial on variational autoencoders. arXiv:1606.05908   
Giusti, C., Guarnera, L., Casu, M., & Battiato, S. (2025). Fraud is not just rarity: A causal prototype attention approach to realistic synthetic oversampling. arXiv:2507.14706   
Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). Deep learning (Vol. 1) (vol. 2). MIT press Cambridge.   
Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., & Bengio, Y. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems, 27   
Han, S., Jung, H., Yoo, P. D., Provetti, A., & Cali, A. (2024). Note: non-parametric oversampling technique for explainable credit scoring. Scientific Reports, 14(1), 26070.   
Hou, W. H., Wang, X. K., Zhang, H. Y., Wang, J. Q., & Li, L. (2020). A novel dynamic ensemble selection classifier for an imbalanced data set: An application for credit risk assessment. Knowledge-Based Systems, 208, 106462.   
Khatir Hussin Adam, A. A., & Bee, M. (2022). Machine learning models and data-balancing techniques for credit scoring: What is the best combination? Risks, 10(9), 169.   
Kingma, D.P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv:1312.6114https://api.seman ticscholar.org/CorpusID:216078090   
Kruppa, J., Schwarz, A., Arminger, G., & Ziegler, A. (2013). Consumer credit risk: Individual probability estimates using machine learning. Expert Systems with Applications, 40(13), 5125–5131.   
Kun, Z., Weibing, F., & Jianlin, W. (2020). Default identification of p2p lending based on stacking ensemble learning. In: 2020 2nd international conference on economic management and model engineering (icemme) (pp. 992–1006).   
Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. European Journal of Operational Research, 247(1), 124–136.   
Longadge, R., & Dongre, S. (2013). Class imbalance problem in data mining review. arXiv:1305.1707   
Mancisidor, R. A., Kampffmeyer, M., Aas, K., & Jenssen, R. (2021). Learning latent representations of bank customers with the variational autoencoder. Expert Systems with Applications, 164, 114020.   
Monshizadeh, M., Khatri, V., Gamdou, M., Kantola, R., & Yan, Z. (2021). Improving data generalization with variational autoencoders for network traffic anomaly detection. IEEE Access, 9, 56893–56907.   
Mou, Y., Pu, Z., Feng, D., Luo, Y., Lai, Y., Huang, J., & Xiao, F. (2024). Cost-aware credit-scoring framework based on resampling and feature selection. Computational Economics, 1–26.   
Muslim, M. A., Nikmah, T. L., Pertiwi, D. A. A., Dasril, Y., et al. (2023). New model combination metalearner to improve accuracy prediction p2p lending with stacking ensemble learning. Intelligent Systems with Applications, 18, 200204.   
Pandey, P., & Bandhu, K. C. (2022). A credit risk assessment on borrowers classification using optimized decision tree and knn with bayesian optimization. International Journal of Information Technology, 14(7), 3679–3689.   
Pandey, T.N., Jagadev, A.K., Mohapatra, S.K., & Dehuri, S. (2017). Credit risk analysis using machine learning classifiers. In: 2017 international conference on energy, communication, data analytics and soft computing (icecds) (pp. 1850–1854).   
Rao, C., Liu, Y., & Goh, M. (2023). Credit risk assessment mechanism of personal auto loan based on psoxgboost model. Complex & Intelligent Systems, 9(2), 1391–1414.   
Salmi, M., Atif, D., Oliva, D., Abraham, A., & Ventura, S. (2024). Handling imbalanced medical datasets: review of a decade of research. Artificial Intelligence Review, 57(10), 273.   
Sertkaya, M.E., Ergen, B., & Togacar, M. (2019). Diagnosis of eye retinal diseases based on convolutional neural networks using optical coherence images. In: 2019 23rd international conference electronics (pp. 1–5).   
Tingfei, H., Guangquan, C., & Kuihua, H. (2020). Using variational auto encoding in credit card fraud detection. IEEE Access, 8, 149841–149853.   
Wang, K., Li, M., Cheng, J., Zhou, X., & Li, G. (2022). Research on personal credit risk evaluation based on xgboost. Procedia Computer Science, 199, 1128–1135.   
Wang, L., Zheng, J., Yao, J., & Chen, Y. (2024). A multi-stage integrated model based on deep neural network for credit risk assessment with unbalanced data. Kybernetes   
Xiao, J., Li, S., Tian, Y., Huang, J., Jiang, X., & Wang, S. (2025). Example dependent cost sensitive learning based selective deep ensemble model for customer credit scoring. Scientific Reports, 15(1), 6000.   
Xiao, J., Zhong, Y., Jia, Y., Wang, Y., Li, R., Jiang, X., & Wang, S. (2024). A novel deep ensemble model for imbalanced credit scoring in internet finance. International Journal of Forecasting, 40(1), 348–372.   
Yin, W., Kirkulak-Uludag, B., Zhu, D., & Zhou, Z. (2023). Stacking ensemble method for personal credit risk assessment in peer-to-peer lending. Applied Soft Computing, 142, 110302.

Zhang, X., Yu, L., Yin, H., & Lai, K. K. (2022). Integrating data augmentation and hybrid feature selection for small sample credit risk assessment with high dimensionality. Computers & Operations Research, 146, 105937.

Publisher's Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.

# Authors and Affiliations

# Dalia ATIF1

Dalia ATIF atif.dalia@cu-tipaza.dz Economics, University Center of Tipaza, Tipaza 42000, Algeria