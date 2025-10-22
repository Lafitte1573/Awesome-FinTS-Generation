# Desensitized Financial Data Generation Based on Generative Adversarial Network and Differential Privacy

Fan Zhang, Luyao Wang, and Xinhong Zhang\*

Abstract: Artificial intelligence has been widely used in the financial field, such as credit risk assessment, fraud detection, and stock prediction. Training deep learning models requires a significant amount of data, but financial data often contains sensitive information, some of which cannot be disclosed. Acquiring large amounts of financial data for training deep learning models is a pressing issue that needs to be addressed. This paper proposes a Noise Visibility Function-Differential Privacy Generative Adversarial Network (NVF-DPGAN) model, which generates privacy preserving data similar to the original data, and can be applied to data augmentation for deep learning. This study conducts experiments using financial data from China Stock Market & Accounting Research (CSMAR) database. It compares the generated data with real data from various perspectives, including mean, probability density distribution, and correlation. The experimental results show that the two datasets exhibit similar characteristics. A time series forecasting model is trained on the generated data and the real data separately, and their prediction results are closely aligned. NVF-DPGAN model is feasible and practical in terms of financial data enhancement and privacy protection. This method can also be generalized to other fields, such as the privacy protection of medical data.

Key words:  data desensitization; Generative Adversarial Network (GAN); differential privacy; noise visibility function

# 1 Introduction

With the development of computer science, the financial field has widely utilized artificial intelligence technology. Research in finance has been greatly expanded by deep learning. Deep learning has been applied to credit risk assessment[1], fraud detection[2], and stock prediction[3]. The training of deep learning requires a large amount of data, and financial data usually contains a large amount of personal sensitive information, such as identity card numbers, bank account numbers, etc., which will cause a great threat to personal privacy and financial security if leaked or misused. To prevent the leakage of personal privacy information, the accessibility of most of the financial data is not authorized, which leads to difficulties in data acquisition and insufficient amount of data available for data analysis tasks. Therefore, how to meet scientific research needs, such as data analysis and protecting private information from being leaked, has been a challenge that cannot be ignored.

In the financial domain, data enhancement and data desensitization are two key requirements. Financial data usually requires extended datasets to improve model training, especially when facing data scarcity or imbalance problems, data augmentation can extend the dataset and improve model training. Customer privacy needs to be protected in financial data sharing and analysis to prevent leakage of sensitive information. Data desensitization aims to protect customers’ sensitive information and ensure the privacy and security of data during sharing and analysis. This paper proposes a Noise Visibility Function-Differential Privacy Generative Adversarial Network (NVFDPGAN) model, which is the ideal tool that can satisfy these two needs. By generating high-quality synthetic data, it not only enhances the dataset, but also ensures privacy protection of the generated data and avoids leaking sensitive information of the original data.

Regarding data privacy protection, one solution involves employing cryptographic encryption algorithms. However, this approach can restrict the circulation of data, consequently reducing its utility and value. To address this problem, a series of anonymization-based privacy protection models have been proposed, such as $k$ -anonymity[4], $l .$ -diversity[5], etc., which address the limitations of data circulation in encryption and decryption algorithms. However, they are unable to quantify the privacy protection strength thereby preventing quantitative analysis of the model, and require updating the model design according to evolving attacks. In fact, anonymization-based privacy protection models have been proven to recover sensitive data information using background knowledge attacks, combination attacks, and other attack models, which in turn lead to the leakage of users’ private information.

Another solution is to add perturbation to the model, i.e., adding random noise during model training. This method will make the output results have a certain degree of deviation from the real results, thus preventing attackers from reasoning maliciously. Dwork[6] proposed differential privacy preservation technique as a representative method of perturbation technique. With rigorous theoretical proofs and quantifiable privacy protection strengths, as well as the ability to meet the privacy protection requirements under arbitrary background knowledge, this technique has been widely used in various fields, such as healthcare and finance[7, 8]. Sun et al.[9] proposed a “counter-empirical attacking” mechanism that can generate traces of “attacking” behavior. By training the adversarial learning problem, an appropriate scoring function can be learned to be robust to the attacking activity traces that are trying to violate the empirical criteria. Zhang et al.[10] proposed Graph Masked AutoEncoders (GMAEs). As a traditional selfsupervised graph representation model, GMAE achieves state-of-the-art performance on both the graph classification task and the node classification task under common downstream evaluation protocols. Sun et al.[11] presented a novel multi-task prompting method for graph models. The prompting idea from Natural Language Processing (NLP) can be seamlessly introduced into the graphics area. Meta-learning is introduced to effectively learn a better initialization for the multi-task prompt of graphs, making the prompt framework more reliable and generic for different tasks.

In recent years, Generative Adversarial Network (GAN) has been rapidly developing in the field of artificial intelligence. GAN is a generative model that can generate high-quality fake samples through continuous training and resampling. Based on the powerful learning ability of GAN for complex features of raw data, some scholars have begun to consider combining it with differential privacy. If GAN is trained under differential privacy, it can theoretically generate unlimited amount of data that are highly usable and satisfy the differential privacy characteristics, which can alleviate the problem of lack of high-quality data in some scenarios, while realizing privacy protection. Xie et al.[12] proposed Differentially Private Generative Adversarial Network (DPGAN), which ensures that the generated data satisfy differential privacy by adding noise to the gradient of the discriminator. Acs et al.[13] utilized the differential privacy kernel $k$ -means algorithm to divide the training data into $k$ clusters, then trained a generative neural network on each cluster individually, and finally mixed them together to simulate the distribution of the training data. Guo et al.[14] improved the quality of generated data by incorporating class labels as auxiliary information to the DPGAN model. The work FinDiff proposed by Sattarov et al.[15] uses diffusion modeling to generate financial tabular data, demonstrating its potential for handling highdimensional data and capturing complex data distributions. However, the diffusion model’s training and generation process are relatively complex and inefficient to generate data, which may limit it when dealing with large-scale data. Khadka et al.[16] generated synthetic data using combinatorial testing and a Variational AutoEncoder (VAE), showing advantages in terms of data diversity and generation quality. However, VAE-generated data may suffer from inferior quality to GAN-generated data, especially in terms of detail and fidelity. The diffusion generative models are more suitable for generating image data. GAN model is still a suitable choice for financial data generation. Compared to the complex generation process of diffusion model, the generation process of GAN is more efficient and applicable to the needs of large-scale data generation. Existing methods of generating financial data rarely involve data desensitization.

Based on the aforementioned discussions, this paper proposes a privacy-preserving GAN model for generating financial data. This model generates highquality data without revealing sensitive features of the original data, all the while ensuring that downstream data analysis tasks remain unaffected. As well as achieving privacy protection, the model can also be used to extend the dataset for deep learning, which provides a method for small-sample financial data research based on deep learning.

The main contributions of this paper are summarized as follows:

(1) NVF-DPGAN model is proposed for generating large amount of desensitized financial data; (2) Differential privacy is introduced in the process of generating financial data; (3) Noise visibility function is introduced to improve the quality of generated data.

# 2 Method

# 2.1 GAN

GAN is a generative model for unsupervised learning proposed by Goodfellow et al.[17] in 2014. The structure of GAN is shown in Fig. 1. Inspired by the idea of binary zero-sum games, GAN consists of mutually adversarial generators $G$ and discriminators $D , G$ and $D$ achieve the generation of new data close to the original data distribution through an adversarial game learning process. The input to $G$ is noise, and the goal is to generate fake samples that are similar to the real samples. The input to $D$ is real data versus generated data, and the goal is to determine whether its input comes from the dataset generated by $G$ or the real dataset. $G$ and $D$ play an adversarial game to achieve their respective goals. In the course of game, $G$ and $D$ continuously improve their generating ability and discrimination ability, and finally reach the Nash equilibrium state. The optimization function of the adversarial training process can be expressed as follows:

$$
\begin{array} { r } { \underset { G } { \operatorname* { m i n } } \underset { D } { \operatorname* { m a x } } V \left( D , G \right) = E _ { x \sim \mathcal { P } _ { \mathrm { d a t a } } } [ \log D \left( \boldsymbol { x } \right) ] + } \\ { E _ { z \sim \mathcal { P } _ { z } } [ 1 - \log D \left( G ( \boldsymbol { z } ) \right) ] } \end{array}
$$

where $V$ represents loss function, $E$ represents the expectation, $\mathcal { P } _ { \mathrm { d a t a } }$ and $\mathcal { P } _ { z }$ represent the real data distribution and the generated data distribution, respectively, $x$ is the real data sample, $z$ is the noise sample, and $G \left( z \right)$ is the data generated by the generator.

The specific steps of adversarial training as follows: Select a batch of real samples of size $m$ and a batch of random variables, the random variables are passed through $G$ to get the generation samples. Keep the $G$ weights unchanged, and use the stochastic gradient ascent method to compute and update the weights of the $D$ network by

$$
\Delta \theta _ { D } = \nabla _ { \theta _ { D } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left[ \log D \left( x ^ { i } \right) + \log \left( 1 - D \left( G \left( z ^ { i } \right) \right) \right) \right]
$$

Then train $G$ , select a batch of random variables to get the generation samples through $G$ , keep the $D$ weights unchanged, and use the stochastic gradient descent method to calculate and update the $D$ network’s weights by

![](images/74d81ddbbeedbc20c1f6b7a3c8097fe0ba001f11924ec893bd60da52627b9ee6.jpg)  
Fig. 1    Structure of GAN.

$$
\Delta \theta _ { G } = - \nabla _ { \theta _ { G } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \log \left[ 1 - D \left( G \left( z ^ { i } \right) \right) \right]
$$

to calculate and update the weights of $G$ network. The Nash equilibrium state of adversarial training is $\mathcal { P } _ { g } = \mathcal { P } _ { \mathrm { d a t a } }$ . $D$ randomly guesses that the samples belong to the real samples or generated samples with at most $50 \%$ probability.

In Eqs. (2) and (3), $\Delta$ ∇represents difference, represents gradient function, $\theta _ { G }$ and $\theta _ { D }$ represent the parameters of generator and discriminator, respectively, $x ^ { i }$ denotes the $i$ -th sample of real data, and $z ^ { i }$ denotes the $i \cdot$ -th sample of noise.

Theoretically, GAN can generate any data as long as there are enough training samples. In practice, the original GAN model is prone to problems, such as vanishing gradient or pattern collapse, because it is too free and leads to uncontrollable training process and results. Vanishing gradient refers to the fact that GAN needs to train the generator and the discriminator separately, the optimization between the two cannot be well synchronized. If the discriminator is always able to correctly distinguish real samples from the generated samples, the discriminator can recognize the samples generated by the generator no matter how realistic they are, at this time, the loss drops to zero, resulting in no learning by the generator, and the phenomenon of gradient vanishing occurs. The vanishing gradient problem leads to a decrease in the quality of the generated samples. Pattern collapse when the generator learns a parameter setting to generate a sample that the discriminator considers realistic, the sample is easy to fool the discriminator, then the generator may repeatedly generate the same pseudo-sample, and finally always generate the same sample points. It will result in the generator only generating part of the real data, reducing the diversity of the generated samples.

Researchers have studied the above problems and have proposed solutions. WGAN proposed by Arjovsky et al.[18] solves the problem of unstable GAN training, almost avoids the phenomenon of crashing patterns, and ensures the diversity of generated samples. In addition, various variants based on the original GAN model have been gradually proposed. Conditional GAN (CGAN)[19] adds constraints on the basis of the original GAN, which controls the problem of too much freedom of the GAN and enables the network to generate samples in the given direction. Super-Resolution Generative Adversarial Network (SRGAN), proposed by Ledig et al.[20], is a superresolution reconstruction algorithm for images, traditional reconstruction algorithms tend to lack the sensory satisfaction of the details at high resolution, SRGAN generates images that can achieve better expected fidelity. Yu et al.[21] combined Convolutional Neural Network (CNN) with GAN and proposed Deep Convolutional Generative Adversarial Network (DCGAN), which uses convolutional neural networks in the generator and discriminator feature extraction layers instead of the multilayer perceptron in the original GAN. CNN is typically used in supervised learning rather than unsupervised domains, while DCGAN creatively combines convolutional neural networks with generative adversarial networks in the unsupervised learning domain[22−24].

# 2.2 Differential privacy

Differential privacy is a strict privacy definition pioneered by Dwork et al.[25] in 2006. In the era of big data, attackers can easily obtain large amounts of background knowledge through various means, leading to serious privacy breaches. Differential privacy is usually used to protect a particular data record on a dataset. Perturbing the search results based on the dataset by adding random noise, which in turn makes the search results distorted. It is guaranteed that inserting or deleting a record on the dataset will not cause much change to the search result, making it impossible for an attacker to infer the exact sensitive privacy information of an individual in the dataset based on the search result. The risk of privacy disclosure of individual data records to the dataset is kept within a given range.

Differential privacy is a very attractive quantitative privacy approach designed to provide guarantees for the computational process of sensitive data. Privacy is ensured by using the following promise to guarantee randomness: We consider an algorithm to be differentially private if the participation of any record in the database (corresponding to an individual) does not significantly modify the likelihood of the output[26]. Dwork et al.[25] provided a complete discussion of the concepts related to differential privacy and also gave a rigorous quantitative assessment of the risk of privacy leakage, giving it a rigorous mathematical theory support.

Definition 1 Neighboring datasets. If there exist two datasets $D$ and $D ^ { \prime }$ , which differ in the number of data records by at most one, and the $L _ { 1 }$ - paradigm distance between the two datasets satisfies

$$
\left\| D - D ^ { \prime } \right\| \leqslant 1
$$

then the two datasets are called neighboring datasets.   
$\left\| \cdot \right\|$ represents $L _ { 1 }$ -norm in Formula (4).

Definition 2 $\varepsilon$ -Differential privacy.

Given a randomized algorithm $f : D \to R$ , $S \subseteq R$ is a subset of the set of all possible output values of algorithm $f$ . For any pair of adjacent datasets $D$ and $D ^ { \prime }$ , if algorithm satisfies

$$
\operatorname* { P r } \left[ f \left( D \right) \in S \right] \leqslant \operatorname* { P r } \left[ f \left( D ^ { \prime } \right) \in S \right] \times \mathrm { e } ^ { \varepsilon }
$$

then it is said that the algorithm satisfies $\varepsilon$ -differential privacy, where $\mathrm { P r } \left[ \cdot \right]$ represents probability, $e$ represents natural constant, and $\varepsilon$ is used to control the degree of privacy protection of the algorithm, which is called the privacy budget. The differential privacy mechanism controls the privacy loss of the algorithm within a limited range, the smaller $\varepsilon$ , the better the privacy protection of the algorithm. Commonly used perturbation mechanisms are Laplace mechanism[25], exponential mechanism[27], and Gaussian mechanism[28, 29], in which the noise size depends on the sensitivity of the algorithm.

Definition 3 Global sensitivity.

For any two neighboring datasets $D$ and $D ^ { \prime }$ , given a query function $f : D \to R$ , the global sensitivity $S _ { f }$ is defined as

$$
M \left( d \right) = f \left( D \right) + N \left( 0 , \left( S _ { f } \sigma \right) ^ { 2 } I \right)
$$

where $N \left( 0 , \left( S _ { f } \sigma \right) ^ { 2 } I \right)$ is Gaussian noise with mean 0 and variance $\left( S _ { f } \sigma \right) ^ { 2 }$ , and $I$ is the unit matrix. If $\delta ^ { 2 } \geqslant 4 / 5 \times \exp { ( - \sigma \varepsilon ^ { 2 } / 2 ) }$ and $ { \varepsilon } \in ( 0 , 1 )$ , then algorithm $M$ satisfies $( \varepsilon , \delta )$ -differential privacy.

In this paper, the Gaussian mechanism is used to implement the addition of differential privacy noise. Noise perturbations are added during the training of the discriminator, rather than adding noise directly to the final parameters, which would reduce the utility of the data.

# Definition 5 Privacy loss.

For any two neighboring datasets $D$ and $D ^ { \prime }$ , there exists a random function $f$ . The privacy loss of $f$ is defined as follows:

$$
C \left( f , D , D ^ { \prime } \right) \overset { \Delta } { = } \log \frac { \operatorname* { P r } \left[ f \left( D \right) \right] } { \operatorname* { P r } \left[ f \left( D ^ { \prime } \right) \right] }
$$

where the function $f$ determines the probability $\mathrm { P r } \left[ \cdot \right]$ .

# 2.3 Noise visibility function

The Noise Visibility Function (NVF) was proposed by Voloshynovskiy[30−32] in the context of image denoising as well as watermarking estimation using Maximum A Posteriori (MAP) estimation. It is defined as follows:

$$
\mathrm { N V F } \left( i , j \right) = \frac { w \left( i , j \right) } { w \left( i , j \right) + \sigma _ { x } ^ { 2 } \left( i , j \right) }
$$

$$
S _ { f } = \operatorname* { m a x } \left\| f \left( D \right) - f \left( D ^ { \prime } \right) \right\| _ { 2 }
$$

where $\lVert \cdot \rVert _ { 2 }$ represents $L _ { 2 }$ -norm.

Global sensitivities provide an upper bound for the design of privacy mechanisms. The size of each query function’s sensitivity is determined by its own properties. For query functions with small sensitivities, a small amount of noise can be added to mask the privacy impact of deletion of a data record. For query functions with large sensitivity, a large amount of noise needs to be added to realize privacy protection. However, the addition of a significant amount of noise can potentially affect the usability of the data, leading to a balance issue between data utility and privacy. The size of the added noise plays a decisive role in the differential privacy preservation of the data.

Definition 4 Gaussian mechanism.

MThe function perturbation algorithm enables differential privacy preservation by adding noise to a function $f$ based on a Gaussian distribution,

where $1 \leqslant i , j \leqslant N . N \times N$ represents the image size of a local region in pixel, $\sigma _ { x } ^ { 2 } \left( i , j \right)$ is the contrast of a localized region of the image,

$$
\sigma _ { x } ^ { 2 } \left( i , j \right) = \frac { 1 } { N ^ { 2 } } \sum _ { k = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \left[ x \left( i , j \right) - \bar { x } \left( i , j \right) \right] ^ { 2 }
$$

$$
\bar { x } ( i , j ) = \frac { 1 } { N ^ { 2 } } \sum _ { k = 1 } ^ { N } \sum _ { l = 1 } ^ { N } x \left( i , j \right)
$$

where $x ( i , j )$ represents the pixel value at $( i , j )$ of the image, and $\bar { x } \left( x , y \right)$ denotes the local mean of the image.

$w \left( i , j \right)$ in Eq. (9) is called the weight function and depends on the shape parameter $\gamma$ . The expression for $w \left( i , j \right)$ is given below:

$$
w \left( i , j \right) = \gamma \left[ \eta \left( \gamma \right) \right] ^ { \gamma } \frac { 1 } { \left. r \left( i , j \right) \right. ^ { 2 - \gamma } }
$$

where

$$
r \left( i , j \right) = \frac { x \left( i , j \right) - \bar { x } \left( i , j \right) } { \sigma _ { x } }
$$

$$
\eta \left( T \right) = \sqrt { \frac { T \left( \frac { 3 } { \gamma } \right) } { T \left( \frac { 1 } { \gamma } \right) } }
$$

$$
\varGamma ( t ) = \int _ { 0 } ^ { \infty } \mathrm { e } ^ { - u } u ^ { t - 1 } \mathrm { d } u
$$

where $\sigma _ { x } ( i , j )$ denotes the local standard deviation of the image, $T \left( \cdot \right)$ is Gamma function, $t$ is the parameter of Gamma function, and $u$ is the integral variable.

From Eq. (9), it can be seen that NVF is inversely proportional to the local energy of the image and is a Texture Masking Function (TMF). NVF is a function that reflects the local texture masking of the image and indicates the degree of sensitivity of each pixel in the image to the noise. The NVF takes a value in the range of 0 to 1. When the value of the NVF is small, it indicates that the local region of the image at the pixel has a complex texture and can allow larger noise, i.e., the noise is less visible. When the NVF value is large, the image is sensitive to noise at that pixel.

In practical calculations, the NVF value of each pixel point is calculated at that pixel point in a localized image centered at that pixel point. Once the NVF value is calculated, the sensitivity of each pixel point to the watermark (noise) can be estimated based on this noise visibility function. When the NVF value is large, more watermark energy can be embedded at that pixel. When the NVF value is small, less watermark energy is embedded. The permissible level of distortion at each pixel point can be calculated based on NVF as per the following equation:

$$
\begin{array} { l } { { \Delta \left( i , j \right) = \left( 1 - { \bf N V F } \left( i , j \right) \right) \cdot S _ { 0 } + { \bf N V F } \left( i , j \right) \cdot S _ { 1 } = } } \\ { { \displaystyle \left( 1 - \frac { w \left( i , j \right) } { w \left( i , j \right) + \sigma _ { x } ^ { 2 } \left( i , j \right) } \right) \cdot S _ { 0 } + \frac { w \left( i , j \right) } { w \left( i , j \right) + \sigma _ { x } ^ { 2 } \left( i , j \right) } \cdot S _ { 1 } } } \end{array}
$$

where $S _ { 0 }$ and $S _ { 1 }$ represent the maximum distortion allowed in the textured and flat regions of the image, respectively. In the actual calculation, we take $S _ { 0 } = 3 0$ and $S _ { 1 } = 3$ . It can be seen from Eq. (16) that the NVF value tends to 1 in the flat region, i.e., the first term of equation tends to 0, so that the permissible distortion at the pixel mainly depends on the smaller value of $S _ { 1 }$ among $S _ { 0 }$ and $S _ { 1 }$ , i.e., the permissible modification of this pixel is of smaller magnitude. Whereas in texture complex region NVF value tends to 0, the distortion allowed at this pixel mainly depends on $S _ { 1 }$ , i.e., the pixel is allowed to be modified with a larger magnitude. This means that it is possible to embed more watermarking energy in texture complex regions and less in flat regions.

In summary, this paper defines a new gradient cropping threshold by combining NVF with global sensitivity due to the fact that NVF can reflect the sensitivity of a certain place to noise addition.

# 2.4 NVF-DPGAN

In this paper, we propose an NVF-DPGAN model that combines differential privacy preserving mechanism, noise visibility function and generative adversarial network. The NVF-DPGAN model generates privacypreserving financial data that can satisfy both data enhancement and data desensitization requirements.

Traditional data privacy protection methods tend to over-clean the original data, which greatly reduces the usability of the data. Unable to resist reconstruction attacks[33], reconstruction attacks can utilize statistical properties and hidden information in the data to restore the original data, thereby exposing the privacy of the training samples. Data generated using GAN do not have an explicit one-to-one relationship with the original data, thus even if the data is partially leaked, it does not directly correspond to the real data. This results in stronger privacy protection and a substantial increase in privacy compared to traditional methods. Membership inference attack[34] refers to an attacker’s attempt to infer whether a given sample is used for model training, i.e., one of the “members” of the training data. In some scenarios, membership inference attacks can have serious consequences. For example, for a diagnostic model constructed from patient data for an infectious disease, if a person’s medical data are inferred to be the training data for the model, it means that the person may have the infectious disease.

Membership inference attacks are capable of reconstructing or recovering training samples from machine learning models, and are popular methods for measuring privacy breaches in deep learning models. Given a model (called a target model), membership inference attacks aim to infer whether data samples are in the model’s training dataset. The generators and discriminators of GAN models, which are deep learning models, are vulnerable to membership inference attacks. In addition, the high-complexity of the GAN model increases the risk of privacy leakage as it remembers certain training samples, leading to the concentration of the learned distribution on these samples. Differential privacy can significantly alleviate this impact. By its definition, if a model’s training process adheres to the principle of differential privacy, then the probability of the same model being trained with or without a particular sample in the training dataset will remain closely consistent.

During the training of the model, only the discriminator is exposed to real data with the generator adjusting the parameters depending on the feedback by the discriminator. To achieve privacy protection, the discriminator needs to be trained under differential privacy protection. Subsequently, the output of the differential privacy algorithm is processed in a dataindependent manner to ensure that the output still maintains the differential privacy properties. As a result, the entire network adheres to the guarantee of differential privacy. The overall architecture of the proposed generative adversarial network model for financial data privacy protection is shown in Fig. 2, and the network parameter settings of the generator and discriminator are shown in Table 1. Instead of adding noise directly to the final parameters, the model adds Gaussian noise to the training process of the discriminator to ensure data usability while achieving differential privacy protection. On the other hand, only the training of the discriminator will directly touch the real data, also, compared to the generator, the discriminator has a simple feature structure and fewer common parameters, which makes it easy to estimate the privacy loss.

The generator network is shown in Fig. 3. The generator uses 100-dimensional Gaussian noise as input and generates data of size $( 2 2 \times 8 )$ as output by continuous de-convolution. The data distribution is adjusted to approach a standard normal distribution with mean 0 and variance 1 using batch normalization.

This normalization operation helps to alleviate the previously mentioned problems of vanishing gradients and gradient explosion, allowing gradients to propagate better through the network and reducing gradient correlation during training. Simultaneously, batch normalization stabilizes parameters within the network, reducing inter-parameter correlations. This enables the use of higher learning rates for training, accelerating convergence speed, and enhancing model stability. ReLU is chosen as the activation function, because it has a constant gradient when the input is greater than 0. The gradient is not affected by the vanishing gradient or gradient explosion in the back propagation process, which allows for better propagation and a more stable training process. Moreover, the computation of ReLU function is very simple and efficient, which can accelerate the training and inference process of neural network compared to other activation functions. In addition, based on the generator loss of the original GAN, this paper adds a regularized loss constraint with the following generator loss function:

$$
L _ { G } = \operatorname* { m i n } \left\{ E _ { z \sim \mathcal { P } _ { z } } \left[ \log \left( 1 - D \left( G \left( z \right) \right) \right] + \operatorname* { l n f } _ { - } \mathrm { l o s s } \left( \mathbf { \Delta } \right) \right\} \right.
$$

where $L _ { G }$ represents the generator loss function, and ( ) lnf_loss is to measure the difference between real and generated samples by calculating the difference in gradient magnitude between them. It can help the generator learn to generate data with richer details and more realistic. The objective of the generator’s loss function is to minimize $\log { ( 1 - D \left( G \left( z \right) \right) + \log { \underline { { \ } } \log { ( \ ) } } }$ , in which the former part indicates that the generator’s objective is to make the discriminator consider the generated samples to be real samples as much as possible, i.e., $D \left( G \left( z \right) \right)$ is equal to 1 as much as possible; and the latter part indicates to minimize the difference between the generated samples and the real samples as much as possible, so as to make the generated samples more real.

![](images/7e85b58bf636c6c67b465de9f1a3b4776c8330a2f587202b34f81903fa7db921.jpg)  
Fig. 2    Structure of NVF-DPGAN medel.

Table 1    Network parameter of NVF-DPGAN model.   

<table><tr><td>Layer</td><td>Generator</td><td>Discriminator</td></tr><tr><td>Layer 1</td><td>UpConv (3, 1)</td><td>Conv (4, 4)</td></tr><tr><td>Layer 2</td><td>UpConv (2, 2)</td><td>Conv (4, 4)</td></tr><tr><td>Layer 3</td><td>UpConv (2, 2)</td><td>Conv (5,2)</td></tr><tr><td>Layer 4</td><td>UpConv (4, 4)</td><td>1</td></tr><tr><td>Layer 5</td><td>UpConv (1, 3)</td><td>1</td></tr></table>

# 3 Result

# 3.1 Dataset

The data for this study are obtained from China Stock Market & Accounting Research Database (CSMAR Database) at https://cn.gtadata.com/. The experimental data are derived from various securities spanning from 1991 to 2021, including 34 financial variables, such as TotalAssets, TotalCurrentAssets, and CashandCashEquivalents. The total number of original data is 59 161. In this study, the original dataset is first cleaned for use in this experiment, the data cleaning steps are as follows:

Step 1: Rename all columns according to their meaning instead of using the original unintuitive numbering.

Step 2: Remove redundant columns and select suitable columns for the study.

This paper chooses the following financial attributes as research objects:

(1) Among the many attributes, TotalAssets, which is the sum of the value of all assets of a company at a given point in time, is an important indicator for assessing financial condition and health. Investors can assess the trend of its financial health by comparing TotalAssets at different points in time. Steadily increasing TotalAssets may indicate that the company is doing well, while a downward trend may hint financial problems. TotalAssets is also a core component of the balance sheet and is compared to total liabilities; when they are equal, it means there is no debt. In addition, investors can use it to compare the financial position of different companies, which is significant for choosing investment targets or making industry comparisons.

(2) TotalCurrentAssets is the total amount of assets that can be converted to cash or consumed in the short term. TotalCurrentAssets is critical in assessing a company’s short-term solvency and the soundness of its operating activities. Higher TotalCurrentAssets implies higher short-term debt repayment capacity. Also, sufficient TotalCurrentAssets may indicate that the company is financially sound and better able to cope with short-term risks.

(3) CashandCashEquivalents is cash held on hand by the company and highly liquid financial assets that can be quickly converted to cash. It reflects a company’s ability to repay its debts and operate in a short term. A moderate level of CashandCashEquivalents can show a company’s good management ability, especially in the face of uncertain economic environment or industry cycle fluctuations, having enough cash reserves can mitigate the risks faced by the company. CashandCashEquivalents is also used to monitor the company’s cash flow position. As cash flows fluctuate, the level of a company’s money funds may change, which is an important concern for investors.

(4) OriginalCostofFixedAssets is commonly used in company’s asset value assessment, depreciation calculation, asset management, financial analysis, and asset evaluation. To a certain extent, it can reflect the company’s investment scale and strength, and help to understand the company’s investment status and operational efficiency, which is an important indicator

![](images/93c2c9bef659da976c1139b58b9caeb01db576f443d8813bb3846d791733a4d8.jpg)  
Fig. 3    Structure of generator. The numbers in the figure represent data dimensions.

in financial data.

(5) TotalLiabilities is often combined with TotalAssets for balance sheet analysis. It is a key indicator for assessing the financial soundness and solvency of a company, as well as an important measure of its capital structure. A high TotalLiabilities may indicate that the company relies mainly on debt financing and faces higher pressure to repay its debts, while a lower TotalLiabilities may mean that the company relies more on shareholders’ capital. TotalLiabilities is an important indicator when understanding the distribution of a company’s debt and making investment decisions.

(6) TotalCurrentLiabilities is an important indicator of a company’s financial position and repayment ability. It is commonly used in the evaluation of the company’s solvency, liquidity assessment, capital structure analysis, and financial ratio calculation. It is one of the important indicators for analyzing and judging the company’s financial status.

(7) ShareCapital is an important component belonging to the Owners’ Equity section. It reflects the company’s shareholding structure, the company’s share issuance and repurchase. It can also be used to calculate a company’s market capitalization, which is an important indicator of a company’s size and shareholders’ equity.

(8) MainOperatingRevenue is the total of all revenues generated by a company through its principal business activities. MainOperatingRevenue is an important measure of a company’s operating performance and is usually directly related to the company’s main products or services. It can reflect how the company’s business is changing, and a consistently growing main operating revenue may mean that the company’s business is expanding and growing. It also reflects the degree of diversity in the company’s business and the company’s profitability.

Based on the above considerations, this paper selects TotalAssets, TotalCurrentAssets, CashandCashEquivalents, OriginalCostofFixedAssets, TotalLiabilities, TotalCurrentLiabilities, ShareCapital, and MainOperatingRevenue for the study.

Step 3: Obtain data of each stock only for the years from 2000 to 2021, so that each stock code has the same amount of data.

Step 4: Fill in missing values. The data are mapped into a matrix $( 2 2 \times 8 )$ for each stock code, where 22 represents the number of years from 2000 to 2021 and 8 represents the eight financial variables. Examples of cleaned data are listed in Table 2, which shows financial variables of 22 years of a stock code.

# 3.2 NVF-DPGAN model training

The experimental environment is Python 3.8. The hardware devices are Intel (R) Core (TM) i7-10700K processor and NVIDIA GeForce RTX 2080Ti. The generator and discriminator are trained with 16 500 iterations using Adam optimization algorithm with a learning rate of 0.000 01 and batch_size of 32. Upon completion of training, the trained generator generated 22 000 financial data for comparison with real data.

Adam is a high performance gradient descent algorithm. It is combined with differential privacy for training NVF-DPGAN model. The specific implementation is as follows: the model uses the Adam gradient descent algorithm to update the model parameters during the neural network training process, and at the same time, Gaussian noise is added during the update process to achieve the purpose of differential privacy. The whole training process of the model is: Gradient trimming and adding noise updating the discriminator updating the generator.

The purpose of gradient cropping is to control the size of the gradient and avoid the gradient explosion problem. If the gradient trimming threshold is too small, it will lead to over-trimming of the gradient, which may lose some useful gradient information, thus affecting the convergence and performance of the model. However, if the gradient cropping threshold is too large, the gradient may still exceed the preset range and the purpose of controlling the gradient size cannot be achieved. So how to determine the threshold of gradient cropping? From Definition 3, it can be seen that the global sensitivity provides an upper bound for the design of the privacy mechanism, which determines the upper limit of the noise that can be added. Traditional differential privacy GAN models use the global sensitivity as a threshold for gradient cropping, and the magnitude of cropping can be determined based on the model’s sensitivity to parameter changes.

In this paper, the global sensitivity is combined with the noise visibility function so that they jointly determine the cropping threshold. As described in

Table 2    Example of cleaned data.   

<table><tr><td>Year</td><td>TotalAssets</td><td></td><td>TotaCurrent- CEquivdlcsh</td><td>OrinaACostof-</td><td>Liabtalties</td><td>TotaCuriert</td><td>Sapireal</td><td>MaiRoverating-</td></tr><tr><td>2000</td><td>562</td><td>503</td><td>100</td><td>36</td><td>266</td><td>253</td><td>63</td><td>378</td></tr><tr><td>2001</td><td>648</td><td>606</td><td>81</td><td>34</td><td>336</td><td>305</td><td>63</td><td>446</td></tr><tr><td>2002</td><td>822</td><td>774</td><td>119</td><td>39</td><td>479</td><td>306</td><td>63</td><td>457</td></tr><tr><td>2003</td><td>1056</td><td>1021</td><td>97</td><td>33</td><td>580</td><td>478</td><td>140</td><td>638</td></tr><tr><td>2004</td><td>1553</td><td>1517</td><td>313</td><td>27</td><td>923</td><td>628</td><td>227</td><td>767</td></tr><tr><td>2005</td><td>2199</td><td>1988</td><td>325</td><td>26</td><td>1341</td><td>1086</td><td>372</td><td>1056</td></tr><tr><td>2006</td><td>4851</td><td>4468</td><td>1074</td><td>56</td><td>3150</td><td>2188</td><td>437</td><td>1785</td></tr><tr><td>2007</td><td>10009</td><td>9543</td><td>1705</td><td>68</td><td>6617</td><td>4877</td><td>687</td><td>3539</td></tr><tr><td>2008</td><td>11 924</td><td>11 346</td><td>1998</td><td>136</td><td>8042</td><td>6455</td><td>1100</td><td>4074</td></tr><tr><td>2009</td><td>13 761</td><td>13032</td><td>2300</td><td>145</td><td>9220</td><td>6806</td><td>1100</td><td>4865</td></tr><tr><td>2010</td><td>21564</td><td>20552</td><td>3782</td><td>132</td><td>16105</td><td>12 965</td><td>1100</td><td>5046</td></tr><tr><td>2011</td><td>29 621</td><td>28265</td><td>3424</td><td>171</td><td>22838</td><td>20072</td><td>1100</td><td>7122</td></tr><tr><td>2012</td><td>37880</td><td>36277</td><td>5229</td><td>177</td><td>29666</td><td>25983</td><td>1100</td><td>10244</td></tr><tr><td>2013</td><td>47921</td><td>44205</td><td>4437</td><td>228</td><td>37377</td><td>32892</td><td>1101</td><td>13 426</td></tr><tr><td>2014</td><td>50841</td><td>46 481</td><td>6272</td><td>268</td><td>39252</td><td>34565</td><td>1104</td><td>14552</td></tr><tr><td>2015</td><td>61130</td><td>54702</td><td>5318</td><td>541</td><td>47499</td><td>42006</td><td>1105</td><td>19318</td></tr><tr><td>2016</td><td>83067</td><td>72130</td><td>8703</td><td>752</td><td>66900</td><td>58000</td><td>1104</td><td>23840</td></tr><tr><td>2017</td><td>116 535</td><td>101755</td><td>17412</td><td>820</td><td>97867</td><td>84736</td><td>1104</td><td>24014</td></tr><tr><td>2018</td><td>152858</td><td>129507</td><td>18842</td><td>1323</td><td>129296</td><td>112 191</td><td>1104</td><td>29 442</td></tr><tr><td>2019</td><td>172993</td><td>143 899</td><td>16620</td><td>1468</td><td>145935</td><td>127261</td><td>1130</td><td>36535</td></tr><tr><td>2020</td><td>186918</td><td>154739</td><td>19523</td><td>1523</td><td>151933</td><td>131749</td><td>1162</td><td>41588</td></tr><tr><td>2021</td><td>193 864</td><td>160027</td><td>14935</td><td>1578</td><td>154587</td><td>131145</td><td>1163</td><td>44976</td></tr></table>

Section 2.3, $\mathrm { N V F } \left( i , j \right)$ reflects the sensitivity of the position to noise, that is, the amount of noise that can NVFbe added; The larger the $( i , j )$ is, the less noise can be added; while the smaller the $\mathrm { N V F } \left( i , j \right)$ is, the more noise can be added, and the value of the $\mathrm { N V F } \left( i , j \right)$ is located in the range of (0, 1). This paper integrates the noise visibility function with global sensitivity, introducing a new clipping threshold, denoted as $( 1 - \mathrm { N V F } ( i , j ) ) \cdot M$ , that allows both factors to jointly control the cropping threshold. This approach enables a more accurate gradient clipping. While determining the global sensitivity $M$ , $( 1 - \mathrm { N V F } ( i , j ) )$ is used as a coefficient to control, so as to realize giving different cropping thresholds at locations with different acceptance of noise. Parameters that are more sensitive to noise will have smaller cropping thresholds to limit their gradient changes. The less sensitive parameters will have larger cropping thresholds to allow for greater variation in their gradients. This approach helps balance the gradient cropping across various parameters of the model. It retains valuable gradient information while ensuring that the extent of cropping adapts to the sensitivity level of different parameters.

This paper introduces random noise to the gradients, ensuring that attackers cannot discern whether specific data points are present in the training dataset. This achieves the goal of privacy protection. There are several options for noise addition, and the commonly used noise perturbation mechanisms are the Laplace mechanism, the exponential mechanism, and the Gaussian mechanism[35]. The Laplace and Gaussian mechanisms are mainly suitable for perturbing the query results of single-valued data records, while the exponential mechanism is mainly suitable for nonnumerical data query perturbation. In this paper, differential privacy preservation is realized using Gaussian mechanism. Both generators and discriminators are essentially differentiable functions, which makes continuous differentiable Gaussian noise easier to mathematically derive and computationally handle during processing. In addition, compared to Laplace noise, Gaussian noise typically has less impact on query results for the same level of privacy protection. Moreover, Gaussian noise exhibits heavier tails, resulting in fewer outliers and extreme values, thereby leading to smoother and more stable query results.

During the training of the NVF-DPGAN model, only the training of the discriminator requires access to real data. To achieve differential privacy, discriminator training needs to be privatized. During discriminator updates, first, samples are drawn from the original data and gradients are computed. After determining the discriminator’s parameters, during the stochastic gradient descent process, Gaussian noise is added and gradients are cropped. The generator will be trained after the parameters of the discriminator are updated. Samples are drawn from the noise distribution $\mathcal { P } _ { z }$ to update the generator parameters. Simultaneously, the loss of privacy during the training process is statistically computed. The algorithm iterates in a form of adversarial learning until the accumulated privacy loss exceeds the total privacy budget or the iteration process concludes, at which point the algorithm terminates.

As described in Section 2.1, the generator training direction is to increase $D \left( G \left( z \right) \right)$ and decrease $D \left( x \right)$ . The generator objective function is to minimize $\log { ( 1 - D \left( G \left( z \right) \right) ) } + \operatorname* { l n f } _ { - } \log { ( { } ) }$ . The discriminator is trained in the direction of maximizing $D \left( G \left( z \right) \right)$ and minimizing $D \left( x \right)$ . The discriminator objective function is to maximize $\log { ( D ( x ) ) } + \log { ( 1 - D ( G ( z ) ) ) }$ . The values of $D \left( G \left( z \right) \right)$ and $D \left( x \right)$ eventually fluctuate around 0.5 as GAN training proceeds. The discriminator is not sure whether the input data are generated or not. At this point the generator and the discriminator reach a Nash Equilibrium, the model reaches the optimal state. The variation process of generator and discriminator loss during NVF-DPGAN training is shown in Fig. 4.

# 3.3 Experimental results

The privacy-preserving data generated based on the NVF-DPGAN model and the real data have the same characteristics in terms of statistical significance. The average value comparison between the generated data and the original data is shown in Table 3. Generated data and real data are indeed not entirely identical, resulting in subtle differences in their statistical properties. Such distinctions can even manifest within the same dataset. For instance, consider splitting the “TotalAssets” column of real data into two parts: the mean of the first portion being 2499.867 81 and the mean of the second portion being 1222.207 28. Differences can still be observed between the two. Figure 5 shows the data presentation of real data and NVF-DPGAN generated data. The first row is the real data and the second row is the synthetic data, including MonetaryFunds, OriginalFixedAssets, Equity, and MainBusinessRevenue. Figure 6 shows the normalized histograms of real data and NVF-DPGAN generated data. Normalized histogram actually depicts the probability densities distribution of data. The probability densities of real data and NVF-DPGAN generated data are very similar. This to some extent implies that these two datasets have similar data features, indirectly affirming the effectiveness of the generated data.

![](images/203537c9bf2e17edac88cf5f510311dc4761732334009edc4684e1e84c16dfea.jpg)  
Fig. 4    Generator and discriminator loss during NVFDPGAN training.

Table 3    Average value of the generated data versus the real data for the comparison of NVF-DPGAN to DPGAN.   

<table><tr><td>Variable</td><td>Real data</td><td>DPGAN</td><td>NVF-DPGAN</td></tr><tr><td>TotalAssets</td><td>1861.037 54</td><td>1846.208 13</td><td>1844.209 83</td></tr><tr><td>TotalCurrentAssets</td><td>649.066 95</td><td>647.875 63</td><td>646.766 73</td></tr><tr><td>CashandCashEquivalents</td><td>184.805 14</td><td>185.128 36</td><td>185.188 35</td></tr><tr><td>OriginalCostofFixedAssets</td><td>291.391 39</td><td>295.248 98</td><td>294.247 77</td></tr><tr><td>TotalLiabilities</td><td>1371.437 46</td><td>1352.845 36</td><td>1355.553 73</td></tr><tr><td>TotalCurrentLiabilities</td><td>561.009 33</td><td>561.825 89</td><td>561.625 78</td></tr><tr><td> ShareCapital</td><td>99.258 25</td><td>99.902 96</td><td>99.692 90</td></tr><tr><td> MainOperatingRevenue</td><td>718.118 43</td><td>722.034 16</td><td>722.034 08</td></tr></table>

![](images/3322fb0f14bff6efb8cf76a7cfd985d7492ced8878eaad2404552a258a6aaf08.jpg)  
Fig. 5    Data presentation of real data and NVF-DPGAN generated data. (a−d) are the real data of MonetaryFunds, OriginalFixedAssets, Equity, and MainBusinessRevenue, respectively, and $\mathbf { \left( e - h \right) }$ are the synthetic data of the corresponding four financial variables.

![](images/a1b90d192cc100c9002c51c20812dad3237a4e91e7cc925479c8264fcdb98970.jpg)  
Fig. 6    Normalized histograms of real data and NVF-DPGAN generated data.

Heatmap is a chart form that displays data density, correlations, or distribution through color variations[36]. By observing color changes, one can discern the intensity and distribution of data in different regions. The heatmap reflects the correlations between data points, where the depth of colors can indicate the degree of correlation between distinct variables, thus aiding in observing relationships among variables. Heatmaps of real data and NVF-DPGAN generated data are shown in Fig. 7, from which the correlation between the generated data and the original data as well as the change of correlation coefficient are very consistent, which again side by side confirms the validity of the generated data.

![](images/5e5c2dd6d6f6e4650485363614c7276c9a3e3ae8e9cdbe5963b0b6bb642b0c2c.jpg)  
Fig. 7    Heatmap of real data (a) and NVF-DPGAN generated data (b).

# 3.4 Comparison experiments

VAE and GAN methods are selected for comparative experiments to verify the performance of the proposed model. The data of comparative experiments are all from the same dataset (CSMAR database). The data generated by the three models are processed using four machine learning methods, Decision Tree (DT), Logistic Regression (LR), Random Forest (RF), and Support Vector Machine (SVM). Three indicators are calculated, which are accuracy, precision, and recall.

The result of the comparative experiments is shown in Table 4. Although the differential privacy mechanism is introduced in the proposed NVFDPGAN model, it is slightly better than the GAN method in three indicators, and comprehensively superior to the VAE method. The introduction of noise visibility function improves the quality of generated data and the performance of the model. The advantage of NVF-DPGAN method is that it can provide a large amount of desensitization data on the premise of guaranteeing the same performance.

# 3.5 Ablation experiments

Kullback-Leibler (KL) divergence, also known as relative entropy or information divergence, is a measure of the dissimilarity between two probability distributions. It measures the amount of additional information required to fit the true distribution $p$ using a probability distribution $q$ . It is defined as follows:

$$
\mathrm { K L } ( p \| q ) = \sum _ { i } p ( i ) \log \frac { p ( i ) } { q ( i ) }
$$

where $p$ and $q$ denote the probability of the true and || fitted distributions on an event, respectively. “ ” represents the logical OR operation. From the above equation, it can be find that the KL divergence is asymmetric, i.e., $\operatorname { K L } \left( q \parallel p \right) \neq \operatorname { K L } \left( p \parallel q \right)$ . This means that when using $\mathrm { K L }$ divergence to measure the variability between two probability distributions, the results are affected by the choice of the baseline distribution. This is obviously an undesired phenomenon. To address this, the Jensen-Shannon (JS) divergence is introduced. The JS divergence is a variation of the KL divergence, but in comparison to KL divergence, JS divergence is characterized by symmetry and continuity. It is defined as follows:

$$
( p \nparallel q ) = \frac { 1 } { 2 } \mathrm { K L } \left( p \left\| \frac { p + q } { 2 } \right) + \frac { 1 } { 2 } \mathrm { K L } \left( q \right\| \frac { p + q } { 2 } \right)
$$

The JS divergence metric is a measure of the similarity between two probability distributions. When the two probability distributions are very similar, the JS divergence is close to 0. When the two probability distributions are more different, the JS divergence value increases.

To demonstrate the effectiveness of integrating NVF, an ablation study comparing NVF-DPGAN to DPGAN (without NVF) is conducted. Generate the same scale of data using the NVF-DPGAN and DPGAN with the same parameter Settings. The data generated by DPGAN model and the real data are used to train a time series prediction model, and the same test set is used to test the two models, and the effect of the two trained models is compared. JS divergence is used to prove the validity of the generated data, and the experimental results of real data and generated data are shown in Table 5. When the sample size is 22 000, the JS divergences of the data generated by DPGAN and the data generated by NVF-DPGAN are 0.2124 and 0.1778, respectively. The data generated by DPGAN model are not much different from the model trained using the real data, but they are still slightly inferior compared with the data generated by NVF-DPGAN, and the Mean Square Error (MSE) between the predicted value and the true value is 0.04.

# 4 Conclusion

In many studies, generated data can be employed to augment the source dataset. In certain research scenarios that involve privacy concerns related to the source dataset, generated data can be used as a substitute for the source dataset during analysis and processing. This method to some extent addresses the challenge posed by the presence of sensitive information in financial data, which often hinders its application in deep learning. In this paper, NVFDPGAN model is proposed that combines differential privacy preserving mechanism, noise visibility function and generative adversarial network. The theoretical contribution of this paper is to introduce differential privacy in the process of generating financial data and introduce noise visibility function to improve the quality of generated data, which can meet the requirements of privacy protection and data enhancement at the same time. The practical contribution of this paper is that the NVF-DPGAN model can generate high quality financial data, and the synthetic financial data has similar statistical features to the original data. These desensitized financial data can be used for deep learning training to promote the application of deep learning in the financial field. The approach presented in this paper can be generalized to other fields, such as the medical field, which is more concerned with privacy protection.

Table 4    Comparison with other data generation methods.   

<table><tr><td rowspan="2">Method</td><td colspan="3">VAE</td><td colspan="3">GAN</td><td colspan="3">NVF-DPGAN</td></tr><tr><td> Accuracy</td><td>Precision</td><td>Recall</td><td> Accuracy</td><td>Precision</td><td>Recall</td><td>Accuracy</td><td>Precision</td><td>Recall</td></tr><tr><td>DT</td><td>80.89</td><td>59.62</td><td>43.96</td><td>86.34</td><td>71.71</td><td>60.18</td><td>88.89</td><td>53.24</td><td>72.88</td></tr><tr><td>LR</td><td>83.9</td><td>71.55</td><td>75.79</td><td>89.81</td><td>81.14</td><td>76.76</td><td>90.98</td><td>84.66</td><td>82.31</td></tr><tr><td>RF</td><td>83.64</td><td>69.46</td><td>75.79</td><td>89.91</td><td>84.3</td><td>82.21</td><td>90.17</td><td>90.68</td><td>91.19</td></tr><tr><td>SVM</td><td>85.27</td><td>80.58</td><td>74.66</td><td>90.63</td><td>85.27</td><td>87.41</td><td>90.98</td><td>88.59</td><td>87.52</td></tr></table>

Table 5    Comparison of JS divergence for the real data with the data generated by DPGAN and NVF-DPGAN.   

<table><tr><td>Data</td><td>Sample size</td><td> JS divergence</td></tr><tr><td>Real data</td><td>22 000</td><td>1</td></tr><tr><td>DPGAN</td><td>22 000</td><td>0.2124</td></tr><tr><td>NVF-DPGAN</td><td>22 000</td><td>0.1778</td></tr></table>

# References

Y. Y. Qu, L. Y. Gong, and W. He, A method of online[1] shopping risk assessment based on RNN-DBN, (in Chinese), J. Harbin Univ. Sci. Technol., vol. 24, no. 4, pp. 105–109, 2019.   
[2] Y. H. Xu, Financial fraud detection models based on autoencoder and adversarial generative learning, (in Chinese), Master dissertation, Nanchang University, Nanchang, China, 2022.   
[3] Y. Liu, G. Hao, Z. Zou, and S. Wu, Stock price prediction method based on sentiment analysis and generative adversarial network, (in Chinese), J. Hunan Univ. (Nat. Sci.), vol. 49, no. 111–118, 2022.   
[4] P. Samarati, Protecting respondents identities in microdata release, IEEE Trans. Knowledge Data Eng., vol. 13, no. 6, pp. 1010–1027, 2001.   
[5] A. Machanavajjhala, J. Gehrke, D. Kifer, and M. Venkitasubramaniam, L-diversity: Privacy beyond kanonymity, in Proc. $2 2 ^ { n d }$ Int. Conf. Data Engineering, Atlanta, GA, USA, 2006, pp. 1–36.   
[6] C. Dwork, Differential privacy, in Proc. $3 3 ^ { r d }$ Int. Colloquium on Automata, Languages and Programming, Venice, Italy, 2006, pp. 1–12.   
[7] B. Wang, W. Li, A. Bradlow, E. Bazuaye, and A. T. Y. Chan, Improving triaging from primary care into secondary care using heterogeneous data-driven hybrid machine learning, Decis. Support Syst., vol. 166, p. 113899, 2023.   
[8] L. S. Lin, Y. S. Lin, D. C. Li, and Y. H. Liu, Improved learning performance for small datasets in high dimensions by new dual-net model for non-linear interpolation virtual sample generation, Decis. Support Syst., vol. 172, p. 113996, 2023.   
[9] X. Sun, H. Cheng, H. Dong, B. Qiao, S. Qin, and Q. Lin, Counter-empirical attacking based on adversarial reinforcement learning for time-relevant scoring system, IEEE Trans. Knowledge Data Eng., doi: 10.1109/TKDE.2023.3341430.   
S. Zhang, H. Chen, H. Yang, X. Sun, P. S. Yu, and G. Xu,[10] Graph masked autoencoders with transformers, arXiv preprint arXiv: 2202.08391, 2022.   
X. Sun, H. Cheng, J. Li, B. Liu, and J. Guan, All in one:[11] Multi-task prompting for graph neural networks, in Proc. 29th ACM SIGKDD Conf. Knowledge Discovery and Data Mining, Long Beach, CA, USA, 2023, pp. 2120–2131.   
[12] L. Xie, K. Lin, S. Wang, F. Wang, and J. Zhou, Differentially private generative adversarial network, arXiv preprint arXiv: 1802.06739, 2018.   
G. Acs, L. Melis, C. Castelluccia, and E. De Cristofaro,[13] Differentially private mixture of generative neural networks, IEEE Trans. Knowledge Data Eng., vol. 31, no. 6, pp. 1109–1121, 2019.   
P. Guo, S. Zhong, and K. Chen, Adaptive selection[14] method of differential privacy GAN gradient clipping thresholds, (in Chinese), Chin. J. Network Inf. Secur., vol. 4, no. 5, p. 2018041, 2018.   
T. Sattarov, M. Schreyer, and D. Borth, FinDiff: Diffusion[15] models for financial tabular data generation, in Proc. 4th ACM Int. Conf. AI in Finance, Brooklyn, NY, USA, 2023, pp. 64–72.   
[16] K. Khadka, J. Chandrasekaran, Y. Lei, R. N. Kacker, and D. R. Kuhn, Synthetic data generation using combinatorial testing and variational autoencoder, in Proc. 2023 IEEE Int. Conf. Software Testing, Verification and Validation Workshops (ICSTW), Dublin, Ireland, 2023, pp. 228–236.   
I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D.[17] Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, Generative adversarial nets, in Proc. $2 7 ^ { t h }$ Int. Conf. Neural Information Processing Systems, Montreal, Canada, 2014, pp. 2672–2680.   
M. Arjovsky, S. Chintala, and L. Bottou, Wasserstein[18] generative adversarial networks, in Proc. $3 4 ^ { t h }$ Int. Conf. Machine Learning, Sydney, NSW, Australia, 2017, pp. 214–223.   
M. Mirza and S. Osindero, Conditional generative[19] adversarial nets, arXiv preprint arXiv: 1411.1784, 2014.   
[20] C. Ledig, L. Theis, F. Huszár, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, et al., Photo-realistic single image super-resolution using a generative adversarial network, in Proc. 2017 IEEE Conf. Computer Vision and Pattern Recognition, Honolulu, HI, USA, 2017, pp. 105–114.   
Y. Yu, Z. Gong, P. Zhong, and J. Shan, Unsupervised[21] representation learning with deep convolutional neural network for remote sensing images, in Proc. 9th Int. Conf. Image and Graphics, Shanghai, China, 2017, pp. 97–108.   
[22] F. Zhang, Y. Zhang, X. Zhu, X. Chen, H. Du, and X. Zhang, PregGAN: A prognosis prediction model for breast cancer based on conditional generative adversarial networks, Computer Methods and Programs in Biomedicine, vol. 224, p. 107026, 2022.   
Z. Qin, Q. Chen, Y. Ding, T. Zhuang, Z. Qin, and K. K. R.[23] Choo, Segmentation mask and feature similarity loss guided GAN for object-oriented image-to-image translation, Inf. Process. Manage., vol. 59, no. 3, p. 102926, 2022.   
[24] G. Rizzo and T. H. M. Van, Adversarial text generation with context adapted global knowledge and a self-attentive discriminator, Inf. Process. Manage., vol. 57, no. 6, p. 102217, 2020.   
C. Dwork, F. McSherry, K. Nissim, and A. Smith,[25] Calibrating noise to sensitivity in private data analysis, J. Priv. Confident., vol. 7, no. 3, pp. 17–51, 2017.   
A. D. Sarwate and K. Chaudhuri, Signal processing and[26] machine learning with differential privacy: Algorithms and challenges for continuous data, IEEE Signal Process. Maga., vol. 30, no. 5, pp. 86–94, 2013.   
F. McSherry and K. Talwar, Mechanism design via[27] differential privacy, in Proc. $4 8 ^ { t h }$ Annu. IEEE Symp. Foundations of Computer Science (FOCS’07), Providence, RI, USA, 2007, pp. 94–103.   
C. Dwork and A. Roth, The algorithmic foundations of[28] differential privacy, Found. Trends, nos. 3&4, pp. 211–407, 2014.   
[29] F. Zhang, Y. Zhang, and X. Zhang, Desensitization method of meteorological data based on differential privacy protection, J. Cleaner Prod., vol. 389, p. 136117, 2023.   
[30] S. Voloshynovskiy, S. Pereira, V. Iquise, and T. Pun, Attack modelling: Towards a second generation watermarking benchmark, Signal Process., vol. 81, no. 6, pp. 1177–1214, 2001.   
[31] M. Kutter, S. V. Voloshynovskiy, and A. Herrigel, Watermark copy attack, in Proc. SPIE 3971, Security and Watermarking of Multimedia Contents II, San Jose, CA, USA, 2000, pp. 371–380.   
[32] S. V. Voloshynovskiy, S. Pereira, A. Herrigel, N. Baumgartner, and T. Pun, Generalized watermarking attack based on watermark estimation and perceptual remodulation, in Proc. SPIE, doi: 10.1117/12.384990.   
[33] M. Fredrikson, E. Lantz, S. Jha, S. Lin, D. Page, and T. Ristenpart, Privacy in pharmacogenetics: An end-to-end case study of personalized warfarin dosing, in Proc. $2 3 ^ { r d }$ USENIX Security Symp., San Diego, CA, USA, 2014, pp. 17–32.   
[34] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, Membership inference attacks against machine learning models, in Proc. 2017 IEEE Symp. Security and Privacy (SP), San Jose, CA, USA, 2017, pp. 3–18.   
[35] C. Dwork, A firm foundation for private data analysis, Commun. ACM, vol. 54, no. 1, pp. 86–95, 2011.   
[36] N. F. Fernandez, G. W. Gundersen, A. Rahman, M. L. Grimes, K. Rikova, P. Hornbeck, and A. Ma’ayan, Clustergrammer, a web-based heatmap visualization and analysis tool for high-dimensional biological data, Sci. Data, vol. 4, no. 1, p. 170151, 2017.

![](images/b16bc87093bbf80360c5852d3cc00354b7074910e704120f76c736821868bf22.jpg)

Fan Zhang received the BEng degree from North China University, China in 1989, the MEng degree from Jiangsu University, China in 2001, and the PhD degree in computer application technology from Beijing University of Technology, China in 2005. He is currently a professor at School of Computer and Information

Engineering, Henan University, China. He also works for Henan Key Laboratory of Big Data Analysis and Processing, China, Henan Engineering Laboratory of Spatial Information Processing, China, and the Huaihe Hospital of Henan University, China. In addition, he is the vice president of the Graphic and Image Society of Henan Province, China, the director of the Image Processing and Pattern Recognition Institute of Henan University, China, the first academic leader of key disciplines of medical imaging in Henan Province, China, the distinguished professor of Huaihe Hospital of Henan University, China. He was a visiting professor at Harvard University, USA in 2010–2011. His research focuses on artificial intelligence, medical image processing, pattern recognition, information security, and bioinformatics.

![](images/a3b8a7f5fb1f3efaad14d2d7491481acf5cb291f15b34f332b68182c2e413d63.jpg)

Luyao Wang received the BEng and MEng degrees from Henan University, China in 2021 and 2024, respectively. She is currently a PhD candidate at School of Artificial Intelligence, Beijing Normal University, China. Her research focuses on digital image processing, information security, and artificial intelligence.

Xinhong Zhang received the BEng and MEng degrees from Henan University, China in 1990 and 2006, respectively. She is currently a professor at School of Software, Henan University, China. She also works at Image Processing and Pattern Recognition Institute, Henan University, China. In addition, she is the director of the Graphic and Image Society of Henan. Her research focuses on digital image processing, pattern recognition, bioinformatics, and artificial intelligence.

![](images/9d19e0075ca780b2ae5ac026d6c31587b9ee4f793557c3317daa6a5cee616b6d.jpg)