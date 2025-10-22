# Pattern recognition with limited data: an AI model inspired by BCR-Net and SwitchNet

Ali Sever Department of Computer Science, The University of North Carolina at Charlotte, Charlotte, North Carolina, USA

# Abstract

Purpose – This research aims to develop a novel deep learning framework that effectively operates with limited data by integrating principles from inverse problem mathematics and neural networks. Inspired by BCR-Net and SwitchNet, the study introduces a pattern recognition system (PRS) that enhances traditional deep learning approaches through a data-driven, flexible architecture. By combining inverse problem methodologies with advanced neural network modules, this model seeks to overcome the challenges posed by small datasets, providing superior performance in various pattern recognition tasks compared to existing methods.

Design/methodology/approach – This study introduces a deep learning model tailored for small datasets by integrating inverse problem (IP) frameworks with neural network modules. Drawing on principles from physics and mathematics, the approach combines BCR-Net and SwitchNet architectures to enhance data augmentation in deep neural networks (DNNs). The model leverages IP’s ability to represent high-dimensional functions flexibly, using data to guide the selection of IP parameters. The effectiveness of the proposed Pattern Recognition System (PRS) is validated through a series of experiments on diverse pattern recognition tasks. The PRS performance is then benchmarked against traditional methods to demonstrate its superior capability in handling limited data scenarios.

Findings – The research demonstrates that integrating inverse problem (IP) frameworks with deep neural networks (DNNs) significantly enhances their performance on small datasets. The proposed Pattern Recognition System (PRS), inspired by BCR-Net and SwitchNet, effectively utilizes $\mathrm { I P }$ for flexible data augmentation, guiding parameter selection based on available data. Experimental validation across various pattern recognition tasks reveals that PRS outperforms traditional deep learning methods, showcasing superior accuracy and robustness in limited data scenarios. This novel approach underscores the potential of combining advanced neural architectures with mathematical principles to overcome the challenges posed by small datasets in deep learning.

Research limitations/implications – While the proposed Pattern Recognition System (PRS) demonstrates significant improvements in handling small datasets, it may face limitations in generalizability across highly diverse or significantly larger data contexts. The reliance on inverse problem (IP) frameworks and their parameterization might also introduce computational complexity and require domain-specific expertise for optimal implementation. Additionally, the integration of IP with neural networks, though beneficial for data augmentation, might limit scalability when transitioning to real-time applications or larger-scale datasets. Future research should explore broader applicability, the simplification of the model for practical deployment and further refinement to enhance efficiency and scalability across different domains.

Practical implications – The research offers a transformative approach for organizations dealing with small datasets by providing a robust deep learning framework that excels in limited data environments. The integration of inverse problem (IP) frameworks with neural network modules allows the proposed Pattern Recognition System (PRS) to achieve superior performance in tasks that typically struggle with insufficient data. This advancement enables businesses and institutions to leverage their existing small-scale datasets more effectively, enhancing predictive accuracy and decision-making processes. Additionally, the model’s flexibility and enhanced data augmentation capabilities make it suitable for a wide range of applications, from healthcare diagnostics to financial forecasting, where data scarcity is a common challenge.

Social implications – This research offers significant social benefits by democratizing the power of deep learning for communities and organizations with limited data resources. The proposed Pattern Recognition System (PRS) enables effective AI solutions even in data-scarce environments, fostering inclusivity and equal access to advanced technological capabilities. This can empower small businesses, non-profits and underserved sectors to leverage AI for social good, improving outcomes in healthcare, education and community services where data is often limited. By reducing the dependency on large datasets, this approach can help bridge the digital divide, ensuring more equitable access to the benefits of AI advancements across diverse social landscapes.

Originality/value – This research introduces a groundbreaking approach by seamlessly integrating principles from physics and inverse problem (IP) mathematics into deep learning (DL) models, specifically designed to thrive in small data environments. Unlike traditional DL methods that depend on large datasets, the proposed model – drawing inspiration from BCR-Net and SwitchNet – offers a novel architecture that enhances data augmentation and parameter selection through IP frameworks. This innovative combination not only expands the applicability of DL to data-scarce contexts but also demonstrates superior performance in pattern recognition tasks. The study’s originality lies in its unique fusion of advanced mathematical principles with cutting-edge neural network design, providing significant value to fields constrained by limited data.

Keywords Pattern recognition, Inverse problems, Machine learning, Deep neural network, Data augmentation Paper type Research article

# 1. Introduction

Despite advancements in deep learning (DL) for large-scale data analysis, many AI applications lack access to extensive datasets [1]. This challenge is evident in personalized systems, where DNNs rely on small datasets from search histories, inventory reports, and biometric measurements. Since deep learning focuses on mastering knowledge, personalization demands precise modeling while avoiding overfitting when working with small datasets. For example, in a recent MAPI survey, $5 8 \%$ of research participants reported that the most significant barrier to deploying AI solutions pertained to a paucity of data resources [2]. Therefore, making deep learning of AI work with small data has become a significant predicament. Limited research has been done in this area, even though there is a great need to design intelligent pattern recognition solutions in many application domains [3, 4]. Consequently, in this paper, we design a Pattern Recognition System (PRS) that recognizes patterns with small data using Deep Neural Network (DNN) and IP. Some inverse problems can be viewed as pattern recognition problems [5], thus making them amenable to deep neural networks. Deep learning (Deep Neural Networks, DNNs) has seen a remarkable surge in capabilities, emerging as highly effective instruments for addressing pattern recognition challenges, particularly in domains such as medical image segmentation and classification [6, 7]. Based on deep neural networks, deep learning methods have demonstrated superior performance and increased reliability compared to traditional approaches in a wide range of applications. Nonetheless, achieving these advanced results comes at the cost of requiring more hardware resources [8]. However, the multi-layer structure of DNNs can be adapted to learn effectively even with limited data. To allow IP-enabled PRS to make compelling predictions, we investigate how inverse problems connect to deep learning with small data.

We designed a model inspired by the BCR-Net and SwitchNet to successfully discover the relationship between deep learning and IP. We leverage and adapt a variant of BCR-Net and SwitchNet [9–11]. The adopted version is naturally sparse and computationally efficient. We also design a model that captures the underlying patterns in the region of interest (ROI). As a sparse model, it selectively captures the essential relationships between input and output variables and among the output variables themselves. Constructing a sparse PRS model focuses on learning the crucial dependencies that aid in predicting patterns, disregarding unnecessary ones.

To evaluate the performance of PRS, we applied it to the pattern recognition-based reconstruction of an inhomogeneous object. The BCR and IP-based deep learning model of PRS only requires a small amount of data to recognize patterns, making it an easily deployable model in practice. Furthermore, our experiments demonstrate that the PRS model performs well in effectively uncovering data dependencies, making it a desirable and practically feasible prediction tool due to its low computational requirements. Also, IP-based PRS, due to its sparse nature and ability to capture only the necessary dependencies between the input and output variables, is an interpretable model desirable for many AI models. Combining these two tools in one pattern recognition descriptor shows that IP-based augmentation of our model can improve the average pattern recognition performance.

# 2. Related work

In order to address the adverse effects of limited data availability and its associated costs, numerous researchers have recently directed their efforts toward developing systems capable of functioning effectively with smaller datasets. Numerous studies have explored the challenges of learning from limited data, and our approach builds on four key areas that have emerged in this domain: transfer learning and autoencoder-based initialization, data augmentation and compact architecture, physics-informed modeling, and meta-learning strategies for few-shot learning.

Transfer learning remains one of the most effective approaches for low-resource scenarios. In this setting, models trained on large-scale datasets are fine-tuned on a smaller target domain, leveraging the reusability of early-layer features across tasks [12, 13]. Closely related, stacked autoencoders have been used to pretrain deep neural networks in small-data regimes, enabling better weight initialization and convergence [14].

Data augmentation and model regularization techniques have also been extensively explored to improve generalization when labeled samples are scarce. Data-augmentationassisted deep learning has proven effective in classifying and monitoring contexts [15]. At the same time, elastic-net and sparsity-promoting structures such as partially connected DNNs offer competitive results in synthetic and practical tasks [16]. FuCiTNet, a GAN-inspired model, fuses class-inherent transformations to enhance generalization from few examples [17]. Further advancements include the development of feature generalization layers (FGLs) [18], multi-branch and parallel convolutional networks [19–21], and domain-specific augmentation strategies tailored to healthcare and bioinformatics applications [22–25].

Recent efforts have also focused on embedding domain knowledge or structural priors directly into network architectures. Our work is influenced by BCR-Net and SwitchNet, which model integral operators and form the foundation of our architecture. Wavelet-based decomposition has long been used to compress operators and represent functions efficiently in hierarchical bases, a principle that underpins the BCR-Net architecture. More recent work has explored incorporating wavelet structures directly into convolutional neural networks to enhance feature extraction and classification performance [26]. In the broader class of physicsinformed neural networks (PINNs), deep architecture has been applied to forward and inverse problems with embedded physical constraints [27]. Hybrid approaches combining residual learning with structured priors have also gained traction [28]. Additional related works include YOLO-based segmentation models [29], transformer-inspired SegFormer architectures [30], and hybrid methods such as PCA-GOA for masked face recognition tasks [31, 32]. Unlike those models, our inverse formulation embeds mathematical operator structure directly into the architecture.

Complementing these efforts are meta-learning and few-shot learning frameworks, which seek to generalize across tasks using minimal data per class. Prototypical Networks operate by embedding inputs into a metric space and classifying based on proximity to class prototypes [33]. Matching Networks extend this paradigm by incorporating attention-based weighting of support-query relationships [34]. MAML (Model-Agnostic Meta-Learning) focuses on learning initialization parameters that enable fast adaptation to new tasks through just a few gradient updates [35]. While powerful, these meta-learning techniques typically require access to large task distributions and extensive episodic training. In contrast, our approach leverages structured inverse operators to support learning under data scarcity without relying on external tasks or data sources.

In summary, prior research demonstrates a range of practical and theoretical strategies for addressing pattern recognition under small-data constraints. Within this context, our approach aims to contribute a unified, inverse-operator-guided deep learning framework that supports generalization in limited-data settings.

# 3. Inverse problem-based model architecture and dataset design

In this section, we explore the development of a model capable of performing pattern prediction tasks with limited data. Specifically, we aim to reconstruct an unknown, complex function in high-dimensional space by effectively matching available data. Additionally, we provide a comprehensive overview of the data generated, which serves as a means to validate the performance of our model. The PRS model is evaluated on both synthetic datasets and the BBBC real-world dataset to test its real-world applicability.

3.1 PRS model architecture and inverse problem formulation

In this section, we aim to outline the model to predict patterns based on synthetic data generated from regions of interest (ROI). Pattern recognition involves reconstructing an unknown high-dimensional nonlinear function that matches data, which can be formulated as a classic inverse problem. Inverse problems are generally modeled using physics-based principles, which can aid in reconstructing high-dimensional nonlinear functions with available data. However, inverse problems typically do not require significant data compared to DNN. Inverse map framework for DNN can solve pattern recognition problems with small data.

We frame the pattern recognition task as an inverse problem by leveraging BCR-Net and SwitchNet. This study introduces an AI-powered model integrating inverse problem (IP) techniques with neural network architectures for enhanced small-data learning. This novel approach to the pattern recognition problem demonstrates empirically that it performs better than the SVM model, which is the most widely used pattern recognition technique. In the following section, we elaborate on the rationale behind selecting the aforementioned model and highlight the distinctive features of our system.

3.2 Evaluation datasets: synthetic and real-world (BBBC022)

In this paper, we use two groups of synthetic data sets for experimental evaluation of the proposed method. The first of these data sets is based on the inclusion function in which we have two small objects to recognize in the domain; for the experimental evaluation of our proposed model, we employ two sets of synthetic data. The first set revolves around the inclusion function, wherein we have two small objects within the domain that need to be recognized in the domain $[ - 2 , 2 ]$ , shown in Figure 1.

The input values for our synthetic data sets are uniformly generated within the range [�2,2]. The corresponding target values are derived using the characteristic functions of the two objects involved in the inclusion function,

$$
| x + ( 1 , 0 . 2 ) | < 1 / 4 \mathrm { a n d } | x - ( 1 , 0 . 4 ) | < 1 / 4
$$

In the second data set, a classification problem having two manifolds is shown in Figure 2. The goal is to detect the location of the blue and purple classes. Specifically, we look for the center of these two objects. Table 1 dataset Summary summarizes the characteristics of both datasets used. The inclusion dataset is fully synthetic, while the manifold set comes from the BBBC022 real-world dataset. Each set was divided into $8 0 \%$ training and $2 0 \%$ testing samples.

The dataset comes from the Broad Bioimage Benchmark Collection (BBBC022) and comprises fluorescence microscopy images of human U2OS cells, capturing the effects of small-molecule treatments on cell morphology. The dataset structure and annotation details are available in the Supplementary Material. It includes high-quality images and manually labeled ground-truth annotations, serving as a gold standard for evaluating machine learning models in image processing and pattern recognition. These expert-verified annotations enable precise assessment of our model’s performance in image-based profiling tasks. In this case, the class priors and class-conditional densities are known, allowing for the straightforward determination of corresponding values for input data [27, 28].

![](images/0a6fdf9b0730305496060865bbf68e4d8282a75f946798e195f993050f33e9fb.jpg)  
Figure 1. Two small objects in the domain [�2,2] for pattern recognition experiments. Source: Author’s own creation

![](images/edf66a5b3736ab71f3c4543c2d3861f86f191e45d17c45bee87efd613eb23b73.jpg)  
Figure 2. Two small manifolds created by synthetic measurement data. Source: Author’s own creation

Table 1. Dataset summary   

<table><tr><td>Dataset</td><td>Source</td><td>Data type</td><td>Total samples</td><td>Train/test split</td><td>Description</td></tr><tr><td>Inclusion set</td><td>Synthetic</td><td>Numerical</td><td>1,000</td><td>800/200</td><td>Two-object detection in domain [-2,2]</td></tr><tr><td>Manifold set</td><td>BBBC022 (real)</td><td>Image Microscopy</td><td>800</td><td>600/200</td><td>Cell morphology classification</td></tr><tr><td colspan="4">Source(s): Author&#x27;s own creation</td><td></td><td></td></tr></table>

# 4. Design of neural network as an inverse problem

A core challenge for inverse problems and artificial intelligence (AI) is reconstructing an unknown high-dimensional nonlinear function that matches data. Inverse problems generally have physics-inspired models to represent them, which may help reconstruct highdimensional non-linear functions with the help of data [29, 30]. However, inverse problems do not require much data compared to DNN; if we use an inverse map framework for DNN, we may be able to solve them with small data. In other words, inverse frameworks inspired by physics help us solve DNN problems with small data. The inverse map architecture in this context serves as a natural data augmentation method.

We first introduce deep learning through the lens of an inverse map architecture, which offers a potential solution to the challenge of requiring large datasets. Unlike many applications in deep learning, such as image processing and pattern recognition, which require a massive amount of data, the required amount for inverse problems may be minimal. We plan to leverage an inverse map to design an NN module based on an integral operator, assemble a DNN from this module, and then use data to train the weights of the inverse map. The IP framework has two major strengths: it works well with limited data, and it is naturally suited for regression problems where the goal is to recover continuous values or parameters from indirect measurements. As a result, pattern recognition can be framed as a supervised regression problem, enabling precise reconstruction even in data-scarce conditions. In this approach, data and parameters work together, enhancing the overall predictive performance. This better architecture, motivated by physics and mathematics, can even work with small data.

To describe a physical situation in mathematical terms, one has to make a certain idealization so that the problem may be amenable to mathematical treatment. Without oversimplification, the model’s success is governed by how close to reality the idealized model is. Various mathematical models have been used in pattern recognition to explain the dynamic behavior of a system. Many involve the formulation and solution of inverse problems, which are generally modeled by considering the model’s components, design variables, and system parameters as deterministic values. In the inverse map framework used here, the dominant factor influencing the accuracy of pattern identification is attributed to the contrast estimator. Here, we briefly describe how the pattern recognition problem of NN can be converted into a suitable inverse problem [5]. In the framework of NN, we have the following integral equation of the first kind

$$
A f ( x ) = g ( x ) , x \in \Omega _ { k }
$$

where

$$
A f ( x ) = \int _ { \Omega } K ( x , y ) f ( y ) d y a n d K ( x , y ) = { \frac { 1 } { \left( 2 \pi \right) ^ { 2 } } } | x - y | ^ { - 2 }
$$

Moreover, $A$ is an operator from $L _ { \infty } ( \Omega )$ into $L _ { \infty } ( \Omega _ { k } )$ . The integral equation mentioned above belongs to the first kind of Fredholm integral equations, featuring a kernel of Riesz type. Significantly, the operator $A$ in (2.1) can be classified as a convolution integral operator employing the Riesz-type kernel. Initially, it is crucial to note that the equation represents a highly ill-posed problem, yet a unique solution $f$ to equation (2.1) exists and is unique. Furthermore, a logarithmic stability estimate supports the solution’s stability [5]. In order to address the issue of ill-posedness, instead of Equation (2.1), we solve the regularized equation:

$$
\Bigl ( A ^ { * } A + \alpha I \Bigr ) f ( x ) = A ^ { * } g ( x )
$$

where $\alpha$ is a regularization parameter and $I$ is the identity matrix.

In (2.2), we may consider $f ( x ) \approx ( A ^ { * } A + \alpha \pmb { I } ) ^ { - 1 } \mathtt { A } ^ { * } g ( x ) .$ As described in [6, 7], first use the SwitchNet module to represent $A$ and its adjoint $A ^ { * }$ , and then second, represent $( A ^ { * } A + \alpha I ) ^ { - 1 }$ integral operator by BCRNet to define the neural network architecture as

$$
g ( x ) - > S w i t c h N e t - > B C R N e t - > f ( x )
$$

A simplified overview of the proposed Pattern Recognition System (PRS) model is provided in Figure 3. Input data $g ( x )$ is first processed by SwitchNet, which approximates the operators $A$ and $A ^ { * }$ . The output is then passed to BCR-Net, which approximates the inverse operator $( A ^ { * } A + \alpha I ) ^ { - 1 }$ . This structured design utilizes low-rank approximations and wavelet-based decompositions to produce the predicted output, $f ( x )$ .

We suggest a combined neural network architecture, incorporating BCRNet, a newly developed neural network architecture inspired by the non-standard form proposed, and SwitchNet, a neural network model equipped with an Inverse Map. Although it is feasible to address this problem using a fully connected network, the parameter counts increase exponentially with the input and output data dimensions. By exploiting the inherent low-rank structure of the inverse conductivity problem defined in (2.2), the introduction of BCRNet and SwitchNet architectures significantly reduces the number of parameters involved, thereby streamlining the training process.

In inverse map $g ( x )  ( x ) , f ( x ) \approx ( A ^ { * } A + \alpha \pmb { I } ) ^ { - 1 } A ^ { * } g ( x )$ where $A ^ { * }$ and $\left( A ^ { * } A + \alpha I \right) ^ { - 1 }$ can be represented with a 1D CNN and BCR-Net, respectively.

We now discuss the motivation modules SwitchNet and BCRNet and show how to use the new modules to correctly solve these newest problem applications. All successful neural network (NN) architectures, such as fully connected, convolutional, recurrent, and residual networks, are fundamentally grounded in underlying principles of mathematics and physics. Convolutional neural networks are commonly used for image processing tasks, recurrent neural networks for speech and sequence data, and residual neural networks represent one of the most advanced architectures in deep learning. Fully connected NN has a dense operator behind it, whereas the convolution NN has a translation-invariant operator. All these neural networks have mathematics behind them.

The natural question is, given the integral operator in the inverse problem that was stated in (2.2), what would the neural network architecture be associated with the integral operator? We have the right building blocks if we can build the neural network architecture for this integral operator. When represented as a matrix, the integral operator $A$ exhibits a key property [Figure 4]: Its off-diagonal blocks are numerically low-rank, allowing for a sparse representation. This enables these blocks to be replaced with efficient numerical approximations, significantly enhancing computational performance.

![](images/c0c1d4dd39a77956568d37da655d72d0fca685f9e7863cc6f6721bbfa37a7ce8.jpg)  
Figure 3. Architecture of PRS model integrating SwitchNet and BCR-Net for inverse operator learning. Source: Author’s own creation

![](images/2d5ecaf4f0daa300e8dfb288e679367512ce4b11c03b33251f2fa6c5a0c6dddf.jpg)  
Figure 4. Sparse structure of integral operator A with low-rank off-diagonal blocks for efficiency. Source: Author’s own creation

Therefore, the plan is to find a convenient sparse-compact representation of the integral operator. Here, we use the BCR-net method inspired by the non-standard wavelet form proposed by Beylkin, Coffman, and Rokhlin [8] for an approximation of a non-linear integral operator given as an nxn matrix A utilizing merely tens of thousands of parameters in the following way:

For a given integral operator, the square matrix can be written as a product of three matrices because of the low rank of the off-diagonal blocks. In Figure 5, the first and last matrices represent the redundant wavelet transformation. This one-to-one orthogonal mapping preserves the degrees of freedom while retaining wavelet and scaling coefficients within the transformation. This transformation increases the size by a factor of two. At the same time, the middle matrix, known as the nonstandard form, consists of three diagonally dominant matrices that provide an efficient approximation of the original operator, forming the core idea of BCR-Net.

A natural question arises: why is this linked to neural networks? As shown in Figure 6(a–c), the computation at each scale can be effectively encoded within a neural network architecture.

![](images/73f15292d53c575d34ce338603a2b78b810947130b123c21deb341227f886a7d.jpg)  
Figure 5. BCR-Net architecture leveraging wavelet-based decomposition for compact operator representation. Source: Author’s own creation

![](images/e7dd3024ac266f50ff07f6841667428869490da28258e8abc67d1c2e710171e4.jpg)  
Figure 6. (a) Neural network formulation of BCR, translating integral operator computations into structured layers, (b) Three-layer BCR model leveraging wavelet-based decomposition, (c) Generalized BCR-Net with ReLU activations for enhanced non-linearity and feature extraction. Source: Author’s own creation

When the input vector is applied, it corresponds to a three-layer network. The first step involves multiplying two diagonal matrices, and assuming these operators are translationinvariant, this operation transforms into a convolution, mapping one channel to two channels. The middle operator acts as a transformation from two channels to two, followed by a final step that reduces the two channels back to one. This structure defines the neural network architecture for a single layer. The complete neural network is constructed by stacking these layers sequentially. The top layer represents the finest.

scale, where wavelet coefficients are computed. These coefficients are combined with scaling coefficients to propagate information to the next layer. The three-layer structure is illustrated in Figure 6(b).

However, for significant problems, one may have many layers. The idea of transforming this into a neural network is to insert a few more internal layers and introduce a non-linearity by inserting ReLUs, so this will give us a nonlinear generalized version of BCR-net shown in 6(c).

Given an $n x n$ integral operator A, we may approximate nonlinear operator A by partitioning it into $n ^ { 1 / 2 } x n ^ { 1 / 2 }$ , applying low-rank approximation to each block, and generalizing this via inserting ReLUs [Figure 7].

# 5. Experimental evaluation, benchmarking, and analysis

In this section, we investigate the efficacy of the PRS learning algorithm introduced in the previous sections. One way to objectively test such a model is to apply it to computational pattern recognition network models for which the mechanisms are entirely known, as described in Section 3.2.

![](images/74d159fd65c113e6fa94d98ea0a5b7bd11d5860f6381073d1e74464eb2148d30.jpg)  
Figure 7. Neural network representation of integral operator $A$ with low-rank approximations. Source: Author’s own creation

In this study, we denote ‘real object’ by $f _ { r }$ and ‘computed object’ by $f _ { c }$ Generating $g ( x )$ in the integral equation (2.1) is essential to evaluate the method. Initially, we define $f ( x )$ and subsequently compute $g ( x )$ numerically. Afterward, we utilize the numerical values of $g ( x )$ as our dataset and extract the underlying pattern within the region of interest.

To ensure transparency and facilitate reproducibility, we present a high-level pseudocode and a formal algorithmic description of the proposed Pattern Reconstruction System (PRS) model. The pseudocode captures the essential conceptual steps of the methodology, abstracted for clarity. At the same time, the accompanying algorithm provides a more detailed procedural breakdown aligned with the mathematical framework introduced in Section 2.

The pseudocode outlines the construction of training pairs using the inverse operator A, the generation of forward data, model training, and performance evaluation. In contrast, the formal algorithm incorporates dataset discretization and numerical computation of the integral operator as defined in Equation (2.1), offering a comprehensive view of the PRS training and prediction pipeline.

Both formats are presented side by side in Table 2 to aid in comparative interpretation.

Table 3 presents our experiments’ detailed model configurations and hyperparameter settings. These include the choice of optimizer, learning rate, batch size, regularization coefficient, network architecture, and activation functions. The configurations were selected based on standard practices and empirical tuning to ensure fairness and optimal performance across all models.

The first group of experiments was carried out for synthetic data sets on recognizing two small objects in the domain $[ - 2 , 2 ] \mathbf { x } [ - 2 , 2 ]$ . The reconstruction closely approximates $\alpha$ the inclusion domain, providing a highly accurate representation [Figure 8]. Based on the presented reconstructions, the numerical experiments mentioned earlier have demonstrated an intense match between the predicted and actual inclusion domains. The model generated each inclusion domain with an accuracy of $9 2 . 6 7 \%$ .

In the second set of experiments, we aimed to detect the centers of the blue and purple classes in the classification problem illustrated in Figure 2. In Figure 8, the center of the location of the objects was well produced, and the proposed PRS model could distinguish between the two objects. To optimize efficiency, the key factors to adjust in Eq. (2.1) are the regularization parameter and the number of grid points. These parameters play a crucial role in the overall performance of the process.

Table 2. Pseudocode and algorithm for the PRS model   

<table><tr><td>Pseudocode 1: PRS model</td><td>Algorithm 1: PRS model (formal description)</td></tr><tr><td>Input:</td><td>Input:</td></tr><tr><td>i. Training dataset Dtraining</td><td>i. Three distinct synthetic datasets and corresponding PRS</td></tr><tr><td>ii. Inverse operator A</td><td>model ii. Discretization levels N= {N1,.,Nn} per dataset</td></tr><tr><td>Output: Reconstructed object pattern 1. Specify target function f using Draining and</td><td>Output: Synthesized decision set Dsynthesized 1. Specify f in Equation (2.1) using Dtraining and A</td></tr><tr><td>operatorA 2. Numerically compute g(x) = Af(x)</td><td>2.Numerically evaluate the integral in (2.1) to compute G,</td></tr><tr><td></td><td>then tune to obtain Gtuned</td></tr><tr><td>3.Train PRS using (g(x),f(x)) pairs 4. Predict fc for test samples</td><td>3. Use Gtuned to compute predicted function fc</td></tr><tr><td>5. Compare predicted feand true fr tocompute5.Return Dsyntheszed</td><td>4. Compare f. and reference fr to generate Dsynthesized</td></tr><tr><td>accuracy metrics Source(s):Author&#x27;s own creation</td><td></td></tr></table>

able 3. Model hyperparameters and architectural configurations used in PRS, SVM, and LSTM experime   

<table><tr><td>Parameter</td><td>PRS</td><td>SVM</td><td>LSTM</td></tr><tr><td>Learning rate</td><td>0.001</td><td>N/A</td><td>0.001</td></tr><tr><td>Batch size</td><td>64</td><td>N/A</td><td>64</td></tr><tr><td>Epochs</td><td>100</td><td>N/A</td><td>100</td></tr><tr><td>Regularization coefficient (@）</td><td>0.01</td><td>0.1(C parameter)</td><td>0.001</td></tr><tr><td>Architecture</td><td>3-layer (inverse mapping + dense)</td><td>Linear kernel</td><td>2-layer</td></tr><tr><td>Activation function</td><td>ReLU</td><td>N/A</td><td>LSTM+dense Tanh</td></tr><tr><td colspan="4">Source(s):Author&#x27;s own creation</td></tr></table>

![](images/58e7bf9a443c0dd35493d2311fd88360741294346b0dfc7f14241209761cbad6.jpg)  
Figure 8. Reconstructed inclusion domain (synthetic data). Source: Author’s own creation

To validate the reconstruction ability of the proposed model, we used the above-described synthetically generated data whose distributions are known (Figure 9(a) and (c)). The results of this experiment are presented in Figure 9(b) and (d). The PRS model demonstrated strong and consistent performance across all three datasets SOD, SMD, and MMD, as defined in Table 4, spanning tasks from small-object classification to complex multi-manifold recognition. To comprehensively evaluate PRS, we compared it with two established baseline models, SVM and LSTM, using seven metrics: Average Squared Error (ASE), Misclassification Rate (MR), Kolmogorov–Smirnov statistic (KS), Precision, Recall, F1-score, and ROC-AUC.

As shown in Table 4, PRS outperforms both SVM and LSTM in most metrics across all datasets. For instance, in the MMD dataset, which presents the highest complexity with 10 classes, PRS achieves a lower ASE (0.0519), lower MR (0.0680), and higher KS (0.6600), indicating better alignment between predicted and actual distributions. Furthermore, PRS leads in classification metrics with a Precision of 0.90, a Recall of 0.91, an F1-score of 0.905, and an ROC-AUC of 0.92, surpassing both SVM and LSTM.

These trends hold across the SOD and SMD datasets as well, where PRS consistently yields better or comparable ASE and MR, and shows powerful results in Recall and ROC-AUC, which are crucial for high-sensitivity applications. KS statistics, a measure of distributional similarity, also confirms PRS’s advantage in maintaining output stability and accuracy across classes.

The ROC analysis presented in Figure 10 highlights the superior discriminative performance of the PRS model in the MMD object recognition task. Compared to SVM and LSTM, PRS consistently achieves higher sensitivity across a wide range of specificity thresholds. Notably, aligning the PRS validation curve (dashed line) with its training and test curves indicates minimal overfitting and robust generalization to unseen data. These results underscore PRS’s effectiveness in handling complex, multi-manifold classification challenges and its advantage in data-scarce environments.

To get a well-rounded comparison, we tested the PRS model against traditional approaches LSTM and SVM, as well as more advanced techniques such as Prototypical Networks and Transfer Learning, which are known to perform well with limited data. These comparisons helped us better understand where PRS regarding accuracy and efficiency. While Table 5 shows that PRS takes longer to train than the other four models, this drawback is offset by not requiring repeated cross-validation to tune complexity parameters. Additionally, as it generates models with fewer variables, the computational time for evaluating test points, often the primary concern in practical scenarios, is significantly reduced. Based on the average training times for the BBBC022 dataset, the PRS model demonstrates a competitive balance between performance and computational efficiency. While its training times, 62.3 seconds for Dataset 1 (Inclusion) and 75.1 seconds for Dataset 2 (Manifold), are higher than those of simpler models like SVM (15.8s/20.3s) and LSTM $( 3 8 . 2 \mathrm { s } / 4 6 . 7 \mathrm { s } )$ , PRS outperforms them in terms of accuracy, making the extra training cost justifiable. Compared to Prototypical Networks $( 5 0 . 0 { \mathrm { s } } / 5 8 . 0 { \mathrm { s } } )$ and Transfer Learning (70.0s/85.0s), PRS sits comfortably in the middle, offering accuracy nearly on par with Transfer Learning but at a slightly lower training cost. This makes PRS a reasonable alternative to established models, particularly in scenarios where high pattern recognition accuracy is essential. However, complete reliance on heavy pretraining (as in Transfer Learning) is less desirable.

![](images/9860f70aab66999bc592de2916d0d17c4942df5610d212a6dfa03f44823c4896.jpg)  
Figure 9. (a, c) Ground truth manifold structures; (b, d) PRS-predicted class centers. Source: Author’s own creation

The PRS model demonstrates strong performance on controlled datasets, yet its generalization to noisy or highly imbalanced real-world data has not been extensively tested. Training time is higher than conventional models, though offset by lower testing cost. Future work will explore model robustness to noise and domain transfer.

Table 6 compares the performance of five different models, LSTM, Prototypical Networks, Transfer Learning, SVM, and PRS on the BBBC022 dataset (MMD), a benchmark for pattern recognition in bioimage analysis. Transfer Learning led the pack with a top accuracy of $9 2 \%$ but PRS followed closely at $9 1 \%$ , showing it can match the best while offering unique strengths in adaptability and computational efficiency. Prototypical Networks also delivered solid results at $8 8 \%$ . LSTM and SVM lagged with $8 2 \%$ and $8 0 \%$ , respectively, highlighting their limitations in this setting.

4. Dataset details and classification accuracy of PRS , SVM, and L STM models   

<table><tr><td>Dataset</td><td>Domain</td><td>#Classes</td><td>Size</td><td>Model</td><td>ASE</td><td>MR</td><td>KS</td><td>Precision</td><td>Recall</td><td>F1- score</td><td>ROC- AUC</td></tr><tr><td>SOD</td><td>Small objects</td><td>2</td><td>5.2k</td><td>PRS</td><td>0.0850</td><td>0.0690</td><td>0.6740</td><td>0.91</td><td>0.92</td><td>0.915</td><td>0.94</td></tr><tr><td></td><td></td><td></td><td></td><td>SVM</td><td>0.0725</td><td>0.0700</td><td>0.6750</td><td>0.88</td><td>0.89</td><td>0.885</td><td>0.91</td></tr><tr><td></td><td></td><td></td><td></td><td>LSTM</td><td>0.0870</td><td>0.0750</td><td>0.6880</td><td>0.86</td><td>0.87</td><td>0.865</td><td>0.90</td></tr><tr><td>SMD</td><td>2-manifolds</td><td>2</td><td>6.1k</td><td>PRS</td><td>0.0625</td><td>0.0710</td><td>0.6556</td><td>0.93</td><td>0.94</td><td>0.935</td><td>0.95</td></tr><tr><td></td><td></td><td></td><td></td><td>SVM</td><td>0.0549</td><td>0.0650</td><td>0.5963</td><td>0.89</td><td>0.88</td><td>0.885</td><td>0.90</td></tr><tr><td></td><td></td><td></td><td></td><td>LSTM</td><td>0.0951</td><td>0.0620</td><td>0.6543</td><td>0.85</td><td>0.86</td><td>0.855</td><td>0.89</td></tr><tr><td>MMD</td><td>Multi-manifolds</td><td>10</td><td>6.1k</td><td>PRS</td><td>0.0519</td><td>0.0680</td><td>0.6600</td><td>0.90</td><td>0.91</td><td>0.905</td><td>0.92</td></tr><tr><td></td><td></td><td></td><td></td><td>SVM</td><td>0.0629</td><td>0.0750</td><td>0.5078</td><td>0.87</td><td>0.86</td><td>0.865</td><td>0.88</td></tr><tr><td></td><td>Source(s):Author&#x27;s own creation</td><td></td><td></td><td>LSTM</td><td>0.0950</td><td>0.0700</td><td>0.6956</td><td>0.84</td><td>0.83</td><td>0.835</td><td>0.87</td></tr></table>

![](images/8fa2a5a4afb453f217867440ccc60b294b8c118d2dc1683bd01187ce9858c5c4.jpg)  
Figure 10. ROC curves of PRS, SVM, and LSTM models on the MMD dataset. Source: Author’s own creation

Table 5. Average training time (in seconds)   

<table><tr><td>Model</td><td>Dataset 1 (inclusion)</td><td>Dataset 2 (manifold)</td></tr><tr><td>LSTM</td><td>38.2</td><td>46.7</td></tr><tr><td>Prototypical networks</td><td>50.0</td><td>58.0</td></tr><tr><td>Transfer learning</td><td>70.0</td><td>85.0</td></tr><tr><td>SVM</td><td>15.8</td><td>20.3</td></tr><tr><td>PRS</td><td>62.3</td><td>75.1</td></tr><tr><td>Source(s):Author&#x27;s own creation</td><td></td><td></td></tr></table>

![](images/544c3824f8beae6fd79e7d84c392851e92edd00e8cc6e28306f29718e7573ba7.jpg)  
Table 6. Accuracy comparison on BBBC022 dataset (MMD)

Overall, PRS proves to be a strong and reliable contender for pattern recognition tasks in biomedical imaging.

# 6. Conclusion and future work

The experiment showcases our framework’s successful implementation and effectiveness while demonstrating the practicality of IP-based PRS. The comparative findings reveal that it achieves $9 0 \%$ accuracy across multiple applications. Furthermore, we have compared our IPbased PRS model to baseline models, LSTM and SVM, and advanced models for small data scenarios such as Prototypical Networks and Transfer learning. Our results surpassed those of SVM, LSTM, and Prototypical Networks by $1 1 \%$ , $9 \%$ , and $3 \%$ , respectively. On the other hand, Transfer Learning did better than the PRS by $3 \%$ , but the PRS model offers a favorable trade-off between accuracy and training efficiency. The proposed inverse map framework for DNN makes it possible for the model to achieve strong generalization from limited data. That makes it well-suited for small data prediction problems where both predictive accuracy and computational efficiency are critical.

Nevertheless, the model has limitations, highlighting the need for further examination.

Future work will explore domain generalization, adaptability to noisy data, better numerics, and extending PRS to more complex and heterogeneous datasets. Integrating physics-based operators into deep learning offers a promising path for improving performance in data-scarce applications.

Additionally, the PRS model was developed using moderate computing power (a midrange GPU setup), and its sparse architecture allows it to maintain practical training times without requiring specialized hardware. This makes PRS a cost-effective and accessible solution for small-data problems in academic or clinical settings.

Overall, we aim for this approach to serve as both a challenge and a benchmark. By leveraging the close relationship between inverse problems and pattern recognition, we can create an innovative learning algorithm suitable for handling small data sets.

# Supplementary material

The supplementary material for this article can be found online.

# References

1. Ali A, Qadir J, Rasool RU, Sathiaseelan A, Zwitter A, Crowcroft J. Big data for development: applications and techniques. Big Data Anal. 2016; 1: 2. doi: 10.1186/s41044-016-0002-4.   
2. Betancourt A. Making AI work with small data. IndustryWeek; 2020. Available from: https:// www.industryweek.com/   
3. Hu G, Peng X, Yang Y, Hospedales TM, Verbeek J. Frankenstein: learning deep face representations using small data. IEEE Trans Image Process. 2018; 27(1): 293-303. doi: 10.1109/ TIP.2017.2756450.   
4. Anaby-Tavor A, Carmeli B, Goldbraich E, Kantor A, Kour G, Shlomov S, Tepper N, Zwerdling N. Do not have enough data? Deep learning to the rescue. Proc AAAI Conf Artif Intell. 2020; 34(5): 7383-90. doi: 10.1609/aaai.v34i05.6233.   
5. Sever A. An inverse problem approach to pattern recognition in industry. Appl Comput Inform. 2015; 11(1): 1-12. doi: 10.1016/j.aci.2014.02.004.   
6. Esteva A, Kuprel B, Novoa RA, Ko J, Swetter SM, Blau HM, Thrun S. Dermatologist-level classification of skin cancer with deep neural networks. Nature. 2017; 542(7639): 115-18. doi: 10.1038/nature21056.   
7. Bychkov D, Linder N, Turkki R, Nordling S, Kovanen PE, Verrill C, Walliander M, Lundin M, Haglund C, Lundin J. Deep learning-based tissue analysis predicts outcome in colorectal cancer. Sci Rep. 2018; 8(1): 3395. doi: 10.1038/s41598-018-21758-3.   
8. Kim GB, Jung KH, Lee Y, Kim HJ, Kim N, Jun S, Seo JB, Lynch DA. Comparison of shallow and deep learning methods on classifying the regional pattern of diffuse lung disease. J Digit Imaging. 2017; 31(4): 415-24. doi: 10.1007/s10278-017-0028-9.   
9. Fan Y, Bohorquez CO, Ying L. BCR-Net: a neural network based on the nonstandard wavelet form. J Comput Phys. 2019; 384: 1-15. doi: 10.1016/j.jcp.2019.02.002.   
10. Khoo Y, Ying L. SwitchNet: a neural network model for forward and inverse scattering problems. SIAM J Sci Comput. 2019; 41(5): A3182-201. doi: 10.1137/18m1222399.   
11. Beylkin G, Coifman R, Rokhlin V. Fast wavelet transforms and numerical algorithms I. Commun Pure Appl Math. 1991; 44(2): 141-83. doi: 10.1002/cpa.3160440202.   
12. Shin H, Roth HR, Gao M, Lu L, Xu Z, Nogues I, Yao J, Mollura D, Summers RM. Deep convolutional neural networks for computer-aided detection: CNN architectures, dataset characteristics and transfer learning. IEEE Trans Med Imaging. 2016; 35(5): 1285-98. doi:   
10.1109/tmi.2016.2528162.   
13. Yosinski Y, Clune J, Bengio Y, Lipson H. How transferable are features in deep neural networks? Adv Neural Inf Process Syst. 2014; 27: 3320-28.   
14. Hinton GE, Salakhutdinov RR. Reducing the dimensionality of data with neural networks. Science. 2006; 313(5786): 504-7. doi: 10.1126/science.1127647.   
15. Yu Y, Si X, Hu C, Zhang J. A review of recurrent neural networks: LSTM cells and network architectures. Neural Comput. 2019; 31(7): 1235-70. doi: 10.1162/neco_a_01199.   
16. Krizhevsky A, Sutskever I, Hinton GE. ImageNet classification with deep convolutional neural networks. Commun ACM. 2017; 60(6): 84-90. doi: 10.1145/3065386.   
17. Hudson DA, Manning CD. Compositional attention networks for machine reasoning. In: International Conference on Learning Representations (ICLR); 2018.   
18. Yang M, Zhang T, Xu C, Fan X, Guo F. A fast and accurate feature generalization network for lowsample image classification. Pattern Recognit. 2020; 107: 107506. doi: 10.1016/ j.patcog.2020.107506.   
19. Liu S, Qi S, Zhu Y. Multi-branch deep learning network for remote sensing image classification. Remote Sens. 2019; 11(6): 666.   
20. Chen H, Wang Y, Liu W. Improved CNN-based hyperspectral image classification using multibranch residual network. Sensors. 2020; 20(7): 1882.   
21. Li X, Guo Y, Jin L. Multi-branch fusion network for vehicle classification. IEEE Access. 2020; 8:   
125870-8.   
22. Shorten S, Khoshgoftaar TM. A survey on image data augmentation for deep learning. J Big Data.   
2019; 6(1): 1-48. doi: 10.1186/s40537-019-0197-0.   
23. Wang Z, Meng Q, Zhu Q. Automatic data augmentation for medical image classification with deep learning. IEEE Access. 2020; 8: 148161-73.   
24. Ravi S, Wong A, Brudno M, Berthelot M, Andreu-Perez J, Lo B, Yang GZ. Deep learning for health informatics. IEEE J Biomed Health Informatics. 2017; 21(1): 4-21. doi: 10.1109/ jbhi.2016.2636665.   
25. Esteva P, Kuprel B, Novoa RA, Ko J, Swetter SM, Blau HM, Thrun S. Dermatologist-level classification of skin cancer with deep neural networks. Nature. 2017; 542(7639): 115-18. doi:   
10.1038/nature21056.   
26. Fan J, Li J, Wang Y, Zhao H. Learning wavelet-based convolutional neural networks for image classification. IEEE Trans Image Process. 2020; 29: 7307-22.   
27. Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. J Comput Phys. 2019; 378: 686-707. doi: 10.1016/j.jcp.2018.10.045.   
28. Kingma DP, Mohamed S, Rezende DJ, Welling M. Semi-supervised learning with deep generative models. Adv Neural Inf Process Syst. 2014; 27: 3581-89.   
29. Bouzid A, Taghezout M, Dornaika F, Chen H. Automated on-site broiler live weight estimation through YOLO-based segmentation. Comput Electronics Agric. 2023; 205: 107572. doi: 10.1016/ j.compag.2022.107572.   
30. Yang H, Zhai Y, Tang W, He X. Semantic segmentation of microbial alterations based on SegFormer. Front Microbiol. 2022; 13: 1023456.   
31. Dineshkumar R, Kumaravel A, Devaraj D. A novel hybrid approach to masked face recognition using robust PCA and GOA optimizer. Neural Comput Appl. 2023; 35: 25915-32.   
32. Dineshkumar R, Kumaravel A. Innovative hybrid approach for masked face recognition using pretrained mask detection and segmentation, robust PCA, and KNN classifier. Multimedia Tools Appl. 2023; 82: 12961-83.   
33. Snell J, Swersky K, Zemel R. Prototypical networks for few-shot learning. Adv Neural Inf Process Syst. 2017; 30: 4077-87.   
34. Vinyals O, Blundell C, Lillicrap T, Wierstra D, Kavukcuoglu K. Matching networks for one shot learning. Adv Neural Inf Process Syst. 2016; 29: 3630-8.   
35. Finn C, Abbeel P, Levine S. Model-agnostic meta-learning for fast adaptation of deep networks. In: Proceedings of the 34th International Conference on Machine Learning; 2017. p. 1126-35.