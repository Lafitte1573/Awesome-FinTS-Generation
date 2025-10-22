# NMTucker: Non-linear Matryoshka Tucker Decomposition for Financial Time Series Imputation

Uras Varolgunes\*   
Dan Zhou\*   
uv27@njit.edu   
dz239@njit.edu   
New Jersey Institute of Technology   
New Jersey, United States

Dantong Yu New Jersey Institute of Technology New Jersey, United States dtyu@njit.edu

Ajim Uddin New Jersey Institute of Technology New Jersey, United States au76@njit.edu

# ABSTRACT

Missing values in financial time series data are of paramount importance in financial modeling and analysis.Appropriately handling missing data is essential to ensure the accuracy and reliability of financial models and forecasts.In this paper,we focus on datasets containing multiple attributes of different firms across time,such as firm fundamentals or characteristics,which can be represented as three dimensional tensors with the dimensions time,firm and attribute.Hence,the task of imputing missing values for these datasets can also be formulated as a tensor completion problem. Tensor completion has a wide range of applications,including link prediction, recommendation,and scientific data extrapolation. The widely used completion algorithms, CP and Tucker decompositions, factorize an N-order tensor into N embedding matrices and use multi-linearity among the factors to reconstruct the tensor.Realworld data are often highly sparse and involve complex interactions beyond simple N-order linearity; they demand models capable of capturing latent variables and their non-linear multi-way interactions.We designan algorithm,called Non-Linear Matryoshka Tucker Completion (NMTucker), that uses element-wise Tucker decomposition, multi-layer perceptrons,and non-linear activation functions to solve these challenges and ensure its scalability. To avoid the overfitting problem with existing neural network-based tensor algorithms,we develop a novel strategy that recursively decomposes a tucker core into smaller ones,reduces the number of trainable parameters,and regularizes the complexity.Its structure is similar to Matryoshka dolls of decreasing size in which one is nested inside another.We conduct experiments to show that NMTucker effectively mitigates overfitting and demonstrate its superior generalization capability (up to $5 3 . 9 1 \%$ less RMSE) in comparison with the state-of-the-art models in multiple tensor completion tasks.

# CCS CONCEPTS

· Computing methodologies Factorization methods; Regularization; Learning latent representations; $\bullet$ Computer systems organization Embedded systems.

# KEYWORDS

Non-linear Tensor Decomposition, Sparse Tensor Completion, Data Imputation,Financial Time Series

# ACMReference Format:

Uras Varolgunes,Dan Zhou,Dantong Yu,and Ajim Uddin.2023.NMTucker: Non-linear Matryoshka Tucker Decomposition for Financial Time Series Imputation.In 4thACMInternational Conference onAI in Finance (ICAIF '23),November 27-29,2023,Brooklyn,NY,USA.ACM,New York,NY,USA, 8pages.https://doi.org/10.1145/3604237.3626909

# 1INTRODUCTION

Financial datasets often contain missing values due to various reasons,such as irregular reporting frequency, infrequent trading activity,and corporate actions like mergers,acquisitions,or restructuring.For instance,our financial analysts’ forecast of earnings per share (EPS) data [2o] has a substantial missing rate of over $9 5 \%$ The high missing rate arises from multiple factors, including analysts intentionally skipping reports for certain firms,personal or professional biases towards specific companies,or their tendency to follow,stop following,and re-follow firms [32]. These missing values can be spread across different features and time points, necessitating critical imputation steps to ensure a complete and usable dataset.

Additionally, financial datasets are typically characterized by a multitude of variables or features, representing various aspects of financial instruments,economic indicators,or market parameters. This complexity often results in high-dimensional data.For example, financial statements typically include multiple variables representing various aspects of a company's financial health, performance, and position.Representing such firm accounting fundamentals or firm characteristics data in a tensor-based structure allows the detection of latent factors,dependencies,causality,and interactions across firms over time.

Given these characteristics,tensor completion offers a compeling solution to recover missing values in multi-way data arrays, making it well-suited for imputing financial data. Tensors, as multi-dimensional arrays, naturally capture complex relationships present in financial datasets.By leveraging the tensor structure and exploiting multi-dimensional correlations,tensor completion effectively imputes missing values,which traditional econometric imputation techniques like mean imputation,regression-based methods,or principal component analysis [3,5,14,26] may struggle to accomplish.

Most tensor completion algorithms are based on two types of tensor decompositions: Tucker [30] and CANDECOMP/PARAFAC (CP)[6,16].CP decomposition is considered to be a particular case of Tucker, where the interaction core is a super diagonal hypercube. CP requires that all embedding matrices corresponding to each order of a tensor have the uniform rank to support the Hadamard products among the latent vectors and aggregation.

This uniform rank assumption can be problematic in practice, because in real-world data,each order of a tensor represents a different entity from the real-world, such as a movie,director, user, location or time.Thus,it is unreasonable to assume homogeneity among their ranks.Tucker-based algorithms remove this constraint and allow flexible multi-linear relationships. $N _ { ☉ }$ -order Tucker completion generalizes matrix factorization and decomposes a single tensor cell into $N$ latent vectors of any dimensionality and the multi-way interactions among these vectors.

The Tucker model does not encode all information solely into the factor matrices.Instead,it distributes the learned knowledge between the core tensor and the factor matrices,allowing complex multi-linear relationships to be discovered and further decomposed.Also,Tucker reconstruction is essentially a higher-order principal component analysis and has a well-defined performance guarantee with the error bound derived from the well-established Eckart-Young-Mirsky [24] theorem on low-rank matrix approximation and covariance analysis.Despite the advantages,Tucker decomposition has several limitations,including its linearity assumption that incurs high bias to real-world data with complex non-linear interactions and the overfitting problem posed by the large size of the core tensor when the original tensor has a high orderand is extremely sparse.In this paper, we introduce a novel regularization method which reduces the freedom of the core tensor by recursively decomposing it.The main strategy is to design a deep layered architecture to reduce the number of redundant trainable parameters while preserving the model's expressive power.

More specifically,we propose NMTucker, a novel Tucker completion algorithm that mines the patterns in non-linear complex data distributions while avoiding the overfitting problem with a recursive decomposition strategy.We introduce our novel recursive decomposition inspired by Russian Matryoshka dolls of decreasing size embedded one inside another. Considering the expensive time cost of applying alternating least squares (ALS) to perform multiple decompositions simultaneously, we instead stack multiple layers of decompositions (Figure 1),where each layer efficiently learns an $n$ -mode Tucker factorizationand the associated embedding matrices.By using smaller ranks within the lower layers,we are able to reconstruct the original core matrix using a smaller number of parameters,while simultaneously allowing a more expressive functional form.We evaluated our algorithm for tensor completion on three real-world financial datasets: firms'earnings per share (EPS) from financial analyst forecast data,firms'fundamentals,and firms' characteristics. The results demonstrate consistent performance superiority, reducing imputation RMSE by up to $5 3 . 9 1 \%$

![](images/973bebfd43540de907e02c95e3075d165e2ef121490258f0e6bb0f1b86db5b92.jpg)  
Figure 1: The 3-layer Neural Network for Matryoshka Tucker Decomposition and Tensor Completion, $\begin{array} { r l } { \dot { g } ^ { ( 1 ) } } & { { } \in } \end{array}$ $\mathbb { R } ^ { R _ { 1 } ^ { ( 1 ) } \times R _ { 2 } ^ { ( 1 ) } \times R _ { 3 } ^ { ( 1 ) } }$ . Factor matrices $U ^ { ( l ) } ~ \in ~ \bar { \mathbb { R } ^ { ( l ) } } \times R _ { 1 } ^ { ( l + 1 ) }$ ， $V ^ { ( l ) } ~ \in$ $\mathbb { R } ^ { R _ { 2 } ^ { ( l ) } \times R _ { 2 } ^ { ( l + 1 ) } }$ and w(Rx contain trainable parameters, while $\mathcal { G } ^ { ( 2 ) }$ and $\mathcal { G } ^ { ( 3 ) }$ store the intermediate results computed during the forward propagation using the elements of the previous layers.The training inputs for the network model are key-value pairs of the index tuple $( i _ { 1 } , i _ { 2 } , i _ { 3 } )$ and the corresponding observation $x _ { i _ { 1 } i _ { 2 } i _ { 3 } }$ is the target.

· We propose a novel non-linear tensor completion model to capture latent factors.   
·We design a multi-layer decomposition strategy to mitigate the overfitting problem for tensor factorization and completion.   
· We conduct tensor completion experiments on real-world data and show that our model outperforms the state-of-theart models.   
· We compare the regularization performance between NMTucker and the standard Lasso and show the superiority of the NMTucker based regularization.

# 2RELATED WORK

The literature related to this paper can be grouped into two broader categories,first,missing value imputation in financial data and second tensor factorization and completion algorithms.

Financial data often contains incomplete observations,which can be found in asset pricing characteristics,proprietary banking,and pricing data.To handle missing data, finance researchers commonlyresort to procedures like mean imputation or forward filling [3,14].However, relying solely on these simple imputation methods without considering the underlying data distribution or multi-dimensional correlation in high-dimensional data can compromise the reliability of the imputed values.For instance,mean imputation may distort the data distribution by filling missing values with the average,and forward filling might overlook potential temporal dependencies.Other approaches in finance utilize traditional econometric methods,such as principal component analysis [5,26]. However, these methods typically assume that the data follows a multivariate normal distribution or is missing at random, making them sensitive to deviations from these assumptions and potentially introducing bias in the analysis.The academic community has recently shown growing interest in utilizing advanced methods to impute missing financial data, driven by significant advancements in the application of machine learning and deep neural networks for handling missing value imputation.For instance, in [15],the authors proposed machine learning techniques, such as tree-based methods and neural networks,to measure asset risk premiums and recover missing firm fundamentals. Additionally, [8] presented a purity-based k nearest neighbor algorithm, enhancing the performance of missing value imputation and its application in financial distress prediction.Moreover, in [31,33],the authors employed tensor-based factorization for filling in missing data in analyst earning forecasts. These innovative approaches demonstrate the promising potential of advanced techniques in improving the accuracy and reliability of missing data imputation in the financial domain.

Tensor factorization and completion algorithms are mostly based on CANDECOMP/PARAFAC (CP) and Tucker factorization. In lowrank tensor completion, the goal is to impute the missing entries of a tensor such that the rank of the reconstructed tensor is as lowas possible while keeping the difference between the observed entries and their reconstructed versions small.This problem is not convex due to the non-convexity of the rank function.[13,23,29] propose different convex relaxations of the problem and methods to solve the relaxed problems.For factorization-based completion algorithms,it is costly to perform update operations on the factors. To alleviate this issue,[27] uses parallelization and [1,2,9,25] propose methods which perform row-wise factor updates by utilizing stochastic gradient descent.We also benefit from this strategy in our NMTucker model. The models mentioned to this point focus on capturing multilinear relationships between variables,while NMTucker is designed to capture nonlinear patterns in the data.

Recent work has shown the effectiveness of neural network based algorithms for tensor/matrix completion,factorization,and classification.[1o] integrate Tucker factorization and neural networks for image feature extraction and classification.[11] performs matrix factorization by replacing the linear dot product operation between two vector inputs with a multi-layer perceptron (MLP).[21] uses MLP and Bayesian updating,[34] uses LSTMand MLP together to do tensor factorization.Without proper regularization,MLP based methods tend to overfit sparse training data.[22] proposes a tensor completion model which uses a convolutional neural network (CNN) to enable a parameter sharing scheme to reduce overfitting. In this paper, we present our network based NMTucker and show its efficacy in sparse settings due to its unique regularization strategy based on a layered Tucker decomposition.

# 3PRELIMINARIES

We first introduce the common symbols used by NMTucker in Table 1.Tucker factorization essentially is a higher-order principal component analysis.For a third-order tensor $\boldsymbol { \chi } ^ { \prime } \in \mathbb { R } ^ { I _ { 1 } \times I _ { 2 } \times \hat { I _ { 3 } } }$ ,Tucker factorization decomposes the tensor $\chi$ into a core tensor $\mathcal { G } ^ { \mathrm { ~ \scriptsize ~ \in ~ } }$ $\mathbb { R } ^ { R _ { 1 } \times R _ { 2 } \times R _ { 3 } }$ and three factor matrices $U \in \mathbb { R } ^ { I _ { 1 } \times R _ { 1 } }$ ， $V ~ \in ~ \mathbb { R } ^ { I _ { 2 } \times R _ { 2 } }$ $W \in \mathbb { R } ^ { I _ { 3 } \times R _ { 3 } }$ $R _ { 1 } , R _ { 2 } , R _ { 3 }$ are the corresponding ranks for the three factor matrices.Tucker factorization decomposes the third-order tensor $\chi$ as follows:

$$
\pmb { \chi } \approx \pmb { \mathcal { G } } \times _ { 1 } U \times _ { 2 } V \times _ { 3 } W = [ [ \pmb { \mathcal { G } } ; U , V , W ] ]
$$

Table 1: Descriptions for commonly used symbols.   

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td>f</td><td>Tucker operator</td></tr><tr><td>Fo</td><td>nonlinear Tucker operator</td></tr><tr><td>f</td><td>element-wise nonlinear Tucker operator</td></tr><tr><td></td><td>Frobenius norm of tensor X</td></tr><tr><td>Ω</td><td>the indices i(1),·,i(@) of X&#x27;s valid entries</td></tr><tr><td>P(X)</td><td>X with zero entries except for locations i∈Ω</td></tr><tr><td></td><td>X ∈ RIXI2×…×IN order-N tensor with shapeI1 × I2 × . × IN</td></tr><tr><td>Xii2….iN</td><td>tensor entry of X at index (i1,i2,·.,iN)</td></tr><tr><td>X(n)</td><td>mode-n unfolding (matricization) of tensor X</td></tr><tr><td>U(@),v@),w()</td><td>factor matrices in network layer l</td></tr><tr><td>Ui:</td><td>i-th row of matrix U</td></tr><tr><td>G(1)</td><td>core tensor of layer l</td></tr><tr><td>R</td><td>rank of the i-th factor matrix of layer l</td></tr></table>

where $\times _ { i }$ indicates tensor $i$ -mode product by multiplying a tensor bya matrix (or a vector) in mode i.A full treatment of Tucker decomposition and tensor mode product appears in [19].

For all observed entries in $\chi$ and their associated indices, we define the index set $\Omega \ = \ \{ ( i _ { 1 } , i _ { 2 } , i _ { 3 } ) | x _ { i _ { 1 } i _ { 2 } i _ { 3 } }$ is obserued}.Tucker completion algorithm tries to find the factor matrices $U , V , W$ and the core tensor $\mathcal { G }$ such that the following objective function is minimized:

$$
\operatorname* { m i n } { \left\| { P _ { \Omega } ( \hat { X } - X ) } \right\| _ { F } ^ { 2 } } , \ w h e r e \ \hat { X } = [ [ \mathscr { G } ; U , V , W ] ] ,
$$

where the operator $P _ { \Omega }$ returns a tensor that has the same size as its input tensor and is defined as:

$$
( P _ { \Omega } ( X ) ) _ { i _ { 1 } i _ { 2 } i _ { 3 } } = { \left\{ \begin{array} { l l } { x _ { i _ { 1 } i _ { 2 } i _ { 3 } } , } & { { \mathrm { i f ~ } } ( i _ { 1 } , i _ { 2 } , i _ { 3 } ) \in \Omega } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

Problem Definition.For any given $x _ { i _ { 1 } , i _ { 2 } , i _ { 3 } }$ ,the element-wise Tucker completion defines a non-linear function $\begin{array} { r l } { x _ { i _ { 1 } , i _ { 2 } , i _ { 3 } } } & { { } = } \end{array}$ $f ( i _ { 1 } , i _ { 2 } , i _ { 3 } )$ on an $N _ { ☉ }$ -way data cube (for this example, $N = 3$ ).Here $\mathbf { i } = \left( i _ { 1 } , i _ { 2 } , i _ { 3 } \right)$ is an index into factor matrices $U , V , W$ . We only observe the values of a fraction of the tensor during training and attempt to predict the unknown entries.Many regression and classification problems fall into this problem definition and use the learned embeddings of three entities $U _ { i _ { 1 } ; } , V _ { i _ { 2 } ; } , W _ { i _ { 3 } }$ : to predict their interaction result $\hat { x } _ { i _ { 1 } i _ { 2 } i _ { 3 } }$ in Figure 1 as follows:

$$
\begin{array} { l } { { \hat { x } _ { i _ { 1 } i _ { 2 } i _ { 3 } } = f ( i _ { 1 } i _ { 2 } i _ { 3 } ) = \displaystyle \sum _ { r _ { 1 } = 1 } ^ { R _ { 1 } } \sum _ { r _ { 2 } = 1 } ^ { R _ { 2 } } \sum _ { r _ { 3 } = 1 } ^ { R _ { 3 } } g _ { r _ { 1 } r _ { 2 } r _ { 3 } } u _ { i _ { 1 } r _ { 1 } } v _ { i _ { 2 } r _ { 2 } } w _ { i _ { 3 } r _ { 3 } } } } \\ { { = \mathcal { G } \times _ { 1 } U _ { i _ { 1 } , : } \times _ { 2 } V _ { i _ { 2 } , : } \times _ { 3 } W _ { i _ { 3 } ; } , } } \end{array}
$$

Eqn 4 naturally decomposes the full-panel data imputation into element-wise reconstructions,enabling us to leverage SGD and neural networks to learn reliable embeddings and their interactions.

# 4METHODOLOGY

In the section,we first introduce our one layer non-linear NMTucker network model and extend NMTucker to multiple layers in Figure 1.We leverage deep neural networks (DNN) to design an elegant solution for recursive Tucker reconstruction, where the tensor cores in the higher layers depend on the tensor cores in the lower layers and apply stochastic gradient descent algorithm to learn the embedding matrices from a batch of training samples.

# 4.1Network Architecture for Non-linear Tucker Completion

We choose a third-order tensor $\chi$ of shape $( I _ { 1 } , I _ { 2 } , I _ { 3 } )$ to represent the output layer of NMTucker. The key idea of NMTucker is nonlinear Tucker completion by adding non-linear activation functions $\sigma$ into the standard element-wise Tucker completion in Equation 4. Our NMTucker algorithm implements the following equation:

$$
\begin{array} { r l } & { \hat { x } _ { i _ { 1 } i _ { 2 } i _ { 3 } } = f ( i _ { 1 } , i _ { 2 } , i _ { 3 } ) } \\ & { \qquad = \pmb { \mathscr { G } } ^ { ( L ) } \times _ { 1 } ^ { \sigma } U _ { i _ { 1 } ; } ^ { ( L ) } \times _ { 2 } ^ { \sigma } V _ { i _ { 2 } ; } ^ { ( L ) } \times _ { 3 } ^ { \sigma } W _ { i _ { 3 } ; } ^ { ( L ) } } \\ & { \qquad = \sigma ( \sigma ( \sigma ( \pmb { \mathscr { G } } ^ { ( L ) } \times _ { 1 } U _ { i _ { 1 } , } ^ { ( L ) } ) \times _ { 2 } V _ { i _ { 2 } ; } ^ { ( L ) } ) \times _ { 3 } W _ { i _ { 3 } ; } ^ { ( L ) } ) } \\ & { \qquad = [ [ \pmb { \mathscr { G } } ^ { ( L ) } ; U _ { i _ { 1 } ; } ^ { ( L ) } , V _ { i _ { 2 } ; } ^ { ( L ) } , W _ { i _ { 3 } ; } ^ { ( L ) } ] ] ^ { \sigma } } \end{array}
$$

where $U _ { i _ { 1 } : } ^ { ( L ) }$ denotes the $i _ { 1 }$ -th row ofthe factor matrix $U$ of output layer L and 1 ≤ i1 ≤ I.The same notation is applied to V(L) and w(L). $\times _ { i } ^ { \sigma }$ indicates a tensor $i$ mode product followed by an activation function $\sigma$ ,for $i \in \{ 1 , 2 , 3 \}$ .We describe the architecture in Figure 2 for learning the embedding matrices and the core tensor $\mathcal { G }$ as follows.

The modified Tucker completion consists of an embedding module and a non-linear tucker multiplication module.The embedding module takes the input index of a tensor entry and generates embedding vectors.The Tucker multiplication module implements the multiplications between embedding vectors and each matricization of a tensor.Here one matricization essentially defines a fully-connected neural network layer (FC) with neuron weights being matrix elements.For a third-order tensor, the output layer needs three embedding modules and three multiplication modules. Figure 2 shows how to chain three embedding modules and three fully connected network for Tucker completion. Given a training input-output pair, $( i _ { 1 } , i _ { 2 } , i _ { 3 } ; x _ { i _ { 1 } i _ { 2 } i _ { 3 } } )$ ,the input variables cascade down through the three networks and are turned into a reconstructed tensor entry $\widehat { x } _ { i _ { 1 } , i _ { 2 } , i _ { 3 } }$ in Equation 5.

Embedding module In the three embedding modules,we frst randomly initialize the weights of three factor matrices at the beginning of training, namely U(L）∈ RlR), $V ^ { ( L ) } \ \in \ \mathbb { R } ^ { I _ { 2 } \times R _ { 2 } ^ { ( L ) } }$ W(L）∈R5xRL, ,each of which corresponds to one tensor order. For a tensor entry $x _ { i _ { 1 } i _ { 2 } i _ { 3 } } \in X$ located at the coordinate $( i _ { 1 } , i _ { 2 } , i _ { 3 } )$ we extract three embedding vectors $U _ { i _ { 1 } , : } ^ { ( L ) } , V _ { i _ { 2 } , : } ^ { ( L ) }$ and W(） for the indices $i _ { 1 }$ ,i2,i3,respectively.

Nonlinear Tucker multiplication module We use three fullyconnected layers in Figure 2 to implement the nonlinear tucker multiplication module that multiplies one mode unfolding of the corresponding tensor $\mathcal { G }$ (in red, green,or purple in Figure 2)and the feature vectors U(L), $V _ { i _ { 2 } , : } ^ { ( L ) }$ and $\overset { - } { W } _ { i _ { 3 } , : } ^ { ( L ) }$ . The multiplication cascades down the three fully-connected network layers (Algorithm1, lines 10 -13).The first FC layer calculates the 1-mode matricization of the core tensor G(L) (inred inFigure 2)and feature vectorU(L) (line 12) and generates outputs to each neuron in the output layer. The nonlinear Tucker multiplication module first initializes the weights of the first FC network with the mode-1 unfolding of $\mathcal { G } ^ { ( L ) }$ （204 with shape $( R _ { 1 } ^ { ( L ) } , R _ { 2 } ^ { ( L ) } \times R _ { 3 } ^ { ( L ) } )$ xR(） (line 11). Then, the first FC network takes the input vector U(L） and uses the weight matrix to map it to the output vector.

$( 1 , R _ { 2 } ^ { ( L ) } , R _ { 3 } ^ { ( L ) } )$ ure 2). It willater be rearranged as a matrix of shape $( R _ { 2 } ^ { ( L ) } , R _ { 3 } ^ { ( L ) } \times 1 )$ to be used as the weight matrix of the second FC network (line 11). Similarly,we initialize the weight matrixof third FC network and produce the tensor output.Note that we perform the element-wise reconstruction.The shape of tensor shrinks by one after each FC network,as shown in Figure 2.Finally, the output of the non-linear tucker multiplication module is a scalar $\widehat { x } _ { i _ { 1 } , i _ { 2 } , i _ { 3 } }$ defined by Eqn. 5.

# 4.2Implicit Regularization with Multiple Layer Recursive NMTucker

In this section, we will introduce a novel implicit regularization method to reconstruct the core tensor $\mathcal { G } ^ { ( L ) }$ in the output layer from the previous $L - 1$ hidden layers.The main idea of our regularization method is to recursively decompose tucker cores into smaller ones,reducing the number of redundant trainable parameters,and regularizing the complexity.

Let $\chi \in \bar { \mathbb { R } } ^ { I _ { 1 } \times I _ { 2 } \times I _ { 3 } }$ denote a third-order tensor we want to complete. Each layer $l$ $\left( 1 \leq l < L \right)$ in the NMTucker network defines a non-linear neural Tucker operator ${ \mathcal { F } } ^ { \sigma }$ which is an inverse function of Tucker decomposition mapping core tensor $\mathcal { G } ^ { ( l ) }$ to $\mathcal { G } ^ { ( l + 1 ) }$ in Equation 6.

$$
\begin{array} { r l } & { \mathcal { G } ^ { ( l + 1 ) } = \mathcal { F } ^ { \sigma } ( \mathcal { G } ^ { ( l ) } , { U } ^ { ( l ) } , { V } ^ { ( l ) } , { W } ^ { ( l ) } ) } \\ & { \quad \quad \quad = \mathcal { G } ^ { ( l ) } \times _ { 1 } ^ { \sigma } { U } ^ { ( l ) } \times _ { 2 } ^ { \sigma } { V } ^ { ( l ) } \times _ { 3 } ^ { \sigma } { W } ^ { ( l ) } } \\ & { \quad \quad = [ [ \mathcal { G } ^ { ( l ) } ; { U } ^ { ( l ) } , { V } ^ { ( l ) } , { W } ^ { ( l ) } ] ] ^ { \sigma } } \end{array}
$$

Given a tensor entry 9i,iz,i3 $g _ { i _ { 1 } , i _ { 2 } , i _ { 3 } } ^ { ( l + 1 ) } \ \in \ { \mathcal G } ^ { ( l + 1 ) }$ located at the coordinate $( i _ { 1 } , i _ { 2 } , i _ { 3 } )$ , the element-wise reconstruction is as follows:

$$
g _ { i _ { 1 } , i _ { 2 } , i _ { 3 } } ^ { ( l + 1 ) } = \mathcal { G } ^ { ( l ) } \times _ { 1 } ^ { \sigma } U _ { i _ { 1 } , : } ^ { ( l ) } \times _ { 2 } ^ { \sigma } V _ { i _ { 2 } , : } ^ { ( l ) } \times _ { 3 } ^ { \sigma } W _ { i _ { 3 } , : } ^ { ( l ) }
$$

Each element in the core tensor $\mathcal { G } ^ { ( l + 1 ) }$ can be reconstructed from the lower layer based on Eqn 7.Therefore in the multi-layer NMTucker, all $L$ layers have the same network architecture.Core tensors $\{ \boldsymbol { G } ^ { ( l ) } | 2 \leq \dot { l } \leq L \}$ are not trainable parameters. They are computed during forward pass,effectively reducing model complexity.

# 4.3 Parameter Efficiency of $L$ -layer NMTucker

Recursive decomposition of core tensors leads to a parameterefficient Tucker model.When we adda new layer into an NMTucker model,we use new trainable factor matrices and a smaller core tensor to represent/replace the bigger core tensor in the upper layer. There will be less trainable parameters than before because the model saves itself from training a large core tensor by training a smaller core and some factor matrices instead. The advantage of NMTucker is that it uses this to improve its generalization capability.For the architecture in Figure 1, the number of trainable parameters of $L$ -layerNMTucker for $N$ -th order tensor is calculated as follows:

![](images/6e83bdde5c0c98ee3b069e8e927c93e88a25e3784dcd01b9f3ddc20e8c49381d.jpg)  
Figure 2: Non-linear Tucker Completion Neural network for element-wise reconstruction on $\boldsymbol { X } \in \mathbb { R } ^ { I _ { 1 } \times I _ { 2 } \times I _ { 3 } }$ . The inputs to the network form an indexing vector of index $( i _ { 1 } , i _ { 2 } , i _ { 3 } )$ to the embedding matrices $U ^ { ( L ) }$ $V ^ { ( L ) }$ and $W ^ { ( L ) }$ . The comma-separated values in parentheses represent the shape of a tensor or its unfoldings.

$$
\begin{array} { r } { \underbrace { R _ { 1 } ^ { ( 1 ) } \times R _ { 2 } ^ { ( 1 ) } \times R _ { 3 } ^ { ( 1 ) } } _ { G ^ { ( 1 ) } } + \underbrace { R _ { 1 } ^ { ( 2 ) } \times R _ { 1 } ^ { ( 1 ) } } _ { U ^ { ( 1 ) } } + \underbrace { R _ { 2 } ^ { ( 2 ) } \times R _ { 2 } ^ { ( 1 ) } } _ { V ^ { ( 1 ) } } + \underbrace { R _ { 3 } ^ { ( 2 ) } \times R _ { 3 } ^ { ( 1 ) } } _ { W ^ { ( 1 ) } } } \\ { + \underbrace { R _ { 1 } ^ { ( 3 ) } \times R _ { 1 } ^ { ( 2 ) } } _ { U ^ { ( 2 ) } } + \underbrace { R _ { 2 } ^ { ( 3 ) } \times R _ { 2 } ^ { ( 2 ) } } _ { V ^ { ( 2 ) } } + \underbrace { R _ { 3 } ^ { ( 3 ) } \times R _ { 3 } ^ { ( 2 ) } } _ { W ^ { ( 2 ) } } } \\ { + \underbrace { I _ { 1 } \times R _ { 1 } ^ { ( 3 ) } } _ { U ^ { ( 3 ) } } + \underbrace { I _ { 2 } \times R _ { 2 } ^ { ( 3 ) } } _ { V ^ { ( 3 ) } } + \underbrace { I _ { 3 } \times R _ { 3 } ^ { ( 3 ) } } _ { W ^ { ( 3 ) } } } \end{array}
$$

Note that only the factor matrices and the core tensor in the first layer, ${ \mathcal { G } } ^ { ( 1 ) }$ , contain trainable weights.Let $\pmb { \chi } \in \mathbb { R } ^ { I _ { 1 } \times \cdots \times I _ { N } }$ denote the original $_ \mathrm { N }$ order tensor.For an $L$ -layerNMTucker model where the output of the $L$ -th layer is the reconstructed tensor $\hat { X }$ let $R _ { i } ^ { ( L + 1 ) } = I _ { i }$ for each $i = 1 , \cdots , N$ for compactness.Then we use the following general equation (Eqn. 9) to calculate the number of trainable parameters inamulti-layerNMTucker model.

$$
\prod _ { i = 1 } ^ { N } R _ { i } ^ { ( 1 ) } + \sum _ { j = 1 } ^ { L } \sum _ { i = 1 } ^ { N } R _ { i } ^ { ( j + 1 ) } R _ { i } ^ { ( j ) }
$$

# 5EXPERIMENTS

In this section,we conduct experiments to validate the performance of our model. Specifically,we verify that NMTucker shows improvement over linear Tucker Decomposition models,has the generalization capacity to overcome the overfitting problem and

Input :Tensor $\boldsymbol { X } \in \mathbb { R } ^ { I _ { 1 } \times I _ { 2 } \times I _ { 3 } }$ to be completed Core tensor dimensionalites R𝑖) for $( l = 1 , \cdots , L )$ ， $( i = 1 , 2 , 3$ index set of obseryed entries $\Omega$ in Tensor $\chi$ Output:Factor Matrices $A _ { i } ^ { ( l ) }$ $( l = 1 , \cdots , L )$ ， $( i = 1 , 2 , 3 )$ and core tensor G(1) 1 Inializefactor matricesAl) $A _ { i } ^ { ( l ) } \in \mathbb { R } ^ { R _ { i } ^ { ( l + 1 ) } \times R _ { i } ^ { ( l ) } }$ J for $( l = 1 , \cdots , L ) , ( i = 1 , 2 , 3 )$ and ${ \mathcal { G } } ^ { ( 1 ) }$ where $R _ { i } ^ { ( L + 1 ) } = I _ { i }$ ； 2 repeat 3 for $\alpha = \forall ( i _ { 1 } , i _ { 2 } , i _ { 3 } ) \in \Omega$ do 4 for $l \gets 1$ to $L - 1$ do 5 $\Upsilon  g ^ { ( l ) }$ 6 for $n \gets 1$ to 3 do // equation 6 7 [ Y ←σ(Y xn A) 8 G(l+1) ←Y; // update next layer's core 9 Y←g(L) 10 for $n \gets 1$ to 3 do 11 $\mathsf { Y } \gets \mathsf { Y } _ { ( n ) }$ ； // matricization 12 Y ←g(A()Y) 13 $\mathsf { Y } \gets \mathsf { F o l d } ( Y )$ //fold vector to tensor 14 $\hat { x } _ { i _ { 1 } i _ { 2 } i _ { 3 } } \gets \Upsilon$ 15 calculate loss $\mathcal { L }$ w.r.t $\frac { 1 } { 2 } ( x _ { i _ { 1 } i _ { 2 } i _ { 3 } } - \hat { x } _ { i _ { 1 } i _ { 2 } i _ { 3 } } ) ^ { 2 }$ 16 Update all $A ^ { ( l _ { i } ) }$ and $\mathcal { G } ^ { ( 1 ) }$ based on $\mathcal { L }$

17 until maximum number of epochs or early stopping;

scales to the large,sparse and high-dimensional datasets with stable performance and efficiency.

Table 2: Summary of the datasets   

<table><tr><td>Dataset</td><td>Shape</td><td>Observed</td></tr><tr><td>Firm fundamentals</td><td>(32,180,19)</td><td>99.13%</td></tr><tr><td>Firm characteristics</td><td>(60,100,202)</td><td>62.60%</td></tr><tr><td>EPS Forecast</td><td>(32, 180, 173)</td><td>4.30%</td></tr></table>

# 5.1 Analysis of Real Data

We evaluate the generalization capabilityof NMTucker and other methods using real-world datasets.We construct three NMTucker models with different numbers of layers: NMTucker1,NMTucker, and NMTucker3,which have 1,2 and 3 layers respectively. To compare our NMTucker with a basic regularization method, we construct NMTucker-L1 by adding L1 regularization on the core tensor $\mathcal { G }$ of our one layer NMTucker1 model.We choose three real-world finance datasets in our experiments: firm accounting fundamentals,financial analyst EPS (earnings per share) forecasts, and firm characteristics.Below,we provide the descriptions for these datasets.

Firm Accounting fundamentals [quater, firm, fundamental]. We select 19 accounting variables following established literature [4,12,17,31]. Quarterly accounting fundamental variables are collected from COMPUSTAT and CRSP.Foranalysts'EPS forecast and firm fundamental datasets,we consider 180 firms for the period from Q1-2010 to Q4-2017.Firm Characteristics [month,firm, characteristic].We use the open-source cross-sectional asset pricing data from [7].The monthly data from December 1925 to June 2022 contains 2O2 predictor variables that are identified in the literature as proven predictors.We sample 1oo random firms from this dataset and use the last 5 years of data.Financial Analyst EPS Forecast [quater, firm,analyst].The quarterly financial analysts'EPS forecast data are collected from Thomson Reuters I/B/E/S and are available at https://wrds-www.wharton.upenn.edu/. This data is highly sparse,with almost $9 6 \%$ missing values. To make our analysis more consistent and meaningful,we remove analysts who have been in business for less than four years or made less than 200 predictions in their entire careers.All datasets were normalized with min-max normalization across the time dimension, i.e., the features are normalized within each firm across time.Table 2 summarizes the properties of the three datasets.

# Baseline methods

We evaluate our NMTucker1,NMTucker2,NMtucker3 and NMTucker-L1 models and compare their performance with other state-of-the-art models designed for sparse tensor completion.The baseline models include the linear model P-Tucker [27] and the neural network based nonlinear CoSTCo [22].

Evaluation We measure the performance using three evaluation metrics,namely,root mean squared error (RMSE),mean absolute error (MAE) and mean absolute percentage error (MAPE).For MAPE, wea $\begin{array} { r } { M A P E = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { \left| y _ { i } - \hat { y _ { i } } \right| } { m a x ( | y _ { i } | , \delta ) } } \end{array}$ $\delta = 0 . 1$

To test the model performance on all datasets,we randomly sample $8 0 \%$ of the non-missing entries as the training set, and the remaining $2 0 \%$ as the test set. For all datasets,we randomly select $1 0 \%$ of the training set as the validation set and we report the average results over 10 different splits.Details of allhyperparameter tuning and our model's implementation are provided at Appendix A.1.

Each model has a different architecture,so a fair comparison should be based on the number of parameters rather than rank.So, we ensure the number of trainable parameters are uniform across all models for each experiment.In tables 3,4,5 we present the approximate number of parameters used in the experiments,where krepresents a thousand.Ranks column in the tables indicates the rank of the factor matrices in each layer of the model.For example, (9,10) means that the ranks in the first and second layer (output layer) are 9 and 10,respectively.Note that we use uniform ranks for all dimensions within an NMTucker layer.

Result Table 3 shows the tensor completion results on the Firm Fundamentals dataset.The performance of our NMTucker models is better overall than all the baselines in all experiments with one exception. Specifically,NMTucker2 exhibits $1 . 1 2 \%$ $1 2 . 1 3 \%$ less RMSE than P-Tucker and $4 . 2 3 \% - 1 8 . 9 8 \%$ less RMSE than CosTCo across different model sizes.For the smaller 3k models,we observe that NMTucker1 has the best overall performance.This is potentially due to heavy regularization not being necessary with a small number of parameters,so adding more decomposition layers cannot improve the performance of NMTucker1.For the larger 13k and 75k sized models,we observe thatNMTucker-L1is better than NMTucker1 in all metrics,showing that L1 regularization on the core tensor can be useful for larger models.Additionally,NMTucker2 has a significantly better performance than NMTucker-L1, showing the effectiveness of our decomposition based regularization strategy compared to basic regularization.Finally, the comparison between the linear P-Tucker model and NMTucker1 indicates that a nonlinear transfer function in Tucker decomposition can introduce flexibility for better fitting.

Table 5 shows the results for the EPS Forecast dataset.The results are similar to the the Firm Fundamentals dataset,except that P-Tucker performs better than NMTucker1 for 5k models.Additionally,we observe that for NMTucker2 with16k parameters is able to outperform NMTucker1 with 81k parameters,showing that is even possible to achieve better performance with less parameters using the recursive decomposition strategy.

Table 4 shows the results for the Firm Characteristics dataset. NMTucker2 consistently outperforms all baselines in all metrics bya significant margin,with improvements over the baselines being much larger compared to other datasets.This experiment has models with larger number of parameters and these results indicate that NMTucker's advantages may be more prominent for larger models.Furthermore,NMTucker2 demonstrates remarkable results, achievinga substantial $2 8 \%$ to $4 2 \%$ reduction in RMSE compared to the linear method P-Tucker.This improved performance can be attributed to NMTucker2's capability to introduce non-linearity into the model.Firm characteristics data often exhibits non-linear relationships due to the complex nature of the underlying factors that influence these characteristics.Recognizing and accounting for the non-linear relationships in firmcharacteristics data is crucial for accurate missing value imputation.

Overall, we observe that NMTucker1 outperforms the nonlinear baseline CoSTCo, suggesting that Tucker decomposition may be more effective compared to CP decomposition for use with nonlinear methods.In addition,our three layer models do not bring much improvement on these three datasets,indicating that adding redundant layers of decomposition can be too restrictive and hurt performance.

Table 3: Model performance comparison for Firm Accounting Fundamental   

<table><tr><td>Metric</td><td colspan="3">RMSE</td><td colspan="3">MAE</td><td colspan="3">MAPE</td><td colspan="3">Ranks</td></tr><tr><td>Model/parameters (~)</td><td>3k</td><td>13k</td><td>75k</td><td>5k</td><td>13k</td><td>75k</td><td>3k</td><td>13k</td><td>75k</td><td>3k</td><td>13k</td><td>75k</td></tr><tr><td>P-Tucker</td><td>0.2219</td><td>0.1965</td><td>0.1857</td><td>0.1666</td><td>0.1421</td><td>0.1287</td><td>68.66</td><td>58.67</td><td>52.90</td><td>10</td><td>20</td><td>40</td></tr><tr><td>CoSTCo</td><td>0.2288</td><td>0.2080</td><td>0.2044</td><td>0.1664</td><td>0.1470</td><td>0.1425</td><td>62.98</td><td>55.24</td><td>52.94</td><td>10</td><td>20</td><td>40</td></tr><tr><td>NMTucker1</td><td>0.2114</td><td>0.1859</td><td>0.1727</td><td>0.1576</td><td>0.1353</td><td>0.1228</td><td>65.14</td><td>55.44</td><td>50.40</td><td>10</td><td>20</td><td>40</td></tr><tr><td>NMTucker2</td><td>0.2194</td><td>0.18010.1656</td><td></td><td>0.1645</td><td>0.1291</td><td>0.1154</td><td>68.18</td><td>53.07</td><td>47.09</td><td>（9,10）</td><td>(18,25)</td><td>(36,80)</td></tr><tr><td>NMTucker3</td><td>0.2419</td><td>0.2230</td><td>0.2493</td><td>0.1875</td><td>0.1662</td><td>0.1934</td><td>79.37</td><td>69.01</td><td>82.46</td><td></td><td>(8,9,10)(162025)(35,5070)</td><td></td></tr><tr><td>NMTucker-L1</td><td>0.2192</td><td>0.1854</td><td>0.1666</td><td>0.1650</td><td>0.1345</td><td>0.1179</td><td>68.31</td><td>55.20</td><td>48.45</td><td>10</td><td>20</td><td>40</td></tr></table>

Table 4: Model performance comparison for Firm Characteristics   

<table><tr><td>Metric</td><td colspan="2">RMSE</td><td colspan="2">MAE</td><td colspan="2">MAPE</td><td colspan="2">Ranks</td></tr><tr><td>Model/parameters (~)</td><td>80k</td><td>550k</td><td>80k</td><td>550k</td><td>80k</td><td>550k</td><td>80k</td><td>550k</td></tr><tr><td>P-Tucker</td><td>0.2736</td><td>0.2689</td><td>0.1956</td><td>0.1717</td><td></td><td>83.9072.67</td><td>40</td><td>80</td></tr><tr><td>CoSTCo</td><td>0.2605</td><td>0.2401</td><td>0.1992</td><td>0.1789</td><td>82.20</td><td>71.02</td><td>40</td><td>80</td></tr><tr><td>NMTucker1</td><td>0.2207</td><td>0.1747</td><td>0.1662</td><td>0.127168.7851.13</td><td></td><td></td><td>40</td><td>80</td></tr><tr><td>NMTucker2</td><td></td><td></td><td>0.19490.15600.14240.1041</td><td></td><td></td><td>58.541.46</td><td>(36,80）</td><td>(78,150)</td></tr><tr><td>NMTucker3</td><td>0.2323</td><td>0.2060</td><td>0.1765</td><td>0.151473.8362.29</td><td></td><td></td><td></td><td>(30,70,80)(75,120,150)</td></tr><tr><td>NMTucker-L1</td><td>0.2260</td><td>0.1798</td><td>0.1712</td><td>0.132271.3653.41</td><td></td><td></td><td>40</td><td>80</td></tr></table>

Table 5: Model performance comparison for EPs Forecasts   

<table><tr><td>Metric</td><td colspan="3">RMSE</td><td colspan="3">MAE</td><td colspan="3">MAPE</td><td colspan="3">Ranks</td></tr><tr><td>Model/parameters (~)</td><td>5k</td><td>16k</td><td>81k</td><td>5k</td><td>16k</td><td>81k</td><td>5k</td><td>16k</td><td>81k</td><td>5k</td><td>16k</td><td>81k</td></tr><tr><td>P-Tucker</td><td>0.1306</td><td>0.1185</td><td>0.1313</td><td>0.0889</td><td>0.0798</td><td>0.0852</td><td>31.03</td><td>27.52</td><td>28.93</td><td>10</td><td>20</td><td>40</td></tr><tr><td>CoSTCo</td><td>0.1407</td><td>0.1368</td><td>0.1278</td><td>0.0982</td><td>0.0937</td><td>0.0871</td><td>34.13</td><td>31.76</td><td>29.17</td><td>10</td><td>20</td><td>40</td></tr><tr><td>NMTucker1</td><td>0.1354</td><td>0.1248</td><td>0.1259</td><td>0.0930</td><td>0.0839</td><td>0.0861</td><td>31.74</td><td>28.19</td><td>29.39</td><td>10</td><td>20</td><td>40</td></tr><tr><td>NMTucker2</td><td>0.2089</td><td>0.11810.1194</td><td></td><td>0.1586</td><td></td><td>0.07880.0787</td><td></td><td></td><td>59.7826.6227.12</td><td>（9,10)</td><td>(18,24)</td><td>(35.80)</td></tr><tr><td>NMTucker3</td><td>0.1615</td><td>0.1654</td><td>0.1815</td><td>0.1164</td><td>0.1168</td><td>0.1303</td><td>41.11</td><td>41.27</td><td>46.00</td><td></td><td>(8,9,10)(16,20,24)</td><td>(36,40,60)</td></tr><tr><td>NMTucker-L1</td><td>0.1941</td><td>0.1408</td><td>0.1178</td><td>0.1468</td><td>0.0984</td><td>0.0789</td><td>57.26</td><td>35.11</td><td>27.59</td><td>10</td><td>20</td><td>40</td></tr></table>

# 5.2Additional Experiment for Theoretical Validation

To better understand the effect of the recursive decomposition strategy, we compare the training and validation curves of NMTucker1 and NMTucker2 from the Firm Characteristics experiment.Both models use the same number of parameters $( \approx 5 5 0 k )$ for consistency. In Figure 3,the plots on the left and right show the training and validation curves for NMTucker1 (blue) and NMTucker2 (orange), respectively.NMTucker2,which has two decomposition layers,is more robust to overfitting with a much smaller gap between the training and the validation loss than the one layered NMTucker1.

![](images/38c22e0f9833d999d5edcbd11fa4e2631ab9fbd91a7bfb8054cffeb7ef639211.jpg)  
Figure 3: Training and Validation MSE v.s.epochs on the Firm Characteristicsdataset

We hypothesize that the multi-layer strategy can have a regularizing effect because it reduces the freedom of the core tensor in the next layer.

# 6CONCLUSION

We propose NMTucker, a novel Tucker completion algorithm, that extracts the non-linear patterns from complex data distributions while avoiding the overfitting problem with a recursive decomposition strategy. NMTucker also utilizes SGD during training to avoid the full-scale tensor decomposition and reconstructs a tensor with onlya small number of observed entries.Experiments show that NMTucker reduces imputation error in comparison with the state-of-the-art completion methods.In future work,we aim to develop a general practice in choosing appropriate tensor ranks in each Matryoshka layer for robustness and confrm its potential in enhancing downstream analysis tasks.

# REFERENCES

[1] Evrim Acar,Daniel M.Dunlavy,Tamara G.Kolda,and Morten Morup.2011. Scalable tensor factorizations for incomplete data. Chemometrics and Intelligent Laboratory Systems 106,1(2011),41-56.Multiway and Multiset Data Analysis.   
[2]Ivana Balazevic,Carl Alen,and Timothy M.Hospedales.2019. TuckER: Tensor Factorization for Knowledge Graph Completion.arXiv e-prints,Article arXiv:1901.09590 (Jan 2019),arXiv:1901.09590 pages.arXiv:1901.09590 Ccs.LG]   
[3] Turan G Bali,Amit Goyal, Dashan Huang,Fuwei Jiang,and Quan Wen. 2021. Different strokes: Return predictability across stocks and bonds with machine learningand big data.Swiss Finance Institute,Research Paper Series 20-11o (2021).   
[4] Ryan TBalland Eric Ghysels.2018.Automated Earnings Forecasts: Beat Analysts or Combine and Conquer? Management Science 64,10 (2018),4936-4952.   
[5]Ercument Cahan, Jushan Bai, and Serena Ng.2023.Factor-based imputation of missing values and covariances in panel data of large dimensions.Journal of Econometrics 233,1(2023),113-131.   
[6] J.Douglas Carroll and Jih-Jie Chang.1970.Analysis of individual differences in multidimensional scaling via an n-way generalization of“Eckart-Young”decomposition. Psychometrika 35 (1970),283-319.   
[7] Andrew Y Chen and Tom Zimmermann. 2021. Open source cross-sectional asset pricing. Critical Finance Review,Forthcoming (2021).   
[8] Ching-Hsue Cheng,Chia-Pang Chan,and Yu-Jheng Sheu. 2019.A novel puritybased k nearest neighbors imputation method and its application in financial distress prediction. Engineering Applications ofArtificial Intelligence 81 (2019), 283-299.   
[9]Dehua Cheng,Richard Peng,Yan Liu,and Ioakeim Perros.2016. SPALS: Fast Alternating Least Squares via Implicit Leverage Scores Sampling.In Advances in Neural Information Processing Systems 29.   
[10] J.Chien and Y.Bao.2018.Tensor-factorized neural networks. IEEE Transactions on Neural Networks and Learning Systems 29,5 (2018),1998-2011.   
[11] Gintare Karolina Dziugaite and Daniel M.Roy. 2015.Neural Network Matrix Factorization. CoRR abs/1511.06443 (2015).arXiv:151106443http://arxiv.org/ abs/1511.06443   
[12]EugeneFFamaand KennethRFrench.2006.Profitabilityinvestmentandaverage returns. Journal offinancial economics 82,3 (2o06),491-518.   
[13] Silvia Gandy,Benjamin Recht,and Isao Yamada.2011. Tensor completion and lown-rank tensor recovery via convex optimization. Inverse Problems 27,2,Article 025010 (Feb.2011),025010 pages.https://doi.org/10.1088/0266-5611/27/2/025010   
[14] Shihao Gu, Bryan Kelly,and Dacheng Xiu.2020.Empirical asset pricing via machine learning. The Review ofFinancial Studies 33,5 (2020),2223-2273.   
[15] Shihao Gu, Bryan Kelly,and Dacheng Xiu.202o.Empirical asset pricing via machine learning. The Review ofFinancial Studies 33,5 (2020),2223-2273.   
[16] R.A. Harshman. 197o. Foundations of the PARAFAC procedure: Models and conditions for an "explanatory" multi-modal factor analysis.UCLA Working Papers in Phonetics 16 (1970),1-84.   
[17]Kewei Hou,Mathijs A Van Dijk,and Yinglei Zhang.2012.The implied cost of capital: A new approach. Journal of Accounting and Economics 53,3 (2012), 504-526.   
[18] Diederik Kingma and Jimmy Ba. 2014. Adam: A Method for Stochastic Optimization.International Conference on Learning Representations (12 2014).   
[19] Tamara G.Kolda and Brett W. Bader. 20o9. Tensor Decompositions and Applications.SIAM Rev.51,3 (2009),455-500．https://doi.0rg/10.1137/07070111X arXiv:https://doi.org/10.1137/07070111X   
[20] Charles MC Lee and Eric CSo.2017. Uncovering expected returns: Information in analyst coverageproxies.Journal ofFinancial Economics124,2 (2017),331-348.   
[21] Bin Liu,Lirong He,YingmingLi, Shandian Zhe,and Zenglin Xu.2018.NeuralCP: Bayesian Multiway Data Analysis with Neural Tensor Decomposition. Cognitive Computation 10 (2018),1051-1061.   
[22] Hanpeng Liu, Yaguang Li, Michael Tsang,and Yan Liu.2019. CoSTCo: A Neural Tensor Completion Model for Sparse Tensors.In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &Data Mining (Anchorage,AK, USA) (KDD ‘19).Association for Computing Machinery,New York, NY,USA,324-334.https://doi.0rg/10.1145/3292500.3330881   
[23] J.Liu,P.Musialski,P.Wonka,andJ.Ye.2013.Tensor Completion for Estimating Missing Values in Visual Data.IEEE Transactions on Patern Analysis and Machine Intelligence 35,1(2013),208-220.   
[24] Dijun Luo,Chris Ding,and Heng Huang. 2011．Are Tensor Decomposition Solutions Unique? On the Global Convergence HOSVD and ParaFac Algorithms. In Advances in Knowledge Discovery and Data Mining,Joshua Zhexue Huang, Longbing Cao,and Jaideep Srivastava (Eds.).148-159.   
[25] Takanori Maehara, Kohei Hayashi,and Ken-ichi Kawarabayashi. 2016.Expected Tensor Decomposition with Stochastic Gradient Descent.In Proceedings of the Thirtieth AAAi Conference on Artificial Intelligence (Phoenix,Arizona) (AAAI'16). AAAI Press,1919-1925.   
[26] Rebecca Morger. 2022.Imputation Algorithms with Principal Component Analysis for Financial Data. (2022).   
[27] Sejoon Oh,Namyong Park, Lee Sael,and U Kang.2017.Scalable Tucker Factorization for Sparse Tensors -Algorithms and Discoveries.CoRR abs/1710.02261 (2017).arXiv:1710.02261 http://arxiv.org/abs/1710.02261   
[28]Adam Paszke and et al.2019.PyTorch: An Imperative Style,High-Performance Deep Learning Library.In Advances in Neural Information Processing Systems 32, H. Wallach and et al. (Eds.).Curran Associates,Inc.,8024-8035.   
[29] Bernardino Romera-Paredes and Massimiliano Pontil. 2013．A New Convex Relaxation for Tensor Completion.In Advances in Neural Information Processing Systems 26.   
[30] Ledyard R. Tucker.1966. Some mathematical notes on three-mode factor analysis. Psychometrika 31,3(01 Sep 1966),279-311. https://doi.org/10.1007/BF02289464   
[31]Ajim Uddin, Xinyuan Tao,Chia-Ching Chou,and Dantong Yu. 202o.Nonlinear Tensor Completion Using Domain Knowledge:An Application in Analysts' Earnings Forecast.In 2020 International Conference onDataMining Workshops (ICDMW).IEEE,377-384.   
[32] Ajim Uddin, Xinyuan Tao,Chia-Ching Chou,and Dantong Yu.2022.Are missing values important for earnings forecasts?A machine learning perspective. Quantitative finance 22,6 (2022),1113-1132.   
[33]Ajim Uddin,Xinyuan Tao,Chia-Ching Chou,and Dantong Yu.2022.Machine Learning for Earnings Prediction: A Nonlinear Tensor Approach for Data Integration and Completion.In Proceedings ofthe Third ACMInternational Conference on AI in Finance.282-290.   
[34] Xian Wu, Baoxu Shi, Yuxiao Dong,Chao Huang,and Nitesh V.Chawla. 2018. Neural Tensor Factorization. CoRR abs/1802.04416 (2018)．arXiv:1802.04416 http://arxiv.org/abs/1802.04416

# ARESEARCHMETHODS

# A.1 Hyper-parameter and Implementation Details

We implement NMTucker in PyTorch [28].For all the experiments, we use the Adam optimizer [18] with the default configuration. We use MSE as the loss function and set early stopping on the validation loss with patience $= 5 0$ .We tune the learning rate from (1e-4,3e-4,1e-3)and batch size from(256,512,1024).For NMTuckerL1,we tune the L1 regularization coefficient on the core tensor weights from(1e-7,1e-6,1e-5,1e-4).For CoSTCo [22],we use the original implementation from the authors.We tune the the learning rate from (1e-4,3e-4,1e-3) and batch size from (256,512,1024) and set patience $= 2 0$ For P-Tucker[27],we also use the original implementation.We tune the L2 regularization parameter $\lambda$ from (1e-7,1e-6,1e-5,1e-4,1e-3,1e-2)and the optional orthogonalization parameter for the factor matrices.