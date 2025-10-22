# CFTNet: a robust credit card fraud detection model enhanced by counterfactual data augmentation

Menglin Kong1 $\cdot$ Ruichen Li1 $\cdot$ Jia Wang2 $\cdot$ Xingquan Li3 • Shengzhong Jin1 $\cdot$ Wanying Xie1 • Muzhou Hou1 • Cong Cao1 $\textcircled{1}$

Received: 9 August 2023 / Accepted: 22 January 2024 / Published online: 26 February 2024   
$©$ The Author(s), under exclusive licence to Springer-Verlag London Ltd., part of Springer Nature 2024

# Abstract

Establishing a reliable credit card fraud detection model has become a primary focus for academia and the financial industry. The existing anti-fraud methods face challenges related to low recall rates, inaccurate results, and insufficient causal modeling ability. This paper proposes a credit card fraud detection model based on counterfactual data enhancement of the triplet network. Firstly, we convert the problem of generating optimal counterfactual explanations (CFs) into a policy optimization of agents in the discrete–continuous mixed action space, thereby ensuring the stable generation of optimal CFs. The triplet network then utilizes the feature similarity and label difference of positive example samples and CFs to enhance the learning of the causal relationship between features and labels. Experimental results demonstrate that the proposed method improves the accuracy and robustness of the credit card fraud detection model, outperforming existing methods. The research outcomes are of significant value for both credit card anti-fraud research and practice while providing a novel approach to causal modeling issues across other fields.

Keywords Credit card fraud detection $\cdot$ Counterfactual data augmentation $\cdot$ Reinforcement learning $\cdot$ Deep neural network

# 1 Introduction

The economy’s growth has led to the widespread adoption of credit cards as a payment method. However, this progress has also resulted in the problem of credit card fraud, where illicit transactions are conducted using falsified or stolen credit card information [1]. Therefore, it is crucial to develop an effective anti-fraud system that can detect and prevent credit card fraud [2]. The anti-fraud field has two distinct features: firstly, the data are often imbalanced [3, 4]; secondly, anti-fraud systems must handle multiple data dimensions. Therefore, anti-fraud systems must be capable of processing high-dimensional, sparse data and accurately identifying fraud-related features [5, 6]. Antifraud applications have specific requirements, including high accuracy, real-time processing, and interpretability [7–10]. Models for detecting fraud can be developed using various approaches, such as statistical models, artificial intelligence (AI) methods, rule-based methods, and risk assessment [11–14].

In recent years, data-driven anti-fraud models have gained significant attention due to their high performance and precision advantages. Despite some achievements, traditional rule-based and data-driven methods employed in credit card fraud detection still have limitations due to the complexity and diversity of the task. The performance of machine learning (ML) or deep learning (DL) models depends on the quality and quantity of training data [15]. However, the scarcity of fraud-positive samples for model training in credit card fraud detection makes it challenging for models to capture the characteristics of fraud samples. Moreover, anti-fraud models are faced with the root cause issue inherent in all classification tasks [16–18]: the functional relationship established is not entirely causal, and spurious correlation modeling can significantly impact classification accuracy. Recent research [19] has highlighted the problem of poor robustness in many DL models caused by models’ functional relationships based on spurious correlation rather than causality. To address this issue, counterfactual explanations (CFs) reflecting causal relationships among variables in data have been proposed to enhance the robustness and interpretability of DL models [20–23].

We argue that CFs can be utilized to mitigate spurious correlation modeling in classification tasks, thereby constructing more robust anti-fraud models. In this paper, we propose a counterfactual triplet neural network model (CFTNet) augmented by CFs to enhance the performance of DL models in credit card fraud detection. Specifically, our approach involves two stages: first, obtaining the optimal CFs of the fraud-positive samples via a Markov decision process (MDP), the reinforcement learning (RL) algorithm is employed for policy optimization; after that, the optimal CFs are introduced to the model training process alongside positive samples to encourage the model to prioritize features with causal relationships with the label, thereby improving the model’s causal modeling ability. Introducing CFs in credit card fraud detection can enhance the model’s ability to capture causal relationships and expand the training samples, providing possibilities for further improving DL model performance. We compare our approach with traditional and state-of-the-art (SOTA) ML/DL-based methods and analyze their advantages and limitations. Detailed ablation experiments also evaluate the proposed model’s parameter sensitivity and causal modeling capabilities. Our contributions are as follows:

• We re-analyze the credit card fraud detection problem from the causal inference perspective and identify the root cause of the poor robustness of traditional ML/DL

classification models across different datasets. Inspired by backdoor adjustment of causal inference, we argue that representation disentanglement is the key to enhancing model performance. We propose a credit card fraud classification framework based on a counterfactual triplet network (CFTNet). Our framework introduces contrastive learning (CL) loss to disentangle the representations of the model in a self-supervised manner. CFTNet can achieve more accurate and robust classification via representation disentanglement. We transform the calculation process of the optimal CFs into an MDP and then resolve an optimal counterfactual generator based on the P-DQN model via the RL algorithm. Our experiments show that the classification performance of CFTNet surpasses that of a series of traditional and SOTA methods, and CFTNet’s performance on datasets with different sample ratios is more robust.

This article is structured as follows: Sect. 2 introduces related work. Then, Sect. 3 provides formulations for credit card fraud detection from the causal inference perspective and defines optimal CFs related to our method. Section 4 describes the details of the proposed CFTNet. Next, Sect. 5 presents the experimental setup and results analysis. Finally, Sect. 6 concludes the whole paper and discusses future work.

# 2 Related work

In this section, we provide an overview of related work in terms of general (supervised) ML techniques for fraud detection and network-based fraud detection. The development of ML and DL-based anti-fraud strategy models has been a popular research direction. Sailusha et al. [24] compared the effectiveness of two popular algorithms, random forest (RF) and AdaBoost, in detecting credit card fraud. The study concluded that RF was more effective than AdaBoost. In addition to methods comparing, some researchers attempted to enhance the classification capability of the model from the perspectives of model ensembling or feature engineering. Bin Sulaiman et al. [25] demonstrated that the artificial neural network (ANN) model based on the federated learning framework achieves higher accuracy. Saia et al. [26] demonstrated through experimentation that the introduction of multidimensional space analysis benefits the model in addressing sample imbalance issues. Salekshahrezaee et al. [27] attempted various preprocessing methods to assist machine learning models in accomplishing fraud detection tasks and demonstrated the superiority of contrastive autoencoder (CAE) and random under-sampling (RUS) techniques.

However, the overall performance of ML algorithms remains unsatisfactory, indicating the need to explore DL algorithms for credit card fraud detection. Harwani et al. [28] developed a novel hybrid model that combines selforganizing mapping (SOM) and artificial neural network (ANN) to achieve higher accuracy than either of the two models alone. Voican et al. [29] employed deep neural networks (DNN) to detect imposter scams, leveraging their ability to understand, judge, and learn the expected behavior of users to detect credit card fraud more accurately. The proposed DNN model achieved a final classification accuracy of $9 9 . 7 6 \%$ . Nguyen et al. [30] conducted a comprehensive study of DL models for credit card fraud detection, comparing their performance with various ML algorithms on three financial datasets. The results demonstrated that DL methods outperformed ML algorithms.

Although existing results are promising, the inherent problem of poor robustness caused by ML/DL modelsfunctional relationships based on spurious correlation is grievous. To address this issue, Van Belle et al. [31] propose a novel network-based credit card fraud detection method based on node representation learning (NRL). Moreover, CFs reflecting causal relationships among variables in data have been employed to enhance the robustness and interpretability of DL models [20–23].

# 3 Preliminary work

In this section, we revisit the credit card fraud detection problem from the perspective of causal graphs of causal inference. We discuss the limitations of existing ML/DL methods for fitting spurious correlations between variables in causal graphs and suggest further improvements. Furthermore, we provide an intuitive illustration and formal definition of the optimal CFs used in our framework.

# 3.1 Analysis of credit card fraud detection from the perspective of causality

The task of detecting credit card fraud requires an antifraud model that can accurately classify positive and negative samples in a dataset characterized by class imbalance, where the number of positive samples is significantly smaller than that of negative samples, and high dimensionality, where multiple features are available. A key challenge in this scenario is identifying the features (independent variables) that have a causal relationship with the label (dependent variable), i.e., whether a transaction is fraudulent or not, from many available features. Previous studies [32–34] have highlighted the importance of this issue. To address this challenge, we propose using causal graphs [35], a powerful tool from the field of causal inference, to analyze the mechanism of variable generation and causal relationships between variables. This paper presents a causal graph tailored explicitly to the credit card fraud detection scenario, as illustrated in Fig. 1.

Figure 1 depicts the credit card fraud detection problem from the perspective of causal graphs. The binary label variable indicating fraudulent activity is represented by $Y _ { : }$ ， while $X$ represents the risk representation capturing critical features for predicting fraud. The confounding variable $Z$ represents features that are irrelevant to predicting fraud. Figure 1a shows that the confounding variable $Z$ influences the label variable $Y$ through two paths: $Z \to Y$ and $Z \to X \to Y$ . This creates a ‘‘backdoor path’’ between $X$ and $Y$ mediated by Z. Conventional ML/DL approaches typically rely on learning prediction functions based on the observed correlation between $X$ and Y. However, such correlations may be spurious and caused by the confounding variable $Z ,$ , leading to suboptimal performance of the learned prediction function on the test set or datasets with different data distributions.

Ensuring that the ML/DL models capture the true causal relationship between $X$ and $Y$ is crucial in improving their robustness. This can be achieved by intervening in the generation mechanism of $X$ using backdoor adjustment techniques from the field of causal inference [36], as illustrated in Fig. 1b. This renders $X$ statistically independent of the confounding variable $Z$ . By incorporating the modified causal graph illustrated in Fig. 1b, we can obtain a probability distribution of $Y$ that accurately models the causal relationship between the variables. The formula for this distribution is as follows:

$$
\begin{array} { l } { P ( Y = y | \mathrm { d o } ( X = x ^ { * } ) , Z ) } \\ { \displaystyle = \sum _ { Z } P ( Y = y | X = x ^ { * } , Z = z ) P ( Z = z | X = x ^ { * } ) , } \end{array}
$$

where $\mathrm { d } \mathbf { o } ( X = x ^ { * } )$ signifies the intervention on the variable $X$ such that it assumes the value $x ^ { * }$ . Given that $Z$ and $X$ are statistically independent in Fig. 1b, Equation (1) can be alternatively expressed as the following Eq. (2):

![](images/a9888f7085921ccb682ab516ba39715c4e3f298e667dfa56c0c8845a22d7e1fe.jpg)  
Fig. 1 a Causal graph in credit card fraud scenario; b Modified causal graph in credit card fraud scenario

$$
\begin{array} { r l } & { \qquad P ( \boldsymbol { Y } = \boldsymbol { y } \mid \mathbf { d o } ( \boldsymbol { X } = x ^ { * } ) , Z ) } \\ & { = \sum _ { Z } P ( \boldsymbol { Y } = \boldsymbol { y } \mid \boldsymbol { X } = x ^ { * } , Z = z ) P ( Z = z ) } \\ & { \qquad \triangleq \sum _ { z } W ( Z = z ) * f ( x ^ { * } , z ) . } \end{array}
$$

In Eq. (2), the function $f ( \mathbf { x } ^ { * } , \mathbf { z } )$ can be estimated using a conventional ML/DL model, whereas $W ( Z = z )$ denotes the weight of the confounding variable for a given value $z$ . Equation (2) can be interpreted as a weighted sum of the predicted conditional probability of the label $Y$ for a sample, where the risk representation $X$ is set to $x ^ { * }$ and its confounding variable $Z$ is assigned different values $z$ .

# 3.2 Formal definition of optimal CFs

This section details the mathematical definition of the optimal CFs used in our framework by formulating and solving a constrained optimization problem. For discussion purposes, we define CFs in a binary classification scenario. We assume that $\mathcal { X } \subseteq \mathbb { R } ^ { n }$ is the feature space of the input, and $\mathcal { V } = 0 , 1$ is the label space of the output. We also assume the existence of a prediction model $h _ { \omega } : { \mathcal { X } } { \mapsto } { \mathcal { Y } }$ that maps the feature vector $\pmb { x } = ( x _ { 1 } , . . . , x _ { n } ) \in \mathcal { X }$ to the label to which it belongs, i.e., $h _ { \omega } ( { \pmb x } ) = { \boldsymbol y } \in \mathcal { V }$ . For a particular sample $\boldsymbol { x }$ , its corresponding CFs are defined as changing some of its features from ${ \mathcal { F } } \subseteq \{ 1 , . . . , n \}$ such that $h _ { \omega } ( \widetilde { \mathbf { x } } ) \neq h _ { \omega } ( \pmb { x } )$ . To ensure that the obtained CFs have the property of a complex negative sample, i.e., distributed on the classification boundary of $h _ { \omega }$ , it is also necessary to find the optimal CFs $\tilde { { \mathbf { x } } } ^ { * }$ of $\boldsymbol { x }$ . This is achieved by adding the minimum perturbation on the original input such that $h _ { \omega } ( \widetilde { \mathbf { x } } ^ { * } ) \neq h _ { \omega } ( \pmb { x } )$ . The model-based optimal CFs generation process is shown in Fig. 2.

We can obtain such optimal CFs $\tilde { { \mathbf { x } } } ^ { * }$ using a model-based approach. Let $g _ { \theta } : { \mathcal { X } } { \mapsto } { \mathcal { X } }$ be a CFs generator with the parameter $\theta$ whose input is $\boldsymbol { x }$ , and output is its corresponding CFs $\widetilde { \pmb { x } } = g _ { \theta } ( \pmb { x } )$ . Given a sample set ${ \mathcal { D } } \sim p _ { \mathrm { d a t a } } ( { \pmb x } )$ , the loss function of $g _ { \theta }$ for generating a CFs is defined as follows:

$$
\mathcal { L } _ { \mathrm { C F } } ( g _ { \theta } ; \mathcal { D } , h _ { \omega } ) = \frac { 1 } { | \mathcal { D } | } \sum _ { x \in \mathcal { D } } \ell _ { \mathrm { o b j } } ( \boldsymbol { x } , \boldsymbol { g } _ { \theta } ( \boldsymbol { x } ) ; h _ { \omega } ) + \lambda \ell _ { \mathrm { r e g } } ( \boldsymbol { x } , \boldsymbol { g } _ { \theta } ( \boldsymbol { x } ) ) .
$$

The first term in Eq. (3) penalizes the modified sample $g _ { \boldsymbol { \theta } } ( \pmb { x } )$ that fails to flip the label. Suppose $C \subseteq X$ is a subset of samples that fail to roll the label, i.e., $C = \{ \pmb { x } \in \pmb { \chi } \mid h _ { \omega } ( \pmb { x } ) = h _ { \omega } ( g _ { \theta } ( \pmb { x } ) ) \}$ , the $\ell _ { \mathrm { o b j } }$ term is equivalent to the following formula:

$$
\ell _ { \mathrm { o b j } } ( { \pmb x } , g _ { \theta } ( { \pmb x } ) ; h _ { \omega } ) = \mathbb { 1 } _ { C } ( { \pmb x } ) ,
$$

where $\mathbb { 1 } _ { C } ( { \pmb x } )$ is a 0–1 indicator function. Subsequent $\ell _ { \mathrm { r e g } } :$ $\mathcal { X } \times \mathcal { X } { \longmapsto } \mathbb { R } _ { > 0 }$ can be an arbitrary distance metric function, constraining $\tilde { x }$ be close to $\boldsymbol { x }$ . $L ^ { 1 }$ -norm is selected in our implementation. $\lambda$ is a hyperparameter that controls the importance tradeoff between $\ell _ { \mathrm { o b j } }$ and $\ell _ { \mathrm { r e g } }$ . In addition, to guarantee the validity of the CFs, the number of features that can be perturbed for a sample should also be strictly limited, i.e., $| { \mathcal { F } } | \leq m$ , where $1 \leq m \leq n$ .

![](images/df72748ba43a1d9c9796b702d59c13165bd09dad6f558fe43e2121f2854f787c.jpg)  
Fig. 2 Model-based optimal CFs generation process

Then the solution problem of the optimal CFs generator $g ^ { * } = g _ { \theta ^ { * } }$ can be reformulated in the following constrained optimization problem:

$$
\pmb { \theta } ^ { * } = \arg \operatorname* { m i n } _ { \theta } \{ \mathcal { L } _ { \mathrm { C F } } ( g _ { \theta } ; \mathcal { D } , h _ { \omega } ) \} ,
$$

The optimal CFs generated based on $g ^ { * }$ are $\tilde { x } ^ { * } = g ^ { * } ( x )$ . By calculating $\tilde { x } ^ { * } - x$ , we can regard the most varied features as being highly important for the prediction of $h _ { \omega }$ .

# 4 Proposed method

This section presents the counterfactual triple network (CFTNet) framework for credit card fraud classification. The framework generates the optimal CFs to create new hard-negative samples by modifying the original samples in the dataset [20–22]. In our methodology, we first reformulate the generation of optimal CFs as an MDP. The solution of the optimal CFs generator is equivalent to optimizing the RL agent. Specifically, we use P-DQN [23] to obtain the optimal CFs in this paper. After acquiring the optimal CFs for the entire dataset, we propose the CFTNet model, a deep DL-based feature extraction and classification architecture that can be applied to tasks such as class imbalance/fine-grained classification, where traditional ML/DL methods may not perform well. The optimal CFs generation process based on reinforcement learning P

DQN model is shown in Fig. 3. In CFTNet, we leverage the unique properties of positive samples, CFs, and negative samples to design a special $\ell _ { \mathrm { s i m } }$ during model training. This helps to disentangle the representations and satisfy the independent relationship between variables in the modified causal graph (Fig. 1b). To further strengthen the disentanglement of variables, we obtain the classification loss $\ell _ { \mathrm { w c l s } }$ weighted by instance-level disentanglement intensity. The model trained in this way can achieve prediction based on causality in the inference stage. The formulation of $\ell _ { \mathrm { w c l s } }$ is consistent with the simplified form of the probability distribution in Eq. (2).

# 4.1 Optimal CFs generation based on P-DQN model

This section introduces the optimal CFs generation algorithm based on the RL P-DQN model. First, we define the intermediate variables involved in the algorithm in the context of RL.

# 4.1.1 Formulation of Markov’s decision process

To introduce the reinforcement learning method used in this paper, in this section, we reformulate the process of calculating the optimal CFs $\tilde { { \mathbf { x } } } ^ { * }$ of $\boldsymbol { x } \in \mathcal { D }$ in the Eq. (3) as a MDP.

Status $( S )$ . At each time step $t .$ , the state of the agent is $S _ { t } = s _ { t }$ , where $s _ { t } = ( x _ { t } , f _ { t } ) \in S$ represents the perturbed sample $x _ { t }$ and the modified feature set $f _ { t }$ . Specifically, $f _ { t } \in \{ 0 , 1 \} ^ { | \mathcal { F } | }$ is a binary vector of dimension $| \mathcal F |$ , where $f _ { t } [ k ] = 1$ indicates that the $k$ -th feature has been modified before the $t \cdot$ -th step. We set $x _ { 0 } = x$ when $t = 0$ , and $f _ { 0 } = 0 ^ { | \mathcal { F } | }$ . If $h _ { \omega } ( { \pmb x } _ { t } ) \neq h _ { \omega } ( { \pmb x } )$ , the agent ends with $\tilde { { \mathbf { x } ^ { * } } } = x _ { t }$ for $\boldsymbol { x }$ . Otherwise, the agent performs the next action $A _ { t } = a _ { t }$ : selecting one feature from the set of features that have not been modified to change and determining the magnitude of the change.

![](images/ad8f00393f9f145f49bddcdafe3c3a90f5a9e7c4523a30f55c3bd97b399a5734.jpg)  
Fig. 3 Optimal CFs generation process based on reinforcement learning P-DQN model

• Discrete–continuous mixing action $( \mathcal { A } )$ . For any time step $t$ , the agent first selects an action $k _ { t }$ from the discrete action set $\mathcal { F } _ { t } \subset \mathcal { F } = \mathcal { F } \backslash \bigcup _ { j = 0 } ^ { t - 1 } k _ { j }$ , i.e., one feature from the set of features that have not been modified at step $t$ . The agent then selects the low-level action $\nu _ { k _ { t } } \in \mathbb { R }$ to determine the degree of modification of the feature $k _ { t }$ . The action at step $t$ is $a _ { t } = \left( k _ { t } , \nu _ { k _ { t } } \right)$ , and the discrete–continuous mixing action space at this time is $\mathcal { A } _ { t } = \{ ( k _ { t } , \nu _ { k _ { t } } ) \ | \ k _ { t } \in \mathcal { F } _ { t } , \nu _ { k _ { t } } \in \mathbb { R } \} ,$ . Transfer function $( \mathcal { T } )$ . After selecting the action $a _ { t } = \left( k _ { t } , \nu _ { k _ { t } } \right)$ , the probability of transferring from state $s _ { t }$ to $s _ { t + 1 }$ can be expressed as follows: $\begin{array} { r l r } & { } & { { \mathcal { T } } ( ( x _ { t } , f _ { t } ) , a _ { t } , ( x _ { t + 1 } , f _ { t + 1 } ) ) } \\ & { } & { = \left\{ \begin{array} { l l } { 1 , \mathrm { ~ i f ~ } { \pmb x } _ { t + 1 } [ k _ { t } ] = { \pmb x } _ { t } [ k _ { t } ] + \nu _ { k _ { t } } \wedge f _ { t + 1 } [ k _ { t } ] = 1 } \\ { 0 , \mathrm { ~ o t h e r w i s e } } \end{array} \right. } \end{array}$ ð6Þ Reward (r). Based on the formulation in the formula (3), this paper defines the reward of the MDP as follows:

$$
r ( s _ { t } , a _ { t } ) = \left\{ \begin{array} { l l } { 1 - \lambda \Big ( \ell _ { \mathrm { r e g } } ^ { t } - \ell _ { \mathrm { r e g } } ^ { t - 1 } \Big ) , \mathrm { i f } h _ { \omega } ( \pmb x _ { t } ) \neq h _ { \omega } ( \pmb x ) } \\ { - \lambda \Big ( \ell _ { \mathrm { r e g } } ^ { t } - \ell _ { \mathrm { r e g } } ^ { t - 1 } \Big ) , \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

where $\lambda$ is a hyperparameter that weighs the importance of achieving the counterfactual goal (flipping the sample class label) against keeping the generated CFs as close as possible to the original sample. Defining the reward above will encourage the agent to flip the predicted outcome of $h _ { \omega }$ by adding the smallest possible perturbation to the original sample.

• Policy $( \pi _ { \boldsymbol { \theta } } )$ . To maximize the expected reward for the MDP described above, we define the policy with the parameter $\theta$ . It has been proved in [22] that finding the optimal policy $\pi ^ { * } = \pi _ { \theta ^ { * } }$ , for a given environment $h _ { \omega }$ (black-box model in the Fig. 3) is equivalent to solving the optimal CFs generator $g ^ { * }$ that minimizes the counterfactual loss in the Eq. (3).

# 4.1.2 Policy optimization

This paper solves the optimal strategy $\pi ^ { * }$ based on the PDQN model. Considering the discrete–continuous mixing action space defined in this paper, when calculating the action value at step $t .$ , the continuous parameter $\nu _ { k _ { t } }$ is related to the selected discrete action $k _ { t }$ . That is, there is $\nu _ { k _ { t } } = \arg \operatorname* { s u p } _ { \nu } Q ( s _ { t } , k _ { t } , \nu )$ when given the state $s _ { t }$ and discrete action $k _ { t }$ , and the choice of the continuous parameter can be written as a function $\nu _ { k _ { t } } ^ { Q } ( s _ { t } )$ related to the action value function $Q$ . We define a DNN $Q ^ { \theta _ { 1 } } ( s _ { t } , k _ { t } , \nu _ { k _ { t } } )$ that fits the actual action value function while using a deterministic policy network $\nu _ { k _ { t } } ^ { \theta _ { 2 } } \left( s _ { t } \right)$ to directly output the optimal continuous parameter given the discrete action $k _ { t }$ and state $s _ { t }$ , where $\theta _ { 1 }$ and $\theta _ { 2 }$ are the parameters of the two neural networks. Stochastic gradient descent (SGD) is used to solve the following loss function to obtain the optimal values $\theta _ { 1 } ^ { * }$ and $\theta _ { 2 } ^ { * }$ :

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \boldsymbol { Q } } ( \pmb { \theta } _ { 1 } ) = \big [ \pmb { Q } ^ { \theta _ { 1 } } ( s _ { t } , k _ { t } , \nu _ { k _ { t } } ) - y _ { t } \big ] ^ { 2 } , \mathcal { L } _ { \boldsymbol { \pi } } ( \pmb { \theta } _ { 2 } ) } \\ { = - \displaystyle \sum _ { k _ { t } \in \mathcal { F } } \pmb { Q } \Big ( s _ { t } , k _ { t } , \nu _ { k _ { t } } ^ { \theta _ { 2 } } ( s _ { t } ) \Big ) , } \end{array}
$$

where $y _ { t }$ is an $\mathbf { n }$ -step TD target commonly used in RL algorithms based on value functions. Theoretically, as the loss in Eq. (7) decreases, the output of the $Q$ network converges to the real optimal action value function, and the deterministic policy network outputs the optimal continuous action. The detailed process of training P-DQN, i.e., the optimal CFs generator, is shown in the Algorithm 1.

![](images/77165ae924bfbdd2f8c8243375c0da46bf5eecc1fc72aee3f2d55fe6fd7e4479.jpg)  
Fig. 4 The framework of the CFTNet model

$\theta _ { 1 } \gets$ $Q ^ { \theta _ { 1 } }$ $\theta _ { 2 } \gets$ $v _ { k _ { t } } ^ { \theta _ { 2 } }$ $\mathcal { M } $ initialize the replay buffer $i \gets 1$   
5:while $i \leqslant$ max_epochs do $x \in \mathcal { D }$ ， sample a training instance $_ { x }$ from $\mathcal { D }$ $\begin{array} { l } { x _ { 0 } \gets x } \\ { s _ { 0 } = ( x _ { 0 } , f _ { 0 } ) } \\ { t \gets 0 } \end{array}$   
10: for $t \leqslant T$ do $T = 5 0 , 0 0 0$ ）   
11: 1 $v _ { k _ { t } } \gets v _ { k _ { t } } ^ { \theta _ { 2 } } ( s _ { t } )$ ,Compute the continuous parameter   
12: $a _ { t } \gets ( k _ { t } , v _ { k _ { t } } )$ , Select the discrete action by $\epsilon$ -greedy   
13: $r _ { t } , s _ { t + 1 } \gets \mathcal { T } ( s _ { t } , a _ { t } )$ ，The agent gets the reward and observes the next state   
14: $\mathcal { M }  ( \{ s _ { t } \} , \{ a _ { t } \} , \{ r _ { t } ^ { \prime } \} , \{ s _ { t + 1 } \} )$ ，Store transition into the replay buffer   
15: $B \in { \mathcal { M } }$ ,Randomly sample batch $B$ from $\mathcal { M }$   
16: $\theta _ { 1 }  \theta _ { 1 } - \gamma _ { 1 } \Delta \mathcal { L } _ { Q } ^ { \prime } ( \theta _ { 1 } )$ , Update the parameters $\theta _ { 1 }$ of the Q network using SGD   
17: $\theta _ { 2 }  \theta _ { 2 } - \gamma _ { 2 } \Delta \mathcal { L } _ { \pi } ^ { \prime } ( \theta _ { 2 } )$ ，Update the parameters $\theta _ { 2 }$ of the policy network using SGD   
18: $t \gets t + 1$   
19: end for   
20: $i  i + 1$   
21:endwhile return the optimal parameters $\theta _ { 1 } ^ { * } , \theta _ { 2 } ^ { * }$ of both networks

# 4.2 Counterfactual triplet network

This section presents the CFTNet structure for the credit card fraud detection binary classification task, as illustrated in Fig. 4. The input to the network is a triplet consisting of a positive sample $\pmb { x } _ { i } \in \mathbb { R } ^ { n }$ , its corresponding optimal CFs $\tilde { \pmb { x } } _ { i } ^ { * } \in \mathbb { R } ^ { n }$ , and a normal negative sample $\pmb { x } _ { j } \in \mathbb { R } ^ { n }$ randomly selected from the negative sample set. These three feature vectors are fed into the feature encoder $f _ { \omega }$ to obtain their respective embeddings, $f _ { \omega } ( \pmb { x } _ { i } ) \in \mathbb { R } ^ { d }$ , $f _ { \omega } ( \tilde { \pmb { x } } _ { i } ^ { * } ) \in \mathbb { R } ^ { d }$ , and $f _ { \omega } ( \pmb { x } _ { j } ) \in \mathbb { R } ^ { d }$ , where $d \gg n$ . The feature encoder $f _ { \omega }$ is a three-layer DNN; the formula is as follows:

$$
f _ { \omega } ( { \pmb x } _ { i } ) = f _ { \omega , 3 } \big ( \omega _ { 3 } \cdot { \pmb f } _ { \omega , 2 } \big ( \omega _ { 2 } \cdot { f } _ { \omega , 1 } \big ( \omega _ { 1 } \cdot { \pmb x } _ { i } + b _ { 1 } \big ) + b _ { 2 } \big ) + b _ { 3 } \big ) ,
$$

where $\omega = \left\{ \omega _ { 1 } , \omega _ { 2 } , \omega _ { 3 } , b _ { 1 } , b _ { 2 } , b _ { 3 } \right\}$ is parameters of the DNN, $f _ { \omega , 1 } , f _ { \omega , 2 } , f _ { \omega , 3 }$ is a nonlinear activation function with the following formualtion:

$$
f ( x ) = \operatorname* { m a x } ( 0 , x ) .
$$

The DNN-based feature encoders use layer-by-layer linear combinations and nonlinear mappings to map inputs to higher-dimensional representation spaces. Experimental results have shown that after feature extraction, the resulting representation $f _ { \omega } ( { \pmb x } )$ is more informative and can improve the performance of downstream classification/regression tasks [37]. However, many dimensions of the vector $f _ { \omega } ( { \pmb x } )$ may contain redundant information that does not contribute to the objective of credit card fraud detection. To achieve more accurate and robust classification, the model needs to fit a function from $X$ and $Z$ to $Y$ in a feature space that achieves representation disentanglement, where $X$ and $Z$ are statistically independent variables. CFTNet introduces the representation separation module $\mathrm { S e p } ( \cdot )$ to disentangle the representations of positive samples and CFs, $f _ { \omega } ( \pmb { x } _ { i } )$ and $f _ { \omega } ( \tilde { x } _ { i } ^ { * } )$ , by filtering out redundant information that is not critical to Y. The positive sample contains the most discriminative information. On the contrary, the CFs modify the positive samples in a way that preserves most of the irrelevant information but flips the most discriminative information. For the positive sample, we have the following formulation:

$$
\begin{array} { r l } & { r _ { i } , z _ { i } = \mathrm { S e p } _ { \varpi } ( f _ { \omega } ( \pmb { x } _ { i } ) ) , } \\ & { r _ { i } = \sigma ( \varpi \cdot f _ { \omega } ( \pmb { x } _ { i } ) ) \cdot f _ { \omega } ( \pmb { x } _ { i } ) , } \\ & { z _ { i } = ( 1 - \sigma ( \pmb { \varpi } \cdot f _ { \omega } ( \pmb { x } _ { i } ) ) ) \cdot f _ { \omega } ( \pmb { x } _ { i } ) . } \end{array}
$$

In the positive sample, $r _ { i }$ and $z _ { i }$ represent the risk and confounding representation vectors, respectively, corresponding to $X$ and $Z$ in Fig. 1b. The separation module in CFTNet employs a fully connected (FC) layer parameterized by $\varpi \in \mathbb { R } ^ { d \times d }$ , where $\begin{array} { r } { \sigma ( x ) = \frac { 1 } { 1 + \exp ( - x ) } } \end{array}$ maps the range of values of $x$ to the interval $( 0 , 1 )$ . The representation separation module in CFTNet effectively filters information by separating the risk representation $X$ from the confounding representation $Z$ in the positive sample and its corresponding CFs. During the final prediction, CFTNet only utilizes $r _ { i }$ and $\tilde { r } _ { i } ^ { * }$ corresponding to $X$ in Fig. 1b to calculate the probability of a sample being positive. To this end, a three-layer FC network $\phi ( \cdot )$ is employed as the classifier:

$$
\begin{array} { r l } & { \hat { y _ { i } } = \phi ( \pmb { r } _ { i } ) , } \\ & { \hat { \widetilde { y } } _ { i } ^ { * } = \phi \left( \widetilde { \pmb { r } } _ { i } ^ { * } \right) , } \\ & { \hat { y _ { j } } = \phi \left( f _ { \omega } \left( \pmb { x } _ { j } \right) \right) . } \end{array}
$$

where $\hat { y _ { i } } , \hat { \tilde { y } } _ { i } ^ { * }$ , and $\hat { y _ { j } }$ denote the model’s predictions for the positive sample in CFTNet, its CFs, and the randomly selected negative sample, respectively, indicating the probability of engaging in credit card fraud. To obtain optimal model parameters, represented as $\{ \omega ^ { * } , \varpi ^ { * } , \phi ^ { * } \}$ , CFTNet minimizes the cross-entropy loss function, taking into account the ground truth of the three types of samples, which is formulated as follows:

$$
\ell _ { \mathrm { c l s } } = \sum _ { i = 1 } ^ { | B | } \sum _ { k \in \{ i , i ^ { * } , j \} } y _ { k } \log ( \hat { y _ { k } } ) + ( 1 - y _ { k } ) \log ( 1 - \hat { y _ { k } } ) .
$$

In Eq. (13), $| B |$ denotes the batch size in the model’s forward and backward processes, where $y _ { i } = 1$ , $y _ { i ^ { * } } = y _ { j } = 0$ . By performing mini-batch SGD to optimize the loss function, CFTNet is trained to make highly confident predictions for positive samples indicating credit card fraud while assigning low probabilities to CFs and negative examples.

# 4.3 Representation disentanglement based on triplet contrast learning

To achieve finer representation disentanglement between positive samples, CFs, and negative samples based on their similarity in the confounding representation $Z$ and difference in the risk representation $X$ , and to attain statistical independence between the variables in Fig. 1b, this paper proposes a regularization term $\ell _ { \mathrm { s i m } }$ based on the CL paradigm. In particular, the positive sample has a class label of 1 and contains the risk representation $X _ { i }$ and confounding representation $Z _ { i }$ . The corresponding optimal CFs have a class label of 0 and a risk representation $X _ { i ^ { * } }$ different from the positive sample, but a confounding representation $Z _ { i ^ { * } }$ highly similar to $Z _ { i }$ of the positive sample. Traditional ML/DL models based on observed correlation may fit spurious correlation between the confounding representation $Z$ and the class label Y, leading to misclassification between CFs and positive samples. Furthermore, the randomly selected negative sample $X _ { j }$ also has a class label of 0 and a risk representation similar to CFs, which enhances the risk representation in CFs by increasing their semantic similarity. This, in turn, enhances the risk representation in the positive sample by reducing the semantic similarity between the risk representation of CFs and the positive sample. The risk representation $X _ { i }$ of the disentangled positive sample satisfies the semantic similarity constraint in Fig. 5, which is essential for predicting the class label (detecting fraud behavior).

![](images/9b12a9dc32ac6e5d131eec906747a4bfb797e4acd83b59ebdc744be280a9be40.jpg)  
Fig. 5 The representation vectors of the three types of samples

This paper adopts a self-supervised learning approach based on CL [38] for CFTNet to learn a metric space, in which the representation vectors satisfy the semantic similarity constraint illustrated in Fig. 5. To this end, the semantic similarity regularization term $\ell _ { \mathrm { s i m } }$ is proposed, as shown in Eq. (14):

$$
\begin{array} { r l } & { \ell _ { \mathrm { s i m } } = - \displaystyle \sum _ { i = 1 } ^ { \lfloor B \rfloor } } \\ & { \log \frac { \exp { \left( \frac { s \left( z _ { i } , z _ { i } ^ { * } \right) } { \tau } \right) } + \exp { \left( \frac { s \left( z _ { i } ^ { * } , f _ { o } \left( x _ { j } \right) \right) } { \tau } \right) } } { \exp { \left( \frac { s \left( r _ { i } , r _ { i } * \right) } { \tau } \right) } + \exp { \left( \frac { s \left( r _ { i } , z _ { i } \right) } { \tau } \right) } + \exp { \left( \frac { s \left( r _ { i } , f _ { o } \left( x _ { j } \right) \right) } { \tau } \right) } } , } \end{array}
$$

where $\tau$ is a temperature coefficient that adjusts the importance of each term in the loss. The notation $r$ and $z$ have the same meaning as in Sect. 3.2, and $s ( \cdot , \cdot )$ is a similarity metric function, with cosine similarity used in the experiments in our implementation. To further enhance representation disentanglement and improve the accuracy and robustness of CFTNet’s predictions, this paper proposes instance re-weighting to improve the classifier $\phi ( \cdot )$ during model training. The weight assigned to each instance is $s ( z _ { i } , z _ { i } ^ { * } )$ , where instances with higher weights have higher semantic similarity between the confounding representations of the positive sample and its CFs. The instances with disentangled $r _ { i }$ that is closer to the risk representation without redundant information for the classification task are more confidently predicted by the model and, thus, they are given greater weights in the total classification loss $\ell _ { \mathrm { w c l s } }$ . The formula for $\ell _ { \mathrm { w c l s } }$ is as follows:

$$
\ell _ { \mathrm { w c l s } } = \sum _ { i = 1 } ^ { | B | } \sum _ { k \in \{ i , i ^ { * } , j \} } w _ { k } ( y _ { k } \log ( \hat { y _ { k } } ) + ( 1 - y _ { k } ) \log ( 1 - \hat { y _ { k } } ) ) ,
$$

the instance weight $w _ { k }$ in the formula (15) is calculated as follows:

$$
w _ { k } = \left\{ { \begin{array} {} { c } { s ( z _ { i } , z _ { i } ^ { * } ) , { \mathrm { ~ i f ~ } } k = i } \\ { 1 , { \mathrm { ~ o t h e r w i s e . ~ } } } \end{array} } \right.
$$

Equation (15), which defines the weighted classification loss $\ell _ { \mathrm { w c l s } }$ , has a similar form to Eq. (2) from the perspective of intervening in the generation mechanism of variables. $\ell _ { \mathrm { w c l s } }$ can be viewed as a backdoor adjustment of the risk representation $X$ in Fig. 1b to eliminate undesired pathways between $X$ and $Y$ induced by $Z .$ . This adjustment mitigates the negative effect of spurious correlation on model prediction. This approach provides an unbiased estimation of the class label $Y$ distribution, reflecting the actual causal relationship between $X , Z ,$ , and $Y$ . To optimize the CFTNet model, we replace $\ell _ { \mathrm { c l s } }$ with $\ell _ { \mathrm { w c l s } }$ in Sect. 3.2, as shown in Eq. (17). Here, $\mathscr { X }$ is a hyperparameter controlling the relative importance of two parts of the loss function, and $\Omega$ is a regularized term limiting the complexity of model parameters; the final loss function is as follows:

$$
\ell _ { \mathrm { t o t a l } } = \ell _ { \mathrm { w c l s } } + \alpha \ell _ { \mathrm { s i m } } + \Omega .
$$

# 5 Experiments

This section presents a series of comparative and ablation experiments to evaluate the effectiveness and robustness of the proposed CFTNet for credit card fraud detection. The experiments aim to address the following research questions:

• RQ1: Does the proposed CFTNet significantly outperform traditional ML/DL and SOTA methods in the accuracy of credit card fraud detection?   
$R Q 2$ : Can counterfactual data augmentation enhance the robustness of DL models under varying degrees of class imbalance?   
• RQ3: What is the contribution of each submodule in the proposed CFTNet to overall performance improvement? How do different hyperparameter values affect CFTNet performance? RQ4: What is the causal modeling capability of CFTNet and does it capture the causal relationship between key features and class labels?

# 5.1 Dataset and experimental setup

# 5.1.1 Dataset visualization

The dataset used in this research is sourced from the UCI Machine Learning Repository [39]. It consists of transaction data from European cardholders in September 2013, covering two days of transaction records with 284,807 transactions. Among these, 492 transactions are fraudulent and serve as positive examples for classification, accounting for only $0 . 1 7 2 \%$ of all transactions. The dataset features have been transformed by PCA to reduce the dimension, except for features ‘time’ and ‘transaction amount’, resulting in 28 features. The dataset is characterized by a high degree of class imbalance (extremely unbalanced positive and negative samples) and high anonymity (ambiguous feature meanings).

First, we calculate the Spearman correlation coefficient between features and visualize the feature correlation within fraudulent and non-fraudulent samples, as shown in Fig. 6. It can be observed that the correlation between nonfraudulent sample features is very low. In contrast, the correlation between fraudulent sample features is high, indicating significant differences in feature distribution between the two groups. Moreover, we visualize the distribution of features for samples with different labels as shown in Fig. 7, where the orange part is the distribution of non-fraudulent sample features, and the blue part is the distribution of fraudulent sample features. It can be seen from the figure that the distribution of the same feature condition on different labels is quite different. For example, the value of the feature $\bf \tilde { V } 4 \ '$ of the non-fraud sample is distributed around 0 without an obvious tail. In contrast, the ‘V4’ of the fraud sample is distributed between 0 and 10, with a specific right tail. It can be inferred that the distribution of feature conditions on different labels has significant differences, which makes the binary classification task feasible.

![](images/92c20175095477f331c06ba3bc0a590b3d108fd6f8dd83e4c00ef396b95d684f.jpg)  
Fig. 6 Visualization of correlations between features concerning different classes. The darker the color in the box, the stronger the correlation between the features (both positive and negative), and the lighter the color, the weaker the correlation between the features (color figure online)   
Fig. 7 Visualization of the distribution of feature condition on different labels

# 5.1.2 Dataset preprocessing

Before the model training process, we downsample the dataset to different positive and negative sample ratios. Sect. 5.3 uses downsampled data with a positive-to-negative sample ratio of 1:50 to generate CFs and evaluate model performance. For model robustness evaluation, we use re-sampled datasets with positive-to-negative sample ratios of 1:25, 1:50, 1:100, 1:150, and 1:200 to assess whether the model’s classification performance is robust under varying sample ratios. All experiments are conducted using ten-fold cross-validation on the corresponding sampled dataset, split at 8:1:1 for training, validation, and testing. We ensure consistency of sample ratios in the training, validation, and test sets during the sampling process.

# 5.1.3 Baseline models

In this section, we present the baseline models employed in the comparison experiments and model robustness evaluations, all of which are drawn from recent literature on credit card fraud detection research and represent SOTA’s performance on that task. To ensure the reproducibility of the experimental results, we report the hyperparameters used in these baseline models in Sect. 5.1.4.

XGBoost [31]: A two-stage fraud detection model that combines transaction data features with network features obtained from graph representation learning for fraud detection. Specifically, we use DeepWalk [40] to obtain network features, followed by classification with XGBoost. RandomForest [41]: Five classical ML/DL methods have been fairly compared in [41], which uses the same dataset as ours, where RF shows the optimal performance among all the methods.   
• SVM [42]: Dheepa et al. [42] use a support vector machine (SVM) as a classifier to detect fraud and use viable component extraction technique to solve the class imbalance problem to improve the detection accuracy further. DNN [29]: Leveraging the qualities of DL that enable an increasingly high level of abstract representation in data and raw patterns, Voican et al. [29] introduces a DNN model capable of recognizing patterns of user behavior. Conv-DNN [29]: An improved version of the DNN in [29], using a convolutional kernel instead of the multilayer perceptron (MLP) to encode raw features. Logistic [43]: Five classical ML methods are evaluated on a dataset (the same as ours) after balanced class adaptation using the SMOTE technique, where the logistic regression (LR) achieves optimal performance. LightGBM [31]: A two-stage fraud detection model that combines transaction data features with network features obtained from graph representation learning for fraud detection. Specifically, we use DeepWalk [40] to obtain network features, followed by classification with LightGBM.   
• KNN [43]: $K$ Nearest Neighbors (KNN) is a simple

computational method that stores every accessible instance and characterizes new cases based on a distance metric and existing instances’ features, and is widely used in fraud detection tasks.

# 5.1.4 Hyperparameter setup

For the convenience of reproduction, we give the hyperparameter setup of CFTNet in Table 1.

In addition, we also give hyperparameter settings obtained by cross-validation of some baseline models in Table 2, where Conv-DNN is a DNN model with two layers of one-dimensional convolution kernels added at the input, the convolution kernel size of the convolutional layer is given in the Table 2, and the models or hyperparameters not mentioned in the Table 2 are kept the same as the related research.

# 5.2 Evaluation metrics

The objective of the anti-fraud classification task is to identify positive cases accurately, and a high recall rate for positive samples is often a primary requirement for the classification model. To balance the importance of recall and precision in classification, this paper introduces the $F _ { 2 }$ score as an evaluation metric alongside recall rate and accuracy. For a fair comparison with current credit card fraud detection methods, we also use the classic metric AUC for evaluating the performance of binary classification models. Therefore, we use recall rate, $F _ { 2 }$ score, and accuracy rate as evaluation criteria for the classification model’s performance, focusing on $F _ { 2 }$ score and recall rate as primary evaluation metrics.

$$
\begin{array} { r l } & { \mathrm { R e c a l l } = \displaystyle \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } , ( } \\ & { \mathrm { P r e c i s i o n } = \displaystyle \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } , ( } \\ & { F _ { 2 } - \mathrm { s c o r e } = ( 1 + \beta ^ { 2 } ) \displaystyle \frac { \mathrm { P r e c i s i o n } * \mathrm { R e c a l l } } { \beta ^ { 2 } * \mathrm { P r e i c i s i o n } + \mathrm { R e c a l l } } ( \beta = 2 ) , } \end{array}
$$

Table 1 CFTNet hyperparameter settings   

<table><tr><td>Hyperparameter</td><td>Meaning</td><td>Hyperparameter setup</td></tr><tr><td>FC_dim</td><td>Representation vector dimensions</td><td>32</td></tr><tr><td>learning_rate</td><td>Learning rate</td><td>0.01</td></tr><tr><td>Activate_Func</td><td>Activation function</td><td>ReLu</td></tr><tr><td>Batchsize</td><td>Training batches</td><td>300</td></tr><tr><td>T</td><td>Temperature coeficient</td><td>0.95</td></tr><tr><td>α</td><td>Balance weight of loss function</td><td>0.5</td></tr></table>

Table 2 Hyperparameter settings of baseline models   

<table><tr><td>Model</td><td>Hyperparameter</td><td>Meaning</td><td>Hyperparameter stup</td></tr><tr><td rowspan="3">XGBoost [31]</td><td>n_estimators</td><td>Number of trees</td><td>50</td></tr><tr><td>lr</td><td>Learning rate</td><td>0.01</td></tr><tr><td>epoch</td><td>1</td><td>100</td></tr><tr><td rowspan="3">Random Forest [43]</td><td>n_estimators</td><td>Number of trees</td><td>50</td></tr><tr><td>lr</td><td>Learning rate</td><td>0.01</td></tr><tr><td>epoch</td><td></td><td>100</td></tr><tr><td rowspan="3">LightGBM [31]</td><td>n_estimators</td><td>Number of trees</td><td>50</td></tr><tr><td>lr</td><td>Learning rate</td><td>0.01</td></tr><tr><td>epoch</td><td>1</td><td>100</td></tr><tr><td rowspan="2">SVM [42]</td><td>C</td><td>Penalty weight</td><td>10</td></tr><tr><td>kernel</td><td>Types of kernel functions</td><td>RBF</td></tr><tr><td rowspan="5">DNN [29]</td><td>FC_dim</td><td>Representation vector dimensions</td><td>32</td></tr><tr><td>Optimizer</td><td>1</td><td>SGD</td></tr><tr><td>Batchsize</td><td>Training batches</td><td>300</td></tr><tr><td>lr</td><td>Learning rate</td><td>0.01</td></tr><tr><td>epoch</td><td>1</td><td>100</td></tr><tr><td rowspan="6">Conv-DNN [29]</td><td>kernel_size</td><td>One-dimensional convolution kernel size</td><td>16/12</td></tr><tr><td>FC_dim</td><td>Representation vector dimensions</td><td>32</td></tr><tr><td>Optimizer</td><td>1</td><td>SGD</td></tr><tr><td>Batchsize</td><td>Training batches</td><td>300</td></tr><tr><td>lr</td><td>Learning rate</td><td>0.01</td></tr><tr><td>epoch</td><td>1</td><td>100</td></tr></table>

$$
\mathrm { A U C } = \frac { \sum \mathrm { p r e d } _ { \mathrm { p o s } } > \mathrm { p r e d } _ { \mathrm { n e g } } } { ( \mathrm { T P } + \mathrm { F N } ) \times ( \mathrm { T N } + \mathrm { F P } ) . }
$$

The model’s performance can be evaluated and described comprehensively based on the above criteria, where TP refers to the number of true positive samples correctly classified by the model, FN refers to the number of false negative samples (positive samples misclassified as negative), and FP refers to the number of false positive samples (negative samples misclassified as positive); and the numerator $\sum \mathrm { p r e d } _ { \mathrm { p o s } } > \mathrm { p r e d } _ { \mathrm { n e g } }$ in Eq. (21) is the count of combinations where positive samples exceed negative samples in terms of the prediction scores of the model. We calculate the above evaluation metrics in all experimental setups.

Table 3 Results of model comparison   

<table><tr><td>Model</td><td>F2-score</td><td>Recall</td><td>Precision</td><td>AUC</td></tr><tr><td>CFTNet</td><td>0.978</td><td>0.980</td><td>0.960</td><td>0.989</td></tr><tr><td>XGBoost [31]</td><td>0.964</td><td>0.960</td><td>0.980</td><td>0.964</td></tr><tr><td>RandomForest [41]</td><td>0.944</td><td>0.940</td><td>0.959</td><td>0.970</td></tr><tr><td>SVM [42]</td><td>0.933</td><td>0.940</td><td>0.904</td><td>0.969</td></tr><tr><td>DNN [29]</td><td>0.916</td><td>0.920</td><td>0.900</td><td>0.939</td></tr><tr><td>Conv-DNN [29]</td><td>0.911</td><td>0.900</td><td>0.957</td><td>0.950</td></tr><tr><td>Logistic [43]</td><td>0.894</td><td>0.880</td><td>0.956</td><td>0.939</td></tr><tr><td>LightGBM [31]</td><td>0.927</td><td>0.920</td><td>0.958</td><td>0.949</td></tr><tr><td>KNN [43]</td><td>0.830</td><td>0.800</td><td>0.975</td><td>0.899</td></tr></table>

The boldness here indicates that the performance of our CFTNet algorithm is superior to the existing model in most of the evaluation indicators and has no other specific significance

# 5.3 Model comparison experiments (RQ1)

To address $R Q I$ , we comprehensively evaluate CFTNet’s performance. A series of baseline models are implemented, trained, and evaluated for comparison in the same procedure as CFTNet. The performance of these models is then compared to CFTNet’s on the test set to determine whether our proposed model outperforms the others.

Table 3 presents the evaluation results of CFTNet and the baseline models. The experimental results demonstrate that CFTNet achieves a very high accuracy rate while maintaining higher recall and AUC than other models, which is not attainable in the baseline models. Numerically, CFTNet achieves the highest $F _ { 2 }$ score of 0.978, AUC of 0.989, and recall of 0.980, with a precision of 0.960, only the precision slightly lower than XGBoost (0.980).

This shows that CFTNet is able to accurately detect positive samples (i.e., fraud) in the dataset with a low false positive rate. This is because the structural design of our special triplet network fully utilizes the feature similarity and label difference between positive samples, CFs, and negative samples to mine the causal relationship between features and labels, thus giving classification results with a higher confidence level. However, XGBoost has a marginally lower recall than CFTNet, and KNN has a recall of only 0.800, significantly lower than CFTNet. This indicates that the classification results of the above models are more or less overfitted. Both the network features used in the XGBoost [31] and the resampling technique used in the KNN [43] rely heavily on the data distribution in the training set, which cannot help the model robustly and effectively learn the causal relationship between features and labels. Overall, these results demonstrate that CFTNet outperforms other models in classification at a 1:50 sample ratio setting.

# 5.4 Model robustness evaluation(RQ2)

To address RQ2, we compare the classification performance of different models under varying sample ratios both horizontally and vertically to observe the models’ robustness. The classification performance of baseline models becomes volatile when the negative sample size increases sharply while the positive sample size remains unchanged. In theory, when the negative sample size changes within a specific range while the positive sample size remains the same, the model’s accuracy will improve to a certain extent due to the increase in sample size. The recall rate does not vary significantly, but when the negative sample size becomes too large, the class imbalance problem will manifest as a significant decrease in the recall rate. Based on the above analysis, the model’s ability to maintain robust performance across different sample ratios is crucial for evaluating its properties. We will experimentally demonstrate whether CFTNet is more robust than other models.

![](images/3382df8da969d584fd6b9c155b2376b85cb5f279ecddd65dd2a15efe1ef4b8e6.jpg)  
Fig. 8 The $F _ { 2 }$ score distribution of the model under different sample ratios

Table 4 shows the recall rate of the models at different sample ratios, with the optimal results highlighted in bold. Figure 8 visualizes the $F _ { 2 }$ scores of different models at varying sample ratios, where the upper and lower bounds indicate the highest and lowest scores, respectively. The greater the distance between the bounds, the less robust the method, while a dot indicates an outlier, where the gap between the bounds is either 1.5 times greater than the upper quartiles or 1.5 times less than the lower quartiles of the model under different sample ratios, signifying a precarious method. Figure 8 shows that XGBoost and Random Forest have outliers, while the other models have a long distance between their upper and lower bounds. CFTNet, with a short distance between its bounds, is more robust than the other models.

Numerically, the recall of most models is unstable under varying sample ratios. For instance, the recall of the RF model can reach 0.940 at a sample ratio of 1:50, but it drops below 0.80 when the ratio becomes 1:100, which is also the indirect cause of its outlier. In contrast, CFTNet’s recall rate fluctuates slightly but remains above 0.97, demonstrating that it is more robust than other models across different sample ratio settings.

Table 4 Results of model robustness evaluation under different sample ratios (Recall)   

<table><tr><td>Ratio</td><td>CFTNet</td><td>XGBoost [31]</td><td>LightGBM[31]</td><td>KNN [43]</td><td>RF [41]</td><td>DNN [29]</td><td>Logistic [43]</td><td>SVM [42]</td><td>Conv-DNN [29]</td></tr><tr><td>1:25</td><td>0.980</td><td>0.860</td><td>0.840</td><td>0.720</td><td>0.860</td><td>0.860</td><td>0.840</td><td>0.840</td><td>0.820</td></tr><tr><td>1:50</td><td>0.980</td><td>0.960</td><td>0.920</td><td>0.800</td><td>0.940</td><td>0.920</td><td>0.880</td><td>0.940</td><td>0.900</td></tr><tr><td>1:100</td><td>0.980</td><td>0.800</td><td>0.760</td><td>0.660</td><td>0.780</td><td>0.780</td><td>0.780</td><td>0.740</td><td>0.740</td></tr><tr><td>1:150</td><td>0.960</td><td>0.860</td><td>0.820</td><td>0.660</td><td>0.860</td><td>0.820</td><td>0.800</td><td>0.780</td><td>0.820</td></tr><tr><td>1:200</td><td>0.970</td><td>0.880</td><td>0.900</td><td>0.840</td><td>0.900</td><td>0.880</td><td>0.880</td><td>0.780</td><td>0.900</td></tr></table>

Table 5 Illustration of variants   

<table><tr><td>Model</td><td>Meaning</td></tr><tr><td>DNN</td><td>Variant without CFs and weighted classification loss</td></tr><tr><td>SimsameNet</td><td>Variant without CFs</td></tr><tr><td>TripletNet</td><td>Variant without weighted classification loss</td></tr><tr><td>CFTNet</td><td>Model with all submodules</td></tr></table>

![](images/a2591cd4a1f2288dfb974bc4acdfb1dcf9e34eace6010f5dd66252ecc21c50b2.jpg)  
Fig. 9 Results of ablation study

# 5.5 Ablation study and parameter sensitivity (RQ3)

To address $R Q 3$ , we investigate whether each submodule of the proposed CFTNet model contributes to its classification performance improvement. We conducted an ablation study using a 1:50 sampling dataset. For each submodule of CFTNet, we train and evaluate a model that removes the submodule and observe the classification performance of CFTNet with and without each submodule. Table 5 provides the names and meanings of each variant.

Figure 9 shows the improvement of model performance by introducing different submodules, and it can be seen that each submodule has a specific improvement in $F _ { 2 }$ score, recall, and precision. Numerically, the recall of SimsameNet obtained after adding weighted loss increases by $4 . 3 4 \%$ compared with the original DNN, the recall of TripletNet model obtained after adding counterfactual samples increases by $6 . 5 2 \%$ , and the recall rate of the CFTNet with all submodules increases by $6 . 8 5 \%$ , which is undoubtedly a strong proof of the effectiveness of CFTNet model submodules.

Furthermore, we conduct a sensitivity analysis on critical hyperparameters with balance weight $\mathscr { X }$ and embedding dimensions $d$ , as depicted in Fig. 10. To this end, we investigate the values of $d$ in the set $\{ 4 , 8 , 1 6 , 3 2 , 6 4 \}$ and $\mathscr { X }$ in the set $\{ 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 . 0 , 1 , 2 5 \}$ while fixing the remaining hyperparameter in each trial. Despite slight performance fluctuations as depicted in Fig. 10, our proposed method, CFTNet, exhibits robustness to hyperparameter selection. These findings imply that the choice of hyperparameters does not significantly impact the performance of our method.

# 5.6 Study on causal modeling capability based on SHAP method (RQ4)

The ability of causal modeling means that the model can select the most influential features for the classification task from many features and pay high attention to them while ignoring or reducing the weight of unimportant features. To evaluate the ability of the proposed CFTNet to model causal relationships in the data, we investigate CFTNet’s attention to different features of all samples based on the SHAP method [44, 45]. The SHAP method is an effective way to interpret model features proposed by Lundberg, which gets the importance of each feature by calculating the Shapley value of each feature. The Shapley value is the core of the SHAP method; the SHAP method interprets the output of the model as the sum of the attribution value (i.e., Shapley value) of each input feature [46], which can be expressed by the following formula for the input of a single instance sample:

![](images/7b782b311366fb8b20881f7c1569319db6790523266544fc72906e385d2c1df6.jpg)  
Fig. 10 The impact of hyperparameter changes on the model performance a Embedding dimension $d$ ; b Balance weight $\mathscr { X }$   
(a)Embedding dimension $^ d$

![](images/056224195a4de8e2b9aa132a012b863dfe7f0941fee98bacd7668d3420253eee.jpg)  
(b）Balance weight $\alpha$

![](images/b1e220eefb265f2ffed7af61e3ee83f43728c193a76eac2c0298430d7647fc1e.jpg)  
Fig. 11 Feature importance graph obtained by SHAP method a Logistic regression b CFTNet. A brighter color (both red and blue) indicates a larger absolute value of the feature for that sample, and a

$$
g ( x ) = \phi _ { 0 } + \Sigma _ { j = 1 } ^ { M } \phi _ { j } ,
$$

where $g ( \cdot )$ is the model being interpreted, $\phi _ { 0 }$ is a constant related to the type of the model, $M$ is the total number of features, $\phi _ { j }$ is the Shapley value of the $j$ th feature.

Figure 11 shows how the feature importance of the LR and CFTNet obtained by the SHAP method on the test set sample changes with the feature value. According to the previous analysis, models that capture causality in data should pay high attention to essential features and ignore unimportant ones. This means that the change in the value of crucial features is strongly correlated with the shift in Shapley’s value. In contrast, the evolution of trivial features is not correlated with the change in Shapley value. As seen from Fig. 11, compared with the LR model, the distribution diagram of essential features of CFTNet has more full colors, which means that the value of important features strongly correlates with the change of Shapley value.

![](images/fadc7a1e26f5590fe378c80903ac51355bd8797e02c29e27b8331be71167cdbd.jpg)  
darker color means a smaller absolute value of the feature for that sample (color figure online)

The color of the distribution of unimportant features below presents a mixed state, which means there is no correlation between the value of trivial features and the change of Shapley value. To sum up, CFTNet shows better causal modeling ability than traditional models.

# 5.7 Model evaluation

In the preceding analysis, we comprehensively validate the effectiveness of the model and its submodules in terms of classification performance and robustness.

During training, the model utilizes a triplet network architecture, however, it shares similarities with a conventional DNN structure while predicting. This implies that the model maintains the same inference speed as DNN. With the exception of a few rare scenarios (in some cases the performance improvement against simpler models is of the scale of $2 \%$ ), the model achieves superior classification performance while ensuring consistent inference speed compared to simpler models.

Furthermore, experiments based on Shapley demonstrate that CFTNet exhibits stronger interpretability for individual samples compared to simpler ML models (LR), substantiating the superiority of the model.

# 6 Conclusion and future work

This paper proposes a credit card fraud detection model based on counterfactual data augmentation via a triplet network. Our model addresses the challenges faced by existing anti-fraud methods, such as low recall rates, inaccurate results, and insufficient causal modeling ability. By converting the problem of CFs into a policy optimization of agents, we ensure the stable generation of optimal CFs in a discrete–continuous mixed action space. The triplet network effectively utilizes the feature similarity and label difference of positive example samples and CFs to enhance the learning of the causal relationship between features and labels. Experimental results demonstrate that our proposed method significantly outperforms existing methods. Specifically, Sect. 5.3 highlights the excellent performance in recall and AUC; furthermore, the robustness of the model is proved in Sects. 5.4; after that, 5.5 proves the necessity of the submodule and the low parameter sensitivity of the model; at last, Sect. 5.6 provides interpretability analysis from the perspective of Shapley. In short, this study contributes to the field of credit card anti-fraud research and practice by providing a novel approach to causal modeling issues. By addressing the limitations of traditional rule-based and data-driven methods, our model shows promise in detecting fraudulent transactions accurately and efficiently.

In terms of future work, there are several avenues for further exploration. Firstly, it would be beneficial to investigate the generalizability of our model to different domains beyond credit card fraud detection. Assessing its performance in other financial fraud detection scenarios or even in non-financial domains can provide valuable insights into the applicability and effectiveness of our approach. Additionally, incorporating real-time processing capabilities and more interpretable features into our model would enhance its practical utility. Furthermore, exploring the use of other DL algorithms or combining multiple models to create an ensemble approach could potentially improve the overall performance of the fraud detection system.

Acknowledgements This study was supported by the Natural Science Foundation of Hunan Province of China(grant number 2022JJ30673), Key R &D Program of Hunan Province(grant number 2023DK2003), Foundation of Department of Science and Technology of Hunan Province(grant number 2022GK3003), and the Graduate Innovation Project of Central South University(2023XQLH032, 2023ZZTS0304).

Author Contributions MK contributed to conceptualization, methodology, software, validation, formal analysis, investigation, data curation, and writing an original draft. RL contributed to investigation, resources, and data curation. JW and WX performed data curation. XL and SJ contributed to software. MH contributed to resources, writing, and review editing. CC contributed to methodology, investigation, resources, data curation, writing, and review editing.

Data availability Data will be made available on request.

# Declarations

Conflict of interest The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# References

1. Delamaire L, Abdou H, Pointon J (2009) Credit card fraud and detection techniques: a review. Banks Bank Syst 4(2):57–68   
2. Song R, Huang L, Cui W, Oskarsdottir M, Vanthienen J (2020) Fraud detection of bulk cargo theft in port using Bayesian network models. Appl Sci 10(3):1056   
3. Mishra KN, Pandey SC (2021) Fraud prediction in smart societies using logistic regression and k-fold machine learning techniques. Wireless Pers Commun 119:1341–1367   
4. Jiang C, Lu W, Wang Z, Ding Y (2023) Benchmarking state-ofthe-art imbalanced data learning approaches for credit scoring. Expert Syst Appl 213:118878   
5. Butaru F, Chen Q, Clark B, Das S, Lo AW, Siddique A (2016) Risk and risk management in the credit card industry. J Bank Financ 72:218–239   
6. Thabtah F, Hammoud S, Kamalov F, Gonsalves A (2020) Data imbalance in classification: experimental evaluation. Inf Sci 513:429–441   
7. Awoyemi JO, Adetunmbi AO, Oluwadare SA (2017) Credit card fraud detection using machine learning techniques: a comparative analysis. In: 2017 international conference on computing networking and informatics (ICCNI), pp 1–9 . IEEE   
8. Khine AA, Khin HW (2020) Credit card fraud detection using online boosting with extremely fast decision tree. In: 2020 IEEE conference on computer applications (ICCA), pp 1–4. IEEE   
9. Agarwal R, Melnick L, Frosst N, Zhang X, Lengerich B, Caruana R, Hinton GE (2021) Neural additive models: interpretable machine learning with neural nets. Adv Neural Inf Process Syst 34:4699–4711   
10. Bockel-Rickermann C, Verdonck T, Verbeke W (2023) Fraud analytics: a decade of research organizing challenges and solutions in the field. Expert Syst Appl 120605   
11. Carta S, Fenu G, Recupero DR, Saia R (2019) Fraud detection for e-commerce transactions by employing a prudential multiple consensus model. J Inf Secur Appl 46:13–22   
12. Fanai H, Abbasimehr H (2023) A novel combined approach based on deep autoencoder and deep classifiers for credit card fraud detection. Expert Syst Appl 217:119562   
13. Aftabi SZ, Ahmadi A, Farzi S (2023) Fraud detection in financial statements using data mining and GAN models. Expert Syst Appl 227:120144   
14. Settipalli L, Gangadharan G (2023) WMTDBC: an unsupervised multivariate analysis model for fraud detection in health insurance claims. Expert Syst Appl 215:119259   
15. Mirtaheri M, Abu-El-Haija S, Morstatter F, Ver Steeg G, Galstyan A (2021) Identifying and analyzing cryptocurrency manipulations in social media. IEEE Trans Comput Soc Syst 8(3):607–617   
16. Wang X, Cui P, Zhu W (2021) Out-of-distribution generalization and its applications for multimedia. In: Proceedings of the 29th ACM international conference on multimedia, pp 5681–5682   
17. Cui P, Shen Z, Li S, Yao L, Li Y, Chu Z, Gao J (2020) Causal inference meets machine learning. In: Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pp 3527–3528   
18. Kuang K, Cui P, Athey S, Xiong R, Li B (2018) Stable prediction across unknown environments. In: Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp 1617–1626   
19. Cui P, Athey S (2022) Stable learning establishes some common ground between causal inference and machine learning. Nat Mach Intell 4(2):110–115   
20. Mothilal RK, Sharma A, Tan C (2020) Explaining machine learning classifiers through diverse counterfactual explanations. In: Proceedings of the 2020 conference on fairness, accountability, and transparency, pp 607–617   
21. Karimi AH, Barthe G, Balle B, Valera I (2020) Model-agnostic counterfactual explanations for consequential decisions. In: International conference on artificial intelligence and statistics, pp 895–905. PMLR   
22. Chen Z, Silvestri F, Wang J, Zhu H, Ahn H, Tolomei G (2022) Relax: reinforcement learning agent explainer for arbitrary predictive models. In: Proceedings of the 31st ACM international conference on information & knowledge management, pp 252–261   
23. Xiong J, Wang Q, Yang Z, Sun P, Han L, Zheng Y, Fu H, Zhang T, Liu J, Liu H (2018) Parametrized deep q-networks learning: reinforcement learning with discrete-continuous hybrid action space. arXiv preprint arXiv:1810.06394   
24. Sailusha R, Gnaneswar V, Ramesh R, Rao GR (2020) Credit card fraud detection using machine learning. In: 2020 4th international conference on intelligent computing and control systems (ICICCS), pp 1264–1270 . IEEE   
25. Bin Sulaiman R, Schetinin V, Sant P (2022) Review of machine learning approach on credit card fraud detection. Human-Centric Intell Syst 2(1–2):55–68   
26. Saia R(2018) Unbalanced data classification in fraud detection by introducing a multidimensional space analysis. In: IoTBDS, pp 29–40   
27. Salekshahrezaee Z, Leevy JL, Khoshgoftaar TM (2023) The effect of feature extraction and data sampling on credit card fraud detection. J Big Data 10(1):6   
28. Harwani H, Jain J, Jadhav C, Hodavdekar M (2020) Credit card fraud detection technique using hybrid approach: an amalgamation of self organizing maps and neural networks. Int Res J Eng Technol (IRJET) 7(2020)   
29. Voican O (2021) Credit card fraud detection using deep learning techniques. Inf Econ 25(1)   
30. Nguyen TT, Tahir H, Abdelrazek M, Babar A (2020) Deep learning methods for credit card fraud detection. arXiv preprint arXiv:2012.03754   
31. Van Belle R, Baesens B, De Weerdt J (2023) CATCHM: a novel network-based credit card fraud detection method using node representation learning. Decis Support Syst 164:113866   
32. RamaKalyani K, UmaDevi D (2012) Fraud detection of credit card payment system by genetic algorithm. Int J Sci Eng Res 3(7):1–6   
33. Jain Y, Tiwari N, Dubey S, Jain S (2019) A comparative analysis of various credit card fraud detection techniques. Int J Recent Technol Eng 7(5):402–407   
34. Phua C, Lee V, Smith K, Gayler R (2010) A comprehensive survey of data mining-based fraud detection research. Artif Intell Revi 33(3):229–246   
35. Pearl J (2000) Models reasoning and inference. vol 19, Cambridge University Press, Cambridge, p 3   
36. Pearl J, Mackenzie D (2018) The book of why: the new science of cause and effect. Basic Books, New York   
37. He K, Zhang X, Ren S, Sun J (2016) Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition, pp 770–778   
38. Chen T, Kornblith S, Norouzi M, Hinton G (2020) A simple framework for contrastive learning of visual representations. In: International conference on machine learning, pp 1597–1607. PMLR   
39. Dal Pozzolo A, Caelen O, Johnson RA, Bontempi G (2015) Calibrating probability with undersampling for unbalanced classification. In: 2015 IEEE symposium series on computational intelligence, pp 159–166. IEEE   
40. Perozzi B, Al-Rfou R, Skiena S (2014) Deepwalk: Online learning of social representations. In: Proceedings of the 20th ACM SIGKDD international conference on knowledge discovery and data mining, pp 701–710   
41. Prusti D, Rath SK (2019) Fraudulent transaction detection in credit card by applying ensemble machine learning techniques. In: 2019 10th international conference on computing, communication and networking technologies (ICCCNT), pp 1–6. IEEE   
42. Dheepa V, Dhanapal R (2012) Behavior based credit card fraud detection using support vector machines. ICTACT J Soft Comput 2(4):391–397   
43. Sasank JS, Sahith GR, Abhinav K, Belwal M (2019) Credit card fraud detection using various classification and sampling techniques: a comparative study. In: 2019 international conference on communication and electronics systems (ICCES), pp 1713–1718. IEEE   
44. Le T-T-H, Kim H, Kang H, Kim H (2022) Classification and explanation for intrusion detection system based on ensemble trees and shap method. Sensors 22(3):1154   
45. Zhang K, Xu P, Zhang J(2020) Explainable AI in deep reinforcement learning models: a shap method applied in power system emergency control. In: 2020 IEEE 4th conference on energy internet and energy system integration (EI2), pp 711–716 . IEEE   
46. Winter E (2002) The shapley value. Handbook of game theory with economic applications, vol 3, pp 2025–2054

Publisher’s Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.