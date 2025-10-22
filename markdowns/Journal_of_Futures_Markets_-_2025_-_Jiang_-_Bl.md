# Black-Scholes Meet Imitation Learning: Evidence From Deep Hedging in China

Fuwei Jiang1 $\textcircled{1}$ 丨 Jie Kang²丨 Ruzheng Tian² 丨 Qingdong Xu³

Scholfercocteoci Economicsadgtetraiedoaogt Beijing, China

Correspondence: Qingdong Xu (qingdongxu.cn@gmail.com)

Received: 9 August 2024丨Revised: 27 January 2025丨Accepted: 23 April 2025

Keywords: deep reinforcement learning limitation learning 丨option hedging

# ABSTRACT

This paper introduces an imitation learning deep hedging (ILDH)algorithm, which bridges the Black-Scholes-Merton (BSM) model with deep reinforcement learning (DRL)to addressthe option hedging problem in incomplete real markets. By leveraging imitation learning,the DRL agent optimizes its hedging policy using both frely explored action samples based on real trading data and corrsponding action demonstrations derived from the BSM model. These demonstrations serve as data augmentation,enabling the agent to develop a meaningful policy even with arelatively small taining data set and enhancing the managementoftailrisk.Empiricalresults showthat ILDHachieves higher profit,lowerrisk,and lowercostintheChinese stock index options market,as compared with other deep hedging algorithms and traditional delta hedging method.This outperformance is robust acrosscaland put options,diferenttransactioncost conditions,andvarying levels ofrsk aversion.

JEL Classification: G130, C160

# 1Introduction

Option hedging is crucial for risk management in the finance industry.The primary method for option hedging is delta hedging, derived from the Black-Scholes-Merton (BSM) model (Black and Scholes l973;Merton 1973),where delta represents the partial derivative of the option price with respect to the underlying asset. However, delta hedging struggles with handling transaction costs, market frictions,and liquidity constraints.Moreover,the restrictive assumptions of the BSM model, such as the price of underlying asset following Geometric Brownian Motion (GBM） and implied volatility remaining constant,stray significantly from real market conditions. It turns out that well-trained traders would make their own adjustments to delta hedging to achieve better hedge performance (Buehler et al. 20l9). This implies that the execution of model-based option hedging strategies in real markets necessitates the incorporation of more complex considerations, thereby requiring manual intervention.

Recent research has shown that deep hedging,based on deep reinforcement learning (DRL),offers a promising alternative solution (Buehler et al. 20l9；Cao et al. 202l;Giurca and Borovkova 202l; Halperin 2020; Marzban et al.2023;Mikkilä and Kanniainen 2023; Zhang and Huang 202l).DRL is a machine learning algorithm that optimizes decision-making policy through trial-and-error interactions with the environment. It integrates the feature extraction ability of deep Learning (DL) with the decision-making ability of reinforcement learning (RL), enabling the development of end-to-end strategies in complex environments (Hambly et al. 2023; Singh et al.2022).This approach allows the deep hedging agent to learn decisionmaking policies for rebalancing option hedging positions in incomplete real markets without imposing any restrictive assumptions.Furthermore,compared to other data-driven and non-parametric models,deep hedging is capable of considering the consequences of decision-making, thereby more effectively addressing the transaction cost problem.

However,applying this data-driven deep hedging method in real option markets presents two distinct challenges.First,it requires a substantial number of samples to learn a meaningful policy,yet real trading data is often limited.To ensure a sufficient sample size for training,most previous studies on deep hedging have relied on model-based simulated environments or synthetic data. This method,however,imposes restrictive assumptions on the simulated environments or synthetic data and doesn't guarantee that the benefits observed in simulations will carry over to real-world applications. Second,deep hedging agent struggles with managing tail risk events due to their rare occurrence.The rarity of tail shocks restricts the agent's ability to develop effective strategies for managing them,which is a pervasive problem for all data-driven models.

This paper aims to address the challenges of limited real trading samples and tail risk problems in deep hedging by leveraging imitation learning.Imitation learning is a framework in which a DRL agent learns from both free explorations and expert demonstrations,see Hussein et al.(20l7) for a survey. By providing expert demonstrations, the agent can quickly learn tasks that are challenging for itself but relatively easy for (human) experts,such as an autonomous driving agent learning from human drivers (Le Le Mero et al. 2O22) or AlphaGo learning from Go manuals (Silver et al. 20l6). For the option hedging problem,classic theoretical models,such as the BSM model, serve as valuable sources of demonstrations to be imitated. These theoretical models are not constrained by sample quantity limitations and remain robust in handling rare shocks. Thus，by imitating demonstrations from these theoretical models,the issues of sample scarcity and tail risks in deep hedging can be effectively mitigated.

We propose an imitation learning deep hedging (ILDH) algorithm to implement the above idea. During the training process of ILDH, for each option hedging episode, BSM-based hedging position demonstrations are included alongside the hedging positions made by the DRL agent. This training process allows the agent to optimize its neural networks,that is,learn its option hedging policy,using a combined data set of BSM-based demonstrations and agent exploration samples.This approach is analogous to how human traders might reference BSM-based option hedging positions to refine their own hedging positions. Essentially, this technique acts as a form of data augmentation, using BSM-based demonstrations to address the limitations of real trading data.Moreover, since the BSM model is robust to tail shocks,imitating BSM-based demonstrations can enhance the deep hedging agent's ability to manage rare tail shocks. Our results show that the ILDH algorithm significantly increases option hedging returns while reducing both return volatility and tail losses.

We further extend the information available to the agent from just option current state to historical trading information,by adding a memory structure to the ILDH algorithm.This extension is motivated by the recognition that option hedging is more accurately modeled as a partially observable Markov decision process (POMDP) rather than a Markov decision process (MDP). Traditionally,DRL requires the decision-making problem to follow MDP,which assumes that all relevant information needed for optimal decision-making should be captured in the current state (Wang et al. 2022). However, we can only observe limited option hedging information in option hedging problem,such as option price,underlying asset price, and time to maturity (TTM).Crucial but hardly observable information, such as investor sentiment,market liquidity,and external economic shocks,remain hidden within the price sequence formed by transactions, significantly impacting option hedging performance.Furthermore,option prices and implied volatility tend to correlate with past underlying asset returns (Bollerslev 2006; Christie 1982; French et al. 1987;Hull and White 2Ol7), implying that past underlying asset price information impacts the future option hedging performance. Thus, we model the option hedging process as a POMDP rather than an MDP and integrate a memory structure into the neural network of ILDH algorithm.This approach allows the DRL agent to better infer the non-fully observable current state in option hedging and make more informed hedging decisions.

Our ILDH algorithm consists of three components: a core DRL algorithm,an imitation learning module,and a memory structure.First,we use the twin delayed deep deterministic policy gradient (TD3) algorithm as the core ILDH component. TD3 is well-suited for handling problems with continuous state and action spaces,making it ideal for option hedging.Its use of clipped double Q-learning and delayed policy updates helps prevent overly rapid iterations and improves robustness against noisy feedback,which is particularly beneficial for the highly noisy financial time series data. Second,our imitation learning module employs the behavior cloning (BC) algorithm,specifically the DAgger algorithm (Ross et al. 20l1). This algorithm combines BSM-based demonstrations with agent exploration samples and stores them in an experience replay buffer for resampling and training.Thirdly,we integrate a recurrent neural network (RNN) structure into the TD3 algorithm's neural networks to incorporate historical trading information into the agent's decision-making process.

In our empirical study,we follow previous research (Mikkilä and Kanniainen 2023; Zhang and Huang 2021) by using intraday 30-min high-frequency data to train and test the ILDH algorithm.The algorithm is highly adaptable to real market conditions and can handle various hedging frequencies and option types,as these differences can be managed through data normalization and hedging environment setting adjustments. The chosen 30-min hedging frequency aligns with the frequent rebalancing required for delta hedging and reflects typical trading patterns of human traders.In addition,to assess the reliability of ILDH under limited trading data conditions and ensure consistency between the training and testing environments,we deliberately choose a shorter training data set length. Specifically,we focus on the CSI 3oo stock index option, the most actively traded stock index option in the Chinese market, and conduct training and testing for both call and put options using 3 years of trading data.

Our results show that, for the CSI 300 stock index option data set, ILDH outperforms both the latest purely data-driven deep hedging algorithm (empirical deep hedging,EDH) (Mikkilä and Kanniainen 2023） and model-driven deep hedging (MDDH) approach,significantly surpassing the classic BSM-based delta hedging strategy.We also assess ILDH's performance under different transaction cost ratios and risk aversion levels, finding that it maintains robust performance across these diferent market conditions for both call and put options.Finally,we conduct ablation experiments to demonstrate the contributions of imitation learning and memory structure to the overall performance of ILDH.

The paper is structured as follows: Section 2 provides a literature review on deep hedging.In Section 3,we model the option hedging process as a POMDP and develop the ILDH algorithm proposed in this paper. Section 4 provides the details of the data and presents the empirical results.The final section concludes this paper.

# 2|Literature Review

Our study directly contributes to a growing literature on deep hedging from algorithmic innovation and practical application feasibility. Deep hedging,first proposed by Buehler et al. (2019) to address option hedging in the presence of transaction costs, shows superior hedging performance over the BSM model in simulated markets.Following this,numerous studies have emerged, focusing on algorithmic improvements (Cao et al.2021; Du et al. 2020; Halperin 2019,2020; Kolm and Riter 2019; Zhang and Huang 2021),combining deep hedging with other option hedging frameworks (Carbonneau and Godin 2O2l;Horvath et al.2021; Lütkebohmert et al. 2022),and extending its application to similar derivatives (Carbonneau 2O21).Despite these advancements,the empirical part of these studies is limited to simulated environments or synthetic data to ensure sufficient training samples.While this method meets the data requirements of DRL,it still imposes restrictive assumptions inherent in model-based simulations and fails to fully exploit the model-free advantages of DRL.Following Mikkilä and Kanniainen (2023), our sturdy improves the practical application feasibility of deep hedging by using real trading data for training and testing, allowing the agent to learn strategies directly from real market environments.By leveraging BSM-based demonstrations through imitation learning,we address the challenge of limited real trading data and accelerate the training of deep hedging. Moreover,unlike previous deep hedging approaches,our approach does not restrict the hedging options to be at-the-money or of particular moneyness, thereby considerably expanding the range of deep hedging applications.Additionally,we are the first to validate that deep hedging empirically performs equally well for put options as compared to call options.

This paper is related to the extensive application of machine learning in option pricing and hedging(Amilon 20o3;Garcia and Gencay 2000; Hutchinson et al. 1994; Ivascu 2021; Huang et al.2022, 2023,Hong et al. 2024). Instead of deriving option price from no-arbitrage or other economic and statistical assumptions,these studies consider option price as a functional mapping output based on information about the option and the underlying asset (Ivascu 2021).By doing so, they enable the direct use of data-driven non-parametric models for option pricing and avoid errors caused by the inappropriate assumptions of traditional models. Our work builds upon the idea of these machine learning algorithms and introduces the prior knowledge of classic parametric models through imitation learning,while still retaining the advantage of the data-driven approach in avoiding errors from restrictive assumptions.

This paper also contributes to the application of reinforcement learning in solving decision-making problems in economics and finance. DRL algorithms have been widely applied to solving the problem of optimal policy in macroeconomics, game theory, and finance; see Charpentier et al. (2023); Hambly et al. (2023) for surveys.Unlike traditional methods that rely on stochastic process modeling，DRL algorithms address these problems directly through interaction with the environment,offering rich flexibility in handling complex environments.However, it also makes DRL prone to over-fitting and less efective in dealing with unseen scenarios. In contrast, traditional theoretical models provide valuable insights but struggle to fit real-world complexity.Imitation learning emerges as a promising method that bridges the gap between these two paradigms. Our work demonstrates how imitation learning facilitates the application of deep hedging to incomplete real markets with limited trading data.Importantly, the applicability of imitation learning can be extended beyond option hedging to various similar problems, such as portfolio optimization and trade execution.Remarkably,imitation learning does not necessarily require a wellestablished theoretical model for the specific problem at hand; it merely relies on demonstrations,which can originate from either classic models or human experts.

# 3」 Methodology

In this section，we model the option hedging process as a POMDP and define the reward function for optimizing the DRL decision-making policy. We then derive the RNN-TD3 algorithm, showing how to learn an option hedging strategy using the DRL algorithm.Lastly,we propose the ILDH algorithm, which bridges the BSM model with the DRL algorithm through imitation learning.

# 3.1| Option Hedging Process for DRL

A POMDP can be defined as a tuple: $\{ S , A , T , R , \Omega , O \}$ including the state space $S$ ，action space $A$ ,transition probability model $T$ ,reward function $R$ ，observation model $\Omega$ ，and observation space $O$ .At each time step, the environment is in a state $s _ { t } \in S$ ,the agent takes action $a _ { t } \in A$ based on the observation $o _ { t } \in O$ of the state $s _ { t }$ . The environment then transitions to a new state $s _ { t + 1 }$ with probability $T ( s _ { t + 1 } | s _ { t } , a _ { t } )$ ,while the agent receivesaconsequentreward $r _ { t + 1 } = R ( s _ { t } , a _ { t } , s _ { t + 1 } )$ and the next observation $o _ { t + 1 }$ with probability $\Omega ( o _ { t + 1 } | s _ { t + 1 } , a _ { t } )$ . The transition probability model $T$ and observation model $\Omega$ are determined by the environment and remain unobservable,existing within market trading data in the context of the option hedging problem. Since the states in POMDP are also unobservable,we proceed to define action space,observation space,and reward function.

We begin by defining the option hedging portfolio and its profit and loss (P&L).Following Kolm and Ritter (2019) and Mikkilä and Kanniainen (2023),we assume that the hedging portfolio shorts one unit of the option at the initial time $t = 0$ of each episode.Between $t = 0$ and $t = T - 1$ ,the DRL agent decides the number of units of the underlying asset to hold at each time $t$ ,denoted as $n _ { t }$ .At the end of the episode $t = T$ ,it sells the held units of the underlying asset and simultaneously buys back the corresponding units of the option to close the position. Assuming that the price of the underlying asset at time $t$ is $S _ { t }$ and the price of the option is $P _ { t }$ ,the P&L of the hedging portfolio at time $t$ contains two parts.The first part is the transaction cost $C o s t _ { t }$ incurred when adjusting the holdings from $n _ { t - 1 }$ to $n _ { t }$ . The second part is the interval return from holding $n _ { t }$ units of the underlying asset and $^ { - 1 }$ unit of the option until $t + 1$ .The formula for P&L at time $t$ can be expressed as follows:

$$
P n L _ { t } = - C o s t _ { t } + [ - ( P _ { t + 1 } - P _ { t } ) + n _ { t } \times ( S _ { t + 1 } - S _ { t } ) ] ,
$$

where the transaction cost $C o s t _ { t }$ ,given a transaction cost rate $c$ ， is defined as1:

$$
C o s t _ { t } = \left\{ \begin{array} { l l } { c \times n _ { 0 } \times S _ { 0 } \quad } & { t = 0 } \\ { c \times \vert n _ { t } - n _ { t - 1 } \vert \times S _ { t } \quad t > 0 } \end{array} \right. .
$$

# Action Space:

Based on the above option hedging process,the action of the deep hedging agent,at time $t$ ,is simply the number of units of the underlying asset $n _ { t }$

$$
a _ { t } = ( n _ { t } ) .
$$

# Observation Space:

Previous studies(Cao et al. 202l; Giurca and Borovkova 202l；Halperin 2019；Kolm and Ritter 2019; Mikkila and Kanniainen 2023） typically construct the observation space basically based on the information necessary for delta hedging according to the BSM model, including moneyness $\left( K / S _ { t } \right)$ ,TTMt，implied volatility $( I V _ { t } )$ and current held units of the underlying asset $\left( n _ { t - 1 } \right)$ . In this paper, we expand upon the above information by including option price $\left( P _ { t } \right)$ ，, resulting in the following deep hedging observation space:

$$
o _ { t } = \left\{ \begin{array} { l } { \left( \frac { K } { S _ { 0 } } , T T M _ { 0 } , I V _ { 0 } , 0 , P _ { 0 } \right) } \\ { \left( \frac { K } { S _ { t } } , T T M _ { t } , I V _ { t } , n _ { t - 1 } , P _ { t } \right) } \end{array} \right. .
$$

Here, $n _ { t - 1 }$ assists the agent in understanding the current hedging position, while $P _ { t }$ is provides additional information. This enables the ILDH agent, through its memory structure,to more accurately infer the current state or predict future option prices based on the available observation information.

# Reward Function:

The objective of the agent's decision-making policy at each step is to maximize its expected cumulative reward.Hence,the formulation of the reward function plays a decisive role in DRL.

Drawing from the above option hedging process,our hedging optimization objective is simply maximizing the expected profit $\mathbb { E } ( P n L )$ ,while minimizing the risk,as measured by the standard deviation of the profit $\mathrm { S D } ( P n L )$ . It can be expressed by the following hedging objective:

$$
\operatorname* { m a x } \mathbb { E } ( P n L ) - \xi \times \operatorname { S D } ( P n L ) .
$$

Here, $\xi$ denotes the risk-aversion ratio.However, $\mathbb { E } ( P n L )$ and $\mathrm { S D } ( P n L )$ can only be estimated at the end of each hedging episode,resulting in extremely sparse reward feedback that only occurs at the last action. Sparse rewards pose a significant challenge in training the agent in DRL,as the agent requires real-time feedback for each action $a _ { t }$ to enhance policy training efficiency.Hence,following Mikkilä and Kanniainen (2023),we construct the reward function $r _ { t + 1 }$ based on the portfolio's $P n L _ { t }$

$$
r _ { t + 1 } = \left\{ \begin{array} { l l } { P n L _ { t } - \xi \times | P n L _ { t } | } \\ { P n L _ { t } - \xi \times | P n L _ { t } | + \mathbb { E } ( P n L ) } & { \ t < T - 1 } \\ { \quad - \xi \times \mathrm { S D } ( P n L ) } \end{array} \right.
$$

In each step,we use $| P n L _ { t } |$ rather than $( P n L _ { t } ) ^ { 2 }$ as the substitute for the incomputable $\mathrm { S D } ( P n L )$ to better maintain the stability of reward.This reward function aims to guide the agent to maximize the positive profit while minimizing the fluctuation of the profit in each step.Moreover,at the end of the hedge episode, the agent should consider both the expected profit and risk, thereby orienting the agent's policy to the option hedging optimization objective.

In summary, the process of option deep hedging defined in this paper can be illustrated in Figure 1.

# 3.2 RNN-TD3 Algorithm

This section introduces the RNN-TD3 algorithm,which serves as the core neural network structure for ILDH.Based on the option hedging process described earlier,we will present the objective of the RNN-TD3 algorithm,explaining how it optimizes the agent's action policy and implement it into a neural network framework.

In DRL,the goal of the agent is to learn a decision-making policy $a = \pi ( s )$ that maximizes the expected cumulative reward,without prior knowledge of the state transition probability function and the reward function of the environment. Specifically,the agent aims to maximize the discounted expected cumulative reward, denoted as $\begin{array} { r } { \mathbb { E } \left( \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } r _ { t + k + 1 } \right) } \end{array}$ where y denotes the discount factor,representing the agent's consideration of future potential rewards influenced by the current action.

To find out the optimal strategy,or to evaluate a policy,we can define the state value function $V _ { \pi } ( s _ { t } )$ ，which represents the discounted expected cumulative reward when facing state $s _ { t }$ at time $t$ and following the policy $\pi$ afterward.

![](images/ddcaf53eef255ed18eb8073285e259e903925494ffc06831e4f5fab96545dac2.jpg)

Hedging Process:

![](images/f331a1b3a074dce8c3c86235e56701c5cb976a5d2a100557b25dd14dceb47e48.jpg)  
FIGURE 1| Deep hedging diagram.

$$
V _ { \pi } ( s _ { t } ) = \mathbb { E } \left[ \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } r _ { t + k + 1 } \middle | s _ { t } , \pi \right] ,
$$

$$
= \mathbb { E } [ r _ { t + 1 } + \gamma V _ { \pi } ( s _ { t + 1 } ) \vert s _ { t } , \pi ] ,
$$

$$
\begin{array} { r } { = \sum _ { a _ { t } } \pi ( s _ { t } ) \sum _ { s _ { t + 1 } } P ( s _ { t + 1 } | s _ { t } , a _ { t } ) ( r _ { t + 1 } + \gamma V _ { \pi } ( s _ { t + 1 } ) | s _ { t } , \pi ) . } \end{array}
$$

Then, the optimal policy $\pi ^ { * }$ is the one that maximizes the state value for all states:

$$
V _ { \pi ^ { * } } ( s ) \geq V _ { \pi } ( s ) \quad \forall s , \pi .
$$

Similarly, the action-value function $Q _ { \pi } ( s _ { t } , a _ { t } )$ can be defined as the discounted expected cumulative reward when facing state $s _ { t }$ at time $t$ ，choosing the action $a _ { t }$ ，and following the policy $\pi$ thereafter:

$$
Q _ { \pi } ( s _ { t } , a _ { t } ) = \mathbb { E } [ r _ { t + 1 } + \gamma V _ { \pi } ( s _ { t + 1 } ) \vert s _ { t } , a _ { t } , \pi ] ,
$$

$$
= \sum _ { s _ { t + 1 } } P ( s _ { t + 1 } | s _ { t } , a _ { t } ) [ r _ { t + 1 } + \gamma V _ { \pi } ( s _ { t + 1 } ) | s _ { t } , a _ { t } , \pi ] ,
$$

$$
= \sum _ { s _ { t + 1 } } P ( s _ { t + 1 } | s _ { t } , a _ { t } ) \Bigg [ r _ { t + 1 } + \gamma \sum _ { a _ { t + 1 } } \pi ( s _ { t + 1 } ) Q _ { \pi } ( s _ { t + 1 } , a _ { t + 1 } ) \Bigg ] .
$$

Hence,the optimal policy $\pi ^ { * }$ can be expressed as the policy that always chooses the action that maximizes the action-value function $Q _ { \pi ^ { * } } ( s , a )$ for any given state $s$

$$
\pi ^ { * } ( s ) = \arg \operatorname* { m a x } _ { \mathrm { a } } Q _ { \pi ^ { * } } ( s , a ) .
$$

The equations for the state-value funct $\mathrm { i o n } V _ { \pi } ( s _ { t } )$ and the action-value function $Q _ { \pi } ( s _ { t } , a _ { t } )$ described above are known as the Bellman equations.Once the transition probability function and reward function of the MDP system are known，the

Bellman equations can be solved using dynamic programming (DP),such as value function iteration.However,in practice, the complexity and stochastic nature of real-world problems often make it impossible to obtain the transition probabilities for all states in a MDP system.Therefore,RL algorithms were developed to address such problems.

A foundational RL algorithm is the Q-Learning algorithm (Watkins 1989),which is a type of temporal difference (TD) method. This algorithm seeks to find the optimal strategy by finding the action-value function Q. From equation(8),we can derive the objective Q-value of one step,which is the step reward plus the discounted optimal Q-value of the next step, $r _ { t + 1 } + \gamma \operatorname* { m a x } _ { \mathrm { a } _ { t + 1 } } Q ( s _ { t + 1 } , a _ { t + 1 } )$ . Thus,according to the contraction mapping theorem, the Q-function can be updated iteratively by minimizing the difference between the target and current Q-values until convergence,as follows:

$$
\begin{array} { c } { Q ( s _ { t } , a _ { t } ) \gets Q ( s _ { t } , a _ { t } ) + \alpha \left[ r _ { t + 1 } + \gamma \operatorname* { m a x } _ { a _ { t + 1 } } Q ( s _ { t + 1 } , a _ { t + 1 } ) \right. } \\ { \left. - \ Q ( s _ { t } , a _ { t } ) \right] , } \end{array}
$$

where $\alpha$ denotes the step size of updating,also known as the learning rate.

In practical computation,for problems with discrete and finite state and action spaces,two-dimensional tabular arrays can be used to store and iteratively update the Q-function.However, for continuous or high-dimensional state-action spaces,such as option hedging in this paper,traditional methods are insufficient. While DRL leverages DL's ability to approximate complex high-dimensional functions,making it a potential solution.

The deep Q-network (DQN） algorithm，one of the most important and earliest DRL algorithms proposed by Mnih et al. (2013) at DeepMind,combines Q-learning with DL to extend its applicability to continuous state spaces.DQN use a convolutional neural network(Q-network) to approximate the actionvalue function (Q-function). Similar to the iterative process of Q-learning, the loss function for the Q-network is as follows:

$$
\begin{array} { r l } & { L ( \theta ^ { Q } ) = \mathbb { E } _ { ( s _ { t } , a _ { t } , r _ { t + 1 } , s _ { t + 1 } ) \sim U ( D ) } } \\ & { \quad \left[ \left( r _ { t + 1 } + \gamma \operatorname* { m a x } _ { a _ { t + 1 } } Q \big ( s _ { t + 1 } , a _ { t + 1 } | \theta ^ { Q ^ { ' } } \big ) - Q ( s _ { t } , a _ { t } | \theta ^ { Q } ) \right) ^ { 2 } \right] , } \end{array}
$$

where $U ( D )$ denotes the distribution of MDP samples $( s _ { t } , a _ { t } , r _ { t + 1 } , s _ { t + 1 } )$ ,and $\theta ^ { Q }$ and ${ \boldsymbol { \theta } } ^ { Q ^ { \prime } }$ are the parameters of training Q-network and target Q-network,respectively. The target network is periodically updated from the training Q-network during the iteration process.In addition, the algorithm employs an experience replay technique to improve sample efficiency，by cutting the MDP process into step samples $( s _ { t } , a _ { t } , r _ { t + 1 } , s _ { t + 1 } )$ ,then storing them into a replay buffer,and randomly sampling batches from the replay buffer to train the Q-network. This technique underpins later advancements in DRL algorithms,including the one used in this paper.

In contrast to the iteratively updating approach of DQN, the deterministic policy gradient (DPG) algorithm proposed by OpenAI (Silver et al. 20l4) focuses on learning the optimal deterministic policy $\pi _ { \theta } ( s )$ directly from the perspective of parameterized policy functions,through applying the gradient ascent method to maximize the Q-values.The optimization objective function $J ( \pi _ { \theta } )$ is based on the expected cumulative reward under the deterministic policy $\pi _ { \theta } ( s )$ defined as:

$$
\begin{array} { r } { J ( \pi _ { \theta } ) = \int _ { S } \rho ^ { \pi } ( s ) r ( s , \pi _ { \theta } ( s ) ) \mathrm { d } s , } \end{array}
$$

$$
= \mathbb { E } _ { s \sim \rho ^ { \pi } } [ r ( s , \pi _ { \theta } ( s ) ) ] ,
$$

where $\rho ^ { \pi }$ denotes the state transition probability under $\pi _ { \theta } ( s )$ According to the DPG theory,for MDPs where Q-functions and policy function gradients exist,the policy gradient of the objective function can be expressed as:

$$
\nabla _ { \theta ^ { \pi } } J \left( \pi _ { \theta } \right) = \int _ { S } \rho ^ { \pi } \left( s \right) \nabla _ { \theta } \pi _ { \theta } ( s ) \nabla _ { a } Q ^ { \pi } \left( s , a \vert _ { a = \pi _ { \theta } ( s ) } \right) \mathrm { d } s ,
$$

$$
= \mathbb { E } _ { s \sim \rho ^ { \pi } } \left[ \nabla _ { \theta } \pi _ { \theta } ( s ) \nabla _ { a } Q ^ { \pi } \left( s , a \vert _ { a = \pi _ { \theta } ( s ) } \right) \right] .
$$

Hence,the optimal policy can be iteratively updated using policy gradient ascent.In practical policy iteration,the DPG algorithm is typically combined with the widely-used actorcritic neural network architecture. The critic network （Qfunction) is updated iteratively using traditional Q-learning algorithms, such as the SARSA algorithm (Rummery and Niranjan 1994),to evaluate the policy. Meanwhile,the Actor network (the deterministic policy function $\pi _ { \theta , \ l }$ ）isupdated through random sampling and policy gradient ascent to select the optimal action.

Inspired by DQN and based on DPG, the deep DPG (DDPG) algorithm (Lillicrap et al. 2Ol5） uses neural networks to approximate both the Q-function and the policy function. Similar to DQN,the Critic network parameters $\theta ^ { Q }$ are updated by minimizing the following loss function $L ( \theta ^ { Q } )$ ，while the Actor network parameters $\theta ^ { \pi }$ are updated via policy gradient ascent:

$$
\begin{array} { r l } & { L ( \theta ^ { Q } ) = \mathbb { E } _ { \{ ( s _ { t } , a _ { t } , r _ { t + 1 } , s _ { t + 1 } ) i \} i = 1 } } \\ & { \quad \bigg [ \bigg ( r _ { t + 1 } + \gamma Q \Big ( s _ { t + 1 } , \pi \left( s _ { t + 1 } | \theta ^ { \pi ^ { ' } } \right) | \theta ^ { Q ^ { ' } } \Big ) - Q ( s _ { t } , a _ { t } | \theta ^ { Q } ) \bigg ) ^ { 2 } \bigg ] , } \end{array}
$$

$$
\begin{array} { r l } & { \nabla _ { \theta } \pi J \approx \mathbb { E } _ { \{ ( s _ { t } , a _ { t } , r _ { t + 1 } , s _ { t + 1 } ) _ { i } \} _ { i = 1 } ^ { N } } \Big [ \nabla _ { a } Q ( s , a | \theta ^ { Q } ) | _ { s = s _ { t } , a = \pi ( s _ { t } ) } } \\ & { \nabla _ { \theta ^ { \pi } } \pi ( s | \theta ^ { Q } ) | _ { s = s _ { t } } \Big ] , } \end{array}
$$

where $\{ ( s _ { t } , a _ { t } , r _ { t + 1 } , s _ { t + 1 } ) _ { i } \} _ { i = 1 } ^ { N }$ denotes the batch of samples from the replay buffer, and ${ \theta } ^ { Q ^ { ' } }$ and $\theta ^ { \pi }$ are the target networks during iteration process,which are updated from $\theta ^ { Q }$ and $\theta ^ { \pi }$ with a soft update parameter $\tau$ ：

$$
\begin{array} { r c l } { { } } & { { } } & { { \theta ^ { Q ^ { ' } }  \tau \theta ^ { Q } + ( 1 - \tau ) \theta ^ { Q ^ { ' } } , } } \\ { { } } & { { } } & { { } } \\ { { } } & { { } } & { { \theta ^ { \pi ^ { ' } }  \tau \theta ^ { \pi } + ( 1 - \tau ) \theta ^ { \pi ^ { ' } } . } } \end{array}
$$

Building on DDPG,the TD3 algorithm (Fujimoto et al. 2018) was proposed to enhance its effciency by addressing practical issues encountered during its application.The key improvements include,deploying clipped double Q-learning to mitigate Q-value overestimation,reducing the update frequency of the actor network to avert rapid but blind iteration,and using target policy smoothing to prevent the Q-function from being limited by wrong peaks. These improvements enable the TD3 algorithm to outperform other algorithms in most continuous space DRL tasks,making it the primary choice for the application of deep hedging in this paper.

As discussed earlier,a memory structure should be integrated into the DRL algorithm to address the option hedging problem as a POMDP.This improves the agent's decision-making capabilities by providing access to historical state (observation)-action data.Inspired by Meng et al.(202l),and based on TD3,this paper introduces the RNN-TD3 algorithm as the central neural network structure of ILDH.

The structure of the RNN-TD3 is illustrated in Figure 2. The agent's memory consists of historical observations and actions,denoted as $h _ { t } ^ { l } = o _ { t - l } , a _ { t - l + 1 } , . . . , o _ { t } , a _ { t - 1 }$ . The sequential historical information $h _ { t } ^ { l }$ is preprocessed with a RNN structure before being merged with the current information $o _ { t }$ and $a _ { t }$ for input into the Actor and Critic networks. Similar to the structure of DDPG, the Critic network $Q ( o _ { t } , a _ { t } , h _ { t } ^ { l } )$ in RNN-TD3 can be optimized by minimizing the difference between the current $\mathrm { Q }$ -value and the target Q-value.Its loss function is defined as:

$$
\begin{array} { r l } & { L ( \theta ^ { Q _ { j } } ) = \mathbb { E } _ { \left\{ \left( h _ { t } ^ { l } , o _ { t } , a _ { t } , r _ { t + 1 } , o _ { t + 1 } \right) _ { i } \right\} _ { i = 1 } ^ { N } } } \\ & { \qquad \left( r _ { t + 1 } + \gamma \displaystyle \operatorname* { m i n } _ { j = 1 , 2 } Q _ { j } ^ { \prime } \Big ( o _ { t + 1 } , a ^ { - } , h _ { t + 1 } ^ { l } \Big ) - Q _ { j } \Big ( o _ { t } , a _ { t } , h _ { t } ^ { l } \Big ) \right) ^ { 2 } . } \end{array}
$$

![](images/0594e5d519bbb31a02d121db072e19b7e4749ef7ccc56f817500e587a532abbe.jpg)  
FIGUR2丨Thisfigureplots theactor-riticnetworkstructureofNN-TD3.FCdnotesthefullyoectedlayersofneuralnetwoksRN denotes the layers of recurrent neural networks.

Here, $a ^ { - } = \pi ( o _ { t + 1 } , h _ { t + 1 } ^ { l } ) + \epsilon$ denotes the policy action plus the random noise $\boldsymbol { \mathfrak { t } } \sim \mathrm { c l i p } ( \mathbb { N } ( 0 , \sigma ) , - c , c )$ ，where $\sigma$ and $c$ are the standard deviation and noise clipping range parameters, respectively. $Q _ { j = 1 , 2 }$ refers to the two critic networks in double Q-learning.

Correspondingly, the objective of the Actor network $\pi ( o _ { t } , h _ { t } ^ { l } )$ is to maximize the $\mathrm { Q }$ value evaluated by the Critic network. It is defined by Equation (19),which would be updated through gradient ascent,as discussed above.

$$
\operatorname* { m a x } _ { \theta ^ { \pi } } \mathbb { E } _ { \left\{ \left( h _ { t } ^ { l } , o _ { t } , \right) _ { i } \right\} _ { i = 1 } ^ { N } } Q _ { j } \left( o _ { t } , \pi \left( o _ { t } , h _ { t } ^ { l } \right) , h _ { t } ^ { l } \right) .
$$

# 3.3| Imitation Learning and Delta Hedging

In this section,we introduce the imitation learning algorithm used in this paper and propose the ILDH algorithm.

# Algorithm1DAGGER

Initial dataset of demonstrations $\mathcal { D }$ , and expert policy $\pi ^ { * }$   
Intial policy $\hat { \pi } _ { 1 }$ and $\{ \beta _ { i } \}$ such hat $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \beta _ { i } ^ { \setminus } \to 0 } \end{array}$ .   
for $i = 1$ to $N$ Let $\pi _ { i } = \beta _ { i } \pi ^ { * } + ( 1 - \beta _ { i } ) \hat { \pi } _ { i }$ Sample $T$ -step trajectories using $\pi _ { i }$ Get dataset $\mathcal { D } _ { i } = \{ ( s , \pi ^ { * } ( s ) ) \}$ of visited states by $\pi _ { i }$ and actions given by expert. Aggregate datasets: $\mathcal { D }  \mathcal { D } \cup \mathcal { D } _ { i }$ ： Train classifier $\hat { \pi } _ { i + 1 }$ on $\mathcal { D }$ ：   
end for

The ILDH algorithm is primarily inspired by the Dagger (Data set Aggregation) algorithm,a specific type of BC imitation learning algorithm proposed by Ross et al.(20ll).The Dagger algorithm allows the agent to explore and collect state samples based on its own policy，while simultaneously receiving expert demonstrations for each encountered state.

These exploration samples and expert demonstrations are then aggregated to form a data set,which is used to train the agent's policy. This approach,akin to a Follow-The-Leader method, significantly reduces the amount of training data required to achieve satisfactory performance.Ross et al. (2011） have shown that the Dagger algorithm outperforms other imitation learning algorithms,particularly in datalimited environments.

The ILDH algorithm draws on the Dagger algorithm,with the BSM model serving as the“expert”model.By utilizing the Dagger algorithm,demonstrations derived from the BSM model enable the agent to converge more quickly to a meaningful hedging strategy from its purely random initial strategy.This demonstration guidance effect is particularly valuable in options hedging,where financial time series are highly noisy and direct reward-based learning is less effective compared to nonfinancial tasks.In addition,the “expert” demonstrations can be replaced with alternative model-based hedging strategies,such as those based on the SABR or Heston model. We choose the BSM delta hedging strategy as the source of demonstrations due to its dominant position in the industry and its long-standing validation.Further details of the BSM model-based delta hedging strategy are provided in Appendix A.

At last, the ILDH algorithm is proposed in Algorithm 2. Given that the trajectories in historical real market trading data are fixed,the agent's actions will influence the hedging positions and rewards,but will not affect the transitions of other state information,such as option prices or TTM.Hence,to accelerate the training process of ILDH,we precompute all BSM-based demonstrations within the training data set and store them in the experience replay buffer for ILDH agent training. Notably, during the agent's testing phase,the corresponding demonstrations are withheld to preserve the integrity of the testing conditions and prevent the disclosure of option pricing information.Hyperparameter details for the ILDH algorithm are provided in Table Al in Appendix.

# Algorithm 2 Imitation Learning Deep Hedging

Initialize critic networks $Q _ { \theta _ { 1 } }$ ， $Q _ { \theta _ { 2 } }$ , and actor network $\pi _ { \phi }$ with random parameters $\theta _ { 1 }$ $\theta _ { 2 }$ $\phi$ （201   
Initialize target networks $\theta _ { t a r g e t , 1 }  \theta _ { 1 }$ $\theta _ { t a r g e t , 2 }  \theta _ { 2 }$ $\phi _ { t a r g e t }  \phi$   
Initialize replay buffer $\boldsymbol { \it { \it B } }$   
for each training episode: Reset environment and receive initial observation $o _ { 0 }$ for $t = 0$ to $T - 1$ Select action with BSM model $a _ { t } = B S M . d e l t a ( o _ { t } )$ Scale $a _ { t } = s c a l e ( a _ { t } , a _ { L o w } , a _ { H i g h } )$ to proper size Execute action $a _ { t }$ and observe reward $r _ { t + 1 }$ and next observation $o _ { t + 1 }$ Store $( o _ { t } , a _ { t } , r _ { t + 1 } , o _ { t + 1 } )$ in replay buffer $\mathcal { B }$ end for   
end for   
for each training episode: Reset environment and receive initial observation $o _ { 0 }$ for $t = 0$ to $T - 1$ Select action with exploration noise $a _ { t } = \mathrm { c l i p } ( \pi ( o _ { t } , h _ { t } ^ { l } ) + \in , a _ { L o w } , a _ { H i g h } )$ , where $\mathsf { \epsilon } \sim \mathbb { N } ( 0 , \sigma ) , h _ { t } ^ { l } = o _ { t - l } , a _ { t - l - 1 } , . . . , o _ { t } , a _ { t - 1 }$ Execute action $a _ { t }$ and observe reward $r _ { t + 1 }$ and next observation $o _ { t + 1 }$ Collect and update $h _ { t + 1 } ^ { l } = o _ { t - l + 1 } , a _ { t - l } , . . . , o _ { t + 1 } , a _ { t }$ with memory length $l$ Store $( o _ { t } , a _ { t } , r _ { t + 1 } , o _ { t + 1 } )$ in replay buffer $\mathcal { B }$ end for if replay buffer $\mathcal { B }$ size $>$ Min size: Randomly sample a batch of transition $\{ ( h _ { t } ^ { l } , o _ { t } , a _ { t } , r _ { t + 1 } , h _ { t + 1 } ^ { l } , o _ { t + 1 } ) _ { i } \} _ { i = 1 } ^ { N } \mathrm { f r o m } ~ \mathcal { B }$ Compute target actions $a _ { t a r g e t , t + 1 } = \mathrm { c l i p } ( \pi _ { t a r g e t } ( o _ { t + 1 } ) + \mathrm { c l i p } ( \in , - c , c ) , a _ { L o w } , a _ { H i g h } ) , \in \sim \mathbb { N } ( 0 , \sigma )$ Compute target Q-value with target network $y ( r _ { t + 1 } , o _ { t + 1 } , \breve { h } _ { t + 1 } ^ { l } ) \dot { = } r _ { t + 1 } + \gamma \operatorname* { m i n } _ { j = 1 , 2 } \breve { Q } _ { \theta _ { t a r g e t , j } } ( o _ { t + 1 } , a _ { t a r g e t , t + 1 } , h _ { t + 1 } ^ { l } )$ Update Q-functions using gradient descent, $\begin{array} { r } { \hat { \nabla _ { \theta _ { j } } } \frac { 1 } { | N | } \sum _ { ( h _ { t } ^ { l } , o _ { t } , a _ { t } , r _ { t + 1 } , h _ { t + 1 } ^ { l } , o _ { t + 1 } ) _ { i } } ^ { \bar { N } } ( y ^ { - } ( r _ { t + 1 } , o _ { t + 1 } , h _ { t + 1 } ^ { l } ) - Q _ { j } ( o _ { t } , a _ { t } , h _ { t } ^ { l } ) ) ^ { 2 } \mathrm { f o r } \quad j = 1 , 2 } \end{array}$ if update-times mod policy-delay $= 0$ Update policy using gradient ascent $\begin{array} { r } { \nabla _ { \phi } \frac { 1 } { | N | } \sum _ { ( o _ { t } , h _ { t } ^ { l } ) _ { i } } ^ { \hat { N } } \dot { Q } _ { \theta _ { j } } ( o _ { t } , \pi ( o _ { t } , h _ { t } ^ { l } ) , h _ { t } ^ { l } ) } \end{array}$ 14 Update target networks with $\begin{array} { l } { { \theta _ { t a r g e t , j }  \tau \theta _ { j } + ( 1 - \tau ) \theta _ { t a r g e t , j } \quad \mathrm { f o r } \quad j = 1 , 2 } } \\ { { \phi _ { t a r g e t }  \tau \phi + ( 1 - \tau ) \phi _ { t a r g e t } } } \end{array}$ end if end if   
end for

# 4|Empirical Result

In this section,we first describe the CSI 300 stock index option data used for training and testing.We then present the out-ofsample testing results of the ILDH algorithm,comparing its performance with EDH,MDDH,delta hedging,and no hedging. Following this,we analyze the performance of ILDH under different transaction cost ratios and risk-aversion levels.Finally, we assess the improvement of imitation learning and memory structure on ILDH through an ablation experiment.

# 4.1Data

We conduct the training and testing of the ILDH algorithm using 30-min intraday trading data of the CSI 30o stock index option.The chosen hedging frequency of $3 0 { \cdot } \mathrm { m i n }$ (8 times per day) aligns with the frequent rebalancing required for deltahedging(Baltussen et al. 202l;Hull and White 2017; Mikkila and Kanniainen 2O23),which is typical or even low for human traders.Additionally, the inherent lower information density per interval in high-frequency data leads to more stable state transition probabilities,which align well with the environment stability requirements of DRL. The CSI 3Oo index option is selected due to its high liquidity and representativeness as the most actively traded financial option in China.

We collect all the available CSI 3oo stock index call and put option contracts and the underlying stock index data from January1,2021 to Deceber 31,2023 from the Wind database.The training data set includes the first2 years of data (January1,2021 to December 31,2022),while the testing data set covers the last year (January 1,2023 to December 31,2023).As discussed earlier,a relatively shorter training period was chosen to ensure comparability between the training and testing environments and to evaluate the ILDH algorithm with limited data.

To address the liquidity problem in option hedging transactions,several reasonable constraints are imposed on TTM and moneyness,with reference to Giurca and Borovkova (2021); Mikkila and Kanniainen (2023); Zhang and Huang (2021).Given the unpredictability of future moneyness,this paper focuses on the initial moneyness of each hedging episode,limiting it to a continuous range from O.9 to $1 . 1 ( 0 . 9 \le K / S _ { 0 } \le 1 . 1 )$ ，rather than using specific discrete values.For TTM,we restrict the data to options with a maturity between 10 and 60 trading days,prioritizing front-month contracts due to their higher liquidity.

In line with the previous settings of the option hedging process, we divide the trading data into discrete hedging episodes for training and testing.Each hedging episode lasts 1 week （5 trading days),with intraday hedging occurring every $3 0 \mathrm { { m i n } }$ ， resulting in 8 hedging steps per day and 40 hedging steps per episode. To avoid duplication of trading episodes,the trading data for call or put options that meet the TTM and moneyness constraints is divided into episodes every 5 trading days.

At last, the call option training and test sets contain 3,352 and 1,678 hedging episodes,respectively. For put options，the training and test sets consist of 3,466 and 1,699 hedging episodes,respectively.Descriptive statistics for these datasets are provided in Table A2 in Appendix.

# 4.2| Deep Hedging Empirical Performance

In this section,we analyze the empirical performance of ILDH compared to EDH as defined in Mikkilä and Kanniainen (2023), MDDH,delta hedging based on the BSM model,and no hedging as a basic benchmark.Specifically,ILDH uses both real data samples and corresponding“expert”demonstrations based on the BSM model for training.In contrast,EDH is a purely datadriven RL deep hedging model trained exclusively on real data, while MDDH is a purely model-driven RL deep hedging approach trained on simulated trading data generated by the BSM model.Details of the simulated data generation process are provided in Appendix B.

We compared the option hedging performance in terms of both profit and risk metrics of the hedging portfolio over single episodes.Profit metrics include the Reward (Equation 6),P&L (Equation 1),and Cost (Equation 2),calculated by summing the corresponding metrics of all hedging actions within each episode.Risk metrics include the standard deviation (Std P&L), Value-at-Risk (VaR,Equation 26),and Conditional Value-atRisk (CVaR,Equation 27), derived by computing the respective metrics across hedging actions per episode.

The formulas for VaR and CVaR are as follows,with the confidence level denoted as $\alpha = 9 5 \%$

$$
\operatorname { V a R } _ { \alpha } ( P n L ) = - \operatorname* { i n f } \{ X : P ( P n L \leq X ) \geq 1 - \alpha \} ,
$$

$$
\mathrm { C V a R } _ { \alpha } = - E ( X | X \leq - \mathrm { V a R } _ { \alpha } ) .
$$

Note that all metrics are averaged across all hedge episodes in the test data set. To ensure comparability between hedging episodes and mitigate distortions caused by extreme option or underlying asset prices,P&L and Cost are scaled by the underlying asset price at the beginning of each hedge episode. Consequently,derived metrics such as Reward,Std P&L,VaR, and CVaR are standardized accordingly.After scaling,P&L can be interpreted as the average weekly return (i.e., profit divided by the starting asset price)of each hedging model,given the five-trading-day episode length.Metrics other than Reward are expressed as percentages for easier comparison.

Table 1 reports the out-of-sample performance of ILDH and benchmark models using call and put option datasets. Overall, ILDH demonstrates superior performance,achieving higher returns and lower risk metrics per hedging episode compared to other models.EDH performs slightly worse than ILDH,but significantly outperforms both MDDH and delta hedging. MDDH exhibits performance close to but slightly worse than delta hedging.

In the upper panel of Table 1, ILDH achieves an average total reward of $- 0 . 5 2 7 2$ per episode in the call option data set, significantly outperforming all benchmarks. ILDH's P&L per episode is $0 . 1 0 1 0 \%$ ,higher than both MDDH $( 0 . 0 4 6 7 \% )$ and Delta Hedging $( 0 . 0 4 3 5 \%$ ),while its transaction cost is $0 . 0 2 3 6 \%$ ,lower than MDDH $( 0 . 0 2 7 5 \% )$ and Delta Hedging $( 0 . 0 3 1 7 \% )$ ，indicating that ILDH provides relatively superior hedging returns.For risk metrics, the P&L standard deviation of ILDH is $0 . 1 0 7 4 \%$ per hedging episode, with $\mathrm { V a R }$ and CVaR at $0 . 1 3 1 6 \%$ and $0 . 2 5 9 0 \%$ ,respectively. These values are the lowest among all of the other benchmark models, suggesting that ILDH yields lower portfolio volatility and tail risk. These results highlight ILDH's effectiveness in addressing the option hedging problem and its advantages over both purely datadriven and MDDH,as wellas traditional delta hedging.

When comparing the performance among the other benchmark models,EDH underperforms ILDH but demonstrates significantly higher returns and lower risks than MDDH and delta hedging.For example,EDH achieves a reward of $- 0 . 5 4 0 8$ and a P&L standard deviation of $0 . 1 1 0 7 \%$ ,close to ILDH's results.In comparison,MDDH and Delta Hedging show rewards near $- 0 . 6 5 \%$ and a P&L standard deviation of approximately $0 . 1 2 \%$ These findings suggest that data-driven RL (EDH) can derive option hedging strategies that outperform delta hedging based on the BSM model, consistent with the findings of Mikkilä and Kanniainen (2023). Moreover, it underscores the importance of training RL models on real trading data,as option hedging strategies (MDDH) optimized for simulated environments do not guarantee optimal performance in real-world conditions.

Meanwhile,MDDH's performance is comparable to but slightly worse than delta hedging,with a reward of $- 0 . 6 9 8 7$ compared to $- 0 . 6 5 5 2$ for delta hedging,and a P&L standard deviation of $0 . 1 2 1 8 \%$ versus $0 . 1 1 8 6 \%$ .It suggests that MDDH successfully learns a strategy in the BSM-simulated environment that approximates delta hedging.This finding aligns with theoretical expectations,as delta hedging is the optimal strategy in the BSM framework,and demonstrates DRL's ability to learn optimal option hedging strategies under a given environment. However, it also underscores the limitations of model-driven RL when applied to real trading data,where market conditions deviate from the assumptions of the BSM model.

TABLE1丨 Out of sample performance of ILDH,EDH,delta hedging and no hedging.   

<table><tr><td rowspan="2"></td><td colspan="6">Call options</td></tr><tr><td>Reward</td><td>P&amp;L</td><td>Std P&amp;L</td><td>Cost</td><td>VaR</td><td>CVaR</td></tr><tr><td>ILDH</td><td>-0.5272</td><td>0.1010%</td><td>0.1074%</td><td>0.0236%</td><td>0.1316%</td><td>0.2590%</td></tr><tr><td>EDH</td><td>-0.5408</td><td>0.1163%</td><td>0.1107%</td><td>0.0191%</td><td>0.1365%</td><td>0.2694%</td></tr><tr><td>MDDH</td><td>-0.6987</td><td>0.0467%</td><td>0.1218%</td><td>0.0275%</td><td>0.1688%</td><td>0.2748%</td></tr><tr><td>Delta Hedging</td><td>-0.6562</td><td>0.0435%</td><td>0.1186%</td><td>0.0317%</td><td>0.1640%</td><td>0.2676%</td></tr><tr><td>No Hedging</td><td>二</td><td>0.1823%</td><td>0.1410%</td><td>二</td><td>0.1732%</td><td>0.3480%</td></tr><tr><td></td><td colspan="6">Put options</td></tr><tr><td>ILDH</td><td>-0.5435</td><td>-0.0247%</td><td>0.1078%</td><td>0.0340%</td><td>0.1440%</td><td>0.2545%</td></tr><tr><td>EDH</td><td>-0.6110</td><td>-0.0121%</td><td>0.1098%</td><td>0.0330%</td><td>0.1542%</td><td>0.2545%</td></tr><tr><td>MDDH</td><td>-0.6392</td><td>0.0029%</td><td>0.1122%</td><td>0.0240%</td><td>0.1609%</td><td>0.2593%</td></tr><tr><td>Delta Hedging</td><td>-0.6213</td><td>0.0011%</td><td>0.1120%</td><td>0.0306%</td><td>0.1622%</td><td>0.2587%</td></tr><tr><td>No Hedging</td><td>1</td><td>-0.0738%</td><td>0.1430%</td><td>1</td><td>0.1916%</td><td>0.3289%</td></tr></table>

Note:eu g tdP&ta edgepisode hedgeepisode.Tefrmulasforallmetricspresentedinsubsequenttablesfolowthesmeayasinthistable.Tetransactionostissetto $\mathbf { c } = \mathbf { 0 . 0 3 \% }$ ,the risk averse ratio $\xi = 2$ and the confidence level of VaR and CVaR is $9 5 \%$

The lower part of Table 1 presents the performance of each model on the put option data set,where ILDH consistently achieves higher profits and lower risk metrics compared to the other models,thereby affirming ILDH's robustness across different option types.The put option data set serves as a robustness test,motivated by the following two key considerations: First,although there is call-put parity in the BSM framework,the formulas for delta calculation differ between call and put options,allowing for validating the feasibility of DRL in learning distinct hedging strategies. Second,given identical market conditions,there is a significant disparity in profitability and trend dynamics between shorting call and put options,providing the DRL agent with diverse training and testing environments.For instance,under the bear market condition in our test data set, the P&L of shorting call options is $0 . 1 8 2 3 \%$ as shown in the row of“No Hedging,”while the P&L of shorting put options is $- 0 . 0 7 3 8 \%$ .These disparities emphasize the importance of evaluating ILDH across different option types and market conditions to ensure their practical applicability.

To further validate the effectiveness of ILDH in managing tail risk,Figure 3 presents the VaR and CVaR of P&L per episode for ILDH,EDH,MDDH,and delta hedging,across different levels of implied volatility,delta,moneyness,and TTM.

The results reveal that,across all models,lower implied volatility,delta,moneyness,and TTM correspond to lower VaR and CVaR.Specifically, lower implied volatility and delta mean lower variance of the underlying asset, leading to smaller hedging errors and lower portfolio tail risk,as evidenced by the lower VaR and CVaR values. Similarly, lower moneyness and shorter TTM enhance the predictability of option price,making hedging deep in-the-money and near-expiry options less challenging and thus reducing tail risk.

Among the models,ILDH consistently achieves the lowest VaR and CVaR under all conditions,demonstrating its robustness in tail risk management. EDH performs slightly underperforms ILDH,while MDDH and delta hedging exhibit significantly higher VaR and CVaR,with comparable performance between them.Notably,the advantage of ILDH over delta hedging is particularlypronounced for different TTMlevels,where ILDH exhibits significantly lower VaR and CVaR.In contrast, the differences across other factors,such as moneyness and implied volatility,are relatively smaller. This finding suggests that the effectiveness of ILDH in tail risk control is especially sensitive to TTM,which can be attributed to its enhanced ability to forecast implied volatility and the memory structure embedded in the model.

# 4.3| Influence of Transaction Cost and Risk Averse Ratio

In this subsection,we rerun the ILDH model and other benchmark models with different transaction costs and risk aversion ratios to analyze its robustness.

Transaction costs and investor risk aversion significantly affect the optimal hedging strategy for a given option contract (Monoyios 20o4).Higher transaction costs may lead to a more serious trade-off between hedging effectiveness and the cost of adjusting option hedge positions.Meanwhile,higher risk aversion prioritizes risk control over profit maximization.In our option hedging POMDP framework,changes in transaction costs and the risk-aversion ratio directly modify the reward function for the DRL agent, guiding it toward different hedging action policies.Therefore,it is crucial to examine whether ILDH maintains its advantages over EDH,MDDH,and delta hedging under different transaction cost and risk-aversion settings.

![](images/f4c7abfae36c891dce26678457cc39f683ffcae0467c52f1f809bb6e31109bd8.jpg)  
FIGURE3isfeotthfplecllotiodggperforaefoHE,DH,ndeltadgigis CVaR,calculaedesofdailitltaosdtaritachubottstfs CVaRfteoellititetitel confidence level of VaR and CVaR is $9 5 \%$

Table 2 reports the empirical performance of ILDH, EDH,MDDH, and delta hedging,under different risk-aversion ratios $( \xi )$ . According to the reward function (Equation 6), $r _ { t } = P n L _ { t } - \xi \times | P n L _ { t } |$ where a higher risk aversion ratio $( \xi )$ directs the agent to focus on minimizing P&L fluctuations rather than maximizing positive P&L. In other words,as $\xi$ increases,the volatility of the hedging portfolio's P&L should decrease,leading to lower standard deviation, VaR, and CVaR.The results in Table 2 support this expectation: as the risk-aversion ratio $( \xi )$ increases,the standard deviation of P&L, VaR,and CVaR progressively decrease.

Comparing ILDH with other benchmark models,the empirical results in Table 2 demonstrate that ILDH consistently outperforms EDH,MDDH,and delta hedging across different risk aversion levels. For $\xi > 0$ ,ILDH excels in risk control metrics, such as the lowest standard deviation of P&L,VaR,and CVaR,although it shows marginally lower performance in profit metrics (e.g.,P&L and Cost) compared to EDH. Nevertheless, ILDH achieves significantly higher P&L than delta hedging,demonstrating a wellbalanced trade-off between risk control and profitability.

However, when $\xi = 0$ , where the DRL agent focuses exclusively on maximizing positive P&L without penalizing risk,both ILDH and EDH exhibit higher risk metrics (e.g., Std P&L, VaR, and CVaR） compared to delta hedging，without achieving superior profits.This highlights the importance of selecting an appropriate risk-aversion ratio to guide the DRL agent toward a rational and effective option hedging strategy.

Table 3 provides the empirical performance of ILDH,rerun EDH,and delta hedging,under different transaction cost ratios. According to the P&L calculation (Equation 1) and Cost calculation (Equation 2),an increase in the transaction cost ratio directly raises the cost of adjusting hedging positions,which affects the optimal hedging action policy. The results in Table 3 show that,as the transaction cost ratio increases,both Cost and P&L change proportionally,while risk metrics such as Std-P&L, VaR,and CVaR show only minor fluctuations.

TABLE 2|Performance of ILDH and EDH under different risk-averse ratio.   

<table><tr><td></td><td>Risk-averse ratio</td><td>Reward</td><td>P&amp;L</td><td>Std P&amp;L</td><td>Cost</td><td>VaR</td><td>CVaR</td></tr><tr><td>ILDH</td><td>=0</td><td>0.0066</td><td>0.0647%</td><td>0.1270%</td><td>0.0332%</td><td>0.1758%</td><td>0.2926%</td></tr><tr><td rowspan="6"></td><td>=1</td><td>-0.2668</td><td>0.0959%</td><td>0.1098%</td><td>0.0309%</td><td>0.1359%</td><td>0.2657%</td></tr><tr><td>=2</td><td>-0.5272</td><td>0.1010%</td><td>0.1074%</td><td>0.0236%</td><td>0.1316%</td><td>0.2590%</td></tr><tr><td>=3</td><td>-0.8065</td><td>0.0969%</td><td>0.1075%</td><td>0.0251%</td><td>0.1330%</td><td>0.2587%</td></tr><tr><td>=5</td><td>-1.3399</td><td>0.0989%</td><td>0.1073%</td><td>0.0235%</td><td>0.1324%</td><td>0.2578%</td></tr><tr><td>m=0</td><td>-0.0054</td><td>-0.0524%</td><td>0.2088%</td><td>0.0341%</td><td>0.2976%</td><td>0.4541%</td></tr><tr><td>=1</td><td>-0.2898</td><td>0.0977%</td><td>0.1134%</td><td>0.0248%</td><td>0.1460%</td><td>0.2713%</td></tr><tr><td rowspan="6">MDDH</td><td>=2</td><td>-0.5408</td><td>0.1163%</td><td>0.1107%</td><td>0.0191%</td><td>0.1365%</td><td>0.2694%</td></tr><tr><td>=3</td><td>-0.8085</td><td>0.1093%</td><td>0.1106%</td><td>0.0217%</td><td>0.1365%</td><td>0.2681%</td></tr><tr><td>=5</td><td>-1.3564</td><td>0.1128%</td><td>0.1101%</td><td>0.0197%</td><td>0.1356%</td><td>0.2680%</td></tr><tr><td>m=0</td><td>-0.0061</td><td>-0.0593%</td><td>0.2194%</td><td>0.0308%</td><td>0.3139%</td><td>0.4760%</td></tr><tr><td>m=1</td><td>-0.3513</td><td>0.0452%</td><td>0.1231%</td><td>0.0277%</td><td>0.1709%</td><td>0.2775%</td></tr><tr><td>=2</td><td>-0.6978</td><td>0.0477%</td><td>0.1217%</td><td>0.0267%</td><td>0.1685%</td><td>0.2743%</td></tr><tr><td></td><td>=3</td><td>-1.0616</td><td>0.0452%</td><td>0.1229%</td><td>0.0279%</td><td>0.1706%</td><td>0.2769%</td></tr><tr><td></td><td>=5</td><td>-1.7622</td><td>0.0463%</td><td>0.1223%</td><td>0.0272%</td><td>0.1696%</td><td>0.2757%</td></tr><tr><td colspan="2">Delta Hedging</td><td>-0.6562</td><td>0.0435%</td><td>0.1186%</td><td>0.0317%</td><td>0.1640%</td><td>0.2676%</td></tr><tr><td colspan="2">No Hedging</td><td>1</td><td>0.1823%</td><td>0.1410%</td><td>0</td><td>0.1732%</td><td>0.3480%</td></tr></table>

Note:istabe (EDH),model-driven deep hedging (MDDH),delta hedging andno hedging,across different risk-averse ratio $\xi$ .Each row corresponds to the result of the respective model, retained and tested under different setting of risk-aversion ratio $\xi$ .All the metrics are calculated in the same way as shown in Tabel 1.The transaction cost is $\mathbf { c } = \mathbf { 0 . 0 3 \% }$ and the confidence level of VaR and CVaR is $9 5 \%$

TABLE 3丨 Performance of ILDH and EDH under different transaction cost ratio.   

<table><tr><td></td><td>Transaction cost ratio</td><td>Reward</td><td>P&amp;L</td><td>Std P&amp;L</td><td>Cost</td><td>VaR</td><td>CVaR</td></tr><tr><td rowspan="5">ILDH</td><td>c=0</td><td>-0.5170</td><td>0.1297%</td><td>0.1084%</td><td>0</td><td>0.1313%</td><td>0.2623%</td></tr><tr><td>c = 0.01%</td><td>-0.5223</td><td>0.1188%</td><td>0.1078%</td><td>0.0089%</td><td>0.1311%</td><td>0.2599%</td></tr><tr><td>c = 0.03%</td><td>-0.5272</td><td>0.1010%</td><td>0.1074%</td><td>0.0236%</td><td>0.1316%</td><td>0.2590%</td></tr><tr><td>C = 0.10%</td><td>-0.5370</td><td>0.0293%</td><td>0.1076%</td><td>0.0911%</td><td>0.1354%</td><td>0.2603%</td></tr><tr><td>C = 0.30%</td><td>-0.5524</td><td>-0.1078%</td><td>0.1113%</td><td>0.2237%</td><td>0.1441%</td><td>0.2784%</td></tr><tr><td rowspan="5">EDH</td><td>c=0</td><td>-0.5274</td><td>0.1385%</td><td>0.1126%</td><td>0</td><td>0.1367%</td><td>0.2751%</td></tr><tr><td>c = 0.01%</td><td>-0.5365</td><td>0.1370%</td><td>0.1140%</td><td>0.0063%</td><td>0.1390%</td><td>0.2801%</td></tr><tr><td>C = 0.03%</td><td>-0.5408</td><td>0.1163%</td><td>0.1107%</td><td>0.0191%</td><td>0.1365%</td><td>0.2694%</td></tr><tr><td>C = 0.10%</td><td>-0.5387</td><td>0.0906%</td><td>0.1146%</td><td>0.0538%</td><td>0.1406%</td><td>0.2839%</td></tr><tr><td>C = 0.30%</td><td>-0.6018</td><td>-0.0883%</td><td>0.1110%</td><td>0.2017%</td><td>0.1497%</td><td>0.2719%</td></tr><tr><td rowspan="4">MDDH</td><td>c=0</td><td>-0.7058</td><td>0.0722%</td><td>0.1233%</td><td>0.0000%</td><td>0.1704%</td><td>0.2767%</td></tr><tr><td>c = 0.01%</td><td>-0.7028</td><td>0.0632%</td><td>0.1224%</td><td>0.0094%</td><td>0.1694%</td><td>0.2748%</td></tr><tr><td>c = 0.03%</td><td>-0.6978</td><td>0.0477%</td><td>0.1217%</td><td>0.0267%</td><td>0.1685%</td><td>0.2743%</td></tr><tr><td>C = 0.10%</td><td>-0.7123</td><td>-0.0152%</td><td>0.1228%</td><td>0.0894%</td><td>0.1724%</td><td>0.2784%</td></tr><tr><td rowspan="5">Delta Hedging</td><td>C = 0.30%</td><td>-0.7698</td><td>-0.1995%</td><td>0.1294%</td><td>0.2684%</td><td>0.1913%</td><td>0.3032%</td></tr><tr><td>c=0</td><td>-0.6528</td><td>0.0752%</td><td>0.1186%</td><td>0</td><td>0.1631%</td><td>0.2666%</td></tr><tr><td>c = 0.01%</td><td>-0.6539</td><td>0.0646%</td><td>0.1186%</td><td>0.0106%</td><td>0.1634%</td><td>0.2670%</td></tr><tr><td>c = 0.03%</td><td>-0.6562</td><td>0.0435%</td><td>0.1186%</td><td>0.0317%</td><td>0.1640%</td><td>0.2676%</td></tr><tr><td>c = 0.10%</td><td>-0.6659</td><td>-0.0303%</td><td>0.1189%</td><td>0.1055%</td><td>0.1668%</td><td>0.2707%</td></tr><tr><td></td><td>C = 0.30%</td><td>-0.7011</td><td>-0.2414%</td><td>0.1217%</td><td>0.3166%</td><td>0.1798%</td><td>0.2888%</td></tr></table>

te e ng of transaction cost ratioc.Al the metrics are calculated in thesame wayasshown in Tabel1.Theriskaverseratio is $\xi = 2$ and the confidence level of VaR and CVaR is $9 5 \%$

Comparing ILDH, EDH, MDDH, and delta hedging in Table 3 reveals that the changes in transaction cost ratio do not impact the superiority of IDLH over other models. Despite rising transaction costs,the risk metrics for ILDH remain lower than those for EDH,MDDH,and delta hedging,with no significant change in the relative differences.This suggests that IDLH's advantage in risk control,relative to other benchmark models, remains robust across different transaction cost ratios. However, excessively high transaction costs can significantly reduce P&L, ultimately rendering option hedging strategies unprofitable.

# 4.4| Improvement of Imitation Learning and Memory Structure

In this subsection,we further evaluate the contributions of imitation learning and memory structure to the hedging performance of ILDH through an ablation study.

Ablation study is an attribution analysis method used in machine learning to assess the impact of specific model components by removing them and observing the resulting effect on performance.Building on this approach,we conduct the following ablation study: First,we compare the empirical results of both ILDH and EDH, with and without a memory structure.Next, we compare these results to delta hedging to quantify the relative improvement of each deep hedging approach.Finally,we analyze the incremental effects of imitation learning and memory structure,by comparing the relative improvement differences across these models.This analysis enables us to disentangle the individual contributions of imitation learning and memory structure to the overall performance of deep hedging.

Table 4 presents the results of ablation study for each step, demonstrating that the integration of imitation learning and memory structure significantly enhances the performance of deep hedging across all metrics.

Regarding the hedging reward,the combination of imitation learning increases the reward increment of deep hedging relative to delta hedging. Specifically, for EDH without memory structure, the reward increment is $1 5 . 9 6 \%$ ,which rises to $1 6 . 3 8 \%$ for ILDH without memory structure,representing an improvement of $2 . 6 3 \%$ When the memory structure is incorporated,the reward increment increases from $1 7 . 5 8 \%$ (EDH with memory structure） to $1 9 . 6 6 \%$ (ILDH with memory structure)， corresponding to an improvement of $1 1 . 8 2 \%$ ，Additionally, the utilization of memory structure increases the reward increment of deep hedging relative to delta hedging by $1 0 . 1 0 \%$ without imitation learning,and by $1 9 . 9 7 \%$ with it. These results indicate that both imitation learning and memory structure contribute to higher rewards in deep hedging, with their joint utilization yielding the greatest improvement. This improvement is likely due to memory structure enhancing the agent's ability to capture temporal dependencies,albeit with increased sample requirementsfor training largerneural networks.Imitation learning complements the increased sample requirements by providing supplementary demonstrations based on the BSM model, thereby improving the agent's learning efficiency and hedging performance.

TABLE 4丨 Ablation study for imitation learning and memory structure.   

<table><tr><td colspan="2"></td><td>Reward</td><td>P&amp;L</td><td>Std P&amp;L</td><td>Cost</td><td>VaR</td><td>CVaR</td></tr><tr><td colspan="8">Panel 1: Empirical performance</td></tr><tr><td>With memory</td><td>ILDH</td><td>-0.5272</td><td>0.1010%</td><td>0.1074%</td><td>0.0236%</td><td>0.1316%</td><td>0.2590%</td></tr><tr><td></td><td>EDH</td><td>-0.5408</td><td>0.1163%</td><td>0.1107%</td><td>0.0191%</td><td>0.1365%</td><td>0.2694%</td></tr><tr><td>Without memory</td><td>ILDH</td><td>-0.5487</td><td>0.1038%</td><td>0.1098%</td><td>0.0300%</td><td>0.1367%</td><td>0.2657%</td></tr><tr><td></td><td>EDH</td><td>-0.5514</td><td>0.1255%</td><td>0.1162%</td><td>0.0180%</td><td>0.1431%</td><td>0.2858%</td></tr><tr><td>Delta hedging</td><td></td><td>-0.6562</td><td>0.0435%</td><td>0.1186%</td><td>0.0317%</td><td>0.1640%</td><td>0.2676%</td></tr><tr><td colspan="8">Panel 2: Comparison deep hedging/delta hedging</td></tr><tr><td>With memory</td><td>ILDH</td><td>19.66%</td><td>132.02%</td><td>-9.45%</td><td>-25.46%</td><td>-19.77%</td><td>-3.22%</td></tr><tr><td></td><td>EDH</td><td>17.58%</td><td>167.19%</td><td>-6.71%</td><td>-39.60%</td><td>-16.81%</td><td>0.67%</td></tr><tr><td>Without memory</td><td>ILDH</td><td>16.38%</td><td>138.50%</td><td>-7.40%</td><td>-5.26%</td><td>-16.63%</td><td>-0.72%</td></tr><tr><td></td><td>EDH</td><td>15.96%</td><td>188.27%</td><td>-2.06%</td><td>-43.21%</td><td>-12.78%</td><td>6.81%</td></tr><tr><td colspan="8">Panel 3: Comparison with/without memory</td></tr><tr><td>ILDH</td><td></td><td>19.97%</td><td>-4.68%</td><td>27.72%</td><td>384.11%</td><td>18.88%</td><td>346.63%</td></tr><tr><td>EDH</td><td></td><td>10.10%</td><td>-11.20%</td><td>225.20%</td><td>-8.37%</td><td>31.52%</td><td>1</td></tr><tr><td colspan="8">Panel 4: Comparison with/without imitation learning</td></tr><tr><td>With memory</td><td></td><td>11.82%</td><td>-21.03%</td><td>40.74%</td><td>-35.70%</td><td>17.67%</td><td>1</td></tr><tr><td>Without memory</td><td></td><td>2.63%</td><td>-26.43%</td><td>258.35%</td><td>-87.83%</td><td>30.18%</td><td>一</td></tr></table>

Note:Pael mertruceles improvemtofrefeor reltiveeli witouto panel 3and panel4,dueto incongruentsigns between itsoldand newvalues,making percentage increasecalculationunfeasible.

Turning to the risk and profit metrics, the combination of memory structure and imitation learning significantly enhances the risk management of ILDH and EDH,while slightly reducing their profitable advantages compared to delta hedging. For instance, memory structure increases the reduction of deep hedging in StdP&L relative to delta hedging by $2 7 . 7 2 \%$ ，from $- 7 . 4 0 \%$ (ILDH without memory structure) to $- 9 . 4 5 \%$ (ILDH with memory structure).Similarly,it raises the decrements of VaR and CVaR by $1 8 . 8 8 \%$ and $3 4 6 . 6 3 \%$ ,respectively.However, it also reduces the P&L increment of ILDH relative to delta hedging from $1 3 8 . 5 0 \%$ to $1 3 2 . 0 2 \%$ 、For EDH,memory structure reduces the decrement of Std-P&L in EDH relative to delta hedging by $2 2 5 . 2 0 \%$ (from $- 2 . 0 6 \%$ to $- 6 . 7 1 \%$ ，yet the P&L increment decreases from $1 8 8 . 2 7 \%$ to $1 6 7 . 1 9 \%$ (a reduction of $1 1 . 2 0 \%$ ).Similar trade-offs between profit and risk control can be observed in the improvement brought by imitation learning to ILDH and EDH. Notably, imitation learning brings the greatest improvement in risk control when embedded with EDH without memory structure,raising its Std P&L decrement relative to delta hedging from $- 2 . 0 6 \%$ to $- 7 . 4 0 \%$ (an increase of $2 5 8 . 3 5 \%$ ), despitea $2 6 . 3 4 \%$ reduction in the P&L increment.

Moreover,imitation learning markedly improves ILDH's tail risk management. For both ILDH with and without memory structure, the utilization of imitation learning raises the VaR increment relative to delta hedging by $1 7 . 6 7 \%$ and $3 0 . 1 8 \%$ ，respectively. It also shifts the CVaR increment of ILDH relative to delta hedging from negative to positive,demonstrating enhanced tail risk control.A rational explanation for this improvement is as follows: The trading samples with high potential profit loss,which indicate tail risks,are infrequent in option trading datasets.As a result,it is difficult for the DRL agent to learn an optimal strategy for managing tail risk purely through trial-and-error attempts on these insufficient samples.Imitation learning addresses this problem by providing the DRL agent with BSM-based demonstrations of tail shocks, thereby enhancing its capability to manage tail risks.

In summary,both memory structure and imitation learning improve the performance of deep hedging by trading off minor profits for better risk control.Among the two,imitation learning achieves more substantial improvements,particularly in managing tail risks. These findings also underscore the complementary roles of memory structure and imitation learning in enhancing the robustness and effectiveness of deep hedging strategies.

# 5 Conclusion

Imitation learning bridges the gap between deep hedging and theoretical option pricing models,by leveraging option hedging demonstrations from these models.The main contribution of this paper is the proposal of an ILDH algorithm with a memory structure, which demonstrates superior performance over EDH (Mikkilä and Kanniainen 2023),MDDH,and delta hedging.Furthermore,this empirical outperformance remains robust across call and put options,different levels of risk aversion and transaction cost,without constraints on specific moneyness or TTM.

Our ablation study shows that both imitation learning and memory structure contribute to reducing portfolio risk in deep hedging,as evidenced by lower standard deviation,VaR,and CVaR of hedging profits.Attribution analysis reveals that imitation learning enhances the ability of deep hedging to handle tail risk,by incorporating additional tail loss hedging demonstrations from the BSM model.Meanwhile,memory structure improves the predicting and decision-making capability of deep hedging by including historical information.

Another notable contribution of this paper is to showcase how imitation learning can integrate DRL with theoretical financial models through ILDH. Traditional financial models are built on solid theoretical foundations and have undergone extensive validation.However, their strict assumptions limit their applicability in real-world markets without manual adjustments. While DRL has the potential to learn the optimal solutions in complex environments,effectively managing diverse information and nonlinear relationships that theoretical models struggle with.Yet, it is often constrained by limited sample data and faces challenges in handling rare risk events.Imitation learning enables DRL to inherit the powerful generalization capability of theoretical models while bypassing their strict assumptions,by providing demonstrations.Furthermore,it preserves DRL's ability to explore and learn superior strategies without being confined by the limitations of the theoretical models.

On the other hand, imitation learning offers a potential solution to the challenges of limited data and tail risk shocks in DRL and other AI applications in economics and finance. These AI models typically require substantial amounts of data to capture complex relationships or nonlinear patterns,yet economic and financial data are often limited.Imitation learning can augment these data, making it particularly valuable in circumstances where data is scarce. Moreover,data-driven AI models commonly encounter difficulties in managing tail risk shocks that are rare or even inexperienced.By utilizing demonstrations based on classic theoretical models,imitation learning can leverage their valuable insights to facilitate the learning of AI models in handling such rare events.

# Acknowledgments

The authors are listed alphabetically with equal contributions,so all the authors are the co-first authors of this paper.Jiang acknowledges financial support from the National Natural Science Foundation of China (No.72072193,72342019) and the National Social Science Fund of China (22&ZD063).

# Conflicts of Interest

The authors declare no conflicts of interest.

# Data Availability Statement

The data that support the findings of this study are available from the corresponding author upon reasonable request or can be accessed through the Wind database (https://www.wind.com.cn). The code for the algorithm introduced in this study is available from the corresponding author by request.

# Endnotes

1Transaction costs for shorting options are ignored,as the option price is relatively small and the position is shorted only once at the start of each hedging episode.The main cost of shorting options is margin requirements,but given the five-trading-days hedging per episode in our empirical analysis,the corresponding interest costs are also negligible.

References   
Amilon,H.20o3.“A Neural Network Versus Black-Scholes:A Comparison of Pricing and Hedging Performances.” Journal of Forecasting 22,no.4: 317-335.   
Baltussen,G.,Z.Da,S. Lammers,and M.Martens.2021.“Hedging Demand and Market Intraday Momentum.” Journal of financial Economics 142, no.1: 377-403.   
Black,F.,and M. Scholes.1973.“The Pricing of Options and Corporate Liabilities.”Journal of political Economy 81,no.3: 637-654.   
Bollerslev,T.2O06.“Leverage and Volatility Feedback Effects In HighFrequency Data.” Journal of Financial Econometrics 4,no.3: 353-384. Buehler,H.,L.Gonon,J. Teichmann,and B.Wood.2019.“Deep Hedging.”Quantitative Finance 19,no.8: 1271-1291.   
Cao,J.,J. Chen,J. Hull,and Z. Poulos. 2021.“Deep Hedging of Derivatives Using Reinforcement Learning.”The Journal of Financial Data Science 3, no. 1: 10-27.   
Carbonneau,A.2o21.“Deep Hedging of Long-Term Financial Derivatives.”Insurance: Mathematics and Economics 99: 327-340.   
Carbonneau,A.,and F.Godin.2O21.“Equal Risk Pricing of Derivatives With Deep Hedging.”Quantitative Finance 21,no.4: 593-608.   
Charpentier,A.，R.Elie,and C.Remlinger.2023.“Reinforcement Learning In Economics and Finance.”Computational Economics 62: 425-462. https://doi.0rg/10.1007/s10614-021-10119-4.   
Christie,A.1982.“The Stochastic Behavior of Common Stock Variances:Value,Leverage and Interest Rate Effects.”Journal of financial Economics 10,no.4: 407-432.   
Du,J.,M. Jin,P.N.Kolm,G.Ritter,Y.Wang,and B. Zhang. 2020. “Deep Reinforcement Learning for Option Replication and Hedging.” Journal ofFinancial Data Science 2,no.4: 44-57.   
French,K.R.,G.W.Schwert,and R.F.Stambaugh.1987.“Expected Stock Returns and Volatility.” Journal of financial Economics 19,no.1: 3-29. Fujimoto,S.，H.Hoof,and D.Meger.2018.“Addressing Function Approximation Error in Actor-Critic Methods.”In Proceedings of the 35th International Conference on Machine Learning,PMLR Vol.80,1587-1596. Garcia，R.， and R. Gencay.20o.“Pricing and Hedging Derivative Securities With Neural Networks and a Homogeneity Hint.”Journal of Econometrics 94, no.1-2: 93-115.   
Giurca, A.,and S. Borovkova. 2021. Delta Hedging of Derivatives Using Deep Reinforcement Learning. Available at SSRN 3847272.   
Halperin,I.2019.“The QLBS Q-Learner Goes NuQlear:Fitted Q Iteration,Inverse Rl,and Option Portfolios.” Quantitative Finance 19,no.9: 1543-1553.   
Halperin,I. 2020.“QLBS: Q-Learner in the Black-Scholes (-Merton) Worlds.”Journal of Derivatives 28,no.1:99-122.   
Hambly,B.,R.Xu,and H. Yang.2023.“Recent Advances in Reinforcement Learning In Finance.”Mathematical Finance 33, no.3: 437-503.   
Horvath,B., J. Teichmann,and Z. Zuric.2021.“Deep Hedging Under Rough Volatility.”Risks 9,no.7: 138.   
Hong,Y.,F.Jiang,L.Meng,and B.Xue.2024.“Forecasting Inflation Using Economic Narratives.” Journal of Business & Economic Statistics 43,no. 1: 216-231.   
Huang,D.,F. Jiang,K.Li, G. Tong,and G. Zhou.2022.“Scaled PCA: A New Approach to Dimension Reduction.” Management Science 68, no. 3: 1678-1695.

Huang,D.,F.Jiang,K.Li,G. Tong,and G.Zhou.2023.“Are Bond ReturnsPredictable With Real-Time Macro Data!" Journalof Econometrics 237, no.2: 105438.

Hull,J.,and A.White.2017.“Optimal Delta Hedging for Options.” Journal of Banking& Finance 82:180-190.   
Hussein,A.,M.M.Gaber,E.Elyan,and C.Jayne.2o17.“Imitation Learning:A Survey ofLearning Methods.”ACMComputing Surveys 50, no. 2: 1-35.   
Hutchinson, J. M.,A.W.Lo,and T.Poggio.1994.“A Nonparametric Approach to Pricing and Hedging Derivative Securities via Learning Networks.” Journal of Finance 49,no.3: 851-889.   
Ivascu，F.2021.“Option Pricing Using Machine Learning.”Expert Systems with Applications 163:113799.   
Kolm,P.N.,and G. Ritter. 2019.“Dynamic Replication and Hedging: A Reinforcement Learning Approach.” Journal of Financial Data Science 1, no. 1: 159-171.   
Lillicrap,T.P.,J.J.Hunt,A. Pritzel,et al. 2015.Continuous Control With Deep Reinforcement Learning. arXiv preprint arXiv:1509.02971. Lutkebohmert, E.,T. Schmidt,and J. Sester. 2022.“Robust Deep Hedging.”Quantitative Finance 22, no.8: 1465-1480.   
Marzban, S.,E.Delage,and J.Y.-M.Li.2023.“Deep Reinforcement Learning for Option Pricing and Hedging Under Dynamic Expectile Risk Measures.”Quantitative Finance 23,no.10: 1411-1430.   
Meng，L.，R.Gorbet，and D.Kulic.2021．Memory-Based Deep Reinforcement Learning for POMDPs.2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).   
Le Mero,L., D.Yi, M. Dianati,and A. Mouzakitis. 2022.“A Survey on Imitation Learning Techniques for End-to-End Autonomous Vehicles.” IEEE Transactions on Inteligent Transportation Systems 23,no.9: 14128-14147.   
Merton,R. C.1973.“Theory of Rational Option Pricing.”Bell Journal of Economics and Management Science 4:141-183.   
Mikkilä,O.,and J.Kanniainen.2023.“Empirical Deep Hedging.” Quantitative Finance 23,no.1: 111-122.   
Mnih,V.,K.Kavukcuoglu,D. Silver, et al.2013.Playing Atari With Deep Reinforcement Learning. arXiv preprint arXiv:1312.5602.   
Monoyios,M. 2004.“Option Pricing With Transaction Costs Using a Markov Chain Approximation.”Journal of Economic Dynamics and Control 28, no. 5: 889-913.   
Ross,S., G. Gordon,and D. Bagnell.2011. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics.   
Rummery,G.A.,and M. Niranjan. 1994. On-Line Q-Learning Using Connectionist Systems. University of Cambridge,Department of Engineering Cambridge.   
Silver,D.,A. Huang,C.J. Maddison,et al. 2016.“Mastering the Game of Go With Deep Neural Networks and Tree Search.”Nature 529, no. 7587: 484-489.   
Silver,D.,G.Lever,N.Heess,T.Degris,D.Wierstra,and M.Riedmiller. 2014.“Deterministic Policy Gradient Algorithms.”In Proceedings of the 31st International Conference on Machine Learning,PMLR,Vol.3,2387-2395. Singh,V., S.-S.Chen, M. Singhania, B. Nanavati, A. kar,and A. Gupta. 2022.“How Are Reinforcement Learning and Deep Learning Algorithms Used for Big Data Based Decision Making In Financial Industries-A Review and Research Agenda.” International Journal of Information Management Data Insights 2,no.2:100094.   
Wang, X., S. Wang, X. Liang, et al. 2022.“Deep Reinforcement Learning: A Survey." In IEEE Transactions on Neural Networks and Learning Systems,Vol.35,5064-5078. Watkins, C.J.C.H.1989.Learning From Delayed Rewards PhD Thesis.   
University of Cambridge.

Zhang,J.,and W.Huang. 2021.“Option Hedging Using LSTM-RNN: An Empirical Analysis.”Quantitative Finance 21, no.10: 1753-1772.

# Appendix A Option Delta Hedging Based on the BSM Model

The option delta hedging position can be derived from the pricing formula of the BSM model:

$$
V _ { t , \mathrm { c a l l } } = S _ { t } \mathrm { N } ( d _ { 1 } ) - K e ^ { - r ( T - t ) } \mathrm { N } ( d _ { 2 } ) ,
$$

$$
V _ { t , \mathrm { p u t } } = K e ^ { - r ( T - t ) } \mathrm { N } ( - d _ { 2 } ) - S _ { t } \mathrm { N } ( - d _ { 1 } ) ,
$$

$$
d _ { 1 , 2 } = \frac { \log S _ { t } / K + r ( T - t ) \pm \frac { 1 } { 2 } \sigma _ { t } ^ { 2 } ( T - t ) } { \sigma _ { t } \sqrt { ( T - t ) } } ,
$$

where $N ( \cdot )$ denotes the standard normal cumulative distribution function,and the hedging positions should be,

$$
n _ { t , \mathrm { c a l l } } = \frac { \partial V _ { \mathrm { c a l l } } } { \partial S } = \mathrm { N } ( d _ { \mathrm { l } } ) ,
$$

$$
n _ { t , \mathrm { p u t } } = \frac { \partial V _ { \mathrm { p u t } } } { \partial S } = \mathrm { N } ( d _ { \mathrm { l } } ) - 1 .
$$

Here,the only parameter needed to be estimated is the volatility of underlying asset $\sigma _ { t }$ .Since we are using the high-frequency option trading data in this paper,the implied volatility for last time step $\sigma _ { t - 1 } ^ { \prime }$ would be a proper estimation of $\sigma _ { t }$ ：

$$
V _ { t - 1 } ( \sigma _ { t - 1 } ^ { \prime } , \cdot ) = V _ { t - 1 , \mathrm { m a r k e t , } }
$$

$$
\sigma _ { t } = \sigma _ { t - 1 } ^ { \prime } .
$$

TABLE A1| Hyperparameters of ILDH.  

<table><tr><td>Hyperparameters Value or range</td></tr><tr><td> TD3 model hyperparameters</td></tr><tr><td>Gamma (γ) 0.999</td></tr><tr><td>Soft update parameter (τ) 0.005</td></tr><tr><td>Episode length 40</td></tr><tr><td>History length (l) 8</td></tr><tr><td>Action limit [-1,1]</td></tr><tr><td>Action noise 0.2</td></tr><tr><td>Actor policy delay 4</td></tr><tr><td>Critic target network update noise 0.2</td></tr><tr><td>Critic target network update noise clip 0.5</td></tr><tr><td>Actor-critic networks hyperparameters</td></tr><tr><td>Critic networks memory state pre-RNN hidden layers (256)</td></tr><tr><td>Critic networks memory state RNN hidden layers (256，256)</td></tr><tr><td>Critic networks memory state after RNN hidden layers (256)</td></tr><tr><td>Critic networks current state RNN hidden layers (256,256)</td></tr><tr><td>Critic networks combine RNN hidden layers (512)</td></tr><tr><td>Critic networks learning rate 10-4</td></tr><tr><td>Critic networks weight decay 10-5</td></tr><tr><td>Actor networks memory state pre-RNN hidden layers (256)</td></tr><tr><td>Actor networks memory state RNN hidden layers (256，256)</td></tr><tr><td>Actor networks memory state after RNN hidden layers (256)</td></tr><tr><td>Actor networks current state RNN hidden layers (256,256)</td></tr><tr><td>Actor networks combine RNN hidden layers (512)</td></tr><tr><td>Actor networks learning rate 10-4</td></tr><tr><td>Actor networks weight decay 10-5</td></tr><tr><td>Replay buffer hyperparameters</td></tr><tr><td>Buffer maximize size 640,000</td></tr><tr><td>Buffer minimize size 320,000</td></tr><tr><td>Batch size 1024</td></tr></table>

TABLE A2丨 Descriptive statistics of CSI 300 stock index call and put option data set.   

<table><tr><td></td><td colspan="6">Call options</td></tr><tr><td></td><td>Mean</td><td>Std</td><td>Max</td><td>Min</td><td>Skewness</td><td>Kurtosis</td></tr><tr><td>Stock index-test</td><td>3853.36</td><td>25.51</td><td>3898.33</td><td>3808.01</td><td>0.0196</td><td>-0.5357</td></tr><tr><td>Stock index-train</td><td>4584.92</td><td>42.59</td><td>4656.67</td><td>4502.86</td><td>-0.1415</td><td>-0.4479</td></tr><tr><td>Option—test</td><td>133.27</td><td>13.24</td><td>157.36</td><td>111.41</td><td>0.2093</td><td>-0.3030</td></tr><tr><td>Option- train</td><td>177.76</td><td>21.92</td><td>217.55</td><td>140.25</td><td>0.1546</td><td>-0.4189</td></tr><tr><td>Implied volatility—test</td><td>0.1712</td><td>0.0170</td><td>0.2046</td><td>0.1367</td><td>0.0254</td><td>0.3703</td></tr><tr><td>Implied volatility—train</td><td>0.1775</td><td>0.0214</td><td>0.2178</td><td>0.1354</td><td>0.0047</td><td>0.3249</td></tr><tr><td></td><td colspan="6">Put options</td></tr><tr><td>Stock index—test</td><td>3853.38</td><td>25.48</td><td>3898.29</td><td>3808.03</td><td>0.0168</td><td>-0.5377</td></tr><tr><td>Stock index-train</td><td>4585.22</td><td>42.55</td><td>4656.91</td><td>4502.98</td><td>-0.1437</td><td>-0.4326</td></tr><tr><td>Option-test</td><td>126.63</td><td>13.75</td><td>151.57</td><td>105.06</td><td>0.2392</td><td>-0.2623</td></tr><tr><td>Option—train</td><td>164.75</td><td>21.65</td><td>209.23</td><td>131.08</td><td>0.3441</td><td>-0.1943</td></tr><tr><td>Implied volatility-test</td><td>0.1677</td><td>0.0149</td><td>0.1992</td><td>0.1380</td><td>0.1558</td><td>0.4485</td></tr><tr><td>Implied volatility-train</td><td>0.2268</td><td>0.0145</td><td>0.2600</td><td>0.1976</td><td>0.1565</td><td>0.3106</td></tr></table>

Note: All statistic variables arecalculated based on single hedge episode and averaged over the data set,respectively.

# Appendix B

Details of BSM model based simulated trading data generating process

To enable MDDH to learn trading strategies from simulated data that are applicable to real market data,we aim to construct a simulation process based on the BSM model that closely resembles actual trading behavior. The process is as follows:

First,under the assumptions of the BSM model, the underlying asset price in the simulated data follows a GBM:

Finally, the constraints on moneyness,TTM,hedging frequency,length of per hedging episode,transaction costs,and other settings in the simulated data are set to match those of real trading data.For each simulated option hedging episode,the moneyness and TTM of the contract are randomly chosen within given constraints.

$$
\frac { d S } { S } = \mu d t + \sigma d z ,
$$

where $S$ is the asset price,and $\mu$ and $\sigma$ are the drift and volatility parameters，respectively. Here, $\mu$ represents the long-term average return of the asset, and $\sigma$ represents its volatility.

Second,the option pricing formulas used are consistent with those in Appendix A.

Third,the BSM model assumes constant implied volatility,which does not align with real trading data.To address this discrepancy,we model implied volatility in the simulated data as a random walk.Specifically, we set the initial implied volatility as a random variable drawn from a normal distribution,with mean and standard deviation matching those of real trading data.Furthermore,the change in implied volatility at each step follows a random walk,with the change's mean and standard deviation also aligned with real trading data.

$$
\sigma _ { t } = \sigma _ { t - 1 } + \varepsilon ,
$$

$$
\varepsilon \sim \mathrm { N } \Big ( \mathrm { m e a n } \big ( \sigma _ { t } ^ { \mathrm { r e a l } } - \sigma _ { t - 1 } ^ { \mathrm { r e a l } } \big ) , \mathrm { s t d } \big ( \sigma _ { t } ^ { \mathrm { r e a l } } - \sigma _ { t - 1 } ^ { \mathrm { r e a l } } \big ) \Big ) ,
$$

$$
\sigma _ { 0 } \sim \mathrm { N } \Big ( \mathrm { m e a n } \big ( \sigma _ { 0 } ^ { \mathrm { r e a l } } \big ) , \mathrm { s t d } \big ( \sigma _ { 0 } ^ { \mathrm { r e a l } } \big ) \Big ) ,
$$

where $\sigma ^ { \mathrm { { r e a l } } }$ denotes the implied volatility from the real training data.