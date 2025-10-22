# TimeHF: Billion-Scale Time Series Models Guided by Human Feedback

Yongzhi Qi 1 Hao Hu 1 Dazhou Lei 2 Jianshen Zhang 1 Zhengxin Shi 1 Yulin Huang 1 Zhengyu Chen 1 Xiaoming Lin 1 Zuo-Jun Max Shen 3 4

# Abstract

Time series neural networks perform exceptionally well in real-world applications but encounter challenges such as limited scalability, poor generalization, and suboptimal zero-shot performance. Inspired by large language models, there’s interest in developing large time series models (LTM) to address these issues. However, current methods struggle with training complexity, adapting human feedback, and achieving high predictive accuracy. We introduce TimeHF, a novel pipeline for creating LTMs with 6 billion parameters, incorporating human feedback. We use patch convolutional embedding to capture long time series information and design a human feedback mechanism called time-series policy optimization. Deployed in JD.com’s supply chain, TimeHF handles automated replenishment for over 20,000 products, improving prediction accuracy by $3 3 . 2 1 \%$ over existing methods. This work advances LTM technology and shows significant industrial benefits.

# 1. Introduction

Benefiting from the success of large models in fields such as language processing (Dubey et al., 2024) and computer vision (Kirillov et al., 2023), large-scale time series models (LTM) have rapidly developed in recent years.

Roughly, two typical strategies can be adopted to construct LTM: one involves designing appropriate prompts and tokenizers to discretize and align the time-series modality with the textual modality, followed by leveraging pre-trained large language models (LLMs) for inference (Zhou et al., 2023); the other is based on the transformer architecture, formulating a pure LTM from scratch (Woo et al., 2024).

![](images/d85b5438635ee52fc3521a8ec57129534701693b8dadeae615a27b212cf08377.jpg)  
Figure 1. Left: Traditional time series foundation models, while performing well in common scenarios, may be overly sensitive to noises in training datasets, resulting in hallucinations in complex scenarios. Right: With RLHF, feedback contrast pairs—constructed by interpretable small models created by human experts—guide the model to gradually shift toward more accurate predictions.

Scaling laws state that increasing the size of models and training datasets typically leads to performance improvements, which have been widely validated in language and vision domains (Alabdulmohsin et al., 2022). However, existing publicly available time series datasets generally suffer from limited data volume and strong regularity, small models can already effectively capture these regular patterns, making it difficult for scaling laws to take effect. Currently, the largest pure time series model contains only 710 million parameters (Ansari et al., 2024), significantly smaller than the large language models which reach up to 405 billion parameters (Dubey et al., 2024).

Although LTM may perform well in some common scenarios, time series can exhibit various patterns or distributions across different fields. In a specific domain such as forecasting sales of a long-tail product, prediction models must be adjusted or tuned to mitigate hallucinations (Xu et al., 2024), preventing inaccurate predictions or the emergence of outliers. Popular tuning methods in the LTM domain include Supervised Fine-Tuning (SFT) (Liu et al., 2024) and Retrieval-Augmented Generation (RAG) (Yang et al., 2024). To align with human preferences, Reinforcement Learning from Human Feedback (RLHF) (Knox and Stone, 2011) directly optimizes model behavior using human feedback. RLHF has demonstrated superior performance in complex tasks and dynamic environments and has been widely applied in fields such as text generation (Ouyang et al., 2022).

In this paper, we propose a novel framework for building LTMs to address the aforementioned challenges (see Figure 1 for an overview). Leveraging JD.com’s extensive sales data, we construct a high-quality time series dataset that spans a variety of scenarios, including standard products, new products, seasonal items, bestsellers, long-tail items, and intermittent items. This diverse dataset significantly enhances the richness and complexity of training data, thereby enabling the development of the LTMs of greater scale. We train PCLTMs with 300M, 1.6B, and 6B parameters, zeroshot experiments demonstrate their superior performance.

To further effectively incorporate human expertise into LTMs, we propose the first RLHF approach for time series prediction models, termed Timeseries policy optimization (TPO). It utilizes specialized predictive models (small models) built by human experts as proxies for expert knowledge and cognition to generate prediction pairs reflecting diverse human preferences in complex time series scenarios. Compared to state-of-the-art models (GPT4TS), our framework achieves an average reduction in Mean Squared Error (MSE) of $7 . 0 9 \%$ . Our contributions are summarized as follows:

• We propose a standard for constructing large-scale, high-quality time series datasets, including data augmentation such as high-dimensional aggregated and interpretable component prediction data, data balancing, diversity ranking, etc. Based on this standard, we construct a large-scale time series dataset of 210B data points with intricate and diverse temporal patterns.

• To address the intricate cross-dependencies in complex, long time series, we introduce a novel patch-based convolutional large time series model (TimeHF). Based on this approach, We propose the first billion-scale pure time series model and successfully achieve industrial deployment in JD.com.

• We propose Timeseries policy optimization (TPO), the first RLHF framework applicable to time series forecasting, which enables LTMs to learn the tacit knowledge of experienced time-series experts. Numerical studies show that our framework can improve the zeroshot capabilities of LTMs significantly.

# 2. Related Work

# 2.1. Pretraining Dataset in LTMs

Table 1 provides a summary of the key differences between recent LTMs, they can be broadly categorized into two types. The first type relies on foundational models from the language or image domains, assuming inherent similarities in time series data (Gruver et al., 2024; Zhou et al., 2023; Jin et al., 2023; Rasul et al., 2023; Ansari et al., 2024; Chen et al., 2024), requiring additional data transformation modules. The second type, including TimesFM (Das et al., 2023), MoiRAI (Woo et al., 2024), and Timer (Liu et al., 2024). These models can extract deeper insights from time series data, but they all treat patches as independent segments, limiting the model’s ability to capture inter-patch information.

Table 1. Comparison of Our Method with Existing LTMs   

<table><tr><td>CATEGORY</td><td>METHOD</td><td></td><td>TRAINING TRAINABLE ZERSAST SFT RLHF</td><td></td><td></td></tr><tr><td>LLM-BASED TIMELLM</td><td></td><td>1</td><td>6M</td><td></td><td></td></tr><tr><td></td><td>GPT4TS</td><td>-</td><td>4M</td><td></td><td></td></tr><tr><td>PURE LTM</td><td>CHRONOS</td><td>84B</td><td>710M</td><td></td><td>√</td></tr><tr><td></td><td>MOIRAI</td><td>28B</td><td>311M</td><td></td><td></td></tr><tr><td></td><td>TIMER</td><td>28B</td><td>67M</td><td></td><td></td></tr><tr><td></td><td>MOMENT</td><td>1B</td><td>385M</td><td></td><td></td></tr><tr><td></td><td>TIMESFM</td><td>100B</td><td>200M</td><td></td><td></td></tr><tr><td></td><td>TIMEHF</td><td>210B</td><td>6B</td><td></td><td></td></tr></table>

Pre-training data are crucial for LTMs, but currently available public time series datasets are insufficient in richness and scale, and existing construction methods are simplistic, lacking emphasis on data quality. These issues may lead to insufficient diversity and balance in the data.

Common datasets for LTMs include TSLib (Wu et al., 2022), the Monash Database (Godahewa et al., 2021), the GluonTs library (Alexandrov et al., 2020), and the LOTSA dataset (Woo et al., 2024). However, these publicly available datasets are far smaller than those used for LLMs. For example, LOTSA integrates various accessible datasets but contains only 27B observations, while TimesFM uses a mixed dataset of 100B observations. While TimesFM’s synthetic data reflects common patterns, it sacrifices some authenticity. Our pretraining dataset combines public datasets with JD.com sales data, leveraging aggregation, predictive synthesis, and advanced data balancing and processing techniques.

# 2.2. Reinforcement Learning from Human Feedback

SFT and RLHF are two popular strategies to enhance LLMs accuracy and adaptability for specific tasks. Compared to SFT, RLHF achieves better results with less data (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022). Many studies enhance reinforcement learning strategies in RLHF for LLMs, such as Proximal Policy Optimization (PPO) and direct Preference Optimization (DPO) (Schulman et al., 2017; Rafailov et al., 2024; Ahmadian et al., 2024). Recognizing the critical role of RLHF in optimizing LLMs, we propose a specialized RLHF framework TPO, tailored for pure LTMs. This framework significantly improves both effectiveness and efficiency compared to traditional reinforcement learning methods. Table 2 compares TPO with other reinforcement learning approaches.

Table 2. Comparison of TPO & Other Reinforcement Learning Models   

<table><tr><td>MODEL</td><td>ACTOR MODEL</td><td>CRITIC MODEL</td><td>REWARD MODEL</td><td>ONLINE LEARNING</td><td>EFFICIENCY</td></tr><tr><td>PPO</td><td>REQUIRED</td><td>REQUIRED</td><td>REQUIRED</td><td>&gt;&gt;</td><td>Low</td></tr><tr><td>RLOO</td><td>REQUIRED</td><td>NOT REQUIRED</td><td>REQUIRED</td><td></td><td>MEDIUM</td></tr><tr><td>DPO</td><td>REQUIRED</td><td>NOT REQUIRED</td><td>NOT REQUIRED</td><td>×</td><td>HIGH</td></tr><tr><td>TPO</td><td>REQUIRED</td><td>NOT REQUIRED</td><td>NOT REQUIRED</td><td>√</td><td>HIGH</td></tr></table>

# 3. Method

Inspired by the widely adopted training procedures of LLMs, we design a three-step framework for LTMs: (1) base model training, (2) supervised fine-tuning, (3) reinforcement learning from human feedback. Figure 2 shows the whole framework of our method.

# 3.1. Base Model: Patch Convolutional Large Timeseries Model

The proposed Patch Convolutional Large Timeseries Model (PCLTM) employs a patch-based approach (potentially overlapping patches) and utilizes a masked encoder architecture for time series modeling. The core Transformer module follows an encoder-only architecture, incorporating various advancements from the state-of-the-art LLM architectures. We divide the input into patches and project them into vector representations. To capture intricate inter-patch information across different channels, we design a network based on convolutional layers. Further, we employ Rotary Position Embedding (ROPE) for temporal position encoding, and Grouped Query Attention (GQA), a group attention mechanism with temporal position encoding to facilitate better embedding. During training, we focus on prediction for each time point. Typical point prediction loss such as Mean Squared Error (MSE) can be utilized to update the gradients.

# 3.1.1. CROSS-PATCH PROJECTION LAYERS

Conventional patch encoding methods typically use linear mapping where each embedding vector only represents the information within the patch itself. Time series data, however, often has long-term dependencies, requiring stronger handling of cross-patch information. Therefore, we aim to incorporate information from a broader temporal window—beyond just the data within a single patch—right at the embedding stage. We design a convolution-based network, called patch conv module to achieve this end.

Suppose the patch length is $d _ { p }$ and input is divided into $s$ patches. An input sequence $\boldsymbol { y } _ { 1 : L } \in \mathbb { R } ^ { L }$ after patching can be represented as:

$$
y _ { p a t c h } \in \mathbb { R } ^ { s \times d _ { p } } .
$$

Then we encode all patches by convolutional layers with cross-patch channels. This requires transforming from $\mathbb { R } ^ { d _ { p } }$

to $\mathbb { R } ^ { d _ { e } }$ , implying each patch is transformed to a larger vector size incorporating temporal information of a longer period. See Figure 3 for an illustration of the cross-patch projection.

$$
y _ { p a t c h . e m b e d } = \mathrm { p a t c h C o n v } ( y _ { p a t c h } ) \in \mathbb { R } ^ { s \times d _ { e } }
$$

# 3.1.2. GROUP ATTENTION MECHANISM WITH TEMPORAL POSITION ENCODING

We employ GQA to reduce the burdensome computation cost of vast amounts of attention parameters. Let $y _ { t i m e . i n d e x } \ \in \ \mathbb { R } ^ { s \times 1 }$ denote the positions of all patches. Then the query, key, and value are represented by:

$$
\begin{array} { r l } & { q _ { i } = W _ { q } y _ { \mathrm { p a t c h \mathrm { . } e m b e d } , i } \in \mathbb { R } ^ { d _ { e } } } \\ & { k _ { i } = W _ { k } y _ { \mathrm { p a t c h \mathrm { . } e m b e d } , i } \in \mathbb { R } ^ { d _ { e } } } \\ & { v _ { i } = W _ { v } y _ { \mathrm { p a t c h \mathrm { . } e m b e d } , i } \in \mathbb { R } ^ { d _ { e } } } \end{array}
$$

where $W _ { q } , W _ { k }$ , and $W _ { v }$ are the parameters of the linear transformation layers for $q _ { i } , k _ { i }$ , and $v _ { i }$ $( i = 1 , . . . , s )$ ), respectively.

To incorporate sequential information, we further apply ROPE to both $q _ { i }$ and $k _ { i }$ :

$$
\begin{array} { r l } & { \mathrm { A t t e n } _ { i , j } = \mathrm { s o f t m a x } ( { q _ { i } } ^ { T } R _ { i - j } k _ { j } ) \in \mathbb { R } ^ { d _ { e } } , } \\ & { R _ { i - j } = \mathrm { R O P E } ( y _ { \mathrm { t i m e - i n d e x } } ) _ { i - j } \in \mathbb { R } ^ { d _ { e } \times d _ { e } } . } \end{array}
$$

where $R _ { i - j }$ is the rotary encoding matrix. After a FeedForward Network (FFN) which is a common module in transformers, the output of the transformer is:

$$
y _ { \mathrm { a t t e n . f n } } = \mathrm { F F N } ( \mathrm { A t t e n } * v ) \in \mathbb { R } ^ { s \times d _ { e } } .
$$

# 3.1.3. OUTPUT LAYERS

For the final output, an additional mapping module is needed for proper output dimensions, i.e., $\mathbb { R } ^ { s \times d _ { e } } \to \mathbb { R } ^ { H }$ . We adopt a flattening operation and then use an MLP for a prediction of length $H$ :

$$
\boldsymbol { \hat { y } } _ { L + 1 : L + H } = \mathbf { M L P } \left( f l a t t e n \left( y _ { a t t e n \_ f f n } \right) \right) \in \mathbb { R } ^ { H } .
$$

# 3.2. Supervised Fine-Tuning

SFT is one of the most common methods for enhancing the performance of large models. Due to the diversity of downstream tasks, pre-training datasets typically contain a large amount of time series data in various domains. For a specific scenario, collecting domain-specific datasets and applying SFT techniques can significantly improve prediction accuracy. We also construct a bunch of fine-tuning datasets to conduct SFT. Our SFT experiments on both proprietary and public datasets demonstrate that SFT has significant advantages in improving time-series forecasting performance. Let ySF T1:L be the input data for SFT, then the SFT task is given by:

![](images/0ae08d0ac76409541ef4b37b41538108a829a97adbd36422c4ad34f08d1d43eb.jpg)  
Figure 2. Model architecture of our method.

![](images/1ee0502fb0d738aa26c53143658a3c10cb27cc7907fc5fa9fbba6daeb382f760.jpg)  
Figure 3. Cross-patch convolutional projection layers.

$$
\hat { y } _ { L + 1 : L + H } ^ { S F T } = f _ { \theta } ^ { S F T } ( y _ { 1 : L } ^ { S F T } ) .
$$

# 3.3. Timeseries Policy Optimization

Reinforcement learning has been widely used for LLM finetuning, enabling the models to learn from human preferences and enhance their performance. The RL objective (Ouyang et al., 2022)for LLMs is:

$$
\begin{array} { r l } & { \mathsf { o b j e c t i v e } ( \phi ) = } \\ & { \mathbb { E } _ { ( x , y ) \sim D _ { \pi _ { \phi } ^ { R L } } } \Bigg [ r _ { \theta } ( x , y ) - \beta \log \Bigg ( \frac { \pi _ { \phi } ^ { R L } ( y \mid x ) } { \pi ^ { S F T } ( y \mid x ) } \Bigg ) \Bigg ] } \\ & { + \gamma \mathbb { E } _ { x \sim D _ { \mathrm { p r e t a i n } } } \left[ \log \left( \pi _ { \phi } ^ { R L } ( x ) \right) \right] } \end{array}
$$

where πRϕ is the learned RL policy, $\pi ^ { S F T } ( y \mid x )$ is the pre-trained supervised fine-tuning model, $D _ { \mathrm { p r e t r a i n } }$ is the pre-training distribution, $\beta$ is the coefficient for the Kullback–Leibler (KL) reward, and $\gamma$ is the coefficient for the pre-training loss. These coefficients control the strength of the KL penalty and the pre-training gradient, respectively.

The input and output of LTMs consist of continuous numerical values, presenting a fundamental difference from language models. Additionally, most pure LTMs typically employ loss functions such as Mean Squared Error or Quantile Loss, which are designed to perform multi-step predictions and are incompatible with TD errors. Furthermore, these models do not provide probabilistic outputs, making it impossible to compute probabilities or metrics such as KL divergence. As a result, conventional reinforcement learning frameworks like PPO and RLOO cannot be directly applied to LTMs (except probabilistic time series models).

To address this, we propose TPO, a reinforcement learning framework specifically designed for pure LTMs. TPO adapts the reinforcement learning architecture of LLMs to time series by incorporating a standardized RLHF data format and an advantage function based on REINFORCE. Compared to PPO and RLOO, TPO is more efficient and better suited for time-series models. Unlike PPO, which requires training three models (policy, critic, and reward), and RLOO, which involves training both policy and reward models, TPO trains only a single RL policy model. This design significantly reduces computational complexity and costs. The TPO objective is defined as:

$$
\begin{array} { r l } & { \mathrm { o b j e c t i v e } ( \phi ) = } \\ & { \hat { A } ( y _ { 1 : L } ^ { R L } , \hat { y } _ { L + 1 : L + H } ^ { R L } ) \frac { \pi _ { R L } ( f _ { \phi } ^ { R L _ { k } } ( y _ { 1 : L } ^ { R L } ) ; \mu _ { \phi } ^ { R L _ { k } } , \sigma _ { R L } ) } { \pi _ { R L } ( f _ { \phi } ^ { R L _ { \mathrm { o l d } } } ( y _ { 1 : L } ^ { R L } ) ; \mu _ { \phi } ^ { R L _ { \mathrm { o l d } } } , \sigma _ { R L } ) } } \\ & { \quad +  \gamma ( \alpha \mathbf { M S E } ( f _ { \phi } ^ { R L } ( y _ { 1 : L } ^ { R L } ) , \hat { y } _ { L + 1 : L + H } ^ { \mathrm { c h o s e n } }    } \\ & { \quad   + \omega \mathbf { M S E } ( f _ { \phi } ^ { R L } ( y _ { 1 : L } ^ { R L } ) , \hat { y } _ { L + 1 : L + H } ^ { \mathrm { r e j e c t } } ) ) } \end{array}
$$

where ˆ  yRL1:L , yˆRLL+1:L+H  is the advantage value estimated from the prediction $\hat { y } _ { L + 1 : L + H } ^ { R L }$ given the input $y _ { 1 : L } ^ { R L }$ , $\pi _ { R L }$ represents the probability of the prediction from RL strategy, and $\begin{array} { r l r } & { } & { \frac { \pi _ { R L } ( f _ { \phi } ^ { R L _ { k } } ( y _ { 1 : L } ^ { R L } ) ; \mu _ { \phi } ^ { R L _ { k } } , \sigma _ { R L } ) } { \pi _ { R L } ( f _ { \phi } ^ { R L _ { \mathrm { o l d } } } ( y _ { 1 : L } ^ { R L } ) ; \mu _ { \phi } ^ { R L _ { \mathrm { o l d } } } , \sigma _ { R L } ) } } \end{array}$ is the ratio of the new policy to the old policy. The parameter $k$ is the number of policy update steps in each iteration, $\gamma$ controls the weight of the time-series loss, and $\alpha$ and $\omega$ control the weights of the good and bad prediction losses (in feedback contrast pair), respectively.

Next, we will provide a detailed explanation of the TPO framework, highlighting its key components and discussing the advantages it offers over traditional reinforcement learning methods.

# 3.3.1. FEEDBACK CONTRAST PAIR

For the input, in addition to the original time-series input $y _ { 1 : L } ^ { R L } \in \mathbb { R } ^ { \bar { L } }$ from the RLHF dataset, we add a feedback contrast pair which includes a good prediction and a bad prediction $\left[ \boldsymbol { \hat { y } } _ { L + 1 : L + H } ^ { \mathrm { c h o s e n } } , \boldsymbol { \hat { y } } _ { L + 1 : L + H } ^ { \mathrm { r e j e c t e d } } \right] \in \left[ \mathbb { R } ^ { H } , \mathbb { R } ^ { H } \right]$ . The feedback contrast pair is constructed from a bunch of time-series models built by JD.com’s data analysts who have abundant experience on specific forecasting scenarios. That is, these tailored time series models are believed to reflect the cognitive insights, experience, and preferences of these people in forecasting tasks. The objective of fine-tuning on feedback contrast pair is to adjust the LTM model’s predictions to be more aligned with the desired outcomes, thereby forcing LTM to learn from the tacit knowledge of human experts.

# 3.3.2. PROBABILISTIC PREDICTION

Most time series models produce deterministic predictions without probabilities or uncertainties, making it impossible to compute KL divergence or policy probability loss in reinforcement learning. To address this issue, we develop a generic probabilistic prediction component suitable for all time series models (not limited to LTMs). This component assumes that both time series predictions and ”good”-”bad” predictions follow a normal distribution $N ( \mu , 1 )$ . The mean $\mu$ can be directly computed from the model’s predicted values, allowing for a rapid generation of prediction probabilities. $\pi _ { R L }$ , πchosen, and $\pi _ { \mathrm { r e j e c t e d } }$ can be derived from this approach.

# 3.3.3. ADVANTAGE FUNCTION

Since our time series large model directly outputs multi-step forecasts, we avoid using TD error in the RL phase. Instead, we adopt a REINFORCE-inspired approach, using an advantage function to quantify the improvement over a baseline reward. The goal is to fine-tune LTMs to closely align with the original SFT model while encouraging predictions that approach the ”good prediction.” Larger deviations from the baseline reward yield greater advantages, guiding the model toward outputs that better reflect human expertise.

The advantage function is defined as:

$$
\hat { A } \left( y _ { 1 : L } ^ { R L } , \hat { y } _ { L + 1 : L + H } ^ { R L } \right) = R \left( y _ { 1 : L } ^ { R L } , \hat { y } _ { L + 1 : L + H } ^ { R L } \right) - b _ { t s }
$$

where $R \left( y _ { 1 : L } ^ { R L } , \hat { y } _ { L + 1 : L + H } ^ { R L } \right)$ is the reward function which consists of two parts:

$$
\begin{array} { r l } & { R \left( y _ { 1 : L } ^ { R L } , \hat { y } _ { L + 1 : L + H } ^ { R L } \right) = r ( y _ { 1 : L } ^ { R L } , \hat { y } _ { L + 1 : L + H } ^ { R L } ) } \\ & { - \beta \log \left( \frac { \pi _ { R L } \left( f _ { \phi } ^ { R L } \left( y _ { 1 : L } ^ { R L } \right) ; \mu _ { \phi } ^ { R L } , \sigma _ { R L } \right) } { \pi _ { R L } \left( f _ { \phi } ^ { S F T } \left( y _ { 1 : L } ^ { R L } \right) ; \mu _ { \phi } ^ { R L } , \sigma _ { R L } \right) } \right) } \end{array}
$$

The first term in Eq. (11), $\begin{array} { r l } { r \left( y _ { 1 : L } ^ { R L } , \hat { y } _ { L + 1 : L + H } ^ { R L } \right) } & { { } = } \end{array}$ $\log \left( \pi _ { c h o s e n } ( f _ { \phi } ^ { R L } \left( y _ { 1 : L } ^ { R L } \right) ; \mu _ { \phi } ^ { c h o s e n } , \sigma _ { c h o s e n } ) \right)$ , represents the probability of the current prediction within the distribution of “good” predictions. Clearly, the higher this probability, the better the prediction will align with the “good prediction.” The second term is the KL divergence, measuring the deviation between the RL model and the SFT model, to prevent excessive divergence during the RL phase. The coefficient $\beta$ controls the weight of the KL reward term.

In contrast to RLOO’s baseline reward, which is derived through online sampling, the baseline reward in our design aims to reduce the variance in model learning. By using the SFT model’s prediction (i.e., the “bad prediction”) as the baseline reward, we ensure that the computed difference reflects a gain toward the “good prediction.” This approach avoids the computational costs associated with the sampling method used in RLOO, resulting in more efficient learning:

$$
b _ { t s } = \log \left( \pi _ { r e j e c t e d } ( f _ { \phi } ^ { R L } \left( y _ { 1 : L } ^ { R L } \right) ; \mu _ { \phi } ^ { r e j e c t e d } , \sigma _ { r e j e c t e d } ) \right) .
$$

# 4. Experiments

# 4.1. Construction of Large-scale, High-quality Time Series Datasets

Scaling up datasets is a proven strategy for improving the performance and generalizability of large time-series models. Time-series data vary in complexity: high-dimensional data often include patterns like trends and seasonality, while low-dimensional data are influenced by random factors such as timing and promotions. By carefully designing datasets, valuable information from diverse sources can be effectively captured.

In this study, we propose a standard for constructing largescale, high-quality time-series datasets. The dataset is divided into three components: pretraining data, supervised fine-tuning (SFT) data, and reinforcement learning with human feedback (RLHF) data. The pretraining dataset combines JD.com’s proprietary sales data, public datasets, and synthetic data, resulting in a large-scale dataset of 210 billion data points with intricate temporal patterns. Details of the pretraining dataset are provided in Appendix A.1.

To adapt the pre-trained model to downstream tasks, we construct fine-tuning datasets tailored to specific scenarios, with further details in Appendix A.2. The RLHF dataset consists of feedback contrast pairs, where each sample includes a good prediction and a bad prediction, as described in Appendix A.3.

# 4.2. Experiment Setting

For the JD.com dataset, we evaluate forecasting performance on over 30,000 products, using data from April 2024 onward as the testing period. Forecast Accuracy (FA) is adopted as the evaluation metric, with the current JD.com online algorithm (JD ONLINE) serving as a baseline. JD ONLINE is an ensemble model combining machine learning methods (e.g., XGBoost) and deep learning models. We compare this with three proposed methods: PCLTM, PCLTM $^ +$ SFT, and PCLTM $^ +$ SFT+TPO.

For the public datasets, we evaluate performance using test data from four datasets in the TSLib database: ETTh1, ETTh2, ETTm1, and ETTm2. Metrics used include Mean Squared Error (MSE) and Mean Absolute Error (MAE), with detailed calculation formulas provided in Appendix B.1. Comparisons are made against the state-of-the-art finetuned LTM (GPT4TS) and five advanced full-shot timeseries models: PatchTST, Autoformer, ITransformer, DLinear, and Informer. Model configurations and training details are detailed in Appendix B.2.

# 4.3. Results

# 4.3.1. RESULTS ON JD.COM’S DATASET

We use a model size of 300M for efficiency to compare our method and the baseline (JD ONLINE). A detailed comparison of different model sizes is provided later in Section 4.4. Table 3 shows the comparison, and the results indicate that our method consistently outperforms JD ONLINE. Specifically, the incremental improvements from PCLTM, SFT, and TPO demonstrate that each component contributes to the overall performance, validating the effectiveness of our framework. The greatest improvement was achieved with PCLTM $1 { + } \mathrm { S F T { + } T P O }$ , resulting in a $1 4 . 7 7 \%$ increase in prediction accuracy.

Table 3. Model and Relative Accuracy Improvement   

<table><tr><td>MODEL</td><td>RELATIVE ACCURACY IMPROVEMENT</td></tr><tr><td>JD_ONLINE</td><td>-</td></tr><tr><td>PCLTM(300M)+ZERO SHOT</td><td>9.47%</td></tr><tr><td>PCLTM(300M)+SFT</td><td>10.98%</td></tr><tr><td>PCLTM(300M)+SFT+TPO</td><td>14.77%</td></tr></table>

Table 4 presents the accuracy improvements of our method across different scenarios. Notable performance gains are observed in the best-selling, seasonal, long-tail, and new product scenarios. For the $\mathrm { P C L T M + S F T + T P O }$ model, the relative optimization ranges from $3 . 9 9 \%$ to $2 9 . 5 1 \%$ , with the most substantial improvement seen in the long-tail product scenario.

Table 4. Performance Improvement Across Different Scenarios   

<table><tr><td>SCENARIO</td><td>MODEL</td><td>RELATIVE ACCURACY IMPROVEMENT</td></tr><tr><td rowspan="4">BESTSELLERS</td><td>JD_ONLINE</td><td></td></tr><tr><td>PCLTM(300M)+ZERO SHOT</td><td>16.86%</td></tr><tr><td>PCLTM(300M)+SFT</td><td>17.99%</td></tr><tr><td>PCLTM(300M)+SFT+TPO</td><td>20.08%</td></tr><tr><td rowspan="4">SEASONAL ITEMS</td><td>JD_ONLINE</td><td></td></tr><tr><td>PCLTM(300M)+ZERO SHOT</td><td>6.25%</td></tr><tr><td>PCLTM(300M)+SFT</td><td>9.62%</td></tr><tr><td>PCLTM(300M)+SFT+TPO</td><td>14.42%</td></tr><tr><td rowspan="4">LONG-TAIL PRODUCTS</td><td>JD.ONLINE</td><td></td></tr><tr><td>PCLTM(300M)+ZERO SHOT</td><td></td></tr><tr><td></td><td>7.45%</td></tr><tr><td>PCLTM(300M)+SFT+TPO</td><td>29.51%</td></tr><tr><td rowspan="4">NEW PRODUCTS</td><td>JD_ONLINE</td><td></td></tr><tr><td>PCLTM(300M)+ZERO SHOT</td><td>-0.84%</td></tr><tr><td>PCLTM(300M)+SFT</td><td>0.84%</td></tr><tr><td>PCLTM(300M)+SFT+TPO</td><td>3.99%</td></tr></table>

To visually present the prediction performance of our method, we display the prediction results of both our method and JD Online across different products in Appendix D.1. To demonstrate the effects of SFT and TPO, we also show the predictions of PCLTM, PCLTM+SFT, and PCLTM $+ \mathrm { S F T + T P O }$ in Appendix D.2.

Deployment on JD.com. In December 2024, we successfully deployed our proposed LTM on JD.com. The model is currently being used for automated replenishment across 20,000 SKUs. Compared to the previous online forecasting system, the model has delivered a substantial $3 3 . 2 1 \%$ improvement in prediction accuracy.

# 4.3.2. RESULTS ON PUBLIC DATASETS

On several public datasets, our method, in a zero-shot setting, performs comparably to or slightly better than the current state-of-the-art fine-tuned LTM (e.g., GPT4TS) and leading deep learning time series models. The performance metrics, including MSE and MAE, are presented in Table 5 and more detail in AppendixC, with the best results marked in bold and the second-best results underlined. The results indicate a progressive improvement as we incorporate SFT and $\mathrm { S F T + T P O }$ , with our model consistently ranking among the top 2 in most cases. This demonstrates the superiority of our proposed TimeHF framework.

# 4.4. Ablation Analysis

We conduct a detailed ablation study on our framework and the relative accuracy improvement for the model without each component is shown in Table 6. Overall, each component we propose contributes positively to the overall performance. Among them, TPO provides the most significant improvement; removing TPO greatly affects the model’s predictive accuracy. The second most impactful component is the temporal positional encoding, followed by PatchConv and SFT.

We also compare different RLHF methods to investigate the effect of TPO. Table 7 shows the relative accuracy improvement of $\mathrm { P C L T M } ( 3 0 0 \mathbf { M } ) { + } \mathrm { S F T }$ compared with different RLHF methods or TPO under various parameter settings, along with training efficiency. We find that TPO demonstrates an obvious advantage ( $1 2 . 3 4 \%$ - $1 4 . 7 7 \%$ over other mainstream RL frameworks in time series tasks. Furthermore, removing either the MSE loss or the MSE rejected loss from the TPO objective function results in a noticeable decline in relative accuracy. This highlights the necessity of the proposed feedback contrast pair. These pairs allow LTMs to learn from the rich forecasting expertise of JD.com’s business experts, leading to further improvements in accuracy.

In terms of training efficiency, our TPO method shows a significant improvement compared to other common RL methods. While DPO demonstrates the highest efficiency, TPO closely follows with only a marginal difference of 0.02 it/s. In terms of predictive performance, however, TPO significantly surpasses DPO by over 3.5 percentage points.

# 4.5. Hyperparameter Analysis

We calculate the metrics for PCLTMs with different parameter counts and present them in Table 8. As shown, the accuracy improves with the increase in the number of parameters, further validating the scaling law. This suggests that for LTM, ultra-large amounts of parameters are a key factor in improving prediction performance. As mentioned earlier, large-scale models require vast amounts of highquality time series data for effective training and fine-tuning, which also highlights the necessity of our carefully designed pretraining, SFT, and RLHF datasets.

We analyze the model performance under different learning rates, sequence lengths, patch sizes, and KL coefficients, as shown in Figure 4. Proper hyperparameters can effectively improve prediction accuracy.

![](images/6f5380cbcdbf53ff775c13c0a9748dc1b181b17e1e5a02a42a1d95501cb4f689.jpg)  
Figure 4. Impact of Hyperparameters on Accuracy Improvement

We tested the effects of different settings for the hyperparameters $\alpha$ and $\omega$ in the MSE loss of the TPO objective function. Table 9 shows that the prediction accuracy improves when the parameters are set to (0.9, 0.1) and (0.8, 0.2).

# 5. Conclusion

In this paper, we present a novel training framework for large-scale time series models $\mathrm { ( P C L T M + S F T + T P O ) }$ . PCLTM is a base LTM, and it is the first time series model at the billion-parameter scale. Its zero-shot performance outperforms the current state-of-the-art fine-tuned large models, such as GPT4TS, as well as fully-supervised forecasting models across various time series datasets. We also propose TPO, an RLHF framework that is tailored for pure LTMs. Our experiments show that TPO consistently outperforms existing RLHF frameworks (such as PPO and RLOO) in predictive performance. Our methodology has practical impacts. The proposed method has already been deployed on JD.com’s supply chain system for an automated replenishment process.

Table 5. Performance Comparison of Our Method and Baselines Using MSE And MAE   

<table><tr><td>DATASET</td><td>METRICS</td><td colspan="3">PCLTM(300M) ZERO SHOT SFT</td><td>GPT4TS SFT</td><td>PATCHTST</td><td>AUTOFORMER</td><td>ITRANSFORMER FULL SHOT</td><td>DLINEAR</td><td>INFORMER</td></tr><tr><td rowspan="2">ETTH1</td><td>MSE</td><td>0.3999</td><td>0.3645</td><td>SFT+TPO 0.3503</td><td>0.3697</td><td>0.3625</td><td>0.4452</td><td>0.3746</td><td>0.3826</td><td>0.7144</td></tr><tr><td>MAE</td><td>0.3994</td><td>0.3924</td><td>0.3819</td><td>0.4000</td><td>0.3851</td><td>0.4536</td><td>0.3818</td><td>0.3921</td><td>0.7934</td></tr><tr><td rowspan="2">ETTH2</td><td>MSE</td><td>0.2262</td><td>0.2199</td><td>0.2175</td><td>0.2419</td><td>0.2347</td><td>0.2839</td><td>0.2623</td><td>0.2967</td><td>1.1863</td></tr><tr><td>MAE</td><td>0.3028</td><td>0.2986</td><td>0.2956</td><td>0.3170</td><td>0.3011</td><td>0.3502</td><td>0.3321</td><td>0.3714</td><td>0.9168</td></tr><tr><td rowspan="2">ETTM1</td><td>MSE</td><td></td><td></td><td></td><td></td><td>0.2915</td><td></td><td>0.3226</td><td></td><td></td></tr><tr><td>MAE</td><td>0.2834 0.3355</td><td>0.2692 0.3257</td><td>0.2606 0.3216</td><td>0.2621 0.3212</td><td>0.3550</td><td>0.4567 0.4495</td><td>0.3565</td><td>0.3388 0.3614</td><td>0.7000 0.9571</td></tr><tr><td rowspan="2">ETTM2</td><td></td><td>0.1417</td><td></td><td>0.1285</td><td>0.1468</td><td>0.1573</td><td>0.1968</td><td>0.1665</td><td>0.1778</td><td></td></tr><tr><td>MSE MAE</td><td>0.2319</td><td>0.1295 0.2214</td><td>0.2182</td><td>0.2396</td><td>0.2464</td><td>0.2976</td><td>0.2615</td><td>0.2910</td><td>0.5592 0.4749</td></tr></table>

Table 6. Ablation Analysis: Relative Accuracy Improvement of Our Framework with the Exclusion of Each Component   

<table><tr><td>MODEL CONFIGURATION</td><td>RELATIVE ACCURACY IMPROVEMENT</td></tr><tr><td>PCLTM(300M)+SFT+TPO</td><td>14.77%</td></tr><tr><td>W/O PATCHCONV</td><td>12.87%</td></tr><tr><td>W/O ROPE</td><td>11.87%</td></tr><tr><td>W/O SFT</td><td>13.47%</td></tr><tr><td>w/O TPO</td><td>10.98%</td></tr></table>

Table 9. Relative Accuracy Improvement for Different $( \alpha , \omega )$ Values in TPO Objective   

<table><tr><td>(a,w)</td><td>RELATIVE ACCURACY IMPROVEMENT</td></tr><tr><td>(1.0,0.0)</td><td>13.23%</td></tr><tr><td>(0.9,0.1)</td><td>14.29%</td></tr><tr><td>(0.8,0.2)</td><td>14.77%</td></tr><tr><td>(0.7,0.3)</td><td>13.76%</td></tr><tr><td>(0.6,0.4)</td><td>11.40%</td></tr><tr><td>(0.5,0.5)</td><td>10.84%</td></tr><tr><td>(0.4,0.6)</td><td>9.89%</td></tr></table>

Table 7. Relative Accuracy Improvement and Training Efficiency of RLHF methods   

<table><tr><td>Model/Configuration</td><td>Relative Accuracy Improvement</td><td>(itr/s, Single GPU) Training Efficiency</td></tr><tr><td>PCLTM(300m)+SFT+PPO</td><td>10.98%</td><td>0.4</td></tr><tr><td>PCLTM(300m)+SFT+DPO</td><td>11.15%</td><td>1.86</td></tr><tr><td>PCLTM(300m)+SFT+RLOO</td><td>11.36%</td><td>1.05</td></tr><tr><td>PCLTM(300m)+SFT+TPO( = 0)</td><td>12.34%</td><td>1.75</td></tr><tr><td>PCLTM(300m)+SFT+TPO (ω = 0)</td><td>13.23%</td><td>1.84</td></tr><tr><td>PCLTM(300m)+SFT+TPO</td><td>14.77%</td><td>1.84</td></tr></table>

Table 8. Model Size and Relative Accuracy Improvement   

<table><tr><td>MODEL SIZE</td><td>RELATIVE ACCURACY IMPROVEMENT</td></tr><tr><td>PCLTM(300M)</td><td>9.47%</td></tr><tr><td>PCLTM(1.6B)</td><td>10.22%</td></tr><tr><td>PCLTM(6B)</td><td>10.68%</td></tr></table>

Our work provides a new perspective for building smarter LTMs, specifically by incorporating human expert feedback into the training process using RLHF methods. We hope that future research will further explore this direction and continue to advance the integration of human knowledge into time series modeling.

# Impact Statement

This work introduces a novel approach to time series modeling by combining pretraining, supervised fine-tuning, and reinforcement learning with human feedback. Our proposed framework marks a significant advancement in predictive accuracy for diverse time series tasks, consistently outperforming state-of-the-art methods in both zero-shot and fine-tuned settings. Notably, it demonstrates remarkable improvements in sales prediction across a variety of products, showcasing its versatility and robustness.

The practical implications of this research are profound. In the supply chain domain, even marginal improvements in forecasting accuracy can yield substantial financial gains. At JD.com, for example, a $10 \%$ relative improvement in prediction accuracy translates into annual cost savings of tens of millions of yuan and sales growth worth billions or even hundreds of billions of yuan. The model introduced in this paper has already been fully deployed at JD.com, achieving over a $30 \%$ enhancement in forecasting accuracy and delivering extraordinary economic benefits, reinforcing its real-world impact.

Beyond these tangible results, this research lays the foundation for integrating expert feedback into time series forecasting, paving the way for scalable and adaptable solutions across various domains. We believe this framework offers immense potential for future applications, and we anticipate that subsequent studies will build upon these findings to unlock even greater opportunities in both academia and industry.

# Acknowledgements

The authors would like to extend their sincere gratitude for the support received from Computing Department of JD.com, with particular thanks to Zhen Chen, XiaoKun Zhu, Zhaolong Xing, Yang Pei, Qian Yu.

References   
Arash Ahmadian, Chris Cremer, Matthias Galle, Marzieh ´ Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ust ¨ un, ¨ and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.   
Ibrahim M Alabdulmohsin, Behnam Neyshabur, and Xiaohua Zhai. Revisiting neural scaling laws in language and vision. Advances in Neural Information Processing Systems, 35:22300–22312, 2022.   
Alexander Alexandrov, Konstantinos Benidis, Michael Bohlke-Schneider, Valentin Flunkert, Jan Gasthaus, Tim Januschowski, Danielle C Maddix, Syama Rangapuram, David Salinas, Jasper Schulz, et al. Gluonts: Probabilistic and neural time series modeling in python. Journal of Machine Learning Research, 21(116):1–6, 2020.   
Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815, 2024.   
Mouxiang Chen, Lefei Shen, Zhuo Li, Xiaoyun Joy Wang, Jianling Sun, and Chenghao Liu. Visionts: Visual masked autoencoders are free-lunch zero-shot time series forecasters. arXiv preprint arXiv:2408.17253, 2024.   
Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.   
Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. arXiv preprint arXiv:2310.10688, 2023.   
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.   
Rakshitha Godahewa, Christoph Bergmeir, Geoffrey I Webb, Rob J Hyndman, and Pablo Montero-Manso. Monash time series forecasting archive. arXiv preprint arXiv:2105.06643, 2021.

Nate Gruver, Marc Finzi, Shikai Qiu, and Andrew G Wilson. Large language models are zero-shot time series forecasters. Advances in Neural Information Processing Systems, 36, 2024.

Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, YuanFang Li, Shirui Pan, et al. Time-llm: Time series forecasting by reprogramming large language models. arXiv preprint arXiv:2310.01728, 2023.   
Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.   
W Bradley Knox and Peter Stone. Augmenting reinforcement learning with human feedback. In ICML 2011 Workshop on New Developments in Imitation Learning (July 2011), volume 855, 2011.   
Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Timer: Generative pre-trained transformers are large time series models. In Forty-first International Conference on Machine Learning, 2024.   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.   
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.   
Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Arian Khorasani, George Adamopoulos, Rishika Bhagwatkar, Marin Bilos, Hena Ghonia, Nadhir Hassen, Anderson ˇ Schneider, et al. Lag-llama: Towards foundation models for time series forecasting. In R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models, 2023.   
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.   
Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler,

Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei,

and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020.   
Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers. arXiv preprint arXiv:2402.02592, 2024.   
Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2dvariation modeling for general time series analysis. arXiv preprint arXiv:2210.02186, 2022.   
Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. Hallucination is inevitable: An innate limitation of large language models. arXiv preprint arXiv:2401.11817, 2024.   
Silin Yang, Dong Wang, Haoqi Zheng, and Ruochun Jin. Timerag: Boosting llm time series forecasting via retrieval-augmented generation. arXiv preprint arXiv:2412.16643, 2024.   
Tian Zhou, Peisong Niu, Liang Sun, Rong Jin, et al. One fits all: Power general time series analysis by pretrained lm. Advances in neural information processing systems, 36:43322–43355, 2023.

# A. Dataset Detail

# A.1. Pretrain Dataset Detail

The entire dataset consists of JD.com’s proprietary sales data, publicly available datasets, and synthetic data. The JD dataset includes extensive sales information, such as lifecycle, promotions, marketing activities, unexpected events, and inventory status, adding complexity to the data. Public datasets cover industries like electricity, solar energy, weather, and transportation, featuring strong regularity with common time series patterns like seasonality and trends. Synthetic datasets are created using high-dimensional aggregation and interpretable methods, ensuring high accuracy and enabling the model to effectively learn the knowledge.the data have totaling approximately 488 billion observational data points. The data set undergoes a series of processing steps, including labeling, quality filtering, deduplication, diversity ranking, and data balancing, resulting in a final pretraining dataset with approximately 210 billion observational points JD.com Dataset. The majority of the data is sourced from JD.com’s sales data, covering various categories such as food and clothing over the past three years. This data contains approximately 382B observational points.

Public Datasets. We also incorporate data from the Monash Time Series Database and the TSLib Database, expanding the training samples through random segments. This contributes around 8B observational points.

Synthetic Datasets. We use two data augmentation methods to construct synthetic data: (1) Interpretable component prediction based on JD.com and public dataset time series. We perform component forecasting on historical time series, with components including baseline, seasonality, promotions, and holidays. The presence of components helps the model learn different time series characteristics more distinctly. (2) high-dimensional aggregated, time series are aggregated across different horizons (e.g. week, month) and different dimensions (e.g. category, brand, region), enriching the dataset’s diversity and heterogeneity. This dataset contains approximately 98 billion observational points.

Pretraining Data Processing. The base data undergoes the following five steps, with steps 1-4 specifically applied to the non-public data: (1) Labeling: Each sample in the non-public datasets is labeled with some metrics, such as series length, average daily sales, and zero-sales proportion. These labels are used to describe the characteristics of each time series. (2) Quality Filtering: Time series are evaluated for quality based on the labeled features, and those with excessively short lengths or polluted sales are removed, improving the overall data quality. (3) Deduplication: The data is randomly grouped and clustered by time series. A subset of samples within each cluster is retained to ensure diversity without redundancy. (4) Diversity Ranking: The data is reordered based on time series labels, ensuring each batch contains diverse time series with varying characteristics. (5) Data Balancing: We set different proportions for each data source: $20 \%$ synthetic data, $4 \%$ public datasets, and $76 \%$ JD.com data. We also balance aggregated time series $( 3 0 \% )$ with regular time series $( 7 0 \% )$ . Resampling is performed to ensure the final dataset adheres to these balance configurations. After evaluating the results from multiple data ratios on the traditional time series models, the optimal ratio configuration is selected for final integration into the pretraining dataset.

# A.2. SFT Dataset Detail

To better adapt the pre-trained model for downstream tasks, we construct fine-tuning datasets tailored to specific scenarios.

For JD.com’s scenario, we divide the data based on product types into four categories: best-sellers, seasonal items, long-tail products, and newly launched items. Fine-tuning datasets were constructed for each category.

• Best-sellers: Products with relatively high sales but lower quantity in terms of total product volume.   
• Seasonal items: Products with significant seasonality, e.g., those popular in summer but less so in winter, or vice versa.   
• Long-tail products: Items with low average daily sales and a high proportion of zero sales.   
• Newly launched items: Products with less than six months of historical sales data.

We select data ratios best suited for each category. For instance, in the best-seller category where the time series is more regular and stable, we reduce the proportion of synthetic data and increase the proportion of real data. The final fine-tuning dataset for JD.com’s scenario has an average of approximately 4B data points. For the public data scenario, we finetune the model using training data from the four datasets in the TSLib database: ETTh1, ETTh2, ETTm1, and ETTm2.

# A.3. RLHF Dataset Detail

In the RLHF fine-tuning dataset, each sample is a feedback contrast pair consisting of a good prediction and a bad prediction. For example, given historical data for the past 6 days, the goal is to forecast the next 3 days. The sample includes the 6 historical observations, with both the good and bad predictions being the 3 forecasted points for the future.

In the JD.com data scenario, business experts from various sectors, such as large supermarkets, books, and 3C electronics, provide forecasts based on their experience and expertise. These forecasts are paired with predictions from the large model and evaluated by the experts. The predictions with smaller forecasting errors are labeled as good predictions, while those with larger errors are labeled as bad predictions. A total of 4532 pairs were collected for this purpose as shown in Table 10.

For public datasets, we apply multiple prediction methods on the validation set of public datasets, including the outputs of the large model and comparative time series models (PatchTST, iTransformer, Autoformer). Predictions are manually labeled by algorithm engineers to form feedback contrast pairs. The number of samples for each public dataset is shown in Table 11.

Table 10. Number of Feedback Contrast Pairs for JD.com’s sectors   

<table><tr><td>SECTOR</td><td>#FEEDBACK CONTRAST PAIRS</td></tr><tr><td>HOME APPLIANCES</td><td>634</td></tr><tr><td>DIGITAL PRODUCTS</td><td>310</td></tr><tr><td>BOOKS</td><td>786</td></tr><tr><td>FASHION</td><td>47</td></tr><tr><td>PHARMACEUTICALS</td><td>531</td></tr><tr><td>AUTOMOBILES</td><td>513</td></tr><tr><td>FOOD&amp;LIFESTYLE</td><td>1,711</td></tr><tr><td>TOTAL</td><td>4,532</td></tr></table>

Table 11. Number of Feedback Contrast Pairs for Public Datasets   

<table><tr><td>PUBLIC DATASET</td><td>#FEEDBACK CONTRAST PAIRS</td></tr><tr><td>ETTH1</td><td>2,683</td></tr><tr><td>ETTH2</td><td>1,110</td></tr><tr><td>ETTM1</td><td>4,218</td></tr><tr><td>ETTM2</td><td>2,574</td></tr></table>

# B. Evaluation Detail

B.1. Evaluation formula

$$
\mathrm { F A } = 1 - \frac { \left| \sum _ { h = l t } ^ { l t + b p } y _ { l + h } - \sum _ { h = l t } ^ { l t + b p } \hat { y } _ { l + h } \right| } { \sum _ { h = l t } ^ { l t + b p } \left| y _ { l + h } \right| }
$$

$$
\mathrm { M S E } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \left( y _ { l + h } - \hat { y } _ { l + h } \right) ^ { 2 }
$$

$$
\mathrm { M A E } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } | y _ { l + h } - \hat { y } _ { l + h } |
$$

where $H$ represents the forecasting horizon, i.e., the number of steps ahead for prediction. $y _ { h }$ and $\hat { y } _ { h }$ denote the actual and predicted values at step $h$ , respectively. $l t$ and $b p$ refer to the delivery lead time and the procurement cycle, respectively.

# B.2. Model configuration and training details

Model configuration and training details are shown in Table 12 and 13.

Table 12. Model Configuration   

<table><tr><td>MODEL SIZE</td><td>300M</td><td>1.6B</td><td>6B</td></tr><tr><td>SEQ_LEN</td><td>512</td><td>512</td><td>512</td></tr><tr><td>PATCH_LEN</td><td>21</td><td>21</td><td>21</td></tr><tr><td>LAYERS</td><td>24</td><td>32</td><td>20</td></tr><tr><td>DMODEL</td><td>1024</td><td>2048</td><td>5120</td></tr><tr><td>HEAD</td><td>16</td><td>16</td><td>16</td></tr></table>

Table 13. Training Details   

<table><tr><td>MODEL SIZE</td><td>300M</td><td>1.6B</td><td>6B</td></tr><tr><td>#GPUS</td><td>4</td><td>8</td><td>12</td></tr><tr><td>BATCH SIZE</td><td>4*32</td><td>8*4</td><td>12*4</td></tr><tr><td>LEARNING RATE</td><td>1×10-4</td><td>1×10-4</td><td>3 ×10-4</td></tr><tr><td>DEEP SPEED</td><td>OR NTAGE1</td><td>STAGE1</td><td>STAGE2</td></tr></table>

# C. Performance Comparison of Our Method and Baselines

Table 14. Performance Comparison of Our Method and Baselines Using MSE   

<table><tr><td>DATASET</td><td>HORIZON</td><td colspan="3">PCLTM(300M)</td><td>GPT4TS SFT</td><td>PATCHTST</td><td>AUTOFORMER</td><td>ITRANSFORMER FULL SHOT</td><td>DLINEAR</td><td>INFORMER</td></tr><tr><td>ETTH1</td><td>21</td><td>ZERO SHOT 0.4281</td><td>SFT 0.3548</td><td>SFT+TPO 0.3291</td><td>0.3622</td><td>0.3110</td><td>0.4414</td><td>0.3082</td><td>0.3792</td><td>0.6198</td></tr><tr><td></td><td>96</td><td>0.3716</td><td>0.3741</td><td>0.3714</td><td>0.3771</td><td>0.4140</td><td>0.4490</td><td>0.4410</td><td>0.3860</td><td>0.8090</td></tr><tr><td></td><td>AVG</td><td>0.3999</td><td>0.3645</td><td>0.3503</td><td>0.3697</td><td>0.3625</td><td>0.4452</td><td>0.3746</td><td>0.3826</td><td>0.7144</td></tr><tr><td>ETTH2</td><td>21</td><td>0.1754</td><td>0.1647</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.1418</td></tr><tr><td></td><td>96</td><td>0.2769</td><td>0.2751</td><td>0.1609 0.2741</td><td>0.1943 0.2895</td><td>0.1674 0.3020</td><td>0.2218 0.3460</td><td>0.2276 0.2970</td><td>0.2604 0.3330</td><td>1.2308</td></tr><tr><td></td><td>AVG</td><td>0.2262</td><td>0.2199</td><td>0.2175</td><td>0.2419</td><td>0.2347</td><td>0.2839</td><td>0.2623</td><td>0.2967</td><td>1.1863</td></tr><tr><td>ETTM1</td><td>21</td><td></td><td>0.2150</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.6909</td></tr><tr><td></td><td>96</td><td>0.2248 0.3419</td><td>0.3233</td><td>0.2002 0.3210</td><td>0.2270 0.2972</td><td>0.2539 0.3290</td><td>0.4083 0.5050</td><td>0.3111 0.3340</td><td>0.3326 0.3450</td><td>0.7090</td></tr><tr><td></td><td>AVG</td><td>0.2834</td><td>0.2692</td><td>0.2606</td><td>0.2621</td><td>0.2915</td><td>0.4567</td><td>0.3226</td><td>0.3388</td><td>0.7000</td></tr><tr><td>ETTM2</td><td>21</td><td>0.1046</td><td>0.0911</td><td>0.0904</td><td>0.1206</td><td>0.1396</td><td>0.1385</td><td>0.1529</td><td>0.1626</td><td>0.5092</td></tr><tr><td></td><td>96</td><td>0.1787</td><td>0.1678</td><td>0.1665</td><td>0.1730</td><td>0.1750</td><td>0.2550</td><td>0.1800</td><td>0.1930</td><td>0.6092</td></tr><tr><td></td><td>AVG</td><td>0.1417</td><td>0.1295</td><td>0.1285</td><td>0.1468</td><td>0.1573</td><td>0.1968</td><td>0.1665</td><td>0.1778</td><td>0.5592</td></tr></table>

Table 15. Performance Comparison of Our Method and Baselines Using MAE   

<table><tr><td>DATASET</td><td>HORIZON</td><td colspan="3">PCLTM(300M)</td><td>GPT4TS SFT</td><td>PATCHTST</td><td>AUTOFORMER</td><td>ITRANSFORMER FULL SHOT</td><td>DLINEAR</td><td>INFORMER</td></tr><tr><td>ETTH1</td><td>21</td><td>ZERO SHOT 0.4028</td><td>SFT 0.3878</td><td>SFT+TPO 0.3680</td><td>0.3973</td><td>0.3511</td><td>0.4482</td><td>0.3586</td><td>0.3842</td><td>0.7488</td></tr><tr><td></td><td>96</td><td>0.3960</td><td>0.3970</td><td>0.3958</td><td>0.4026</td><td>0.4190</td><td>0.4590</td><td>0.4050</td><td>0.4000</td><td>0.8380</td></tr><tr><td></td><td>AVG</td><td>0.3994</td><td>0.3924</td><td>0.3819</td><td>0.4000</td><td>0.3851</td><td>0.4536</td><td>0.3818</td><td>0.3921</td><td>0.7934</td></tr><tr><td>ETTH2</td><td>21</td><td>0.2687</td><td>0.2612</td><td>0.2563</td><td>0.2860</td><td>0.2541</td><td>0.3124</td><td>0.3152</td><td>0.3557</td><td>0.8636</td></tr><tr><td></td><td>96</td><td>0.3368</td><td>0.3359</td><td>0.3348</td><td>0.3479</td><td>0.3480</td><td>0.3880</td><td>0.3490</td><td>0.3870</td><td>0.9700</td></tr><tr><td></td><td>AVG</td><td>0.3028</td><td>0.2986</td><td>0.2956</td><td>0.3170</td><td>0.3011</td><td>0.3502</td><td>0.3321</td><td>0.3714</td><td>0.9168</td></tr><tr><td>ETTM1</td><td>21</td><td>0.2939</td><td></td><td></td><td></td><td></td><td></td><td>0.3450</td><td></td><td></td></tr><tr><td></td><td>96</td><td>0.3770</td><td>0.2889 0.3625</td><td>0.2829 0.3602</td><td>0.2925 0.3499</td><td>0.3429 0.3670</td><td>0.4239 0.4750</td><td>0.3680</td><td>0.3508 0.3720</td><td>0.9341 0.9800</td></tr><tr><td></td><td>AVG</td><td>0.3355</td><td>0.3257</td><td>0.3216</td><td>0.3212</td><td>0.3550</td><td>0.4495</td><td>0.3565</td><td>0.3614</td><td>0.9571</td></tr><tr><td>ETTM2</td><td>21</td><td>0.1980</td><td>0.1876</td><td>0.1860</td><td>0.2171</td><td>0.2337</td><td>0.2561</td><td>0.2590</td><td>0.2900</td><td>0.4197</td></tr><tr><td></td><td>96</td><td>0.2658</td><td>0.2551</td><td>0.2504</td><td>0.2620</td><td>0.2590</td><td>0.3390</td><td>0.2640</td><td>0.2920</td><td>0.5300</td></tr><tr><td></td><td>AVG</td><td>0.2319</td><td>0.2214</td><td>0.2182</td><td>0.2396</td><td>0.2464</td><td>0.2976</td><td>0.2615</td><td>0.2910</td><td>0.4749</td></tr></table>

# D. Case study

D.1. the prediction results of both our method and JD Online across different products

![](images/27dc08c9abe9fcc0394c18544944a3bce536f2c36758af6c389bb7ebe5f2e561.jpg)  
Figure 5. Prediction Comparison Between Our Method and JD Online Across Different Products

# D.2. Prediction Results of PCLTM, PCLTM+SFT, and PCLTM+SFT+TPO

![](images/214e257e7dea308666c861d7dac10c53d1f6f3fa5d1524232b31fd704730fea7.jpg)  
Figure 6. Impact of SFT and TPO: Prediction Results of PCLTM, PCLTM+SFT, and PCLTM $^ +$ SFT+TPO