# Large Language Models for Financial Time Series Forecasting

Miguel Noguer i Alonso, Rodolfo Pereira Franklin Artificial Intelligence Finance Institute

March 4, 2025

# Abstract

This paper investigates the performance of various Large Language Models (LLMs) for time series forecasting, with a particular focus on a newly added model and its results. Models such as TimeGPT, NBEATS, NHITS, PatchTST, and KAN are tested against stock data from major companies including Google, Apple, Amazon, JPMorgan, and Meta. By utilizing metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE), the study assesses model performance across different market conditions. While TimeGPT and its Long Horizon variant exhibit strong performance in stable environments, specialized models like KAN and PatchTST demonstrate effectiveness in more complex data scenarios. The addition of a new model further highlights the importance of adapting LLMs for time series analysis, especially in volatile markets. The results underscore the potential of LLMs for accurate forecasting but also indicate areas for further model refinement.

# 1 Introduction

Time Series Analysis is a subfield of stochastic processes, with the first appearance of autoregressive models applied in a work by G. U. Yule and J. Walker in the 1920s and 1930s. It saw significant advancement with Box and Jenkins (1970), who introduced a systematic approach for constructing stationary ARMA models. Today, it finds applications in various fields such as Finance, Demand Planning, Astronomy, etc. However, each of these tasks requires extensive specialized knowledge, which contrasts with the idea of LLMs (Large Language Models) that aim to be generalists and can be applied to various scenarios, like GPT-3 Brown et al. (2020), GPT-4 OpenAI (2023), Gemini Team (2024), and many others. Several fields have benefited greatly from advances in LLMs, such as the field of natural language processing, but the area of time series has not yet reaped such positive impacts. Brown et al. (2020) demonstrated that LLMs have an admirable capacity for transfer learning with few or no examples, making them strong candidates for performing generalizable forecasting in different fields without the need for retraining with specific data for each task, which is entirely contrary to the more commonly used models today. As LLMs are pre-trained on massive datasets, they have also shown the ability to perform new tasks with few examples, which can enable predictions in areas where we have a small amount of data, which is a barrier today when it comes to specialized models, as it usually requires a large amount of data to increase their predictive power. However, despite their advantages, it is a challenge to use LLMs in the context of time series analysis due to the type of data involved. LLMs operate on discrete tokens, while time series data are inherently continuous. Furthermore, the knowledge and reasoning capabilities to interpret time series patterns are not naturally present in the pre-training of LLMs. With this open field of research, new studies are emerging that aim to fill this gap and explore the potential of LLMs in this area. case study

In this work, we aim to evaluate the performance of frameworks that have adapted Language Models for time series forecasting, particularly the Time-LLM proposed by Jin et al. (2024). The core idea is to reprogram the input time series into text prototype representations that are more naturally suited to language models’ capabilities. To further enhance the model’s reasoning ability concerning time series, the Prompt-as-prefix (PaP) was also proposed, a novel idea in enriching the input time series with additional context and providing task instructions in the modality of natural language Jin et al. (2024). Through an evaluation conducted in their work, they were able to show that large language models can act as effective time series learners with few examples and zero examples when adopted through this reprogramming approach, outperforming specialized forecasting models.

![](images/749f284cfe6900302c63471ccc3e50135e30c9c3b6ec19546453a11dbf6e2a01.jpg)  
Figure 1: Schematic illustration of reprogramming large language models (LLMs) in comparison of (a) task-specific learning.

As mentioned earlier, most time series forecasting models were and are developed for specific tasks, and although such models achieve good results, they lack versatility and generalization power. However, with computational advancements, increased processing power, and the emergence of more powerful hardware, pre-trained models have arisen, within which model reprogramming Chen (2023) emerged, which is the approach aligned with Time-LLM. A difference with TimeLLM is that unlike the LLM4TS proposed by Chang et al. (2024), it does not edit the input time series and does not perform fine-tuning on the base LLM. Instead, a reprogramming of the time series with the original data modality is proposed, along with prompts to unlock the potential of LLMs as effective time series machines.

# 3 Methodology

Jin et al. (2024) focused on reprogramming a visible foundation language model to embeddings, such as Llama Touvron et al. (2023) and GPT-2 Radford et al. (2019), for general time series forecasting without the need for any fine-tuning of the base model. Given a sequence of historical observations $X \in \mathbb { R } ^ { N \times T }$ consisting of $N$ different univariate variables over $T$ time steps, they attempted to reprogram a large language model $f \left( \cdot \right)$ to understand the input time series and accurately predict the readings at $H$ future time steps, denoted by $\hat { Y } \in \mathbb { R } ^ { N \times H }$ , with the overall goal of minimizing the mean squared errors between the actual values $Y$ and the predictions, i.e., $\begin{array} { r } { \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \lvert \lvert \hat { Y } _ { h } - Y _ { h } \rvert \rvert _ { F } ^ { 2 } } \end{array}$ .

In their methodology, three main components are presented: the first is input transformation, the second is a pre-trained and frozen LLM, and lastly, output projection. Initially, a multivariate time series is divided into N univariate time series, which are processed independently Nie et al. (2023). The $i$ -th series is denoted as $X ^ { ( i ) } \in R ^ { 1 \times T }$ which undergoes normalization, segmentation, and embedding before being reprogrammed with learned text prototypes to align the source and target modalities. Next, we enhance the LLM’s reasoning ability on time series by activating it together with the reprogrammed segments to generate output representations, which are projected into the final predictions $\hat { Y } ^ { ( i ) } \in R ^ { 1 \times H }$ .

It was observed that only the parameters of the light input transformation and output projection are updated, while the base language model remains frozen. The TIME-LLM is fine-tuned directly and becomes easily usable with just a small set of time series and a few training epochs, maintaining high efficiency and requiring fewer resources compared to creating large specialized models from scratch or finetuning those models.

# 3.1 Model Structure

# 3.1.1 TIME-LLM: Theoretical Foundations and Methodology

The application of Large Language Models (LLMs) to time series data introduces new challenges that require the adaptation of these models to work efficiently with temporal information. TimeLLM proposes a novel approach that leverages the sequence modeling capacity of LLMs and tailors their architecture to time series forecasting, which is traditionally handled by specialized models like ARIMA, LSTMs, and Transformers. By incorporating patch reprogramming, cross-modal embeddings, and novel prompting strategies, TimeLLM seeks to harness the best of both worlds — the power of LLMs and the specific demands of time series analysis.

Input Embedding and Normalization: Time series data, by nature, is subject to non-stationarity, meaning that its statistical properties, such as mean and variance, can shift over time. This poses a significant challenge for traditional machine learning models, which typically assume stationarity. To address this, TimeLLM utilizes Reversible Instance Normalization (RevIN) ?, a technique specifically designed for time series that normalizes the input data with $N ( 0 , 1 )$ distributions. RevIN allows for both normalization and the ability to reverse this normalization when outputting predictions, preserving important temporal characteristics.

Dividing the time series into overlapping or non-overlapping segments is another crucial step in the TimeLLM pipeline. By tokenizing the data into patches of length $L _ { p }$ , TimeLLM captures local semantic information. The sliding step $S$ controls how much overlap there is between patches, and by choosing $S$ carefully, the model can ensure sufficient granularity in capturing temporal patterns. This segmentation process also reduces the overall computational complexity by transforming a long sequence of time series into a manageable sequence of tokens that can be embedded into a lower-dimensional space.

Patch Reprogramming for Time Series: Patch reprogramming is one of the most innovative aspects of TimeLLM. Traditional LLMs are trained on natural language data, where tokens have semantic meaning. However, time series data is fundamentally different, making it challenging to directly apply LLMs to this domain. Patch reprogramming solves this by transforming patches of time series data into a space where LLMs can operate. This transformation involves reprogramming the patch embeddings $\hat { \mathbf { X } } _ { P } ^ { ( i ) }$ into the latent space of a pre-trained language model’s word embeddings $\mathbf { E }$ .

A key insight behind patch reprogramming is the analogy to transfer learning. In transfer learning, models trained on a source task are adapted to a target task, often with minimal retraining. Similarly, patch reprogramming adapts time series data to the input space of pre-trained LLMs without the need to re-train the LLM from scratch. This allows the TimeLLM model to leverage the vast representational power of LLMs, which have been trained on massive corpora, while focusing on the specific nuances of time series data.

Moreover, this cross-modal adaptation addresses the challenges of domainspecific features in time series, such as periodicity, seasonality, and trends, by embedding the patches in a way that reflects both the temporal structure and the underlying LLM representations. Figure 2 illustrates the comparison between Patch-as-Prefix (PaP) and Prompt-as-Prefix (PaP) methods, showcasing how patch embeddings can be reprogrammed for different modalities.

![](images/3744e8175ec8fec958c927ff2c00f62d0fb325f3c282d5e1736d1b039b41342b.jpg)  
Figure 2: Patch-as-Prefix versus Prompt-as-Prefix

Multi-head Cross-attention for Temporal Alignment: Cross-attention plays a pivotal role in TimeLLM, enabling the model to align time series data with pre-trained language representations. By using multi-head cross-attention, TimeLLM can focus on different temporal regions in the time series, effectively learning which parts of the sequence are most relevant for the prediction task at hand. For each attention head $k$ , query matrices $\mathbf { Q } _ { k } ^ { \left( i \right) }$ , key matrices $\mathbf { K } _ { k } ^ { \left( i \right) }$ , and value matrices V(i)k a re computed, which allows the model to attend to different temporal patches and language embeddings simultaneously.

This cross-attention mechanism ensures that the model can learn complex relationships within the time series while benefiting from the contextual information encoded in the language model’s embeddings. Each attention head specializes in capturing different types of patterns (e.g., short-term vs. long-term dependencies), which enhances the model’s ability to generalize across different forecasting horizons. The ability to capture both local and global dependencies is critical for making accurate time series predictions, especially in applications like climate forecasting, financial market prediction, and sensor data analysis.

Prompt-as-Prefix (PaP) for Enhanced Task Adaptability: Promptas-Prefix (PaP) introduces a novel way of guiding LLMs to process time series data effectively. Prompts, which are typically used to provide context in natural language tasks, are adapted here to act as prefixes that contain information about the structure and task of the time series data. This approach has two main benefits: (1) it allows the model to dynamically adapt to different forecasting tasks by providing relevant context in the form of prompts, and (2) it leverages the LLM’s ability to process structured information efficiently.

In TimeLLM, prompts are designed to include not only the task instruction (e.g., forecasting future values) but also relevant dataset statistics such as trends, seasonal components, and known patterns. Figure 4 provides an example of a time series prompt that helps the LLM focus on critical aspects of the input data. This method bypasses some of the challenges inherent in using LLMs for numerically precise tasks, as it allows the model to interpret time series data more flexibly without relying on direct numerical tokenization.

The Wind Speed and Direction Data (WSDD) reflects the atmospheric conditions influencing renewable energy generation. Each data point consists of the target wind speed and 4 meteorological features...

Below is the information about the input time series:

# [BEGINDATA]

[DOMAIN]: We typically observe that wind speeds increase significantlyin the late afternoon, leading to higher energy output in wind farms.

[Instruction]: Predict the next ${ \mathrm { - H } } >$ steps given the previous ${ < } \mathsf { T } >$ steps of wind speed and direction dataprovided.

[Statistics]: The input has a minimum of <min_val>,a maximum of <max_val>,and a median of <median_val>.The overall trend is <upward or downward>.The top three lags are <lag_val>. [END DATA]

$\gg$ and $\cdot$ are task-specific configurationsand calculated inputstatistics.

The PaP strategy also introduces the concept of task-specific prompts that can vary based on the domain of the time series data (e.g., financial markets, weather forecasting). This flexibility allows TimeLLM to generalize well across multiple domains, providing task-specific guidance without the need for extensive retraining or fine-tuning. Additionally, prompts help to mitigate the precision challenges associated with LLMs in numerical tasks, as they encourage the model to focus on higher-level patterns rather than individual numeric values.

Theoretical Justifications: The theoretical framework of TimeLLM is grounded in several well-established concepts in machine learning and deep learning. The use of self-attention in the backbone of LLMs, combined with cross-attention mechanisms, allows TimeLLM to model both short-term dependencies (within each

patch) and long-term dependencies (across the entire sequence). This hierarchical modeling approach is critical for effectively capturing the multi-scale nature of time series data.

Additionally, the reprogramming of patches into a language embedding space can be understood through the lens of cross-modal transfer learning. TimeLLM exploits the semantic richness of LLM embeddings, which have been trained on a vast corpus of text data, to enhance its ability to process and understand temporal patterns. This aligns with recent advancements in multimodal learning, where models trained on one modality (e.g., text) are adapted to perform tasks in another modality (e.g., time series).

Output Projection and Forecasting: After processing the reprogrammed patches through the LLM’s architecture, TimeLLM projects the output back into the time series domain. This involves flattening the output of the attention layers and applying a linear transformation to produce the final forecasts $\hat { \mathbf { Y } } ^ { ( i ) }$ . The output projection step ensures that the predictions are aligned with the original time series data, preserving the temporal ordering and structure of the input.

TimeLLM’s output layer is designed to handle both point forecasts and probabilistic forecasts, making it versatile for a wide range of time series applications. Whether forecasting stock prices, electricity demand, or temperature, TimeLLM provides a flexible and powerful framework for generating accurate predictions based on learned temporal representations.

The methodology introduced here represents a significant advancement in applying LLMs to time series forecasting, offering a new way to blend the strengths of language models with the specific requirements of temporal data. Future extensions of this work could involve integrating external knowledge sources, such as knowledge graphs, or hybridizing TimeLLM with traditional time series models like ARIMA or exponential smoothing to enhance its robustness across different forecasting scenarios.

# 3.1.2 NBEATS

Oreshkin et al. (2020) propose a basic building block with a fork architecture. This section provides a detailed description of the operation of the $\ell$ -th block. The $\ell$ -th block accepts an input $\mathbf { x } _ { \ell }$ and produces two outputs: $\hat { \mathbf { x } } _ { \ell }$ , referred to as the backcast, and $\hat { \mathbf { y } } _ { \ell }$ , the forward forecast.

![](images/6c6d041ab92ba47982f4202961a604f118fed92e9e6c02bad55d8b6e9d3a803f.jpg)  
Figure 4: NBeats Architecture   
Source: Oreshkin et al. (2020)

The first block in the model takes the overall model input, a history lookback window of a certain length, as its respective $\mathbf { x } _ { \ell }$ . The input window length is typically set to a multiple of the forecast horizon $H$ , with lengths ranging from $2 H$ to $7 H$ . For subsequent blocks, the input $\mathbf { x } _ { \ell }$ is the residual output from the previous blocks.

Internally, each block consists of two main parts. The first part is a fully connected network that produces forward $\theta _ { \ell } ^ { f }$ and backward $\theta _ { \ell } ^ { b }$ expansion coefficients. The second part involves the backward $g _ { \ell } ^ { b }$ and forward $g _ { \ell } ^ { f }$ basis layers, which use these coefficients to project onto basis functions and generate the backcast $\hat { \mathbf { x } } _ { \ell }$ and the forecast $\hat { \mathbf { y } } _ { \ell }$ .

The operation of the first part of the $\ell \cdot$ -th block is described by the following equations:

$$
\begin{array} { r l } & { \mathbf { h } _ { \ell , 1 } = F C _ { \ell , 1 } \left( \mathbf { x } _ { \ell } \right) , } \\ & { \mathbf { h } _ { \ell , 2 } = F C _ { \ell , 2 } \left( \mathbf { x } _ { \ell , 1 } \right) , } \\ & { \mathbf { h } _ { \ell , 3 } = F C _ { \ell , 3 } \left( \mathbf { x } _ { \ell , 2 } \right) , } \\ & { \mathbf { h } _ { \ell , 4 } = F C _ { \ell , 4 } \left( \mathbf { x } _ { \ell , 3 } \right) , } \\ & { \theta _ { \ell } ^ { b } = \mathrm { L I N E A R } _ { \ell } ^ { b } \left( \mathbf { h } _ { \ell , 4 } \right) , } \\ & { \theta _ { \ell } ^ { f } = \mathrm { L I N E A R } _ { \ell } ^ { f } \left( \mathbf { h } _ { \ell , 4 } \right) . } \end{array}
$$

Here, the LINEAR layer is a linear projection, where $\theta _ { \ell } ^ { f } = \mathbf { W } _ { \ell } ^ { f } \mathbf { h } _ { \ell , 4 }$ . The FC layer is a fully connected layer with ReLU non-linearity, described as $\mathbf { h } _ { \ell , 1 } =$ RELU $( \mathbf { W } _ { \ell , 1 } \mathbf { x } _ { \ell } + \mathbf { b } _ { \ell , 1 } )$ . The first part of the architecture aims to predict forward expansion coefficients $\theta _ { \ell } ^ { f }$ to optimize the accuracy of the partial forecast $\hat { \mathbf { y } } _ { \ell }$ , and backward coefficients $\theta _ { \ell } ^ { b }$ to help subsequent blocks by removing irrelevant components from their input.

The second part of the block maps the expansion coefficients $\theta _ { \ell } ^ { f }$ and $\theta _ { \ell } ^ { b }$ to outputs via basis layers, defined by the following equations:

$$
\begin{array} { r l } & { \hat { \bf { y } } _ { \ell } = \displaystyle \sum _ { i = 1 } ^ { \dim \left( \theta _ { \ell , i } ^ { f } \right) } \theta _ { \ell , i } ^ { f } { \bf { v } } _ { i } ^ { f } , } \\ & { \hat { \bf { x } } _ { \ell } = \displaystyle \sum _ { i = 1 } ^ { \dim \left( \theta _ { \ell } ^ { b } \right) } \theta _ { \ell , i } ^ { b } { \bf { v } } _ { i } ^ { b } . } \end{array}
$$

In these equations, $\mathbf { v } _ { i } ^ { f }$ and $\mathbf { v } _ { i } ^ { b }$ are the forecast and backcast basis vectors, and $\theta _ { \ell , i } ^ { f }$ is the $i$ -th element of $\theta _ { \ell } ^ { f }$ . The functions $g _ { \ell } ^ { b }$ and $g _ { \ell } ^ { f }$ are designed to provide sufficiently rich sets of basis vectors such that the outputs can be accurately represented by varying expansion coefficients $\theta _ { \ell } ^ { f }$ and $\theta _ { \ell } ^ { b }$ . The specific choices for $g _ { \ell } ^ { b }$ and $g _ { \ell } ^ { f }$ , including whether they are learnable or set to particular functional forms.

# 3.1.3 NHITS

Challu et al. (2022) propose an approach called N-HiTS, as detailed in this section. The high-level diagram and main operational principles are illustrated in Figure 5. Their method builds upon the Neural Basis Expansion Analysis Oreshkin et al. (2020) but introduces several enhancements to improve accuracy and computational efficiency, particularly for long-horizon forecasting. The core of their approach involves multirate sampling of the input signal and multi-scale synthesis of the forecast, resulting in a hierarchical forecast construction. This significantly reduces computational demands while enhancing forecasting accuracy.

![](images/eacfb5b40154e9590a65abd81047fb632044e19f060e1a829dae2ec963c2f551.jpg)  
Figure 5: N-HiTS Architecture   
Source: Challu et al. (2022)

Like N-BEATS, N-HiTS performs local nonlinear projections onto basis functions across multiple blocks. Each block comprises a multilayer perceptron (MLP) that learns to generate coefficients for the backcast and forecast outputs of its basis. The backcast output is utilized to clean the inputs for subsequent blocks, while the forecast outputs are summed to produce the final prediction. The blocks are organized into stacks, each stack specializing in learning a distinct characteristic of the data using a unique set of basis functions. The overall network input, $y _ { t - L : t }$ , includes $L$ lags.

N-HiTS is structured with $S$ stacks, each containing B blocks. Each block features an MLP that predicts forward and backward basis coefficients. The following subsections introduce the novel components of their architecture, with the stack index $s$ omitted for brevity.

It is proposed the use of a MaxPool layer at the input of each block $\ell$ to enable the block to focus on analyzing components of the input at a specific scale. The kernel size $k _ { \ell }$ of the MaxPool layer is crucial in this process. A larger $k _ { \ell }$ tends to remove high-frequency or small-time-scale components from the input, forcing the block to concentrate on analyzing large-scale, low-frequency content. This approach, termed multi-rate signal sampling, ensures that each block’s MLP processes an input signal at a different effective sampling rate. This method allows blocks with larger pooling kernel sizes to focus on the critical large-scale components necessary for consistent long-horizon forecasts.

Moreover, multi-rate processing reduces the input width for most blocks’ MLPs, which in turn limits the memory footprint, computational load, and the number of learnable parameters. This reduction helps alleviate overfitting while maintaining the original receptive field. The operation, given the block $\ell$ input $y _ { t - L : t , \ell }$ (where the input to the first block $\ell = 1$ is the network-wide input $y _ { t - L : t , 1 } \equiv y _ { t - L : t }$ ), is formalized as follows:

$$
y _ { t - L : t , \ell } ^ { ( p ) } = \mathrm { M a x P o o l } \left( y _ { t - L : t , \ell } , k _ { \ell } \right)
$$

It is described a process of Non-Linear Regression in which, after subsampling, block $\ell$ processes its input to nonlinearly regress forward interpolation MLP coefficients, denoted as $\theta _ { \ell } ^ { b }$ . The block first learns a hidden vector $h _ { \ell } \in \mathbb { R } ^ { N _ { h } }$ , which is then linearly projected to obtain the coefficients:

$$
\begin{array} { r l } & { \quad h _ { \ell } = M L P _ { \ell } \left( y _ { t - L : t , \ell } ^ { ( p ) } \right) } \\ & { \quad \theta _ { \ell } ^ { f } = L I N E A R ^ { f } \left( h _ { \ell } \right) } \\ & { \quad \theta _ { \ell } ^ { b } = L I N E A R ^ { b } \left( h _ { \ell } \right) } \end{array}
$$

These coefficients are subsequently used to synthesize the backcast $\tilde { y } _ { t - L : t , \ell }$ and forecast $\hat { y } _ { t + 1 : t + H , \ell }$ outputs of the block, following the process described by the authors.

A Hierarchical Interpolation method was proposed to address the increasing computational requirements and unnecessary complexity in multi-horizon forecasting models as the forecast horizon $H$ grows. Typically, in such models, the neural network prediction’s dimensionality equals the horizon’s size, leading to significant computational inflation. To mitigate this, the authors introduce temporal interpolation, where the dimensionality of interpolation coefficients is controlled by an expressiveness ratio $r _ { \ell }$ , which determines the number of parameters per unit of output time. Specifically, they define the dimensionality of the forward interpolation coefficients as $| \theta | _ { \ell } ^ { f } = \lceil r _ { \ell } H \rceil$ .

To recover the original sampling rate and predict all $H$ points in the horizon, they employ a temporal interpolation function $g$ :

$$
\begin{array} { r l } & { \hat { y } _ { r , \ell } = g \left( \tau , \theta _ { \ell } ^ { f } \right) , \qquad \forall \tau \in \left\{ t + 1 , \dots , t + H \right\} , } \\ & { \tilde { y } _ { \tau , \ell } = g \left( \tau , \theta _ { \ell } ^ { b } \right) , \qquad \forall \tau \in \left\{ t - L , \dots , t \right\} . } \end{array}
$$

Interpolation can vary in smoothness, with $g ~ \in ~ \mathcal { C } ^ { 0 } , \mathcal { C } ^ { 1 } , \mathcal { C } ^ { 2 }$ . For concreteness, the authors define a linear interpolator $g \in { \mathcal { C } } ^ { 1 }$ and a time partition $\tau =$ $\{ t + 1 , t + 1 + 1 / { r _ { \ell } } , \ldots , t + H - 1 / { r _ { \ell } } , t + H \}$ as follows:

$$
\begin{array} { r l r } & { } & { g \left( \tau , \theta \right) = \theta \left[ t _ { 1 } \right] + \left( \frac { \theta \left[ t _ { 2 } \right] - \theta \left[ t _ { 1 } \right] } { t _ { 2 } - t _ { 1 } } \right) \left( \tau - t _ { 1 } \right) , } \\ & { } & { t _ { 1 } = \arg \underset { t \in \mathcal { T } : t \leq \tau } { \operatorname* { m i n } } \left( \tau - t \right) , \quad t _ { 2 } = t _ { 1 } + \frac { 1 } { r _ { \ell } } . } \end{array}
$$

The hierarchical interpolation is implemented by distributing expressiveness ratios across blocks, synchronized with multi-rate sampling. Blocks closer to the input have smaller $r _ { \ell }$ and larger $k _ { \ell }$ , leading to low-granularity signals through aggressive interpolation and focusing on more smoothed, sub-sampled inputs. The final hierarchical forecast $\hat { y } _ { t + 1 : t + H }$ is obtained by summing the outputs of all blocks, assembling interpolations at different time-scale hierarchy levels.

Each block specializes in a specific scale of input and output signals, creating a structured hierarchy of interpolation granularity. The authors propose using exponentially increasing expressiveness ratios to handle various frequency bands while controlling the number of parameters. Alternatively, each stack can model a different known cycle of the time-series (e.g., weekly, daily) using a corresponding $r _ { \ell }$ . The backcast residual formed at a previous hierarchy scale is subtracted from the input of the next hierarchy level, sharpening the focus on signals outside the band already addressed by earlier hierarchy members.

$$
\widehat { y } _ { t + 1 : t + H } = \sum _ { \ell = 1 } ^ { L } \widehat { y } _ { t + 1 : t + H , \ell } y _ { t - L : t , \ell + 1 } = y _ { t - L : t , \ell } - \widetilde { y } _ { t - L : t , \ell }
$$

The hierarchical interpolation approach offers theoretical guarantees, as demonstrated in Appendix A, where the authors show that it can approximate infinitely dense horizons, provided the interpolating function $g$ uses projections onto informed multi-resolution functions $V _ { w }$ , and the forecast relationships are smooth.

The Neural Basis Approximation Theorem, which addresses the ability to approximate a forecast mapping $\mathcal { V } \left( \cdot | y _ { t - L : t } \right) : \left[ 0 , 1 \right] ^ { L } \gets \left\{ \begin{array} { r l r l } \end{array} \right.$ , where the forecast functions $\mathcal { F } = \{ \mathcal { V } ( \tau ) : [ 0 , 1 ]  \mathbb { R } \} = \mathcal { L } ^ { 2 } ( [ 0 , 1 ] )$ represent an infinite/dense horizon and are square integrable. They assume that multi-resolution functions $V _ { w } = \{ \phi _ { w , h } \left( \tau \right) = \phi \left( 2 ^ { w } \left( \tau - h \right) \right) \left| w \in \mathbb { Z } , h \in 2 ^ { - w } \times \left[ 0 , \ldots , 2 ^ { w } \right] \right\}$ can arbitrarily approximate $\mathcal { L } ^ { 2 } \left( \left[ 0 , 1 \right] \right)$ , and that the projection $\operatorname { P r o j } _ { V _ { w } } \left( \mathcal { V } \left( \tau \right) \right)$ varies smoothly on $y _ { t - L : t }$ .

Under these conditions, the forecast mapping $\mathcal { V } \left( y _ { t - L : t } \right)$ can be arbitrarily approximated by a neural basis expansion that learns a finite number of multiresolution coefficients $\widehat { \theta } _ { w , h }$ . This is expressed as follows: for any $\epsilon > 0$ ,

$$
\int \left| \mathcal { V } \left( \tau \middle | y _ { t - L : t } \right) - \sum _ { w , h } \hat { \theta } _ { w , h } \left( y _ { t - L : t } \right) \phi _ { w , h } \left( \tau \right) \right| d \tau \leq \epsilon
$$

The authors provide examples of multi-resolution functions $V _ { w }$ , such as piecewise constants, piece-wise linear functions, and splines, all of which possess arbitrary approximation capabilities.

# 3.1.4 PatchTST

Nie et al. (2023) propose a method for forecasting future values given a collection of multivariate time series samples with a lookback window $L$ . The PatchTST model they introduce utilizes a vanilla Transformer encoder as its core architecture, as illustrated in Figure 6.

Figure 6: A case study of multivariate time series forecasting on Traffic dataset. The prediction horizon is 96. Results with different look-back window $L$ and number of input tokens $N$ are reported. The best result is in bold and the second best is underlined. Down-sampled means sampling every 4 step and adding the last value. All the results are from supervised training except the best result which uses self-supervised learning.

<table><tr><td colspan="4"> Running time (s) with L = 336</td></tr><tr><td>Dataset</td><td>w. patch</td><td>w.o. patch</td><td>Gain</td></tr><tr><td>Traffic</td><td>464</td><td>10040</td><td>x 22</td></tr><tr><td>Electricity</td><td>300</td><td>5730</td><td>x 19</td></tr><tr><td>Weather</td><td>156</td><td>680</td><td>x4</td></tr></table>

<table><tr><td>Models</td><td>L</td><td>N</td><td>patch</td><td>method</td><td>MSE</td></tr><tr><td rowspan="5">Chanel-independet</td><td>96</td><td>96</td><td></td><td></td><td>0.518</td></tr><tr><td></td><td></td><td></td><td>down-sampled</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3330</td><td>903亿</td><td>厂</td><td></td><td>87</td></tr><tr><td>336</td><td>42</td><td></td><td> self-supervised</td><td>0.349</td></tr><tr><td>Channel-mixing FEDFormer</td><td>336</td><td>336</td><td></td><td></td><td>0.597</td></tr><tr><td>DLinear</td><td>336</td><td>336</td><td></td><td></td><td>0.410</td></tr></table>

Source: Nie et al. (2023)

They consider a univariate time series of length $L$ starting at time index 1, denoted as $x _ { 1 : L } ^ { ( i ) }$ , where $i = 1 , \ldots , M$ . The input $( x _ { 1 } , \dots , x _ { L } )$ is split into $M$ univariate series $\boldsymbol { x } ^ { ( i ) } \in \mathbb { R } ^ { 1 \times L }$ , which are fed independently into the Transformer backbone according to a channel-then produces prediction results $\hat { \boldsymbol { x } } ^ { ( i ) } = ( \hat { x } _ { L + 1 } ^ { ( i ) } , \dots , \hat { x } _ { L + T } ^ { ( i ) } ) \in \mathbb { R } ^ { 1 \times T }$ sformer backbone.

The method involves dividing each input univariate time series $\boldsymbol { x } ^ { ( i ) }$ into patches, which can be either overlapped or non-overlapped. With patch length $P$ and stride $S$ , the patching process generates a sequence of patches $\boldsymbol { x } _ { p } ^ { ( i ) } \in \mathbb { R } ^ { P \times N }$ , where $N$ is the number of patches. The use of patches reduces the number of input tokens from $L$ to approximately $L / S$ , significantly decreasing memory usage and computational complexity, thereby allowing the model to handle longer historical sequences and improve forecasting performance.

The Transformer encoder maps the observed signals to latent representations. The patches are mapped to the Transformer latent space of dimension $\boldsymbol { D }$ via a trainable linear projection $W _ { p } \in \mathbb { R } ^ { D \times P }$ , with a learnable additive position encoding $W _ { p o s } \in \mathbb { R } ^ { D \times N }$ to monitor the temporal order of patches. The multi-head attention mechanism then processes these patches through query, key, and value matrices:

$$
\begin{array} { r } { \left( O _ { h } ^ { ( i ) } \right) ^ { T } = \mathrm { A t t e n t i o n } \left( Q _ { h } ^ { ( i ) } , K _ { h } ^ { ( i ) } , V _ { h } ^ { ( i ) } \right) } \\ { = \mathrm { S o f t m a x } \left( \frac { Q _ { h } ^ { ( i ) } K _ { h } ^ { ( i ) ^ { T } } } { \sqrt { d _ { k } } } \right) V _ { h } ^ { ( i ) } } \end{array}
$$

The multi-head attention block includes BatchNorm layers and a feedforward network with residual connections, producing the representation $\boldsymbol { z } ^ { ( i ) } \in \mathbb { R } ^ { D \times N }$ . A flatten layer with a linear head is then used to obtain the final prediction result $\hat { x } ^ { ( i ) }$ .

The model uses MSE loss to measure the discrepancy between predictions and ground truth, averaged over $M$ time series:

$$
\mathcal { L } = \mathbb { E } _ { x } \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \lvert | \hat { \mathbf { x } } _ { L + 1 : L + T } ^ { ( i ) } - \mathbf { x } _ { L + 1 : L + T } \rvert | _ { 2 } ^ { 2 } .
$$

Nie et al. (2023) also introduce an instance normalization technique to mitigate the distribution shift effect between training and testing data. This involves normalizing each time series instance $\mathbf { x } ^ { ( i ) }$ to have zero mean and unit standard deviation before patching, with the mean and deviation added back to the output prediction.

In the context of representation learning, PatchTST is applied for self-supervised learning to extract high-level abstract representations from unlabeled data. The method employs a masked autoencoder approach, where portions of the input sequence are randomly masked, and the model is trained to reconstruct the missing content. This approach addresses challenges such as masking at single time steps and the design complexity of the output layer for forecasting tasks. By using nonoverlapping patches and masking a subset of them, the model is trained to recover the masked patches with MSE loss. Each time series develops its own latent representation, which can be cross-learned via a shared weight mechanism, allowing for flexible pre-training data that might not be feasible with other approaches. For the Time-LLM, NBEATS, NHiTS and Patch TST models we used the code available at https://github.com/marcopeix/time-series-analysis.

# 3.1.5 Chronos

Ansari et al. (2024) introduce Chronos, a framework that adapts existing language model architectures and training procedures to probabilistic time series forecasting. While both natural language and time series share sequential properties, they differ significantly in their representation — natural language consists of words from a finite vocabulary, whereas time series are real-valued. This difference requires specific modifications, particularly concerning tokenization, to apply language models to time series data. However, the design philosophy behind Chronos is to make minimal changes to the model architectures and training procedures since transformer models have already shown exceptional performance on language tasks.

Chronos tackles the challenge of tokenizing time series data by introducing a scaling and quantization process. Consider a time series $x _ { 1 : C + H } = [ x _ { 1 } , \ldots , x _ { C + H } ]$ , where the first $C$ time steps represent the historical context and the remaining $H$ represent the forecast horizon. In language models, inputs are tokens from a finite vocabulary, so time series data must be mapped into discrete tokens. This is achieved by first scaling and then quantizing the real-valued observations.

Scaling aims to normalize the time series to optimize deep learning models. In the case of Chronos, normalization is crucial for facilitating the quantization process. Chronos uses mean scaling, which normalizes individual time series entries by the mean of the absolute values in the historical context:

$$
\tilde { x } _ { i } = \left( x _ { i } - m \right) / s .
$$

Here, $m = 0$ and $\begin{array} { r } { s = { \frac { 1 } { C } } \sum _ { i = 1 } ^ { C } | x _ { i } | } \end{array}$ . This method, while not the only possible approach, is effective in practice and is commonly used in time series applications. Quantization converts the scaled real-valued time series, $\tilde { x } _ { 1 : C + H } = | \tilde { x } _ { 1 } , \dots , \tilde { x } _ { C + H } |$ , into a finite set of discrete tokens that can be processed by language models. Chronos does this by selecting $B$ bin centers $c _ { 1 } < . . . < c _ { B }$ on the real line and defining $B - 1$ edges $b _ { i }$ that separate the bins. The quantization function $q : \mathbb { R }  \{ 1 , 2 , \ldots , B \}$ maps real values to bins, while the dequantization function $d : \{ 1 , 2 , \dots , B \}  \mathbb { R }$ maps discrete tokens back to real values:

$$
\begin{array} { r } { q ( x ) = \left\{ \begin{array} { l l l } { 1 } & { \mathrm { i f ~ } - \infty \leq x < b _ { 1 } , } \\ { 2 } & { \mathrm { i f ~ } b _ { 1 } \leq x < b _ { 2 } , } \\ { \vdots } \\ { B } & { \mathrm { i f ~ } b _ { B - 1 } \leq x < \infty , } \end{array} \right. \quad \mathrm { a n d } \quad d ( j ) = c _ { j } . } \end{array}
$$

This approach, while simple, has certain limitations. For instance, the prediction range is constrained between the bin centers $[ c _ { 1 } , c _ { B } ]$ , which could hinder the model’s ability to predict values for time series with strong trends. The authors explore this limitation further in their experiments. In addition to the time series tokens, Chronos includes two special tokens, PAD and EOS, in its vocabulary. The PAD token is used to pad time series to a fixed length and replace missing values, while the EOS token marks the end of the sequence. This setup facilitates training and inference using language modeling libraries.

Chronos uses both encoder-decoder models (like T5) and decoder-only models (like GPT-2). No architectural changes are needed, except for adjusting the vocabulary size to match the number of bins used for quantization.

The objective function for Chronos is based on the categorical distribution over the quantized tokens. The model learns to predict $p \left( z _ { C + h + 1 } | z _ { 1 : C + h } \right)$ , where $z _ { 1 : C + h }$ is the tokenized time series. Chronos minimizes the cross-entropy loss between the true quantized values and the predicted distribution:

$$
\ell \left( \boldsymbol { \theta } \right) = - \sum _ { h = 1 } ^ { H + 1 } \sum _ { i = 1 } ^ { | \mathcal { V } _ { t s } | } \boldsymbol { 1 } _ { z _ { \left( C + h + 1 \right) } = i } \log p _ { \boldsymbol { \theta } } \left( z _ { C + h + 1 } = i | z _ { 1 : C + h } \right) .
$$

This loss function is standard in language modeling tasks. However, it does not take into account the distance between bins; the model is expected to learn these relationships implicitly from the training data. Chronos generates probabilistic forecasts by autoregressively sampling from the predicted distribution. The sampled tokens are dequantized and then unscaled to generate real-valued predictions. The dequantization process uses the function $d$ to convert tokens into real values, and these values are then rescaled using the inverse of the mean scaling transformation.

Given the relatively limited availability of high-quality time series datasets compared to natural language data, Chronos introduces two data augmentation techniques: TSMix and KernelSynth.

TSMix is an adaptation of Mixup, a technique commonly used in image classification. It generates new time series data by creating convex combinations of $k$ randomly sampled time series. The convex combination is given by:

$$
\tilde { x } _ { 1 : l } ^ { T S M i x } = \sum _ { i = 1 } ^ { k } \lambda _ { i } \tilde { x } _ { 1 : l } ^ { ( i ) } ,
$$

where the weights $\lambda _ { i }$ are sampled from a symmetric Dirichlet distribution. This process enhances the diversity of the training dataset by combining different patterns from existing time series. KernelSynth, on the other hand, generates synthetic time series using Gaussian Processes (GPs). GPs are defined by a mean function $m ( t )$ and a positive definite kernel $k ( t , t ^ { \prime } )$ . KernelSynth constructs a composite kernel by sampling from a set of basis kernels (linear, RBF, and periodic) and applying random operations ( $^ +$ or $\times$ ) to combine them. A synthetic time series is then generated by sampling from the GP prior. This method is particularly useful for creating time series with diverse patterns that may not be present in the available real data.

In summary, Chronos adapts the powerful capabilities of language models to time series forecasting through careful tokenization, scaling, and quantization. By leveraging existing model architectures and enhancing training data with augmentation techniques, Chronos provides a flexible and robust solution for handling time series data across various domains.

For the implementation we used https://github.com/amazon-science/chronosforecasting.

# 3.1.6 iTransformer

Liu et al. (2024) introduce the iTransformer, a novel approach for multivariate time series forecasting. The objective is to predict future time steps ${ \textbf { Y } } =$ $\left\{ \mathbf { x } _ { T + 1 } , \ldots , \mathbf { x } _ { T + S } \right\} \in \mathbb { R } ^ { S \times N }$ using historical observations ${ \bf X } = \{ { \bf x } _ { 1 } , \ldots , { \bf x } _ { T } \} \in \mathbb { R } ^ { T \times N }$ , where $T$ represents the time steps and $N$ the number of variates. Importantly, realworld time series often contain variates that do not align perfectly in time due to systematic lags or differences in physical measurements and distributions. This can lead to significant complexity when forecasting future values.

The iTransformer architecture builds on the encoder-only version of the Transformer (Vaswani et al., 2017), which includes embedding, projection, and Transformer blocks. Unlike most Transformer-based models that treat multiple variates at a single time point as tokens, iTransformer views the entire time series of each variate as a token. This approach challenges the conventional use of encoderdecoder Transformers, suggesting that heavy architectures may not be necessary for effective time series forecasting. Instead, the iTransformer focuses on representation learning and identifying correlations between multivariate series. The responsibility for generating the predicted time series is placed on simple linear layers, which have been shown to perform well in previous research.

The process of forecasting the future series of a specific variate $\hat { \mathbf { Y } } _ { : , n }$ based on the historical series $\mathbf { X } _ { : , n }$ is described by the following equations:

$$
\mathbf { h } _ { n } ^ { 0 } = \operatorname { E m b e d d i n g } ( \mathbf { X } _ { : , n } ) ,
$$

$$
\mathbf { H } ^ { l + 1 } = \operatorname { T r m B l o c k } ( \mathbf { H } ^ { l } ) , \quad l = 0 , \dots , L - 1 ,
$$

$$
\hat { \mathbf { Y } } _ { : , n } = \operatorname { P r o j e c t i o n } ( \mathbf { h } _ { n } ^ { L } ) ,
$$

where ${ \bf H } = \{ { \bf h } _ { 1 } , \dots , { \bf h } _ { N } \} \in \mathbb { R } ^ { N \times D }$ contains the embedded tokens, and the embedding $\mathbb { R } ^ { T } \to \mathbb { R } ^ { D }$ and projection $\mathbb { R } ^ { D } \to \mathbb { R } ^ { S }$ are both implemented using multi-layer perceptrons (MLPs). Self-attention is applied to capture the relationships between the variates, and the series is processed independently by a shared feed-forward network at each Transformer block layer. Notably, iTransformer eliminates the need for explicit positional embeddings, as the sequence order is inherently stored in the neuron structure of the feed-forward network.

One of the key advantages of iTransformer is its flexibility in utilizing various attention mechanisms. As multivariate correlations are central to time series forecasting, the architecture can incorporate efficient attention mechanisms, such as those proposed by Li et al. (2021) and Wu et al. (2022). This adaptability allows the model to handle different numbers of variates during training and inference, offering a scalable solution to complex forecasting tasks.

The components of iTransformer include layer normalization, feed-forward networks, and self-attention, all carefully adapted for time series data. Layer normalization ensures that each variate token is normalized individually, thus minimizing discrepancies caused by differences in physical measurements and distributions. The normalization formula is:

$$
\mathrm { L a y e r N o r m } ( \mathbf { H } ) = \left\{ \frac { \mathbf { h } - \mathrm { M e a n } ( \mathbf { h } _ { n } ) } { \sqrt { \mathrm { V a r } ( \mathbf { h } _ { n } ) } } \bigg | n = 1 , \ldots , N \right\} .
$$

This approach is particularly effective for handling non-stationary problems, as it prevents the oversmoothing of time series that can occur when tokens are normalized across time steps.

The feed-forward network (FFN) is essential for encoding token representations and extracting complex patterns from time series. In iTransformer, the FFN is applied to the series representation of each variate. By stacking multiple blocks, iTransformer encodes the observed time series and decodes them into future representations using dense non-linear connections. This structure allows iTransformer to capture essential properties of time series, such as amplitude, periodicity, and frequency spectra. The identical linear operation applied to independent time series in iTransformer also reflects recent advancements in linear forecasters and channel independence.

Self-attention in iTransformer plays a crucial role in capturing the relationships between different variates. The self-attention mechanism generates query, key, and value matrices ${ \bf Q } , { \bf K } , { \bf V } \in \mathbb { R } ^ { N \times d _ { k } }$ , where $d _ { k }$ is the dimension of the projected space. For each variate token, the attention score between tokens $i$ and $j$ is computed as:

$$
\mathbf { A } _ { i , j } = \left( \mathbf { Q } \mathbf { K } ^ { \top } / \sqrt { d _ { k } } \right) _ { i , j } \propto \mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { j } ,
$$

where $ { \mathbf { A } } ~ \in ~ \mathbb { R } ^ { N \times N }$ represents the multivariate correlations between the tokens. This approach allows iTransformer to weigh the importance of correlated variates more heavily when generating the next representation.

In summary, Liu et al. (2024) propose iTransformer as a flexible and efficient model for multivariate time series forecasting. By embedding entire time series as tokens, using self-attention to model correlations, and relying on MLPs for prediction, iTransformer offers a powerful solution for complex forecasting tasks. The architecture’s ability to incorporate different attention mechanisms and handle varying numbers of variates makes it suitable for a wide range of real-world applications. Experimentally, iTransformer demonstrates superior performance compared to traditional models, providing a promising approach to time series forecasting.

For the implementation we used https://github.com/thuml/iTransformer explained in Liu et al. (2023).

# 3.1.7 KAN for Time Series

Xu et al. (2024) introduced Kolmogorov-Arnold Networks (KAN), a novel neural network architecture grounded in the Kolmogorov-Arnold representation theorem. This theorem provides a theoretical foundation for expressing multivariate continuous functions through compositions of univariate functions, enabling the decomposition of complex functions into simpler and more interpretable components.

The Kolmogorov-Arnold representation theorem asserts that any multivariate continuous function can be decomposed into a finite sum of compositions of univariate functions. Formally, the theorem is expressed as:

$$
f \left( x _ { 1 } , \ldots , x _ { n } \right) = \sum _ { q = 1 } ^ { 2 n + 1 } \Phi _ { q } \left( \sum _ { p = 1 } ^ { n } \phi _ { q , p } \left( x _ { p } \right) \right)
$$

In KAN, traditional linear weights in neural networks are replaced with splineparametrized univariate functions. Unlike conventional Multi-Layer Perceptrons (MLPs), which utilize fixed activation functions at the nodes, KAN applies adaptive and learnable activation functions on the edges between nodes. These activation functions are parameterized as B-spline curves, which dynamically adjust during training to better capture the underlying data patterns. This unique structure allows KAN to effectively model complex nonlinear relationships within the data. Formally, a KAN layer is defined as:

$$
\Phi = \{ \phi _ { q , p } | p = 1 , 2 , \ldots , n _ { \mathrm { i n } } , \quad q = 1 , 2 , \ldots , n _ { \mathrm { o u t } } \}
$$

where $\phi _ { q , p }$ are parametrized functions with learnable parameters. To enhance its modeling capabilities, deeper KAN architectures are constructed by composing multiple KAN layers:

$$
K A N ( x ) = ( \Phi _ { L - 1 } \circ \Phi _ { L - 2 } \circ \cdot \cdot \cdot \circ \Phi _ { 0 } ) ( x ) ,
$$

where each $\Phi _ { l }$ represents a KAN layer. The depth of the network allows it to capture more intricate patterns and dependencies in the data, with each layer transforming the input through a series of learnable functions $\phi _ { q , p }$ , thereby making the network highly adaptable and powerful.

Building on the KAN framework, Xu et al. (2024) proposed two specialized models for time series forecasting: Temporal Kolmogorov-Arnold Networks (T-KAN) and Multivariate Temporal Kolmogorov-Arnold Networks (MT-KAN). These models address specific challenges in univariate and multivariate time series forecasting, respectively.

# T-KAN

T-KAN is designed to handle univariate time series data with the primary objectives of predicting future values and detecting as well as tracking concept drift. The architecture of T-KAN is based on a two-layer network structure, where each layer comprises spline-parametrized univariate functions. These functions model the relationships between consecutive time steps, allowing the network to adaptively learn temporal patterns within the data. The output of T-KAN at time step $t$ , denoted as $\hat { S } _ { t + T }$ , is given by:

$$
\hat { S } _ { t + T } = \sum _ { q = 1 } ^ { 2 n + 1 } \Phi _ { q } \left( \sum _ { p = 1 } ^ { n } \phi _ { q , p } \left( S _ { t - h + p } \right) \right)
$$

where $S _ { t - h + p }$ represents past observations at previous time steps, and $\phi _ { q , p }$ are the spline-parametrized univariate functions. Both $\Phi _ { q }$ and $\phi _ { q , p }$ are adaptively learned during the training process, enabling T-KAN to effectively model complex, nonlinear temporal dependencies.

To facilitate training, a sliding window approach is employed to traverse the time series, creating input-output pairs from each window. For example, using two historical time steps to predict the next time step. Different KAN structures and activation functions represent different concepts, allowing the identification of concept drift by observing variations in KAN models. This ensemble of evolving KAN models constitutes T-KAN, which effectively captures and adapts to concept drift in time series data.

Symbolic Regression for Interpretability

To enhance interpretability, symbolic regression is incorporated into T-KAN by fitting mathematical expressions to the learnable activation functions. This approach generates human-readable models that explain the underlying patterns in the data, facilitating an understanding of how past observations influence future predictions. Consequently, T-KAN not only achieves high predictive accuracy but also maintains transparency.

MT-KAN

MT-KAN extends the KAN framework to multivariate settings by modeling interactions between multiple time series. Similar to T-KAN, MT-KAN uses splineparametrized univariate functions to capture temporal dependencies. Additionally, it incorporates mechanisms to model cross-variable interactions, enabling the exploitation of relationships among different variables. The architecture involves flattening and stacking the historical steps of each variable as inputs, which are then processed through $L$ layers of KAN to produce forecasts for each variable. Formally, the output of MT-KAN at time step $t + T$ , denoted as $\hat { \mathbf { S } } _ { t + T }$ , is given by:

$$
\hat { \mathbf { S } } _ { t + T } = \sum _ { q = 1 } ^ { 2 n + 1 } \Phi _ { q } \left( \sum _ { p = 1 } ^ { h } \sum _ { k = 1 } ^ { m } \phi _ { q , p , k } \left( \mathbf { S } _ { t - h + p , k } \right) \right)
$$

where $\mathbf { S } _ { t - h + p , k }$ represents the past observations of the $k$ -th variable at previous time steps, and $\phi _ { q , p , k }$ are the spline-parametrized univariate functions. This structure enables MT-KAN to effectively capture both temporal dependencies and cross-variable interactions within the data, resulting in more accurate multivariate time series forecasts.

Overall, the introduction of T-KAN and MT-KAN demonstrates the versatility and power of the KAN framework in addressing complex forecasting challenges in both univariate and multivariate time series data.

For the implementation we used https://www.datasciencewithmarco.com/blog/kolmogorovarnold-networks-kans-for-time-series-forecasting.

# 4 Performance Metrics

In this section, we discuss the performance metrics used to evaluate the accuracy of our forecasting model. The primary metrics considered are the Mean Absolute Error (MAE), the Mean Squared Error (MSE), and the Root Mean Squared Error (RMSE). These metrics are widely used in time series forecasting and regression tasks due to their effectiveness in quantifying the difference between predicted and actual values.

# 4.1 Mean Absolute Error (MAE)

The Mean Absolute Error (MAE) is a metric that measures the average magnitude of errors in a set of predictions, without considering their direction. It is calculated

as the mean of the absolute differences between predicted values $\hat { y } _ { i }$ and actual values $y _ { i }$ :

$$
\mathrm { M A E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left| \hat { y } _ { i } - y _ { i } \right|
$$

where $n$ represents the total number of observations. The MAE is a linear score, which means that all individual differences are weighted equally. It provides an intuitive measure of the accuracy of the model, with lower MAE values indicating better performance.

# 4.2 Mean Squared Error (MSE)

The Mean Squared Error (MSE) is another widely used metric that measures the average of the squares of the errors, giving more weight to larger errors. It is defined as:

$$
\mathrm { M S E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( \hat { y } _ { i } - y _ { i } \right) ^ { 2 }
$$

The MSE penalizes larger errors more severely than the MAE, making it particularly sensitive to outliers. This characteristic can be useful when it is important to minimize large deviations in predictions. However, it also means that the MSE may be disproportionately influenced by a small number of large errors.

# 4.3 Root Mean Squared Error (RMSE)

The Root Mean Squared Error (RMSE) is the square root of the MSE, providing a measure of error that is in the same units as the original data. It is calculated as follows:

$$
{ \mathrm { R M S E } } = { \sqrt { { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \left( { \hat { y } } _ { i } - y _ { i } \right) ^ { 2 } } }
$$

The RMSE is a useful metric for comparing the residuals or differences between the predicted and actual values in a way that is easier to interpret than the MSE. Like the MSE, the RMSE is sensitive to large errors, but it is often preferred in cases where the interpretability of the error magnitude is important.

# 4.4 Comparative Analysis

When selecting a performance metric, it is important to consider the specific characteristics of the task at hand. The MAE provides a straightforward interpretation of average error magnitude, making it a good choice for applications where all errors are treated equally. On the other hand, the MSE and RMSE emphasize larger errors, making them more suitable for scenarios where minimizing significant deviations is crucial. By comparing these metrics, one can gain a comprehensive understanding of model performance, balancing between sensitivity to outliers (MSE and RMSE) and simplicity of interpretation (MAE).

# 5 Results and Discussion

# 5.1 Data Description

The dataset utilized in this study comprises stock prices from several major companies, including Google, Apple, Amazon, JPMorgan, and Meta. The data were sourced from the Yahoo Finance website, covering the period from September 15, 2014, to September 12, 2024. This extensive dataset provides nearly a decade of market activity, enabling a robust evaluation of the models’ forecasting capabilities across various economic cycles, including periods of growth, stability, and volatility.

The selection of these specific companies is strategic. Google, Apple, Amazon, JPMorgan, and Meta are among the most influential and valuable companies globally, each representing different sectors of the economy. By analyzing the stock prices of these companies, we can assess the models’ performance in diverse market environments, which is essential for understanding their generalizability and robustness.

These companies also offer a wealth of historical data, which is crucial for training and testing time series forecasting models. Their stock prices are influenced by various factors, including technological advancements, consumer behavior, global economic trends, and regulatory changes. This complexity makes their data an excellent test case for evaluating the performance of advanced models like TimeGPT and TimeGPT Long Horizon. Moreover, the practical implications of accurate forecasts for these stocks are significant, given their market impact and the investment opportunities they represent.

It is important to note that the focus of this study was not on performing an exhaustive comparison with traditional time series models like ARIMA or LSTM. Instead, the aim was to explore and evaluate the capabilities of newer methodologies, particularly those based on large language models (LLMs). While ARIMA and LSTM have proven effective in various forecasting tasks, this study prioritizes the investigation of novel approaches that leverage LLMs for time series analysis. Therefore, comparisons to traditional methods were intentionally omitted to maintain the focus on assessing the innovations presented by these advanced models.

The dataset has been divided into two distinct sets: a training set containing data from September 15, 2014, to August 21, 2024, and a test set with data from August 22, 2024, to September 12, 2024.

# 5.2 Model Performance Comparison

In this section, we present and analyze the results obtained from the different models employed in our study, with a particular focus on TimeGPT and TimeGPT Long Horizon, the models that are central to this research. These models are compared against more recent and specialized models, including NBEATS, NHITS, PatchTST, Chronos, and iTransformer, to evaluate their performance in time series forecasting tasks. The key metrics used for comparison are the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE), as shown in Table 1. These metrics provide a comprehensive understanding of the models’ accuracy and the nature of their forecasting errors.

Table 1: Performance Metrics for Time Series Forecasting Models   

<table><tr><td>Company</td><td>Model</td><td>MAE</td><td>MSE</td><td>RMSE</td></tr><tr><td rowspan="7">Google</td><td>TimeGPT</td><td>7.9756</td><td>100.5047</td><td>10.0252</td></tr><tr><td>TimeGPT Long Horizon</td><td>9.9456</td><td>152.0701</td><td>12.3317</td></tr><tr><td>NBEATS</td><td>6.5214</td><td>62.1442</td><td>7.8832</td></tr><tr><td>NHITS</td><td>7.1026</td><td>72.3332</td><td>8.5049</td></tr><tr><td>PatchTST</td><td>12.3553</td><td>205.5839</td><td>14.3382</td></tr><tr><td>KAN</td><td>9.4451</td><td>132.4405</td><td>11.5083</td></tr><tr><td>Chronos</td><td>11.9338</td><td>199.9801</td><td>14.1414</td></tr><tr><td rowspan="7">Apple</td><td>TimeGPT</td><td>3.5246</td><td>14.9491</td><td>3.8664</td></tr><tr><td>TimeGPT Long Horizon</td><td>4.2878</td><td>27.4146</td><td>5.2359</td></tr><tr><td>NBEATS</td><td>6.0787</td><td>43.0489</td><td>6.5612</td></tr><tr><td>NHITS</td><td>3.5188</td><td>15.3355</td><td>3.9161</td></tr><tr><td>PatchTST</td><td>2.5341</td><td>9.0335</td><td>3.0056</td></tr><tr><td>KAN</td><td>4.8047</td><td>35.2323</td><td>5.9357</td></tr><tr><td>Chronos</td><td>4.0832</td><td>19.8610</td><td>4.4566</td></tr><tr><td rowspan="7">Amazon</td><td>TimeGPT</td><td>3.7368</td><td>21.7460</td><td>4.6633</td></tr><tr><td>TimeGPT Long Horizon</td><td>4.5977</td><td>28.2077</td><td>5.3111</td></tr><tr><td>NBEATS</td><td>10.1972</td><td>132.3228</td><td>11.5032</td></tr><tr><td>NHITS</td><td>5.4653</td><td>52.1561</td><td>7.2219</td></tr><tr><td>PatchTST</td><td>9.2496</td><td>120.4989</td><td>10.9772</td></tr><tr><td>KAN</td><td>3.9976</td><td>23.8527</td><td>4.8839</td></tr><tr><td>Chronos</td><td>4.2931</td><td>30.2459</td><td>5.4996</td></tr><tr><td rowspan="7">JPMorgan</td><td>TimeGPT</td><td>5.1384</td><td>42.8329</td><td>6.5447</td></tr><tr><td>TimeGPT Long Horizon</td><td>6.2243</td><td>70.6893</td><td>8.4077</td></tr><tr><td>NBEATS</td><td>8.0474</td><td>119.2264</td><td>10.9191</td></tr><tr><td>NHITS</td><td>5.7487</td><td>56.8632</td><td>7.5408</td></tr><tr><td>PatchTST</td><td>6.9111</td><td>68.4265</td><td>8.2720</td></tr><tr><td>KAN</td><td>5.6636</td><td>36.7187</td><td>6.0596</td></tr><tr><td>Chronos</td><td>5.2749</td><td>38.5122</td><td>6.2058</td></tr><tr><td rowspan="7">Meta</td><td>TimeGPT</td><td>19.8728</td><td>471.8949</td><td>21.7231</td></tr><tr><td>TimeGPT Long Horizon</td><td>28.0251</td><td>875.1672</td><td>29.5832</td></tr><tr><td>NBEATS</td><td>72.0364</td><td>5674.8426</td><td>75.3316</td></tr><tr><td>NHITS</td><td>56.9580</td><td>3717.3620</td><td>60.9702</td></tr><tr><td>PatchTST</td><td>60.2837</td><td>4064.3690</td><td>63.7524</td></tr><tr><td>KAN</td><td>25.0792</td><td>789.8932</td><td>28.1050</td></tr><tr><td>Chronos</td><td>28.3446</td><td>1067.5623</td><td>32.6736</td></tr></table>

# 5.3 Performance of Forecasting Models Across Different Companies

In this section, we analyze and compare the performance of various time series forecasting models, including TimeGPT, TimeGPT Long Horizon, NBEATS, NHITS, PatchTST, and KAN, across five major companies: Google, Apple, Amazon, JPMorgan, and Meta. The evaluation metrics considered are Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE), as presented in Table 1.

# 5.3.1 Google

For Google, NBEATS exhibits the lowest MAE (6.5214), MSE (62.1442), and RMSE (7.8832), indicating superior accuracy compared to other models. TimeGPT and NHITS also perform competitively with MAE values of 7.9756 and 7.1026, respectively. TimeGPT Long Horizon shows slightly higher errors (MAE: 9.9456, MSE: 152.0701, RMSE: 12.3317), suggesting reduced accuracy over extended forecasting periods. PatchTST and KAN have the highest errors among the models for Google, with MAE values of 12.3553 and 9.4451, respectively. Overall, while NBEATS leads in performance, TimeGPT remains a strong contender in forecasting Google’s stock prices.

# 5.3.2 Apple

In the case of Apple, PatchTST achieves the lowest MAE (2.5341), MSE (9.0335), and RMSE (3.0056), outperforming all other models. NHITS follows closely with MAE, MSE, and RMSE values of 3.5188, 15.3355, and 3.9161, respectively. TimeGPT and TimeGPT Long Horizon show moderate performance with MAE values of 3.5246 and 4.2878. NBEATS and KAN exhibit higher errors (MAE: 6.0787 and 4.8047), indicating less effective forecasting for Apple’s stock prices. PatchTST stands out as the most accurate model for Apple, demonstrating its capability to capture the underlying trends in less volatile stock data.

# 5.3.3 Amazon

For Amazon, KAN achieves the lowest MAE (3.9976), MSE (23.8527), and RMSE (4.8839), indicating robust performance. TimeGPT and TimeGPT Long Horizon also perform well with MAE values of 3.7368 and 4.5977, respectively. NHITS shows moderate accuracy (MAE: 5.4653), while PatchTST and NBEATS have higher errors (MAE: 9.2496 and 10.1972). KAN outperforms NBEATS and PatchTST, demonstrating its effectiveness in handling Amazon’s stock data, which may involve more complex patterns compared to Apple.

# 5.3.4 JPMorgan

In the context of JPMorgan, NBEATS achieves the lowest MAE (8.0474), MSE (119.2264), and RMSE (10.9191), indicating strong forecasting capabilities. TimeGPT and NHITS also perform reasonably well with MAE values of 5.1384 and 5.7487, respectively. TimeGPT Long Horizon shows higher errors (MAE: 6.2243, MSE: 70.6893, RMSE: 8.4077) compared to TimeGPT. PatchTST and KAN exhibit comparable performance with MAE values of 6.9111 and 5.6636. Overall, while NBEATS leads in accuracy, KAN and TimeGPT also demonstrate effective forecasting for JPMorgan’s financial data.

# 5.3.5 Meta

For Meta, all models exhibit higher error metrics compared to other companies, reflecting the increased volatility and complexity of Meta’s stock data. TimeGPT records a MAE of 19.8728, MSE of 471.8949, and RMSE of 21.7231. TimeGPT Long Horizon further increases these errors with MAE, MSE, and RMSE values of 28.0251, 875.1672, and 29.5832, respectively. KAN shows slightly better performance with MAE of 25.0792, MSE of 789.8932, and RMSE of 28.1050. NHITS and PatchTST report MAE values of 56.9580 and 60.2837, respectively, indicating significant challenges in accurately forecasting Meta’s stock prices. NBEATS performs the worst among the models for Meta, with MAE, MSE, and RMSE values of 72.0364, 5674.8426, and 75.3316. These results highlight the difficulty of forecasting highly volatile stocks and suggest that all models, including TimeGPT and KAN, require further refinement to handle such complexities effectively.

# 5.3.6 Overall Comparative Analysis

Across all companies, TimeGPT and TimeGPT Long Horizon demonstrate strong performance in less volatile markets such as Apple and Amazon, with competitive MAE, MSE, and RMSE values. NBEATS and NHITS also perform well in certain contexts but show significant variability across different companies. PatchTST excels in specific scenarios like Apple but struggles with more volatile stocks like Meta. KAN offers competitive performance, particularly in handling complex and moderately volatile data, as seen with Amazon and JPMorgan.

However, all models encounter substantial challenges when forecasting highly volatile stocks like Meta, evidenced by elevated error metrics across the board. This underscores the inherent difficulty in predicting such markets and highlights the need for model enhancements to better capture the dynamics of volatile stock data.

TimeGPT Long Horizon generally exhibits higher error metrics compared to TimeGPT, especially in extended forecasting periods, indicating a trade-off between forecast horizon and accuracy. Nonetheless, both models maintain reasonable performance in stable market conditions, making them suitable for a wide range of forecasting applications. The inclusion of KAN data provides additional insights, showcasing its potential as a viable alternative with competitive accuracy in less volatile environments.

Overall, the comparative analysis reveals that while TimeGPT and KAN are effective in stable markets, their performance diminishes in highly volatile conditions. This suggests the importance of selecting appropriate models based on the specific characteristics of the stock data and the market environment, as well as the potential for further model refinement to enhance forecasting accuracy in complex scenarios.

# 5.4 Visualization of Results

To further illustrate the differences in model performance, we present graphical comparisons of the actual versus predicted values across the different models for all companies studied, including Google, Apple, Amazon, JPMorgan, and Meta. These visualizations highlight how each model’s predictions align with the actual data, providing a visual context to the numerical metrics discussed.

![](images/5d71104a0997b15887aea3870b1b5402846077804c1d4f87745532408d4d6720.jpg)  
Figure 7: Google - Actual vs. Predicted Values

![](images/647c413dfb3fa5365c7de76abedcff2c9ed960af4385a09f64015d0e39d59356.jpg)  
Figure 8: Apple - Actual vs. Predicted Values

![](images/598acef47bcfe494d89cd60211ce44a94ea3bd410dbd79569cadfc7eefc5a016.jpg)  
Figure 9: Amazon - Actual vs. Predicted Values

![](images/88538cc87ccf906229566b2ab01b355fb0c1318be35c894f22fb3a59aab893f7.jpg)  
Figure 10: JPMorgan - Actual vs. Predicted Values

![](images/44b3dd04fabb551d0f39b9ed5edfc82e72c7b648e06c6d92515e933520a3fc9b.jpg)  
Figure 11: Meta - Actual vs. Predicted Values

In the graphs, it is evident that TimeGPT Long Horizon and NHITS maintain closer alignment with the actual values, particularly in periods where there are significant fluctuations, such as in Google’s stock. This consistency is reflected in their lower error metrics. In contrast, other models like NBEATS and PatchTST show more deviation from the actual values, particularly in more volatile stocks like Meta and Apple, which correlates with their higher MAE, MSE, and RMSE scores.

The visual comparison underscores the importance of selecting the right model for the specific forecasting task at hand. While TimeGPT Long Horizon and NHITS excel in scenarios with significant variability, other models like NBEATS and PatchTST may be more suited to data with clearer trends and patterns. The choice of model should be guided by the characteristics of the data and the specific requirements of the forecasting task.

# 5.5 Conclusion

This paper provides valuable insights into the application of Large Language Models (LLMs) for time series forecasting, particularly in financial markets. TimeGPT and its Long Horizon variant demonstrate strong performance in stable market environments, while specialized models like KAN and PatchTST show potential in handling more complex data scenarios. However, all models encounter challenges when forecasting highly volatile stocks, as seen in the case of Meta, where error metrics increase significantly. This underscores the need for further refinement and adaptation of LLM-based models to better handle volatile market conditions. Overall, the results highlight the promising role of LLMs in time series forecasting, but selecting the appropriate model based on market characteristics remains essential. Future work should continue to explore advanced techniques to improve accuracy and robustness in diverse forecasting scenarios. One important avenue would be specialized LLM Times series models only trained in financial time series data.

# References

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., Zschiegner, J., Maddix, D. C., Wang, H., Mahoney, M. W., Torkkola, K., Wilson, A. G., Bohlke-Schneider, M., and Wang, Y. (2024). Chronos: Learning the language of time series.   
Box, G. E. and Jenkins, G. M. (1970). Time series analysis: forecasting and control.   
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. (2020). Language models are few-shot learners.   
Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F., Mergenthaler-Canseco, M., and Dubrawski, A. (2022). N-hits: Neural hierarchical interpolation for time series forecasting.   
Chang, C., Wang, W.-Y., Peng, W.-C., and Chen, T.-F. (2024). Llm4ts: Aligning pre-trained llms as data-efficient time-series forecasters.   
Chen, P.-Y. (2023). Model reprogramming: Resource-efficient cross-domain machine learning.   
Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X., Chen, P.-Y., Liang, Y., Li, Y.-F., Pan, S., and Wen, Q. (2024). Time-llm: Time series forecasting by reprogramming large language models.   
Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., and Long, M. (2023). itransformer: Inverted transformers are effective for time series forecasting. arXiv preprint arXiv:2310.06625.   
Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., and Long, M. (2024). itransformer: Inverted transformers are effective for time series forecasting.   
Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers.   
OpenAI (2023). Gpt-4 technical report.   
Oreshkin, B. N., Carpov, D., Chapados, N., and Bengio, Y. (2020). N-beats: Neural basis expansion analysis for interpretable time series forecasting.   
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019). Language models are unsupervised multitask learners.   
Team, G. (2024). Gemini: A family of highly capable multimodal models.   
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi\`ere, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. (2023). Llama: Open and efficient foundation language models.   
Xu, K., Chen, L., and Wang, S. (2024). Kolmogorov-arnold networks for time series: Bridging predictive power and interpretability.