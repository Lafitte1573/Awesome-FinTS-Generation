# Natural Language Processing and Deep Learning for Bankruptcy Prediction: An End-to-End Architecture

GIANFRANCO LOMBARDO 1, ANDREA BERTOGALLI 2, SERGIO CONSOLI 3, AND DIEGO REFORGIATO RECUPERO 4

1Department of Engineering and Architecture, University of Parma, 43124 Parma, Italy   
2Department of Electronics, Information and Bioengineering, Politecnico di Milano, 20133 Milan, Italy   
3European Commission, Joint Research Centre (DG JRC), 21027 Ispra, Italy   
4Department of Mathematics and Computer Science, University of Cagliari, 09124 Cagliari, Italy

Corresponding author: Gianfranco Lombardo (gianfranco.lombardo@unipr.it)

ABSTRACT Machine and Deep Learning methods are widely adopted to predict corporate bankruptcy events for their effectiveness. Bankruptcy prediction is commonly modeled as a binary classification task over accounting data where the positive label is associated with companies with a high likelihood of bankruptcy and the negative label with a low risk of failure. Most of the models mainly focus on exploiting accounting, stock market data, and data augmentation to deal with the intrinsic unbalance of this task. More recently, financial reports such as the US SEC annual reports have been investigated for feature engineering to boost the accuracy of the classification task. However, these approaches only marginally leverage Natural Language Processing advanced techniques to improve the prediction, by usually only leveraging dictionary-based approaches and word frequencies for feature engineering. These fixed features suffer from concept drift over time leading to weaker predictive models by missing a data-driven architecture to extract text disclosures from financial reports to improve the task. This paper aims to fill the gap between the bankruptcy prediction domain and the recent advances in Natural Language Processing by proposing a Transformer-based architecture that combines: a) a text summarization module that extracts text disclosures from financial reports, over time, by leveraging the self-attention mechanism and learning which contents are more valuable for the prediction in a text communication; b) a multivariate time series modeling for accounting data that is aligned and optimized along with the text module. In this way, the architecture benefits from both data sources for the prediction and ensures continual model adaptation over time. We focused on public companies listed in the American stock market with a dataset including 6190 companies from 1999 to 2018. We have deeply analyzed the contribution of the two proposed modules, the accounting time series module (Accuracy $78 \%$ ) and the text disclosures module (Accuracy $8 1 \%$ ) to finally prove that a unique model that can leverage both data sources at the same time achieves better performance (Accuracy $8 7 . 5 \%$ ). The architecture also outperforms the other baselines for Recall of default events (0.84) and for type II error (16.12).

INDEX TERMS Bankruptcy prediction, deep learning, text disclosure, transformer, text classification, SEC filing.

# I. INTRODUCTION

Within the financial domain, corporate bankruptcy represents one of the worst scenarios for investors and creditors suffering financial damage that cannot be understated since it may

The associate editor coordinating the review of this manuscript and approving it for publication was Rajeeb Dey

further propagate large-scale economic recessions. Indeed, as shown in the 2007-2008 financial crisis, these negative events deeply impacted and influenced the entire economy with strong negative social costs.1

Thanks to the widespread diffusion and evolution of machine and deep learning technologies, today, we have strong tools to monitor, analyze, and identify patterns that may bring potential bankruptcy [1].

The bankruptcy prediction task is often modeled as a binary classification task, where a positive label is associated with a company with a high likelihood of bankruptcy in the next year, and a negative label is associated with healthy companies with a low risk of failure. The main challenge in this domain is having a classifier presenting a high recall over the bankruptcy class by minimizing the number of false positives as much as possible.

An accurate bankruptcy forecasting model is valuable to different stakeholders, such as: i) regulators, to monitor the financial health of institutions and curb systemic risks; ii) stock market investors, to avoid unproductive investments and loss; iii) bank and credit lenders, to define their credit risk models; iv) academics, that need to estimate corporate distress risk to calibrate theoretical financial model, such as explaining anomalies in the standard Capital Asset Pricing Model (CAPM) [2].

The most consolidated approach for bankruptcy prediction is considering each firm under two main points of view: a) the financial status over the year by considering the accounting data from the company’s financial statement; b) the firms’ stock market trading information when considering a public company. A common element between accounting data and stock market data is their numeric and well-structured format (tabular data). Several successful ML and DL models exploiting this information have been proposed in recent years, with the best results achieved by leveraging Bagging and Boosting models [3] and Neural Networks [4], [5]. However, ML and DL models usually need large datasets for their effective training and suffer when class imbalance is strong, as is the case in the context of bankruptcy since default events are quite rare. This is a clear limitation, especially when the most important class the model should recognize is the one less represented in the dataset [4].

Data augmentation may contribute to solving the problem with methods using input data to produce features with patterns that resemble real-world events. However, since these methods rely on a few samples of corporate default for the case of bankruptcy dataset, they usually contribute to improving the recall over the bankruptcy class by also providing, as a side effect, a high number of false positives [6]. To improve predictive performance, ensuring a moderate and small number of false positives while dealing with a strong class imbalance is employing more data sources, for example, the financial information coming from text communications that companies use to convey information to the public and investors [7].

Public companies in the American stock market must fill out annual reports to the Security Exchange Commission - SEC.2 Analysts and investors largely consume these documents for their text disclosures. Despite their public availability and easy accessibility, dealing with this kind of data is still challenging because of their length and the need to involve Natural Language Processing (NLP) techniques in the analysis.

However, most of the research considering text communications as additional features to predict bankruptcy mainly focuses on feature extraction in terms of simple word frequencies or by classifying the tone and sentiment of SEC annual reports’ items according to financial-adapted dictionaries. Nevertheless, feature engineering and the manual extraction of features from financial documents can be a limit for the adoption of these models in real scenarios since managers are often incentivized to obfuscate valuable contents with additional boilerplate words or by rephrasing contents to be less identifiable by leveraging fixed features over time [2]. In light of this, designing ML models that learn the probability distribution of each word in wide contexts can reduce the above-mentioned problem [8]. The recent advances in NLP, such as Transformer-based models [9], have still been poorly investigated in the domain of bankruptcy prediction. Few attempts are limited to feature extraction (e.g., sentiment analysis of financial reports) and do not leverage the wide possibilities offered by masked Language models and permutation language models [10] which can learn word embeddings by leveraging the self-attention mechanism taking into account semantics and relationships among words in a sentence. Moreover, the wide diffusion of pre-trained Transformers over millions of heterogeneous documents reduces the dependency of the model from training with only past documents of the same companies and over the use of specific words inside them.

However, the current Transformer models exhibit a fixed-length input that is usually around a few hundred words. Therefore, for long documents transformers require a strategy to detect the most important parts to be processed by the neural network by preserving the context and the most valuable information for bankruptcy prediction.

A second current gap is the lack of an architecture for bankruptcy prediction that can simultaneously leverage accounting time series and textual content and be oriented toward real-scenario applications that require two requirements for continuous adaptation of the predictive models:

• Information extraction from documents should be done without making limiting assumptions about the textual content, such as considering specific chapters of particular financial reports [7], [11]. Removing limiting specifications over the documents enables the continuous adaptation of these models over time and the development of ad-hoc sub-models for specific sectors if needed. • Features from documents should be extracted by taking advantage of Deep Learning capabilities and data-driven procedures that select the best word features while searching for the optimal decision boundaries to distinguish bankruptcy samples from healthy ones by also leveraging other data sources (end-to-end training) that can depict an overall picture about the health status of a firm (e.g., accounting time series). Current approaches mostly propose off-line procedures and feature engineering steps independent from the prediction performance, which can easily lead to bias and concept drift problems over time [12].

In light of this context, the main contributions of this paper are the following:

1) A Transformer-based architecture that combines an NLP module for processing financial reports without any prior assumption about the document format and a multivariate recurrent network (RNN) for modeling accounting data that is aligned and optimized along with the text module.   
2) On the NLP side, we propose a methodology to identify the most promising parts of a financial report to extract valuable text disclosures for bankruptcy prediction from financial documents by leveraging traditional NLP techniques and the self-attention mechanism.   
3) The NLP module acts as a text summarization pipeline that, given a long financial report, achieves single document embedding from Transformers, overcoming the current length limits of these models that require splitting long documents into several chunks that are processed separately.   
4) The multivariate RNN models independently each accounting variable with a sub-network for each financial time series that are aligned. This module is based on an adaptation of the model presented in [13].   
5) For the experiments, We focused on public companies listed in the American stock market with a dataset including 6190 companies from 1999 to 2018.   
6) We have deeply analyzed the contribution of the two proposed modules, the accounting time series module (Accuracy $78 \%$ ) and the text disclosures module (Accuracy $8 1 \%$ ) to finally prove that a unique model that can leverage both data sources with an end-toend training achieves better performance (Accuracy $8 7 . 5 \% )$ . The architecture also outperforms the other baselines for the Recall of default events (0.84) and for type II errors (16.12).

The aligned datasets for the accounting data and the corporate annual reports will be available on request for further investigations.

The remainder of this paper is organized as follows. Section II revises the state of the art for bankruptcy prediction and highlights the current open challenges that motivate this research work. Section III introduces the two data sources we leveraged; Section IV presents the end-to-end architecture we propose in this paper, while Section V and VI describe the sub-modules of the architecture and motivate the main design choices. Section VII reports the computational experiments to prove the benefits of our architecture. Finally, Section VIII ends the paper with conclusions and future works on where we are headed.

# II. RELATED WORKS

Bankruptcy prediction has been a critical issue in credit risk for decades. Several approaches proposed in the literature can be divided into four groups depending on the modeling perspective: statistic linear models, machine learning approaches, ensemble learning approaches, and the recent Deep Learning models. In the following, we will revise all the methods by also discussing the advantages and disadvantages of each one by highlighting at the end the current missings.

# A. STATISTIC AND MACHINE LEARNING MODELS

The first group includes the traditional statistic linear models represented by simple models that use a small number of variables with strong interpretability [14] but poor performance in terms of accuracy, such as Logistic Regression and Multiple Discriminant Analysis [15]. These methods consider accounting variables and financial ratios as input with the goal of estimating early warning scores for corporate bankruptcy.

The second group includes single machine learning models introducing non-linear functions to solve the bankruptcy prediction as a binary classification task with Support Vector Machine and Multi-layer perceptron [1]. These methods have improved short-term predictions’ performance by introducing less interpretable models but with poor performance when considering long-term predictions. [16]

# B. ENSEMBLE MODELS

The third group is the one characterized by the introduction of ensemble learning models [3], [17], [18] such as Bagging, Boosting, Random Forest and Gradient Boosting Decision Tree. These methods enhance bankruptcy prediction performance for several reasons we can summarize in the following points:

1) Robustness to Noisy Data: Financial data can be noisy, with outliers and errors. Ensemble models are typically more robust to noisy data because they rely on multiple base models (e.g., decision trees) and can handle outliers and errors better than linear models [19]   
2) Handling Imbalanced Data: Bankruptcy prediction datasets are often imbalanced, with fewer bankruptcies than non-bankruptcies. Ensemble models can handle imbalanced datasets by adjusting the class weights or using techniques like bagging and boosting, which help in better capturing the minority class [20], [21], [22].   
3) Model Averaging: Ensemble models combine the predictions of multiple base models, reducing the risk of overfitting. Linear models may struggle with overfitting in complex datasets, leading to poor generalization performance [23].   
4) Feature Importance: Ensemble models can provide insights into feature importance, helping analysts understand which variables are most relevant for bankruptcy prediction. This information can be valuable for risk assessment and decision-making [24]

However, ensemble models perform less for long-term prediction when the accounting data are provided as multiple time series or when the aim is predicting the bankruptcy some years ahead of the event [4], [25]

# C. DEEP LEARNING MODELS & NLP

The fourth group is the most recent one and is the one that is coming up with the advances in Deep Learning. These models benefit from a large amount of data to detect complex patterns [26]. However, they also introduce the need to leverage multiple data sources (data fusion) with heterogeneous formats to avoid overfitting problems while keeping the desired qualities already presented for the third group’s models [27]. Currently, Deep Learning models outperform the previous approaches with the side effect of introducing a high computational cost, and complex procedures to design and implement deep networks [25] and to identify and process other data sources beyond accounting variables. For example, in [12] the authors use financial statement data of Japanese listed companies and transform the numerical financial ratio data into grayscale images to be adapted to a Convolutional Neural Network (CNN) characteristics. This model outperforms all the traditional machine learning techniques presented in Sections II-A and II-B. However, treating financial statements as images introduces two main problems:

1) Variables order to build the image: the order in which the variables are arranged severely affects the model’s ability to learn meaningful patterns and makes the procedure less adaptable over time and for out-ofdistribution new samples   
2) Sampling Bias: To avoid overfitting the model when dealing with a strong class imbalance, the authors generated synthetic data with a weighted average of bankruptcy samples. However, this process relies on the characteristics and distributions of the existing dataset. This can result in an inaccurate representation of the real-world data and lead to biased predictions and an increase of false positives in the prediction [6].

More recently, in [13], the authors presented a Multi-head LSTM recurrent neural network that outperforms ensemble models and feed-forward neural networks in the bankruptcy prediction task by modeling financial variables separately with independent recurrent units, enabling them to achieve better results using short-length time-series of accounting data. The RNN module of our end-to-end architecture is inspired by that work.

On the other hand, the solution can be represented using additional data sources beyond accounting variables, such as text communications, which are widely used in financial analysis. Corporate reports are the most important way for external investors to understand a firm’s operating conditions and development trends. The most studied reports are the ones provided by the SEC authority for public companies in the American stock market. This is mainly due to the large-capitalization of the American market but also to the simple procedures of retrieving data through the SEC Edgar platform in different plain-text formats.

Multiple research focused on the type of text-derived features that can be extracted from financial reports and can be grouped as the following:

• Tone and readability: Authors in [28] have proved that a firm’s earnings management level is negatively associated with the readability of the ‘‘Item 7-Management Discussion and Analysis’’ (MD&A) section of the 10-K annual reports and that, in general, annual report readability is a good indicator of future earnings performance of a company. Authors in [29] have further investigated the value of annual reports introducing different measures, including readability, evaluative content, and visual aids to evaluate the health of a company. Researchers in [30] have considered the more recent Transformer-based neural network, a.k.a. BERT [31], to extract the sentiment of the Section MD&A of 10-K reports and have proved that Transformers have improved performance on sentiment analysis rather than word embeddings models for firm health evaluation. • Content similarity over time: Authors in [2] showed that building a portfolio in the American stock market by considering the companies that exhibit changes in terms of words overlapping among consecutive annual reports (10)-K) leads to abnormal returns. In the same line, others in [8] have proved that analyzing the cosine similarity between two consecutive annual reports by leveraging word embeddings leads to outperforming the market and that when the documents keep being similar over time is a healthy signal for the firm. Authors in [32] highlighted the importance of considering the semantics of words more than the intersection of the words in consecutive reports because it reduces the risk of undetected important contents, either because of changing the words or because of just rephrasing.

# D. OPEN PROBLEMS IN NLP FOR BANKRUPTCY

A major lack is that current advances in NLP, such as Transformer networks and pre-trained models, have not been properly exploited in this context. However, these models can leverage a wide knowledge of words and contexts in different domains. They may contribute to reducing the dependence on specific datasets and the concept drift when the need to adapt the models over time arises.

To the best of our knowledge, Mai et al. [7] have proposed the first and only effective deep learning architecture to predict bankruptcy by leveraging accounting variables and textual disclosures from 10-K documents. However, text disclosures are extracted off-line and without considering the text features that can effectively improve the bankruptcy prediction task by relying on the extraction from the MD&A section considering word frequencies with the traditional TF-IDF technique [33] and word embeddings [34]. The combination of financial variables for the year before the event and text features is achieved with a CNN.

Fixed rules to extract word features (e.g., leveraging dictionary-based approaches or word frequencies) are currently one reason that makes current approaches less adaptable over time [8]. Indeed, it is now common for corporate managers to be incentivized to hide important information in the reports with linguistic techniques, such as using boiler-plate words, using sentiment features, and leveraging word frequencies [2]. A data-driven procedure that extracts features from accounting and text data sources while learning patterns for bankruptcy prediction (end-to-end training) could alleviate the above problems.

Moreover, the importance of each item of an annual report for the specific case of bankruptcy prediction has been poorly investigated by often considering only Item 7 (i.e., Section MD&A) as for earnings prediction and without considering that the formal structure in terms of items for an SEC annual report is only suggested and thus not mandatory. Therefore, other document parts can also contain valuable information for bankruptcy prediction. For the same reason, features engineering from financial documents should focus on communicative values and readability over the entire report and not only on the specific popular MD&A section [35].

In light of this, the research work presented in this paper aims to fill these gaps in the literature between the advances in NLP and the bankruptcy prediction task by proposing:

• A NLP methodology to evaluate which items (chapter) of an annual report are more promising for bankruptcy prediction and that reduce the noise contents with a specific strategy based on document-subtraction over time and sentiment analysis; • An end-to-end Deep Learning architecture that can take advantage of the two data sources by simultaneously considering accounting variables in the form of small-time series and the text communications contents coming from financial reports, without limiting assumptions for the only 10-K reports.

# III. DATA

This section introduces the different data we leveraged to train and evaluate our proposed architecture. We considered 6190 public companies in the US stock market (New York Stock Exchange and NASDAQ) with data available in the period from 1999 to 2018. We leveraged two different data sources: accounting variables in the form of time series and the SEC annual reports (10)-K filings) as text communications with valuable information to predict bankruptcy events for such companies. These are described in the following. The dataset is publicly available on GitHub3

# A. ACCOUNTING DATA

As for the accounting data, we referred to the publicly available dataset proposed in [4]. This dataset includes time series of 18 accounting variables for 8,262 public companies (Table 1). Each time series has a maximum length of 5 years; therefore, the dataset provides the possibility of learning to predict bankruptcy from one to five years before the event.

TABLE 1. The 18 accounting variables considered to predict bankruptcy events.   

<table><tr><td>Variablename</td><td>Description</td></tr><tr><td>Current assets</td><td>All the assets of a company that are expected to be sold or used as a result of standard business operations over the next year</td></tr><tr><td>Cost of goods sold</td><td>Ted</td></tr><tr><td>Depreciation and amortization</td><td>Depreciation refers to the lossof value of atangible fixed asset over time (such as property.machinery, buildings,and plant). Amortization refers to the loss of value of intangible assets over time.</td></tr><tr><td>EBITDA</td><td>Earnings before interest,taxes, depreciation and aia:s net income</td></tr><tr><td>Inventory</td><td></td></tr><tr><td>Net Income</td><td>The overall profitability of acompany after all expenses and costs have been deducted from total revenue.</td></tr><tr><td>Total Receivables</td><td>Balance of money due to a firm for goods or services delivered or used but not yet paid for by customers.</td></tr><tr><td>Market value</td><td>Price an asset gets in a marketplace. In our dataset it refers to the market capitalization since companies are publicly traded in the stock market</td></tr><tr><td>Net sales</td><td>Sumof acowaesssminusts</td></tr><tr><td>Total assets</td><td>All the assets,or items of value,abusiness owns Company&#x27;s loans and other liabilities</td></tr><tr><td>Total Long term debt</td><td>that will not become due within one year of the balance sheet date</td></tr><tr><td>EBIT</td><td>Earnings before interest, taxes Profit a business makes after subtracting</td></tr><tr><td>Gross Profit</td><td>all the costs are related to manufacturing and selling its products or services</td></tr><tr><td>Total Current Liabilities</td><td>Sum of accounts payable,accrued liabilities and taxes such as Bonds payable at the end of the year, salaries,and commissions</td></tr><tr><td>Retained Earnings</td><td>The amount of profit a company has left over after paying allits direct costs,indirect costs, income taxes and its dividends to shareholders</td></tr><tr><td>Total Revenue</td><td>The amount of income that a business made from all sales before subtracting expenses. It may include interest and dividends from investments</td></tr><tr><td>Total Liabilities</td><td>Teds</td></tr><tr><td>Total Operating Expenses</td><td>itnorma bsinssincurs throigh</td></tr></table>

Each company has a label for each fiscal year available according to its next year’s status. According to SEC, a company in the American market is declared bankrupt in two cases according to some special text communications that firms’ management must fill for the Security Exchange Commission in case of financial insolvency:

• When the firm’s management files the ‘‘SEC Chapter 11’’ to declare the need to reorganize its business: management continues to run the day-to-day business operations, but all significant business decisions must be approved by a bankruptcy court.   
• When the firm’s management files the ‘‘SEC Chapter $7 ^ { \bullet \bullet }$ : the company stops all operations and goes completely out of business.

In both cases, the fiscal year before the chapter filing is labeled as ‘‘Bankruptcy’’ (mapped as label 1 in our classification task). Otherwise, the company is considered healthy (mapped as label 0 in our classification task).

# B. 10-K SEC FILINGS

A 10-K filing is an annual report that any public company with an income over 10 million dollars and a class of shares held by more than 2,000 people must fill for SEC requirements. These reports are publicly available in HTML and annually published on the EDGAR SEC platform in the first months of the fiscal year.4 A 10-K report summarizes the company’s performance in the stock market, the performance of the business, and the balance sheet in the previous fiscal year. It is organized into four parts with 20 items (sections). Each section covers a specific financial topic, as presented in Table 2.

![](images/806e480ba7d2c1857bac64c201be7cca80baf31568fd75501a17bd4b412946be.jpg)  
FIGURE 1. Rate of bankruptcy in the dataset (2000-2019) with financial variables in the period (1999-2018). The next subdivision in training, validation, and testing is highlighted with different colors.

Feeding a DL model with an entire 10-K document is not trivial because of the generally extensive length of such documents. For this reason, one of the aims of this paper consists of identifying the most promising items (chapters) of the document that may contain valuable information for bankruptcy prediction. In light of this, we have leveraged a pre-processing pipeline to divide each 10-K into its 20 items. Although SEC suggests a specific document format to improve readability and analysis, several companies have introduced their own format and organization of the contents in the last few years. This condition makes the pre-processing to retrieve the items’ division more complex. Therefore, we designed a regex-based parser that divides the content into the SEC items by looking for specific keywords or by analyzing their Table of Contents (when available). We applied the same parsing methodology proposed in [36] to the annual reports corresponding to the 8,262 companies related to the accounting data described in the previous paragraph. The automatic parsing cannot be performed when the underlying annual report does not respect the standard SEC format. Therefore, the dataset is reduced from 8,262 to 6,190 companies. Moreover, as a first pre-processing step, all the images and tables have been deleted by filtering the HTML content of the document. For our purposes, we have collected two consecutive annual reports for each company for the last two years reported in the accounting variables dataset (thus, each analyzed company is at least three years old).

TABLE 2. 10-K report generic structure.   

<table><tr><td>Part1</td><td></td></tr><tr><td>Item 1</td><td>Business</td></tr><tr><td>Item 1A</td><td>Risk Factors</td></tr><tr><td>Item 1B</td><td>Unresolved Staff Comments</td></tr><tr><td>Item 2</td><td>Properties</td></tr><tr><td>Item 3</td><td>Legal Proceedings</td></tr><tr><td>Item 4</td><td>Mine SafetyDisclosures</td></tr><tr><td>Part2</td><td></td></tr><tr><td>Item5</td><td>Market</td></tr><tr><td>Item 6</td><td>Consolidated Financial Data</td></tr><tr><td>Item 7</td><td>Management&#x27;s Discussion and Analysis of</td></tr><tr><td></td><td>Financial Condition and Results of Operations</td></tr><tr><td>Item 7A</td><td>Quantitative and Qualitative Disclosures about Market Risks</td></tr><tr><td>Item 8 Item 9</td><td>Financial Statements Changes in and Disagreements With Accountants</td></tr><tr><td></td><td>on Accounting and Financial Disclosure</td></tr><tr><td>Item 9A</td><td>Controlsand Procedures</td></tr><tr><td>Item 9B</td><td>Other Information</td></tr><tr><td>Part3</td><td></td></tr><tr><td>Item 10</td><td>Directors, Executive Officers,and Corporate Governance</td></tr><tr><td>Item 11</td><td>Executive Compensation</td></tr><tr><td>Item 12</td><td>Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters</td></tr><tr><td>Item 13</td><td>Certain Relationships and Related Transactions,</td></tr><tr><td>Item 14</td><td>and Director Independence Principal Accounting Fees and Services</td></tr><tr><td>Part 4</td><td></td></tr><tr><td>Item 15</td><td>Exhibits,Financial Statement Schedules Signatures</td></tr></table>

# C. DATASET COMPOSITION

Each sample of the dataset represents the financial status of a company according to the accounting data of the last 3 financial years and the corresponding two 10-K reports available. In this way, we can leverage both data sources to evaluate the bankruptcy likelihood.

The dataset regarding companies is split into a training set, a validation set, and a test set on a temporal basis as follows:

• The training set refers to the data from 1999 to 2014. It is composed of 2,739 healthy samples and 403 default cases. • The validation set refers to the data of 2015. It is composed of 206 healthy samples and 27 default cases. • The test set refers to the data from 2016 to 2019. It is composed of 2,743 healthy samples and 72 default cases.

As expected, the dataset is highly imbalanced since more healthy companies exist than bankrupted ones. Figure 1 shows the distribution over the years of bankrupt samples in the dataset. Indeed, companies that go bankrupt each year represent, on average, the $1 \%$ of the companies available in the market.

# IV. END-TO-END ARCHITECTURE

This section introduces the proposed end-to-end architecture that performs bankruptcy prediction by simultaneously leveraging accounting time series and text content from financial reports. Figure 2 depicts the detailed structure of the architecture; each block will be described in detail in the next sections. In brief, the proposed end-to-end architecture is a combined model composed of two main modules:

1) The multi-head long short-term memory (LSTM) neural network [37], which models each accounting variable independently with several LSTMs equal to the number of financial variables considered. It comprises 18 lstm-based heads, where each one processes one of the variables reported in Table 1 in the form of a time series of the last three years available for the company.   
2) The NLP model, mainly based on Transformers [9], that process, as input, the 10-K annual report referred to the last year of the accounting time series processed by the multi-head LSTM. Optionally, this module also takes as input the annual report referred to the previous year for the ‘‘document subtraction’’ operation described in Section V.

For example, if the goal is predicting if company $X$ will fail in 2018 (time $t$ ), for the multi-head LSTM it will be considered the 18 accounting time series referred to as $( t - 1 , t - 2 , t - 3 )$ , which correspond respectively to financial sheets from 2015, 2016, 2017.

On the other hand, the NLP module will consider the annual report for the fiscal year 2017 $( t - 1 )$ and the one from 2016 $( t - 2 )$ for the document subtraction.

The NLP part requires a pre-processing step that is the core of our NLP methodology to support Transformer models with long documents. We referred to this pre-processing step as ‘‘Summarization’’, because it has the fundamental duty of performing an extractive text summarization that shortens the document’s length while preserving the readability and the context of each sentence. Indeed, this step has two main benefits: on the one hand, it acts as an attention strategy that forces the neural network to focus only on the most informative content. On the other hand, it reduces the document’s effective length to meet the maximum length requirement of most current Transformer networks.

The use of Transformers classically requires splitting the original document into several chunks. However, the higher the number of chunks, the more the Transformer suffers from losing a global context. Our desiderata is, therefore, to have the minimum number of chunks while preserving the information to predict bankruptcy.

Once obtained, the text summary is then split into a few chunks that become the input of a pre-trained model that acts as a document embedding generator. The embedding generator maps the text content into a distributed vector representation [38] according to the Distributional hypothesis that linguistic items with similar distributions have similar meanings [31]. As models for the embedding generator, we have evaluated the DistillBert pre-trained Transformer [39] and the Doc2Vec [40] algorithm. We have evaluated different merging strategies to have a unique embedding that compresses the information for all the chunks.

The two modules were first trained separately for predicting bankruptcy, and afterward, they were concatenated into a single model and fine-tuned again on the bankruptcy prediction task. In our research, the bankruptcy prediction is modeled as a binary classification task where the positive class (1) is associated with a high risk of bankruptcy in the next year, while the negative class (0) is associated with a low risk of failure.

In light of this, the overall architecture considers the two data sources for the bankruptcy but also optimizes the latent representation of the documents directly to predict the bankruptcy with a data-driven procedure and without recurring to features engineering. This point is highlighted because it is the key functionality that enables us to remove the limits of previously presented models reported in Section II, making the architecture suitable for any financial report. It could also use multiple reports to compute the final text embedding.

The next sections describe the single components of the architecture. In particular, Section V describes the NLP module, Section VI discusses the multi-head recurrent neural network that processes the accounting time series. Section VII presents our computational experiments, where we also explain how the NLP and time series modules are arranged to have the final end-to-end architecture.

# V. NLP MODULE

Text communications in finance have various formats and lengths [41]. The NLP module processes a financial report to extract the most relevant parts and to extract a document embedding for the bankruptcy prediction task. Our NLP module can be easily generalized for whatever kind of financial report, given that most of the financial reports required by stock market regulators and institutions worldwide present similar characteristics. The only sufficient (not necessary) condition is providing text contents through time for one of the operations, that we named ‘‘Document subtraction’’. Since we leveraged the SEC annual reports (10-K fillings), we will refer to the structure of these documents.

The NLP module can be divided into two main submodules:

• Extractive text summarization module: which acts as a pre-processor of the text communications and aims at discarding the non-relevant information from the original documents. The main goal of this submodule is also to shorten the original document length to process it with a transformer network while preserving most of the original information.

![](images/4327e3c4f90fc896e4ca0be8bbe079da1849ca12e000ecb4e1e7a9e39a21d957.jpg)  
FIGURE 2. The end-to-end architecture proposed in this research work. The architecture simultaneously leverages accounting time-series (multi-head LSTM) and financial reports with an attention strategy to the most informative contents for bankruptcy prediction, without any assumption over the document.

• Document embedding module: which has the duty of mapping the summary achieved with the previous submodule into a dense vector representation (document embedding) that represents the NLP features for the end-to-end architecture for bankruptcy prediction. If the original document does not require a text summarization (e.g., the report is a short communication), this submodule would take the original document as input. This sub-module is presented in Section V-B

# A. EXTRACTIVE TEXT SUMMARIZATION

The text summarization module plays a fundamental role in shortening the original content by preserving the most informative content as much as possible and reducing redundancy and noise information that does not contribute to bankruptcy prediction. According to [42], 10-K reports have grown substantially longer (from 20K to $4 0 \mathrm { k }$ words on average), more complex, and less readable in the last twenty years. A similar pattern of using highly complex language is also evident in financial news. Thus, shortening the content is necessary to support the subsequent task of document embedding generation since most of the algorithms and techniques that can be leveraged for this activity provide a limited input length. For example, the DistillBERT model [39] that we leveraged for embedding generation can handle inputs limited to 512 tokens, which approximately corresponds to the number of words that can be handled.

The extractive text summarization consists of three different operations:

• Topic selection • Document subtraction • Sentiment analysis

# 1) TOPIC SELECTION

This first operation involved in our text summarization methodology consists of dividing the report into sections according to the financial topic and then preserving in the summary only sentences belonging to topics that are more promising to contribute to bankruptcy prediction. Since 10- K documents are already divided into items with specific topics (see Table 2), we have directly proceeded with the topic selection. For this off-line pre-processing step, we have leveraged traditional NLP techniques for a prior decision about which items’ content is preserved in the summary and which is not.

For this preliminary task, we have compared a Bagof-Word encoding (BoW) [33] of the items and a vocabulary-based encoding using the Loughran & McDonald’s financial dictionary (M&L) [43]. This dictionary provides the following NLP features for each item: number of words, percentage of words expressing negative feeling, percentage of words expressing a positive feeling, percentage of words expressing a feeling of uncertainty, percentage of words expressing an argumentative feeling, percentage of words expressing an order (strong modal verbs), percentage of words expressing a suggestion (weak modal verbs), percentage of words expressing an imposition feeling, number of alphabetic letters, number of numbers, the average number of syllables per word, the average length of words. For both encondings, we trained a Logistic classifier.

We randomly balanced the training and validation sets regarding the number of healthy and bankrupt companies. Since our aim was to understand which items of a 10-K report would be more promising for their contents, we evaluated the performance of each item’s content as inputs for predicting bankruptcy. We trained independent classifiers, two for each item: one using the BoW encoding and one with the M&L encoding. To compare the models, we performed ten different runs and considered the average accuracy reached in the bankruptcy prediction task. Figure 3 depicts the results of this preliminary analysis achieved with both encodings for each item included in the SEC report. We finally considered the items that enabled us to achieve the highest accuracy over the balanced validation sets. According to the accuracy achieved by this preliminary experiment, we can conclude that the most promising items to predict bankruptcy events are items 1, 5, and 7, which correspond respectively to the general description of the business and revenues (item 1), the current trends that the company is facing over the stock market (item 5), and, finally, the management discussion which discusses future plans and market risks (item 7). We should highlight that for the analysis, we have merged item 1 with item 1A, item 1B, and item 7 with item 7A because not all companies present these specific items in their documents.

Given these results, we have considered only the sentences belonging to such items for the text summarization. In addition, we have also evaluated with this methodology the general contribution of the entire document, proving that, without deep learning models, the SEC reports provide, in general, relevant insights for bankruptcy prediction (the average accuracy with BoW is equal to $6 5 \%$ ).

# 2) DOCUMENT SUBTRACTION

The second step we have leveraged for the extractive text summarization is based on the year-on-year (yoy) similarities among consecutive reports that have been deeply analyzed by [2] and [8]. It has emerged that it is a common procedure for most companies’ management to copy and paste contents from the previous reports when filling the new ones, especially for the generic consideration still valid for the firm. Therefore, detecting yoy changes is crucial to focus the ML models only on the new relevant content. We have referred to such an operation as ‘‘document subtraction’’. Given two text sequences of arbitrary length, the subtraction consists of determining which parts have been either modified, added, or deleted, and matching the equal parts. This operation enables us to further reduce the content of the document achieved with the topic selection, since most companies exhibit a yoy similarity of around $90 \%$ , meaning that $90 \%$ of the copies report is the same.

# 3) SENTIMENT ANALYSIS

The third step that concludes the text summarization module is related to sentiment analysis. We hypothesize that neutral contents express less information to predict bankruptcy and should be discarded. Indeed, neutral content is usually used more for narrative reasons and to report descriptions and details, rather than negative risks or positive revenue [29]. For this step, we have leveraged a first pre-trained Transformer, FinBERT [44], which is a pre-trained NLP model to analyze the sentiment of financial texts. It is built by further training the BERT language model in the finance domain, using a large financial corpus and thereby fine-tuning it for financial sentiment classification.

To use FinBERT for the sentiment analysis, it is required to divide the text summary achieved with the ‘‘document subtraction’’ into text chunks of 256 tokens to deal with the maximum input length supported by the model. The Transformer classifies each chunk into three possible classes, i.e. {Positive, Neutral, Negative}. The chunks classified as neutral have been then discarded.

This step concludes the text summarization, finally providing a text dataset composed of a diverse number of text chunks for each company.

# B. DOCUMENT EMBEDDING

Once the original report has been summarized into $n$ chunks, the architecture aims at learning an optimal embedding for the summary. We have evaluated two different approaches to achieve such embeddings for this task:

• Doc2Vec [40]: This algorithm learns distributed vector representations for paragraphs, regardless of their length, while learning word feature vectors using a shallow and linear neural network.

• DistilBERT [39]: This pre-trained model leverages transfer learning and Knowledge Distillation [45] during the pre-training phase to obtain a BERT model $40 \%$ smaller than the original large language model, which can be easily deployed in a production environment with similar performances to the original BERT Transformer network. Despite that, DistilBert remains a model with about 66 million parameters. We have fine-tuned such a model to perform the bankruptcy prediction task given each chunk that composes the text summary since the model has an input length limited to 512 tokens. We have achieved an embedding for each chunk by extracting the latent representation of each chunk in the last hidden layer of the network that maximizes the linear separability between healthy samples and bankrupted ones. Since in this way we have obtained chunk-level embeddings, we have then evaluated two different policies to merge these embeddings and obtain the summary’s document embedding.

The two approaches offer various benefits and return a document embedding with a different quality. Doc2Vec offers more flexibility in terms of input and output length, while DistilBERT, in general, offers a better quality of embedding. Moreover, the document embedding leveraging DistilBERT by optimizing the bankruptcy prediction is one of our main research contributions. In fact, DistilBERT is not designed for document embedding but only as a large language model to be fine-tuned for specific classification tasks and therefore does not explicitly keep into account semantics or words’ order.

![](images/f3cafe1f68193bebae7241bebb85cc317d441d35b680e2c16004b91be20355de.jpg)  
FIGURE 3. Preliminary topic selection results: (a) Average accuracy per item using the BoW encoding. (b) Average accuracy per item using the Loughran & McDonald’s encoding.

# 1) Doc2Vec EMBEDDING

In Doc2Vec, word embedding of word $i$ is learned by predicting the surrounding words (window) so that similar words should have similar embeddings in terms of spatial proximity. Thus, the more distant words are, the less they are related to the current word to be embedded. Words are fed into the linear neural network, and the output layer is implemented with a hierarchical Softmax function [1]. Values from the hidden layers are then the resulting embedding vectors. This means that syntax and semantics are captured as the indirect result of predicting the next word in a sentence. Given a sequence of training words $w _ { 1 } , w _ { 2 } , w _ { 3 } , \ldots , w _ { T }$ , the objective is to maximize the average log probability:

$$
{ \frac { 1 } { T } } \sum _ { t = k } ^ { T - k } \log p \left( w _ { t } | w _ { t - k } , . . . , w _ { t + k } \right)
$$

where:

$$
p \left( w _ { t } | w _ { t - k } , \ldots , w _ { t + k } \right) = \frac { e ^ { y _ { w _ { t } } } } { \sum _ { i } e ^ { y _ { i } } }
$$

In other words, it learns a word feature vector by predicting its context in a window of surrounding words preserving the order. Each output of $y _ { i }$ is the unnormalized log-probability for each output word $i$ , and it is computed as:

$$
y = b + U h \left( w _ { t - k } , \dots , w _ { t + k } ; W \right)
$$

where $U , b$ are the Softmax parameters. The functional $h$ is constructed by a concatenation or an average of the word vectors extracted from the weights matrix $W$ .

The document embedding in Doc2Vec is estimated by introducing a special token in the text that acts as an additional word and as memory to remember what is missing from the current context (window of surrounding words). Thus, the paragraph token has an associated paragraph vector to be learned. This vector is shared with all the windows that the algorithm uses to learn the other word embeddings for the same document. Each word feature vector that compounds the W word vector matrix is shared across the other paragraphs.

# 2) DistilBERT DOCUMENT EMBEDDING

DistilBERT leverages a knowledge distillation learning process [45], which is a compression technique in which a compact model, referred to as ‘‘the student’’, is trained to reproduce the behavior of a larger model (‘‘the teacher’’) that, in this case, corresponds to a large BERT uncased model. DistilBERT is trained with a distillation loss over the soft target probabilities of the teacher, i.e.

$$
{ L _ { c e } } = \sum _ { i } { t _ { i } } * \log ( { s _ { i } } ) ,
$$

where $t _ { i }$ is the probability for the class estimated by BERT with the Softmax activation function and $s _ { i }$ is the one estimated by the student DistilBERT. The final training loss of the student is a linear combination of the distillation loss $L _ { c e }$ and the supervised NLP training loss, i.e. the ‘‘masked language modeling loss’’ of the original BERT [31]. The masked loss is used for self-supervising learning by solving a masked language task where $15 \%$ of all input tokens are randomly selected for replacement with a special [MASK] token. The model has to predict the the original value of the masked words.

DistilBERT exhibits the same general architecture as BERT in terms of input processing and in terms of the last tensor shape that we have leveraged to extract the chunk embeddings. Most of the operations used in the Transformer architecture (linear layer and layer normalization) preserve the self-attention mechanism, although highly optimized by learning from half of the original BERT layers.

The embedding generation process begins with a tokenization phase where, for each chunk, we add the special tokens [CLS] and [SEP] that the Transformer uses for the self-attention mechanism. To extract the embedding of each chunk, we need to access DistilBERT’s last hidden_state after it has been fine-tuned using our training set and validated with the validation set. The last hidden_state is a three-dimensional tensor where, on the $\mathbf { X }$ -axis, we have the number of tokens for each chunk, on the y-axis, instead, we have the number of chunks, and finally, on the z-axis, we have the embedding dimension (i.e., 768). Only the [CLS] token is necessary for our purposes since we need to embed the whole chunk corresponding to the tensor’s first row.

Once our fine-tuned DistilBERT has computed the embeddings for all the chunks, every company is described by a variable number $n$ of vectors according to the number of chunks with 256 tokens composing the text summary. We have finally evaluated two merging strategies to achieve a unique embedding vector for each company.

# 3) EMBEDDING MERGING STRATEGIES

To grant the end-to-end property of the architecture, every company has to be described with an input with the same shape for the NLP part. Consequently, we combine all the embeddings for each company into a single vector of 768 features. For this purpose, two main techniques have been evaluated: the Hadamard product [46] and the element-wise mean. The idea behind these two methods derives from [47].

The Hadamard product (also known as Schur product or element-wise product) can be defined as a binary operator that, given two vectors of equal size, outputs a vector with the same size of the inputs, where each element $i$ is the product between the $i \cdot$ -th elements in the input vectors.

$$
\nu \left. \boldsymbol { \Theta } \right. u = \left[ \begin{array} { l } { \nu _ { 0 } } \\ { \nu _ { 1 } } \\ { \vdots } \\ { \nu _ { n } } \end{array} \right] \overset { \textstyle \left[ \begin{array} { l } { \boldsymbol { u } _ { 0 } } \\ { \boldsymbol { u } _ { 1 } } \\ { \vdots } \\ { \boldsymbol { u } _ { n } } \end{array} \right] } { \mathop { : } } = \left[ \begin{array} { l } { \nu _ { 0 } \cdot u _ { 0 } } \\ { \nu _ { 1 } \cdot u _ { 1 } } \\ { \vdots } \\ { \nu _ { n } \cdot u _ { n } } \end{array} \right]
$$

To apply the Hadamard product to reduce the embedding vectors, we iterate the product on the vectors and, finally, obtain a single vector of 768 elements. Formally, we have:

$$
E _ { C I K } = E ^ { ( 0 ) } \mapsto \cdots \mapsto E ^ { ( n ) } = \left[ \begin{array} { c } { E _ { 0 } ^ { ( 0 ) } \cdot E _ { 0 } ^ { ( 1 ) } \cdot \hdots \cdot E _ { 0 } ^ { ( n ) } } \\ { E _ { 1 } ^ { ( 0 ) } \cdot E _ { 1 } ^ { ( 1 ) } \cdot \hdots \cdot E _ { 1 } ^ { ( n ) } } \\ { \vdots } \\ { E _ { 7 6 8 } ^ { ( 0 ) } \cdot E _ { 7 6 8 } ^ { ( 1 ) } \cdot \hdots \cdot E _ { 7 6 8 } ^ { ( n ) } } \end{array} \right]
$$

where $E _ { C I K }$ is the company embedding.

The other evaluated technique consists of computing the vector $E _ { C I K }$ using the element-wise average, iterated, as for the Hadamard product, on the $n$ vectors for each company. We can formalize this operation as follows:

$$
E _ { C I K } = \frac { 1 } { n } \sum _ { i = 0 } ^ { n } E ^ { ( i ) } = \frac { 1 } { n } \left[ \begin{array} { c } { E _ { 0 } ^ { ( 0 ) } + E _ { 0 } ^ { ( 1 ) } + . . . + E _ { 0 } ^ { ( n ) } } \\ { E _ { 1 } ^ { ( 0 ) } + E _ { 1 } ^ { ( 1 ) } + . . . + E _ { 1 } ^ { ( n ) } } \\ { \vdots } \\ { E _ { 7 6 8 } ^ { ( 0 ) } + E _ { 7 6 8 } ^ { ( 1 ) } + . . . + E _ { 7 6 8 } ^ { ( n ) } } \end{array} \right]
$$

Also, in this case, $E _ { C I K }$ is the company embedding.

We compared the two techniques by leveraging them to create a single document embedding of the result achieved with the text summarization. We then provided such embedding as input to the final network achieved by fine-tuning DistilBERT for bankruptcy prediction, presented in the next sub-section. The comparison between the two techniques, in terms of average accuracy over ten runs, has revealed that both are good ways to join the embedding vectors. However, merging vectors with the Hadamard product revealed a side effect. Indeed, when the number of chunks available is high, the Hadamard becomes unstable. This instability is because if we multiply a number greater than one for various times, the product tends to diverge; on the other hand, if we multiply various times a number between zero and one, the product tends to converge. This behavior can lead to an exploding gradient problem when training the NLP model [37].

TABLE 3. Hyper-parameters for the DistilBERT fine-tuning identified with the random and grid searches.   

<table><tr><td>Number of hidden layers</td><td>1</td></tr><tr><td>Neurons per layer</td><td>5</td></tr><tr><td>Early stopping patience</td><td>10</td></tr><tr><td>Batch size</td><td>30</td></tr><tr><td>Epochs</td><td>326</td></tr><tr><td>Adam learning rate</td><td>0.0001</td></tr></table>

# 4) DistilBERT FINE-TUNING

We have fine-tuned DistilBERT in our architecture by replacing the loss function employed in the masked language task with a binary-cross-entropy loss function to predict bankruptcy, given the unique embedding extracted for each company in the previous step. Fine-tuning is performed by exposing DistilBERT to the contents from 10-K reports and by adding a feed-forward network (FF network) that takes as input the resulting 768 unique embedding vectors computed by the element-wise average layer. In this way, the embedding strategy does not introduce any learnable weight and is excluded in the Back-propagation learning algorithm.

The hyper-parameters of this feed-forward neural network have been identified by combining a random search with a grid search. The hyper-parameters search involves the average of 10 different initializations and runs of DistilBERT $+$ FF Network for each configuration. The configuration that achieves the highest accuracy on the validation set has been selected as the best one. Table 3 shows the optimization result. The Hadamard embedding strategy presents a high variance and worst results in terms of accuracy when compared to the element-wise average strategy, which results in being the best option.

Figure 4 depicts the average accuracy achieved for the bankruptcy prediction task, given the feature vectors of 10-K reports obtained using Doc2Vec and DistilBERT, respectively, using the Hadamard product and element-wise average. Results over a randomly balanced validation set prove that averaging the chunk embeddings from DistilBERT is the optimal strategy. In addition, they also preliminary prove that analyzing 10-K reports may enrich traditional accounting-based classifiers for bankruptcy prediction since we have achieved an $80 \%$ accuracy on average over a randomly balanced validation set by using the summary generated with our methodology.

![](images/2e2ca988957526beb78d01830ac37bb143aa1dd99eafa45bdcf4d6369becaf27.jpg)  
FIGURE 4. Comparison of document embedding techniques as feature vectors for bankruptcy prediction. DistilBERT $^ +$ Hadamard (orange), Doc2Vec (blue), DistilBERT $+ \mathbf { E } \mathbf { l }$ .wise average (red). The accuracy is computed as the average of 10 runs on the validation set (2012-2014).

# VI. TIME-SERIES MODULE

In this section, we present the second component of the architecture, which has the purpose of processing and predicting bankruptcy by leveraging accounting time series data presented in Section III-A. Since the goal is to leverage the last three years of accounting variables, we have designed this module using a Recurrent Neural Network (RNN) [26], [48]. RNNs are a special class of artificial neural networks where connections between nodes form a directed or an undirected graph along a temporal sequence [48]. This allows them to exhibit temporal dynamic behavior and to use their internal state (memory) to process variable-length sequences of inputs. RNNs analyze sequences of values in the form $x ^ { ( 1 ) } , \bar { x } ^ { ( 2 ) } . . . . . , x ^ { ( t ) }$ . This capability arises from the network’s utilization of shared parameters across various recurrent iterations. This parameter sharing facilitates the preservation of generalization across the sequence, as identical weights are employed for every time index, in contrast to a conventional fully connected feed-forward network, which would necessitate distinct parameters for each input feature. Here, the time index indicates the position within the sequence [48]. Furthermore, the model size does not increase with the size of the input, which theoretically can be of any length.

A RNN is generally composed of a single unit of processing that produces an output $y$ at each time step and has recurrent connections in general from the hidden units $a ^ { < j > }$ and optionally from the output. When an RNN has only recurrent connections from the hidden units, it processes an entire sequence and then produces a single output.

Input, output, and recurrent hidden states are propagated using different weight matrices whose elements are learned during training using the back-propagation through time algorithm [49]. Equations 5 and 6 describe the internal behavior of an RNN unit. The initial hidden state $a ^ { 0 }$ generally equals zero. In general, for a generic time index $j$ the hidden state is computed as a weighted sum of the previous state $\boldsymbol { a } ^ { j - 1 }$ and the current input $x ^ { j }$ plus a bias term. After that, an activation function $\sigma$ is applied to the result as in fully connected networks. The output at each time step only depends on the current internal state plus a different bias $b _ { y }$ . The two activation functions to estimate the hidden state and the output may differ.

$$
\begin{array} { l } { { a ^ { < j > } = \sigma ( W _ { a a } \cdot a ^ { < j - 1 > } + W _ { x a } \cdot x ^ { < j > } + b _ { a } ) } } \\ { { y ^ { < j > } = \sigma ( W _ { y a } \cdot a ^ { < j > } + b _ { y } ) } } \end{array}
$$

In this way, RNNs can process entire sequences and can use contextual information when mapping inputs into outputs.

In our application, each company is described by 18 different time series, each with a length equal to 3 years, as previously depicted in Figure 2. It is well-known that LSTMs perform better than GRUs when the sequence in input is short [50], as in our case. For this reason, we have chosen LSTMs as base units for the network in our architecture.

Moreover, we have leveraged a multi-head LSTM, where the network is a composition of different LSTMs, each one modeling independently a different accounting time series. This choice is motivated because, in our preliminary experiments, the performance was better when leveraging a multi-head network rather than a unique LSTM with a matrix input, especially on heavy unbalanced tasks such as bankruptcy prediction.

We could also evaluate more than 3 years of accounting variables. However, increasing the number of years reduces the firm’s samples available in the dataset since few companies have data for a large number of years.

# VII. COMPUTATIONAL EXPERIMENTS

This section describes the main results achieved regarding the generalization capabilities over time of our algorithm for bankruptcy prediction using the test set (2015-2019).

In our experiments, we have trained all the components using data from 1999 until 2011 and have validated all the models using randomly balanced validation sets with data from 2012 until 2014. Finally, we have performed a deep evaluation using a test set with real data from the American market between 2015 and 2019. Since the test set suffers from imbalance, we have performed evaluations with two different settings:

1) On twenty random balanced test sets to compute the average accuracy of our model;   
2) On the entire imbalanced test set for a global picture of the generalization error.

For both cases, balanced and imbalanced, we performed an ablation study, where we have compared the single architecture’s components: the NLP module (Section V) and the multi-head LSTM (Section VI), with the entire end-to-end architecture, that is described in the following subsections.

Moreover, we compared our model with the existing models presented in SectionII.

# A. METRICS

We have evaluated the bankruptcy prediction as a binary prediction task where the positive class (1) indicates bankruptcy in the next year, while the negative class (0) means that a company has been classified as healthy in the next year. For the entire test set, we have used different metrics that look at the imbalanced condition of our bankruptcy prediction task. Consider the following quantities for our case:

• True Positive (TP): The number of actually defaulted companies that have been correctly predicted as bankrupted;   
• False Negative (FN): The number of actually defaulted companies that have been wrongly predicted as healthy firms;   
• True Negative (TN): The number of actually healthy companies that have been correctly predicted as healthy;   
• False Positive (FP): The number of actually healthy companies that have been wrongly predicted as bankrupted by the model.

In a balanced classification problem, the classical metric to evaluate a model performance is the Accuracy, defined in the formula as

$$
A c c u r a c y = \frac { T P + T N } { T P + T N + F P + F N } .
$$

However, considering our highly imbalanced problem setting, we have computed the widely adopted Precision, Recall, and $F _ { 1 }$ scores [1] for each class. This is highlighted because, in predicting bankruptcy, an error has a different cost depending on the class that has been incorrectly predicted. The cost of predicting a company going into default as healthy is much higher than the cost of predicting a company that will default as healthy.

In light of this, the precision achieved for a class is the accuracy of that class’ predictions. The Recall (Sensitivity) is the ratio of the class instances that are correctly detected as such by the classifier. The $F _ { 1 }$ score is the harmonic mean of precision and recall: whereas the regular mean treats all values equally, the harmonic one gives much weight to low values. Consequently, we obtain a large $F _ { 1 }$ score for a certain class only if its precision and recall are high. Equations 8, 9 and 10 report how these quantities are computed for the positive class. On formulas:

$$
\begin{array} { c } { { P r e c i s i o n = \displaystyle \frac { T P } { T P + F P } , } } \\ { { R e c a l l = \displaystyle \frac { T P } { T P + F N } , } } \\ { { F _ { 1 } s c o r e = \displaystyle \frac { 2 } { \frac { T } { P r e c i s i o n } + \frac { 1 } { R e c a l l } } . } } \end{array}
$$

The definition of Precision, Recall, and $F _ { 1 }$ score for the negative class can be obtained from equations 8 and 9 by considering the true negatives and false negatives instead of true positives and false positives, and in equations 9 by considering the false positives instead of the false negatives.

Moreover, we have reported three global metrics that have been selected because they enable an overall evaluation of the classifier on both classes [1]:

• The macro $F _ { 1 }$ score is computed as the arithmetic mean of the $F _ { 1 }$ score of all the classes.

• The micro $F _ { 1 }$ score is used to assess the quality of multi-label binary problems. It measures the $F _ { 1 }$ score of the aggregated contributions of all classes giving the same importance to each sample.

We also considered the Area Under the Curve (AUC) which measures the ability of a classifier to distinguish between classes and is used as a summary of the Receiver Operating Characteristic (ROC) Curve. The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. It is worth noting that AUC can be misleading for high-imbalanced datasets [51] like ours. Indeed, the AUC can reach a high value because the model correctly classifies health companies (the most representative class) rather than bankrupt ones). For this reason, we have also used two further metrics that are often evaluated as benchmarks in bankruptcy prediction models because they do not suffer from any imbalance between the two classes. Since bankruptcy is a rare event, using overall accuracy metrics to measure a model’s performance can be misleading without appropriate analysis since they assume that type I error (Equation 11) and type $\mathbf { I I }$ error (Equation 12) are equally costly. Actually, the cost of false negatives is much greater than the cost of false positives for a financial institution. In light of this, we have explicitly computed and reported type I and type $\mathrm { I I }$ errors:

$$
\begin{array} { r } { T y p e \textit { I } \textit { e r r o r } = \frac { F P } { T N + F P } , } \\ { T y p e \textit { I I } \textit { e r r o r } = \frac { F N } { T P + F N } . } \end{array}
$$

Type II error reports the percentage of bankruptcy samples wrongly classified as healthy. Instead, the type I error outlines the percentage of healthy samples wrongly classified as bankrupt. The two errors are inversely proportional, and therefore, it is unsuitable to minimize both errors. However, the main desiderata for the bankruptcy prediction task is minimizing the type II error by keeping the false positives and, thus, the type I error as small as possible.

Finally, we considered two other metrics: the geometric mean of true positive and true negative rates (G-Mean) [52] and the balanced accuracy for imbalance learning (BAC) [53]. The two metrics give a global picture when comparing models and considering the right trade-off between type I and II errors [51]. The G-Mean (Eq. 13) aims to balance the classification performances of the majority and minority classes. A poor performance in predicting the positive examples will lead to a low G-mean value, even if the negative examples are correctly classified.

BAC (Eq. 14) is an easier-interpreted and simplified version of the G-Mean that performs the arithmetic mean between true positive and negative rates. When the classifier performs equally in both classes or when the classes are equally balanced, this measure is equivalent to conventional accuracy (Eq. 7). However, if the conventional accuracy is high only because the classifier takes advantage of an imbalanced test set, the balanced accuracy will be lower.

![](images/8a5bf2a9d357366b8f64e11cebfaba94b1c4ce68e8629126333bb3917b883c9e.jpg)  
FIGURE 5. The concatenation network architecture for the end-to-end training.

$$
G - M e a n = { \sqrt { { \frac { T P } { T P + F N } } * { \frac { T N } { T N + F P } } } } = { \sqrt { T P R * T N R } }
$$

$$
B A C = { \frac { T P R * T N R } { 2 } }
$$

# B. END-TO-END TRAINING

Our main goal is to design an architecture that can take advantage of leveraging both accounting time series and text communications while predicting the financial status of a firm in the next year. We have presented the two modules we have designed and optimized separately in Sections V and VI.

First, the two models have been trained and validated independently to model the probability distribution of bankruptcy given the accounting time series and the document embedding of the last available 10-K report summary separately.

To achieve a global architecture, we have removed the output layers of the two modules and proceeded with a fine-tuning of their concatenation with an early-stopping setting [26] and with an additional hyperparameters search for the concatenation network. The fine-tuning architecture is depicted in Figure 5. The best concatenation network on the validation set has been identified as the one with just one hidden layer with 20 non-linear neurons.

# C. ABLATION STUDY

We have investigated the contributions of each single component of our proposed architecture. In light of this, we compared the results we achieved with the NLP module (Section V) and the multi-head LSTM (Section VI), with the entire end-to-end architecture within the 20 randomly balanced test-sets. Moreover, we have also compared the two different embedding strategies we leveraged for the NLP module: the Hadamard and the element-wise average. Table 4 shows the result of this analysis. Our time-series module leveraging a multi-head LSTM achieves an average accuracy equal to $78 \%$ . Text communications processed by our NLP module, as in Section V, provide useful insights for bankruptcy prediction independently from accounting variables, with better performance when the embedding of the extracted summary is achieved using the element-wise average over the chunks (accuracy $8 1 \%$ ). This result is, as expected, in line with the results achieved over the validation set (2012-2014). Our end-to-end architecture with the element-wise averaging embedding strategy, presented in Section VII-B, achieves the best result $87 \%$ accuracy). For this reason the element-wise averaging has been choice as the final setting for our end-to-end architecture.

# D. COMPARISON WITH THE STATE OF THE ART

To the best of our knowledge, none of the models presented in the scientific literature have been made publicly available nor open source. In light of this we implemented the state of the art models, presented in Section II to reproduce their results on the American dataset, using the same hyper-parameters reported in their respective research papers.

For the balanced case, we have compared our architecture with the other models that only consider accounting variables, such as the Altman model [54], which is still widely considered as a benchmark, but also the Boosting techniques (Gradient Boosting, AdaBoost and XGBoost) and Random Forest presented for bankruptcy prediction in [3] and [4]. Moreover, we compared with the Deep Learning architecture presented in [13], which basically inspired our multi-head LSTM module.

Table 5 shows the results of this comparison. The Altman Model achieves $7 5 \%$ of average accuracy over the twenty test sets. Boosting models with 500 estimators, as suggested in [3], [17], and [18], underperform the Altman model and, therefore, have been discarded for the next evaluations. In line with the results in the literature, Random Forest with 500 estimators is the only predictive model that achieves better results with an average accuracy of $7 6 . 3 \%$ Moreover, it should be noticed that the second highest accuracy is achieved with our architecture when leveraging the Hadamard embedding strategy $83 \%$ accuracy). These results prove that leveraging both data sources simultaneously enables better prediction and reduces the number of wrong predictions. Unfortunately, a direct comparison with other models leveraging NLP techniques, such as [7], is not trivial because most of the details for the few models presented are not publicly available and however they leverage different text-based data sources rather than the entire SEC annual reports.

This result has been further investigated in the second scenario, where the test set is evaluated overall, by considering its intrinsic imbalance between healthy and bankrupted samples.

For the imbalanced case, we aimed to investigate how the models performed in the real scenario of the American market between 2015 and 2019. In particular, we have analyzed the performance when considering the entire market with all the industry sectors to exclude possible bias in the balanced case. In this context, we leveraged the metrics for imbalance learning presented in Section VII-A.

We compared our final architecture (End-to-End architecture with an element-wise average of Table 5) with its single modules (multi-head LSTM and NLP module) and the

TABLE 4. Ablation study with the balanced test-set (Average accuracy of 20 randomly balanced test sets between 2015 and 2019).   

<table><tr><td>Model</td><td>Configuration</td><td>Average accuracy</td></tr><tr><td>Multi-head LSTM</td><td>■</td><td>78%</td></tr><tr><td>NLP module</td><td>DistilBERT+Element-wise AVG</td><td>81%</td></tr><tr><td>End-to-end architecture with Hadamard</td><td>LSTM+DistilBERT+Hadamard</td><td>83%</td></tr><tr><td> End-to-end architecture with El.wise avg</td><td>LSTM+DistilBERT+Element-wise AVG</td><td>87.5%</td></tr></table>

TABLE 5. Average accuracy of 20 randomly balanced test sets between 2015 and 2019.   

<table><tr><td>Model</td><td>Configuration</td><td>Average accuracy</td></tr><tr><td>GradientBoosting</td><td>500 estimators as in [3]</td><td>72.6%</td></tr><tr><td>AdaBoost</td><td>500 estimators as in [4]</td><td>73.75%</td></tr><tr><td>XGBoost</td><td>500 estimators as in [3]</td><td>74.8%</td></tr><tr><td>Altman Model</td><td>Settings from [14]</td><td>75%</td></tr><tr><td>Logistic Regression</td><td>as in [3]</td><td>75.3%</td></tr><tr><td>Random Forest</td><td>500 estimatorsas in [3]</td><td>76.3%</td></tr><tr><td>Multi-head LSTM</td><td>18 recurrent neural nets as in [13]</td><td>78%</td></tr><tr><td> End-to-end architecture with El.wise avg</td><td>LSTM+DistilBERT+Element-wise AVG</td><td>87.5%</td></tr></table>

TABLE 6. Architecture’s performance over the entire test set compared with its single components (Multi-head LSTM) and NLP.   

<table><tr><td>Model</td><td>Type II eror</td><td>Type Ierror</td><td>Recall</td><td>Precision</td><td>F1-macro</td><td>F1-micro</td><td>AUC</td><td>BAC</td><td>G-Mean</td></tr><tr><td>Random Forest</td><td>20.96</td><td>39.64</td><td>0.79</td><td>0.04</td><td>0.45</td><td>0.65</td><td>0.75</td><td>0.70</td><td>0.69</td></tr><tr><td>Multi-head LSTM</td><td>37.96</td><td>12.74</td><td>0.63</td><td>0.11</td><td>0.56</td><td>0.86</td><td>0.90</td><td>0.75</td><td>0.74</td></tr><tr><td>NLP module</td><td>24.19</td><td>37.25</td><td>0.75</td><td>0.05</td><td>0.43</td><td>0.63</td><td>0.65</td><td>0.69</td><td>0.69</td></tr><tr><td>Ourarchitecture</td><td>16.12</td><td>29.53</td><td>0.84</td><td>0.07</td><td>0.48</td><td>0.71</td><td>0.79</td><td>0.77</td><td>0.76</td></tr></table>

TABLE 7. Confusion matrices for the test set (2015-2019).   

<table><tr><td>Model</td><td>TP</td><td>FP</td><td>TN</td><td>FN</td></tr><tr><td>Random Forest</td><td>49</td><td>949</td><td>1445</td><td>13</td></tr><tr><td>Multi-headLSTM</td><td>39</td><td>305</td><td>2089</td><td>23</td></tr><tr><td>NLP module</td><td>47</td><td>892</td><td>1502</td><td>15</td></tr><tr><td>Our architecture</td><td>52</td><td>707</td><td>1687</td><td>10</td></tr></table>

Random Forest model that achieved the best result within the balanced case. Tables 6 and 7 report the results achieved within this scenario. The results can be summarized as the following:

• Our end-to-end architecture achieves the best results when considering the BAC $( 0 . 7 6 \% )$ and the G-Mean $( 0 . 7 6 \% )$ , meaning that the model can effectively and equally perform well on both classes. • The architecture achieves the lowest type II error $( 1 6 . 1 2 \ \ \% )$ , showing a good ability to recognize bankruptcy events one year before the events.

These performance reasons can be easily detected by considering the results of the two modules (Multi-head LSTM and NLP) to make further considerations. By leveraging only the accounting variables (Multi-head LSTM), the obtained recall for the bankruptcy class is low (column Recall of Table 6). At the same time, recall is high over the healthy class, according to the confusion matrix in Table 7. The opposite expected result can be found for the case of Random Forest, which shows a higher recall for bankruptcy but with a high number of false positives (healthy companies that are wrongly predicted as bankruptcy).

However, depending on the stakeholder, false positives and negatives are not equally important in bankruptcy prediction. Indeed, in this field, a financial stakeholder needs to minimize type II errors, which consider the false negatives (bankrupted companies that have been predicted as healthy). As shown in the tables, the Multi-head LSTM with only accounting variables fails in that sense.

On the other side, the NLP model that leverages only text communications exhibits a better trade-off in terms of performance with a higher recall over bankruptcy events. However, the model fails to minimize the type I error (false positives), presenting a low precision for bankruptcy. However, regarding type I errors, the NLP model performs slightly better than the random forest model.

The two components are almost complementary: by leveraging accounting variables, we can better identify healthy companies, while by using text communications, we can better detect the risky ones with an associated risk of failure. Given this, the lower accuracy for the LSTM reported in Table 5 is more explainable.

The best average accuracy reported by the final end-toend architecture is achieved by taking advantage of the two complementary modules, which enable to reach a very low type II error $( 1 6 \% )$ with a high recall over the bankruptcy examples and with a better trade-off in terms of false positives that lead to a micro $F _ { 1 }$ equal to 0.71. Therefore, text communications improve the decision boundaries by better separating bankruptcy examples from healthy ones in terms of the decision boundary. The end-to-end architecture improves this behavior by also preserving a smaller number of false positives.

# VIII. CONCLUSION AND FUTURE WORKS

This research has described an end-to-end architecture for the bankruptcy prediction problem by leveraging accounting variables and financial documents. We have proposed a general NLP methodology to deal with long financial reports, such as the ones considered for the American market: the SEC 10-K annual reports. From our reported computational experiments, we can conclude that bankruptcy prediction can be effectively improved by also considering text communications and, in particular, large language models outperform traditional NLP models for prediction and document embedding generation. Finally, we have proved that building a deep neural network that can be trained end-to-end with both data sources outperforms the single models without an ensemble learning approach. Furthermore, the proposed architecture is flexible and can deal with longer accounting time series and different financial reports since the NLP methodology can be easily generalized. Limitations of this study that should be reported are related to the few current possibilities to validate this architecture in other financial domains because of the lack of multi-modal datasets (based on different data sources) for bankruptcy prediction and in general data from other countries. At the same time, the design of the architecture using data from 1999 to 2011, its tuning using data from 2012 to 2014 and its evaluation with a real imbalanced scenario with data from 2015 to 2018, prove the ability of such an architecture to adapt over time since the period analyzed involves different financial crisis and growth periods.

Future works might involve different lines of research: i) financial news and stock market prices can be considered as additional data sources which may improve the prediction, especially in some industries and sectors; ii) the large language models domain is rapidly growing and thus, more complex pre-trained models could either simplify the NLP methodology or enable more detailed analyses; iii) transfer learning and self-supervised approaches will be further exploited to improve the prediction from accounting data and to overcome the current limitations encountered with imbalance datasets; iv) The Expected value of perfect information (EVPI) should be considered to define whether or not adopting such technology considering the chance that the decision turns out to be wrong.

# REFERENCES

[1] S. Consoli, D. R. Recupero, and M. Saisana, Data Science for Economics and Finance: Methodologies and Applications. Cham, Switzerland: Springer, 2021.   
[2] L. Cohen, C. Malloy, and Q. Nguyen, ‘‘Lazy prices,’’ J. Finance, vol. 75, no. 3, pp. 1371–1415, Jun. 2020.   
[3] F. Barboza, H. Kimura, and E. Altman, ‘‘Machine learning models and bankruptcy prediction,’’ Expert Syst. Appl., vol. 83, pp. 405–417, Oct. 2017.   
[4] G. Lombardo, M. Pellegrino, G. Adosoglou, S. Cagnoni, P. M. Pardalos, and A. Poggi, ‘‘Machine learning for bankruptcy prediction in the American stock market: Dataset and benchmarks,’’ Future Internet, vol. 14, no. 8, p. 244, Aug. 2022.   
[5] M. Moscatelli, F. Parlapiano, S. Narizzano, and G. Viggiano, ‘‘Corporate default forecasting with machine learning,’’ Expert Syst. Appl., vol. 161, Dec. 2020, Art. no. 113567.   
[6] G. Ranjbaran, D. R. Recupero, G. Lombardo, and S. Consoli, ‘‘Leveraging augmentation techniques for tasks with unbalancedness within the financial domain: A two-level ensemble approach,’’ EPJ Data Sci., vol. 12, no. 1, pp. 1–31, Jul. 2023.   
[7] F. Mai, S. Tian, C. Lee, and L. Ma, ‘‘Deep learning models for bankruptcy prediction using textual disclosures,’’ Eur. J. Oper. Res., vol. 274, no. 2, pp. 743–758, Apr. 2019.   
[8] G. Adosoglou, G. Lombardo, and P. M. Pardalos, ‘‘Neural network embeddings on corporate annual filings for portfolio selection,’’ Expert Syst. Appl., vol. 164, Feb. 2021, Art. no. 114053.   
[9] A. Vaswani, ‘‘Attention is all you need,’’ in Proc. Adv. Neural Inf. Process. Syst., 2017.   
[10] Z. Yang, ‘‘XLNet: Generalized autoregressive pretraining for language understanding,’’ 2019, arXiv:1906.08237.   
[11] H. Kim, H. Cho, and D. Ryu, ‘‘Corporate bankruptcy prediction using machine learning methodologies with a focus on sequential data,’’ Comput. Econ., vol. 59, no. 3, pp. 1231–1249, Mar. 2022.   
[12] T. Hosaka, ‘‘Bankruptcy prediction using imaged financial ratios and convolutional neural networks,’’ Expert Syst. Appl., vol. 117, pp. 287–299, Mar. 2019.   
[13] M. Pellegrino, G. Lombardo, G. Adosoglou, S. Cagnoni, P. M. Pardalos, and A. Poggi, ‘‘A multi-head LSTM architecture for bankruptcy prediction with time series accounting data,’’ Future Internet, vol. 16, no. 3, p. 79, Feb. 2024.   
[14] E. I. Altman, ‘‘Financial ratios, discriminant analysis and the prediction of corporate bankruptcy,’’ J. Finance, vol. 23, no. 4, p. 589, Sep. 1968.   
[15] J. A. Ohlson, ‘‘Financial ratios and the probabilistic prediction of bankruptcy,’’ J. Accounting Res., vol. 18, no. 1, p. 109, 1980.   
[16] L. Nanni and A. Lumini, ‘‘An experimental comparison of ensemble of classifiers for bankruptcy prediction and credit scoring,’’ Expert Syst. Appl., vol. 36, no. 2, pp. 3028–3033, Mar. 2009.   
[17] D. Liang, C.-C. Lu, C.-F. Tsai, and G.-A. Shih, ‘‘Financial ratios and corporate governance indicators in bankruptcy prediction: A comprehensive study,’’ Eur. J. Oper. Res., vol. 252, no. 2, pp. 561–572, Jul. 2016.   
[18] D.-N. Wang, L. Li, and D. Zhao, ‘‘Corporate finance risk prediction based on LightGBM,’’ Inf. Sci., vol. 602, pp. 259–268, Jul. 2022.   
[19] A. Mashrur, W. Luo, N. A. Zaidi, and A. Robles-Kelly, ‘‘Machine learning for financial risk management: A survey,’’ IEEE Access, vol. 8, pp. 203203–203223, 2020.   
[20] S. Carta, A. Ferreira, D. R. Recupero, M. Saia, and R. Saia, ‘‘A combined entropy-based approach for a proactive credit scoring,’’ Eng. Appl. Artif. Intell., vol. 87, Jan. 2020, Art. no. 103292.   
[21] S. Carta, A. Ferreira, D. Reforgiato Recupero, and R. Saia, ‘‘Credit scoring by leveraging an ensemble stochastic criterion in a transformed feature space,’’ Prog. Artif. Intell., vol. 10, no. 4, pp. 417–432, Dec. 2021.   
[22] S. M. Carta, G. Fenu, A. Ferreira, D. R. Recupero, and R. Saia, ‘‘A two-step feature space transforming method to improve credit scoring performance,’’ in Proc. Int. Joint Conf. Knowl. Discovery, Knowl. Eng. Knowl. Manage., 2019.   
[23] D. Liang, C.-F. Tsai, H.-Y. Lu, and L.-S. Chang, ‘‘Combining corporate governance indicators with stacking ensembles for financial distress prediction,’’ J. Bus. Res., vol. 120, pp. 137–146, Nov. 2020.   
[24] W. Lin, Y. Lu, and C. Tsai, ‘‘Feature selection in single and ensemble learning-based bankruptcy prediction models,’’ Expert Syst., vol. 36, no. 1, Feb. 2019, Art. no. e12335.   
[25] Y. Qu, P. Quan, M. Lei, and Y. Shi, ‘‘Review of bankruptcy prediction using machine learning and deep learning techniques,’’ Proc. Comput. Sci., vol. 162, pp. 895–899, Jan. 2019.   
[26] Y. LeCun, Y. Bengio, and G. Hinton, ‘‘Deep learning,’’ Nature, vol. 521, no. 7553, pp. 436–444, 2015.   
[27] A. Chaudhuri and S. K. Ghosh, Bankruptcy Prediction Through Soft Computing Based Deep Learning Technique. Cham, Switzerland: Springer, 2017.   
[28] K. Lo, F. Ramos, and R. Rogo, ‘‘Earnings management and annual report readability,’’ J. Accounting Econ., vol. 63, no. 1, pp. 1–25, 2017.   
[29] A. Seebeck and D. Kaya, ‘‘The power of words: An empirical analysis of the communicative value of extended auditor reports,’’ Eur. Accounting Rev., vol. 32, no. 5, pp. 1185–1215, Oct. 2023.   
[30] A. G. Kim and S. Yoon, ‘‘Corporate bankruptcy prediction with domain-adapted BERT,’’ in Proc. 3rd Workshop Econ. Natural Lang. Process., 2021.   
[31] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, ‘‘BERT: Pre-training of deep bidirectional transformers for language understanding,’’ 2018, arXiv:1810.04805.   
[32] G. Adosoglou, S. Park, G. Lombardo, S. Cagnoni, and P. M. Pardalos, ‘‘Lazy network: A word embedding-based temporal financial network to avoid economic shocks in asset pricing models,’’ Complexity, vol. 2022, no. 1, Jan. 2022.   
[33] M. Lan, C. Lim Tan, J. Su, and Y. Lu, ‘‘Supervised and traditional term weighting methods for automatic text categorization,’’ IEEE Trans. Pattern Anal. Mach. Intell., vol. 31, no. 4, pp. 721–735, Apr. 2009.   
[34] M. J. Kusner, Y. Sun, N. I. Kolkin, and K. Q. Weinberger, ‘‘From word embeddings to document distances,’’ in Proc. 32nd Int. Conf. Mach. Learn., vol. 2, 2015, pp. 957–966.   
[35] T.-K. Chen, H.-H. Liao, G.-D. Chen, W.-H. Kang, and Y.-C. Lin, ‘‘Bankruptcy prediction using machine learning models with the text-based communicative value of annual reports,’’ Expert Syst. Appl., vol. 233, Dec. 2023, Art. no. 120714.   
[36] G. Lombardo, G. Trimigno, M. Pellegrino, and S. Cagnoni, ‘‘Language models fine-tuning for automatic format reconstruction of SEC financial filings,’’ IEEE Access, vol. 12, pp. 31249–31261, 2024.   
[37] S. Hochreiter and J. Schmidhuber, ‘‘Long short-term memory,’’ Neural Comput., vol. 9, no. 8, pp. 1735–1780, Nov. 1997.   
[38] H. Li, ‘‘Language models: Past, present, and future,’’ Commun. ACM, vol. 65, no. 7, pp. 56–63, 2022.   
[39] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, ‘‘DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter,’’ 2019, arXiv:1910.01108.   
[40] Q. V. Le and T. Mikolov, ‘‘Distributed representations of sentences and documents,’’ in Proc. 31st Int. Conf. Mach. Learn., vol. 32, Jan. 2014, pp. 1188–1196.   
[41] L. Barbaglia, S. Consoli, S. Manzan, D. R. Recupero, M. Saisana, and L. T. Pezzoli, ‘‘Data science technologies in economics and finance: A gentle walk-in,’’ in Data Science for Economics and Finance. Cham, Switzerland: Springer, 2021, pp. 1–17.   
[42] D. Lesmy, L. Muchnik, and Y. Mugerman, ‘‘Doyoureadme? Temporal trends in the language complexity of financial reporting,’’ in Temporal Trends in the Language Complexity of Financial Reporting. Amsterdam, The Netherlands: Elsevier, 2019.   
[43] T. M. Arnold, ‘‘When is a liability not a liability? Textual analysis, dictionaries, and 10-ks,’’ CFA Dig., vol. 41, no. 2, pp. 57–59, May 2011.   
[44] D. Araci, ‘‘FinBERT: Financial sentiment analysis with pre-trained language models,’’ 2019, arXiv:1908.10063.   
[45] G. Hinton, O. Vinyals, and J. Dean, ‘‘Distilling the knowledge in a neural network,’’ 2015, arXiv:1503.02531.   
[46] E. Million, ‘‘The Hadamard product,’’ Course Notes, vol. 3, no. 6, pp. 1–7, 2007.   
[47] A. Grover and J. Leskovec, ‘‘node2vec: Scalable feature learning for networks,’’ 2016, arXiv:1607.00653.   
[48] A. Graves and J. Schmidhuber, ‘‘Offline handwriting recognition with multidimensional recurrent neural networks,’’ in Proc. Adv. Neural Inf. Process. Syst., 2008, pp. 545–552.   
[49] A. Gruslys, R. Munos, I. Danihelka, M. Lanctot, and A. Graves, ‘‘Memoryefficient backpropagation through time,’’ in Proc. Adv. Neural Inf. Process. Syst., 2016.   
[50] S. Yang, X. Yu, and Y. Zhou, ‘‘LSTM and GRU neural network performance comparison study: Taking yelp review dataset as an example,’’ in Proc. Int. Workshop Electron. Commun. Artif. Intell. (IWECAI), Jun. 2020, pp. 98–101.   
[51] H. He and Y. Ma, Imbalanced Learning: Foundations, Algorithms, and Applications. Hoboken, NJ, USA: Wiley, 2013.   
[52] M. Kubat and S. Matwin, ‘‘Addressing the curse of imbalanced training sets: One-sided selection,’’ in Proc. ICML, vol. 97, 1997, p. 179.   
[53] D. R. Velez, B. C. White, A. A. Motsinger, W. S. Bush, M. D. Ritchie, S. M. Williams, and J. H. Moore, ‘‘A balanced accuracy function for epistasis modeling in imbalanced datasets using multifactor dimensionality reduction,’’ Genetic Epidemiol., vol. 31, no. 4, pp. 306–315, May 2007.   
[54] E. I. Altman, E. Hotchkiss, and W. Wang, Corporate Financial Distress, Restructuring, and Bankruptcy: Analyze Leveraged Finance, Distressed Debt, and Bankruptcy. Hoboken, NJ, USA: Wiley, 2019.

![](images/81c409f54e894f4ca2e64062a3c96693f5aa9744073703097466fb4ccbc5da1c.jpg)

ANDREA BERTOGALLI received the bachelor’s degree in computer engineering from the University of Parma. He is currently pursuing the master’s degree in computer science and engineering with a specialization in artificial intelligence with Politecnico di Milano. His research interests include deep learning, particularly in computer vision and NLP.

SERGIO CONSOLI received the Ph.D. degree. He is currently a Scientific Project Officer with European Commission, Joint Research Centre (DG JRC), Ispra, Italy, the Competence Centre on Composite Indicators and Scoreboards, and, formerly with the Centre for Advanced Studies on the project: Big data and forecasting of economic developments. Previously, he was a Senior Scientist with the Data Science Department, Philips Research, Eindhoven, The Netherlands, focusing on advancing automated analytical methods used to extract new knowledge from data for HealthTech applications. Other former experiences include Italian Presidency of the Council of Ministers and the National Research Council of Italy. He also provided ICT consultancy services to Isab, the largest oil refinery in the Mediterranean area. His education and scientific experience fall in the areas of data science, operational research, artificial intelligence, knowledge engineering, semantic reasoning, and machine learning. He is the author of several research publications in peer-reviewed international journals, granted EPO and WIPO patents, edited books, and leading conferences in the fields of his work. He also co-edited two books. He is an Associate Editor of PLOS One and Journal of Big Data.

![](images/4e7768f595cde347033cb00bedc99a508ee6e4b649f4853d60c3480d892243ea.jpg)

![](images/d172fc2a887ff4b0c4a3f62c01db91605bc3954dea5470238f8f1d8f564af335.jpg)

DIEGO REFORGIATO RECUPERO received the Ph.D. degree in computer science from the University of Naples Federico II, Italy, in 2004. From 2005 to 2008 he was a Postdoctoral Researcher with the University of Maryland, College Park, USA. He has been a Full Professor with the Department of Mathematics and Computer Science, University of Cagliari, Italy, since February 2022. He has co-founded six companies within the ICT sector and is actively involved in

![](images/6f0d3a0787dc8fb1affe697f4f9679555ba55c8778dbf7cc0a775dfe8111930e.jpg)

GIANFRANCO LOMBARDO received the Ph.D. degree from the University of Parma, in 2021. In Fall 2019, he was a Visiting Researcher with the Center for Applied Optimization, Herbert Wertheim College of Engineering, University of Florida. He has co-founded Neuraloo Inc., an AI tech startup for analytics in the financial domain. He is currently an Assistant Professor with the Department of Engineering and Architecture, University of Parma. His research interests include deep learning and natural language processing for the financial domain.

European projects and research (with one of his companies he won more than 40 FP7 and H2020 projects). His current research interests include sentiment analysis, semantic web, natural language processing, human–robot interaction, financial technology, and smart grids. He is the author of more than 200 conference papers and journal articles in these research fields, with more than 2800 citations. He won different awards in his career, such as the Marie Curie International Reintegration Grant, the Marie Curie Innovative Training Network, the Best Researcher Award from the University of Catania, the Computer World Horizon Award, the Telecom Working Capital, the Startup Weekend, and the Best Paper Award.