# An LLM-Based Framework for Synthetic Data Generation

Mandeep Goyal and Qusay H. Mahmoud Department of Electrical, Computer, and Software Engineering Ontario Tech University Oshawa, ON, L1G 0C5 Canada {mandeep.goyal,qusay.mahmoud}@ontariotechu.net

Abstract—The demand for high-quality datasets is rapidly increasing across sectors such as healthcare, finance, and cybersecurity, yet challenges like data scarcity and privacy concerns persist. To address this, we introduce a framework for synthetic data generation that empowers users to create realistic datasets while maintaining privacy. The framework leverages fine-tuned Large Language Models (LLMs) and differential privacy techniques, including IBM’s diffprivlib, to generate synthetic data that replicates real-world patterns without exposing sensitive information. A proof-of-concept platform has been constructed to facilitate seamless data generation and augmentation, making it particularly useful in scenarios where original datasets are inaccessible, scarce, or privacy-restricted. The platform supports the creation of datasets across five key categories, employing advanced methods to preserve data integrity while ensuring compliance with stringent privacy standards. By combining cutting-edge AI technologies with robust privacy-preserving techniques, this framework offers a practical solution for researchers and professionals seeking reliable synthetic data to drive innovation in data-sensitive fields.

Keywords—synthetic data, differential privacy, large language models (LLMs), pattern preservation, data augmentation & processing

# I. INTRODUCTION

The need for high-quality, privacy-preserving synthetic data is growing across various industries, as organizations increasingly rely on data-driven insights for decision-making, machine learning, and AI model development. However, realworld datasets often have limitations such as privacy concerns, data scarcity, or regulatory restrictions, especially in sensitive domains such as healthcare and finance. Synthetic data generation presents a viable solution by providing data that preserves the statistical patterns of the original datasets, while ensuring differential privacy and compliance with legal standards. Prior studies have demonstrated the use of synthetic data to address data limitations in areas such as healthcare research and cybersecurity training [1].

This paper introduces a novel platform for synthetic data generation that leverages fine-tuned large language models (LLMs) and differential privacy techniques to produce highquality synthetic datasets across multiple domains, including healthcare, finance, retail, logistics, and cybersecurity. The platform allows users to generate synthetic datasets from scratch or augment existing datasets while maintaining privacy through differential privacy methods. Large Language Models (LLMs) were selected for their ability to generate contextually rich and diverse synthetic data, surpassing traditional models such as GANs and VAEs in adaptability and scalability. Unlike GANs and VAEs, which require resource-intensive retraining for specific data types, LLMs offer a scalable solution that efficiently generates both structured and unstructured datasets. IBM's diffprivlib ensures that the original datasets are not exposed, making the platform a trustworthy tool for generating data for research, training, and other applications. Previous approaches often lacked versatility in handling different data types or failed to adequately protect privacy, which this platform aims to address [2].

With the growing adoption of LLMs for various tasks, finetuning has emerged as a crucial step for optimizing LLMs for specific applications, such as synthetic data generation [3]. By training the models on carefully curated datasets across different industries, our platform ensures the preservation of intricate patterns in the generated data, making it suitable for practical use without compromising privacy. Furthermore, the platform's user-friendly interface makes it accessible to a broad range of users, from researchers to individual practitioners, thereby facilitating seamless data generation and preprocessing.

In this paper, we present the architecture of our synthetic data generation framework, outline the methods used for finetuning LLMs, apply differential privacy, and discuss the potential benefits and challenges associated with using synthetic data in real-world applications. Our findings underscore the value of synthetic data as a means to address the limitations of real-world datasets while preserving privacy, thus contributing to the ongoing research in data-centric AI and privacypreserving machine learning.

The structure of this paper is organized as follows. Section II reviews related work, comparing various frameworks designed for synthetic data generation. Section III outlines the design of our proposed framework, discussing its architecture and methodologies that enable efficient, privacy-preserving data generation. Section IV presents a proof of concept prototype implementation. Section $\mathrm { V }$ discusses the evaluation results. Finally, Section VI concludes the paper and offers ideas for future work.

# II. RELATED WORK

Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Large Language Models (LLMs) are prominent methods for generating synthetic data with unique characteristics. GANs, introduced by Goodfellow et al. in 2014, create synthetic data by training two networks, a generator and a discriminator, in a zero-sum game framework where the generator attempts to produce realistic data while the discriminator differentiates between real and fake data [4]. Although GANs are powerful, they are prone to problems such as mode collapse and require substantial computational resources for training. VAEs, proposed by Kingma and Welling in 2013, approach data generation by mapping input data to a latent space and then decoding it back, generating new samples based on the learned probability distributions [5]. They can effectively generate structured data but often produce blurry outputs for complex datasets, such as images. LLMs, including models such as GPT-3, can generate text-based synthetic data by predicting the next word in a sequence based on vast amounts of pre-trained knowledge [3].

Other frameworks have been developed to generate synthetic data, each with its own strengths and limitations. ATEN specializes in generating realistic synthetic data in healthcare by incorporating statistical constraints, such as age and patient outcomes, to mirror real-world conditions [6]. This structured approach ensures that synthetic electronic healthcare records (RS-EHRs) are created while maintaining patient privacy. However, ATEN's limitations lie in its unidirectional, linear methodology and dependency on predefined parameters, making it less flexible to evolving data requirements. Additionally, ATEN’s data realism can be compromised owing to potential biases from expert feedback and substantial preparation time. Another framework, GeMSyD focuses on synthetic data generation for smart devices that capture humandevice interactions using statistical analysis and machine learning [7]. Although it excels in modeling user interactions, it may not fully capture real-world behavior nuances that lead to discrepancies in model training. Overfitting of synthetic data is also a concern, as GeMSyD's reliance on statistical methods may not generalize well to real-world applications. Its applicability is limited to specific domains, such as smart devices, which reduce its versatility.

ElderSim is another framework that generates synthetic data to enhance human action recognition (HAR) models for the elderly. Utilizing GANs and other advanced techniques, it generates diverse datasets that reflect various conditions, such as lighting and camera angles [8]. However, discrepancies between synthetic and real-world data, particularly for elderly activities, can limit their accuracy in practical applications. The framework also struggles with class imbalances and generalization issues owing to its reliance on synthetic data. Another framework is GenerativeMTD, which is based on a VAE-GAN architecture, that generates synthetic tabular data while preserving privacy by training on pseudo-real data. Its limitations include extended running times during data generation, which restricts its applicability to tabular data, and risks of mode collapse, where minority classes may not be sufficiently represented [9].

The SYNC framework uses Gaussian Copula models to create high-resolution synthetic data from aggregated datasets [10]. Its limitations arise from its reliance on Gaussian Copula models, which may not always capture complex dependencies, and the need for ongoing validation to maintain data integrity. SYNC's computational intensity can be prohibitive for large datasets, and its accuracy depends on the quality of the original data. Another framework, SynGen employs a combination of Gaussian Copula and machine learning techniques to generate synthetic tabular data [11]. Although it enables users to upload data and generate datasets with similar patterns, the generated data often shows only a moderate similarity to the original data, which may not suffice in applications demanding high fidelity, such as healthcare. Its computational requirements can also be significant, making them less accessible to smaller organizations.

SynSys is another framework that uses Hidden Markov Models (HMMs) to simulate activity-labeled smart home sensor data for healthcare [12]. HMMs models sequences of human activities and sensor events, and regression models generate timestamps to mimic real-world behavior, improving machine learning models when labeled data are scarce. Although this framework is highly credible, it often struggles with complex dependencies beyond HMM capabilities and relies on sufficient annotated data for training. Also HMMs may not generalize well to heterogeneous or non-sequential datasets.

Another framework combines generative adversarial networks (GANs) with data processing techniques to create realistic tabular data while ensuring privacy. It aims to address data sharing challenges by generating synthetic data that mirrors the statistical properties of real datasets [13]. However, GANs are computationally intensive and can suffer from mode collapse, which can cause limited sample variety. In addition, privacy-preserving methods may affect the accuracy or utility of synthetic data.

In an attempt to use LLMs to generate synthetic driving datasets, a framework was proposed that simulated diverse driving scenarios to aid autonomous driving model training [14]. This approach provides a broad range of data that reflect different driving conditions. However, LLMs may not fully capture the nuances of real driving scenarios and can introduce biases when not fine-tuned using domain-specific data.

Existing frameworks often have limitations that restrict their applicability, such as domain specificity, high computational requirements, and the requirement for expert-defined parameters. In contrast, our platform leverages pre-trained LLMs that can rapidly generate synthetic data without extensive training times, thereby reducing energy consumption. By integrating fine-tuning with real datasets, our platform ensures that LLMs understand the dependencies across different features, producing synthetic data that retain the original patterns. In addition, our use of differential privacy safeguards user data, allowing users to append the original datasets with confidence. Furthermore, the platform supports data preprocessing and analysis, offering a comprehensive solution that addresses the gaps in the existing frameworks.

# III. FRAMEWORK DESIGN

Our framework is designed to provide synthetic data generation, data pre-processing, and analysis capabilities across various domains, including Healthcare, Finance, Retail, Logistics, and Cybersecurity. The design emphasizes flexibility, efficiency, and privacy preservation. The architecture leverages OpenAI's Large Language Models (LLMs) to handle complex data generation and processing tasks, alongside IBM's diffprivlib library to ensure differential privacy for sensitive data [15][16]. The system’s architecture integrates user input, data processing, and output generation, ensuring that data flows seamlessly through each module and allowing users to handle a wide range of data types and formats with minimal effort.

# A. Architecture and Data Flow

The platform architecture is structured to support various functionalities with a focus on scalability and modularity. Data flows from user input through the differential privacy module (if user data is provided) and then to the synthetic data generation module powered by LLMs. The system’s design allows iterative querying of the LLM, using a loop mechanism that continuously requests additional records if the initially generated data does not meet the desired volume. This iterative approach guarantees that the final synthetic dataset aligns with the user’s data volume requirements and preserves the patterns of the original data. Below Fig. 1 shows the simplified working of the framework architecture.

![](images/c2fe0b34eca0587930dfe49bc0727f18c6d6059b5ca2ec998a36a17cc9423956.jpg)  
Fig. 1. Framework architecture.

1) Differential Privacy Integration: The IBM diffprivlib library is integrated into the design, specifically for handling numerical datasets. When users opt to append their data with synthetic data, differential privacy ensures that the original data is anonymized. This is crucial for maintaining user privacy and preventing any data leakage, especially for sensitive fields.

2) Modular Processing for Data Pre-processing and Analysis: The design includes dedicated modules for data preprocessing and analysis. Each module is capable of batch processing, which not only supports large datasets but also reduces memory and computational requirements. This modularity also ensures that any future extensions, such as predictive modeling or advanced trend analysis, can be added without significant changes to the core architecture.

# B. Synthetic Data Generation

Synthetic data generation is a core component of the platform, designed to meet user-specific requirements using two approaches.

1) User-provided Data with Differential Privacy: When a user provides a dataset, differential privacy is applied to protect individual record privacy. The platform leverages the LLM to analyze and learn from this differentially private data, ensuring that synthetic data generation preserves patterns without risking privacy. If the generated dataset does not meet the required data size, an iterative loop requests more data from the LLM until the desired size is achieved. The fine-tuning process employs a combination of prompt engineering and iterative learning, allowing the LLM to capture intricate data dependencies while adapting to privacy constraints.

2) Pre-defined Categories with User Description: For users without their own data, the platform offers five predefined categories. Users choose a category and input a brief description. Using fine-tuning through prompt engineering, the LLM generates synthetic datasets relevant to the chosen category. Fine-tuning with real-world datasets ensures that the LLM captures dependencies between features, producing realistic data tailored to specific domains.

# IV. PROTOTYPE IMPLEMENTATION

The prototype was implemented in Python and hosted on GitHub Codespaces, thereby enabling collaborative development and efficient testing. The platform integrates three primary functionalities—synthetic data generation, data preprocessing, and analysis—each supported by specific technologies to ensure robust functionality and ease of use.

# A. Development Environment and Tools

Python was chosen as the primary programming language due to its extensive libraries and community support. For LLMbased synthetic data generation, we utilized OpenAI's API, while IBM’s diffprivlib provided differential privacy functionality. GitHub Collaboratories facilitated real-time collaboration and version control, critical for iterative development and testing. The user interface was designed to prioritize accessibility, allowing users to upload datasets, configure privacy parameters, and generate synthetic data through a streamlined workflow. The platform includes guided tooltips, preset configurations, and detailed documentation that ensures the ease of use while accommodating advanced customizations for technical users.

# B. Detailed Component Implementation

1) Synthetic Data Generation: Users upload data through the web interface, where it undergoes differential privacy processing using IBM diffprivlib. A loop function enables iterative requests to the LLM, ensuring that the final dataset meets the specified record count. Looping function is explained in detail in Fig. 2 below. While working with pre-defined categories, each category is tailored with prompts that guide the LLM in generating domain-relevant synthetic data. Fine-tuned prompts improve the LLM’s contextual understanding, enhancing data quality and relevance.

2) Data Pre-processing: To handle large datasets, batch processing was implemented to send data in chunks to the LLM. This approach manages memory efficiently and ensures seamless processing of extensive data. The LLM identifies and treats outliers and missing values, utilizing statistical techniques like mean/median imputation. To improve data balance, synthetic samples are generated for minority classes, if applicable, ensuring a more balanced dataset.

3) Data Analysis: The platform provides basic analysis, including descriptive statistics and visualization recommendations. The LLM can generate Python code for histograms, scatter plots, and other visualizations, offering users insights into their datasets. The LLM generates prompts for DALL-E or other image-generation models to visualize dataset characteristics, offering a comprehensive approach to data analysis. However, this technique isn’t completely reliable for image generation because LLMs can’t understand complex data and can’t generate sophisticated images to represent it.

![](images/17033ae4cf2931d3a4d1eebfc6498563699d310d9f41d7172e2d40632e7702f8.jpg)  
Fig. 2. Synthetic data generation process.

The prototype leverages LLMs and differential privacy to ensure data quality and privacy while reducing computational overhead compared to approaches like GANs or VAEs.

# V. EVALUATION RESULTS

The prototype was used to evaluate the quality of the generated synthetic data under different privacy configurations. The Iris dataset, which consists of 150 rows and 5 columns, was used as a benchmark for this analysis [17]. The Iris dataset is known to contain measurements of iris flowers, making it suitable for testing how well synthetic data generation preserves the statistical properties of the original data.

# A. Differential Privacy and Synthetic Data Generation

The first step involved applying differential privacy to the original dataset. By setting the differential privacy parameter to $\varepsilon = 1 0$ , we achieved a balance in which privacy was relaxed in favor of higher data utility, allowing the synthetic data to more closely match the original patterns. A higher value of ε implies that the noise added to the data was relatively low, leading to a higher similarity between the real and synthetic datasets.

The results show that the synthetic data captured the key statistical patterns present in the original data. The distributions of both the real and synthetic datasets across different features reveal that the synthetic data closely followed the distributions of the original data when ε was set to 10.0.

![](images/9742c6eb0bfbb0414885170d041beb68da931758d0c20b0578eba2590e33306c.jpg)  
Fig. 3. Comparison of statistical properties of real and synthetic data $( \varepsilon = 1 0 . 0 )$ ).

Fig. 3 indicates that LLM-based synthetic data generation effectively retained the underlying patterns of the real dataset under a relatively low privacy setting. Fig. 4 shows four graphs that compare the distributions of the features in the original dataset and the generated synthetic dataset can be placed here. These graphs demonstrate how well the synthetic data preserves the original data patterns.

![](images/4e9600965af661a81ebe9160874b35aa11f04a8f92ec72d4bb432163fa4131f8.jpg)  
Fig. 4. Comparison of frequency distribution between real and synthetic data among all four features of the dataset $\mathbf { \varepsilon } _ { \mathbf { \varepsilon } } ( \mathbf { \varepsilon } _ { \mathbf { \varepsilon } } = 1 0 . 0 )$ ).

# B. Impact of Varying Privacy Settings $( \varepsilon = I . 0 ,$

To understand the effect of different privacy levels on data quality, synthetic data generation was repeated with a lower privacy parameter, $\varepsilon = 1 . 0$ . In this case, a higher degree of noise is introduced into the original dataset, resulting in a greater deviation between the real and synthetic data distributions. As shown in Fig. 5, some generated synthetic data values fell outside the expected range, such as negative lengths, which were not plausible in the context of the Iris dataset.

![](images/b1863ca1d7e9aca50202296de6f4e9b925524964681d6731d7331c1b3a740419.jpg)

The presence of these implausible values occurred because the original data had values close to zero, and the added noise was more substantial with the lower ε setting. This experiment demonstrated that when working with data that contains values near zero, it is advisable to use relatively higher ε values to avoid producing unrealistic synthetic data.

These results indicate that while an accuracy drop is observed at lower epsilon values, it remains within an acceptable range for privacy-sensitive applications, such as in healthcare or finance. This balance highlights the practical usability of the generated data in real-world scenarios, where data utility and privacy must coexist.

# C. Machine Learning Usability Evaluation

To assess the usability of the synthetic data, a machine learning model was trained separately on real and synthetic datasets. The performance of the model trained on the synthetic data was then evaluated on a test set drawn from the original dataset, and the results were compared with those of the model trained on real data.

The findings showed only a slight decrease in model performance when trained on synthetic data compared with real data, indicating that the synthetic data retained sufficient quality and utility for practical machine learning tasks. This further validated that with an appropriate ε value, the platform can generate synthetic data that maintain predictive accuracy similar to real-world datasets.

However, with a decreased value of ε, a noticeable decrease in the model accuracy was observed. The results confirmed that stricter privacy settings can affect data utility, leading to a trade-off between data privacy and quality. Table I. shows the difference between the accuracy and validation loss of real and synthetic data when used for training machine learning models.

Fig. 5. Comparison of statistical properties of real and synthetic data $( \varepsilon = 1 . 0$ ).   
TABLE I. REAL VS. SYNTHETIC DATA FOR TRAINING   

<table><tr><td rowspan="2">Matrix</td><td>Real Data</td><td colspan="4">Synthetic Data</td></tr><tr><td></td><td>ε=10.0</td><td>8=5.0</td><td>8=2.5</td><td>8=1.0</td></tr><tr><td>Accuracy</td><td>95.5%</td><td>91.1%</td><td>83.6%</td><td>82.5%</td><td>77.8%</td></tr><tr><td>Validation Loss</td><td>10.8%</td><td>20.0%</td><td>27.3%</td><td>34.7%</td><td>42.2%</td></tr></table>

# $D$ . Choosing the Right Value of ε

Selecting an appropriate value for the privacy parameter ε is crucial to achieving a balance between data utility and privacy. As ε decreases, the accuracy of the synthetic data in capturing real-world patterns also diminishes, as the added noise begins to obscure subtle data dependencies. Lower ε values enhance privacy by making it more difficult to trace synthetic data back to original entries, yet this comes at the cost of reduced accuracy and increased validation loss. Conversely, higher values of ε yield synthetic data that retains more of the original dataset's patterns, making it more accurate for realworld applications but with less stringent privacy guarantees.

For datasets with values close to zero or with lower variance, higher ε values (such as 5.0 or above) are recommended to maintain utility, as smaller values of ε can disproportionately affect low or narrow-range data. However, when working with datasets where standard deviation is less critical or with larger-scale values, a lower ε value may be effective, providing enhanced privacy without significantly compromising usability. This adaptability in ε selection allows users to tailor privacy settings to the unique attributes of their dataset, ensuring an optimal trade-off between data fidelity and privacy protection.

# $E .$ . Comparison of Synthetic Data Generated from FineTuned Model vs. General LLMs

To compare the proposed framework with other popular LLMs, similar type of tabular data was generated using each LLM. The comparison results in Fig. 6 clearly demonstrate that the synthetic data generated by our platform significantly outperforms the data generated by general-purpose language models such as ChatGPT and Gemini. This achievement can be attributed to the fine-tuning of our large language model (LLM), which was specifically adapted to learn the underlying data distributions and preserve the statistical properties of realworld datasets.

![](images/9075245aefeea5940c98b94283509fdcecc2f7d8daebd702da94e789d5991f3e.jpg)  
Fig. 6. Comparison of the quality of synthetic data generated by the fine-tuned model vs. ChatGPT and Gemini.

This specialized approach ensures that the generated data is not only statistically consistent with the original but also maintains essential characteristics that are critical for data analysis and downstream machine learning tasks. The consistent outperformance of our platform across all evaluation metrics emphasizes its suitability for real-world applications where data quality and representativeness are paramount.

# $F _ { \ l }$ . Performance of Synthetic Data Generation Platform

The evaluation of the synthetic data generation platform focuses on measuring the quality of the generated synthetic datasets and the system's performance in terms of data generation time. For the performance evaluation, the time taken to generate synthetic datasets of various sizes was plotted to understand the scalability of the platform. Fig. 7 shows an analysis of the performance of the platform. The platform was tested 3 times to obtain 5000 records of data.

The time required for synthetic data generation demonstrates a consistent trend across different dry runs, showing the scalability and efficiency of the platform. There is a general proportionality between the number of rows and columns requested, which affects the processing time due to the increased computational complexity associated with more features. The time in seconds is roughly double the number of rows generated, but is affected by the number of columns and other features. The results also show a moderate level of variability, with time differences ranging between $10 \%$ to $20 \%$ across multiple trials. This suggests that the model operates reliably and maintains consistency despite slight fluctuations.

![](images/a3fb7ed12bf6a86d34d2892be0d308eea66fd3ab20974c2f5ac48d5c750adfcc.jpg)  
Fig. 7. Performance of the synthetic data generation platform showing the time taken to generate synthetic data for different row counts.

The platform offers a robust pre-processing capability that accommodates both numerical and string data types, allowing for a comprehensive data cleaning process. The pre-processing pipeline ensures that outliers are identified and handled effectively, and missing values are managed using techniques such as mean/median imputation for numerical features and mode imputation or custom handling for string values. In addition, the platform supports batch processing, enabling the handling of large datasets by dividing them into manageable chunks. This allows for pre-processing of datasets of virtually any size without compromising performance or data quality.

Additionally, the platform was benchmarked against GANbased and copula-based synthetic data generation models. Results demonstrate that GANs require approximately $60 \%$ more computational resources for generating smaller datasets $\mathord { \sim } 1 0 0 0$ rows) and $40 \%$ more for bigger datasets ${ \sim } 4 0 0 0$ rows). While the copula-based models are relatively quick, they exhibited reduced versatility with unstructured data. However, the LLM-based approach shows balanced performance and efficiency. This makes it well-suited for diverse applications without compromising computational feasibility.

# VI. CONCLUSION AND FUTURE WORK

This paper presents a synthetic data generation platform designed to address the limitations of existing data generation tools by leveraging fine-tuned large language models (LLMs) and differential privacy techniques. This platform effectively generates high-quality synthetic datasets across various domains, including healthcare, finance, logistics, cybersecurity, and retail, while preserving data privacy. Through fine-tuning, LLM was able to better understand data dependencies, leading to more accurate synthetic data that closely resemble real-world datasets. The experimental results demonstrate the proposed platform outperformed general-purpose models such as ChatGPT and Gemini in generating synthetic data, particularly in preserving statistical properties and variability. The platform achieved very good results in terms of accuracy and variability retention due to its fine-tuning approach and differential privacy integration.

Future work includes integrating data analysis functionality to enhance the platform’s utility and provide users with valuable insights, visualizations, and code suggestions to understand their datasets

# REFERENCES

[1] M. Goyal and Q. H. Mahmoud, “A Systematic Review of Synthetic Data Generation Techniques Using Generative AI,” Electronics, vol. 13, no. 17, p. 3509, Sep. 2024, doi: https://doi.org/10.3390/electronics13173509.   
[2] N. Patki, R. Wedge and K. Veeramachaneni, "The Synthetic Data Vault," 2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA), Montreal, QC, Canada, 2016, pp. 399-410, doi: 10.1109/DSAA.2016.49.   
[3] T. B. Brown et al., “Language Models Are Few-Shot Learners,” arxiv.org, vol. 4, May 2020, Available: https://arxiv.org/abs/2005.14165   
[4] I. J. Goodfellow et al., “Generative Adversarial Networks,” arXiv.org, Jun. 10, 2014. https://arxiv.org/abs/1406.2661   
[5] D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” arXiv.org, Dec. 20, 2013. https://arxiv.org/abs/1312.6114   
[6] S. McLachlan, K. Dube, T. Gallagher, J. A. Simmonds, and N. Fenton, “Realistic Synthetic Data Generation: The ATEN Framework,” Communications in computer and information science, pp. 497–523, Jan. 2019, doi: https://doi.org/10.1007/978-3-030-29196-9_25.   
[7] R. Tolas, R. Portase, and R. Potolea, “GeMSyD: Generic Framework for Synthetic Data Generation,” Data, vol. 9, no. 1, p. 14, Jan. 2024, doi: https://doi.org/10.3390/data9010014.   
[8] H. Hwang, C. Jang, G. Park, J. Cho, and I.-J. Kim, “ElderSim: A Synthetic Data Generation Platform for Human Action Recognition in Eldercare Applications,” arXiv.org, 2020. https://arxiv.org/abs/2010.14742 (accessed Oct. 16, 2024).   
[9] J. Sivakumar, K. Ramamurthy, M. Radhakrishnan, and D. Won, “GenerativeMTD: A deep synthetic data generation framework for small datasets,” Knowledge Based Systems, pp. 110956–110956, Sep. 2023, doi: https://doi.org/10.1016/j.knosys.2023.110956.   
[10] Z. Li, Y. Zhao, and J. Fu, “SYNC: A Copula based Framework for Generating Synthetic Data from Aggregated Sources,” arXiv.org, 2020. https://arxiv.org/abs/2009.09471 (accessed Oct. 16, 2024).   
[11] A. Kothare, S. Chaube, Y. Moharir, G. Bajodia and S. Dongre, "SynGen: Synthetic Data Generation," 2021 International Conference on Computational Intelligence and Computing Applications (ICCICA), Nagpur, India, 2021, pp. 1-4, doi: 10.1109/ICCICA52458.2021.9697232.   
[12] J. Dahmen and D. Cook, “SynSys: A Synthetic Data Generation System for Healthcare Applications,” Sensors (Basel, Switzerland), vol. 19, no. 5, Mar. 2019, doi: https://doi.org/10.3390/s19051181.   
[13] L. Hansen, N. Seedat, van, and A. Petrovic, “Reimagining Synthetic Tabular Data Generation through Data-Centric AI: A Comprehensive Benchmark,” arXiv.org, 2023. https://arxiv.org/abs/2310.16981 (accessed Oct. 16, 2024).   
[14] J. Guo, C. Chang, Z. Li and L. Li, "Mixing Left and Right-Hand Driving Data in a Hierarchical Framework With LLM Generation," in IEEE Robotics and Automation Letters, vol. 9, no. 10, pp. 8290-8297, Oct. 2024, doi: 10.1109/LRA.2024.3443494.   
[15] OpenAI, “OpenAI API,” platform.openai.com, 2024. https://platform.openai.com/docs/models   
[16] “Diffprivlib v0.5,” GitHub, Dec. 06, 2021. https://github.com/IBM/differential-privacy-library   
[17] R. Fisher. "Iris," UCI Machine Learning Repository, 1936. [Online]. Available: https://doi.org/10.24432/C56C76.