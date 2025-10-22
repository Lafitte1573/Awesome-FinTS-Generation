# Data Augmentation Using BERT-Based Models for Aspect-Based Sentiment Analysis

本文提出了一种基于BERT的文本数据增强方法，用于提升面向方面的情感分析（Aspect-Based Sentiment Analysis, ABSA）模型HAABSA++的性能。具体方法如下：

1. **模型框架：HAABSA++**  
   HAABSA++是一种混合方法，由两个阶段组成：  
   - **第一阶段：基于领域情感本体的规则分类**  
     使用预定义的领域情感本体（domain sentiment ontology）对句子进行情感分类。该本体基于词汇规则和语义关系，仅能识别“正面”和“负面”情感，无法处理“中性”情感，也无法处理情感冲突（同一目标同时被判定为正面和负面）或本体未覆盖的词汇（无匹配）。  
   - **第二阶段：LCR-Rot-hop++神经网络模型（作为后备）**  
     当本体分类结果不确定（如中性、冲突或无匹配）时，调用LCR-Rot-hop++模型进行情感预测。该模型是LCR-Rot模型的改进版本，其核心创新包括：  
     - 重复旋转注意力机制（rotatory attention），以更准确地加权与目标方面相关的词；  
     - 使用上下文相关的BERT词嵌入替代传统的GloVe嵌入；  
     - 引入分层注意力结构，增强模型对上下文的建模能力。

2. **数据增强方法**  
   为解决训练数据稀缺问题，本文比较了五种数据增强技术，均作用于HAABSA++的训练数据，以提升LCR-Rot-hop++模块的泛化能力：

   - **EDA-adjusted（基线）**  
     基于Easy Data Augmentation（EDA）的改进版本，包含三种操作：  
     1. 同义词替换（Synonym Replacement）：使用WordNet和Lesk算法进行词义消歧，选择语境最匹配的同义词；  
     2. 随机插入（Random Insertion）：插入语境相关的同义词；  
     3. 目标词跨句交换（Target Swap）：在相同类别（如“服务”、“食物”）的句子间交换目标方面词，以生成多样化的上下文。  
     该方法为基于词典和语言规则的非神经网络方法，不依赖预训练模型。

   - **BERT**  
     利用BERT的掩码语言建模（MLM）任务进行数据增强：  
     - 对训练句子中的每个词以15%的概率随机掩码；  
     - 使用BERT预测被掩码词的候选词，选择概率最高的非原词作为替换；  
     - 生成的新句子保留原句语义，但词汇表达多样化。  
     此方法未考虑原句的情感标签，可能导致情感信息丢失。

   - **Conditional-BERT (C-BERT)**  
     改进BERT，使增强过程依赖于句子的情感标签：  
     - 将BERT原有的segment embeddings替换为label embeddings（情感标签嵌入）；  
     - 在带标签的语料上对BERT进行微调，使模型在预测掩码词时考虑情感类别；  
     - 增强后仍移除标签，仅保留增强后的句子。  
     此方法旨在保留情感一致性，但因牺牲了句子结构信息（标签嵌入取代了句子段嵌入），可能破坏上下文语义。

   - **BERTprepend**  
     在不修改BERT内部结构的前提下，通过输入格式调整实现标签感知增强：  
     - 将情感标签（如“positive”）作为文本前缀直接拼接到原始句子前，例如：“positive [SEP] The food was amazing.”；  
     - 标签不加入BERT的词汇表，仅作为普通文本序列的一部分；  
     - 在MLM过程中，标签不被掩码，确保其始终保留；  
     - 增强完成后，移除前缀标签，仅保留增强后的原始句子用于训练。  
     此方法在不改变BERT模型结构的情况下，使模型在预测时隐式感知情感类别。

   - **BERTexpand**  
     与BERTprepend类似，但关键区别在于：  
     - 将情感标签（如“positive”）作为独立token加入BERT的词汇表；  
     - 在输入序列中，标签仍作为前缀拼接，但被BERT的WordPiece分词器视为单一token（而非被拆分）；  
     - 在MLM过程中，标签同样不被掩码，确保其稳定性。  
     由于BERT使用WordPiece分词器，且“positive”、“negative”、“neutral”均为完整词，不会被切分，因此BERTexpand与BERTprepend在实际操作中产生完全相同的增强效果。

3. **模型微调与训练设置**  
   - 所有BERT相关模型（BERT、C-BERT、BERTprepend、BERTexpand）均在完整SemEval 2015/2016训练集上进行MLM任务的微调；  
   - 使用80%数据训练，20%数据验证超参数；  
   - 微调10个epoch，采用默认掩码参数（15%掩码率）；  
   - BERTprepend与BERTexpand在输入序列中添加情感标签，但最终增强后的数据移除标签，仅保留句子内容用于HAABSA++训练。

4. **核心创新点**  
   - 首次在HAABSA++框架下系统比较了从传统词典增强（EDA-adjusted）到多种BERT增强方法的性能；  
   - 提出并验证了BERTprepend与BERTexpand两种轻量级标签感知增强策略，无需修改模型结构，仅通过输入格式调整实现情感一致性增强；  
   - 通过实证发现，在不同规模数据集上，不同增强方法表现不同：小数据集（SemEval 2015）上EDA-adjusted最优，大数据集（SemEval 2016）上BERTprepend/BERTexpand最优；  
   - 首次揭示BERTprepend与BERTexpand在WordPiece分词器下因标签不被切分而产生完全一致增强结果的机制。

综上，本文方法的核心是：通过引入标签感知的BERT数据增强技术（尤其是BERTprepend与BERTexpand），在不改变HAABSA++模型结构的前提下，显著提升其在大规模ABSA数据集上的泛化能力。