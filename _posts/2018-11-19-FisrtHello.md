# Inspiration
1. 可以使用Transfer Learning把别人训练好的模型拿过来直接使用，然后结合自己的模型再去进行别的工作
2. 如果能够有详细的数据分析则一定要做来验证自己的Idea

# Motivation
现有的方法都是使用问题和文章全部做Co-Attention之类的操作，但是在训练的时候会面临着效率低的问题，毕竟并不是文章的所有部分都与问题是相关的，所以这里文章提出来先从文章中挑选出来一部分相关的语句，然后再去做QA任务。
> Neural models for question answering (QA) over documents have achieved significant performance improvements. Although effective, these models do not scale to large corpora due to their complex modeling of interactions between the document and the question. Moreover, recent work has shown that such models are sensitive to adversarial inputs. In this paper, we study the minimal context required to answer the question, and find that most questions in existing datasets can be answered with a small set of  sentences. Inspired by this observation, we propose a simple sentence selector to select the minimal set of sentences to feed into the QA model.
# Model
根据Motivation，文章先分析了Motivation的正确性，然后模型Sentence Selector + QA Model来完成任务。
## Corpus Analysis

> 表1说明了90%的问题只与一个句子相关，6%的与两个句子相关，2%的与3个以上句子相关。这样子来看，基本上是可以大概率断定假设是正确的。（PS：这儿的显然是人工判断的，那么肯定是一个小样本抽取，作者是随机选了50个例子来判断的，所以后面如果数据集过大自己也要做分析的话也可以引用这篇文章，并且采用类似的方法去做。）

# Model


## Sentence Selector
目的是为了给每个句子打分，最后归一化以后应该是一个0-1之间的分数。
#### Sentence Encoding
使用S-Reader来做Transfer Learning，来对问题和文章进行Embedding，得到$D \in \mathbb{R}^{h_d \times L_d}$以及$Q \in \mathbb{R}^{h_d \times L_q}$，其中$D$代表文章的Embedding， $Q$代表问题的Embedding，$h_d$是hidden size， $L_d$和$L_q$是对应的句子长度。
计算question-aware sentence embedding：简单的Attention而已。这儿计算的应该是一个句子的embedding。
$$\alpha_i = {\rm softmax} (D_i^T W_1 Q) \in \mathbb{R}^{L_q}$$
$$D_i^q=\sum_i^{L_q} (\alpha_{i,j} Q_j) \in \mathbb{R}^{h_d}$$
随后计算Question Encoding & Sentence Encoding
$$D^{enc}=BiLSTM([D_i;D_i^q]) \in \mathbb{R}^{h \times L_d}$$
$$Q^{enc}=BiLSTM(Q_j) \in \mathbb{R}^{h \times L_q}$$
随后计算$D^{enc}$与$Q^{enc}$之间的分数呗，这儿设计的还挺复杂的：其中的$max$操作应该是按照维度来的。后面自己设计Fusion的时候也可以参考一下别人的设计，自己想确实比较简单了。
$$\beta={\rm softmax}(w^T Q^{enc}) \in \mathbb{R}^{L_q}$$
$$q^{\widetilde{enc}}=\sum_{j=1}^{L_q} (\beta_j Q_j^{enc}) \in \mathbb{R}^h$$
$$\tilde{h}_i=(D_i^{enc}W_2 q^{\widetilde{enc}})\in \mathbb{R}^h$$
$$\tilde{h}=max(\tilde{h}_1, \tilde{h}_2, ... , \tilde{h}_{L_d})$$
$$score=W_3^T \tilde{h} \in \mathbb{R}^2$$
score 代表了问题是否可以被某一个句子回答。
后面，作者对每一个段落中的句子得分进行归一化，并且最后按照设置阈值的方式来选择句子，$Top K$。这里作者也说和别人不一样，别人是取前$K$，而它是设置阈值。

## QA System
这里作者在模型图中解释了，用的就是S-Reader中的组件，所以作者极力降低了这部分的存在感，后面干脆就不写了，只是在图片里面用文字说明了一下。好吧，我也不写了。
# Experiments
这儿的实验感觉不是那么重要。后面具体设置实验的时候再去参考实验设计。大致一想也知道作者会从和别的模型比较，设置阈值选$Top K$，验证自己选择的句子多少以及效果等方面去解释。
