# Background

> Pre-trained word representations are a key component in many natural language understanding models. However, learning high quality representations can be challenging. They should ideally model both:
>
> 1. complex characteristics of word use (e.g., syntax and semantics)
> 2. how these uses vary across linguistic contexts (i.e., to model polysemy).



# Motivation

> + Our representations differ from traditional word type embeddings in that each token is assigned a representation that is a function of the entire input sentences.
> + We use vector derived from a bidirectional LSTM that is trained with a **coupled** language model(LM) objective on a large text corpora. For this reason, we call them **ELMo (Embeddings from Language Models)** representations.
> + Unlike previous approaches for learning contextualized word vectors, ELMo representations are deep, in the sens that they are a function of all of the internal layers of the biLM. More specifically, we learn a linear combination of the vectors stacked above each input word for each end task, which markedly improves performance over just using the top LSTM layer.



In brief:

+ The word vector is derived from a language model (add context).
+ The word vector is not just the top layer, but a weighted sum of all the previous layer (different levels of semantics).
+ The language model is deep...



**Note:**

This paper adopts a biLSTM which is a **coupled** LM. The forward LSTM can not get any information from the backward LSTM. If not, the forward LSTM will know the words to predict...



# Model

ELMo provides a new way to represent word vectors. It consists of two major processes: pre-train ELMo and fine-tuning.  In the pre-training process, ELMo is trained according to an objective function, and then the parameters of the model are saved. During fine-tuning, we feed word embeddings to the saved model and get the output of each layer, then a combination of different layers is adopted to get an ELMo vector for input words. 

Two functions:

+ Learn complex uses of words: syntactics, semantics, etc.
+ Disambiguate the meaning of words using their context.



## Model Pre-training

Like other language models, the authors pre-train ELMo using a deep biLSTM model. The objective is as follows:


$$
L=\sum_{k=1}^N (\log p(t_k|t_1, ..., t_{k-1}; \Theta_x, \overrightarrow{\Theta}_{LSTM}, \Theta_s)) + \log p(t_k|t_{k+1}, ..., t_N; \Theta_x, \overleftarrow{\Theta}_LSTM, \Theta_s)
$$


Two parts represent the loss of forward LSTM and backward LSTM respectively. 

$\Theta_x$: token representation; $\Theta_s$: softmax layer.



For each token $t_k$, a $L$-layer biLM computes a set of $2L+1$ representations:


$$
R_k=\{x_k^{LM}, \overrightarrow{h}_{k,j}^{LM}, \overleftarrow{h}_{k,j}^{LM}|j=1,...,L\}=\{h_{k,j}^{LM}|j=0,...,L\}
$$


$x_k^{LM}$: word vector;  $\overrightarrow{h}_{k,j}^{LM}$ : forward LSTM hidden state; $\overleftarrow{h}_{k,j}^{LM}$: backward LSTM hidden state;



## ELMo Fine-tuning

After pre-training, we can get the ELMo vector as follows:


$$
ELMo_k^{task}=E(R_k;\Theta^{task})=\gamma^{task} \sum_{j=0}^L s_j^{task} h_{k,j}^{LM}
$$


$h_{k,j}^{LM}$: hidden states of layer $j$;  $s_j^{task}$: softmax normalized weights (model parameters, a $L$-dimension vector); $\gamma^{task}$: scalar parameter (hyper-parameter). The sentence "it can also help to apply layer normalization to each biLM layer before weighting". Err... In my opinion, the sentence can be rewritten to "Applying layer normalization to each biLM layer before weighting is helpful". Emmm... Layer normalization is helpful. First, adopt layer normalization on $h_{k,j}^{LM}$ , then weighting $s_j^{task} h_{k,j}^{LM}$ . Emmm... The authors also mention that "Considering that the activations of each biLM layer have a different distribution", so they should express that layer normalization is helpful to obtain the same distribution.



During the fine-tuning process, we can get the EMLo vector for according to the above equation (fix ELMo model),  then we can append it to our model:

+ Add ELMo vector to the word vector, $[x_k; ELMo_k^{task}]$;
+ Add ELMo to the output vector to make prediction, $[h_k;ELMo_k^{task}]$.



Other details ,like regularization, residual network, etc, are not so important as understanding how to train ELMo and add ELMo vectors. Please refer to the original paper.



