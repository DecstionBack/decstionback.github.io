# Background and Motivation

### Background

> + RNNs have been firmly established as state of the art approaches in sequence modeling and transduction problem such as language modeling and machine translation.
>
> + RNN precludes parallelization within training examples.
> + Attention Mechanisms allow modeling of dependencies without regard to distances in the input or output sequences. However, attention mechanisms are used in conjunction with a recurrent network.



### Motivation

> + Propose the **Transformer**, a model architecture eschewing recurrence and instead relying on attention mechanism to draw global dependencies between input and output.
>
> + **Transformer** allows for significantly more parallelization and can reach a new state of the art in translation quality.



# Model Architecture

The **Transformer** follows an encoder-decoder architecture, which encodes an input sequence $x=(x_1, x_2, ..., x_n)$ to a sequence of continuous representation $z=(z_1, z_2, ..., z_n)$  and generates an output sequence $(y_1, y_2, ..., y_m)$  of symbols one element at a time.

![transformer](../img/transformer/transformer.png)



### Encoder

+ Composed of N=6 identical layers
+ Each layer has two sub-layers: Multi-Head Self Attention Mechanism and Point-wise fully connected feed-forward network
+ Employ a residual connection between two sub-layers, the output of each sub-layer is LayerNorm(x+Sublayer(x))



### Decoder

+ Composed of N=6 identical layers
+ in addition to the two sub-layers in encoder, a third sub-layer which performs multi-head attention over **the output of the encoder stack** is added.
+ Employ a residual connection between two sub-layers, the output of each sub-layer is LayerNorm(x+Sublayer(x))
+ Modify the self-attention sub-layer to prevent positions from attending to subsequent positions by masks. This can ensure that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.



In the encoder, sequences are consumed to get a feature representation. So the encoder can be used to obtain sequence vectors for classification tasks. While in the decoder, it consumes sequences before position $i$ to predict the word  in position $i$, so the decoder can be used to generate new sequences. During training time, if no masks are adopted, self attention mechanism will see the words of the whole sequence and there is no need to predict since we have known the whole sequence.



There are many common components in encoder and decoder: Multi-Head Attention, Position-wise fully connected network, etc. I will discuss about these first then dive into other components in the decoder.



### Attention

> An attention function can be described as mapping a query and a set of key-valued pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

Intuition behind this attention formulation:

Given a set of vectors $V$, I'd like to get a weighted sum of $V_i$ according to my query $Q$,  namely, $\sum_i^n a_i V_i$ , constraint to $\sum_i^n a_i=1$, but how to allocate the weights $a_1, a_2, ..., a_n$ properly.  Each $V_i$ has $K_i$ , which is used to calculate the relation between $Q$ and $V$.  We can get the weights $a$ based on $Q$ and $V$ then get the weighted sum of $V$. In the encoder-decoder architecture, $V=K$, while in self attention mechanism, $Q=K=V$.



In this paper, the authors proposed **scaled dot-product attention**, formulated as follows:
$$
Attention(Q,K,V)={\rm softmax} (\frac{QK^T}{\sqrt{d_k}})V
$$
In this formulation, $\sqrt{d_k}$ is the scaling factor. From the dot expression $QK^T=\sum_i^d Q_i K^T_i$ , we can see that if $d$ is large, the sum value can be very large and it will push the softmax function to regions where it has very small gradients (The authors use the word **suspect**). 

### Multi-Head Attention

> Instead of performing a single attention function with d_{model}-dimension keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to d_k, d_k and d_v dimensions, respectively.

I have read a paper "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)" which also adopts similar ideas. They both introduce multiple linear projections to the original embeddings.

In this paper, assuming the input dimension is [B,L,D] (B is batch size, L is sequence length and D is word vector dimension), for each linear projection D-->d  we can get the output [B,L,d]. Performing this linear projection h times we can get h *[B,L,d]. d can be d_k or d_v, K and V are the same and they have the same dimension.

This paper then concatenates the output and once again projects to result in the final values as follows:
$$
MultiHead(Q,K,V)=Concat(head_1, head_2, ... heawd_h) W^O \\
where head_i=Attention(QW_i^Q, KW_i^K, VW_i^V)
$$
**Dimension**:

$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}, W_i^K \in \mathbb{R}^{d_{model} \times d_k}, W_i^V \in \mathbb{R}^{d_{model} \times d_v}$  and  $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

$d_{model} $ is the dimension of MultiHead(Q,K,V) and $d_{model}=h*d_v$.

Project Q,K,V respectively and concatenate the results together, Emmm...Nice! 

QW_i^Q projects Q to d_k, KW_i^k projects K to d_v, VW_i^K also projects V to d_v, then the result dimension should be d_v for each head. h heads should have h\times d_v dimension.

Here, d_k and d_v can have different dimensions.



This paper adopts h=8 heads, $d_{model}=512, so d_k=d_v=d_{model}/h=64$.



### Position-wise Feed-Forward Networks

In the last sub-layer of encoder and decoder, a feed-forward network is applied to perfrom linear transformation as follows:
$$
FFN(x)={\rm max(0,xW_1+b）W_2+b_2}
$$


The input $x$ is the output of Multi-Head Attention with dimension $d_{model}$.  The output dimension is also $d_{model}$. The inner-layer has dimension $d_{ff}=2048$ (Other dimensions are OK).

The authors also put the feed-forward networks in another way:

> Another way of describing this is as two convolutions with kernel size 1.

This is the classical problem. How to implement fully-connected network with filters whose kernel size is 1?



See also: [【机器学习】关于CNN中1×1卷积核和Network in Network的理解](https://blog.csdn.net/haolexiao/article/details/77073258)



Assuming the input dimension is $d_{model}=512$ and the output dimension of the second feed-forward layer is also $d_{model}$. First, we can consider the input as a $1*512$ tensor, then we can adopt $d_{ff}$ number of $1$ filters with depth $512$. Each filter can get a $1$-dimension result. By performing this $d_{ff}$ times, we can get a $1*d_{ff}$ tensor.  Then we can adopt $d_{model}$ of $1$ fitlers with depth $d_{ff}$. After performing this $d_{model}$ times, we can get a $1*d_{model}$ tensor. So the feed-forward network can also be implemented by two convolutions with kernel size $1$. 



### Embedding and Softmax

> Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d-{model}$. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformations, similar to... In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$.

**The Question is "Why do they multiply those weights by $\sqrt{d_{model}}$  ??"**

### Position Encoding

> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject somo information about the relative or absolute position of the tokens in the sequence.

The authors adopt relative position to model token sequences as follow:
$$
PE(pos,2i)=sin(pos/10000^{2i/d_{model}}) \\
PE(pos,2i+1)=cos(pos/10000^{2i/d_{model}})
$$
where $pos$ is the position and $i$ is the dimension. For each fixed dimension $i$, we can get a sinusoid function. We know that for a sin function: $f(t)=Asin(wt)$, the wavelength(or period $T$) equals $2\pi /w$. For the position function,we can get $T=2\pi / (1/10000^{2i/d_{model}})=2\pi * 10000^{2i/d_{model}}$. We can see that $2i/d_{model}\in [0,1]$. So $T\in [2\pi, 10000 \cdot 2\pi]$. More concisely, $T\in\{2\pi, 2*2\pi, 3*2\pi, ..., 9999* 2\pi, 10000*2\pi\}$.

Here, $2*i$ means even dimensions, and $2*i+1$means odd dimensions, so $i\in\{0,1,2,...,d_{model}/2\}$ and $2i/d_{model}\in [0,1]$. Then the functions fo each dimension are as follows: sin, cos, sin, cos,...



**Then, why relative positions not absolute positions?**

> We chose this function because we hypothesized it would allow the model to easily to attend by relative positions, since for any fixed offset $k$, $PE_{pos}+k$ can be represented as  a linear function of $PE_{pos}​$.

Emmm..., it is intuitive and acceptable.



# Pytorch Implementation