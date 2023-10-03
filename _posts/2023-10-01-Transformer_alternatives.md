---
layout: post
title: DRAFT - Review of Alternative Models to Transformers
description: A review of alternative models to Transformers.
summary: A review of alternative models to Transformers.
tags: machine-learning deep-learning language-modelling llms review
minute: 10
---


# Alternatives to Transformer

## Transformer
Space: O(T^2 + Td)<br>
Time: O(T log Td)<br>

Traditional sequential models, like recurrent neural networks (RNNs) and long short-term memory networks (LSTMs), faced challenges in capturing long-range dependencies and parallelizing computations effectively. The Transformer architecture addresses these issues by relying on self-attention mechanisms.

At the core of the Transformer is the self-attention mechanism. Unlike traditional approaches where each element in a sequence is processed one at a time, self-attention allows the model to weigh the importance of different elements in the sequence relative to each other. This enables capturing relationships between distant words in a sentence.

Transfomer has some limitations and constraints in terms of computation and storage. Transformer is based on dot-product attention that computes ```softmax(Q*K.t)``` which is computationally heavy and it needs to store a KV cache that is also heavy in memory at inference. This is a limiting factor especially on problems with long context sizes. Transformers space complexity increases quadratically with the increasing context size.

The Transformer is a key component of the current LLM revolution, and researchers are actively seeking alternatives to address its limitations. While there have been several proposed alternatives, none have been as successful as the original model. Nevertheless, considering the scale of the state-of-the-art LLM problem and the high cost of training these models, even a slight improvement can have a significant impact.

In this article, I will discuss several variants of the Transformer model. I will outline their advantages and disadvantages, analyze their space and time complexities, and provide a detailed comparison of their performances.



## RWKV

Space: O(Td)<br>
Time: O(Td)<br>

Code: https://github.com/BlinkDL/RWKV-LM<br>
Paper: https://arxiv.org/abs/2305.13048<br>
https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py<br>

RWKV is a new approach to RNN models that combines the advantages of RNNs and Transformers while mitigating their known limitations. It introduces several key strategies that allow it to capture locality and long-range dependencies, while addressing limitations of current architectures. RWKV offers a promising and viable solution for handling tasks involving large-scale models with billions of parameters, exhibiting competitive performance at a fraction of the computational cost. If you're interested in improving the memory and computational complexity of transformers in natural language processing tasks, RWKV is definitely worth exploring.

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158150056478789703/image.png?ex=651b32ca&is=6519e14a&hm=4781857cc520c0d1abef45ae00bccbf9cf2fbd492694410c9b031dbad689d279&" alt="Image">
</figure>

RWKV is designed to combine the strengths of RNNs and Transformers while mitigating their known limitations. Compared to RNNs, RWKV offers more efficient parallelizable training and better performance on long-range dependencies by not relying on a single hidden unit to pass the context between different time steps.

Compared to Transformers, RWKV offers linear attention and constant computational and memory complexity during inference, making it more efficient for large-scale models.

There are two main components of a RWKV block. Time-mixing and channel-mixing. Time-mixing works by linearly interpolating between the current input and the previous time step input, which naturally aggregates and gates information in the input channels. The time-mixing block is composed of three equations that compute the values of r, k, and v at each time step, which are then used to compute the WKV computation that plays the role of attention in Transformers. In essence, as time passes and t becomes larger, the vector o_t relies on a  historical record which is indicated by the accumulation of a greater number of terms.

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158150244828188702/image.png?ex=651b32f7&is=6519e177&hm=b4fff3b20f4c38fc5a1d7be4dff01277764b94e74ff1b440087d89e6808988a9" alt="Image">
</figure>

Channel-mixing is another key component of the RWKV architecture that helps it capture locality in sequential data. It works by computing the values of r, k, and o at each time step, which are then used to compute the final output vector. The channel-mixing block is composed of three equations that compute the values of r, k, and o at each time step. The output vector is computed by taking the sigmoid of the receptance r and using it as a "forget gate" to eliminate unnecessary historical information. The final output vector is then computed by multiplying the sigmoid of r with the result of a max pooling operation on k, followed by a squared ReLU activation.

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158150324482211981/image.png?ex=651b330a&amp;is=6519e18a&amp;hm=a27f2cb37111f256b78dfb8b1a1946362f1bc847de4b93e196c57c1807db2ed7&amp;" alt="Image">
</figure>

RWKV comes with certain limitations  RWKV may have performance limitations on tasks that require recalling information over very long contexts due to relaying on a limited vector between time steps as in RNNs as opposed to Transformers having access to all the information at every step by the attention mechanism. One limitation of RWKV is that prompt engineering has become more significant compared to traditional Transformer models. In the RWKV framework, the linear attention mechanism restricts the extent to which prompt information is passed on to the model's continuation. It is also emprically shown that "When the prompts were adjusted from the ones used for GPT to more suitable for RWKV, the F1 measure performance increased even from 44.2% to 74.8%."

In the results, RWKV has shown impressive performance and has even outperformed other models in specific tasks. However, when the task demands a stronger reliance on context, RWKV's performance tends to decrease, leading to underperformance compared to other models.

 RWKV model serves as an outstanding illustration of an open-source project, with the paper mentioning many contributors. It is truly impressive to observe the significant influence that open-source research has had in advancing innovative AI solutions on a grand scale. It appears that efforts are already underway to address certain limitations of RWKV with a new iteration of the model architecture. You can join [their discord]( https://discord.gg/bDSBUMeFpc) if you are willing to get involved in the development process.








## Hyena

 Time: O(NdT (logT + d)) - N is number of projections <br>
 Space Complexity: O(Td)

 Code: https://github.com/HazyResearch/safari <br>
 Paper: https://arxiv.org/abs/2302.10866 <br>
 Blogpost: https://hazyresearch.stanford.edu/blog/2023-03-07-hyena <br>

Hyena addresses the Transformers limitations with their attention operator, which becomes computationally expensive with longer sequences and is unable to access a significant amount of context. Hyena offers a subquadratic alternative to attention by combining long convolutions with data-controlled gating. In various tasks involving recall and reasoning with sequences containing thousands to hundreds of thousands of tokens, Hyena has demonstrated significant improvements in accuracy compared to state-space operators and other methods. Additionally, Hyena has set a new standard for dense-attention-free architectures in language modeling. It achieves Transformer-level quality while reducing required training compute by 20% at a sequence length of 2K. Notably, Hyena operators are also faster, offering twice the efficiency of highly optimized attention operators.

<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147883573452930/Screenshot_2023-09-28_at_15.50.08.png?ex=651b30c4&is=6519df44&hm=2b69bef06a520b2132969521cf327e82d64100f3a6b7d435e8ba265870c1ec41&=&width=2204&height=990" alt="Image description">
</figure>

Hyena first projects the input into a set of vectors ```v, x_1, ..., x_n``` and ```v``` acts like the value vector as in the attention. Then it projects ```v, x_1, ..., x_n``` with learnable filters ```h_1, ..., h_n```. Hyena applies a multiplicative gating interaction to the projected vectors, similar to LSTMs. This gating is used to control the information flow thorugh the recurrance.

<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147883883839538/Screenshot_2023-09-28_at_16.56.27.png?ex=651b30c4&is=6519df44&hm=371aa3bf09a5953f22ab4719e69b6c8ce3abb1530e9c5c8a6a4d069f0af57258&=width=2204&height=466" alt="Image">
</figure>

Hyena uses an implicit long convolution to the gated input, using a set of Hyena filters thar are parametrized by a feed-forward network. This convolution is used to capture long-range dependencies in the input.

<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147884139684053/Screenshot_2023-09-28_at_17.28.21.png?ex=651b30c4&is=6519df44&hm=94592fd5b34bbbf083d352998bc7ec7ec10c836508e2b404c2bfe6b960e002d2&=&width=2204&height=466" alt="Image">
</figure>


<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147884416516166/Screenshot_2023-09-28_at_17.29.11.png?ex=651b30c4&is=6519df44&hm=6ca777f2a5790e4f49e7ea2c2184cc186aed1f0c414444e45f677319c6fca9e8&=&width=2204&height=512" alt="Image description">
</figure>

<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147884634615828/Screenshot_2023-09-28_at_17.29.49.png?ex=651b30c5&is=6519df45&hm=9e787acaef69633b283691dfecdd371581feeddf76d854af1c420ebe6d908955&=&width=2204&height=580" alt="Image">
</figure>

Below is the overall Hyena operator in python as described in the blogpost:

```python
def hyena_filter(t):
	return window(t) * ffn(t) * poitional_encoding(t)

x, v = input_projections(u)
for o in range(hyena_orders):
    h = hyena_filter(L)  # long conv filter parameterized via an MLP
    v = x[o] * fftconv(h, v)  # elem-wise mult & fftconv
)
```

When it comes to language modelling, Hyena is often compared to GPTNeo and RWKV. Hyena exhibits superior performance in few-shot learning, but RWKV outperforms in terms of zero-shot accuracy on SuperGLUE tasks. Moreover, Hyena performs on par with a Transformer model when it comes to WikiText103 dataset perplexity numbers.

In terms of runtime, the cross-over point between Hyena and attention occurs at a length of 2048 and Hyena and flash attention ranges from 4086 to 8196.

<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147884944990327/Screenshot_2023-09-28_at_17.44.48.png?ex=651b30c5&is=6519df45&hm=8efde44c22a32b3404f8c5c80d736f5d71bae95f178637493627289d8da0503a&=&width=2204&height=717"
  alt="Image">
</figure>

In my view, the Hyena model is a intersting approach for extending input length through scalable computing. Nonetheless, further investigations on a larger scale are necessary to confirm its efficacy as a viable alternative to the Transformer model. For now, the RWKV model appears to offer better value in terms of both complexity and performance. However, if the goal is to tackle very lengthy context problems, Hyena could be a promising choice.


## Attention Free Transformer

Time: AFT-simple O(Td), AFT-full O(T^2d) <br>
Space: O(Td) <br>

Paper: https://arxiv.org/abs/2105.14103v2<br>
Code (unofficial): https://github.com/rish-16/aft-pytorch<br>

Attention Free Transformer (AFT) is a new variant of the Transformer model that eliminates the need for dot product self attention, making it compatible with large input and model sizes. AFT takes advantage of locality and spatial weight sharing while maintaining global connectivity, resulting in excellent efficiency. The paper presents experiments on autoregressive modeling tasks and image recognition, demonstrating competitive performance compared to other models. Overall, AFT is a promising approach for efficient and effective deep learning.
Original attention can be implemented as

We can think of AFT as weighted average over values that is combined by the queries with element-wise multiplication instead of a heavy attention matrix. In an Attention-based Feed-Forward Transformer (AFT) layer, the learned position biases are added to the key values. Then, the values are combined with the key using element-wise multiplication. Finally, the resulting values are multiplied with the query in an element-wise manner. Thus it avoids the computationally heavy ```softmax(Q*K.t)``` operation of transformers. "AFT can be interpreted as performing implicit attention with as many heads as feature dimensions, where the attention matrices take a factorized form."

This new operation has a memory complexity linear with respect to both the context size and the dimension of features, making it compatible with both large input and model sizes. The AFT model variants take advantage of the idea of locality and spatial weight sharing while maintaining global connectivity. This allows AFT to capture long-term dependencies and achieve competitive performance on autoregressive modeling tasks and image recognition.

<figure>
    <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158143056378331156/Screenshot_2023-09-29_at_16.02.12.png?ex=651b2c45&is=6519dac5&hm=3f69a901a09a9e90c3b2e6a622ec69603f2d654c8e4be3eca506d08c65d1d3ac&" alt="Image">
</figure>

There are four different versions of AFT. The first version is AFT-simple, which does not utilize position encoding. The second version is AFT-full, which includes regular position encoding. The third version is AFT-local, which incorporates a learned set of relative position biases within a specified window. The fourth version is AFT-conv, which utilizes depth-wise separable convolution and underlined especially for image tasks.

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158143241707864185/Screenshot_2023-09-29_at_16.01.38.png?ex=651b2c72&is=6519daf2&hm=95cf4adb2b085f2a57a00731ec8816c0a3777d6ebcb1b29fd364695ef61322ae&" alt="Screenshot">
	<figcaption>AFT-conv formulation. </figcaption>
</figure>


In terms of results, the paper shows that AFT achieves comparable or better accuracy than traditional Transformer models on various autoregressive modeling tasks and image recognition tasks, while using much smaller memory footprints. AFT also outperforms other efficient Transformer variants such as Linformer and Performer. The paper also demonstrates the effectiveness of AFT on variable size inputs and shows that it is well-suited for pretraining and finetuning workflows in vision tasks.

In general, AFT shows great potential as a substitute for conventional Transformers. It offers substantial reductions in computational requirements and memory usage, all while maintaining high performance. Moreover, AFT serves as the foundation for the development of both Hyena and RWKV.



## Retentive Network

Time: O(Td(b + h)) s.t. b chunk size and h is head dimension <br>
Space: O(T) <br>

Paper: https://arxiv.org/abs/2307.08621<br>
Code (official):  https://github.com/microsoft/torchscale/commit/bf65397b26469ac9c24d83a9b779b285c1ec640b <br>
Code:  https://github.com/syncdoth/RetNet<br>
Code:  https://github.com/Jamie-Stirling/RetNet<br>


RetNet borrows recurrent input processing from RNN and parallel-training of Transformer models and combine them to achieve a compute efficient model. Recurrence enables O(1) inference since it does not need to compute the relation between every input and every other input in the sequence. RetNet applies recurrance chunk-wise to the input to alleviate the representational bottleneck of the regular RNN and model longer samples efficiently.

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158141074603253800/retnet-vs-transformer.webp?ex=651b2a6d&is=6519d8ed&hm=7904fcf6149253cc0742d0ab1ec4f17a78b405adfee7191618e144a0b40b2a28&" alt="Caption" title="Difference between Transformer and RetNet">
  <figcaption>Difference between Transformer and RetNet</figcaption>
</figure>


RetNet introduces a novel approach to replace the softmax operation utilized in self-attention with a Hadamard product. By leveraging a newly introduced **D-matrix and incorporating a GroupNorm operation**, the relative attention weights assigned to each token in the input sequence are determined. Traditionally, the softmax operation plays a crucial role in capturing long-term dependencies and contributes to the remarkable performance of Transformers. However, the computation of softmax, specifically ```softmax(Q * K.t)```, significantly hampers the efficiency of Transformers during inference. This is  due to the storage requirements of a squared ```NxN``` matrix, which grows quadratically with the sequence length.

RetNet utilizes two variants of the same computation, one for training and another for inference. This is the crux of RetNet's functionality. During training, a parallel computation approach is employed to expedite the process, while during inference, a recurrent formulation is utilized instead. I suggest you to check [this post](https://medium.com/ai-fusion-labs/retentive-networks-retnet-explained-the-much-awaited-transformers-killer-is-here-6c17e3e8add8) by Shantanu Chandra who made a better job than the paper explaining how things work.

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158143553000718397/image.png?ex=651b2cbc&is=6519db3c&hm=663ffeedd63406f87d7032c71c9b4884f01602e16e408f7b2de2a00a95cc61d4" alt="Image">
  <figcaption>Training and inference computation.</figcaption>
</figure>


When we compare RetNet to attention-free transformers and RWKV, it retains the element-wise interactions in the sequence with a certain constraint by the retention operation. It keeps the high-dimensional state of the encoded sequence information which they claim to contribute to the performance of the model.

Results show that after ~2.7B parameters RetNet achieves lower perplexity values and out-performs Transformer. Most of the results are reported based on the 6.7B model. RetNet is signifcantly better than Transformer at this scale in zero-shot, few-shot learning.

RetNet replaces the KV cache of Transformers by the propsed recurrance operation and saves memory also chunk-wise retention makes inference significantly scalable with increasing batch size and input length.

They also show that RetNet is computationally way efficient than Transformer and almost on par with Transformer + Flash Attention 1 (needs to compare Flash Attention2 as well). Results show that it uses 3.4x lower memory, 8.4x higher throughput, 15.6x lower latency in relation to a Transformer model.

When compared to the other Transformer alternatives RetNet outperforms all the other models by an important margin on language modelling.

<figure>
<img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158143990630187048/image.png?ex=651b2d24&is=6519dba4&hm=05891731dffc0e053173f55da8865837ad7805293a3c3b29eda0f0e2a7cc8490" alt="Image">
  <figcaption>Comparison with the other models.</figcaption>
</figure>

<br>
## Longnet

Time: O(Td)<br>
Space: O(T/r log T/r d) s.t. r is the attention dilation rate <br>

Paper: https://arxiv.org/pdf/2307.02486.pdf <br>
Code: https://github.com/microsoft/torchscale <br>

LONGNET is a variant of the Transformer model that tackles the issue of scaling sequence length in large language models. It can handle sequences with over **1 billion tokens** while maintaining good performance on shorter sequences. This is accomplished through dilated attention, which enhances the model's ability to attend to distant tokens. LONGNET has advantages such as linear computation complexity, the capability to serve as a distributed trainer for long sequences, and seamless integration with existing Transformer-based optimization. Experimental results confirm its effectiveness on long-sequence modeling as well as general language tasks.


<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158764016307556372/image.png?ex=651d6e96&is=651c1d16&hm=677a5c15b30c83014def640efb5341424801680255ce60bfedd2baf973c24a14" alt="Image">
</figure>

In order to simplify the self-attention layers, LONGNET utilizes dilated attention. This approach involves dividing the input sequence into segments and dilating each segment at a specific rate. By doing so, the model is able to leverage different segment and dilation rates to improve its modeling abilities. The outputs of each segment size and dilation rate combination are then combined through a weighted sum. These weights are determined based on the softmax denominators of each output. This combination of segments and dilation strikes a balance between considering the global context and maintaining efficiency, as dilation serves as an efficient approximation of the attention matrix.

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158766857587806258/image.png?ex=651d713b&is=651c1fbb&hm=40fa139299585deb1af9fb02e23c05007dea236811e3ce34ea5414a81ffe175a&" alt="Image Description">
</figure>

There are two additional techniques that can be employed. One of them, called LONGNET, incorporates varying dilation rates in each attention head to introduce more diversity. This technique also gradually increases the segment lengths and dilation rates in successive layers, allowing for the processing of extremely long input sequences.

Training LONGNET for 1 billion tokens requires distributed training. Due to segment nature, any long text can be segmented and those segments can be distributed on different GPUs and processed in-parallel with a constant communication overhead.

For testing the model, they used the Stack dataset, a source code collection with over 300 programming languages. They showed that LONGNET outperforms a vanilla Transformer model by a large margin in final perplexity and computation. They were able to train LONGNET with 32k context size and the Transformer only 16k.

<figure>
    <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158774319560798238/image.png?ex=651d782e&is=651c26ae&hm=1d47402aaab7e0ee804e53e76179c11d03a48e654cb30bce1dde0c3b91f7a051&" alt="Image">
</figure>

**My 2 cents:** Consider using LONGNET when you need to process very long context or stream outputs.



## MegaByte
https://arxiv.org/abs/2305.07185
https://github.com/lucidrains/MEGABYTE-pytorch



## Multi-Query Attention
 https://arxiv.org/pdf/1911.02150.pdf
 https://medium.com/@nidhibits224/multi-query-attention-speeding-ai-ad8fa1626b82


## Linformer
O(n)



 ## Roformer

  https://arxiv.org/abs/2104.09864

	## Monarch Mixer
	https://hazyresearch.stanford.edu/blog/2023-07-25-m2-bert


## Hierarchical Transformers
https://arxiv.org/pdf/2110.13711.pdf

 ## One Wide Feedforward is All You Need
 https://arxiv.org/abs/2309.01826


## Performer
Time: O(Td^2 log d)
Space: O(Td log d + d^2 lod d)

Paper: https://arxiv.org/abs/2009.14794v4
Code: https://github.com/facebookresearch/xformers/blob/4e096c4afd4d1d377053cdfc6964f67f6435dceb/xformers/components/attention/favor.py#L41

## Reformer
Time: O(T log Td)
Space: O(T log T + Td)

https://arxiv.org/abs/2001.04451
https://blog.research.google/2020/01/reformer-efficient-transformer.html
https://github.com/lucidrains/reformer-pytorch
