---
layout: post
title: A Review for Transformer Variants
description: A review of alternative models to Transformers.
summary: A review of alternative models to Transformers.
tags: machine-learning deep-learning language-modelling llms review
minute: 10
---

<style>
img {
    border-radius: 10px;
}
</style>

In this article, I will explore various alternatives to transformers, considering their architectural improvements, computational efficiency, and performance results across different benchmarks. I intend to continually update this post with new models in the future. If you believe there are any models or important points that should be included or any corrections that need to be made, please feel free to contact me.

## Transformer
Space: `O(T^2 + Td)`
Time: `O(T log Td)`

Traditional sequential models, like recurrent neural networks (RNNs) and long short-term memory networks (LSTMs), faced challenges in effectively capturing long-range dependencies and parallelizing computations. The Transformer architecture addresses these issues by relying on self-attention mechanisms.

At the core of the Transformer is the self-attention mechanism. Unlike traditional approaches, where each element in a sequence is processed one at a time, self-attention allows the model to weigh the importance of different elements relative to each other. This enables capturing relationships between distant words in a sentence.

Transformer has some limitations and constraints in terms of computation and storage. The Transformer is based on dot-product attention that computes ```softmax(Q*K.t)```, which is computationally heavy, and it needs to store a KV cache that is also heavy in memory at inference. This is a limiting factor, especially in problems with extended context sizes. Transformers' space complexity increases quadratically with the increasing context size.

The Transformer is a key component of the current LLM revolution, and researchers are actively seeking alternatives to address its limitations. While there have been several proposed alternatives, the original model has yet to be as successful as the original model. Nevertheless, considering the scale of the state-of-the-art LLM problem and the high cost of training these models, even a slight improvement can have a significant impact.



## RWKV

Space: `O(Td)`
Time: `O(Td)`

👩‍💻 [Code](https://github.com/BlinkDL/RWKV-LM)
📎 [Paper](https://arxiv.org/abs/2305.13048)

RWKV is a new approach that combines the advantages of RNNs and Transformers while mitigating their known limitations. It introduces several key strategies that allow it to capture local and long-range dependencies. RWKV offers a promising and viable solution for handling tasks involving large-scale models with billions of parameters, exhibiting competitive performance at a fraction of the computational cost. If you're interested in improving transformers' memory and computational complexity in natural language processing tasks, RWKV is worth exploring. However, in my experiments, it did not perform as good as transformers in audio tasks.

<figure style="text-align:center;">
    <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158150056478789703/image.png?ex=651b32ca&is=6519e14a&hm=4781857cc520c0d1abef45ae00bccbf9cf2fbd492694410c9b031dbad689d279&" alt="Image" width="75%" height="75%">
</figure>

The aim of RWKV is to harness the advantages of both RNNs and Transformers while addressing their shortcomings. In comparison to RNNs, RWKV provides more efficient parallelizable training and improved performance in capturing long-range dependencies. This is achieved by eliminating the reliance on a single vector to transmit the context between different time steps.

Compared to Transformers, RWKV offers linear attention and constant computational and memory complexity during inference, making it more efficient for large-scale models.

There are two primary components of a RWKV block: time-mixing and channel-mixing. Time-mixing operates by using linear interpolation to blend the current input with the input from the previous time step. This process effectively combines and controls the information in the input channels. The time-mixing block is composed of three equations that compute the values of r, k, and v at each time step, which are then used to calculate the WKV that plays the role of Transformer's attention. Essentially, as time progresses and t increases, the vector o_t becomes increasingly dependent on a historical record.

<figure style="text-align:center;">
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158150244828188702/image.png?ex=651b32f7&is=6519e177&hm=b4fff3b20f4c38fc5a1d7be4dff01277764b94e74ff1b440087d89e6808988a9" alt="Image" width="75%" height="75%">
</figure>

Channel mixing aids in capturing local information effectively. It works by computing the values of r, k, and o at each time step, which are then used to calculate the final output vector. The channel-mixing block comprises three equations that compute the values of r, k, and o at each time step. The output vector is calculated by taking the sigmoid of the receptance r and using it as a "forget gate" to eliminate unnecessary historical information. The final output vector is then computed by multiplying the sigmoid of r with the result of a max pooling operation on k, followed by a squared ReLU activation.

<figure style="text-align:center;">
    <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158150324482211981/image.png?ex=651b330a&amp;is=6519e18a&amp;hm=a27f2cb37111f256b78dfb8b1a1946362f1bc847de4b93e196c57c1807db2ed7&amp;" alt="Image" width="75%" height="75%">
</figure>

RWKV has certain limitations. For instance, it may struggle with tasks that require recalling information over a long context. This is because RWKV relies on a limited window between time steps, whereas Transformers have access to all the information at each step through attention. Another limitation is the prominence of prompt engineering in RWKV. In RWKV, the linear attention mechanism restricts the extent to which prompt information is passed on to the model. Empirical evidence supports this, showing that when prompts were adjusted to be  for RWKV, there was a significant increase in performance, with the F1 measure improving from 44.2% to 74.8%.

The results have demonstrated that RWKV has delivered impressive performance and surpassed other models in certain tasks. Nevertheless, when tasks require a greater emphasis on context, RWKV's performance tends to decline.

RWKV model serves as an outstanding illustration of an open-source project, with the paper mentioning many contributors. It is impressive to observe the significant influence that open-source research has had in advancing innovative AI solutions on a grand scale. Efforts are already underway to address the limitations of RWKV. You can join [their discord]( https://discord.gg/bDSBUMeFpc) if you are willing to get involved in the development process.







## Hyena

 Time: `O(NdT (logT + d))` st. N is the number of projections
 Space Complexity: `O(Td)`

👩‍💻 [Code](https://github.com/HazyResearch/safari)
📎 [Paper](https://arxiv.org/abs/2302.10866)
📎 [Blogpost](https://hazyresearch.stanford.edu/blog/2023-03-07-hyena)

Hyena addresses the Transformer's limitations with their attention operator, which becomes computationally expensive with longer sequences and cannot access a significant amount of context. Hyena offers a subquadratic alternative to attention by combining long convolutions with data-controlled gating. In various tasks involving recall and reasoning with sequences containing thousands to hundreds of thousands of tokens, Hyena has demonstrated significant improvements in accuracy. It achieves Transformer-level quality while reducing required training computed by 20% at a sequence length of 2K. Notably, Hyena operators are also faster, offering twice the efficiency of highly optimized attention operators.

<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147883573452930/Screenshot_2023-09-28_at_15.50.08.png?ex=651b30c4&is=6519df44&hm=2b69bef06a520b2132969521cf327e82d64100f3a6b7d435e8ba265870c1ec41&=&width=2204&height=990" alt="Image description">
</figure>

Hyena first projects the input into a set of vectors ```v, x_1, ..., x_n``` and ```v``` acts like the value vector as in the attention. Then it projects ```v, x_1, ..., x_n``` with learnable filters ```h_1, ..., h_n```. Hyena applies a multiplicative gating interaction to the projected vectors, similar to LSTMs. This gating is used to control the information flow through the recurrence.

<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147883883839538/Screenshot_2023-09-28_at_16.56.27.png?ex=652c5444&is=6519df44&hm=f64159b863bc947cd4d25b99cdcac8c9c621faafbc43461ae53ed6059e110653&=&width=2256&height=478" alt="Image">
</figure>

Hyena uses an implicit long convolution to the gated input, using a set of Hyena filters that are parametrized by a feedforward network. This convolution is used to capture long-range dependencies in the input.

<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147884139684053/Screenshot_2023-09-28_at_17.28.21.png?ex=651b30c4&is=6519df44&hm=94592fd5b34bbbf083d352998bc7ec7ec10c836508e2b404c2bfe6b960e002d2&=&width=2204&height=466" alt="Image">
</figure>


<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147884416516166/Screenshot_2023-09-28_at_17.29.11.png?ex=651b30c4&is=6519df44&hm=6ca777f2a5790e4f49e7ea2c2184cc186aed1f0c414444e45f677319c6fca9e8&=&width=2204&height=512" alt="Image description">
</figure>

<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147884634615828/Screenshot_2023-09-28_at_17.29.49.png?ex=651b30c5&is=6519df45&hm=9e787acaef69633b283691dfecdd371581feeddf76d854af1c420ebe6d908955&=&width=2204&height=580" alt="Image">
</figure>

Below is the overall Hyena operator in Python as described in the blog post:

```python
def hyena_filter(t):
    return window(t) * ffn(t) * poitional_encoding(t)

x, v = input_projections(u)
for o in range(hyena_orders):
    h = hyena_filter(L)  # long conv filter parameterized via an MLP
    v = x[o] * fftconv(h, v)  # elem-wise mult & fftconv
)
```

Regarding language modeling, Hyena is compared to GPTNeo and RWKV. Hyena outperforms in few-shot learning, but RWKV is better in zero-shot accuracy on SuperGLUE tasks. Moreover, Hyena performs on par with a Transformer on language modeling with WikiText103 dataset.

Regarding runtime, the cross-over point between Hyena and attention occurs at 2048, and Hyena and flash attention range from 4086 to 8196.

<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1158147884944990327/Screenshot_2023-09-28_at_17.44.48.png?ex=651b30c5&is=6519df45&hm=8efde44c22a32b3404f8c5c80d736f5d71bae95f178637493627289d8da0503a&=&width=2204&height=717"
  alt="Image">
</figure>

**My 2 cents**: Hyena is an interesting approach for extending input length through scalable computing. Nonetheless, further investigations on a larger scale are necessary to confirm its efficacy as a viable alternative to the Transformer model. For now, the RWKV model offers better value in terms of both complexity and performance. However, if the goal is to tackle lengthy context problems, Hyena could be a promising choice.



## Attention Free Transformer

Time: AFT-simple `O(Td)`, AFT-full `O(T^2d)`
Space: `O(Td)`

📎 [Paper](https://arxiv.org/abs/2105.14103v2)
👩‍💻 [Code (unofficial)](https://github.com/rish-16/aft-pytorch)

Attention Free Transformer (AFT) eliminates the need for dot product self-attention, making it scalable with long inputs and large model sizes. AFT takes advantage of locality and spatial weight sharing while maintaining global connectivity, resulting in excellent efficiency. The paper presents experiments on autoregressive modeling tasks and image recognition, demonstrating competitive performance compared to other models.

AFT is a weighted average over values combined by the queries with element-wise multiplication instead of a heavy attention matrix. In an Attention-based Feedforward Transformer (AFT) layer, the learned position biases are added to the key values. Then, the values are combined with the key using element-wise multiplication. Finally, the resulting values are multiplied with the query element-wise. Thus, it avoids the computationally heavy ```softmax(Q*K.t)``` operation of a Transformer. "AFT can be interpreted as performing implicit attention with as many heads as feature dimensions, where the attention matrices take a factorized form."

<figure>
    <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158143056378331156/Screenshot_2023-09-29_at_16.02.12.png?ex=651b2c45&is=6519dac5&hm=3f69a901a09a9e90c3b2e6a622ec69603f2d654c8e4be3eca506d08c65d1d3ac&" alt="Image">
</figure>

There are four different versions of AFT. The first version is AFT-simple, which does not utilize position encoding. The second version is AFT-full, which includes regular position encoding. The third version is AFT-local, incorporating a learned set of relative position biases within a specified window. The fourth version is AFT-conv, which utilizes depth-wise separable convolution and is proposed especially for image tasks.

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158143241707864185/Screenshot_2023-09-29_at_16.01.38.png?ex=651b2c72&is=6519daf2&hm=95cf4adb2b085f2a57a00731ec8816c0a3777d6ebcb1b29fd364695ef61322ae&" alt="Screenshot">
    <figcaption>AFT-conv formulation. </figcaption>
</figure>


In terms of results, the paper shows that AFT achieves comparable or better accuracy than traditional Transformers on various autoregressive modeling tasks and image recognition tasks while using much smaller memory footprints. AFT also outperforms other efficient Transformer variants such as Linformer and Performer. The paper also demonstrates the effectiveness of AFT on variable-length inputs and shows that it is well-suited for pretraining and finetuning workflows in vision tasks.

In general, AFT shows potential as a substitute for conventional Transformers. It substantially reduces computational requirements and memory usage, all while maintaining high performance. Moreover, AFT serves as the foundation for the development of both Hyena and RWKV.



## Retentive Network

Time: `O(Td(b + h))` s.t. b chunk size and h is head dimension
Space: `O(T)`

📎 [Paper](https://arxiv.org/abs/2307.08621)
👩‍💻 [Official Code](https://github.com/microsoft/torchscale/commit/bf65397b26469ac9c24d83a9b779b285c1ec640b)
👩‍💻 [Code 1](https://github.com/syncdoth/RetNet)
👩‍💻 [Code 2](https://github.com/Jamie-Stirling/RetNet)

RetNet borrows recurrent inference from RNN and parallel-training from Transformer, combining them to achieve an efficient model. Recurrent models facilitate O(1) inference as they do not require modeling the relationship between each input and every other input in the sequence. RetNet applies chunk-wise recurrence to alleviate the representational bottleneck of RNNs and effectively model longer context.

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158141074603253800/retnet-vs-transformer.webp?ex=651b2a6d&is=6519d8ed&hm=7904fcf6149253cc0742d0ab1ec4f17a78b405adfee7191618e144a0b40b2a28&" alt="Caption" title="Difference between Transformer and RetNet">
  <figcaption>Difference between Transformer and RetNet</figcaption>
</figure>


RetNet introduces a novel approach to replace the softmax operation utilized in self-attention with a Hadamard product. By leveraging a newly introduced **D-matrix and incorporating a GroupNorm operation**, the relative attention weights assigned to each token in the input sequence are determined. Traditionally,  softmax  plays a crucial role in capturing long-term dependencies and contributes to the remarkable performance of Transformers.

In RetNet, training and inference use of different flows that result in the same computation. In the training phase, a parallel formulation is utilized, while in the inference phase, a recurrent formulation.I suggest you check [this post](https://medium.com/ai-fusion-labs/retentive-networks-retnet-explained-the-much-awaited-transformers-killer-is-here-6c17e3e8add8) by Shantanu Chandra who did a better job than the paper explaining how things work.

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158143553000718397/image.png?ex=651b2cbc&is=6519db3c&hm=663ffeedd63406f87d7032c71c9b4884f01602e16e408f7b2de2a00a95cc61d4" alt="Image">
  <figcaption>Training and inference computation.</figcaption>
</figure>


When we compare RetNet to attention-free transformers and RWKV, it retains the element-wise interactions in the sequence with the retention operation. It keeps the high-dimensional state of the encoded sequence information, which they claim to contribute to the model performance.

Results show that after ~2.7B parameters, RetNet achieves lower perplexity and outperforms Transformer. Most of the results are reported based on the 6.7B model. RetNet is significantly better than Transformer at this scale in zero-shot, few-shot learning.

RetNet replaces the KV cache of Transformers with recurrence and saves memory. Also, chunk-wise retention makes inference significantly scalable with increasing batch size and input length.

They also show that RetNet is computationally way more efficient than Transformer and almost on par with Transformer + Flash Attention 1 (needs to compare Flash Attention2). Results show that it uses 3.4x lower memory, 8.4x higher throughput, and 15.6x lower latency concerning a Transformer model.

When compared to the other Transformer alternatives, RetNet outperforms all the different models by a big margin on language modeling.

<figure>
<img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158143990630187048/image.png?ex=651b2d24&is=6519dba4&hm=05891731dffc0e053173f55da8865837ad7805293a3c3b29eda0f0e2a7cc8490" alt="Image">
  <figcaption>Comparison with the other models.</figcaption>
</figure>


<br>
## Longnet

Time: `O(Td)`
Space: `O(T/r log T/r d)` s.t. r is the attention dilation rate

📎 [Paper](https://arxiv.org/pdf/2307.02486.pdf)
👩‍💻 [Code](https://github.com/microsoft/torchscale)

LONGNET  is designed to tackle longer sequence lengths. It can handle sequences with over **1 billion tokens** while maintaining good performance on shorter sequences. This is accomplished through dilated attention, which enhances the model's ability to attend to distant tokens. LONGNET has advantages such as linear computation complexity, the capability to serve as a distributed trainer for long sequences. Experiments confirm its effectiveness.


<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158764016307556372/image.png?ex=651d6e96&is=651c1d16&hm=677a5c15b30c83014def640efb5341424801680255ce60bfedd2baf973c24a14" alt="Image">
</figure>

To simplify the self-attention layers, LONGNET utilizes dilated attention. This approach involves dividing the input sequence into segments and dilating each segment at a specific rate. By doing so, the model is able to leverage different segment lengths and dilation rates to improve its modeling abilities. The outputs of each segment size and dilation rate pairs are then combined through a weighted sum. These weights are determined based on the softmax denominators of each output. Combination of using segments and dilated attention strikes a balance between considering the global context and maintaining efficiency, as dilation serves as an efficient approximation of the attention matrix.

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158766857587806258/image.png?ex=651d713b&is=651c1fbb&hm=40fa139299585deb1af9fb02e23c05007dea236811e3ce34ea5414a81ffe175a&" alt="Image Description">
</figure>

Two more tricks LONGNET employs for better modeling. It incorporates varying dilation rates in each attention head for more diversity. It also gradually increases the segment lengths and dilation rates in successive layers, allowing for the processing extremely long input sequences with a larger increasing receptive window in later layers.

To train LONGNET on 1 billion tokens, distributed training is necessary. LONGNET divides the inputs into segments, which are then distributed across different GPUs. These segments are processed simultaneously, with a constant communication overhead.

They used the Stack dataset to test the model, a source code collection with over 300 programming languages. They showed that LONGNET outperforms a vanilla Transformer model by a large margin in  perplexity and computation. They were able to train LONGNET with 32k context size whereas the Transformer only 16k.

<figure>
    <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1158774319560798238/image.png?ex=651d782e&is=651c26ae&hm=1d47402aaab7e0ee804e53e76179c11d03a48e654cb30bce1dde0c3b91f7a051&" alt="Image">
</figure>

**My 2 cents:** Consider using LONGNET when processing a long context or stream outputs.



<br>
## MegaByte
Time: `O(T ^ (4/3)  d)`
Space: `O(T log Td)`

📎 [Paper](https://arxiv.org/abs/2305.07185)
👩‍💻 [Code](https://github.com/lucidrains/MEGABYTE-pytorch)

<figure>
  <img src="https://cdn.discordapp.com/attachments/1158141030080716891/1159125345593737246/image.png?ex=651ebf19&is=651d6d99&hm=987d8fe538501b0d3970b50846dfa7f154a721edbf771dac74f8a9da7708675c&" alt="image">
</figure>

MEGABYTE is a "multiscale decoder architecture that enables end-to-end differentiable modeling of sequences of over one million bytes". MEGABYTE utilizes byte values to directly model data, requiring the ability to effectively capture a lengthy context. To achieve this, it divides sequences into patches and employs a local model for each patch, while also incorporating a global model between patches. By doing so, MEGABYTE enables sub-quadratic self-attention, facilitates larger feedforward layers without incurring additional computational cost, and enhances parallelism during decoding. As a result, MEGABYTE delivers improved performance for training and generation efficiently.

MEGABYTES offers several advantages, including sub-quadratic self-attention, per patch feedforward layers, and parallel decoding. The sub-quadratic self-attention is achieved by dividing the input into smaller "patches,". This reduces the self-attention cost to `O(T^(4/3) d)`.

They note that in a Transformer, the feedforward layers consume about 98% of the FLOPs. MEGABYTES addresses this by replacing multiple passes of these layers with a single pass, utilizing a larger linear layer.

Furthermore, the use of patches also introduces a level of parallelism. As a result, they found that their 1.5B parameter model is 40% faster than a 350M Transformer model.

The MEGABYTE system is composed of three main components:
- patch embedder, which converts the patch sequences into a representation that considers the context.
- global Transformer that encodes the contextualized inputs.
- local Transformer model that takes each output from the global model and predicts the output tokens in an auto-regressive manner.

MEGABYTE is applied to language modeling, image modeling, and audio modeling. The cool thing is that it is trained by the raw byte values (hence the name). It is compared to PerceiverAR and a Transformer baseline. In all tasks, it outperforms both and is competitive with models that use tokenizers to discretize the input.

The ablation analysis shows the importance of having both local and global models. If one of these components is absent, there is a notable decline in performance.

**My 2 cents:** I find learning from raw bytes and utilizing multi-stage transformers intriguing. This approach can potentially revolutionize language model systems (LLMs). By eliminating tokenization models, we can bridge the gap between computers and models, paving the way for developing new generation LLM-based operating systems.

In addition, I'd like to try MEGABYTE for text-to-speech. I believe it is well-suited to learn local and global relations better than Transfomers for TTS.

**Edit**: Looks like [UniAudio](https://arxiv.org/abs/2310.00704) did it.



## Noteworthy Mentions

Here are a few other noteworthy models that I won't delve into further since they have yet to gain much traction in the community or are simple tricks that don't require much explanation.

### Multi-Query Attention
📎[Paper](https://arxiv.org/pdf/1911.02150.pdf)
👩‍💻[Code](https://github.com/knotgrass/attention/blob/main/attn/attention.py)

Using shared key and value vectors among attention heads reduces the memory overhead at inference by reducing the size of the KV cache.


### Linformer
📎 [Paper](https://arxiv.org/abs/2006.04768v3)
👩‍💻 [Code](https://github.com/facebookresearch/fairseq/tree/main/examples/linformer)

A linear time self-attention is achieved by breaking down the scaled dot-product attention into multiple smaller attentions using linear projections. Together, these operations create a low-rank factorization of the original attention mechanism.


## Roformer
 📎 [Paper](https://arxiv.org/abs/2104.09864)
👩‍💻 [Code](https://huggingface.co/docs/transformers/model_doc/roformer)

"Rotary Position Embedding, or RoPE, is a type of position embedding which encodes absolute positional information with rotation matrix and naturally incorporates explicit relative position dependency in self-attention formulation."


## One Wide Feedforward is All You Need
📎 [Paper](https://arxiv.org/abs/2309.01826)

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;One Wide Feedforward is All You Need&quot; from Apple<br><br>- FFN parameters are redundant in the Transformer <br>- Remove FFN on decoder<br>- Share an FFN in encoder<br>- Slight accuracy drop<br>- Scale back the model to the org size. <br>- Improved accuracy and latency<a href="https://t.co/2Q5hFe7RRA">https://t.co/2Q5hFe7RRA</a></p>&mdash; erogol 🐸💬 (@erogol) <a href="https://twitter.com/erogol/status/1701633558316535883?ref_src=twsrc%5Etfw">September 12, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

It is suggested that the Feedforward Network (FFN) are unnecessary and redundant in Transformers. As a result, the FFN is removed from the Transformer decoder, shared in the encoder. Even though there is a small decrease in accuracy as a result of this change, when the model is scaled back to its original size, it leads to enhanced accuracy and decreased latency. They report 18.5% speed-up using this technique.


## Performer
Time: `O(Td^2 log d)`
Space: `O(Td log d + d^2 lod d)`

📎 [Paper](https://arxiv.org/abs/2009.14794v4)
👩‍💻 [Code](https://github.com/facebookresearch/xformers/blob/4e096c4afd4d1d377053cdfc6964f67f6435dceb/xformers/components/attention/favor.py#L41)

Performer can "estimate" regular dot-product attention using an approach called "Fast attention via positive orthogonal random features" FAVOR+. FAVOR+ combines low-rank approximation, matrix factorization, and matrix decomposition; then the space and time complexity becomes much more linear.


## Reformer
Time: `O(T log Td)`
Space: O`(T log T + Td)`

[📎Paper](https://arxiv.org/abs/2001.04451)
[👩‍💻Code (unofficial)](https://github.com/lucidrains/reformer-pytorch)

Reformer model incorporates three techniques to improve efficiency. First, it uses "reversible residuals" to reduce memory consumption by storing only one copy of the intermediate activation that can be used to reproduce the activations of the earlier layers by the model parameters. This helps minimize the memory overhead. Second, it splits values into chunks, saving memory in FFT layers and make the inference more efficient. Lastly, Reformer uses locality-sensitive hashing to approximate the attention matrix for a more efficient runtime.


## Monarch Mixer

👩‍💻[Blog](https://hazyresearch.stanford.edu/blog/2023-07-25-m2-bert)
👩‍💻[Code](https://github.com/HazyResearch/m2)

"Monarch Mixer uses monarch matrices for a sub-quadratic model in sequence length and model dimension. The idea is to replace the major elements of a Transformer with Monarch matrices — which are a class of structured matrices that "generalize the FFT and are sub-quadratic, hardware-efficient, and expressive."  In Monarch Mixer, they use layers built up from Monarch matrices to mix across the sequence (replacing the Attention operation) and across the model dimension (replacing the dense MLP).

## Conformers

📎 [Paper](https://arxiv.org/abs/2005.08100)
👩‍💻 [Code (unofficial)](https://github.com/sooftware/conformer)

The Conformer is a variant designed for speech recognition. While the Transformer excels at capturing global relationships, it is less effective than convolutional layers in capturing local information. To address this, the Conformer augments the Transformer model by adding convolutional layers between the attention module and the final feedforward layer. As a result, the Conformer achieves significantly better performance than previous Transformer and CNN-based models, setting new state-of-the-art on ASR.


### Efficient Streaming LMs with Attention Sinks

📎 [Paper](https://arxiv.org/pdf/2309.17453v1.pdf)

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Efficient Streaming Language Models with Attention Sinks <br><br>- Softmax in attention forces sum to 1 <br>- Thus it always attends first tokens <br>- Add learnable start tokens (sinks) <br>- Sliding window context with sinks <br>- Stable, fast, scalable inference! <a href="https://t.co/xNG4asnxWc">https://t.co/xNG4asnxWc</a></p>&mdash; erogol 🐸💬 (@erogol) <a href="https://twitter.com/erogol/status/1708811519511744899?ref_src=twsrc%5Etfw">October 2, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

This looks similar to Longnet, but they keep a set of learnable tokens - sinks - at the beginning of the generated sequence, observing that it improves stability and performance even if you window the attention computation.

### Simplifying Transformer Blocks

📎 [Paper](https://arxiv.org/pdf/2311.01906.pdf)

<figure>
  <img src="https://media.discordapp.net/attachments/1158141030080716891/1174358243426979920/image.png?ex=65674d57&is=6554d857&hm=4c029263a3d055efcb5420b09cd8a957837c82ae3399c50fecce5e7dbce45cd3&=&width=2254&height=1342" alt="image">
</figure>

Proposed model removes attention sub-block skip connection, value and projection parameters, MLP skip connection and normalization layers. They report 15% faster training and 15% fewer parameters with no loss in performance.
One caveat is that they only experimented with relatively smaller models (200-300m parameters) and it is not clear if this approach scales to larger models. Also, experiments are done in a limited number of tasks. However,
it is easy to implement and worth trying at least for the model scales and tasks mentioned in the paper.