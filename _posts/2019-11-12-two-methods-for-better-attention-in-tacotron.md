---
layout: post
title: "Two Attention Methods for Better Alignment with Tacotron"
description: "In this post, I like to introduce two methods that worked well in my experience for better attention"
tags: alignment attention github graves ljspeech
minute: 5
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

In this post, I like to introduce two methods that worked well in my experience for better attention alignment in Tacotron models. If you like to try your own you can visit **Mozilla TTS.** The first method is [Bidirectional Decoder](https://arxiv.org/abs/1907.09006) and the second is [Graves Attention](https://arxiv.org/abs/1308.0850) (Gaussian Attention) with small tweaks.

## Bidirectional Decoder

![](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_1024/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2019/11/image-1024x626.png)

from [the paper](https://arxiv.org/pdf/1907.09006.pdf)

Bidirectional decoding uses an extra decoder which takes the encoder outputs in the reverse order and then, there is an extra loss function that compares the output states of the forward decoder with the backward one. With this additional loss, the forward decoder models what it needs to expect for the next iterations. In this regard, the backward decoder punishes bad decisions of the forward decoder and vice versa.

Intuitionally, if the forward decoder fails to align the attention, that would cause a big loss and ultimately it would learn to go monotonically through the alignment process with a correction induced by the backward decoder. Therefore, this method is able to prevent "catastrophic failure" where the attention falls apart in the middle of a sentence and it never aligns again.

At the inference time, the paper suggests to us only the forward decoder and demote the backward decoder. However, it is possible to think more elaborate ways to combine these two models.

![](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_727/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2019/11/grafik.png)

Example attention figures of both of the decoders.

There are 2 main pitfalls of this method. The first, due to additional parameters of the backward decoder, it is slower to train this model (almost 2x) and this makes a huge difference especially when the reduction rate is low (number of frames the model generates per iteration). The second, if the backward decoder penalizes the forward one too harshly, that causes prosody degradation in overall. The paper suggests activating the additional loss just for fine-tuning, due to this.

My experience is that Bidirectional training is quite robust against alignment problems and it is especially useful if your dataset is hard. It also aligns almost after the first epoch. Yes, at inference time, it sometimes causes pronunciation problems but I solved this by doing the opposite of the paper's suggestion. I finetune the network without the additional loss for just an epoch and everything started to work well.

## Graves Attention

Tacotron uses Bahdenau Attention which is a content-based attention method. However, it does not consider location information, therefore, it needs to learn the monotonicity of the alignment just looking into the content which is a hard deal. Tacotron2 uses Location Sensitive Attention which takes account of the previous attention weights. By doing so, it learns the monotonic constraint. But it does not solve all of the problems and you can still experience failures with long or out of domain sentences.

Graves Attention is an alternative that uses content information to decide how far it needs to go on the alignment per iteration. It does this by using a mixture of Gaussian distribution.

Graves Attention takes the context vector of time t-1 and passes it through couple of fully connected layers ([FC > ReLU > FC] in our model) and estimates step-size, variance and distribution weights for time t. Then the estimated step-size is used to update the mean of Gaussian modes. Analogously, mean is the point of interest t the alignment path, variance is attention window over this point of interest and distribution weight is the importance of each distribution head.

![(g,b,k) = FC(ReLU(FC(c))))) \\
\delta = softplus(k)\\
\sigma = exp(-b)\\
w = softmax(g) \\
\mu_{t} = \mu_{t-1} + \delta \\
\alpha_{i,j} = \sum_{k} w_{k} exp\left(-\frac{(j-\mu_{i,k})^2}{2\sigma_{i,k})}\right ) ](https://cdn.shortpixel.ai/client/q_glossy,ret_img/https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_93493aa3a8fd60746a835205383f76d2.gif)(g,b,k) = FC(ReLU(FC(c))))) \\
\delta = softplus(k)\\
\sigma = exp(-b)\\
w = softmax(g) \\
\mu\_{t} = \mu\_{t-1} + \delta \\
\alpha\_{i,j} = \sum\_{k} w\_{k} exp\left(-\frac{(j-\mu\_{i,k})^2}{2\sigma\_{i,k})}\right )

I try to formulate above how I compute the alignment in my implementation. ![g, b, k](https://cdn.shortpixel.ai/client/q_glossy,ret_img/https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_2f5ec82ae01562c497b4138a8d980eaa.gif) are intermediate values. ![\delta](https://cdn.shortpixel.ai/client/q_glossy,ret_img/https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_77a3b715842b45e440a5bee15357ad29.gif) is the step size, ![\sigma](https://cdn.shortpixel.ai/client/q_glossy,ret_img/https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_a2ab7d71a0f07f388ff823293c147d21.gif) is the variance, ![w_{k}](https://cdn.shortpixel.ai/client/q_glossy,ret_img/https://web.archive.org/web/2020/http://erogol.com/wp-content/plugins/latex/cache/tex_4c7b679e0d3f8ac1664adbcec246e1b9.gif)w\_{k} is the distribution weight for the GMM node k. (You can also check the [code](https://github.com/mozilla/TTS/blob/dev/layers/common_layers.py#L147)).

Some other versions are explained [here](https://arxiv.org/abs/1910.10288) but so far I found the above formulation works for me the best, without any NaNs in training. I also realized that with the best-claimed method in this paper, one of the distribution nodes overruns the others in the middle of the training and basically, attention starts to run on a single Gaussian head.

![](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_1024/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2019/11/grafik-1-1024x500.png)

Test time attention plots with Graves Attention. The attention looks softer due to the scale differences of the values. In the first step, the attention is weight is big since distributions have almost uniform weights. And they differ as the attention runs forward.

The benefit of using GMM is to have more robust attention. It is also computationally light-weight compared to both bidirectional decoding and normal location attention. Therefore, you can increase your batch size and possibly converge faster.

The downside is that, although my experiments are not complete, GMM's not provided slightly worse prosody and naturalness compared to the other methods.

## Comparison

Here I compare Graves Attention, Bidirectional Decoding and Location Sensitive Attention trained on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset. For the comparison, I used the set of sentences provided by [this work](https://arxiv.org/abs/1905.09263). There are in total of 50 sentences.

Bidirectional Decoding has 1, Graves attention has 6, Location Sensitive Attention has 18, Location Sensitive Attention with inference time windowing has 11 failures out of these 50 sentences.

In terms of prosodic quality, in my opinion, Location Sensitive Attention > Bidirectional Decoding > Graves Attention > Location Sensitive Attention with Windowing. However, I should say the quality difference is hardly observable in LJSpeech dataset. I also need to point out that, it is a hard dataset.

If you like to try these methods, all these are implemented on [Mozilla TTS](https://github.com/mozilla/TTS) and give it a try.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.