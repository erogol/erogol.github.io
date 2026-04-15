---
layout: post
title: "Gradual Training with Tacotron for Faster Convergence"
description: "Tacotron is a commonly used Text-to-Speech architecture"
tags: deeplearning mozilla research tacotron tacotron2
minute: 4
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

Tacotron is a commonly used Text-to-Speech architecture. It is a very flexible alternative over traditional solutions. It only requires text and corresponding voice clips to train the model. It avoids the toil of fine-grained annotation of the data. However, Tacotron might also be very time demanding to train, especially if you don't know the right hyperparameters, to begin with. Here, I like to share a gradual training scheme to ease the training difficulty. In my experiments, it provides faster training, tolerance for hyperparameters and more time with your family.

In summary, Tacotron is an Encoder-Decoder architecture with Attention. it takes a sentence as a sequence of characters (or phonemes) and it outputs sequence of spectrogram frames to be ultimately converted to speech with an additional vocoder algorithm (e.g. Griffin-Lim or WaveRNN). There are two versions of Tacotron. Tacotron is a more complicated architecture but it has fewer model parameters as opposed to Tacotron2. Tacotron2 is much simpler but it is ~4x larger (~7m vs ~24m parameters). To be clear, so far, I mostly use gradual training method with Tacotron and about to begin to experiment with Tacotron2 soon.

![](https://user-images.githubusercontent.com/10332831/50150681-f8a33c80-02be-11e9-93eb-b894209ceed7.png)

Tacotron architecture (Thx @[yweweler](https://github.com/yweweler) for the figure)

Here is the trick. Tacotron has a parameter called 'r' which defines the number of spectrogram frames predicted per decoder iteration. It is a useful parameter to reduce the number of computations since the larger 'r', the fewer the decoder iterations. But setting the value to high might reduce the performance as well. Another benefit of higher r value is that the alignment module stabilizes much faster. If you talk someone who used Tacotron, he'd probably know what struggle the attention means. So finding the right trade-off for 'r' is a great deal. In the original Tacotron paper, authors used 'r' as 2 for the best-reported model. They also emphasize the challenge of training the model with r=1.

Gradual training comes to the rescue at this point. What it means is that we set 'r' initially large, such as 7. Then, as the training continues, we reduce it until the convergence. This simple trick helps quite magically to solve two main problems. The first, it helps the network to learn the monotonic attention after almost the first epoch. The second, it expedites convergence quite much. As a result, the final model happens to have more stable and resilient attention without any degrigation of performance. You can even eventually let the network to train with r=1 which was not even reported in the original paper.

Here, I like to share some results to prove the effectiveness. I used LJspeech dataset for all the results. The training schedule can be summarized as follows. (You see I also change the batch\_size but it is not necessary if you have enough GPU memory.)

> "gradual\_training": [[0, 7, 32], [10000, 5, 32], [50000, 3, 32], [130000, 2, 16], [290000, 1, 8]] # [start\_step, r, batch\_size]

Below you can see the attention at validation time after just 1K iterations with the training schedule above.

![](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_1024/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2019/09/attention_1k-1024x640.png)

Tacotron after 950 steps on LJSpeech. Don't worry about the last part, it is just because the model does not know where to stop initially.

Next, let's check the model training curve and convergence.

![](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_1024/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2019/09/Screenshot-2019-09-20-at-13.34.19-1024x583.png)

(Ignore the plot in the middle.) You see here the model jumping from r=7 to r=5. There is obvious easy gain after the jump.

![](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_1024/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2019/09/Screenshot-2019-09-20-at-13.41.28-1024x507.png)

Test time model results after 300K. r=1 after 290K steps.

![](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_1024/https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2019/09/Screenshot-2019-09-20-at-14.10.23-1024x531.png)

Here is the training plot until ~300K iterations.   
 (For some reason I could not move the first plot to the end)

You can listen to [voice examples](https://soundcloud.com/user-565970875/sets/gradual-training-results) generated with the final model using GriffinLim vocoder. I'd say the quality of these examples is quite good to my ear.

It was a short post but if you like to replicate the results here, you can visit our repo [Mozilla TTS](https://github.com/mozilla/TTS) and just run the training with the provided config.json file. Hope, imperfect documentation on the repo would help you. Otherwise, you can always ask for help creating an issue or on [Mozilla TTS Discourse](https://discourse.mozilla.org/c/tts) page. There are some other cool things in the repo that I also write about in the future. Until next time..!

**Disclaimer**: In this post, I just wanted to briefly share a trick that I find quite useful in my TTS work. Please feel free to share your comments. This work might be a more legit research work in the future.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Text to Speech Deep Learning Architectures](http://www.erogol.com/text-speech-deep-learning-architectures/ "Text to Speech Deep Learning Architectures")