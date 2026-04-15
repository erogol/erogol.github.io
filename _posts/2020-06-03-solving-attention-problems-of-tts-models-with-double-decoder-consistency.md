---
layout: post
title: "Solving Attention Problems of TTS Models with Double Decoder Consistency"
description: "Model Samples: <https://erogol"
tags: deep-learning mozilla-tts text-to-speech text2speech tts
minute: 15
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

**Model Samples:** <https://erogol.github.io/ddc-samples/>

**Colab Notebook (PyTorch):** [link](https://colab.research.google.com/drive/1u_16ZzHjKYFn1HNVuA4Qf_i2MMFB9olY?usp=sharing)

**Colab Notebook (Tensorflow):** [link](https://colab.research.google.com/drive/1LgQpdbgLHjyjTxgs6LHaKH_yl0luxkhT?usp=sharing)

Despite the success of the latest attention based end2end text2speech (TTS) models, they suffer from attention alignment problems at inference time. They occur especially with long-text inputs or out-of-domain character sequences. Here I like to propose a novel technique to fight against these alignment problems which I call Double Decoder Consistency (DDC) (with a limited creativity). DDC consists of two decoders that learn synchronously with different reduction factors. We use the level of consistency of these decoders to attain better attention performance.

[](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2020/07/ls1.mp4)

One of Shakespeare’s works read by the DDC model.

## End-to-End TTS Models with Attention

Good examples of attention based TTS models are Tacotron and Tacotron2 **[1][2]**. Tacotron2 is also the main architecture used in this work. These models comprise a sequence-to-sequence architecture with an encoder, an attention-module, a decoder and an additional stack of layers called Postnet. The encoder takes an input text and computes a hidden representation from which the decoder computes predictions of the target acoustic feature frames. A context-based attention mechanism is used to align the input text with the predictions. Finally, decoder predictions are passed over the Postnet which predicts residual information to improve the reconstruction performance of the model. In general, mel-spectrograms are used as acoustic features to represent audio signals in a lower temporal resolution and perceptually meaningful way.

Tacotron proposes to compute multiple non-overlapping output frames by the decoder. You are able to set the number of output frames per decoder step which is called ‘reduction rate’ (r). Larger the reduction rate, fewer the number of decoder steps required for the model to produce the same length output. Thereby, the model achieves faster training convergence and easier attention alignment, as explained in [1]. However, larger r values also produce smoother output frames and therefore, reduce the frame-level details.

Although these models are used in TTS systems for more natural-sounding speech, they frequently suffer from attention alignment problems, especially at inference time, because of out-of-the-domain words, long input texts, or intricacies of the target language. One solution is to use larger r for a better alignment however, as note above, it reduces the quality of the predicted frames. DDC tries to mitigate these attention problems by acting on these observations to find a suitable architecture finding the middle ground.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2020/06/model_overview-1-678x1024.png)

Fig1. This is an overview of the model used in this work. (Excuse my artwork).

The bare-bone model used in this work is formalized as follows:

![\[\{h_l\}^L_{l=1} = Encoder(\{x_l\}^L_{l=1})\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-b5dd6a47c9431750186cb5699bcf6891_l3.svg "Rendered by QuickLaTeX.com")

![\[p_t = Prenet(o_{t-1})\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-be04afd9efdaea4e748ed9dc0999756d_l3.svg "Rendered by QuickLaTeX.com")

     ![\[q_t = concat(p_t, c_{t-1})\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-0f2f592b7196d879b1df717c3b131d1b_l3.svg "Rendered by QuickLaTeX.com")

     ![\[a_t = Attention(q_t, \{h_l\}^L_{l=1})\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-5b68bd9b3f5088d020e39654d306fd14_l3.svg "Rendered by QuickLaTeX.com")

     ![\[c_t = \sum_{l}a_{t,l}h_l\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-2df9d72c1e5868116526239511c879e0_l3.svg "Rendered by QuickLaTeX.com")

     ![\[o_t = RNNs(c_t), \quad   o_t = \{f_{t.r}, ..., f_{t.r + r}\}\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-f9a8319b982b40e0c96370fdf7448657_l3.svg "Rendered by QuickLaTeX.com")

     ![\[\{o_t\}^T_{t=1}, \{a_t\}^{T}_{t=1} = Decoder(\{h_i\}^L_{i=1}; r)\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-17b89e16f87a9cf9220ca5a860fa0f57_l3.svg "Rendered by QuickLaTeX.com")

     ![\[\{f^D_k\}^K_{k=1} = reshape(\{o_t\}^T_{t=1})\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-eb7e5c09a62e87be1bb46102555a0d4b_l3.svg "Rendered by QuickLaTeX.com")

     ![\[\{f^P_k\}^K_{k=1} = Postnet((\{f^D_k\}^K_{k=1})\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-48764a4f14acaeef082af79609e0a482_l3.svg "Rendered by QuickLaTeX.com")

     ![\[L = ||f^P - y || + ||f^D - y||\quad(loss)\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-c7eb9a14722b203cf323c49011f65e4f_l3.svg "Rendered by QuickLaTeX.com")

![{y_k}<em>{k=1}^K](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-a3f838c9d312aa2c7d9eefb76dcf9f38_l3.svg "Rendered by QuickLaTeX.com") is a sequence of acoustic feature frames. ![{x_l}</em>{l=1}^L](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-ba526546fe5577a4fc9d346cfda5dc7f_l3.svg "Rendered by QuickLaTeX.com") is a sequence of characters or phonemes, from which we compute sequence of encoder outputs ![{h_l}_{l=1}^L](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-f4b81111a28b0e7887ecc1e811c0a4c7_l3.svg "Rendered by QuickLaTeX.com"). ![r](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-a05d21231b95a3cca8e9f374ca9465cb_l3.svg "Rendered by QuickLaTeX.com") is the reduction factor which defines the number of output frames per decoder step. Attention alignments, query vector and encoder output at decoder step ![t](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-b6b0f0002c48236f543c70e54ebf1a27_l3.svg "Rendered by QuickLaTeX.com") are donated by ![a_t](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-57bf9355df4f653d1b478df374aabe2d_l3.svg "Rendered by QuickLaTeX.com"), ![o_t](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-c1512f2f14776868a5bb0e30187f2d5f_l3.svg "Rendered by QuickLaTeX.com"), ![q_t](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-9a261fa7980e921f89fcad6f272df246_l3.svg "Rendered by QuickLaTeX.com"), ![o_t](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-c1512f2f14776868a5bb0e30187f2d5f_l3.svg "Rendered by QuickLaTeX.com") respectively. Also, ![o_t](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-c1512f2f14776868a5bb0e30187f2d5f_l3.svg "Rendered by QuickLaTeX.com") defines a set of output frames whose size changed by ![r](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-a05d21231b95a3cca8e9f374ca9465cb_l3.svg "Rendered by QuickLaTeX.com"). Total number of decoder steps is donated by ![T](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-b1a11ab53e5f7012604c3d23b2b42749_l3.svg "Rendered by QuickLaTeX.com").

Note that teacher forcing is applied at training. Therefore, ![K=T*r](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-683aa858c350287e2ba02e09cba2b8d9_l3.svg "Rendered by QuickLaTeX.com") at training time. However, the decoder is instructed to stop at inference by a separate network (Stopnet) which predicts a value in a range [0, 1]. If its prediction is larger than a defined threshold, the decoder stops inference.

## Double Decoder Consistency

DDC bases on two decoders working simultaneously with different reduction factors (r). One decoder (coarse) works with a large, and the other decoder (fine) works with a small reduction factor.

DDC is designed to settle the trade-off between the attention alignment and the predicted frame quality tunned by the reduction factor. In general, standard models have more robust attention performance with a larger r but due to the smoothing effect of multiple-frames prediction per iteration, final acoustic features are coarser compared to lower reduction factor models.

DDC combines these two properties at training time as it uses the coarse decoder to guide the fine decoder to preserve the attention performance without a loss of precision in acoustic features. DDC achieves this by introducing an additional loss function comparing the attention vectors of these two decoders.

For each training step, both decoders compute their relative attention vectors and the outputs. Due to the differences in their respective r values, their attention vectors are in different lengths. The coarse decoder produces a shorter vector compared to the fine decoder. In order to mitigate this, we interpolate the coarse attention vector to match the length of the fine attention vector. After having them in the same length we use a loss function to penalize the difference in the alignments. This loss is able to synchronize two decoders with respect to their alignments.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2020/06/DDC_overview-1024x814.png)

Fig2. DDC model architecture.

The two decoders take the same input from the encoder. They also compute the outputs in the same way except they use different reduction factors. The coarse decoder uses a larger reduction factor compared to the fine decoder. These two decoders are trained with separate loss functions comparing their respective outputs with the real feature frames. The only interaction between these two decoders is the attention loss applied to compare their respective attention alignments.

     ![\[\{{f^{D_f}}_k\}^K_{k=1}, \{a^f_t\}^{T_f}_{t=1} = Decoder_F(\{h_i\}^L_{i=1}; r_f)\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-5af8f6f1b8112b225eaf79ef881394f7_l3.svg "Rendered by QuickLaTeX.com")

     ![\[\{{f^{D_c}}_k\}^K_{k=1}, \{a^c_t\}^{T_c}_{t=1} = Decoder_C(\{h_i\}^L_{i=1}; r_c)\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-9b41d2500ef96df72b30cc1fd5fd0eb6_l3.svg "Rendered by QuickLaTeX.com")

     ![\[{\{a^\prime^c_t\}^{T_f}_{t=1}} = interpolate(\{a^c_t\}^{T_c}_{t=1})\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-ab9d83804a79444c69fb3443a646ecaa_l3.svg "Rendered by QuickLaTeX.com")

     ![\[L_{DDC}= ||a^F - a^C||\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-d67b7c8e986d559c868f4ae1c1da779b_l3.svg "Rendered by QuickLaTeX.com")

     ![\[L_{model} = ||f^P - y || + ||f^{D_f} - y||+ ||f^{D_c} - y|| + ||a^F - a^C||\]](https://web.archive.org/web/2020/http://erogol.com/wp-content/ql-cache/quicklatex.com-4166324d227799a90876c42f236b1379_l3.svg "Rendered by QuickLaTeX.com")

## Other Model Updates

### Batch Norm Prenet

Prenet is an important part of Tacotron like auto-regressive models. It projects model output frames before passing to the decoder. Essentially, it computes an embedding space of the feature (spectrogram) frames by which the model de-factors the distribution of upcoming frames.

I replace the original Prenet (PrenetDropout) with the one using Batch Normalization [3] (PrenetBN) after each dense layer and I remove Dropout layers. Dropout is necessary for learning attention, especially when the data quality is low. However, it causes problems at inference due to distributional differences between training and inference time. Using Batch Normalization is a good alternative. It avoids the issues of Dropout and also provides a certain level of regularization due to the noise of batch-level statistics. It also normalizes computed embedding vectors and generates a well-shaped embedding space.

### Gradual Training

I use **gradual training** scheme for the model training. I’ve introduced the **gradual training** in a [previous blog post](https://erogol.com/gradual-training-with-tacotron-for-faster-convergence/). In short, we start the model training with a larger reduction factor and gradually reduce it as the model saturates.

Gradual Training shortens the total training time significantly and yields better attention performance due to its progression from coarse to fine information levels.

### Recurrent PostNet at inference

The Postnet is the part of the network applied after the Decoder to improve the Decoder predictions before the vocoder. Its output is summed with the Decoder’s to be the final output of the model. Therefore, it predicts a residual which improves the Decoder output. So we can also apply Postnet more than one time assuming, it computes useful residual information for each time. I applied this trick only at inference and observe that, up to a certain number of iterations, it improves the performance. For my experiments, I set the number of iterations to 2.

### MB-Melgan Vocoder with Multiple Random Window Discriminator

As a vocoder, I use Multi-Band Melgan [11] generator. It is trained with Multiple Random Window Discriminator (RWD)[13] different than the original work [11] where they used Multi-Scale Melgan Discriminator (MSMD)[12].

The main difference between these two is that RWD uses audio level information and MSMD uses spectrogram level information. More specifically, RWD comprises multiple convolutional networks each takes different length audio segments with different sampling rates and performs classification whereas MSMD uses convolutional networks to perform the same classification on STFT output of the target voice signal.

In my experiments, I observed better RWD yields better results with more natural and less abberated voice.

## Related Work

Guided attention [4] uses a soft diagonal mask to force the attention alignment to be diagonal. As we do, it uses this constant mask at training time to penalize the model with an additional loss term. However, due to its constant nature, it dictates a constant prior to the model which does not always to be true, especially long sentences with various pauses. It also causes skipping in my experiments which are tried to be solved by using a windowing approach at inference time in their work.

Using multiple decoders is initially introduced by [5]. They use two decoders that run in forward and backward directions through the encoder output. The main problem with this approach is that because of the use of two decoders with identical reduction factors, it is almost 2 times slower to train compared to a vanilla model. We solve the problem by using the second decoder with a higher reduction rate. It accelerates the training significantly and also gives the user the opportunity to choose between the two decoders depending on run-time requirements. DDC also does not use any complex scheduling or multiple loss signals that aggravates the model training.

Lately, new TTS models introduced by [7][8][9][10] predicting output duration directly from the input characters. These models train a duration-predictor or use approximation algorithms to find the duration of each input character. However, as you listen to their samples, it is observed that these models lead to degraded timbre and naturalness. This is because of the indirect hard alignment produced by these models. However, models with soft-attention modules can adaptively emphasize different parts of the speech producing a more natural speech.

## Results and Experiments

### Experiment Setup

All the experiments are performed using **LJspeech** dataset [6] . I use a sampling-rate of 22050 Hz and mel-scale spectrograms as the acoustic feature. Mel-spectrograms are computed with hop-length 256, window-length 1024. Mel-spectrograms are normalized into [-4, 4]. You can see the used audio parameters below in [Mozilla TTS](https://github.com/mozilla/TTS) config format.

```python
// AUDIO PARAMETERS
    "audio":{
        // stft parameters
        "num_freq": 513,         // number of stft frequency levels. Size of the linear spectogram frame.
        "win_length": 1024,      // stft window length in ms.
        "hop_length": 256,       // stft window hop-lengh in ms.
        "frame_length_ms": null, // stft window length in ms.If null, 'win_length' is used.
        "frame_shift_ms": null,  // stft window hop-lengh in ms. If null, 'hop_length' is used.

        // Audio processing parameters
        "sample_rate": 22050,   // DATASET-RELATED: wav sample-rate. If different than the original data, it is resampled.
        "preemphasis": 0.0,     // pre-emphasis to reduce spec noise and make it more structured. If 0.0, no -pre-emphasis.
        "ref_level_db": 20,     // reference level db, theoretically 20db is the sound of air.

        // Silence trimming
        "do_trim_silence": true,// enable trimming of slience of audio as you load it. LJspeech (false), TWEB (false), Nancy (true)
        "trim_db": 60,          // threshold for timming silence. Set this according to your dataset.

        // MelSpectrogram parameters
        "num_mels": 80,         // size of the mel spec frame.
        "mel_fmin": 0.0,        // minimum freq level for mel-spec. ~50 for male and ~95 for female voices. Tune for dataset!!
        "mel_fmax": 8000.0,     // maximum freq level for mel-spec. Tune for dataset!!

        // Normalization parameters
        "signal_norm": true,    // normalize spec values. Mean-Var normalization if 'stats_path' is defined otherwise range normalization defined by the other params.
        "min_level_db": -100,   // lower bound for normalization
        "symmetric_norm": true, // move normalization to range [-1, 1]
        "max_norm": 4.0,        // scale normalization to range [-max_norm, max_norm] or [0, max_norm]
        "clip_norm": true,      // clip normalized values into the range.
    },
```

I used **Tacotron2**[2] as the base architecture with **location-sensitive attention** and applied all the model updates expressed above. The model is trained for 330k iterations and it took 5 days with a single GPU although the model seems to produce satisfying quality after only 2 days of training with DDC. I used a gradual training schedule shown below. The model starts with r=7 and batch-size 64 and gradually reduces to r=1 and batch-size 32. The coarse decoder is set r=7 for the whole training.

```python
"gradual_training": [[0, 7, 64], [1, 5, 64], [50000, 3, 32], [130000, 2, 32], [290000, 1, 32]], // [first_step, r, batch_size]
```

I trained MB-Melgan vocoder using real spectrograms up to 1.5M steps, which took 10 days on a single GPU machine. For the first 600K iterations, it is pre-trained with only the supervised loss as in [11] and than the discriminator is enabled for the rest of the training. I do not apply any learning rate schedule and I used 1e-4 for the whole training.

### DDC Attention Performance

**Fig3**. shows the validation alignments of the fine and the coarse decoders which have r=1 and r=7 respectively. We observe that two decoders show almost identical attention alignments with a slight roughness with the coarse decoder due to the interpolation.

DDC significantly shortens the time required to learn the attention alignmet. In my experiments, the model is able to align just **after 1k steps** as opposed to ~8k steps with normal location-sensitive attention.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2020/06/image-1-1024x487.png)

**Fig3**. Attention Alignments of the fine decoder (left) and interpolated the coarse (right)   
decoder.

At the inference time, we ignore the coarse decoder and use only the fine decoder. Below (**Fig.4**) depicts the model outputs and attention alignments at inference time with 4 different sentences that are not seen at training time. This shows us that the fine decoder is able to generalize successfully on novel sentences.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2020/06/image-1024x496.png)

Fig4. DDC model outputs and attention alignments at test time.

I used **50 hard-sentences** introduced by [7] to check the attention quality of the DDC model. As you see in the notebook below (Open it on Colab to listen to Griffin-Lim based voice samples), the DDC model performs without any alignment problems. It is the first model, to my knowledge, which performs flawlessly on these sentences.

### Recurrent Postnet

In Fig5. we see the average L1 difference between the real mel-spectrogram and the model prediction for each Postnet iteration. The results improve until the 3rd iteration. We also observe that some of the artifacts after the first iteration are removed by the second iteration that yields a better L1 value. Therefore, we see here how effective the iterative application of the Posnet to improve the final model predictions.

[![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2020/06/recurrent_postnet-1024x717.jpg)](https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2020/06/recurrent_postnet.jpg)

Fig5. (Click on the figure to see larger) Difference between real mel-spectrogram and the Postnet prediction for each iteration. We see that the results improve until the 3rd iteration and some of the artifacts are smoothen at the second iteration. Please pay attention to the scale differences among the figures.

## Future Work

First of all I hope this section would not be “here are the things we’ve not tried and will not try” section.

There are specifically three aspects of DDC which I like to investigate more. The first is sharing the weights between the fine and the coarse decoders to reduce the total number of model parameters and observing how the shared weights benefit from different resolutions.

The second is to measure the level of complexity required by the coarse decoder. That is, how much simpler the coarse architecture can get without performance loss.

Finally, I like to try DDC with the different model architectures.

## Conclusion

Here I tried to summarize a new method that significantly accelerates model training, provides steadfast attention alignment and provides a choice in a spectrum of quality and speed switching between the fine and the coarse decoders at inference. The user can choose depending on run-time requirements.

You can replicate all this work using Mozilla TTS. You can also see voice samples and Colab Notebooks from the links above. Let me know how it goes if you try DDC in your project.

If you like to cite this work:

*Gölge E. (2020) Solving Attention Problems of TTS models with Double Decoder Consistency. erogol.com/solving-attention-problems-of-tts-models-with-double-decoder-consistency/*

#### references

[1] Wang, Y., Skerry-Ryan, R., Stanton, D., Wu, Y., Weiss, R. J., Jaitly, N., Yang, Z., Xiao, Y., Chen, Z., Bengio, S., Le, Q., Agiomyrgiannakis, Y., Clark, R., & Saurous, R. A. (2017). *Tacotron: Towards End-to-End Speech Synthesis*. 1–10. https://doi.org/10.21437/Interspeech.2017-1452

[2] Shen, J., Pang, R., Weiss, R. J., Schuster, M., Jaitly, N., Yang, Z., Chen, Z., Zhang, Y., Wang, Y., Skerry-Ryan, R., Saurous, R. A., Agiomyrgiannakis, Y., & Wu, Y. (2017). *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions*. 2–6. http://arxiv.org/abs/1712.05884

[3] Ioffe, S., & Szegedy, C. (n.d.). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*.

[4] Tachibana, H., Uenoyama, K., & Aihara, S. (2017). *Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention*. http://arxiv.org/abs/1710.08969

[5] Zheng, Y., Wang, X., He, L., Pan, S., Soong, F. K., Wen, Z., & Tao, J. (2019). *Forward-Backward Decoding for Regularizing End-to-End TTS*. http://arxiv.org/abs/1907.09006

[6] Keith Ito, The LJ Speech Dataset (2017) https://keithito.com/LJ-Speech-Dataset/

[7] Ren, Y., Ruan, Y., Tan, X., Qin, T., Zhao, S., Zhao, Z., & Liu, T.-Y. (2019). *FastSpeech: Fast, Robust and Controllable Text to Speech*. http://arxiv.org/abs/1905.09263

[8] Kim, J., Kim, S., Kong, J., & Yoon, S. (2020). *Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search*. http://arxiv.org/abs/2005.11129

[9] Ren, Y., Hu, C., Qin, T., Zhao, S., Zhao, Z., & Liu, T.-Y. (2020). *FastSpeech 2: Fast and High-Quality End-to-End Text-to-Speech*. 1–11. http://arxiv.org/abs/2006.04558

[10] Miao, C., Liang, S., Chen, M., Ma, J., Wang, S., & Xiao, J. (2020). *Flow-TTS: A Non-Autoregressive Network for Text to Speech Based on Flow*. 7209–7213. https://doi.org/10.1109/icassp40776.2020.9054484

[11] Yang, G., Yang, S., Liu, K., Fang, P., Chen, W., & Xie, L. (2020). *Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech*. http://arxiv.org/abs/2005.05106

[12] Bińkowski, M., Donahue, J., Dieleman, S., Clark, A., Elsen, E., Casagrande, N., Cobo, L. C., & Simonyan, K. (2019). *High Fidelity Speech Synthesis with Adversarial Networks*. 1–17. http://arxiv.org/abs/1909.11646

[13] Bińkowski, M., Donahue, J., Dieleman, S., Clark, A., Elsen, E., Casagrande, N., Cobo, L. C., & Simonyan, K. (2019). *High Fidelity Speech Synthesis with Adversarial Networks*. 1–17. http://arxiv.org/abs/1909.11646

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

No related posts.