---
layout: post
title: "Speech to Text Deep Learning Architectures"
description: " Small Intro"
tags: baidu deep learning deep-voice google mozilla
minute: 13
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

### Small Intro. and Background

Recently, I started at Mozilla Research. I am really excited to be a part of a small but a great team working hard to solve important ML problems with a great deal of being open-sourced. It is the first place, I've even been where people warn you to license your code to enable other to use it freely. Oxymoron by the first sight, isn't it.

Before my presence, our team already released the best known [open-sourced STT (Speech to Text) implementation](https://github.com/mozilla/DeepSpeech) based on Tensorflow. The next step is to improve the current Baidu's Deep Speech architecture and also implement a new TTS (Text to Speech) solution that complements the whole conversational AI agent. So after these two projects, anyone around the world will be able to create his own Alexa without any commercial attachment. Which is the real way to democratize AI, at least I believe it is.

Up until now, I worked on variety of data types and ML problems, except audio. Now it is time learn a it. And the first thing to do is a comprehensive literature review (like a boss). Here I like to share top notch DL architectures dealing with TTS (Text to Speech). I also invite you to our [Github repository](https://github.com/mozilla/TTS) hosting PyTorch implementation of the first version of our implementation. (We switched to PyTorch for obvious reasons). It is a work in progress and please feel free to comment and contribute.

Below I like to share my pinpoint summary of the well-known TTS papers which are by no means complete but useful to highlight important aspects of these papers. Let's start.

### **Glossary**

* **Phonemes** : units of sounds, we pronounce as we speak. Necessary since very similar words in letter might be pronounced very differently (e.g. "Rough" "Though")
* **Vocoder**: part of the system decoding from features to audio signals. Wave is used in Deep Voice at that stage.
* **Fundamental Frequency - F0:** lowest frequency of a periodic waveform describing the pitch of the sound.
* **Autoregressive Model:** Specifies a model depending linearly on its own outputs and on a parameter set which can be approximated.
* **Query, Key, Value:** Key is used by attention module to compute attention weights. Value is the vector stipulated by the attention weights to compute the module output. Query vector is the hidden state of the decoder.
* **Grapheme**: Cool way to say character.
* **Error Modes**: Sub-optimal status for the attention block where it is not able to escape.
* **Monotonic Attention**: Use only a limited scope of nodes close in time to the output step. It improves performance for TTS since there is a certain relation btw the output at time t and the input at time t. However, it is not that reasonable for translation problem since words orders might no be the same. <https://arxiv.org/pdf/1704.00784.pdf>
* **MOS**: Mean Opinion Score. Crows-source the evaluation process with native speakers. It is not easy to measure, especially for a layman.
* **Context vector**: Output of an attention module which summarizes multiple time-step output of the encoder.
* **Hann Window Function:** https://en.wikipedia.org/wiki/Window\_function#Hann\_window
* **Teacher Forcing:** Providing model's **expected** output at time t as a input at time t+1. It is controlled ground-truth feedback as a teacher does to a student.
* **Casual convolution**: Convolution which does not foresee the future units given the reference time step T which we like to predict next. In practice, it is implemented by setting right padding orientation to to normal convolution layers.

### **Deep Voice (25 Feb 2017)**

* Text to phonemes. Deterministically computed with a dictionary. Or **Seq2Seq model** to deal with the unseen words.  
  + <http://www.speech.cs.cmu.edu/cgi-bin/cmudict> CMU dictionary
* The same phoneme might hold different durations in different words. We need to predict the duration. It is sequence depended.
* Fundamental frequency for the pitch of the each phoneme. It is sequence depended.
* Frequency + Phonemes + Duration = Voice synthesis. It is done via Google's WaveNet.
* Models  
  + Segmentation Model  
    - Segment audio signal to phonemes.
    - CTC loss
    - Predict phoneme pairs due to probability mass
    - Inputs:  
      * Audio clip of “It was early spring”
      * Phonemes (label)  
        + [IH1, T, ., W, AA1, Z, ., ER1, L, IY0, ., S, P, R, IH1, NG, .]
    - *Outputs****:***
      * Pairs of Phonemes with their start time
        + [(IH1, T, 0:00), (T, ., 0:01), (., W, 0:02), (W, AA1, 0:025), (NG, ., 0:035)]
  + Fundamental Freq & Duration Models  
    - Segmentation model predictions are the labels for these models.
    - Inputs:  
      * Phonemes
    - Outputs**:**  
      * Duration, Probability, F0 for each phoneme; [H, 0.1, 25hz], ...
  + Audio Synthesizer Model
    - Simplified WaveNet
    - Inputs:  
      * Duration and F0 for phonemes + audio signals (labels)
    - Outputs:  
      * Synthesis audio signal

### **Deep Voice 2 (24 May 2017)**

* Using WaveNet as vocoder (synthesis from features to audio signal)
* Experiments with Tacotron using WaveNet
* Speaker embedding   
  + Provide speaker embedding as;  
    - initial hidden state for RNNs
    - concatenate the embedding with input features for every RNN steps.
    - multiply layer activation with the embedding
* Text => Phoneme Dictionary => Phonemes => S+Durations => Upsamling => S+ Frequency => S+Vocal => Speech
* Segmentation Model  
  + Finding phoneme boundaries
  + Predict phoneme pairs as a seq2seq problem
  + Deep Voice 1 model + BN + residual connections
* Duration Model  
  + Labels are discretized buckets of log-scale duration.
  + CRF is used to employ the dependence btw phonemes while predicting the sequence duration.
* Frequency Model   
  + Use Praat software to extract fundamental frequency
  + Input: upsampled features by the duration predicted by the duration model.
  + Output: normalized f => F0 computed by speaker specific mean and std.
* Synthesis Model
  + Inputs: F0 and phonemes computed by Duration and Frequency models.
  + Output: Synthesis speeches
* Differences from Deep Voice 1:  
  + Speaker embedding
  + Separation of duration and frequency models

### **Deep Voice 3 (20 Oct 2017)**

![](https://www.evernote.com/shard/s146/res/f5251234-6d09-445f-9388-f3036df73b4f/deep_voivce3_detailed.png)

Deep Voice 3 architecture at a high-level.

* Fully convolutional
* Text -> Mel-band spectrogram -> Audio
* WaveNet > WORLD > Griffin-Lim by the measure of MOS
* WaveNet 3x , WORLD 40X real-time in CPU
* Results are comparable to Tacotron but DeepVoice 3 faster in training ??
* Monotonic Attention
* Complex text preprocessing - A tools called Gentle used to find pauses btw words.“Either way%you should shoot/very slowly%.” - long pause after 'way', short pause after 'short'
* Joint representation of Chars and Phonemes
  + Fixing pronunciation errors by using a Phoneme dict.
  + Character embedding
  + Phoneme embedding
  + Char + Phoneme embedding
* Encode Model  
  + Given the embedding compute trainable internal representations.
  + Input:  
    - Char, Phoneme or Char + Phoneme embedding vector
  + Output:  
    - Encoded internal representation separated into Key, Value vectors.
* Attention Block  
  + Monotonic attention
  + Query, Key, Value\*
  + Fixed window Softmax
  + Positional Encoding (allows attention on sequence without RNN)
* Decoder Model  
  + Input:  
    - Attended feature vector + Previous decoder output
  + Output:  
    - next R audio feature frames
    - binary final frame prediction
* Converter  
  + Spectogram to Audio
  + Wavenet, WORLD or Griffin-Lim
* Differences from Deep Voice 2  
  + Fully convolutional
  + WORLD vocoder in deployment
  + No Separate Duration, Frequency, Segmentation models. End-2-End as it can get 🙂
* My 2 cents  
  + At a high-level, Easy and End2End but so many nitty-gritty details under the hood.  
    - Weight norm for conv layers hidden in Appendix
    - Junction of inputs; chars+phonemes
    - Speaker embedding - trained how and when ?
  + Claimed to be fast in training but I don't think fast in hyper-tunning and experimenting.
  + Like that part 🙂  
    - "Running inference with a TensorFlow graph turns out to be prohibitively expensive, averaging approximately 1 QPS 9."
  + On average all Deep Voice implementation might be hard for small teams since each paper has at least 8 people devoting fully day time on it .
  + It is also strange path that they first separate duration and frequency model on Deep Voice 2 then they completely resolve it into the whole end2end architecture. Probably Tacotron influence.

### **Tacotron (6 April 2017)**

![](https://www.evernote.com/shard/s146/res/1719309d-c03a-4ec6-9256-fc71c886df55/tacotron.png)

Tacotron architecture.

* End2End
* Faster than WaveNet
* Character sequence => Audio Spectrogram => Synthesized Audio
* Example Results : <https://google.github.io/tacotron/publications/tacotron/index.html>
* Decoder predicts r frames one at a time
* Faster training   
  + Faster convergence
* Decoder is autoregressive  
  + Training with every r-th ground truth frame of time t-1.
  + Inference by providing the r-th prediction at time t-1 for the time step t
* Griffin-Lim synthesizer
* Pre-Net
* Encoder Model
  + CBHG module
  + input: Embedding vector
  + output: encoded representation
* Attention Module  
  + input: encoded representation, decoder query
  + output: context vector
* Decoder Model  
  + input: context vector, previous r-th frame output
  + output: r spectrogram frames
* Post-Net
  + CBHG module
  + input: decoded r spectrogram frames
  + output: general audio frame can be synthesized to waveform
* Parameters:  
  + 50 ms frame length
  + 12.5 ms frame shift ?
  + 2048 points Fourier Transform
  + 24 kHz sampling for all experiments
  + Adam optimizer
  + r=2 and r=5 also works well
  + LR 0.001 -> 0.0005 -> 0.0003 -> 0.0001 after 500K -> 1M -> 2M steps
  + L1 loss for decoder and post-processing net with equal weights
  + 32 batch size with max-length padded sequences
* My 2cents:
  + Attention module might be changed with Monotonic Attention used in Deep Voice 3
  + Griffin-Lim can be switched to WaveNet for better precision, WORLD for faster results.
  + Very straight forward from a high-level to implement
  + No support for multi-speaker.
  + CBHG is we tried and it worked kind of thing.
  + However Tacotron 2 results are much better regarding the prosody and sound.

### **Tacotron 2 (16 Dec 2017)**

![](https://www.evernote.com/shard/s146/res/7e4cbe6b-4319-4bdb-a30e-52121abff259/tacotron2.png)

Tacotron 2 overview.

* Examples: <https://google.github.io/tacotron/publications/tacotron2/index.html>
* WaveNet vocoder
* Text -> Mel-spectrogram -> Audio
* In lie of linguistic features of WaveNet, Tacotron uses intermediate representation used from characters.
* Location sensitive attention
* Parameters:  
  + 50ms frames, 12,5 ms frame shift, a Hann Window function \*\*
  + "We transform the STFT magnitude to the mel-scale using an 80 channel mel filterbank spanning 125 Hz to 7.6 kHz, followed by log dynamic range compression"
* Mean Squared Error
* **Stop token prediction,** concatenation of context vector and the feedback network from the prev. time step is projected to a scalar and passed through a sigmoid function to predict the stop token.
* Mixture of Logistic dist. instead of Softmax predictions.
* 2 stage training; train feature network then WaveNet as vocoder
* Teacher forcing is used for models
* **Diff Tacotron 1**
  + Normal convolution layers over CBHG modules.
  + WaveNet vocoder
  + Location sensitive attention to achieve monotonic change of attention weights.
  + GRU -> LSTM
  + Stop token prediction
* **My 2 cents**:  
  + MOS is not a good measure since test time recordings are not meant to be separate from the training set. The exact same sentence or very similar sequences might appear in the Test set. So the success of the model is mostly based on its memorization capabilities of the large dataset.
  + No speaker embedding.
  + All the dataset is a single person speech recorded by a professional.
  + They try to reduce WaveNet computation cost by learning the internal representation by a Encode - Decoder architecture.
  + Used character embedding does not say anything about pronunciation. Voice based embedding might be useful.

### **WaveNet (16 Sep 2016)**

* Casual convolution
* Dilated convolution
* Really slow for a real-life application.

(if someone provide me a summary happy to append here. I knew this paper in advance, therefore I skip it here.)

### Voice Loop (20 July 2017)

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2018/02/VoiceLoop.jpeg)

* No need for speech text alignment due to the encoder-decoder architecture.
* **No encoding** is performed for the input text sequence. At each time step, only the corresponding embedding vector for the given character (phoneme) is used for the upper computations.
* Context computation is done by a **attentive memory buffer** in lieu of RNN. For each time step a new vector is added to buffer as the earliest is removed. The buffer keeps the long-term information by using the the autoregressive connection to compute the new time step vector.
* Using **GMM based attention model** (Graved 2013) which ensure the monotonic attention.
* Speaker embedding is used for voice adaptation. The embedding is done by a multi-class network trained on speaker identification.
* All networks in the architecture are **fully-connected**.
* **WORLD** vocoder is in use for audio synthesis.
* Auto-regressive connection is exploited with teacher-forcing which introduces a combination of the real ground-truth and the previous time-step network output. They discuss using the ground-truth forcing network to predict the next time step only.
* My two cents:
  + Memory buffer is a cool idea but I am not sure how to employ at cool-start.
  + Performing no encoding on text sequence seems out of the way when the input is a list of characters instead of phonemes (they used phonemes).
  + One important nuance is the way of teacher-forcing but needs validation on my side.

### **Datasets**

* VCTK from CSTR  
  + 109 speakers
  + ~44 hours
  + 48 kHZ sampling
  + <http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html>
* LibriSpeech  
  + <http://www.openslr.org/12/>
  + 1000 hours
  + Clean and Noisy sets
  + 2484 speakers
* LJ Speech Dataset from Keith Ito  
  + <https://keithito.com/LJ-Speech-Dataset/>
  + 1 speaker
  + ~24 hours
* [Blizzard Challenge 2012](http://www.synsig.org/index.php/Blizzard_Challenge_2012)  
  + [Toshiba Research Europe Ltd, Cambridge Research Laboratory](http://www.toshiba-europe.com/research/crl/index.html)

### **Final Words**

DeepSpeech models seem really complicated. There are many things to consider (Maybe number of people on the paper is a good indicator). I especially suggest you to read the appendixes of these papers before doing anything. On the other hand, DeepSpeech stands as a more generic solution expanding to different languages and speakers. This is where you need to decide your requirements and then choose wisely.

Tacotron models are much more simpler. This might also stems from the brevity of the papers. So might be deceiving to this end. Also it is hard to compare since they only use internal dataset to show comparative results. Nevertheless, Tacotron is my initial choice to start TTS due to its simplicity.

After all, in essence, it is easy to implement a TTS system. Main components are;

* Projecting audio into a succinct representation space (e.g. Spectrogram).
* Define the encoder architecture with less possible use of RNNs. (for the sake of run-time efficiency)
* Define a smart attention module aligning to the incremental nature of TTS problem (e.g. Monotonic Attention)
* Define the decoder with less use of RNNs.
* Find the right vocoder algorithm. It has a big impact at the final results. I warned you.

It is also breath taking to see the whole improvement in a single year. It maybe stems from the commercial importance of TTS systems with the emergence of Alexa type voice assistances.

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Harnessing Deep Neural Networks with Logic Rules](http://www.erogol.com/harnessing-deep-neural-networks-with-logic-rules/ "Harnessing Deep Neural Networks with Logic Rules")
2. [ParseNet: Looking Wider to See Better](http://www.erogol.com/parsenet-looking-wider-see-better/ "ParseNet: Looking Wider to See Better")
3. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
4. [Paper review - Understanding Deep Learning Requires Rethinking Generalization](http://www.erogol.com/paper-review-understanding-deep-learning-requires-rethinking-generalization/ "Paper review - Understanding Deep Learning Requires Rethinking Generalization")