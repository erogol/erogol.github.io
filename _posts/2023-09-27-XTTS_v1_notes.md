---
layout: post
title: XTTSv1 - technical notes
description: XTTS technical details for model, training and performance.
summary: XTTS technical details for model, training and performance.
tags: machine-learning TTS coqui.ai open-source XTTS
minute: 5
---


<figure>
    <img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffe629b36-9239-401c-8e4c-dbff981f670d_512x512" width="100%" />
    <figcaption>"Smart electric cars in French, Rococo style, classical style, oil on canvas"</figcaption>
</figure>

# XTTS v1 technical notes


🎮 [XTTS Demo](https://huggingface.co/spaces/coqui/xtts)<br>
👨‍💻 [XTTS Code](https://github.com/coqui-ai/TTS)<br>
💬 [![Dicord](https://img.shields.io/discord/1037326658807533628?color=%239B59B6&label=chat%20on%20discord)](https://discord.gg/5eXr5seRrv)

XTTS is a versatile Text-to-speech model that offers natural-sounding voices in 13 different languages. One of its unique features is the ability to clone voices across languages using just a 3-second audio sample.

Currently, XTTS-v1 supports the following 13 languages: English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, and Chinese (Simplified).

TTS introduces innovative techniques that simplify cross-language voice cloning and multi-lingual speech generation. These techniques eliminate the need for extensive training data that spans countless hours and have a parallel dataset to be able to do cross-language voice cloning.

## XTTS Architecture

XTTS builds upon the recent advancements in autoregressive models, such as Tortoise, Vall-E, and Soundstorm, which are based on language models trained on discrete audio representations. XTTS utilizes a VQ-VAE model to discretize the audio into audio tokens. Subsequently, it employs a GPT model to predict these audio tokens based on the input text and speaker latents. The speaker latents are computed by a stack of self-attention layers. The output of the GPT model is passed on to a decoder model that outputs the audio signal. We employ the Tortoise methodology for XTTS-v1, which combines a diffusion model and UnivNet vocoder. This approach involves using the diffusion model to transform the GPT outputs into spectrogram frames, and then utilizing UnivNet to generate the ultimate audio signal.

## XTTS training

To convert characters to input IDs, the input text is passed through a BPE tokenizer. Additionally, a special token representing the target language is placed at the beginning of the text. As a result, the final input is structured as follows: ```[bots], [lang], t1, t2, t3 ... tn, [eots]```. ```[bots]``` represents begining of the text sequence and ```eots``` represents the end.

The speaker latents are obtained by applying a series of attention layers to an input mel-spectrogram. The speaker encoder processes the mel-spectrogram and generates a speaker latent vector for each frame of the spectrogram ```s1, s2, ..., sk```. These latent vectors are then used to condition the model on the speaker. Instead of averaging or pooling these vectors, we directly pass them to the model as a sequence. Consequently, the length of the input sequence is proportional to the duration of the speaker's audio sample. As a result, the final conditioning input is composed of ```s1, s2, ..., sk, [bots], [lang], t1, t2, t3, ..., tn, [eots]```.

We append the audio tokens to the input sequence above to create each of the training samples. So each sample becomes ```s1, s2, ..., sk, [bots], [lang], t1, t2, t3, ..., tn, [eots], [boas], a1, a2, ..., a_l, [eoas]```. ```[boas]``` == beginning of audio sequence and ```[eoas]```...

XTTS-v1 can be trained in 3 stages. First, we train the VQVAE model, then the GPT model, and finally an audio decoder.

XTTS is trained with 16k hours of data mostly consisting of public datasets. We use all the datasets from the beginning and balance the data batches by language. In order to compute speaker latents, we used audio segments ranging from 3 to 6 seconds in length.

In our training process, we utilize a learning rate of 1e-4 in combination with the AdamW optimizer. Additionally, we employ a step-wise scheduling approach to decrease the learning rate to 1e-5 after 750k steps. The entire training process consists of approximately 1 million steps. As a result, the final model comprises approximately 750 million parameters.

## XTTS v1

XTTS utilizes various unique techniques for;

- Better cloning and speaker consistency
- Cross-language cloning without a language parallel voice dataset
- Learning languages without excessive amounts or training datasets.

In XTTS-v1, the speaker latents are learned from a portion of the ground-truth audio. However, when the segment is used directly, the model tends to cheat by simply copying it to the output. To avoid this, we divide the segment into smaller chunks and shuffle them before inputting them to the speaker encoder. The encoder then calculates a speaker latent for each frame of the spectrogram, which we use to condition the GPT model based on the speaker information

XTTS combines Tortoise and Vall-E to be able to clone the voice and keep it consistent between each run. Tortoise is very good at cloning however, you are likely to get different voices for each run since it only relies on a single vector. Therefore, XTTS uses a sequence of latent vectors that transfers more information about the speaker and leads to better cloning and consistency between runs.

Vall-E had exceptional cloning abilities. However, in order to achieve cross-language voice cloning with Vall-E-X, both the reference audio transcript and a parallel dataset are necessary. On the other hand, our method for conditioning XTTS does not rely on transcripts, and despite not using a parallel dataset, the model is capable of seamlessly transferring voices across different languages.

It is important to mention that the use of language tokens is essential in reducing the data and training time required to learn a language. These tokens act as a strong indicator at the beginning of the generation process and assist the model in being conditioned on the specific language. Language tokens proved to be beneficial in enabling XTTS to learn a language using just 150 hours long dataset, which is considerably smaller when compared to other auto-regressive TTS models.


## Performance Notes

- The model outputs 24khz audio.
- It is often difficult to pronounce acronyms correctly. It is more likely to be successful if you separate the individual characters by space.
- Numbers are better handled by converting them to text.
- Sometimes the reference speaker audio might be copied in the output, especially when the input text is the same or very similar to the text of the reference.
- Output quality is strongly tied to the reference audio quality. A good reference is around 4-6 secs, clean from anything but speech.
- Using a cartoonish voice for audio references can potentially cause the model to fail, as these references stand out as outliers in relation to the training dataset.
- Model context is limited to 604 audio and 402 text tokens. 604 audio tokens correspond to ~12 seconds of audio. We recommend keeping the input text under 250 characters.
- We recommend using a GPU with at least 8GB of memory.

## Future plans

- Our team has plans to make the training and fine-tuning code publicly available in the future. However, we don't have a specific timeline for this release at the moment.
- In the next update, we will introduce streaming support with a latency of less than 0.6 seconds.
- We are actively expanding the range of languages supported by our system. If you're interested in contributing, please don't hesitate to contact us. The upcoming release is likely to include Japanese, Korean, and Hungarian.
- We are currently developing YTTS (next generation XTTS) to improve the speed and efficiency of inference and training. While we don't have a timetable, we welcome any assistance you may be able to provide..

## Using XTTS

To start using XTTS, all you need to do is pip install TTS.

Here is a sample Python code. Please see [the docs](https://tts.readthedocs.io/en/latest/models/xtts.html) for more details.

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=True)

python# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="/path/to/target/speaker.wav",
                language="en")
```


## Acknowledgements
Big thanks to the Coqui team and the community for their work and support. James Betker for [Tortoise](https://github.com/neonbjb/tortoise-tts) that showed all the
the right way to do TTS. The HuggingFace team for their amazing work and support with the XTTS release. All the people and organizations behind public voice datasets, especially Common Voice.

<br>
<br>

## Play with XTTS

<iframe
	src="https://coqui-xtts.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
