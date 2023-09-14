---
layout: post
title: XTTS - the best open-source TTS
description: XTTS open-source release.
summary: XTTS is getting open-sourced, the best open-source text-to-speech model we’ve released so far.
tags: machine-learning TTS coqui.ai open-source
minute: 2
---

<figure>
    <img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faaf94eec-aabb-413a-b903-38a721abbf42_1024x1024" width="100%" />
    <figcaption>"Smart electric cars in French, Rococo style, classical style, oil on canvas"</figcaption>
</figure>

We recently open-sourced XTTS, the best open-source text-to-speech model we’ve released so far.

XTTS uses the latest generative AI techniques to deliver faster, higher-quality speech in 13 different languages. It is now accessible with 🐸TTS.

XTTS offers…

- Voice cloning with just a 3-second audio clip.
- Emotion and style transfer by cloning.
- Cross-language voice cloning.
- Multi-lingual speech generation.
- 24khz sampling rate.

As of now, XTTS-v1 supports the following languages: **English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, and Chinese.**

To start using XTTS, all you need to do is `pip install TTS`.

Here is a sample Python code and please [see the docs](https://tts.readthedocs.io/en/latest/index.html#) for more details.

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="/path/to/target/speaker.wav",
                language="en")
```