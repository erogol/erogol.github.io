---
layout: post
title: XTTS v2 Notes
description: XTTSv2 technical notes.
summary: XTTSv2 technical notes.
tags: machine-learning TTS coqui.ai open-source XTTS
minute: 5
---

🐸 [Github](https://github.com/coqui-ai/TTS)
🤗 [Demo](https://huggingface.co/spaces/coqui/xtts)
🤖 [Model card](https://huggingface.co/coqui/XTTS-v2)
💬 [Discord](https://discord.gg/rfw3CBFWUV)

We recently released XTTSv2 with 🐸TTS v0.20, and here I go over the relevant details of the model.

XTTSv2 uses the same backbone as XTTSv1. It is a GPT2 model that predicts audio tokens computed by a pre-trained Discrete VAE model. The core update is changing the way we condition the model on the speaker information with a Perceiver model. In our model, the Perceiver inputs a mel0spectrogram and produces 32 latent vectors representing speaker information to prefix the GPT decoder.

We observed that the Perceiver captures the speaker characteristics better than a simple encoder like Tortoise or speech prompting like Vall-E. It also provides consistent model outputs between different runs, alleviating speaker shifting between different model runs.

The Perceiver allows the use of multiple references without any length limits. This way, capturing different aspects of the target speaker and even combining other speakers to create a unique voice is possible.

We switched to a HifiGAN model to compute the final audio signal from the GPT2 outputs. Compared to standard multi-stage models like VallE and SoundStorm, it considerably reduced the inference latency.

XTTSv2 can achieve less than 150ms streaming latency with a pure Pytorch implementation on a consumer-grade GPU, significantly faster than known open-source and commercial solutions.

XTTSv2 comes with additional languages, making a total of 16 languages.

I thank our community, who helped us create new datasets and evaluate the model for their native languages.

XTTSv2 is trained with more data and better-tuned hyper-parameters, achieving better loss curves.

We primarily use publicly available datasets for the training. We intentionally did not crawl the entire web. Some may consider this foolish in a competitive environment. However, we respect everyone's work and want to maintain this respect.

This approach also helps us keep our work and models accessible to enterprise and private users without worrying about future problems due to the training data.

Overall, XTTSv2 is an improvement in every way. It offers better cloning and audio quality, additional Hungarian and Korean languages, and more expressive and natural outputs.

This new release has received a great reception from our community and feedback. Give XTTSv2 a try!

We are actively working on the new version. We plan to expand the model's capabilities and add even more languages. Our great community is helping us with this endeavor. If you want to join us, we are on Discord.

Best :)

### References and Acknowledgements

XTTSv1: https://erogol.com/2023/09/27/xtts-v1-notes
VallE: https://arxiv.org/abs/2301.02111
Tortoise: https://github.com/neonbjb/tortoise-tts
DALL-E: https://arxiv.org/abs/2102.12092
Perceiver: https://arxiv.org/abs/2103.03206
