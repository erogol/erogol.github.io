---
layout: post
title: Training XTTS for Voice Conversion
description: Training XTTS for Voice Conversion
summary: Training XTTS for Voice Conversion
tags: machine-learning TTS open-source XTTS
minute: 2
---

Here I wrote a short note on my experiment with training XTTS for Voice Conversion.

Briefly here is what I tried:

- Use quantized Hubert as content encoder to extract speaker-independent content features.
- Replace the text conditioning in the XTTS with the content features.
- Train the model with speech to speech manner.

Source speech is quantized with Hubert and used to condition the model along side with the target speaker latents extracted as in the original XTTS.
Here, I did not use a parallel dataset. I used the same sample for both source and target speakers at training. Training input is `[source_spk_latents, source_content_features]` and the model predicts `source_spk_speech`.

At inference, model input is `[target_spk_latents, content_features]` and the expected output is the target speaker speech.

I used the same XTTSv2 model with the same hyper-parameters. I trained the model for 10k steps.

Results are promising. Transfer of the timber is almost perfect but the speaking style is not fully transferred from the target speaker. I think there are two possible reasons for this:

1. The content features are not fully speaker independent.
2. Speaker latents do not capture the style as good as needed.

One inference time trick I plan to try is that together with the speaker latents, use target speaker sample to prompt the model output. So the input would be [target_spk_latents, content_features, target_spk_prompt]. Here `content_features` is computed from `[target_spk_speech, source_spk_speech]`.
