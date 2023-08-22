---
layout: post
title: Newsletter-1
description: ML notes, news, and open-source.
summary:
tags: machine-learning mlops newsletter news research
minute: 5
---

(This is the lazier version of my [substack](https://erogol.substack.com/).)

A group of robots working together in a robot factory, French, Rococo style, classical style, oil on canvas
Hi everyone. This is the first newsletter send-out…
**What's this:** This newsletter will be sharing my notes about ML in a (hopefully) more organized manner.
**Why yet another newsletter:** I read and think about ML, AI, and Tech and take notes to myself. ML is also my job. I train ML models for many different tasks and domains. Also, I used to have a blog (nuked by Digital Ocean) and I miss it. So I decided to start writing again. But I'm also lazy, so it was easier to reformat my notes and publish them.
Maybe there is enough space for one more newsletter. Who knows.
**The content:** Mostly ML research, a bit of open-source, rare tech news, and no "This is Game Changer" buzz (maybe some?).
**How regular:** I don't trust myself to be very consistent, but I'll try to prepare it every other week.
**Contact me:** You can find me on Twitter (or X whatever), LinkedIn, GitHub and see more content on my home page.
So let's dive in…
Thanks for reading Machine Learns! Subscribe for free to receive new posts and support my work.

So let's dive in

## Bits & Pieces

### No GIL Python. Maybe ??

- New proposal for Python to make GIL optional. Dream of some Python devs.
- **What’s GIL:** The GIL (global interpreter lock) is a mutex that restricts the execution of multiple threads in a Python process, allowing only one thread to execute Python bytecode at a time.
- **Why GIL**: It simplifies memory management in the Python interpreter (the most widely used implementation of Python),
- **Why not GIL**: also limits the potential parallelism that can be achieved through multi-threading, especially in CPU-bound tasks.
- **What's new:** New promising [PEP 703](https://peps.python.org/pep-0703/), suggests making the GIL optional with a new flag `--disable-gil, which would allow developers to choose whether they want to enable or disable it for their Python programs.
- **Key insights:** GIL stops achieving full parallelism in Python programs. It is why many devs don't like it. It is a bottleneck for getting the best performance for trending ML and scientific computing workflows.
- **Thinking:** Probably this will take some time to be implemented even if it is accepted. But it is a great step forward especially for AI applications where there is need for parallelism. However, Python politics are complicated 😄. Thanks to Sam Gross who authored the proposal.

### ****AI2 Dolma: 3 Trillion Token Open Corpus for Language Model Pretraining****

👉 [Data](https://huggingface.co/datasets/allenai/dolma)<br>
👉 [Blogpost](https://blog.allenai.org/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64)<br>
👉 [Datasheet](https://drive.google.com/file/d/12gOf5I5RytsD159nSP7iim_5zN31FCXq/view)<br>
👉 [Code](https://github.com/allenai/dolma)

Allen Institute for AI has released a new dataset for pre-training language models. They also open-sourced the code that they used to format this dataset. Data is under the [AI 2 ImPACT license](https://allenai.org/impact-license). I don’t know if it is a coincidence but Dolma is also one of my favorite [dishe](https://www.google.com/search?sca_esv=558576318&sxsrf=AB5stBj-WoSZ_13PBzv_QlE2b7Z3q4_I7g:1692547838450&q=dolma&tbm=isch&source=lnms&sa=X&ved=2ahUKEwi2-LOF0OuAAxVmSPEDHVsCDacQ0pQJegQICBAB&biw=1470&bih=834&dpr=2)s.

## Research

### RRHF:Align LLMS with Human Feedback using Ranking

👉 [Paper](pdf)<br>
👉 [Code](https://github.com/GanjinZero/RRHF)

- A new framework for RRHF to align LLMs with Human Feedback using Ranking Loss.
- **What’s RRHF:** This is an alternative to the famous RLHF (what makes ChatGPT). Instead of using PPO (Proximal Policy Optimization), RRHF teaches the model the ranking of the desired outputs based on human feedback.
- **How it works**
    - Obtain outputs from the LLM.
    - Score those outputs using human feedback or based on other criteria.
    - Calculate the likelihood of each output generated by the LLM.
    - Adjust the LLM so that it assigns higher probabilities to outputs with higher scores.
- **Why RRHF over RLHF**
    - Robust against hyper-parameter selection.
    - Compatible with different scoring mechanisms.
    - No need for an explicit reward model as LLM models act as a reward model.
    - Easier to scale and apply with smaller datasets.
    - Can work with any fine-tuning technique.

### Bayesian Flow Nets

👉 [Paper](https://arxiv.org/abs/2308.07037)

This is the new paper from Alex Graves’ (one of the founders of Attention) after 5 years of silence. This paper proposes a new generative model that can work with discrete and continuous data.

We'll see if it will take another 5 years for someone to create the next level of AI based on his work 😄.

- **What's BFN:** Bayesian Flow Networks (BFNs) are a type of generative model that use Bayesian inference and neural networks to produce high-quality samples. BFNs are simpler than other models, and can be used for different types of data, such as diffusion.
- **What's not:** BFN is not a drop-in replacement for Diffusion models, at least not for now. There is a need for larger-scale experiments with more real-life problems.
- **About Flow nets:** Flow models are generally underappreciated compared to Diffusion and GAN models. I think they are much easier to train than GANs, faster than Diffusion at inference, and provide exact probability values without sampling.

### Instruction Back Translation

👉 [Paper](https://arxiv.org/pdf/2308.06259.pdf)

- **What’s IBT:** IBT is a way to synthetically generate data for instruction fine-tuning. It is similar to pseudo labeling that was earlier applied successfully for Speech Recognition.
- **How it works:**
    - There is a small amount of seed data with (instruction, output) pairs and a big chunk without instructions called unlabelled data. (You can think of instructions == prompts)
    - **Augmentation:** Create synthetic instructions by a base model like LLaMA or ChatGPT for the unlabelled data.
    - Score the synthetic pairs by the same model with a specific prompt as the king model to rate the samples.
    - **Curation:** Add a subset of top-scored synthetic (instruction, output) pairs to the training set.
    - Fine-tune the model one more round with the updated dataset.
    - Repeat the process
- **Results:** Just starting with 3200 human annotated seed samples, on the Alpaca leaderboard, fine-tuned models outperform all other non-distilled instruction-following models.
- **Thinking:** If you have a limited amount of data and especially a specific domain, this approach might give you an easy way to fine-tune your model. However, I am not sure how it’d perform as the domain get wider since there is also parallel research that shows that synthetic data is likely to drift the model with further fine-tuning. This might be a problem in cases where the task is sensitive to hallucination, and factual correctness.

### SpeechX - One model for many audio tasks.

👉 [Project page](https://www.microsoft.com/en-us/research/project/speechx/)<br>
👉 [Paper](https://arxiv.org/pdf/2308.06873.pdf)

[SpeechX](https://www.microsoft.com/en-us/research/project/speechx/) is a new model from Microsoft that can perform multiple speech tasks. This is my field I go deeper about this one.

**What’s new:** SpeechX is an audio language model that is trained to do noise suppression, speaker removal, target speech extraction, zero-shot text-to-speech, and speech editing.

**How it works:**

- SpeechX follows the LLM trend (VallE, AudioLM…) and uses an LLM to model audio and handle multiple tasks based on input phonemes and task-based prompting.
- Phonemes are optional depending on the tasks.
- A task-based prompt is a special task token prepended to prompt audio tokens.
- The content of the prompt audio and the task token change based on the task.
- [phonemes, task-token, audio prompt] is the input to the model
- The model is trained to generate the target audio tokens. For TTS it is the speech and noise suppression of the clean audio.
- Audio is converted to codes by using the EnCodec model.
- The language model predicts the target audio codes and these codes are converted back to audio by using the EnCodec decoder.
- Basically, by just changing the task tokens and the orientation of the inputs, the model is able to perform all the listed tasks.
- This model is trained with 60k hours of English speech with 7k speakers like the earlier model Vall-E.

**Key insights:** It is interesting to see that we can perform a variety of tasks with the same model by just changing small bits in the input. Of course, with enough data.

**Results:** Looking at the results SpeechX is out-performing or competitive with the expert models. However, listening to the TTS samples (this is my field), I’d tell there is more to be desired in terms of audio quality which can be addressed by a better audio encoding model. (EnCoded is not the best for speech in my experience)

We also see the effect of transfer learning. The model performs better when it is initialized from a pre-trained VallE model that is initially trained for TTS.

## More reads

- ****[Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.15647)****
- **[AdaTape: Foundation model with adaptive computation and dynamic read-and-write](https://ai.googleblog.com/2023/08/adatape-foundation-model-with-adaptive.html)**
- **[Attention is off by one](https://www.evanmiller.org/attention-is-off-by-one.html)**

## Open Source

### Interspeech 2023 papers

👉 [Repo](https://github.com/DmitryRyumin/INTERSPEECH-2023-Papers)

Interspeech 2023 is happening or happened recently. It is one of the most important conferences in audio and speech tech. In the repo, they organized the papers into different topics and give links to the papers and codes. Shout out to **[𝙳𝚖𝚒𝚝𝚛𝚢 𝚁𝚢𝚞𝚖𝚒𝚗](https://github.com/DmitryRyumin)** and the contributors.

### HuggingFace's Torch Replacement in Rust (Candle)

👉 [Code](https://github.com/huggingface/candle)

Huggingface released a new library called Candle which ports Torch kernels to Rust. It is quite early in development but got much attention in a short time.

Rust is awesome but it misses a solid ML framework. Either Python will get rid of GIL or Rust will get a solid ML framework. I'm betting on Rust.

### 🐸 CoquiTTS - Text-to-Speech in >1100 languages.

👉 [Code](https://github.com/coqui-ai/TTS)<br>
👉 [Docs](https://tts.readthedocs.io/en/latest/)

🐸TTS is the library, I spend years developing it. Started when I was at Mozilla then forked it when I co-founded [coqui.ai](https://coqui.ai/). It is like Transformers but for TTS. Many different model implementations, utilities to train new models, and pre-trained models that are downloaded >1m monthly. So if you need a voice your thing, give this a try.

### Diffiner - Sony

👉 [Code](https://github.com/sony/diffiner)<br>
👉 [Paper](https://www.isca-speech.org/archive/interspeech_2023/sawata23_interspeech.html)

This is a speech enhancement model based on a diffusion model. They provide a pre-trained model and training code.

### Quivr - Second Brain with GenAI

👉 [Code](https://github.com/StanGirard/quivr)

“Quivr, your second brain, utilizes the power of GenerativeAI to store and retrieve unstructured information. Think of it as Obsidian, but turbocharged with AI capabilities.”

Basically, Quivr helps you store any media content and create a database that you can chat with. This is a very-well thought project with the Apache 2.0 license.

## Extras

### Debate on AI - Bengio, Tegmark vs Mitchell, Lecun
👉 [Youtube](https://www.youtube.com/watch?v=144uOfr4SYA)

The debate about "Existential Risk of AI". It’s a long bit but worth it if you are into the discussion.

### A Talk about Unsupervised Learning
👉 [Youtube](https://www.youtube.com/watch?v=AKMuA_TVz3A)

An interesting talk from one of the ChatGPT creators Ilya Sutskever (OpenAI). He gives his own perspective on unsupervised learning and draws some parallels with data compression, and Kolmogorov complexity for the theory of unsupervised learning.

### Baldusgate 3 🎮

I play games and probably you too. This is big if you are into role-playing games. And probably I’m playing it as you read these lines. 

Just imagine the next-gen game with NPCs connected to ChatGPT…