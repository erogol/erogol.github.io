---
layout: post
title: "Fish Audio Drama 3: a 35B MoE TTS whose codebook decoder attends over all past frames"
description: "A hybrid Gated DeltaNet + MoE backbone, a 13-layer codebook-MTP audio decoder, and RL with one learned preference reward. Open weights, about 3B active in the backbone."
tags: machine-learning tts speech-synthesis moe model-check fish-audio
minute: 9
---

<a href="/assets/images/fish-drama-3-architecture.png"><img src="/assets/images/fish-drama-3-architecture.png" width="100%" alt="Fish Audio Drama 3 architecture" /></a>

*Drama 3 architecture. Shapes and parameter counts from config.json and the safetensors headers; flow from [arXiv 2610.18010](https://arxiv.org/abs/2610.18010).*

*A hybrid Gated DeltaNet + MoE backbone, a 13-layer codebook-MTP audio decoder, and RL with one learned preference reward. Open weights, about 3B active in the backbone.*

Fish Audio released Drama 3 on 13 October 2026 with weights, a technical report ([arXiv 2610.18010](https://arxiv.org/abs/2610.18010)) and launch-day serving in [SGLang-Omni](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html). It is a 35.9B-parameter speech model initialized from Qwen3.5-35B-A3B, with about 3B active per token in the backbone. Fish Audio calls it the largest open-weight speech generation model by total parameters. The new part is the audio decoder. Each acoustic codebook gets its own full-width layer that attends causally over the whole sequence, reference audio included. With 7.8x the parameters of Fish Audio S2 (about 4.6B), it decodes one stream at 271 frames/s on an H200, against 177 for S2 in the same test.

```python
h = shared_layers(backbone_hidden, q0_next, sum_emb(q[2:10]))   # 4 shared layers
for k in range(1, 10):                                          # 9 acoustic codebooks
    h, q[k] = mtp_layer[k](h, q[k - 1])                         # causal over all past frames
```

## Overview

Parameter counts and backbone shapes come from config.json and the safetensors headers. Codec and decoder behavior come from the report.

| Property | Value |
|-|-|
| Total params | 35.90B (backbone 34.68B, audio decoder 1.22B) |
| Active params | backbone 2.95B without the input embedding table (3.47B with it), plus the 1.22B decoder, which runs in full every frame |
| Backbone | 40 layers, width 2048, 30 Gated DeltaNet + 10 gated full attention, pattern 3 + 1 |
| Full attention | 16 query heads, 2 KV heads, head dim 256, output gate, partial RoPE 0.25 |
| MoE | every layer, 256 experts of width 512, top-8, plus 1 gated shared expert |
| Precision | routed experts FP8 E4M3 with 128x128 block scales (32.2B params, 90%), rest BF16 |
| Checkpoint | 39.6 GB on disk |
| Codec | DAC-based, 44.1 kHz, 2048-sample frames = 21.5 frames/s, 10 codebooks of 4,096 |
| Audio decoder | 13 layers, 4 shared + 9 per codebook, full width |
| Languages | about 80, same coverage as S2 per the report (83 listed on the card) |
| License | Fish Audio Community License v1.0 |

## What is new vs S2

- **Backbone.** A hybrid linear-attention MoE replaces S2's dense Qwen3 model.
- **Feedback.** Only codes 0 and 1 of each frame go back into the backbone. S2 summed all ten.
- **Decoder.** A codebook-MTP decoder that attends along time replaces S2's per-frame 4-layer Fast AR.
- **One model, more tasks.** Multi-turn revision from notes, scene dubbing, editing by infilling and streaming text input are new.
- **RL.** One learned human-preference reward. No ASR, speaker-similarity or quality rewards.

## Architecture

### Codec and backbone

The S2 tokenizer is reused unchanged. It is built on the Descript Audio Codec and is fully causal, so encode and decode both stream. Code 0 is semantic, codes 1 to 9 are residual VQ levels. One frame is one sequence position. There is no audio encoder, so reference audio enters only as codec tokens.

The backbone stacks blocks of three Gated DeltaNet layers and one gated full-attention layer (layers 3, 7, ..., 39). Only the 10 full-attention layers keep a KV cache. The report gives them the job of exact retrieval over long context, for example a reference clip or a line far back in a script. Qwen3.5's native MTP layer is replaced by the audio decoder.

### Partial Codebook Fusion

At audio frame t the backbone input is (E_LM[s(q0)] + E1[q1]) / sqrt(2). The report motivates feeding back only two codes as a way to limit exposure bias from the model's own sampled fine codes. Codes 2 to 9 skip the backbone and go straight into the decoder.

### Codebook-MTP decoder

At position t the backbone predicts the next frame's semantic code q0(t+1). The decoder completes the same frame at the same position.

- **Layer.** A merge projection of the normed code embedding and normed hidden state (2d to d), then one block with gated full attention and a dense SwiGLU at width 2048, causal over the whole sequence.
- **4 shared layers.** Each injects q0(t+1) through its own table. The first also adds the sum of codes 2 to 9 of input frame t. No output head.
- **9 per-codebook layers.** Layer k reads the code k-1 just sampled and emits code k through its own head.
- **Training.** All codes teacher-forced. The decoder is one extra causal pass over the full sequence, parallel in time, on the backbone's RoPE and masks. Its loss flows into the backbone.
- **Inference.** One backbone step plus 13 single-position decoder steps per frame.

```
input at position t: text token, or audio frame t
            |
  backbone: 10 x (3 Gated DeltaNet + 1 full attn), MoE top-8 of 256
  audio input = (E_LM[q0(t)] + E1[q1(t)]) / sqrt(2)
            |-------------------------> q0(t+1)
            v                              |
  shared layers x4 <-- q0(t+1), + E(q2..q9 of frame t) at layer 1
            |
  MTP 1 -> q1(t+1) -> MTP 2 -> q2(t+1) ... MTP 9 -> q9(t+1)
            |
  q0(t+1), q1(t+1) -> backbone input at position t+1
  q0..q9 -> causal DAC decoder -> 44.1 kHz audio
```

The layers stack along the codebook axis and pass the hidden state upward, like DeepSeek-V3's MTP modules. S2's Fast AR saw the past only through one projected backbone vector. Here every codebook prediction attends directly to the fine codes of the reference and of all earlier frames. The report argues this carries reference timbre into every codebook. It has no ablation that isolates the decoder, so I take that as a design argument. The price is 13 extra KV caches. Drama 3 has 23 layers whose KV cache grows with context, against 36 for S2.

### Tokens and controls

Examples from the report's control table:

{% raw %}
```
[Reference 0]
Text: <|speaker:0|> transcript of the clip
Speech: {{audio codes}}

<|speaker:0|> Ready? <|speaker:1|> Always.
Well [sighs] fine, you win. [laughs]
Promise me <whisper>you won't tell her</whisper>.
[with bright eager excitement] We won!
[softer and warmer versus previous] ...        (later turns)
Welcome back to the show. [language: en] [audio tokens: 65]
```
{% endraw %}

- **Direction.** Point tags act at a position, scoped tags on a span, style prefixes on a line. All free natural language. Language and duration hints are never spoken.
- **Structure tokens.** 16 speaker tokens, a line separator, FIM markers for infilling, and an end-of-turn marker that carries the next text chunk as look-ahead for streaming input.
- **Editing.** The API takes `previous_takes`, so an edit is a continuation with earlier takes in context. Infilling regenerates a window from the target text and the audio tokens on both sides. Its loss always covers the window's first and last frames, so seams get supervision.

## Training

Init from the Qwen3.5 language model, new decoder. Several hundred billion tokens of quality-controlled transcribed speech plus text documents. Optimizer Muon. Examples are packed into 128K-token rows with document masking, so the lengths below are per-example limits.

| Stage | Max example length | Adds |
|-|-|-|
| Pre-train 1 | 8K tokens | text documents + multi-turn TTS conversations |
| Pre-train 2 | 8K tokens | same data, batch doubled, LR decay deferred |
| Pre-train 3 | 16K tokens | long-form recordings, streaming interleave, infilling, numbered references |
| SFT | 32K tokens | directions, dialogue, scenes |
| RL | 4,096 frames per take (rollout cap) | preference reward, decoder frozen |

- **Loss.** L = L_text + 0.5 L_sem + 0.5 L_cb, with L_cb the mean over nine codebooks. Each term is a mean over its own positions, so the text to speech balance does not drift with batch mix.
- **Reward model.** A smaller speech LM pre-trained through Stage 2, scalar head on the closing token. It reads prompt plus take tokens, so no codec decode, ASR or speaker model runs in the reward path.
- **Why one reward.** On Expresso, speaker similarity prefers the default reading over the whispered one in 99.7% of pairs. The reward model, given the direction, prefers the whisper in 62.2%.
- **Policy update.** GRPO with Dr. GRPO mean-centered advantages, no KL, no reference model. Reward is each take's win rate within its group of 8. DAPO clip 0.2 and 0.28. RL loss covers only the semantic codes and the closing token.
- **Off-policy control.** Rollouts run FP8 experts, the trainer BF16. One optimizer step per rollout batch plus rollout routing replay aim to keep the importance ratio near 1.
- **Scale.** 1,536 prompts x 8 takes = 12,288 takes per step, constant LR, few steps. Small-batch, many-step runs drifted toward reward over-optimization.

RL alone, with no transcription reward, moves Seed-TTS-Eval test-en first-take WER from 3.49 to 2.20 and flagged takes from 3.40% to 1.84%.

## Inference and serving

The report explains the speedup over S2 with less work per frame. At batch 1, Drama 3 reads about 6.0 GB of weights per frame against 15.6 GB for S2, and runs 13 sequential layer passes after the backbone step against 40.

| Read per frame | Drama 3 | S2 |
|-|-|-|
| Backbone without routed experts | 2.9 GB | 7.3 GB |
| 8 active routed experts | 1.0 GB | n/a |
| Audio decoder or Fast AR | 2.1 GB | 8.3 GB |

- **Single stream.** 271 frames/s decode rate on H200 (12.6x real time), 295 on B300 with MXFP8 experts.
- **FP8.** Serving uses the same block-scaled FP8 experts as RL rollouts, +36% per-stream rate at concurrency 32 over BF16 experts.
- **Kernel census.** A B300 frame took 4.0 ms and 1,171 kernels, but moving its data needs only 0.76 ms. About 830 kernels that move almost no data took 2.7 ms. CUDA graphs and PDL do not remove that cost, so the team deleted or fused kernels.
- **Agent-run kernel work.** Coding agents run each optimization round under an engineer-written contract. Isolated kernel speed and serving speed disagreed 9 times.
- **Prefill-decode disaggregation.** KV and DeltaNet state move inside the node to a decode GPU. On RTX PRO 6000 a PD unit carries 24 streams per GPU with no stall and a p1 per-stream rate of at least 35 frames/s, against about 10 for one GPU. Production runs this.

Model card, H200, streaming, one pre-encoded reference. RTF is request start to last chunk over audio length. The card's 166.7 frames/s at concurrency 1 differs from the report's 271 frames/s decode rate, and neither source reconciles the two.

| Concurrency | Frames/s | TTFA p50 | RTF |
|-|-|-|-|
| 1 | 166.7 | 88.9 ms | 0.131 |
| 8 | 678.1 | 153.2 ms | 0.256 |
| 32 | 1,244.7 | 304.6 ms | 0.551 |
| 64 | 1,687.3 | 450.9 ms | 0.801 |

The cookbook reports about 72% more throughput and 37% lower median TTFA at 256 streams with the vocoder on a second GPU. Server defaults cap each reference at 60 s and each prompt at 8,192 rows.

## Results

Most baselines are quoted from their papers. ElevenLabs v4, Inworld TTS-2, Soniox and Cartesia were run by Fish Audio through public APIs. Takes that fail signal checks (silence, runaway length) are regenerated before scoring.

| Benchmark | Drama 3 | Comparison |
|-|-|-|
| MiniMax multilingual, 24 languages | lower error than OmniVoice in 15, ElevenLabs v4 in 9 | lowest of all 7 systems in zh, nl, en, de, tr |
| English WER (same set) | 1.13 | ElevenLabs v4 1.56 |
| LibriSpeech-PC cloning WER / SIM | 1.85 / 0.605 (0.633 with 3 refs) | OmniVoice 1.30 / 0.729, Qwen3-TTS 1.60 / 0.704 |
| Streaming vs offline, test-en WER | 1.86 vs 1.83, SIM 0.640 both | same model |
| Infill vs Cartesia Sonic-3 API, zh WER (basic, both pasted with crossfade) | 3.69 | 8.64 |
| Infill, Ming-Freeform en insertion WER | 3.03 | CosyEdit2 1.90 |
| Step-Audio-Edit emotion accuracy, 3 revisions | 55.5% to 58.2% | Step-Audio-EditX 53.6% to 69.3% |
| Step-Audio-Edit style accuracy, 3 revisions | 76.5% to 81.8% | Step-Audio-EditX 45.6% to 63.8% |

The Step-Audio-Edit rows use a Gemini-2.5-Pro judge. Fish Audio ran Drama 3 and quoted EditX. Weak spots are Cantonese CER 22.70 (OmniVoice 17.71), Greek 3.39 and Romanian 4.94 against 0.56 and 1.60 for ElevenLabs v4. Results are mixed. Drama 3 leads on multilingual error rates against most systems, and trails specialist systems on cloning similarity and several editing metrics.

## Limits and open questions

- **No human evaluation in the report.** Task numbers come from ASR, speaker embeddings and an LLM judge. For a model tuned on human preference, MOS or A/B tests are the missing number.
- **Data is vague.** No hours, no speech to text split, no compute, no RL step count.
- **Revision is slow to move.** Emotion accuracy rises 2.7 points over three notes, against 15.7 for Step-Audio-EditX.
- **Context.** Config allows 262,144 positions, but training stops at 32K tokens.
- **Decoder RL.** Decoder weights stay frozen in RL. Whether tuning the fine codes would add quality is untested.
