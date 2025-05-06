---
layout: post
title: BlaGPT notes on Softpick, Canon Layers, Parallel Transformer
description: Notes after benchmarking BlaGPT on Softpick, Canon Layers, Parallel Transformer
tags: machine-learning research deep-learning, transformer, large-language-models
minute: 3
---

Recently, I've been testing different architectural modifications in my BlaGPT benchmark to see how they affect performance. I wanted to share what I've learned about three interesting techniques: Softpick, Canon Layers, and Parallel Transformer blocks.

## [Softpick:](https://arxiv.org/abs/2504.20966) A Different Approach to Attention

Softpick offers an interesting alternative to the standard softmax function in attention blocks. It has a key difference in how it handles values:

- It allows zero values in the numerator
- Negative values can contribute to the denominator

Why does this matter? This approach prevents attention sinks - a phenomenon where attention gets stuck focusing on specific tokens (usually early ones). The math properties remain similar to regular softmax, so it doesn't disrupt the model's learning dynamics too drastically.

In my testing, Softpick resulted in a slightly worse validation loss compared to standard softmax. However, it completely eliminated attention sinks, which could be valuable for certain applications or longer context lengths.

## [Canon Layers:](https://physics.allen-zhu.com/part-4-architecture-design/part-4-1) Mixing History with Current State

Canon Layers are essentially causal 1D convolutions that combine the current hidden state with previous states. The kernel size determines how many previous states get mixed in.

This isn't entirely new - the RWKV architecture used a similar concept years ago. However, the recent Canon Layers paper demonstrates how these layers can boost performance when added to transformer blocks in various configurations.

One particularly interesting finding is that Canon Layers help models without positional encoding (NoPE) perform on par with models using RoPE (Rotary Positional Encoding).

My experiments confirmed that adding Canon Layers before Attention and MLP blocks noticeably improved model performance. There seems to be something valuable about explicitly mixing information from nearby tokens in this way.

## [Parallel Transformer Blocks:](https://arxiv.org/abs/2204.02311) Efficiency Gains

Traditional transformer blocks process data sequentially: first through attention, then through an MLP. Parallel Transformer blocks take a different approach by running these operations simultaneously and combining their outputs:

```
z = x + MLP(x) + Attention(x)
```

This design was implemented in Google's PaLM models. The main advantage is efficiency - it reduces memory usage and speeds up processing without sacrificing model capability.

Some also claim that Google's Gemini models use a similar architecture.

My tests showed that Parallel Transformer blocks delivered validation loss comparable to the baseline sequential approach, but with approximately 15% faster training time. That's a significant speedup without any performance penalty.

## Conclusion

After running these experiments, here's what I found most valuable:

1. **Canon Layers** provide a clear performance improvement when placed before Attention and MLP blocks
2. **Softpick** eliminates attention sinks but at a slight cost to validation loss
3. **Parallel Transformer blocks** maintain performance while offering a substantial 15% speedup

All the code for these experiments is available in the [BlaGPT](https://github.com/erogol/BlaGPT) repository.
