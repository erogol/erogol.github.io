---
layout: post
title: Model check - KyutaiTTS - Streaming Text-to-Speech with Delayed Streams Modeling
description: An in-depth analysis of KyutaiTTS, a revolutionary streaming text-to-speech system that generates audio word-by-word with just 220ms latency using innovative Delayed Streams Modeling.
tags: machine-learning text-to-speech streaming tts voice-cloning
minute: 12
---

Text-to-Speech (TTS) systems have traditionally struggled with the trade-off between quality and latency. Most high-quality systems require processing the entire text before generating audio, while streaming approaches often sacrifice naturalness. KyutaiTTS breaks this paradigm with a novel approach that delivers high-quality, streaming audio generation with unprecedented low latency.

## The Innovation

KyutaiTTS introduces "Delayed Streams Modeling" - a sophisticated state machine approach that enables word-by-word audio generation with just **220ms latency**. This isn't just an incremental improvement; it's a fundamental rethinking of how streaming TTS should work.

### Key Capabilities

**Ultra-Low Latency Streaming**: The system generates audio incrementally as text arrives, with each word rendered in just 220ms. This makes real-time conversation possible without the awkward pauses of traditional TTS.

**Voice Cloning with Minimal Data**: Using just 10 seconds of reference audio, KyutaiTTS can clone any voice while maintaining the streaming capability. This opens up possibilities for personalized, real-time applications.

**Multi-Speaker Support**: The system handles up to 5 different voices simultaneously with automatic voice switching based on context. Perfect for dialogue, storytelling, or multi-participant scenarios.

**Production Ready**: Supports 32 concurrent users on a single GPU, making it viable for real-world applications at scale.

## Technical Architecture

### The Core Model

At its heart, KyutaiTTS is a **1.6 billion parameter transformer** that supports both English and French. The model uses the Mimi codec operating at 12.5 Hz with 16 codebooks, providing efficient audio representation while maintaining quality.

### Delayed Streams Modeling

The breakthrough comes from the state machine design that solves the fundamental challenge of streaming TTS: how do you generate coherent audio when you don't know what's coming next in the text?

The system uses special control tokens:
- `new_word`: Signals the start of a new word boundary
- `pad`: Maintains timing and rhythm during pauses
- `zero`: Controls silence and spacing between words

This approach allows the model to:
1. **Process text incrementally** without waiting for complete sentences
2. **Maintain audio quality** despite not knowing future context
3. **Preserve natural timing** and prosody in real-time generation
4. **Handle interruptions** and dynamic text changes gracefully

### Multi-Backend Implementation

KyutaiTTS isn't just a research prototype. The team has implemented multiple backends:
- **PyTorch**: Full-featured research and development
- **MLX**: Optimized for Apple Silicon
- **Rust**: High-performance production deployment

This multi-backend approach ensures the technology can be deployed across different hardware environments and use cases.

## Real-World Applications

The combination of low latency, voice cloning, and production readiness opens up numerous applications:

**Real-Time Conversation**: AI assistants can now speak as naturally as humans, without the robotic pauses that break conversational flow.

**Live Translation**: Combine with real-time translation for natural, voice-preserved cross-language communication.

**Interactive Gaming**: NPCs can speak with unique voices without pre-recording massive voice banks.

**Accessibility Tools**: Real-time reading assistance with personalized voices for users with visual impairments.

**Content Creation**: Streamers and podcasters can generate multiple character voices on-the-fly.

## The Technical Challenge

Traditional streaming TTS faces several fundamental problems:

1. **Context Dependency**: Speech prosody often depends on future words (e.g., the rise at the end of a question)
2. **Timing Precision**: Word boundaries must align perfectly with audio output
3. **Quality vs. Speed**: Previous streaming approaches sacrificed naturalness for speed
4. **Memory Management**: Streaming systems must be stateful but memory-efficient

KyutaiTTS addresses each of these through its state machine design, which maintains just enough context to generate natural prosody while being responsive to real-time text input.

## Performance Metrics

The system achieves impressive benchmarks:
- **Latency**: 220ms per word (industry leading)
- **Concurrency**: 32 users per GPU (production viable)
- **Voice Quality**: Maintains naturalness in streaming mode
- **Cloning Accuracy**: High fidelity with just 10-second references

## Research Impact

KyutaiTTS represents more than just a better TTS system - it's a new paradigm for real-time AI interaction. The delayed streams modeling approach could be applied to other sequential generation tasks where low latency is critical.

The open-source release of both code and model weights demonstrates a commitment to advancing the field. This isn't just academic research; it's production-ready technology that can be deployed today.

## Looking Forward

As conversational AI becomes more prevalent, the demand for natural, low-latency speech synthesis will only grow. KyutaiTTS sets a new standard for what's possible in streaming TTS.

The system's ability to handle multiple voices, clone new speakers quickly, and maintain conversation-like timing makes it a crucial building block for the next generation of AI interfaces.

For developers and researchers working on conversational AI, this represents a significant leap forward in making AI speech as natural and responsive as human conversation.

## Resources

- **Code Repository**: [github.com/kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- **Research Paper**: [arxiv.org/abs/2410.00037](https://arxiv.org/abs/2410.00037)  
- **Model Weights**: [huggingface.co/kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr)

---

*Originally published on my [Substack](https://erogol.substack.com/p/model-check-kyutaitts-streaming-text) - August 2, 2025*