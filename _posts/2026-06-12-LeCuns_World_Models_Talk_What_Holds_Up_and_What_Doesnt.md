---
layout: post
title: LeCun's World Models Talk - What Holds Up and What Doesn't
description: Yann LeCun gave a talk at ETH Zürich on world models. I checked his claims against the actual papers — where he is right, where he overreaches.
tags: machine-learning world-models jepa v-jepa lecun ai research representation-learning
minute: 8
---

Yann LeCun gave a talk at ETH Zürich last week, "World Models: Enabling the next AI revolution". I went through the whole thing and checked his claims against the actual papers. Here is where he is right, where he overreaches.

The talk is his standard world-models pitch, updated with new results from 2025: LeJEPA/SIGReg, V-JEPA intuitive physics, and the launch of his startup AMI Labs. It is also, in effect, the founding pitch for that startup. Keep that in mind while reading.

## The argument in one paragraph

LLMs and autoregressive generative models are the wrong substrate for physical-world AI. Humans and animals learn world models from observation with very little data. AI should do the same: learn representations from video with joint-embedding architectures (JEPA), then plan by optimizing actions against a learned world model at inference time, MPC style, with guardrail objectives baked into the optimization. Token-space reasoning is a hack. Pixel-space generation wastes capacity. Academia should stop working on LLMs.

## Where he is right

The sample-efficiency gap is real. A teenager learns to drive in about 20 hours. Self-driving programs have millions of hours of data and there is still no Level 5 system. This is Moravec's paradox and it has not gone away. [Chollet made the same argument formally](https://arxiv.org/abs/1911.01547) years ago: intelligence is skill-acquisition efficiency, not skill itself.

Latent prediction beats pixel prediction for representations. This is the strongest technical thread in the talk and the evidence backs it:

- For frozen-encoder evaluation, joint-embedding methods (DINOv2/v3, I-JEPA, V-JEPA) consistently beat pixel-reconstruction methods like MAE.
- Latent diffusion won for the same reason: predict in representation space, not pixel space.
- DeepMind's [Physics-IQ benchmark](https://arxiv.org/abs/2502.11831) found exactly what LeCun claims: current video generators achieve visual realism with "a striking lack of physical understanding".
- V-JEPA learns intuitive physics without supervision. [Garrido et al. 2025](https://arxiv.org/abs/2502.11831) show that V-JEPA's prediction error spikes on physically impossible videos. Object permanence, solidity, gravity, all measurable through violation-of-expectation, the same protocol developmental psychologists use on infants. Works even at 115M params. This is a real, checkable result. One caveat: DeepMind's [PLATO](https://www.nature.com/articles/s41562-022-01394-8) showed something similar in 2022 with object-centric inductive biases, so "first" framing would be too strong. V-JEPA's version is cleaner because it needs no object-level supervision.
- LeJEPA is honest work. [SIGReg](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun) constrains embeddings toward an isotropic Gaussian via random 1-D projections, killing collapse with a single hyperparameter. What I liked in the talk: he flags the weak spots himself. Empirical information measures are upper bounds when you would want lower bounds ("we cross our fingers"), and the recovery theorem only holds if the latent ground truth is actually Gaussian. More speakers should caveat their own methods like this.
- Hierarchical planning is unsolved and he says so. "If you're starting a PhD on this topic, this is a good topic." Correct. HRL and the options framework exist, nothing robust and general does.

## Where he overreaches

**"Intrinsically safe, you can't jailbreak a system like this."** This is the weakest claim in the talk. Guardrails-as-cost-functions are only as good as the learned costs and the learned world model. The [specification gaming literature](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) is a catalog of optimizers exploiting exactly this kind of imperfect objective. And in the Q&A he says the guardrail heads are tiny projections "trained with a very small number of samples". A planner searching for low-energy action sequences is precisely the adversary that finds the blind spots of a small learned head. "More controllable than RLHF" would be defensible. "Can't jailbreak" is not.

**"We don't reason in token space" proves less than he thinks.** The human side is solid: [Fedorenko's work](https://www.nature.com/articles/s41586-024-07522-w) shows the brain's language network is dissociated from reasoning. Humans really don't think in language. But "not how humans do it" is not evidence for "won't work". RL-trained chain-of-thought ([DeepSeek-R1](https://www.nature.com/articles/s41586-025-09422-z) made Nature's cover for it) went from nothing to IMO gold level in two years, clearing several bars LeCun publicly set for autoregressive models. Planes don't flap wings.

**"Don't work on generative world models."** [DreamerV3](https://arxiv.org/abs/2301.04104) learns a generative latent world model with a reconstruction loss and solves Minecraft diamond collection from scratch. Genie-style models are actively used for agent training. The successful ones predict in latent space, which is closer to his JEPA position than he admits. The evidence supports "predict in representation space", not "generative is a dead end".

**"Academia has nothing to bring to LLMs."** DPO, FlashAttention, and the RLHF lineage all came out of academia, all post-scale-era. The reasonable core is that academia can't compete on pretraining scale. The categorical version is just false. And note the incentive: the under-explored area he points students toward is his own research program.

## My 2 cents

The reliability of his claims degrades with distance from his own lab. The JEPA results are good science with honest caveats. The field-level claims are mixed. The predictions about everyone else's research program have a documented track record of being wrong about LLMs, and the talk does not update on that record, while doubling as the prospectus for a company whose premise requires LLMs to plateau.

Right about how to build world models. Overconfident about why everyone else is wrong.

One more thing. "Humans learn from less data" is the part everyone nods along to, and I think it is wrong. Humans learn with better abstractions, but building those abstractions consumes an enormous data stream. Run the numbers: at roughly 10 effective frames per second over 12 waking hours, a child processes on the order of 150 billion visual frames by age 4, before you count audio, touch, and proprioception. The retina alone streams an estimated 10 Mbit/s to the brain ([Koch et al., Current Biology 2006](https://www.cell.com/current-biology/fulltext/S0960-9822(06)01521-X)). LeCun himself makes the bandwidth version of this argument when arguing against text-only training, then keeps the "humans learn from little data" framing when arguing against LLMs. You can't have both. The honest claim is that humans learn from less labeled data and fewer task-specific trials, on top of a sensory pretraining corpus that dwarfs any video dataset.