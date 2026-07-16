---
layout: post
title: I let an AI agent do architecture research on my GPT repo
description: Two weeks, ~120 experiments, six confirmed keeps, and a validation loss drop from 3.2354 to 3.1965. The interesting part was keeping it honest.
tags: machine-learning llm bla-gpt autoresearch agent training architecture
minute: 10
---

For the past two weeks, an AI agent ran autonomous architecture research on [BlaGPT](https://github.com/erogol/BlaGPT), my GPT-2-scale playground for testing LLM techniques. It ran about 100 short proxy experiments and 22 full training runs, implemented techniques from papers, kept its own ledger, and pushed everything to the repo. **Validation loss went from 3.2354 to 3.1965.**

The agent is Keche, my personal AI assistant that lives on my server and runs long unattended missions. It is not OpenClaw, Hermes, or Claude Code; it is something I built myself, and it deserves its own post later. This was its longest mission yet.

To be clear about what "found" means here: none of the winning techniques are original inventions. They come from papers and from the speedrun community, and the agent's contribution was implementing them from source, testing them under a fixed protocol, and deciding what survives at this scale. That is most of what human empirical research is too. This post covers what survived, what died, and the harness that made the numbers trustworthy, which is the part I actually care about. Getting an agent to produce plausible diffs is easy. Getting numbers you can believe from a system that ran all night unsupervised is not.

## The setup

BlaGPT is a small, hackable GPT training repo: GPT-2-scale models, FineWeb data, one config per idea. I have used it for a while to benchmark techniques from papers, and its README is a long list of "tried it, here is the loss."

The autoresearch ran in two phases on a rented 8x H100 machine. Phase one gave each experiment **exactly 600 seconds of pure training time**, with compilation and validation excluded from the clock, then recorded validation loss on a fixed shard. Cheap breadth: about a hundred experiments sweeping configs, attention variants, embeddings, and training curricula. Phase two took the survivors and trained each candidate for **a full 5100 steps** with the normal recipe: 8 GPUs, global batch 512, sequence length 1024, fixed validation tokens. Phase two is the ledger that counts, and it is the one this post reports.

The agent worked inside durable tmux sessions, committed every experiment to a branch, and appended every result to a TSV file in the repo. When a night ended, the full history sat in git: configs, diffs, logs, and every keep-or-discard decision.

## The harness

An LLM agent optimizing a metric will find every crack you leave open, not out of malice, but because **reward hacking is just gradient descent with extra steps**. The rules came down to four ideas, and **twice during the run they caught a fake record I would probably have shipped**. The details are in the graveyard section.

Frozen invariants. Some things the agent may never touch: the validation shard, the validation token count, the tokenization, the data order. Random initialization only, which means `torch.load`, `from_pretrained`, and friends are banned inside experiment diffs. A guard script greps every diff for violations before a result can be recorded, and **a violation voids the run regardless of the loss it posted**.

Keep/discard discipline. A candidate is kept only if it beats the current best. The noise floor, estimated from repeated runs of identical configs in the ledger, is roughly ±0.003, and any improvement smaller than that needs a second seed before the keep row is written. The complementary rule: **any improvement much larger than the noise floor is a bug until proven otherwise**. Inspect the diff for eval leakage before celebrating.

A separate review pass. The instance that writes a diff does not crown its own result; an independent pass reviews the diff before a keep stands. One caveat: both passes run on the same model family, so their blind spots are correlated. This is a smell test, not true independence.

Durability. The ledger is append-only, past runs are immutable, and every run directory stores its exact config and diff against the parent commit. If the machine dies, the history survives in git. This rule turned out to matter more than I expected.

None of this is exotic. It is the hygiene you would demand from a junior researcher, except that an agent needs it enforced mechanically from minute one, because nobody is watching at 3am.

## What survived

The full-training phase started from a baseline of 3.2354, the combined keeps of the proxy phase. Six changes made it through:

| id | technique (lineage) | val loss (seeds) | delta |
|---|---|---|---|
| baseline | combined proxy-phase keeps | 3.2354 | — |
| F74 | per-head attention-sink prior (GOAT, Litman & Guo 2026) | 3.2352 / 3.2298 | −0.0056 |
| F77 | GatedNorm, rank 16 (Qiu et al. 2026) | 3.2230 | −0.0068 |
| F82F84 | HybridNorm + gated attention (Zhuo et al. 2025) | 3.2128 | −0.0102 |
| F85 | learnable value residual (ResFormer lineage) | 3.2011 | −0.0117 |
| F87 | U-net long skips (modded-nanogpt lineage) | 3.1979 | −0.0032 |
| F90r | learning rate ×1.4 on the new stack | 3.1972 / 3.1965 | −0.0014 |

Two notes on reading this table. Where two numbers appear, they are independent seeds from the mandatory confirmation rerun; single-number rows beat the previous best by more than the noise floor, so no rerun was required under the protocol. And the last row is individually within noise even after confirmation: both of its seeds landed below the previous best, which is why it was kept, but if you strike it the headline gain shrinks from 0.039 to 0.037. I am comfortable with either number. Per-technique writeups with paper links live in the repo's [`techniques/`](https://github.com/erogol/BlaGPT/tree/main/techniques) directory.

The pattern surprised me. Not the attention variants. **Normalization and residual-path changes did nearly all the heavy lifting**, while attention experiments, which is where I would have placed my own bets going in, lost outright or came back within noise almost every single time.

The final row is worth a story regardless of its size. After five architecture changes, the old learning rate was no longer optimal, and the agent scheduled the sweep itself. Its first attempt silently failed because the config key never took effect. It noticed. It recorded the run as `discard_noop_config`, reran the sweep with the setting actually applied, and confirmed with a second seed before writing the keep row.

The proxy phase contributed structural wins that carried into the baseline too, the biggest being the removal of per-layer token embeddings, a component an earlier experiment had added: **159M fewer parameters and 14% more steps in the same time budget, at equal loss**. Cleaning up your own earlier keep is less glamorous than a new mechanism, and just as valuable.

Everything is in the repo: [`ar/full_results.tsv`](https://github.com/erogol/BlaGPT/blob/main/ar/full_results.tsv) holds the ledger, and `ar/runs/` holds the per-run diffs and configs.

## The graveyard

**About 70% of full runs were discards.** The Aurora optimizer was catastrophically worse, adding 0.09 to the loss. The NAG residual lost by 0.04, and the agent left a note diagnosing why: the scale signal it introduces gets erased by the final RMSNorm before reaching the logits. PoPE positional embeddings, pre-affine RMSNorm, affine-scaled attention, and block-attention residuals all came back flat or worse. One depth-softmax idea went straight to NaN. Lipschitz-constrained training never got GPU time at all, because the agent read the paper first and noticed it only matches baseline accuracy at a vacuous Lipschitz bound. Reading before implementing saved a GPU-day.

The graveyard also answers a question worth asking directly: did the agent invent anything? Its mission brief explicitly said to invent, not just reimplement, and it tried. The NAG residual readout, the depth-softmax variant that went to NaN, the block-attention residuals, and a tapered MLP width schedule were its own constructions or hybrids rather than paper implementations. Every one of them died. At this scale and budget, **published mechanisms beat the agent's original ideas without exception**, which says something about the current ceiling of agent-driven invention, or about a two-week budget, or both.

And here are the two false wins the harness rules caught. HybridNorm alone, and later an "aggregate all winning changes" run, both posted the best numbers seen up to that point. Both failed. The aggregate run posted 3.1961 on its first seed and 3.1987 on the rerun, landing behind the standing best of 3.1979. **Seed variance is exactly the size of a fake breakthrough**, and a ledger that trusted single runs would carry two false results with later experiments chasing them.

![Full-training runs](/assets/images/blagpt_autoresearch_chart_v2.png)
*Every full-training run in the ledger: keeps (green), discards (red), and the running best. The marked point beat the best and still died on its rerun. Data: [ar/full_results.tsv](https://github.com/erogol/BlaGPT/blob/main/ar/full_results.tsv), July 2026.*

The discard rate also answers a question I had going in: does a 600-second proxy predict a 5100-step run? Weakly. The proxy phase was good at killing broken ideas cheaply and terrible at ranking the survivors, which is why the full-training phase exists.

## War stories

The stories are where the "autonomous" part gets tested, so here are three from the commit log.

At one point, two research agent instances were briefly active on the same machine. They noticed each other and one stood down, leaving handoff notes in [`ar/program.md`](https://github.com/erogol/BlaGPT/blob/main/ar/program.md): which experiments it had recorded, a pitfall in a pending implementation, and a paper it recommended skipping. I did not mediate this. I found out from the commits, and the note is still in the repo.

The training machine had no GitHub credentials. That was deliberate. The agent's workaround was to package git bundles, ship them through a relay host, and push from a credentialed box, a route it built on its own the first time a push failed. It did this after every keep, because it knew the rental had an expiry date.

Then the machine died anyway. Mid-run, while I was writing this post, taking an unfinished experiment with it. **Everything committed and pushed survived. Everything else did not.**

## What this cost, and what I'd tell a skeptic

Compute came to roughly 40 hours of 8x H100 time, **about $1,300** at Lambda's on-demand rate of $3.99 per GPU-hour, proxy phase included. A failed full run costs about $25, which is what made the rerun rule affordable. My own time was a day or two on the harness, plus a handful of interventions, mostly correcting config choices the agent made without asking.

Now the limitations. 120 sequential decisions against one fixed validation shard is a selection pressure that seed reruns do not address. The fix is a one-shot evaluation on a held-out shard, which the machine died before running; it is the first job of the next one. And 3.1965 has no external reference point, since BlaGPT's recipe and token budget match nobody else's. The number belongs to no leaderboard but my own.

What the agent was genuinely better at than me: it never skipped a confirmation rerun, never left a run unrecorded, and logged its own failures as exactly what they were, including the NaN and the no-op config. Humans skip the boring parts eventually. It also worked around infrastructure the way a decent engineer would, and the one place it fell short, knowing when a research direction was exhausted, was covered by a mechanical rule that closes a family after enough straight discards.

A human researcher with the same GPU hours would have made sharper bets and fewer of them. Whether that trade favors the human depends on how much your time is worth, and mine went into writing rules once instead of babysitting runs for two weeks.

The harness, the ledger, and every diff live in the repo under [`ar/`](https://github.com/erogol/BlaGPT/tree/main/ar). If you want to point your own agent at it, the frozen rules are in `program.md`, and I would genuinely like to see whether a different agent finds a different stack.