# Chapter 3: Private Voting — GNMax and RDP

## The Idea

Ch2's weights leak which source mattered. GNMax adds calibrated noise. RDP tracks budgets. Build the core of [autodp](https://github.com/yuxiangw/autodp) from scratch.

This chapter also upgrades the app into a real product: Flask, URLs, budget controls. And introduces the second running example.

## The Conflicting Facts Dataset (introduced here, used through Ch8)

The rooms dataset teaches mechanics. This dataset teaches WHY attribution matters.

```
Source A (Wikipedia, current): "The population of Lagos is 16.6 million (2023 estimate)"
Source B (Wikipedia, outdated): "The population of Lagos is 8.0 million (2006 census)"
Question: "What is the population of Lagos?"
```

The model picks one. The bars show which. If it picks the wrong one, you can see it. You can set a lower budget on Source B to limit its influence. This is the "holy shit" moment — attribution isn't academic when the model is citing outdated data.

## What the Student Builds

1. **GNMax** — noise on vote tallies, argmax of noisy scores, threshold filtering (~50 lines)
2. **RDP Accountant** — built step by step:
   - Naive ε: "budget runs out in 5 queries"
   - Rényi DP: "now lasts 15"
   - Multi-α optimization: "now 18"
   - Per-source accounting (weight² scaling): "low-weight sources last 30+"
3. **The real app** — upgrade from Ch1's minimal page to Flask + proper frontend (~300 lines)
   - URL input (fetch real documents)
   - Per-source ε budget controls (or ∞ for track-only)
   - Spend meters, exhaustion markers
   - Streaming (SSE)
   - The conflicting-facts demo as a preset
4. **The conflicting facts demo** — paste two Wikipedia-style sources that disagree, see which one the model believes, set budgets to limit the untrusted one

### The Artifact

`http://localhost:5001` — a real app. Paste URLs OR use the preset examples (rooms, conflicting facts). Budget controls. Colored bars. Privacy guarantees. This is the reward for three chapters of work.

## Key Ideas

1. **GNMax:** noise on vote tallies
2. **RDP composition is tighter than naive ε**
3. **Per-source cost ∝ weight²**
4. **Attribution matters when sources conflict.** The conflicting-facts example makes this visceral.
5. **This is the system that beat SOTA on HLE** — [Deep Voting](https://attribution-based-control.ai/chapter2.html)

## Assets Inherited (from Ch2)

- FTPL weighting, the minimal app, ensemble pipeline, rooms dataset

## Assets Produced (for Ch4)

- `rdp_accountant.py` — reused in Ch5 for single-model DP
- The real app (Flask + frontend) — upgraded in every later chapter
- URL fetching pipeline
- The conflicting-facts dataset
- The observation: "running N models is expensive" → Ch4
