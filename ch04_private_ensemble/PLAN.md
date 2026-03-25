# Chapter 4: Private Ensemble — GNMax and RDP

## The Idea

Ch3's weights leak which source mattered. GNMax adds calibrated noise to the vote tally. RDP tracks per-source privacy budgets. Build the core of [autodp](https://github.com/yuxiangw/autodp) from scratch.

## What the Student Builds

1. **GNMax** — Gaussian noise on vote tallies, argmax of noisy scores (~50 lines)
2. **RDP Accountant** — built step by step, each piece motivated by failure of the previous:
   - Naive ε composition: "budget runs out in 5 queries"
   - Rényi DP: "now lasts 15 queries"
   - Multi-α optimization: "now 18 queries"
   - Per-source accounting: "source 7 lasts 30+ because its weight is low"
3. **Budget exhaustion** — sources drop out when ε is spent

### The Artifact

`ch04.py` + `rdp_accountant.py` — ~200 lines. Run 50 queries, watch budgets deplete, see each piece of the RDP accountant earn its keep.

## Key Ideas

1. **GNMax:** noise on vote tallies (simple — it's just a vector of scores)
2. **RDP composition is tighter than naive ε**
3. **Per-source cost depends on weight²** — low-weight sources are almost free
4. **Threshold filtering:** abstaining costs no privacy
5. **This is the same system that beat SOTA on HLE**

## Assets Inherited (from Ch3)

- FTPL weighting, ensemble pipeline, `corpus.py`

## Assets Produced (for Ch5)

- `rdp_accountant.py` — reused in Ch8 for single-model DP too
- The complete private ensemble pipeline that Ch5 wraps in a web UI
