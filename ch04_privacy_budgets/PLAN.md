# Chapter 4: Noise That Proves Something — Privacy Budgets

## The Idea

Ch3 gave us the sensitivity bound: "source i can change the output by at most L." Now we add Gaussian noise with std proportional to L, and we can **prove** that the output doesn't reveal too much about any single source. This is differential privacy.

This chapter rebuilds the core of [autodp](https://github.com/yuxiangw/autodp) from scratch, piece by piece, each piece motivated by a concrete failure of the previous approach.

**The progression within this chapter:**
1. Naive Gaussian mechanism → works but wastes budget (ε adds up linearly)
2. Rényi DP → tighter composition (ε grows sub-linearly)
3. Multi-α optimization → even tighter (try many Rényi orders, pick the best)
4. Per-individual accounting → tightest (each source's cost depends on its actual norm)

Each step is motivated by running the previous version and seeing it fail: "the budget runs out too fast" → "RDP fixes that" → "but it's still loose" → "per-individual accounting fixes that."

## What the Student Builds

### Part 1: The Gaussian Mechanism (~50 lines)

- On the **linear model** from Ch2 (where L ≈ 4.7, noise is gentle)
- Add noise ~ N(0, σ² · L²) to the output
- Show: with σ=1, the output is barely perturbed
- Introduce ε: "this noise level gives us (ε, δ)-DP with ε = L²/(2σ²)"
- Run 20 queries. Naive composition: total ε = 20 × per-query ε. Budget blows up.

### Part 2: Rényi DP Accountant (~100 lines)

- The student sees: "naive composition wastes budget because it assumes worst-case correlation between queries"
- Build the RDP accountant: track ρ(α) instead of ε
- RDP-to-(ε,δ) conversion: ε = ρ(α) + log(1/δ)/(α-1)
- Same 20 queries, now RDP composition: total ε is much smaller
- "We just got 3× more queries for the same budget"

### Part 3: Multi-α Optimization (~50 lines)

- Different α values give different ε bounds for the same ρ
- Try α ∈ {1.5, 2, 3, 4, 5, 8, 10, 16, 20, 32, 50, 64, 100}
- Pick the α that minimizes ε for our specific δ
- Even tighter budget. A few more lines of code, meaningful improvement.

### Part 4: Per-Individual Accounting (~150 lines)

- Feldman & Zrnic (NeurIPS 2021): each source's RDP cost depends on its actual embedding norm, not the worst case
- Source i's cost: ρᵢ(α) = α · L² · ||xᵢ||² / (2σ²)
- Sources with small embeddings cost less → their budgets last longer
- Build `PerIndividualAccountant` class: tracks per-source RDP across queries
- Budget exhaustion: when source i exceeds ε_max, rollback and mute it

### Part 5: Apply to GPT-2 (~100 lines)

- Take the Lipschitz bound from Ch3
- Calibrate σ so that 50 tokens of generation stays within budget
- Generate text. Watch sources fade and die.
- On the linear model: noise is gentle, text is good
- On GPT-2: L is 10^83, noise is astronomical, text is garbage → "we need tighter bounds or smaller models" (motivates Ch6-7)

### The Artifact

`ch04.py` + `rdp_accountant.py` + `dp_inference.py`. The student builds autodp from scratch, each piece justified by running the previous version and seeing it fail.

## Key Ideas

1. **Differential privacy:** "the output barely changes whether your data is in or out"
2. **Gaussian mechanism:** noise ~ N(0, σ² · L²) where L = Lipschitz bound from Ch3
3. **Naive composition wastes budget:** ε_total = n × ε_per_query (linear growth)
4. **RDP composition is tighter:** ρ_total = Σρᵢ, then convert to ε (sub-linear)
5. **Multi-α optimization:** different Rényi orders give different bounds; pick the tightest
6. **Per-individual accounting:** each source has its own ρ based on its actual norm
7. **The privacy-utility tradeoff:** big L → big noise → useless output
8. **Budget exhaustion:** when ε is spent, the source is muted (embedding blended toward neutral)

## Assets Inherited (from Ch3)

- `lipschitz_tensor.py` and `gpt2_lipschitz.py` — provide the L bound
- The linear model (for gentle-noise demos)
- Leave-one-out (for comparing DP attribution vs empirical)
- `corpus.py` and visualization

## Assets Produced (for Ch5)

- `rdp_accountant.py` — the RDP accountant with multi-α optimization
- `dp_inference.py` — per-individual accounting + budget-driven muting
- The student's understanding of the full DP pipeline: sensitivity → noise → accounting → budgets
- A working DP generation system that Ch5 wraps in a web UI

## Pacing Note

**Each sub-part should run and show a concrete improvement.** Part 1: "budget runs out in 5 queries." Part 2: "now it lasts 15 queries." Part 3: "now 18." Part 4: "now some sources last 30+ because they have small norms." The student sees each piece of math earn its keep.
