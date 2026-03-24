# Chapter 5: The Citation Engine

## The Idea

Everything from Ch1-4 in a web app. This is the first time the student sees attribution as an interactive, visual, real-time experience — not just numbers in a terminal. The app is the capstone of the "math" half of the course. Everything after this is about making it faster and connecting it to the real world.

**Keep this chapter SHORT on new concepts.** It's glue code. The concepts are all from Ch1-4. The new thing is: seeing it live.

## What the Student Builds

A Flask web app with:

- **Source sidebar:** Add/edit/remove source documents. Each gets a color.
- **Budget controls:** Per-source ε input. "∞" = track without enforcing (just show the attribution colors).
- **Streaming generation:** Server-Sent Events, one token at a time.
- **Per-token influence bars:** For each generated token, colored bars showing relative influence of each source. Widest bar = most influential source for that token.
- **Hover tooltips:** Mouse over any token to see exact influence values per source.
- **Spend meters:** Cumulative ε spent per source. Budget bar fills up.
- **Exhaustion markers:** When a source's budget runs out, it fades in the sidebar and gets a marker in the token stream.

### The Artifact

`http://localhost:5001` — Load the 10 animal-room sources. Type "The hamster is in the". Hit Generate. Watch the bars pulse. Hover over tokens. See the Hamster Report dominate. Set budgets, watch sources exhaust.

## Key Ideas

1. **SSE streaming** for token-by-token generation
2. **"Track without enforcing" mode:** set ε=∞, still see attribution bars (leave-one-out only, no noise)
3. **Relative influence bars:** each token's bars sum to 100% width — shows which source matters MOST for this specific token
4. **The UX question:** how do you present ε (a Greek letter from cryptography) to people who aren't cryptographers?

## Assets Inherited (from Ch4)

- `dp_inference.py` — DP generation with per-source budgets
- `rdp_accountant.py` — RDP accounting
- `gpt2_lipschitz.py` + `lipschitz_tensor.py` — Lipschitz-tracked forward pass
- `gpt2.py` — plain forward pass (used for leave-one-out in track-only mode)
- `corpus.py` — the 10 animal documents
- Leave-one-out function from Ch2

## Assets Produced (for Ch6)

- `chat_app.py` — Flask server with `/generate` SSE endpoint
- `chat_ui.html` — the full frontend (vanilla JS, no build step)
- The complete working app that Ch6 will accelerate with MLX
- The observation: "generating is slow (~600ms/token on numpy). I wish it were faster." → Ch6

## Pacing Note

This should feel like a reward chapter. The student has done the hard math in Ch3-4. Now they get to SEE it. Keep new concepts minimal. The "wow" moment is the first time the bars animate in the browser, not a new equation.
