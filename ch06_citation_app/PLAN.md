# Chapter 6: The Citation App + URLs

## The Idea

The student has a working private ensemble (Ch3-5). Time to make it real: a web app where you paste URLs, the system fetches documents, spins up an ensemble, and gives attributed answers with citations. This is the first "product" moment.

## What the Student Builds

### Part 1: URL → Documents (~40 lines)

- Fetch HTML from URLs
- Extract readable text (strip boilerplate)
- Each URL becomes a "source" in the ensemble

### Part 2: The Web App (~300 lines)

- Flask + vanilla JS (no build step)
- **URL input:** paste URLs in the sidebar, each gets a color
- **Budget controls:** per-source ε (or ∞ for track-only)
- **Ensemble generation:** each source goes to its own model instance, they vote
- **Streaming:** SSE, one token at a time, colored bars show which source's model won each vote
- **Citations:** [1][2][3] inline, linking back to source URLs
- **Spend meters:** cumulative ε per source, budget bars fill up

### Part 3: Track-Only Mode (~20 lines)

- Set all budgets to ∞
- No noise, no budget enforcement
- Just see the attribution colors — which source drives each token
- This is the mode most users will use first

### The Artifact

`http://localhost:5001` — paste 3 Wikipedia URLs about different animals. Ask "Which animal lives in the kitchen?" See the answer with [1] citing the cat article. Colored bars show the cat source dominating. Budget meters track spend.

## Key Ideas

1. **The ensemble pipeline from Ch3-5, now with a UI.** No new math — this is glue code.
2. **URLs as sources.** Real documents, not pasted snippets.
3. **Track-only mode:** attribution visualization without privacy enforcement. The "just show me" mode.
4. **Citations:** map ensemble weights back to source URLs.
5. **This app will grow:** Ch8 adds single-model mode, Ch10 adds GPU acceleration, Ch11 adds SOTA models.

## Assets Inherited (from Ch5)

- `rdp_accountant.py` — RDP accounting
- GNMax voting + FTPL weighting
- Budget management
- `corpus.py` and the bar-chart pattern

## Assets Produced (for Ch7+)

- `chat_app.py` + `chat_ui.html` — the web app (reused and extended in every later chapter)
- URL fetching pipeline
- The working citation engine that later chapters improve
- The observation: "this is slow because we run N separate models" → motivates single-model path (Ch7-8) and GPU acceleration (Ch10)

## Pacing Note

This is a reward chapter. The student has done the hard DP math in Ch5. Now they get to SEE it in a browser. Keep new concepts minimal. The "wow" moment is pasting a real URL and seeing attributed citations appear.
