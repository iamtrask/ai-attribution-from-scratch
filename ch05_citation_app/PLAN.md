# Chapter 5: The Citation App + URLs

## The Idea

The student has a working private ensemble (Ch2-4). Time to make it real: a web app where you paste URLs, fetch documents, spin up an ensemble, and get attributed answers with citations. **First working product — chapter 5.**

## What the Student Builds

1. **URL → Documents** — fetch HTML, extract text (~40 lines)
2. **The Web App** — Flask + vanilla JS (~300 lines)
   - URL sidebar with per-source colors
   - Budget controls (ε or ∞ for track-only)
   - Ensemble generation with streaming (SSE)
   - Per-token influence bars (relative, colored)
   - Citations: [1][2][3] linking to source URLs
   - Spend meters and exhaustion markers
3. **Track-only mode** — set ε=∞, just see attribution colors, no noise

### The Artifact

`http://localhost:5001` — paste 3 Wikipedia URLs. Ask a question. Get attributed answers with citations. This is the reward chapter.

## Key Ideas

1. **No new math.** This is glue code around Ch2-4.
2. **URLs as sources.** Real documents, not pasted snippets.
3. **Track-only mode:** attribution without enforcement.
4. **This app grows:** Ch8 adds single-model mode, Ch10 adds GPU, Ch11 adds SOTA models.

## Assets Inherited (from Ch4)

- `rdp_accountant.py`, GNMax, FTPL, budget management, `corpus.py`

## Assets Produced (for Ch6+)

- `chat_app.py` + `chat_ui.html` — extended in every later chapter
- URL fetching pipeline
- The observation: "running N models is expensive" → motivates single-model path (Ch6-8)
