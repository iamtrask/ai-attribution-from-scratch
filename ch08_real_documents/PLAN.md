# Chapter 8: Real Documents from URLs

## The Idea

Everything up to now uses pasted text snippets. Real usage means: paste a URL, get attributed answers with citations linking back to specific paragraphs.

This chapter connects the attribution engine to the real world.

## What the Student Builds

### Part 1: URL → Text → Chunks

- Fetch HTML from URL
- Extract readable text (strip nav, ads, boilerplate)
- Split into semantic chunks (paragraph-level, ~200 tokens each)
- Each chunk becomes a "source" in our attribution system

### Part 2: Chunk-Level Attribution

- Same leave-one-out, but now each chunk is a source
- A Wikipedia article with 15 paragraphs → 15 sources to attribute
- Show: paragraph 7 drove the answer, paragraph 3 contributed a little, the rest didn't matter

### Part 3: Citation Generation

- Map attribution back to source URLs and paragraph positions
- Auto-generate [1], [2], [3] inline citations
- Link each citation to the specific paragraph in the specific document

### Part 4: Hierarchical Budgets

- Per-URL ε budget that subdivides across that URL's chunks
- "This Wikipedia article gets ε=1000 total, spread across its 15 paragraphs"
- High-influence paragraphs spend budget faster

### The Artifact

The Ch5 app with a URL input field. Paste 3 Wikipedia URLs. Ask a question. Get an answer with [1][2][3] citations. Click a citation → jumps to the source paragraph. Colored bars show attribution at the chunk level.

## Key Ideas

1. **Chunking strategies:** too small = noisy attribution, too big = vague. ~200 tokens is the sweet spot.
2. **Retrieval vs attribution:** RAG retrieves relevant docs (which to include in context). Attribution tells you which ones the model actually used (which drove the output).
3. **Cross-document attribution:** when two sources say the same thing, leave-one-out may under-attribute both (removing one doesn't change the output because the other covers it). Discuss Shapley values as the theoretically correct but exponentially expensive solution.
4. **Trust scores:** attribution strength × source reliability. A highly-attributed Wikipedia source is more trustworthy than a highly-attributed random blog post.

## Assets Inherited (from Ch7)

- KV-cache-aware attribution (fast enough for 15+ chunks)
- Batched ablation
- The full app (Ch5) running on SOTA models (Ch7)
- The complete DP pipeline (Ch4)

## Assets Produced

- URL fetching + text extraction pipeline
- Chunking module
- Citation generation system
- The final artifact: a fully functional, GPU-accelerated, DP-aware citation engine that works with real web documents

## Pacing Note

This chapter is more engineering than math. The new concepts (chunking, citations) are intuitive. The student should spend most of their time using the tool, not building it. Consider providing more starter code for the URL/HTML pipeline and focusing the notebook on the attribution-specific decisions (chunk size, hierarchical budgets, cross-doc attribution).
