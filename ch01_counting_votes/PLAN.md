# Chapter 1: Counting Votes

## The Idea

A language model predicts the next word. An n-gram model does this by counting: "how many times did 'kitchen' appear after 'the cat is in the' in the training data?" Each training example is a **vote**. Attribution is trivial — just look at who voted.

## What the Student Builds

A bigram model trained on 10 source documents. Each source contributes counts. When the model predicts "kitchen," trace exactly which sources contributed that count.

### The Artifact

`ch01.py` — ~60 lines of pure Python. Train a bigram model, generate text, print a bar chart showing which source voted for each predicted token. Perfect, exact, free attribution.

## Key Ideas

1. **A language model is a voting system.** Training examples vote on what the next token should be.
2. **In an n-gram model, votes are transparent.** Look them up in the count table.
3. **Attribution = "which sources cast votes for the predicted token?"**
4. **This is perfect, exact, and free.** Savor it.

## Assets Produced (for Ch2)

- The animal-in-room corpus (10 source documents, reused in every chapter)
- The bar-chart attribution visualization (reused everywhere)
- The "source → vote → prediction" mental model
- `corpus.py` — shared data file
