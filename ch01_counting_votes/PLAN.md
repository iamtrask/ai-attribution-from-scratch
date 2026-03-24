# Chapter 1: Counting Votes

## The Idea

A language model predicts the next word. An n-gram model does this by counting: "how many times did 'kitchen' appear after 'the cat is in the' in the training data?" Each training example that contains that pattern is a **vote**. Attribution is trivial — just look at who voted.

## What the Student Builds

A bigram language model trained on 10 source documents (the animal-in-room corpus). Each source contributes counts to the bigram table. When the model predicts "kitchen" after "the", we trace exactly which sources contributed that count.

### The Artifact

`ch01.py` — ~60 lines of pure Python (no numpy). Trains a bigram model, generates text, and for each predicted token, prints a bar chart showing which source document voted for it. Perfect, exact, free attribution.

### Example Output

```
Prompt: "The hamster is in the"
Predicted: "bedroom"

Source attribution for "bedroom":
  Hamster Report ████████████████████ 100%
  Cat Report
  Dog Report
  ...
```

## Key Ideas

1. **A language model is a voting system.** Training examples vote on what the next token should be.
2. **In an n-gram model, votes are transparent.** You can look them up in the count table.
3. **Attribution = "which sources cast votes for the predicted token?"**
4. **This is perfect, exact, and free.** Savor it. It won't last.

## Assets Produced (for Ch2)

- The animal-in-room corpus (10 source documents, reused in every chapter)
- The bar-chart attribution visualization (text-based, pattern reused everywhere)
- The concept of "source → vote → prediction" as the mental model for attribution
- `corpus.py` — shared data file with the 10 animal documents

## Prerequisites

- Python only. No numpy. No ML knowledge needed (though Karpathy's series provides useful context).

## Pacing Note

This chapter should feel almost too easy. That's intentional. The student should finish it thinking "ok, attribution is just counting." Chapter 2 breaks that assumption.
