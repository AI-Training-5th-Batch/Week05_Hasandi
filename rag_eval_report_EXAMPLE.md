# RAG Maintenance Assistant: Evaluation Report
*Model deliverable example (RAG service plus evaluation)*

## What it does
Answers operator maintenance questions from the plant handbook (8 documents, 25
retrievable passages) and refuses questions the handbook does not cover. Exposed as
`RAGService.ask()`, which returns the answer plus a trace (sources, similarity,
latency).

## Setup
Local and CPU-only. Sentence-embedding retrieval returns the top 3 passages; a small
instruction-tuned model generates a grounded answer. The model sits behind a single
`generate()` function, so a hosted API can be swapped in without touching the rest of
the pipeline.

## Results (17-question golden set: 14 answerable, 3 out of scope)

| Metric | Score |
|---|---|
| Retrieval hit-rate (top 3) | ~1.00 |
| Answer quality (semantic similarity to reference) | ~0.97 |
| Refusal accuracy (out-of-scope questions) | ~1.00 |
| Prompt quality: v1 naive vs v2 grounded | ~0.92 vs ~0.97 |

## Guardrail
A retrieval-confidence threshold of 0.45 separates in-scope from out-of-scope
questions cleanly on this set (answerable questions retrieve at similarity ~0.57 and
above; out-of-scope questions stay at ~0.35 and below). Out-of-scope questions are
refused before any model call.

Importantly, the prompt instruction alone ("if the answer is not in the context, say
I don't know") did not reliably stop the small model from answering out-of-scope
questions. The programmatic guardrail, not the prompt, is doing the safety work. This
is the key lesson: enforce safety in code.

## Observability
Every call logs latency, prompt size, approximate tokens, and an estimated cost as if
using a hosted API. A refused call has near-zero latency and zero token cost, because
the guardrail short-circuits before generation. Increasing the number of retrieved
chunks inflates prompt size and cost, so k = 3 is a deliberate balance rather than a
default.

## When to tune retrieval, the prompt, or the guardrail
- Wrong or missing context (low retrieval hit-rate): fix retrieval (chunking,
  embedder, k, a re-ranker). A prompt change cannot fix a retrieval miss.
- Right context but a weak answer (low answer quality): fix the prompt or use a larger
  generation model.
- Confident answers to out-of-scope questions: fix the guardrail, since small models
  often ignore a refusal instruction.

## Recommendation and caveats
Ship with the grounded prompt (v2) and the 0.45 guardrail. This is a small, clean
handbook, which flatters retrieval; on real, larger documents expect a lower
hit-rate, so add a re-ranker and re-tune the threshold before production. Monitor
refusal rate and latency in the observability log.
