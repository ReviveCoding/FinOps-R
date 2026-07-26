# Retrieval and planner stage 1

This stage replaces the earlier rule-only evidence approximation with a governed CPU reference path. It does not claim that Qwen or BGE has been evaluated yet.

## Included

- Strict policy-card compilation with source, source date, effective interval, applicability, permitted simulator actions, and content checksum.
- Decision-time filtering for both policy cards and case artifacts.
- Deterministic BM25 with frozen tokenization, parameters, corpus hash, and tie-breaking.
- Dense cosine ranking and reciprocal-rank fusion primitives for the later BGE benchmark.
- Strict `ActionPacket/1.0` parsing and fail-closed enforcement.
- A deterministic planner that emits an action only when a retrieved card supports it and the action is permitted; otherwise it emits `HOLD_FOR_REVIEW`.
- Prompt construction that treats retrieved text as untrusted data and does not expose an external execution capability.

## Public grounding boundary

The source cards contain short paraphrases of SEC and DTCC material. They are experimental retrieval documents, not legal advice or autonomous compliance interpretations. The DTCC 9:00 p.m. cutoff card is scoped to the referenced institutional workflow and is not generalized to every trade or venue.

## Build and verify

```powershell
python scripts\build_policy_retrieval.py
python scripts\verify_retrieval_planner_stage1.py
```

The acceptance gate requires seven governed cards, seven retrieval queries, recall@3 of 1.0, mean reciprocal rank of at least 0.90, and zero future-effective or future-observable evidence results.

## Deferred to stage 2

- Download and freeze exact BGE model and tokenizer revisions.
- Benchmark BM25, BGE, and hybrid retrieval on a larger locked query set.
- Profile `Qwen/Qwen3-4B` on Linux with a pinned Transformers version and non-thinking generation.
- Create balanced ActionPacket supervised examples, perform QLoRA profiling, and train only after the zero-shot gate is recorded.
- Freeze model, tokenizer, prompt, adapter, retrieval-index, and environment manifests before locked evaluation.

The attached FUNSD, DocVQA, EDGAR, HMDA, M5, Instacart, Criteo, SEC financial-statement, and Kleister files are not inputs to this stage because they do not measure the FinOps-R post-trade retrieval or action contract.
