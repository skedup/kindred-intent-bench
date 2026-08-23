# IE3 OpenAI adapter recovery

This directory preserves a post-freeze corrective rerun of the OpenAI cross-provider reference. It
does not replace the first frozen run in [`../ie3-test`](../ie3-test/README.md), change the Gemini
primary verdict, or spend test results on Prompt/model selection. The intervention and stopping rule
were recorded in [`PLAN.md`](PLAN.md) before the two full recovery arms ran.

## What changed

The first run's B2b call-1 and B3 stage A both completed for all 112 cases, but every routing call
returned HTTP 400. A diagnostic request identified the first rejected branch: an array inside
`candidate_intents.anyOf` lacked `items`. OpenAI's Structured Outputs subset also excludes
`uniqueItems`. Fix `c31bda4` supplies `items` on both branches, removes `uniqueItems` only from the
OpenAI request schema, and keeps duplicate rejection in the local `Prediction` model. A one-case
probe changed from HTTP 400 `invalid_json_schema` to HTTP 200 `completed` before this rerun.

No case, Gold label, taxonomy, few-shot, Prompt, model, reasoning effort, output cap, retry policy, or
repair policy changed.

## Original failure and corrected result

| arm | run | successful | HEM | decision Macro-F1 | total tokens | P50 / P95 latency |
|---|---|---:|---:|---:|---:|---:|
| B2b | original frozen | 0/112 | 0.0000 | 0.0000 | incomplete | 2934.5 / 4595.4 ms |
| B2b | corrected post-freeze | 111/112 | 0.9911 | 0.9977 | 849,459 | 4855.9 / 7888.9 ms |
| B3 | original frozen | 0/112 | 0.0000 | 0.0000 | incomplete | 3406.6 / 4994.6 ms |
| B3 | corrected post-freeze | 111/112 | 0.9464 | 0.9453 | 684,173 | 5678.4 / 7917.8 ms |

Each corrected arm reused exactly 112 successful first-stage cache records and made exactly 112 new
routing calls. B3's `formed-intentions.jsonl` is byte-identical to the original frozen artifact
(`9610a61f...bc8adc0`), so stage A was not resampled. Both arms have one `incomplete` prediction on
`kir-pilot-v2-0011`; all call usage is present, and neither arm has a schema-invalid prediction.

## Within-model comparison

All differences are B3 minus B2b over the same 112 cases with 10,000 paired cluster-bootstrap draws:

| metric | point difference | 95% interval |
|---|---:|---:|
| hierarchical exact match | -0.0446 | [-0.0940, -0.0087] |
| in-scope intent Macro-F1 | -0.0341 | [-0.1250, 0.0000] |
| near-OOS recall | 0.0000 | [0.0000, 0.0000] |
| no-intent recall | -0.1176 | [-0.3529, 0.0000] |

B3 produced three known-intent false rejects and two no-intent overactions that B2b avoided. This is
useful descriptive evidence that semantic decomposition is not automatically better on a capable
model. It is not a strict equal-compute causal result: B2b used 849,459 tokens and B3 used 684,173,
an absolute difference of 19.46%, so the preregistered matrix status is `budget-confounded`. OpenAI
also remains reference-only and has no global verdict authority.

## Artifacts and offline reproduction

- [`seven-run-matrix.json`](seven-run-matrix.json) combines the unchanged original five non-OpenAI
  cells with the two corrected cells.
- [`b2b-vs-b3-bootstrap.json`](b2b-vs-b3-bootstrap.json) contains the paired intervals and source
  hashes.
- `evaluation/{b2b,b3}` contains deterministic metrics, confusion tables, and bad cases.
- `cross_provider_reference/{b2b,b3}` contains predictions, calls, manifests, and first-stage
  provenance/evidence logs.

Rebuild and verify every derived artifact without a Provider call:

```bash
uv run python scripts/build_ie3_openai_recovery.py
```
