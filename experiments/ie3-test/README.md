# IE3 frozen-test run bundle

This directory is the first and only run of the frozen `kir-pilot-v2-ie3` experiment over the
112-case non-blind frozen test split. The seven-run checker validates every LLM cell against the
dataset/experiment hashes, model and adapter identities, call order, actual usage, prediction
universe, output hashes, and B2a-to-B2b call-1 reuse.

| role / arm | HEM | successful predictions | total tokens | estimated cost |
|---|---:|---:|---:|---:|
| B0 | 0.5625 | 112/112 | n/a | n/a |
| B1 | 0.6786 | 112/112 | n/a | n/a |
| Gemini primary B2a | 1.0000 | 112/112 | 372,827 | $0.3020 |
| Gemini primary B2b | 0.9911 | 112/112 | 722,254 | $0.5952 |
| Gemini primary B3 | 0.9643 | 112/112 | 573,297 | $0.4844 |
| DeepSeek weak B2b | 0.6696 | 75/112 | 805,465 | not registered |
| DeepSeek weak B3 | 0.2679 | 30/112 | 470,974 | not registered |
| OpenAI reference B2b | 0.0000 | 0/112 | incomplete | not registered |
| OpenAI reference B3 | 0.0000 | 0/112 | incomplete | not registered |

The matrix is complete, but the strict comparison contract is not: Gemini and DeepSeek have
absolute B2b/B3 token differences of 20.6% and 41.5%, above the preregistered 10% maximum. OpenAI
is `usage-incomplete`. Therefore none of these runs support a strict equal-compute causal claim.

DeepSeek exposes the intended weak-model failure mode instead of hiding it: B2b has 37 failed
predictions; B3 has 82, including 65 stage-A `incomplete` failures. OpenAI one-stage and B3 stage A
both succeed for 112/112 cases, while every verifier/stage-B routing call returns HTTP 400 without
usage metadata. This strongly suggests a compatibility gap between the frozen routing response
schema and that Responses endpoint; the stored error type does not contain enough detail to assert
the exact rejected schema keyword. The frozen run is preserved without retries or post-test fixes.

This bundle contains predictions, per-call metadata, manifests, cache provenance, B3 formed-intent
logs, and `seven-run-matrix.json`. It does not contain credentials or raw Provider responses.
