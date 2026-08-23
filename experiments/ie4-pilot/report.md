# IE4 Pilot Report

> Auto analysis is complete; the frozen 48-row semantic-preservation audit is awaiting human review.

## Decision

Global verdict: **inconclusive** (`primary_decision` only). Reasons: `budget_confounded`.

The weak-model and cross-provider results are diagnostic replications. They are not averaged, voted, or allowed to replace the preregistered primary model.

## Frozen-test runs

| Role | Arm | Model | HEM | Decision Macro-F1 | ID Macro-F1 | Near-OOS recall | No-intent recall | Failures | P50 ms | P95 ms | Tokens | Cost USD |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | B0 | none | 0.562 | 0.533 | 0.751 | 0.364 | 0.588 | 0 | — | — | — | — |
| embedding | B1 | google_generative_language/gemini-embedding-001 | 0.679 | 0.579 | 0.868 | 0.455 | 0.824 | 0 | 360.9 | 608.0 | — | — |
| primary_decision | B2a | google_generative_language/gemini-3.6-flash | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 1276.7 | 1563.6 | 372,827 | 0.3020 |
| primary_decision | B2b | google_generative_language/gemini-3.6-flash | 0.991 | 0.991 | 0.990 | 1.000 | 1.000 | 0 | 2827.1 | 3540.4 | 722,254 | 0.5952 |
| primary_decision | B3 | google_generative_language/gemini-3.6-flash | 0.964 | 0.954 | 0.981 | 1.000 | 0.882 | 0 | 2625.4 | 2975.3 | 573,297 | 0.4844 |
| weak_decision | B2b | deepseek/deepseek-v4-flash | 0.670 | 0.767 | 0.843 | 0.636 | 0.588 | 37 | 5466.1 | 9144.9 | 805,465 | — |
| weak_decision | B3 | deepseek/deepseek-v4-flash | 0.268 | 0.399 | 0.446 | 0.091 | 0.353 | 82 | 5031.2 | 7870.6 | 470,974 | — |
| cross_provider_reference | B2b | openai/gpt-5.6-luna | 0.991 | 0.998 | 0.990 | 1.000 | 1.000 | 1 | 4855.9 | 7888.9 | 849,459 | — |
| cross_provider_reference | B3 | openai/gpt-5.6-luna | 0.946 | 0.945 | 0.956 | 1.000 | 0.882 | 1 | 5678.4 | 7917.8 | 684,173 | — |

## B3 - B2b paired comparisons

Intervals are 95% paired cluster-bootstrap intervals with the frozen seed and 10,000 draws.

| Role | Authority | Δ HEM [95% CI] | Δ ID Macro-F1 [95% CI] | Δ near-OOS [95% CI] | Δ no-intent [95% CI] | Token difference | Verdict | Reasons |
|---|---|---|---|---|---|---:|---|---|
| primary_decision | global | -0.027 [-0.071, +0.000] | -0.010 [-0.054, +0.000] | +0.000 [+0.000, +0.000] | -0.118 [-0.353, +0.000] | 20.6% | inconclusive | budget_confounded |
| weak_decision | diagnostic | -0.402 [-0.525, -0.273] | -0.397 [-0.603, -0.202] | -0.545 [-0.818, -0.273] | -0.235 [-0.533, +0.062] | 41.5% | inconclusive | budget_confounded, incomplete_cost |
| cross_provider_reference | diagnostic | -0.045 [-0.094, -0.009] | -0.034 [-0.125, +0.000] | +0.000 [+0.000, +0.000] | -0.118 [-0.353, +0.000] | 19.5% | inconclusive | budget_confounded, incomplete_cost |

## Audit and interpretation boundary

- The 24 frozen cases are audited for action, object, and horizon preservation on B3 Stage A for the primary and weak roles (48 rows).
- The audit is diagnostic only. It does not change the tri-state verdict and must not be used to tune the test prompts.
- A budget-confounded verdict is a contract result: it blocks causal attribution to the B3 structure even when point estimates or intervals look favorable or unfavorable.

## Limitations

- The 160-case dataset is synthetic, balanced, and non-blind; it is not production traffic.
- Only 112 frozen test cases contribute to formal test metrics; small slices retain wide intervals.
- Cached predictions make evaluation reproducible, but hosted model inference is not fully replayable.
- Weak and cross-provider pricing is not registered, so their cost gates remain incomplete.
- No production Kindred runtime, user data, or Activity authority is changed by this pilot.
