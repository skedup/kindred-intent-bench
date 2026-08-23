# DeepSeek V4 Pro post-freeze robustness

## Outcome

DeepSeek V4 Pro did **not** rescue the frozen weak-model result under the same 512-token-per-call contract. This is a diagnostic result and has no effect on the Gemini primary verdict.

| Model | Arm | Success | Failures | HEM | P50 ms | P95 ms | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek V4 Pro | B2b | 31/112 | 81 | 0.2768 | 9394.9 | 15608.7 | 624,353 |
| DeepSeek V4 Pro | B3 | 8/112 | 104 | 0.0714 | 9961.3 | 15126.2 | 346,507 |
| DeepSeek V4 Flash (frozen reference) | B2b | 75/112 | 37 | 0.6696 | — | — | — |
| DeepSeek V4 Flash (frozen reference) | B3 | 30/112 | 82 | 0.2679 | — | — | — |

Pro minus Flash is descriptive only: B2b HEM -0.3929 and B3 HEM -0.1964. The model comparison was added after the original freeze and is not a new verdict authority.

## Within-Pro B3 minus B2b

Diagnostic verdict: **inconclusive**. Reasons: `budget_confounded`, `incomplete_cost`.

| Metric | Difference [95% paired cluster-bootstrap CI] |
|---|---:|
| HEM | -0.2054 [-0.3033, -0.1101] |
| ID intent Macro-F1 | -0.3297 [-0.4735, -0.1533] |
| Near-OOS recall | -0.3636 [-0.6364, -0.0909] |
| No-intent recall | +0.0000 [-0.1765, +0.1765] |

B2b/B3 token difference is 44.50%; pricing is not registered. The pre-registered contract therefore stops at `inconclusive` before structural attribution.

## Output-cap evidence

- B2b: 81 incomplete calls; 81 were at 511-512+ output tokens under the 512-token cap.
- B3: 104 incomplete calls; 104 were at 511-512+ output tokens under the 512-token cap.

This strongly associates the failures with output-budget exhaustion under the frozen reasoning-enabled configuration. It does not establish that Pro is intrinsically less capable than Flash; a larger-cap run would be a different, separately frozen experiment.

## Semantic-audit decision

Only 4/24 frozen semantic-audit cases produced a successful Pro Stage A formed intention; 20/24 failed before there was an intention to judge. An additional Pro human-audit sheet is therefore not added: it would mainly duplicate the output-cap failure signal rather than measure semantic preservation. The preregistered 48-row Primary + Flash audit resumes unchanged.

## Boundaries

- This post-freeze diagnostic cannot replace the preregistered Flash weak role.
- The 512-token cap is held fixed for comparability but is poorly matched to Pro output.
- Provider pricing is not registered, so the diagnostic cost gate is incomplete.
- The synthetic balanced test set is not production traffic.
- No result changes Gemini primary authority or the existing IE4 global verdict.
