# IE4.5A Primary Budget Attribution

> Offline diagnostic only: provider_calls=0 and the IE4 verdict is unchanged.

## Observed arm totals

| Arm | Logical calls | Incremental provider calls | Input tokens | Output tokens | Retry tokens | Total tokens | Estimated cost USD | P50 case latency ms | P95 case latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B2b | 224 | 112 | 704,427 | 17,827 | 0 | 722,254 | 0.59517150 | 2827.1 | 3540.4 |
| B3 | 224 | 224 | 555,139 | 18,158 | 0 | 573,297 | 0.48444675 | 2625.4 | 2975.3 |

B3 minus B2b is -148,957 total tokens (20.62% absolute difference), -0.11072475 USD, -171.7 ms at paired P50, and +295.1 ms at paired P95.

## Stage-pair attribution

| Position | B2b stage | B3 stage | Δ input | Δ output | Δ retry | Δ total | Δ cost USD |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | call_1_one_stage | stage_a | -178,864 | +1,606 | +0 | -177,258 | -0.12812550 |
| 2 | call_2_verifier | stage_b | +29,576 | -1,275 | +0 | +28,301 | +0.01740075 |

The observed gap is driven by the first call: Stage A uses substantially fewer input tokens than the one-stage B2b call, while Stage B is larger than the B2b verifier and partially offsets that saving. Output usage is nearly equal overall.

## Contract boundary

- The registered contract says B2b sees the full taxonomy in both calls, while B3 hides it in Stage A and exposes it in Stage B.
- API usage supports exact arm, stage, and case accounting. It cannot uniquely split input tokens into taxonomy, prompt instructions, schema, few-shot, and context components.
- The first-call difference is therefore associated with the registered architecture, but it is not a component-level causal estimate of taxonomy tokens.
- B2b counts both logical calls for fair resource accounting even though its first-call results were reused from the B2a cache; incremental provider-call counts are reported separately.
- This artifact does not modify or recompute the IE4 global verdict.

## Reproduction

```bash
uv run intentbench ie45 budget-attribution
```
