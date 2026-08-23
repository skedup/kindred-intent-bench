# DeepSeek V4 Pro post-freeze robustness plan

## Question

Does the large failure rate and B3 regression observed with the preregistered
`deepseek-v4-flash` weak role persist when the same B2b/B3 protocol is run with
`deepseek-v4-pro`?

## Interpretation boundary

- This is a post-freeze diagnostic robustness experiment, not a replacement for the original
  seven-run matrix.
- Gemini 3.6 Flash remains the only global verdict authority.
- DeepSeek V4 Flash remains the preregistered weak-model result.
- The existing frozen test, Gold labels, taxonomy, few-shot cases, prompts, output schemas,
  two-call limit, 512-token per-call cap, no-retry policy, and reject-on-schema-error policy are
  unchanged.
- The implementation reuses the `weak_decision` runner slot only because that slot owns the
  DeepSeek adapter contract. Reports must label the result `deepseek_pro_robustness`, not treat Pro
  as the preregistered weak model.
- Pro results are reported within-model as B3 minus B2b. They are not pooled with Gemini, Flash, or
  OpenAI and cannot change the IE4 global verdict.

## Frozen intervention

The derived experiment lock is
`configs/kir-pilot-v2-ie3-deepseek-pro-robustness.yaml`. Relative to the original IE3 lock, only
these semantic fields change:

1. experiment ID;
2. weak-role selection rationale;
3. weak-role model from `deepseek-v4-flash` to `deepseek-v4-pro`.

All artifact hashes and evaluation contracts remain identical. The provider must report the exact
model identity `deepseek-v4-pro`; an alias or fallback is recorded as a contract failure.

## Execution and stopping rule

1. Run one synthetic readiness probe for the derived weak role.
2. Populate one-stage-compatible call-1 cache over all 112 frozen test cases.
3. Run one complete B2b pass and one complete B3 pass over the same 112 cases.
4. Preserve every provider, incomplete, and schema failure in the prediction universe.
5. Do not retry individual cases or change prompts/model parameters after seeing results.
6. Generate deterministic metrics, confusion, badcases, paired 10,000-draw cluster bootstrap,
   latency/token summaries, and a diagnostic conclusion.

Provider pricing is not registered in the original contract, so cost remains explicitly
incomplete. Any B2b/B3 token difference above 10% is still reported as budget-confounded.

## Outputs

- `provider-readiness.json`
- `weak_decision/{b2b,b3}` raw reduced run artifacts
- `evaluation/{b2b,b3}` deterministic evaluator bundles
- `b2b-vs-b3-bootstrap.json`
- `report.json` and `README.md`

The current 48-row primary/Flash semantic audit is paused until these automatic Pro results are
available. A Pro audit, if useful, will be an additional diagnostic sheet rather than a replacement.
