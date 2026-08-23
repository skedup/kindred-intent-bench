# IE3 OpenAI adapter recovery plan

## Status and purpose

This is a post-freeze corrective rerun, not a replacement for the first frozen IE3 run. The
original OpenAI B2b and B3 artifacts remain authoritative evidence that the registered adapter
could not submit its routing schema to the OpenAI Responses API. The recovery run asks the narrower
question: after correcting that Provider compatibility defect, what automatic B2b/B3 results does
the same frozen cross-provider reference produce?

The recovery remains reference-only. It cannot change the Gemini primary verdict, tune a Prompt,
select a different model, or be combined with another Provider by voting or averaging.

## Trigger evidence

- Original bundle commit: `f0be33bbe92bb2b1b98867d9f6ce6f6dca7c807a`.
- Original B2b manifest SHA-256:
  `b1babba1badf21f52d70c396ac98ec8a5931d36bcf3ff31892905c2a4b19bb5b`.
- Original B3 manifest SHA-256:
  `4c90a966fa8958efe62cc61ebf6c9bdfabbd8c8c7bc24d037ff302535274e12f`.
- In both arms, all 112 first-stage calls succeeded and all 112 routing calls returned HTTP 400
  without billable usage metadata.
- A one-case diagnostic request using the legacy schema returned HTTP 400
  `invalid_json_schema`, with `text.format.schema` pointing to the empty
  `candidate_intents.anyOf` array branch because it lacked `items`.
- The corrected schema returned HTTP 200 `completed` on the same endpoint and model. Request IDs
  are `req_f3a40dbabd9f480f92ca11138a50ad2d` (legacy) and
  `req_605fee818c8d4caf87ff185ac871c7e9` (corrected).

## Single permitted intervention

Fix commit: `c31bda4090b47882d9811ebfb504a6482c8bd3b3`.

The OpenAI routing schema now supplies `items` on both array branches and omits unsupported
`uniqueItems`; duplicate candidates remain rejected by the local `Prediction` contract. Other
Provider schemas are unchanged.

- Legacy routing schema SHA-256:
  `f637e51031a65f3425f4fdb0c1009795325a63b3bad9bb7852d80d96aefde7e6`.
- Corrected OpenAI routing schema SHA-256:
  `f5689b961006831591c2392232ee7e511cec6db5d6043e05e40312166c446ce9`.

## Frozen factors and run policy

- Dataset freeze SHA-256:
  `e079e192837da06daa8a00ab4cf73174ee4463bdeeb9b3f0271e52929597d9aa`.
- Experiment lock SHA-256:
  `7df5f4c3e5fa4098ffdf6bb0ed68b9618c7dc01d92beaa736fb5c83c9d54f52b`.
- Keep the frozen test cases, Gold labels, taxonomy, few-shots, Prompts, `gpt-5.6-luna`, reasoning
  effort, 512-token per-call cap, call order, no-retry policy, and reject-on-schema-failure policy.
- Reuse only the 112 successful B2b call-1 and 112 successful B3 stage-A cache records from the
  original run. The changed schema identity must force exactly the formerly rejected routing calls
  back to the Provider.
- Execute B2b once and B3 once after this plan is committed. Do not tune from recovery outputs or
  resample failures.
- Write results under this directory. Do not overwrite `experiments/ie3-test`.

## Required report

Report HEM, decision Macro-F1, success/failure counts, token totals, B2b/B3 token-difference status,
latency, confusion and bad cases. Build a corrected seven-cell matrix that reuses the original five
non-OpenAI cells and points to the two recovery cells. The report must show the original adapter
failure and corrected result side by side and label the latter post-freeze.
