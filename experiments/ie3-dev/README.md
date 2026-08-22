# IE3 dev Prompt selection evidence

This directory preserves the four registered Gemini dev runs used by
`configs/kir-pilot-v2-ie3-prompt-selection.yaml`. They are selection evidence from the frozen
48-case dev split, not frozen-test or production results.

The receipt selects B2b v2 and B3 v2 after one structural revision. B2b v2 has 47/48 HEM with no
failed prediction; B3 v2 has 48/48 HEM with 48 valid `formed-intentions.jsonl` records. The selected
dev runs used 309,138 and 245,110 total tokens respectively, an absolute difference of about 20.7%.
This is above the preregistered 10% strict-comparison threshold, so a formal run with the same token
relationship must be reported as `budget-confounded`; the dev scores do not waive that gate.

Each candidate directory includes predictions, per-call metadata, its run manifest, and deterministic
evaluation artifacts. B2b also includes call-1 cache provenance; B3 includes the taxonomy-free formed
intention log. Raw Provider responses and credentials are not stored.
