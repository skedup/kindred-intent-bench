# Kindred Activity Grounding Audit

> Status: `runtime captured / taxonomy v2 authored / 160 reviews complete / v2 adjudication pending`
>
> Audit date: 2026-08-22

## 1. Authority and scope

The authority for this audit is the Kindred distribution used by the running Mac Heart, not the public source checkout.
The probe imported the same installed Python distribution used by the running process and resolved Activity and Action
packages through Kindred's production loaders.

The captured authority is:

- Kindred distribution `0.3.1`;
- internal source commit `0d6f9aeddfc18bf001c99f4dc3f6c2d2ad8225e7`;
- private overlay SHA-256 `71604b8d131ef115c513faeb249143bd176a2a87a567f48f6fa472a359d8d641`;
- 8 strictly loadable Activities and 15 strictly loadable Actions.

The machine-readable snapshot is
[`configs/kindred-activity-grounding-v1.yaml`](../configs/kindred-activity-grounding-v1.yaml). It records each loaded
manifest/SKILL package path and content hash. Workbench evaluation remains independent from Kindred at runtime; this is a
maintainer-triggered capture followed by a static, versioned snapshot.

This is an intent-contract audit, not a full end-to-end execution test. Manifest, SKILL, Action closure and the installed
catalog determine the routable Activity semantics. Provider health and task success remain outside the current benchmark.

## 2. Mapping unit

The classifier predicts a high-level Activity goal, not the next atomic Action. An incidental unsupported step does not by
itself make an otherwise supported Activity OOS. For example, delivery can be a means to an at-home meal, while the
classified goal remains `eat_at_home`. Conversely, when the unsupported behavior is itself the primary action—editing an
existing image, running, private messaging, or viewing an online exhibition—the intent is OOS.

This distinction is required to keep Activity recognition separate from Action planning and capability execution.

## 3. Grounded catalog

| Activity | Loaded Actions | Grounded core boundary |
|---|---|---|
| `create_picture` | `draw` | Form a new local visual work from a concrete mental image; existing-image editing is not included. |
| `dine_out` | preparation, `walk/ride`, `eat` | Eating or drinking at an outside food venue is the goal; venue-only errands are not included. |
| `eat_at_home` | `walk/ride`, `prepare_food`, `eat` | Return or remain home to prepare and eat a meal; food-information tasks without eating are not included. |
| `play_xiaohongshu` | `use_xhs`, `compose`, `draw`, `publish_xhs` | Browse, search, public interaction, creation and public publishing are supported; private messages are not. |
| `reach_out_to_user` | `compose`, `send` | Proactively message the current user; sending to arbitrary third parties is not included. |
| `rest` | `walk/ride`, preparation, `sleep` | Stop to recover through lying down, resting or sleep; a different relaxation activity is not automatically rest. |
| `take_a_walk` | preparation, `walk` | Walking itself is the nearby outdoor experience; running, cycling and destination tasks are not included. |
| `visit_cultural_place` | preparation, `walk/ride`, `explore_place` | Reach and experience a real cultural place; online viewing and doorway errands are not included. |

The grounded experiment taxonomy is
[`configs/kindred-activity-intents-v2.yaml`](../configs/kindred-activity-intents-v2.yaml). It is deliberately authored
from the runtime contract instead of copying manifest descriptions verbatim: manifests define the product, while the
taxonomy makes its classification boundaries and examples explicit.

## 4. Existing data impact

All 160 candidate contexts were reviewed against the grounded catalog, with explicit contract review for all 80 in-scope
and 16 near-OOS cases. The machine-readable decision record is
[`data/kir-pilot-v1/grounding-impact.yaml`](../data/kir-pilot-v1/grounding-impact.yaml).

The principal result is:

- 159 draft hierarchical labels can remain;
- `draft-kir-pilot-0087` must change from `oos` to `in_scope/play_xiaohongshu`;
- its `near_oos`, sibling and hard-negative relationships must be removed or replaced;
- the XHS positive slice lacks publish and public-interaction coverage;
- one genuine XHS near-OOS replacement is required to keep two boundary probes per Activity.

For the seven disagreements previously routed to `product_policy_required`, the runtime contract resolves every one:

| Candidate | Grounded result | What the disagreement means |
|---|---|---|
| `0087` public XHS post | `in_scope/play_xiaohongshu` | Old taxonomy and draft were stale; the human label was correct. |
| `0081`, `0082` existing-image editing | `oos` | `create_picture` only exposes new-work `draw`; human labels need revision. |
| `0094` running | `oos` | `take_a_walk` requires walking; the human label needs revision. |
| `0088` XHS private message | `oos` | `use_xhs` explicitly excludes private messages; the human label needs revision. |
| `0085` recipe organization | `oos` | It is a concrete current action, so `no_intent` is not applicable. |
| `0086` menu comparison | `oos` | It is concrete and uniquely outside the catalog, so `ambiguous` is not applicable. |

Therefore `product_policy_required=0` for this packet. A near-OOS disagreement must first be checked against grounded
product evidence; only an unresolved contract gap may be escalated as a new product-policy decision.

## 5. Migration rule

The old review workspace remains historical evidence and must not be silently rebound to the new candidate/taxonomy hash.
Migration has:

1. preserve old candidates, responses and the stale adjudication packet unchanged;
2. generated corrected candidates against taxonomy v2;
3. carried forward 158 human labels only for identical normalized contexts, with explicit migration provenance;
4. completed and imported a two-row blind delta review for the new public-interaction and private-message contexts;
5. generated a new adjudication packet whose recommendations distinguish grounded reviewer error, taxonomy drift and a
   genuinely unresolved product boundary.

Gold materialization and IE1.3 freeze remain gated on the resulting v2 adjudication.
