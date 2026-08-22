# IE2 frozen-dev reduced reference bundle

This directory is the versionable, secret-free reference output for the selected IE2 configuration.
It contains normalized predictions and cache records, hashes, usage, latency, metrics, badcases, and
the B1-to-B2a descriptive paired bootstrap artifact. It contains no Provider response body, API key,
production Kindred data, or frozen-test prediction.

The B2a cache bundle is portable input for the future B2b call-1 restore path. Every record embeds the
v2 immutable call contract and, for the one-time contract migration, the legacy source cache key. The
provenance artifact maps each destination key to its normalized Prediction hash and raw-response hash.

The manifests were generated before the implementation commit, so `git_worktree_dirty` is expected.
Auditability does not rely on that field: the B2a manifest records exact hashes for the adapter,
normalizer, cache, runner, Provider contract, and schema source files. CI verifies those hashes and all
bundle cross-references against the checked-out tree. After this bundle is committed, a fresh clone can
recompute every evaluator artifact without Provider access.

Interpretation limits:

- Results are from the synthetic, non-blind frozen dev selection set, not frozen test or production.
- The paired bootstrap interval is descriptive on the same selection set used for B1 thresholds and
  B2a Prompt selection; it is not a generalization confidence interval.
- A cache rerun proves pipeline and artifact reproducibility, not repeated model-sampling stability.
- The 30-case cluster-held-out metric removes every bootstrap cluster represented in the 12 few-shot
  examples; the 36-case case-ID-held-out metric removes only the examples themselves.
