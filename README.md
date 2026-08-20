# Kindred Intent Bench

An offline evaluation workbench with optional networked model runners for open-set intent recognition and Activity routing
in a stateful Agent. It evaluates routing when the input already contains observable intent evidence; it does not assign a
single Gold Activity to spontaneous autonomous choice.

Cached predictions make scoring reproducible. Provider inference itself may not be exactly replayable.

The project is currently in the design phase. It does not read Kindred production data, call live Capabilities, or modify
the Kindred runtime.

## Design

See [docs/design.md](docs/design.md) for the task definition, Gold-label contract, baseline matrix, evaluation metrics,
statistical gates, implementation plan, and portfolio scope.

The executable delivery sequence and acceptance gates are defined in
[docs/implementation-plan.md](docs/implementation-plan.md).
