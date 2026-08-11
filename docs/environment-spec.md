# SwitchDoor environment specification

## Purpose

SwitchDoor is a controlled hidden-parameter environment. A single frame makes
the physical state visible, while the rule governing state transitions remains
latent. An agent or model must use an ordered action-observation history to
identify that rule.

This is closer to visual system identification than ordinary image
classification: observations show what the world looks like; interventions
reveal how the world works.

## State and geometry

The static world is a 7×7 grid containing:

- walls;
- one red and one blue switch;
- one red and one blue door;
- one goal cell;
- one initial agent cell.

The dynamic state is

```text
(agent position, red door open, blue door open, terminated)
```

Both doors are closed initially. `terminated` is true exactly when the agent
occupies the goal.

## Actions and transitions

The action vocabulary is:

```text
MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, INTERACT
```

Moves are deterministic. A move into a wall, the grid boundary, or a closed
door leaves the state unchanged. Reaching the goal terminates the episode;
subsequent actions are no-ops.

`INTERACT` is a no-op away from a switch. On a switch, its effect is controlled
by two hidden factors:

1. `switch_mapping`
   - `same_color`: a switch controls the door of the same color;
   - `cross_color`: a switch controls the opposite-colored door.
2. `door_response`
   - `open_only`: the affected door is set to open;
   - `toggle`: the affected door changes between open and closed.

Their Cartesian product yields C0–C3. The exact executable definition lives in
`src/switchdoor/core.py`; wrappers must call it rather than duplicate it.

## Observations

The environment exposes exact symbolic states and two deterministic RGB
rendering profiles:

- `legacy`: preserves the original renderer geometry;
- `salient`: makes doors visually stronger while preserving object semantics.

Supported resolutions are 448×448 and 896×896. Rendering is implemented without
external image libraries. Identical symbolic inputs and renderer parameters
produce identical PNG bytes.

`SwitchDoorEnv.state`, `SwitchDoorEnv.oracle_rule`, generated `Episode` objects,
and `private.jsonl` are privileged simulator interfaces. A visual rule-learning
agent should receive rendered observations and actions only. The explicit
`oracle_rule` name is intentional: reading it inside a policy or model input is
label leakage.

`oracle_shortest_plan` provides a deterministic breadth-first-search reference
for tasks that need an executable upper bound. It also receives the true rule
and is therefore privileged; it must not be presented as a learned planner.

## Paired root generator

`GenerationConfig` deterministically produces a sequence of scenario roots.
Within one root, geometry, initial state, and actions are held fixed while C0,
C1, C2, and C3 are replayed. The generator audits that:

- all four histories use the same shell;
- every history uniquely identifies its true rule;
- the diagnostic sequence does not reach the goal;
- all four histories end in the same symbolic state with both doors open.

The last condition prevents a final-state shortcut: the rule is identifiable
from the transition history, not merely from the endpoint.

Two layout families are registered:

- `near`: switches are one Manhattan step apart;
- `far`: switches are two Manhattan steps apart.

Two action families vary repeated interaction counts while preserving the same
rule-identification contract:

- `core`: `(1,3)`, `(3,1)`, or `(3,3)` interactions at the two switches;
- `five`: `(1,5)` or `(5,1)` interactions.

These names describe generator families, not train/test roles. Users may assign
any non-empty split token and should keep seeds and roots disjoint when making
statistical generalization claims.

## Dataset information boundary

`materialize_dataset` writes three aligned manifests. `public.jsonl` contains
only model-facing frame paths and actions. `labels.jsonl` contains the hidden
rule. `private.jsonl` contains the full world and replay trace needed for audit.
Datasets produced by the CLI also bind the exact generator version, seed,
identity seed, split, layout family, and action family so the audit can rebuild
the registered roots rather than trusting provenance text.

This separation does not itself provide access control. A training pipeline
must decide which files are available to the model and preserve that boundary.
The receipt detects accidental corruption and binds the generated files; it is
not a signature against an adversary who can rewrite both data and receipt.

## Claim boundary

SwitchDoor supports controlled claims about this finite two-factor environment.
Success does not by itself demonstrate open-ended rule induction, general world
modeling, or planning. Results should identify the root as the independent unit
when C0–C3 share one generated shell.

## Provenance

Version 0.1 extracts the transition system, paired generator semantics, and
native renderer used by the `visual-rule-induction` SwitchDoor experiments at
commit `16b82a1186819ac88ce9c74c602935fbc29052aa`. The standalone package removes
model-training dependencies and broadens split names, while golden regression
tests preserve the registered world and rendering behavior.
