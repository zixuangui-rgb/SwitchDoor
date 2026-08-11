"""Reference oracle planning utilities built only on the public transition API."""

from __future__ import annotations

from collections import deque

from .core import (
    Action,
    RuleConfig,
    WorldSpec,
    WorldState,
    initial_state,
    step,
    validate_state_for_spec,
)


def oracle_shortest_plan(
    spec: WorldSpec,
    rule: RuleConfig,
    *,
    start: WorldState | None = None,
) -> tuple[Action, ...] | None:
    """Return a deterministic shortest plan to the goal, or ``None``.

    This helper is privileged because it receives the true hidden rule. It is a
    reference executor and upper bound, not an observation available to an
    agent performing rule induction.
    """

    origin = initial_state(spec) if start is None else start
    validate_state_for_spec(spec, origin)
    if origin.terminated:
        return ()
    frontier = deque([origin])
    parents: dict[WorldState, tuple[WorldState, Action] | None] = {origin: None}
    while frontier:
        current = frontier.popleft()
        for action in Action:
            candidate = step(spec, current, action, rule)
            if candidate in parents:
                continue
            parents[candidate] = (current, action)
            if candidate.terminated:
                plan: list[Action] = []
                cursor = candidate
                while parents[cursor] is not None:
                    previous, taken = parents[cursor]
                    plan.append(taken)
                    cursor = previous
                return tuple(reversed(plan))
            frontier.append(candidate)
    return None
