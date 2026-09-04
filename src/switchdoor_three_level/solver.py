"""Exact finite-state solver used to generate and audit oracle trajectories."""

from __future__ import annotations

from collections import deque

from .model import ACTIONS, Action, Color, Rule, WorldSpec, WorldState, initial_state, step


def replay(
    spec: WorldSpec,
    rule: Rule,
    actions: tuple[Action, ...] | list[Action],
) -> tuple[WorldState, ...]:
    current = initial_state(spec)
    states = [current]
    for action in actions:
        current = step(spec, current, action, rule)
        states.append(current)
    return tuple(states)


def shortest_plan(
    spec: WorldSpec,
    rule: Rule,
    *,
    required_first_switch: Color | None = None,
) -> tuple[Action, ...] | None:
    """Return a shortest successful plan.

    When ``required_first_switch`` is set, exactly one effective INTERACT is
    permitted and it must occur at that switch.  This is the registered L1
    information-gathering trace, not a rule-aware shortcut.
    """

    start = initial_state(spec)
    queue: deque[tuple[WorldState, bool, tuple[Action, ...]]] = deque([(start, False, ())])
    seen = {(start, False)}
    while queue:
        state, interacted, path = queue.popleft()
        if state.terminated and (required_first_switch is None or interacted):
            return path
        for action in ACTIONS:
            next_interacted = interacted
            if action is Action.INTERACT and required_first_switch is not None:
                if interacted or spec.switch_at(state.agent) is not required_first_switch:
                    continue
                next_interacted = True
            candidate = step(spec, state, action, rule)
            if candidate == state:
                continue
            key = (candidate, next_interacted)
            if key in seen:
                continue
            seen.add(key)
            queue.append((candidate, next_interacted, (*path, action)))
    return None


def first_effective_switch(
    spec: WorldSpec,
    rule: Rule,
    actions: tuple[Action, ...] | list[Action],
) -> Color | None:
    state = initial_state(spec)
    for action in actions:
        candidate = step(spec, state, action, rule)
        if action is Action.INTERACT and candidate != state:
            return spec.switch_at(state.agent)
        state = candidate
    return None


def first_opened_door(
    spec: WorldSpec,
    rule: Rule,
    actions: tuple[Action, ...] | list[Action],
) -> Color | None:
    state = initial_state(spec)
    for action in actions:
        candidate = step(spec, state, action, rule)
        if action is Action.INTERACT and candidate != state:
            if not state.red_door_open and candidate.red_door_open:
                return Color.RED
            if not state.blue_door_open and candidate.blue_door_open:
                return Color.BLUE
        state = candidate
    return None
