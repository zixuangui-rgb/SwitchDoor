from __future__ import annotations

import pytest

from switchdoor import (
    ALL_RULES,
    C0,
    C1,
    C2,
    C3,
    Action,
    Color,
    Position,
    SwitchDoorEnv,
    WorldSpec,
    WorldState,
    compatible_rules,
    execute,
    oracle_shortest_plan,
    parse_rule_json,
    step,
)


def simple_spec() -> WorldSpec:
    return WorldSpec(
        walls=frozenset({Position(1, 0)}),
        switches=(
            (Color.RED, Position(1, 1)),
            (Color.BLUE, Position(1, 2)),
        ),
        doors=(
            (Color.RED, Position(2, 1)),
            (Color.BLUE, Position(2, 2)),
        ),
        goal=Position(3, 3),
        agent_start=Position(1, 1),
    )


def test_canonical_rule_factorial() -> None:
    assert [rule.public_dict() for rule in ALL_RULES] == [
        {"switch_mapping": "same_color", "door_response": "open_only"},
        {"switch_mapping": "same_color", "door_response": "toggle"},
        {"switch_mapping": "cross_color", "door_response": "open_only"},
        {"switch_mapping": "cross_color", "door_response": "toggle"},
    ]


@pytest.mark.parametrize(
    ("rule", "first", "second"),
    [
        (C0, (True, False), (True, False)),
        (C1, (True, False), (False, False)),
        (C2, (False, True), (False, True)),
        (C3, (False, True), (False, False)),
    ],
)
def test_interaction_semantics(rule, first, second) -> None:
    spec = simple_spec()
    state = WorldState(agent=spec.switch_map[Color.RED])
    state = step(spec, state, Action.INTERACT, rule)
    assert (state.red_door_open, state.blue_door_open) == first
    state = step(spec, state, Action.INTERACT, rule)
    assert (state.red_door_open, state.blue_door_open) == second


def test_blocked_moves_and_off_switch_interaction_are_noops() -> None:
    spec = simple_spec()
    start = WorldState(agent=Position(1, 1))
    assert step(spec, start, Action.MOVE_LEFT, C0) == start
    neutral = WorldState(agent=Position(0, 0))
    assert step(spec, neutral, Action.INTERACT, C0) == neutral


def test_environment_wrapper_matches_pure_executor() -> None:
    spec = simple_spec()
    actions = (Action.INTERACT, Action.MOVE_RIGHT, Action.INTERACT)
    expected = execute(spec, actions, C1)
    env = SwitchDoorEnv(spec, C1)
    observed = env.rollout(actions)
    assert observed == expected
    assert env.reset() == expected[0]


def test_oracle_planner_returns_an_executable_shortest_plan() -> None:
    spec = simple_spec()
    plan = oracle_shortest_plan(spec, C0)
    assert plan is not None
    assert len(plan) == 4
    assert execute(spec, plan, C0)[-1].terminated


def test_compatible_rules_recovers_unique_rule() -> None:
    spec = simple_spec()
    actions = (
        Action.INTERACT,
        Action.INTERACT,
        Action.MOVE_RIGHT,
        Action.INTERACT,
        Action.INTERACT,
    )
    states = execute(spec, actions, C3)
    assert compatible_rules(spec, actions, states) == (C3,)


def test_strict_rule_parser_rejects_extra_or_duplicate_fields() -> None:
    assert (
        parse_rule_json('{"switch_mapping":"cross_color","door_response":"toggle"}')
        == C3
    )
    with pytest.raises(ValueError):
        parse_rule_json(
            '{"switch_mapping":"same_color","switch_mapping":"cross_color",'
            '"door_response":"toggle"}'
        )
    with pytest.raises(ValueError):
        parse_rule_json(
            '{"switch_mapping":"same_color","door_response":"toggle","x":1}'
        )
