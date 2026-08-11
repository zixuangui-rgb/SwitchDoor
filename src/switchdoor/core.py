"""Deterministic, self-contained contracts for the 7x7 SwitchDoor task.

This module is the source of truth for state transitions. Generation,
rendering, wrappers, and audits should call these functions instead of
reimplementing the game rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

GRID_SIZE = 7
PUBLIC_EPISODE_SCHEMA_VERSION = "switchdoor-dynamic-rgb-v1"
PRIVATE_EPISODE_SCHEMA_VERSION = "switchdoor-dynamic-private-v1"


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"

    @property
    def other(self) -> "Color":
        return Color.BLUE if self is Color.RED else Color.RED


class SwitchMapping(str, Enum):
    SAME_COLOR = "same_color"
    CROSS_COLOR = "cross_color"


class DoorResponse(str, Enum):
    OPEN_ONLY = "open_only"
    TOGGLE = "toggle"


class Action(str, Enum):
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    INTERACT = "INTERACT"


ACTION_DELTA: dict[Action, tuple[int, int]] = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
}


class RuleClass(str, Enum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


_RULE_CLASS_BY_FIELDS: dict[tuple[SwitchMapping, DoorResponse], RuleClass] = {
    (SwitchMapping.SAME_COLOR, DoorResponse.OPEN_ONLY): RuleClass.C0,
    (SwitchMapping.SAME_COLOR, DoorResponse.TOGGLE): RuleClass.C1,
    (SwitchMapping.CROSS_COLOR, DoorResponse.OPEN_ONLY): RuleClass.C2,
    (SwitchMapping.CROSS_COLOR, DoorResponse.TOGGLE): RuleClass.C3,
}


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value), key=repr)
        extra = sorted(set(value) - expected, key=repr)
        raise ValueError(f"{label} fields differ; missing={missing}, extra={extra}")


def _require_plain_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _require_opaque_token(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be a non-empty token without whitespace")
    return value


@dataclass(frozen=True, order=True, slots=True)
class Position:
    row: int
    col: int

    def __post_init__(self) -> None:
        _require_plain_int(self.row, label="row")
        _require_plain_int(self.col, label="col")
        if not (0 <= self.row < GRID_SIZE and 0 <= self.col < GRID_SIZE):
            raise ValueError(
                f"position ({self.row}, {self.col}) is outside the "
                f"{GRID_SIZE}x{GRID_SIZE} grid"
            )

    def to_dict(self) -> dict[str, int]:
        return {"row": self.row, "col": self.col}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Position":
        _require_exact_keys(value, {"row", "col"}, label="position")
        return cls(
            row=_require_plain_int(value["row"], label="position.row"),
            col=_require_plain_int(value["col"], label="position.col"),
        )


@dataclass(frozen=True, slots=True)
class RuleConfig:
    config_id: RuleClass
    switch_mapping: SwitchMapping
    door_response: DoorResponse

    def __post_init__(self) -> None:
        try:
            config_id = RuleClass(self.config_id)
            switch_mapping = SwitchMapping(self.switch_mapping)
            door_response = DoorResponse(self.door_response)
        except (TypeError, ValueError) as exc:
            raise ValueError("rule contains an unsupported enum value") from exc
        if self.config_id is not config_id:
            object.__setattr__(self, "config_id", config_id)
        if self.switch_mapping is not switch_mapping:
            object.__setattr__(self, "switch_mapping", switch_mapping)
        if self.door_response is not door_response:
            object.__setattr__(self, "door_response", door_response)
        expected = _RULE_CLASS_BY_FIELDS[(switch_mapping, door_response)]
        if config_id is not expected:
            raise ValueError(
                f"{config_id.value} does not match "
                f"{switch_mapping.value}/{door_response.value}"
            )

    def target_door(self, switch_color: Color) -> Color:
        color = Color(switch_color)
        if self.switch_mapping is SwitchMapping.SAME_COLOR:
            return color
        return color.other

    def public_dict(self) -> dict[str, str]:
        """Return the canonical two-factor rule representation."""

        return {
            "switch_mapping": self.switch_mapping.value,
            "door_response": self.door_response.value,
        }

    def private_dict(self) -> dict[str, str]:
        return {
            "config_id": self.config_id.value,
            **self.public_dict(),
        }

    @classmethod
    def from_public_mapping(cls, value: Mapping[str, Any]) -> "RuleConfig":
        _require_exact_keys(
            value,
            {"switch_mapping", "door_response"},
            label="public rule",
        )
        try:
            switch_mapping = SwitchMapping(value["switch_mapping"])
            door_response = DoorResponse(value["door_response"])
        except (TypeError, ValueError) as exc:
            raise ValueError("public rule contains an unsupported value") from exc
        return RULE_BY_CLASS[_RULE_CLASS_BY_FIELDS[(switch_mapping, door_response)]]

    @classmethod
    def from_private_mapping(cls, value: Mapping[str, Any]) -> "RuleConfig":
        _require_exact_keys(
            value,
            {"config_id", "switch_mapping", "door_response"},
            label="private rule",
        )
        public = cls.from_public_mapping(
            {
                "switch_mapping": value["switch_mapping"],
                "door_response": value["door_response"],
            }
        )
        try:
            claimed = RuleClass(value["config_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError("private rule config_id is invalid") from exc
        if claimed is not public.config_id:
            raise ValueError("private rule config_id disagrees with its two fields")
        return public


C0 = RuleConfig(
    RuleClass.C0,
    SwitchMapping.SAME_COLOR,
    DoorResponse.OPEN_ONLY,
)
C1 = RuleConfig(
    RuleClass.C1,
    SwitchMapping.SAME_COLOR,
    DoorResponse.TOGGLE,
)
C2 = RuleConfig(
    RuleClass.C2,
    SwitchMapping.CROSS_COLOR,
    DoorResponse.OPEN_ONLY,
)
C3 = RuleConfig(
    RuleClass.C3,
    SwitchMapping.CROSS_COLOR,
    DoorResponse.TOGGLE,
)
ALL_RULES: tuple[RuleConfig, ...] = (C0, C1, C2, C3)
RULE_BY_CLASS: dict[RuleClass, RuleConfig] = {
    rule.config_id: rule for rule in ALL_RULES
}
RULE_BY_ID: dict[str, RuleConfig] = {rule.config_id.value: rule for rule in ALL_RULES}


def rule_from_mapping(value: Mapping[str, Any]) -> RuleConfig:
    return RuleConfig.from_public_mapping(value)


def canonical_rule_completion(rule: RuleConfig) -> str:
    """Return the sole canonical JSON spelling of a rule."""

    return json.dumps(
        rule.public_dict(),
        ensure_ascii=True,
        separators=(",", ":"),
    )


CANONICAL_RULE_COMPLETIONS: tuple[str, ...] = tuple(
    canonical_rule_completion(rule) for rule in ALL_RULES
)


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_rule_json(text: str) -> RuleConfig:
    """Strictly parse one JSON object and reject duplicate/extra fields."""

    if not isinstance(text, str):
        raise TypeError("rule completion must be text")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("rule completion is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("rule completion must be one JSON object")
    return rule_from_mapping(value)


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """Rule-free static geometry for one episode."""

    walls: frozenset[Position]
    switches: tuple[tuple[Color, Position], tuple[Color, Position]]
    doors: tuple[tuple[Color, Position], tuple[Color, Position]]
    goal: Position
    agent_start: Position

    def __post_init__(self) -> None:
        if not isinstance(self.goal, Position):
            raise TypeError("goal must be a Position")
        if not isinstance(self.agent_start, Position):
            raise TypeError("agent_start must be a Position")
        if not isinstance(self.walls, frozenset) or any(
            not isinstance(position, Position) for position in self.walls
        ):
            raise TypeError("walls must be a frozenset of Position values")
        if not isinstance(self.switches, tuple) or len(self.switches) != 2:
            raise TypeError("switches must contain exactly two color-position pairs")
        if not isinstance(self.doors, tuple) or len(self.doors) != 2:
            raise TypeError("doors must contain exactly two color-position pairs")

        switch_map = self._validated_color_positions(
            self.switches,
            label="switches",
        )
        door_map = self._validated_color_positions(self.doors, label="doors")
        object.__setattr__(
            self,
            "switches",
            tuple((color, switch_map[color]) for color in (Color.RED, Color.BLUE)),
        )
        object.__setattr__(
            self,
            "doors",
            tuple((color, door_map[color]) for color in (Color.RED, Color.BLUE)),
        )
        static_objects = [
            switch_map[Color.RED],
            switch_map[Color.BLUE],
            door_map[Color.RED],
            door_map[Color.BLUE],
            self.goal,
        ]
        if len(set(static_objects)) != len(static_objects):
            raise ValueError("switches, doors, and goal must be pairwise distinct")
        if self.walls & set(static_objects):
            raise ValueError("walls overlap a switch, door, or goal")
        if self.agent_start in self.walls:
            raise ValueError("agent_start overlaps a wall")
        if self.agent_start in set(door_map.values()):
            raise ValueError("agent_start overlaps a closed initial door")
        if self.agent_start == self.goal:
            raise ValueError("agent_start must differ from goal")

    @staticmethod
    def _validated_color_positions(
        pairs: tuple[tuple[Color, Position], tuple[Color, Position]],
        *,
        label: str,
    ) -> dict[Color, Position]:
        result: dict[Color, Position] = {}
        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(f"{label} entries must be (Color, Position) tuples")
            raw_color, position = pair
            try:
                color = Color(raw_color)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} contains an invalid color") from exc
            if not isinstance(position, Position):
                raise TypeError(f"{label} positions must be Position values")
            if color in result:
                raise ValueError(f"{label} contains duplicate color {color.value}")
            result[color] = position
        if set(result) != {Color.RED, Color.BLUE}:
            raise ValueError(f"{label} must contain red and blue exactly once")
        if len(set(result.values())) != 2:
            raise ValueError(f"{label} positions must be distinct")
        return result

    @property
    def switch_map(self) -> dict[Color, Position]:
        return dict(self.switches)

    @property
    def door_map(self) -> dict[Color, Position]:
        return dict(self.doors)

    def switch_at(self, position: Position) -> Color | None:
        for color, candidate in self.switches:
            if candidate == position:
                return color
        return None

    def door_at(self, position: Position) -> Color | None:
        for color, candidate in self.doors:
            if candidate == position:
                return color
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "grid_size": GRID_SIZE,
            "walls": [position.to_dict() for position in sorted(self.walls)],
            "switches": {
                color.value: position.to_dict()
                for color, position in sorted(
                    self.switches,
                    key=lambda item: item[0].value,
                )
            },
            "doors": {
                color.value: position.to_dict()
                for color, position in sorted(
                    self.doors,
                    key=lambda item: item[0].value,
                )
            },
            "goal": self.goal.to_dict(),
            "agent_start": self.agent_start.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorldSpec":
        _require_exact_keys(
            value,
            {
                "grid_size",
                "walls",
                "switches",
                "doors",
                "goal",
                "agent_start",
            },
            label="world spec",
        )
        if (
            _require_plain_int(
                value["grid_size"],
                label="world_spec.grid_size",
            )
            != GRID_SIZE
        ):
            raise ValueError(f"world grid_size must equal {GRID_SIZE}")
        walls = value["walls"]
        switches = value["switches"]
        doors = value["doors"]
        goal = value["goal"]
        agent_start = value["agent_start"]
        if (
            not isinstance(walls, list)
            or any(not isinstance(item, Mapping) for item in walls)
            or not isinstance(switches, Mapping)
            or not isinstance(doors, Mapping)
            or not isinstance(goal, Mapping)
            or not isinstance(agent_start, Mapping)
        ):
            raise ValueError("world spec contains an invalid nested value")
        _require_exact_keys(
            switches,
            {"red", "blue"},
            label="world switches",
        )
        _require_exact_keys(doors, {"red", "blue"}, label="world doors")
        parsed_walls = tuple(Position.from_mapping(item) for item in walls)
        if len(set(parsed_walls)) != len(parsed_walls):
            raise ValueError("serialized walls contain duplicates")
        return cls(
            walls=frozenset(parsed_walls),
            switches=tuple(
                (
                    color,
                    Position.from_mapping(switches[color.value]),
                )
                for color in (Color.RED, Color.BLUE)
            ),
            doors=tuple(
                (
                    color,
                    Position.from_mapping(doors[color.value]),
                )
                for color in (Color.RED, Color.BLUE)
            ),
            goal=Position.from_mapping(goal),
            agent_start=Position.from_mapping(agent_start),
        )


def spec_from_mapping(value: Mapping[str, Any]) -> WorldSpec:
    return WorldSpec.from_mapping(value)


@dataclass(frozen=True, slots=True)
class WorldState:
    agent: Position
    red_door_open: bool = False
    blue_door_open: bool = False
    terminated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.agent, Position):
            raise TypeError("agent must be a Position")
        _require_bool(self.red_door_open, label="red_door_open")
        _require_bool(self.blue_door_open, label="blue_door_open")
        _require_bool(self.terminated, label="terminated")

    def door_open(self, color: Color) -> bool:
        active = Color(color)
        return self.red_door_open if active is Color.RED else self.blue_door_open

    def with_agent(
        self,
        position: Position,
        *,
        terminated: bool | None = None,
    ) -> "WorldState":
        return WorldState(
            agent=position,
            red_door_open=self.red_door_open,
            blue_door_open=self.blue_door_open,
            terminated=self.terminated if terminated is None else terminated,
        )

    def with_door(self, color: Color, is_open: bool) -> "WorldState":
        _require_bool(is_open, label="is_open")
        active = Color(color)
        return WorldState(
            agent=self.agent,
            red_door_open=(is_open if active is Color.RED else self.red_door_open),
            blue_door_open=(is_open if active is Color.BLUE else self.blue_door_open),
            terminated=self.terminated,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "agent": self.agent.to_dict(),
            "doors": {
                "red": "open" if self.red_door_open else "closed",
                "blue": "open" if self.blue_door_open else "closed",
            },
            "terminated": self.terminated,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorldState":
        _require_exact_keys(
            value,
            {"agent", "doors", "terminated"},
            label="world state",
        )
        agent = value["agent"]
        doors = value["doors"]
        if not isinstance(agent, Mapping) or not isinstance(doors, Mapping):
            raise ValueError("world state contains an invalid nested value")
        _require_exact_keys(doors, {"red", "blue"}, label="door state")
        if doors["red"] not in {"open", "closed"}:
            raise ValueError("red door state must be open or closed")
        if doors["blue"] not in {"open", "closed"}:
            raise ValueError("blue door state must be open or closed")
        return cls(
            agent=Position.from_mapping(agent),
            red_door_open=doors["red"] == "open",
            blue_door_open=doors["blue"] == "open",
            terminated=_require_bool(
                value["terminated"],
                label="world_state.terminated",
            ),
        )


def state_from_mapping(value: Mapping[str, Any]) -> WorldState:
    return WorldState.from_mapping(value)


def validate_state_for_spec(spec: WorldSpec, state: WorldState) -> None:
    """Reject physically impossible states before executing or rendering."""

    if state.agent in spec.walls:
        raise ValueError("agent occupies a wall")
    occupied_door = spec.door_at(state.agent)
    if occupied_door is not None and not state.door_open(occupied_door):
        raise ValueError("agent occupies a closed door")
    if state.terminated != (state.agent == spec.goal):
        raise ValueError("terminated must be true exactly when the agent is at goal")


def initial_state(spec: WorldSpec) -> WorldState:
    state = WorldState(agent=spec.agent_start)
    validate_state_for_spec(spec, state)
    return state


def _movement_target(state: WorldState, action: Action) -> Position | None:
    row_delta, col_delta = ACTION_DELTA[action]
    row = state.agent.row + row_delta
    col = state.agent.col + col_delta
    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        return None
    return Position(row, col)


def step(
    spec: WorldSpec,
    state: WorldState,
    action: Action,
    rule: RuleConfig,
) -> WorldState:
    """Apply one deterministic action under one of C0--C3.

    Movement into a boundary, wall, or closed door is a no-op.  INTERACT is a
    no-op unless the agent is standing on a switch.  Once the goal is reached,
    all subsequent actions are no-ops.
    """

    if not isinstance(action, Action):
        raise TypeError("action must be an Action enum")
    if rule not in ALL_RULES:
        raise ValueError("rule must be one of the canonical C0-C3 objects")
    validate_state_for_spec(spec, state)
    if state.terminated:
        return state

    if action in ACTION_DELTA:
        target = _movement_target(state, action)
        if target is None or target in spec.walls:
            return state
        door_color = spec.door_at(target)
        if door_color is not None and not state.door_open(door_color):
            return state
        result = WorldState(
            agent=target,
            red_door_open=state.red_door_open,
            blue_door_open=state.blue_door_open,
            terminated=target == spec.goal,
        )
        validate_state_for_spec(spec, result)
        return result

    if action is not Action.INTERACT:
        raise ValueError(f"unsupported action: {action!r}")
    switch_color = spec.switch_at(state.agent)
    if switch_color is None:
        return state
    target_color = rule.target_door(switch_color)
    current_open = state.door_open(target_color)
    next_open = (
        True if rule.door_response is DoorResponse.OPEN_ONLY else not current_open
    )
    result = state.with_door(target_color, next_open)
    validate_state_for_spec(spec, result)
    return result


def execute(
    spec: WorldSpec,
    actions: Iterable[Action],
    rule: RuleConfig,
    *,
    start: WorldState | None = None,
) -> tuple[WorldState, ...]:
    """Return state 0 followed by one state for every supplied action."""

    current = initial_state(spec) if start is None else start
    validate_state_for_spec(spec, current)
    states = [current]
    for action in actions:
        current = step(spec, current, action, rule)
        states.append(current)
    return tuple(states)


def replay_matches(
    spec: WorldSpec,
    actions: Sequence[Action],
    observed_states: Sequence[WorldState],
    candidate_rule: RuleConfig,
) -> bool:
    if len(observed_states) != len(actions) + 1 or not observed_states:
        return False
    try:
        replayed = execute(
            spec,
            actions,
            candidate_rule,
            start=observed_states[0],
        )
    except (TypeError, ValueError):
        return False
    return replayed == tuple(observed_states)


def compatible_rules(
    spec: WorldSpec,
    actions: Sequence[Action],
    observed_states: Sequence[WorldState],
) -> tuple[RuleConfig, ...]:
    """Return every C0--C3 rule exactly compatible with an observed trace."""

    return tuple(
        candidate
        for candidate in ALL_RULES
        if replay_matches(spec, actions, observed_states, candidate)
    )


def state_signature(state: WorldState) -> tuple[int, int, bool, bool, bool]:
    return (
        state.agent.row,
        state.agent.col,
        state.red_door_open,
        state.blue_door_open,
        state.terminated,
    )


def door_signature(state: WorldState) -> tuple[bool, bool]:
    return state.red_door_open, state.blue_door_open


def reachable_when_doors_open(spec: WorldSpec) -> bool:
    """Check geometry reachability while treating both doors as traversable."""

    frontier = [spec.agent_start]
    seen = {spec.agent_start}
    while frontier:
        current = frontier.pop()
        if current == spec.goal:
            return True
        for row_delta, col_delta in ACTION_DELTA.values():
            row = current.row + row_delta
            col = current.col + col_delta
            if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
                continue
            candidate = Position(row, col)
            if candidate in seen or candidate in spec.walls:
                continue
            seen.add(candidate)
            frontier.append(candidate)
    return False


def reachable_states(
    spec: WorldSpec,
    rule: RuleConfig,
) -> frozenset[WorldState]:
    """Exhaustively enumerate the finite states reachable from initial_state."""

    start = initial_state(spec)
    frontier = [start]
    seen = {start}
    while frontier:
        current = frontier.pop()
        for action in Action:
            candidate = step(spec, current, action, rule)
            if candidate in seen:
                continue
            seen.add(candidate)
            frontier.append(candidate)
    return frozenset(seen)


@dataclass(frozen=True, slots=True)
class Episode:
    """One private, fully replayable episode.

    The public manifest must use :meth:`public_dict`; it deliberately omits the
    scenario identifiers, geometry, symbolic states, and ground-truth rule.
    """

    episode_id: str
    scenario_group_id: str
    root_family_id: str
    split: str
    rule: RuleConfig
    spec: WorldSpec
    actions: tuple[Action, ...]
    states: tuple[WorldState, ...]
    template_name: str

    def __post_init__(self) -> None:
        _require_opaque_token(self.episode_id, label="episode_id")
        _require_opaque_token(
            self.scenario_group_id,
            label="scenario_group_id",
        )
        _require_opaque_token(self.root_family_id, label="root_family_id")
        _require_opaque_token(self.template_name, label="template_name")
        _require_opaque_token(self.split, label="split")
        if self.rule not in ALL_RULES:
            raise ValueError("episode rule must be one of C0-C3")
        if not isinstance(self.actions, tuple) or any(
            not isinstance(action, Action) for action in self.actions
        ):
            raise TypeError("episode actions must be a tuple of Action values")
        if not isinstance(self.states, tuple) or any(
            not isinstance(state, WorldState) for state in self.states
        ):
            raise TypeError("episode states must be a tuple of WorldState values")
        if len(self.states) != len(self.actions) + 1:
            raise ValueError("episode must have exactly one more state than action")
        if self.states[0] != initial_state(self.spec):
            raise ValueError("episode must start at agent_start with both doors closed")
        expected = execute(
            self.spec,
            self.actions,
            self.rule,
            start=self.states[0],
        )
        if expected != self.states:
            raise ValueError("episode symbolic trace disagrees with fixed executor")

    def public_dict(self, frame_paths: Iterable[str]) -> dict[str, object]:
        paths = tuple(frame_paths)
        if len(paths) != len(self.states):
            raise ValueError("frame path count must equal state count")
        if any(
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\\" in path
            or ".." in path.split("/")
            for path in paths
        ):
            raise ValueError("frame paths must be safe, non-empty relative paths")
        return {
            "schema_version": PUBLIC_EPISODE_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "input": {
                "frame_paths": list(paths),
                "actions": [action.value for action in self.actions],
            },
        }

    def label_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "target": self.rule.public_dict(),
        }

    def private_dict(self) -> dict[str, object]:
        return {
            "schema_version": PRIVATE_EPISODE_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "scenario_group_id": self.scenario_group_id,
            "root_family_id": self.root_family_id,
            "split": self.split,
            "rule": self.rule.private_dict(),
            "world_spec": self.spec.to_dict(),
            "actions": [action.value for action in self.actions],
            "symbolic_state_trace": [state.to_dict() for state in self.states],
            "template_name": self.template_name,
        }

    @classmethod
    def from_private_mapping(cls, value: Mapping[str, Any]) -> "Episode":
        _require_exact_keys(
            value,
            {
                "schema_version",
                "episode_id",
                "scenario_group_id",
                "root_family_id",
                "split",
                "rule",
                "world_spec",
                "actions",
                "symbolic_state_trace",
                "template_name",
            },
            label="private episode",
        )
        if value["schema_version"] != PRIVATE_EPISODE_SCHEMA_VERSION:
            raise ValueError("private episode schema version differs")
        rule_value = value["rule"]
        spec_value = value["world_spec"]
        action_values = value["actions"]
        state_values = value["symbolic_state_trace"]
        if (
            not isinstance(rule_value, Mapping)
            or not isinstance(spec_value, Mapping)
            or not isinstance(action_values, list)
            or not isinstance(state_values, list)
            or any(not isinstance(item, Mapping) for item in state_values)
        ):
            raise ValueError("private episode contains an invalid nested value")
        try:
            actions = tuple(Action(item) for item in action_values)
        except (TypeError, ValueError) as exc:
            raise ValueError("private episode contains an invalid action") from exc
        return cls(
            episode_id=value["episode_id"],
            scenario_group_id=value["scenario_group_id"],
            root_family_id=value["root_family_id"],
            split=value["split"],
            rule=RuleConfig.from_private_mapping(rule_value),
            spec=WorldSpec.from_mapping(spec_value),
            actions=actions,
            states=tuple(WorldState.from_mapping(item) for item in state_values),
            template_name=value["template_name"],
        )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
