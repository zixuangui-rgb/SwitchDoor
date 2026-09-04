"""Dynamic-size SwitchDoor state transition with one permanent-open rule bit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

Position = tuple[int, int]


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"


class SwitchMapping(str, Enum):
    SAME_COLOR = "same_color"
    CROSS_COLOR = "cross_color"


class Action(str, Enum):
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    INTERACT = "INTERACT"


ACTIONS: tuple[Action, ...] = tuple(Action)
MOVE_DELTAS: dict[Action, Position] = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
}


def _position(value: object, *, label: str) -> Position:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{label} must be [row, col]")
    return value[0], value[1]


@dataclass(frozen=True, slots=True)
class Rule:
    switch_mapping: SwitchMapping

    @classmethod
    def from_value(cls, value: str | SwitchMapping) -> Rule:
        return cls(SwitchMapping(value))

    def target_door(self, switch_color: Color) -> Color:
        if self.switch_mapping is SwitchMapping.SAME_COLOR:
            return switch_color
        return Color.BLUE if switch_color is Color.RED else Color.RED

    def to_dict(self) -> dict[str, str]:
        return {
            "switch_mapping": self.switch_mapping.value,
            "door_response": "open_only",
        }


@dataclass(frozen=True, slots=True)
class WorldSpec:
    grid_size: int
    walls: frozenset[Position]
    red_switch: Position
    blue_switch: Position
    red_door: Position
    blue_door: Position
    goal: Position
    agent_start: Position

    def validate(self) -> None:
        if isinstance(self.grid_size, bool) or not isinstance(self.grid_size, int):
            raise TypeError("grid_size must be an integer")
        if self.grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        positions = (
            self.red_switch,
            self.blue_switch,
            self.red_door,
            self.blue_door,
            self.goal,
            self.agent_start,
        )
        if len(set(positions)) != len(positions):
            raise ValueError("switches, doors, goal, and agent start must be distinct")
        if any(not self.in_bounds(position) for position in self.walls | set(positions)):
            raise ValueError("world contains an out-of-bounds position")
        if self.walls & set(positions):
            raise ValueError("a special position overlaps a wall")
        boundary = {
            (row, col)
            for row in range(self.grid_size)
            for col in range(self.grid_size)
            if row in {0, self.grid_size - 1} or col in {0, self.grid_size - 1}
        }
        if not boundary <= self.walls:
            raise ValueError("the outer boundary must be closed by walls")

    def in_bounds(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def switch_at(self, position: Position) -> Color | None:
        if position == self.red_switch:
            return Color.RED
        if position == self.blue_switch:
            return Color.BLUE
        return None

    def door_at(self, position: Position) -> Color | None:
        if position == self.red_door:
            return Color.RED
        if position == self.blue_door:
            return Color.BLUE
        return None

    def switch_position(self, color: Color) -> Position:
        return self.red_switch if color is Color.RED else self.blue_switch

    def door_position(self, color: Color) -> Position:
        return self.red_door if color is Color.RED else self.blue_door

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_size": self.grid_size,
            "walls": [list(position) for position in sorted(self.walls)],
            "switches": {
                "red": list(self.red_switch),
                "blue": list(self.blue_switch),
            },
            "doors": {
                "red": list(self.red_door),
                "blue": list(self.blue_door),
            },
            "goal": list(self.goal),
            "agent_start": list(self.agent_start),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorldSpec:
        if set(value) != {"grid_size", "walls", "switches", "doors", "goal", "agent_start"}:
            raise ValueError("world spec has missing or extra fields")
        grid_size = value["grid_size"]
        walls = value["walls"]
        switches = value["switches"]
        doors = value["doors"]
        if isinstance(grid_size, bool) or not isinstance(grid_size, int):
            raise TypeError("world grid_size must be an integer")
        if (
            not isinstance(walls, list)
            or not isinstance(switches, dict)
            or not isinstance(doors, dict)
        ):
            raise TypeError("world spec has invalid nested fields")
        if set(switches) != {"red", "blue"} or set(doors) != {"red", "blue"}:
            raise ValueError("world spec must contain red and blue switches and doors")
        parsed_walls = [_position(item, label="wall") for item in walls]
        if len(parsed_walls) != len(set(parsed_walls)):
            raise ValueError("world spec contains duplicate walls")
        spec = cls(
            grid_size=grid_size,
            walls=frozenset(parsed_walls),
            red_switch=_position(switches["red"], label="red switch"),
            blue_switch=_position(switches["blue"], label="blue switch"),
            red_door=_position(doors["red"], label="red door"),
            blue_door=_position(doors["blue"], label="blue door"),
            goal=_position(value["goal"], label="goal"),
            agent_start=_position(value["agent_start"], label="agent_start"),
        )
        spec.validate()
        return spec

    @classmethod
    def from_grid(cls, rows: list[str]) -> WorldSpec:
        if not rows or any(not isinstance(row, str) or len(row) != len(rows) for row in rows):
            raise ValueError("grid must be a non-empty square list of strings")
        positions: dict[str, Position] = {}
        walls: set[Position] = set()
        allowed = {"#", ".", "r", "b", "R", "B", "G", "A"}
        for row_index, row in enumerate(rows):
            for col_index, token in enumerate(row):
                if token not in allowed:
                    raise ValueError(f"unsupported grid token: {token!r}")
                if token == "#":
                    walls.add((row_index, col_index))
                elif token != ".":
                    if token in positions:
                        raise ValueError(f"duplicate grid token: {token!r}")
                    positions[token] = (row_index, col_index)
        if set(positions) != {"r", "b", "R", "B", "G", "A"}:
            raise ValueError("grid must contain r, b, R, B, G, and A exactly once")
        spec = cls(
            grid_size=len(rows),
            walls=frozenset(walls),
            red_switch=positions["r"],
            blue_switch=positions["b"],
            red_door=positions["R"],
            blue_door=positions["B"],
            goal=positions["G"],
            agent_start=positions["A"],
        )
        spec.validate()
        return spec


@dataclass(frozen=True, slots=True)
class WorldState:
    agent: Position
    red_door_open: bool = False
    blue_door_open: bool = False
    terminated: bool = False

    def door_open(self, color: Color) -> bool:
        return self.red_door_open if color is Color.RED else self.blue_door_open

    def with_door_open(self, color: Color) -> WorldState:
        return WorldState(
            agent=self.agent,
            red_door_open=self.red_door_open or color is Color.RED,
            blue_door_open=self.blue_door_open or color is Color.BLUE,
            terminated=self.terminated,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": list(self.agent),
            "doors": {
                "red": "open" if self.red_door_open else "closed",
                "blue": "open" if self.blue_door_open else "closed",
            },
            "terminated": self.terminated,
        }


def validate_state(spec: WorldSpec, state: WorldState) -> None:
    if not spec.in_bounds(state.agent) or state.agent in spec.walls:
        raise ValueError("agent occupies an invalid cell")
    occupied_door = spec.door_at(state.agent)
    if occupied_door is not None and not state.door_open(occupied_door):
        raise ValueError("agent occupies a closed door")
    if state.terminated != (state.agent == spec.goal):
        raise ValueError("terminated must equal whether the agent occupies the goal")


def initial_state(spec: WorldSpec) -> WorldState:
    spec.validate()
    state = WorldState(agent=spec.agent_start)
    validate_state(spec, state)
    return state


def step(spec: WorldSpec, state: WorldState, action: Action, rule: Rule) -> WorldState:
    """Apply one deterministic action; open doors never close."""

    if not isinstance(action, Action):
        raise TypeError("action must be an Action")
    if not isinstance(rule, Rule):
        raise TypeError("rule must be a Rule")
    validate_state(spec, state)
    if state.terminated:
        return state

    if action in MOVE_DELTAS:
        delta_row, delta_col = MOVE_DELTAS[action]
        destination = (state.agent[0] + delta_row, state.agent[1] + delta_col)
        if not spec.in_bounds(destination) or destination in spec.walls:
            return state
        door_color = spec.door_at(destination)
        if door_color is not None and not state.door_open(door_color):
            return state
        result = WorldState(
            agent=destination,
            red_door_open=state.red_door_open,
            blue_door_open=state.blue_door_open,
            terminated=destination == spec.goal,
        )
        validate_state(spec, result)
        return result

    switch_color = spec.switch_at(state.agent)
    if switch_color is None:
        return state
    result = state.with_door_open(rule.target_door(switch_color))
    validate_state(spec, result)
    return result


def status(state: WorldState) -> str:
    return "WIN" if state.terminated else "CONTINUE"
