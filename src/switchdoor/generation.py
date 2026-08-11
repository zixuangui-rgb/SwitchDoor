"""Deterministic procedural generation for paired SwitchDoor scenarios.

The generator creates a *root* by fixing one layout and one action script, then
replaying that shell under C0--C3.  This makes the hidden rule the only causal
difference inside a root and is useful for rule induction, representation
learning, planning, and counterfactual evaluation.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from typing import cast

from .core import (
    ACTION_DELTA,
    ALL_RULES,
    Action,
    Color,
    Episode,
    Position,
    WorldSpec,
    canonical_json_bytes,
    compatible_rules,
    execute,
    reachable_when_doors_open,
)

GENERATOR_VERSION = "switchdoor-paired-generator-v1"


class LayoutFamily(str, Enum):
    """Registered switch-spacing families used by the reference generator."""

    NEAR = "near"
    FAR = "far"

    @property
    def switch_distance(self) -> int:
        return 1 if self is LayoutFamily.NEAR else 2


class ActionFamily(str, Enum):
    """Registered interaction-count families for diagnostic histories."""

    CORE = "core"
    FIVE = "five"

    @property
    def interaction_patterns(self) -> tuple[tuple[int, int], ...]:
        if self is ActionFamily.CORE:
            return ((1, 3), (3, 1), (3, 3))
        return ((1, 5), (5, 1))


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Inputs that fully determine a sequence of generated roots."""

    seed: int
    split: str = "train"
    layout_family: LayoutFamily = LayoutFamily.NEAR
    action_family: ActionFamily = ActionFamily.CORE
    id_seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            not isinstance(self.split, str)
            or not self.split
            or any(character.isspace() for character in self.split)
        ):
            raise ValueError("split must be a non-empty token without whitespace")
        object.__setattr__(self, "layout_family", LayoutFamily(self.layout_family))
        object.__setattr__(self, "action_family", ActionFamily(self.action_family))
        if self.id_seed is not None and (
            isinstance(self.id_seed, bool) or not isinstance(self.id_seed, int)
        ):
            raise TypeError("id_seed must be an integer or None")

    @property
    def resolved_id_seed(self) -> int:
        return self.seed if self.id_seed is None else self.id_seed

    def to_dict(self) -> dict[str, object]:
        return {
            "generator_version": GENERATOR_VERSION,
            "seed": self.seed,
            "id_seed": self.resolved_id_seed,
            "split": self.split,
            "layout_family": self.layout_family.value,
            "action_family": self.action_family.value,
        }


@dataclass(frozen=True, slots=True)
class ScenarioShell:
    """Rule-free geometry and diagnostic action script for one root."""

    scenario_group_id: str
    root_family_id: str
    split: str
    candidate_index: int
    spec: WorldSpec
    actions: tuple[Action, ...]
    template_name: str
    layout_family: LayoutFamily
    action_family: ActionFamily
    first_switch_color: Color
    movement_delta: tuple[int, int]
    blocked_delta: tuple[int, int]
    layout_sha256: str
    shell_sha256: str


@dataclass(frozen=True, slots=True)
class ScenarioGroup:
    """The C0--C3 counterfactual quartet for one fixed shell."""

    shell: ScenarioShell
    episodes: tuple[Episode, Episode, Episode, Episode]

    def __post_init__(self) -> None:
        if tuple(episode.rule for episode in self.episodes) != ALL_RULES:
            raise ValueError("scenario group must contain C0-C3 in order")
        for episode in self.episodes:
            if (
                episode.scenario_group_id != self.shell.scenario_group_id
                or episode.root_family_id != self.shell.root_family_id
                or episode.spec != self.shell.spec
                or episode.actions != self.shell.actions
                or episode.split != self.shell.split
            ):
                raise ValueError("episodes do not share the frozen scenario shell")


DIRECTION_TO_ACTION: dict[tuple[int, int], Action] = {
    delta: action for action, delta in ACTION_DELTA.items()
}
DIRECTIONS: tuple[tuple[int, int], ...] = tuple(DIRECTION_TO_ACTION)


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _opaque(prefix: str, seed: int, value: str) -> str:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()
    return f"{prefix}_{digest[:20]}"


def _border_walls() -> set[Position]:
    result: set[Position] = set()
    for index in range(7):
        result.update(
            {
                Position(0, index),
                Position(6, index),
                Position(index, 0),
                Position(index, 6),
            }
        )
    return result


def _position_add(position: Position, delta: tuple[int, int]) -> Position | None:
    row = position.row + delta[0]
    col = position.col + delta[1]
    if not (0 <= row < 7 and 0 <= col < 7):
        return None
    return Position(row, col)


def _perpendiculars(delta: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    row, col = delta
    return (-col, row), (col, -row)


def _sample_spec(
    rng: random.Random,
    *,
    distance: int,
    movement_delta: tuple[int, int],
    blocked_delta: tuple[int, int],
    first_color: Color,
) -> WorldSpec:
    first_candidates: list[Position] = []
    for row in range(1, 6):
        for col in range(1, 6):
            first = Position(row, col)
            current = first
            for _ in range(distance):
                next_position = _position_add(current, movement_delta)
                if next_position is None or not (
                    1 <= next_position.row <= 5 and 1 <= next_position.col <= 5
                ):
                    break
                current = next_position
            else:
                first_candidates.append(first)
    if not first_candidates:
        raise RuntimeError("no legal switch-pair start exists")

    for _ in range(512):
        first = rng.choice(first_candidates)
        path = [first]
        current = first
        for _ in range(distance):
            next_position = _position_add(current, movement_delta)
            if next_position is None:
                raise AssertionError("validated movement path left the grid")
            current = next_position
            path.append(current)
        second = path[-1]
        block_targets = {
            target
            for target in (
                _position_add(first, blocked_delta),
                _position_add(second, blocked_delta),
            )
            if target is not None
        }
        if block_targets & set(path):
            continue

        switches = (
            (Color.RED, first if first_color is Color.RED else second),
            (Color.BLUE, first if first_color is Color.BLUE else second),
        )
        required_internal_walls = {
            target
            for target in block_targets
            if 1 <= target.row <= 5 and 1 <= target.col <= 5
        }
        reserved = set(path) | required_internal_walls
        candidates = [
            Position(row, col)
            for row in range(1, 6)
            for col in range(1, 6)
            if Position(row, col) not in reserved
        ]
        if len(candidates) < 6:
            continue
        red_door, blue_door, goal = rng.sample(candidates, k=3)
        static = {first, second, red_door, blue_door, goal}
        wall_candidates = [
            item for item in candidates if item not in static and item not in path
        ]
        extra_count = min(rng.randint(2, 6), len(wall_candidates))
        spec = WorldSpec(
            walls=frozenset(
                _border_walls()
                | required_internal_walls
                | set(rng.sample(wall_candidates, k=extra_count))
            ),
            switches=switches,
            doors=((Color.RED, red_door), (Color.BLUE, blue_door)),
            goal=goal,
            agent_start=first,
        )
        if reachable_when_doors_open(spec):
            return spec
    raise RuntimeError("could not sample a reachable SwitchDoor layout")


def _factor_assignment(
    config: GenerationConfig,
    candidate_index: int,
) -> tuple[tuple[int, int], Color, tuple[int, int], tuple[int, int]]:
    combinations = [
        (pattern, color, movement, blocked)
        for pattern in config.action_family.interaction_patterns
        for color in (Color.RED, Color.BLUE)
        for movement in DIRECTIONS
        for blocked in _perpendiculars(movement)
    ]
    return combinations[candidate_index % len(combinations)]


def build_shell(config: GenerationConfig, candidate_index: int) -> ScenarioShell:
    """Build one deterministic rule-free shell."""

    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise TypeError("candidate_index must be an integer")
    if candidate_index < 0:
        raise ValueError("candidate_index must be non-negative")
    interaction_counts, first_color, movement_delta, blocked_delta = _factor_assignment(
        config, candidate_index
    )
    rng = random.Random(config.seed + candidate_index * 1_000_003)
    spec = _sample_spec(
        rng,
        distance=config.layout_family.switch_distance,
        movement_delta=movement_delta,
        blocked_delta=blocked_delta,
        first_color=first_color,
    )
    first_count, second_count = interaction_counts
    actions = (
        *((Action.INTERACT,) * first_count),
        DIRECTION_TO_ACTION[blocked_delta],
        *(
            (DIRECTION_TO_ACTION[movement_delta],)
            * config.layout_family.switch_distance
        ),
        *((Action.INTERACT,) * second_count),
        DIRECTION_TO_ACTION[blocked_delta],
    )
    if not 7 <= len(actions) <= 10:
        raise AssertionError("registered history must contain 7-10 actions")
    layout_sha256 = _hash(spec.to_dict())
    template_name = (
        f"i{first_count}_i{second_count}_d{config.layout_family.switch_distance}_"
        f"b{DIRECTION_TO_ACTION[blocked_delta].value.lower()}"
    )
    shell_sha256 = _hash(
        {
            "split": config.split,
            "world_spec": spec.to_dict(),
            "actions": [action.value for action in actions],
            "template_name": template_name,
        }
    )
    group_id = _opaque(
        "grp",
        config.resolved_id_seed,
        f"{config.split}|{candidate_index}|{shell_sha256}",
    )
    return ScenarioShell(
        scenario_group_id=group_id,
        root_family_id=_opaque("root", config.resolved_id_seed, f"{group_id}|root"),
        split=config.split,
        candidate_index=candidate_index,
        spec=spec,
        actions=actions,
        template_name=template_name,
        layout_family=config.layout_family,
        action_family=config.action_family,
        first_switch_color=first_color,
        movement_delta=movement_delta,
        blocked_delta=blocked_delta,
        layout_sha256=layout_sha256,
        shell_sha256=shell_sha256,
    )


def build_scenario_group(
    config: GenerationConfig,
    candidate_index: int,
) -> ScenarioGroup:
    """Build and audit the C0--C3 quartet for one shell."""

    shell = build_shell(config, candidate_index)
    episodes = []
    for rule in ALL_RULES:
        episodes.append(
            Episode(
                episode_id=_opaque(
                    "ep",
                    config.resolved_id_seed,
                    f"{shell.scenario_group_id}|{rule.config_id.value}",
                ),
                scenario_group_id=shell.scenario_group_id,
                root_family_id=shell.root_family_id,
                split=shell.split,
                rule=rule,
                spec=shell.spec,
                actions=shell.actions,
                states=execute(shell.spec, shell.actions, rule),
                template_name=shell.template_name,
            )
        )
    group = ScenarioGroup(
        shell=shell,
        episodes=cast(tuple[Episode, Episode, Episode, Episode], tuple(episodes)),
    )
    audit_scenario_group(group)
    return group


def audit_scenario_group(group: ScenarioGroup) -> None:
    """Fail if a paired root is ambiguous or contains endpoint shortcuts."""

    terminal_states: set[bytes] = set()
    for episode in group.episodes:
        if any(state.terminated for state in episode.states):
            raise ValueError("diagnostic history reached the goal")
        if compatible_rules(episode.spec, episode.actions, episode.states) != (
            episode.rule,
        ):
            raise ValueError("history is not uniquely identifiable")
        terminal_states.add(canonical_json_bytes(episode.states[-1].to_dict()))
    if len(terminal_states) != 1:
        raise ValueError("terminal state differs across C0-C3")
    final = group.episodes[0].states[-1]
    if not final.red_door_open or not final.blue_door_open:
        raise ValueError("diagnostic history must end with both doors open")


def generate_scenario_groups(
    config: GenerationConfig,
    count: int,
) -> tuple[ScenarioGroup, ...]:
    """Generate ``count`` unique, audited roots in deterministic order."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 1:
        raise ValueError("count must be positive")
    groups: list[ScenarioGroup] = []
    seen_layouts: set[str] = set()
    seen_shells: set[str] = set()
    candidate_index = 0
    while len(groups) < count:
        if candidate_index > max(10_000, count * 50):
            raise RuntimeError("could not generate enough unique roots")
        group = build_scenario_group(config, candidate_index)
        candidate_index += 1
        if (
            group.shell.layout_sha256 in seen_layouts
            or group.shell.shell_sha256 in seen_shells
        ):
            continue
        groups.append(group)
        seen_layouts.add(group.shell.layout_sha256)
        seen_shells.add(group.shell.shell_sha256)
    return tuple(groups)
