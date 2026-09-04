"""Stateful live wrapper around the registered pure SwitchDoor transition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import EXPECTED_ACTIONS, load_config, validate_config
from .layouts import MaterializedLevel, materialize_levels
from .model import Action, Rule, SwitchMapping, WorldState, initial_state, status, step
from .render import render_rgb


class LiveEnvError(RuntimeError):
    """Raised when a live episode lifecycle operation is invalid."""


@dataclass(frozen=True, slots=True)
class PublicObservation:
    """The exact model-visible observation; no episode identity or private state."""

    rgb_frame: bytes
    available_actions: tuple[str, ...]
    status: str
    previous_action: str | None


@dataclass(frozen=True, slots=True)
class StepResult:
    """One public observation plus runner-only termination metadata."""

    observation: PublicObservation
    level_done: bool
    episode_done: bool


class ThreeLevelSwitchDoorEnv:
    """Run one registered L1-L3 episode without exposing future or private data.

    The hidden rule and the three materialized layouts persist for the whole
    episode.  Dynamic world state resets between levels.  Reader, belief,
    planner, scoring, and action-budget state intentionally live outside this
    environment.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = load_config() if config is None else validate_config(config)
        self._levels: tuple[MaterializedLevel, ...] | None = None
        self._rule: Rule | None = None
        self._level_index: int | None = None
        self._state: WorldState | None = None
        self._previous_action: Action | None = None

    def reset(
        self,
        root_seed: int,
        mapping: str | SwitchMapping,
    ) -> PublicObservation:
        """Start a fresh three-level episode and return the initial L1 pixels."""

        if isinstance(root_seed, bool) or not isinstance(root_seed, int):
            raise TypeError("root_seed must be an integer")
        try:
            rule = Rule.from_value(mapping)
        except (TypeError, ValueError) as exc:
            raise ValueError("mapping must be same_color or cross_color") from exc

        self._levels = materialize_levels(self._config, root_seed)
        self._rule = rule
        self._level_index = 0
        self._state = initial_state(self._levels[0].spec)
        self._previous_action = None
        return self._observation()

    def step(self, action: str | Action) -> StepResult:
        """Apply exactly one model action and return its real public consequence."""

        self._require_started()
        assert self._state is not None
        if self._state.terminated:
            raise LiveEnvError(
                "cannot step after WIN; advance_level or reset before another action"
            )
        try:
            parsed_action = action if isinstance(action, Action) else Action(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid action: {action!r}") from exc

        level = self._current_level()
        assert self._rule is not None
        self._state = step(level.spec, self._state, parsed_action, self._rule)
        self._previous_action = parsed_action
        level_done = self._state.terminated
        episode_done = level_done and self._is_last_level()
        return StepResult(
            observation=self._observation(),
            level_done=level_done,
            episode_done=episode_done,
        )

    def advance_level(self) -> PublicObservation:
        """Enter the next level after WIN without consuming a model action."""

        self._require_started()
        assert self._state is not None
        assert self._level_index is not None
        assert self._levels is not None
        if not self._state.terminated:
            raise LiveEnvError("cannot advance before the current level reaches WIN")
        if self._is_last_level():
            raise LiveEnvError("cannot advance after the final level")

        self._level_index += 1
        self._state = initial_state(self._levels[self._level_index].spec)
        self._previous_action = None
        return self._observation()

    def _require_started(self) -> None:
        if (
            self._levels is None
            or self._rule is None
            or self._level_index is None
            or self._state is None
        ):
            raise LiveEnvError("reset must be called before using the live environment")

    def _current_level(self) -> MaterializedLevel:
        self._require_started()
        assert self._levels is not None
        assert self._level_index is not None
        return self._levels[self._level_index]

    def _is_last_level(self) -> bool:
        self._require_started()
        assert self._levels is not None
        assert self._level_index is not None
        return self._level_index == len(self._levels) - 1

    def _observation(self) -> PublicObservation:
        level = self._current_level()
        assert self._state is not None
        rgb = render_rgb(level.spec, self._state, self._config["render"])
        return PublicObservation(
            rgb_frame=rgb,
            available_actions=tuple(EXPECTED_ACTIONS),
            status=status(self._state),
            previous_action=(
                None if self._previous_action is None else self._previous_action.value
            ),
        )
