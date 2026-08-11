"""Small dependency-free interactive wrapper around the transition system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .core import (
    Action,
    RuleConfig,
    WorldSpec,
    WorldState,
    execute,
    initial_state,
    validate_state_for_spec,
)
from .core import (
    step as transition,
)
from .rendering import render_png, render_rgb


@dataclass(frozen=True, slots=True)
class Transition:
    """One executed action and its exact symbolic consequence."""

    before: WorldState
    action: Action
    after: WorldState

    @property
    def changed(self) -> bool:
        return self.before != self.after

    @property
    def terminated(self) -> bool:
        return self.after.terminated

    @property
    def reached_goal(self) -> bool:
        return not self.before.terminated and self.after.terminated


class SwitchDoorEnv:
    """Interactive stateful view of one fixed SwitchDoor world.

    The environment deliberately does not prescribe a reward function.  Users
    can define task-specific rewards from :class:`Transition`; the original
    hidden-rule benchmark only requires exact state transitions and rendering.
    """

    def __init__(self, spec: WorldSpec, rule: RuleConfig) -> None:
        if not isinstance(spec, WorldSpec):
            raise TypeError("spec must be a WorldSpec")
        if not isinstance(rule, RuleConfig):
            raise TypeError("rule must be a RuleConfig")
        self._spec = spec
        self._rule = rule
        self._state = initial_state(spec)

    @property
    def spec(self) -> WorldSpec:
        return self._spec

    @property
    def oracle_rule(self) -> RuleConfig:
        """Return the privileged rule; never expose this to a rule-learning agent."""

        return self._rule

    @property
    def state(self) -> WorldState:
        return self._state

    def reset(self, state: WorldState | None = None) -> WorldState:
        """Reset to the canonical initial state or an explicitly supplied state."""

        next_state = initial_state(self._spec) if state is None else state
        validate_state_for_spec(self._spec, next_state)
        self._state = next_state
        return self._state

    def step(self, action: Action | str) -> Transition:
        """Execute one action under the environment's hidden rule."""

        try:
            parsed = action if isinstance(action, Action) else Action(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown SwitchDoor action: {action!r}") from exc
        before = self._state
        self._state = transition(self._spec, before, parsed, self._rule)
        return Transition(before=before, action=parsed, after=self._state)

    def rollout(self, actions: Iterable[Action | str]) -> tuple[WorldState, ...]:
        """Execute actions in place and return state 0 followed by every result."""

        states = [self._state]
        for action in actions:
            states.append(self.step(action).after)
        return tuple(states)

    def expected_rollout(
        self, actions: Iterable[Action], *, start: WorldState | None = None
    ) -> tuple[WorldState, ...]:
        """Pure replay helper that does not mutate the environment."""

        return execute(self._spec, actions, self._rule, start=start or self._state)

    def render_rgb(self, *, renderer: str = "salient", resolution: int = 448) -> bytes:
        return render_rgb(
            self._spec,
            self._state,
            renderer=renderer,
            resolution=resolution,
        )

    def render_png(self, *, renderer: str = "salient", resolution: int = 448) -> bytes:
        _, png = render_png(
            self._spec,
            self._state,
            renderer=renderer,
            resolution=resolution,
        )
        return png
