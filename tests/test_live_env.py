from __future__ import annotations

import copy
import json
from dataclasses import fields
from pathlib import Path

import pytest

from switchdoor_three_level.config import (
    EXPECTED_ACTIONS,
    ConfigError,
    load_config,
    validate_config,
)
from switchdoor_three_level.generator import generate_dataset
from switchdoor_three_level.layouts import materialize_levels
from switchdoor_three_level.live_env import (
    LiveEnvError,
    PublicObservation,
    StepResult,
    ThreeLevelSwitchDoorEnv,
)
from switchdoor_three_level.model import Action, Color, Rule, SwitchMapping, initial_state
from switchdoor_three_level.render import encode_png, render_rgb
from switchdoor_three_level.solver import replay, shortest_plan


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture()
def fast_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    value["render"]["resolution"] = 128
    value["render"]["png_compression_level"] = 1
    return validate_config(value)


def _png_from_observation(observation: PublicObservation, config: dict) -> bytes:
    return encode_png(
        observation.rgb_frame,
        config["render"]["resolution"],
        config["render"]["png_compression_level"],
    )


def test_config_freezes_exact_model_visible_allowlist(config: dict) -> None:
    assert config["task"]["model_visible_observation_fields"] == [
        "rgb_frame",
        "available_actions",
        "status",
        "previous_action",
    ]

    changed = copy.deepcopy(config)
    changed["task"]["model_visible_observation_fields"].append("root_id")
    with pytest.raises(ConfigError, match="model-visible observation fields"):
        validate_config(changed)


def test_reset_returns_only_registered_public_fields(fast_config: dict) -> None:
    root_seed = 4107
    env = ThreeLevelSwitchDoorEnv(fast_config)
    observation = env.reset(root_seed, SwitchMapping.SAME_COLOR)

    assert [field.name for field in fields(PublicObservation)] == [
        "rgb_frame",
        "available_actions",
        "status",
        "previous_action",
    ]
    assert [field.name for field in fields(StepResult)] == [
        "observation",
        "level_done",
        "episode_done",
    ]
    assert isinstance(observation.rgb_frame, bytes)
    assert len(observation.rgb_frame) == 128 * 128 * 3
    assert observation.available_actions == tuple(EXPECTED_ACTIONS)
    assert observation.status == "CONTINUE"
    assert observation.previous_action is None
    assert not hasattr(observation, "root_id")
    assert not hasattr(observation, "level_done")

    first_level = materialize_levels(fast_config, root_seed)[0]
    expected = render_rgb(
        first_level.spec,
        initial_state(first_level.spec),
        fast_config["render"],
    )
    assert observation.rgb_frame == expected


def test_lifecycle_errors_and_legal_noop_are_explicit(fast_config: dict) -> None:
    env = ThreeLevelSwitchDoorEnv(fast_config)
    with pytest.raises(LiveEnvError, match="reset must be called"):
        env.step(Action.MOVE_UP)
    with pytest.raises(LiveEnvError, match="reset must be called"):
        env.advance_level()
    with pytest.raises(TypeError, match="root_seed"):
        env.reset(True, SwitchMapping.SAME_COLOR)
    with pytest.raises(ValueError, match="mapping"):
        env.reset(19, "not_a_mapping")

    initial = env.reset(19, SwitchMapping.SAME_COLOR)
    with pytest.raises(LiveEnvError, match="cannot advance before"):
        env.advance_level()
    with pytest.raises(ValueError, match="invalid action"):
        env.step("JUMP")

    # The agent starts away from both switches, so INTERACT is a legal, costly no-op.
    result = env.step(Action.INTERACT)
    assert result.observation.rgb_frame == initial.rgb_frame
    assert result.observation.previous_action == "INTERACT"
    assert result.level_done is False
    assert result.episode_done is False


@pytest.mark.parametrize("mapping", list(SwitchMapping))
def test_live_episode_matches_registered_transition_across_all_levels(
    fast_config: dict,
    mapping: SwitchMapping,
) -> None:
    root_seed = 998_271
    rule = Rule(mapping)
    levels = materialize_levels(fast_config, root_seed)
    env = ThreeLevelSwitchDoorEnv(fast_config)
    observation = env.reset(root_seed, mapping)

    for level_index, level in enumerate(levels):
        expected_initial = render_rgb(
            level.spec,
            initial_state(level.spec),
            fast_config["render"],
        )
        assert observation.rgb_frame == expected_initial
        assert observation.status == "CONTINUE"
        assert observation.previous_action is None

        required_switch = Color.RED if level_index == 0 else None
        plan = shortest_plan(level.spec, rule, required_first_switch=required_switch)
        assert plan is not None
        expected_states = replay(level.spec, rule, plan)

        result: StepResult | None = None
        for action_index, action in enumerate(plan, start=1):
            result = env.step(action)
            assert result.observation.rgb_frame == render_rgb(
                level.spec,
                expected_states[action_index],
                fast_config["render"],
            )
            assert result.observation.previous_action == action.value
            assert result.level_done is (action_index == len(plan))
            assert result.episode_done is (
                level_index == len(levels) - 1 and action_index == len(plan)
            )

        assert result is not None
        assert result.observation.status == "WIN"
        with pytest.raises(LiveEnvError, match="cannot step after WIN"):
            env.step(Action.INTERACT)

        if level_index < len(levels) - 1:
            observation = env.advance_level()
        else:
            with pytest.raises(LiveEnvError, match="final level"):
                env.advance_level()


def test_paired_live_twins_share_initial_pixels_and_flip_real_feedback(
    fast_config: dict,
) -> None:
    root_seed = 88_204
    levels = materialize_levels(fast_config, root_seed)
    level = levels[0]
    same_env = ThreeLevelSwitchDoorEnv(fast_config)
    cross_env = ThreeLevelSwitchDoorEnv(fast_config)

    same_initial = same_env.reset(root_seed, SwitchMapping.SAME_COLOR)
    cross_initial = cross_env.reset(root_seed, SwitchMapping.CROSS_COLOR)
    assert same_initial.rgb_frame == cross_initial.rgb_frame

    plan = shortest_plan(
        level.spec,
        Rule(SwitchMapping.SAME_COLOR),
        required_first_switch=Color.RED,
    )
    assert plan is not None
    interaction_index = plan.index(Action.INTERACT)
    for action in plan[:interaction_index]:
        same_result = same_env.step(action)
        cross_result = cross_env.step(action)
        assert same_result.observation.rgb_frame == cross_result.observation.rgb_frame

    same_feedback = same_env.step(Action.INTERACT).observation
    cross_feedback = cross_env.step(Action.INTERACT).observation
    assert same_feedback.previous_action == "INTERACT"
    assert cross_feedback.previous_action == "INTERACT"
    assert same_feedback.rgb_frame != cross_feedback.rgb_frame


def test_live_frames_are_byte_identical_to_generated_oracle_replay(
    tmp_path: Path,
    fast_config: dict,
) -> None:
    output = tmp_path / "generated"
    generate_dataset(
        output,
        seed=771,
        mapping="cross_color",
        config=fast_config,
    )
    index = json.loads((output / "index.jsonl").read_text(encoding="utf-8"))
    public = json.loads((output / index["public"]).read_text(encoding="utf-8"))
    private = json.loads((output / index["private"]).read_text(encoding="utf-8"))
    episode_dir = output / "episodes" / index["episode_id"]

    env = ThreeLevelSwitchDoorEnv(fast_config)
    observation = env.reset(private["root_seed"], private["rule"]["switch_mapping"])

    for level_index, (public_level, private_level) in enumerate(
        zip(public["levels"], private["levels"], strict=True)
    ):
        stored_initial = episode_dir / public_level["observations"][0]["rgb_frame"]
        assert _png_from_observation(observation, fast_config) == stored_initial.read_bytes()

        for step_index, action in enumerate(private_level["oracle_actions"], start=1):
            result = env.step(action)
            stored_frame = episode_dir / public_level["observations"][step_index]["rgb_frame"]
            assert (
                _png_from_observation(result.observation, fast_config) == stored_frame.read_bytes()
            )

        assert result.level_done is True
        assert result.episode_done is (level_index == len(public["levels"]) - 1)
        if not result.episode_done:
            observation = env.advance_level()
