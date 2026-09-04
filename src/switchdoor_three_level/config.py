"""Load and validate the single machine-readable generator contract."""

from __future__ import annotations

import json
import sysconfig
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the machine contract no longer expresses this protocol."""


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_CONFIG_PATH = PACKAGE_ROOT / "config" / "experiment.json"
_INSTALLED_CONFIG_PATH = (
    Path(sysconfig.get_path("data"))
    / "share"
    / "switchdoor-three-level"
    / "config"
    / "experiment.json"
)
DEFAULT_CONFIG_PATH = (
    _SOURCE_CONFIG_PATH if _SOURCE_CONFIG_PATH.is_file() else _INSTALLED_CONFIG_PATH
)

EXPECTED_ACTIONS = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "INTERACT"]
EXPECTED_MODEL_VISIBLE_OBSERVATION_FIELDS = [
    "rgb_frame",
    "available_actions",
    "status",
    "previous_action",
]
EXPECTED_LEVEL_IDS = ["L1", "L2", "L3"]
EXPECTED_LEVEL_SIZES = [7, 7, 11]
REQUIRED_TOKENS = {"r", "b", "R", "B", "G", "A"}
ALLOWED_TEMPLATE_TOKENS = REQUIRED_TOKENS | {"#", "."}


def _plain_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be an object")
    return value


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a list")
    return value


def _validate_template(level: Mapping[str, Any], expected_size: int) -> None:
    level_id = level.get("level_id")
    template = _list(level.get("template"), label=f"{level_id}.template")
    if len(template) != expected_size:
        raise ConfigError(f"{level_id}.template must have {expected_size} rows")
    if any(not isinstance(row, str) or len(row) != expected_size for row in template):
        raise ConfigError(f"{level_id}.template must be a square string grid")
    tokens = [token for row in template for token in row]
    if set(tokens) - ALLOWED_TEMPLATE_TOKENS:
        raise ConfigError(f"{level_id}.template contains unsupported tokens")
    for token in REQUIRED_TOKENS:
        if tokens.count(token) != 1:
            raise ConfigError(f"{level_id}.template must contain one {token!r}")
    if any(token != "#" for token in template[0] + template[-1]):
        raise ConfigError(f"{level_id}.template top and bottom boundaries must be walls")
    if any(row[0] != "#" or row[-1] != "#" for row in template):
        raise ConfigError(f"{level_id}.template side boundaries must be walls")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if _plain_int(config.get("schema_version"), label="schema_version") != 1:
        raise ConfigError("schema_version must equal 1")
    if config.get("protocol_id") != "switchdoor-three-level-generator-v1":
        raise ConfigError("protocol_id changed")

    task = _mapping(config.get("task"), label="task")
    if _plain_int(task.get("levels_per_episode"), label="levels_per_episode") != 3:
        raise ConfigError("every episode must contain exactly three levels")
    if task.get("level_sizes") != EXPECTED_LEVEL_SIZES:
        raise ConfigError("level sizes must remain [7, 7, 11]")
    if task.get("shared_hidden_factor") != "switch_mapping":
        raise ConfigError("the only shared hidden factor must be switch_mapping")
    if task.get("switch_mapping_values") != ["same_color", "cross_color"]:
        raise ConfigError("switch mapping values changed")
    if task.get("door_response") != "open_only":
        raise ConfigError("door response must remain open_only")
    if task.get("actions") != EXPECTED_ACTIONS:
        raise ConfigError("registered action order changed")
    if task.get("model_visible_observation_fields") != EXPECTED_MODEL_VISIBLE_OBSERVATION_FIELDS:
        raise ConfigError("model-visible observation fields changed")

    generation = _mapping(config.get("generation"), label="generation")
    if generation.get("layout_randomization") != "independent_d4_transform_per_level":
        raise ConfigError("layout randomization must remain independent D4 transforms")
    if generation.get("level_1_probe_switch") != "red":
        raise ConfigError("L1 must probe the red switch")
    if generation.get("paired_rule_twins_supported") is not True:
        raise ConfigError("paired rule twins must remain supported")
    if (
        _plain_int(
            generation.get("default_episode_roots"),
            label="default_episode_roots",
        )
        < 1
    ):
        raise ConfigError("default_episode_roots must be positive")

    render = _mapping(config.get("render"), label="render")
    resolution = _plain_int(render.get("resolution"), label="render.resolution")
    if resolution < 128:
        raise ConfigError("render.resolution must be at least 128")
    compression = _plain_int(
        render.get("png_compression_level"),
        label="render.png_compression_level",
    )
    if not 0 <= compression <= 9:
        raise ConfigError("PNG compression level must lie in [0, 9]")
    palette = _mapping(render.get("palette"), label="render.palette")
    expected_palette = {
        "background",
        "floor",
        "wall",
        "grid_line",
        "dark",
        "red",
        "blue",
        "goal",
        "agent",
    }
    if set(palette) != expected_palette:
        raise ConfigError("render.palette keys changed")
    for name, raw_color in palette.items():
        values = _list(raw_color, label=f"render.palette.{name}")
        if len(values) != 3 or any(
            isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255
            for channel in values
        ):
            raise ConfigError(f"render.palette.{name} must be an RGB triplet")

    levels = _list(config.get("levels"), label="levels")
    if len(levels) != 3 or any(not isinstance(level, Mapping) for level in levels):
        raise ConfigError("levels must contain exactly three objects")
    if [level.get("level_id") for level in levels] != EXPECTED_LEVEL_IDS:
        raise ConfigError("level IDs must remain L1, L2, L3")
    if [level.get("grid_size") for level in levels] != EXPECTED_LEVEL_SIZES:
        raise ConfigError("level grid sizes changed")
    if [level.get("oracle_policy") for level in levels] != [
        "probe_red_then_reach_goal",
        "shortest_goal_plan",
        "shortest_goal_plan",
    ]:
        raise ConfigError("registered oracle policies changed")
    for level, size in zip(levels, EXPECTED_LEVEL_SIZES, strict=True):
        _validate_template(level, size)

    return json.loads(json.dumps(config))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    source = DEFAULT_CONFIG_PATH if path is None else Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load config: {source}") from exc
    return validate_config(_mapping(value, label="config"))
