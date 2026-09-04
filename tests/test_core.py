from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from switchdoor_three_level.cli import main
from switchdoor_three_level.config import ConfigError, load_config, validate_config
from switchdoor_three_level.generator import (
    GenerationError,
    _opaque_id,
    _paired_mapping_order,
    audit_config,
    generate_dataset,
    validate_dataset,
)
from switchdoor_three_level.layouts import TRANSFORMS, materialize_level
from switchdoor_three_level.model import (
    Action,
    Color,
    Rule,
    SwitchMapping,
    initial_state,
    step,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture()
def fast_config(config: dict) -> dict:
    value = copy.deepcopy(config)
    value["render"]["resolution"] = 128
    value["render"]["png_compression_level"] = 1
    return validate_config(value)


def _index(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "index.jsonl").read_text().splitlines()]


def _files(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_registered_contract_is_exactly_three_levels(config: dict) -> None:
    assert config["task"]["level_sizes"] == [7, 7, 11]
    assert [level["level_id"] for level in config["levels"]] == ["L1", "L2", "L3"]
    assert config["task"]["door_response"] == "open_only"
    assert config["task"]["switch_mapping_values"] == ["same_color", "cross_color"]

    changed = copy.deepcopy(config)
    changed["task"]["level_sizes"] = [7, 7, 13]
    with pytest.raises(ConfigError, match="level sizes"):
        validate_config(changed)


def test_open_only_transition_and_mapping(config: dict) -> None:
    level = materialize_level(config["levels"][0], "identity")
    start = initial_state(level.spec)
    at_red = step(level.spec, start, Action.MOVE_UP, Rule(SwitchMapping.SAME_COLOR))
    assert level.spec.switch_at(at_red.agent) is Color.RED

    same = step(level.spec, at_red, Action.INTERACT, Rule(SwitchMapping.SAME_COLOR))
    assert same.red_door_open is True and same.blue_door_open is False
    assert step(level.spec, same, Action.INTERACT, Rule(SwitchMapping.SAME_COLOR)) == same

    cross = step(level.spec, at_red, Action.INTERACT, Rule(SwitchMapping.CROSS_COLOR))
    assert cross.red_door_open is False and cross.blue_door_open is True


def test_all_d4_layouts_are_balanced_and_solvable(config: dict) -> None:
    report = audit_config(config)
    assert report["status"] == "CONFIG_VALID"
    assert report["audited_cells"] == 3 * len(TRANSFORMS) * 2
    cells: dict[tuple[str, str], dict[str, int]] = {}
    for row in report["rows"]:
        key = (row["level_id"], row["transform"])
        cells.setdefault(key, {})[row["switch_mapping"]] = row["oracle_actions"]
    assert all(lengths["same_color"] == lengths["cross_color"] for lengths in cells.values())


def test_generate_one_root_writes_three_replayable_levels(
    tmp_path: Path,
    fast_config: dict,
) -> None:
    output = tmp_path / "single"
    report = generate_dataset(output, seed=17, mapping="same_color", config=fast_config)
    assert report["status"] == "DATASET_VALID"
    assert report["episode_roots"] == 1
    assert report["episodes"] == 1
    assert report["levels"] == 3

    index = _index(output)
    public = json.loads((output / index[0]["public"]).read_text())
    private = json.loads((output / index[0]["private"]).read_text())
    assert set(public) == {"schema_version", "protocol_id", "levels"}
    assert public["schema_version"] == 2
    assert [level["grid_size"] for level in public["levels"]] == [7, 7, 11]
    public_text = json.dumps(public, sort_keys=True)
    for private_name in (
        "episode_id",
        "root_id",
        "switch_mapping",
        "door_response",
        "world",
        "walls",
        "oracle_actions",
        "target_action",
    ):
        assert private_name not in public_text
    assert private["rule"] == {
        "switch_mapping": "same_color",
        "door_response": "open_only",
    }
    assert private["levels"][0]["oracle_actions"].count("INTERACT") == 1
    assert validate_dataset(output, config=fast_config) == report


def test_paired_identity_does_not_encode_mapping() -> None:
    assert _paired_mapping_order(2) == (
        SwitchMapping.SAME_COLOR,
        SwitchMapping.CROSS_COLOR,
    )
    assert _paired_mapping_order(3) == (
        SwitchMapping.CROSS_COLOR,
        SwitchMapping.SAME_COLOR,
    )


def test_paired_twins_share_initial_pixels_and_flip_later_switches(
    tmp_path: Path,
    fast_config: dict,
) -> None:
    output = tmp_path / "paired"
    report = generate_dataset(output, seed=23, paired=True, config=fast_config)
    assert report["episodes"] == 2
    assert report["paired"] is True

    entries = _index(output)
    payloads = []
    for entry in entries:
        old_mapping_ids = {
            f"episode_{_opaque_id(entry['root_id'], mapping.value)}" for mapping in SwitchMapping
        }
        assert entry["episode_id"] not in old_mapping_ids
        public = json.loads((output / entry["public"]).read_text())
        private = json.loads((output / entry["private"]).read_text())
        initial = [
            (
                output / "episodes" / entry["episode_id"] / level["observations"][0]["rgb_frame"]
            ).read_bytes()
            for level in public["levels"]
        ]
        payloads.append((private, initial))

    by_mapping = {payload[0]["rule"]["switch_mapping"]: payload for payload in payloads}
    assert by_mapping["same_color"][1] == by_mapping["cross_color"][1]
    assert [
        level["first_effective_switch"] for level in by_mapping["same_color"][0]["levels"][1:]
    ] == ["red", "red"]
    assert [
        level["first_effective_switch"] for level in by_mapping["cross_color"][0]["levels"][1:]
    ] == ["blue", "blue"]


def test_generation_is_byte_deterministic(tmp_path: Path, fast_config: dict) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    generate_dataset(left, episode_roots=2, seed=91, config=fast_config)
    generate_dataset(right, episode_roots=2, seed=91, config=fast_config)
    assert _files(left) == _files(right)


def test_validation_rejects_changed_root_denominator(tmp_path: Path, fast_config: dict) -> None:
    output = tmp_path / "bad_denominator"
    generate_dataset(output, seed=7, config=fast_config)
    generation_path = output / "generation.json"
    generation = json.loads(generation_path.read_text())
    generation["episode_roots"] = 2
    generation_path.write_text(json.dumps(generation), encoding="utf-8")
    with pytest.raises(GenerationError, match="denominator"):
        validate_dataset(output, config=fast_config)


def test_validation_rejects_world_transform_mismatch(tmp_path: Path, fast_config: dict) -> None:
    output = tmp_path / "bad_transform"
    generate_dataset(output, seed=11, config=fast_config)
    entry = _index(output)[0]
    private_path = output / entry["private"]
    private = json.loads(private_path.read_text())
    original = private["levels"][0]["transform"]
    private["levels"][0]["transform"] = TRANSFORMS[
        (TRANSFORMS.index(original) + 1) % len(TRANSFORMS)
    ]
    private_path.write_text(json.dumps(private), encoding="utf-8")
    with pytest.raises(GenerationError, match="registered template transform"):
        validate_dataset(output, config=fast_config)


def test_validation_rejects_public_identity_shortcuts(tmp_path: Path, fast_config: dict) -> None:
    output = tmp_path / "bad_public_identity"
    generate_dataset(output, seed=13, config=fast_config)
    entry = _index(output)[0]
    public_path = output / entry["public"]
    public = json.loads(public_path.read_text())
    public["episode_id"] = entry["episode_id"]
    public_path.write_text(json.dumps(public), encoding="utf-8")
    with pytest.raises(GenerationError, match="leaks private fields"):
        validate_dataset(output, config=fast_config)


def test_nonempty_output_fails_without_deleting_user_data(
    tmp_path: Path,
    fast_config: dict,
) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    with pytest.raises(GenerationError, match="not empty"):
        generate_dataset(output, config=fast_config)
    assert marker.read_text(encoding="utf-8") == "user data"


def test_cli_smoke(tmp_path: Path, fast_config: dict) -> None:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(fast_config), encoding="utf-8")
    output = tmp_path / "cli"
    assert main(["--config", str(config_path), "audit-config"]) == 0
    assert (
        main(
            [
                "--config",
                str(config_path),
                "generate",
                "--output",
                str(output),
                "--seed",
                "5",
            ]
        )
        == 0
    )
    assert main(["--config", str(config_path), "validate", "--input", str(output)]) == 0
