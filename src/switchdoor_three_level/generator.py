"""Generate, audit, and replay-validate three-level SwitchDoor datasets."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .config import EXPECTED_ACTIONS, load_config, validate_config
from .layouts import TRANSFORMS, MaterializedLevel, materialize_level, materialize_levels
from .model import Action, Color, Rule, SwitchMapping, WorldSpec, status
from .render import render_png
from .solver import first_effective_switch, first_opened_door, replay, shortest_plan

DATASET_SCHEMA_VERSION = 2


class GenerationError(RuntimeError):
    """Raised when generation or replay validation cannot preserve the contract."""


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        (json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )
    path.write_bytes(payload)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationError(f"cannot read JSON: {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationError(f"cannot read JSONL: {path}") from exc


def _plan_for_level(level: MaterializedLevel, rule: Rule) -> tuple[Action, ...]:
    required = Color.RED if level.oracle_policy == "probe_red_then_reach_goal" else None
    plan = shortest_plan(level.spec, rule, required_first_switch=required)
    if plan is None:
        raise GenerationError(f"{level.level_id} is unsolvable under {rule.switch_mapping.value}")
    final_state = replay(level.spec, rule, plan)[-1]
    if not final_state.terminated:
        raise GenerationError(f"{level.level_id} oracle plan does not reach the goal")
    return plan


def _expected_controller(rule: Rule) -> Color:
    return Color.RED if rule.switch_mapping is SwitchMapping.SAME_COLOR else Color.BLUE


def audit_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Exhaustively qualify all 24 transformed layouts under both mappings."""

    active = load_config() if config is None else validate_config(config)
    rows: list[dict[str, Any]] = []
    for level_config in active["levels"]:
        per_transform_lengths: dict[str, dict[str, int]] = defaultdict(dict)
        for transform in TRANSFORMS:
            level = materialize_level(level_config, transform)
            for mapping in SwitchMapping:
                rule = Rule(mapping)
                plan = _plan_for_level(level, rule)
                first_switch = first_effective_switch(level.spec, rule, plan)
                opened_door = first_opened_door(level.spec, rule, plan)
                if level.level_id == "L1":
                    if sum(action is Action.INTERACT for action in plan) != 1:
                        raise GenerationError("L1 must contain exactly one probe interaction")
                    if first_switch is not Color.RED:
                        raise GenerationError("L1 must first interact with the red switch")
                    if opened_door is not rule.target_door(Color.RED):
                        raise GenerationError("L1 probe did not reveal the registered mapping")
                else:
                    if first_switch is not _expected_controller(rule):
                        raise GenerationError(
                            f"{level.level_id} first switch does not flip with the mapping"
                        )
                    if opened_door is not Color.RED:
                        raise GenerationError(
                            f"{level.level_id} oracle must first open the target red door"
                        )
                per_transform_lengths[transform][mapping.value] = len(plan)
                rows.append(
                    {
                        "level_id": level.level_id,
                        "grid_size": level.spec.grid_size,
                        "transform": transform,
                        "switch_mapping": mapping.value,
                        "oracle_actions": len(plan),
                        "first_switch": first_switch.value,
                        "first_opened_door": opened_door.value,
                    }
                )
        for transform, lengths in per_transform_lengths.items():
            if lengths["same_color"] != lengths["cross_color"]:
                raise GenerationError(
                    f"{level_config['level_id']} {transform} confounds mapping with plan length"
                )
    return {
        "protocol_id": active["protocol_id"],
        "status": "CONFIG_VALID",
        "levels": 3,
        "transforms_per_level": len(TRANSFORMS),
        "mappings": len(SwitchMapping),
        "audited_cells": len(rows),
        "rows": rows,
    }


def _opaque_id(*parts: object, length: int = 16) -> str:
    canonical = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:length]


def _paired_mapping_order(root_seed: int) -> tuple[SwitchMapping, SwitchMapping]:
    """Assign paired rules to anonymous slots using private root state."""

    mappings = (SwitchMapping.SAME_COLOR, SwitchMapping.CROSS_COLOR)
    order_bit = int(_opaque_id("paired_mapping_order", root_seed), 16) % 2
    return mappings if order_bit == 0 else tuple(reversed(mappings))


def _episode_payloads(
    *,
    config: dict[str, Any],
    output: Path,
    root_id: str,
    root_seed: int,
    episode_id: str,
    rule: Rule,
    levels: tuple[MaterializedLevel, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    episode_dir = output / "episodes" / episode_id
    public_levels: list[dict[str, Any]] = []
    supervision: list[dict[str, Any]] = []
    private_levels: list[dict[str, Any]] = []

    for level in levels:
        plan = _plan_for_level(level, rule)
        states = replay(level.spec, rule, plan)
        observations: list[dict[str, Any]] = []
        for step_index, state in enumerate(states):
            relative_frame = Path("frames") / level.level_id / f"{step_index:03d}.png"
            frame_path = episode_dir / relative_frame
            _rgb, png = render_png(level.spec, state, config["render"])
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame_path.write_bytes(png)
            observations.append(
                {
                    "step_index": step_index,
                    "rgb_frame": relative_frame.as_posix(),
                    "png_sha256": hashlib.sha256(png).hexdigest(),
                    "available_actions": list(EXPECTED_ACTIONS),
                    "status": status(state),
                    "previous_action": None if step_index == 0 else plan[step_index - 1].value,
                }
            )
            if step_index < len(plan):
                supervision.append(
                    {
                        "episode_id": episode_id,
                        "level_id": level.level_id,
                        "step_index": step_index,
                        "rgb_frame": relative_frame.as_posix(),
                        "target_action": plan[step_index].value,
                    }
                )

        first_switch = first_effective_switch(level.spec, rule, plan)
        opened_door = first_opened_door(level.spec, rule, plan)
        if first_switch is None or opened_door is None:
            raise GenerationError(f"{level.level_id} oracle trace lacks an effective interaction")
        public_levels.append(
            {
                "level_id": level.level_id,
                "grid_size": level.spec.grid_size,
                "observations": observations,
            }
        )
        private_levels.append(
            {
                "level_id": level.level_id,
                "purpose": level.purpose,
                "transform": level.transform,
                "world": level.spec.to_dict(),
                "oracle_actions": [action.value for action in plan],
                "oracle_action_count": len(plan),
                "first_effective_switch": first_switch.value,
                "first_opened_door": opened_door.value,
            }
        )

    public = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "protocol_id": config["protocol_id"],
        "levels": public_levels,
    }
    private = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "protocol_id": config["protocol_id"],
        "episode_id": episode_id,
        "root_id": root_id,
        "root_seed": root_seed,
        "rule": rule.to_dict(),
        "levels": private_levels,
    }
    return public, supervision, private


def generate_dataset(
    output: str | Path,
    *,
    episode_roots: int = 1,
    seed: int = 0,
    mapping: str = "random",
    paired: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one or more complete three-level episode roots."""

    active = load_config() if config is None else validate_config(config)
    if isinstance(episode_roots, bool) or not isinstance(episode_roots, int) or episode_roots < 1:
        raise GenerationError("episode_roots must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise GenerationError("seed must be an integer")
    if mapping not in {"random", "same_color", "cross_color"}:
        raise GenerationError("mapping must be random, same_color, or cross_color")
    if paired and mapping != "random":
        raise GenerationError("paired generation cannot also force one mapping")

    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise GenerationError(f"output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    audit_config(active)

    rng = random.Random(seed)
    index_rows: list[dict[str, Any]] = []
    for root_index in range(episode_roots):
        root_seed = rng.getrandbits(63)
        root_id = (
            f"root_{root_index:06d}_"
            f"{_opaque_id(active['protocol_id'], seed, root_index, root_seed)}"
        )
        levels = materialize_levels(active, root_seed)
        if paired:
            mappings = _paired_mapping_order(root_seed)
        elif mapping == "random":
            mappings = (rng.choice(tuple(SwitchMapping)),)
        else:
            mappings = (SwitchMapping(mapping),)

        for episode_slot, active_mapping in enumerate(mappings):
            episode_id = f"episode_{_opaque_id(root_id, 'slot', episode_slot)}"
            public, supervision, private = _episode_payloads(
                config=active,
                output=destination,
                root_id=root_id,
                root_seed=root_seed,
                episode_id=episode_id,
                rule=Rule(active_mapping),
                levels=levels,
            )
            episode_dir = destination / "episodes" / episode_id
            _write_json(episode_dir / "public.json", public)
            _write_jsonl(episode_dir / "supervision.jsonl", supervision)
            _write_json(episode_dir / "private.json", private)
            index_rows.append(
                {
                    "episode_id": episode_id,
                    "root_id": root_id,
                    "public": f"episodes/{episode_id}/public.json",
                    "supervision": f"episodes/{episode_id}/supervision.jsonl",
                    "private": f"episodes/{episode_id}/private.json",
                }
            )

    generation = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "protocol_id": active["protocol_id"],
        "seed": seed,
        "episode_roots": episode_roots,
        "episodes": len(index_rows),
        "paired": paired,
        "mapping_mode": mapping,
    }
    _write_json(destination / "generation.json", generation)
    _write_jsonl(destination / "index.jsonl", index_rows)
    return validate_dataset(destination, config=active)


def _assert_public_firewall(public: Mapping[str, Any]) -> None:
    banned = {
        "episode_id",
        "root_id",
        "switch_mapping",
        "door_response",
        "rule",
        "world",
        "walls",
        "switches",
        "doors",
        "goal",
        "agent_start",
        "transform",
        "oracle_actions",
        "target_action",
    }

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            overlap = set(value) & banned
            if overlap:
                raise GenerationError(f"public observation leaks private fields: {sorted(overlap)}")
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(public)


def validate_dataset(
    input_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay every generated action and byte-verify every rendered observation."""

    active = load_config() if config is None else validate_config(config)
    base = Path(input_path)
    generation = _read_json(base / "generation.json")
    index_rows = _read_jsonl(base / "index.jsonl")
    if not isinstance(generation, dict):
        raise GenerationError("generation metadata must be an object")
    if generation.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise GenerationError("generated dataset schema version changed")
    if generation.get("protocol_id") != active["protocol_id"]:
        raise GenerationError("generated protocol ID differs from active config")
    if generation.get("episodes") != len(index_rows) or not index_rows:
        raise GenerationError("index denominator differs from generation metadata")
    episode_roots = generation.get("episode_roots")
    if isinstance(episode_roots, bool) or not isinstance(episode_roots, int) or episode_roots < 1:
        raise GenerationError("generation metadata has an invalid root denominator")
    paired = generation.get("paired")
    if not isinstance(paired, bool):
        raise GenerationError("generation metadata has an invalid paired flag")
    expected_episodes = episode_roots * (2 if paired else 1)
    if len(index_rows) != expected_episodes:
        raise GenerationError("episode denominator differs from root and pairing metadata")

    root_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_frames = 0
    total_transitions = 0
    for index in index_rows:
        episode_id = index.get("episode_id")
        root_id = index.get("root_id")
        if not isinstance(episode_id, str) or not isinstance(root_id, str):
            raise GenerationError("index contains an invalid episode or root ID")
        public = _read_json(base / index["public"])
        private = _read_json(base / index["private"])
        supervision = _read_jsonl(base / index["supervision"])
        if not isinstance(public, dict) or not isinstance(private, dict):
            raise GenerationError("public and private episode payloads must be objects")
        _assert_public_firewall(public)
        if set(public) != {"schema_version", "protocol_id", "levels"}:
            raise GenerationError("public episode payload contains unregistered fields")
        if (
            public.get("schema_version") != DATASET_SCHEMA_VERSION
            or private.get("schema_version") != DATASET_SCHEMA_VERSION
        ):
            raise GenerationError("episode schema version changed")
        if (
            public.get("protocol_id") != active["protocol_id"]
            or private.get("protocol_id") != active["protocol_id"]
        ):
            raise GenerationError("episode protocol ID differs from active config")
        if private.get("episode_id") != episode_id:
            raise GenerationError("episode identity mismatch")
        if private.get("root_id") != root_id:
            raise GenerationError("root identity mismatch")
        public_levels = public.get("levels")
        private_levels = private.get("levels")
        if not isinstance(public_levels, list) or not isinstance(private_levels, list):
            raise GenerationError("episode levels must be lists")
        if len(public_levels) != 3 or len(private_levels) != 3:
            raise GenerationError("episode does not contain exactly three levels")
        rule_value = private.get("rule")
        if not isinstance(rule_value, dict) or rule_value.get("door_response") != "open_only":
            raise GenerationError("private rule does not preserve open_only")
        rule = Rule.from_value(rule_value.get("switch_mapping"))

        supervision_by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in supervision:
            supervision_by_level[row["level_id"]].append(row)

        initial_pngs: list[bytes] = []
        for expected_index, (public_level, private_level, level_config) in enumerate(
            zip(public_levels, private_levels, active["levels"], strict=True)
        ):
            level_id = level_config["level_id"]
            if (
                public_level.get("level_id") != level_id
                or private_level.get("level_id") != level_id
            ):
                raise GenerationError("level ordering or identity changed")
            if public_level.get("grid_size") != active["task"]["level_sizes"][expected_index]:
                raise GenerationError("public grid size differs from the registered size")
            spec = WorldSpec.from_dict(private_level["world"])
            actions = tuple(Action(value) for value in private_level["oracle_actions"])
            registered_level = materialize_level(level_config, private_level.get("transform"))
            if spec != registered_level.spec:
                raise GenerationError(
                    "private world differs from its registered template transform"
                )
            if private_level.get("purpose") != registered_level.purpose:
                raise GenerationError("private level purpose differs from the registered purpose")
            expected_actions = _plan_for_level(registered_level, rule)
            if actions != expected_actions:
                raise GenerationError("oracle actions differ from the registered exact solver")
            if private_level.get("oracle_action_count") != len(actions):
                raise GenerationError("private oracle action count changed")
            states = replay(spec, rule, actions)
            if not states[-1].terminated:
                raise GenerationError(f"{episode_id}/{level_id} does not terminate at the goal")
            observations = public_level.get("observations")
            if not isinstance(observations, list) or len(observations) != len(states):
                raise GenerationError("observation count does not close over the replay")
            targets = supervision_by_level[level_id]
            if len(targets) != len(actions):
                raise GenerationError("supervision denominator differs from oracle actions")

            for step_index, (observation, state) in enumerate(
                zip(observations, states, strict=True)
            ):
                if observation.get("step_index") != step_index:
                    raise GenerationError("observation step index changed")
                expected_previous = None if step_index == 0 else actions[step_index - 1].value
                if observation.get("previous_action") != expected_previous:
                    raise GenerationError("public previous action does not match replay")
                if observation.get("available_actions") != EXPECTED_ACTIONS:
                    raise GenerationError("available actions changed")
                if observation.get("status") != status(state):
                    raise GenerationError("public status does not match replay")
                frame_path = base / "episodes" / episode_id / observation["rgb_frame"]
                _rgb, expected_png = render_png(spec, state, active["render"])
                try:
                    observed_png = frame_path.read_bytes()
                except OSError as exc:
                    raise GenerationError(f"cannot read frame: {frame_path}") from exc
                if observed_png != expected_png:
                    raise GenerationError("frame bytes do not match deterministic replay")
                if observation.get("png_sha256") != hashlib.sha256(expected_png).hexdigest():
                    raise GenerationError("frame SHA-256 does not match PNG bytes")
                if step_index == 0:
                    initial_pngs.append(expected_png)
                total_frames += 1

            for step_index, target in enumerate(targets):
                if target.get("step_index") != step_index:
                    raise GenerationError("supervision step order changed")
                if target.get("episode_id") != episode_id or target.get("level_id") != level_id:
                    raise GenerationError("supervision identity differs from its episode")
                if target.get("rgb_frame") != observations[step_index].get("rgb_frame"):
                    raise GenerationError("supervision frame differs from its observation")
                if target.get("target_action") != actions[step_index].value:
                    raise GenerationError("supervision target differs from oracle replay")
            first_switch = first_effective_switch(spec, rule, actions)
            first_door = first_opened_door(spec, rule, actions)
            if private_level.get("first_effective_switch") != (
                None if first_switch is None else first_switch.value
            ) or private_level.get("first_opened_door") != (
                None if first_door is None else first_door.value
            ):
                raise GenerationError("private interaction audit differs from replay")
            if level_id == "L1":
                if sum(action is Action.INTERACT for action in actions) != 1:
                    raise GenerationError("L1 must contain exactly one probe interaction")
                if first_switch is not Color.RED or first_door is not rule.target_door(Color.RED):
                    raise GenerationError("L1 probe contract failed")
            elif first_switch is not _expected_controller(rule) or first_door is not Color.RED:
                raise GenerationError(f"{level_id} rule-conditioned controller contract failed")
            total_transitions += len(actions)

        root_groups[root_id].append(
            {
                "mapping": rule.switch_mapping.value,
                "initial_pngs": initial_pngs,
                "private_levels": private_levels,
            }
        )

    if len(root_groups) != episode_roots:
        raise GenerationError("observed root denominator differs from generation metadata")

    if paired:
        for root_id, episodes in root_groups.items():
            if len(episodes) != 2 or {item["mapping"] for item in episodes} != {
                "same_color",
                "cross_color",
            }:
                raise GenerationError(f"paired root is incomplete: {root_id}")
            left, right = episodes
            if left["initial_pngs"] != right["initial_pngs"]:
                raise GenerationError(f"paired root initial pixels differ: {root_id}")
            left_worlds = [level["world"] for level in left["private_levels"]]
            right_worlds = [level["world"] for level in right["private_levels"]]
            if left_worlds != right_worlds:
                raise GenerationError(f"paired root symbolic layouts differ: {root_id}")
    else:
        for root_id, episodes in root_groups.items():
            if len(episodes) != 1:
                raise GenerationError(f"non-paired root is duplicated: {root_id}")

    return {
        "protocol_id": active["protocol_id"],
        "status": "DATASET_VALID",
        "episode_roots": len(root_groups),
        "episodes": len(index_rows),
        "levels": len(index_rows) * 3,
        "frames": total_frames,
        "transitions": total_transitions,
        "paired": paired,
    }
