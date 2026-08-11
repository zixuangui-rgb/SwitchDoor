"""Portable dataset materialization and model-free integrity audits."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import ALL_RULES, Episode, canonical_json_bytes, compatible_rules
from .generation import (
    GENERATOR_VERSION,
    GenerationConfig,
    ScenarioGroup,
    audit_scenario_group,
    generate_scenario_groups,
)
from .rendering import audit_rendered_state, png_dimensions, render_png

DATASET_SCHEMA_VERSION = "switchdoor-dataset-v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(raw: str) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_nonfinite,
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")


def read_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            raise ValueError(f"blank JSONL row at line {line_number}")
        value = _strict_json_loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object")
        result.append(value)
    return tuple(result)


def _frame_key(episode: Episode, state_index: int) -> str:
    return _sha256_bytes(
        canonical_json_bytes(
            {
                "world_spec": episode.spec.to_dict(),
                "state": episode.states[state_index].to_dict(),
            }
        )
    )


def materialize_dataset(
    output_dir: str | Path,
    groups: Iterable[ScenarioGroup],
    *,
    renderer: str = "salient",
    resolution: int = 448,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    """Write a self-contained dataset without overwriting an existing path.

    The output separates model-visible histories, labels, and fully private
    replay records. Identical symbolic frames are stored once by content key.
    """

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    materialized_groups = tuple(groups)
    if not materialized_groups:
        raise ValueError("at least one scenario group is required")
    if (
        generation_config is not None
        and materialized_groups
        != generate_scenario_groups(generation_config, len(materialized_groups))
    ):
        raise ValueError("generation config does not reproduce supplied groups")
    root_ids: set[str] = set()
    episode_ids: set[str] = set()
    layout_hashes: set[str] = set()
    split_names: set[str] = set()
    for group in materialized_groups:
        audit_scenario_group(group)
        if group.shell.root_family_id in root_ids:
            raise ValueError("duplicate root_family_id")
        root_ids.add(group.shell.root_family_id)
        layout_hashes.add(group.shell.layout_sha256)
        split_names.add(group.shell.split)
        for episode in group.episodes:
            if episode.episode_id in episode_ids:
                raise ValueError("duplicate episode_id")
            episode_ids.add(episode.episode_id)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        frames_dir = temporary / "frames"
        frames_dir.mkdir()
        public_rows: list[dict[str, Any]] = []
        label_rows: list[dict[str, Any]] = []
        private_rows: list[dict[str, Any]] = []
        frame_receipts: dict[str, dict[str, Any]] = {}
        for group in materialized_groups:
            for episode in group.episodes:
                frame_paths: list[str] = []
                for state_index, state in enumerate(episode.states):
                    frame_id = _frame_key(episode, state_index)
                    relative_path = f"frames/{frame_id}.png"
                    frame_paths.append(relative_path)
                    if frame_id in frame_receipts:
                        continue
                    rgb, png = render_png(
                        episode.spec,
                        state,
                        renderer=renderer,
                        resolution=resolution,
                    )
                    audit_rendered_state(
                        episode.spec,
                        state,
                        rgb,
                        renderer=renderer,
                        resolution=resolution,
                    )
                    path = temporary / relative_path
                    path.write_bytes(png)
                    frame_receipts[frame_id] = {
                        "path": relative_path,
                        "png_sha256": _sha256_bytes(png),
                    }
                public_rows.append(episode.public_dict(frame_paths))
                label_rows.append(episode.label_dict())
                private_rows.append(episode.private_dict())

        public_path = temporary / "public.jsonl"
        labels_path = temporary / "labels.jsonl"
        private_path = temporary / "private.jsonl"
        _write_jsonl(public_path, public_rows)
        _write_jsonl(labels_path, label_rows)
        _write_jsonl(private_path, private_rows)
        receipt_body: dict[str, Any] = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "renderer": renderer,
            "resolution": resolution,
            "splits": sorted(split_names),
            "root_count": len(root_ids),
            "episode_count": len(episode_ids),
            "frame_count": len(frame_receipts),
            "generation": (
                None if generation_config is None else generation_config.to_dict()
            ),
            "audits": {
                "counterfactual_pairing": True,
                "unique_rule_per_history": True,
                "rendered_state_visibility": True,
            },
            "root_family_ids": sorted(root_ids),
            "layout_sha256": sorted(layout_hashes),
            "manifests": {
                "public.jsonl": _sha256_file(public_path),
                "labels.jsonl": _sha256_file(labels_path),
                "private.jsonl": _sha256_file(private_path),
            },
            "frames": dict(sorted(frame_receipts.items())),
        }
        receipt = {
            **receipt_body,
            "dataset_identity": _sha256_bytes(canonical_json_bytes(receipt_body)),
        }
        _write_json(temporary / "dataset_receipt.json", receipt)
        os.replace(temporary, destination)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_private_episodes(dataset_dir: str | Path) -> tuple[Episode, ...]:
    root = Path(dataset_dir)
    return tuple(
        Episode.from_private_mapping(row) for row in read_jsonl(root / "private.jsonl")
    )


def _audit_episode_quartets(episodes: tuple[Episode, ...]) -> None:
    by_root: dict[str, list[Episode]] = {}
    for episode in episodes:
        by_root.setdefault(episode.root_family_id, []).append(episode)
    for root_id, members in by_root.items():
        ordered = sorted(members, key=lambda item: item.rule.config_id.value)
        if len(ordered) != 4 or tuple(item.rule for item in ordered) != ALL_RULES:
            raise ValueError(f"root {root_id} does not contain C0-C3 exactly once")
        reference = ordered[0]
        for episode in ordered:
            if (
                episode.scenario_group_id != reference.scenario_group_id
                or episode.spec != reference.spec
                or episode.actions != reference.actions
                or episode.split != reference.split
            ):
                raise ValueError(f"root {root_id} is not counterfactually paired")
            if compatible_rules(episode.spec, episode.actions, episode.states) != (
                episode.rule,
            ):
                raise ValueError(f"root {root_id} contains an ambiguous history")
        if len({item.states[-1] for item in ordered}) != 1:
            raise ValueError(f"root {root_id} has a rule-dependent endpoint")
        final = ordered[0].states[-1]
        if not final.red_door_open or not final.blue_door_open:
            raise ValueError(f"root {root_id} does not end with both doors open")


def audit_dataset(
    dataset_dir: str | Path,
    *,
    rerender: bool = True,
) -> dict[str, Any]:
    """Recompute manifests, symbolic traces, pairings, and optional PNGs."""

    root = Path(dataset_dir).resolve()
    receipt = _strict_json_loads(
        (root / "dataset_receipt.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != DATASET_SCHEMA_VERSION
    ):
        raise ValueError("dataset receipt schema differs")
    body = dict(receipt)
    claimed_identity = body.pop("dataset_identity", None)
    if claimed_identity != _sha256_bytes(canonical_json_bytes(body)):
        raise ValueError("dataset receipt identity differs")
    manifests = receipt.get("manifests")
    frames = receipt.get("frames")
    if not isinstance(manifests, dict) or not isinstance(frames, dict):
        raise ValueError("dataset receipt is incomplete")
    if receipt.get("audits") != {
        "counterfactual_pairing": True,
        "unique_rule_per_history": True,
        "rendered_state_visibility": True,
    }:
        raise ValueError("dataset audit receipt differs")
    if set(manifests) != {"public.jsonl", "labels.jsonl", "private.jsonl"}:
        raise ValueError("dataset manifest registry differs")
    for relative_path, expected in manifests.items():
        if not _is_sha256(expected):
            raise ValueError(f"manifest hash is invalid: {relative_path}")
        if _sha256_file(root / relative_path) != expected:
            raise ValueError(f"manifest hash differs: {relative_path}")
    for frame_id, frame_receipt in frames.items():
        if not _is_sha256(frame_id) or not isinstance(frame_receipt, dict):
            raise ValueError("frame registry entry is invalid")
        if set(frame_receipt) != {"path", "png_sha256"}:
            raise ValueError("frame registry fields differ")
        if frame_receipt["path"] != f"frames/{frame_id}.png" or not _is_sha256(
            frame_receipt["png_sha256"]
        ):
            raise ValueError("frame registry identity differs")

    public_rows = read_jsonl(root / "public.jsonl")
    label_rows = read_jsonl(root / "labels.jsonl")
    episodes = load_private_episodes(root)
    _audit_episode_quartets(episodes)
    generation = receipt.get("generation")
    if generation is not None:
        if not isinstance(generation, dict) or set(generation) != {
            "generator_version",
            "seed",
            "id_seed",
            "split",
            "layout_family",
            "action_family",
        }:
            raise ValueError("generation provenance fields differ")
        if generation["generator_version"] != GENERATOR_VERSION:
            raise ValueError("generator version differs")
        generation_config = GenerationConfig(
            seed=generation["seed"],
            id_seed=generation["id_seed"],
            split=generation["split"],
            layout_family=generation["layout_family"],
            action_family=generation["action_family"],
        )
        expected_episodes = tuple(
            episode
            for group in generate_scenario_groups(
                generation_config, receipt["root_count"]
            )
            for episode in group.episodes
        )
        if episodes != expected_episodes:
            raise ValueError("private episodes differ from registered generation")
    public_by_id = {row.get("episode_id"): row for row in public_rows}
    label_by_id = {row.get("episode_id"): row for row in label_rows}
    private_by_id = {episode.episode_id: episode for episode in episodes}
    if not (
        len(public_by_id) == len(public_rows)
        and len(label_by_id) == len(label_rows)
        and len(private_by_id) == len(episodes)
        and set(public_by_id) == set(label_by_id) == set(private_by_id)
    ):
        raise ValueError("public, label, and private episode identities differ")

    seen_frame_ids: set[str] = set()
    for episode_id, episode in private_by_id.items():
        if label_by_id[episode_id] != episode.label_dict():
            raise ValueError("label row differs from private replay truth")
        public_input = public_by_id[episode_id].get("input")
        if not isinstance(public_input, dict):
            raise ValueError("public input is missing")
        frame_paths = public_input.get("frame_paths")
        if not isinstance(frame_paths, list):
            raise ValueError("public frame paths are missing")
        if public_by_id[episode_id] != episode.public_dict(frame_paths):
            raise ValueError("public row differs from private replay truth")
        for state_index, (relative_path, state) in enumerate(
            zip(frame_paths, episode.states, strict=True)
        ):
            frame_id = _frame_key(episode, state_index)
            expected_path = f"frames/{frame_id}.png"
            if relative_path != expected_path or frame_id not in frames:
                raise ValueError("frame identity differs from symbolic state")
            path = root / relative_path
            expected_hash = frames[frame_id].get("png_sha256")
            if _sha256_file(path) != expected_hash:
                raise ValueError("frame PNG hash differs")
            if png_dimensions(path.read_bytes()) != (
                receipt["resolution"],
                receipt["resolution"],
            ):
                raise ValueError("frame dimensions differ")
            if rerender:
                expected_rgb, expected_png = render_png(
                    episode.spec,
                    state,
                    renderer=receipt["renderer"],
                    resolution=receipt["resolution"],
                )
                if path.read_bytes() != expected_png:
                    raise ValueError("frame differs from deterministic rerender")
                audit_rendered_state(
                    episode.spec,
                    state,
                    expected_rgb,
                    renderer=receipt["renderer"],
                    resolution=receipt["resolution"],
                )
            seen_frame_ids.add(frame_id)
    if seen_frame_ids != set(frames):
        raise ValueError("frame registry has missing or extra entries")
    observed_roots = {episode.root_family_id for episode in episodes}
    observed_layouts = {
        _sha256_bytes(canonical_json_bytes(episode.spec.to_dict()))
        for episode in episodes
    }
    observed_splits = {episode.split for episode in episodes}
    if (
        receipt.get("root_family_ids") != sorted(observed_roots)
        or receipt.get("layout_sha256") != sorted(observed_layouts)
        or receipt.get("splits") != sorted(observed_splits)
    ):
        raise ValueError("dataset root, layout, or split registry differs")
    if (
        receipt.get("episode_count") != len(episodes)
        or receipt.get("frame_count") != len(frames)
        or receipt.get("root_count") != len(observed_roots)
    ):
        raise ValueError("dataset counts differ")
    return {
        "valid": True,
        "dataset_identity": receipt["dataset_identity"],
        "roots": receipt["root_count"],
        "episodes": receipt["episode_count"],
        "frames": receipt["frame_count"],
        "rerendered": rerender,
    }
