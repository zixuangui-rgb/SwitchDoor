from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from switchdoor.cli import main
from switchdoor.core import canonical_json_bytes
from switchdoor.dataset import audit_dataset, materialize_dataset, read_jsonl
from switchdoor.generation import GenerationConfig, generate_scenario_groups


def test_materialized_dataset_round_trips_and_rejects_tampering(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    groups = generate_scenario_groups(GenerationConfig(seed=41), 2)
    receipt = materialize_dataset(destination, groups)
    assert receipt["root_count"] == 2
    assert receipt["episode_count"] == 8
    assert audit_dataset(destination) == {
        "valid": True,
        "dataset_identity": receipt["dataset_identity"],
        "roots": 2,
        "episodes": 8,
        "frames": receipt["frame_count"],
        "rerendered": True,
    }
    assert len(read_jsonl(destination / "public.jsonl")) == 8
    assert len(read_jsonl(destination / "labels.jsonl")) == 8
    assert len(read_jsonl(destination / "private.jsonl")) == 8
    public_text = (destination / "public.jsonl").read_text(encoding="utf-8")
    assert "switch_mapping" not in public_text
    assert "door_response" not in public_text
    assert "root_family_id" not in public_text
    assert "world_spec" not in public_text
    with pytest.raises(FileExistsError):
        materialize_dataset(destination, groups)

    public_path = destination / "public.jsonl"
    public_path.write_text(public_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash differs"):
        audit_dataset(destination)


def test_audit_rejects_coherently_rehashed_broken_quartet(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    materialize_dataset(
        destination,
        generate_scenario_groups(GenerationConfig(seed=42), 1),
    )
    private_path = destination / "private.jsonl"
    rows = list(read_jsonl(private_path))
    rows[0]["root_family_id"] = "root_alien"
    private_path.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )

    receipt_path = destination / "dataset_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifests"]["private.jsonl"] = hashlib.sha256(
        private_path.read_bytes()
    ).hexdigest()
    body = dict(receipt)
    body.pop("dataset_identity")
    receipt["dataset_identity"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")

    with pytest.raises(ValueError, match="does not contain C0-C3"):
        audit_dataset(destination, rerender=False)


def test_cli_generate_audit_and_rules(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "cli-data"
    assert (
        main(
            [
                "generate",
                "--output",
                str(destination),
                "--roots",
                "1",
                "--seed",
                "9",
            ]
        )
        == 0
    )
    generated = json.loads(capsys.readouterr().out)
    assert generated["roots"] == 1
    receipt = json.loads(
        (destination / "dataset_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["generation"] == {
        "generator_version": "switchdoor-paired-generator-v1",
        "seed": 9,
        "id_seed": 9,
        "split": "train",
        "layout_family": "near",
        "action_family": "core",
    }
    assert main(["audit", "--dataset", str(destination)]) == 0
    audited = json.loads(capsys.readouterr().out)
    assert audited["valid"] is True
    assert main(["rules"]) == 0
    rules = json.loads(capsys.readouterr().out)
    assert set(rules) == {"C0", "C1", "C2", "C3"}


def test_jsonl_reader_rejects_duplicate_keys_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"x":1,"x":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_jsonl(path)
    path.write_text('{"x":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        read_jsonl(path)
