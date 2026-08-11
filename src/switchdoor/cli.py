"""Command-line entry points for generation and integrity auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .core import ALL_RULES
from .dataset import audit_dataset, materialize_dataset
from .generation import (
    ActionFamily,
    GenerationConfig,
    LayoutFamily,
    generate_scenario_groups,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="switchdoor",
        description="Generate and audit deterministic SwitchDoor environments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="materialize a paired dataset")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--roots", type=int, required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--id-seed", type=int)
    generate.add_argument("--split", default="train")
    generate.add_argument(
        "--layout-family",
        choices=[item.value for item in LayoutFamily],
        default=LayoutFamily.NEAR.value,
    )
    generate.add_argument(
        "--action-family",
        choices=[item.value for item in ActionFamily],
        default=ActionFamily.CORE.value,
    )
    generate.add_argument(
        "--renderer", choices=("legacy", "salient"), default="salient"
    )
    generate.add_argument("--resolution", type=int, choices=(448, 896), default=448)

    audit = commands.add_parser("audit", help="verify a materialized dataset")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument(
        "--skip-rerender",
        action="store_true",
        help="verify hashes and symbolic replay without recomputing pixels",
    )

    commands.add_parser("rules", help="print the four canonical hidden rules")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        config = GenerationConfig(
            seed=args.seed,
            id_seed=args.id_seed,
            split=args.split,
            layout_family=LayoutFamily(args.layout_family),
            action_family=ActionFamily(args.action_family),
        )
        groups = generate_scenario_groups(config, args.roots)
        receipt = materialize_dataset(
            args.output,
            groups,
            renderer=args.renderer,
            resolution=args.resolution,
            generation_config=config,
        )
        result = {
            "dataset_identity": receipt["dataset_identity"],
            "output": str(args.output.resolve()),
            "roots": receipt["root_count"],
            "episodes": receipt["episode_count"],
            "frames": receipt["frame_count"],
        }
    elif args.command == "audit":
        result = audit_dataset(args.dataset, rerender=not args.skip_rerender)
    elif args.command == "rules":
        result = {rule.config_id.value: rule.public_dict() for rule in ALL_RULES}
    else:  # pragma: no cover - argparse owns the command set
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
