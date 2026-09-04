"""Command-line interface for generation and contract validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .config import ConfigError, load_config
from .generator import GenerationError, audit_config, generate_dataset, validate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="switchdoor-three-level",
        description="Generate deterministic three-level SwitchDoor episodes.",
    )
    parser.add_argument("--config", help="optional experiment.json path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit-config", help="audit every template transform and rule")

    generate = subparsers.add_parser("generate", help="generate and immediately validate data")
    generate.add_argument("--output", required=True)
    generate.add_argument("--episode-roots", type=int)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument(
        "--mapping",
        choices=("random", "same_color", "cross_color"),
        default="random",
    )
    generate.add_argument(
        "--paired",
        action="store_true",
        help="emit same/cross twins for every layout root",
    )

    validate = subparsers.add_parser("validate", help="replay and byte-verify generated data")
    validate.add_argument("--input", required=True)

    play_web = subparsers.add_parser(
        "play-web",
        help="launch the local dependency-free three-level Web console",
    )
    play_web.add_argument("--host", default="127.0.0.1")
    play_web.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "play-web":
            from .web_app import run_web_server

            return run_web_server(args.host, args.port, config=config)
        if args.command == "audit-config":
            result = audit_config(config)
        elif args.command == "generate":
            roots = (
                config["generation"]["default_episode_roots"]
                if args.episode_roots is None
                else args.episode_roots
            )
            result = generate_dataset(
                args.output,
                episode_roots=roots,
                seed=args.seed,
                mapping=args.mapping,
                paired=args.paired,
                config=config,
            )
        elif args.command == "validate":
            result = validate_dataset(args.input, config=config)
        else:  # pragma: no cover - argparse enforces the command set.
            raise AssertionError(args.command)
    except (ConfigError, GenerationError, OSError, ValueError) as exc:
        print(
            json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False), file=sys.stderr
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
