"""Command line entry point for the offline frozen RootFS manager."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .manager import RuntimeManager, RuntimeManagerError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dci_bench.runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="import or reuse an offline RootFS bundle")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--bundle", type=Path)
    prepare.add_argument("--offline", action="store_true", help="require local-only preparation")
    prepare.add_argument("--preflight", action="store_true", help="run bwrap tool probes")
    verify = commands.add_parser("verify", help="verify an existing cached RootFS")
    verify.add_argument("--config", required=True, type=Path)
    verify.add_argument("--preflight", action="store_true", help="run bwrap tool probes")
    inspect = commands.add_parser("inspect", help="inspect and verify a cached RootFS")
    inspect.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manager = RuntimeManager.from_config(arguments.config)
        if arguments.command == "prepare":
            result = manager.prepare(bundle=arguments.bundle, offline=True, preflight=arguments.preflight).to_payload()
        elif arguments.command == "verify":
            result = manager.verify(full=True, preflight=arguments.preflight).to_payload()
        else:
            result = manager.inspect()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (RuntimeManagerError, ValueError) as exc:
        print(f"dci_bench.runtime: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
