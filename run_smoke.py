"""Run offline smoke checks for the benchmark stand."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class SmokeCommand:
    name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline smoke checks")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run offline experiment configs instead of only validate/dry-run checks.",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Also run pytest, Ruff, compileall, and pip check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print smoke commands without executing them.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for full smoke experiment artifacts.",
    )
    return parser.parse_args()


def build_smoke_commands(
    full: bool = False,
    quality: bool = False,
    output_dir: str | None = None,
) -> list[SmokeCommand]:
    """Build deterministic offline smoke commands."""
    py = sys.executable
    commands = [
        SmokeCommand("list_strategies", [py, "run_benchmark.py", "--list-strategies"]),
        SmokeCommand("list_providers", [py, "run_benchmark.py", "--list-providers"]),
        SmokeCommand(
            "provider_dry_run",
            [
                py,
                "check_providers.py",
                "--embedding-provider",
                "mock",
                "--embedding-model",
                "mock:12",
                "--llm-provider",
                "openrouter",
                "--llm-model",
                "openai/gpt-4.1-mini",
            ],
        ),
        SmokeCommand(
            "mock_embedding_live",
            [
                py,
                "check_providers.py",
                "--run",
                "--skip-llm",
                "--embedding-provider",
                "mock",
                "--embedding-model",
                "mock:12",
            ],
        ),
        SmokeCommand(
            "quick_benchmark_print",
            [
                py,
                "quick_benchmark.py",
                "data/custom_benchmark.jsonl",
                "--provider",
                "mock",
                "--model",
                "mock:64",
                "--strategies",
                "dynamic_semantic,token_text",
                "--print-command",
            ],
        ),
        SmokeCommand(
            "static_validate",
            [py, "run_experiments.py", "configs/static_smoke.yaml", "--validate-only"],
        ),
        SmokeCommand(
            "custom_validate",
            [py, "run_experiments.py", "configs/custom_smoke.yaml", "--validate-only"],
        ),
        SmokeCommand(
            "custom_optuna_validate",
            [py, "run_experiments.py", "configs/custom_optuna_smoke.yaml", "--validate-only"],
        ),
        SmokeCommand(
            "multi_dataset_optuna_validate",
            [py, "run_experiments.py", "configs/multi_dataset_optuna_smoke.yaml", "--validate-only"],
        ),
    ]

    if full:
        full_output_dir = Path(output_dir) if output_dir else Path("results") / "smoke" / _timestamp()
        commands.extend(
            [
                SmokeCommand(
                    "custom_experiment",
                    [
                        py,
                        "run_experiments.py",
                        "configs/custom_smoke.yaml",
                        "--output-dir",
                        str(full_output_dir / "custom_smoke"),
                    ],
                ),
                SmokeCommand(
                    "custom_optuna_experiment",
                    [
                        py,
                        "run_experiments.py",
                        "configs/custom_optuna_smoke.yaml",
                        "--output-dir",
                        str(full_output_dir / "custom_optuna_smoke"),
                    ],
                ),
                SmokeCommand(
                    "multi_dataset_optuna_experiment",
                    [
                        py,
                        "run_experiments.py",
                        "configs/multi_dataset_optuna_smoke.yaml",
                        "--output-dir",
                        str(full_output_dir / "multi_dataset_optuna_smoke"),
                    ],
                ),
            ]
        )

    if quality:
        commands.extend(
            [
                SmokeCommand("pytest", [py, "-m", "pytest", "tests"]),
                SmokeCommand("ruff", [py, "-m", "ruff", "check", "."]),
                SmokeCommand(
                    "compileall",
                    [
                        py,
                        "-m",
                        "compileall",
                        "-q",
                        "check_providers.py",
                        "quick_benchmark.py",
                        "run_benchmark.py",
                        "run_comparison.py",
                        "run_experiments.py",
                        "run_optuna.py",
                        "run_smoke.py",
                        "summarize_results.py",
                        "src",
                        "tests",
                    ],
                ),
                SmokeCommand("pip_check", [py, "-m", "pip", "check"]),
            ]
        )

    return commands


def run_command(command: SmokeCommand) -> int:
    print(f"\n[smoke] {command.name}: {' '.join(command.argv)}")
    process = subprocess.run(command.argv, text=True)
    print(f"[smoke] {command.name}: rc={process.returncode}")
    return process.returncode


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    args = parse_args()
    commands = build_smoke_commands(
        full=args.full,
        quality=args.quality,
        output_dir=args.output_dir,
    )

    print(f"Smoke commands: {len(commands)}")
    if args.dry_run:
        for command in commands:
            print(f"{command.name}: {' '.join(command.argv)}")
        return 0

    for command in commands:
        returncode = run_command(command)
        if returncode != 0:
            return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
