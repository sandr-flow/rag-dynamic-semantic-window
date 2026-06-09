"""Run benchmark/Optuna experiment matrices from YAML or JSON config."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.experiment_config import build_commands, load_experiment_config, validate_experiment_config
from src.result_summary import (
    build_leaderboard,
    summarize_files,
    write_leaderboard_csv,
    write_summary_csv,
    write_summary_jsonl,
)

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Run configured experiment matrix")
    parser.add_argument("config", type=str, help="Path to YAML/JSON experiment config")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for experiment logs and manifest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated commands without executing them",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and print generated command names without writing outputs.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running later commands after a failure",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    issues = validate_experiment_config(config, base_dir=Path(__file__).parent)
    if issues:
        for issue in issues:
            print(f"[{issue.level.upper()}] {issue.message}")
        if any(issue.level == "error" for issue in issues):
            return 2

    commands = build_commands(config)

    experiment_name = config.get("name") or Path(args.config).stem
    if args.validate_only:
        print(f"Experiment config OK: {experiment_name}")
        print(f"Commands: {len(commands)}")
        for index, command in enumerate(commands, start=1):
            print(f"[{index}] {command.name}: {' '.join(command.argv)}")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or Path("results") / "experiments" / f"{experiment_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": experiment_name,
        "config_path": str(Path(args.config)),
        "config_sha256": _config_sha256(config),
        "started_at": datetime.now().isoformat(),
        "dry_run": args.dry_run,
        "continue_on_failure": args.continue_on_failure,
        "command_count": len(commands),
        "runner": {
            "cwd": str(Path(__file__).parent),
            "python": sys.executable,
        },
        "runs": [],
    }

    print(f"Experiment: {experiment_name}")
    print(f"Output: {output_dir}")
    print(f"Commands: {len(commands)}")

    for index, command in enumerate(commands, start=1):
        run_id = f"{index:03d}_{command.name}"
        log_path = output_dir / f"{run_id}.log"
        print(f"\n[{index}/{len(commands)}] {' '.join(command.argv)}")
        benchmark_files_before = _benchmark_result_files()

        run_record = {
            "id": run_id,
            "name": command.name,
            "argv": command.argv,
            "env": command.env,
            "log_path": str(log_path),
            "expected_artifacts": _expected_artifacts(command.argv),
            "status": "dry_run" if args.dry_run else "pending",
        }

        if args.dry_run:
            manifest["runs"].append(run_record)
            continue

        env = os.environ.copy()
        env.update(command.env)
        start = time.time()
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command.argv,
                cwd=Path(__file__).parent,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

        run_record["returncode"] = process.returncode
        run_record["elapsed_seconds"] = round(time.time() - start, 3)
        run_record["status"] = "ok" if process.returncode == 0 else "failed"
        if "run_benchmark.py" in command.argv:
            new_results = sorted(_benchmark_result_files() - benchmark_files_before)
            run_record["benchmark_results"] = [str(path) for path in new_results]
        manifest["runs"].append(run_record)

        print(f"Status: {run_record['status']} ({run_record['elapsed_seconds']}s)")
        if process.returncode != 0 and not args.continue_on_failure:
            break

    manifest["finished_at"] = datetime.now().isoformat()

    benchmark_result_paths = [
        path
        for run in manifest["runs"]
        for path in run.get("benchmark_results", [])
    ]
    if benchmark_result_paths:
        rows = summarize_files(benchmark_result_paths)
        summary_csv = output_dir / "benchmark_summary.csv"
        summary_jsonl = output_dir / "benchmark_summary.jsonl"
        write_summary_csv(rows, summary_csv)
        write_summary_jsonl(rows, summary_jsonl)
        leaderboard_csv = output_dir / "benchmark_leaderboard.csv"
        write_leaderboard_csv(build_leaderboard(rows), leaderboard_csv)
        grouped_leaderboard_csv = output_dir / "benchmark_leaderboard_by_dataset.csv"
        write_leaderboard_csv(
            build_leaderboard(rows, group_by=DEFAULT_LEADERBOARD_GROUP_BY),
            grouped_leaderboard_csv,
        )
        manifest["benchmark_summary"] = {
            "csv": str(summary_csv),
            "jsonl": str(summary_jsonl),
            "leaderboard_csv": str(leaderboard_csv),
            "grouped_leaderboard_csv": str(grouped_leaderboard_csv),
            "rows": len(rows),
        }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest: {manifest_path}")
    failed = [run for run in manifest["runs"] if run["status"] == "failed"]
    return 1 if failed else 0


def _benchmark_result_files() -> set[Path]:
    return set((Path(__file__).parent / "results").glob("benchmark_*.json"))


def _config_sha256(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expected_artifacts(argv: list[str]) -> dict[str, list[str]]:
    """Return output paths implied by a generated command."""
    command_name = Path(argv[1]).name if len(argv) > 1 else ""
    artifacts: dict[str, list[str]] = {}
    if command_name == "prepare_corpus.py":
        _add_flag_artifact(artifacts, "corpus", argv, "--corpus-path")
        _add_flag_artifact(artifacts, "articles", argv, "--articles-output-path")
        _add_flag_artifact(artifacts, "questions", argv, "--questions-output-path")
    elif command_name == "run_optuna.py":
        _add_flag_artifact(artifacts, "best_params_json", argv, "--best-params-json")
        _add_flag_artifact(artifacts, "best_params_txt", argv, "--best-params-txt")
        _add_flag_artifact(artifacts, "output_dir", argv, "--output-dir")
        storage_path = _sqlite_storage_artifact(_flag_value(argv, "--storage"))
        if storage_path:
            artifacts["storage"] = [storage_path]
    elif command_name == "run_benchmark.py":
        artifacts["benchmark_results"] = ["results/benchmark_*.json"]
    return artifacts


def _add_flag_artifact(
    artifacts: dict[str, list[str]],
    key: str,
    argv: list[str],
    flag: str,
) -> None:
    value = _flag_value(argv, flag)
    if value:
        artifacts[key] = [value]


def _flag_value(argv: list[str], flag: str) -> str | None:
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(argv):
        return None
    return argv[value_index]


def _sqlite_storage_artifact(storage: str | None) -> str | None:
    if not storage or not storage.startswith("sqlite:///"):
        return None
    path = storage.removeprefix("sqlite:///")
    if not path or path == ":memory:":
        return None
    return path


DEFAULT_LEADERBOARD_GROUP_BY = [
    "dataset_name",
    "embedding_provider",
    "embedding_model",
    "llm_provider",
    "llm_model",
]


if __name__ == "__main__":
    raise SystemExit(main())
