"""Interactive wrapper for quick benchmark runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.providers import DEFAULT_EMBEDDING_MODELS, provider_catalog
from src.strategy_registry import parse_strategy_ids, strategy_catalog


@dataclass
class QuickBenchmarkSelection:
    source: str
    dataset_path: str | None
    dataset_name: str | None
    embedding_provider: str
    embedding_model: str
    strategies: str
    top_k: int
    metric_k: int | None


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive quick benchmark launcher")
    parser.add_argument("dataset_path", nargs="?", help="Custom JSON/JSONL dataset path")
    parser.add_argument("--model", type=str, default=None, help="Embedding model")
    parser.add_argument("--provider", type=str, default=None, help="Embedding provider")
    parser.add_argument("--strategies", type=str, default=None, help="Strategy ids, default, or all")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--metric-k", type=int, default=None)
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print generated run_benchmark.py command without executing it.",
    )
    return parser.parse_args()


def discover_datasets(data_dir: str | Path = "data") -> list[Path]:
    """Return local custom dataset candidates."""
    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.glob("*.jsonl")
        if path.name not in {"articles.jsonl", "questions.jsonl"}
    )


def build_benchmark_command(selection: QuickBenchmarkSelection) -> list[str]:
    """Build the low-level benchmark command from a quick selection."""
    argv = [sys.executable, "run_benchmark.py", "--source", selection.source]
    if selection.dataset_path:
        argv.extend(["--dataset-path", selection.dataset_path])
    if selection.dataset_name:
        argv.extend(["--dataset-name", selection.dataset_name])
    argv.extend(
        [
            "--embedding-provider",
            selection.embedding_provider,
            "--embedding-model",
            selection.embedding_model,
            "--strategies",
            ",".join(parse_strategy_ids(selection.strategies)),
            "--top-k",
            str(selection.top_k),
        ]
    )
    if selection.metric_k is not None:
        argv.extend(["--metric-k", str(selection.metric_k)])
    return argv


def interactive_selection(args) -> QuickBenchmarkSelection:
    """Collect benchmark choices interactively."""
    interactive = not args.print_command
    dataset_path = args.dataset_path
    source = "custom" if dataset_path else (_choose_source() if interactive else "static")
    dataset_name = None
    if source == "custom":
        dataset_path = dataset_path or _choose_dataset_path()
        dataset_name = Path(dataset_path).stem

    embedding_provider = args.provider or (_choose_embedding_provider() if interactive else "huggingface")
    embedding_model = args.model or (
        _choose_embedding_model(embedding_provider)
        if interactive
        else DEFAULT_EMBEDDING_MODELS[embedding_provider]
    )
    strategies = args.strategies or (_choose_strategies() if interactive else "default")
    top_k = args.top_k or (_ask_int("Top K", default=5) if interactive else 5)
    metric_k = args.metric_k if args.metric_k is not None else top_k

    return QuickBenchmarkSelection(
        source=source,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        strategies=strategies,
        top_k=top_k,
        metric_k=metric_k,
    )


def _choose_source() -> str:
    return _choose("Dataset source", ["custom", "static"], default_index=0)


def _choose_dataset_path() -> str:
    datasets = discover_datasets()
    if not datasets:
        return _ask_text("Dataset path", required=True)

    options = [path.as_posix() for path in datasets] + ["enter another path"]
    choice = _choose("Dataset", options, default_index=0)
    if choice == "enter another path":
        return _ask_text("Dataset path", required=True)
    return choice


def _choose_embedding_provider() -> str:
    providers = [item["id"] for item in provider_catalog()["embeddings"]]
    preferred = providers.index("huggingface") if "huggingface" in providers else 0
    return _choose("Embedding provider", providers, default_index=preferred)


def _choose_embedding_model(provider: str) -> str:
    default_model = DEFAULT_EMBEDDING_MODELS[provider]
    return _ask_text(f"Embedding model for {provider}", default=default_model)


def _choose_strategies() -> str:
    preset = _choose("Strategies", ["default", "all", "custom"], default_index=0)
    if preset != "custom":
        return preset

    strategies = strategy_catalog()
    print("\nAvailable strategies:")
    for index, strategy in enumerate(strategies, start=1):
        print(f"  {index}. {strategy['id']} - {strategy['description']}")
    return _ask_text(
        "Comma-separated strategy ids",
        default="naive,token_text,dynamic_semantic",
    )


def _choose(label: str, options: list[str], default_index: int = 0) -> str:
    print(f"\n{label}:")
    for index, option in enumerate(options, start=1):
        suffix = " [default]" if index - 1 == default_index else ""
        print(f"  {index}. {option}{suffix}")
    while True:
        raw = input(f"{label} [{default_index + 1}]: ").strip()
        if not raw:
            return options[default_index]
        try:
            index = int(raw)
        except ValueError:
            if raw in options:
                return raw
            print("Enter a number from the list or an exact option value.")
            continue
        if 1 <= index <= len(options):
            return options[index - 1]
        print("Enter a valid option number.")


def _ask_text(label: str, default: str | None = None, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("Value is required.")


def _ask_int(label: str, default: int) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Enter an integer.")
            continue
        if value > 0:
            return value
        print("Enter a positive integer.")


def main() -> int:
    args = parse_args()
    selection = interactive_selection(args)
    command = build_benchmark_command(selection)
    print("\nCommand:")
    print(" ".join(command))
    if args.print_command:
        return 0

    confirm = input("\nRun benchmark? [Y/n]: ").strip().lower()
    if confirm in {"n", "no"}:
        return 0
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
