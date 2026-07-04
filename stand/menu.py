"""Interactive console menu — the primary entry point.

No flags: it lists prepared artifacts and walks you through one combination,
then runs it in-process. Everything offered here must already be prepared.
"""

from __future__ import annotations

from src.strategy_registry import ALL_STRATEGY_IDS, DEFAULT_STRATEGY_IDS, parse_strategy_ids

from . import artifacts
from .runconfig import RunConfig
from .runner import run


def _choose(label: str, options: list[str], default_index: int = 0) -> int:
    """Prompt to pick one option; return its index."""
    print(f"\n{label}:")
    for i, option in enumerate(options):
        marker = " [default]" if i == default_index else ""
        print(f"  {i + 1}. {option}{marker}")
    while True:
        raw = input(f"> [{default_index + 1}]: ").strip()
        if not raw:
            return default_index
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  Enter a number 1-{len(options)}.")


def _ask_int(label: str, default: int) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("  Enter a positive integer.")


def _choose_strategies() -> list[str]:
    presets = ["default", "all", "custom"]
    choice = presets[_choose("Strategy set", presets, default_index=0)]
    if choice == "default":
        return list(DEFAULT_STRATEGY_IDS)
    if choice == "all":
        return list(ALL_STRATEGY_IDS)
    print("  Available:", ", ".join(ALL_STRATEGY_IDS))
    raw = input("  Comma-separated ids [naive,dynamic_semantic]: ").strip()
    return parse_strategy_ids(raw or "naive,dynamic_semantic")


def run_menu() -> int:
    """Drive a single benchmark run via interactive prompts."""
    print("=" * 64)
    print("Dynamic Semantic Window — benchmark stand")
    print("=" * 64)

    datasets = artifacts.list_datasets()
    if not datasets:
        print("\nNo prepared datasets yet. Create one, e.g.:")
        print("  python -m stand prepare-dataset --source qasper --name qasper_val "
              "--num-articles 5 --qa-provider mistral")
        return 1
    dataset = datasets[_choose("Dataset", [d.label() for d in datasets])].name

    embeddings = artifacts.list_embeddings()
    embedding = embeddings[_choose("Embedding model", [e.label() for e in embeddings])].name

    strategies = _choose_strategies()

    modes = ["shared", "per_document"]
    index_mode = modes[_choose(
        "Index mode",
        ["shared (one corpus collection, primary)",
         "per_document (per-doc collection, diagnostic)"],
    )]

    if artifacts.has_tuned(dataset, embedding):
        param_opts = ["default", "tuned (per-domain best params)"]
        params = ["default", "tuned"][_choose("Dynamic params", param_opts, default_index=1)]
    else:
        params = "default"
        print("\nDynamic params: default (no tuned artifact for this dataset+embedding)")

    top_k = _ask_int("\nTop-K", default=5)

    config = RunConfig(
        dataset=dataset,
        embedding=embedding,
        strategies=strategies,
        index_mode=index_mode,
        params=params,
        top_k=top_k,
    )

    print("\nReady to run:")
    print(f"  dataset={dataset}  embedding={embedding}  mode={index_mode}  params={params}")
    print(f"  strategies={', '.join(strategies)}  top_k={top_k}")
    if input("Run? [Y/n]: ").strip().lower() in {"n", "no"}:
        return 0

    print()
    run(config)
    return 0
