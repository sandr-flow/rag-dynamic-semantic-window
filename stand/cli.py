"""Thin dispatcher: interactive menu by default, scriptable subcommands for CI.

    python -m stand                      # interactive menu (primary entry point)
    python -m stand list                 # show prepared artifacts
    python -m stand prepare-dataset ...  # build a dataset artifact
    python -m stand prepare-embedding ...# register/warm an embedding model
    python -m stand tune ...             # per-domain Optuna tuning
    python -m stand run ...              # run one combination non-interactively
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from . import artifacts, paths

load_dotenv()

# Windows consoles often default to cp1252; Wikipedia titles are UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _cmd_list(_: argparse.Namespace) -> int:
    datasets = artifacts.list_datasets()
    embeddings = artifacts.list_embeddings()
    print("Datasets:")
    for info in datasets or []:
        print(f"  - {info.label()}")
    if not datasets:
        print("  (none — run: python -m stand prepare-dataset ...)")
    print("\nEmbedding models:")
    for info in embeddings:
        print(f"  - {info.label()}")
    print("\nTuned params:")
    tuned = sorted(paths.TUNED_DIR.glob("*/manifest.json")) if paths.TUNED_DIR.exists() else []
    for path in tuned:
        print(f"  - {path.parent.name}")
    if not tuned:
        print("  (none — run: python -m stand tune ...)")
    return 0


def _cmd_prepare_dataset(args: argparse.Namespace) -> int:
    from .prepare import prepare_dataset

    prepare_dataset(
        source=args.source,
        name=args.name,
        num_articles=args.num_articles,
        min_length=args.min_length,
        questions_per_article=args.questions_per_article,
        split=args.split,
        skip=args.skip,
        qa_provider=args.qa_provider,
        qa_model=args.qa_model,
        qa_delay=args.qa_delay,
        dataset_path=args.dataset_path,
    )
    return 0


def _cmd_harden_dataset(args: argparse.Namespace) -> int:
    from .prepare import harden_dataset

    harden_dataset(
        source_name=args.source_name,
        target_name=args.name,
        qa_provider=args.qa_provider,
        qa_model=args.qa_model,
        qa_delay=args.qa_delay,
        max_overlap=args.max_overlap,
    )
    return 0


def _cmd_extrahard_dataset(args: argparse.Namespace) -> int:
    from .prepare import extrahard_dataset

    extrahard_dataset(
        source_name=args.source_name,
        target_name=args.name,
        partners_per_question=args.partners_per_question,
        pair_seed=args.seed,
    )
    return 0


def _cmd_prepare_embedding(args: argparse.Namespace) -> int:
    from .prepare import prepare_embedding

    prepare_embedding(
        name=args.name,
        provider=args.provider,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        warm=not args.no_warm,
    )
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    from .tune import tune

    tune(
        args.dataset,
        args.embedding,
        n_trials=args.n_trials,
        soft_token_limit=args.soft_token_limit,
        rebuild_corpus=args.rebuild_corpus,
        phantom_window=args.phantom_window,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        hpo_config=args.hpo_config,
        index_mode=args.index_mode,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    import json

    from src.strategy_registry import parse_strategy_ids

    from .runconfig import RunConfig
    from .runner import run

    dynamic_overrides = {}
    if args.dynamic_overrides:
        dynamic_overrides = json.loads(args.dynamic_overrides)
        if not isinstance(dynamic_overrides, dict):
            raise ValueError("--dynamic-overrides must be a JSON object")

    config = RunConfig(
        dataset=args.dataset,
        embedding=args.embedding,
        strategies=parse_strategy_ids(args.strategies),
        index_mode=args.index_mode,
        params=args.params,
        top_k=args.top_k,
        metric_k=args.metric_k,
        limit=args.limit,
        tuned_dataset=args.tuned_dataset,
        dynamic_overrides=dynamic_overrides,
    )
    run(config)
    return 0


def _cmd_significance(args: argparse.Namespace) -> int:
    from src.significance import format_comparisons

    from .analysis import significance_report

    comparisons = significance_report(
        args.result,
        baseline_result_path=args.baseline_result,
        target=args.target,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    if not comparisons:
        print("No baselines to compare against.")
        return 0
    for line in format_comparisons(comparisons, n_resamples=args.resamples):
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stand", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List prepared datasets, embeddings, and tuned params.")

    p_ds = sub.add_parser("prepare-dataset", help="Build a reusable dataset artifact.")
    p_ds.add_argument(
        "--source", choices=["wikipedia", "qasper", "financebench", "custom"], required=True
    )
    p_ds.add_argument("--name", required=True, help="Artifact name.")
    p_ds.add_argument(
        "--num-articles",
        type=int,
        default=None,
        help="Documents to fetch (default: 5; financebench defaults to all filings).",
    )
    p_ds.add_argument("--min-length", type=int, default=2000)
    p_ds.add_argument("--questions-per-article", type=int, default=3)
    p_ds.add_argument("--split", default="validation", help="QASPER split.")
    p_ds.add_argument("--skip", type=int, default=0)
    p_ds.add_argument("--qa-provider", default=None, help="LLM provider for QA generation.")
    p_ds.add_argument("--qa-model", default=None, help="LLM model for QA generation.")
    p_ds.add_argument("--qa-delay", type=float, default=1.1)
    p_ds.add_argument("--dataset-path", default=None, help="Source file for --source custom.")

    p_hard = sub.add_parser(
        "harden-dataset",
        help="Paraphrase questions of an existing dataset away from answer lexis.",
    )
    p_hard.add_argument("--source-name", required=True, help="Existing dataset artifact.")
    p_hard.add_argument("--name", required=True, help="Name for the hardened artifact.")
    p_hard.add_argument("--qa-provider", default=None, help="LLM provider for paraphrasing.")
    p_hard.add_argument("--qa-model", default=None, help="LLM model for paraphrasing.")
    p_hard.add_argument("--qa-delay", type=float, default=1.1)
    p_hard.add_argument(
        "--max-overlap",
        type=float,
        default=0.35,
        help="Max content-token overlap between question and answer sentence.",
    )

    p_extra = sub.add_parser(
        "extrahard-dataset",
        help="Build cross-document compound questions from an existing dataset.",
    )
    p_extra.add_argument("--source-name", required=True, help="Existing QA dataset artifact.")
    p_extra.add_argument("--name", required=True, help="Name for the extrahard artifact.")
    p_extra.add_argument(
        "--partners-per-question",
        type=int,
        default=2,
        help="Random cross-doc partners sampled per base question.",
    )
    p_extra.add_argument("--seed", type=int, default=42, help="RNG seed for pair sampling.")

    p_emb = sub.add_parser("prepare-embedding", help="Register/warm an embedding model.")
    p_emb.add_argument("--name", required=True)
    p_emb.add_argument("--provider", required=True)
    p_emb.add_argument("--model", default=None)
    p_emb.add_argument("--api-key-env", default=None)
    p_emb.add_argument("--base-url", default=None)
    p_emb.add_argument("--no-warm", action="store_true", help="Register without downloading weights.")

    p_tune = sub.add_parser("tune", help="Per-domain Optuna tuning of dynamic_semantic.")
    p_tune.add_argument("--dataset", required=True)
    p_tune.add_argument("--embedding", required=True)
    p_tune.add_argument("--n-trials", type=int, default=100)
    p_tune.add_argument("--soft-token-limit", type=int, default=1200)
    p_tune.add_argument("--rebuild-corpus", action="store_true")
    p_tune.add_argument("--phantom-window", type=int, default=1)
    p_tune.add_argument("--train-ratio", type=float, default=0.70)
    p_tune.add_argument("--split-seed", type=int, default=42)
    p_tune.add_argument(
        "--hpo-config",
        default=None,
        help="YAML/JSON file with search_space and objective policy overrides.",
    )
    p_tune.add_argument(
        "--index-mode",
        choices=["per_document", "shared"],
        default="per_document",
        help=(
            "Retrieval mode the objective is computed in. Use 'shared' to tune "
            "under the same cross-document competition the shared benchmark "
            "measures. Extrahard datasets are always shared."
        ),
    )

    p_run = sub.add_parser("run", help="Run one combination non-interactively.")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--embedding", default="mock")
    p_run.add_argument("--strategies", default="default")
    p_run.add_argument("--index-mode", choices=["shared", "per_document"], default="shared")
    p_run.add_argument("--params", choices=["default", "tuned"], default="default")
    p_run.add_argument(
        "--tuned-dataset",
        default=None,
        help="With --params tuned: load tuned artifact from this dataset name "
        "instead of --dataset (same --embedding).",
    )
    p_run.add_argument("--top-k", type=int, default=5)
    p_run.add_argument("--metric-k", type=int, default=None)
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument(
        "--dynamic-overrides",
        default=None,
        help="JSON object with dynamic_semantic param overrides applied on top "
        'of --params, e.g. \'{"min_window": 0, "max_expand": 0}\' for ablations.',
    )

    p_sig = sub.add_parser(
        "significance",
        help="Paired bootstrap CIs (dHR/dMRR vs baselines) from saved result JSONs.",
    )
    p_sig.add_argument("result", help="Result JSON containing the target strategy rows.")
    p_sig.add_argument(
        "--baseline-result",
        default=None,
        help="Optional result JSON supplying baseline rows (same dataset, "
        "index mode, and question order).",
    )
    p_sig.add_argument("--target", default="Dynamic Semantic")
    p_sig.add_argument("--resamples", type=int, default=10000)
    p_sig.add_argument("--seed", type=int, default=42)

    return parser


_HANDLERS = {
    "list": _cmd_list,
    "prepare-dataset": _cmd_prepare_dataset,
    "harden-dataset": _cmd_harden_dataset,
    "extrahard-dataset": _cmd_extrahard_dataset,
    "prepare-embedding": _cmd_prepare_embedding,
    "tune": _cmd_tune,
    "run": _cmd_run,
    "significance": _cmd_significance,
}


def main(argv: list[str] | None = None) -> int:
    paths.ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        from .menu import run_menu

        try:
            return run_menu()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return 130

    try:
        return _HANDLERS[args.command](args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
