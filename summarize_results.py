"""Summarize benchmark result files into CSV/JSONL."""

import argparse
from pathlib import Path

from src.result_summary import (
    build_leaderboard,
    summarize_files,
    write_leaderboard_csv,
    write_summary_csv,
    write_summary_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize benchmark JSON results")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Benchmark JSON files. If omitted, scans --results-dir.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to scan for benchmark_*.json when paths are omitted.",
    )
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--jsonl", type=str, default=None)
    parser.add_argument("--leaderboard-csv", type=str, default=None)
    parser.add_argument("--grouped-leaderboard-csv", type=str, default=None)
    parser.add_argument(
        "--leaderboard-group-by",
        type=str,
        default="dataset_name,embedding_provider,embedding_model,llm_provider,llm_model",
        help="Comma-separated columns for grouped leaderboard ranks.",
    )
    parser.add_argument(
        "--rank-metric",
        type=str,
        default=None,
        help="Metric column to rank by. Defaults to avg_ndcg@metric_k, then HR/MRR fallback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [Path(path) for path in args.paths]
    if not paths:
        paths = sorted(Path(args.results_dir).glob("benchmark_*.json"))

    rows = summarize_files(paths)
    leaderboard = build_leaderboard(rows, quality_metric=args.rank_metric)
    group_by = [item.strip() for item in args.leaderboard_group_by.split(",") if item.strip()]
    grouped_leaderboard = build_leaderboard(rows, quality_metric=args.rank_metric, group_by=group_by)
    output_all_defaults = not args.csv and not args.jsonl and not args.leaderboard_csv and not args.grouped_leaderboard_csv
    csv_path = args.csv or ("results/benchmark_summary.csv" if output_all_defaults else None)
    jsonl_path = args.jsonl or ("results/benchmark_summary.jsonl" if output_all_defaults else None)
    leaderboard_path = args.leaderboard_csv or (
        "results/benchmark_leaderboard.csv" if output_all_defaults else None
    )
    grouped_leaderboard_path = args.grouped_leaderboard_csv or (
        "results/benchmark_leaderboard_by_dataset.csv" if output_all_defaults else None
    )

    if csv_path:
        write_summary_csv(rows, csv_path)
    if jsonl_path:
        write_summary_jsonl(rows, jsonl_path)
    if leaderboard_path:
        write_leaderboard_csv(leaderboard, leaderboard_path)
    if grouped_leaderboard_path:
        write_leaderboard_csv(grouped_leaderboard, grouped_leaderboard_path)

    print(f"Files: {len(paths)}")
    print(f"Rows: {len(rows)}")
    if csv_path:
        print(f"CSV: {csv_path}")
    if jsonl_path:
        print(f"JSONL: {jsonl_path}")
    if leaderboard_path:
        print(f"Leaderboard CSV: {leaderboard_path}")
    if grouped_leaderboard_path:
        print(f"Grouped leaderboard CSV: {grouped_leaderboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
