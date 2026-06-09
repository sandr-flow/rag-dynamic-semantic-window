"""Run Optuna hyperparameter optimization for Dynamic Semantic Expander.

This script loads a pre-computed corpus and runs Optuna trials to find
optimal hyperparameters. All similarity computations are pre-cached,
so trials run extremely fast (no model.encode() calls).

Usage:
    python run_optuna.py [--n-trials 200] [--study-name dynamic_semantic_hpo]

Prerequisites:
    Run prepare_corpus.py first to create the cached corpus.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

try:
    import optuna
except ImportError:
    print("[ERROR] Optuna not installed. Run: pip install optuna")
    exit(1)

from src.cached_expander import CachedDynamicExpander, evaluate_retrieval
from src.config import DEFAULT_CORPUS_CONFIG, DEFAULT_OPTUNA_CONFIG
from src.corpus_data import CorpusData
from src.hpo_config import HPOSettings, load_hpo_settings, suggest_params


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_OPTUNA_CONFIG.n_trials,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=DEFAULT_OPTUNA_CONFIG.study_name,
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=DEFAULT_CORPUS_CONFIG.corpus_cache_path,
        help="Path to cached corpus",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="dynamic_semantic",
        help="Strategy to optimize. Cached Optuna currently supports dynamic_semantic.",
    )
    parser.add_argument(
        "--target-clusters",
        type=int,
        default=5,
        help="Target number of retrieved clusters/chunks.",
    )
    parser.add_argument(
        "--soft-token-limit",
        type=int,
        default=1200,
        help="Soft average-token limit used by the objective.",
    )
    parser.add_argument(
        "--hpo-config",
        type=str,
        default=None,
        help="YAML/JSON file with search_space and objective policy overrides.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_study.db",
        help="SQLite storage path for persistence (default: sqlite:///optuna_study.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for best params and visualization artifacts. Defaults to results/.",
    )
    parser.add_argument(
        "--best-params-json",
        type=str,
        default=None,
        help="Path for JSON best params artifact. Defaults to <output-dir>/best_params.json.",
    )
    parser.add_argument(
        "--best-params-txt",
        type=str,
        default=None,
        help="Path for text best params artifact. Defaults to <output-dir>/best_params.txt.",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip parameter importance visualization",
    )
    return parser.parse_args()


def load_corpus(corpus_path: str) -> CorpusData:
    """Load pre-computed corpus from disk."""
    with open(corpus_path, "rb") as f:
        return pickle.load(f)


def ensure_sqlite_storage_parent(storage: str) -> None:
    """Create parent directory for sqlite:///path storage URIs."""
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return
    db_path = storage.removeprefix(prefix)
    if not db_path or db_path == ":memory:":
        return
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def create_objective(
    corpus: CorpusData,
    target_clusters: int = 5,
    hpo_settings: HPOSettings | None = None,
):
    """
    Create Optuna objective function with corpus closure.
    
    The objective evaluates hyperparameters by running the cached expander
    on all questions and computing average Hit Rate and token count.
    
    Args:
        corpus: Pre-computed corpus data.
        optuna_config: Optuna configuration with search ranges.
    
    Returns:
        Objective function for Optuna.
    """
    # Build article lookup
    article_lookup = {a.article_id: a for a in corpus.articles}
    hpo_settings = hpo_settings or load_hpo_settings()
    objective_policy = hpo_settings.objective
    
    def objective(trial: optuna.Trial) -> float:
        """Evaluate a single set of hyperparameters."""
        params = suggest_params(trial, hpo_settings.search_space)
        threshold = params["threshold"]
        skip_threshold = params["skip_threshold"]
        relevance_threshold_pct = params["relevance_threshold_pct"]
        min_window = params["min_window"]
        max_expand = params["max_expand"]
        merge_gap = params["merge_gap"]
        
        # Evaluate on all questions
        total_hits = 0
        total_mrr = 0.0  # Mean Reciprocal Rank
        total_tokens = 0
        num_valid_questions = 0
        
        for question in corpus.questions:
            article = article_lookup.get(question.article_id)
            if not article:
                continue
            
            # Create cached expander for this question
            expander = CachedDynamicExpander(
                neighbor_sims=article.neighbor_sims,
                sentence_sims=question.sentence_sims,
                top_k_indices=question.top_k_indices,
                threshold=threshold,
                skip_threshold=skip_threshold,
                min_window=min_window,
                max_expand=max_expand,
                relevance_threshold_pct=relevance_threshold_pct,
                merge_gap=merge_gap,
                target_clusters=target_clusters,
            )
            
            # Expand and evaluate
            clusters = expander.expand_and_retrieve()
            metrics = evaluate_retrieval(
                clusters=clusters,
                answer_sentence_idx=question.answer_sentence_idx,
                sentences=article.sentences,
            )
            
            if question.answer_sentence_idx >= 0:
                total_hits += int(metrics["hit"])
                # MRR: 1/rank if hit, 0 otherwise
                if metrics["hit"] and metrics["rank"] > 0:
                    total_mrr += 1.0 / metrics["rank"]
                num_valid_questions += 1
            total_tokens += metrics["tokens"]
        
        # Compute averages
        if num_valid_questions == 0:
            return objective_policy.invalid_score
        
        avg_hr = total_hits / num_valid_questions
        avg_mrr = total_mrr / num_valid_questions
        avg_tokens = total_tokens / len(corpus.questions) if corpus.questions else 0
        
        # Primary: HR (higher is better) - scaled 0-100
        score = avg_hr * objective_policy.hr_weight
        
        # Secondary: MRR bonus - up to 10
        score += avg_mrr * objective_policy.mrr_weight
        
        # Token efficiency bonus/penalty
        if avg_tokens <= objective_policy.soft_token_limit:
            # Bonus for being under limit (up to 5)
            token_bonus = (
                (objective_policy.soft_token_limit - avg_tokens)
                / objective_policy.soft_token_limit
                * objective_policy.token_bonus_weight
            )
            score += token_bonus
        else:
            # Soft penalty for being over limit
            excess = avg_tokens - objective_policy.soft_token_limit
            token_penalty = excess * objective_policy.token_penalty_per_token
            score -= token_penalty
        
        # Log intermediate values for analysis
        trial.set_user_attr("hit_rate", avg_hr)
        trial.set_user_attr("mrr", avg_mrr)
        trial.set_user_attr("avg_tokens", avg_tokens)
        trial.set_user_attr("score", score)
        trial.set_user_attr("tokens_ok", avg_tokens <= objective_policy.soft_token_limit)
        
        return score
    
    return objective


def run_optuna() -> int:
    """Main Optuna optimization loop."""
    args = parse_args()
    
    print("=" * 60)
    print("Optuna Hyperparameter Optimization")
    print("=" * 60)

    if args.strategy.replace("-", "_") not in {"dynamic", "dynamic_semantic"}:
        raise ValueError("Cached Optuna currently supports only dynamic_semantic")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be >= 1")

    hpo_settings = load_hpo_settings(
        path=args.hpo_config,
        soft_token_limit=args.soft_token_limit,
    )
    
    # Load corpus
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        raise ValueError(f"Corpus not found: {corpus_path}. Run prepare_corpus.py first.")
    
    print(f"\n[INFO] Loading corpus from {corpus_path}...")
    corpus = load_corpus(str(corpus_path))
    print(f"   [OK] Loaded {len(corpus.articles)} articles, {len(corpus.questions)} questions")
    print(f"   Source: {getattr(corpus, 'source', 'unknown')}")
    print(f"   Embeddings: {getattr(corpus, 'embedding_provider', 'unknown')}/{getattr(corpus, 'embedding_model', 'unknown')}")
    if args.hpo_config:
        print(f"   HPO config: {args.hpo_config}")
    print(f"   Objective soft token limit: {hpo_settings.objective.soft_token_limit}")
    
    # Count valid questions (with answer index found)
    valid_questions = sum(1 for q in corpus.questions if q.answer_sentence_idx >= 0)
    print(f"   [OK] Valid questions (answer found): {valid_questions}/{len(corpus.questions)}")
    
    # Create study
    storage = args.storage
    if storage:
        ensure_sqlite_storage_parent(storage)
        print(f"\n[INFO] Using persistent storage: {storage}")
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    
    # Create objective
    objective = create_objective(
        corpus,
        target_clusters=args.target_clusters,
        hpo_settings=hpo_settings,
    )
    
    # Run optimization
    print(f"\n[INFO] Running {args.n_trials} trials...")
    start_time = time.time()
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 60)
    print("[OK] Optimization complete!")
    print("=" * 60)
    
    best = study.best_trial
    tokens_ok = best.user_attrs.get('tokens_ok', False)
    print(f"\nBest trial #{best.number}:")
    print(f"   Score: {best.value:.4f}")
    print(f"   Hit Rate: {best.user_attrs.get('hit_rate', 0):.4f}")
    print(f"   MRR: {best.user_attrs.get('mrr', 0):.4f}")
    print(f"   Avg Tokens: {best.user_attrs.get('avg_tokens', 0):.1f} (soft limit: {hpo_settings.objective.soft_token_limit}) {'[OK]' if tokens_ok else '[WARN]'}")
    
    print("\nBest hyperparameters:")
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/args.n_trials:.2f}s per trial)")
    
    # Save best params
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = Path(args.best_params_txt) if args.best_params_txt else output_dir / "best_params.txt"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, "w") as f:
        f.write(f"# Best Optuna trial #{best.number}\n")
        f.write(f"# Score: {best.value:.4f}\n")
        f.write(f"# Hit Rate: {best.user_attrs.get('hit_rate', 0):.4f}\n")
        f.write(f"# MRR: {best.user_attrs.get('mrr', 0):.4f}\n")
        f.write(f"# Avg Tokens: {best.user_attrs.get('avg_tokens', 0):.1f}\n")
        f.write(f"# Tokens OK: {best.user_attrs.get('tokens_ok', False)}\n\n")
        for key, value in best.params.items():
            f.write(f"{key} = {value}\n")
    
    print(f"\n[INFO] Best params saved to: {params_path}")

    params_json_path = Path(args.best_params_json) if args.best_params_json else output_dir / "best_params.json"
    params_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "strategy": "dynamic_semantic",
                "study_name": args.study_name,
                "trial_number": best.number,
                "score": best.value,
                "metrics": {
                    "hit_rate": best.user_attrs.get("hit_rate", 0),
                    "mrr": best.user_attrs.get("mrr", 0),
                    "avg_tokens": best.user_attrs.get("avg_tokens", 0),
                    "tokens_ok": best.user_attrs.get("tokens_ok", False),
                },
                "objective": {
                    "target_clusters": args.target_clusters,
                    "soft_token_limit": hpo_settings.objective.soft_token_limit,
                },
                "hpo": hpo_settings.to_jsonable(),
                "corpus": {
                    "path": str(corpus_path),
                    "source": getattr(corpus, "source", "unknown"),
                    "embedding_provider": getattr(corpus, "embedding_provider", "unknown"),
                    "embedding_model": getattr(corpus, "embedding_model", "unknown"),
                    "articles": len(corpus.articles),
                    "questions": len(corpus.questions),
                },
                "params": best.params,
            },
            f,
            indent=2,
        )
    print(f"   [OK] JSON saved to: {params_json_path}")
    
    # Parameter importance visualization
    if not args.no_viz:
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            print("\n[INFO] Generating parameter importance plot...")
            
            # Parameter importance
            fig_importance = plot_param_importances(study)
            importance_path = output_dir / "param_importance.html"
            fig_importance.write_html(str(importance_path))
            print(f"   [OK] Saved to: {importance_path}")
            
            # Optimization history
            fig_history = plot_optimization_history(study)
            history_path = output_dir / "optimization_history.html"
            fig_history.write_html(str(history_path))
            print(f"   [OK] Saved to: {history_path}")
            
        except ImportError:
            print("\n[WARN] Install plotly for visualizations: pip install plotly")
        except Exception as e:
            print(f"\n[WARN] Could not generate visualizations: {e}")
    
    # Print suggested config update
    print("\nSuggested config.py update:")
    print("```python")
    print("@dataclass")
    print("class ExpansionConfig:")
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f"    {key}: float = {value:.4f}")
        else:
            print(f"    {key}: int = {value}")
    print("```")
    return 0


def main() -> int:
    """CLI entrypoint with concise configuration errors."""
    try:
        return run_optuna()
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
