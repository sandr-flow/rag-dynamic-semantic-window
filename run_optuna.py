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
import pickle
import time
from pathlib import Path



try:
    import optuna
except ImportError:
    print("❌ Optuna not installed. Run: pip install optuna")
    exit(1)

from src.cached_expander import CachedDynamicExpander, evaluate_retrieval
from src.config import DEFAULT_CORPUS_CONFIG, DEFAULT_OPTUNA_CONFIG
from src.corpus_data import CorpusData


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
        "--storage",
        type=str,
        default="sqlite:///optuna_study.db",
        help="SQLite storage path for persistence (default: sqlite:///optuna_study.db)",
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


def create_objective(corpus: CorpusData, optuna_config=DEFAULT_OPTUNA_CONFIG):
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
    
    def objective(trial: optuna.Trial) -> float:
        """Evaluate a single set of hyperparameters."""
        
        # Threshold parameters (tunable)
        threshold = trial.suggest_float(
            "threshold",
            optuna_config.threshold_range[0],
            optuna_config.threshold_range[1],
            step=0.001,
        )
        skip_threshold = trial.suggest_float(
            "skip_threshold",
            optuna_config.skip_threshold_range[0],
            optuna_config.skip_threshold_range[1],
            step=0.001,
        )
        relevance_threshold_pct = trial.suggest_float(
            "relevance_threshold_pct",
            optuna_config.relevance_threshold_pct_range[0],
            optuna_config.relevance_threshold_pct_range[1],
            step=0.001,
        )
        
        # Window parameters (tunable)
        min_window = trial.suggest_int(
            "min_window",
            optuna_config.min_window_range[0],
            optuna_config.min_window_range[1],
        )
        max_expand = trial.suggest_int(
            "max_expand",
            optuna_config.max_expand_range[0],
            optuna_config.max_expand_range[1],
        )
        merge_gap = trial.suggest_int(
            "merge_gap",
            optuna_config.merge_gap_range[0],
            optuna_config.merge_gap_range[1],
        )
        
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
                target_clusters=5,  # Fixed target for fair comparison
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
            return -9999.0
        
        avg_hr = total_hits / num_valid_questions
        avg_mrr = total_mrr / num_valid_questions
        avg_tokens = total_tokens / len(corpus.questions) if corpus.questions else 0
        
        # HR-maximize objective with soft token penalty:
        # Goal: maximize HR, minimize tokens, soft penalty for tokens > 1200
        
        soft_token_limit = 1200
        
        # Primary: HR (higher is better) - scaled 0-100
        score = avg_hr * 100
        
        # Secondary: MRR bonus - up to 10
        score += avg_mrr * 10
        
        # Token efficiency bonus/penalty
        if avg_tokens <= soft_token_limit:
            # Bonus for being under limit (up to 5)
            token_bonus = (soft_token_limit - avg_tokens) / soft_token_limit * 5
            score += token_bonus
        else:
            # Soft penalty for being over limit
            excess = avg_tokens - soft_token_limit
            token_penalty = excess * 0.01  # Gradual penalty
            score -= token_penalty
        
        # Log intermediate values for analysis
        trial.set_user_attr("hit_rate", avg_hr)
        trial.set_user_attr("mrr", avg_mrr)
        trial.set_user_attr("avg_tokens", avg_tokens)
        trial.set_user_attr("score", score)
        trial.set_user_attr("tokens_ok", avg_tokens <= soft_token_limit)
        
        return score
    
    return objective


def main():
    """Main Optuna optimization loop."""
    args = parse_args()
    
    print("=" * 60)
    print("Optuna Hyperparameter Optimization")
    print("=" * 60)
    
    # Load corpus
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        print(f"❌ Corpus not found: {corpus_path}")
        print("   Run prepare_corpus.py first to create the corpus.")
        return
    
    print(f"\n📦 Loading corpus from {corpus_path}...")
    corpus = load_corpus(str(corpus_path))
    print(f"   ✅ Loaded {len(corpus.articles)} articles, {len(corpus.questions)} questions")
    
    # Count valid questions (with answer index found)
    valid_questions = sum(1 for q in corpus.questions if q.answer_sentence_idx >= 0)
    print(f"   ✅ Valid questions (answer found): {valid_questions}/{len(corpus.questions)}")
    
    # Create study
    storage = args.storage
    if storage:
        print(f"\n💾 Using persistent storage: {storage}")
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    
    # Create objective
    objective = create_objective(corpus)
    
    # Run optimization
    print(f"\n🔬 Running {args.n_trials} trials...")
    start_time = time.time()
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 60)
    print("✅ Optimization complete!")
    print("=" * 60)
    
    best = study.best_trial
    tokens_ok = best.user_attrs.get('tokens_ok', False)
    print(f"\n📊 Best trial #{best.number}:")
    print(f"   Score: {best.value:.4f}")
    print(f"   Hit Rate: {best.user_attrs.get('hit_rate', 0):.4f}")
    print(f"   MRR: {best.user_attrs.get('mrr', 0):.4f}")
    print(f"   Avg Tokens: {best.user_attrs.get('avg_tokens', 0):.1f} (soft limit: 1200) {'✅' if tokens_ok else '⚠️'}")
    
    print("\n🎯 Best hyperparameters:")
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n⏱️ Total time: {elapsed:.1f}s ({elapsed/args.n_trials:.2f}s per trial)")
    
    # Save best params
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    params_path = results_dir / "best_params.txt"
    with open(params_path, "w") as f:
        f.write(f"# Best Optuna trial #{best.number}\n")
        f.write(f"# Score: {best.value:.4f}\n")
        f.write(f"# Hit Rate: {best.user_attrs.get('hit_rate', 0):.4f}\n")
        f.write(f"# MRR: {best.user_attrs.get('mrr', 0):.4f}\n")
        f.write(f"# Avg Tokens: {best.user_attrs.get('avg_tokens', 0):.1f}\n")
        f.write(f"# Tokens OK: {best.user_attrs.get('tokens_ok', False)}\n\n")
        for key, value in best.params.items():
            f.write(f"{key} = {value}\n")
    
    print(f"\n💾 Best params saved to: {params_path}")
    
    # Parameter importance visualization
    if not args.no_viz:
        try:
            from optuna.visualization import plot_param_importances, plot_optimization_history
            
            print("\n📊 Generating parameter importance plot...")
            
            # Parameter importance
            fig_importance = plot_param_importances(study)
            importance_path = results_dir / "param_importance.html"
            fig_importance.write_html(str(importance_path))
            print(f"   ✅ Saved to: {importance_path}")
            
            # Optimization history
            fig_history = plot_optimization_history(study)
            history_path = results_dir / "optimization_history.html"
            fig_history.write_html(str(history_path))
            print(f"   ✅ Saved to: {history_path}")
            
        except ImportError:
            print("\n⚠️ Install plotly for visualizations: pip install plotly")
        except Exception as e:
            print(f"\n⚠️ Could not generate visualizations: {e}")
    
    # Print suggested config update
    print("\n📝 Suggested config.py update:")
    print("```python")
    print("@dataclass")
    print("class ExpansionConfig:")
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f"    {key}: float = {value:.4f}")
        else:
            print(f"    {key}: int = {value}")
    print("```")


if __name__ == "__main__":
    main()
