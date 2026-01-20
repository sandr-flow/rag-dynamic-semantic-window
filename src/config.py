"""Configuration for Dynamic Semantic Expander and retrieval strategies."""

from dataclasses import dataclass, field


@dataclass
class ExpansionConfig:
    """Configuration for semantic expansion algorithm.
    
    Controls how the DynamicSemanticExpander expands context around seed nodes.
    """
    
    # Core expansion thresholds
    threshold: float = 0.85
    """Minimum cosine similarity for expansion beyond min_window."""
    
    skip_threshold: float = 0.85
    """Threshold for sentence skipping (bridging low-similarity gaps)."""
    
    relevance_threshold_pct: float = 0.70
    """Query relevance threshold as percentage of max seed score (0.70 = 70%)."""
    
    # Window size limits
    max_expand: int = 7
    """Maximum sentences to expand in each direction from seed."""
    
    min_window: int = 3
    """Minimum neighbors to always include (hybrid safety net)."""
    
    min_chunk_length: int = 20
    """Minimum character length for valid chunks (filters garbage)."""
    
    # Cluster management
    target_clusters: int = 5
    """Target number of clusters for deduplication and backfill."""
    
    merge_gap: int = 2
    """Maximum gap between clusters to merge (2 = merge if <= 2 sentences apart)."""


@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive threshold mechanism.
    
    Controls how expansion threshold decays with distance from seed,
    adjusted by local score density.
    """
    
    enabled: bool = True
    """Enable adaptive threshold (decay with distance)."""
    
    decay_lambda_sparse: float = 0.5
    """Decay rate λ for sparse regions (faster decay)."""
    
    decay_lambda_dense: float = 0.3
    """Decay rate λ for dense regions (slower decay, more context)."""
    
    density_threshold: float = 0.4
    """Threshold for considering region 'dense' (ratio of high-scoring neighbors)."""
    
    density_score_ratio: float = 0.6
    """Neighbor is 'high-scoring' if score > seed_score * this ratio."""
    
    floor_multiplier: float = 0.7
    """Floor: don't decay below base_threshold * this multiplier."""
    
    gradient_cliff_factor: float = -2.0
    """Stop expansion if gradient accelerates beyond this (semantic cliff detection)."""


@dataclass
class SeedValidationConfig:
    """Configuration for seed validation mechanism.
    
    Uses Modified Z-score and local semantic support to filter
    outlier seeds (lone rangers) that may be false positives.
    """
    
    enabled: bool = False
    """Enable seed validation (currently disabled - was too aggressive)."""
    
    mod_z_threshold: float = 5.0
    """Modified Z-score threshold for outlier detection."""
    
    support_threshold_pct: float = 0.50
    """Relative threshold for local support (neighbor_sim > seed_sim * this)."""
    
    min_local_support: int = 0
    """Minimum supporting neighbors required (0 = only mod_z matters)."""
    
    window_size: int = 3
    """Window size for local support check (±N neighbors)."""
    
    early_position_threshold: int = 10
    """First N sentences get relaxed validation (introductions are often unique)."""
    
    min_scores_for_validation: int = 3
    """Minimum number of scores required for meaningful statistics."""
    
    mad_epsilon: float = 1e-10
    """Epsilon for MAD calculation to avoid division by zero."""
    
    mad_constant: float = 0.6745
    """Constant for Modified Z-score: 0.6745 ≈ 1/Φ^(-1)(0.75)."""


@dataclass
class NaiveChunkingConfig:
    """Configuration for Naive Chunking strategy (baseline)."""
    
    chunk_size: int = 128
    """Target chunk size in tokens."""
    
    chunk_overlap: int = 20
    """Overlap between consecutive chunks."""


@dataclass
class FixedWindowConfig:
    """Configuration for Fixed Sentence Window strategy (control)."""
    
    window_size: int = 3
    """Number of sentences to include on each side of center."""
    
    window_metadata_key: str = "window"
    """Metadata key for storing window context."""
    
    original_text_metadata_key: str = "original_text"
    """Metadata key for storing original sentence text."""


@dataclass
class SemanticSplitterConfig:
    """Configuration for Semantic Splitter strategy."""
    
    buffer_size: int = 1
    """Buffer size for semantic chunking."""
    
    breakpoint_percentile_threshold: int = 80
    """Percentile threshold for detecting semantic breakpoints."""


@dataclass
class DynamicSemanticConfig:
    """Configuration for Dynamic Semantic strategy (experiment)."""
    
    phantom_window: int = 1
    """Context window for phantom embeddings (0 = disabled)."""
    
    prefetch_multiplier: int = 2
    """Two-pass retrieval: fetch top_k * multiplier, then filter to top_k."""


@dataclass
class RetrievalConfig:
    """Top-level retrieval configuration."""
    
    top_k: int = 5
    """Number of results to retrieve."""


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    mistral_rps_delay: float = 1.1
    """Delay between Mistral API calls (seconds) to respect 1 RPS limit."""
    
    default_min_article_length: int = 2000
    """Minimum article length in characters."""
    
    default_num_questions: int = 3
    """Default number of QA pairs to generate per article."""
    
    default_num_articles: int = 5
    """Default number of articles to fetch for benchmark."""


@dataclass
class DynamicSemanticExpanderConfig:
    """Complete configuration for DynamicSemanticExpander.
    
    Aggregates all sub-configurations for easy initialization.
    """
    
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    adaptive_threshold: AdaptiveThresholdConfig = field(default_factory=AdaptiveThresholdConfig)
    seed_validation: SeedValidationConfig = field(default_factory=SeedValidationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


# Default configurations
DEFAULT_EXPANSION_CONFIG = ExpansionConfig()
DEFAULT_ADAPTIVE_THRESHOLD_CONFIG = AdaptiveThresholdConfig()
DEFAULT_SEED_VALIDATION_CONFIG = SeedValidationConfig()
DEFAULT_NAIVE_CHUNKING_CONFIG = NaiveChunkingConfig()
DEFAULT_FIXED_WINDOW_CONFIG = FixedWindowConfig()
DEFAULT_SEMANTIC_SPLITTER_CONFIG = SemanticSplitterConfig()
DEFAULT_DYNAMIC_SEMANTIC_CONFIG = DynamicSemanticConfig()
DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()
DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()
