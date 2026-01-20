# Mathematical Foundations for Dynamic Semantic Chunker

Advanced mathematical approaches for robust semantic window expansion.

## Table of Contents

- [A. Seed Validation (Outlier Detection)](#a-seed-validation-outlier-detection)
- [B. Adaptive Expansion Threshold](#b-adaptive-expansion-threshold-dynamic-decay)
- [C. Semantic Coherence (Gap Bridging)](#c-semantic-coherence-gap-bridging)
- [D. Score Space Normalization](#d-score-space-normalization)
- [Final Architecture](#final-architecture)

---

## A. Seed Validation (Outlier Detection)

> [!CAUTION]
> **Status: DEFERRED / NEEDS RETHINKING**
> 
> Experimental results showed this approach is **too aggressive** for RAG retrieval.
> In benchmarks, valid seeds (first sentences, direct answers) were incorrectly rejected
> because they lacked "local semantic support" (neighboring sentences on different topics).
> 
> The assumption that high-scoring isolated sentences are "noise" doesn't hold for:
> - Introductory sentences (position 0-10)
> - Factual statements answering specific questions
> - Short articles with few sentences
> 
> **Recommendation:** Disable or significantly relax before using in production.

### Recommendation: Modified Z-score via MAD

**Why:**

- Standard Z-score is sensitive to outliers (outliers themselves affect μ and σ)
- IQR is too coarse for small K (Top-10)
- MAD (Median Absolute Deviation) is robust to outliers

**Formula:**

```
MAD = median(|score_i - median(scores)|)
Modified Z-score = 0.6745 × (score_i - median(scores)) / MAD
```

**Validity Criterion:**

```python
# Seed is valid if:
modified_z > 2.5  # classic threshold for outlier detection

# But! Add "lone ranger" check:
local_support = count(scores in window ±2 from seed where score > seed × 0.7)
is_valid = (modified_z > 2.5) AND (local_support >= 1)
```

### Lone Ranger Detection

```python
def validate_seed(seed_idx, scores, embeddings):
    """
    Validate seed using statistical and semantic support criteria.
    
    Args:
        seed_idx: Index of the seed sentence.
        scores: Array of relevance scores.
        embeddings: Array of sentence embeddings.
    
    Returns:
        True if seed is valid, False otherwise.
    """
    # 1. Statistical validation
    mad = np.median(np.abs(scores - np.median(scores)))
    mod_z = 0.6745 * (scores[seed_idx] - np.median(scores)) / mad
    
    # 2. Local semantic support
    window = range(max(0, seed_idx-3), min(len(scores), seed_idx+4))
    neighbors = [scores[i] for i in window if i != seed_idx]
    
    # Check: is there a "semantic aura"?
    support_threshold = scores[seed_idx] * 0.65  # relative threshold
    local_support = sum(1 for s in neighbors if s > support_threshold)
    
    return (mod_z > 2.5) and (local_support >= 2)
```

---

## B. Adaptive Expansion Threshold (Dynamic Decay)

### Recommendation: Exponential Decay with Local Calibration

**Base Formula:**

```
threshold(distance) = score_seed × exp(-λ × distance)

where λ = -ln(α) / max_distance
α — allowed energy drop (e.g., 0.5 for 50% of seed)
```

**Adaptive Approach (more effective):**

```python
def adaptive_expansion_threshold(seed_score, local_scores, distance):
    """
    Threshold based on local noise and decay gradient.
    
    Args:
        seed_score: Relevance score of the seed sentence.
        local_scores: Array of scores in the local window.
        distance: Distance from the seed sentence.
    
    Returns:
        Adaptive threshold value for expansion.
    """
    # 1. Estimate local noise (std in window)
    local_std = np.std(local_scores)
    noise_floor = np.median(local_scores) + 1.5 * local_std
    
    # 2. Adaptive decay rate based on density
    score_density = len([s for s in local_scores if s > seed_score * 0.6]) / len(local_scores)
    λ = 0.3 if score_density > 0.4 else 0.5  # decay slower in dense areas
    
    # 3. Combined threshold
    decay_threshold = seed_score * np.exp(-λ * distance)
    adaptive_threshold = max(decay_threshold, noise_floor)
    
    return adaptive_threshold
```

### Gradient-Based Stop Criterion

```python
def should_stop_expansion(scores_sequence):
    """
    Stop on sharp gradient drop (semantic cliff).
    
    Args:
        scores_sequence: Sequence of scores during expansion.
    
    Returns:
        True if expansion should stop, False otherwise.
    """
    if len(scores_sequence) < 3:
        return False
    
    # Compute gradient (derivative)
    gradients = np.diff(scores_sequence)
    
    # Detect "cliff" — sharp trend change
    if len(gradients) >= 2:
        gradient_change = gradients[-1] / (gradients[-2] + 1e-10)
        # If drop accelerated 2x+ — stop
        return gradient_change < -2.0
    
    return False
```

---

## C. Semantic Coherence (Gap Bridging)

### Recommendation: Sliding Window Coherence Score

**Formula:**

```python
def coherence_score(embeddings, window_size=3):
    """
    Evaluate local coherence via sliding window.
    
    Args:
        embeddings: Array of sentence embeddings.
        window_size: Size of the sliding window.
    
    Returns:
        List of coherence scores for each window position.
    """
    coherence_scores = []
    
    for i in range(len(embeddings) - window_size + 1):
        window = embeddings[i:i+window_size]
        
        # Option 1: Average pairwise similarity
        pairwise = []
        for j in range(len(window)):
            for k in range(j+1, len(window)):
                pairwise.append(cosine_similarity(window[j], window[k]))
        
        avg_coherence = np.mean(pairwise)
        coherence_scores.append(avg_coherence)
    
    return coherence_scores
```

### Gap Bridging Criterion

```python
def can_bridge_gap(prev_segment, gap_sentences, next_segment, embeddings):
    """
    Determine if gap can be included between two semantic clusters.
    
    Args:
        prev_segment: Indices of the previous segment.
        gap_sentences: Indices of the gap sentences.
        next_segment: Indices of the next segment.
        embeddings: Array of all sentence embeddings.
    
    Returns:
        True if gap forms a valid bridge, False otherwise.
    """
    # Evaluate gap coherence with both sides
    gap_embs = embeddings[gap_sentences]
    
    coherence_with_prev = cosine_similarity(
        gap_embs[0], 
        embeddings[prev_segment[-1]]
    )
    coherence_with_next = cosine_similarity(
        gap_embs[-1], 
        embeddings[next_segment[0]]
    )
    
    # Gap is valid if it forms a "bridge"
    min_bridge_strength = 0.6  # can be made adaptive
    return (coherence_with_prev > min_bridge_strength and 
            coherence_with_next > min_bridge_strength)
```

### Alternative (Faster): Centroid-Based

```python
def centroid_coherence(segment_embeddings):
    """
    Compute coherence as distance to segment centroid.
    
    Args:
        segment_embeddings: Array of embeddings for the segment.
    
    Returns:
        Coherence score (1 - std of distances to centroid).
    """
    centroid = np.mean(segment_embeddings, axis=0)
    distances = [cosine_similarity(emb, centroid) for emb in segment_embeddings]
    
    # Low variance = high coherence
    coherence = 1 - np.std(distances)
    return coherence
```

---

## D. Score Space Normalization

### Problem

Different embedding models have different cosine score distributions:

| Model   | Score Range   |
|---------|---------------|
| BGE     | [0.6 - 0.95]  |
| OpenAI  | [0.4 - 0.85]  |
| Cohere  | [0.5 - 0.92]  |

### Solution 1: Temperature Scaling

```python
def temperature_calibrate(scores, temperature=1.0):
    """
    Normalize via temperature for distribution sharpness control.
    
    Args:
        scores: Array of similarity scores.
        temperature: Scaling factor (empirically tuned per model).
            - temperature < 1 → sharper distribution
            - temperature > 1 → smoother distribution
    
    Returns:
        Temperature-scaled normalized scores.
    """
    exp_scores = np.exp(scores / temperature)
    return exp_scores / np.sum(exp_scores)
```

### Solution 2: Quantile Normalization (Recommended)

```python
def quantile_normalize(scores, reference_quantiles=None):
    """
    Map distribution to reference via quantiles.
    
    Args:
        scores: Array of similarity scores.
        reference_quantiles: Reference distribution (calibrated on dev set).
    
    Returns:
        Quantile-normalized scores.
    """
    if reference_quantiles is None:
        # Reference distribution (calibrated on dev set)
        reference_quantiles = np.array([0.3, 0.5, 0.7, 0.85, 0.95])
    
    # Rank current scores
    ranks = np.argsort(np.argsort(scores))
    quantile_positions = ranks / (len(scores) - 1)
    
    # Interpolate to reference space
    normalized = np.interp(
        quantile_positions,
        np.linspace(0, 1, len(reference_quantiles)),
        reference_quantiles
    )
    
    return normalized
```

### Solution 3: Robust Min-Max with Outlier Clipping

```python
def robust_minmax(scores, percentile_range=(5, 95)):
    """
    Min-Max normalization with robust boundaries.
    
    Args:
        scores: Array of similarity scores.
        percentile_range: Percentile range for clipping outliers.
    
    Returns:
        Robust min-max normalized scores clipped to [0, 1].
    """
    lower = np.percentile(scores, percentile_range[0])
    upper = np.percentile(scores, percentile_range[1])
    
    normalized = (scores - lower) / (upper - lower)
    return np.clip(normalized, 0, 1)
```

---

## Final Architecture

```python
class AdaptiveSemanticExpander:
    """
    Complete adaptive semantic expansion pipeline.
    
    Combines all mathematical foundations into a unified processor
    for robust context window expansion.
    """
    
    def __init__(self, model_calibration=None):
        """
        Initialize expander with optional model-specific calibration.
        
        Args:
            model_calibration: Dict with model-specific parameters.
        """
        self.calibration = model_calibration or self._default_calibration()
    
    def process(self, top_k_results, embeddings, query_embedding):
        """
        Process top-k results into expanded semantic windows.
        
        Args:
            top_k_results: Dict with 'indices' and 'scores' from retrieval.
            embeddings: Array of all document sentence embeddings.
            query_embedding: Query embedding vector.
        
        Returns:
            Final merged context with gap bridging applied.
        """
        # 1. Normalize scores for the model
        normalized_scores = quantile_normalize(
            top_k_results['scores'],
            self.calibration['quantiles']
        )
        
        # 2. Validate seeds
        valid_seeds = [
            idx for idx, score in enumerate(normalized_scores)
            if self.validate_seed(idx, normalized_scores, embeddings)
        ]
        
        # 3. Adaptive expansion for each seed
        expanded_windows = []
        for seed_idx in valid_seeds:
            window = self.expand_window(
                seed_idx, 
                normalized_scores, 
                embeddings
            )
            expanded_windows.append(window)
        
        # 4. Merge with gap bridging
        final_context = self.merge_with_bridging(expanded_windows, embeddings)
        
        return final_context
```

---

## References

- Modified Z-score: Iglewicz, B. and Hoaglin, D.C. (1993). "How to Detect and Handle Outliers"
- MAD (Median Absolute Deviation): Robust statistics for outlier detection
- Quantile Normalization: Commonly used in bioinformatics for cross-sample normalization
