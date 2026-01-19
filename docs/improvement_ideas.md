# Dynamic Semantic Improvement Ideas

Benchmark baseline (200 articles, 3 questions each):
- **Dynamic Semantic**: HR@5=0.85, MRR=0.74, Tokens=431
- **Naive Chunking**: HR@5=0.94, MRR=0.79, Tokens=569
- **Fixed Window**: HR@5=0.94, MRR=0.80, Tokens=1201

**Goal**: Close the HR gap (~9%) while staying under Fixed Window token count.

---

## 1. Query-Aware Expansion ⭐ (Recommended First)

**Problem**: Current expansion checks similarity between *adjacent sentences*. If there's a stylistic jump (e.g., "The battle was fierce." → "Casualties numbered 5,000."), we stop expanding even though both are query-relevant.

**Solution**: Also check `similarity(query_embedding, neighbor)`. If neighbor is directly relevant to the query, include it regardless of adjacency score.

```python
# Hybrid check
if similarity(current, neighbor) >= threshold:
    include(neighbor)
elif similarity(query, neighbor) >= query_threshold:  # NEW
    include(neighbor)
```

---

## 2. Multi-Seed Merging

**Problem**: With `top_k=5`, we get 5 separate clusters. They only merge if overlapping.

**Solution**: 
- **Proximity merge**: If clusters are within N sentences, merge them.
- **Query-coherence merge**: Merge if both cluster centroids are highly similar to query.

---

## 3. Adaptive Threshold

**Problem**: Fixed 0.6 threshold doesn't account for document length/complexity.

**Solution**: `threshold = 0.5 + 0.1 * log10(len(text)/1000)`
- Short articles (2k chars): ~0.53
- Long articles (20k chars): ~0.63

---

## 4. Bi-Directional Priority

**Problem**: We expand left and right equally, potentially wasting budget on weaker direction.

**Solution**: First expand in direction with higher initial similarity, then switch.

---

## 5. Context Window Overlap

**Problem**: Adjacent clusters may have gaps at boundaries, losing transition context.

**Solution**: Always include ±1 sentence overlap between merged clusters.

---

## Priority Order

1. **Query-Aware Expansion** — likely biggest impact on HR gap
2. **Proximity Merge** — simple, catches fragmented relevant context
3. **Adaptive Threshold** — low risk, may help with diverse article lengths
