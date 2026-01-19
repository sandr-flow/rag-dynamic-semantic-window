# Concept Design: Dynamic Semantic Window for RAG

## 1. Problem Statement

Traditional RAG systems suffer from suboptimal context boundaries:

- **Fixed Chunking** splits text at arbitrary positions, often mid-sentence
- **Fixed Window** padding includes irrelevant neighbors or misses related content
- Both approaches result in low Signal-to-Noise Ratio in retrieved context

## 2. Proposed Solution

Implement **Dynamic Semantic Window** — a post-retrieval expansion algorithm that:

1. Indexes each sentence as a separate node
2. Retrieves top-k most relevant sentences
3. Greedily expands each result by including semantically similar neighbors

The expansion continues in both directions while `cosine_similarity(current, neighbor) > threshold`.

## 3. Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Framework | LlamaIndex v0.10+ | Core orchestration |
| Embeddings | HuggingFace (`BAAI/bge-small-en-v1.5` or `intfloat/multilingual-e5-large`) | Fast, good semantic preservation |
| Vector Store | ChromaDB (local) or `SimpleVectorStore` (in-memory) | Demo-friendly |
| LLM | Mistral (`mistral-small-latest`/`mistral-large-latest`) or Ollama (local) | For answer generation and evaluation |

## 4. Architecture

### 4.1 Comparison Strategies

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: Source Text                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  A. Naive Chunking  │  │  B. Fixed Window    │  │  C. Dynamic Window  │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤
│ SentenceSplitter    │  │ SentenceWindow      │  │ Per-sentence index  │
│ chunk_size=256      │  │ NodeParser          │  │ + Custom Expander   │
│ overlap=20          │  │ window_size=3       │  │ threshold=0.75      │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
           │                        │                        │
           ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Benchmark Comparison                            │
│         (Token Count, Coherence Score, Answer Quality)              │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Dynamic Expansion Algorithm

```
┌────────────────────────────────────────────────────────────────┐
│                    Retrieved Sentence (seed)                   │
└────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│   Expand Left     │                     │   Expand Right    │
│  via prev_id      │                     │  via next_id      │
└───────────────────┘                     └───────────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│ cosine(cur, prev) │                     │ cosine(cur, next) │
│   >= threshold?   │                     │   >= threshold?   │
└───────────────────┘                     └───────────────────┘
     │         │                               │         │
    YES        NO                             YES        NO
     │         │                               │         │
     ▼         ▼                               ▼         ▼
  Include    Stop                           Include    Stop
  & repeat   expansion                      & repeat   expansion
```

## 5. Implementation Details

### 5.1 Data Preparation

Each sentence becomes a `TextNode` with metadata:

```python
{
    "node_id": "sent_042",
    "text": "Quantum entanglement occurs when particles...",
    "embedding": [0.023, -0.156, ...],  # Pre-computed
    "metadata": {
        "prev_id": "sent_041",
        "next_id": "sent_043",
        "source_doc": "quantum_mechanics.txt",
        "position": 42
    }
}
```

### 5.2 Custom PostProcessor

```python
class DynamicSemanticExpander(BaseNodePostprocessor):
    """
    Expand retrieved nodes by including semantically similar neighbors.
    
    Attributes:
        docstore: Access to all indexed nodes.
        threshold: Minimum cosine similarity for expansion (default: 0.75).
        max_expand: Maximum nodes to include per direction (default: 5).
    """
    
    docstore: BaseDocumentStore
    threshold: float = 0.75
    max_expand: int = 5

    def _postprocess_nodes(
        self, 
        nodes: list[NodeWithScore], 
        query_bundle: QueryBundle
    ) -> list[NodeWithScore]:
        expanded = []
        
        for node_score in nodes:
            cluster = self._expand_cluster(node_score.node)
            merged_text = self._merge_cluster(cluster)
            
            new_node = TextNode(
                text=merged_text,
                metadata={"source_nodes": [n.node_id for n in cluster]}
            )
            expanded.append(NodeWithScore(node=new_node, score=node_score.score))
        
        return self._deduplicate(expanded)
    
    def _expand_cluster(self, seed_node: TextNode) -> list[TextNode]:
        cluster = [seed_node]
        
        # Expand left
        current = seed_node
        for _ in range(self.max_expand):
            prev_id = current.metadata.get("prev_id")
            if not prev_id:
                break
            prev_node = self.docstore.get_node(prev_id)
            similarity = cosine_similarity(current.embedding, prev_node.embedding)
            if similarity < self.threshold:
                break
            cluster.insert(0, prev_node)
            current = prev_node
        
        # Expand right (analogous)
        current = seed_node
        for _ in range(self.max_expand):
            next_id = current.metadata.get("next_id")
            if not next_id:
                break
            next_node = self.docstore.get_node(next_id)
            similarity = cosine_similarity(current.embedding, next_node.embedding)
            if similarity < self.threshold:
                break
            cluster.append(next_node)
            current = next_node
        
        return cluster
```

### 5.3 Deduplication Logic

When multiple seed sentences belong to the same semantic region, their clusters may overlap. Deduplication merges overlapping clusters:

```python
def _deduplicate(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Merge clusters that share common source nodes."""
    seen_ids = set()
    result = []
    
    for node_score in nodes:
        source_ids = set(node_score.node.metadata["source_nodes"])
        if not source_ids & seen_ids:
            result.append(node_score)
            seen_ids.update(source_ids)
    
    return result
```

## 6. Evaluation Plan

### 6.1 Test Dataset

- Complex, structured text (e.g., Wikipedia article on Quantum Mechanics)
- 10 manually crafted questions covering different sections
- Questions designed to require multi-sentence context

### 6.2 Metrics

| Metric | Computation |
|--------|-------------|
| **Token Count** | `tiktoken.encode(context).length` |
| **Intra-Cluster Similarity** | Mean pairwise cosine similarity within cluster |
| **Boundary Coherence** | Manual annotation: Does context contain complete thoughts? |
| **Answer F1** | Token overlap between generated and reference answers |

### 6.3 Benchmark Output

```json
{
  "question": "What is quantum entanglement?",
  "strategies": {
    "naive_chunking": {
      "context": "...partial text...",
      "token_count": 312,
      "coherence_score": 0.67
    },
    "fixed_window": {
      "context": "...padded text...",
      "token_count": 458,
      "coherence_score": 0.72
    },
    "dynamic_window": {
      "context": "...expanded text...",
      "token_count": 289,
      "coherence_score": 0.91
    }
  }
}
```

## 7. Expected Outcomes

| Aspect | Naive Chunking | Fixed Window | Dynamic Window |
|--------|----------------|--------------|----------------|
| Context Quality | Truncated thoughts | Sometimes irrelevant padding | Complete semantic units |
| Token Efficiency | Medium | High (often wasteful) | Optimal (adaptive) |
| Coherence | Low | Medium | High |

### Trade-offs

- **Threshold tuning**: Too low = over-expansion; too high = fragmentation
- **Compute overhead**: Extra similarity calculations during post-processing
- **Edge cases**: Very long homogeneous sections may expand excessively

## 8. Success Criteria

1. ✅ Benchmark script executes without errors
2. ✅ Dynamic strategy shows higher coherence scores than baselines
3. ✅ Dynamic strategy uses fewer or equal tokens for comparable answer quality
4. ✅ Visualization notebook demonstrates clear boundary differences
5. ✅ Results exportable as JSON/Markdown for documentation

## 9. Future Extensions

- **Adaptive Threshold**: Learn optimal threshold per document type
- **Bidirectional Attention**: Use query embedding to guide expansion
- **Hierarchical Expansion**: Expand paragraphs, then sentences within
- **Streaming Expansion**: Progressive context loading for long documents
