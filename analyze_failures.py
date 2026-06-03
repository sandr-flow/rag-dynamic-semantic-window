"""Analyze failures to find cases with insufficient chunks."""

import json
from pathlib import Path


def analyze_failures():
    """Count failures where Dynamic Semantic has less than 5 chunks."""
    failures_dir = Path(__file__).parent / "results" / "failures"
    
    if not failures_dir.exists():
        print("No failures directory found")
        return
    
    total_failures = 0
    insufficient_chunks = 0
    chunk_counts = {}
    
    for failure_file in failures_dir.glob("*.json"):
        try:
            with open(failure_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            total_failures += 1
            
            dynamic_semantic = data.get("strategies", {}).get("Dynamic Semantic", {})
            chunks = dynamic_semantic.get("chunks", [])
            num_chunks = len(chunks)
            
            # Count chunk distribution
            chunk_counts[num_chunks] = chunk_counts.get(num_chunks, 0) + 1
            
            if num_chunks < 5:
                insufficient_chunks += 1
                
        except Exception as e:
            print(f"Error reading {failure_file.name}: {e}")
    
    print(f"\n📊 Failure Analysis")
    print("=" * 40)
    print(f"Total failures: {total_failures}")
    print(f"Dynamic Semantic with < 5 chunks: {insufficient_chunks}")
    print(f"Percentage: {insufficient_chunks / total_failures * 100:.1f}%\n")
    
    print("Chunk distribution:")
    for num, count in sorted(chunk_counts.items()):
        pct = count / total_failures * 100
        print(f"  {num} chunks: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    analyze_failures()
