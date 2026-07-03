"""Benchmark stand for the custom dynamic semantic chunking strategy.

Two layers:
- prep (heavy, run once): build reusable artifacts on disk — datasets, embedding
  registrations, tuned hyperparameters.
- run (light): pick from prepared artifacts and benchmark one combination.

The interactive menu (``python -m stand``) is the primary entry point; the same
work is reachable non-interactively via ``stand.runner.run`` for CI/reproducibility.
"""
