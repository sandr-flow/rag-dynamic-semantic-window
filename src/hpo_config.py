"""Configurable Optuna search space and objective policy."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src.config import DEFAULT_OPTUNA_CONFIG, OptunaConfig

# Cosine thresholds do not need milli-precision; 0.01 also keeps Optuna's
# float grid free of binary dust like 0.8979999999999999.
DEFAULT_FLOAT_STEP = 0.01


@dataclass
class ParameterSpec:
    """One Optuna parameter range."""

    type: str
    low: int | float
    high: int | float
    step: int | float | None = None


@dataclass
class ObjectivePolicy:
    """Scoring policy for cached dynamic semantic HPO.

    Score is ``HR*hr_weight + MRR*mrr_weight + under_bonus - overage_penalty``.

    Below ``soft_token_limit`` a small savings bonus is a sub-HR tie-breaker
    (see ``token_tiebreak_fraction``). Above it the penalty grows as
    ``(excess / limit) ** token_overage_power``, scaled in units of one HR
    question. There is no hard wall: ~1.25× the reference is still cheaper
    than one extra hit, while ~1.5× already costs more than one hit and
    2.5× is several questions expensive. That gap is what pushes Optuna
    toward threshold-driven expansion instead of a fat fixed window.
    Set ``token_overage_scale`` to 0 to disable.
    """

    soft_token_limit: int = 1200
    hr_weight: float = 100.0
    mrr_weight: float = 0.01
    token_bonus_weight: float = 0.0
    token_tiebreak_fraction: float = 0.5
    token_penalty_per_token: float = 0.01
    token_overage_power: float = 2.0
    token_overage_scale: float = 10.0
    invalid_score: float = -9999.0

    def token_bonus_weight_for(self, num_valid_questions: int) -> float:
        """Effective per-unit weight of the normalized under-budget bonus."""
        if self.token_tiebreak_fraction > 0 and num_valid_questions > 0:
            return self.token_tiebreak_fraction * self.hr_weight / num_valid_questions
        return self.token_bonus_weight

    def token_overage_penalty(self, avg_tokens: float, num_valid_questions: int) -> float:
        """Accelerating penalty for tokens above ``soft_token_limit``, in score units."""
        if (
            self.token_overage_scale <= 0
            or num_valid_questions <= 0
            or self.soft_token_limit <= 0
            or avg_tokens <= self.soft_token_limit
        ):
            return 0.0
        excess_ratio = (avg_tokens - self.soft_token_limit) / self.soft_token_limit
        one_question = self.hr_weight / num_valid_questions
        return one_question * self.token_overage_scale * (
            excess_ratio ** self.token_overage_power
        )


def compute_objective_score(
    *,
    avg_hr: float,
    avg_mrr: float,
    avg_tokens: float,
    num_valid_questions: int,
    policy: ObjectivePolicy,
) -> tuple[float, bool]:
    """Objective score for one config, plus whether it sat at or under the reference.

    No hard wall: over-reference configs stay in the same ranking. Under the
    reference, token savings are a sub-HR tie-breaker. Above it the penalty
    accelerates with excess so ~1.25× is cheap and ~1.5× is already more
    than one hit. ``tokens_ok`` is informational, not a veto.
    """
    tokens_ok = avg_tokens <= policy.soft_token_limit
    score = avg_hr * policy.hr_weight + avg_mrr * policy.mrr_weight
    if avg_tokens <= policy.soft_token_limit:
        bonus_weight = policy.token_bonus_weight_for(num_valid_questions)
        if bonus_weight and policy.soft_token_limit > 0:
            savings = (policy.soft_token_limit - avg_tokens) / policy.soft_token_limit
            score += savings * bonus_weight
    else:
        score -= policy.token_overage_penalty(avg_tokens, num_valid_questions)
    return score, tokens_ok


@dataclass
class HPOSettings:
    """Resolved HPO settings."""

    search_space: dict[str, ParameterSpec]
    objective: ObjectivePolicy
    raw_config: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "search_space": {key: asdict(value) for key, value in self.search_space.items()},
            "objective": asdict(self.objective),
            "raw_config": self.raw_config,
        }


def load_hpo_settings(
    path: str | None = None,
    optuna_config: OptunaConfig = DEFAULT_OPTUNA_CONFIG,
    soft_token_limit: int | None = None,
    strategy: str = "dynamic_semantic",
) -> HPOSettings:
    """Load HPO settings from optional YAML/JSON config."""
    settings = HPOSettings(
        search_space=search_space_for(strategy, optuna_config),
        objective=ObjectivePolicy(),
        raw_config={},
    )

    if path:
        raw = _load_mapping(Path(path))
        settings.raw_config = raw
        if "search_space" in raw:
            settings.search_space.update(_parse_search_space(raw["search_space"]))
        if "objective" in raw:
            settings.objective = _replace_objective(settings.objective, raw["objective"])

    if soft_token_limit is not None:
        settings.objective = replace(settings.objective, soft_token_limit=soft_token_limit)

    return settings


def suggest_params(trial, search_space: dict[str, ParameterSpec]) -> dict[str, int | float]:
    """Suggest all parameters from the configured search space."""
    params: dict[str, int | float] = {}
    for name, spec in search_space.items():
        if spec.type == "int":
            params[name] = trial.suggest_int(name, int(spec.low), int(spec.high), step=int(spec.step or 1))
        elif spec.type == "float":
            kwargs = {}
            if spec.step is not None:
                kwargs["step"] = float(spec.step)
            raw = trial.suggest_float(name, float(spec.low), float(spec.high), **kwargs)
            params[name] = quantize_float(raw, spec.step)
        else:
            raise ValueError(f"Unsupported Optuna parameter type for {name}: {spec.type}")
    return params


def _float_decimals(step: float | None) -> int:
    if step is None or step <= 0:
        return _float_decimals(DEFAULT_FLOAT_STEP)
    return max(0, min(6, int(round(-math.log10(step)))))


def quantize_float(value: float, step: float | None = DEFAULT_FLOAT_STEP) -> float:
    """Snap a float onto the search grid and drop binary rounding dust."""
    grid = DEFAULT_FLOAT_STEP if step is None else float(step)
    if grid <= 0:
        return round(float(value), _float_decimals(DEFAULT_FLOAT_STEP))
    decimals = _float_decimals(grid)
    snapped = math.floor(float(value) / grid + 0.5) * grid
    return round(snapped, decimals)


def quantize_params(
    params: dict[str, Any],
    search_space: dict[str, ParameterSpec] | None = None,
) -> dict[str, Any]:
    """Round float search params onto their grid so saved configs stay readable."""
    quantized = dict(params)
    for name, value in params.items():
        spec = search_space.get(name) if search_space else None
        if spec is not None and spec.type != "float":
            continue
        if not isinstance(value, float):
            continue
        step = spec.step if spec is not None else DEFAULT_FLOAT_STEP
        quantized[name] = quantize_float(value, step)
    return quantized


def default_search_space(optuna_config: OptunaConfig = DEFAULT_OPTUNA_CONFIG) -> dict[str, ParameterSpec]:
    return {
        "threshold": _float_spec(optuna_config.threshold_range),
        "skip_threshold": _float_spec(optuna_config.skip_threshold_range),
        "relevance_threshold_pct": _float_spec(optuna_config.relevance_threshold_pct_range),
        "min_window": _int_spec(optuna_config.min_window_range),
        "max_expand": _int_spec(optuna_config.max_expand_range),
        "merge_gap": _int_spec(optuna_config.merge_gap_range),
    }


def search_space_for(
    strategy: str,
    optuna_config: OptunaConfig = DEFAULT_OPTUNA_CONFIG,
) -> dict[str, ParameterSpec]:
    """Default Optuna search space for one retrieval strategy."""
    from src.strategy_registry import normalize_strategy_id

    strategy_id = normalize_strategy_id(strategy)
    if strategy_id == "dynamic_semantic":
        return default_search_space(optuna_config)
    if strategy_id in {"naive", "token_text"}:
        # Overlap stays strictly below the minimum chunk_size so every
        # sample is a valid splitter config without a separate constraint.
        return {
            "chunk_size": ParameterSpec(type="int", low=128, high=512, step=32),
            "chunk_overlap": ParameterSpec(type="int", low=0, high=80, step=8),
        }
    if strategy_id == "fixed_window":
        return {
            "window_size": ParameterSpec(type="int", low=1, high=6, step=1),
        }
    if strategy_id == "semantic_splitter":
        return {
            "buffer_size": ParameterSpec(type="int", low=1, high=3, step=1),
            "breakpoint_percentile_threshold": ParameterSpec(
                type="int", low=50, high=95, step=5
            ),
        }
    raise ValueError(f"No HPO search space for strategy '{strategy}'")


def _float_spec(value: tuple[float, float]) -> ParameterSpec:
    return ParameterSpec(type="float", low=value[0], high=value[1], step=DEFAULT_FLOAT_STEP)


def _int_spec(value: tuple[int, int]) -> ParameterSpec:
    return ParameterSpec(type="int", low=value[0], high=value[1], step=1)


def _load_mapping(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8-sig") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(f)
        else:
            payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("HPO config must be a mapping")
    return payload


def _parse_search_space(payload: dict[str, Any]) -> dict[str, ParameterSpec]:
    if not isinstance(payload, dict):
        raise ValueError("search_space must be a mapping")

    parsed = {}
    for name, value in payload.items():
        parsed[name] = _parse_parameter_spec(name, value)
    return parsed


def _parse_parameter_spec(name: str, value: Any) -> ParameterSpec:
    if isinstance(value, list | tuple):
        if len(value) not in {2, 3}:
            raise ValueError(f"Search range for {name} must have 2 or 3 values")
        low, high = value[0], value[1]
        step = value[2] if len(value) == 3 else None
        param_type = "int" if isinstance(low, int) and isinstance(high, int) else "float"
        if param_type == "float" and step is None:
            step = DEFAULT_FLOAT_STEP
        return ParameterSpec(type=param_type, low=low, high=high, step=step)

    if isinstance(value, dict):
        if "low" not in value or "high" not in value:
            raise ValueError(f"Search range for {name} must include low and high")
        param_type = value.get("type")
        if not param_type:
            param_type = "int" if isinstance(value["low"], int) and isinstance(value["high"], int) else "float"
        step = value.get("step")
        if param_type == "float" and step is None:
            step = DEFAULT_FLOAT_STEP
        return ParameterSpec(
            type=str(param_type),
            low=value["low"],
            high=value["high"],
            step=step,
        )

    raise ValueError(f"Unsupported search range for {name}: {value}")


def _replace_objective(policy: ObjectivePolicy, payload: dict[str, Any]) -> ObjectivePolicy:
    if not isinstance(payload, dict):
        raise ValueError("objective must be a mapping")
    allowed = ObjectivePolicy.__dataclass_fields__.keys()
    values = {key: value for key, value in payload.items() if key in allowed}
    return replace(policy, **values)
