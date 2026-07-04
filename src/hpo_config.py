"""Configurable Optuna search space and objective policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src.config import DEFAULT_OPTUNA_CONFIG, OptunaConfig


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

    Above ``soft_token_limit`` a hard wall applies (see
    :func:`compute_objective_score`). Below it the score is
    ``HR*hr_weight + MRR*mrr_weight + token_bonus``. When
    ``token_tiebreak_fraction > 0`` the token bonus is scaled so its maximum
    possible value is that fraction of one HR question's worth
    (``hr_weight / num_valid_questions``). Because the fraction is < 1, token
    savings can never outweigh even a single caught question: the objective is
    HR first, token savings as a strict sub-HR tie-breaker, MRR last. Set the
    fraction to 0 to fall back to the static ``token_bonus_weight`` (legacy).
    """

    soft_token_limit: int = 1200
    hr_weight: float = 100.0
    mrr_weight: float = 0.01
    token_bonus_weight: float = 0.0
    token_tiebreak_fraction: float = 0.5
    token_penalty_per_token: float = 0.01
    invalid_score: float = -9999.0

    def token_bonus_weight_for(self, num_valid_questions: int) -> float:
        """Effective per-unit weight of the normalized token-savings bonus."""
        if self.token_tiebreak_fraction > 0 and num_valid_questions > 0:
            return self.token_tiebreak_fraction * self.hr_weight / num_valid_questions
        return self.token_bonus_weight


def compute_objective_score(
    *,
    avg_hr: float,
    avg_mrr: float,
    avg_tokens: float,
    num_valid_questions: int,
    policy: ObjectivePolicy,
) -> tuple[float, bool]:
    """Objective score for one config, plus whether it respected the budget.

    Hard wall: an over-budget config scores below ``invalid_score`` by its
    excess, so it is strictly worse than any in-budget config while still
    ordered by how far over it is. In budget: HR dominates, token savings act
    as a sub-HR tie-breaker, MRR breaks remaining ties.
    """
    tokens_ok = avg_tokens <= policy.soft_token_limit
    if not tokens_ok:
        return policy.invalid_score - (avg_tokens - policy.soft_token_limit), False

    score = avg_hr * policy.hr_weight + avg_mrr * policy.mrr_weight
    bonus_weight = policy.token_bonus_weight_for(num_valid_questions)
    if bonus_weight and policy.soft_token_limit > 0:
        token_bonus = (
            (policy.soft_token_limit - avg_tokens) / policy.soft_token_limit * bonus_weight
        )
        score += token_bonus
    return score, True


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
) -> HPOSettings:
    """Load HPO settings from optional YAML/JSON config."""
    settings = HPOSettings(
        search_space=default_search_space(optuna_config),
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
    params = {}
    for name, spec in search_space.items():
        if spec.type == "int":
            params[name] = trial.suggest_int(name, int(spec.low), int(spec.high), step=int(spec.step or 1))
        elif spec.type == "float":
            kwargs = {}
            if spec.step is not None:
                kwargs["step"] = float(spec.step)
            params[name] = trial.suggest_float(name, float(spec.low), float(spec.high), **kwargs)
        else:
            raise ValueError(f"Unsupported Optuna parameter type for {name}: {spec.type}")
    return params


def default_search_space(optuna_config: OptunaConfig = DEFAULT_OPTUNA_CONFIG) -> dict[str, ParameterSpec]:
    return {
        "threshold": _float_spec(optuna_config.threshold_range),
        "skip_threshold": _float_spec(optuna_config.skip_threshold_range),
        "relevance_threshold_pct": _float_spec(optuna_config.relevance_threshold_pct_range),
        "min_window": _int_spec(optuna_config.min_window_range),
        "max_expand": _int_spec(optuna_config.max_expand_range),
        "merge_gap": _int_spec(optuna_config.merge_gap_range),
    }


def _float_spec(value: tuple[float, float]) -> ParameterSpec:
    return ParameterSpec(type="float", low=value[0], high=value[1], step=0.001)


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
            step = 0.001
        return ParameterSpec(type=param_type, low=low, high=high, step=step)

    if isinstance(value, dict):
        if "low" not in value or "high" not in value:
            raise ValueError(f"Search range for {name} must include low and high")
        param_type = value.get("type")
        if not param_type:
            param_type = "int" if isinstance(value["low"], int) and isinstance(value["high"], int) else "float"
        step = value.get("step")
        if param_type == "float" and step is None:
            step = 0.001
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
