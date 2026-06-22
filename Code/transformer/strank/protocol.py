"""Versioned experiment protocol identifiers and configuration checks."""

from __future__ import annotations

from typing import Mapping

from .scaling import scaling_protocol

SYNTHETIC_TRANSFORMER_PROTOCOL_VERSION = "synthetic_transformer_publication_protocol_v4"
WHITENING_ABLATION_PROTOCOL_VERSION = "synthetic_transformer_whitening_publication_protocol_v4"

SUPPORTED_ALLOCATION_RULES = {
    "uniform",
    "uniform_fill",
    "effective_rank",
    "soft_dimension",
    "gradient_norm",
    "marginal_gain_raw",
    "marginal_gain_edge",
    "marginal_gain_soft",
}


def _integer_list(values, label: str, *, positive: bool = False) -> list[int]:
    result = [int(value) for value in values]
    if not result:
        raise ValueError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate values: {result}")
    lower_bound = 1 if positive else 0
    if any(value < lower_bound for value in result):
        adjective = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must contain only {adjective} integers")
    if result != sorted(result):
        raise ValueError(f"{label} must be sorted in ascending order")
    return result


def validate_synthetic_transformer_config(cfg: Mapping) -> dict:
    """Validate invariants that should fail before an expensive experiment.

    The function remains backward-compatible with legacy single-scaling runs,
    while enforcing every corrected field that is present. Strict publication
    sample-size and null-calibration requirements are enforced by the release
    validator, where the complete multi-run design is available.
    """

    if not isinstance(cfg, Mapping):
        raise ValueError("experiment config must be a mapping")
    for section in ("run", "task", "model", "base_train", "calibration", "sites", "lora", "budgets"):
        if not isinstance(cfg.get(section), Mapping):
            raise ValueError(f"missing or invalid config section: {section}")

    protocol_cfg = cfg.get("protocol", {})
    if not isinstance(protocol_cfg, Mapping):
        raise ValueError("protocol must be a mapping")
    scaling = scaling_protocol(cfg["lora"], protocol_cfg)

    ranks = _integer_list(cfg["lora"].get("ranks", []), "lora.ranks")
    if ranks[0] != 0:
        raise ValueError("lora.ranks must include rank 0 as the first value")
    reference_rank = scaling["reference_rank"]
    if reference_rank is not None and int(reference_rank) not in ranks:
        raise ValueError("lora.scale_reference_rank must be present in lora.ranks")

    sites = [str(value) for value in cfg["sites"].get("include", [])]
    if not sites or any(not site for site in sites):
        raise ValueError("sites.include must contain at least one non-empty module name")
    if len(set(sites)) != len(sites):
        raise ValueError("sites.include contains duplicate module names")

    budgets = _integer_list(
        cfg["budgets"].get("param_budgets", []),
        "budgets.param_budgets",
        positive=True,
    )
    rules = [str(value) for value in cfg["budgets"].get("rules", [])]
    if not rules:
        raise ValueError("budgets.rules must not be empty")
    if len(set(rules)) != len(rules):
        raise ValueError("budgets.rules contains duplicate rule names")
    unknown_rules = sorted(set(rules) - SUPPORTED_ALLOCATION_RULES)
    if unknown_rules:
        raise ValueError(f"budgets.rules contains unsupported rules: {unknown_rules}")

    matched_rules = [str(value) for value in cfg["budgets"].get("cost_matched_rules", [])]
    if len(set(matched_rules)) != len(matched_rules):
        raise ValueError("budgets.cost_matched_rules contains duplicate rule names")
    missing_matched = sorted(set(matched_rules) - set(rules))
    if missing_matched:
        raise ValueError(
            "budgets.cost_matched_rules are absent from budgets.rules: "
            f"{missing_matched}"
        )
    invalid_matched = sorted(set(matched_rules) & {"uniform", "uniform_fill"})
    if invalid_matched:
        raise ValueError(
            "cap baselines cannot be declared as cost-matched candidate rules: "
            f"{invalid_matched}"
        )

    primary_rule = str(protocol_cfg.get("primary_allocation_rule", ""))
    primary_reference = str(
        protocol_cfg.get("primary_reference_rule", "uniform_exact_cost")
    )
    if primary_rule and primary_rule not in rules:
        raise ValueError("protocol.primary_allocation_rule is absent from budgets.rules")
    if primary_rule and primary_rule not in matched_rules:
        raise ValueError(
            "protocol.primary_allocation_rule must have an exact-cost comparator"
        )
    if matched_rules and primary_reference != "uniform_exact_cost":
        raise ValueError(
            "protocol.primary_reference_rule must be uniform_exact_cost when "
            "cost-matched comparisons are enabled"
        )

    adaptation_replicates = int(protocol_cfg.get("adaptation_replicates", 1))
    sweep_replicates = int(protocol_cfg.get("sweep_replicates", 1))
    if adaptation_replicates < 1 or sweep_replicates < 1:
        raise ValueError("adaptation_replicates and sweep_replicates must be at least 1")

    primary_metric = str(protocol_cfg.get("primary_metric", "final_val_loss"))
    independent_unit = str(protocol_cfg.get("independent_unit", "task_seed_run"))
    if primary_metric != "final_val_loss":
        raise ValueError("protocol.primary_metric must be final_val_loss")
    if independent_unit != "task_seed_run":
        raise ValueError("protocol.independent_unit must be task_seed_run")

    calibration = cfg["calibration"]
    n_boot = int(calibration.get("null_bootstrap", 8))
    quantile = float(calibration.get("null_quantile", 0.995))
    uncertainty_resamples = int(calibration.get("null_uncertainty_resamples", 1000))
    confidence = float(calibration.get("null_uncertainty_confidence", 0.95))
    if n_boot <= 0:
        raise ValueError("calibration.null_bootstrap must be positive")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("calibration.null_quantile must lie in (0, 1]")
    if uncertainty_resamples < 0:
        raise ValueError("calibration.null_uncertainty_resamples must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("calibration.null_uncertainty_confidence must lie in (0, 1)")

    if int(cfg["base_train"].get("steps", 0)) < 0 or int(cfg["lora"].get("steps", 0)) < 0:
        raise ValueError("training step counts must be non-negative")

    return {
        **scaling,
        "ranks": ranks,
        "sites": sites,
        "budgets": budgets,
        "rules": rules,
        "cost_matched_rules": matched_rules,
        "primary_allocation_rule": primary_rule,
        "primary_reference_rule": primary_reference,
        "primary_metric": primary_metric,
        "independent_unit": independent_unit,
        "adaptation_replicates": adaptation_replicates,
        "sweep_replicates": sweep_replicates,
        "null_bootstrap": n_boot,
        "null_quantile": quantile,
        "null_uncertainty_resamples": uncertainty_resamples,
        "null_uncertainty_confidence": confidence,
    }
