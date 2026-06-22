from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

SCALING_MODES = ("standard", "fixed_update_scale", "rslora")
_ALIASES = {
    "standard": "standard",
    "alpha_over_r": "standard",
    "fixed": "fixed_update_scale",
    "fixed_scale": "fixed_update_scale",
    "fixed_update_scale": "fixed_update_scale",
    "rslora": "rslora",
    "rs_lora": "rslora",
    "alpha_over_sqrt_r": "rslora",
}


@dataclass(frozen=True)
class LoraScaling:
    """Resolved LoRA scaling for one module rank.

    ``reference_alpha`` is the alpha used by the original standard-alpha/r
    experiment. ``reference_rank`` defines a common anchor
    ``scale_at_reference_rank = reference_alpha / reference_rank``. The fixed
    and rsLoRA-style conditions are matched to the standard condition at that
    rank, avoiding an arbitrary overall rescaling between conditions.
    """

    rank: int
    scaling_mode: str
    reference_alpha: float
    reference_rank: int | None
    scale_at_reference_rank: float | None
    effective_alpha: float
    lora_scale: float

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def canonical_scaling_mode(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"unknown LoRA scaling mode {value!r}; expected one of {SCALING_MODES}"
        )
    return _ALIASES[key]


def normalise_scaling_modes(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        raw_modes = ["standard"]
    elif isinstance(value, str):
        raw_modes = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_modes = [str(part).strip() for part in value if str(part).strip()]
    if not raw_modes:
        raise ValueError("at least one LoRA scaling mode is required")
    modes = [canonical_scaling_mode(mode) for mode in raw_modes]
    if len(set(modes)) != len(modes):
        raise ValueError(f"duplicate LoRA scaling modes are not allowed: {modes}")
    return modes


def scaling_protocol(lora_cfg: Mapping, protocol_cfg: Mapping | None = None) -> dict:
    modes = normalise_scaling_modes(
        lora_cfg.get("scaling_modes", lora_cfg.get("scaling_mode"))
    )
    reference_alpha = float(lora_cfg.get("alpha", 8.0))
    if not math.isfinite(reference_alpha) or reference_alpha <= 0:
        raise ValueError("lora.alpha must be finite and positive")

    raw_reference_rank = lora_cfg.get("scale_reference_rank")
    reference_rank = int(raw_reference_rank) if raw_reference_rank is not None else None
    if reference_rank is not None and reference_rank <= 0:
        raise ValueError("lora.scale_reference_rank must be positive")
    if any(mode != "standard" for mode in modes) and reference_rank is None:
        raise ValueError(
            "lora.scale_reference_rank is required for fixed_update_scale or "
            "rslora so all scaling conditions share a declared anchor"
        )

    protocol_cfg = protocol_cfg or {}
    primary = canonical_scaling_mode(
        str(protocol_cfg.get("primary_scaling_mode", modes[0]))
    )
    if primary not in modes:
        raise ValueError(
            f"protocol.primary_scaling_mode={primary!r} is not present in "
            f"lora.scaling_modes={modes}"
        )

    scale_at_reference_rank = (
        reference_alpha / reference_rank if reference_rank is not None else None
    )
    return {
        "scaling_modes": modes,
        "primary_scaling_mode": primary,
        "reference_alpha": reference_alpha,
        "reference_rank": reference_rank,
        "scale_at_reference_rank": scale_at_reference_rank,
    }


def resolve_lora_scaling(
    rank: int,
    reference_alpha: float,
    scaling_mode: str = "standard",
    reference_rank: int | None = None,
) -> LoraScaling:
    rank = int(rank)
    reference_alpha = float(reference_alpha)
    scaling_mode = canonical_scaling_mode(scaling_mode)
    if rank < 0:
        raise ValueError("LoRA rank must be non-negative")
    if not math.isfinite(reference_alpha) or reference_alpha <= 0:
        raise ValueError("reference_alpha must be finite and positive")
    if reference_rank is not None:
        reference_rank = int(reference_rank)
        if reference_rank <= 0:
            raise ValueError("reference_rank must be positive")
    if scaling_mode != "standard" and reference_rank is None:
        raise ValueError(f"reference_rank is required for {scaling_mode} scaling")

    scale_at_reference_rank = (
        reference_alpha / reference_rank if reference_rank is not None else None
    )
    if rank == 0:
        return LoraScaling(
            rank=0,
            scaling_mode=scaling_mode,
            reference_alpha=reference_alpha,
            reference_rank=reference_rank,
            scale_at_reference_rank=scale_at_reference_rank,
            effective_alpha=0.0,
            lora_scale=0.0,
        )

    if scaling_mode == "standard":
        lora_scale = reference_alpha / rank
    elif scaling_mode == "fixed_update_scale":
        assert scale_at_reference_rank is not None
        lora_scale = scale_at_reference_rank
    else:  # reference-matched rsLoRA-style 1/sqrt(rank) scaling
        assert reference_rank is not None and scale_at_reference_rank is not None
        lora_scale = scale_at_reference_rank * math.sqrt(reference_rank / rank)

    return LoraScaling(
        rank=rank,
        scaling_mode=scaling_mode,
        reference_alpha=reference_alpha,
        reference_rank=reference_rank,
        scale_at_reference_rank=scale_at_reference_rank,
        effective_alpha=lora_scale * rank,
        lora_scale=lora_scale,
    )
