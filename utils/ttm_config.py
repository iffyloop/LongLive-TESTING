from __future__ import annotations

from typing import Any, Dict


def normalize_ttm_config(raw_ttm: Any, num_denoising_steps: int) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "enabled": False,
        "mode": "file",
        "tweak_index": 0,
        "tstrong_index": num_denoising_steps,
        "threshold": 0.5,
        "mask_dilate": 0,
        "warmup_blocks": 0,
        "apply_after_prompt_switch": True,
        "deterministic_ref_noise": True,
        "ref_noise_seed": 0,
        "stream_decode": False,
        "use_cached_decode": False,
    }

    if raw_ttm is None:
        return cfg

    for key in cfg:
        if hasattr(raw_ttm, key):
            cfg[key] = getattr(raw_ttm, key)
        elif isinstance(raw_ttm, dict) and key in raw_ttm:
            cfg[key] = raw_ttm[key]

    cfg["enabled"] = bool(cfg["enabled"])
    cfg["mode"] = str(cfg["mode"])
    cfg["tweak_index"] = int(cfg["tweak_index"])
    cfg["tstrong_index"] = int(cfg["tstrong_index"])
    cfg["threshold"] = float(cfg["threshold"])
    cfg["mask_dilate"] = int(cfg["mask_dilate"])
    cfg["warmup_blocks"] = int(cfg["warmup_blocks"])
    cfg["apply_after_prompt_switch"] = bool(cfg["apply_after_prompt_switch"])
    cfg["deterministic_ref_noise"] = bool(cfg["deterministic_ref_noise"])
    cfg["ref_noise_seed"] = int(cfg["ref_noise_seed"])
    cfg["stream_decode"] = bool(cfg["stream_decode"])
    cfg["use_cached_decode"] = bool(cfg["use_cached_decode"])

    if not (0.0 <= cfg["threshold"] <= 1.0):
        raise ValueError("ttm.threshold must be in [0, 1]")
    if cfg["mask_dilate"] < 0:
        raise ValueError("ttm.mask_dilate must be >= 0")
    if cfg["warmup_blocks"] < 0:
        raise ValueError("ttm.warmup_blocks must be >= 0")
    if cfg["mode"] not in ("file", "stream"):
        raise ValueError("ttm.mode must be 'file' or 'stream'")

    if cfg["enabled"]:
        if cfg["tweak_index"] < 0 or cfg["tweak_index"] >= num_denoising_steps:
            raise ValueError("ttm.tweak_index out of range")
        if (
            cfg["tstrong_index"] <= cfg["tweak_index"]
            or cfg["tstrong_index"] > num_denoising_steps
        ):
            raise ValueError("ttm.tstrong_index must be in (tweak_index, num_steps]")

    return cfg
