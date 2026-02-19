from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_spatial_bcthw(
    data_bcthw: torch.Tensor, target_h: int, target_w: int, mode: str
) -> torch.Tensor:
    b, c, t, h, w = data_bcthw.shape
    flat = data_bcthw.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    if mode == "nearest":
        flat = F.interpolate(flat, size=(target_h, target_w), mode=mode)
    else:
        flat = F.interpolate(
            flat,
            size=(target_h, target_w),
            mode=mode,
            align_corners=False,
        )
    return flat.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()


def prepare_motion_latents(
    motion_bcthw: torch.Tensor,
    vae,
    target_shape_btc_hw,
    *,
    target_pixel_hw,
    drop_first_latent: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Encode motion pixels to latents and align to [B, T, C, H, W]."""
    target_b, target_t, _, target_h, target_w = target_shape_btc_hw

    pixel_h, pixel_w = target_pixel_hw

    motion = motion_bcthw.to(device=device, dtype=torch.float32)
    motion = _resize_spatial_bcthw(motion, pixel_h, pixel_w, mode="bicubic")
    motion = motion * 2.0 - 1.0
    ref_latents = vae.encode_to_latent(motion)

    if drop_first_latent:
        if ref_latents.shape[1] <= 1:
            raise ValueError("Cannot drop first latent: encoded reference is too short")
        ref_latents = ref_latents[:, 1:]

    if ref_latents.shape[1] != target_t:
        raise ValueError(
            f"Motion temporal mismatch after VAE encode: got {ref_latents.shape[1]}, expected {target_t}"
        )

    # Keep a final spatial alignment safeguard in latent space.
    if ref_latents.shape[3] != target_h or ref_latents.shape[4] != target_w:
        ref_bcthw = ref_latents.permute(0, 2, 1, 3, 4)
        ref_bcthw = _resize_spatial_bcthw(ref_bcthw, target_h, target_w, mode="bicubic")
        ref_latents = ref_bcthw.permute(0, 2, 1, 3, 4).contiguous()

    if ref_latents.shape[0] != target_b:
        raise ValueError(
            f"Motion batch mismatch: got {ref_latents.shape[0]}, expected {target_b}"
        )
    return ref_latents.to(dtype=dtype)


def prepare_motion_mask(
    mask_b1thw: torch.Tensor,
    target_shape_btc_hw,
    *,
    target_pixel_hw,
    temporal_factor: int,
    drop_first_latent: bool,
    threshold: float,
    dilate: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert mask to latent-aligned binary mask [B, T, 1, H, W]."""
    target_b, target_t, _, target_h, target_w = target_shape_btc_hw

    pixel_h, pixel_w = target_pixel_hw

    mask = mask_b1thw.to(device=device, dtype=torch.float32)
    mask = _resize_spatial_bcthw(mask, pixel_h, pixel_w, mode="nearest")
    mask = (mask > threshold).to(dtype=torch.float32)

    if temporal_factor > 1:
        if not drop_first_latent:
            if target_t == 1:
                if mask.shape[2] < 1:
                    raise ValueError("Mask has no frames for target_t=1")
                mask = mask[:, :, :1]
            else:
                expected = 1 + (target_t - 1) * temporal_factor
                if mask.shape[2] != expected:
                    raise ValueError(
                        f"Mask temporal mismatch: got {mask.shape[2]}, expected {expected}"
                    )
                first = mask[:, :, :1]
                rest = mask[:, :, 1::temporal_factor]
                if rest.shape[2] != target_t - 1:
                    raise ValueError(
                        f"Mask temporal sampling mismatch: got {rest.shape[2]}, expected {target_t - 1}"
                    )
                mask = torch.cat([first, rest], dim=2)
        else:
            expected = 1 + target_t * temporal_factor
            if mask.shape[2] != expected:
                raise ValueError(
                    f"Mask temporal mismatch: got {mask.shape[2]}, expected {expected}"
                )
            mask = mask[:, :, 1::temporal_factor]
            if mask.shape[2] != target_t:
                raise ValueError(
                    f"Mask temporal sampling mismatch: got {mask.shape[2]}, expected {target_t}"
                )
    else:
        if mask.shape[2] != target_t:
            raise ValueError(
                f"Mask temporal mismatch: got {mask.shape[2]}, expected {target_t}"
            )

    mask = _resize_spatial_bcthw(mask, target_h, target_w, mode="nearest")

    if dilate > 0:
        kernel = 2 * dilate + 1
        bt, h, w = mask.shape[0] * mask.shape[2], mask.shape[3], mask.shape[4]
        flat = mask.permute(0, 2, 1, 3, 4).reshape(bt, 1, h, w)
        flat = F.max_pool2d(flat, kernel_size=kernel, stride=1, padding=dilate)
        mask = flat.reshape(mask.shape[0], target_t, 1, h, w).permute(0, 2, 1, 3, 4)

    if mask.shape[0] != target_b:
        raise ValueError(
            f"Mask batch mismatch: got {mask.shape[0]}, expected {target_b}"
        )

    return mask.permute(0, 2, 1, 3, 4).to(dtype=dtype)


def blend_ttm(
    latents: torch.Tensor, ref_noisy: torch.Tensor, motion_mask: torch.Tensor
) -> torch.Tensor:
    """Blend in latent space with motion mask.

    Shapes:
      latents/ref_noisy: [B, T, C, H, W]
      motion_mask:       [B, T, 1, H, W]
    """
    bg_mask = 1.0 - motion_mask
    return latents * bg_mask + ref_noisy * motion_mask
