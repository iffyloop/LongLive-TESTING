from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from torchvision.io import read_video


class TTMInputProvider(ABC):
    @abstractmethod
    def get_motion_chunk(
        self, start_frame: int, num_frames: int, batch_size: int
    ) -> torch.Tensor:
        """Return motion tensor with shape [B, C, T, H, W], value range [0, 1]."""

    @abstractmethod
    def get_mask_chunk(
        self, start_frame: int, num_frames: int, batch_size: int
    ) -> torch.Tensor:
        """Return mask tensor with shape [B, 1, T, H, W], value range [0, 1]."""


def _load_video_tensor(path: str) -> torch.Tensor:
    frames, _, _ = read_video(path, pts_unit="sec")
    if frames.numel() == 0:
        raise ValueError(f"No frames found in video: {path}")
    # [T, H, W, C] -> [C, T, H, W]
    return (frames.float() / 255.0).permute(3, 0, 1, 2).contiguous()


def _slice_with_repeat(
    video_cthw: torch.Tensor, start_frame: int, num_frames: int
) -> torch.Tensor:
    total_frames = int(video_cthw.shape[1])
    if start_frame >= total_frames:
        last = video_cthw[:, -1:, :, :]
        return last.repeat(1, num_frames, 1, 1)

    end = min(total_frames, start_frame + num_frames)
    chunk = video_cthw[:, start_frame:end, :, :]
    if chunk.shape[1] < num_frames:
        pad = chunk[:, -1:, :, :].repeat(1, num_frames - int(chunk.shape[1]), 1, 1)
        chunk = torch.cat([chunk, pad], dim=1)
    return chunk


def _slice_strict(
    video_cthw: torch.Tensor, start_frame: int, num_frames: int, *, name: str
) -> torch.Tensor:
    total_frames = int(video_cthw.shape[1])
    end = start_frame + num_frames
    if start_frame < 0:
        raise ValueError(f"{name}: start_frame must be >= 0, got {start_frame}")
    if end > total_frames:
        raise ValueError(
            f"{name}: requested frames [{start_frame}, {end}) exceed available {total_frames} frames"
        )
    return video_cthw[:, start_frame:end, :, :]


def _ensure_mask_1cthw(mask_cthw: torch.Tensor) -> torch.Tensor:
    # [C, T, H, W] -> [1, T, H, W]
    if mask_cthw.shape[0] == 1:
        return mask_cthw
    return mask_cthw.max(dim=0, keepdim=True).values


class FileTTMInputProvider(TTMInputProvider):
    def __init__(self, motion_video_path: str, mask_video_path: str):
        self.motion_video = _load_video_tensor(motion_video_path)
        self.mask_video = _ensure_mask_1cthw(_load_video_tensor(mask_video_path))

    def get_motion_chunk(
        self, start_frame: int, num_frames: int, batch_size: int
    ) -> torch.Tensor:
        chunk = _slice_strict(
            self.motion_video,
            start_frame,
            num_frames,
            name="motion_signal_video_path",
        ).unsqueeze(0)
        if batch_size > 1:
            chunk = chunk.repeat(batch_size, 1, 1, 1, 1)
        return chunk

    def get_mask_chunk(
        self, start_frame: int, num_frames: int, batch_size: int
    ) -> torch.Tensor:
        chunk = _slice_strict(
            self.mask_video,
            start_frame,
            num_frames,
            name="motion_signal_mask_path",
        ).unsqueeze(0)
        if batch_size > 1:
            chunk = chunk.repeat(batch_size, 1, 1, 1, 1)
        return chunk


class QueueTTMInputProvider(TTMInputProvider):
    """
    Placeholder streaming provider interface.
    Implementations can subclass this and fetch chunks from a realtime queue/socket.
    """

    def get_motion_chunk(
        self, start_frame: int, num_frames: int, batch_size: int
    ) -> torch.Tensor:
        raise NotImplementedError("QueueTTMInputProvider is an interface placeholder")

    def get_mask_chunk(
        self, start_frame: int, num_frames: int, batch_size: int
    ) -> torch.Tensor:
        raise NotImplementedError("QueueTTMInputProvider is an interface placeholder")
