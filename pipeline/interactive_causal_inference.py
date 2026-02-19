# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Dict, List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import (
    gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
)
from pipeline.causal_inference import CausalInferencePipeline
import torch.distributed as dist
from utils.debug_option import DEBUG
from utils.ttm_ops import blend_ttm, prepare_motion_latents, prepare_motion_mask
from utils.ttm_stream import TTMInputProvider


class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(
            args, device, generator=generator, text_encoder=text_encoder, vae=vae
        )
        self.global_sink = getattr(args, "global_sink", False)

    def _get_vae_temporal_factor(self) -> int:
        temporal_downsample = getattr(self.vae.model, "temperal_downsample", None)
        if temporal_downsample is None:
            return 1
        factor = 1
        for use_downsample in temporal_downsample:
            if bool(use_downsample):
                factor *= 2
        return max(1, int(factor))

    def _get_vae_spatial_factor(self) -> int:
        encoder = getattr(self.vae.model, "encoder", None)
        downsamples = getattr(encoder, "downsamples", None)
        if downsamples is None:
            return 8

        factor = 1
        for layer in downsamples:
            mode = getattr(layer, "mode", None)
            if mode in ("downsample2d", "downsample3d"):
                factor *= 2

        return max(1, int(factor))

    # Internal helpers
    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()

        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        # recache
        if current_start_frame == 0:
            return

        num_recache_frames = (
            current_start_frame
            if self.local_attn_size == -1
            else min(self.local_attn_size, current_start_frame)
        )
        recache_start_frame = current_start_frame - num_recache_frames

        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == "cpu":
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(
            f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}"
        )

        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size,
        )

        context_timestep = (
            torch.ones(
                [batch_size, num_recache_frames], device=device, dtype=torch.int64
            )
            * self.args.context_noise
        )

        self.generator.model.block_mask = block_mask

        # recache
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
            )

        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        # encode all prompts
        print(text_prompts_list)
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device("cpu") if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype,
        )

        # initialize caches
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(
            f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})"
        )

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(
            f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}"
        )
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0  # current segment index
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if DEBUG:
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

        for current_num_frames in all_num_frames:
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                self._recache_after_switch(
                    output, current_start_frame, cond_list[segment_idx]
                )
                if DEBUG:
                    print(
                        f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                    )
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
                print(f"segment_idx: {segment_idx}")
                print(
                    f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}"
                )
            cond_in_use = cond_list[segment_idx]

            noisy_input = noise[
                :, current_start_frame : current_start_frame + current_num_frames
            ]

            # ---------------- Spatial denoising loop ----------------
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            # Record output
            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred.to(output.device)

            # rerun with clean context to update cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # Update frame pointer
            current_start_frame += current_num_frames

        # Standard decoding
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video

    def inference_stream_ttm(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        ttm_provider: TTMInputProvider,
        ttm_cfg: Dict,
        return_latents: bool = False,
        low_memory: bool = False,
        on_video_chunk: Optional[Callable[[torch.Tensor, int], None]] = None,
    ):
        """Interactive inference with TTM latent control and optional chunk callback."""
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        assert batch_size == 1, "TTM streaming path currently supports batch_size=1"

        num_blocks = num_output_frames // self.num_frame_per_block
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device("cpu") if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype,
        )

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(
            f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})"
        )

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        tweak_index = int(ttm_cfg["tweak_index"])
        tstrong_index = int(ttm_cfg["tstrong_index"])
        warmup_blocks = int(ttm_cfg["warmup_blocks"])
        threshold = float(ttm_cfg["threshold"])
        mask_dilate = int(ttm_cfg["mask_dilate"])
        deterministic_ref_noise = bool(ttm_cfg["deterministic_ref_noise"])
        apply_after_prompt_switch = bool(ttm_cfg["apply_after_prompt_switch"])
        use_cached_decode = bool(ttm_cfg.get("use_cached_decode", False))
        temporal_factor = self._get_vae_temporal_factor()
        spatial_factor = self._get_vae_spatial_factor()

        step_count = len(self.denoising_step_list)
        if not (0 <= tweak_index < tstrong_index <= step_count):
            raise ValueError(
                f"Invalid TTM schedule: tweak_index={tweak_index}, tstrong_index={tstrong_index}, num_steps={step_count}"
            )

        rng = None
        if deterministic_ref_noise:
            rng = torch.Generator(device=noise.device)
            rng.manual_seed(int(ttm_cfg["ref_noise_seed"]))

        block_idx = 0
        for current_num_frames in all_num_frames:
            switched_in_this_block = False
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                self._recache_after_switch(
                    output, current_start_frame, cond_list[segment_idx]
                )
                switched_in_this_block = True
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
            cond_in_use = cond_list[segment_idx]

            if temporal_factor > 1:
                if current_start_frame == 0:
                    motion_start_frame = 0
                    motion_num_frames = 1 + (current_num_frames - 1) * temporal_factor
                    drop_first_latent = False
                else:
                    motion_start_frame = (current_start_frame - 1) * temporal_factor
                    motion_num_frames = 1 + current_num_frames * temporal_factor
                    drop_first_latent = True
            else:
                motion_start_frame = current_start_frame
                motion_num_frames = current_num_frames
                drop_first_latent = False

            motion_chunk = ttm_provider.get_motion_chunk(
                motion_start_frame, motion_num_frames, batch_size
            )
            mask_chunk = ttm_provider.get_mask_chunk(
                motion_start_frame, motion_num_frames, batch_size
            )

            target_shape = (
                batch_size,
                current_num_frames,
                num_channels,
                height,
                width,
            )
            target_pixel_hw = (height * spatial_factor, width * spatial_factor)

            ref_latents = prepare_motion_latents(
                motion_chunk,
                self.vae,
                target_shape,
                target_pixel_hw=target_pixel_hw,
                drop_first_latent=drop_first_latent,
                dtype=noise.dtype,
                device=noise.device,
            )
            motion_mask = prepare_motion_mask(
                mask_chunk,
                target_shape,
                target_pixel_hw=target_pixel_hw,
                temporal_factor=temporal_factor,
                drop_first_latent=drop_first_latent,
                threshold=threshold,
                dilate=mask_dilate,
                dtype=noise.dtype,
                device=noise.device,
            )

            if rng is None:
                fixed_noise = torch.randn_like(ref_latents)
            else:
                fixed_noise = torch.randn(
                    ref_latents.shape,
                    device=ref_latents.device,
                    dtype=ref_latents.dtype,
                    generator=rng,
                )

            start_timestep = self.denoising_step_list[tweak_index]
            start_timestep_tensor = start_timestep * torch.ones(
                [batch_size * current_num_frames],
                device=noise.device,
                dtype=torch.long,
            )
            noisy_input = self.scheduler.add_noise(
                ref_latents.flatten(0, 1),
                fixed_noise.flatten(0, 1),
                start_timestep_tensor,
            ).unflatten(0, ref_latents.shape[:2])

            denoised_pred = None
            last_timestep = None
            for index in range(tweak_index, step_count):
                current_timestep = self.denoising_step_list[index]
                last_timestep = current_timestep
                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )

                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=cond_in_use,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                if index < step_count - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noise_for_step = torch.randn_like(denoised_pred.flatten(0, 1))
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        noise_for_step,
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])

                    ttm_active = (
                        index < tstrong_index
                        and block_idx >= warmup_blocks
                        and (apply_after_prompt_switch or not switched_in_this_block)
                    )
                    if ttm_active:
                        ref_noisy = self.scheduler.add_noise(
                            ref_latents.flatten(0, 1),
                            fixed_noise.flatten(0, 1),
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, ref_latents.shape[:2])
                        noisy_input = blend_ttm(noisy_input, ref_noisy, motion_mask)

            if denoised_pred is None or last_timestep is None:
                raise RuntimeError("TTM inference did not produce denoised predictions")

            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred.to(output.device)

            context_timestep = (
                torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                )
                * self.args.context_noise
            )
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if on_video_chunk is not None:
                chunk_video = self.vae.decode_to_pixel(
                    denoised_pred.to(noise.device), use_cache=use_cached_decode
                )
                chunk_video = (chunk_video * 0.5 + 0.5).clamp(0, 1)
                on_video_chunk(chunk_video, current_start_frame)

            current_start_frame += current_num_frames
            block_idx += 1

        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video
