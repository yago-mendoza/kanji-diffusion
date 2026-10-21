"""The latent denoiser and the noise schedule."""

from __future__ import annotations

from diffusers import DDPMScheduler, UNet2DConditionModel

PRESETS = {
    # Fits a laptop CPU: 17M parameters, ~2.4 s per batch of 64 on 8 Zen 5 cores.
    "small": dict(block_out_channels=(64, 128, 192), layers_per_block=2),
    # For a GPU.
    "base": dict(block_out_channels=(128, 256, 384), layers_per_block=2),
}


def build_unet(latent_size: int, latent_channels: int, text_dim: int, preset: str = "small") -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=latent_size,
        in_channels=latent_channels,
        out_channels=latent_channels,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=text_dim,
        attention_head_dim=8,
        norm_num_groups=32,
        **PRESETS[preset],
    )


def build_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
        clip_sample=False,
    )
