"""
ScragVAE Loader Node

Loads the community ScragVAE decoder (scragnog/Ace-Step-1.5-ScragVAE) as a
drop-in replacement for the stock ACEStep 1.5 VAE. ScragVAE fixes the metallic
high-frequency sibilance caused by perceptual HF de-emphasis in the original
VAE's spectral loss, resulting in cleaner 10–20 kHz reconstruction.

Usage: wire the stock ACEStep VAE into this node, then wire the output VAE
into any node that accepts a VAE (Preprocess, SFT generation, etc.).
"""

import copy
import logging
from pathlib import Path

import torch
import safetensors.torch

from ..modules.model_downloader import get_acestep_models_dir, ensure_scrag_vae

logger = logging.getLogger("FL_AceStep_Training")


class FL_AceStep_ScragVAELoader:
    """
    ScragVAE Loader

    Replaces the ACEStep 1.5 VAE decoder with ScragVAE's improved weights,
    which fix high-frequency metallic artifacts in generated audio.

    Connect the stock ACEStep VAE here; the output is a standard VAE type
    compatible with all downstream nodes (Preprocess Dataset, generation, etc.).

    Auto-downloads scragnog/Ace-Step-1.5-ScragVAE on first use (~675 MB).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "HuggingFace token (leave blank if not needed)",
                }),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "FL AceStep/Models"

    def load(self, vae, hf_token=""):
        token = hf_token.strip() or None
        models_dir = get_acestep_models_dir()
        scragvae_dir = models_dir / "scragvae"
        weights_path = scragvae_dir / "diffusion_pytorch_model.safetensors"

        # Download if needed
        if not weights_path.exists():
            logger.info("ScragVAE weights not found — downloading...")
            success, msg = ensure_scrag_vae(models_dir, token=token)
            if not success:
                raise RuntimeError(f"ScragVAE download failed: {msg}")
            logger.info(msg)

        if not weights_path.exists():
            raise RuntimeError(
                f"Expected ScragVAE weights at {weights_path} but file not found "
                "after download. Check the download directory."
            )

        logger.info(f"Loading ScragVAE weights from {weights_path}")

        # Load ScragVAE state dict on CPU to avoid occupying VRAM during the swap
        scrag_sd = safetensors.torch.load_file(str(weights_path), device="cpu")
        logger.info(f"ScragVAE state dict: {len(scrag_sd)} keys")

        # Deep-copy the ComfyUI VAE so we don't modify the original
        # (the user may have the stock VAE wired elsewhere in the same workflow)
        new_vae = copy.deepcopy(vae)

        # The first_stage_model is the underlying AutoencoderOobleck.
        # Move it to CPU first so the weight swap doesn't touch VRAM.
        fsm = new_vae.first_stage_model
        original_device = next(fsm.parameters()).device
        if original_device != torch.device("cpu"):
            fsm = fsm.cpu()
            new_vae.first_stage_model = fsm

        missing, unexpected = fsm.load_state_dict(scrag_sd, strict=False)

        if missing:
            logger.warning(f"ScragVAE load — missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            logger.warning(f"ScragVAE load — unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        # Rebuild the ModelPatcher so ComfyUI's memory management tracks the
        # new model instance rather than the deep-copied one.
        try:
            import comfy.model_patcher
            new_vae.patcher = comfy.model_patcher.ModelPatcher(
                new_vae.first_stage_model,
                load_device=vae.patcher.load_device,
                offload_device=vae.patcher.offload_device,
            )
        except Exception as e:
            logger.warning(f"Could not rebuild ModelPatcher for ScragVAE ({e}); using deepcopy patcher")

        logger.info("ScragVAE loaded — high-frequency reconstruction improved")
        return (new_vae,)
