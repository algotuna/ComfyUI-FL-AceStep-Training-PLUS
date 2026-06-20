"""
ACEStep VAE Loader Node

Single node for selecting the ACEStep 1.5 VAE variant. Wire your checkpoint
VAE in, pick a variant from the dropdown, and the output is ready to use
downstream. ScragVAE auto-downloads on first use (~675 MB).

Variants:
  stock    — pass the checkpoint VAE through unchanged
  ScragVAE — swap in the community fine-tuned decoder that fixes the metallic
             high-frequency sibilance caused by the original VAE's spectral loss
"""

import copy
import logging

import torch
import safetensors.torch

from ..modules.model_downloader import get_acestep_models_dir, ensure_scrag_vae

logger = logging.getLogger("FL_AceStep_Training")

VAE_VARIANTS = ["stock", "ScragVAE (HQ)"]


class FL_AceStep_VAELoader:
    """
    ACEStep VAE Loader

    Select the VAE variant to use for preprocessing and generation.
    Wire the checkpoint VAE in once; toggle between stock and ScragVAE
    without rewiring your workflow.

    ScragVAE fixes the metallic high-frequency sibilance in SFT-generated
    tracks by improving 10-20 kHz reconstruction (scragnog/Ace-Step-1.5-ScragVAE).
    Auto-downloads on first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "variant": (VAE_VARIANTS, {"default": "stock"}),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "HuggingFace token (only needed for gated repos)",
                }),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "FL AceStep/Models"

    def load(self, vae, variant="stock", hf_token=""):
        if variant == "stock":
            return (vae,)

        # ScragVAE path
        token = hf_token.strip() or None
        models_dir = get_acestep_models_dir()
        weights_path = models_dir / "scragvae" / "diffusion_pytorch_model.safetensors"

        if not weights_path.exists():
            logger.info("ScragVAE not cached — downloading (~675 MB)...")
            success, msg = ensure_scrag_vae(models_dir, token=token)
            if not success:
                raise RuntimeError(f"ScragVAE download failed: {msg}")
            logger.info(msg)

        if not weights_path.exists():
            raise RuntimeError(
                f"Expected ScragVAE weights at {weights_path} but file missing after download."
            )

        logger.info(f"Applying ScragVAE weights from {weights_path}")
        scrag_sd = safetensors.torch.load_file(str(weights_path), device="cpu")

        # Deep-copy so the original VAE in the workflow is unaffected
        new_vae = copy.deepcopy(vae)

        fsm = new_vae.first_stage_model
        if next(fsm.parameters()).device != torch.device("cpu"):
            fsm.cpu()

        missing, unexpected = fsm.load_state_dict(scrag_sd, strict=False)
        if missing:
            logger.warning(f"ScragVAE: {len(missing)} missing keys (expected if encoder-only checkpoint)")
        if unexpected:
            logger.warning(f"ScragVAE: {len(unexpected)} unexpected keys")

        try:
            import comfy.model_patcher
            new_vae.patcher = comfy.model_patcher.ModelPatcher(
                new_vae.first_stage_model,
                load_device=vae.patcher.load_device,
                offload_device=vae.patcher.offload_device,
            )
        except Exception as e:
            logger.warning(f"Could not rebuild ModelPatcher ({e}); deepcopy patcher retained")

        logger.info("ScragVAE ready")
        return (new_vae,)
