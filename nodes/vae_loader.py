"""
ACEStep VAE Loader Node

Standalone VAE loader for ACEStep 1.5. Picks a variant from a dropdown and
loads it directly — no input VAE connection required. Downloads from HuggingFace
on first use if the weights are not already present.

Variants:
  stock    — ACE-Step/Ace-Step1.5 VAE (standard quality)
  ScragVAE — scragnog/Ace-Step-1.5-ScragVAE (improved 10-20 kHz reconstruction,
             fixes metallic high-frequency sibilance in SFT outputs)
"""

import logging

import safetensors.torch

from ..modules.model_downloader import (
    get_acestep_models_dir,
    ensure_main_model,
    ensure_scrag_vae,
)

logger = logging.getLogger("FL_AceStep_Training")

VAE_VARIANTS = ["stock", "ScragVAE (HQ)"]


class FL_AceStep_VAELoader:
    """
    ACEStep VAE Loader

    Loads the ACEStep 1.5 VAE directly — no need to pipe a VAE from a
    checkpoint loader. Select a variant and wire the output VAE into any
    downstream node (Preprocess Dataset, generation, etc.).

    Auto-downloads weights on first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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

    def load(self, variant="stock", hf_token=""):
        import comfy.sd

        token = hf_token.strip() or None
        models_dir = get_acestep_models_dir()

        if variant == "stock":
            weights_path = models_dir / "vae" / "diffusion_pytorch_model.safetensors"
            if not weights_path.exists():
                logger.info("Stock ACEStep VAE not found — downloading main model (~several GB)...")
                success, msg = ensure_main_model(models_dir, token=token)
                if not success:
                    raise RuntimeError(f"Stock VAE download failed: {msg}")
        else:
            weights_path = models_dir / "scragvae" / "diffusion_pytorch_model.safetensors"
            if not weights_path.exists():
                logger.info("ScragVAE not cached — downloading (~675 MB)...")
                success, msg = ensure_scrag_vae(models_dir, token=token)
                if not success:
                    raise RuntimeError(f"ScragVAE download failed: {msg}")

        if not weights_path.exists():
            raise RuntimeError(
                f"VAE weights not found at {weights_path} after download attempt. "
                "Check your internet connection and models directory."
            )

        logger.info(f"Loading {variant} VAE from {weights_path}")
        sd = safetensors.torch.load_file(str(weights_path), device="cpu")
        vae = comfy.sd.VAE(sd=sd)
        logger.info(f"{variant} VAE ready")
        return (vae,)
