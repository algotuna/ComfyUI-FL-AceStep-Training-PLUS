"""
ACEStep VAE Loader Node

Standalone VAE loader for ACEStep 1.5 — no input VAE connection required.
Pick a variant; the file is fetched on first use and a ComfyUI VAE is built
from it directly (the hosted files are already in ComfyUI key format, so no
diffusers->comfy conversion is needed).

Variants:
  ScragVAE (HQ) — community fine-tuned decoder that fixes the metallic
                  high-frequency sibilance in SFT outputs (improved 10-20 kHz
                  reconstruction). Hosted on this repo's GitHub releases,
                  sha256-verified, faithful re-key of scragnog/Ace-Step-1.5-ScragVAE.
  stock         — official ACE-Step 1.5 VAE (Comfy-Org repackaged).
"""

import logging

import safetensors.torch

from ..modules.model_downloader import (
    get_acestep_models_dir,
    ensure_scrag_vae,
    ensure_stock_comfy_vae,
    VAE_HQ_DIRNAME,
    SCRAGVAE_FILENAME,
    STOCK_VAE_FILENAME,
)

logger = logging.getLogger("FL_AceStep_Training")

VAE_VARIANTS = ["ScragVAE (HQ)", "stock"]


class FL_AceStep_VAELoader:
    """
    ACEStep VAE Loader

    Loads an ACEStep 1.5 VAE directly — no need to pipe a VAE in from a
    checkpoint loader. Select a variant and wire the output VAE into any
    downstream node (Preprocess Dataset, generation, etc.).

    ScragVAE (the default) fixes the metallic high-frequency sibilance in
    SFT-generated tracks. Weights auto-download on first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variant": (VAE_VARIANTS, {"default": "ScragVAE (HQ)"}),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "HuggingFace token (only if the stock download is rate-limited)",
                }),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "FL AceStep/Models"

    def load(self, variant="ScragVAE (HQ)", hf_token=""):
        import comfy.sd

        token = hf_token.strip() or None
        models_dir = get_acestep_models_dir()

        if variant.startswith("ScragVAE"):
            success, msg = ensure_scrag_vae(models_dir, token=token)
            filename = SCRAGVAE_FILENAME
        else:
            success, msg = ensure_stock_comfy_vae(models_dir, token=token)
            filename = STOCK_VAE_FILENAME

        if not success:
            raise RuntimeError(msg)

        weights_path = models_dir / VAE_HQ_DIRNAME / filename
        if not weights_path.exists():
            raise RuntimeError(f"VAE file missing after download: {weights_path}")

        logger.info(f"Loading {variant} VAE from {weights_path}")
        sd = safetensors.torch.load_file(str(weights_path), device="cpu")
        vae = comfy.sd.VAE(sd=sd)
        logger.info(f"{variant} VAE ready")
        return (vae,)
