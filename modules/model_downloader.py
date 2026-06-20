"""
ACE-Step Model Downloader

Handles automatic downloading of models from HuggingFace Hub
and manages model paths for ComfyUI integration.
"""

import os
import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Dict

try:
    from huggingface_hub import snapshot_download, HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False

logger = logging.getLogger("FL_AceStep_Training")

# Main model repository containing core components
MAIN_MODEL_REPO = "ACE-Step/Ace-Step1.5"

# Model registry: maps local directory names to HuggingFace repo IDs
MODEL_REGISTRY: Dict[str, str] = {
    # DiT models
    "acestep-v15-turbo": MAIN_MODEL_REPO,  # Part of main repo
    "acestep-v15-turbo-shift1": "ACE-Step/acestep-v15-turbo-shift1",
    "acestep-v15-turbo-shift3": "ACE-Step/acestep-v15-turbo-shift3",
    "acestep-v15-sft": "ACE-Step/acestep-v15-sft",
    "acestep-v15-base": "ACE-Step/acestep-v15-base",
    "acestep-v15-turbo-continuous": "ACE-Step/acestep-v15-turbo-continuous",
    # LLM models
    "acestep-5Hz-lm-0.6B": "ACE-Step/acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-1.7B": MAIN_MODEL_REPO,  # Part of main repo
    "acestep-5Hz-lm-4B": "ACE-Step/acestep-5Hz-lm-4B",
    # VAE and text encoder (part of main repo)
    "vae": MAIN_MODEL_REPO,
    "Qwen3-Embedding-0.6B": MAIN_MODEL_REPO,
}

# ---------------------------------------------------------------------------
# High-quality VAE assets for the FL AceStep VAE Loader node.
#
# Both files are in ComfyUI key format (`decoder.layers.N...`) so they load
# directly via comfy.sd.VAE(sd=...) with no diffusers->comfy key conversion.
#
# ScragVAE is mirrored on THIS repo's GitHub releases (so the project owns the
# asset rather than depending on a third-party HF repo staying up). It is a
# bit-for-bit faithful re-key of scragnog/Ace-Step-1.5-ScragVAE: all 365
# tensors were verified identical by full-content hash; only key names and
# singleton-dim shapes differ. Original work (c) scragnog, MIT.
#
# Stock comes from Comfy-Org's official repackaged ComfyUI files.
# ---------------------------------------------------------------------------
VAE_HQ_DIRNAME = "vae_hq"

SCRAGVAE_FILENAME = "ace_1.5_scrag_vae.safetensors"
SCRAGVAE_URL = (
    "https://github.com/algotuna/ComfyUI-FL-AceStep-Training-PLUS/"
    "releases/download/scragvae-v1.0/ace_1.5_scrag_vae.safetensors"
)
SCRAGVAE_SHA256 = "b89c6a9b0ab8e799923a2d35ee69810fbb07124f9e86540bf00870163a8e0b0c"

STOCK_VAE_FILENAME = "ace_1.5_vae.safetensors"
STOCK_VAE_URL = (
    "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/"
    "resolve/main/split_files/vae/ace_1.5_vae.safetensors"
)
STOCK_VAE_SHA256 = None  # official file, not pinned

# Components included in the main model download
MAIN_MODEL_COMPONENTS = [
    "acestep-v15-turbo",
    "vae",
    "Qwen3-Embedding-0.6B",
    "acestep-5Hz-lm-1.7B",
]


def get_acestep_models_dir(custom_dir: Optional[str] = None) -> Path:
    """Get the ACE-Step models directory path."""
    if custom_dir:
        return Path(custom_dir)

    if COMFYUI_AVAILABLE:
        return Path(folder_paths.models_dir) / "acestep"
    else:
        # Fallback to current directory
        return Path("./models/acestep")


def check_model_exists(model_name: str, models_dir: Optional[Path] = None) -> bool:
    """Check if a model exists in the models directory."""
    if models_dir is None:
        models_dir = get_acestep_models_dir()

    model_path = models_dir / model_name

    # Check for config.json or model index as indicator
    config_file = model_path / "config.json"
    index_file = model_path / "model_index.json"

    return config_file.exists() or index_file.exists()


def check_main_model_exists(models_dir: Optional[Path] = None) -> bool:
    """Check if all main model components exist."""
    if models_dir is None:
        models_dir = get_acestep_models_dir()

    for component in MAIN_MODEL_COMPONENTS:
        if not check_model_exists(component, models_dir):
            return False
    return True


def download_model(
    repo_id: str,
    local_dir: Path,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """Download a model from HuggingFace."""
    if not HF_AVAILABLE:
        return False, "huggingface_hub not installed. Run: pip install huggingface-hub"

    try:
        logger.info(f"Downloading from {repo_id} to {local_dir}...")

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token,
        )

        return True, f"Successfully downloaded to {local_dir}"
    except Exception as e:
        error_msg = f"Failed to download {repo_id}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def ensure_main_model(
    models_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Ensure the main ACE-Step model is available, downloading if necessary.

    The main model includes:
    - acestep-v15-turbo (default DiT model)
    - vae (audio encoder/decoder)
    - Qwen3-Embedding-0.6B (text encoder)
    - acestep-5Hz-lm-1.7B (default LM model)
    """
    if models_dir is None:
        models_dir = get_acestep_models_dir()

    models_dir.mkdir(parents=True, exist_ok=True)

    if check_main_model_exists(models_dir):
        return True, "Main model is available"

    logger.info("=" * 60)
    logger.info("Main model not found. Starting automatic download...")
    logger.info("This may take a while depending on your internet connection...")
    logger.info("=" * 60)

    return download_model(MAIN_MODEL_REPO, models_dir, token)


def ensure_dit_model(
    model_name: str,
    models_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """Ensure a specific DiT model is available."""
    if models_dir is None:
        models_dir = get_acestep_models_dir()

    # Check if it's the default turbo model (part of main)
    if model_name == "acestep-v15-turbo":
        if check_model_exists(model_name, models_dir):
            return True, f"DiT model '{model_name}' is available"
        return ensure_main_model(models_dir, token)

    # Check if it's a known sub-model
    if model_name in MODEL_REGISTRY:
        model_path = models_dir / model_name

        if check_model_exists(model_name, models_dir):
            return True, f"DiT model '{model_name}' is available"

        repo_id = MODEL_REGISTRY[model_name]
        if repo_id == MAIN_MODEL_REPO:
            # This model is part of main, ensure main is downloaded
            return ensure_main_model(models_dir, token)

        logger.info(f"DiT model '{model_name}' not found. Downloading...")
        return download_model(repo_id, model_path, token)

    return False, f"Unknown DiT model: {model_name}"


def ensure_lm_model(
    model_name: str = "acestep-5Hz-lm-1.7B",
    models_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """Ensure an LM model is available for audio understanding."""
    if models_dir is None:
        models_dir = get_acestep_models_dir()

    # Default LM is part of main model
    if model_name == "acestep-5Hz-lm-1.7B":
        if check_model_exists(model_name, models_dir):
            return True, f"LM model '{model_name}' is available"
        return ensure_main_model(models_dir, token)

    # Other LM variants
    if model_name in MODEL_REGISTRY:
        model_path = models_dir / model_name

        if check_model_exists(model_name, models_dir):
            return True, f"LM model '{model_name}' is available"

        repo_id = MODEL_REGISTRY[model_name]
        logger.info(f"LM model '{model_name}' not found. Downloading...")
        return download_model(repo_id, model_path, token)

    return False, f"Unknown LM model: {model_name}"


def ensure_vae(
    models_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """Ensure the VAE model is available."""
    if models_dir is None:
        models_dir = get_acestep_models_dir()

    if check_model_exists("vae", models_dir):
        return True, "VAE is available"

    # VAE is part of main model
    return ensure_main_model(models_dir, token)


def ensure_text_encoder(
    models_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """Ensure the text encoder (Qwen3-Embedding) is available."""
    if models_dir is None:
        models_dir = get_acestep_models_dir()

    if check_model_exists("Qwen3-Embedding-0.6B", models_dir):
        return True, "Text encoder is available"

    # Text encoder is part of main model
    return ensure_main_model(models_dir, token)


def get_model_path(model_name: str, models_dir: Optional[Path] = None) -> Optional[Path]:
    """Get the path to a model if it exists."""
    if models_dir is None:
        models_dir = get_acestep_models_dir()

    model_path = models_dir / model_name

    if check_model_exists(model_name, models_dir):
        return model_path

    return None


def list_available_models(models_dir: Optional[Path] = None) -> Dict[str, bool]:
    """List all known models and their availability status."""
    if models_dir is None:
        models_dir = get_acestep_models_dir()

    status = {}
    for model_name in MODEL_REGISTRY.keys():
        status[model_name] = check_model_exists(model_name, models_dir)

    return status


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Compute the sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _stream_download(
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """
    Stream a single file to `dest` with progress logging and optional
    sha256 verification. Downloads to a .part file and atomically renames
    on success so an interrupted download never leaves a truncated file
    that looks complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    headers = {"User-Agent": "FL-AceStep-Training"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("content-length", 0))
        got = 0
        next_log = 0
        with open(tmp, "wb") as f:
            while True:
                buf = resp.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                if total and got >= next_log:
                    logger.info(f"  {dest.name}: {got / 1e6:.0f}/{total / 1e6:.0f} MB")
                    next_log += max(total // 10, 1)

    if expected_sha256:
        actual = _sha256_file(tmp)
        if actual.lower() != expected_sha256.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"sha256 mismatch for {dest.name}: expected {expected_sha256}, got {actual}. "
                "The download may be corrupted or tampered with; not installing."
            )

    tmp.replace(dest)
    return dest


def ensure_scrag_vae(
    models_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Ensure the ScragVAE weights are available, downloading from this repo's
    GitHub release if necessary.

    ScragVAE is a community fine-tune of the ACEStep 1.5 VAE decoder that
    improves high-frequency reconstruction (10-20 kHz), fixing the metallic
    sibilance in generated audio. The hosted file is in ComfyUI key format
    and sha256-pinned (see SCRAGVAE_SHA256).
    """
    if models_dir is None:
        models_dir = get_acestep_models_dir()

    dest = models_dir / VAE_HQ_DIRNAME / SCRAGVAE_FILENAME
    if dest.exists():
        return True, f"ScragVAE already available at {dest}"

    logger.info(f"ScragVAE not found. Downloading from {SCRAGVAE_URL}")
    try:
        _stream_download(SCRAGVAE_URL, dest, expected_sha256=SCRAGVAE_SHA256, token=token)
        return True, f"ScragVAE downloaded to {dest}"
    except Exception as e:
        error_msg = f"Failed to download ScragVAE: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def ensure_stock_comfy_vae(
    models_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Ensure the stock ACE-Step 1.5 VAE (ComfyUI format) is available,
    downloading Comfy-Org's official repackaged file if necessary.
    """
    if models_dir is None:
        models_dir = get_acestep_models_dir()

    dest = models_dir / VAE_HQ_DIRNAME / STOCK_VAE_FILENAME
    if dest.exists():
        return True, f"Stock VAE already available at {dest}"

    logger.info(f"Stock VAE not found. Downloading from {STOCK_VAE_URL}")
    try:
        _stream_download(STOCK_VAE_URL, dest, expected_sha256=STOCK_VAE_SHA256, token=token)
        return True, f"Stock VAE downloaded to {dest}"
    except Exception as e:
        error_msg = f"Failed to download stock VAE: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
