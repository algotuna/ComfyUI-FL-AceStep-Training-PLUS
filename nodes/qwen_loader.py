"""FL_AceStep_QwenAudioLoader.

Loads Qwen2-Audio-7B-Instruct (Apache-2.0) and emits a QWEN_AUDIO_TAGGER
handler for the tagger node. Registration does no heavy I/O; the model is
downloaded (first run) / loaded lazily inside the tagger node.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("FL_AceStep_Training")

try:
    from comfy.utils import ProgressBar
except Exception:  # noqa: BLE001
    ProgressBar = None

DEVICE_OPTIONS = ["auto", "cuda", "cpu"]
_DEFAULT_MODEL = "Qwen2-Audio-7B-Instruct"


class FL_AceStep_QwenAudioLoader:
    """Load Qwen2-Audio for open-vocabulary music tagging.

    Qwen "listens" to each track and proposes genre/instrument/mood tags with
    no pre-curated label list, so it scales to large, unaudited datasets where
    you cannot enumerate every instrument in advance.

    Pick the model from the dropdown (scanned from models/acestep/). The
    default entry auto-downloads Qwen2-Audio-7B-Instruct into that folder on
    first use. The 7B model loads via accelerate device_map and is offloaded
    after a tagging pass by the tagger node.
    """

    @classmethod
    def _scan_qwen_models(cls):
        """List Qwen model dirs under models/acestep/, plus the default entry."""
        found = []
        try:
            import folder_paths  # noqa: PLC0415
            acestep_dir = os.path.join(folder_paths.models_dir, "acestep")
        except Exception:  # noqa: BLE001
            acestep_dir = os.path.join("models", "acestep")

        if os.path.isdir(acestep_dir):
            for name in sorted(os.listdir(acestep_dir)):
                d = os.path.join(acestep_dir, name)
                if os.path.isfile(os.path.join(d, "config.json")) and "qwen" in name.lower():
                    found.append(name)

        # Always offer the default so first run (empty folder) is selectable
        # and triggers the auto-download.
        if _DEFAULT_MODEL not in found:
            found.insert(0, _DEFAULT_MODEL)
        return found

    @classmethod
    def INPUT_TYPES(cls):
        models = cls._scan_qwen_models()
        return {
            "required": {
                "model": (models, {
                    "default": models[0],
                    "tooltip": (
                        "Qwen2-Audio model from models/acestep/. The default "
                        "'Qwen2-Audio-7B-Instruct' auto-downloads (~16GB) on first "
                        "use. After a download, refresh the graph to see it listed."
                    ),
                }),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
            },
            "optional": {
                "custom_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional override: absolute, or relative to models/",
                    "tooltip": (
                        "Leave empty to use the dropdown. Set this to load a Qwen "
                        "model from a non-standard location — an absolute path, or a "
                        "path relative to your models/ directory."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("QWEN_AUDIO_TAGGER",)
    RETURN_NAMES = ("tagger",)
    FUNCTION = "load"
    CATEGORY = "FL AceStep/Loaders"

    def load(self, model, device, custom_path=""):
        import torch  # noqa: PLC0415
        from ..modules.qwen_tagger import load_qwen_audio_tagger  # noqa: PLC0415

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # custom_path wins; otherwise the dropdown selection under models/acestep.
        if custom_path and custom_path.strip():
            model_path = custom_path.strip()
        else:
            model_path = os.path.join("acestep", model)

        pbar = ProgressBar(1) if ProgressBar else None
        tagger = load_qwen_audio_tagger(device=device, model_path=model_path)
        if pbar:
            pbar.update(1)

        # Only the processor is resident now; the 7B model is loaded lazily by
        # the tagger node after it evicts other models, to avoid a 16GB OOM.
        logger.info("Qwen2-Audio processor ready (model loads when tagging starts)")
        return (tagger,)
