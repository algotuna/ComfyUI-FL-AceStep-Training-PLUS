"""FL_AceStep_QwenAudioLoader.

Loads Qwen2-Audio-7B-Instruct (Apache-2.0) and emits a QWEN_AUDIO_TAGGER
handler for the tagger node. Registration only does no I/O; the heavy load
happens lazily inside load() on first execution.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("FL_AceStep_Training")

try:
    from comfy.utils import ProgressBar
except Exception:  # noqa: BLE001
    ProgressBar = None

DEVICE_OPTIONS = ["auto", "cuda", "cpu"]


class FL_AceStep_QwenAudioLoader:
    """Load Qwen2-Audio for open-vocabulary music tagging.

    Qwen "listens" to each track and proposes genre/instrument/mood tags with
    no pre-curated label list, so it scales to large, unaudited datasets where
    you cannot enumerate every instrument in advance.

    The 7B model is heavy; it loads via accelerate device_map (CPU offload on a
    16GB card) and is offloaded after a tagging pass by the tagger node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty to auto-download Qwen2-Audio-7B-Instruct",
                }),
            },
        }

    RETURN_TYPES = ("QWEN_AUDIO_TAGGER",)
    RETURN_NAMES = ("tagger",)
    FUNCTION = "load"
    CATEGORY = "FL AceStep/Loaders"

    def load(self, device, model_path=""):
        import torch  # noqa: PLC0415
        from ..modules.qwen_tagger import load_qwen_audio_tagger  # noqa: PLC0415

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        pbar = ProgressBar(1) if ProgressBar else None
        tagger = load_qwen_audio_tagger(device=device, model_path=model_path)
        if pbar:
            pbar.update(1)

        # Only the processor is resident now; the 7B model is loaded lazily by
        # the tagger node after it evicts other models, to avoid a 16GB OOM.
        logger.info("Qwen2-Audio processor ready (model loads when tagging starts)")
        return (tagger,)
