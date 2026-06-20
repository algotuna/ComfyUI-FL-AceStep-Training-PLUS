"""Qwen2-Audio open-vocabulary music tagger.

Phase 1 of the tagging pipeline. Qwen2-Audio "listens" to each track and
returns free-form structured tags (genre / instruments / moods / a short
description) WITHOUT any pre-curated label list — it discovers the vocabulary
from the audio itself. This is the answer to the "I can't pre-list every
instrument in 100 unaudited songs" problem.

Licensing: Qwen2-Audio-7B-Instruct is Apache-2.0 (commercial use permitted).

Memory: the 7B model is heavy for a 16GB card, so it loads via accelerate's
device_map (CPU offload as needed). Because an accelerate-dispatched model
cannot be hand-moved to CPU afterwards, offload() FULLY RELEASES the model to
free VRAM before the rest of the dataset pipeline runs; tag_audio() lazily
reloads it if called again. bitsandbytes 8-bit is deliberately avoided: no
guaranteed sm_120 (Blackwell) kernels.

Heavy imports (transformers, librosa, torch) live INSIDE methods so a missing
optional dependency never breaks node registration at ComfyUI startup.
"""

from __future__ import annotations

import gc
import json
import logging
import re

logger = logging.getLogger("FL_AceStep_Training")

QWEN_AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

# The discovery prompt. We ask for STRICT JSON so the output parses
# deterministically. Qwen is told to use concise, lowercase, conventional
# tag words (so aggregation downstream can dedupe synonyms) and to only list
# instruments it is reasonably confident it can hear (curbs hallucination).
_DISCOVERY_PROMPT = (
    "Listen to this music and identify its style. Respond with ONLY a JSON "
    "object, no prose before or after, in exactly this form:\n"
    '{"genre": ["..."], "instruments": ["..."], "moods": ["..."], '
    '"description": "one short sentence"}\n\n'
    "Rules:\n"
    "- Use concise, lowercase, conventional tag words (e.g. \"drum and bass\", "
    "\"electric guitar\", \"oud\", \"energetic\").\n"
    "- genre: 1-3 entries, most specific first.\n"
    "- instruments: only instruments you can actually hear; name the specific "
    "instrument if you can (e.g. \"oud\" not \"string instrument\").\n"
    "- moods: 1-3 mood/energy words.\n"
    "- Do not invent instruments you are unsure about."
)


class QwenAudioTagger:
    """Handler wrapping Qwen2-Audio for open-vocabulary music tagging.

    The processor is kept resident (it is light); the model is released by
    offload() and reloaded on demand so a tagging pass can hand VRAM back to
    the preprocessing/training stages on a 16GB card.
    """

    def __init__(self, processor, device, source, model_id=QWEN_AUDIO_MODEL_ID):
        self.processor = processor
        self.device = device
        self.source = source          # local dir or HF hub id
        self.model_id = model_id
        self.model = None             # loaded lazily / released by offload()

    # -- lifecycle --------------------------------------------------------

    def _load_model(self):
        """(Re)load the model if it is not currently resident."""
        if self.model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import Qwen2AudioForConditionalGeneration  # noqa: PLC0415

        logger.info(f"Loading Qwen2-Audio model from {self.source} (device={self.device})")
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.source,
                dtype=torch.bfloat16,
                device_map="auto",        # accelerate handles 16GB offload
            )
        else:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.source, dtype=torch.float32
            ).to("cpu")
        self.model.eval()

    def offload(self):
        """Fully release the model and free its VRAM.

        A device_map model keeps accelerate hooks that hold its GPU tensors
        alive, so merely dropping the reference can leave ~12GB resident and
        OOM the next stage (preprocessing). Remove the hooks first, drop the
        reference, then hard-clear the CUDA cache. Reloaded lazily on the next
        tag_audio() call.
        """
        model = self.model
        self.model = None
        if model is not None:
            try:
                from accelerate.hooks import remove_hook_from_module  # noqa: PLC0415
                remove_hook_from_module(model, recurse=True)
            except Exception:  # noqa: BLE001
                pass
            del model
        gc.collect()
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass

    # -- tagging ----------------------------------------------------------

    def _run_processor(self, text, audio):
        """Call the processor, tolerating the audio=/audios= API variation.

        Current transformers expects audio=[...]; older Qwen2-Audio builds used
        audios=[...]. Critically, the WRONG name is ignored with a warning
        (not a TypeError), which silently drops the audio - so we must use the
        current name first and only fall back on a genuine TypeError.
        """
        try:
            return self.processor(
                text=text, audio=[audio], return_tensors="pt", padding=True
            )
        except TypeError:
            return self.processor(
                text=text, audios=[audio], return_tensors="pt", padding=True
            )

    def tag_audio(self, audio_path: str, max_new_tokens: int = 256) -> dict:
        """Tag one audio file. Returns a normalised dict:

            {"genre": [...], "instruments": [...], "moods": [...],
             "description": "..."}

        Never raises for ordinary failures — on error returns empty lists so a
        single bad file does not abort the whole dataset pass.
        """
        import librosa  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self._load_model()

        try:
            sr = self.processor.feature_extractor.sampling_rate
            audio, _ = librosa.load(audio_path, sr=sr, mono=True)

            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": _DISCOVERY_PROMPT},
                ]},
            ]
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self._run_processor(text, audio)
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            # Strip the prompt tokens, decode only the completion.
            generated = generated[:, inputs.input_ids.size(1):]
            response = self.processor.batch_decode(
                generated, skip_special_tokens=True
            )[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Qwen tagging failed for {audio_path}: {exc}")
            return {"genre": [], "instruments": [], "moods": [], "description": ""}

        return self._parse_tags(response)

    @staticmethod
    def _parse_tags(response: str) -> dict:
        """Extract the JSON object from Qwen's reply and normalise it.

        Robust to surrounding prose / markdown fences: grabs the first {...}
        block. Falls back to empty lists if no valid JSON is present.
        """
        result = {"genre": [], "instruments": [], "moods": [], "description": ""}

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            logger.warning(f"Qwen returned no JSON object; raw: {response[:200]!r}")
            return result

        raw_json = match.group()
        data = None
        # Some Qwen outputs come back with backslash-escaped quotes
        # (e.g. {\"genre\": ...}), which is not valid JSON. Try the raw text
        # first, then an unescaped variant, before giving up.
        for candidate in (raw_json, raw_json.replace('\\"', '"').replace("\\'", "'")):
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            logger.warning(f"Qwen JSON did not parse; raw: {raw_json[:200]!r}")
            return result

        def _clean_list(value):
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                return []
            out = []
            for item in value:
                tag = str(item).strip().lower()
                if tag and tag not in out:
                    out.append(tag)
            return out

        result["genre"] = _clean_list(data.get("genre"))
        result["instruments"] = _clean_list(data.get("instruments"))
        result["moods"] = _clean_list(data.get("moods"))
        desc = data.get("description", "")
        result["description"] = str(desc).strip() if isinstance(desc, (str, int, float)) else ""
        return result


def resolve_qwen_dir(model_path: str = "") -> str:
    """Resolve the local directory the Qwen model lives in / downloads to.

    - empty            -> <models_dir>/acestep/Qwen2-Audio-7B-Instruct
    - relative path    -> joined under <models_dir> (e.g. "diffusion_models/
                          qwen/Qwen2-Audio-7B-Instruct")
    - absolute path    -> used as-is
    """
    import os  # noqa: PLC0415

    try:
        import folder_paths  # noqa: PLC0415
        models_dir = folder_paths.models_dir
    except Exception:  # noqa: BLE001
        models_dir = os.path.abspath("./models")

    model_path = (model_path or "").strip()
    if not model_path:
        return os.path.join(models_dir, "acestep", "Qwen2-Audio-7B-Instruct")
    if os.path.isabs(model_path):
        return model_path
    return os.path.join(models_dir, model_path)


def ensure_qwen_downloaded(target_dir: str, repo_id: str = QWEN_AUDIO_MODEL_ID) -> str:
    """Download the model into target_dir (clean filenames) if missing.

    Uses snapshot_download with a real local dir so files land as
    config.json / *.safetensors / tokenizer.json etc. in the ComfyUI tree,
    not as HF content-addressed cache blobs. Returns target_dir.
    """
    import os  # noqa: PLC0415

    if os.path.exists(os.path.join(target_dir, "config.json")):
        return target_dir

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Downloading {repo_id} -> {target_dir} (first run, ~16GB)...")
    try:
        snapshot_download(repo_id=repo_id, local_dir=target_dir,
                          local_dir_use_symlinks=False)
    except TypeError:
        # local_dir_use_symlinks was removed in newer huggingface_hub, which
        # already copies real files into local_dir by default.
        snapshot_download(repo_id=repo_id, local_dir=target_dir)
    logger.info(f"Qwen2-Audio ready at {target_dir}")
    return target_dir


def load_qwen_audio_tagger(device: str = "cuda", model_id: str = QWEN_AUDIO_MODEL_ID,
                           model_path: str = "") -> QwenAudioTagger:
    """Load the Qwen2-Audio processor and return a QwenAudioTagger.

    The model is downloaded into the ComfyUI models tree on first use (clean
    filenames, not HF cache blobs) and loaded from there. Only the light
    processor is held here; the 7B model loads lazily on the first tag_audio()
    (after the tagger node evicts other models) and is freed by offload().

    device: "cuda" or "cpu" (resolve "auto" before calling).
    model_path: target directory (see resolve_qwen_dir); empty = default
        <models_dir>/acestep/Qwen2-Audio-7B-Instruct.
    """
    from transformers import AutoProcessor  # noqa: PLC0415

    target_dir = resolve_qwen_dir(model_path)
    ensure_qwen_downloaded(target_dir, model_id)

    logger.info(f"Loading Qwen2-Audio processor from {target_dir}")
    processor = AutoProcessor.from_pretrained(target_dir)
    return QwenAudioTagger(processor=processor, device=device, source=target_dir, model_id=model_id)
