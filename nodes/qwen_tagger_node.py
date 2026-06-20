"""FL_AceStep_QwenTagger.

Open-vocabulary music tagging for an ACE-Step dataset using Qwen2-Audio.

Flow:
  1. Discovery  - Qwen tags every track (genre / instruments / moods / desc),
                  with no pre-curated label list. Results are cached to an
                  editable review JSON, and the tagger is offloaded to free VRAM.
  2. Aggregate  - the union of all discovered tags becomes a corpus vocabulary
                  (written into the review JSON for inspection / editing).
  3. Apply      - per-sample tags are assembled into a consistent caption and
                  written to each sample (genre / instruments / moods / caption).

Review mode (user's choice, via auto_apply):
  * auto_apply = True   -> discover (if needed) and apply in one run.
  * auto_apply = False  -> discover, write the review JSON, and STOP so you can
                           edit it. Re-run with auto_apply = True to apply the
                           edited tags (discovery is read from the cache, so
                           Qwen is not re-run).

Set force_rediscover = True to ignore an existing cache (e.g. after changing
the dataset) and re-run Qwen.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("FL_AceStep_Training")

try:
    from comfy.utils import ProgressBar
except Exception:  # noqa: BLE001
    ProgressBar = None

try:
    import folder_paths
except Exception:  # noqa: BLE001
    folder_paths = None


class FL_AceStep_QwenTagger:
    DESCRIPTION = (
        "Tag a dataset's musical style with Qwen2-Audio (open-vocabulary, "
        "Apache-2.0). Discovers genre/instrument/mood tags from the audio "
        "itself - no pre-built label list - then writes consistent captions. "
        "auto_apply OFF stops after discovery so you can edit the review JSON; "
        "auto_apply ON applies immediately."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("ACESTEP_DATASET",),
                "tagger": ("QWEN_AUDIO_TAGGER",),
                "auto_apply": ("BOOLEAN", {
                    "default": False,
                    "label_on": "auto (apply now)",
                    "label_off": "manual (review first)",
                }),
            },
            "optional": {
                "review_file": ("STRING", {
                    "default": "./output/acestep/tagging_review.json",
                    "multiline": False,
                    "placeholder": "Where discovered tags are cached for review",
                }),
                "force_rediscover": ("BOOLEAN", {"default": False}),
                "only_unlabeled": ("BOOLEAN", {"default": False}),
                "max_genres": ("INT", {"default": 2, "min": 1, "max": 5}),
                "max_instruments": ("INT", {"default": 5, "min": 1, "max": 12}),
                "max_moods": ("INT", {"default": 2, "min": 0, "max": 5}),
                "include_description": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("ACESTEP_DATASET", "STRING", "STRING")
    RETURN_NAMES = ("dataset", "review_path", "status")
    FUNCTION = "tag"
    CATEGORY = "FL AceStep/Dataset"

    # ------------------------------------------------------------------ #

    def tag(
        self,
        dataset,
        tagger,
        auto_apply,
        review_file="./output/acestep/tagging_review.json",
        force_rediscover=False,
        only_unlabeled=False,
        max_genres=2,
        max_instruments=5,
        max_moods=2,
        include_description=False,
    ):
        samples = dataset.samples
        if not samples:
            return (dataset, "", "No samples to tag")

        review_path = self._resolve_review_path(review_file)

        # ---- Step 1: discovery (cache-aware) ----------------------------
        cache = None
        if not force_rediscover and os.path.exists(review_path):
            cache = self._load_cache(review_path)
            if cache:
                logger.info(f"Loaded cached tags from {review_path}")

        if cache is None:
            cache = self._discover(samples, tagger, only_unlabeled)
            # Free VRAM as soon as the model's work is done.
            if hasattr(tagger, "offload"):
                try:
                    tagger.offload()
                    logger.info("Offloaded Qwen tagger to CPU after discovery")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Could not offload Qwen tagger: {exc}")
            self._write_cache(review_path, cache)

        per_sample = cache.get("samples", {})

        # ---- Step 2: manual review gate ---------------------------------
        if not auto_apply:
            status = (
                f"Discovery complete for {len(per_sample)} samples. Review/edit "
                f"{review_path}, then re-run with auto_apply = ON to apply. "
                f"(Nothing applied yet.)"
            )
            logger.info(status)
            return (dataset, review_path, status)

        # ---- Step 3: apply ----------------------------------------------
        tag_position = getattr(getattr(dataset, "metadata", None), "tag_position", "prepend")
        applied = 0
        for sample in samples:
            tags = per_sample.get(sample.id)
            if not tags:
                continue
            self._apply_to_sample(
                sample, tags, max_genres, max_instruments, max_moods,
                include_description, tag_position,
            )
            applied += 1

        status = f"Applied tags to {applied}/{len(samples)} samples"
        logger.info(status)
        return (dataset, review_path, status)

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def _discover(self, samples, tagger, only_unlabeled) -> dict:
        # Evict ComfyUI's resident models (DiT, VAE, text encoder) before Qwen's
        # ~12GB load, or they collide and OOM on a 16GB card. They reload on
        # demand when Preprocess runs after this node.
        try:
            import comfy.model_management as mm  # noqa: PLC0415
            mm.unload_all_models()
            mm.soft_empty_cache()
            logger.info("Unloaded ComfyUI models to make room for Qwen2-Audio")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not unload ComfyUI models before Qwen: {exc}")

        targets = [
            s for s in samples
            if not (only_unlabeled and (s.labeled or s.caption))
        ]
        logger.info(f"Qwen discovery over {len(targets)} samples...")
        pbar = ProgressBar(len(targets)) if ProgressBar else None

        per_sample = {}
        for i, sample in enumerate(targets):
            try:
                tags = tagger.tag_audio(sample.audio_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Tagging failed for {sample.filename}: {exc}")
                tags = {"genre": [], "instruments": [], "moods": [], "description": ""}
            per_sample[sample.id] = tags
            logger.info(
                f"[{i + 1}/{len(targets)}] {sample.filename}: "
                f"genre={tags.get('genre')} instruments={tags.get('instruments')}"
            )
            if pbar:
                pbar.update(1)

        return {
            "samples": per_sample,
            "corpus_vocabulary": self._aggregate(per_sample),
        }

    @staticmethod
    def _aggregate(per_sample: dict) -> dict:
        """Union of all discovered tags -> a canonical corpus vocabulary.

        Written into the review file so you can see (and prune) the full tag
        set your dataset produced, instead of pre-imagining one.
        """
        vocab = {"genre": set(), "instruments": set(), "moods": set()}
        for tags in per_sample.values():
            for key in vocab:
                for t in tags.get(key, []) or []:
                    if isinstance(t, str) and t.strip():
                        vocab[key].add(t.strip().lower())
        return {k: sorted(v) for k, v in vocab.items()}

    # ------------------------------------------------------------------ #
    # Apply
    # ------------------------------------------------------------------ #

    def _apply_to_sample(self, sample, tags, max_genres, max_instruments,
                         max_moods, include_description, tag_position):
        genres = (tags.get("genre") or [])[:max_genres]
        instruments = (tags.get("instruments") or [])[:max_instruments]
        moods = (tags.get("moods") or [])[:max_moods] if max_moods else []

        sample.genre = ", ".join(genres)
        sample.instruments = ", ".join(instruments)
        sample.moods = ", ".join(moods)
        sample.caption = self._build_caption(
            sample, genres, instruments, moods,
            tags.get("description", "") if include_description else "",
            tag_position,
        )
        sample.labeled = True

    @staticmethod
    def _build_caption(sample, genres, instruments, moods, description, tag_position) -> str:
        """Assemble a consistent, prompt-aligned caption from the tags.

        Order: [trigger], genres, instruments, moods, [bpm], [key], [desc].
        The custom_tag is placed per the dataset's tag_position convention so
        the trigger word matches how the rest of the pipeline uses it.
        """
        parts = []
        for g in genres:
            parts.append(g)
        for ins in instruments:
            parts.append(ins)
        for m in moods:
            parts.append(m)
        if sample.bpm:
            parts.append(f"{sample.bpm} BPM")
        if sample.keyscale:
            parts.append(str(sample.keyscale))

        caption = ", ".join(p for p in parts if p)
        if description:
            caption = f"{caption}. {description}" if caption else description

        tag = (sample.custom_tag or "").strip()
        if tag:
            if tag_position == "replace":
                caption = tag
            elif tag_position == "append":
                caption = f"{caption}, {tag}" if caption else tag
            else:  # prepend (default)
                caption = f"{tag}, {caption}" if caption else tag
        return caption

    # ------------------------------------------------------------------ #
    # Review-file I/O
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_review_path(review_file: str) -> str:
        review_file = (review_file or "").strip() or "./output/acestep/tagging_review.json"
        if os.path.isabs(review_file):
            return review_file
        # Anchor relative paths at the ComfyUI output dir when available so the
        # file lands somewhere predictable regardless of cwd.
        if folder_paths is not None:
            try:
                base = folder_paths.get_output_directory()
                rel = review_file.lstrip("./").replace("output/", "", 1)
                return os.path.join(base, rel)
            except Exception:  # noqa: BLE001
                pass
        return os.path.abspath(review_file)

    @staticmethod
    def _load_cache(path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "samples" in data:
                return data
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not read review cache {path}: {exc}")
        return None

    @staticmethod
    def _write_cache(path: str, cache: dict):
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote tagging review file: {path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not write review cache {path}: {exc}")
