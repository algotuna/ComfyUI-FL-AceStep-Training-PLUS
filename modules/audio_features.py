"""DSP music-feature estimation (BPM + key) via librosa.

Commercial-safe: librosa is ISC-licensed. These replace the 5Hz-LLM's
bpm/key guesses so the Qwen tagger pipeline can stand alone.

Quality, honestly:
  * BPM  - good. Tempo is a periodicity measurement DSP does well, better than
           an LLM over lossy codes. Main failure mode is OCTAVE ERRORS (half or
           double the true tempo).
  * KEY  - mediocre. librosa has no key detector; this implements the standard
           Krumhansl-Schmuckler chroma-profile correlation. It frequently
           confuses relative major/minor and perfect-fifth neighbours. Treat as
           a best-effort hint, not ground truth (the tagger exposes a toggle).

Heavy imports (librosa, numpy) are inside the functions so a missing optional
dependency never breaks node registration.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("FL_AceStep_Training")

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (major / minor).
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Analysis sample rate. 22050 is librosa's default and plenty for tempo/chroma.
_ANALYSIS_SR = 22050


def estimate_bpm(audio_path: str, max_duration: float = 120.0):
    """Return an integer BPM estimate, or None on failure.

    Analyses up to max_duration seconds (tempo is stationary enough that the
    first couple of minutes are representative and keep this fast).
    """
    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        y, sr = librosa.load(audio_path, sr=_ANALYSIS_SR, mono=True, duration=max_duration)
        if y.size == 0:
            return None
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # librosa may return a scalar or a 1-element array depending on version.
        tempo = float(np.atleast_1d(tempo)[0])
        if tempo <= 0:
            return None
        return int(round(tempo))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"BPM estimation failed for {audio_path}: {exc}")
        return None


def estimate_key(audio_path: str, max_duration: float = 120.0):
    """Return a key string like 'E major' / 'A minor', or '' on failure.

    Krumhansl-Schmuckler: average chroma over time, correlate against all 24
    rotated major/minor profiles, pick the best. Mediocre accuracy by design;
    see module docstring.
    """
    try:
        import librosa  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        y, sr = librosa.load(audio_path, sr=_ANALYSIS_SR, mono=True, duration=max_duration)
        if y.size == 0:
            return ""

        # CENS chroma is smoothed and robust to dynamics/timbre.
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)  # [12]
        if not np.any(chroma_mean):
            return ""

        major = np.array(_KS_MAJOR)
        minor = np.array(_KS_MINOR)

        best_score = -np.inf
        best_key = ""
        for i in range(12):
            maj_profile = np.roll(major, i)
            min_profile = np.roll(minor, i)
            maj_corr = float(np.corrcoef(chroma_mean, maj_profile)[0, 1])
            min_corr = float(np.corrcoef(chroma_mean, min_profile)[0, 1])
            if maj_corr > best_score:
                best_score, best_key = maj_corr, f"{_PITCH_CLASSES[i]} major"
            if min_corr > best_score:
                best_score, best_key = min_corr, f"{_PITCH_CLASSES[i]} minor"

        return best_key
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Key estimation failed for {audio_path}: {exc}")
        return ""
