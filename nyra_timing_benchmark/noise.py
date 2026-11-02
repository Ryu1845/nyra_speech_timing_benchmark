"""Additive-noise conditions for robustness evaluation.

A :class:`NoiseCondition` turns a clean sample into a noisy one at a target SNR:

* ``gaussian``  -- white Gaussian noise,
* ``<category>`` -- a random excerpt from ``<noise_dir>/<category>/*`` (any
  format ``soundfile`` reads), resampled to the speech rate and looped if short.

SNR is computed over the whole (chunked) sample:
``10*log10(P_speech / P_noise)`` with ``P = mean(x**2)``. This needs only the
audio, so external submitters can reproduce the exact noisy inputs from their own
licensed copy with ``nyra-timing make-noisy``.

Noise is deterministic per sample: the RNG is seeded from
``(seed, noise, key)`` -- deliberately *not* the SNR -- so every SNR of a sweep
uses the same noise realisation and only its gain changes. If the mixture would
clip, the whole mixture is scaled down (the SNR is unchanged).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

NOISE_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".sph", ".aiff", ".aif")
CLEAN = "clean"
GAUSSIAN = "gaussian"
_PEAK = 0.99


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Read audio as mono float32 in [-1, 1]."""
    import soundfile as sf

    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        import librosa

        y, sr = librosa.load(path, sr=None, mono=True)
        return y.astype(np.float32), int(sr)
    return data.mean(axis=1).astype(np.float32), int(sr)


def write_wav(path: str, audio: np.ndarray, sr: int) -> str:
    import soundfile as sf

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
    return path


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


def sample_seed(seed: int, noise: str, key: str) -> int:
    digest = hashlib.sha256(f"{seed}|{noise}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x, dtype=np.float64))) if x.size else 0.0


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Return ``speech + g*noise`` with ``g`` chosen so the mix has ``snr_db``."""
    ps, pn = power(speech), power(noise)
    if ps == 0.0 or pn == 0.0:
        return speech.copy()
    gain = np.sqrt(ps / (pn * 10.0 ** (snr_db / 10.0)))
    mix = speech.astype(np.float64) + gain * noise[: len(speech)]
    peak = float(np.max(np.abs(mix)))
    if peak > _PEAK:
        mix *= _PEAK / peak
    return mix.astype(np.float32)


class NoiseBank:
    """Noise files grouped by category directory, loaded lazily and cached."""

    def __init__(self, noise_dir: str):
        self.noise_dir = noise_dir
        self._cache: Dict[Tuple[str, int], np.ndarray] = {}

    def files(self, category: str) -> List[str]:
        d = os.path.join(self.noise_dir, category)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Noise category directory not found: {d}")
        out = []
        for root, _, names in os.walk(d):
            out += [
                os.path.join(root, n)
                for n in names
                if n.lower().endswith(NOISE_EXTENSIONS)
            ]
        if not out:
            raise FileNotFoundError(f"No audio files in noise category: {d}")
        return sorted(out)

    def categories(self) -> List[str]:
        return sorted(
            n
            for n in os.listdir(self.noise_dir)
            if os.path.isdir(os.path.join(self.noise_dir, n))
        )

    def _load(self, path: str, sr: int) -> np.ndarray:
        k = (path, sr)
        if k not in self._cache:
            audio, file_sr = load_audio(path)
            self._cache[k] = _resample(audio, file_sr, sr)
        return self._cache[k]

    def segment(
        self, category: str, n: int, sr: int, rng: np.random.Generator
    ) -> np.ndarray:
        files = self.files(category)
        audio = self._load(files[int(rng.integers(len(files)))], sr)
        if audio.size == 0:
            return np.zeros(n, dtype=np.float32)
        if audio.size < n:
            audio = np.tile(audio, int(np.ceil(n / audio.size)) + 1)
        start = int(rng.integers(audio.size - n + 1))
        return audio[start : start + n]


@dataclass(frozen=True)
class NoiseCondition:
    noise: str = CLEAN
    snr_db: Optional[float] = None
    seed: int = 0

    def __post_init__(self):
        if self.noise != CLEAN and self.snr_db is None:
            raise ValueError(f"Noise '{self.noise}' requires an SNR.")

    @property
    def is_clean(self) -> bool:
        return self.noise == CLEAN

    @property
    def label(self) -> str:
        if self.is_clean:
            return CLEAN
        return f"{self.noise}_snr{self.snr_db:g}"

    def to_dict(self) -> Dict:
        if self.is_clean:
            return {"noise": CLEAN}
        return {"noise": self.noise, "snr_db": float(self.snr_db), "seed": self.seed}

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "NoiseCondition":
        d = d or {}
        return cls(
            noise=d.get("noise", CLEAN),
            snr_db=d.get("snr_db"),
            seed=int(d.get("seed", 0)),
        )

    def noise_for(
        self,
        key: str,
        n: int,
        sr: int,
        bank: Optional[NoiseBank] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(sample_seed(self.seed, self.noise, key))
        if self.noise == GAUSSIAN:
            return rng.standard_normal(n).astype(np.float32)
        if bank is None:
            raise ValueError(
                f"Noise category '{self.noise}' needs a noise directory "
                f"(--noise-dir DIR with DIR/{self.noise}/*.wav)."
            )
        return bank.segment(self.noise, n, sr, rng)

    def apply(
        self,
        speech: np.ndarray,
        sr: int,
        key: str,
        bank: Optional[NoiseBank] = None,
    ) -> np.ndarray:
        if self.is_clean:
            return speech
        noise = self.noise_for(key, len(speech), sr, bank)
        return mix_at_snr(speech, noise, float(self.snr_db))


def render_sample(
    src_path: str,
    dst_path: str,
    key: str,
    condition: NoiseCondition,
    bank: Optional[NoiseBank] = None,
    chunk: Optional[Tuple[float, float]] = None,
) -> str:
    """Write the (optionally chunked) sample with ``condition`` applied as WAV."""
    audio, sr = load_audio(src_path)
    if chunk is not None:
        a = max(0, int(round(chunk[0] * sr)))
        b = min(len(audio), int(round(chunk[1] * sr)))
        audio = audio[a:max(b, a + 1)]
    return write_wav(dst_path, condition.apply(audio, sr, key, bank), sr)


def parse_snrs(value: str) -> List[Optional[float]]:
    """Parse ``"clean,20,10,0"`` into ``[None, 20.0, 10.0, 0.0]``."""
    out: List[Optional[float]] = []
    for tok in value.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.append(None if tok in (CLEAN, "inf") else float(tok))
    return out
