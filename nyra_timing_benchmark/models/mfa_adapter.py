"""Montreal Forced Aligner (MFA) adapter.

MFA aligns a whole corpus directory in a single invocation, so this adapter
overrides :meth:`predict_many` to prepare one corpus (16 kHz mono wav + ``.lab``
transcript per utterance), run ``mfa align`` once, and parse every TextGrid. This
is dramatically faster than aligning one file per process. Requires an MFA install
with the ``english_mfa`` acoustic model and dictionary.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .base import PredictItem, TimingModel

TARGET_SR = 16_000


def _normalize(word: str) -> str:
    w = (word or "").strip().lower()
    w = w.rstrip(".,;:!?\"'")
    w = w.lstrip("\"'")
    return w


def _safe_uid(key: str) -> str:
    return key.replace("/", "_").replace(".", "_").replace(" ", "_")


def _parse_textgrid_words(tg_path: Path) -> List[Tuple[str, float, float]]:
    text = tg_path.read_text(encoding="utf-8", errors="replace")
    tiers = re.split(r"item\s*\[\d+\]", text)
    word_tier = None
    for tier in tiers:
        if re.search(r'name\s*=\s*"words?"', tier, re.IGNORECASE):
            word_tier = tier
            break
    if word_tier is None:
        return []
    intervals = re.findall(
        r"xmin\s*=\s*([\d.eE+\-]+)\s*xmax\s*=\s*([\d.eE+\-]+)\s*"
        r'(?:text|intervals\s*\[.*?\].*?text)\s*=\s*"([^"]*)"',
        word_tier,
        re.DOTALL,
    )
    out: List[Tuple[str, float, float]] = []
    for xmin_s, xmax_s, word in intervals:
        word = word.strip()
        if word and word not in ("", "sp", "sil", "<eps>"):
            out.append((word, float(xmin_s), float(xmax_s)))
    return out


class MfaModel(TimingModel):
    requires_transcript = True

    def __init__(
        self,
        mfa_bin: str = "/home/ubuntu/miniconda3/envs/mfa_env/bin/mfa",
        acoustic_model: str = "english_mfa",
        dictionary: str = "english_mfa",
        jobs: int = 8,
        beam: Optional[int] = None,
        retry_beam: Optional[int] = None,
    ):
        if not os.path.exists(mfa_bin):
            raise ImportError(
                f"MFA binary not found at {mfa_bin}. Install MFA and the "
                f"{acoustic_model} model/dictionary."
            )
        self.name = f"mfa-{acoustic_model}" + (f"-beam{beam}-{retry_beam}" if beam else "")
        self.beam = beam
        self.retry_beam = retry_beam
        self.mfa_bin = mfa_bin
        self.mfa_env_bin = str(Path(mfa_bin).parent)
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        self.jobs = jobs

    def _prep_wav(self, src: str, dst: Path) -> None:
        try:
            import torchaudio
        except ImportError:
            from ..noise import _resample, load_audio, write_wav

            audio, sr = load_audio(src)
            write_wav(str(dst), _resample(audio, sr, TARGET_SR), TARGET_SR)
            return

        wav, sr = torchaudio.load(src)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        torchaudio.save(str(dst), wav, TARGET_SR)

    def _words_for(self, item: PredictItem) -> List[str]:
        if item.words:
            return [_normalize(w) for w in item.words if _normalize(w)]
        if item.transcript:
            return [_normalize(w) for w in item.transcript.split() if _normalize(w)]
        return []

    def predict(
        self,
        audio_path: str,
        transcript: Optional[str] = None,
        words: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        item = PredictItem(
            key="utt",
            audio_path=audio_path,
            words=list(words or []),
            transcript=transcript or "",
        )
        return self.predict_many([item], progress_every=0).get("utt", [])

    def predict_many(
        self, items: Sequence[PredictItem], progress_every: int = 25
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        work_root = Path(tempfile.mkdtemp(prefix="nyra_mfa_"))
        corpus = work_root / "corpus"
        aligned = work_root / "aligned"
        tmp = work_root / "tmp"
        corpus.mkdir(parents=True, exist_ok=True)

        # uid <-> key mapping (uids must be unique and filesystem-safe).
        uid_to_key: Dict[str, str] = {}
        try:
            for item in items:
                words = self._words_for(item)
                if not words:
                    continue
                uid = _safe_uid(item.key)
                if uid in uid_to_key:
                    uid = f"{uid}_{len(uid_to_key)}"
                try:
                    self._prep_wav(item.audio_path, corpus / f"{uid}.wav")
                except Exception as e:
                    print(f"  audio prep failed for {item.key}: {e}")
                    continue
                (corpus / f"{uid}.lab").write_text(" ".join(words))
                uid_to_key[uid] = item.key

            if not uid_to_key:
                return {}

            print(f"  MFA: aligning {len(uid_to_key)} utterances in one pass...")
            env = dict(os.environ)
            env["PATH"] = self.mfa_env_bin + ":" + env.get("PATH", "")
            cmd = [
                self.mfa_bin,
                "align",
                "--clean",
                "--overwrite",
                "--single_speaker",
                "-t",
                str(tmp),
                "-j",
                str(self.jobs),
            ]
            if self.beam:
                config = work_root / "align_config.yaml"
                config.write_text(f"beam: {self.beam}\nretry_beam: {self.retry_beam}\n")
                cmd += ["--config_path", str(config)]
            cmd += [
                str(corpus),
                self.dictionary,
                self.acoustic_model,
                str(aligned),
            ]
            result = subprocess.run(cmd, text=True, env=env, capture_output=True)
            if result.returncode not in (0, 1):
                print(f"  MFA exit code {result.returncode}: {result.stderr[-500:]}")

            out: Dict[str, List[Tuple[str, float, float]]] = {}
            n_found = 0
            for uid, key in uid_to_key.items():
                tg = aligned / f"{uid}.TextGrid"
                if tg.exists():
                    out[key] = _parse_textgrid_words(tg)
                    n_found += 1
            print(f"  MFA: parsed {n_found}/{len(uid_to_key)} TextGrids")
            return out
        finally:
            shutil.rmtree(work_root, ignore_errors=True)
