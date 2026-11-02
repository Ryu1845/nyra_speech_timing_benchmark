"""Minimal audio slicing for chunked datasets.

Uses only the standard library (``wave``) so the core package stays dependency-
light. Buckeye parent recordings are PCM WAV, which ``wave`` handles directly.
"""

from __future__ import annotations

import tempfile
import wave
from typing import Optional


def slice_wav(
    src_path: str,
    start_s: float,
    end_s: float,
    out_path: Optional[str] = None,
) -> str:
    """Write the ``[start_s, end_s]`` segment of a PCM WAV to a new file.

    Returns the output path (a temp file if ``out_path`` is not given). The caller
    is responsible for deleting temp files.
    """
    with wave.open(src_path, "rb") as w:
        framerate = w.getframerate()
        n_frames = w.getnframes()
        start_frame = max(0, int(round(start_s * framerate)))
        end_frame = min(n_frames, int(round(end_s * framerate)))
        if end_frame <= start_frame:
            end_frame = min(n_frames, start_frame + 1)
        w.setpos(start_frame)
        frames = w.readframes(end_frame - start_frame)
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()

    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = tmp.name
        tmp.close()

    with wave.open(out_path, "wb") as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.writeframes(frames)
    return out_path
