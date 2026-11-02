#!/usr/bin/env python3
"""Generate word timings with Cartesia Ink-Whisper STT.

POST https://api.cartesia.ai/stt with ``model=ink-whisper`` and
``timestamp_granularities[]=word`` returns ``words: [{word, start, end}]``
(seconds). Reads ``CARTESIA_API_KEY``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import requests

from common import add_common_args, load_env_file, require_key, run_provider

PROVIDER_ID = "cartesia_ink_whisper"
API_URL = "https://api.cartesia.ai/stt"
CARTESIA_VERSION = "2025-04-16"


def build_transcriber(api_key: str, model: str, language: str):
    headers = {"X-API-Key": api_key, "Cartesia-Version": CARTESIA_VERSION}

    def transcribe(audio_path: Path) -> Tuple[List[Tuple[str, float, float]], dict]:
        with audio_path.open("rb") as f:
            response = requests.post(
                API_URL,
                headers=headers,
                data=[
                    ("model", model),
                    ("language", language),
                    ("timestamp_granularities[]", "word"),
                ],
                files={"file": (audio_path.name, f, "audio/wav")},
                timeout=180,
            )
        response.raise_for_status()
        data = response.json()
        triples: List[Tuple[str, float, float]] = []
        for w in data.get("words") or []:
            text = (w.get("word") or "").strip()
            start, end = w.get("start"), w.get("end")
            if text and start is not None and end is not None:
                triples.append((text, float(start), float(end)))
        return triples, data

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="ink-whisper")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    load_env_file()
    api_key = require_key("CARTESIA_API_KEY", args.api_key)
    run_provider(
        PROVIDER_ID, build_transcriber(api_key, args.model, args.language), args
    )


if __name__ == "__main__":
    main()
