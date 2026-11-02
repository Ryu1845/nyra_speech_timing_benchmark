#!/usr/bin/env python3
"""Generate word timings with the xAI Grok speech-to-text API.

POST https://api.x.ai/v1/stt returns ``words: [{text, start, end}]`` (seconds).
Reads ``XAI_API_KEY``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import requests

from common import (
    add_common_args,
    load_env_file,
    require_key,
    run_provider,
)

PROVIDER_ID = "xai_grok_stt"
API_URL = "https://api.x.ai/v1/stt"


def build_transcriber(api_key: str, language: str):
    def transcribe(audio_path: Path) -> Tuple[List[Tuple[str, float, float]], dict]:
        with audio_path.open("rb") as f:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                data={"filler_words": "true", "format": "false", "language": language},
                files={"file": (audio_path.name, f, "audio/wav")},
                timeout=180,
            )
        response.raise_for_status()
        data = response.json()
        triples: List[Tuple[str, float, float]] = []
        for w in data.get("words") or []:
            text = (w.get("text") or "").strip()
            start, end = w.get("start"), w.get("end")
            if text and start is not None and end is not None:
                triples.append((text, float(start), float(end)))
        return triples, data

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    load_env_file()
    api_key = require_key("XAI_API_KEY", args.api_key)
    run_provider(PROVIDER_ID, build_transcriber(api_key, args.language), args)


if __name__ == "__main__":
    main()
