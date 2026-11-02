#!/usr/bin/env python3
"""Generate word timings with Deepgram Nova-3.

POST https://api.deepgram.com/v1/listen returns
``results.channels[0].alternatives[0].words: [{word, start, end}]`` (seconds).
Reads ``DEEPGRAM_API_KEY``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import requests

from common import add_common_args, load_env_file, require_key, run_provider

PROVIDER_ID = "deepgram_nova3"
API_URL = "https://api.deepgram.com/v1/listen"


def build_transcriber(api_key: str, model: str, language: str):
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"}
    params = {
        "model": model,
        "language": language,
        "smart_format": "false",
        "punctuate": "false",
        "filler_words": "true",
    }

    def transcribe(audio_path: Path) -> Tuple[List[Tuple[str, float, float]], dict]:
        response = requests.post(
            API_URL,
            headers=headers,
            params=params,
            data=audio_path.read_bytes(),
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        triples: List[Tuple[str, float, float]] = []
        try:
            alt = data["results"]["channels"][0]["alternatives"][0]
        except (KeyError, IndexError):
            return triples, data
        for w in alt.get("words") or []:
            text = (w.get("word") or "").strip()
            start, end = w.get("start"), w.get("end")
            if text and start is not None and end is not None:
                triples.append((text, float(start), float(end)))
        return triples, data

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="nova-3")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    load_env_file()
    api_key = require_key("DEEPGRAM_API_KEY", args.api_key)
    run_provider(
        PROVIDER_ID, build_transcriber(api_key, args.model, args.language), args
    )


if __name__ == "__main__":
    main()
