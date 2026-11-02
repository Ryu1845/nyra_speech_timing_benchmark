#!/usr/bin/env python3
"""Generate word timings with Inworld STT (inworld/inworld-stt-1).

POST https://api.inworld.ai/stt/v1/transcribe with ``includeWordTimestamps``
returns ``transcription.wordTimestamps: [{word, startTimeMs, endTimeMs}]``.

NOTE: Inworld documents word timestamps as "coming soon" / not yet populated, so
this provider may return an empty ``words`` list even on success. Reads
``INWORLD_API_KEY`` (a Basic auth token).
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import List, Tuple

import requests

from common import add_common_args, load_env_file, require_key, run_provider

PROVIDER_ID = "inworld_stt1"
API_URL = "https://api.inworld.ai/stt/v1/transcribe"


def build_transcriber(api_key: str, model_id: str, language: str):
    def transcribe(audio_path: Path) -> Tuple[List[Tuple[str, float, float]], dict]:
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        payload = {
            "audio_data": {"content": audio_b64},
            "transcribe_config": {
                "model_id": model_id,
                "language_code": language,
                "include_word_timestamps": True,
                "audio_encoding": "LINEAR16",
                "sample_rate_hertz": 16000,
            },
        }
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Basic {api_key}"},
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        transcription = data.get("transcription") or data
        triples: List[Tuple[str, float, float]] = []
        for w in transcription.get("wordTimestamps") or []:
            text = (w.get("word") or "").strip()
            start_ms = w.get("startTimeMs")
            end_ms = w.get("endTimeMs")
            if text and start_ms is not None and end_ms is not None:
                triples.append((text, float(start_ms) / 1000.0, float(end_ms) / 1000.0))
        return triples, data

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model-id", default="inworld/inworld-stt-1")
    parser.add_argument("--language", default="en-US")
    args = parser.parse_args()

    load_env_file()
    api_key = require_key("INWORLD_API_KEY", args.api_key)
    run_provider(
        PROVIDER_ID,
        build_transcriber(api_key, args.model_id, args.language),
        args,
    )


if __name__ == "__main__":
    main()
