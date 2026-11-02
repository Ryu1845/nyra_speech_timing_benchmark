#!/usr/bin/env python3
"""Generate word timings with Microsoft MAI-Transcribe-1.5 (Azure fast transcription).

POST {endpoint}/speechtotext/transcriptions:transcribe?api-version=2025-10-15 with
``wordLevelTimestampsEnabled`` returns ``phrases[].words[]`` with
``offsetMilliseconds`` / ``durationMilliseconds``.

IMPORTANT: per Azure docs, ``mai-transcribe-1.5`` (enhanced mode) does NOT emit
word-level timestamps — only segment/phrase-level. In that case ``words`` comes
back empty. Pass ``--no-enhanced`` to use the standard fast-transcription model,
which DOES return word timings, if you need word-level output from Azure.

Reads ``AZURE_AI_FOUNDRY_API_KEY`` (or ``MAI_TRANSCRIBE_15_KEY``) and
``MAI_TRANSCRIBE_15_ENDPOINT``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import requests

from common import add_common_args, load_env_file, require_key, run_provider

PROVIDER_ID = "mai_transcribe_15"
API_VERSION = "2025-10-15"
DEFAULT_ENDPOINT = "https://eastus.api.cognitive.microsoft.com"
LOCALES = {"en": "en-US", "de": "de-DE"}


def build_definition(locale: str, enhanced: bool) -> dict:
    definition = {
        "locales": [locale],
        "wordLevelTimestampsEnabled": True,
    }
    if enhanced:
        definition["enhancedMode"] = {
            "enabled": True,
            "model": "mai-transcribe-1.5",
            "transcribeStyle": "verbatim",
        }
    return definition


def _parse_words(data: dict) -> List[Tuple[str, float, float]]:
    triples: List[Tuple[str, float, float]] = []
    for phrase in data.get("phrases") or []:
        for w in phrase.get("words") or []:
            text = (w.get("text") or "").strip()
            off = w.get("offsetMilliseconds")
            dur = w.get("durationMilliseconds")
            if text and off is not None and dur is not None:
                start = float(off) / 1000.0
                triples.append((text, start, start + float(dur) / 1000.0))
    return triples


def build_transcriber(api_key: str, endpoint: str, locale: str, enhanced: bool):
    url = (
        f"{endpoint.rstrip('/')}/speechtotext/transcriptions:transcribe"
        f"?api-version={API_VERSION}"
    )
    definition = build_definition(locale, enhanced)

    def transcribe(audio_path: Path) -> Tuple[List[Tuple[str, float, float]], dict]:
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        for attempt in range(3):
            with audio_path.open("rb") as f:
                response = requests.post(
                    url,
                    headers=headers,
                    files={
                        "audio": (audio_path.name, f, "audio/wav"),
                        "definition": (None, json.dumps(definition), "application/json"),
                    },
                    timeout=300,
                )
            if response.ok:
                data = response.json()
                return _parse_words(data), data
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            response.raise_for_status()
        return [], {"error": "exhausted retries"}

    return transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--no-enhanced",
        action="store_true",
        help="Use the standard fast-transcription model (word timings) instead of MAI.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MAI_TRANSCRIBE_15_ENDPOINT", DEFAULT_ENDPOINT),
    )
    args = parser.parse_args()

    load_env_file()
    endpoint = os.environ.get("MAI_TRANSCRIBE_15_ENDPOINT", args.endpoint)
    api_key = args.api_key or os.environ.get("MAI_TRANSCRIBE_15_KEY", "")
    if not api_key:
        api_key = require_key("AZURE_AI_FOUNDRY_API_KEY")
    locale = LOCALES.get(args.language, args.language)
    run_provider(
        PROVIDER_ID,
        build_transcriber(api_key, endpoint, locale, enhanced=not args.no_enhanced),
        args,
    )


if __name__ == "__main__":
    main()
