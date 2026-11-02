# Provider timing generation

Call hosted speech-to-text APIs to produce **word-level timings**, then score them
with the benchmark. Mirrors the structure of the verbatim speech benchmark: one
small `generate_<provider>.py` per provider, all sharing `common.py`.

Each run writes two artifacts per dataset:

- `predictions/<provider>_<dataset>.json` — canonical submission format
  (schema v1, `metadata.deployment = "hosted_api"`), consumable by
  `nyra-timing score`.
- `generation/outputs/raw/<provider>_<dataset>.json` — the raw provider response
  per sample plus the parsed word timings (for auditing).

## Setup

```bash
pip install -r generation/requirements.txt
```

API keys are read from the research repo `.env` by default
(`/home/ubuntu/code/research/.env`); override the location with `NYRA_ENV_FILE`,
or pass `--api-key` explicitly. Required env vars per provider:

| script | provider | env var | word timings? |
| --- | --- | --- | --- |
| `generate_xai_grok_stt.py` | xAI Grok STT | `XAI_API_KEY` | yes |
| `generate_cartesia_ink_whisper.py` | Cartesia Ink-Whisper | `CARTESIA_API_KEY` | yes |
| `generate_deepgram_nova3.py` | Deepgram Nova-3 | `DEEPGRAM_API_KEY` | yes |
| `generate_inworld_stt1.py` | Inworld STT-1 | `INWORLD_API_KEY` | **no** (API returns empty `wordTimestamps` — "coming soon") |
| `generate_mai_transcribe_15.py` | Azure MAI-Transcribe-1.5 | `AZURE_AI_FOUNDRY_API_KEY` | **no** (MAI enhanced mode is segment-level only; use `--no-enhanced` for word timings from the standard fast-transcription model) |

The ElevenLabs **forced-alignment** API is a forced aligner, not an STT
provider, so it is a built-in model adapter instead of a script here:
`ELEVENLABS_API_KEY=... nyra-timing run --model elevenlabs_fa --dataset timit`
(also works with `robustness` for noise sweeps).

## Run

```bash
# Default: both TIMIT (test) and Buckeye, resumable.
python generation/generate_deepgram_nova3.py --dataset all

# A single dataset / a few samples for a smoke test.
python generation/generate_xai_grok_stt.py --dataset timit --max-samples 5 --out-dir /tmp/gentest
```

Common flags (see `common.py`): `--dataset {timit,buckeye,all}`, `--split`,
`--data-root`, `--max-samples N`, `--out-dir DIR`, `--no-resume`, `--api-key`.
Buckeye chunks are sliced from their parent recording automatically, so timings
are chunk-relative and line up with the ground truth.

## Score

```bash
nyra-timing score --dataset timit --predictions predictions/deepgram_nova3_timit.json
nyra-timing compare predictions/*_timit.json
```

## Add a provider

Copy any `generate_<provider>.py`, implement
`transcribe(audio_path) -> (List[(word, start_s, end_s)], raw_response)`, set a
`PROVIDER_ID`, and call `run_provider(PROVIDER_ID, transcribe, args)`. The shared
runner handles dataset iteration, Buckeye slicing, resume, and both output files.
