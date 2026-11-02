import json

from nyra_timing_benchmark.schema import (
    load_predictions,
    validate_submission,
    write_predictions,
)

DURS = {"a": 2.0, "b": 3.0}
META = {"system_type": "forced_aligner", "uses_reference_text": True,
        "description": "d", "model_url": "u", "license": "mit"}


def _sub(preds, **kw):
    d = {"schema_version": "1.0", "model": "m", "dataset": "timit",
         "metadata": META, "predictions": preds}
    d.update(kw)
    return d


def test_valid_submission():
    r = validate_submission(_sub({"a": [{"word": "x", "start": 0.1, "end": 0.5}],
                                   "b": []}), DURS, "timit")
    assert r.ok and not r.warnings


def test_errors_and_warnings():
    r = validate_submission(
        _sub({"a": [{"word": "x", "start": 0.5, "end": 0.1}], "zzz": []},
             metadata={"system_type": "asr"}),
        DURS,
        "timit",
    )
    joined = " ".join(r.errors)
    assert "system_type" in joined and "invalid times" in joined and "not in the dataset" in joined
    assert any("no prediction" in w for w in r.warnings)


def test_times_past_audio_end():
    r = validate_submission(_sub({"a": [{"word": "x", "start": 100, "end": 140}], "b": []}), DURS)
    assert not r.ok and any("systematic" in e for e in r.errors)
    many = {f"k{i}": [{"word": "x", "start": 0.1, "end": 0.2}] for i in range(9)}
    many["a"] = [{"word": "x", "start": 2.8, "end": 2.9}]
    r = validate_submission(_sub(many), {**{k: 1.0 for k in many}, "a": 2.0})
    assert r.ok and any("starting more than" in w for w in r.warnings)
    r = validate_submission(_sub({"a": [{"word": "x", "start": 1.5, "end": 4.0}], "b": []}), DURS)
    assert r.ok and any("ending more than" in w for w in r.warnings)


def test_noisy_condition_needs_snr():
    r = validate_submission(_sub({"a": [], "b": []}, condition={"noise": "gaussian"}), DURS)
    assert any("snr_db" in e for e in r.errors)


def test_roundtrip_gz(tmp_path):
    p = str(tmp_path / "p.json.gz")
    write_predictions(p, {"a": (["x"], [(0.1, 0.2)])}, "m", "timit",
                      {"noise": "gaussian", "snr_db": 10, "seed": 0}, META)
    f = load_predictions(p)
    assert f.predictions["a"] == (["x"], [(0.1, 0.2)])
    assert f.condition["snr_db"] == 10 and f.metadata["system_type"] == "forced_aligner"


def test_legacy_file_loads(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"model": "m", "dataset": "timit",
                             "predictions": {"a": [{"word": "x", "start": 0, "end": 1}]}}))
    f = load_predictions(str(p))
    assert f.condition == {"noise": "clean"} and f.metadata == {}
