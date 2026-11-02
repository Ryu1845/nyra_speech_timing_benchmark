import numpy as np
import pytest

from nyra_timing_benchmark.noise import (
    NoiseBank,
    NoiseCondition,
    load_audio,
    mix_at_snr,
    parse_snrs,
    power,
    render_sample,
    write_wav,
)

SR = 16000


def _tone(seconds=1.0, amp=0.1):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.mark.parametrize("snr", [30, 10, 0, -5])
def test_gaussian_mix_hits_target_snr(snr):
    speech = _tone()
    cond = NoiseCondition("gaussian", snr)
    mix = cond.apply(speech, SR, "k")
    noise = mix - speech
    assert 10 * np.log10(power(speech) / power(noise)) == pytest.approx(snr, abs=0.05)


def test_noise_is_deterministic_and_shared_across_snrs():
    speech = _tone()
    a = NoiseCondition("gaussian", 10).apply(speech, SR, "k") - speech
    b = NoiseCondition("gaussian", 10).apply(speech, SR, "k") - speech
    c = NoiseCondition("gaussian", 0).apply(speech, SR, "k") - speech
    other = NoiseCondition("gaussian", 10).apply(speech, SR, "k2") - speech
    np.testing.assert_array_equal(a, b)
    assert np.corrcoef(a, c)[0, 1] > 0.999
    assert abs(np.corrcoef(a, other)[0, 1]) < 0.1


def test_clipping_rescales_without_changing_snr():
    speech = _tone(amp=0.9)
    mix = mix_at_snr(speech, np.random.default_rng(0).standard_normal(len(speech)), 0)
    assert np.max(np.abs(mix)) <= 0.99 + 1e-6


def test_file_noise_category(tmp_path):
    (tmp_path / "babble").mkdir()
    write_wav(str(tmp_path / "babble" / "n.wav"), np.random.default_rng(1).standard_normal(8000).astype(np.float32) * 0.1, 8000)
    bank = NoiseBank(str(tmp_path))
    assert bank.categories() == ["babble"]
    speech = _tone(2.0)
    mix = NoiseCondition("babble", 5).apply(speech, SR, "k", bank)
    assert len(mix) == len(speech)
    noise = mix - speech
    assert 10 * np.log10(power(speech) / power(noise)) == pytest.approx(5, abs=0.2)


def test_render_sample_chunk(tmp_path):
    src = str(tmp_path / "s.wav")
    write_wav(src, _tone(3.0), SR)
    out = render_sample(src, str(tmp_path / "o.wav"), "k", NoiseCondition("gaussian", 20), chunk=(1.0, 2.5))
    audio, sr = load_audio(out)
    assert sr == SR and len(audio) == int(1.5 * SR)


def test_parse_snrs_and_labels():
    assert parse_snrs("clean,20,5,-5") == [None, 20.0, 5.0, -5.0]
    assert NoiseCondition().label == "clean"
    assert NoiseCondition("gaussian", 10).label == "gaussian_snr10"
    assert NoiseCondition.from_dict(NoiseCondition("x", 2.5, 3).to_dict()) == NoiseCondition("x", 2.5, 3)
    with pytest.raises(ValueError):
        NoiseCondition("gaussian")
