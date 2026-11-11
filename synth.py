#!/usr/bin/env python3
"""synth.py — standalone round-8 synthesis for tts_test_bot.

The vendored piper-plus G2P (hinglish fork, byte-identical to what
trained the model) plus the round-8 ONNX. No Plan_B, no test_battery
dependency — this folder is the whole voice. The encode path mirrors
battery.SynthEngine (the proven production path): same phonemizer, same
PiperEncoder IDs, same feed names, same prosody features.

+4 dB presentation gain is baked in (2026-11-06 shootout: it clears all
the battery's loudness-gate failures with no clipping). Speaking speed
is a length_scale parameter per call — the UI's slider maps straight to
it (1.0 fast … 1.3 slow; the tested sweet spot is 1.10–1.25).
"""
import json
import logging
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # vendored g2p package lives here
MODEL_DIR = HERE / "model"
ONNX = MODEL_DIR / "navgurukul_voice.onnx"
CONFIG = MODEL_DIR / "navgurukul_voice.config.json"

GAIN_DB = 4.0                            # presentation gain, shootout-verified
GAIN = 10.0 ** (GAIN_DB / 20.0)

# ---- audio gate thresholds (r8 gate_wav.py, battery-tuned) ----
MIN_DUR = 0.5                            # s
MAX_SAT_FRAC = 0.01                      # samples at |x| >= 32760
RNG_RMS = (500, 15000)                   # 16-bit body rms
MIN_FLATNESS = 0.0025                    # tone killer
MAX_ZCR = 0.35                           # noise-like ceiling
WHITE_FLAT, WHITE_ZCR = 0.5, 0.3         # white-noise pattern


class SynthEngine:
    """r8 voice: vendored hinglish G2P -> ids -> ONNX -> pcm.

    Proven-path faithful copy of battery.SynthEngine (test_battery),
    minus the slow toggle — length_scale is passed per synthesize call
    so the UI slider can set it directly.
    """

    def __init__(self, onnx: Path = ONNX, config: Path = CONFIG):
        # the canonical encode path warns per dropped space/punct token;
        # it drops them by design — silence the noise, keep the path
        logging.getLogger("piper_plus_g2p").setLevel(logging.ERROR)
        import onnxruntime as ort
        from piper_plus_g2p.encode.encoder import PiperEncoder
        from piper_plus_g2p.multilingual import (MultilingualPhonemizer,
                                                 UnicodeLanguageDetector)

        self.cfg = json.loads(config.read_text(encoding="utf-8"))
        self.sr = self.cfg["audio"]["sample_rate"]
        self.session = ort.InferenceSession(str(onnx),
                                            providers=["CPUExecutionProvider"])
        self.wanted = {i.name for i in self.session.get_inputs()}
        langs = list(self.cfg.get("language_id_map", {}).keys())
        self.lang2id = self.cfg.get("language_id_map", {})
        self.phonemizer = MultilingualPhonemizer(langs,
                                                 default_latin_language="en")
        self.detector = UnicodeLanguageDetector(langs,
                                                default_latin_language="en")
        self.encoder = PiperEncoder(self.cfg["phoneme_id_map"])
        inf = self.cfg.get("inference", {})
        self.ns = inf.get("noise_scale", 0.667)
        self.nw = inf.get("noise_w", 0.8)

    def detect_lid(self, text: str) -> int:
        has_kana = self.detector.has_kana(text)
        counts: dict[str, int] = {}
        for ch in text:
            lang = self.detector.detect_char(ch, context_has_kana=has_kana)
            if lang is not None:
                counts[lang] = counts.get(lang, 0) + 1
        if not counts:
            return 0
        return self.lang2id.get(max(counts, key=lambda k: counts[k]), 0)

    def build_feeds(self, text: str, ls: float) -> dict:
        tokens, prosody_list = self.phonemizer.phonemize_with_prosody(text)
        ids, prosody = self.encoder.encode_with_prosody(tokens, prosody_list)
        n = len(ids)
        rows = [[p.a1, p.a2, p.a3] if p is not None else [0, 0, 0]
                for p in prosody]
        while len(rows) < n:
            rows.append([0, 0, 0])
        available = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([n], dtype=np.int64),
            "scales": np.array([self.ns, ls, self.nw], dtype=np.float32),
            "sid": np.array([0], dtype=np.int64),
            "lid": np.array([self.detect_lid(text)], dtype=np.int64),
            "prosody_features": np.array(rows[:n], dtype=np.int64).reshape(1, n, 3),
            "speaker_embedding": np.zeros((1, 256), dtype=np.float32),
            "speaker_embedding_mask": np.zeros((1, 1), dtype=np.int64),
        }
        return {k: v for k, v in available.items() if k in self.wanted}

    def synthesize(self, text: str, ls: float):
        """text + length_scale -> (pcm float32 in [-1, 1], synth_ms)."""
        feeds = self.build_feeds(text, ls)
        t0 = time.perf_counter()
        out = self.session.run(None, feeds)
        ms = (time.perf_counter() - t0) * 1000
        return np.asarray(out[0], dtype=np.float32).reshape(-1), ms


# ----------------------------------------------------------------------
# acoustic QC — sliding-window voice check + 6-point gate (battery-proven)
# ----------------------------------------------------------------------

def voice_check(pcm, sr):
    """Sliding 0.5s windows: rms<200 = energy collapse; flat>=0.004 and
    zcr>=0.10 = noise smear. Thresholds from the 2026-11-03 cluster
    findings (diag_cluster.py)."""
    events = []
    step = sr // 2
    for i in range(0, len(pcm) - step + 1, step):
        chunk = pcm[i:i + step]
        rms = float(np.sqrt((chunk ** 2).mean())) * 32767
        win = chunk * np.hanning(len(chunk))
        spec = np.abs(np.fft.rfft(win)) ** 2
        spec = spec[spec > 1e-12]
        flat = float(np.exp(np.log(spec).mean()) / spec.mean())
        zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)
        t = i / sr
        if rms < 200:
            events.append("collapse@%.1fs(rms=%d)" % (t, rms))
        elif flat >= 0.004 and zcr >= 0.10:
            events.append("smear@%.1fs" % t)
    return events


def stats(sr: int, x: np.ndarray):
    x = x - x.mean()
    n = len(x)
    zcr = float(np.mean(np.abs(np.diff(np.sign(x)))) / 2)
    win = int(sr * 0.025)
    hop = win // 2
    spec = np.zeros(win // 2 + 1)
    cnt = 0
    for i in range(0, max(1, n - win), hop):
        spec += np.abs(np.fft.rfft(x[i:i + win] * np.hanning(win))) ** 2
        cnt += 1
    spec /= max(1, cnt)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    m = (freqs >= 100) & (freqs <= 8000)
    flat = float(np.exp(np.mean(np.log(spec[m] + 1e-12))) /
                 (np.mean(spec[m]) + 1e-12))
    cent = float(np.sum(freqs[m] * spec[m]) / np.sum(spec[m]))
    return zcr, flat, cent


def gate_audio(path: Path) -> tuple[int, float, dict]:
    """6-point gate over the written wav -> (n_pass, dur_s, info)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        x = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
             .astype(np.float64) / 32768.0)
    dur = len(x) / sr
    sat = float(np.mean(np.abs(x) >= 32760 / 32768))
    zcr, flat, cent = stats(sr, x)
    body = x[int(0.1 * len(x)):int(0.9 * len(x))]
    rms = float(np.sqrt(np.mean(body ** 2))) * 32768
    checks = [
        dur >= MIN_DUR,
        sat <= MAX_SAT_FRAC,
        RNG_RMS[0] <= rms <= RNG_RMS[1],
        flat >= MIN_FLATNESS,
        zcr <= MAX_ZCR,
        not (flat > WHITE_FLAT and zcr > WHITE_ZCR),
    ]
    return (sum(checks), dur,
            {"sat_pct": sat * 100, "rms": rms, "flat": flat, "zcr": zcr})


def render_reply(eng, clean, out_dir: Path, n, ls: float):
    """Synthesize (with baked +4 dB) + write wav + voice-check + gate
    -> (wavpath, events, n_pass, dur). ls = length_scale (the speed
    slider's value: 1.0 fast … 1.3 slow)."""
    pcm, ms = eng.synthesize(clean, ls)
    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    wavp = wavs / ("turn_%04d.wav" % n)
    pcm = pcm * GAIN                     # presentation gain, baked in
    with wave.open(str(wavp), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(eng.sr)
        fh.writeframes((np.clip(pcm, -1, 1) * 32767)
                       .astype(np.int16).tobytes())
    events = voice_check(pcm, eng.sr)
    n_pass, dur, info = gate_audio(wavp)
    return wavp, events, n_pass, dur
