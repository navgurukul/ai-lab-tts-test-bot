# tts_test_bot — Arzoo voice test chat (internal team test)

A small chat window: you type a line, Arzoo replies in text **and** the
round-8 voice. Purpose: team testing of the voice, the speaking speed,
and the LLM behind the answers.

- **Which LLM answers** is set in `.env` — one key uncommented, the
  other commented. The app uses the uncommented one. Title bar shows
  which is active. Both keys are already in `.env` (DeepSeek active).
- **Speaking speed** is the slider (1.0 = fast … 1.3 = slow).
  The tested sweet spot is **1.10–1.25**; the default is 1.20.
- **Loudness** is already taken care of: +4 dB gain is baked in.

The gray lines under each reply show the voice check (collapse/smear/home/phantom/Desktop/traning-tts/version_9/LLM_TTS_RESPONSE/r9_1_texts.xlsx
events), the audio gate score (x/6), and the speed used. Every turn is
logged to `out/conversation.jsonl` — please paste those lines into your
feedback notes if anything sounds off.

## Setup (once)

1. Install Python 3.10+ from python.org/home/phantom/Desktop/traning-tts/version_9/LLM_TTS_RESPONSE/r9_1_texts.xlsx (tkinter is included; do not
   use the Microsoft Store build).
2. Open a terminal in this folder and run:

   ```
   pip install onnxruntime numpy g2p_en
   ```

That's all — the voice model and the G2P frontend are inside this
folder (`model/`, `piper_plus_g2p/`).

## Run

Double-click `run.bat`, or in a terminal:

```
python tts_test_bot.py
```

The greeting is spoken on startup (a few seconds to load the voice).

## Switching the LLM

Edit `.env` with Notepad. Example — to switch to Gemini:

```
# DEEPSEEK_API_KEY=sk-...        <- comment out DeepSeek
GEMINI_API_KEY=AIza...           <- uncomment Gemini
```

The app must be restarted after editing `.env`.

## What to test

1. General chat — does the voice sound natural and clear?
2. The **speed slider** — try 1.10, 1.15, 1.20, 1.25 on the same line.
   Which sounds best to you?
3. Words from the voice's problem list (cluster words like *padhna,
   seekhna, chahte, hoon*) — are they clear at your chosen speed?
4. Hinglish vs English replies (the Auto/Hinglish/English buttons).

## Folder contents

| Item | What |
|---|---|
| `tts_test_bot.py` | the chat window |
| `bot_core.py` | LLM brain (DeepSeek + Gemini), voice-safety rules, reply wrapper |
| `synth.py` | round-8 voice synthesis (+4 dB, voice check, audio gate) |
| `model/` | NavGurukul round-8 voice (ONNX + config) |
| `piper_plus_g2p/` | the G2P frontend that trained the voice (vendored, MIT — LICENSE.md included) |
| `voice_rules.json` | voice-safety word rules (prompt + wrapper share this) |
| `.env` | the two API keys (comment-control) |
| `out/` | conversation log + wavs per turn |
