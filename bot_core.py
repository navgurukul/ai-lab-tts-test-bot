#!/usr/bin/env python3
"""bot_core.py — the chatbot brain for tts_test_bot (internal team test).

Two providers only, chosen by tts_test_bot/.env comment-control: the
uncommented key wins. DeepSeek and Gemini share one JSON contract
({"reply", "notes"}) and the same guardrail system prompt, generated
from voice_rules.json so prompt and wrapper can never drift apart.

The system prompt carries the TTS-ready rules adopted from the
2026-11-10 LLM-inference research assessment (Roman-only output,
aa/ee/oo long-vowel spelling, numbers as spoken words, short
conversational sentences, no markdown/emojis) — the wrapper below is
the deterministic backstop.
"""
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULES_PATH = HERE / "voice_rules.json"

# DeepSeek
API_URL = "https://api.deepseek.com/chat/completions"
DS_MODEL = "deepseek-chat"

# Gemini — key goes in the x-goog-api-key header, never in the URL
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "%s:generateContent")
GEMINI_MODEL = "gemini-3.8-flash"   # current Flash; same model the voice
                                    # battery trusts for STT

# ----------------------------------------------------------------------
# .env comment-control — the uncommented key picks the provider
# ----------------------------------------------------------------------

def _env_entries(path):
    """{VAR: value} for uncommented KEY= lines in an .env file. Commented
    lines are ignored — that's the on/off switch for the provider."""
    entries = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            v = v.split(" #")[0].rstrip()   # tolerate inline comments
            if v:
                entries[k.strip()] = v.strip()
    return entries


class TutorLLM:
    """The chatbot's brain, chosen by the uncommented key in .env.
    Both uncommented -> DeepSeek wins. Neither -> a clear error."""

    def __init__(self):
        app = _env_entries(HERE / ".env")
        ds_key = app.get("DEEPSEEK_API_KEY")
        gem_key = app.get("GEMINI_API_KEY")
        if ds_key:
            self.provider = "deepseek"
            self.model = DS_MODEL
            self.key = ds_key
        elif gem_key:
            self.provider = "gemini"
            self.model = GEMINI_MODEL
            self.key = gem_key
        else:
            raise SystemExit(
                "No API key found. In tts_test_bot/.env, uncomment either "
                "DEEPSEEK_API_KEY or GEMINI_API_KEY (comment the other) "
                "and re-run.")

    def chat(self, system, msgs):
        """One call -> (reply, notes). msgs = alternating user/assistant
        turns ending with the user's line."""
        if self.provider == "gemini":
            return _gemini(self.key, system, msgs, self.model)
        return _deepseek(self.key, system, msgs, self.model)


# ----------------------------------------------------------------------
# voice rules — single source of truth (prompt + wrapper both read it)
# ----------------------------------------------------------------------

def load_rules():
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def build_guardrails(rules):
    """The system prompt, with the banned-word list generated from
    voice_rules.json so prompt and wrapper can never drift apart."""
    ban_lines = []
    for word in rules["words_hng"]:
        subs = [rules["words_hng"][word]]
        en_sub = rules["words_en"].get(word, "")
        if en_sub and en_sub not in subs:
            subs.append(en_sub)
        ban_lines.append("   %s -> %s" % (word, " / ".join(subs)))
    for word, hint in rules["phrase_hints"].items():
        ban_lines.append("   %s -> rephrase with %s" % (word, hint))
    ban = "\n".join(ban_lines)
    return """\
You are Arzoo, NavGurukul's friendly AI learning assistant. You are \
interviewing a student about their learning goals, asking warm follow-up \
questions like a good tutor. You reply in 2-4 short sentences (each under \
12 words), in the same language the student used.

Your replies are spoken aloud by a text-to-speech voice. Follow these \
voice-safety rules exactly:

1. Latin script ONLY (Roman Hinglish or English). Never output Devanagari \
or any non-Latin script, and never use emojis, quotation marks, markdown \
or bullet lists. Even when the student writes pure Hindi in Devanagari, \
your reply must be Roman transliteration (Latin script), e.g. write \
"Aap kya seekhna chahte hain" - never "आप क्या सीखना चाहते हैं".
2. NEVER use these words - the voice cannot say them clearly. Use the \
substitute that matches your reply language instead:
%s
3. NEVER use the phrase "lag rahi" - say "chal rahi" or "jaari" instead.
4. If you introduce yourself, use exactly this phrase: "Main Arzoo hoon, \
aapki padhaai ki saathi." - never say the name alone; otherwise refer to \
yourself as "main".
5. Prefer short everyday words; when two words mean the same, pick the \
one with fewer consonant clusters.
6. Prefer colloquial everyday Hindi over formal or Sanskritized words \
(e.g. "kitaab" not "pustak", "paani" not "jal", "kaam" not "udyog"). \
Colloquial words are also easier for the voice to say.
7. Write numbers as spoken words in your reply's language (e.g. \
"pachchees" or "twenty five", never "25").
8. Spell Hindi long vowels with doubled letters - aa, ee, oo (e.g. \
"paani", "seekhna", "hoon", never "pani", "sikhna", "hon"). The voice \
reads single and double vowels differently.

Output ONLY this JSON object, no other text:
{"reply": "<your reply>", "notes": "<the banned words you avoided and what \
you used instead, comma-separated; write 'none' if none>"}
""" % ban


# ----------------------------------------------------------------------
# deterministic post-LLM wrapper
# ----------------------------------------------------------------------

HINDI_MARKERS = {"hai", "hain", "hoon", "ho", "karo", "karna", "karne",
                 "karti", "karta", "main", "aap", "tum", "mujhe", "mera",
                 "meri", "mere", "ka", "ki", "ke", "se", "mein", "nahi",
                 "nahin", "kya", "aur", "yeh", "woh", "bhi", "toh", "kab",
                 "kaise", "kyun", "seekhna", "padhaai", "kitaab", "sikha",
                 "aapki", "aapka", "hum", "ham", "sakte", "sakti", "sakta",
                 "rahi", "raha", "gaya", "gayi", "hoga", "hogi", "kaam",
                 "accha", "bahut", "ek", "theek", "sahi"}


def detect_lang(text):
    words = set(re.findall(r"[a-zA-Z']+", text.lower()))
    return "hng" if len(words & HINDI_MARKERS) >= 2 else "en"


def num2words(n):
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty",
            "seventy", "eighty", "ninety"]

    def under100(x):
        if x < 20:
            return ones[x]
        t, u = divmod(x, 10)
        return tens[t] + (" " + ones[u] if u else "")

    if n < 100:
        return under100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        return ones[h] + " hundred" + (" and " + under100(r) if r else "")
    if n < 1000000:
        t, r = divmod(n, 1000)
        return num2words(t) + " thousand" + (" " + num2words(r) if r else "")
    return str(n)


def _case_rep(m, rep):
    got = m.group(0)
    return rep[0].upper() + rep[1:] if got[0].isupper() else rep


def wrap_reply(text, rules):
    """Deterministic pass over the LLM reply -> (clean, changes)."""
    changes = []
    lang = detect_lang(text)
    t = text
    # 1. phrase patterns (grammar-safe)
    for pat, rep in rules.get("patterns_" + lang, {}).items():
        t, k = re.subn(pat, rep, t, flags=re.IGNORECASE)
        if k:
            changes.append("%s -> %s" % (pat, rep))
    # 2. word-level substitutions (case-preserving, word boundaries)
    for word, rep in rules.get("words_" + lang, {}).items():
        t, k = re.subn(r"\b" + re.escape(word) + r"\b",
                       lambda m, rep=rep: _case_rep(m, rep), t,
                       flags=re.IGNORECASE)
        if k:
            changes.append("%s -> %s" % (word, rep))
    # 3. sibilant spelling fixes
    for word, rep in rules["sibilants"].items():
        t, k = re.subn(r"\b" + re.escape(word) + r"\b",
                       lambda m, rep=rep: _case_rep(m, rep), t,
                       flags=re.IGNORECASE)
        if k:
            changes.append("%s -> %s (sibilant)" % (word, rep))
    # 4. digits -> words (English replies only)
    if lang == "en":
        t, k = re.subn(r"\b\d{1,6}\b", lambda m: num2words(int(m.group(0))),
                       t)
        if k:
            changes.append("digits -> words x%d" % k)
    # 5. cleanup: emojis, markdown, stray quotes, whitespace
    t = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", t)
    t = t.replace("*", "").replace("#", "").replace("`", "").replace('"', "")
    t = re.sub(r"\s+", " ", t).strip(" -–—")
    return t, changes


# ----------------------------------------------------------------------
# transports (curl subprocess — Python urllib hangs on this machine;
# keys are never printed, and stderr is never shown to the user)
# ----------------------------------------------------------------------

def _curl_post(url, headers, body):
    """POST JSON via curl. 3 attempts, hardened against the flaky
    network. Returns the parsed JSON dict, or {} on failure. Never
    print stderr — curl failure messages can echo the URL containing
    the key."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as fh:
        json.dump(body, fh)
        tmp = fh.name
    try:
        proc = None
        for attempt in range(3):
            try:
                hdrs = [h for pair in headers.items()
                        for h in ("-H", "%s: %s" % pair)]
                proc = subprocess.run(
                    ["curl", "-s", "-m", "120"] + hdrs +
                    ["--data-binary", "@" + tmp, url],
                    capture_output=True, text=True, timeout=130)
                if proc.returncode == 0:
                    break
            except subprocess.TimeoutExpired:
                proc = None
            time.sleep(3 * (attempt + 1))
        if proc is not None and proc.returncode == 0:
            try:
                return json.loads(proc.stdout or "{}") or {}
            except ValueError:
                return {}
        return {}
    finally:
        Path(tmp).unlink(missing_ok=True)


def _parse_json_text(text):
    """Leniently pull {"reply", "notes"} out of an LLM answer."""
    t = (text or "").strip().strip("`")
    if t.lower().startswith("json"):
        t = t[4:].strip()
    try:
        data = json.loads(t)
        return (str(data.get("reply", "")).strip(),
                str(data.get("notes", "")).strip())
    except ValueError:
        pass
    # near-JSON fallback (unquoted keys, trailing prose):
    m = re.search(r'"?reply"?\s*:\s*"((?:[^"\\]|\\.)*)"', t,
                  re.IGNORECASE | re.DOTALL)
    if m:
        reply = m.group(1)
        n = re.search(r'"?notes"?\s*:\s*"((?:[^"\\]|\\.)*)"', t,
                      re.IGNORECASE | re.DOTALL)
        notes = n.group(1) if n else ""
        return reply.strip(), notes.strip()
    return t, ""


def _deepseek(key, system, msgs, model):
    """One DeepSeek chat/completions call -> (reply, notes)."""
    body = {"model": model,
            "messages": [{"role": "system", "content": system}] + msgs,
            "temperature": 0.4, "max_tokens": 400,
            "response_format": {"type": "json_object"}, "stream": False}
    resp = _curl_post(API_URL,
                      {"Content-Type": "application/json",
                       "Authorization": "Bearer " + key}, body)
    if "choices" not in resp:
        if not resp:
            msg = "no response from DeepSeek (network flaky? retry)"
        else:
            msg = str(resp.get("error", {}).get("message", resp))[:200]
        if "402" in msg or "balance" in msg.lower():
            raise SystemExit("DeepSeek account has no balance - top up, "
                             "then re-run.")
        raise RuntimeError("DeepSeek API error: " + msg)
    content = (resp["choices"][0]["message"].get("content") or "")
    return _parse_json_text(content)


def _gemini(key, system, msgs, model):
    """One Gemini generateContent call -> (reply, notes). Key goes in the
    x-goog-api-key header (never in the URL), same curl transport as the
    battery's proven Gemini path."""
    contents = []
    for m in msgs:
        role = {"user": "user", "assistant": "model"}.get(m["role"], "user")
        contents.append({"role": role,
                         "parts": [{"text": m["content"]}]})
    body = {"system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                # gemini-3.8-flash thinks by default and maxOutputTokens
                # INCLUDES thinking tokens — a tight cap truncates the
                # visible JSON. Budget 1024 so the reply always fits, and
                # ask for the lowest thinking level.
                "temperature": 0.4, "maxOutputTokens": 1024,
                "response_mime_type": "application/json",
                "thinkingConfig": {"thinkingLevel": "low"}}}
    resp = _curl_post(GEMINI_URL % model,
                      {"Content-Type": "application/json",
                       "x-goog-api-key": key}, body)
    if "candidates" not in resp:
        if not resp:
            msg = "no response from Gemini (network flaky? retry)"
        else:
            msg = str(resp.get("error", {}).get("message", resp))[:200]
        low = msg.lower()
        if "429" in msg or "quota" in low or "exhausted" in low:
            raise SystemExit("Gemini key has no quota left: " + msg)
        raise RuntimeError("Gemini API error: " + msg)
    parts = resp["candidates"][0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts
                   if isinstance(p, dict))
    t = (text or "").strip()
    if t.startswith("{") and not t.endswith("}"):
        # truncated mid-JSON (thinking ate the budget) — retry, don't speak it
        raise RuntimeError("Gemini JSON truncated (%d chars): %.40s"
                           % (len(t), t))
    return _parse_json_text(text)
