#!/usr/bin/env python3
"""tts_test_bot.py — Arzoo voice-test chat window (internal team test).

Type a line, Arzoo answers in text and the round-8 voice. This is the
simple team version of the tutor app: chat + a speaking-speed slider +
language choice. The speed slider maps straight to the synth
length_scale (1.0 = fastest … 1.3 = slowest; the tested sweet spot is
1.10–1.25, default 1.20). +4 dB loudness gain is already baked in.

Which LLM answers is chosen by tts_test_bot/.env — the uncommented key
wins (DeepSeek or Gemini). The title bar shows the active provider.

Every turn is also logged to out/conversation.jsonl (spoken text, speed,
voice-check + gate results) so testers can paste findings back easily.

Run (any python 3.10+ with onnxruntime, numpy, g2p_en installed):
  python tts_test_bot.py
Or double-click run.bat. --smoke opens the window briefly and closes it
(construct test, no LLM call, no audio).
"""
import json
import sys
import threading
import tkinter as tk
import winsound
from pathlib import Path
from tkinter import scrolledtext

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bot_core as bc  # noqa: E402  (LLM brain, guardrails, wrapper)
import synth            # noqa: E402  (r8 engine, +4 dB, voice check, gate)

GREETING = ("Namaste! Main Arzoo hoon, aapki padhaai ki saathi. "
            "Aap aaj kya padhna chahte hain?")
OUT = HERE / "out"
LOG = OUT / "conversation.jsonl"


class ArzooApp:
    def __init__(self, root, smoke=False):
        self.root = root
        self.smoke = smoke
        root.title("Arzoo — Team Voice Test (r8 voice)")
        root.geometry("760x560")

        # transcript
        self.chat = scrolledtext.ScrolledText(root, wrap=tk.WORD,
                                              font=("Segoe UI", 11),
                                              state=tk.DISABLED)
        self.chat.tag_configure("you", foreground="#1a6b3c")
        self.chat.tag_configure("arzoo", foreground="#1a3c6b")
        self.chat.tag_configure("meta", foreground="#8a8a8a",
                                font=("Segoe UI", 9))
        self.chat.tag_configure("warn", foreground="#b04000",
                                font=("Segoe UI", 9))
        self.chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        # reply-language choice
        bar = tk.Frame(root)
        bar.pack(fill=tk.X, padx=8)
        self.lang = tk.StringVar(value="auto")
        for val, label in (("auto", "Auto"), ("hng", "Hinglish"),
                           ("en", "English")):
            tk.Radiobutton(bar, text=label, value=val, variable=self.lang,
                           font=("Segoe UI", 9)).pack(side=tk.LEFT,
                                                      padx=(0, 10))

        # speed slider — length_scale 1.00–1.30, sweet spot 1.10–1.25
        speedbar = tk.Frame(root)
        speedbar.pack(fill=tk.X, padx=8, pady=(4, 0))
        tk.Label(speedbar, text="Speaking speed:",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.speed = tk.DoubleVar(value=1.20)
        tk.Scale(speedbar, from_=1.00, to=1.30, resolution=0.05,
                 orient=tk.HORIZONTAL, variable=self.speed, length=280,
                 showvalue=True, font=("Segoe UI", 9),
                 command=self._on_speed).pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(speedbar, text="1.0 = fast … 1.3 = slow",
                 font=("Segoe UI", 8), fg="#8a8a8a").pack(side=tk.LEFT,
                                                          padx=6)

        # input row
        row = tk.Frame(root)
        row.pack(fill=tk.X, padx=8, pady=(4, 8))
        self.entry = tk.Entry(row, font=("Segoe UI", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda e: self.send())
        tk.Button(row, text="Send", command=self.send, width=8,
                  font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=(6, 0))

        self.status = tk.Label(root, text="starting...", anchor="w",
                               font=("Segoe UI", 9), fg="#555555")
        self.status.pack(fill=tk.X, padx=10, pady=(0, 4))

        # shared state (engine load takes a few seconds)
        self.llm = bc.TutorLLM()
        self.rules = bc.load_rules()
        self.eng = synth.SynthEngine()
        self.history = []          # last turns for LLM context
        self.n = 0                 # wav / log counter
        self.busy = not smoke      # greeting keeps it busy until spoken
        self._on_speed(self.speed.get())

        self.say("Arzoo", GREETING)
        if not smoke:
            threading.Thread(target=self._greet, daemon=True).start()
            self.status.config(text="ready — type below and press Enter")
        else:
            self.status.config(text="smoke test")

    # ---- transcript (UI thread only) ----

    def say(self, who, text):
        tag = "arzoo" if who == "Arzoo" else "you"
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, "%s> %s\n\n" % (who, text), tag)
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    def meta(self, text, warn=False):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text + "\n", "warn" if warn else "meta")
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _on_speed(self, v):
        self.root.title("Arzoo — Team Voice Test (r8 · %s · speed %.2f)"
                        % (self.llm.provider, float(v)))

    # ---- turn flow (worker threads, UI updates via root.after) ----

    def send(self):
        if self.busy:
            return
        user = self.entry.get().strip()
        if not user:
            return
        self.entry.delete(0, tk.END)
        self.say("You", user)
        self.busy = True
        self.status.config(text="Arzoo is thinking...")
        ls = self.speed.get()
        threading.Thread(target=self._turn, args=(user, ls),
                         daemon=True).start()

    def _greet(self):
        try:
            self._render_turn(GREETING, "", [], GREETING, "<greeting>",
                              self.speed.get())
        except Exception as e:
            self.root.after(0, self._fail, str(e))
        finally:
            self.root.after(0, self._idle)

    def _turn(self, user, ls):
        try:
            hint = {"auto": "the same language the student used",
                    "hng": "Roman Hinglish",
                    "en": "English"}[self.lang.get()]
            msgs = self.history[-6:] + [{"role": "user", "content":
                                         'Student says: "%s"\nReply in %s.'
                                         % (user, hint)}]
            reply, notes = self.llm.chat(bc.build_guardrails(self.rules),
                                         msgs)
            if not reply:
                self.root.after(0, self.meta,
                                "(LLM returned an empty reply)", True)
                return
            clean, changes = bc.wrap_reply(reply, self.rules)
            self._render_turn(clean, notes, changes, reply, user, ls)
            self.history.append({"role": "user", "content": user})
            self.history.append({"role": "assistant", "content": reply})
        except Exception as e:
            self.root.after(0, self._fail, str(e))
        finally:
            self.root.after(0, self._idle)

    def _render_turn(self, clean, notes, changes, raw, user, ls):
        """Synth + wav + voice check + gate (worker thread), then show it."""
        self.n += 1
        n = self.n
        self.root.after(0, self.status.config, "Arzoo is speaking...")
        wavp, events, n_pass, dur = synth.render_reply(self.eng, clean,
                                                       OUT, n, ls)
        winsound.PlaySound(str(wavp),
                           winsound.SND_FILENAME | winsound.SND_ASYNC)
        OUT.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"turn": n, "user": user, "reply": raw,
                                 "clean": clean, "notes": notes,
                                 "wrapper": changes, "speed": ls,
                                 "wav": str(wavp), "events": events,
                                 "gate": "%d/6" % n_pass},
                                ensure_ascii=False) + "\n")

        def finish():
            self.say("Arzoo", clean)
            if notes and notes.lower() != "none":
                self.meta("  [dodged: %s]" % notes)
            if changes:
                self.meta("  [wrapper: %s]" % ", ".join(changes))
            qc = "clean" if not events else ", ".join(events)
            self.meta("  voice check: %s · gate %d/6 · speed %.2f · %.1fs"
                      % (qc, n_pass, ls, dur), warn=bool(events))
        self.root.after(0, finish)

    def _fail(self, msg):
        self.meta("[error: %s]" % msg, True)
        self.status.config(text="ready — type below and press Enter")

    def _idle(self):
        self.busy = False
        self.status.config(text="ready — type below and press Enter")


def main():
    smoke = "--smoke" in sys.argv
    root = tk.Tk()
    ArzooApp(root, smoke=smoke)
    if smoke:
        root.after(400, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
