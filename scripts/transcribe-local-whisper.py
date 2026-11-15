#!/usr/bin/env python3
"""Transcribe a recording locally with faster-whisper (no API key, no upload).

Writes the kit's common transcript format (see docs/TOOLS-AND-API-KEYS.md):
  { "text", "audio_duration_secs", "language", "words": [{text, start, end, type}] }
with word times in seconds on the source recording's timeline.

Usage:
  npm run transcribe:local -- <input-media> [output.json] [--model small.en] [--language en]

Defaults: model "small.en" (English); output "<input-name>.transcript.json" next
to the input. Use --model small (or medium) with --language for other languages.
Environment: WHISPER_VENV is the Python with faster-whisper (default
/opt/whisper-venv); HF_HOME is the model cache (default /opt/whisper-models).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def media_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return round(float(out), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--language", default=None, help="e.g. en; default: detect (or English for *.en models)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    src = args.input.resolve()
    if not src.is_file():
        sys.exit(f"Input not found: {src}")
    out = (args.output or src.with_name(src.stem + ".transcript.json")).resolve()
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}; pass a new output filename.")

    os.environ.setdefault("HF_HOME", "/opt/whisper-models")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from faster_whisper import WhisperModel  # imported after HF_HOME is set

    language = args.language or ("en" if args.model.endswith(".en") else None)
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=args.threads)
    segments, info = model.transcribe(
        str(src), language=language, word_timestamps=True,
        vad_filter=False,  # keep real pauses; cut-silences decides what to remove
        condition_on_previous_text=False,
    )

    words = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if not text:
                continue
            words.append({"text": text, "start": round(w.start, 3), "end": round(w.end, 3), "type": "word"})
        print(f"\r  transcribed {seg.end:7.1f}s / {info.duration:.1f}s", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    result = {
        "text": " ".join(w["text"] for w in words),
        "audio_duration_secs": media_duration(src),
        "language": info.language,
        "source": {"file": src.name, "engine": "faster-whisper", "model": args.model},
        "words": words,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(words)} words to {out}")


if __name__ == "__main__":
    main()
