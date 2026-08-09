"""Pre-flight check. Run before the private dataset drops, not after.

Every item here is something that fails silently: the pipeline keeps running
and produces a plausible answer that is wrong. Exit code is non-zero if any
required item is missing.
"""
from __future__ import annotations
import importlib, os, shutil, subprocess, sys

REQUIRED = ["pdfplumber", "pandas"]
OCR_PKGS = ["pypdfium2", "pytesseract", "PIL"]

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
problems = 0
warnings = 0


def line(state: str, text: str, hint: str = ""):
    global problems, warnings
    print(f"[{state}] {text}")
    if hint:
        print(f"        {hint}")
    if state == BAD:
        problems += 1
    elif state == WARN:
        warnings += 1


def main() -> int:
    print("python:", sys.version.split()[0], "\n")

    for mod in REQUIRED:
        try:
            importlib.import_module(mod)
            line(OK, f"{mod} importable")
        except Exception:
            line(BAD, f"{mod} missing", "pip install -r requirements.txt")

    ocr_ready = True
    for mod in OCR_PKGS:
        try:
            importlib.import_module(mod)
        except Exception:
            ocr_ready = False
    if ocr_ready:
        line(OK, "OCR python packages present")
    else:
        line(BAD, "OCR python packages missing",
             "pip install pypdfium2 pytesseract Pillow")

    binary = shutil.which("tesseract")
    if not binary:
        line(BAD, "tesseract binary not found",
             "apt-get install -y tesseract-ocr tesseract-ocr-rus  — six pages in "
             "the corpus are images; without OCR their tables vanish silently")
    else:
        try:
            langs = subprocess.run(["tesseract", "--list-langs"],
                                   capture_output=True, text=True).stdout
        except Exception:
            langs = ""
        if "rus" in langs:
            line(OK, f"tesseract at {binary} with Russian pack")
        else:
            line(WARN, f"tesseract at {binary} without Russian pack",
                 "apt-get install -y tesseract-ocr-rus — Latin names and digits "
                 "still parse, Cyrillic headings do not")

    if os.environ.get("ANTHROPIC_API_KEY"):
        line(OK, "ANTHROPIC_API_KEY set — LLM formula fallback active")
    else:
        line(WARN, "ANTHROPIC_API_KEY not set",
             "an unrecognised covenant formula will score zero instead of "
             "falling back to the model")

    print()
    if problems:
        print(f"{problems} blocking problem(s), {warnings} warning(s) — fix before running")
        return 1
    print(f"ready ({warnings} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
