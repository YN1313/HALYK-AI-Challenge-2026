"""Stage 1-2: PDF -> text, document classification, borrower routing."""
from __future__ import annotations
import os, re, glob, json, hashlib
import concurrent.futures as cf

import pdfplumber

try:
    import pypdfium2 as pdfium
    import pytesseract
    from PIL import Image
    OCR = True
except Exception:                                   # pragma: no cover
    OCR = False


# ---------------------------------------------------------------- extraction

def _extract_one(path: str, cache_dir: str) -> tuple[str, str]:
    key = hashlib.md5(open(path, "rb").read()).hexdigest()
    cp = os.path.join(cache_dir, key + ".txt")
    if os.path.exists(cp):
        return path, open(cp, encoding="utf-8").read()
    try:
        parts = []
        scanned = []
        with pdfplumber.open(path) as pdf:
            for i, pg in enumerate(pdf.pages):
                body = pg.extract_text() or ""
                parts.append(body)
                # A page carrying an image and almost no characters is a scan.
                # Its content is invisible to text extraction, and in this
                # corpus scanned pages hold load-bearing tables — ownership
                # structures, collateral coverage, EBITDA add-backs.
                if pg.images and len(pg.chars) < 10:
                    scanned.append(i)
        for i in scanned:
            ocr = _ocr_page(path, i)
            if ocr:
                parts.append(ocr)
        text = "\n".join(parts)
    except Exception as exc:  # never crash the pipeline on one bad file
        text = ""
        print(f"[warn] extract failed {path}: {exc}")
    os.makedirs(cache_dir, exist_ok=True)
    open(cp, "w", encoding="utf-8").write(text)
    return path, text


def ocr_languages() -> str:
    """Language string for tesseract, degrading gracefully.

    With the Russian pack the whole page is recoverable, headings included.
    Without it, Latin entity names and digits still survive — enough for the
    ownership and collateral tables, which is why the table parsers key on
    structure rather than on wording.
    """
    if not OCR:
        return ""
    try:
        langs = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"
    if "rus" in langs:
        return "rus+eng"
    print("[warn] tesseract has no Russian pack: scanned pages will be read "
          "in Latin only. Install tesseract-ocr-rus for full coverage.")
    return "eng"


_LANGS: str | None = None


def _ocr_page(path: str, index: int, scale: float = 3.0) -> str:
    """Transcribe one scanned page."""
    global _LANGS
    if not OCR:
        print(f"[warn] scanned page {index} in {os.path.basename(path)} "
              f"cannot be read: OCR unavailable")
        return ""
    if _LANGS is None:
        _LANGS = ocr_languages()
    try:
        pdf = pdfium.PdfDocument(path)
        img = pdf[index].render(scale=scale).to_pil()
        return pytesseract.image_to_string(img, lang=_LANGS)
    except Exception as exc:
        print(f"[warn] OCR failed on {os.path.basename(path)} p{index}: {exc}")
        return ""


def extract_all(doc_dir: str, cache_dir: str, workers: int = 8) -> dict[str, str]:
    files = sorted(glob.glob(os.path.join(doc_dir, "*.pdf")))
    out: dict[str, str] = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        for path, text in ex.map(lambda f: _extract_one(f, cache_dir), files):
            out[os.path.basename(path)] = text
    return out


# ------------------------------------------------------------ classification

# Account identifiers are not all of one shape: alongside ACC-7801 the ledger
# carries TELE-4471. The authoritative list is the ledger's own account_id
# column, so routing is driven by that rather than by a guessed prefix.
# Sub-accounts (ACC-7801-05) appear only in decoy documents and must never
# route a document to a borrower.
ACC_RE = re.compile(r"\b([A-Z]{2,6}-\d{3,6})\b(?!-\d)")

DOC_TYPES = ("LOAN", "LOAN_SUPERSEDED", "KYC", "AUDIT", "TREASURY", "NOISE")


def classify(text: str) -> str:
    head = text[:600]
    # Documents appear in Russian and in English, and the superseded marker
    # must be tested first: a prior-year copy also calls itself an agreement.
    if re.search(r"НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ|Заменена и изложена в новой редакции|"
                 r"SUPERSEDED\s*—?\s*PRIOR-YEAR|Amended and restated by the current",
                 head, re.I):
        return "LOAN_SUPERSEDED"
    if re.search(r"ИСПОЛНИТЕЛЬНЫЙ ЭКЗЕМПЛЯР|EXECUTION COPY", head, re.I):
        return "LOAN"
    # The dossier is identified by its own registration number, not by the
    # ownership table: on scanned pages that table is invisible to text
    # extraction and only reappears after OCR.
    if re.search(r"KYC-|Знай своего клиента|голосующих прав|"
                 r"Know Your Customer|voting rights", text, re.I):
        return "KYC"
    if re.search(r"Служебная записка казначейства|КАЗНАЧЕЙСТВО ГРУППЫ|"
                 r"GROUP TREASURY|Treasury memorandum", text[:1200], re.I):
        return "TREASURY"
    if re.search(r"АУДИТОРСКОЕ ДЕЛО|Примечания к финансовой отчётности|"
                 r"СОГЛАСОВАННЫХ ПРОЦЕДУР|выводы по классификации операций|"
                 r"Registered Auditors|Notes to the financial statements|"
                 r"AGREED-UPON PROCEDURES", text[:2500], re.I):
        return "AUDIT"
    return "NOISE"


def account_of(text: str, known: set[str] | None = None) -> str | None:
    """Borrower account for a document, or None if it is unattributable.

    A document is routed only when it names exactly one account that the
    ledger actually knows. Restricting to known accounts matters: decoy
    documents cite plausible-looking identifiers that belong to nobody.
    """
    accs = set(ACC_RE.findall(text))
    if known is not None:
        accs &= known
    return accs.pop() if len(accs) == 1 else None


def build_index(docs: dict[str, str], known: set[str] | None = None) -> dict:
    """{account: {doctype: [filenames]}} plus a list of unrouted documents."""
    index: dict[str, dict[str, list[str]]] = {}
    orphans: list[str] = []
    for name, text in docs.items():
        dt = classify(text)
        acc = account_of(text, known)
        if acc is None or dt == "NOISE":
            orphans.append(name)
            continue
        index.setdefault(acc, {}).setdefault(dt, []).append(name)
    return {"by_account": index, "orphans": orphans}
