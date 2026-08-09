"""Stage 3-4: extract typed facts from routed documents.

Everything here returns Decimal or None. Nothing here does covenant maths.
"""
from __future__ import annotations
import re
from decimal import Decimal
from dataclasses import dataclass, field


MONEY = r"\$\s?([0-9][0-9,]*(?:\.\d{2})?)"
RATIO = r"([0-9]+(?:\.\d+)?)\s?x"


def money(s: str) -> Decimal:
    return Decimal(s.replace(",", "").replace("$", "").strip())


# ------------------------------------------------------------------ covenants

@dataclass
class Covenant:
    clause: str            # "6.1"
    title: str
    text: str
    limit: Decimal | None = None
    limit_kind: str | None = None   # "money" | "ratio"
    direction: str | None = None    # "max" | "min"
    doc: str = ""


# Clause numbering is not fixed. Most agreements put the financial covenants
# in article 6, one puts them in article 5, and some carry a fourth covenant.
# The submission template is authoritative about which clauses exist for a
# borrower, so extraction is driven by that list rather than by an assumed
# article number. Marker words appear in both languages.
CLAUSE_RE = re.compile(r"(?:Пункт|Clause|Section|Статья)\s+(\d{1,2}\.\d{1,2})\s+(.{0,120})",
                       re.S)

# The last covenant in an article has no clause marker after it, so its chunk
# would otherwise swallow the rest of the agreement — thousands of words of
# negative covenants and boilerplate that contain enough stray wording to
# invert the direction of the test and turn an absolute limit into a ratio.
ARTICLE_END = re.compile(r"(?:Статья|Article|ARTICLE)\s+\d{1,2}\s*[—–\-]")


def parse_covenants(loan_text: str, doc: str = "",
                    wanted: list[str] | None = None) -> dict[str, Covenant]:
    """Covenants keyed by clause number.

    `wanted` restricts extraction to the clauses the template asks for, which
    keeps boilerplate clauses (governing law, notices, assignment) out even
    when they share the numbering style.
    """
    marks = [(m.start(), m.group(1), " ".join(m.group(2).split()))
             for m in CLAUSE_RE.finditer(loan_text)]
    out: dict[str, Covenant] = {}
    for i, (pos, clause, title) in enumerate(marks):
        if wanted is not None and clause not in wanted:
            continue
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(loan_text)
        nxt = ARTICLE_END.search(loan_text, pos + 20)
        if nxt and nxt.start() < stop:
            stop = nxt.start()
        chunk = " ".join(loan_text[pos:stop].split())
        title = re.split(r"\.\s", title)[0]
        cov = Covenant(clause=clause, title=title, text=chunk, doc=doc)
        cov.direction = _direction(cov.text)
        cov.limit, cov.limit_kind = _limit(cov.text, cov.direction)
        # A later occurrence of the same number is a cross-reference, not the
        # covenant itself; the first, longest match wins.
        if clause not in out or len(cov.text) > len(out[clause].text):
            out[clause] = cov
    return out


def _direction(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("не менее", "не ниже", "минимальн", "minimum",
                              "не допускать снижения", "at least", "no less than",
                              "not fall below")):
        return "min"
    return "max"


def _limit(text: str, direction: str) -> tuple[Decimal | None, str | None]:
    """Primary threshold of the covenant.

    Springing and conditional covenants carry two numbers (a trigger and a
    limit). A ratio is always the covenant limit; a money figure alongside it
    is the trigger.
    """
    ratios = re.findall(RATIO, text)
    moneys = re.findall(MONEY, text)
    if ratios:
        return Decimal(ratios[0]), "ratio"
    if moneys:
        return money(moneys[0]), "money"
    return None, None


def trigger_of(text: str) -> Decimal | None:
    """Money threshold that switches a springing covenant on."""
    if "только при условии" not in text and "применяется" not in text:
        return None
    m = re.findall(MONEY, text)
    return money(m[0]) if m else None


# ---------------------------------------------------------------------- KYC

@dataclass
class KYCFacts:
    threshold: Decimal | None = None
    holdings: dict[str, Decimal] = field(default_factory=dict)
    doc: str = ""

    named: set[str] = field(default_factory=set)   # stated outright, no table
    conclusive: bool = False                       # dossier settles the question

    def related(self) -> set[str]:
        if self.named or self.conclusive:
            return set(self.named)
        if self.threshold is None:
            return set()
        return {k for k, v in self.holdings.items() if v >= self.threshold}


# Percentage tables are extracted without relying on Cyrillic. On scanned
# pages an English-only OCR model garbles every Russian word but leaves Latin
# entity names and digits intact, so the structure — not the wording — has to
# carry the parse:
#
#   a data row  ends with a percentage        "Zhetysu Capital Partners LLP 31.2%"
#   the rule    has a percentage mid-sentence "... владеет 25.0% и более ..."
#
# Entity names in this dataset also carry irregular punctuation, and it is
# systematically the controlling holder that is written irregularly:
#   Syrdarya Capital Holding, LLP   "Saryarka Capital Partners" LLP
#   Aral Capital Partners, LLP      Ulytau Capital LLP.
ROW_RE = re.compile(r"^(.{3,70}?)\s+(\d{1,3}(?:[.,]\d+)?)\s?%\s*$")
INLINE_PCT = re.compile(r"(\d{1,3}(?:[.,]\d+)?)\s?%\s*\S")

# Signatures survive OCR because they are matched on fragments, not words.
OWNERSHIP_SIG = ("голосующ", "ronoc", "ronocy", "voting")
COLLATERAL_SIG = ("залог", "3alor", "3aslor", "aslore", "актив", "AKTHBOB", "pledge")


def _tables(text: str) -> list[tuple[dict[str, Decimal], Decimal | None]]:
    """Every percentage table in the document, with its threshold."""
    out = []
    rows: dict[str, Decimal] = {}
    threshold = None
    for line in text.splitlines():
        line = line.strip()
        m = ROW_RE.match(line)
        if m:
            name = re.sub(r"[\"«»]", " ", m.group(1)).strip(" ,.")
            if len(name) >= 3 and not name.isdigit():
                rows[name] = Decimal(m.group(2).replace(",", "."))
            continue
        inline = INLINE_PCT.search(line)
        if inline and rows:
            threshold = Decimal(inline.group(1).replace(",", "."))
            out.append((rows, threshold))
            rows, threshold = {}, None
    if rows:
        out.append((rows, threshold))
    return out


def _signature(text: str, table_names) -> str:
    """Which kind of table this is, judged by the words around it."""
    anchor = min((text.find(n) for n in table_names if text.find(n) >= 0),
                 default=-1)
    window = text[max(0, anchor - 400):anchor] if anchor >= 0 else text[:1500]
    low = window.lower()
    if any(sig.lower() in low for sig in COLLATERAL_SIG):
        return "collateral"
    if any(sig.lower() in low for sig in OWNERSHIP_SIG):
        return "ownership"
    return "unknown"


# Dossiers come in three shapes, and only the first is a table:
#   1. an ownership table with a percentage threshold
#   2. numbered records naming a counterparty and its status outright
#   3. an explicit statement that no related parties exist
# The third matters most: it is a positive finding, not missing data, and any
# heuristic that guesses related parties must stand down in front of it.
RECORD_HEAD = re.compile(r"(?:Запись|Record)\s*\d+\.\s*")
RECORD_NAME = re.compile(r"[«\"]([^»\"]{3,70})[»\"]")
NONE_FOUND = re.compile(
    r"[Сс]вязанные стороны среди контрагентов не выявлены|"
    r"[Сс]вязанных сторон не выявлено|no related parties (?:were )?identified", re.I)

AFFILIATE_WORDS = ("АФФИЛИРОВАННОЕ", "СВЯЗАННОЙ СТОРОНОЙ", "AFFILIATE",
                   "RELATED PARTY")
UNRESTRICTED_WORDS = ("НЕОГРАНИЧЕННОЙ ДОЧЕРНЕЙ", "НЕОГРАНИЧЕННАЯ ДОЧЕРНЯЯ",
                      "UNRESTRICTED SUBSIDIAR")
RESTRICTED_WORDS = ("ОГРАНИЧЕННОЙ ДОЧЕРНЕЙ", "ОГРАНИЧЕННАЯ ДОЧЕРНЯЯ",
                    "RESTRICTED SUBSIDIAR")


def parse_records(text: str) -> tuple[set[str], set[str], bool]:
    """(affiliates, unrestricted subsidiaries, dossier is conclusive).

    A record is conclusive either way: naming an affiliate establishes one,
    and naming a Restricted Subsidiary establishes that this counterparty is
    *not* an affiliate. Both stop the guessing.
    """
    affiliates: set[str] = set()
    unrestricted: set[str] = set()
    conclusive = bool(NONE_FOUND.search(text))

    for part in RECORD_HEAD.split(text)[1:]:
        body = " ".join(part.split())
        head = body[:300]
        name = RECORD_NAME.search(head)
        if not name:
            continue
        who = name.group(1).strip()
        # Unrestricted is checked first: its wording contains the restricted one.
        # Subsidiary records answer a different question — whether a party sits
        # inside the security perimeter — so they must not suppress an ownership
        # table that is present in the same dossier. Only a record that speaks
        # to affiliation settles affiliation.
        if any(w in head for w in UNRESTRICTED_WORDS):
            unrestricted.add(who)
        elif any(w in head for w in RESTRICTED_WORDS):
            pass
        elif any(w in head for w in AFFILIATE_WORDS):
            affiliates.add(who)
            conclusive = True
    return affiliates, unrestricted, conclusive


def parse_kyc(text: str, doc: str = "") -> KYCFacts:
    affiliates, _, conclusive = parse_records(text)
    if conclusive:
        k = KYCFacts(threshold=None, holdings={}, doc=doc)
        k.named = affiliates
        k.conclusive = True
        return k

    best = KYCFacts(doc=doc)
    for rows, threshold in _tables(text):
        if not rows or threshold is None:
            continue
        if _signature(text, rows) == "collateral":
            continue
        if len(rows) > len(best.holdings):
            best = KYCFacts(threshold=threshold, holdings=rows, doc=doc)
    return best


def parse_unrestricted(text: str, doc: str = "") -> set[str]:
    named = parse_records(text)[1]
    if named:
        return named
    return _parse_unrestricted_table(text, doc)


def _parse_unrestricted_table(text: str, doc: str = "") -> set[str]:
    """Subsidiaries outside the lender's security perimeter.

    A subsidiary whose pledged-asset share falls below the stated floor sits
    outside the collateral perimeter and counts as Unrestricted.
    """
    for rows, threshold in _tables(text):
        if not rows or threshold is None:
            continue
        if _signature(text, rows) != "collateral":
            continue
        return {k for k, v in rows.items() if v < threshold}
    return set()


# -------------------------------------------------------------------- audit

@dataclass
class Adjustment:
    """An auditor instruction about one ledger row.

    The row is identified either by transaction id or — more often — by the
    pair (amount, counterparty), because auditors write about sums paid to
    named parties, not about internal identifiers.
    """
    kind: str                        # "cutoff" | "reclass" | "amount"
    to_category: str | None
    note: str
    txn_id: str | None = None
    amount: Decimal | None = None
    new_amount: Decimal | None = None
    counterparty: str | None = None
    doc: str = ""


TXN_RE = re.compile(r"\b(TXN-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d+)\b")

CUTOFF_HINTS = ("относится к услугам, оказанным", "относится к периоду",
                "оказаны в", "выходит за пределы", "не относится к ковенантному периоду",
                "исключена из ковенантного периода", "исключено из ковенантного периода",
                "отсрочено и исключено", "переходят только в")

# An auditor who *considered* a reclassification and declined it has changed
# nothing. Several covenants say so explicitly ("реклассификации, рассмотренные
# и отклонённые аудиторами, в расчёт не принимаются"), and the paragraph reads
# almost exactly like an accepted one — same verb, same amount, same target
# category. Rejection must be detected before the verb is trusted.
REJECTION_HINTS = (
    "первоначальная классификация", "классификация сохраняется",
    "сохраняется", "не производилась", "не производится",
    "корректировка не требуется", "корректировка для целей ковенантов не требуется",
    "рассмотрено и отклонено", "рассмотрена и отклонена", "отклонён",
    "рассматривалась на предмет", "рассматривался на предмет",
)
RECLASS_HINTS = ("переклассифиц", "переквалифиц", "реклассифиц", "относится к статье",
                 "подлежит отнесению", "отнесена")

CATEGORY_WORDS = {
    # Stems, not full words: the auditor writes "Страховые премии", not
    # "страхование", and a stem that is too long silently matches nothing.
    "процент": "interest", "финансир": "financing", "капитальн": "capex",
    "операционн": "opex", "выручк": "revenue", "аренд": "lease",
    "персонал": "payroll", "оплату труда": "payroll", "заработн": "payroll",
    "коммунальн": "utilities", "налог": "tax", "маркетинг": "marketing",
    "страхов": "insurance", "связи": "telecom",
}


DRAFT_MARKERS = ("ПРОЕКТ — ПРОМЕЖУТОЧНАЯ", "НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ ПОЗИЦИЕЙ",
                 "заменена окончательным отчётом", "Направлена для обсуждения")


def is_draft(text: str) -> bool:
    """Interim auditor working papers are superseded by the final report."""
    return any(m in text[:1500] for m in DRAFT_MARKERS)


AMOUNT_FIX = re.compile(
    r"(TXN-[A-Z0-9]+-\d+).{0,220}?фактическая сумма операции составляет\s*"
    r"\$\s?([0-9][0-9,]*(?:\.\d{2})?)\s*\((расход|поступление)\)", re.S)


def parse_treasury(text: str, doc: str = "") -> list[Adjustment]:
    """Treasury memos carry amounts that never reached the ledger export.

    A row whose amount is blank is not zero — it is unknown, and the covenant
    figure is wrong until the real number is supplied from here.
    """
    out = []
    flat = " ".join(text.split())
    for m in AMOUNT_FIX.finditer(flat):
        value = Decimal(m.group(2).replace(",", ""))
        out.append(Adjustment(
            kind="amount", to_category=None, txn_id=m.group(1),
            new_amount=-value if m.group(3) == "расход" else value,
            note=flat[max(0, m.start() - 60):m.end()][:400], doc=doc))
    return out


def parse_audit(text: str, doc: str = "") -> list[Adjustment]:
    """Extract auditor instructions from a covenant supplement or AUP report.

    Superseded interim working papers are skipped entirely. Paragraphs that
    merely cross-reference another report ("вывод изложен в отчёте №...") carry
    no instruction and are skipped too, so the authoritative document wins.
    """
    if is_draft(text):
        return []

    out: list[Adjustment] = []
    parts = re.split(r"\n?\((\d+\.\d+)\)\s", text)
    chunks = [parts[i + 1] for i in range(1, len(parts) - 1, 2)] if len(parts) > 2 else []
    for chunk in chunks:
        flat = " ".join(chunk.split())
        if CROSS_REF.search(flat):
            continue

        if any(h in flat.lower() for h in REJECTION_HINTS):
            continue

        kind = ("cutoff" if any(h in flat for h in CUTOFF_HINTS) else
                "reclass" if any(h in flat for h in RECLASS_HINTS) else "note")
        if kind == "note":
            continue

        to_cat = _target_category(flat)
        ids = TXN_RE.findall(flat)
        amt = AMOUNT_RE.search(flat)
        cp = COUNTERPARTY_RE.search(flat)

        if ids:
            for tid in dict.fromkeys(ids):
                out.append(Adjustment(kind=kind, to_category=to_cat, note=flat[:400],
                                      txn_id=tid, doc=doc))
        elif amt:
            out.append(Adjustment(
                kind=kind, to_category=to_cat, note=flat[:400],
                amount=Decimal(amt.group(1).replace(",", "")),
                counterparty=cp.group(1).strip(' "«»') if cp else None,
                doc=doc))
    return out


CROSS_REF = re.compile(r"изложен в отчёте|в настоящих примечаниях не повторяется|"
                       r"отобрана для проверки классификации")
AMOUNT_RE = re.compile(r"\$\s?([0-9][0-9,]*(?:\.\d{2})?)")
COUNTERPARTY_RE = re.compile(r"контрагенту\s+[«\"]?([A-Za-z][^,.»\"]*)")


def _target_category(flat: str) -> str | None:
    """Category the auditor moves the amount *into*.

    The sentence names both the original and the target classification, so the
    text after the reclassification verb is what counts.
    """
    m = re.search(r"переклассифиц\w*|переквалифиц\w*|реклассифиц\w*", flat, re.I)
    tail = flat[m.end():] if m else flat
    low = tail.lower()
    hits = [(low.index(w), c) for w, c in CATEGORY_WORDS.items() if w in low]
    return min(hits)[1] if hits else None


# -------------------------------------------------------------------- FX

# The covenants say foreign amounts are translated "по курсу, раскрытому
# аудитором" — there is no rate table, only a worked example buried in the
# supplement: an invoice of 72,146.75 EUR settled by a payment of $83,690.23.
# The rate is the quotient of the two.
FX_RE = re.compile(
    r"([0-9][0-9,]*(?:\.\d+)?)\s*(EUR|USD|GBP|KZT)\b.{0,160}?"
    r"\$\s?([0-9][0-9,]*(?:\.\d{2})?)", re.S)


def parse_fx(text: str, doc: str = "") -> dict[str, Decimal]:
    """Exchange rates to USD, keyed by currency code."""
    rates: dict[str, Decimal] = {}
    flat = " ".join(text.split())
    for m in FX_RE.finditer(flat):
        foreign, code, usd = m.group(1), m.group(2), m.group(3)
        if code == "USD":
            continue
        try:
            f = Decimal(foreign.replace(",", ""))
            u = Decimal(usd.replace(",", ""))
        except Exception:
            continue
        if f <= 0:
            continue
        rate = u / f
        # A plausible FX rate, not a coincidence of two unrelated numbers.
        if Decimal("0.2") < rate < Decimal("5"):
            rates.setdefault(code, rate)
    return rates


# ----------------------------------------------------------- EBITDA add-backs

# Auditors disclose one-off items in two ways: by naming the transaction, or in
# a table of amounts governed by a materiality floor ("разовыми признаются
# статьи в сумме не менее $500,000.00"). Items below the floor are not added
# back — the floor is part of the covenant arithmetic, not commentary.
ADDBACK_TXN = re.compile(
    r"(TXN-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d+)\s*\(\$\s?([0-9][0-9,]*(?:\.\d{2})?)\)"
    r"[^.]{0,120}?разов\w*\s+стать\w*,?\s+прибавляем", re.S)
ADDBACK_ROW = re.compile(r"\$\s?([0-9][0-9,]*(?:\.\d{2})?)")
ADDBACK_FLOOR = re.compile(
    r"[Рр]азовыми[^.]{0,80}?не менее\s*\$\s?([0-9][0-9,]*(?:\.\d{2})?)")
ADDBACK_SECTION = re.compile(r"Корректировки EBITDA|разовые статьи, выявленные")


def parse_addbacks(text: str, doc: str = "") -> Decimal:
    """Total added back to EBITDA for covenant purposes."""
    if is_draft(text):
        return Decimal(0)
    flat = " ".join(text.split())

    named = {m.group(1): Decimal(m.group(2).replace(",", ""))
             for m in ADDBACK_TXN.finditer(flat)}
    if named:
        return sum(named.values(), Decimal(0))

    start = ADDBACK_SECTION.search(flat)
    if not start:
        return Decimal(0)
    tail = flat[start.end():]
    floor_m = ADDBACK_FLOOR.search(tail)
    floor = Decimal(floor_m.group(1).replace(",", "")) if floor_m else Decimal(0)
    body = tail[:floor_m.start()] if floor_m else tail[:800]
    total = Decimal(0)
    for m in ADDBACK_ROW.finditer(body):
        v = Decimal(m.group(1).replace(",", ""))
        if v >= floor:
            total += v
    return total
