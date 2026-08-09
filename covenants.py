"""Stage 6: deterministic covenant evaluation.

No LLM touches this file. Every evaluator returns (actual, evidence_txn_id).
Status is derived from actual vs limit by the caller, so the comparison logic
lives in exactly one place.
"""
from __future__ import annotations
import re
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

from . import ledger as L


Q2 = Decimal("0.01")


def q(x: Decimal) -> float:
    return float(Decimal(x).quantize(Q2, rounding=ROUND_HALF_UP))


def _safe_div(a: Decimal, b: Decimal) -> Decimal:
    return a / b if b else Decimal(0)


# --------------------------------------------------------------- 6.1 family
# Each entry: (regex on covenant title/text, evaluator)

def cap_intensity(ctx) -> tuple[Decimal, str | None]:
    """capex / (opex + lease)"""
    r = ctx.rows
    return _safe_div(L.total(r, "capex"),
                     L.total(r, "opex") + L.total(r, "lease")), None


def interest_cover(ctx) -> tuple[Decimal, str | None]:
    """(revenue - opex) / interest"""
    r = ctx.rows
    ebitda = L.total(r, "revenue", "in") - L.total(r, "opex")
    return _safe_div(ebitda, L.total(r, "interest")), ctx.reclass_into("interest")


def cover_of_applications(ctx) -> tuple[Decimal, str | None]:
    """(revenue + financing inflows) / (opex + capex)"""
    r = ctx.rows
    src = L.total(r, "revenue", "in") + L.total(r, "financing", "in")
    use = L.total(r, "opex") + L.total(r, "capex")
    return _safe_div(src, use), None


def springing_leverage(ctx) -> tuple[Decimal, str | None]:
    """financing inflows / EBITDA, only tested if inflows exceed the trigger."""
    r = ctx.rows
    fin = L.total(r, "financing", "in")
    ebitda = L.total(r, "revenue", "in") - L.total(r, "opex")
    return _safe_div(fin, ebitda), None


def rp_share_of_opex(ctx) -> tuple[Decimal, str | None]:
    r = ctx.rows
    rp = abs(sum(L.payments_to(r, ctx.related).amount, Decimal(0)))
    return _safe_div(rp, L.total(r, "opex")), None


def largest_overhead(ctx) -> tuple[Decimal, str | None]:
    """max(payroll, utilities) — tested individually, not summed."""
    r = ctx.rows
    cands = {c: L.total(r, c) for c in ("payroll", "utilities")}
    cat = max(cands, key=cands.get)
    return cands[cat], None


def _fallback(ctx) -> tuple[Decimal, str | None]:
    """Unknown 6.1 shape: fall back on the biggest named category in the text."""
    r = ctx.rows
    for word, cat in (("капитальн", "capex"), ("выручк", "revenue"),
                      ("персонал", "payroll"), ("оплату труда", "payroll"),
                      ("операционн", "opex")):
        if word in ctx.cov.text.lower():
            sign = "in" if cat == "revenue" else "out"
            return L.total(r, cat, sign), None
    return Decimal(0), None


DISPATCH_61 = [
    (r"capital intensity|капиталоёмкости",            cap_intensity),
    (r"покрытия процентов|interest cover",            interest_cover),
    (r"cover of applications|покрытия.*источник",     cover_of_applications),
    (r"springing|поступлений по финансированию к",    springing_leverage),
    (r"доля платежей связанным сторонам в операционн", rp_share_of_opex),
    (r"наибольш.*накладн|individual overhead",        largest_overhead),
]


# --------------------------------------------------------------- 6.2 family

def revenue_min(ctx) -> tuple[Decimal, str | None]:
    return L.total(ctx.rows, "revenue", "in"), ctx.cutoff_on("revenue")


def category_max(ctx) -> tuple[Decimal, str | None]:
    """Max spend on a named category. The category is named in the text."""
    cat = ctx.named_category() or "capex"
    return L.total(ctx.rows, cat), ctx.reclass_into(cat)


def overhead_ceiling(ctx) -> tuple[Decimal, str | None]:
    return largest_overhead(ctx)


# --------------------------------------------------------------- 6.3 family

def related_party_abs(ctx) -> tuple[Decimal, str | None]:
    r = ctx.rows
    hits = L.payments_to(r, ctx.related)
    total = abs(sum(hits.amount, Decimal(0)))
    ev = ctx.decisive_related(hits, ctx.cov.limit)
    return total, ev


def related_party_ratio(ctx) -> tuple[Decimal, str | None]:
    r = ctx.rows
    hits = L.payments_to(r, ctx.related)
    rp = abs(sum(hits.amount, Decimal(0)))
    rev = L.total(r, "revenue", "in")
    ratio = _safe_div(rp, rev)
    limit_abs = (ctx.cov.limit or Decimal(0)) * rev
    ev = ctx.decisive_related(hits, limit_abs)
    return ratio, ev


# ------------------------------------------------------------------ context

class Ctx:
    def __init__(self, cov, rows: pd.DataFrame, related: set[str], adjustments,
                 unrestricted: set[str] | None = None,
                 addbacks: Decimal | None = None):
        self.cov = cov
        self.rows = rows
        self.related = related
        self.adjustments = adjustments
        self.unrestricted = unrestricted or set()
        self.addbacks = addbacks or Decimal(0)

    def named_category(self) -> str | None:
        t = self.cov.text.lower()
        for word, cat in (("капитальные затраты", "capex"),
                          ("операционные расходы", "opex"),
                          ("расходы на персонал", "payroll"),
                          ("оплату труда", "payroll"),
                          ("коммунальн", "utilities"),
                          ("маркетинг", "marketing"),
                          ("страхован", "insurance"),
                          ("аренд", "lease"),
                          ("выручк", "revenue")):
            if word in t:
                return cat
        return None

    def reclass_into(self, category: str) -> str | None:
        for a in self.adjustments:
            if a.kind == "reclass" and a.to_category == category:
                return a.txn_id
        return None

    def cutoff_on(self, category: str) -> str | None:
        for a in self.adjustments:
            if a.kind == "cutoff":
                hit = self.rows[self.rows.txn_id == a.txn_id]
                if len(hit) and hit.iloc[0].category == category:
                    return a.txn_id
        return None

    def decisive_related(self, hits: pd.DataFrame, limit) -> str | None:
        """The single payment whose removal flips the verdict.

        Not the largest and not the last: the one that is individually
        necessary for the breach.
        """
        if limit is None or hits.empty:
            return None
        total = abs(sum(hits.amount, Decimal(0)))
        if total <= limit:
            return None
        decisive = [r.txn_id for r in hits.itertuples()
                    if total - abs(r.amount) <= limit]
        return decisive[0] if len(decisive) == 1 else None


def ebitda(rows) -> Decimal:
    return L.total(rows, "revenue", "in") - L.total(rows, "opex")


def q4_revenue(ctx):
    """Revenue recognised in the fourth quarter only."""
    r = ctx.rows
    q4 = r[(r.date >= "2025-10-01") & (r.date <= "2025-12-31")]
    return L.total(q4, "revenue", "in"), ctx.cutoff_on("revenue")


def insurance_cover(ctx):
    """insurance premiums / (lease + utilities)"""
    r = ctx.rows
    return _safe_div(L.total(r, "insurance"),
                     L.total(r, "lease") + L.total(r, "utilities")), None


def tax_utility_to_ebitda(ctx):
    r = ctx.rows
    return _safe_div(L.total(r, "tax") + L.total(r, "utilities"), ebitda(r)), None


def ebitda_margin(ctx):
    r = ctx.rows
    return _safe_div(ebitda(r), L.total(r, "revenue", "in")), None


def capex_to_ebitda(ctx):
    r = ctx.rows
    return _safe_div(L.total(r, "capex"), ebitda(r)), None


def payroll_obligations(ctx):
    return L.total(ctx.rows, "payroll"), None


def transferred_assets_share(ctx):
    """capital assets moved to unrestricted subsidiaries / total capex"""
    r = ctx.rows
    moved = L.payments_to(r, ctx.unrestricted)
    amt = abs(sum(moved.amount, Decimal(0)))
    ev = moved.iloc[0].txn_id if len(moved) == 1 else None
    return _safe_div(amt, L.total(r, "capex")), ev


# A covenant is dispatched on what it *says*, never on its clause number.
# Clause numbers are cell addresses only: 6.3 is a related-party test for most
# borrowers but a capital-expenditure ceiling for others, and 6.1 takes a
# different shape for every borrower in the set.
DISPATCH = [
    # ratio and structural tests
    (r"capital intensity|капиталоёмкости",             cap_intensity),
    (r"выручк.*за четвёртый|четвёртый.*квартал",       q4_revenue),
    (r"страховых премий|страховое покрытие",           insurance_cover),
    (r"налоговой и коммунальной нагрузки|налогов и коммунальных", tax_utility_to_ebitda),
    (r"рентабельность по ebitda|скорректированной ebitda к выручке", ebitda_margin),
    (r"капитальных затрат группы к ebitda",            capex_to_ebitda),
    (r"обязательства по персоналу",                    payroll_obligations),
    (r"неограниченным дочерним|неограниченных дочерних", transferred_assets_share),
    (r"покрытия процентов|interest cover",             interest_cover),
    (r"cover of applications|покрытия.*источник",      cover_of_applications),
    (r"springing|поступлений по финансированию к",     springing_leverage),
    (r"доля платежей связанным сторонам в операционн", rp_share_of_opex),
    (r"накладн|overhead",                              overhead_ceiling),
    # related-party tests
    (r"proportion of revenue|связанным сторонам.*от выручки|"
     r"аффилированных лиц.*от выручки",                related_party_ratio),
    (r"платежи связанным сторонам|related-party payments|"
     r"аффилированных и связанных сторон",             related_party_abs),
    # category tests
    (r"минимальн.*выручк|minimum revenue|выручк.*не ниже", revenue_min),
    (r"максимальные расходы|maximum.*expense|расходы по категории", category_max),
]


def evaluate(ctx) -> tuple[Decimal, str | None]:
    blob = (ctx.cov.title + " " + ctx.cov.text).lower()
    for pattern, fn in DISPATCH:
        if re.search(pattern, blob):
            return fn(ctx)
    return _fallback(ctx)


def status_of(actual: Decimal, cov, ctx) -> str:
    """COMPLIANT/BREACH from actual vs limit, honouring springing triggers."""
    if cov.limit is None:
        return "COMPLIANT"
    trig = _trigger(cov, ctx)
    if trig is not None and not trig:
        return "COMPLIANT"          # covenant not switched on
    if cov.direction == "min":
        return "BREACH" if actual < cov.limit else "COMPLIANT"
    return "BREACH" if actual > cov.limit else "COMPLIANT"


def _trigger(cov, ctx):
    if "только при условии" not in cov.text:
        return None
    m = re.search(r"\$\s?([0-9][0-9,]*(?:\.\d{2})?)", cov.text)
    if not m:
        return None
    thr = Decimal(m.group(1).replace(",", ""))
    fin = L.total(ctx.rows, "financing", "in")
    return fin > thr
