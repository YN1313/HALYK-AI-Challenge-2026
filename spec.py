"""Covenant specification + deterministic executor.

The point of this module: a covenant is described as *data*, and one executor
runs any covenant so described. That decouples "which formula is this" from
"how do I compute it".

The formula recogniser (regex table in covenants.py) and the LLM formula
parser (llm.py) both emit a Spec. Neither of them ever computes anything —
all arithmetic happens here, in Decimal, deterministically.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal

from . import ledger as L


# A term is one addend of a numerator or denominator.
#   {"category": "capex"}            outflows booked to capex
#   {"category": "revenue", "sign": "in"}   inflows booked to revenue
#   {"special": "ebitda"}            revenue(in) - opex(out)
#   {"special": "related_party"}     outflows to KYC-related counterparties
#   {"special": "largest_overhead"}  max(payroll, utilities), tested singly
Term = dict


@dataclass
class Spec:
    """Executable description of a financial covenant."""
    numerator: list[Term] = field(default_factory=list)
    denominator: list[Term] = field(default_factory=list)   # empty => plain sum
    direction: str = "max"                                   # "max" | "min"
    limit: Decimal | None = None
    limit_kind: str = "money"                                # "money" | "ratio"
    period: str = "full_year"                                # "full_year" | "q4"
    trigger_terms: list[Term] = field(default_factory=list)  # springing covenants
    trigger_above: Decimal | None = None
    trigger_denominator: list[Term] = field(default_factory=list)
    # Dual tests: a breach requires both legs (mode "and") or either leg
    # ("or"). The reported figure stays the primary leg — the covenant's own
    # headline measure — while the second leg only gates the verdict.
    second_numerator: list[Term] = field(default_factory=list)
    second_denominator: list[Term] = field(default_factory=list)
    second_limit: Decimal | None = None
    second_direction: str = "max"
    combine: str | None = None                               # "and" | "or"
    source: str = "rules"                                    # "rules" | "llm"

    def is_ratio(self) -> bool:
        return bool(self.denominator)


PERIODS = {
    "full_year": ("2025-01-01", "2025-12-31"),
    "q1": ("2025-01-01", "2025-03-31"),
    "q2": ("2025-04-01", "2025-06-30"),
    "q3": ("2025-07-01", "2025-09-30"),
    "q4": ("2025-10-01", "2025-12-31"),
}


def _window(rows, period: str):
    lo, hi = PERIODS.get(period, PERIODS["full_year"])
    return rows[(rows.date >= lo) & (rows.date <= hi)]


def _term_value(term: Term, rows, ctx) -> Decimal:
    special = term.get("special")
    if special == "ebitda":
        return L.total(rows, "revenue", "in") - L.total(rows, "opex")
    if special == "ebitda_adjusted":
        return (L.total(rows, "revenue", "in") - L.total(rows, "opex")
                + getattr(ctx, "addbacks", Decimal(0)))
    if special == "related_party":
        hits = L.payments_to(rows, ctx.related)
        return abs(sum(hits.amount, Decimal(0)))
    if special == "unrestricted_transfers":
        hits = L.payments_to(rows, getattr(ctx, "unrestricted", set()))
        return abs(sum(hits.amount, Decimal(0)))
    if special == "largest_overhead":
        return max(L.total(rows, "payroll"), L.total(rows, "utilities"))
    if special == "max_quarter_marketing":
        return _max_quarter(rows, ctx, "marketing")
    if special == "max_quarter_revenue":
        return _max_quarter(rows, ctx, "revenue")

    cat = term.get("category")
    if not cat:
        return Decimal(0)
    sign = term.get("sign") or ("in" if cat in ("revenue", "financing") else "out")
    return L.total(rows, cat, sign)


def _max_quarter(rows, ctx, category: str) -> Decimal:
    """Largest single-quarter total. A quarterly cap is tested per quarter,
    so the reported figure is the worst quarter, not the annual sum."""
    sign = "in" if category == "revenue" else "out"
    best = Decimal(0)
    for q in ("q1", "q2", "q3", "q4"):
        best = max(best, L.total(_window(rows, q), category, sign))
    return best


def _side(terms: list[Term], rows, ctx) -> Decimal:
    """Sum of terms. A term marked negate is subtracted, which lets a covenant
    say 'revenue less the largest overhead line' without a special case."""
    out = Decimal(0)
    for t in terms:
        v = _term_value(t, rows, ctx)
        out += -v if t.get("negate") else v
    return out


def execute(spec: Spec, ctx) -> tuple[Decimal, str | None]:
    """Compute the covenant's actual value and its evidence transaction."""
    rows = _window(ctx.rows, spec.period)

    num = _side(spec.numerator, rows, ctx)
    if spec.is_ratio():
        den = _side(spec.denominator, rows, ctx)
        actual = num / den if den else Decimal(0)
    else:
        actual = num

    return actual, _evidence(spec, rows, ctx, actual)


def triggered(spec: Spec, ctx) -> bool:
    """Springing covenants only bite once their trigger is exceeded."""
    if spec.trigger_above is None or not spec.trigger_terms:
        return True
    rows = _window(ctx.rows, spec.period)
    value = _side(spec.trigger_terms, rows, ctx)
    if spec.trigger_denominator:
        den = _side(spec.trigger_denominator, rows, ctx)
        value = value / den if den else Decimal(0)
    return value > spec.trigger_above


def _second_leg(spec: Spec, ctx) -> bool:
    """Whether the second condition of a dual test is met."""
    if not spec.second_numerator or spec.second_limit is None:
        return True
    rows = _window(ctx.rows, spec.period)
    num = _side(spec.second_numerator, rows, ctx)
    if spec.second_denominator:
        den = _side(spec.second_denominator, rows, ctx)
        num = num / den if den else Decimal(0)
    if spec.second_direction == "min":
        return num < spec.second_limit
    return num > spec.second_limit


def status(spec: Spec, actual: Decimal, ctx) -> str:
    if spec.limit is None:
        return "COMPLIANT"
    if not triggered(spec, ctx):
        return "COMPLIANT"
    if spec.direction == "min":
        first = actual < spec.limit
    else:
        # Forbidding an amount to *exceed* a limit is not breached at equality.
        first = actual > spec.limit

    if spec.combine == "and":
        return "BREACH" if (first and _second_leg(spec, ctx)) else "COMPLIANT"
    if spec.combine == "or":
        return "BREACH" if (first or _second_leg(spec, ctx)) else "COMPLIANT"
    return "BREACH" if first else "COMPLIANT"


# ------------------------------------------------------------------ evidence

def _mentions_related(spec: Spec) -> bool:
    return any(t.get("special") in ("related_party", "unrestricted_transfers")
               for t in spec.numerator)


def _evidence(spec: Spec, rows, ctx, actual: Decimal) -> str | None:
    """The single transaction whose removal flips the verdict.

    Only meaningful where the covenant sums a set of identified payments.
    For ratio and aggregate tests the key holds null, so returning None is
    both correct and free.
    """
    if spec.limit is None or not _mentions_related(spec):
        return None
    if status(spec, actual, ctx) != "BREACH":
        return None

    special = next(t["special"] for t in spec.numerator if t.get("special") in
                   ("related_party", "unrestricted_transfers"))
    names = ctx.related if special == "related_party" else getattr(ctx, "unrestricted", set())
    hits = L.payments_to(rows, names)
    if hits.empty:
        return None

    total = abs(sum(hits.amount, Decimal(0)))
    # limit expressed as a ratio of a denominator -> convert to an amount
    if spec.is_ratio():
        den = _side(spec.denominator, rows, ctx)
        threshold = spec.limit * den
    else:
        threshold = spec.limit

    decisive = [r.txn_id for r in hits.itertuples()
                if total - abs(r.amount) <= threshold]
    return decisive[0] if len(decisive) == 1 else None


# ------------------------------------------------------- parsing LLM output

CATEGORIES = {"revenue", "capex", "opex", "lease", "payroll", "utilities", "transfer",
              "interest", "tax", "insurance", "marketing", "telecom",
              "financing", "related", "other"}
SPECIALS = {"ebitda", "ebitda_adjusted", "related_party",
            "unrestricted_transfers", "largest_overhead",
            "max_quarter_marketing", "max_quarter_revenue"}


def _clean_terms(raw) -> list[Term]:
    out: list[Term] = []
    for t in raw or []:
        if not isinstance(t, dict):
            continue
        if t.get("special") in SPECIALS:
            term = {"special": t["special"]}
            if t.get("negate"):
                term["negate"] = True
            out.append(term)
        elif t.get("category") in CATEGORIES:
            term = {"category": t["category"]}
            if t.get("sign") in ("in", "out"):
                term["sign"] = t["sign"]
            if t.get("negate"):
                term["negate"] = True
            out.append(term)
    return out


def from_json(obj: dict) -> Spec | None:
    """Validate an LLM-produced spec. Anything malformed is rejected, not patched."""
    try:
        num = _clean_terms(obj.get("numerator"))
        if not num:
            return None
        limit = obj.get("limit")
        return Spec(
            numerator=num,
            denominator=_clean_terms(obj.get("denominator")),
            direction="min" if obj.get("direction") == "min" else "max",
            limit=Decimal(str(limit)) if limit is not None else None,
            limit_kind="ratio" if obj.get("limit_kind") == "ratio" else "money",
            period=obj.get("period") if obj.get("period") in PERIODS else "full_year",
            trigger_terms=_clean_terms(obj.get("trigger_terms")),
            trigger_above=(Decimal(str(obj["trigger_above"]))
                           if obj.get("trigger_above") is not None else None),
            source="llm",
        )
    except Exception:
        return None
