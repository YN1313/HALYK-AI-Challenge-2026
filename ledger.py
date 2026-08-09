"""Stage 5: categorise ledger rows.

The ledger has no category column. Category is carried by the *description*,
never by the counterparty name — counterparties are randomised decoration for
every row except genuine trading and related-party counterparties.

Ordering matters: the first pattern that matches wins, so specific
scenario-level items are tested before the generic decoy families.
"""
from __future__ import annotations
import re
from decimal import Decimal

import pandas as pd


# Categories are carried by the description, and the wording varies widely
# across borrowers: "sales settlement", "bulk cement sales" and "wholesale
# interconnect revenue" are all revenue. Patterns therefore match on the
# substance of the operation rather than on a fixed phrase.
#
# Order matters — the first match wins, so the discriminating cases are tested
# before the broad families:
#   * a transfer of assets to a subsidiary is capital expenditure, but it must
#     be caught before the generic equipment pattern to stay identifiable
#   * repair, refurbishment and maintenance restore an existing asset and are
#     operating costs, never capital expenditure
#   * advisory and management retainers are the related-party channel
SPECIFIC = [
    ("transfer",  r"transfer of .*(?:equipment|assets|machinery|tooling)|"
                  r"(?:equipment|machinery|assets) transfer to (?:group|subsidiar)|"
                  r"intra-group transfer of"),

    ("opex",      r"maintenance|servicing|refurbish|repair|decommission|"
                  r"operating and maintenance|operating costs|"
                  r"cleaning|clearance works|dredging|remediation"),

    # Capital expenditure is an *acquisition*. "Equipment yard lease" and
    # "machinery insurance" name equipment without buying any, so the pattern
    # anchors on the acquiring verb or noun rather than on the asset word.
    ("capex",     r"purchase of |\bpurchase\b|acquisition of |"
                  r"(?:equipment|machinery|plant)\s+(?:purchase|instalment|upgrade|"
                  r"installation)|"
                  r"(?:machinery|equipment)\s+(?:instalment|upgrade)|"
                  r"expansion works|imported .*(?:equipment|machinery)|"
                  r"\w+\s+(?:machinery|equipment)\s+(?:instalment|installation|upgrade|purchase)|"
                  r"aeration equipment|refurbishment programme"),

    ("revenue",   r"sales settlement|revenue settlement|\brevenue\b|\bsales\b|"
                  r"subscription revenue|interconnect revenue|throughput sales"),

    ("financing", r"facility drawdown|loan drawdown|drawdown|"
                  r"promissory note proceeds|note proceeds|bond proceeds|"
                  r"principal repayment"),

    ("related",   r"advisory|retainer|consulting|management fees|"
                  r"intercompany (?:advisory|distribution)|distribution settlement"),

    ("lease",     r"land lease payments"),
]

# Generic decoy families.
GENERIC = [
    ("interest",  r"\binterest\b|coupon|overdraft"),
    ("tax",       r"\btax\b|vat|excise|customs duty|franchise tax|social tax|levy"),
    ("payroll",   r"payroll|staff|crew|wages"),
    ("insurance", r"insurance|indemnity|fidelity bond|insurance broker"),
    ("marketing", r"marketing|media buy|ad campaign|sponsorship|exhibition stand|"
                  r"advertis|collateral run|newsletter"),
    ("utilities", r"electricity|water|gas utility|natural gas|heating|waste|"
                  r"compressed air|utility|metering"),
    ("telecom",   r"telecom|leased line|mobile fleet"),
    ("lease",     r"\blease\b|\brent\b|rental"),
]

PATTERNS = [(c, re.compile(p, re.I)) for c, p in SPECIFIC + GENERIC]


def categorise(description: str) -> str:
    for cat, rx in PATTERNS:
        if rx.search(description):
            return cat
    return "other"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["scenario_id"] = df.txn_id.str.split("-").str[1]
    # A blank amount is unknown, not zero. Decimal("NaN") would silently
    # poison every sum it touches, so unknown rows are flagged and excluded
    # until a treasury memo supplies the real figure.
    df["amount_missing"] = df.amount.isna()
    df["amount"] = df.amount.map(
        lambda x: Decimal("0") if pd.isna(x) else Decimal(str(x)))
    df["category"] = df.description.map(categorise)
    df["date"] = pd.to_datetime(df.date)
    return df


def for_scenario(df: pd.DataFrame, scenario_id: str) -> pd.DataFrame:
    """Rows belonging to one borrower.

    Scenario id is taken from the txn_id prefix, which is authoritative.
    Numeric prefixes (TXN-9001-...) are unrelated noise accounts.
    """
    return df[df.scenario_id == scenario_id].copy()


def convert_currency(rows: pd.DataFrame, rates: dict, log=None,
                     tag: str = "") -> pd.DataFrame:
    """Translate non-USD rows at the rate the auditor disclosed.

    Covenant limits are all in dollars, so a euro row left untranslated is
    understated by whatever the rate exceeds one — small per row, decisive in
    a ratio. A row whose currency has no disclosed rate is left alone and
    logged rather than converted at a guessed rate.
    """
    rows = rows.copy()
    rows["fx_applied"] = None
    for code in rows.currency.unique():
        if code == "USD":
            continue
        mask = rows.currency == code
        rate = rates.get(code)
        if rate is None:
            if log is not None:
                total = abs(sum(rows.loc[mask, "amount"], Decimal(0)))
                log.append(f"{tag}: {int(mask.sum())} row(s) in {code} "
                           f"({total:,.2f}) left untranslated — no rate disclosed "
                           f"by the auditor")
            continue
        rows.loc[mask, "amount"] = rows.loc[mask, "amount"].map(lambda a: a * rate)
        rows.loc[mask, "fx_applied"] = f"{code}@{rate:.6f}"
    return rows


def apply_adjustments(rows: pd.DataFrame, adjustments, log=None) -> pd.DataFrame:
    """Apply auditor instructions to the period's rows.

    Cutoff removes a row from the covenant period; reclass moves it to another
    category. A row is matched by transaction id when the auditor gave one, and
    otherwise by amount — the counterparty name is used to disambiguate, never
    on its own, because amounts are unique in practice and names are not.
    """
    rows = rows.copy()
    rows["excluded"] = False
    rows["adjusted_by"] = None

    for adj in adjustments:
        mask = _match(rows, adj)
        if mask is None or not mask.any():
            if log is not None:
                log.append(f"adjustment unmatched ({adj.doc}): {adj.note[:90]}")
            continue
        if mask.sum() > 1:
            if log is not None:
                log.append(f"adjustment ambiguous ({adj.doc}): {adj.note[:90]}")
            continue
        if adj.kind == "amount" and adj.new_amount is not None:
            rows.loc[mask, "amount"] = adj.new_amount
            rows.loc[mask, "amount_missing"] = False
        elif adj.kind == "cutoff":
            rows.loc[mask, "excluded"] = True
        elif adj.kind == "reclass" and adj.to_category:
            rows.loc[mask, "category"] = adj.to_category
        else:
            continue
        rows.loc[mask, "adjusted_by"] = adj.doc
    return rows


def _match(rows: pd.DataFrame, adj):
    if adj.txn_id:
        return rows.txn_id == adj.txn_id
    if adj.amount is None:
        return None
    m = rows.amount.map(lambda a: abs(abs(a) - adj.amount) < Decimal("0.005"))
    if adj.counterparty and m.sum() > 1:
        key = normalise(adj.counterparty)
        m = m & rows.counterparty.map(lambda c: normalise(c) == key)
    return m


# ------------------------------------------------------------------ measures

def total(rows: pd.DataFrame, category: str, sign: str = "out") -> Decimal:
    """Absolute total of a category. sign='out' = payments, 'in' = receipts."""
    sub = rows[(~rows.excluded) & (~rows.amount_missing) & (rows.category == category)]
    sub = sub[sub.amount < 0] if sign == "out" else sub[sub.amount > 0]
    return abs(sum(sub.amount, Decimal(0)))


def payments_to(rows: pd.DataFrame, names: set[str]) -> pd.DataFrame:
    """Outflows to any of the given counterparties, matched on normalised name."""
    if not names:
        return rows.iloc[0:0]
    keys = {normalise(n) for n in names}
    m = rows.counterparty.map(lambda c: normalise(c) in keys)
    return rows[m & (~rows.excluded) & (~rows.amount_missing) & (rows.amount < 0)]


SUFFIX = re.compile(r"\b(llp|jsc|llc|ltd|ag|inc|corp|corporation|company|co|"
                    r"group|holding|holdings|partners|partner|enterprise|"
                    r"trading house|supply|lp|plc)\b", re.I)
PARENS = re.compile(r"\(.*?\)")


def normalise(name: str) -> str:
    """Canonical key for entity matching.

    Periods are removed before tokenising so that 'L.L.P.' and 'LLP' collapse
    to the same legal-form token; the branch qualifier in parentheses is
    dropped because it identifies a site, not a counterparty.
    """
    s = PARENS.sub(" ", str(name))
    s = s.replace(".", "").replace(",", " ").lower()
    s = SUFFIX.sub(" ", s)
    return " ".join(s.split())
