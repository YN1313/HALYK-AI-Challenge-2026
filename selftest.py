"""Verification harness for the LLM formula parser.

Question it answers: if the rule table did not exist, would the model recover
the same covenant formulas? That is exactly the situation on an unseen dataset,
where the rules will miss shapes they have never encountered.

    export ANTHROPIC_API_KEY=...
    python -m src.selftest --dataset <открытый-датасет>

For every covenant it parses the formula twice — once with rules, once with the
model — and compares both the resulting Spec and the number that Spec produces.
Exits non-zero if agreement falls below the threshold, so it can gate a commit.
"""
from __future__ import annotations
import argparse, sys
from decimal import Decimal

from . import run as R, facts, formulas, llm, covenants as C, ledger as L
from . import spec as SP


def _shape(sp: SP.Spec | None) -> tuple:
    """Comparable fingerprint of a Spec. Term order is irrelevant."""
    if sp is None:
        return ()
    key = lambda ts: tuple(sorted(
        (t.get("category") or t.get("special"), bool(t.get("negate")),
         t.get("sign") or "")
        for t in ts))
    return (key(sp.numerator), key(sp.denominator), sp.direction,
            sp.limit, sp.limit_kind, sp.period)


def _fmt(sp: SP.Spec | None) -> str:
    if sp is None:
        return "—"
    f = lambda ts: "+".join((t.get("category") or t.get("special")) +
                            ("(-)" if t.get("negate") else "") for t in ts) or "·"
    body = f(sp.numerator) + (" / " + f(sp.denominator) if sp.denominator else "")
    return f"{sp.direction} {sp.limit} {sp.limit_kind} [{sp.period}] {body}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--min-agreement", type=float, default=0.80)
    a = ap.parse_args()

    if not llm.available():
        print("ANTHROPIC_API_KEY is not set — the LLM path cannot be verified.\n"
              "Without it the pipeline silently runs on rules alone, which is\n"
              "exactly the failure this harness exists to catch.")
        return 2

    state = R.build(a.dataset, a.cache)
    amap = R.account_map(state["df"])

    jobs = []   # (scenario, clause, cov)
    for scenario, acc in sorted(amap.items()):
        bucket = state["index"]["by_account"].get(acc, {})
        for f in bucket.get("LOAN", [])[:1]:
            for clause, cov in sorted(facts.parse_covenants(state["docs"][f], f).items()):
                jobs.append((scenario, clause, cov))

    print(f"parsing {len(jobs)} covenants with {llm.MODEL} …\n")
    parsed = llm.parse_covenants([c for _, _, c in jobs])
    assert len(parsed) == len(jobs), "parser lost covenants"

    agree = differ = failed = 0
    numeric_agree = numeric_total = 0
    rows = []

    for (scenario, clause, cov), llm_spec in zip(jobs, parsed):
        rule_spec = formulas.recognise(cov)

        if llm_spec is None:
            failed += 1
            verdict = "NO-PARSE"
        elif _shape(rule_spec) == _shape(llm_spec):
            agree += 1
            verdict = "agree"
        else:
            differ += 1
            verdict = "DIFFER"

        # Does the disagreement actually change the answer?
        note = ""
        if rule_spec and llm_spec:
            ctx = _ctx(state, amap, scenario, cov)
            if ctx is not None:
                numeric_total += 1
                ra, _ = SP.execute(rule_spec, ctx)
                la, _ = SP.execute(llm_spec, ctx)
                rs = SP.status(rule_spec, ra, ctx)
                ls = SP.status(llm_spec, la, ctx)
                same = rs == ls and _close(ra, la)
                numeric_agree += same
                if not same:
                    note = f"  rules={rs} {C.q(abs(ra))} | llm={ls} {C.q(abs(la))}"

        rows.append((f"{scenario}/{clause}", verdict, _fmt(rule_spec),
                     _fmt(llm_spec), note))

    for cell, verdict, r, l, note in rows:
        if verdict == "agree":
            continue
        print(f"{cell:9} {verdict}\n    rules: {r}\n    llm:   {l}{note}")

    n = len(jobs)
    rate = agree / n if n else 0
    print(f"\nformula shape:  {agree} agree, {differ} differ, {failed} no-parse "
          f"({rate:.0%} agreement)")
    if numeric_total:
        print(f"same answer:    {numeric_agree}/{numeric_total} "
              f"({numeric_agree / numeric_total:.0%})")

    if rate < a.min_agreement:
        print(f"\nBelow the {a.min_agreement:.0%} threshold. The LLM fallback is not "
              f"trustworthy as written — tighten the prompt in llm.py before relying "
              f"on it for unseen covenants.")
        return 1
    print("\nFallback verified: on unseen covenant shapes the model is likely to "
          "recover a usable formula.")
    return 0


def _close(a: Decimal, b: Decimal) -> bool:
    if a == b:
        return True
    if not a:
        return False
    return abs(abs(a) - abs(b)) / abs(a) < Decimal("0.005")


def _ctx(state, amap, scenario, cov):
    acc = amap.get(scenario)
    bucket = state["index"]["by_account"].get(acc, {})
    kyc = facts.KYCFacts()
    for f in bucket.get("KYC", []):
        k = facts.parse_kyc(state["docs"][f], f)
        if k.threshold is not None and k.holdings:
            kyc = k
    adj = []
    for f in bucket.get("AUDIT", []):
        adj += facts.parse_audit(state["docs"][f], f)
    rows = L.for_scenario(state["df"], scenario)
    if rows.empty:
        return None
    rows = L.apply_adjustments(rows, adj)
    return C.Ctx(cov, rows, kyc.related(), adj)


if __name__ == "__main__":
    sys.exit(main())
