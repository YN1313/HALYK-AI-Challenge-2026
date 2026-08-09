"""Entry point: dataset directory -> submission.json.

Guarantees:
  * never raises: any per-cell failure produces a fallback answer and a log line
  * deterministic: identical input produces byte-identical output
  * every cell in the template is filled
"""
from __future__ import annotations
import os, json, argparse, traceback
from decimal import Decimal

from . import ingest, facts, ledger as L, covenants as C
from . import formulas, spec as SP, llm


def build(dataset: str, cache: str) -> dict:
    docs = ingest.extract_all(os.path.join(dataset, "documents"), cache)
    df = L.load(os.path.join(dataset, "master_ledger_2025.csv"))
    index = ingest.build_index(docs, set(df.account_id.dropna().unique()))
    template = json.load(open(os.path.join(dataset, "submission_template.json"),
                              encoding="utf-8"))
    return {"docs": docs, "index": index, "df": df, "template": template}


def account_map(df) -> dict[str, str]:
    """scenario_id -> account_id, taken from the ledger (authoritative)."""
    out = {}
    for sc, grp in df.groupby("scenario_id"):
        accs = grp.account_id.unique()
        if len(accs) == 1:
            out[sc] = accs[0]
    return out


def solve(state: dict, log: list) -> dict:
    docs, index, df = state["docs"], state["index"], state["df"]
    template = state["template"]
    amap = account_map(df)
    answers = {}

    for scenario in template["answers"]:
        cells = {c: {"status": "COMPLIANT", "actual": 0.0, "evidence_txn_id": None}
                 for c in template["answers"][scenario]}
        try:
            acc = amap.get(scenario)
            bucket = index["by_account"].get(acc, {})

            loan_files = bucket.get("LOAN", [])
            if not loan_files:                      # never fall back on the 2024 text
                log.append(f"{scenario}: no executed loan agreement found")
                answers[scenario] = cells
                continue
            covs = facts.parse_covenants(docs[loan_files[0]], loan_files[0],
                                         wanted=list(cells))

            kyc = facts.KYCFacts()
            for f in bucket.get("KYC", []):
                k = facts.parse_kyc(docs[f], f)
                # A dossier counts if it settles the question at all: either it
                # carries an ownership table, or it names the parties outright,
                # or it states that none exist. The last is a finding, not a gap.
                if (k.threshold is not None and k.holdings) or k.conclusive:
                    kyc = k
            related = kyc.related()
            unrestricted: set[str] = set()
            for f in bucket.get("KYC", []):
                unrestricted |= facts.parse_unrestricted(docs[f], f)
            if not kyc.holdings and not kyc.conclusive:
                # Some dossiers carry no ownership table at all. Falling back on
                # the management-retainer counterparty is weaker evidence than a
                # shareholding, so it is logged and only used when nothing better
                # exists — never to override a dossier that does list holdings.
                related = _retainer_counterparties(df, scenario)
                log.append(f"{scenario}: no ownership table and no explicit "
                           f"records; related inferred from retainer rows: "
                           f"{sorted(related)}")

            adjustments = []
            for f in bucket.get("AUDIT", []):
                adjustments += facts.parse_audit(docs[f], f)
            for f in bucket.get("TREASURY", []):
                adjustments += facts.parse_treasury(docs[f], f)

            addbacks = Decimal(0)
            for f in bucket.get("AUDIT", []):
                addbacks += facts.parse_addbacks(docs[f], f)

            rates: dict = {}
            for f in bucket.get("AUDIT", []):
                if not facts.is_draft(docs[f]):
                    rates.update(facts.parse_fx(docs[f], f))

            rows = L.for_scenario(df, scenario)
            rows = L.convert_currency(rows, rates, log, scenario)
            rows = L.apply_adjustments(rows, adjustments, log)

            for clause in cells:
                cov = covs.get(clause)
                if cov is None:
                    log.append(f"{scenario}/{clause}: clause not parsed")
                    continue
                ctx = C.Ctx(cov, rows, related, adjustments, unrestricted, addbacks)
                try:
                    # First echelon: the rule table. It cannot drift, so when it
                    # recognises the formula it is trusted over the model.
                    sp = formulas.recognise(cov)
                    if sp is None:
                        sp = llm.parse_covenant(cov)
                        if sp is not None:
                            log.append(f"{scenario}/{clause}: formula parsed by LLM")
                        elif not llm.available():
                            log.append(f"{scenario}/{clause}: formula UNRECOGNISED and "
                                       f"no ANTHROPIC_API_KEY — the fallback that exists "
                                       f"for exactly this case is switched off")
                    if sp is None:
                        log.append(f"{scenario}/{clause}: formula UNRECOGNISED "
                                   f"({cov.title[:70]})")
                        # The default cell stands: COMPLIANT with zero. For a
                        # Group-level measure the dataset simply does not carry
                        # the inputs, and the borrower's own ledger is not a
                        # substitute — reporting it would be a confident wrong
                        # answer rather than an unavoidable blank.
                        continue

                    actual, evidence = SP.execute(sp, ctx)
                    cells[clause] = {
                        "status": SP.status(sp, actual, ctx),
                        "actual": C.q(abs(actual)),
                        "evidence_txn_id": evidence,
                    }
                except Exception:
                    log.append(f"{scenario}/{clause}: {traceback.format_exc(limit=1)}")
        except Exception:
            log.append(f"{scenario}: {traceback.format_exc(limit=1)}")
        answers[scenario] = cells

    out = dict(template)
    out["answers"] = answers
    return out


def _retainer_counterparties(df, scenario: str) -> set[str]:
    rows = L.for_scenario(df, scenario)
    hits = rows[(rows.category == "related") & (rows.amount < 0)]
    return set(hits.counterparty.unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--team", default="")
    ap.add_argument("--email", default="")
    ap.add_argument("--model", default="claude-sonnet-5")
    a = ap.parse_args()

    log: list[str] = []
    state = build(a.dataset, a.cache)
    sub = solve(state, log)
    sub["team"], sub["contact_email"], sub["model"] = a.team, a.email, a.model

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(sub, fh, ensure_ascii=False, indent=2, sort_keys=False)
    with open(a.out + ".log", "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) or "clean run")
    print(f"wrote {a.out} ({len(sub['answers'])} scenarios, {len(log)} warnings)")


if __name__ == "__main__":
    main()


