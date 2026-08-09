"""Replica of the official rubric, for local iteration against ground_truth.json.

status 0.50 | actual 0.30 (linear decay to 0 at 5% relative error)
evidence 0.20 — exact match when the key is non-null; when the key is null the
0.20 decays with `actual` on the same scale.
"""
from __future__ import annotations
import json, sys


def cell_score(got: dict, key: dict) -> tuple[float, str]:
    if not isinstance(got, dict):
        return 0.0, "malformed"
    if got.get("status") not in ("COMPLIANT", "BREACH"):
        return 0.0, "bad status"
    if got["status"] != key["status"]:
        return 0.0, f"status {got['status']} != {key['status']}"

    s = 0.50
    a, ka = got.get("actual"), key["actual"]
    if isinstance(a, (int, float)) and ka:
        e = abs(a - ka) / abs(ka)
        frac = max(0.0, 1 - e / 0.05)
    else:
        e, frac = 1.0, 0.0
    s += 0.30 * frac

    if key["evidence_txn_id"] is None:
        s += 0.20 * frac
        note = f"actual err {e:.3%}"
    else:
        ok = got.get("evidence_txn_id") == key["evidence_txn_id"]
        s += 0.20 if ok else 0.0
        note = f"actual err {e:.3%}; evidence {'ok' if ok else 'MISS ' + str(key['evidence_txn_id'])}"
    return s, note


def main(sub_path: str, gt_path: str):
    sub = json.load(open(sub_path, encoding="utf-8"))
    gt = json.load(open(gt_path, encoding="utf-8"))["scenarios"]

    total, n = 0.0, 0
    rows = []
    for sc, block in gt.items():
        for clause, key in block["covenants"].items():
            got = sub.get("answers", {}).get(sc, {}).get(clause, {})
            s, note = cell_score(got, key)
            total += s
            n += 1
            rows.append((sc, clause, s, got.get("status"), key["status"],
                         got.get("actual"), key["actual"], note))

    rows.sort(key=lambda r: r[2])
    print(f"{'cell':10} {'score':>5}  {'got':<10} {'key':<10} "
          f"{'actual':>15} {'expected':>15}  note")
    for sc, cl, s, gs, ks, ga, ka, note in rows:
        print(f"{sc + '/' + cl:10} {s:5.2f}  {str(gs):<10} {ks:<10} "
              f"{str(ga):>15} {ka:>15}  {note}")
    print(f"\nTOTAL {total:.2f} / {n} = {total / n:.1%}")
    ok = sum(1 for r in rows if r[3] == r[4])
    print(f"status accuracy {ok}/{n} = {ok / n:.1%}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
