"""Structural check of submission.json against the template. Run before sending."""
from __future__ import annotations
import json, sys


def main(sub_path: str, tpl_path: str) -> int:
    sub = json.load(open(sub_path, encoding="utf-8"))
    tpl = json.load(open(tpl_path, encoding="utf-8"))
    errs: list[str] = []

    for field in ("team", "contact_email", "model"):
        if not sub.get(field):
            errs.append(f"top-level field '{field}' is empty")

    want, got = tpl["answers"], sub.get("answers", {})
    if set(want) != set(got):
        errs.append(f"scenario keys differ: missing {set(want) - set(got)}, "
                    f"extra {set(got) - set(want)}")

    for sc in want:
        for cl in want[sc]:
            cell = got.get(sc, {}).get(cl)
            if cell is None:
                errs.append(f"{sc}/{cl}: missing")
                continue
            if cell.get("status") not in ("COMPLIANT", "BREACH"):
                errs.append(f"{sc}/{cl}: status {cell.get('status')!r} invalid")
            a = cell.get("actual")
            if not isinstance(a, (int, float)) or isinstance(a, bool):
                errs.append(f"{sc}/{cl}: actual {a!r} is not a number")
            elif a < 0:
                errs.append(f"{sc}/{cl}: actual {a} is negative")
            ev = cell.get("evidence_txn_id")
            if ev is not None and not str(ev).startswith("TXN-"):
                errs.append(f"{sc}/{cl}: evidence {ev!r} malformed")

    if errs:
        print("\n".join(errs))
        print(f"\n{len(errs)} problem(s) — DO NOT SUBMIT")
        return 1
    n = sum(len(v) for v in want.values())
    print(f"OK: {len(want)} scenarios, {n} cells, all filled and well typed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
