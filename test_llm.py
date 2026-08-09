"""Offline tests for everything in the LLM path except the model's judgment.

The transport is replaced by a stub, so these run without an API key and cover
the parts that actually break in production: malformed JSON, hallucinated
category names, missing fields, partial content, and cache behaviour.

    python -m src.test_llm
"""
from __future__ import annotations
import json, os, shutil, tempfile
from decimal import Decimal

from . import llm, spec as SP, formulas, facts


PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'  — ' + detail if detail and not ok else ''}")


# ------------------------------------------------------------ json extraction

def test_json_block():
    print("\n_json_block")
    good = '{"numerator": [{"category": "capex"}]}'
    check("bare json", llm._json_block(good) is not None)
    check("fenced json", llm._json_block("```json\n" + good + "\n```") is not None)
    check("fenced no lang", llm._json_block("```\n" + good + "\n```") is not None)
    check("with preamble",
          llm._json_block("Вот формула:\n" + good) is not None)
    check("trailing prose",
          llm._json_block(good + "\n\nНадеюсь, помог.") is not None)
    check("empty", llm._json_block("") is None)
    check("garbage", llm._json_block("не знаю") is None)
    check("truncated", llm._json_block('{"numerator": [{"cat') is None)


# ----------------------------------------------------------- spec validation

def test_from_json():
    print("\nspec.from_json")
    ok = SP.from_json({"numerator": [{"category": "capex"}],
                       "denominator": [{"category": "opex"}, {"category": "lease"}],
                       "direction": "max", "limit": 0.42, "limit_kind": "ratio"})
    check("well formed", ok is not None and ok.limit == Decimal("0.42"))
    check("source tagged llm", ok is not None and ok.source == "llm")

    check("hallucinated category rejected",
          SP.from_json({"numerator": [{"category": "goodwill_amortisation"}]}) is None)
    check("hallucinated special rejected",
          SP.from_json({"numerator": [{"special": "free_cash_flow"}]}) is None)
    check("empty numerator rejected",
          SP.from_json({"numerator": []}) is None)
    check("missing numerator rejected", SP.from_json({}) is None)
    check("bad limit rejected",
          SP.from_json({"numerator": [{"category": "capex"}],
                        "limit": "около сорока"}) is None)

    mixed = SP.from_json({"numerator": [{"category": "capex"},
                                        {"category": "nonsense"},
                                        {"special": "ebitda"}]})
    check("bad terms dropped, good kept",
          mixed is not None and len(mixed.numerator) == 2)

    bad_dir = SP.from_json({"numerator": [{"category": "capex"}],
                            "direction": "sideways", "limit": 1})
    check("unknown direction defaults to max",
          bad_dir is not None and bad_dir.direction == "max")

    bad_per = SP.from_json({"numerator": [{"category": "revenue"}],
                            "period": "fiscal_h2", "limit": 1})
    check("unknown period defaults to full_year",
          bad_per is not None and bad_per.period == "full_year")

    neg = SP.from_json({"numerator": [{"category": "revenue"},
                                      {"special": "largest_overhead", "negate": True}],
                        "limit": 5000000})
    check("negation preserved",
          neg is not None and neg.numerator[1].get("negate") is True)


# --------------------------------------------------------------- transport

class Stub:
    """Replaces llm._call. Records prompts, returns scripted answers."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, system, user, max_tokens=1200):
        self.calls += 1
        return self.answers.pop(0) if self.answers else None


def test_parse_covenant():
    print("\nparse_covenant against a stub")
    real = llm._call
    cov = facts.Covenant(clause="6.1", title="Maximum Capital Intensity Ratio",
                         text="коэффициент капиталоёмкости не превышал 0.42x")
    try:
        llm._call = Stub('```json\n{"numerator":[{"category":"capex"}],'
                         '"denominator":[{"category":"opex"},{"category":"lease"}],'
                         '"direction":"max","limit":0.42,"limit_kind":"ratio",'
                         '"period":"full_year"}\n```')
        sp = llm.parse_covenant(cov)
        check("fenced answer parsed", sp is not None and sp.limit == Decimal("0.42"))

        llm._call = Stub("Не могу определить формулу.")
        check("prose answer -> None", llm.parse_covenant(cov) is None)

        llm._call = Stub(None)
        check("transport failure -> None", llm.parse_covenant(cov) is None)

        llm._call = Stub('{"numerator":[{"category":"unicorn"}]}')
        check("invalid spec -> None", llm.parse_covenant(cov) is None)

        stub = Stub(*['{"numerator":[{"category":"capex"}],"limit":1}'] * 3)
        llm._call = stub
        out = llm.parse_covenants([cov, cov, cov])
        check("batch aligned to input",
              len(out) == 3 and all(o is not None for o in out) and stub.calls == 3)
    finally:
        llm._call = real


def test_categorise():
    print("\ncategorise against a stub")
    real = llm._call
    try:
        llm._call = Stub('{"Purchase of quayside crane equipment":"capex",'
                         '"Port handling sales settlement":"revenue",'
                         '"Weird thing":"antimatter"}')
        m = llm.categorise(["Purchase of quayside crane equipment",
                            "Port handling sales settlement", "Weird thing"])
        check("valid categories kept", m.get("Purchase of quayside crane equipment") == "capex")
        check("invalid category dropped", "Weird thing" not in m)

        llm._call = Stub(None)
        check("transport failure -> empty", llm.categorise(["x"]) == {})
        check("empty input -> empty", llm.categorise([]) == {})
    finally:
        llm._call = real


def test_cache():
    print("\ncache")
    tmp = tempfile.mkdtemp()
    real_dir, real_key = llm.CACHE_DIR, os.environ.get("ANTHROPIC_API_KEY")
    try:
        llm.CACHE_DIR = tmp
        os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"
        path = llm._cache_path("k")
        os.makedirs(tmp, exist_ok=True)
        json.dump({"text": '{"numerator":[{"category":"capex"}],"limit":7}'},
                  open(path, "w"))
        # a warm cache must not require the network
        got = llm._call("", "")  # key "k" is built from model+system+user, so miss
        check("cache miss without network returns None", got is None)

        # exercise a real hit
        key = f"{llm.MODEL}\x00sys\x00usr"
        json.dump({"text": "cached!"}, open(llm._cache_path(key), "w"))
        check("cache hit served", llm._call("sys", "usr") == "cached!")
    finally:
        llm.CACHE_DIR = real_dir
        if real_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = real_key
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_key_is_silent_but_detectable():
    print("\nheadless behaviour")
    real = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        check("available() False without key", llm.available() is False)
        cov = facts.Covenant(clause="6.1", title="t", text="x")
        check("parse returns None without key", llm.parse_covenant(cov) is None)
    finally:
        if real is not None:
            os.environ["ANTHROPIC_API_KEY"] = real


def test_executor_matches_rules():
    print("\nexecutor accepts a hand-written spec identical to the rule output")
    cov = facts.Covenant(clause="6.1", title="Maximum Capital Intensity Ratio",
                         text="коэффициент капиталоёмкости за период не превышал 0.42x")
    rule = formulas.recognise(cov)
    hand = SP.from_json({"numerator": [{"category": "capex"}],
                         "denominator": [{"category": "opex"}, {"category": "lease"}],
                         "direction": "max", "limit": 0.42, "limit_kind": "ratio",
                         "period": "full_year"})
    same = (rule and hand and rule.limit == hand.limit
            and [t["category"] for t in rule.numerator] ==
                [t["category"] for t in hand.numerator]
            and [t["category"] for t in rule.denominator] ==
                [t["category"] for t in hand.denominator])
    check("rule spec and llm spec are interchangeable", bool(same))


def main():
    for fn in (test_json_block, test_from_json, test_parse_covenant,
               test_categorise, test_cache, test_no_key_is_silent_but_detectable,
               test_executor_matches_rules):
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failing: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
