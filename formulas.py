"""Rule-based formula recogniser.

Maps a covenant's text to a Spec. This is the first echelon: when a pattern
matches, it is trusted over the LLM parser because it cannot drift.
Anything it does not recognise falls through to llm.parse_covenant.
"""
from __future__ import annotations
import re
from decimal import Decimal

from .spec import Spec


C = lambda cat, sign=None: ({"category": cat, "sign": sign} if sign
                            else {"category": cat})
S = lambda name: {"special": name}

REVENUE = C("revenue", "in")
FINANCING = C("financing", "in")
EBITDA = S("ebitda")
RP = S("related_party")


# (pattern, numerator, denominator, period)
EBITDA_ADJ = S("ebitda_adjusted")

RULES: list[tuple[str, list, list, str]] = [
    # --- leverage ----------------------------------------------------------
    # Total debt in this dataset is the financing drawn during the period, and
    # EBITDA carries the auditor's one-off add-backs. Both are stated in the
    # covenant text itself rather than left to accounting convention.
    (r"коэффициент\w* долгов\w* нагрузк|долговая нагрузка|leverage ratio|"
     r"отношение совокупного долга",
     [FINANCING], [EBITDA_ADJ], "full_year"),

    # --- category and asset limits ----------------------------------------
    # "Advisory services" here is a spend category defined by the covenant
    # itself, not the related-party channel: the text defines it as all sums
    # paid for consulting and advisory work, whoever the counterparty is.
    (r"консультационных услуг к ebitda|advisory.*to ebitda|"
     r"консультационн\w* услуг\w*,? оплаченн",
     [C("related")], [EBITDA], "full_year"),

    # A springing rental cap: the ceiling on lease payments only applies once
    # payroll passes a share of revenue.
    (r"springing property rental cap|ограничение совокупных арендных платежей",
     [C("lease")], [], "full_year"),

    # Disposal of assets, measured as an absolute amount rather than a share.
    (r"запрет передачи активов|не отчуждать какие-либо существенные активы|"
     r"disposal of material assets",
     [C("transfer")], [], "full_year"),

    # --- margins and burden ratios ----------------------------------------
    # "Operating margin after personnel" is stated as the excess of revenue
    # over payroll and tax, expressed as a proportion of revenue.
    (r"post-personnel operating margin|маржа после расходов на персонал",
     [REVENUE, {"category": "payroll", "negate": True},
      {"category": "tax", "negate": True}], [REVENUE], "full_year"),

    (r"fiscal burden|налоги и уплаченные проценты",
     [C("tax"), C("interest")], [REVENUE], "full_year"),

    (r"retained financing proceeds|поступления.*по финансированию.*уменьшенные",
     [FINANCING, {"category": "interest", "negate": True},
      {"category": "tax", "negate": True}], [], "full_year"),

    (r"liquidity contribution|вклад в ликвидность",
     [EBITDA, FINANCING], [], "full_year"),

    (r"fixed charge cover|покрытия постоянных платежей",
     [EBITDA, C("lease")], [C("interest"), C("lease")], "full_year"),

    (r"консультационных услуг к ebitda|advisory.*to ebitda",
     [RP], [EBITDA], "full_year"),

    (r"financing and occupancy|расходы на финансирование и содержание",
     [C("interest"), C("lease")], [], "full_year"),

    (r"квартальн\w* лимит маркетинг|quarterly marketing",
     [S("max_quarter_marketing")], [], "full_year"),

    (r"quarterly revenue concentration|квартальн\w* концентрац\w* выручк",
     [S("max_quarter_revenue")], [REVENUE], "full_year"),

    # Two covenants share the name "capital intensity" with different
    # definitions, so the match is on the formula stated in the text rather
    # than on the title: capex over revenue here, capex over operating costs
    # and lease payments below.
    (r"капитальных затрат к выручке|капитальные затраты.*к выручке|"
     r"capital expenditure limitation \(percentage of revenue\)|"
     r"не превышали \d+ процентов совокупной выручки",
     [C("capex")], [REVENUE], "full_year"),

    (r"капитальных затрат к ebitda|capital expenditure.*to ebitda",
     [C("capex")], [EBITDA], "full_year"),

    (r"долговой корзин|permitted debt basket",
     [FINANCING], [], "full_year"),

    # --- ratio tests -------------------------------------------------------
    (r"capital intensity|капиталоёмкости",
     [C("capex")], [C("opex"), C("lease")], "full_year"),

    (r"покрытия процентов|interest cover",
     [EBITDA], [C("interest")], "full_year"),

    (r"cover of applications|покрытия.*источник",
     [REVENUE, FINANCING], [C("opex"), C("capex")], "full_year"),

    (r"springing|поступлений по финансированию к ebitda",
     [FINANCING], [EBITDA], "full_year"),

    (r"доля платежей связанным сторонам в операционн",
     [RP], [C("opex")], "full_year"),

    (r"страховых премий|страховое покрытие расходов",
     [C("insurance")], [C("lease"), C("utilities")], "full_year"),

    (r"налоговой и коммунальной нагрузки|налогов и коммунальных",
     [C("tax"), C("utilities")], [EBITDA], "full_year"),

    (r"рентабельность по ebitda|скорректированной ebitda к выручке",
     [EBITDA_ADJ], [REVENUE], "full_year"),

    (r"капитальных затрат группы к ebitda",
     [C("capex")], [EBITDA], "full_year"),

    (r"неограниченным дочерним|неограниченных дочерних",
     [S("unrestricted_transfers")], [C("capex")], "full_year"),

    (r"proportion of revenue|аффилированных лиц.*от выручки|"
     r"связанным сторонам.*от выручки",
     [RP], [REVENUE], "full_year"),

    (r"покрытия расходов на персонал и коммунальн|покрытие расходов на персонал",
     [REVENUE], [C("payroll"), C("utilities")], "full_year"),

    # --- sum tests ---------------------------------------------------------
    (r"выручк.*за вычетом наибольшей статьи накладных",
     [REVENUE, {"special": "largest_overhead", "negate": True}], [], "full_year"),

    (r"выручк.*за четвёртый|четвёртый.*квартал", [REVENUE], [], "q4"),

    (r"обязательства по персоналу",       [C("payroll")], [], "full_year"),

    (r"накладн|overhead",                 [S("largest_overhead")], [], "full_year"),

    (r"платежи связанным сторонам|related-party payments|"
     r"аффилированных и связанных сторон", [RP], [], "full_year"),

    (r"минимальн.*выручк|minimum revenue|выручк.*не ниже",
     [REVENUE], [], "full_year"),
]

RULES_C = [(re.compile(p, re.I), n, d, per) for p, n, d, per in RULES]

CATEGORY_WORDS = [
    ("капитальные затраты", "capex"), ("операционные расходы", "opex"),
    ("расходы на персонал", "payroll"), ("оплату труда", "payroll"),
    ("коммунальн", "utilities"), ("маркетинг", "marketing"),
    ("страхован", "insurance"), ("аренд", "lease"), ("выручк", "revenue"),
]

MONEY = re.compile(r"\$\s?([0-9][0-9,]*(?:\.\d{1,2})?)")
RATIO = re.compile(r"([0-9]+(?:\.\d+)?)\s?x")
MIN_WORDS = ("не менее", "не ниже", "минимальн", "minimum", "не допускать снижения")


SPECIAL_SHAPES = [
    # Group-level covenants are measured on consolidated statements, not on the
    # borrower's own ledger — the text says so explicitly. Without those
    # statements the measure is unavailable, and reporting the borrower's own
    # capex in its place would be a confident wrong answer.
    (r"капитальн\w* затрат\w* группы|группы за пределами заёмщика|"
     r"капитальных затрат на уровне группы|"
     r"совокупными капитальными затратами консолидированной группы", "group_capex"),
    (r"двойн\w* поддерживающ\w* тест|двойное условие дефолта|"
     r"одновременно.*оба", "dual_and"),
    (r"досрочного погашения за счёт|любого из следующих обстоятельств", "dual_or"),
    (r"insurance cover linked to capital|страховые премии.*если совокупные "
     r"капитальные затраты", "insurance_if_capex"),
    (r"rental cap with insurance proviso|аренд\w* платеж\w*.*не влечёт", "rent_unless_insurance"),
]


def _dual(cov, blob, mode: str) -> Spec | None:
    """Two-legged tests: leverage against interest cover, or leverage against
    a capital-expenditure ceiling. The first ratio in the text is the headline
    measure; the second gates the verdict."""
    nums = [Decimal(x) for x in re.findall(r"([0-9]+(?:\.\d+)?)\s?x", cov.text)]
    money = [Decimal(x.replace(",", "")) for x in MONEY.findall(cov.text)]
    if not nums:
        return None
    first, second = nums[0], (nums[1] if len(nums) > 1 else None)

    sp = Spec(numerator=[FINANCING], denominator=[EBITDA],
              direction="max", limit=first, limit_kind="ratio",
              combine=mode, source="rules")

    if second is not None and re.search(r"ebitda к процентн|покрытия обслуживания|"
                                        r"ebitda к процент", blob):
        sp.second_numerator, sp.second_denominator = [EBITDA], [C("interest")]
        sp.second_limit, sp.second_direction = second, "min"
    elif money:
        sp.second_numerator, sp.second_limit = [C("capex")], money[0]
        sp.second_direction = "max"
    elif second is not None:
        sp.second_numerator, sp.second_denominator = [EBITDA], [C("interest")]
        sp.second_limit, sp.second_direction = second, "min"
    return sp


def _conditional_insurance(cov, blob) -> Spec | None:
    """Insurance floor that only applies once capex passes a trigger."""
    money = [Decimal(x.replace(",", "")) for x in MONEY.findall(cov.text)]
    if len(money) < 2:
        return None
    trigger, floor = money[0], money[1]
    return Spec(numerator=[C("insurance")], direction="min", limit=floor,
                limit_kind="money", trigger_terms=[C("capex")],
                trigger_above=trigger, source="rules")


def _rent_unless_insurance(cov, blob) -> Spec | None:
    """Rent ceiling with a carve-out: breached only if insurance also falls
    short, so the two legs must both hold."""
    money = [Decimal(x.replace(",", "")) for x in MONEY.findall(cov.text)]
    if len(money) < 2:
        return None
    sp = Spec(numerator=[C("lease")], direction="max", limit=money[0],
              limit_kind="money", combine="and", source="rules")
    sp.second_numerator, sp.second_limit = [C("insurance")], money[1]
    sp.second_direction = "min"
    return sp


def recognise(cov) -> Spec | None:
    blob = f"{cov.title} {cov.text}".lower()

    for pattern, shape in SPECIAL_SHAPES:
        if not re.search(pattern, blob):
            continue
        if shape == "group_capex":
            return None                      # measured outside the ledger
        if shape == "dual_and":
            return _dual(cov, blob, "and")
        if shape == "dual_or":
            return _dual(cov, blob, "or")
        if shape == "insurance_if_capex":
            return _conditional_insurance(cov, blob)
        if shape == "rent_unless_insurance":
            return _rent_unless_insurance(cov, blob)


    numerator = denominator = None
    period = "full_year"
    for rx, num, den, per in RULES_C:
        if rx.search(blob):
            numerator, denominator, period = num, den, per
            break

    if numerator is None:
        # "Максимальные расходы по категории" — the category is named in the text
        if re.search(r"максимальные расходы|maximum.*expense|расходы по категории", blob):
            cat = next((c for w, c in CATEGORY_WORDS if w in blob), None)
            if not cat:
                return None
            numerator, denominator = [C(cat)], []
        else:
            return None

    spec = Spec(
        numerator=numerator,
        denominator=denominator,
        period=period,
        direction="min" if any(w in blob for w in MIN_WORDS) else "max",
        **_threshold(cov.text),
        trigger_terms=[FINANCING] if "только при условии" in cov.text else [],
        trigger_above=_trigger(cov.text),
        source="rules",
    )

    # Some conditions are stated as a share of revenue rather than an amount
    # ("применяется только при условии, что расходы на оплату труда превышают
    # 30.0% of Revenue"), so the trigger needs a denominator of its own.
    pct = re.search(r"только при условии[^.]*?расход\w* на оплату труда[^.]*?"
                    r"(\d+(?:\.\d+)?)\s?%", cov.text)
    if pct:
        spec.trigger_terms = [C("payroll")]
        spec.trigger_denominator = [REVENUE]
        spec.trigger_above = Decimal(pct.group(1)) / Decimal(100)
        spec.limit, spec.limit_kind = _money_limit(cov.text), "money"
    return spec


def _money_limit(text: str) -> Decimal | None:
    m = MONEY.search(text)
    return Decimal(m.group(1).replace(",", "")) if m else None


def _threshold(text: str) -> dict:
    """The covenant limit. A ratio, when present, is always the limit;
    any money figure alongside it is the springing trigger."""
    r = RATIO.search(text)
    if r:
        return {"limit": Decimal(r.group(1)), "limit_kind": "ratio"}
    m = MONEY.search(text)
    if m:
        return {"limit": Decimal(m.group(1).replace(",", "")), "limit_kind": "money"}
    return {"limit": None, "limit_kind": "money"}


def _trigger(text: str) -> Decimal | None:
    if "только при условии" not in text:
        return None
    m = MONEY.search(text)
    return Decimal(m.group(1).replace(",", "")) if m else None
