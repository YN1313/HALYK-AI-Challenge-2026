"""LLM layer: parses covenant *formulas* and ledger *categories*.

The model never computes. It converts prose into a machine-checkable
description that spec.execute runs deterministically. Every answer is cached
on disk by content hash, so a rerun costs nothing and produces byte-identical
output.

Runs headless: with no API key available the module returns None for
everything and the pipeline falls back to its rule tables.
"""
from __future__ import annotations
import os, re, json, time, hashlib
import urllib.request
import concurrent.futures as cf

from . import spec as S

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("COVENANT_MODEL", "claude-sonnet-5")
CACHE_DIR = os.environ.get("COVENANT_LLM_CACHE", "cache/llm")


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ------------------------------------------------------------------ transport

def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, hashlib.md5(key.encode()).hexdigest() + ".json")


def _call(system: str, user: str, max_tokens: int = 1200) -> str | None:
    key = f"{MODEL}\x00{system}\x00{user}"
    cp = _cache_path(key)
    if os.path.exists(cp):
        return json.load(open(cp, encoding="utf-8"))["text"]
    if not available():
        return None

    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()

    for attempt in range(4):
        try:
            req = urllib.request.Request(
                API_URL, data=body,
                headers={"content-type": "application/json",
                         "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                         "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.load(resp)
            text = "".join(b.get("text", "") for b in data.get("content", []))
            os.makedirs(CACHE_DIR, exist_ok=True)
            json.dump({"text": text}, open(cp, "w", encoding="utf-8"))
            return text
        except Exception as exc:
            if attempt == 3:
                print(f"[warn] llm call failed: {exc}")
                return None
            time.sleep(2 ** attempt)
    return None


def _json_block(text: str):
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"[{\[].*[}\]]", cleaned, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except Exception:
            return None


# ----------------------------------------------------------- covenant parsing

COVENANT_SYSTEM = """Ты разбираешь текст финансового ковенанта из кредитного договора \
и возвращаешь его формулу в виде JSON. Ты ничего не вычисляешь.

Верни ТОЛЬКО JSON, без пояснений и без markdown-ограждения:

{
  "numerator":     [<term>, ...],
  "denominator":   [<term>, ...],
  "direction":     "max" | "min",
  "limit":         <число>,
  "limit_kind":    "ratio" | "money",
  "period":        "full_year" | "q1" | "q2" | "q3" | "q4",
  "trigger_terms": [<term>, ...],
  "trigger_above": <число или null>
}

<term> — это {"category": "<имя>"} либо {"special": "<имя>"}.

category: revenue, capex, opex, lease, payroll, utilities, interest, tax,
          insurance, marketing, telecom, financing
special:  ebitda                  — выручка минус операционные расходы
          related_party           — платежи связанным сторонам по досье KYC
          unrestricted_transfers  — активы, переданные неограниченным дочерним
          largest_overhead        — наибольшая из статей: персонал / коммунальные

Правила:
- denominator пустой, если ковенант ограничивает сумму, а не отношение.
- direction "min" — если требуется «не менее / не ниже»; иначе "max".
- limit — порог самого ковенанта. Если в тексте есть и коэффициент, и сумма,
  коэффициент почти всегда порог, а сумма — условие срабатывания
  (trigger_above), которое ставится только для условных ковенантов.
- Числитель и знаменатель — списки слагаемых; они складываются."""


def parse_covenant(cov) -> S.Spec | None:
    raw = _call(COVENANT_SYSTEM, f"{cov.title}\n\n{cov.text}")
    obj = _json_block(raw)
    return S.from_json(obj) if isinstance(obj, dict) else None


def parse_covenants(covs: list, workers: int = 8) -> list[S.Spec | None]:
    """Parse many covenants concurrently.

    Returns a list positionally aligned with the input. Keying by id() would be
    wrong: CPython reuses ids, and the same object may legitimately appear twice.
    """
    if not covs:
        return []
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(parse_covenant, covs))


# ----------------------------------------------------------- categorisation

CATEGORY_SYSTEM = """Ты классифицируешь назначения платежей из банковского реестра.

Верни ТОЛЬКО JSON-объект вида {"<описание>": "<категория>", ...}, без пояснений.

Допустимые категории:
  revenue   — поступления от продажи основной продукции или услуг
  capex     — приобретение оборудования и основных средств, капитальные работы
  opex      — эксплуатация и обслуживание производственных объектов
  lease     — аренда и лизинговые платежи
  payroll   — оплата труда
  utilities — электроэнергия, вода, газ, тепло, стоки
  interest  — проценты по займам и кредитам
  tax       — налоги, сборы, пошлины
  insurance — страховые премии
  marketing — реклама и продвижение
  telecom   — связь
  financing — поступления по кредитным линиям и займам
  related   — управленческие и консультационные вознаграждения
  other     — всё остальное

Классифицируй по существу операции, а не по наименованию контрагента."""


def categorise(descriptions: list[str], batch: int = 60) -> dict[str, str]:
    """Category for each description core. Empty dict when the LLM is absent."""
    mapping: dict[str, str] = {}
    chunks = [descriptions[i:i + batch] for i in range(0, len(descriptions), batch)]

    def one(chunk):
        raw = _call(CATEGORY_SYSTEM, "\n".join(chunk), max_tokens=4000)
        obj = _json_block(raw)
        return obj if isinstance(obj, dict) else {}

    if not chunks:
        return mapping
    with cf.ThreadPoolExecutor(4) as ex:
        for part in ex.map(one, chunks):
            for k, v in part.items():
                if v in S.CATEGORIES:
                    mapping[k] = v
    return mapping
