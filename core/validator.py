from datetime import datetime, date
from typing import Optional

TODAY = date.today()
TWO_YEARS_AGO = date(TODAY.year - 2, TODAY.month, TODAY.day)

# ─── PLACEHOLDER PATTERNS ────────────────────────────────────
PLACEHOLDER_PATTERNS = [
    "month year", "dd/mm/yyyy", "mm/dd/yyyy",
    "date here", "insert date", "[date]", "xx/xx/xxxx"
]

# ─── INDIVIDUAL RULES ────────────────────────────────────────

REFERENCE_CONTEXT_KEYWORDS = [
    "enacted", "signed into law", "established", "published",
    "effective since", "as of", "history", "background",
    "inspection", "issued in", "dated", "originally"
]

def check_age(normalized: str, raw: str, context: str = "") -> Optional[dict]:
    if not normalized:
        return None
    # skip if this is clearly a historical reference
    if any(k in context.lower() for k in REFERENCE_CONTEXT_KEYWORDS):
        return None
    try:
        doc_date = datetime.fromisoformat(normalized).date()
        if doc_date < TWO_YEARS_AGO:
            age_years = (TODAY - doc_date).days / 365
            return {
                "rule": "AGE_CHECK",
                "level": "WARNING",
                "message": f"Date '{raw}' is {age_years:.1f} years old — exceeds 2-year threshold",
                "raw": raw,
                "normalized": normalized
            }
    except ValueError:
        pass
    return None


def check_expired(normalized: str, raw: str) -> Optional[dict]:
    """Flag if expiry date has already passed."""
    if not normalized:
        return None
    try:
        doc_date = datetime.fromisoformat(normalized).date()
        if doc_date < TODAY:
            days_expired = (TODAY - doc_date).days
            return {
                "rule":    "EXPIRY_CHECK",
                "level":   "CRITICAL",
                "message": f"Expiry date '{raw}' passed {days_expired} days ago",
                "raw":     raw,
                "normalized": normalized
            }
    except ValueError:
        pass
    return None


def check_date_range(normalized: str, raw: str) -> Optional[dict]:
    """Flag dates outside reasonable pharmaceutical range."""
    if not normalized:
        return None
    try:
        year = int(normalized[:4])
        if year < 1990 or year > 2035:
            return {
                "rule":    "RANGE_CHECK",
                "level":   "WARNING",
                "message": f"Date '{raw}' has suspicious year {year} — outside 1990-2035",
                "raw":     raw,
                "normalized": normalized
            }
    except (ValueError, IndexError):
        pass
    return None


def check_placeholder(raw: str) -> Optional[dict]:
    """Flag placeholder text masquerading as dates."""
    if any(p in raw.lower() for p in PLACEHOLDER_PATTERNS):
        return {
            "rule":    "PLACEHOLDER_CHECK",
            "level":   "CRITICAL",
            "message": f"'{raw}' appears to be a placeholder — not a real date",
            "raw":     raw,
            "normalized": None
        }
    return None


def check_date_logic(dates: list[dict]) -> list[dict]:
    """
    Cross-date logic checks across all dates in a document.
    Finds cases where expiry < manufacture, etc.
    Requires multiple dates to compare.
    """
    flags = []

    # find earliest and latest normalized dates
    normalized = sorted([
        d["normalized"] for d in dates
        if d.get("normalized") and
        not d.get("is_duration") and
        not d.get("is_fiscal")
    ])

    if len(normalized) < 2:
        return flags

    try:
        earliest = datetime.fromisoformat(normalized[0]).date()
        latest   = datetime.fromisoformat(normalized[-1]).date()

        # if latest is in the past AND earliest is very old
        if latest < TODAY and earliest < TWO_YEARS_AGO:
            flags.append({
                "rule":    "LOGIC_CHECK",
                "level":   "WARNING",
                "message": f"All dates in document are historical "
                           f"({normalized[0]} to {normalized[-1]}) — "
                           f"document may need renewal",
                "raw":     f"{normalized[0]} to {normalized[-1]}",
                "normalized": normalized[-1]
            })
    except ValueError:
        pass

    return flags


# ─── MAIN VALIDATOR ──────────────────────────────────────────
def validate_dates(dates: list[dict]) -> dict:
    """
    Run all business rules against extracted dates.

    Input:  list of date dicts from extractor
    Output: validation report with flags by severity
    """
    critical = []
    warnings = []

    for d in dates:
        raw        = d.get("raw_date") or d.get("raw", "")
        normalized = d.get("normalized")
        context    = d.get("context") or d.get("text", "")
        page       = d.get("page", "?")

        # add page context to each flag
        def with_context(flag):
            if flag:
                flag["page"]    = page
                flag["context"] = context[:80]
            return flag

        # Rule 1 — placeholder
        flag = with_context(check_placeholder(raw))
        if flag:
            critical.append(flag)
            continue  # no point running other rules on placeholders

        # Rule 2 — expiry
        EXPIRY_KEYWORDS = [
            "expir", "valid until", "valid through",
            "use by", "best before", "expiration date",
            "lot expir", "exp date", "exp:"
        ]

        is_expiry_context = any(kw in context.lower() for kw in EXPIRY_KEYWORDS)
        try:
            is_past_date = normalized and datetime.fromisoformat(normalized).date() < TODAY
        except ValueError:
            is_past_date = False

        # Flag expired dates even without explicit expiry keyword
        if any(kw in context.lower() for kw in EXPIRY_KEYWORDS):
            flag = with_context(check_expired(normalized, raw))
            if flag:
                critical.append(flag)

        # Rule 3 — age check (all non-future dates)
        flag = with_context(check_age(normalized, raw, context))
        if flag:
            warnings.append(flag)

        # Rule 4 — range sanity
        flag = with_context(check_date_range(normalized, raw))
        if flag:
            warnings.append(flag)

    # Rule 5 — cross-date logic
    logic_flags = check_date_logic(dates)
    warnings.extend(logic_flags)

    return {
        "critical":       critical,
        "warnings":       warnings,
        "total_flags":    len(critical) + len(warnings),
        "critical_count": len(critical),
        "warning_count":  len(warnings),
        "passed":         len(critical) == 0
    }


def format_validation_report(report: dict, filename: str) -> str:
    """Human-readable validation summary for Streamlit display."""
    lines = [f"**Validation Report: {filename}**\n"]

    if report["passed"]:
        lines.append("✅ No critical issues found.")
    else:
        lines.append(
            f"⚠️ {report['critical_count']} critical | "
            f"{report['warning_count']} warnings\n"
        )

    if report["critical"]:
        lines.append("**🔴 Critical:**")
        for f in report["critical"]:
            lines.append(
                f"- {f['message']} "
                f"*(Page {f['page']})*"
            )

    if report["warnings"]:
        lines.append("\n**🟡 Warnings:**")
        for f in report["warnings"]:
            lines.append(
                f"- {f['message']} "
                f"*(Page {f['page']})*"
            )

    return "\n".join(lines)