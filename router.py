def is_structured_query(query: str) -> bool:
    q = query.lower()
    structured_patterns = [
        "what date", "which date", "when was", "when did",
        "what is the date", "expiry date", "expiration date",
        "effective date", "latest date", "earliest date",
        "how many dates", "list all dates", "list dates",
        "show me dates", "flagged dates", "show flagged",
        "ambiguous dates"
    ]
    return any(pattern in q for pattern in structured_patterns)