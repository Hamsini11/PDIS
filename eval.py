"""
eval.py — PDIS Accuracy Evaluation
===================================
Run this after python main.py to measure accuracy.

Metrics:
  - Date Recall:    % of known dates found
  - Date Precision: % of found dates that are correct
  - Flag Accuracy:  % of known flags correctly raised
  - Section Accuracy: % of known sections detected
  - Routing Accuracy: % of queries routed correctly
"""

import json
from pathlib import Path
from datetime import datetime
from core.vector_store import get_all_dates, list_documents, search
from core.validator import validate_dates
from core.router import is_structured_query

# ─── GROUND TRUTH ────────────────────────────────────────────
# These are dates/flags we KNOW exist in these documents.
# Verified manually by reading the actual PDFs.

GROUND_TRUTH = {
    "sample-sdf-document.pdf": {
        "known_dates": [
            "2021-01-26",   # manufacture date page 2
            "2023-01-26",   # expiry lot 17242818 page 2
        ],
        "known_flags": ["EXPIRY_CHECK"],
        "known_sections": [
            "Certificate of Compliance",
            "Test Report"
        ],
        "doc_description": "Vendor SDF — expired lot present"
    },

    "importation-guidance.pdf": {
        "known_dates": [],  # no reliable calendar dates
        "known_flags": ["PLACEHOLDER_CHECK"],
        "known_sections": [
            "Guidance Document",
            "Dear Healthcare Provider Letter"
        ],
        "doc_description": "FDA importation guidance — placeholder date on p.17"
    },

    "DSCSA Pilot Project - Final Report.pdf": {
        "known_dates": [
            "2019-02-08",   # program established date page 1
            "2023-11-27",   # DSCSA deadline
        ],
        "known_flags": [],  # historical doc, no expiry flags expected
        "known_sections": [
            "Regulatory Filing",
            "Test Report"
        ],
        "doc_description": "FDA DSCSA pilot report — 53 pages"
    },

    "OPQ_FY23RSPQ_20250821.pdf": {
        "known_dates": [
            "2025-08-21",   # document date in filename
        ],
        "known_flags": [],
        "known_sections": [
            "Regulatory Filing",
            "Quality Report"
        ],
        "doc_description": "FDA OPQ annual report FY2023"
    }
}

# ─── ROUTING GROUND TRUTH ────────────────────────────────────
# (query, expected_route)
ROUTING_GROUND_TRUTH = [
    ("What is the latest date found across all documents?",   "structured"),
    ("Show me all flagged dates",                             "structured"),
    ("What is the earliest date in DSCSA report?",           "structured"),
    ("What does FDA say about audit trails?",                 "semantic"),
    ("Summarize the OPQ report",                             "semantic"),
    ("What are the DSCSA compliance requirements?",           "semantic"),
    ("What electronic records requirements apply?",           "semantic"),
    ("What dates are in the importation guidance?",          "structured"),
    ("Explain the part 11 regulations",                      "semantic"),
    ("List all dates from sample SDF document",              "structured"),
]


# ─── EVALUATION FUNCTIONS ────────────────────────────────────

def evaluate_date_extraction(filename: str, truth: dict) -> dict:
    """
    Compare extracted dates against known ground truth.
    Returns recall, precision, and F1.
    """
    extracted = get_all_dates(filter_doc=filename)
    extracted_normalized = set(
        d.get("normalized") for d in extracted
        if d.get("normalized")
    )
    known = set(truth["known_dates"])

    if not known:
        return {
            "recall":    None,
            "precision": None,
            "f1":        None,
            "note":      "No ground truth dates defined"
        }

    true_positives  = len(known & extracted_normalized)
    recall          = true_positives / len(known) if known else 0
    precision       = true_positives / len(extracted_normalized) \
                      if extracted_normalized else 0
    f1              = (2 * precision * recall / (precision + recall)) \
                      if (precision + recall) > 0 else 0

    missed = known - extracted_normalized
    extra  = extracted_normalized - known

    return {
        "recall":         round(recall, 3),
        "precision":      round(precision, 3),
        "f1":             round(f1, 3),
        "known_count":    len(known),
        "found_count":    len(extracted_normalized),
        "true_positives": true_positives,
        "missed_dates":   list(missed),
        "extra_dates":    list(extra)[:5]  # cap at 5
    }


def evaluate_flag_accuracy(filename: str, truth: dict) -> dict:
    extracted = get_all_dates(filter_doc=filename)
    validation = validate_dates(extracted)

    raised_rules = set(
        f["rule"]
        for f in validation["critical"] + validation["warnings"]
    )
    known_flags = set(truth["known_flags"])

    if not known_flags:
        return {
            "accuracy": None,
            "note": "No flags expected for this document"
        }

    caught   = known_flags & raised_rules
    missed   = known_flags - raised_rules
    accuracy = len(caught) / len(known_flags)

    return {
        "accuracy":     round(accuracy, 3),
        "known_flags":  list(known_flags),
        "caught_flags": list(caught),
        "missed_flags": list(missed)
    }

def evaluate_section_detection(filename: str, truth: dict) -> dict:
    known_sections = truth["known_sections"]
    if not known_sections:
        return {"accuracy": None, "note": "No sections defined"}

    sections_file = Path("storage/sections.json")
    if not sections_file.exists():
        return {"accuracy": None, "note": "sections.json not found"}

    all_sections = json.loads(sections_file.read_text())
    detected = all_sections.get(filename, [])
    detected_types = [
        s["section_type"] if isinstance(s, dict) else s
        for s in detected
    ]
    
    found = 0
    results = []
    for section in known_sections:
        match = any(section.lower() in d.lower() for d in detected_types)
        if match:
            found += 1
        results.append({"section": section, "detected": match})

    accuracy = found / len(known_sections)
    return {
        "accuracy": round(accuracy, 3),
        "sections": results,
        "detected_types": detected_types
    }

def evaluate_query_routing() -> dict:
    """
    Test query routing accuracy without calling Claude.
    Uses the is_structured_query function directly.
    """

    correct = 0
    results = []

    for query, expected in ROUTING_GROUND_TRUTH:
        predicted = "structured" if is_structured_query(query) else "semantic"
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        results.append({
            "query":     query[:60],
            "expected":  expected,
            "predicted": predicted,
            "correct":   is_correct
        })

    accuracy = correct / len(ROUTING_GROUND_TRUTH)
    wrong = [r for r in results if not r["correct"]]

    return {
        "accuracy": round(accuracy, 3),
        "correct":  correct,
        "total":    len(ROUTING_GROUND_TRUTH),
        "wrong":    wrong
    }


# ─── MAIN EVAL RUNNER ────────────────────────────────────────

def run_full_eval():
    print("\n" + "="*60)
    print("PDIS ACCURACY EVALUATION")
    print(f"Run date: {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    indexed = list_documents()
    overall_date_recalls    = []
    overall_flag_accuracies = []
    overall_section_accuracies = []

    for filename, truth in GROUND_TRUTH.items():
        if filename not in indexed:
            print(f"\n⚠️  SKIPPED: {filename} (not indexed)")
            continue

        print(f"\n── {filename}")
        print(f"   {truth['doc_description']}")

        # Date extraction
        date_result = evaluate_date_extraction(filename, truth)
        if date_result["recall"] is not None:
            print(f"   Date Recall:    {date_result['recall']:.0%} "
                  f"({date_result['true_positives']}/{date_result['known_count']} found)")
            print(f"   Date Precision: {date_result['precision']:.0%}")
            if date_result["missed_dates"]:
                print(f"   ❌ Missed: {date_result['missed_dates']}")
            overall_date_recalls.append(date_result["recall"])
        else:
            print(f"   Date Recall:    {date_result['note']}")

        # Flag accuracy
        flag_result = evaluate_flag_accuracy(filename, truth)
        if flag_result["accuracy"] is not None:
            print(f"   Flag Accuracy:  {flag_result['accuracy']:.0%} "
                  f"({flag_result['caught_flags']} caught)")
            if flag_result["missed_flags"]:
                print(f"   ❌ Missed flags: {flag_result['missed_flags']}")
            overall_flag_accuracies.append(flag_result["accuracy"])
        else:
            print(f"   Flag Accuracy:  {flag_result['note']}")

        # Section detection
        section_result = evaluate_section_detection(filename, truth)
        if section_result["accuracy"] is not None:
            print(f"   Section Detect: {section_result['accuracy']:.0%}")
            overall_section_accuracies.append(section_result["accuracy"])

    # Query routing
    print(f"\n── Query Routing")
    routing = evaluate_query_routing()
    if "error" not in routing:
        print(f"   Routing Accuracy: {routing['accuracy']:.0%} "
              f"({routing['correct']}/{routing['total']})")
        if routing["wrong"]:
            print(f"   ❌ Wrong routes:")
            for w in routing["wrong"]:
                print(f"      '{w['query']}' "
                      f"→ predicted {w['predicted']}, "
                      f"expected {w['expected']}")
    else:
        print(f"   {routing['error']}")

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    if overall_date_recalls:
        avg_recall = sum(overall_date_recalls) / len(overall_date_recalls)
        print(f"  Avg Date Recall:     {avg_recall:.0%}")

    if overall_flag_accuracies:
        avg_flags = sum(overall_flag_accuracies) / len(overall_flag_accuracies)
        print(f"  Avg Flag Accuracy:   {avg_flags:.0%}")

    if overall_section_accuracies:
        avg_sections = sum(overall_section_accuracies) / len(overall_section_accuracies)
        print(f"  Avg Section Detect:  {avg_sections:.0%}")

    if "error" not in routing:
        print(f"  Query Routing:       {routing['accuracy']:.0%}")

    print(f"\n  Target: 97%+")
    print(f"  Model:  NOTE — switch to Sonnet for final eval")
    print("="*60)


if __name__ == "__main__":
    run_full_eval()