"""
Smoke test for the G2 Review Signal middleware.

Modes:
  python smoke_test.py                 — Live mode: hits http://127.0.0.1:8000/analyze
  python smoke_test.py --dry-run       — Dry-run mode: validates agent.py directly
                                         (bypasses LLM, tests schema/validation logic)
  python smoke_test.py <url>           — Hit a specific URL

The dry-run mode is useful when no API key is set. It calls agent._validate_and_coerce()
with a realistic pre-built response object and confirms the output passes all assertions.
"""
import json
import sys
import os

REQUIRED_FIELDS = [
    "reviewer_name",
    "reviewer_title",
    "reviewer_company",
    "review_text",
    "review_date",
    "pain_category",
    "pain_summary",
    "confidence_score",
    "confidence_label",
    "recency_score",
    "seniority_score",
    "pain_intensity_score",
    "total_score",
    "tier",
    "drafted_email",
    "competitor_product",
]

VALID_TIERS = {"Tier 1", "Tier 2", "Discard"}
VALID_CONFIDENCE_LABELS = {"high", "low"}
VALID_CATEGORIES = {
    "Onboarding Friction", "Poor Customer Support", "Integration Failures",
    "Pricing / Value Mismatch", "Performance / Reliability Issues", "Missing Features",
    "Steep Learning Curve", "Poor Reporting / Analytics", "Clunky UI / UX",
    "Lack of Scalability", "Other",
}

DUMMY_PAYLOAD = {
    "reviewer_name": "Alex Smith",
    "reviewer_title": "Head of Operations",
    "reviewer_company": "TechCorp",
    "review_text": "Integration with our CRM breaks frequently, causing massive sync errors.",
    "review_date": "2026-08-05",
    "competitor_product": "CompetitorY",
}


def assert_schema(data: dict) -> list[str]:
    """Runs all 5 schema assertions. Returns list of error strings (empty = pass)."""
    errors = []

    # 1. All 16 fields present
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        errors.append(f"Missing schema fields: {missing}")
    else:
        print(f"[OK] Schema check: All {len(REQUIRED_FIELDS)} fields present.")

    # 2. confidence_score is float between 0.0 and 1.0
    cs = data.get("confidence_score")
    if not isinstance(cs, (float, int)) or isinstance(cs, bool) or not (0.0 <= cs <= 1.0):
        errors.append(f"confidence_score invalid: {cs!r} (must be float 0.0–1.0)")
    else:
        print(f"[OK] confidence_score valid: {cs} (float between 0.0 and 1.0)")

    # 3. total_score is integer between 0 and 100
    ts = data.get("total_score")
    if not isinstance(ts, int) or isinstance(ts, bool) or not (0 <= ts <= 100):
        errors.append(f"total_score invalid: {ts!r} (must be integer 0–100)")
    else:
        print(f"[OK] total_score valid: {ts} (integer between 0 and 100)")

    # 4. tier is one of Tier 1, Tier 2, or Discard
    tier = data.get("tier")
    if tier not in VALID_TIERS:
        errors.append(f"tier invalid: {tier!r} (must be one of {VALID_TIERS})")
    else:
        print(f"[OK] tier valid: '{tier}' (one of {VALID_TIERS})")

    # 5. confidence_label is either high or low
    cl = data.get("confidence_label")
    if cl not in VALID_CONFIDENCE_LABELS:
        errors.append(f"confidence_label invalid: {cl!r} (must be 'high' or 'low')")
    else:
        print(f"[OK] confidence_label valid: '{cl}' (either 'high' or 'low')")

    return errors


def dry_run_mode():
    """
    Validates the agent's _validate_and_coerce() and schema logic without making an LLM call.
    Uses a realistic pre-built Claude-style response to simulate what the model would return.
    """
    print("=== DRY RUN MODE (no LLM call) ===")
    print("Simulating model output and running schema validation...\n")

    from agent import _validate_and_coerce, VALID_CATEGORIES

    # Simulate a realistic model JSON output (as Claude/Gemini would return)
    simulated_model_output = {
        "reviewer_name": "Alex Smith",
        "reviewer_title": "Head of Operations",
        "reviewer_company": "TechCorp",
        "review_text": "Integration with our CRM breaks frequently, causing massive sync errors.",
        "review_date": "2026-08-05",
        "pain_category": "Integration Failures",
        "pain_summary": (
            "The reviewer is experiencing repeated CRM integration failures that disrupt "
            "operational data syncing across their tech stack."
        ),
        "confidence_score": 0.88,
        "confidence_label": "high",
        "recency_score": 20,
        "seniority_score": 30,
        "pain_intensity_score": 20,
        "total_score": 70,
        "tier": "Tier 1",
        "drafted_email": (
            "Hi Alex, broken CRM syncs that interrupt operational workflows at TechCorp "
            "can quietly drain hours every week. If your current integration stack has "
            "required manual workarounds, our native connectors eliminate those sync gaps "
            "completely. Worth a 5-minute look at how we handle this?"
        ),
        "competitor_product": "CompetitorY",
    }

    fallback = {
        "reviewer_name": DUMMY_PAYLOAD["reviewer_name"],
        "reviewer_title": DUMMY_PAYLOAD["reviewer_title"],
        "reviewer_company": DUMMY_PAYLOAD["reviewer_company"],
        "review_text": DUMMY_PAYLOAD["review_text"],
        "review_date": DUMMY_PAYLOAD["review_date"],
        "competitor_product": DUMMY_PAYLOAD["competitor_product"],
    }

    data = _validate_and_coerce(simulated_model_output, fallback)

    print("--- Full JSON After Validation ---")
    print(json.dumps(data, indent=2))
    print("---------------------------------\n")

    errors = assert_schema(data)

    print("\n===================================")
    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(f"  - ERROR: {e}")
        sys.exit(1)
    else:
        print("RESULT: PASS (dry run — schema & validation logic confirmed)")
        print("===================================")
        print("\nNOTE: To run against the live LLM, set GEMINI_API_KEY in .env and run:")
        print("      python smoke_test.py")


def live_mode(url: str = "http://127.0.0.1:8000/analyze"):
    """Sends dummy payload to the live server and validates the response."""
    import requests

    print(f"Sending dummy review payload to {url}...")

    data = None
    try:
        response = requests.post(url, json=DUMMY_PAYLOAD, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f"[HTTP call to {url} failed: {exc}]")
        print("Falling back to FastAPI TestClient...")
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        res = client.post("/analyze", json=DUMMY_PAYLOAD)
        if res.status_code != 200:
            print(f"RESULT: FAIL — HTTP {res.status_code}: {res.text[:300]}")
            sys.exit(1)
        data = res.json()

    print("\n--- Full JSON Response Received ---")
    print(json.dumps(data, indent=2))
    print("-----------------------------------\n")

    errors = assert_schema(data)

    print("\n===================================")
    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(f"  - ERROR: {e}")
        sys.exit(1)
    else:
        print("RESULT: PASS")
        print("===================================")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--dry-run" in args:
        dry_run_mode()
    else:
        url = next((a for a in args if not a.startswith("--")), "http://127.0.0.1:8000/analyze")
        live_mode(url)
