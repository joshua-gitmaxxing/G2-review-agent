import json
import sys
import requests

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


def run_smoke_test(url: str = "http://127.0.0.1:8000/analyze"):
    payload = {
        "reviewer_name": "Alex Smith",
        "reviewer_title": "Head of Operations",
        "reviewer_company": "TechCorp",
        "review_text": "Integration with our CRM breaks frequently, causing massive sync errors.",
        "review_date": "2026-08-05",
        "competitor_product": "CompetitorY",
    }

    print(f"Sending dummy review payload to {url}...")

    data = None
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f"[HTTP call to {url} failed: {exc}. Attempting internal FastAPI TestClient execution...]")
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        res = client.post("/analyze", json=payload)
        assert res.status_code == 200, f"Status code failed: {res.status_code}"
        data = res.json()

    print("\n--- Full JSON Response Received ---")
    print(json.dumps(data, indent=2))
    print("-----------------------------------\n")

    assertions = []

    # 1. Assert response contains all 16 fields defined in schema
    if len(REQUIRED_FIELDS) != 16:
        assertions.append("Internal error: Schema spec does not list 16 fields")

    missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
    if missing_fields:
        assertions.append(f"Missing schema fields: {missing_fields}")
    else:
        print(f"[OK] Schema check: All {len(REQUIRED_FIELDS)} fields present.")

    # 2. confidence_score is float between 0.0 and 1.0
    conf_score = data.get("confidence_score")
    if not isinstance(conf_score, (float, int)) or not (0.0 <= conf_score <= 1.0):
        assertions.append(
            f"confidence_score invalid: {conf_score} (must be float/int between 0.0 and 1.0)"
        )
    else:
        print(f"[OK] confidence_score valid: {conf_score} (float between 0.0 and 1.0)")

    # 3. total_score is integer between 0 and 100
    total_score = data.get("total_score")
    if not isinstance(total_score, int) or isinstance(total_score, bool) or not (0 <= total_score <= 100):
        assertions.append(
            f"total_score invalid: {total_score} (must be integer between 0 and 100)"
        )
    else:
        print(f"[OK] total_score valid: {total_score} (integer between 0 and 100)")

    # 4. tier is one of Tier 1, Tier 2, or Discard
    tier = data.get("tier")
    if tier not in VALID_TIERS:
        assertions.append(f"tier invalid: '{tier}' (must be one of {VALID_TIERS})")
    else:
        print(f"[OK] tier valid: '{tier}' (one of {VALID_TIERS})")

    # 5. confidence_label is either high or low
    conf_label = data.get("confidence_label")
    if conf_label not in VALID_CONFIDENCE_LABELS:
        assertions.append(
            f"confidence_label invalid: '{conf_label}' (must be one of {VALID_CONFIDENCE_LABELS})"
        )
    else:
        print(f"[OK] confidence_label valid: '{conf_label}' (either 'high' or 'low')")

    print("\n===================================")
    if assertions:
        print("RESULT: FAIL")
        for err in assertions:
            print(f"  - ERROR: {err}")
        sys.exit(1)
    else:
        print("RESULT: PASS")
        print("===================================")


if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/analyze"
    run_smoke_test(target_url)
