"""
Offline test suite for company-name and company-size handling (TEST-005).

All tests run completely offline with mocked model responses.
No live Gemini or Apify calls are made.
"""

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from main import app
from agent import (
    AntigravityReviewAgent,
    normalize_company_info,
    is_g2_size_label,
    email_treats_size_as_employer,
)


SAMPLE_MOCK_MODEL_OUTPUT = {
    "reviewer_name": "Jane Doe",
    "reviewer_title": "Head of Operations",
    "reviewer_company": "Acme Corp",
    "reviewer_company_size": "Enterprise",
    "review_text": "CRM integrations constantly disconnect, halting operations.",
    "review_date": "2026-09-01",
    "pain_category": "Integration Failures",
    "pain_summary": "CRM integrations fail frequently, disrupting operational syncs.",
    "confidence_score": 0.90,
    "confidence_label": "high",
    "recency_score": 30,
    "seniority_score": 30,
    "pain_intensity_score": 20,
    "total_score": 80,
    "tier": "Tier 1",
    "drafted_email": (
        "Hi Jane, recurring CRM disconnects drain operational bandwidth every single week. "
        "Our native integrations guarantee continuous data sync without manual maintenance. "
        "Open to exploring a more dependable alternative?"
    ),
    "competitor_product": "CompetitorZ",
}


class TestCompanyFields(unittest.TestCase):
    def setUp(self):
        self.agent = AntigravityReviewAgent()
        self.client = TestClient(app)

    # -----------------------------------------------------------------------
    # 1. Known company name and size
    # -----------------------------------------------------------------------
    @patch("agent._call_llm")
    def test_known_company_name_and_size(self, mock_llm):
        mock_output = dict(SAMPLE_MOCK_MODEL_OUTPUT)
        mock_output["reviewer_company"] = "Acme Corp"
        mock_output["reviewer_company_size"] = "Enterprise"
        mock_llm.return_value = json.dumps(mock_output)

        payload = {
            "reviewer_name": "Jane Doe",
            "reviewer_title": "Head of Operations",
            "reviewer_company": "Acme Corp",
            "reviewer_company_size": "Enterprise",
            "review_text": "CRM integrations constantly disconnect, halting operations.",
            "review_date": "2026-09-01",
            "competitor_product": "CompetitorZ",
        }

        result = self.agent.analyze(payload)

        # Verify fields are preserved accurately
        self.assertEqual(result["reviewer_company"], "Acme Corp")
        self.assertEqual(result["reviewer_company_size"], "Enterprise")
        self.assertNotIn("at Enterprise", result["drafted_email"])
        self.assertNotIn("teams at Enterprise", result["drafted_email"])

    # -----------------------------------------------------------------------
    # 2. Missing company name with known size
    # -----------------------------------------------------------------------
    @patch("agent._call_llm")
    def test_missing_company_name_with_known_size(self, mock_llm):
        mock_output = dict(SAMPLE_MOCK_MODEL_OUTPUT)
        mock_output["reviewer_company"] = None
        mock_output["reviewer_company_size"] = "Small-Business"
        mock_output["drafted_email"] = (
            "Hi Jane, teams dealing with sync errors often waste critical hours each week. "
            "Worth a 5-minute conversation?"
        )
        mock_llm.return_value = json.dumps(mock_output)

        payload = {
            "reviewer_name": "Jane Doe",
            "reviewer_title": "Head of Operations",
            "reviewer_company": None,
            "reviewer_company_size": "Small-Business",
            "review_text": "CRM integrations constantly disconnect, halting operations.",
            "review_date": "2026-09-01",
            "competitor_product": "CompetitorZ",
        }

        result = self.agent.analyze(payload)

        self.assertIsNone(result["reviewer_company"])
        self.assertEqual(result["reviewer_company_size"], "Small-Business")
        self.assertNotIn("Small-Business", result["drafted_email"])
        self.assertNotIn("None", result["drafted_email"])
        self.assertNotIn("Unknown Company", result["drafted_email"])

    # -----------------------------------------------------------------------
    # 3. Both fields missing
    # -----------------------------------------------------------------------
    @patch("agent._call_llm")
    def test_both_fields_missing(self, mock_llm):
        mock_output = dict(SAMPLE_MOCK_MODEL_OUTPUT)
        mock_output["reviewer_company"] = None
        mock_output["reviewer_company_size"] = None
        mock_output["drafted_email"] = (
            "Hi Jane, teams dealing with integration failures frequently look for better alternatives. "
            "Worth a quick chat?"
        )
        mock_llm.return_value = json.dumps(mock_output)

        payload = {
            "reviewer_name": "Jane Doe",
            "reviewer_title": "Head of Operations",
            "reviewer_company": None,
            "reviewer_company_size": None,
            "review_text": "CRM integrations constantly disconnect, halting operations.",
            "review_date": "2026-09-01",
            "competitor_product": "CompetitorZ",
        }

        result = self.agent.analyze(payload)

        self.assertIsNone(result["reviewer_company"])
        self.assertIsNone(result["reviewer_company_size"])
        self.assertNotIn("teams at None", result["drafted_email"])
        self.assertNotIn("Unknown Company", result["drafted_email"])

    # -----------------------------------------------------------------------
    # 4. Legacy company-size value in reviewer_company
    # -----------------------------------------------------------------------
    @patch("agent._call_llm")
    def test_legacy_company_size_value_in_reviewer_company(self, mock_llm):
        mock_output = dict(SAMPLE_MOCK_MODEL_OUTPUT)
        mock_output["reviewer_company"] = None
        mock_output["reviewer_company_size"] = "Mid-Market"
        mock_output["drafted_email"] = (
            "Hi Jane, teams dealing with integration failures often look for better tools. "
            "Worth a quick call?"
        )
        mock_llm.return_value = json.dumps(mock_output)

        # Legacy payload: G2 size label passed in reviewer_company, size field empty
        payload = {
            "reviewer_name": "Jane Doe",
            "reviewer_title": "Director of Operations",
            "reviewer_company": "Mid-Market",
            "reviewer_company_size": None,
            "review_text": "CRM integrations constantly disconnect, halting operations.",
            "review_date": "2026-09-01",
            "competitor_product": "CompetitorZ",
        }

        result = self.agent.analyze(payload)

        # Recognized size must move to reviewer_company_size and leave reviewer_company null
        self.assertIsNone(result["reviewer_company"])
        self.assertEqual(result["reviewer_company_size"], "Mid-Market")
        self.assertNotIn("teams at Mid-Market", result["drafted_email"])
        self.assertNotIn("at Mid-Market", result["drafted_email"])

    @patch("agent._call_llm")
    def test_legacy_g2_employee_range_in_reviewer_company(self, mock_llm):
        mock_output = dict(SAMPLE_MOCK_MODEL_OUTPUT)
        mock_llm.return_value = json.dumps(mock_output)

        payload = {
            "reviewer_name": "Jane Doe",
            "reviewer_company": "Small-Business (50 or fewer emp.)",
            "reviewer_company_size": None,
        }

        result = self.agent.analyze(payload)

        self.assertIsNone(result["reviewer_company"])
        self.assertEqual(result["reviewer_company_size"], "Small-Business (50 or fewer emp.)")

    # -----------------------------------------------------------------------
    # 5. Model inventing a company name despite missing input
    # -----------------------------------------------------------------------
    @patch("agent._call_llm")
    def test_model_inventing_company_name_despite_missing_input(self, mock_llm):
        # The model hallucinates and outputs an invented company name
        hallucinated_model_output = dict(SAMPLE_MOCK_MODEL_OUTPUT)
        hallucinated_model_output["reviewer_company"] = "Invented TechCorp Global"
        mock_llm.return_value = json.dumps(hallucinated_model_output)

        payload = {
            "reviewer_name": "Jane Doe",
            "reviewer_title": "Head of Operations",
            "reviewer_company": None,  # Verified input has NO company
            "reviewer_company_size": "Enterprise",
            "review_text": "CRM integrations constantly disconnect, halting operations.",
            "review_date": "2026-09-01",
            "competitor_product": "CompetitorZ",
        }

        result = self.agent.analyze(payload)

        # Must restore reviewer_company from normalized input; do not trust model invention
        self.assertIsNone(result["reviewer_company"])
        self.assertEqual(result["reviewer_company_size"], "Enterprise")

    # -----------------------------------------------------------------------
    # 6. /analyze response preserving both fields correctly (FastAPI)
    # -----------------------------------------------------------------------
    @patch("agent._call_llm")
    def test_analyze_endpoint_preserves_both_fields(self, mock_llm):
        mock_output = dict(SAMPLE_MOCK_MODEL_OUTPUT)
        mock_output["reviewer_company"] = "DataSystems Inc"
        mock_output["reviewer_company_size"] = "Mid-Market"
        mock_llm.return_value = json.dumps(mock_output)

        payload = {
            "reviewer_name": "Jane Doe",
            "reviewer_title": "Head of Operations",
            "reviewer_company": "DataSystems Inc",
            "reviewer_company_size": "Mid-Market",
            "review_text": "CRM integrations constantly disconnect, halting operations.",
            "review_date": "2026-09-01",
            "competitor_product": "CompetitorZ",
        }

        response = self.client.post("/analyze", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["reviewer_company"], "DataSystems Inc")
        self.assertEqual(data["reviewer_company_size"], "Mid-Market")

    @patch("agent._call_llm")
    def test_analyze_endpoint_serializes_null_for_missing_company(self, mock_llm):
        mock_output = dict(SAMPLE_MOCK_MODEL_OUTPUT)
        mock_output["reviewer_company"] = None
        mock_output["reviewer_company_size"] = "Small-Business"
        mock_llm.return_value = json.dumps(mock_output)

        payload = {
            "reviewer_name": "Jane Doe",
            "reviewer_title": "Head of Operations",
            "reviewer_company": "Unknown Company",  # Placeholder that normalizes to None
            "reviewer_company_size": "Small-Business",
            "review_text": "CRM integrations constantly disconnect, halting operations.",
            "review_date": "2026-09-01",
            "competitor_product": "CompetitorZ",
        }

        response = self.client.post("/analyze", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIsNone(data["reviewer_company"])
        self.assertEqual(data["reviewer_company_size"], "Small-Business")

    # -----------------------------------------------------------------------
    # 7. Exact validation & legitimate phrasing preservation
    # -----------------------------------------------------------------------
    @patch("agent._call_llm")
    def test_teams_at_small_business_full_label_replaced_with_fallback(self, mock_llm):
        mock_output = dict(SAMPLE_MOCK_MODEL_OUTPUT)
        # Invalid generated email treating size with employee range as employer name
        mock_output["drafted_email"] = (
            "Hi Jane, teams at Small-Business (50 or fewer emp.) dealing with CRM sync errors often struggle. "
            "Worth a chat?"
        )
        mock_llm.return_value = json.dumps(mock_output)

        payload = {
            "reviewer_name": "Jane Doe",
            "reviewer_title": "Head of Operations",
            "reviewer_company": None,
            "reviewer_company_size": "Small-Business (50 or fewer emp.)",
            "review_text": "CRM integrations constantly disconnect, halting operations.",
        }

        result = self.agent.analyze(payload)

        # Entire invalid generated email must be replaced by safe generic fallback email
        self.assertNotIn("Small-Business", result["drafted_email"])
        self.assertNotIn("(50 or fewer emp.)", result["drafted_email"])
        self.assertEqual(
            result["drafted_email"],
            "Hi Jane, Dealing with integration issues can create operational friction for teams. "
            "CRM integrations fail frequently, disrupting operational syncs. "
            "Is improving integration reliability currently a priority?"
        )

    @patch("agent._call_llm")
    def test_at_mid_market_full_label_replaced_with_fallback(self, mock_llm):
        mock_output = dict(SAMPLE_MOCK_MODEL_OUTPUT)
        # Invalid generated email treating size with employee range as employer name
        mock_output["drafted_email"] = (
            "Hi Jane, at Mid-Market (51-1000 emp.) organizations, broken syncs drain bandwidth. "
            "Worth a conversation?"
        )
        mock_llm.return_value = json.dumps(mock_output)

        payload = {
            "reviewer_name": "Jane Doe",
            "reviewer_title": "Head of Operations",
            "reviewer_company": None,
            "reviewer_company_size": "Mid-Market (51-1000 emp.)",
            "review_text": "CRM integrations constantly disconnect, halting operations.",
        }

        result = self.agent.analyze(payload)

        # Entire invalid generated email must be replaced by safe generic fallback email
        self.assertNotIn("Mid-Market", result["drafted_email"])
        self.assertNotIn("(51-1000 emp.)", result["drafted_email"])
        self.assertEqual(
            result["drafted_email"],
            "Hi Jane, Dealing with integration issues can create operational friction for teams. "
            "CRM integrations fail frequently, disrupting operational syncs. "
            "Is improving integration reliability currently a priority?"
        )

    @patch("agent._call_llm")
    def test_legitimate_phrases_like_mid_market_teams_preserved(self, mock_llm):
        mock_output = dict(SAMPLE_MOCK_MODEL_OUTPUT)
        # Legitimate descriptive phrase in email
        expected_email = (
            "Hi Jane, mid-market teams dealing with integration failures often switch to our solution. "
            "Worth a 5-minute conversation?"
        )
        mock_output["drafted_email"] = expected_email
        mock_llm.return_value = json.dumps(mock_output)

        payload = {
            "reviewer_name": "Jane Doe",
            "reviewer_title": "Head of Operations",
            "reviewer_company": None,
            "reviewer_company_size": "Mid-Market",
            "review_text": "CRM integrations constantly disconnect, halting operations.",
        }

        # 1. Verify email_treats_size_as_employer does not treat legitimate adjective as employer name (TEST-005)
        self.assertFalse(email_treats_size_as_employer(expected_email, "Mid-Market"))

        # 2. Verify deterministic safe neutral email is generated when no approved outreach context exists (TEST-002)
        result = self.agent.analyze(payload)
        self.assertEqual(
            result["drafted_email"],
            "Hi Jane, Dealing with integration issues can create operational friction for teams. "
            "CRM integrations fail frequently, disrupting operational syncs. "
            "Is improving integration reliability currently a priority?"
        )




if __name__ == "__main__":
    unittest.main()
