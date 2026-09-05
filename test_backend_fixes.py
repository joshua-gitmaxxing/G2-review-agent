"""
Offline test suite for TEST-001, TEST-002, TEST-004, and TEST-006.

All tests run completely offline with mocked model responses.
No live Gemini, Apify, HubSpot, or n8n calls are made.
"""

import json
import re
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from main import app
from agent import (
    AntigravityReviewAgent,
    CATEGORY_TEMPLATES,
    _generate_fallback_email,
    _validate_and_coerce,
    _build_analysis_user_message,
)


class TestBackendFixes(unittest.TestCase):
    def setUp(self):
        self.agent = AntigravityReviewAgent()
        self.client = TestClient(app)

    # -----------------------------------------------------------------------
    # TEST-001: Weak / No-Pain Reviews Must Be Discard
    # -----------------------------------------------------------------------
    @patch("agent._call_llm")
    def test_no_pain_review_with_high_recency_and_seniority_is_discarded(self, mock_llm):
        """
        A review with no actionable pain (pain_intensity_score = 0) must be Discard,
        even if high recency (30) + high seniority (40) would otherwise total 70 (Tier 1).
        """
        mock_analysis = {
            "reviewer_name": "Victoria C-Suite",
            "reviewer_title": "Chief Executive Officer",
            "reviewer_company": "Enterprise Tech",
            "reviewer_company_size": "Enterprise",
            "review_text": "Overall great tool! Solved all our problems and love the UI.",
            "review_date": "2026-09-04",
            "pain_category": "Other",
            "pain_summary": "No actionable pain identified. Review is overwhelmingly positive.",
            "confidence_score": 0.95,
            "confidence_label": "high",
            "recency_score": 30,
            "seniority_score": 40,
            "pain_intensity_score": 0,
            "total_score": 70,  # Model arithmetic might attempt 70
            "tier": "Tier 1",   # Model might attempt Tier 1 based on 70
            "competitor_product": "CompetitorX",
        }
        mock_llm.return_value = json.dumps(mock_analysis)

        payload = {
            "reviewer_name": "Victoria C-Suite",
            "reviewer_title": "Chief Executive Officer",
            "reviewer_company": "Enterprise Tech",
            "reviewer_company_size": "Enterprise",
            "review_text": "Overall great tool! Solved all our problems and love the UI.",
            "review_date": "2026-09-04",
            "competitor_product": "CompetitorX",
        }

        result = self.agent.analyze(payload)

        # Qualification gate forces Discard
        self.assertEqual(result["tier"], "Discard")
        self.assertEqual(result["pain_intensity_score"], 0)
        # Discard must have null drafted_email (TEST-004)
        self.assertIsNone(result["drafted_email"])
        # Email LLM call must NOT be made (mock called exactly once for analysis)
        self.assertEqual(mock_llm.call_count, 1)

    @patch("agent._call_llm")
    def test_low_confidence_with_genuine_pain_eligible_for_human_review(self, mock_llm):
        """
        Keep low-confidence but genuinely painful reviews eligible for the existing human-review route.
        They must NOT be discarded by the qualification gate.
        """
        mock_analysis = {
            "reviewer_name": "Sam LowConf",
            "reviewer_title": "Operations Lead",
            "reviewer_company": "Logistics Co",
            "reviewer_company_size": "Mid-Market",
            "review_text": "Sometimes sync stops working and we cannot figure out why.",
            "review_date": "2026-08-20",
            "pain_category": "Performance / Reliability Issues",
            "pain_summary": "Reviewer experiences intermittent synchronization stops with unclear root cause.",
            "confidence_score": 0.55,  # Low confidence
            "confidence_label": "low",
            "recency_score": 20,
            "seniority_score": 10,
            "pain_intensity_score": 20,  # Genuine pain!
            "total_score": 50,
            "tier": "Tier 2",
            "competitor_product": "CompetitorY",
        }
        mock_llm.return_value = json.dumps(mock_analysis)

        payload = {
            "reviewer_name": "Sam LowConf",
            "reviewer_title": "Operations Lead",
            "reviewer_company": "Logistics Co",
            "reviewer_company_size": "Mid-Market",
            "review_text": "Sometimes sync stops working and we cannot figure out why.",
            "review_date": "2026-08-20",
            "competitor_product": "CompetitorY",
        }

        result = self.agent.analyze(payload)

        # Retains Tier 2 status for human review in Slack, not discarded
        self.assertEqual(result["tier"], "Tier 2")
        self.assertEqual(result["confidence_label"], "low")
        self.assertIsNotNone(result["drafted_email"])
        self.assertEqual(mock_llm.call_count, 1)

    # -----------------------------------------------------------------------
    # TEST-002: Deterministic Unsupported Product Claim Prevention & Outreach Context
    # -----------------------------------------------------------------------
    @patch("agent._call_llm")
    def test_llm_hallucinated_claim_cannot_reach_drafted_email(self, mock_llm):
        """
        Prove that an LLM-hallucinated claim cannot reach drafted_email.
        Even if the model returns a drafted_email full of invented statistics and features,
        the deterministic code generator overwrites it with safe neutral fallback.
        """
        mock_analysis = {
            "reviewer_name": "Jane Lead",
            "reviewer_title": "VP of Sales",
            "reviewer_company": "SalesHub",
            "reviewer_company_size": "Enterprise",
            "review_text": "CRMs disconnect daily, losing critical pipeline data.",
            "review_date": "2026-09-01",
            "pain_category": "Integration Failures",
            "pain_summary": "Daily CRM disconnects cause lost pipeline data.",
            "confidence_score": 0.9,
            "confidence_label": "high",
            "recency_score": 30,
            "seniority_score": 40,
            "pain_intensity_score": 20,
            "total_score": 90,
            "tier": "Tier 1",
            # LLM hallucinating claims not backed by approved outreach context:
            "drafted_email": (
                "Hi Jane, our platform delivers 99.999% uptime, proprietary AI self-healing sync, "
                "and costs 50% less than CompetitorZ. Let's talk tomorrow."
            ),
            "competitor_product": "CompetitorZ",
        }
        mock_llm.return_value = json.dumps(mock_analysis)

        payload = {
            "reviewer_name": "Jane Lead",
            "reviewer_title": "VP of Sales",
            "reviewer_company": "SalesHub",
            "reviewer_company_size": "Enterprise",
            "review_text": "CRMs disconnect daily, losing critical pipeline data.",
            "review_date": "2026-09-01",
            "competitor_product": "CompetitorZ",
        }

        result = self.agent.analyze(payload)
        email = result["drafted_email"]

        # Assert hallucinated claims were prevented in code
        self.assertNotIn("99.999%", email)
        self.assertNotIn("AI self-healing", email)
        self.assertNotIn("50% less", email)
        self.assertNotIn("CompetitorZ", email)
        # Assert never mentions G2
        self.assertNotIn("g2", email.lower())
        # Safe neutral copy was generated using Integration Failures template & verified pain summary
        self.assertEqual(
            email,
            "Hi Jane, Dealing with integration issues can create operational friction for teams at SalesHub. "
            "Daily CRM disconnects cause lost pipeline data. "
            "Is improving integration reliability currently a priority?"
        )

    @patch("agent._call_llm")
    def test_no_context_qualified_leads_make_only_analysis_llm_call(self, mock_llm):
        """
        Prove that no-context qualified leads make only the analysis LLM call (mock_llm.call_count == 1)
        and receive the safe neutral fallback email.
        """
        mock_analysis = {
            "reviewer_name": "Alex Smith",
            "reviewer_title": "Head of Operations",
            "reviewer_company": "CloudCorp",
            "reviewer_company_size": "Mid-Market",
            "review_text": "Integration with our CRM breaks frequently.",
            "review_date": "2026-09-01",
            "pain_category": "Integration Failures",
            "pain_summary": "CRM integrations break frequently.",
            "confidence_score": 0.88,
            "confidence_label": "high",
            "recency_score": 30,
            "seniority_score": 30,
            "pain_intensity_score": 20,
            "total_score": 80,
            "tier": "Tier 1",
            "competitor_product": "CompetitorA",
        }
        mock_llm.return_value = json.dumps(mock_analysis)

        payload = {
            "reviewer_name": "Alex Smith",
            "reviewer_title": "Head of Operations",
            "reviewer_company": "CloudCorp",
            "reviewer_company_size": "Mid-Market",
            "review_text": "Integration with our CRM breaks frequently.",
            "review_date": "2026-09-01",
            "competitor_product": "CompetitorA",
        }

        result = self.agent.analyze(payload)

        self.assertEqual(result["tier"], "Tier 1")
        self.assertEqual(mock_llm.call_count, 1)
        self.assertEqual(
            result["drafted_email"],
            "Hi Alex, Dealing with integration issues can create operational friction for teams at CloudCorp. "
            "CRM integrations break frequently. "
            "Is improving integration reliability currently a priority?"
        )

    @patch("agent._call_llm")
    def test_product_name_only_context_adds_no_capability_claim(self, mock_llm):
        """
        Prove that if only product_name exists without an approved value proposition,
        the generated email does NOT claim the product was built for, solves or improves anything.
        """
        mock_analysis = {
            "reviewer_name": "Jordan Lee",
            "reviewer_title": "Director of IT",
            "reviewer_company": "FinTech Ltd",
            "reviewer_company_size": "Mid-Market",
            "review_text": "Reporting dashboard takes 10 minutes to load.",
            "review_date": "2026-09-01",
            "pain_category": "Poor Reporting / Analytics",
            "pain_summary": "Dashboard takes 10 minutes to load.",
            "confidence_score": 0.9,
            "confidence_label": "high",
            "recency_score": 30,
            "seniority_score": 30,
            "pain_intensity_score": 20,
            "total_score": 80,
            "tier": "Tier 1",
            "competitor_product": "SlowAnalytics",
        }
        mock_llm.return_value = json.dumps(mock_analysis)

        agent_with_prod_only = AntigravityReviewAgent(
            outreach_context={"product_name": "DashFast"}
        )
        payload = {
            "reviewer_name": "Jordan Lee",
            "reviewer_title": "Director of IT",
            "reviewer_company": "FinTech Ltd",
            "reviewer_company_size": "Mid-Market",
            "review_text": "Reporting dashboard takes 10 minutes to load.",
            "review_date": "2026-09-01",
            "competitor_product": "SlowAnalytics",
        }

        result = agent_with_prod_only.analyze(payload)
        email = result["drafted_email"]
        lower_email = email.lower()

        # Must mention product_name
        self.assertIn("DashFast", email)
        # Must NOT claim the product was built for, solves, or improves anything
        self.assertNotIn("built for", lower_email)
        self.assertNotIn("solves", lower_email)
        self.assertNotIn("improves", lower_email)
        self.assertNotIn("guarantee", lower_email)
        self.assertNotIn("faster", lower_email)
        self.assertNotIn("features", lower_email)
        self.assertNotIn("eliminates", lower_email)
        # Exact non-capability phrasing
        self.assertEqual(
            email,
            "Hi Jordan, Navigating reporting and analytics challenges can make tracking progress difficult for teams at FinTech Ltd. "
            "Dashboard takes 10 minutes to load. "
            "Reaching out from DashFast. "
            "Is improving reporting clarity something you are currently evaluating?"
        )
        self.assertEqual(mock_llm.call_count, 1)

    @patch("agent._call_llm")
    def test_approved_value_propositions_appear_without_extra_claims(self, mock_llm):
        """
        Prove that approved value propositions appear deterministically and exact without extra claims,
        paraphrasing, or embellishments.
        """
        mock_analysis = {
            "reviewer_name": "Taylor Swift",
            "reviewer_title": "CTO",
            "reviewer_company": "MediaCorp",
            "reviewer_company_size": "Enterprise",
            "review_text": "Onboarding takes 3 months.",
            "review_date": "2026-09-01",
            "pain_category": "Onboarding Friction",
            "pain_summary": "Onboarding process takes three months.",
            "confidence_score": 0.95,
            "confidence_label": "high",
            "recency_score": 30,
            "seniority_score": 40,
            "pain_intensity_score": 20,
            "total_score": 90,
            "tier": "Tier 1",
            "competitor_product": "OldPlatform",
        }
        mock_llm.return_value = json.dumps(mock_analysis)

        approved_vps = [
            "we provide automated 1-day white-glove migration",
            "dedicated Slack channel with our solutions engineering team",
        ]
        agent_with_vps = AntigravityReviewAgent(
            outreach_context={
                "product_name": "SwiftOnboard",
                "value_propositions": approved_vps,
            }
        )
        payload = {
            "reviewer_name": "Taylor Swift",
            "reviewer_title": "CTO",
            "reviewer_company": "MediaCorp",
            "reviewer_company_size": "Enterprise",
            "review_text": "Onboarding takes 3 months.",
            "review_date": "2026-09-01",
            "competitor_product": "OldPlatform",
        }

        result = agent_with_vps.analyze(payload)
        email = result["drafted_email"]

        self.assertIn("SwiftOnboard", email)
        # Value propositions appear exact and verbatim
        self.assertIn("we provide automated 1-day white-glove migration", email)
        self.assertIn("dedicated Slack channel with our solutions engineering team", email)
        # Check complete deterministic structure
        expected_text = (
            "Hi Taylor, Managing user onboarding friction can quickly create bottlenecks for teams at MediaCorp. "
            "Onboarding process takes three months. "
            "At SwiftOnboard, we provide automated 1-day white-glove migration; "
            "dedicated Slack channel with our solutions engineering team. "
            "Is streamlining the onboarding process currently a priority for your team?"
        )
        self.assertEqual(email, expected_text)
        # No extra claims or hallucinations
        self.assertNotIn("99.9%", email)
        self.assertNotIn("cheapest", email.lower())
        self.assertEqual(mock_llm.call_count, 1)

    def test_different_pain_categories_produce_meaningfully_different_drafts(self):
        """
        Prove that all 11 pain categories produce distinct, tailored drafts with unique hooks and CTAs,
        utilize verified pain_summary where natural, never mention G2, and stay under 100 words.
        """
        all_categories = list(CATEGORY_TEMPLATES.keys())
        drafts = {}
        hooks = set()
        ctas = set()

        sample_summaries = {
            "Onboarding Friction": "User setup took 6 weeks of manual back-and-forth.",
            "Poor Customer Support": "Tickets routinely go unanswered for four business days.",
            "Integration Failures": "CRM connectors fail weekly and drop lead records.",
            "Pricing / Value Mismatch": "Renewal prices jumped 45% without added value.",
            "Performance / Reliability Issues": "Daily server crashes occur during peak traffic.",
            "Missing Features": "No native bulk export functionality is supported.",
            "Steep Learning Curve": "Junior reps take months to navigate the complex menus.",
            "Poor Reporting / Analytics": "Custom dashboard filters yield inaccurate totals.",
            "Clunky UI / UX": "Navigation requires five clicks to reach basic settings.",
            "Lack of Scalability": "System throughput throttles when passing 10k events.",
            "Other": "Legacy system constraints block operational progress.",
        }

        for cat in all_categories:
            data = {
                "reviewer_name": "Jordan Doe",
                "reviewer_company": "Acme Corp",
                "reviewer_company_size": "Mid-Market",
                "pain_category": cat,
                "pain_summary": sample_summaries[cat],
                "tier": "Tier 1",
            }
            email = _generate_fallback_email(data, outreach_context=None)
            self.assertIsNotNone(email)

            # Never mention G2 or any review site
            self.assertNotIn("g2", email.lower())
            self.assertNotIn("review", email.lower())

            # Stay strictly under 100 words
            word_count = len(email.split())
            self.assertLess(word_count, 100, f"Email for {cat} exceeded 100 words: {word_count}")

            # Natural inclusion of pain summary
            self.assertIn(sample_summaries[cat], email)

            # Store and check uniqueness
            drafts[cat] = email
            tmpl = CATEGORY_TEMPLATES[cat]
            hooks.add(tmpl["hook"])
            ctas.add(tmpl["cta"])

        # All 11 emails must be mutually distinct
        self.assertEqual(len(set(drafts.values())), 11)
        # All 11 hooks must be unique
        self.assertEqual(len(hooks), 11)
        # All 11 CTAs must be unique
        self.assertEqual(len(ctas), 11)

    def test_third_person_pain_summary_is_not_inserted_into_email(self):
        """Analyst-style summaries must not appear in recipient-facing copy."""
        data = {
            "reviewer_name": "Jordan Lee",
            "reviewer_company": None,
            "reviewer_company_size": "Small-Business (50 or fewer emp.)",
            "pain_category": "Integration Failures",
            "pain_summary": "The reviewer is experiencing frequent CRM integration failures.",
            "tier": "Tier 2",
        }

        email = _generate_fallback_email(data)

        self.assertNotIn("reviewer", email.lower())
        self.assertNotIn("The reviewer is experiencing", email)

    def test_neutral_fallback_email_when_approved_context_missing(self):
        """When approved outreach context is missing, produce neutral pain-focused draft without product claims."""
        data = {
            "reviewer_name": "Alex Smith",
            "reviewer_company": "CloudCorp",
            "reviewer_company_size": "Mid-Market",
            "pain_category": "Integration Failures",
        }
        email = _generate_fallback_email(data, outreach_context=None)
        # Should be neutral, mentioning pain but making NO product, pricing, or feature claims
        self.assertIn("Hi Alex", email)
        self.assertIn("teams at CloudCorp", email)
        self.assertIn("dealing with integration issues can create operational friction", email.lower())
        self.assertIn("is improving integration reliability currently a priority?", email.lower())
        self.assertNotIn("we guarantee", email.lower())
        self.assertNotIn("our platform delivers", email.lower())
        self.assertNotIn("pricing", email.lower())

    def test_approved_outreach_context_in_fallback_email(self):
        """When approved outreach context is provided, safe fallback can use it."""
        data = {
            "reviewer_name": "Alex Smith",
            "reviewer_company": "CloudCorp",
            "reviewer_company_size": "Mid-Market",
            "pain_category": "Integration Failures",
        }
        ctx = {
            "product_name": "SyncWave",
            "value_propositions": ["we provide zero-maintenance two-way CRM connectors"],
        }
        email = _generate_fallback_email(data, outreach_context=ctx)
        self.assertIn("SyncWave", email)
        self.assertIn("zero-maintenance two-way CRM connectors", email)

    def test_no_context_emails_contain_no_product_references_or_comparative_claims(self):
        """
        Assert that no-context emails across all categories:
        1. Contain no product references (e.g., our product, our platform, our roadmap, we provide).
        2. Contain no comparative capability claims (e.g., more dependable, responsive, intuitive, scalable, modern, transparent).
        3. Do not invent specifics absent from pain_summary (e.g., renewal hikes, inaccurate metrics, downtime).
        4. Every CTA only asks about the buyer's priority, focus, or current evaluation approach.
        """
        prohibited_product_refs = [
            "our product",
            "our platform",
            "our solution",
            "our roadmap",
            "our team",
            "our software",
            "we provide",
            "we offer",
            "we build",
            "we deliver",
            "see how we compare",
            "reaching out from",
            "switch to",
            "in action",
        ]
        prohibited_comparative_claims = [
            "dependable",
            "responsive",
            "intuitive",
            "scalable",
            "modern",
            "transparent",
            "clearer",
            "smoother",
            "faster",
            "cheaper",
            "better",
            "superior",
        ]
        prohibited_invented_specifics = [
            "renewal hike",
            "renewal increase",
            "downtime",
            "latency",
            "crash",
            "inaccurate metric",
            "hand-holding",
            "waiting days",
            "broken data sync",
            "clunky menu",
            "rigid pricing",
            "rigid dashboard",
        ]

        for cat in CATEGORY_TEMPLATES.keys():
            # Test without pain_summary to verify template hook + CTA in isolation
            data_no_summary = {
                "reviewer_name": "Jordan Test",
                "reviewer_company": "Acme Corp",
                "reviewer_company_size": "Mid-Market",
                "pain_category": cat,
                "tier": "Tier 1",
            }
            email = _generate_fallback_email(data_no_summary, outreach_context=None)
            self.assertIsNotNone(email)
            lower_email = email.lower()

            for ref in prohibited_product_refs:
                self.assertFalse(
                    bool(re.search(rf"\b{re.escape(ref)}\b", lower_email)),
                    f"Category '{cat}' email contained prohibited product reference '{ref}': {email}",
                )

            for claim in prohibited_comparative_claims:
                self.assertFalse(
                    bool(re.search(rf"\b{re.escape(claim)}\b", lower_email)),
                    f"Category '{cat}' email contained comparative capability claim '{claim}': {email}",
                )

            for spec in prohibited_invented_specifics:
                self.assertFalse(
                    bool(re.search(rf"\b{re.escape(spec)}\b", lower_email)),
                    f"Category '{cat}' email contained invented specific '{spec}': {email}",
                )

            # Never mention G2 or review sites
            self.assertNotIn("g2", lower_email)
            self.assertNotIn("review", lower_email)

            # CTA must only ask about priority, evaluation, or current approach
            cta = CATEGORY_TEMPLATES[cat]["cta"].lower()
            self.assertTrue(
                any(keyword in cta for keyword in ["priority", "evaluating", "exploring", "focus"]),
                f"Category '{cat}' CTA does not ask about buyer priority or approach: {cta}",
            )
            self.assertTrue(cta.endswith("?"))

    # -----------------------------------------------------------------------
    # TEST-004: Discarded Leads Receive No Emails and No Second LLM Call
    # -----------------------------------------------------------------------
    @patch("agent._call_llm")
    def test_discarded_lead_returns_null_email_and_suppresses_email_llm(self, mock_llm):
        """Discarded leads must return drafted_email as null and spend no LLM call generating one."""
        mock_analysis = {
            "reviewer_name": "Dan Discard",
            "reviewer_title": "Intern",
            "reviewer_company": "SmallCo",
            "reviewer_company_size": "Small-Business",
            "review_text": "Minor typo in the settings menu.",
            "review_date": "2025-01-01",  # > 90 days ago -> 0
            "pain_category": "Clunky UI / UX",
            "pain_summary": "Minor typo in settings.",
            "confidence_score": 0.85,
            "confidence_label": "high",
            "recency_score": 0,
            "seniority_score": 10,
            "pain_intensity_score": 10,
            "total_score": 20,
            "tier": "Discard",
            "competitor_product": "CompetitorZ",
        }
        mock_llm.return_value = json.dumps(mock_analysis)

        payload = {
            "reviewer_name": "Dan Discard",
            "reviewer_title": "Intern",
            "reviewer_company": "SmallCo",
            "reviewer_company_size": "Small-Business",
            "review_text": "Minor typo in the settings menu.",
            "review_date": "2025-01-01",
            "competitor_product": "CompetitorZ",
        }

        result = self.agent.analyze(payload)

        self.assertEqual(result["tier"], "Discard")
        self.assertIsNone(result["drafted_email"])
        self.assertEqual(mock_llm.call_count, 1)

    @patch("agent._call_llm")
    def test_discarded_lead_via_api_endpoint_serializes_null(self, mock_llm):
        """FastAPI endpoint /analyze must serialize drafted_email as JSON null."""
        mock_analysis = {
            "reviewer_name": "Dan Discard",
            "reviewer_title": "Intern",
            "reviewer_company": "SmallCo",
            "reviewer_company_size": "Small-Business",
            "review_text": "Old minor feedback.",
            "review_date": "2025-01-01",
            "pain_category": "Other",
            "pain_summary": "No serious pain.",
            "confidence_score": 0.75,
            "confidence_label": "high",
            "recency_score": 0,
            "seniority_score": 10,
            "pain_intensity_score": 10,
            "total_score": 20,
            "tier": "Discard",
            "competitor_product": "CompetitorZ",
        }
        mock_llm.return_value = json.dumps(mock_analysis)

        payload = {
            "reviewer_name": "Dan Discard",
            "reviewer_title": "Intern",
            "reviewer_company": "SmallCo",
            "reviewer_company_size": "Small-Business",
            "review_text": "Old minor feedback.",
            "review_date": "2025-01-01",
            "competitor_product": "CompetitorZ",
        }

        response = self.client.post("/analyze", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["tier"], "Discard")
        self.assertIsNone(data["drafted_email"])
        # Ensure it literally serializes as null in the raw response text
        self.assertIn('"drafted_email":null', response.text.replace(" ", ""))

    @patch("agent._call_llm")
    def test_tier_1_and_tier_2_generate_email_deterministically_after_qualification(self, mock_llm):
        """Tier 1 and Tier 2 leads must generate an outreach email deterministically with only 1 LLM call."""
        # Test Tier 1
        tier1_analysis = {
            "reviewer_name": "Alice T1",
            "reviewer_title": "VP of Product",
            "reviewer_company": "BigTech",
            "reviewer_company_size": "Enterprise",
            "review_text": "Unusable integrations.",
            "review_date": "2026-09-01",
            "pain_category": "Integration Failures",
            "pain_summary": "Unusable integrations.",
            "confidence_score": 0.9,
            "confidence_label": "high",
            "recency_score": 30,
            "seniority_score": 40,
            "pain_intensity_score": 20,
            "total_score": 90,
            "tier": "Tier 1",
            "competitor_product": "CompA",
        }
        mock_llm.return_value = json.dumps(tier1_analysis)

        result_t1 = self.agent.analyze({"reviewer_name": "Alice T1", "reviewer_company": "BigTech"})
        self.assertEqual(result_t1["tier"], "Tier 1")
        self.assertIsNotNone(result_t1["drafted_email"])
        self.assertIn("teams at BigTech", result_t1["drafted_email"])
        self.assertEqual(mock_llm.call_count, 1)

        # Test Tier 2
        mock_llm.reset_mock()
        tier2_analysis = {
            "reviewer_name": "Bob T2",
            "reviewer_title": "Operations Manager",
            "reviewer_company": "MidTech",
            "reviewer_company_size": "Mid-Market",
            "review_text": "Reporting is confusing.",
            "review_date": "2026-08-15",
            "pain_category": "Poor Reporting / Analytics",
            "pain_summary": "Confusing analytics.",
            "confidence_score": 0.8,
            "confidence_label": "high",
            "recency_score": 20,
            "seniority_score": 20,
            "pain_intensity_score": 10,
            "total_score": 50,
            "tier": "Tier 2",
            "competitor_product": "CompA",
        }
        mock_llm.return_value = json.dumps(tier2_analysis)

        result_t2 = self.agent.analyze({"reviewer_name": "Bob T2", "reviewer_company": "MidTech"})
        self.assertEqual(result_t2["tier"], "Tier 2")
        self.assertIsNotNone(result_t2["drafted_email"])
        self.assertIn("teams at MidTech", result_t2["drafted_email"])
        self.assertEqual(mock_llm.call_count, 1)

    # -----------------------------------------------------------------------
    # TEST-006: Benefits vs Dislikes Structured Inputs
    # -----------------------------------------------------------------------
    @patch("agent._call_llm")
    def test_structured_inputs_prioritize_dislikes_over_likes(self, mock_llm):
        """Explicit dislikes are prioritized when identifying buyer pain."""
        mock_analysis = {
            "reviewer_name": "Charlie User",
            "reviewer_title": "Head of Engineering",
            "reviewer_company": "DevHouse",
            "reviewer_company_size": "Mid-Market",
            "review_text": "Dislikes: API rate limits block us. | Likes: UI is slick.",
            "review_date": "2026-09-02",
            "pain_category": "Performance / Reliability Issues",
            "pain_summary": "API rate limits prevent deployment scaling.",
            "confidence_score": 0.92,
            "confidence_label": "high",
            "recency_score": 30,
            "seniority_score": 30,
            "pain_intensity_score": 20,
            "total_score": 80,
            "tier": "Tier 1",
            "competitor_product": "CompB",
        }
        mock_llm.return_value = json.dumps(mock_analysis)

        payload = {
            "reviewer_name": "Charlie User",
            "reviewer_title": "Head of Engineering",
            "reviewer_company": "DevHouse",
            "reviewer_company_size": "Mid-Market",
            "likes": "UI is slick and easy to navigate.",
            "dislikes": "API rate limits block our production deployment pipeline.",
            "problemsSolved": "Centralized our task tracking.",
            "competitor_product": "CompB",
        }

        result = self.agent.analyze(payload)
        self.assertEqual(result["pain_category"], "Performance / Reliability Issues")
        self.assertEqual(result["tier"], "Tier 1")
        self.assertEqual(mock_llm.call_count, 1)

        # Verify structured inputs were passed to the prompt
        analysis_prompt_arg = mock_llm.call_args_list[0][0][0]
        self.assertIn("STRUCTURED REVIEW FEEDBACK", analysis_prompt_arg)
        self.assertIn("API rate limits block our production deployment pipeline", analysis_prompt_arg)
        self.assertIn("UI is slick and easy to navigate", analysis_prompt_arg)
        self.assertIn("Centralized our task tracking", analysis_prompt_arg)

    @patch("agent._call_llm")
    def test_benefits_and_solved_problems_with_no_dislikes_discarded(self, mock_llm):
        """When review only expresses benefits / solved problems and no unresolved pain, it is Discard."""
        mock_analysis = {
            "reviewer_name": "Happy Customer",
            "reviewer_title": "Director of IT",
            "reviewer_company": "HappyCorp",
            "reviewer_company_size": "Enterprise",
            "review_text": "Likes: It solved all our onboarding issues. | Dislikes: None.",
            "review_date": "2026-09-02",
            "pain_category": "Other",
            "pain_summary": "Reviewer reported no unresolved pain; onboarding problems were solved by competitor.",
            "confidence_score": 0.95,
            "confidence_label": "high",
            "recency_score": 30,
            "seniority_score": 30,
            "pain_intensity_score": 0,  # Solved problem is NOT buyer pain!
            "total_score": 60,
            "tier": "Discard",
            "competitor_product": "CompC",
        }
        mock_llm.return_value = json.dumps(mock_analysis)

        payload = {
            "reviewer_name": "Happy Customer",
            "reviewer_title": "Director of IT",
            "reviewer_company": "HappyCorp",
            "reviewer_company_size": "Enterprise",
            "likes": "It solved all our onboarding issues and support is great.",
            "dislikes": "Nothing, it has been wonderful.",
            "problemsSolved": "Solved our initial onboarding bottleneck completely.",
            "competitor_product": "CompC",
        }

        result = self.agent.analyze(payload)
        self.assertEqual(result["tier"], "Discard")
        self.assertIsNone(result["drafted_email"])
        self.assertEqual(mock_llm.call_count, 1)

    @patch("agent._call_llm")
    def test_backward_compatibility_with_only_review_text(self, mock_llm):
        """When only review_text is provided without structured likes/dislikes, agent functions normally."""
        mock_analysis = {
            "reviewer_name": "Legacy Payload",
            "reviewer_title": "CTO",
            "reviewer_company": "LegacyCorp",
            "reviewer_company_size": "Enterprise",
            "review_text": "Constant server crashes.",
            "review_date": "2026-09-01",
            "pain_category": "Performance / Reliability Issues",
            "pain_summary": "Constant server crashes.",
            "confidence_score": 0.9,
            "confidence_label": "high",
            "recency_score": 30,
            "seniority_score": 40,
            "pain_intensity_score": 30,
            "total_score": 100,
            "tier": "Tier 1",
            "competitor_product": "CompOld",
        }
        mock_llm.return_value = json.dumps(mock_analysis)

        payload = {
            "reviewer_name": "Legacy Payload",
            "reviewer_title": "CTO",
            "reviewer_company": "LegacyCorp",
            "reviewer_company_size": "Enterprise",
            "review_text": "Constant server crashes.",
            "review_date": "2026-09-01",
            "competitor_product": "CompOld",
        }

        result = self.agent.analyze(payload)
        self.assertEqual(result["tier"], "Tier 1")
        self.assertIsNotNone(result["drafted_email"])
        self.assertIn("teams at LegacyCorp", result["drafted_email"])
        self.assertEqual(mock_llm.call_count, 1)


if __name__ == "__main__":
    unittest.main()
