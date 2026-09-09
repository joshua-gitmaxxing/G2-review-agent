# Competitive Displacement Outbound System — Build Brief

**Project type:** GTM Engineering portfolio build  
**Status:** Reconstructed and manually tested end to end  
**Trigger model:** Manual Trigger for controlled runs, demo recording, and credit management  
**Primary outcome:** Turn competitor-review signals into qualified, enrichment-ready contacts with relevant drafted outreach stored in HubSpot.

## 1. Overview

This system uses publicly available G2 reviews of defined competitor products as a competitive-displacement signal.

It collects reviews, identifies the specific pain behind them, scores the potential lead, routes uncertain cases for human review, enriches only leads with sufficient identity data, and stores qualified contacts with drafted outreach in HubSpot.

The system is intentionally designed to avoid two common failure modes:

- spending enrichment credits on incomplete reviewer identities
- generating confident-looking outreach from weak or ambiguous review signals

## 2. Current architecture

```text
Manual Trigger
  -> Wake Render middleware
  -> Wait for service readiness
  -> Apify G2 review collection
  -> Review-ID deduplication across prior executions
  -> FastAPI + Gemini analysis
  -> Discard filter
  -> Confidence routing
      |-- Low confidence -> Slack human review
      |     |-- Approve -> return to qualification and enrichment
      |     `-- Reject -> Google Sheets discard log
      `-- High confidence -> qualification and enrichment
  -> Identity eligibility check
      |-- Insufficient identity -> Google Sheets log
      `-- Eligible -> Clay enrichment
  -> Clay callback webhook
  -> Normalize enrichment response
  -> Email check
      |-- Work email found -> HubSpot contact
      `-- No work email -> Google Sheets log
```

## 3. Tool stack

| Tool | Role |
| --- | --- |
| n8n | Orchestration, routing, deduplication, integrations, and exception handling |
| Apify | G2 review collection |
| FastAPI | Middleware API between n8n and Gemini |
| Gemini | Pain extraction, classification, scoring, and outreach drafting |
| Render | FastAPI middleware deployment |
| Slack | Human review for low-confidence reviews |
| Clay | Company-domain and work-email enrichment |
| HubSpot | Contact and drafted-email storage |
| Google Sheets | Discard, identity, and email-not-found logging |

## 4. Analysis and qualification logic

The middleware receives one review at a time and returns structured data for n8n to route.

### Pain categories

- Onboarding Friction
- Poor Customer Support
- Integration Failures
- Pricing / Value Mismatch
- Performance / Reliability Issues
- Missing Features
- Steep Learning Curve
- Poor Reporting / Analytics
- Clunky UI / UX
- Lack of Scalability
- Other

### Scoring

| Dimension | Maximum score |
| --- | ---: |
| Role seniority | 40 |
| Review recency | 30 |
| Pain intensity | 30 |
| Total | 100 |

| Tier | Score |
| --- | --- |
| Tier 1 | 70-100 |
| Tier 2 | 40-69 |
| Discard | Below 40 |

### Confidence routing

- **High confidence:** score of 0.70 or above. The lead continues automatically.
- **Low confidence:** below 0.70. Reviews with genuine pain go to Slack for a human decision instead of being automatically enriched or discarded.

### Identity protection

Before Clay, the workflow checks for a usable full name and real company identity. G2 company-size labels such as `Mid-Market` and `Enterprise` are not treated as company names.

Leads without enough identity data are logged rather than sent to enrichment.

## 5. Outreach safeguards

Drafted outreach is based on the identified pain but does not mention the G2 review or imply that the reviewer was found through it.

The backend includes safeguards to:

- suppress drafted emails for discarded leads
- keep low-confidence leads available for human review when real pain exists
- prevent unsupported product or comparative claims
- avoid treating missing or company-size data as an employer name
- generate different drafts for materially different pain categories

## 6. Data returned by the middleware

The analysis response contains the reviewer identity fields, review data, pain analysis, confidence data, scoring data, tier, drafted email, and competitor product.

Key fields include:

```text
reviewer_name
reviewer_title
reviewer_company
reviewer_company_size
review_text
review_date
pain_category
pain_summary
confidence_score
confidence_label
recency_score
seniority_score
pain_intensity_score
total_score
tier
drafted_email
competitor_product
```

## 7. Testing completed

The reconstructed workflow was manually tested from the Manual Trigger through each route and final destination.

Validated areas include:

- Render wake-and-wait handling for cold starts
- G2 review collection and review-ID deduplication
- Gemini analysis, score validation, tiering, and email drafting
- discard routing
- high- and low-confidence routing
- Slack approval and rejection paths
- identity eligibility filtering before Clay
- Clay enrichment callback normalization
- HubSpot contact creation when work email exists
- Google Sheets logging for rejected, insufficient-identity, and email-not-found outcomes

The repository also contains regression tests for backend safeguards and company-field handling.

## 8. Operating decision

The workflow remains inactive between demos and uses a Manual Trigger by design. This keeps review collection and Clay enrichment spend under deliberate control.

Activating it later would make the production Clay callback webhook persistently available. It does not need to be active for the current portfolio demo.

## 9. Current limitations and future extension

HubSpot is the final storage destination in the current version. It stores qualified contacts and drafted emails; it does not automatically send outbound messages.

A sequencing tool can be added later after the system is being used for real outreach and there is a clear sending, deliverability, and approval process.

## 10. Repository sources of truth

- `agent.py` — Gemini analysis, validation, scoring, and outreach generation
- `main.py` — FastAPI `/analyze` endpoint
- `workflow-export.json` — final n8n workflow export
- `test_backend_fixes.py` — regression tests for backend safeguards
- `test_company_fields.py` — company and company-size handling tests
- `smoke_test.py` — schema and endpoint smoke test
