# Competitive Displacement Outbound System

A GTM workflow that turns competitor-review signals into qualified, enrichment-ready contacts with personalized outreach drafts.

Built as a portfolio project to demonstrate a complete signal-to-pipeline system: collection, AI analysis, qualification, human review, enrichment, CRM routing, and test coverage.

## Workflow
<img width="1920" height="1080" alt="g2-system" src="https://github.com/user-attachments/assets/ebe5685b-3a86-4c9d-936c-54e309fb21db" />

## The problem

Competitor reviews can reveal real friction: poor reporting, pricing concerns, onboarding issues, unreliable integrations, and other reasons a buyer may be open to an alternative.

The difficult part is turning that raw signal into something useful without manually reading reviews, guessing who is worth pursuing, enriching weak identities, or writing generic outreach.

This system handles that workflow in one controlled pipeline.

## What it does

1. Pulls reviews from defined competitor G2 URLs through Apify.
2. Deduplicates reviews by review ID across previous workflow executions.
3. Sends each review to a FastAPI middleware service running Gemini.
4. Extracts the pain, scores the lead, assigns a tier, and drafts relevant outreach.
5. Removes low-value leads before enrichment.
6. Routes low-confidence but potentially useful reviews to Slack for human approval or rejection.
7. Checks whether the reviewer has enough identity information for enrichment.
8. Sends eligible leads to Clay for domain and work-email enrichment.
9. Receives and normalizes Clay's callback.
10. Creates a HubSpot contact when a work email is found, or logs the outcome to Google Sheets when it is not.

## Architecture

```text
Manual Trigger
  -> Wake Render
  -> Wait for middleware
  -> Apify G2 review collection
  -> Review-ID deduplication
  -> FastAPI + Gemini analysis
  -> Discard filter
  -> Confidence routing
      |-- Low confidence -> Slack human review
      |     |-- Approve -> qualification and enrichment
      |     `-- Reject -> discard log
      `-- High confidence -> qualification and enrichment
  -> Clay enrichment
  -> Clay callback webhook
  -> Email check
      |-- Work email found -> HubSpot contact
      `-- No work email -> Google Sheets log
```

## Qualification logic

Each review is scored across three dimensions:

| Dimension | Maximum score |
| --- | ---: |
| Role seniority | 40 |
| Review recency | 30 |
| Pain intensity | 30 |
| Total | 100 |

Lead tiers:

- Tier 1: 70-100
- Tier 2: 40-69
- Discard: below 40

Confidence is binary:

- High confidence: 0.70 and above
- Low confidence: below 0.70, routed to Slack for review when genuine pain exists

A lead must also have a usable full name and company identity before Clay enrichment. This prevents enrichment spend on incomplete G2 reviewer data.

## Stack

| Tool | Role |
| --- | --- |
| n8n | Workflow orchestration, routing, deduplication, and integrations |
| Apify | G2 review collection |
| FastAPI | Middleware API between n8n and Gemini |
| Gemini | Pain extraction, classification, scoring, and outreach drafting |
| Render | Middleware deployment |
| Slack | Human review for low-confidence leads |
| Clay | Domain and work-email enrichment |
| HubSpot | Contact storage and drafted-email destination |
| Google Sheets | Exception and rejection logging |

## Current status

The system was reconstructed from an existing build and manually tested end to end.

- Manual Trigger is intentional for controlled testing, demo recording, and credit management.
- The workflow remains inactive between demos. It can be activated later if persistent live webhook handling is needed.
- HubSpot stores qualified contacts and drafted emails.
- Sending is intentionally not automated. A sequencing tool can be connected later if needed.

## Repository structure

```text
agent.py                     Gemini analysis, validation, scoring, and email logic
main.py                      FastAPI application and /analyze endpoint
workflow-export.json         Final n8n workflow export
render.yaml                  Render deployment configuration
requirements.txt             Python dependencies
smoke_test.py                Schema and endpoint smoke test
test_backend_fixes.py        Regression tests for analysis and outreach safeguards
test_company_fields.py       Tests for company-name and company-size handling
```

## Running the backend locally

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_key_here
```

Start the API:

```bash
uvicorn main:app --reload
```

The analysis endpoint is available at:

```text
POST /analyze
```

## Tests

Run the regression tests:

```bash
python -m unittest test_backend_fixes
python -m unittest test_company_fields
```

Run the smoke test without making a live model call:

```bash
python smoke_test.py --dry-run
```

## Notes

- The workflow export is provided for architecture and portfolio review. Importing it requires your own n8n credentials and connected tools.
