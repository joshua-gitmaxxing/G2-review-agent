# G2 Review Signal Workflow — Build Brief
**Project Type:** GTM Engineering Portfolio Build  
**Goal:** Demo-ready competitive displacement agent with a recorded walkthrough video  
**Status:** Planning complete. Ready to build.

---

## 1. What This Workflow Does

This is a competitive displacement agent. It monitors G2 reviews for defined competitor products, extracts the pain behind each review, scores and tiers the leads, enriches their contact information, drafts personalized outreach based on their specific pain, and pushes launch-ready contacts into HubSpot.

The workflow solves a real GTM problem: companies know their competitors have frustrated customers, but the process of finding those customers, identifying what they're unhappy about, locating their contact details, and writing a relevant outreach email is manual, slow, and doesn't scale. This workflow automates that entire chain end-to-end.

The signal is G2 review activity. This was chosen deliberately over job posting signals because job posting workflows are oversaturated in GTM portfolios. G2 review-based competitive displacement is differentiated, more sophisticated in its reasoning layer, and directly maps to a documented outbound strategy used by B2B SaaS sales teams.

---

## 2. Full Stack

| Tool | Role | Plan |
|------|------|------|
| n8n | Orchestration — triggers, data movement, node logic | Self-hosted, free |
| Apify | G2 review scraper | Pay-per-run |
| Antigravity (Google) | Agentic reasoning — pain extraction, scoring, campaign drafting | Paid subscription |
| Claude Sonnet 4.6 / Opus 4.6 | LLM inside Antigravity agent | Via Antigravity subscription |
| Python middleware service | Receives POST from n8n, runs Antigravity SDK agent, returns JSON | Built by Antigravity, deployed on Render or Railway |
| Clay | Waterfall contact enrichment | Free tier (100 credits/month) — activate 2-week premium trial for demo recording only |
| HubSpot | Output destination — stores enriched contacts and drafted emails | Free plan |
| Slack | Human-in-the-loop notifications for low confidence reviews | Free |
| Google Sheets | Discard log for low confidence reviews that a human rejects | Free |

---

## 3. Architecture Overview

```
Apify (G2 scraper)
        ↓
n8n (receives structured review data)
        ↓
HTTP Request → Python middleware service (Render/Railway)
        ↓
Antigravity SDK agent runs Claude model
  - Extracts pain
  - Tags pain category
  - Scores confidence
  - Scores lead (recency + seniority + pain intensity)
  - Assigns tier
  - Drafts outreach email
        ↓
Returns structured JSON → n8n
        ↓
Confidence check (n8n)
  ├── High confidence → continues to Clay enrichment
  └── Low confidence → Slack notification (human review)
              ├── Approve → re-enters pipeline via n8n webhook → Clay enrichment
              └── Discard → logs to Google Sheet → workflow ends
        ↓
Clay enrichment (reviewer name + company → domain → email)
        ↓
HubSpot (contact created with all fields populated)
```

---

## 4. Handoff Pattern

n8n does all data plumbing. When the workflow reaches any step requiring reasoning, n8n sends the payload to the Python middleware service via HTTP POST. The Python service passes it to the Antigravity SDK agent. The agent runs the Claude model, completes its task, and the Python service returns a structured JSON response to n8n. n8n picks up and continues.

This is the core integration to validate first before any other build step begins.

---

## 5. Agent Stages

### Stage 1 — Signal Collection (Apify + n8n)
- n8n triggers Apify actor on a schedule
- Apify scrapes G2 reviews for 2–3 defined competitor products
- Returns structured review data: reviewer name, title, company, rating, review body, date
- n8n receives the data and passes it downstream

### Stage 2 — Pain Extraction (Antigravity Agent)
The agent reads the raw review text and extracts:
- The specific pain stated or implied
- The feature or outcome the reviewer wished existed
- The emotional tone (frustrated, neutral, mildly dissatisfied)
- A pain category tag from the defined list (see Section 7)
- A one to two sentence plain English pain summary (used in Slack notifications)
- A confidence score (float 0.0–1.0) representing how clearly the review maps to a single pain category

### Stage 3 — Lead Scoring (Antigravity Agent)
The agent scores each lead across three dimensions:

**Recency (max 30 points)**
- Last 7 days → 30
- 8–30 days → 20
- 31–90 days → 10
- Older than 90 days → 0

**Role Seniority (max 40 points)**
- C-suite / VP / Owner → 40
- Director / Head of → 30
- Manager → 20
- Individual contributor → 10
- Role unknown → 5

**Pain Intensity (max 30 points)**
- Explicit frustration, mentions switching or cancelling → 30
- Clear criticism of a specific feature or process → 20
- Mild dissatisfaction, mixed review → 10
- Mostly positive with minor complaint → 0

**Total score:** 0–100

**Tier assignment:**
- Tier 1 → 70–100
- Tier 2 → 40–69
- Discard → below 40

Discard leads are dropped after scoring. They do not proceed to enrichment.

### Stage 4 — Confidence Check (n8n)
After the agent returns its JSON, n8n evaluates the confidence score:

- **High confidence:** 0.70 and above → lead flows automatically to Clay enrichment
- **Low confidence:** below 0.70 → lead is held, Slack notification fires

The agent must commit to one label. There is no medium tier. This is intentional — forcing a binary decision produces more trustworthy categorizations.

### Stage 5 — Human-in-the-Loop (Slack + n8n)
Fires only for low confidence leads.

Slack message contains:
- Reviewer name, title, company
- Review text
- Pain category the agent was leaning toward
- Confidence score
- Approve and Discard buttons

**If Approve:** Slack sends webhook to n8n → n8n re-enters the lead into the pipeline → continues to Clay enrichment

**If Discard:** Slack sends webhook to n8n → n8n writes the lead to a Google Sheet discard log (reviewer name, company, review text, leaning category, confidence score, timestamp) → workflow ends

### Stage 6 — Enrichment (Clay)
Reviewer name + company name → company domain → email lookup

Waterfall order:
1. Hunter.io
2. Apollo
3. Prospeo

If all three fail → lead is flagged as unenrichable and skipped. Does not get pushed to HubSpot.

Activate Clay 2-week premium trial for this stage. Do not activate the trial until all other stages are working and tested. Use the trial window exclusively for enrichment runs and demo recording.

### Stage 7 — Campaign Generation (Antigravity Agent)
The agent drafts the outreach email using the extracted pain as the hook.

Rules the agent must follow:
- Never reference the G2 review directly or imply the reviewer was found through a review
- Open with the implied pain, not a generic opener
- Tone: direct, peer-to-peer, not salesy
- Length: 100 words maximum
- No subject line generated at this stage — that is a human decision at launch
- Two leads with different pain categories must produce meaningfully different emails — this is the key demo moment

### Stage 8 — HubSpot Push (n8n)
n8n creates a new contact in HubSpot via API with the following fields:

**Standard fields:**
- First name
- Last name
- Company name
- Email address (from Clay enrichment)

**Custom properties (5 of 10 free plan limit used):**
- Pain Category
- Tier
- Confidence Label
- Drafted Email Body
- Competitor Product

HubSpot is a storage layer only. It performs no automation. All logic lives in n8n. When a human wants to launch outreach, they go into HubSpot, filter by Tier 1, read the drafted email on the contact record, and copy it into their sequencing tool manually.

---

## 6. JSON Schema

This is the structured output the Python middleware service returns to n8n after the Antigravity agent completes its reasoning. Every downstream step depends on this schema.

```json
{
  "reviewer_name": "string",
  "reviewer_title": "string",
  "reviewer_company": "string",
  "review_text": "string",
  "review_date": "YYYY-MM-DD",
  "pain_category": "string",
  "pain_summary": "string",
  "confidence_score": "float (0.0 - 1.0)",
  "confidence_label": "high | low",
  "recency_score": "integer (0-30)",
  "seniority_score": "integer (0-40)",
  "pain_intensity_score": "integer (0-30)",
  "total_score": "integer (0-100)",
  "tier": "Tier 1 | Tier 2 | Discard",
  "drafted_email": "string",
  "competitor_product": "string"
}
```

---

## 7. Pain Categories

The agent classifies each review into exactly one of the following categories. If the review does not clearly map to any of them, it must use "Other."

1. Onboarding Friction
2. Poor Customer Support
3. Integration Failures
4. Pricing / Value Mismatch
5. Performance / Reliability Issues
6. Missing Features
7. Steep Learning Curve
8. Poor Reporting / Analytics
9. Clunky UI / UX
10. Lack of Scalability
11. Other

---

## 8. Python Middleware Service Spec

This service is the bridge between n8n and the Antigravity SDK agent. Antigravity should write this service.

**Requirements:**
- Language: Python
- Framework: FastAPI (preferred) or Flask
- Exposes a single POST endpoint (e.g. `/analyze`)
- Accepts a JSON payload containing the raw review data from Apify
- Passes the payload to the Antigravity SDK agent
- Agent runs the Claude Sonnet 4.6 or Opus 4.6 model
- Returns the structured JSON schema defined in Section 6
- Deployed on Render or Railway
- Must be reachable via public URL for n8n to call

**The service must handle:**
- Malformed or incomplete review data (missing title, unknown company) without crashing
- Graceful error responses that n8n can interpret and route appropriately

---

## 9. Build Order

Follow this order exactly. Do not move to the next step until the current one is confirmed working.

**Step 1 — Validate the handshake**
Build a minimal Python service that receives a POST request with dummy review data and returns a hardcoded JSON response matching the schema. Deploy to Render or Railway. Call it from n8n's HTTP Request node. Confirm the full round trip works before writing any real agent logic.

**Step 2 — Build and test the agent reasoning layer**
Wire the real Antigravity SDK agent into the Python service. Test pain extraction, scoring, confidence scoring, tiering, and campaign drafting against real G2 review text. Confirm the JSON output matches the schema exactly.

**Step 3 — Build the n8n workflow**
Wire all n8n nodes: Apify trigger → HTTP Request to Python service → confidence check → Slack notification (low confidence path) → Google Sheet discard log → Clay enrichment → HubSpot push. Test each node in isolation before running the full flow.

**Step 4 — Activate Clay trial**
Only activate when Steps 1–3 are fully working. Use the 2-week window for enrichment testing and demo recording only.

**Step 5 — Record demo walkthrough**
Use real G2 reviews from real competitor products. Show the full pipeline running live. Highlight the pain extraction producing different emails for different pain categories — this is the key moment.

**Step 6 — Package as portfolio piece**
Write-up, LinkedIn post, joshuafubara.dev portfolio page.

---

## 10. Constraints and Known Limits

- HubSpot free plan: 1,000 contacts, 10 custom properties (5 used by this workflow), no native automation, API capped at 100 calls per 10 seconds. All sufficient for demo scale.
- Clay free tier: 100 credits/month. Not enough for production volume. Premium trial handles the demo.
- The Python middleware service must be running and reachable during the demo. Deploy it before recording and confirm it is live.
- The Antigravity agent must never reference the G2 review in drafted outreach. This is a hard rule, not a preference.
- Confidence scoring is binary: high (0.70 and above) or low (below 0.70). No middle tier. This is intentional.
