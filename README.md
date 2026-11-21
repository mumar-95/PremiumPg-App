# Premium PG Assistant

LLM-powered Streamlit assistant for room-seeking visitors and PG residents, with Groq RAG over the house policy document.

## Features

- **Find a Room** — filter by sharing, AC, food plan, amenities, and budget; only rooms with free beds
- **Booking Request** — move-in date + contact details; session-only `REQ-…` IDs with Pending status
- **My Rent** — resident self-lookup by nickname **and** room (own rent/due date/paid status only)
- **Policy / FAQ Chat** — answers grounded in the policy DOCX with section citations; manager referral when uncovered
- **Complaint Tickets** — category, priority, description; session-only `TKT-…` with Open / In Progress / Closed

## Privacy

- Never exposes one resident’s details to another (no directory; dual-key self-lookup only)
- Masks phone numbers in UI and chat output
- Never asks for government ID proofs or numbers
- Resident rent CSV is used only for My Rent self-lookup — never injected into policy chat RAG

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Set your Groq API key (choose one):

**Option A — environment variable**

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_..."
```

**Option B — Streamlit secrets**

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_..."
```

## Run

**Important (Windows):** do not run bare `streamlit` if it points at another Python (e.g. 3.14). Use the project venv:

```powershell
.\.venv\Scripts\activate
python -m streamlit run app.py
```

Or:

```powershell
.\run.ps1
```

App URL: http://localhost:8501

Policy retrieval uses a lightweight local section index (no model download). Groq is called only when you ask a chat question.

## Data

| File | Role |
|------|------|
| `documents/pg_rooms.csv` | Room inventory & availability |
| `documents/Mens_PG_Resident_Policy.docx` | Policy RAG source |
| `documents/synthetic_residents_rent.csv` | Resident rent self-lookup (My Rent tab); never listed in chat |

Bookings and complaints live in browser session state only and reset on refresh.
