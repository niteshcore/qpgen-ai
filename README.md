# QPGen — Smart Question Paper Generator

**QPGen** is a modern, AI-powered question bank and paper generation studio designed for educators to easily manage their question repositories and generate balanced exam papers based on Bloom's Taxonomy.

<p align="center">
  <img src="frontend/assets/hero_preview.png" width="800" alt="QPGen Hero Section">
  <br>
  <img src="frontend/assets/stats_preview.png" width="800" alt="QPGen Stats and Workflow">
  <br>
  <img src="frontend/assets/features_preview.png" width="800" alt="QPGen Features">
  <br>
  <img src="frontend/assets/cta_preview.png" width="800" alt="QPGen CTA">
</p>

## ✨ Core Features

### Question bank
- **Structured bank**: 400+ questions across 10 subjects, tagged by subject, topic, Bloom's level (Remember → Create), difficulty, marks and type (MCQ / Short / Long).
- **Semantic search**: every question is embedded with Gemini (`gemini-embedding-001`, 768-d) and stored in **pgvector** on Postgres (JSON on local SQLite). Search by meaning, not just keywords.
- **Duplicate detection**: new and AI-generated questions are compared against the bank; near-duplicates (cosine ≥ 0.92, calibrated on the seeded bank) trigger a warning before saving.
- **Off-request indexing**: embeddings are computed in the background, so saving a question stays fast.

### AI generation
- **Question generation** with Google Gemini, tuned to subject, difficulty and Bloom's level.
  - A 3-attempt validation loop ensures structured, non-empty JSON.
  - Temperature follows difficulty (Easy 0.3 / Medium 0.5 / Hard 0.8).
  - Structured errors when generation constraints fail.
- **RAG-grounded generation from course material**: upload a syllabus/notes PDF; it is chunked per page and embedded (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`). Questions are generated strictly from the retrieved passages.
  - Citations are verified, not trusted: the model names a passage number and the real page is looked up from it. A bad citation drops the citation, not the question.
  - Saved questions carry `source_document_id` / `source_page`, shown in the UI as "Grounded in *file*, page *N*".

### Paper generation
- **Smart selection algorithm** with three modes: custom marks distribution (exact count per mark value), Bloom's/difficulty-weighted selection, and a backfill pass that tops up remaining marks from the subject pool. Optional MCQ cap.
- **AI-syllabus mode** (`/api/papers/generate-ai-syllabus`) generates the questions a paper needs from a syllabus.
- **Paper editor**: review and edit the question list before finalising.
- **PDF export** (ReportLab), generated on demand.

### Community library
- **Public library** (`/library.html`): papers a creator chooses to share are browsable and downloadable with no account. Filter by subject, marks and question type.
- **Private by default**: the public blueprint only exposes papers with `is_public = true`; nothing behind auth leaks through it. Owners toggle sharing from their dashboard.
- **Live stats** on the landing page and library, computed from the database.

### Access control and operations
- **RBAC**: 20 fine-grained permissions (e.g. `questions.create`, `papers.download_pdf`) grouped into four roles: `admin`, `teacher`, `viewer`, and an open self-serve `user`. Enforced with `@require_permission` / `@require_role` decorators.
- **Teacher subject scoping**: admins assign subjects to teachers; teachers can request more, which admins approve or reject.
- **Audit log**: sensitive actions (login, question and paper changes, sharing) are recorded; logging never breaks the request.
- **JWT authentication** with bcrypt password hashing and auto-redirect on token expiry.
- **Admin panel** for teachers, subject assignments and requests.
- **Migrations** with Alembic (Flask-Migrate); the chain is verified against a fresh Postgres database.
- **CI**: GitHub Actions runs the pytest suite on every push and PR.

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Backend | Python 3.11+, Flask 3, Flask-SQLAlchemy, Flask-Migrate (Alembic), Marshmallow |
| Database | PostgreSQL on Supabase (production), SQLite (local) |
| Vector search | pgvector (HNSW index, cosine distance), Gemini embeddings |
| AI | Google Gemini (`gemini-flash-latest`, `gemini-embedding-001`) |
| Auth | Flask-JWT-Extended, bcrypt, custom RBAC |
| Documents | ReportLab (PDF export), pypdf (PDF ingestion), pandas/openpyxl (bulk upload) |
| Frontend | Vanilla HTML, CSS and ES6+ JavaScript (no framework) |
| Deploy / CI | Vercel (serverless Flask + static frontend), GitHub Actions |

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- A Google AI Studio API key (the free tier is enough)

### Backend setup
1. Create a virtual environment and install dependencies. `requirements.txt` is at the **repo root**:
   ```bash
   python -m venv backend/venv
   source backend/venv/bin/activate   # Windows: backend\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Configure environment variables:
   ```bash
   cp backend/.env.example backend/.env
   ```
   Then fill in `backend/.env`:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   SECRET_KEY=...            # python -c "import secrets; print(secrets.token_hex(32))"
   JWT_SECRET_KEY=...
   # DATABASE_URL=postgresql://...   # omit to use local SQLite
   ```
3. Start the server from `backend/`. On an empty database it creates the schema, roles, permissions, subjects and the admin account:
   ```bash
   cd backend
   python run.py
   ```
   The API is at `http://127.0.0.1:5000/api`, and the server also serves the frontend at `http://127.0.0.1:5000`.
4. Seed the question bank (first setup only), then embed it for semantic search:
   ```bash
   python seed_questions.py
   python backfill_embeddings.py --report     # re-runs only embed new/edited questions
   ```
5. Optional: fill the public library with sample papers (the server must be running; the AI-syllabus ones call Gemini):
   ```bash
   python seed_sample_papers.py
   ```

### Migrations
```bash
cd backend && flask db upgrade
```
The vector migrations run `CREATE EXTENSION IF NOT EXISTS vector` on Postgres.

### Tests
```bash
pip install -r requirements-dev.txt
cd backend && pytest tests/ -v
```
The suite (about 66 tests) covers auth, RBAC, audit logging, paper generation, embeddings and RAG.

### Deployment
`vercel.json` routes `/api/*` to the Flask app (`api/index.py`) and serves `frontend/` as static files. Set `DATABASE_URL`, `GOOGLE_API_KEY`, `SECRET_KEY` and `JWT_SECRET_KEY` in the Vercel project settings.

## 📂 Project Structure

```text
├── api/index.py               # Vercel serverless entry point
├── backend/
│   ├── app/
│   │   ├── models/            # user, role, permission, question, paper, document, audit_log, ...
│   │   ├── routes/            # auth, questions, papers, subjects, teacher, admin, documents, public
│   │   ├── services/          # ai, embedding, rag, document, paper_generator, pdf_generator, audit
│   │   ├── authorization.py   # RBAC decorators
│   │   └── config.py
│   ├── migrations/            # Alembic versions
│   ├── tests/                 # pytest suite
│   ├── seed_*.py              # question / subject / sample-paper seeders
│   ├── backfill_embeddings.py
│   └── run.py                 # local entry point
├── frontend/
│   ├── index.html             # landing + login/register
│   ├── library.html           # public paper library
│   ├── dashboard.html         # stats and your papers
│   ├── generate.html          # paper and AI generation
│   ├── edit-paper.html        # paper editor
│   ├── questions.html         # question bank management
│   ├── profile.html
│   └── admin/                 # teachers and subject requests
├── .github/workflows/ci.yml   # CI
├── requirements.txt / requirements-dev.txt
└── vercel.json
```

## 🔐 Default account
Auto-seeded on first run against an empty database:
- **Admin**: `admin@qpgen.com` / `admin123`, or the value of `ADMIN_DEFAULT_PASSWORD` in `.env`. **Change this before deploying anywhere public.**

New accounts registered at `/api/auth/register` get the open `user` role: they can generate and download papers but not edit the question bank.

## 📚 More
See [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) for architecture, design decisions and resume-ready summary points.

---
*Developed as a Semester 6 Minor Project.*
