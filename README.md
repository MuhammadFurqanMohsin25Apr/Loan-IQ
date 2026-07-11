# Loan IQ: Intelligent Loan Co-Pilot

Loan IQ is a hackathon-ready full-stack starter for a bank loan assistant. It includes a FastAPI backend, a React + Vite frontend, explainable scoring stubs, officer-review actions, and Docker Compose wiring for local demos.

## Project Layout

- `backend/` - FastAPI API, scoring logic, report generation stub
- `frontend/` - React dashboard and applicant portal UI
- `docker-compose.yml` - local orchestration for both services

## Run Locally

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Docker Compose

```bash
docker compose up --build
```

## API Highlights

- `GET /health`
- `GET /api/applications`
- `POST /api/applications`
- `POST /api/uploads/documents`
- `GET /api/uploads/documents`
- `POST /api/models/train`
- `GET /api/models/status`
- `POST /api/applications/{application_id}/decision`
- `GET /api/applications/{application_id}/report`
- `POST /api/applications/{application_id}/chat`

## Notes

- The backend uses ANN, LSTM, and CNN model services for scoring.
- The case summary endpoint uses Gemini when `GEMINI_API_KEY` is set; otherwise it falls back to a local summary.
- Demo queue seeding is disabled by default. Set `SEED_DEMO_CASES=true` only if you want sample records on startup.
- The project is a starter scaffold, not a production lending system.
