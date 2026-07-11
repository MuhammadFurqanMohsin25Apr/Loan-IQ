from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from threading import Thread

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response

from .schemas import (
    ChatRequest,
    ChatResponse,
    DashboardStats,
    DecisionRequest,
    DocumentUploadResponse,
    LoanApplication,
    LoanApplicationCreate,
    ModelStatus,
    ModelTrainRequest,
    ModelTrainResponse,
)
from .services.documents import DocumentBlob, build_document_blob
from .services.report import build_case_report_pdf, generate_case_report
from .services.scoring import evaluate_application
from .services.trainable_models import LoanModelRegistry, load_loan_approval_dataset

app = FastAPI(title="Loan IQ API", version="0.2.0")

default_cors_origins = "http://localhost:5173,http://127.0.0.1:5173"
cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", default_cors_origins).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

applications: dict[str, LoanApplication] = {}
documents: dict[str, DocumentBlob] = {}
model_registry = LoanModelRegistry()


def decision_to_status(decision: str) -> str:
    return {
        "approve": "approved",
        "reject": "rejected",
        "modify": "modified",
        "request_more_documents": "pending_documents",
        "refer": "needs_officer_review",
    }.get(decision, decision)


def resolve_documents(document_ids: list[str]) -> list[DocumentBlob]:
    return [documents[document_id] for document_id in document_ids if document_id in documents]


def build_record(
    payload: LoanApplicationCreate,
    application_id: str | None = None,
    submitted_at: datetime | None = None,
    officer_decision: str | None = None,
    officer_comment: str | None = None,
) -> LoanApplication:
    attached_documents = resolve_documents(payload.document_ids)
    scoring = evaluate_application(payload, attached_documents, model_registry)
    resolved_id = application_id or str(uuid.uuid4())
    resolved_status = decision_to_status(scoring.recommendation.lower())
    if officer_decision:
        resolved_status = decision_to_status(officer_decision)

    return LoanApplication(
        id=resolved_id,
        submitted_at=submitted_at or datetime.now(timezone.utc),
        payload=payload,
        scoring=scoring,
        documents=[blob.document for blob in attached_documents],
        officer_decision=officer_decision,
        officer_comment=officer_comment,
        status=resolved_status,
    )


def rescore_application(application_id: str) -> LoanApplication:
    record = applications.get(application_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Application not found")

    updated = build_record(
        payload=record.payload,
        application_id=record.id,
        submitted_at=record.submitted_at,
        officer_decision=record.officer_decision,
        officer_comment=record.officer_comment,
    )
    applications[application_id] = updated
    return updated


def seed_cases() -> None:
    if applications:
        return

    samples = [
        LoanApplicationCreate(
            applicant_name="Amina Khan",
            applicant_email="amina@example.com",
            employment_type="salaried",
            monthly_income=180000,
            monthly_expenses=92000,
            existing_debts=26000,
            requested_amount=650000,
            tenure_months=36,
            credit_history_years=6.5,
            dependents=2,
            purpose="Home renovation",
            transactions=[
                {"month": "2026-01", "inflow": 182000, "outflow": 114000, "closing_balance": 68000},
                {"month": "2026-02", "inflow": 179000, "outflow": 108000, "closing_balance": 71000},
                {"month": "2026-03", "inflow": 181000, "outflow": 110500, "closing_balance": 70500},
            ],
            document_checks=[
                {"document_type": "salary_slip", "authenticity_confidence": 0.89, "face_match_similarity": 0.91, "tamper_flag": False},
                {"document_type": "national_id", "authenticity_confidence": 0.93, "face_match_similarity": 0.87, "tamper_flag": False},
            ],
        ),
        LoanApplicationCreate(
            applicant_name="Bilal Ahmed",
            applicant_email="bilal@example.com",
            employment_type="business",
            monthly_income=120000,
            monthly_expenses=98000,
            existing_debts=54000,
            requested_amount=1200000,
            tenure_months=48,
            credit_history_years=2.0,
            dependents=4,
            purpose="Inventory expansion",
            transactions=[
                {"month": "2026-01", "inflow": 124000, "outflow": 119000, "closing_balance": 5000},
                {"month": "2026-02", "inflow": 117000, "outflow": 121000, "closing_balance": -4000},
                {"month": "2026-03", "inflow": 119500, "outflow": 125000, "closing_balance": -5500},
            ],
            document_checks=[
                {"document_type": "bank_statement", "authenticity_confidence": 0.61, "face_match_similarity": 0.58, "tamper_flag": True},
                {"document_type": "national_id", "authenticity_confidence": 0.71, "face_match_similarity": 0.62, "tamper_flag": False},
            ],
        ),
    ]

    for index, sample in enumerate(samples, start=1):
        record = build_record(sample, application_id=f"case-{1000 + index}")
        applications[record.id] = record


def train_models(request: ModelTrainRequest | None = None) -> list[ModelStatus]:
    _ = request or ModelTrainRequest()
    return model_registry.train(list(applications.values()), documents)


@app.on_event("startup")
def startup_event() -> None:
    if os.getenv("SEED_DEMO_CASES", "false").lower() == "true":
        seed_cases()
    Thread(target=train_models, daemon=True).start()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/applications", response_model=list[LoanApplication])
def list_applications() -> list[LoanApplication]:
    return sorted(applications.values(), key=lambda item: item.submitted_at, reverse=True)


@app.post("/api/applications", response_model=LoanApplication, status_code=201)
def create_application(payload: LoanApplicationCreate) -> LoanApplication:
    record = build_record(payload)
    applications[record.id] = record
    for document_id in payload.document_ids:
        if document_id in documents:
            documents[document_id].document.application_id = record.id
    applications[record.id] = build_record(
        payload=record.payload,
        application_id=record.id,
        submitted_at=record.submitted_at,
        officer_decision=record.officer_decision,
        officer_comment=record.officer_comment,
    )
    return applications[record.id]


@app.get("/api/applications/{application_id}", response_model=LoanApplication)
def get_application(application_id: str) -> LoanApplication:
    record = applications.get(application_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return record


@app.post("/api/uploads/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    application_id: str | None = Form(default=None),
) -> DocumentUploadResponse:
    blob = await build_document_blob(file)
    document = blob.document
    if application_id and application_id in applications:
        document = document.model_copy(update={"application_id": application_id})

    documents[document.id] = DocumentBlob(document=document, content=blob.content)

    if application_id and application_id in applications:
        current = applications[application_id]
        if document.id not in current.payload.document_ids:
            current.payload.document_ids.append(document.id)
        applications[application_id] = build_record(
            payload=current.payload,
            application_id=current.id,
            submitted_at=current.submitted_at,
            officer_decision=current.officer_decision,
            officer_comment=current.officer_comment,
        )

    prediction = model_registry.document_model.predict(document, blob.content)
    analysis = (
        f"Stored {document.document_kind} upload '{document.filename}' with {document.size_bytes} bytes. "
        f"The document model currently scores it {prediction.score:.3f}."
    )
    return DocumentUploadResponse(
        document=document,
        model_score=prediction.score,
        confidence=prediction.confidence,
        analysis=analysis,
    )


@app.get("/api/uploads/documents", response_model=list[DocumentUploadResponse])
def list_documents() -> list[DocumentUploadResponse]:
    responses: list[DocumentUploadResponse] = []
    for blob in documents.values():
        prediction = model_registry.document_model.predict(blob.document, blob.content)
        responses.append(
            DocumentUploadResponse(
                document=blob.document,
                model_score=prediction.score,
                confidence=prediction.confidence,
                analysis=f"Stored {blob.document.document_kind} upload '{blob.document.filename}'.",
            )
        )
    return sorted(responses, key=lambda item: item.document.created_at, reverse=True)


@app.post("/api/models/train", response_model=ModelTrainResponse)
def retrain_models(request: ModelTrainRequest | None = Body(default=None)) -> ModelTrainResponse:
    statuses = train_models(request)
    return ModelTrainResponse(
        trained_at=datetime.now(timezone.utc),
        model_statuses=statuses,
        sample_count=len(applications) + len(load_loan_approval_dataset()),
    )


@app.get("/api/models/status", response_model=list[ModelStatus])
def model_status() -> list[ModelStatus]:
    return model_registry.statuses()


@app.post("/api/applications/{application_id}/decision", response_model=LoanApplication)
def record_decision(application_id: str, request: DecisionRequest) -> LoanApplication:
    record = applications.get(application_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Application not found")

    payload = record.payload
    if request.decision == "modify":
        updates: dict[str, float | int] = {}
        if request.revised_amount is not None:
            updates["requested_amount"] = request.revised_amount
        if request.revised_tenure_months is not None:
            updates["tenure_months"] = request.revised_tenure_months
        if updates:
            payload = payload.model_copy(update=updates)

    updated = build_record(
        payload=payload,
        application_id=application_id,
        submitted_at=record.submitted_at,
        officer_decision=request.decision,
        officer_comment=request.comment,
    )
    applications[application_id] = updated
    return updated


@app.get("/api/applications/{application_id}/report", response_class=PlainTextResponse)
def get_report(application_id: str) -> PlainTextResponse:
    record = applications.get(application_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return PlainTextResponse(generate_case_report(record), media_type="text/markdown")


@app.get("/api/applications/{application_id}/report/pdf")
def get_report_pdf(application_id: str) -> Response:
    record = applications.get(application_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Application not found")

    pdf_bytes = build_case_report_pdf(record)
    filename = f"Loan-IQ-Request-Docs-{application_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/applications/{application_id}/chat", response_model=ChatResponse)
def chat_about_case(application_id: str, request: ChatRequest) -> ChatResponse:
    record = applications.get(application_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Application not found")

    main_driver = record.scoring.model_outputs[0].model_name if record.scoring.model_outputs else "the model aggregate"
    answer = (
        f"This case is {record.scoring.risk_band.lower()} risk with a {record.scoring.final_score:.3f} final score. "
        f"The main driver is {main_driver}, the cash-flow signal is {record.scoring.cash_flow_score:.3f}, "
        f"and the document authenticity score is {record.scoring.authenticity_score:.3f}. "
        f"Question received: {request.question}"
    )
    return ChatResponse(answer=answer)


@app.get("/api/dashboard/stats", response_model=DashboardStats)
def dashboard_stats() -> DashboardStats:
    records = list(applications.values())
    total_cases = len(records)
    review_queue = sum(1 for record in records if record.status in {"needs_officer_review", "pending_documents", "modified", "refer"})
    approved = sum(1 for record in records if record.status == "approved")
    rejected = sum(1 for record in records if record.status == "rejected")
    average_final_score = sum(record.scoring.final_score for record in records) / total_cases if total_cases else 0.0
    override_rate = sum(1 for record in records if record.officer_decision is not None) / total_cases if total_cases else 0.0
    return DashboardStats(
        total_cases=total_cases,
        review_queue=review_queue,
        approved=approved,
        rejected=rejected,
        average_final_score=round(average_final_score, 3),
        override_rate=round(override_rate, 3),
    )