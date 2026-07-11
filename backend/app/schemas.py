from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionSummary(BaseModel):
    month: str
    inflow: float = Field(ge=0)
    outflow: float = Field(ge=0)
    closing_balance: float


class DocumentCheck(BaseModel):
    document_type: str
    authenticity_confidence: float = Field(ge=0, le=1)
    face_match_similarity: float = Field(ge=0, le=1)
    tamper_flag: bool = False


class LoanApplicationCreate(BaseModel):
    applicant_name: str
    applicant_email: str | None = None
    employment_type: Literal["salaried", "self_employed", "business", "contract", "other"] = "other"
    monthly_income: float = Field(ge=0)
    monthly_expenses: float = Field(ge=0)
    existing_debts: float = Field(ge=0)
    requested_amount: float = Field(ge=0)
    tenure_months: int = Field(ge=6, le=360)
    credit_history_years: float = Field(ge=0)
    dependents: int = Field(default=0, ge=0)
    purpose: str = "Working capital"
    notes: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    transactions: list[TransactionSummary] = Field(default_factory=list)
    document_checks: list[DocumentCheck] = Field(default_factory=list)


class UploadedDocument(BaseModel):
    id: str
    application_id: str | None = None
    filename: str
    content_type: str
    document_kind: Literal["pdf", "id_image"]
    size_bytes: int
    sha256: str
    byte_entropy: float
    ascii_ratio: float
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    document: UploadedDocument
    model_score: float
    confidence: float
    analysis: str


class FeatureAttribution(BaseModel):
    feature: str
    contribution: float


class ModelPrediction(BaseModel):
    model_name: str
    model_version: str
    score: float
    confidence: float
    attributions: list[FeatureAttribution] = Field(default_factory=list)


class ModelStatus(BaseModel):
    model_name: str
    model_version: str
    trained: bool
    sample_count: int
    accuracy: float | None = None
    trained_at: datetime | None = None


class ModelTrainRequest(BaseModel):
    include_seed_cases: bool = True
    include_unattached_documents: bool = False


class ModelTrainResponse(BaseModel):
    trained_at: datetime
    model_statuses: list[ModelStatus]
    sample_count: int


class ScoringOutput(BaseModel):
    final_score: float
    risk_band: str
    recommendation: str
    confidence: float
    fusion_weights: dict[str, float]
    tabular_score: float
    cash_flow_score: float
    authenticity_score: float
    model_outputs: list[ModelPrediction] = Field(default_factory=list)
    shap_explanation: list[dict[str, float | str]]
    time_step_signal: list[dict[str, float | str]]
    document_rationale: str


class LoanApplication(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    submitted_at: datetime
    payload: LoanApplicationCreate
    scoring: ScoringOutput
    documents: list[UploadedDocument] = Field(default_factory=list)
    officer_decision: str | None = None
    officer_comment: str | None = None
    status: str


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject", "modify", "request_more_documents"]
    comment: str | None = None
    revised_amount: float | None = Field(default=None, ge=0)
    revised_tenure_months: int | None = Field(default=None, ge=6, le=360)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


class DashboardStats(BaseModel):
    total_cases: int
    review_queue: int
    approved: int
    rejected: int
    average_final_score: float
    override_rate: float
