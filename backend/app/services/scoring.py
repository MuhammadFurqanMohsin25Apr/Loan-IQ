from __future__ import annotations

from ..schemas import DocumentCheck, LoanApplicationCreate, ScoringOutput
from .documents import DocumentBlob
from .trainable_models import LoanModelRegistry


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def evaluate_application(
    payload: LoanApplicationCreate,
    documents: list[DocumentBlob],
    registry: LoanModelRegistry,
) -> ScoringOutput:
    model_outputs = registry.predict(payload, documents, payload.document_checks)

    tabular_output, cash_flow_output, document_output = model_outputs
    fusion_weights = {"tabular": 0.55, "cash_flow": 0.25, "document": 0.20}
    final_score = clamp(
        tabular_output.score * fusion_weights["tabular"]
        + cash_flow_output.score * fusion_weights["cash_flow"]
        + document_output.score * fusion_weights["document"]
    )

    document_rationale = "Uploaded documents were analyzed by the document authenticity service."
    if payload.document_checks:
        flags = [
            f"{item.document_type}:{'tamper' if item.tamper_flag else 'clean'}"
            for item in payload.document_checks
        ]
        document_rationale = "; ".join(flags)

    if final_score >= 0.75:
        risk_band = "Low"
        recommendation = "Approve"
    elif final_score >= 0.55:
        risk_band = "Medium"
        recommendation = "Refer"
    else:
        risk_band = "High"
        recommendation = "Reject"

    if document_output.score < 0.45:
        recommendation = "Refer"

    confidence = clamp(mean([tabular_output.confidence, cash_flow_output.confidence, document_output.confidence], 0.5))

    time_step_signal = [
        {"month": transaction.month, "signal": round(score, 3)}
        for transaction, score in zip(payload.transactions, _transaction_signals(payload.transactions))
    ]

    return ScoringOutput(
        final_score=round(final_score, 3),
        risk_band=risk_band,
        recommendation=recommendation,
        confidence=round(confidence, 3),
        fusion_weights=fusion_weights,
        tabular_score=tabular_output.score,
        cash_flow_score=cash_flow_output.score,
        authenticity_score=document_output.score,
        model_outputs=model_outputs,
        shap_explanation=[
            {"feature": item.feature, "impact": item.contribution}
            for item in tabular_output.attributions[:5]
        ],
        time_step_signal=time_step_signal,
        document_rationale=document_rationale,
    )


def _transaction_signals(transactions) -> list[float]:
    if not transactions:
        return []

    scores: list[float] = []
    previous_margin: float | None = None
    for transaction in transactions:
        margin = transaction.inflow - transaction.outflow
        savings_rate = clamp(margin / max(transaction.inflow, 1.0))
        balance_health = clamp((transaction.closing_balance + transaction.inflow) / max(transaction.inflow * 2.0, 1.0))
        if previous_margin is None:
            stability = 0.72
        else:
            spread = abs(margin - previous_margin) / max(abs(previous_margin) + abs(margin) + 1.0, 1.0)
            stability = clamp(1.0 - spread)
        scores.append(clamp(0.45 * savings_rate + 0.35 * balance_health + 0.20 * stability))
        previous_margin = margin
    return scores
