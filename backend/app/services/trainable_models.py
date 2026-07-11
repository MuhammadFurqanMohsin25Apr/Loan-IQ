from __future__ import annotations

import hashlib
import csv
import io
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from math import log2
from functools import lru_cache

from PIL import Image
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from ..schemas import DocumentCheck, FeatureAttribution, LoanApplicationCreate, ModelPrediction, ModelStatus, UploadedDocument
from .documents import DocumentBlob


torch.manual_seed(7)
torch.set_num_threads(1)

DEFAULT_LOAN_DATASET_URL = "https://huggingface.co/datasets/13nishit/LoanApprovalPrediction/resolve/main/dataset.csv?download=true"


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def variance(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    avg = mean(values)
    return mean([(value - avg) ** 2 for value in values], default)


def one_hot(employment_type: str) -> list[float]:
    mapping = ["salaried", "self_employed", "business", "contract", "other"]
    return [1.0 if employment_type == label else 0.0 for label in mapping]


def tabular_features(payload: LoanApplicationCreate, documents: list[DocumentBlob]) -> list[float]:
    income = max(payload.monthly_income, 1.0)
    affordability = (payload.monthly_income - payload.monthly_expenses) / income
    dti = payload.existing_debts / income
    request_pressure = payload.requested_amount / max(income * max(payload.tenure_months / 12.0, 1.0), 1.0)
    history = min(payload.credit_history_years / 10.0, 1.0)
    document_count = float(len(documents))
    document_strength = mean([blob.document.byte_entropy for blob in documents], 0.0)
    employment = one_hot(payload.employment_type)
    return [
        affordability,
        dti,
        request_pressure,
        history,
        float(payload.dependents),
        document_count,
        document_strength,
        *employment,
    ]


def sequence_features(payload: LoanApplicationCreate) -> list[list[float]]:
    if not payload.transactions:
        return [[0.0, 0.0, 0.0, 0.0]]

    sequence: list[list[float]] = []
    for transaction in payload.transactions:
        margin = transaction.inflow - transaction.outflow
        sequence.append([transaction.inflow, transaction.outflow, transaction.closing_balance, margin])
    return sequence


def document_kind_from_label(document_type: str) -> str:
    lower = document_type.lower()
    if any(token in lower for token in ("pdf", "statement", "slip", "return", "bank")):
        return "pdf"
    return "id_image"


def _parse_dependents(value: str) -> int:
    cleaned = str(value).strip().replace("+", "")
    try:
        return max(int(cleaned), 0)
    except ValueError:
        return 0


def _parse_credit_history(value: str) -> float:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "yes", "y", "true"}:
        return 8.0
    if text in {"0", "0.0", "no", "n", "false"}:
        return 1.0
    try:
        return float(text)
    except ValueError:
        return 3.0


def _safe_float(value: str | float | int | None, default: float = 0.0) -> float:
    try:
        number = float(value if value is not None else default)
        return default if number != number else number
    except (TypeError, ValueError):
        return default


def _safe_int(value: str | float | int | None, default: int = 0) -> int:
    try:
        number = int(float(value if value is not None else default))
        return default if number != number else number
    except (TypeError, ValueError):
        return default


def _employment_from_dataset(row: dict[str, str]) -> str:
    if str(row.get("Self_Employed", "")).strip().lower() == "yes":
        return "self_employed"
    if str(row.get("Education", "")).strip().lower() == "not graduate":
        return "business"
    return "salaried"


def _loan_status_label(value: str) -> int:
    return 1 if str(value).strip().lower() == "y" else 0


def _row_to_payload(row: dict[str, str], index: int) -> LoanApplicationCreate:
    applicant_income = _safe_float(row.get("ApplicantIncome"), 0.0)
    coapplicant_income = _safe_float(row.get("CoapplicantIncome"), 0.0)
    loan_amount = _safe_float(row.get("LoanAmount"), 0.0)
    loan_term = _safe_int(row.get("Loan_Amount_Term"), 36)
    credit_history = _parse_credit_history(row.get("Credit_History", "1"))
    requested_amount = max(loan_amount * 1000.0, 1.0)
    monthly_income = max(applicant_income + coapplicant_income, 1.0)
    monthly_expenses = max(monthly_income * 0.55, requested_amount / max(loan_term, 12))
    existing_debts = max(requested_amount * 0.22, 0.0)
    dependents = _parse_dependents(row.get("Dependents", "0"))
    purpose = f"Loan approval dataset row {index + 1}"
    property_area = str(row.get("Property_Area", ""))

    transactions = [
        {
            "month": "2024-01",
            "inflow": round(monthly_income * 0.96, 2),
            "outflow": round(monthly_expenses * 0.92, 2),
            "closing_balance": round(monthly_income * 0.16, 2),
        },
        {
            "month": "2024-02",
            "inflow": round(monthly_income * 1.01, 2),
            "outflow": round(monthly_expenses * 0.98, 2),
            "closing_balance": round(monthly_income * 0.18, 2),
        },
        {
            "month": "2024-03",
            "inflow": round(monthly_income * 1.03, 2),
            "outflow": round(monthly_expenses * 1.01, 2),
            "closing_balance": round(monthly_income * 0.2, 2),
        },
    ]

    document_checks = [
        {
            "document_type": "loan_approval_dataset_summary",
            "authenticity_confidence": 0.75 if property_area.lower() != "rural" else 0.68,
            "face_match_similarity": 0.8 if credit_history >= 5 else 0.62,
            "tamper_flag": credit_history < 2,
        }
    ]

    return LoanApplicationCreate(
        applicant_name=str(row.get("Loan_ID", f"dataset-{index + 1}")),
        applicant_email=None,
        employment_type=_employment_from_dataset(row),
        monthly_income=monthly_income,
        monthly_expenses=monthly_expenses,
        existing_debts=existing_debts,
        requested_amount=requested_amount,
        tenure_months=max(6, min(loan_term, 360)),
        credit_history_years=credit_history,
        dependents=dependents,
        purpose=purpose,
        notes=f"Dataset source property area: {property_area}",
        document_ids=[],
        transactions=transactions,
        document_checks=document_checks,
    )


@lru_cache(maxsize=1)
def load_loan_approval_dataset(url: str = DEFAULT_LOAN_DATASET_URL) -> list[tuple[LoanApplicationCreate, int]]:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw_csv = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return []

    reader = csv.DictReader(io.StringIO(raw_csv))
    dataset: list[tuple[LoanApplicationCreate, int]] = []
    for index, row in enumerate(reader):
        payload = _row_to_payload(row, index)
        label = _loan_status_label(row.get("Loan_Status", "N"))
        dataset.append((payload, label))
    return dataset


def _entropy(content: bytes) -> float:
    if not content:
        return 0.0

    counts: dict[int, int] = {}
    for byte in content:
        counts[byte] = counts.get(byte, 0) + 1

    total = len(content)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * log2(probability)
    return round(entropy, 4)


def _ascii_ratio(content: bytes) -> float:
    if not content:
        return 0.0
    printable = sum(1 for byte in content if 32 <= byte <= 126 or byte in {9, 10, 13})
    return round(printable / len(content), 4)


def synthetic_document_blob(payload: LoanApplicationCreate, document_check: DocumentCheck, index: int) -> DocumentBlob:
    seed_text = (
        f"{payload.applicant_name}|{document_check.document_type}|{document_check.authenticity_confidence}|"
        f"{document_check.face_match_similarity}|{document_check.tamper_flag}|{index}"
    )
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    image_bytes = (digest * 32)[: 32 * 32]
    image = Image.frombytes("L", (32, 32), image_bytes)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()
    sha256 = hashlib.sha256(content).hexdigest()
    uploaded_document = UploadedDocument(
        id=sha256[:16],
        application_id=None,
        filename=f"synthetic-{document_check.document_type}-{index}.png",
        content_type="image/png",
        document_kind=document_kind_from_label(document_check.document_type),
        size_bytes=len(content),
        sha256=sha256,
        byte_entropy=_entropy(content),
        ascii_ratio=_ascii_ratio(content),
        created_at=datetime.now(timezone.utc),
    )
    return DocumentBlob(document=uploaded_document, content=content)


def document_tensor_from_blob(blob: DocumentBlob, size: int = 32) -> torch.Tensor:
    content = blob.content
    try:
        with Image.open(BytesIO(content)) as image:
            grayscale = image.convert("L").resize((size, size))
            pixels = torch.tensor(list(grayscale.getdata()), dtype=torch.float32).view(1, size, size)
            return pixels / 255.0
    except Exception:
        if not content:
            content = b"\x00"
        repeated = (content * ((size * size // len(content)) + 1))[: size * size]
        return torch.tensor(list(repeated), dtype=torch.float32).view(1, size, size) / 255.0


def _feature_statistics(vectors: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.tensor(vectors, dtype=torch.float32)
    mean_values = stacked.mean(dim=0)
    std_values = stacked.std(dim=0)
    std_values = torch.where(std_values < 1e-6, torch.ones_like(std_values), std_values)
    return mean_values, std_values


def _sequence_statistics(sequences: list[list[list[float]]]) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = torch.tensor([step for sequence in sequences for step in sequence], dtype=torch.float32)
    mean_values = flattened.mean(dim=0)
    std_values = flattened.std(dim=0)
    std_values = torch.where(std_values < 1e-6, torch.ones_like(std_values), std_values)
    return mean_values, std_values


def _train_binary_classifier(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, epochs: int = 48, learning_rate: float = 1e-2) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()


class TabularANN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class SequenceLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = pack_padded_sequence(sequences, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        attention_logits = self.attention(output).squeeze(-1)
        mask = torch.arange(output.size(1), device=output.device)[None, :] >= lengths[:, None]
        attention_logits = attention_logits.masked_fill(mask, -1e9)
        weights = torch.softmax(attention_logits, dim=1)
        context = torch.sum(output * weights.unsqueeze(-1), dim=1)
        logits = self.head(context).squeeze(-1)
        return logits, weights


class DocumentCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.features(inputs)
        logits = self.head(feature_maps).squeeze(-1)
        return logits, feature_maps


@dataclass(slots=True)
class TabularANNService:
    name: str = "ann_credit_risk"
    feature_names: list[str] = field(
        default_factory=lambda: [
            "affordability",
            "debt_to_income",
            "request_pressure",
            "credit_history",
            "dependents",
            "document_count",
            "document_strength",
            "employment_salaried",
            "employment_self_employed",
            "employment_business",
            "employment_contract",
            "employment_other",
        ]
    )
    model: TabularANN | None = None
    feature_mean: torch.Tensor | None = None
    feature_std: torch.Tensor | None = None
    trained: bool = False
    sample_count: int = 0
    accuracy: float | None = None
    trained_at: datetime | None = None
    version: str = "untrained"

    def fit(self, samples: list[tuple[list[float], int]]) -> None:
        if not samples:
            return

        vectors = [vector for vector, _ in samples]
        labels = torch.tensor([label for _, label in samples], dtype=torch.float32)
        mean_values, std_values = _feature_statistics(vectors)
        features = torch.tensor(vectors, dtype=torch.float32)
        normalized = (features - mean_values) / std_values

        self.feature_mean = mean_values
        self.feature_std = std_values
        self.model = TabularANN(normalized.shape[1])
        _train_binary_classifier(self.model, normalized, labels)

        self.sample_count = len(samples)
        self.trained = True
        self.trained_at = datetime.now(timezone.utc)
        self.version = f"{self.name}-v{self.sample_count}-{int(self.trained_at.timestamp())}"
        self.accuracy = self._accuracy(samples)

    def _accuracy(self, samples: list[tuple[list[float], int]]) -> float:
        if not samples:
            return 0.0
        correct = 0
        for features, target in samples:
            prediction = self.predict(features)
            if (prediction.score >= 0.5) == bool(target):
                correct += 1
        return round(correct / len(samples), 3)

    def predict(self, features: list[float]) -> ModelPrediction:
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            score = clamp(
                0.52
                - features[0] * 0.22
                - features[1] * 0.28
                - features[2] * 0.18
                + features[3] * 0.12
                - features[4] * 0.01
                + features[5] * 0.04
                + features[6] * 0.03
                + features[7] * 0.04
                + features[8] * 0.02
                + features[9] * 0.02
                + features[10] * 0.01
                + features[11] * 0.01
            )
            contributions = [
                -features[0] * 0.22,
                -features[1] * 0.28,
                -features[2] * 0.18,
                features[3] * 0.12,
                -features[4] * 0.01,
                features[5] * 0.04,
                features[6] * 0.03,
                features[7] * 0.04,
                features[8] * 0.02,
                features[9] * 0.02,
                features[10] * 0.01,
                features[11] * 0.01,
            ]
            attribution_pairs = [
                FeatureAttribution(feature=feature, contribution=round(float(contribution), 4))
                for feature, contribution in zip(self.feature_names, contributions)
            ]
            attribution_pairs.sort(key=lambda item: abs(item.contribution), reverse=True)
            confidence = round(min(1.0, 0.52 + abs(score - 0.5) * 0.8), 3)
            return ModelPrediction(
                model_name=self.name,
                model_version=self.version,
                score=round(score, 3),
                confidence=confidence,
                attributions=attribution_pairs[:5],
            )

        input_tensor = torch.tensor(features, dtype=torch.float32)
        normalized = (input_tensor - self.feature_mean) / self.feature_std
        with torch.no_grad():
            logits = self.model(normalized.unsqueeze(0))
            probability = torch.sigmoid(logits).item()

        first_layer = self.model.network[0]
        importance = first_layer.weight.detach().abs().mean(dim=0)
        contributions = normalized * importance
        attribution_pairs = [
            FeatureAttribution(feature=feature, contribution=round(float(contribution), 4))
            for feature, contribution in zip(self.feature_names, contributions)
        ]
        attribution_pairs.sort(key=lambda item: abs(item.contribution), reverse=True)
        confidence = round(min(1.0, 0.55 + abs(probability - 0.5) * 0.9), 3)
        return ModelPrediction(
            model_name=self.name,
            model_version=self.version,
            score=round(probability, 3),
            confidence=confidence,
            attributions=attribution_pairs[:5],
        )


@dataclass(slots=True)
class CashFlowLSTMService:
    name: str = "lstm_cash_flow"
    model: SequenceLSTM | None = None
    feature_mean: torch.Tensor | None = None
    feature_std: torch.Tensor | None = None
    trained: bool = False
    sample_count: int = 0
    accuracy: float | None = None
    trained_at: datetime | None = None
    version: str = "untrained"

    def fit(self, samples: list[tuple[list[list[float]], int]]) -> None:
        if not samples:
            return

        sequences = [sequence for sequence, _ in samples]
        labels = torch.tensor([label for _, label in samples], dtype=torch.float32)
        feature_mean, feature_std = _sequence_statistics(sequences)
        normalized_sequences: list[torch.Tensor] = []
        lengths: list[int] = []
        for sequence in sequences:
            tensor = torch.tensor(sequence, dtype=torch.float32)
            normalized_sequences.append((tensor - feature_mean) / feature_std)
            lengths.append(tensor.shape[0])

        padded = pad_sequence(normalized_sequences, batch_first=True)
        length_tensor = torch.tensor(lengths, dtype=torch.long)

        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.model = SequenceLSTM(padded.shape[-1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(64):
            optimizer.zero_grad()
            logits, _weights = self.model(padded, length_tensor)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        self.sample_count = len(samples)
        self.trained = True
        self.trained_at = datetime.now(timezone.utc)
        self.version = f"{self.name}-v{self.sample_count}-{int(self.trained_at.timestamp())}"
        self.accuracy = self._accuracy(samples)

    def _accuracy(self, samples: list[tuple[list[list[float]], int]]) -> float:
        if not samples:
            return 0.0
        correct = 0
        for sequence, target in samples:
            prediction = self.predict(sequence)
            if (prediction.score >= 0.5) == bool(target):
                correct += 1
        return round(correct / len(samples), 3)

    def predict(self, sequence: list[list[float]]) -> ModelPrediction:
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            margins = [step[3] for step in sequence] if sequence else [0.0]
            closing_balances = [step[2] for step in sequence] if sequence else [0.0]
            score = clamp(0.5 + mean(margins, 0.0) / 100000.0 + mean(closing_balances, 0.0) / 2000000.0)
            attribution_pairs = [
                FeatureAttribution(feature=f"step_{index + 1}", contribution=round(float(margin), 4))
                for index, margin in enumerate(margins)
            ]
            attribution_pairs.sort(key=lambda item: abs(item.contribution), reverse=True)
            confidence = round(min(1.0, 0.52 + abs(score - 0.5) * 0.8), 3)
            return ModelPrediction(
                model_name=self.name,
                model_version=self.version,
                score=round(score, 3),
                confidence=confidence,
                attributions=attribution_pairs[:5],
            )

        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        normalized = (sequence_tensor - self.feature_mean) / self.feature_std
        padded = normalized.unsqueeze(0)
        lengths = torch.tensor([normalized.shape[0]], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            logits, weights = self.model(padded, lengths)
            probability = torch.sigmoid(logits).item()
            attention = weights.squeeze(0)

        attribution_pairs = [
            FeatureAttribution(feature=f"step_{index + 1}", contribution=round(float(weight), 4))
            for index, weight in enumerate(attention)
        ]
        attribution_pairs.sort(key=lambda item: abs(item.contribution), reverse=True)
        confidence = round(min(1.0, 0.55 + abs(probability - 0.5) * 0.9), 3)
        return ModelPrediction(
            model_name=self.name,
            model_version=self.version,
            score=round(probability, 3),
            confidence=confidence,
            attributions=attribution_pairs[:5],
        )


@dataclass(slots=True)
class DocumentCNNService:
    name: str = "cnn_document_authenticity"
    model: DocumentCNN | None = None
    trained: bool = False
    sample_count: int = 0
    accuracy: float | None = None
    trained_at: datetime | None = None
    version: str = "untrained"

    def fit(self, samples: list[tuple[torch.Tensor, int]]) -> None:
        if not samples:
            return

        inputs = torch.stack([image for image, _ in samples], dim=0)
        labels = torch.tensor([label for _, label in samples], dtype=torch.float32)
        self.model = DocumentCNN()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(40):
            optimizer.zero_grad()
            logits, _feature_maps = self.model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        self.sample_count = len(samples)
        self.trained = True
        self.trained_at = datetime.now(timezone.utc)
        self.version = f"{self.name}-v{self.sample_count}-{int(self.trained_at.timestamp())}"
        self.accuracy = self._accuracy(samples)

    def _accuracy(self, samples: list[tuple[torch.Tensor, int]]) -> float:
        if not samples:
            return 0.0
        correct = 0
        for image, target in samples:
            prediction = self._predict_tensor(image)
            if (prediction.score >= 0.5) == bool(target):
                correct += 1
        return round(correct / len(samples), 3)

    def _predict_tensor(self, image: torch.Tensor) -> ModelPrediction:
        if self.model is None:
            average_intensity = float(image.mean().item())
            contrast = float(image.std().item())
            score = clamp(0.5 + (average_intensity - 0.5) * 0.18 + contrast * 0.12)
            attribution_pairs = [
                FeatureAttribution(feature="intensity", contribution=round(average_intensity - 0.5, 4)),
                FeatureAttribution(feature="contrast", contribution=round(contrast, 4)),
            ]
            confidence = round(min(1.0, 0.55 + abs(score - 0.5) * 0.9), 3)
            return ModelPrediction(
                model_name=self.name,
                model_version=self.version,
                score=round(score, 3),
                confidence=confidence,
                attributions=attribution_pairs,
            )
        self.model.eval()
        with torch.no_grad():
            logits, feature_maps = self.model(image.unsqueeze(0))
            probability = torch.sigmoid(logits).item()
            channel_strength = feature_maps.squeeze(0).squeeze(-1).squeeze(-1)

        attribution_pairs = [
            FeatureAttribution(feature=f"channel_{index + 1}", contribution=round(float(value), 4))
            for index, value in enumerate(channel_strength)
        ]
        attribution_pairs.sort(key=lambda item: abs(item.contribution), reverse=True)
        confidence = round(min(1.0, 0.55 + abs(probability - 0.5) * 0.9), 3)
        return ModelPrediction(
            model_name=self.name,
            model_version=self.version,
            score=round(probability, 3),
            confidence=confidence,
            attributions=attribution_pairs[:5],
        )

    def predict(self, document: UploadedDocument, content: bytes) -> ModelPrediction:
        blob = DocumentBlob(document=document, content=content)
        image = document_tensor_from_blob(blob)
        return self._predict_tensor(image)

    def predict_many(self, document_blobs: list[DocumentBlob]) -> ModelPrediction:
        predictions = [self.predict(blob.document, blob.content) for blob in document_blobs]
        if not predictions:
            return ModelPrediction(
                model_name=self.name,
                model_version=self.version,
                score=0.5,
                confidence=0.5,
                attributions=[],
            )

        score = mean([prediction.score for prediction in predictions])
        confidence = mean([prediction.confidence for prediction in predictions])
        attributions: list[FeatureAttribution] = []
        for prediction in predictions:
            attributions.extend(prediction.attributions)
        attributions.sort(key=lambda item: abs(item.contribution), reverse=True)
        return ModelPrediction(
            model_name=self.name,
            model_version=self.version,
            score=round(score, 3),
            confidence=round(confidence, 3),
            attributions=attributions[:5],
        )


def _documents_for_payload(payload: LoanApplicationCreate, document_blobs: list[DocumentBlob]) -> list[DocumentBlob]:
    if document_blobs:
        return document_blobs
    return [synthetic_document_blob(payload, document_check, index) for index, document_check in enumerate(payload.document_checks)]


def _tabular_label(payload: LoanApplicationCreate) -> int:
    income = max(payload.monthly_income, 1.0)
    affordability = (payload.monthly_income - payload.monthly_expenses) / income
    dti = payload.existing_debts / income
    request_pressure = payload.requested_amount / max(income * max(payload.tenure_months / 12.0, 1.0), 1.0)
    score = 0.52 + affordability * 0.22 - dti * 0.28 - request_pressure * 0.18 + min(payload.credit_history_years / 10.0, 1.0) * 0.12
    return 1 if score >= 0.5 else 0


def _sequence_label(payload: LoanApplicationCreate) -> int:
    return 1 if payload.requested_amount <= max(payload.monthly_income * payload.tenure_months, 1.0) and payload.existing_debts <= payload.monthly_income else 0


def _document_label(payload: LoanApplicationCreate, index: int) -> int:
    if index < len(payload.document_checks):
        return 0 if payload.document_checks[index].tamper_flag else 1
    if payload.document_checks:
        return 0 if any(check.tamper_flag for check in payload.document_checks) else 1
    return 1


@dataclass(slots=True)
class LoanModelRegistry:
    tabular_model: TabularANNService = field(default_factory=TabularANNService)
    cash_flow_model: CashFlowLSTMService = field(default_factory=CashFlowLSTMService)
    document_model: DocumentCNNService = field(default_factory=DocumentCNNService)

    def train(self, applications: list, documents: dict[str, DocumentBlob]) -> list[ModelStatus]:
        tabular_samples: list[tuple[list[float], int]] = []
        sequence_samples: list[tuple[list[list[float]], int]] = []
        document_samples: list[tuple[torch.Tensor, int]] = []

        dataset_samples = load_loan_approval_dataset()

        for payload, label in dataset_samples:
            effective_documents = _documents_for_payload(payload, [])
            tabular_samples.append((tabular_features(payload, effective_documents), label))
            sequence_samples.append((sequence_features(payload), label))
            for index, blob in enumerate(effective_documents):
                document_samples.append((document_tensor_from_blob(blob), _document_label(payload, index)))

        for application in applications:
            attached_documents = [documents[document_id] for document_id in application.payload.document_ids if document_id in documents]
            effective_documents = _documents_for_payload(application.payload, attached_documents)
            tabular_samples.append((tabular_features(application.payload, effective_documents), _tabular_label(application.payload)))
            sequence_samples.append((sequence_features(application.payload), _sequence_label(application.payload)))
            for index, blob in enumerate(effective_documents):
                document_samples.append((document_tensor_from_blob(blob), _document_label(application.payload, index)))

        self.tabular_model.fit(tabular_samples)
        self.cash_flow_model.fit(sequence_samples)
        self.document_model.fit(document_samples)

        return self.statuses()

    def predict(self, payload: LoanApplicationCreate, documents: list[DocumentBlob], document_checks: list[DocumentCheck]) -> list[ModelPrediction]:
        effective_documents = documents or [synthetic_document_blob(payload, document_check, index) for index, document_check in enumerate(document_checks)]
        tabular_prediction = self.tabular_model.predict(tabular_features(payload, effective_documents))
        cash_flow_prediction = self.cash_flow_model.predict(sequence_features(payload))
        document_prediction = self.document_model.predict_many(effective_documents)
        return [tabular_prediction, cash_flow_prediction, document_prediction]

    def statuses(self) -> list[ModelStatus]:
        return [
            ModelStatus(
                model_name=self.tabular_model.name,
                model_version=self.tabular_model.version,
                trained=self.tabular_model.trained,
                sample_count=self.tabular_model.sample_count,
                accuracy=self.tabular_model.accuracy,
                trained_at=self.tabular_model.trained_at,
            ),
            ModelStatus(
                model_name=self.cash_flow_model.name,
                model_version=self.cash_flow_model.version,
                trained=self.cash_flow_model.trained,
                sample_count=self.cash_flow_model.sample_count,
                accuracy=self.cash_flow_model.accuracy,
                trained_at=self.cash_flow_model.trained_at,
            ),
            ModelStatus(
                model_name=self.document_model.name,
                model_version=self.document_model.version,
                trained=self.document_model.trained,
                sample_count=self.document_model.sample_count,
                accuracy=self.document_model.accuracy,
                trained_at=self.document_model.trained_at,
            ),
        ]
