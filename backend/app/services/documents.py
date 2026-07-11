from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import UploadFile

from ..schemas import UploadedDocument


@dataclass(slots=True)
class DocumentBlob:
    document: UploadedDocument
    content: bytes


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
        entropy -= probability * math.log2(probability)
    return round(entropy, 4)


def _ascii_ratio(content: bytes) -> float:
    if not content:
        return 0.0
    printable = sum(1 for byte in content if 32 <= byte <= 126 or byte in {9, 10, 13})
    return round(printable / len(content), 4)


def infer_document_kind(filename: str, content_type: str | None) -> str:
    lower_name = filename.lower()
    lower_type = (content_type or "").lower()
    if lower_name.endswith(".pdf") or "pdf" in lower_type:
        return "pdf"
    return "id_image"


async def build_document_blob(file: UploadFile) -> DocumentBlob:
    content = await file.read()
    sha256 = hashlib.sha256(content).hexdigest()
    document = UploadedDocument(
        id=sha256[:16],
        application_id=None,
        filename=file.filename or "upload.bin",
        content_type=file.content_type or "application/octet-stream",
        document_kind=infer_document_kind(file.filename or "upload.bin", file.content_type),
        size_bytes=len(content),
        sha256=sha256,
        byte_entropy=_entropy(content),
        ascii_ratio=_ascii_ratio(content),
        created_at=datetime.now(timezone.utc),
    )
    return DocumentBlob(document=document, content=content)


def document_feature_vector(document: UploadedDocument) -> list[float]:
    return [
        float(document.size_bytes) / 1024.0,
        document.byte_entropy,
        document.ascii_ratio,
        1.0 if document.document_kind == "pdf" else 0.0,
        1.0 if document.document_kind == "id_image" else 0.0,
        float(len(document.filename)),
    ]
