"""Поиск похожих исторических случаев"""

from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "mri_tumor_cases")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def _client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=30.0, check_compatibility=False)


def search_similar_cases(
    embedding: List[float],
    *,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    limit: int = 3,
    exclude_patient_id: Optional[str] = None,
) -> str:
    """
    Ищет ближайших соседей по эмбеддингу.
    Возраст и пол влияют на порядок через rerank, без жесткого фильтра.
    """
    try:
        client = _client()
        if not client.collection_exists(COLLECTION_NAME):
            logger.warning("Qdrant collection %s missing", COLLECTION_NAME)
            return json.dumps([])

        query_filter = None
        if exclude_patient_id:
            query_filter = Filter(
                must_not=[
                    FieldCondition(
                        key="patient_id",
                        match=MatchValue(value=exclude_patient_id),
                    )
                ]
            )

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            query_filter=query_filter,
            limit=max(limit * 4, 12),
        )
        hits = list(response.points)
    except Exception as exc:
        logger.exception("Qdrant search failed: %s", exc)
        return json.dumps([])

    def rerank_key(hit) -> tuple:
        payload = hit.payload or {}
        gender_bonus = 0.0
        if gender and gender not in ("", "unknown"):
            if str(payload.get("gender")) == str(gender):
                gender_bonus = 0.05
        age_bonus = 0.0
        hit_age = payload.get("age", -1)
        if age is not None and age >= 0 and isinstance(hit_age, (int, float)) and hit_age >= 0:
            age_bonus = max(0.0, 0.05 - abs(int(hit_age) - int(age)) * 0.002)
        return (-(float(hit.score) + gender_bonus + age_bonus),)

    hits_sorted = sorted(hits, key=rerank_key)[:limit]
    cases: List[Dict[str, Any]] = []
    for hit in hits_sorted:
        payload = dict(hit.payload or {})
        patient_id = payload.get("patient_id")
        case = {
            **payload,
            "case_id": payload.get("short_id") or patient_id,
            "patient_id": patient_id,
            "score": round(float(hit.score), 4),
        }
        case["summary"] = (
            f"histological_type={case.get('histological_type', '?')}, "
            f"grade={case.get('grade', '?')}, "
            f"location={case.get('location', '?')}"
        )
        cases.append(case)
    return json.dumps(cases, ensure_ascii=False)
