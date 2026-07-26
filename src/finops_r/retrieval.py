from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .artifacts import atomic_json, sha256_file
from .json_safety import require_json_object, strict_json_dumps, strict_json_load
from .policy_cards import CompiledPolicyCard, load_compiled_policy_cards


TOKENIZER_VERSION = "finops_ascii_terms_v1"
INDEX_SCHEMA_VERSION = "1.0"
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> tuple[str, ...]:
    normalized = text.lower().replace("t+1", "tplus1").replace("t+2", "tplus2")
    return tuple(_TOKEN.findall(normalized))


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


class EvidenceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    document_id: str = Field(min_length=1, max_length=160)
    source_type: Literal["POLICY_CARD", "CASE_ARTIFACT"]
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=5000)
    source_url: str | None = Field(default=None, max_length=1000)
    observable_at: datetime
    effective_from: datetime
    effective_to: datetime | None = None
    allowed_actions: tuple[str, ...] = ()
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("observable_at", "effective_from", "effective_to")
    @classmethod
    def timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value, "evidence timestamp")

    @model_validator(mode="after")
    def interval_and_source_are_valid(self) -> "EvidenceDocument":
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be later than effective_from")
        if self.source_type == "POLICY_CARD" and not self.source_url:
            raise ValueError("policy-card evidence requires source_url")
        if self.source_type == "CASE_ARTIFACT" and self.allowed_actions:
            raise ValueError("case artifacts cannot authorize actions")
        return self

    def eligible_at(self, decision_time: datetime) -> bool:
        decision = _utc(decision_time, "decision_time")
        return (
            self.observable_at <= decision
            and self.effective_from <= decision
            and (self.effective_to is None or decision < self.effective_to)
        )


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    document_id: str
    source_type: Literal["POLICY_CARD", "CASE_ARTIFACT"]
    score: float
    rank: int = Field(ge=1)
    allowed_actions: tuple[str, ...]
    checksum: str


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    query_id: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=2000)
    decision_time: datetime
    relevant_ids: tuple[str, ...] = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=100)

    @field_validator("decision_time")
    @classmethod
    def decision_time_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, "decision_time")


def policy_cards_to_documents(
    cards: list[CompiledPolicyCard],
) -> list[EvidenceDocument]:
    documents: list[EvidenceDocument] = []
    for card in cards:
        text = " ".join([card.title, card.applicability, card.paraphrase, " ".join(card.tags)])
        documents.append(
            EvidenceDocument(
                document_id=card.card_id,
                source_type="POLICY_CARD",
                title=card.title,
                text=text,
                source_url=card.source_url,
                observable_at=datetime.combine(card.source_date, datetime.min.time(), timezone.utc),
                effective_from=card.effective_from,
                effective_to=card.effective_to,
                allowed_actions=card.allowed_actions,
                checksum=card.checksum,
            )
        )
    return documents


class BM25Index:
    def __init__(self, documents: list[EvidenceDocument], *, k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("retrieval index requires at least one document")
        if not math.isfinite(k1) or k1 <= 0.0 or not math.isfinite(b) or not 0.0 <= b <= 1.0:
            raise ValueError("BM25 parameters are invalid")
        ids = [document.document_id for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieval document IDs must be unique")
        self.documents = tuple(sorted(documents, key=lambda item: item.document_id))
        self.k1 = float(k1)
        self.b = float(b)
        self._tokens = {item.document_id: tokenize(item.text) for item in self.documents}

    def eligible_documents(self, decision_time: datetime) -> list[EvidenceDocument]:
        return [item for item in self.documents if item.eligible_at(decision_time)]

    def search(self, query: str, decision_time: datetime, top_k: int = 5) -> list[RetrievalHit]:
        if not query.strip() or query != query.strip():
            raise ValueError("query must be non-empty and trimmed")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = tokenize(query)
        eligible = self.eligible_documents(decision_time)
        if not eligible:
            return []
        token_lists = [self._tokens[item.document_id] for item in eligible]
        lengths = np.asarray([len(tokens) for tokens in token_lists], dtype=float)
        average_length = float(lengths.mean()) if len(lengths) else 1.0
        document_frequency = Counter(
            term for tokens in token_lists for term in set(tokens) if term in set(query_terms)
        )
        scores: list[tuple[float, EvidenceDocument]] = []
        n_documents = len(eligible)
        for document, tokens in zip(eligible, token_lists, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse_document_frequency = math.log(1.0 + (n_documents - df + 0.5) / (df + 0.5))
                norm = frequency + self.k1 * (
                    1.0 - self.b + self.b * len(tokens) / max(average_length, 1e-12)
                )
                score += inverse_document_frequency * frequency * (self.k1 + 1.0) / norm
            scores.append((float(score), document))
        ranked = sorted(scores, key=lambda item: (-item[0], item[1].document_id))[:top_k]
        return [
            RetrievalHit(
                document_id=document.document_id,
                source_type=document.source_type,
                score=score,
                rank=rank,
                allowed_actions=document.allowed_actions,
                checksum=document.checksum,
            )
            for rank, (score, document) in enumerate(ranked, start=1)
        ]

    def manifest(self) -> dict[str, Any]:
        records = [item.model_dump(mode="json") for item in self.documents]
        corpus_hash = hashlib.sha256(
            strict_json_dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "retriever": "BM25",
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": self.k1,
            "b": self.b,
            "document_count": len(self.documents),
            "document_ids": [item.document_id for item in self.documents],
            "corpus_sha256": corpus_hash,
            "effective_time_filter": "observable_at<=decision_time and effective interval contains decision_time",
            "tie_break": "document_id ascending",
        }


def rank_dense(
    documents: list[EvidenceDocument],
    document_vectors: dict[str, np.ndarray],
    query_vector: np.ndarray,
    decision_time: datetime,
    top_k: int = 5,
) -> list[RetrievalHit]:
    query = np.asarray(query_vector, dtype=float)
    if query.ndim != 1 or not query.size or not np.isfinite(query).all():
        raise ValueError("query vector must be one finite non-empty vector")
    query_norm = float(np.linalg.norm(query))
    if query_norm <= 0.0:
        raise ValueError("query vector norm must be positive")
    ranked: list[tuple[float, EvidenceDocument]] = []
    for document in documents:
        if not document.eligible_at(decision_time):
            continue
        if document.document_id not in document_vectors:
            raise ValueError(f"missing dense vector for {document.document_id}")
        vector = np.asarray(document_vectors[document.document_id], dtype=float)
        if vector.shape != query.shape or not np.isfinite(vector).all():
            raise ValueError("dense document vectors must be finite and dimension-compatible")
        vector_norm = float(np.linalg.norm(vector))
        score = (
            -1.0
            if vector_norm <= 0.0
            else float(np.dot(query, vector) / (query_norm * vector_norm))
        )
        ranked.append((score, document))
    ranked.sort(key=lambda item: (-item[0], item[1].document_id))
    return [
        RetrievalHit(
            document_id=document.document_id,
            source_type=document.source_type,
            score=score,
            rank=rank,
            allowed_actions=document.allowed_actions,
            checksum=document.checksum,
        )
        for rank, (score, document) in enumerate(ranked[:top_k], start=1)
    ]


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievalHit]], *, top_k: int = 5, constant: int = 60
) -> list[RetrievalHit]:
    if not result_sets or top_k < 1 or constant < 1:
        raise ValueError("hybrid retrieval parameters are invalid")
    scores: dict[str, float] = {}
    exemplars: dict[str, RetrievalHit] = {}
    for results in result_sets:
        seen: set[str] = set()
        for hit in results:
            if hit.document_id in seen:
                raise ValueError("one result set cannot contain duplicate document IDs")
            seen.add(hit.document_id)
            scores[hit.document_id] = scores.get(hit.document_id, 0.0) + 1.0 / (constant + hit.rank)
            exemplars[hit.document_id] = hit
    ordered = sorted(scores, key=lambda item: (-scores[item], item))[:top_k]
    return [
        RetrievalHit(
            **{
                **exemplars[document_id].model_dump(),
                "score": float(scores[document_id]),
                "rank": rank,
            }
        )
        for rank, document_id in enumerate(ordered, start=1)
    ]


def load_retrieval_queries(path: Path) -> list[RetrievalQuery]:
    payload = require_json_object(strict_json_load(path), source=f"retrieval queries {path}")
    if set(payload) != {"schema_version", "queries"} or payload["schema_version"] != "1.0":
        raise ValueError("retrieval query bundle schema is incompatible")
    if not isinstance(payload["queries"], list) or not payload["queries"]:
        raise ValueError("retrieval query bundle must contain queries")
    queries = [RetrievalQuery.model_validate(item) for item in payload["queries"]]
    ids = [query.query_id for query in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("retrieval query IDs must be unique")
    return queries


def benchmark_bm25(index: BM25Index, queries: list[RetrievalQuery]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for query in queries:
        hits = index.search(query.text, query.decision_time, query.top_k)
        hit_ids = [hit.document_id for hit in hits]
        relevant = set(query.relevant_ids)
        found = [
            rank for rank, document_id in enumerate(hit_ids, start=1) if document_id in relevant
        ]
        records.append(
            {
                "query_id": query.query_id,
                "hit_ids": hit_ids,
                "recall_at_k": len(relevant.intersection(hit_ids)) / len(relevant),
                "reciprocal_rank": 0.0 if not found else 1.0 / min(found),
                "future_evidence_count": sum(
                    not next(
                        item for item in index.documents if item.document_id == hit.document_id
                    ).eligible_at(query.decision_time)
                    for hit in hits
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "retriever": "BM25",
        "query_count": len(records),
        "mean_recall_at_k": float(np.mean([item["recall_at_k"] for item in records])),
        "mean_reciprocal_rank": float(np.mean([item["reciprocal_rank"] for item in records])),
        "future_evidence_count": int(sum(item["future_evidence_count"] for item in records)),
        "queries": records,
    }


def build_policy_bm25_artifacts(
    compiled_cards_path: Path, query_path: Path, output_dir: Path
) -> dict[str, Any]:
    cards = load_compiled_policy_cards(compiled_cards_path)
    index = BM25Index(policy_cards_to_documents(cards))
    queries = load_retrieval_queries(query_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "retrieval_index_manifest.json"
    report_path = output_dir / "retrieval_benchmark.json"
    atomic_json(index_path, index.manifest())
    atomic_json(report_path, benchmark_bm25(index, queries))
    return {
        "schema_version": "1.0",
        "index_manifest": index_path.name,
        "index_manifest_sha256": sha256_file(index_path),
        "benchmark": report_path.name,
        "benchmark_sha256": sha256_file(report_path),
    }
