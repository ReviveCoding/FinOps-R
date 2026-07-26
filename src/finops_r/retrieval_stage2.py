from __future__ import annotations

import hashlib
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from .artifacts import sha256_file
from .json_safety import require_json_object, strict_json_dumps, strict_json_load
from .policy_cards import CompiledPolicyCard
from .retrieval import (
    BM25Index,
    EvidenceDocument,
    policy_cards_to_documents,
    rank_dense,
    reciprocal_rank_fusion,
)


STAGE2_SCHEMA_VERSION = "2.0"
STAGE2_BENCHMARK_ID = "finops_policy_retrieval_locked_v2"
STAGE2_BENCHMARK_SHA256 = "8588209b86427771cf3da1bc57db5220abd72a0eeb8b533c6ad09e17d04bc1bf"
BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QWEN_MODEL_ID = "Qwen/Qwen3-4B"
QWEN_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


class Stage2Distractor(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    document_id: str = Field(pattern=r"^benchmark\.case\.[a-z0-9_]+$")
    title: str = Field(min_length=3, max_length=200)
    text: str = Field(min_length=20, max_length=2000)
    observable_at: datetime
    effective_from: datetime
    effective_to: datetime | None = None

    @field_validator("observable_at", "effective_from", "effective_to")
    @classmethod
    def timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value, "distractor timestamp")

    @model_validator(mode="after")
    def interval_is_valid(self) -> "Stage2Distractor":
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("distractor effective interval is invalid")
        return self

    def to_evidence(self) -> EvidenceDocument:
        payload = self.model_dump(mode="json")
        checksum = hashlib.sha256(
            strict_json_dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return EvidenceDocument(
            document_id=self.document_id,
            source_type="CASE_ARTIFACT",
            title=self.title,
            text=f"{self.title} {self.text}",
            observable_at=self.observable_at,
            effective_from=self.effective_from,
            effective_to=self.effective_to,
            allowed_actions=(),
            checksum=checksum,
        )


class Stage2RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    query_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{2,79}$")
    category: Literal[
        "LEXICAL", "PARAPHRASE", "HARD_NEGATIVE", "TEMPORAL_CONFLICT", "MULTI_RELEVANT"
    ]
    text: str = Field(min_length=8, max_length=1000)
    decision_time: datetime
    relevant_ids: tuple[str, ...] = Field(min_length=1, max_length=4)

    @field_validator("decision_time")
    @classmethod
    def decision_time_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, "decision_time")

    @field_validator("relevant_ids")
    @classmethod
    def relevant_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("relevant_ids must be unique")
        return value


class Stage2Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["2.0"]
    benchmark_id: Literal["finops_policy_retrieval_locked_v2"]
    split: Literal["LOCKED_TEST"]
    frozen_at: datetime
    tuning_prohibited: StrictBool
    rrf_constant: Literal[60]
    top_k_values: tuple[Literal[1, 3, 5], Literal[1, 3, 5], Literal[1, 3, 5]]
    query_instruction: Literal["Represent this sentence for searching relevant passages: "]
    distractors: list[Stage2Distractor] = Field(min_length=10)
    queries: list[Stage2RetrievalQuery] = Field(min_length=30)

    @field_validator("frozen_at")
    @classmethod
    def frozen_at_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, "frozen_at")

    @model_validator(mode="after")
    def bundle_is_locked_and_unique(self) -> "Stage2Benchmark":
        if not self.tuning_prohibited:
            raise ValueError("locked test benchmark cannot permit tuning")
        if tuple(self.top_k_values) != (1, 3, 5):
            raise ValueError("top_k_values must be exactly [1, 3, 5]")
        document_ids = [item.document_id for item in self.distractors]
        query_ids = [item.query_id for item in self.queries]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("distractor document IDs must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("stage-2 query IDs must be unique")
        expected_categories = {
            "LEXICAL",
            "PARAPHRASE",
            "HARD_NEGATIVE",
            "TEMPORAL_CONFLICT",
            "MULTI_RELEVANT",
        }
        if {item.category for item in self.queries} != expected_categories:
            raise ValueError("stage-2 benchmark must cover every required query category")
        return self


class DenseEncoder(Protocol):
    dimension: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


def load_stage2_benchmark(path: Path) -> Stage2Benchmark:
    if sha256_file(path) != STAGE2_BENCHMARK_SHA256:
        raise ValueError("locked stage-2 benchmark SHA-256 mismatch")
    payload = require_json_object(strict_json_load(path), source=f"stage-2 benchmark {path}")
    return Stage2Benchmark.model_validate(payload)


def build_stage2_documents(
    cards: list[CompiledPolicyCard], benchmark: Stage2Benchmark
) -> list[EvidenceDocument]:
    documents = policy_cards_to_documents(cards) + [
        distractor.to_evidence() for distractor in benchmark.distractors
    ]
    by_id = {item.document_id: item for item in documents}
    if len(by_id) != len(documents):
        raise ValueError("stage-2 corpus document IDs must be unique")
    for query in benchmark.queries:
        missing = set(query.relevant_ids) - set(by_id)
        if missing:
            raise ValueError(f"query {query.query_id} has unknown relevant IDs: {sorted(missing)}")
        ineligible = [
            document_id
            for document_id in query.relevant_ids
            if not by_id[document_id].eligible_at(query.decision_time)
        ]
        if ineligible:
            raise ValueError(
                f"query {query.query_id} labels ineligible evidence as relevant: {ineligible}"
            )
    return documents


def ranking_metrics(
    hit_ids: list[str],
    relevant_ids: tuple[str, ...],
    top_k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    if not relevant_ids or len(relevant_ids) != len(set(relevant_ids)):
        raise ValueError("ranking metrics require unique non-empty relevant IDs")
    if len(hit_ids) != len(set(hit_ids)):
        raise ValueError("ranking metrics reject duplicate hit IDs")
    relevant = set(relevant_ids)
    output: dict[str, float] = {}
    ranks = [rank for rank, item in enumerate(hit_ids, start=1) if item in relevant]
    output["mrr"] = 0.0 if not ranks else 1.0 / min(ranks)
    for top_k in top_k_values:
        if top_k < 1:
            raise ValueError("top-k values must be positive")
        selected = hit_ids[:top_k]
        output[f"recall_at_{top_k}"] = len(relevant.intersection(selected)) / len(relevant)
        dcg = sum(
            1.0 / math.log2(rank + 1.0)
            for rank, item in enumerate(selected, start=1)
            if item in relevant
        )
        ideal_count = min(len(relevant), top_k)
        ideal = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_count + 1))
        output[f"ndcg_at_{top_k}"] = dcg / ideal
    return output


def _percentile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile requires values and a valid quantile")
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def compare_retrievers(
    documents: list[EvidenceDocument],
    benchmark: Stage2Benchmark,
    encoder: DenseEncoder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if encoder.dimension < 1:
        raise ValueError("dense encoder dimension must be positive")
    bm25 = BM25Index(documents)
    document_texts = [item.text for item in documents]
    vectors = np.asarray(encoder.encode_documents(document_texts), dtype=float)
    if vectors.shape != (len(documents), encoder.dimension) or not np.isfinite(vectors).all():
        raise ValueError("dense document encoder returned an invalid matrix")
    document_vectors = {
        document.document_id: vectors[index] for index, document in enumerate(documents)
    }
    by_id = {item.document_id: item for item in documents}
    pool_k = min(20, len(documents))
    records: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {"BM25": [], "BGE_DENSE": [], "RRF_HYBRID": []}

    for query in benchmark.queries:
        start = time.perf_counter_ns()
        bm25_hits = bm25.search(query.text, query.decision_time, pool_k)
        bm25_ms = (time.perf_counter_ns() - start) / 1_000_000.0

        start = time.perf_counter_ns()
        query_matrix = np.asarray(encoder.encode_queries([query.text]), dtype=float)
        if query_matrix.shape != (1, encoder.dimension) or not np.isfinite(query_matrix).all():
            raise ValueError("dense query encoder returned an invalid matrix")
        dense_hits = rank_dense(
            documents,
            document_vectors,
            query_matrix[0],
            query.decision_time,
            pool_k,
        )
        dense_ms = (time.perf_counter_ns() - start) / 1_000_000.0

        start = time.perf_counter_ns()
        hybrid_hits = reciprocal_rank_fusion(
            [bm25_hits, dense_hits], top_k=pool_k, constant=benchmark.rrf_constant
        )
        fusion_ms = (time.perf_counter_ns() - start) / 1_000_000.0

        for method, hits, latency_ms in (
            ("BM25", bm25_hits, bm25_ms),
            ("BGE_DENSE", dense_hits, dense_ms),
            ("RRF_HYBRID", hybrid_hits, bm25_ms + dense_ms + fusion_ms),
        ):
            hit_ids = [hit.document_id for hit in hits[:5]]
            metrics = ranking_metrics(hit_ids, query.relevant_ids)
            future_count = sum(
                not by_id[hit.document_id].eligible_at(query.decision_time) for hit in hits
            )
            latencies[method].append(latency_ms)
            records.append(
                {
                    "query_id": query.query_id,
                    "category": query.category,
                    "method": method,
                    "decision_time": query.decision_time.isoformat(),
                    "relevant_ids": list(query.relevant_ids),
                    "hit_ids": hit_ids,
                    "latency_ms": float(latency_ms),
                    "future_evidence_count": int(future_count),
                    **metrics,
                }
            )

    candidates: list[dict[str, Any]] = []
    metric_names = ["mrr"] + [
        f"{name}_at_{top_k}" for top_k in benchmark.top_k_values for name in ("recall", "ndcg")
    ]
    for method in ("BM25", "BGE_DENSE", "RRF_HYBRID"):
        selected = [item for item in records if item["method"] == method]
        row: dict[str, Any] = {
            "method": method,
            "query_count": len(selected),
            "future_evidence_count": sum(item["future_evidence_count"] for item in selected),
            "latency_p50_ms": _percentile(latencies[method], 0.50),
            "latency_p95_ms": _percentile(latencies[method], 0.95),
        }
        for metric in metric_names:
            row[metric] = float(statistics.fmean(item[metric] for item in selected))
        candidates.append(row)
    return candidates, records


class BGEEncoder:
    dimension = 384

    def __init__(
        self,
        *,
        cache_dir: Path,
        query_instruction: str,
        device: str = "cuda",
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Stage 2 requires torch and transformers") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("BGE stage requires a CUDA device")
        self._torch = torch
        self.device = torch.device(device)
        self.query_instruction = query_instruction
        self.tokenizer = AutoTokenizer.from_pretrained(
            BGE_MODEL_ID,
            revision=BGE_MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            BGE_MODEL_ID,
            revision=BGE_MODEL_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=False,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()
        hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        if hidden_size != self.dimension:
            raise RuntimeError(f"unexpected BGE embedding dimension: {hidden_size}")

    def _encode(self, texts: list[str], *, batch_size: int = 16) -> np.ndarray:
        if not texts or any(not item or item != item.strip() for item in texts):
            raise ValueError("BGE inputs must be non-empty trimmed strings")
        chunks: list[np.ndarray] = []
        torch = self._torch
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.inference_mode():
                embeddings = self.model(**tokens, return_dict=True).last_hidden_state[:, 0]
                embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
            chunks.append(embeddings.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_instruction + item for item in texts])

    def warmup(self) -> None:
        self.encode_queries(["warmup retrieval query"])
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)

    def peak_vram_bytes(self) -> int:
        if self.device.type != "cuda":
            return 0
        return int(self._torch.cuda.max_memory_allocated(self.device))


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "accelerate",
                "finops-r",
                "huggingface-hub",
                "numpy",
                "pydantic",
                "safetensors",
                "tokenizers",
                "torch",
                "transformers",
            )
        },
    }
    try:
        import torch

        identity["torch_cuda_version"] = torch.version.cuda
        identity["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            identity["gpu"] = {
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
            }
    except ImportError:
        identity["cuda_available"] = False
    return identity
