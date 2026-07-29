from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .contracts import Assessment, CaseSnapshot
from .json_safety import StrictJSONError, require_json_object, strict_json_loads
from .pipeline import assess_payload_with_bundle, load_production_bundle

MAX_ASSESSMENT_JSON_BYTES = 1_048_576


def _strict_request_object(body: bytes) -> dict[str, Any]:
    if len(body) > MAX_ASSESSMENT_JSON_BYTES:
        raise StrictJSONError(f"assessment request body exceeds {MAX_ASSESSMENT_JSON_BYTES} bytes")
    return require_json_object(strict_json_loads(body), source="assessment request body")


def create_app(run_dir: str | Path) -> Any:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise RuntimeError("Install the API extra with: pip install -e '.[api]'") from exc

    resolved = Path(run_dir)
    bundle = load_production_bundle(resolved)
    app = FastAPI(title="FinOps-R", version=__version__)

    @app.middleware("http")
    async def reject_ambiguous_json(request: Request, call_next: Any) -> Any:
        if request.method == "POST" and request.url.path == "/v1/cases/assess":
            try:
                declared_length = request.headers.get("content-length")
                if declared_length is not None and int(declared_length) > MAX_ASSESSMENT_JSON_BYTES:
                    raise StrictJSONError(
                        f"assessment request body exceeds {MAX_ASSESSMENT_JSON_BYTES} bytes"
                    )
                _strict_request_object(await request.body())
            except (StrictJSONError, ValueError):
                return JSONResponse(
                    status_code=400,
                    content={"detail": "invalid or ambiguous JSON"},
                )
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ready",
            "external_execution": False,
            "run_id": bundle["run_id"],
            "model_version": bundle["model_version"],
            "artifact_trust": "verified_local_only",
        }

    @app.post("/v1/cases/assess", response_model=Assessment)
    def assess(payload: CaseSnapshot) -> Assessment:
        result = assess_payload_with_bundle(bundle, payload.model_dump(mode="json"))
        return Assessment.model_validate(result)

    return app
