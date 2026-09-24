"""Background RAG service exposing REST API and Streamable HTTP MCP server."""

import asyncio
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import uvicorn
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .config import (
    ADMIN_TOKEN,
    COLLECTION_NAME,
    MAX_REQUEST_BODY_BYTES,
    MAX_SEARCH_LIMIT,
    SCORE_THRESHOLD,
    SERVICE_HOST,
    SERVICE_PID_FILE,
    SERVICE_PORT,
)
from .core import ConcurrencyError, DimensionMismatchError, RagEngine
from .security import (
    ClientProfile,
    get_profile_by_token,
    validate_host_header,
    validate_origin_header,
)

logger = logging.getLogger("rag_qdrant.service")

# Global engine instance
_engine: Optional[RagEngine] = None


def get_engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine


# Initialize FastMCP server
mcp = FastMCP("rag-qdrant")


def _get_request_profile() -> Optional[ClientProfile]:
    """Retrieve client profile from active HTTP request."""
    try:
        req = get_http_request()
        if req and hasattr(req, "state") and hasattr(req.state, "profile"):
            return req.state.profile
    except Exception:
        pass
    # Default to admin profile if running locally without HTTP request context
    return get_profile_by_token(ADMIN_TOKEN)


@mcp.tool()
def search(
    query: str,
    sources: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Perform semantic search across indexed Markdown documentation.

    Args:
        query: The semantic search query text.
        sources: Optional comma-separated source tag(s) to filter by (e.g. 'project-a,project-b').
        limit: Maximum number of results to return (1-50, default 5).
    """
    engine = get_engine()
    profile = _get_request_profile()
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))
    try:
        results = engine.search(
            query=query,
            sources=sources,
            limit=limit,
            score_threshold=SCORE_THRESHOLD,
            profile=profile,
        )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except PermissionError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Search failed: {e}"}, ensure_ascii=False)


@mcp.tool()
def status() -> str:
    """Inspect Qdrant collection status, vector count, and health."""
    engine = get_engine()
    profile = _get_request_profile()
    try:
        payload = engine.get_status_payload(profile=profile)
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Status check failed: {e}"}, ensure_ascii=False)


@mcp.tool()
def list_sources() -> str:
    """List all indexed knowledge sources, file counts, and vector counts."""
    engine = get_engine()
    profile = _get_request_profile()
    try:
        sources = engine.get_sources_stats(profile=profile)
        return json.dumps(sources, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to list sources: {e}"}, ensure_ascii=False)


@mcp.tool()
def index(
    path: str,
    source: str,
) -> str:
    """Incrementally index Markdown documentation below a directory.

    Requires write permission and that the directory falls within permitted paths.
    """
    engine = get_engine()
    profile = _get_request_profile()
    if not profile or not profile.can_write:
        return json.dumps(
            {"error": "Unauthorized: client profile does not have write/indexing permissions."},
            ensure_ascii=False,
        )

    target_dir = Path(path).resolve()
    try:
        stats = engine.index_directory(
            directory=target_dir,
            source_name=source,
            profile=profile,
        )
        return json.dumps({"status": "ok", "stats": stats}, ensure_ascii=False, indent=2)
    except ConcurrencyError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except PermissionError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Indexing failed: {e}"}, ensure_ascii=False)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Protects against DNS rebinding, cross-origin tampering, payload exhaustion, and unauthorized access."""

    async def dispatch(self, request: Request, call_next):
        # 1. Validate Host header
        host = request.headers.get("host")
        if not validate_host_header(host):
            return JSONResponse({"error": "Forbidden: invalid Host header."}, status_code=403)

        # 2. Validate Origin header
        origin = request.headers.get("origin")
        if not validate_origin_header(origin):
            return JSONResponse({"error": "Forbidden: invalid Origin header."}, status_code=403)

        # 3. Validate content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
            return JSONResponse({"error": "Payload Too Large."}, status_code=413)

        # 4. Check unauthenticated endpoints
        path = request.url.path
        if path in ("/v1/health", "/v1/ready"):
            return await call_next(request)

        # 5. Authenticate via Bearer token, X-API-Key, or query param token
        token = ""
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        elif "x-api-key" in request.headers:
            token = request.headers["x-api-key"].strip()
        elif "token" in request.query_params:
            token = request.query_params["token"].strip()

        profile = get_profile_by_token(token)
        if not profile:
            return JSONResponse(
                {"error": "Unauthorized: invalid or missing authentication token."},
                status_code=401,
            )

        request.state.profile = profile
        return await call_next(request)


def _check_index_json_compatibility(engine: RagEngine, requested_path: Optional[str]) -> Optional[str]:
    """Verify that requested --index-json is compatible with service canonical index file."""
    if not requested_path:
        return None
    req = Path(requested_path).expanduser().resolve()
    canonical = engine.index_json.resolve()

    if req == canonical:
        return None

    return (
        f"The provided --index-json '{requested_path}' does not match the service canonical "
        f"index file ('{canonical.name}')."
    )


# --- REST Handlers ---

async def handle_health(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


async def handle_ready(request: Request) -> Response:
    engine = get_engine()
    ready = engine.fastembed_model is not None
    return JSONResponse({
        "ready": ready,
        "provider": engine.active_provider,
        "batch_size": engine.active_batch_size,
        "collection": COLLECTION_NAME,
        "index_json": str(engine.index_json),
    })


async def handle_status(request: Request) -> Response:
    engine = get_engine()
    profile: ClientProfile = request.state.profile
    index_json_arg = request.query_params.get("index_json")
    mismatch_error = _check_index_json_compatibility(engine, index_json_arg)
    if mismatch_error:
        return JSONResponse({"error": mismatch_error}, status_code=400)

    try:
        payload = engine.get_status_payload(profile=profile)
        return JSONResponse(payload)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def handle_sources(request: Request) -> Response:
    engine = get_engine()
    profile: ClientProfile = request.state.profile
    index_json_arg = request.query_params.get("index_json")
    mismatch_error = _check_index_json_compatibility(engine, index_json_arg)
    if mismatch_error:
        return JSONResponse({"error": mismatch_error}, status_code=400)

    try:
        sources = engine.get_sources_stats(profile=profile)
        return JSONResponse(sources)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def handle_search(request: Request) -> Response:
    engine = get_engine()
    profile: ClientProfile = request.state.profile
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    query = body.get("query")
    if not query:
        return JSONResponse({"error": "Field 'query' is required."}, status_code=400)

    index_json_arg = body.get("index_json")
    mismatch_error = _check_index_json_compatibility(engine, index_json_arg)
    if mismatch_error:
        return JSONResponse({"error": mismatch_error}, status_code=400)

    sources = body.get("source") or body.get("sources")
    limit = body.get("limit", 5)
    try:
        limit = int(limit)
    except ValueError:
        limit = 5
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))

    try:
        results = engine.search(
            query=query,
            sources=sources,
            limit=limit,
            score_threshold=SCORE_THRESHOLD,
            profile=profile,
        )
        return JSONResponse(results)
    except PermissionError as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def handle_index(request: Request) -> Response:
    engine = get_engine()
    profile: ClientProfile = request.state.profile

    if not profile.can_write:
        return JSONResponse(
            {"error": f"Client profile '{profile.name}' does not have write/indexing permissions."},
            status_code=403,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    path_str = body.get("path")
    source = body.get("source")
    if not path_str or not source:
        return JSONResponse(
            {"error": "Both 'path' and 'source' are required for indexing."},
            status_code=400,
        )

    index_json_arg = body.get("index_json")
    mismatch_error = _check_index_json_compatibility(engine, index_json_arg)
    if mismatch_error:
        return JSONResponse({"error": mismatch_error}, status_code=400)

    target_dir = Path(path_str).resolve()
    stream = body.get("stream", False) or request.query_params.get("stream") == "true"

    if not stream:
        try:
            stats = engine.index_directory(
                directory=target_dir,
                source_name=source,
                profile=profile,
            )
            return JSONResponse({"status": "ok", "stats": stats})
        except ConcurrencyError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        except PermissionError as e:
            return JSONResponse({"error": str(e)}, status_code=403)
        except DimensionMismatchError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # Streaming progress response
    async def event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        event_queue = asyncio.Queue()

        def progress_cb(completed, total, filename, action, is_bytes=True, count_str=""):
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {
                    "type": "progress",
                    "completed": completed,
                    "total": total,
                    "filename": filename,
                    "action": action,
                    "is_bytes": is_bytes,
                    "count_str": count_str,
                },
            )

        def plan_cb(plan):
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"type": "plan", "plan": plan},
            )

        def run_index():
            try:
                stats = engine.index_directory(
                    directory=target_dir,
                    source_name=source,
                    profile=profile,
                    progress_cb=progress_cb,
                    plan_cb=plan_cb,
                )
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    {"type": "complete", "stats": stats},
                )
            except Exception as exc:
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    {"type": "error", "error": str(exc)},
                )
            finally:
                loop.call_soon_threadsafe(event_queue.put_nowait, None)

        threading.Thread(target=run_index, daemon=True).start()

        while True:
            event = await event_queue.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


async def handle_stop(request: Request) -> Response:
    profile: ClientProfile = request.state.profile
    if profile.name != "admin":
        return JSONResponse({"error": "Only admin profile can stop the service."}, status_code=403)

    loop = asyncio.get_running_loop()
    loop.call_later(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    return JSONResponse({"status": "shutting_down"})


def create_app() -> Starlette:
    """Create the Starlette application with FastMCP mounted and REST routes."""
    # Pre-initialize engine
    get_engine()

    app = mcp.http_app()
    app.add_middleware(SecurityMiddleware)

    # Register custom REST routes
    routes = [
        Route("/v1/health", handle_health, methods=["GET"]),
        Route("/v1/ready", handle_ready, methods=["GET"]),
        Route("/v1/status", handle_status, methods=["GET"]),
        Route("/v1/sources", handle_sources, methods=["GET"]),
        Route("/v1/search", handle_search, methods=["POST"]),
        Route("/v1/index", handle_index, methods=["POST"]),
        Route("/v1/service/stop", handle_stop, methods=["POST"]),
    ]
    for r in routes:
        app.router.routes.append(r)

    return app


def write_pid_file():
    pid_path = Path(SERVICE_PID_FILE)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")


def remove_pid_file():
    pid_path = Path(SERVICE_PID_FILE)
    if pid_path.is_file():
        try:
            pid_path.unlink()
        except Exception:
            pass


def run_service():
    """Entrypoint to run the service process."""
    write_pid_file()
    try:
        app = create_app()
        uvicorn.run(
            app,
            host=SERVICE_HOST,
            port=SERVICE_PORT,
            log_level="info",
            access_log=False,
        )
    finally:
        remove_pid_file()


if __name__ == "__main__":
    run_service()
