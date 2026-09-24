"""Lightweight HTTP client for communicating with the local rag-qdrant background service."""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from .config import (
    ADMIN_TOKEN,
    PACKAGE_ROOT,
    SERVICE_HOST,
    SERVICE_PID_FILE,
    SERVICE_PORT,
    SERVICE_URL,
)


class ServiceError(RuntimeError):
    """Raised when the service returns an error response."""
    pass


class ServiceUnavailableError(RuntimeError):
    """Raised when the background service cannot be reached or failed to start."""
    pass


class RagServiceClient:
    """Thin HTTP client for the persistent rag-qdrant service."""

    def __init__(
        self,
        base_url: str = SERVICE_URL,
        token: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token or ADMIN_TOKEN or "local-dev-token"
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Host": f"{SERVICE_HOST}:{SERVICE_PORT}",
            "User-Agent": "rag_qdrant_cli/2.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def is_healthy(self) -> bool:
        """Check if service is responding on health endpoint."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/health",
                headers={"Host": f"{SERVICE_HOST}:{SERVICE_PORT}"},
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def is_ready(self) -> bool:
        """Check if service is fully initialized with embedder and collection."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/ready",
                headers={"Host": f"{SERVICE_HOST}:{SERVICE_PORT}"},
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return bool(data.get("ready"))
        except Exception:
            return False
        return False

    def ensure_service_running(
        self,
        max_wait_seconds: float = 25.0,
        notify_cb: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Ensure background service is running; starts it on-demand if absent."""
        if self.is_ready():
            return True

        if notify_cb:
            notify_cb("Starting background rag_qdrant service...")

        # Spawn detached background process
        flags = 0
        if sys.platform == "win32":
            detached_process = 0x00000008
            create_new_process_group = 0x00000200
            flags = detached_process | create_new_process_group

        service_env = os.environ.copy()
        service_env["PYTHONUNBUFFERED"] = "1"
        service_env["PYTHONIOENCODING"] = "utf-8"

        subprocess.Popen(
            [sys.executable, "-m", "rag_qdrant.service"],
            creationflags=flags,
            close_fds=(sys.platform != "win32"),
            cwd=str(PACKAGE_ROOT),
            env=service_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            if self.is_ready():
                if notify_cb:
                    notify_cb("Service ready.")
                return True
            time.sleep(0.3)

        raise ServiceUnavailableError(
            f"rag_qdrant service failed to start or become ready within {max_wait_seconds}s. "
            f"Check logs or start manually with 'python -m rag_qdrant.service'."
        )

    def get_status(self, index_json: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if index_json:
            params["index_json"] = str(index_json)
        query_str = f"?{urllib.parse.urlencode(params)}" if params else ""
        return self._get(f"/v1/status{query_str}")

    def get_sources(self, index_json: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {}
        if index_json:
            params["index_json"] = str(index_json)
        query_str = f"?{urllib.parse.urlencode(params)}" if params else ""
        return self._get(f"/v1/sources{query_str}")

    def search(
        self,
        query: str,
        sources: Optional[str] = None,
        limit: int = 5,
        index_json: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "query": query,
            "limit": limit,
        }
        if sources:
            body["sources"] = sources
        if index_json:
            body["index_json"] = str(index_json)
        return self._post("/v1/search", body)

    def index_directory_stream(
        self,
        directory: Union[str, Path],
        source: str,
        index_json: Optional[Union[str, Path]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream real-time progress events from the indexing service."""
        body: Dict[str, Any] = {
            "path": str(directory),
            "source": str(source),
            "stream": True,
        }
        if index_json:
            body["index_json"] = str(index_json)

        payload = json.dumps(body).encode("utf-8")
        headers = self._headers()
        headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            f"{self.base_url}/v1/index?stream=true",
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=3600.0) as resp:
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str:
                        try:
                            yield json.loads(line_str)
                        except json.JSONDecodeError:
                            pass
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8")
            try:
                err_json = json.loads(error_body)
                raise ServiceError(err_json.get("error", f"HTTP {error.code}")) from error
            except json.JSONDecodeError:
                raise ServiceError(f"HTTP {error.code}: {error_body}") from error
        except urllib.error.URLError as error:
            raise ServiceUnavailableError(f"Could not connect to service: {error}") from error

    def stop_service(self) -> Dict[str, Any]:
        return self._post("/v1/service/stop", {})

    def _get(self, path: str) -> Any:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers=self._headers(),
            method="GET",
        )
        return self._send_request(req)

    def _post(self, path: str, data: Dict[str, Any]) -> Any:
        payload = json.dumps(data).encode("utf-8")
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=payload,
            headers=headers,
            method="POST",
        )
        return self._send_request(req)

    def _send_request(self, req: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8")
            try:
                err_json = json.loads(error_body)
                raise ServiceError(err_json.get("error", f"HTTP {error.code}")) from error
            except json.JSONDecodeError:
                raise ServiceError(f"HTTP {error.code}: {error_body}") from error
        except urllib.error.URLError as error:
            raise ServiceUnavailableError(f"Service unreachable at {self.base_url}: {error}") from error
