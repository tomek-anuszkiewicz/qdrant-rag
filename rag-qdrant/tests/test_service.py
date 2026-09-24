"""Tests for the persistent RAG background service, security middleware, and REST/MCP APIs."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from rag_qdrant.config import ADMIN_TOKEN, AMIGA_TOKEN, DEVNOTES_TOKEN
from rag_qdrant.core import ConcurrencyError, RagEngine
from rag_qdrant.security import (
    ClientProfile,
    get_profile_by_token,
    validate_host_header,
    validate_index_authorization,
    validate_origin_header,
    validate_search_sources,
)
from rag_qdrant.service import create_app


class SecurityValidationTests(unittest.TestCase):
    def test_host_header_validation(self):
        self.assertTrue(validate_host_header("127.0.0.1:6335"))
        self.assertTrue(validate_host_header("localhost:6335"))
        self.assertTrue(validate_host_header("[::1]:6335"))
        self.assertTrue(validate_host_header("localhost"))
        self.assertFalse(validate_host_header("evil.com"))
        self.assertFalse(validate_host_header("example.org:6335"))
        self.assertFalse(validate_host_header(None))

    def test_origin_header_validation(self):
        self.assertTrue(validate_origin_header(None))  # Non-browser clients
        self.assertTrue(validate_origin_header(""))
        self.assertTrue(validate_origin_header("http://127.0.0.1:6335"))
        self.assertTrue(validate_origin_header("http://localhost:6335"))
        self.assertFalse(validate_origin_header("http://attacker.com"))
        self.assertFalse(validate_origin_header("https://evil.org:6335"))

    def test_token_resolution(self):
        admin_prof = get_profile_by_token(ADMIN_TOKEN)
        self.assertIsNotNone(admin_prof)
        self.assertEqual(admin_prof.name, "admin")
        self.assertTrue(admin_prof.can_read)
        self.assertTrue(admin_prof.can_write)

        self.assertIsNone(get_profile_by_token("invalid-secret-token"))

    def test_search_sources_scoping(self):
        # Admin can search all
        admin = ClientProfile(name="admin", token="t1", can_read=True, allowed_search_sources=["*"])
        sources, err = validate_search_sources(None, admin)
        self.assertIsNone(sources)
        self.assertIsNone(err)

        # Amiga profile restricted to amiga and devnotes
        amiga = ClientProfile(name="amiga", token="t2", can_read=True, allowed_search_sources=["amiga", "devnotes"])
        sources, err = validate_search_sources(None, amiga)
        self.assertEqual(sources, ["amiga", "devnotes"])
        self.assertIsNone(err)

        # Requesting allowed source
        sources, err = validate_search_sources(["amiga"], amiga)
        self.assertEqual(sources, ["amiga"])
        self.assertIsNone(err)

        # Requesting forbidden source
        sources, err = validate_search_sources(["secret_docs"], amiga)
        self.assertIsNone(sources)
        self.assertIn("Source 'secret_docs' is not permitted", err)

    def test_index_authorization(self):
        read_only = ClientProfile(name="reader", token="t3", can_read=True, can_write=False)
        allowed, err = validate_index_authorization(Path.cwd(), "amiga", read_only)
        self.assertFalse(allowed)
        self.assertIn("read-only access", err)

        scoped_writer = ClientProfile(
            name="scoped",
            token="t4",
            can_read=True,
            can_write=True,
            allowed_index_sources=["amiga"],
            allowed_index_directories=[str(Path.cwd())],
        )
        allowed, err = validate_index_authorization(Path.cwd(), "amiga", scoped_writer)
        self.assertTrue(allowed)
        self.assertIsNone(err)

        # Wrong source
        allowed, err = validate_index_authorization(Path.cwd(), "other", scoped_writer)
        self.assertFalse(allowed)
        self.assertIn("Source 'other' is not permitted", err)


class ServiceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.headers = {
            "Host": "127.0.0.1:6335",
            "Authorization": f"Bearer {ADMIN_TOKEN}",
        }

    def test_health_endpoint_is_unauthenticated(self):
        resp = self.client.get("/v1/health", headers={"Host": "127.0.0.1:6335"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_ready_endpoint_returns_engine_metadata(self):
        resp = self.client.get("/v1/ready", headers={"Host": "127.0.0.1:6335"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("ready"))
        self.assertEqual(data.get("collection"), "projects_docs")

    def test_host_header_rejection(self):
        resp = self.client.get("/v1/health", headers={"Host": "malicious.com"})
        self.assertEqual(resp.status_code, 403)

    def test_origin_header_rejection(self):
        resp = self.client.get("/v1/health", headers={"Host": "127.0.0.1:6335", "Origin": "http://evil.com"})
        self.assertEqual(resp.status_code, 403)

    def test_status_endpoint_requires_auth(self):
        resp = self.client.get("/v1/status", headers={"Host": "127.0.0.1:6335"})
        self.assertEqual(resp.status_code, 401)

    def test_status_endpoint_with_auth(self):
        resp = self.client.get("/v1/status", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("collection"), "projects_docs")
        self.assertIn("total_vectors", data)

    def test_sources_endpoint_with_auth(self):
        resp = self.client.get("/v1/sources", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)

    def test_search_endpoint_with_query(self):
        resp = self.client.post(
            "/v1/search",
            headers=self.headers,
            json={"query": "test query", "limit": 2},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)

    def test_search_endpoint_rejects_missing_query(self):
        resp = self.client.post(
            "/v1/search",
            headers=self.headers,
            json={"limit": 2},
        )
        self.assertEqual(resp.status_code, 400)

    def test_incompatible_index_json_rejected(self):
        resp = self.client.get(
            "/v1/status?index_json=arbitrary_unauthorized.json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("does not match the service canonical", resp.json()["error"])
