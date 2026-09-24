"""Tests for the persistent RAG background service, security middleware, and REST/MCP APIs."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from rag_qdrant.config import ADMIN_TOKEN
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

    def test_unconfigured_default_token_is_rejected(self):
        with patch("rag_qdrant.security.ADMIN_TOKEN", ""), patch(
            "rag_qdrant.security.CLIENT_PROFILES_JSON", ""
        ):
            self.assertIsNone(get_profile_by_token("local-dev-token"))

    def test_client_profiles_are_read_only(self):
        profile_list = [{
            "name": "reader", "token": "reader-token",
            "allowed_search_sources": ["project-a"],
        }]
        with patch("rag_qdrant.security.CLIENT_PROFILES_JSON", json.dumps(profile_list)):
            profile = get_profile_by_token("reader-token")
            self.assertEqual(profile.name, "reader")
            self.assertEqual(profile.allowed_search_sources, ["project-a"])
            self.assertFalse(profile.can_write)
            self.assertIsNone(get_profile_by_token("unknown-token"))

    def test_client_list_rejects_write_settings(self):
        profile_list = [{
            "name": "reader", "token": "reader-token",
            "allowed_search_sources": ["project-a"], "can_write": True,
        }]
        with patch("rag_qdrant.security.CLIENT_PROFILES_JSON", json.dumps(profile_list)):
            with self.assertRaisesRegex(ValueError, "only name"):
                get_profile_by_token("reader-token")

    def test_duplicate_admin_token_is_rejected(self):
        profile_list = [{
            "name": "other", "token": ADMIN_TOKEN,
            "allowed_search_sources": [],
        }]
        with patch("rag_qdrant.security.CLIENT_PROFILES_JSON", json.dumps(profile_list)):
            with self.assertRaisesRegex(ValueError, "unique"):
                get_profile_by_token(ADMIN_TOKEN)

    def test_client_list_cannot_define_admin(self):
        profile_list = [{
            "name": "admin", "token": "different-token",
            "allowed_search_sources": [],
        }]
        with patch("rag_qdrant.security.CLIENT_PROFILES_JSON", json.dumps(profile_list)):
            with self.assertRaisesRegex(ValueError, "cannot be admin"):
                get_profile_by_token("different-token")

    def test_search_sources_scoping(self):
        # Admin can search all
        admin = ClientProfile(name="admin", token="t1", can_read=True, allowed_search_sources=["*"])
        sources, err = validate_search_sources(None, admin)
        self.assertIsNone(sources)
        self.assertIsNone(err)

        scoped = ClientProfile(name="reader", token="t2", can_read=True, allowed_search_sources=["project-a", "project-b"])
        sources, err = validate_search_sources(None, scoped)
        self.assertEqual(sources, ["project-a", "project-b"])
        self.assertIsNone(err)

        sources, err = validate_search_sources(["project-a"], scoped)
        self.assertEqual(sources, ["project-a"])
        self.assertIsNone(err)

        sources, err = validate_search_sources(["secret_docs"], scoped)
        self.assertIsNone(sources)
        self.assertIn("Source 'secret_docs' is not permitted", err)

    def test_index_authorization(self):
        read_only = ClientProfile(name="reader", token="t3", can_read=True, can_write=False)
        allowed, err = validate_index_authorization(Path.cwd(), "project-a", read_only)
        self.assertFalse(allowed)
        self.assertIn("read-only access", err)

        scoped_writer = ClientProfile(
            name="scoped",
            token="t4",
            can_read=True,
            can_write=True,
            allowed_index_sources=["project-a"],
            allowed_index_directories=[str(Path.cwd())],
        )
        allowed, err = validate_index_authorization(Path.cwd(), "project-a", scoped_writer)
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

    def test_same_index_filename_in_other_directory_is_rejected(self):
        from rag_qdrant.service import _check_index_json_compatibility, get_engine

        engine = get_engine()
        other = engine.index_json.parent / "other" / engine.index_json.name
        self.assertIsNotNone(_check_index_json_compatibility(engine, str(other)))

    def test_incompatible_index_json_rejected(self):
        resp = self.client.get(
            "/v1/status?index_json=arbitrary_unauthorized.json",
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("does not match the service canonical", resp.json()["error"])
