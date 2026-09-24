---
name: service-boundary-review
description: Review changes to RAG service authentication, profile permissions, HTTP/MCP routes, or network exposure for boundary regressions.
---

# Service boundary review

Use when changing `security.py`, service authentication or routes, token configuration, or `docker-compose.yml`, and for an explicit security review. It does not run for ordinary indexing or documentation edits.

Inspect the changed request path from caller to handler. Verify the intended unauthenticated health behavior and denial of data endpoints without a valid token; profile-scoped search and index permissions; Host and Origin validation; canonical index JSON matching; and loopback binding in service and Compose configuration. Choose checks relevant to the changed path rather than treating this as a fixed checklist for every edit.

Run focused negative and positive tests through the in-process test client where possible. Run `tools/harness/preflight.py`, then full unit tests when dependencies are installed. Do not start or mutate a live service merely to finish the review. Report specific files and tests inspected, findings, and any boundary not exercised.
