---
name: docs-code-parity
description: Audit whether rag_qdrant README claims and examples match current CLI, service, security, and Docker behavior when a documentation review is requested.
---

# Documentation and code parity

Use for a requested README or documentation accuracy audit, or when a behavior change directly affects published instructions. Do not run a whole-repository audit for a small unrelated edit.

Choose the affected claims in `README.md` and `rag-qdrant/README.md`. For each, inspect the current code, configuration, or test that establishes it. Check both directions: features or limits claimed by the docs but absent from code, and user-visible behavior missing from docs. In this repository, pay particular attention to CLI flags, the service lifecycle, authentication, source permissions, port binding, MCP transport, and whether performance numbers have current evidence.

Correct claims supported by current evidence. Mark uncertain measurements or live-service claims as unverified rather than inventing results. Preserve unrelated documentation and report the claims checked with their code or test locations.
