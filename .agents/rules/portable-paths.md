---
trigger: always_on
description: Keep committed paths portable and avoid leaking local workstation locations.
---

# Portable paths

Use repository-relative paths, environment variables, or clearly marked placeholders in committed documentation and agent instructions. Do not persist a real user's home directory, local checkout location, or file URI. Runtime state paths in local configuration stay local and ignored by Git.

Tests may use synthetic absolute paths when the path shape is the behavior under test. The preflight catches common local-path leaks in staged added lines; inspect other added paths manually before committing.
