---
hide: # Optional: Hide table of contents on simple pages
  - toc
---

# Contributing Guide 🙌

First off, thanks for taking the time to contribute!  Flock is still early but we value outside input.  We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

---

## 1. Getting Started

1. **Fork** the repo & create your feature branch (`git checkout -b feat/my-awesome-thing`).
2. **Install** dev dependencies: `uv pip install -r requirements.txt && uv pip install -r requirements-dev.txt`.
3. **Run tests**: `pytest -q` (they should pass before and after your change).
4. **Make changes** – keep them atomic and well-documented.
5. **Lint & type-check**: `ruff .` and `mypy src/`.
6. **Commit** following *Conventional Commits* (`feat:`, `fix:`, `docs:` etc.).
7. **Open a PR** against `main`.  Include a description, screenshots, and linked issues.

---

## 2. Docs Contributions

Docs live under `docs/` and are built with **MkDocs Material**.

```bash
mkdocs serve  # live-reload at http://localhost:8000
```

* Use American English.
* Keep sentences short; favour lists & tables.
* Add code blocks with triple-backticks and language.

---

## 3. Coding Standards

* Python `3.10+`.  Use type hints everywhere.
* Follow `ruff` default rules + `black` formatting.
* Keep imports sorted (`ruff format`).
* Write **async-friendly** code (non-blocking I/O).

---

## 4. Tests

* Place new tests in `tests/` mirroring the package path.
* Use `pytest` fixtures instead of duplicating setup code.
* For Temporal code, rely on the *Temporal Test Server* fixture.

---

## 5. Release Process

1. Maintainer bumps version in `pyproject.toml` following *SemVer*.
2. Changelog entry added in `docs/about/changelog.md`.
3. `uv pip install -e .[all] && pytest`.
4. `git tag vX.Y.Z && git push --tags`.
5. GitHub Action publishes to PyPI.

---

Happy hacking! 🐦
