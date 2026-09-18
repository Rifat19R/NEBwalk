# Release checklist

Steps for publishing a new NEBwalk release to PyPI. `NEBwalk` is already
published (current: v0.10.0); trusted publishing via GitHub Actions OIDC is
configured and working.

## Local checks before tagging

```bash
pytest tests/ -v
ruff check src/NEBwalk/
mypy src/NEBwalk
python -m build
twine check dist/*
```

## Publish

1. Bump the version in `pyproject.toml` and `src/NEBwalk/__init__.py`
   (`__version__`) together — they must match.
2. Update `CHANGELOG.md`.
3. Push `main`.
4. Tag and push:

   ```bash
   VERSION=v0.11.0
   git tag -a "$VERSION" -m "NEBwalk $VERSION"
   git push origin main "$VERSION"
   ```

5. The `Publish` workflow (`.github/workflows/publish.yml`) builds the
   sdist/wheel, runs `twine check`, then publishes to PyPI via OIDC trusted
   publishing. No PyPI token is stored anywhere in the repo.

## After release

```bash
pip install --upgrade NEBwalk
python -c "import NEBwalk; print(NEBwalk.__version__)"
```

## PyPI trusted-publisher configuration (reference, already set up)

- PyPI project name: `NEBwalk`
- GitHub owner: `Rifat19R`
- GitHub repository: `NEBwalk`
- Workflow filename: `publish.yml`
- Environment name: `pypi`
