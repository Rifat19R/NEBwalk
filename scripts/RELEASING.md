# Release checklist

Steps for publishing a new nebwalk release to PyPI. `nebwalk` is already
published (current: v0.10.0); trusted publishing via GitHub Actions OIDC is
configured and working.

## Local checks before tagging

```bash
pytest tests/ -v
ruff check src/nebwalk/
mypy src/nebwalk
python -m build
twine check dist/*
```

## Publish

1. Bump the version in `pyproject.toml` and `src/nebwalk/__init__.py`
   (`__version__`) together — they must match.
2. Update `CHANGELOG.md`.
3. Push `main`.
4. Tag and push:

   ```bash
   VERSION=v0.11.0
   git tag -a "$VERSION" -m "nebwalk $VERSION"
   git push origin main "$VERSION"
   ```

5. The `Publish` workflow (`.github/workflows/publish.yml`) builds the
   sdist/wheel, runs `twine check`, then publishes to PyPI via OIDC trusted
   publishing. No PyPI token is stored anywhere in the repo.

## After release

```bash
pip install --upgrade nebwalk
python -c "import nebwalk; print(nebwalk.__version__)"
```

## PyPI trusted-publisher configuration (reference, already set up)

- PyPI project name: `nebwalk`
- GitHub owner: `Rifat19R`
- GitHub repository: `nebwalk`
- Workflow filename: `publish.yml`
- Environment name: `pypi`
