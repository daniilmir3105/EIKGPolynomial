# Release Guide (`eikgp-regressor`)

## 1) Pre-release checklist

- Update version in `pyproject.toml` and `CHANGELOG.md`.
- Ensure project URLs in `pyproject.toml` point to the real repository.
- Run local quality gates:
  - `ruff check .`
  - `python -m mypy eikg`
  - `python -m pytest`

## 2) Build artifacts

```powershell
venv\Scripts\python -m pip install -e ".[dev]"
venv\Scripts\python -m build
venv\Scripts\python -m twine check dist/*
```

## 3) Upload to TestPyPI (recommended)

```powershell
venv\Scripts\python -m twine upload --repository testpypi dist/*
```

Test install:

```powershell
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple eikgp-regressor
```

## 4) Upload to PyPI

```powershell
venv\Scripts\python -m twine upload dist/*
```

## 5) Git release (recommended)

```powershell
git add .
git commit -m "Release 0.1.1"
git tag v0.1.1
git push origin main --tags
```

## Notes

- Use API tokens for Twine auth (`__token__` username).
- Keep `dist/` clean between releases if you rebuild many times.
