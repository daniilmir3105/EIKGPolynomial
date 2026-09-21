# Release Guide (`eikgpolynomial` 0.2.0)

This document is the release checklist for version **0.2.0**, dated **2026-09-21**. The
distribution name used by PyPI and `pip` is `eikgpolynomial`; the Python import package is
`eikg`.

The primary production publishing path is the GitHub Actions workflow
`.github/workflows/publish.yml` with PyPI Trusted Publishing. A manual Twine upload is retained
only as a fallback.

Every artifact must be built afresh from the exact commit that passed CI and will receive the
release tag. Never upload files merely because they already exist in a local `dist` directory.
Artifact timestamps, names, or matching source diffs are not proof of their provenance.

## 1. One-time account and repository setup

1. Create and verify the maintainer accounts on both
   [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/). They are separate services
   and use separate credentials.
2. Enable two-factor authentication on PyPI and TestPyPI. Store the recovery codes in a secure
   location outside the repository.
3. Open https://pypi.org/project/eikgpolynomial/ and
   https://test.pypi.org/project/eikgpolynomial/. For this first upload both pages must be
   missing (HTTP 404). If either project already exists under another account, stop and choose
   a different distribution name. Do not reuse the retired `eikgp-regressor` name.
4. In the GitHub repository, create a protected environment named `pypi`. Configure required
   reviewers or other deployment protections if desired.
5. Register a **pending** GitHub Trusted Publisher before the first upload. The PyPI project
   does not exist yet, so this is done from the account Publishing page
   (https://pypi.org/manage/account/publishing/), not from an existing project. Repeat the same
   pending-publisher form on TestPyPI
   (https://test.pypi.org/manage/account/publishing/) if TestPyPI Trusted Publishing will be
   used. Fill the form with exactly these values:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `eikgpolynomial` |
   | Owner | `daniilmir3105` |
   | Repository name | `EIKGPolynomial` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

The pending publisher is consumed by the first successful Trusted Publishing upload and then
becomes the ordinary publisher for the created `eikgpolynomial` project. Trusted Publishing
exchanges GitHub's short-lived OIDC identity for a short-lived PyPI token; no long-lived
production token needs to be stored in GitHub Secrets.

## 2. Verify the release commit

Before the release commit, confirm all version-bearing documentation is synchronized:

- `pyproject.toml` contains version `0.2.0`;
- the source-tree fallback in `eikg/__init__.py` is `0.2.0`;
- `CHANGELOG.md` contains `## [0.2.0] - 2026-09-21`;
- README installation and release-guide links refer to `0.2.0`;
- this guide names the same version, tag, artifact filenames, and release date.

Confirm that the working tree contains only intended release changes:

```powershell
git status --short
git diff --check
git diff -- pyproject.toml eikg/__init__.py CHANGELOG.md README.md RELEASE.md MANIFEST.in
```

Run every local quality gate from the repository root in a fresh or updated development
environment. The Markdown extra is required so Twine can actually render the README instead of
silently skipping Markdown rendering:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy eikg
python -c "from pathlib import Path; from readme_renderer.markdown import render; assert render(Path('README.md').read_text(encoding='utf-8'), stream=None) is not None"
```

Commit and push the release preparation changes. Do not create the tag yet:

```powershell
git add .
git commit -m "Prepare release 0.2.0"
git push origin main
```

If the default branch is protected, use the normal pull-request and review process instead of
pushing directly. Fetch the remote, require a clean tree, and record the exact release commit:

```powershell
git fetch origin
if (git status --porcelain) { throw "The release worktree is not clean." }
$releaseSha = (git rev-parse HEAD).Trim()
$originSha = (git rev-parse origin/main).Trim()
if ($releaseSha -ne $originSha) { throw "HEAD does not match origin/main." }
$releaseSha
```

Wait for every CI job on exactly `$releaseSha` to complete successfully. Keep that SHA in the
current PowerShell session or copy it into the release record; all later build and tag checks
refer to it.

## 3. Build clean artifacts

Start from the repository root at the recorded green commit. If this is a new PowerShell
session, first set `$releaseSha` to the full SHA recorded in step 2. Require the expected project
root and exact commit before removing the fixed set of local build outputs:

```powershell
if (-not (Test-Path -LiteralPath .\pyproject.toml)) { throw "Run from the repository root." }
if ((git rev-parse HEAD).Trim() -ne $releaseSha) { throw "HEAD is not the green release commit." }
if (git status --porcelain) { throw "The release worktree is not clean." }

$buildOutputs = @(
  ".\build",
  ".\dist",
  ".\eikgpolynomial.egg-info",
  ".\eikgp_regressor.egg-info",
  ".\eikg_regressor.egg-info"
)
foreach ($path in $buildOutputs) {
  if (Test-Path -LiteralPath $path) {
    Remove-Item -LiteralPath $path -Recurse -Force
  }
}
foreach ($path in $buildOutputs) {
  if (Test-Path -LiteralPath $path) { throw "Stale build output remains: $path" }
}

python -m build
```

The new directory must contain exactly these two files:

```text
dist/eikgpolynomial-0.2.0-py3-none-any.whl
dist/eikgpolynomial-0.2.0.tar.gz
```

Reject extra files, confirm that the source commit did not change during the build, validate the
README renderer, and check both artifacts strictly:

```powershell
$expectedArtifacts = @(
  "eikgpolynomial-0.2.0-py3-none-any.whl",
  "eikgpolynomial-0.2.0.tar.gz"
) | Sort-Object
$actualArtifacts = @(Get-ChildItem -LiteralPath .\dist -File | ForEach-Object Name) | Sort-Object
$artifactDifference = Compare-Object $expectedArtifacts $actualArtifacts
if ($artifactDifference) { throw "dist contains missing or unexpected artifacts." }
if ((git rev-parse HEAD).Trim() -ne $releaseSha) { throw "HEAD changed during the build." }

python -c "from pathlib import Path; from readme_renderer.markdown import render; assert render(Path('README.md').read_text(encoding='utf-8'), stream=None) is not None"
python -m twine check --strict `
  .\dist\eikgpolynomial-0.2.0-py3-none-any.whl `
  .\dist\eikgpolynomial-0.2.0.tar.gz
Get-FileHash -Algorithm SHA256 `
  .\dist\eikgpolynomial-0.2.0-py3-none-any.whl, `
  .\dist\eikgpolynomial-0.2.0.tar.gz
```

Save the SHA-256 values with the release record. Do not use `dist/*` for uploads: an old or
unrelated artifact must never be uploaded by accident.

## 4. Test the exact release on TestPyPI

Upload only the two named 0.2.0 artifacts. Use a TestPyPI project-scoped API token when the
project already exists, or an account-scoped token for its first upload. Confirm once more in
the TestPyPI web interface that version 0.2.0 does not already exist:

```powershell
$env:TWINE_USERNAME = "__token__"
python -m twine upload --repository testpypi `
  .\dist\eikgpolynomial-0.2.0-py3-none-any.whl `
  .\dist\eikgpolynomial-0.2.0.tar.gz
Remove-Item Env:TWINE_USERNAME
```

Paste the complete TestPyPI token into Twine's hidden password prompt. Never commit a token,
assign its literal value in a shell command that may enter command history, place it in a
command-line argument, paste it into an issue, or reuse a TestPyPI token on production PyPI.

Create a clean virtual environment. Install runtime dependencies from production PyPI first,
then install the exact TestPyPI artifact with `--no-deps`. This avoids using TestPyPI as an
untrusted dependency source and prevents an unrelated dependency package from that index from
being selected:

```powershell
python -m venv .venv-testpypi
.\.venv-testpypi\Scripts\python.exe -m pip install --upgrade pip
.\.venv-testpypi\Scripts\python.exe -m pip install "numpy>=1.24"
.\.venv-testpypi\Scripts\python.exe -m pip install `
  --index-url https://test.pypi.org/simple/ `
  --no-deps `
  --no-cache-dir `
  eikgpolynomial==0.2.0
.\.venv-testpypi\Scripts\python.exe -c `
  "import eikg; from importlib.metadata import version; assert eikg.__version__ == version('eikgpolynomial') == '0.2.0'; from eikg import CombinatorialPolynomialNetwork, DeepPolyNetwork, DeepPolyNetworkCV, EIKGPolynomialRegressor, EIKGPolynomialRegressorCV; print(eikg.__version__)"
```

Run a small fit/predict smoke test and inspect the TestPyPI project page. Confirm that the README,
license, Python requirement, classifiers, project URLs, and absolute release-guide link render
correctly. Delete the disposable environment after verification.

If any source, metadata, or documentation change is required after the TestPyPI upload, do not
try to replace the 0.2.0 files. TestPyPI also retains filename history. Increment the version,
update every version-bearing file, create a new release commit, rerun CI, record its new SHA,
rebuild from empty output directories, and repeat the TestPyPI check with the new version.

## 5. Tag, create the GitHub Release, and publish to PyPI

After TestPyPI verification, ensure `HEAD` is still the exact green release commit. Create and
push an annotated tag whose version matches `pyproject.toml`:

```powershell
git status --short
git tag -a v0.2.0 -m "Release 0.2.0"
git push origin v0.2.0
```

Create a GitHub Release for the existing `v0.2.0` tag, use the 0.2.0 changelog section as the
release notes, and publish the release. The `release: published` event starts
`.github/workflows/publish.yml`. That workflow builds the artifacts and publishes them through
the protected `pypi` environment and Trusted Publishing.

Review the workflow logs and the resulting PyPI project page. Then verify the production package
from another clean virtual environment:

```powershell
python -m venv .venv-pypi-check
.\.venv-pypi-check\Scripts\python.exe -m pip install --upgrade pip
.\.venv-pypi-check\Scripts\python.exe -m pip install eikgpolynomial==0.2.0
.\.venv-pypi-check\Scripts\python.exe -c `
  "from importlib.metadata import version; from eikg import EIKGPolynomialRegressor; print(version('eikgpolynomial'))"
```

Do not move or recreate `v0.2.0` after publication. If GitHub release publication fails before
PyPI receives any files, diagnose the workflow and rerun the failed job. If PyPI received the
files, inspect the project page before attempting any retry.

## 6. Manual production upload (fallback only)

Use this path only when Trusted Publishing cannot be restored and the exact locally built
artifacts have passed all checks. Create a short-lived, project-scoped production PyPI API token
and keep two-factor authentication enabled:

```powershell
$env:TWINE_USERNAME = "__token__"
python -m twine upload `
  .\dist\eikgpolynomial-0.2.0-py3-none-any.whl `
  .\dist\eikgpolynomial-0.2.0.tar.gz
Remove-Item Env:TWINE_USERNAME
```

Paste the complete production token into Twine's hidden password prompt, then revoke the
temporary token after the upload. Never store it in the repository, workflow file, shell
history, or an unprotected GitHub secret.

## 7. PyPI immutability and release recovery

PyPI does not allow a distribution filename to be reused, even if the original file is deleted.
Published `eikgpolynomial-0.2.0` artifacts therefore cannot be replaced in place. If a published
artifact is incorrect:

1. Do not delete and retry version 0.2.0 expecting the filename to become available.
2. Fix the source and tests.
3. Increment the version, normally to `0.2.1` for a compatible bug fix.
4. Add a changelog entry and repeat this entire checklist with the new version and tag.

A release may be yanked on PyPI to discourage new installations, but yanking also does not make
its filenames reusable.
