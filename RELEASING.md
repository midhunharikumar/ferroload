# Releasing `ferroload` to PyPI

The package is built with [maturin](https://www.maturin.rs/) and published by CI
(`.github/workflows/release.yml`) as cross-platform **abi3** wheels (one wheel per
platform covers CPython 3.9+) plus an sdist.

## One-time setup

1. **Set the project URLs.** Replace `USERNAME` in
   `crates/ferroload-py/pyproject.toml` (`[project.urls]`) and in `mkdocs.yml`
   with your GitHub org/user.

2. **Configure PyPI Trusted Publishing** (recommended — no API token/secret):
   - Create the project on PyPI (or pre-register the name).
   - On PyPI → your project → *Publishing* → *Add a pending publisher*:
     - Owner / Repository: your `USERNAME/ferroload-rs`
     - Workflow name: `release.yml`
     - Environment name: `pypi`
   - In GitHub → repo *Settings* → *Environments*, create an environment named
     `pypi` (the workflow references it). Optionally add required reviewers.

   *Alternative (API token):* create a PyPI token, add it as the repo secret
   `PYPI_API_TOKEN`, and in `release.yml` remove the `id-token: write` permission
   and uncomment the `password: ${{ secrets.PYPI_API_TOKEN }}` line.

## Cut a release

1. Bump the version in **`crates/ferroload-py/Cargo.toml`** (`package.version`) —
   maturin reads the wheel version from the crate. Keep the workspace crates in
   sync if you also publish those. Update `CHANGELOG.md`.

2. Commit, then tag and push:

   ```bash
   git commit -am "Release v0.14.0"
   git tag v0.14.0
   git push origin main --tags
   ```

   The tag triggers `release.yml`: it builds wheels (Linux x86_64/aarch64, macOS
   arm64, Windows x64) + sdist, verifies they import on Python 3.9–3.13, and
   publishes them to PyPI. Publishing also runs when you publish a GitHub Release.

3. Verify:

   ```bash
   pip install --upgrade ferroload
   python -c "import ferroload; print(ferroload.__version__)"
   ```

## Test the build locally first

```bash
bash scripts/build-package.sh        # wheel + sdist into ./dist
python -m twine check dist/*         # validate metadata/long-description
pip install dist/ferroload-*.whl     # smoke-test the wheel
```

To dry-run the publish path, upload to TestPyPI first
(`twine upload --repository testpypi dist/*`).

## Notes

- The published wheel is built with **default features** (no `video`). In-Rust
  video decode needs system ffmpeg, so it isn't portable in a manylinux wheel;
  users who need it build from source (`maturin develop --features video`).
- The Rust library crates (`ferroload-core`/`io`/`codec`) are **not** published
  here — they'd go to crates.io separately (`cargo publish`). Ask if you want a
  crates.io release workflow added.
