# Installation

Ferroload is a Cargo workspace of Rust crates with a maturin-built Python
extension (`ferroload._core`). For Python use you build and install that wheel;
for Rust use you depend on `ferroload-core` directly.

## Python package

It's a maturin mixed package — build and install it into a fresh virtualenv or
conda env:

```bash
cd crates/ferroload-py
pip install maturin
maturin develop --release          # dev install into the active env
# or build a wheel to pip-install elsewhere:
maturin build --release            # -> target/wheels/ferroload-*-abi3-*.whl

python -c "import ferroload; print(ferroload.__version__)"
ferroload --help                   # the CLI is installed too
```

!!! note "Build target on a mounted/output filesystem"
    Some filesystems block the temp-file deletes cargo does while linking. If a
    build fails that way, point the target at local disk:
    ```bash
    export CARGO_TARGET_DIR=/tmp/ferro-target
    ```

## Optional: video decode

In-Rust video decode (`decode_video`, and `videos=`/`columns=[…]` in the loader)
is **feature-gated** because it needs system ffmpeg + clang:

```bash
maturin develop --release --features video
```

=== "Linux"

    Install ffmpeg development packages from your distro (e.g.
    `apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev clang`),
    then build with `--features video`.

=== "macOS (Homebrew)"

    `ffmpeg-sys-next`'s bindgen runs clang, which defaults to `/usr/include`
    where Homebrew does **not** put headers — point it at the brew prefix or you'll
    get `fatal error: '.../libavcodec/avfft.h' file not found`:

    ```bash
    brew install ffmpeg pkg-config
    export BINDGEN_EXTRA_CLANG_ARGS="-I$(brew --prefix)/include"

    # If you build under conda, make sure Homebrew's pkg-config wins:
    #   which pkg-config   # should be under $(brew --prefix)/bin, not miniconda
    export PATH="$(brew --prefix)/bin:$PATH"

    cd crates/ferroload-py
    maturin develop --release --features video
    ```

Without the feature, `decode_video` (and video columns in the loader) raise a
clear error telling you to rebuild — everything else works unchanged.

## Cloud backends (S3 / GCS / Azure)

Streaming datasets from object storage (`Dataset.open("s3://…")` / `gs://` / `az://`)
is behind the `cloud` feature, which builds **all** cloud backends at once (they're
pure-Rust/rustls, so portable — no system libs like ffmpeg):

```bash
cd crates/ferroload-py
maturin develop --release --features cloud        # S3 + GCS + Azure
```

The published wheel already includes `cloud`, so `pip install ferroload` gets it.
Credentials come from the environment (`AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`,
`AZURE_*`). To build a single backend instead, use `--features aws` (or `gcp`/`azure`).

## Rust crates

Build and test the workspace directly:

```bash
cargo test                                       # core + io + codec
cargo test -p ferroload-core --features parquet  # + the parquet/arrow index backend
cargo run  -p ferroload-core --example synthetic_av
```

See [Rust core usage](rust/usage.md) to add `ferroload-core` as a dependency.
