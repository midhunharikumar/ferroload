# Ferroload — API & Documentation Review

Audit of the **implemented** surface (`crates/ferroload-py/src/lib.rs`,
`python/ferroload/loader.py`, `python/ferroload/cli.py`, `ferroload-core`) against
the **docs** (`README.md`, `USAGE.md`, `EXAMPLES.md`, `DESIGN.md`, `python/README_HF.md`).
Grouped as: (A) doc↔code discrepancies, (B) API inconsistencies, (C) a tightened
API spec, (D) prioritized improvements.

> **Status — RESOLVED (as of 0.9.0).** All items below are implemented:
> A1–A5 (README + `lib.rs` docstring corrected), A6 (DESIGN §10 flagged
> aspirational), A7 (`PYTHON_API.md` added), A8 (`decode_audio` exposed),
> A9 (`ferroload.Sampler` + `loader.FerroSampler` wired), A10 (`read_batch` role
> documented). B1 (`Dataset`/`Writer` are the canonical pyclass names;
> `Ferro*` are aliases), B2/B4/B5/B9 (documented in `PYTHON_API.md`),
> B3 (`check_resize` + `(H,W)` documented), B6 (`core_err` →
> `IndexError`/`FileNotFoundError`/`RuntimeError`/`ValueError`),
> B7 (`__getitem__` + `name`/`version`/`modalities()`/`schema()`/`manifest()`;
> the CLI `inspect` now uses them), B8 (`loader.subset_dataset`). C captured in
> `PYTHON_API.md`. D1–D8 done (incl. versions unified to 0.9.0 + `CHANGELOG.md`).
> Remaining (roadmap, not review items): byte-budgeted prefetch + pinned/DLPack.
> The text below is the original audit, kept for reference.

---

## A. Doc ↔ implementation discrepancies

| # | Where | Doc says | Reality | Fix |
|---|---|---|---|---|
| A1 | `README.md` §Python extension | `cargo build` → `libferroload.so`, copy to `ferroload.so`, `import ferroload` | Extension is now `ferroload._core` inside a package; bare-`.so` copy import no longer works; build is via **maturin** | Replace with `maturin develop/build`; drop the copy trick |
| A2 | `README.md` workspace layout | `ferroload-py # PyO3 extension module ferroload` | Mixed package: `python/ferroload/{__init__,loader,cli}.py` + `_core` extension; also missing `notebooks/`, CLI, loader | Update tree |
| A3 | `README.md` Roadmap | "Wire io + codec + sampler into the Python Dataset … (codec not yet wired)" | `decode_many`/`decode_video` are wired; `FerroTorchDataset` exists | Move these to "done"; keep async-prefetch/sampler as open |
| A4 | `README.md` status matrix | "Python bindings (write/read/projection)" | Also: parallel decode, video decode, `read_batch`, `meta_batch`, `subset`, the torch loader, and a CLI + catalog | Expand the row; add CLI/loader rows |
| A5 | `lib.rs` module docstring (ll. 2–6) | "extension module `ferroload`"; "decoding … layer on top in Python" | Module is `_core`; decoding happens **in** the extension (`decode_many`/`decode_video`) | Update docstring |
| A6 | `DESIGN.md` §10 Python API | `ferroload.Dataset("s3://…", world_size, rank)`, `ds[i]`/`ds[a:b]`/`for x in ds`, `ds.subset("SELECT …")` → view, `worker_init_fn`, `collate_fn=train.collate`, `state_dict()` | Implemented: `FerroDataset.open(local)`, `.get(i, modalities)`, `.subset(WHERE)`→`list[int]`; no slicing/iter/worker_init/collate/state_dict; no remote/world_size at open | Mark §10 **aspirational**, or reconcile to the real surface |
| A7 | `USAGE.md` / `EXAMPLES.md` | Document the **Rust core** API only | Accurate for the core crate, but there is **no Python API reference** anywhere; the Python surface is shown only ad-hoc in README/README_HF | Add a dedicated `PYTHON_API.md` |
| A8 | `README.md` status: "Audio decode (WAV/PCM) ✅" | Implies usable | `WavCodec` exists in `ferroload-codec` but is **not exposed in Python** (no `decode_audio`); only raw bytes via `read` | Expose audio decode, or note "Rust-only" |
| A9 | `DESIGN.md` (sampler is central) | Deterministic rank×worker sampler drives loading | The Rust `Sampler` is implemented + tested but **unused by the Python loader**, which relies on torch's `DataLoader`/`DistributedSampler` | Either wire the Rust sampler in, or document that torch's sampler is used |
| A10 | `BENCHMARKS.md` uses `read_batch` | Presented as the batched path | The torch loader (`FerroTorchDataset`) uses `read_many`/`decode_many`, **not** `read_batch` — it's effectively orphaned | Use it in the loader, or fold into `read_many(contiguous=True)` |

---

## B. API inconsistencies (worth tightening)

- **B1 — name stutter.** Inside the `ferroload` package the classes are
  `ferroload.FerroDataset` / `ferroload.FerroWriter`. The `Ferro` prefix is
  redundant once namespaced. Prefer `ferroload.Dataset` / `ferroload.Writer`.
- **B2 — projection param shape.** `get(i, modalities: list[str])` is plural/list,
  but `read/read_many/read_batch/decode_many(modality: str)` are singular. The rule
  (list = projection of a full sample; str = one column) is sound but undocumented.
- **B3 — `resize=(H, W)` order** is unstated in the binding docstrings (it *is*
  height-first); easy to get backwards. Document and assert.
- **B4 — image-centric defaults.** `read/read_many/read_batch/decode_many` default
  `modality="image"`; on a video-only dataset every call must pass `modality=`.
  Either drop the default (force explicit) or make it the sole tensor modality when
  unambiguous.
- **B5 — return-type zoo.** `get`→dict, `read`→`bytes|None`, `read_many`→`list`,
  `read_batch`→`(bytes, spans)`, `decode_many`/`decode_video`→`list[ndarray|None]`,
  `meta_batch`→`dict`, `subset`→`list[int]`. Powerful but must be enumerated in one
  place (see §C).
- **B6 — error mapping.** Every Rust error becomes `ValueError`. Index out of range,
  missing file, and `ReaderTooOld` should map to `IndexError` / `FileNotFoundError`
  / `RuntimeError` respectively.
- **B7 — no Pythonic access / introspection.** `FerroDataset` has `__len__` but no
  `__getitem__` (so `ds[i]` fails; you must call `ds.get(i)`), and no
  `manifest`/`modalities`/`schema`/`version`/`name` accessors — the CLI re-parses
  `manifest.json` by hand because of this.
- **B8 — `subset` is a dead end.** It returns `list[int]`, but the reader has no way
  to be *restricted* to those ids — there's no `Subset` view nor an `indices=` hook
  on the loader, so the result can't feed training without the caller wiring a
  sampler. (The CLI only writes the ids to JSON.)
- **B9 — writer ergonomics.** `declare()` returns `None` and mutates (it
  take()/replace()s the inner builder); fine, but differs from the fluent Rust
  builder and isn't documented as stateful. `add(blobs: dict[str, bytes])` requires
  pre-encoded bytes (no tensor/array input).

---

## C. Tightened API spec (proposed canonical surface)

```text
ferroload.Writer(root: str, name: str)
    .declare(name: str, ext: str, kind="tensor", codec="raw") -> None
    .add(key: str, blobs: dict[str, bytes], meta: dict[str, scalar] | None = None) -> int
    .close() -> None                       # idempotent; required to commit

ferroload.Dataset.open(root: str) -> Dataset
    __len__() -> int
    __getitem__(i) -> dict                  # NEW: alias of get(i)
    name: str  version: int                 # NEW: properties
    modalities: dict[str, dict]  schema: list[dict]  # NEW: from manifest

    # single sample
    get(i, modalities: list[str] | None = None) -> dict
        # {sample_id, basename, <modality>: bytes, <modality>_present: bool, meta: dict}

    # one modality, raw bytes
    read(i, modality: str) -> bytes | None
    read_many(indices: list[int], modality: str) -> list[bytes | None]      # GIL-released
    read_batch(indices, modality) -> tuple[bytes, list[(off:int, len:int)]] # contiguous

    # one modality, decoded (parallel, GIL-released, zero-copy NumPy)
    decode_many(indices, modality="image", resize: (H,W) | None = None) -> list[ndarray|None]
    decode_video(indices, modality="video", num_frames=16, resize=(H,W)|None=None)
        -> list[ndarray|None]               # [T,H,W,3]; needs --features video

    # metadata (no I/O) + subsetting
    meta_batch(indices, keys: list[str]) -> dict[str, ndarray | list]
    subset(where_sql: str) -> list[int]     # ascending sample_ids
    verify() -> int
```

Decisions to lock:
- **Canonical names** `Dataset`/`Writer`; keep `FerroDataset`/`FerroWriter` as
  deprecated aliases for one release.
- **`resize` is `(height, width)`** everywhere; validate `> 0`.
- **Projection** = `modalities: list` (whole sample); **single-column** ops take
  `modality: str`. State this rule in the module docstring.
- **Errors:** `IndexError` for bad `i`, `FileNotFoundError` for missing root/shard,
  `RuntimeError` for `ReaderTooOld`, `ValueError` for predicate/parse errors.

---

## D. Prioritized improvements

1. **Fix the stale `README.md`** (A1–A4) and the `lib.rs` docstring (A5) — these are
   actively misleading (the build/import instructions no longer work). *High, cheap.*
2. **Add `PYTHON_API.md`** as the single Python reference (A7) and mark `DESIGN.md`
   §10 aspirational (A6). *High.*
3. **Drop the `Ferro` prefix** via package-level aliases (B1) and add
   `__getitem__` + `manifest/modalities/schema` properties (B7) — removes the CLI's
   hand-rolled JSON parsing. *Medium.*
4. **Make `subset` usable** (B8): add a `Subset`/`indices=` path so the ids feed the
   loader (and a materialized re-shard option per DESIGN §6). *Medium.*
5. **Expose audio decode** and unify a generic `decode(indices, modality, …)` that
   dispatches on the codec (A8). *Medium.*
6. **Wire the Rust sampler or document the torch one** (A9); decide `read_batch`'s
   fate (B10/A10). *Medium.*
7. **Precise exception types** (B6) and **`resize=(H,W)` validation** (B3). *Low.*
8. **Version hygiene:** `ferroload-py` is `0.8.0` while core/codec/io are `0.1.0`;
   align versions and add a short CHANGELOG. *Low.*

None of these are correctness bugs — the implemented surface works and is tested.
They're consistency/discoverability fixes; items 1–2 are the ones to do first
because the current README would actively mislead a new user.
