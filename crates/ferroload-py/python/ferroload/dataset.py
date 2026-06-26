"""`ferroload.Dataset` — the reader handle.

Wraps the compiled core reader (`_core.Dataset`) and adds:
  - fluent framework views: `.torch()` / `.numpy()` / `.jax()`
  - `.subset(where_sql)` returning a **new (subset) Dataset** by default, or the
    raw `list[int]` of sample ids with `return_indices=True`.

A subset is a lightweight index-remapped view over the same underlying reader:
positions `0..len(subset)` map to the matching global sample ids, so `get`,
`decode_many`, `meta_batch`, the framework views, and even further `.subset()`
all work on it. Core descriptive methods (`name`, `version`, `modalities`,
`schema`, `manifest`, `verify`, `num_shards`) are delegated to the reader.
Use `ds.reader` for the raw `_core.Dataset`, `ds.indices` for the id map.
"""
import io

from ._core import Dataset as _CoreDataset
from .executor import MapPlan, detect_topology, select_executor
from .loader import FerroTorchDataset

# Output kinds for `map`:
#   tensor     -> a new modality, NumPy array serialized to .npy blobs in the layer
#   bytes      -> a new modality, RAW bytes written verbatim to the layer's shards
#                 (e.g. a downloaded mp4/jpg/flac), declared with an ext + codec
#   annotation -> a scalar/string stored inline in the layer index (merged into meta)
_TENSOR_KINDS = {"array", "tensor", "ndarray", "npy", "embedding"}
_ANNO_KINDS = {"scalar", "annotation", "text", "str", "string", "json", "number", "label"}
_BYTES_KINDS = {"bytes", "raw", "blob", "file", "binary"}
# raw-bytes conveniences: a bare kind that implies a sensible (ext, codec)
_BYTES_PRESET = {
    "video": ("mp4", "video"), "audio": ("wav", "audio"),
    "image": ("jpg", "image"), "jpg": ("jpg", "image"),
    "jpeg": ("jpg", "image"), "png": ("png", "image"),
}


class Modality:
    """A tensor/blob output declaration for `map` (DESIGN §13.3): becomes a new
    modality in the layer's shards. `Modality("npy")` stores NumPy arrays; any
    other ext (e.g. `Modality("png", codec="depth16")`, `Modality("mp4",
    codec="video")`) stores the **raw bytes** the fn returns."""
    __slots__ = ("ext", "codec", "kind")

    def __init__(self, ext, codec=None, kind="tensor"):
        self.ext = ext
        self.codec = codec
        self.kind = kind

    def __repr__(self):
        return f"Modality(ext={self.ext!r}, codec={self.codec!r})"


class Annotation:
    """A scalar/annotation output declaration for `map` (DESIGN §13.3): stored
    inline in the layer index and merged into `meta` on read (no shards)."""
    __slots__ = ()

    def __repr__(self):
        return "Annotation()"


def _norm_output(spec):
    """Normalize one output spec into `(kind, ext, codec)` where kind is
    'tensor' | 'bytes' | 'annotation'. Accepts:
      - `Modality(ext, codec)` / `Annotation()` (DESIGN §13.3 typed objects)
      - a string kind: 'array'/'tensor', 'scalar'/'text', 'bytes'/'raw',
        or a media shorthand 'video'/'audio'/'image'/'jpg'/'png'
      - a dict: {'type'|'kind': ..., 'ext': ..., 'codec': ...}
      - a tuple/list: (kind, ext[, codec])
    """
    if isinstance(spec, Annotation):
        return ("annotation", None, None)
    if isinstance(spec, Modality):
        # npy ext => NumPy array sink; any other ext => raw-bytes sink
        if (spec.ext or "npy").lower() == "npy":
            return ("tensor", "npy", spec.codec or "npy")
        return ("bytes", spec.ext, spec.codec or "raw")
    if isinstance(spec, dict):
        kind = (spec.get("type") or spec.get("kind") or "array")
        ext, codec = spec.get("ext"), spec.get("codec")
        return _resolve_output(kind, ext, codec)
    if isinstance(spec, (tuple, list)):
        kind = spec[0]
        ext = spec[1] if len(spec) > 1 else None
        codec = spec[2] if len(spec) > 2 else None
        return _resolve_output(kind, ext, codec)
    return _resolve_output(spec, None, None)


def _resolve_output(kind, ext, codec):
    k = kind.lower() if isinstance(kind, str) else "array"
    if k in _TENSOR_KINDS:
        return ("tensor", ext or "npy", codec or "npy")
    if k in _ANNO_KINDS:
        return ("annotation", None, None)
    if k in _BYTES_PRESET:  # 'video'/'audio'/'image'/... -> raw bytes modality
        pe, pc = _BYTES_PRESET[k]
        return ("bytes", ext or pe, codec or pc)
    if k in _BYTES_KINDS:
        return ("bytes", ext or "bin", codec or "raw")
    raise ValueError(
        f"unknown output kind {kind!r}; use 'array' (tensor), 'scalar' (metadata), "
        f"'bytes'/'video'/'audio'/'image' (raw blob), or a dict {{'type','ext','codec'}}")


def _npy_bytes(arr):
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def _as_tuple(ret, n, what):
    """Validate a multi-output return is a tuple/list of exactly `n` items."""
    if not isinstance(ret, (tuple, list)) or len(ret) != n:
        raise ValueError(
            f"map fn must return {n} {what} (one per output); got {ret!r}")
    return tuple(ret)


def _run_fn(fn, args, out_names, batch_len, batched):
    """Call `fn` with positionally-bound input columns and return
    `{output_name: [values]}` of length `batch_len`.

    Per-sample (`batched=False`): `fn` runs once per row on scalar args and
    returns one value (single output) or a tuple in `outputs` order. Batched:
    `fn` runs once on the column lists and returns one list (single output) or a
    tuple of lists. A `None` value/element is preserved and skipped downstream.
    """
    n_out = len(out_names)
    if batched:
        ret = fn(*args)
        cols = [ret] if n_out == 1 else list(_as_tuple(ret, n_out, "columns"))
        out = {}
        for name, col in zip(out_names, cols):
            col = list(col)
            if len(col) != batch_len:
                raise ValueError(
                    f"map output {name!r} has length {len(col)}, "
                    f"expected {batch_len} (batch size)")
            out[name] = col
        return out
    # per-sample
    out = {k: [] for k in out_names}
    for r in range(batch_len):
        ret = fn(*[a[r] for a in args])
        vals = (ret,) if n_out == 1 else _as_tuple(ret, n_out, "values")
        for k, v in zip(out_names, vals):
            out[k].append(v)
    return out


class Dataset:
    def __init__(self, inner, indices=None):
        self._inner = inner
        self._indices = None if indices is None else [int(x) for x in indices]

    @staticmethod
    def open(root, cache_dir=None):
        """Open a local path or a remote URL (``s3://``, ``gs://``, ``az://``,
        ``file://``, ``memory://``). For remote URLs, shard bytes stream via ranged
        GETs through a local cache at ``cache_dir`` (defaults to ``$FERROLOAD_CACHE``
        or a temp dir). Remote support needs a build with ``--features aws`` (or
        ``gcp``/``azure``)."""
        return Dataset(_CoreDataset.open(root, cache_dir))

    @property
    def reader(self):
        """The underlying `_core.Dataset` (full, unsubset)."""
        return self._inner

    @property
    def indices(self):
        """Global sample ids this view exposes, or `None` if it's the full dataset."""
        return self._indices

    # ---- index remapping (subset position -> global id) ----
    def _g(self, i):
        return self._indices[i] if self._indices is not None else int(i)

    def _gl(self, idxs):
        if self._indices is None:
            return [int(i) for i in idxs]
        return [self._indices[i] for i in idxs]

    # ---- subsetting ----
    def subset(self, where_sql, return_indices=False):
        """Filter by a SQL-ish `WHERE` over metadata. Returns a new (subset)
        `Dataset` by default, or `list[int]` of ids when `return_indices=True`."""
        ids = self._inner.subset(where_sql)            # ascending global ids
        if self._indices is not None:
            allowed = set(self._indices)
            ids = [i for i in ids if i in allowed]
        if return_indices:
            return ids
        return Dataset(self._inner, indices=ids)

    # ---- enrichment (map) ----
    def map(self, fn, inputs, outputs, name=None, batch_size=32, batched=False,
            resume=True, decode=True, resize=None, num_workers=0, progress=False,
            executor=None):
        """Enrich the dataset by running `fn` over it and storing the results as a
        new, additive **layer** (joined on `sample_id`). The base data is never
        rewritten, the pass is **idempotent + resumable**, and the outputs read
        back as ordinary modalities/metadata.

        `fn` is bound to its inputs **positionally**: it receives one argument per
        name in `inputs`, in order, and never references column names itself — so
        the same function is reusable across datasets. By default (`batched=False`)
        `fn` is **unitary** — it takes one sample's value(s) and returns that
        sample's output(s). With `batched=True` it takes one list per input and
        returns one list (or a tuple of lists) of outputs.

            # per-sample (default): generic, column-name-free
            def download(url):     return requests.get(url).content
            def mean_color(img):   return img.mean(axis=(0, 1)).astype("float32")
            ds.map(download,   inputs=["thumbnail_loc"], outputs={"image": Modality("image")})
            ds.map(mean_color, inputs=["image"],         outputs={"emb":   Modality("npy")})

            # multiple outputs -> return a tuple in `outputs` order
            def features(img, label):  return img.mean((0, 1)), f"class{label}"
            ds.map(features, inputs=["image", "label"],
                   outputs={"emb": Modality("npy"), "tag": Annotation()})

            # batched (vectorized) -> return a list, or a tuple of lists
            def emb(imgs):  return [im.mean((0, 1)) for im in imgs]
            ds.map(emb, inputs=["image"], outputs={"emb": Modality("npy")}, batched=True)

        Args:
          inputs:  modality and/or metadata names bound **positionally** to `fn`'s
                   arguments (a bare str is allowed for a single input). Image-codec
                   modalities arrive as `[H,W,C]` uint8 arrays (unless `decode=False`);
                   `.npy` tensor-layer outputs arrive as arrays; other modalities as
                   raw `bytes`; metadata keys as scalars. Absent → `None`.
          outputs: a list of names (all arrays) or a dict `{name: kind}` where kind
                   is one of:
                     - `'array'`/'tensor'  -> a new `.npy`-backed modality (NumPy)
                     - `'bytes'`/'raw', or media shorthands `'video'`/'audio'/
                       'image' -> a new modality storing the **raw bytes** `fn`
                       returns (e.g. a downloaded mp4). Use a dict
                       `{'type':'bytes','ext':'mp4','codec':'video'}` for full
                       control of the file extension/codec.
                     - `'scalar'`/'text'/'annotation' -> metadata
                   Also accepts the typed objects `Modality(...)` / `Annotation()`.
                   The output order defines how `fn`'s returned tuple is unpacked;
                   with a single output, return the value (or column) directly. A
                   returned `None` (per-sample) or `None` element (batched) skips
                   that sample — the layer is sparse.
          name:    layer name (default: `"map_" + "_".join(outputs)`).
          batched: if False (default) `fn` runs once per sample on scalar args; if
                   True it runs once per batch on the input column lists. Image
                   inputs are decoded in parallel in Rust either way, so per-sample
                   mode keeps the fast decode — only `fn` itself runs per row.
          batch_size: rows decoded/read per call (the unit of the resume loop).
          resume:  skip sample_ids already present in the layer (default True).
          decode:  decode image modalities to arrays (default True).
          resize:  optional `(h, w)` applied to decoded images.
          num_workers: reserved (intra-process decode is already parallel across
                   cores inside Rust with the GIL released).
          executor: an `Executor` (see `ferroload.executor`). Default: auto-select
                   from the launch topology — `LocalExecutor` on a single node,
                   `StaticPartitionExecutor` under torchrun/SLURM (each rank writes
                   a layer fragment; a single commit merges them), `RayExecutor`
                   multi-node. `FERROLOAD_EXECUTOR=local|static|ray` overrides.

        Returns a fresh `Dataset` with the new layer visible (the same subset
        view, if `self` is a subset).
        """
        import numpy as np  # noqa: F401  (used by _npy_bytes / asarray)

        inputs = [inputs] if isinstance(inputs, str) else list(inputs)
        if isinstance(outputs, dict):
            norm = {k: _norm_output(v) for k, v in outputs.items()}
        else:
            specs = [outputs] if isinstance(outputs, str) else outputs
            norm = {k: _norm_output("array") for k in specs}
        if not norm:
            raise ValueError("map: `outputs` must declare at least one output")
        # norm: {name: (kind, ext, codec)} with kind in tensor|bytes|annotation
        tensor_outs = [k for k, (kind, *_ ) in norm.items() if kind == "tensor"]
        bytes_outs = [k for k, (kind, *_ ) in norm.items() if kind == "bytes"]
        anno_outs = [k for k, (kind, *_ ) in norm.items() if kind == "annotation"]
        out_names = list(norm.keys())

        layer_name = name or ("map_" + "_".join(norm.keys()))
        mods = self.modalities()  # {name: {ext, kind, codec}}
        # tensor + raw-bytes outputs are blob modalities stored in the layer shards
        mod_decls = {}
        for k in tensor_outs + bytes_outs:
            _, ext, codec = norm[k]
            mod_decls[k] = (ext, "tensor", codec)

        root = self._inner.root
        n = len(self)

        def read_inputs(positions):
            cols = {}
            for inp in inputs:
                m = mods.get(inp)
                if m is not None and decode and m.get("codec") == "image":
                    cols[inp] = self.decode_many(positions, inp, resize)
                elif m is not None and m.get("codec") == "npy":
                    # a tensor layer output (e.g. from a previous map) -> arrays
                    cols[inp] = self.read_arrays(positions, inp)
                elif m is not None:
                    cols[inp] = self.read_many(positions, inp)
                else:  # metadata key
                    cols[inp] = list(self.meta_batch(positions, [inp])[inp])
            return cols

        # process a set of positions into a (possibly partition-local) writer
        def process(writer, positions):
            done = set(writer.existing_ids()) if resume else set()
            written = 0
            for start in range(0, len(positions), batch_size):
                chunk = positions[start:start + batch_size]
                todo = [p for p in chunk if self._g(p) not in done]
                if not todo:
                    continue
                cols = read_inputs(todo)
                args = [cols[i] for i in inputs]      # bind columns -> positional args
                out = _run_fn(fn, args, out_names, len(todo), batched)
                for j, pos in enumerate(todo):
                    sid = self._g(pos)
                    blobs, meta = {}, {}
                    for k in tensor_outs:
                        v = out[k][j]
                        if v is not None:
                            blobs[k] = _npy_bytes(np.asarray(v))
                    for k in bytes_outs:
                        v = out[k][j]
                        if v is not None:
                            blobs[k] = bytes(v)    # raw passthrough (e.g. mp4 bytes)
                    for k in anno_outs:
                        v = out[k][j]
                        if v is not None:
                            meta[k] = v
                    writer.add(sid, blobs or None, meta or None)
                    written += 1
                if progress:
                    print(f"  map[{layer_name}]: +{written}", flush=True)
            return written

        plan = MapPlan(root=root, layer_name=layer_name, modalities=mod_decls,
                       total=n, process=process, sample_id=self._g, progress=progress)
        ex = executor if executor is not None else select_executor(detect_topology())
        ex.run(plan)

        enriched = Dataset.open(root)
        if self._indices is not None:
            return Dataset(enriched._inner, indices=list(self._indices))
        return enriched

    # ---- array read-back (consume .npy tensor outputs from a layer) ----
    def read_array(self, i, modality):
        """Read a tensor-layer (`.npy`) modality for sample `i` as a NumPy array
        (or `None` if absent)."""
        import numpy as np
        raw = self._inner.read(self._g(i), modality)
        return None if raw is None else np.load(io.BytesIO(raw), allow_pickle=False)

    def read_arrays(self, indices, modality):
        """Read a tensor-layer (`.npy`) modality for many indices: list of NumPy
        arrays (`None` where absent)."""
        import numpy as np
        raws = self._inner.read_many(self._gl(indices), modality)
        return [None if r is None else np.load(io.BytesIO(r), allow_pickle=False) for r in raws]

    # ---- framework views (pass self so subsets are respected) ----
    def torch(self, **kwargs):
        return FerroTorchDataset(self, out="torch", **kwargs)

    def numpy(self, **kwargs):
        return FerroTorchDataset(self, out="numpy", **kwargs)

    def jax(self, **kwargs):
        return FerroTorchDataset(self, out="jax", **kwargs)

    def iterable(self, **kwargs):
        """Streaming (`IterableDataset`) view over this dataset — the counterpart
        to `.torch()`/`.numpy()`. Reads contiguous, shard-local blocks sequentially
        and yields through a shuffle buffer (WebDataset-style), which is the
        object-store-friendly access pattern. Hand it to a `DataLoader` *without* a
        sampler. Same `columns=`/`resize=`/`out=` config as the map views, plus
        `shuffle_buffer=`, `block_size=`, and `world_size`/`rank`/`seed` for DDP.
        """
        from .loader import FerroIterableDataset
        return FerroIterableDataset(self, **kwargs)

    # ---- reads (remapped) ----
    def get(self, i, modalities=None):
        return self._inner.get(self._g(i), modalities)

    def read(self, i, modality="image"):
        return self._inner.read(self._g(i), modality)

    def read_many(self, indices, modality="image"):
        return self._inner.read_many(self._gl(indices), modality)

    def read_batch(self, indices, modality="image"):
        return self._inner.read_batch(self._gl(indices), modality)

    def decode_many(self, indices, modality="image", resize=None):
        return self._inner.decode_many(self._gl(indices), modality, resize)

    def decode_audio(self, indices, modality="audio"):
        return self._inner.decode_audio(self._gl(indices), modality)

    def decode_video(self, indices, modality="video", num_frames=16, resize=None):
        # only present on inner when built with the `video` feature
        return self._inner.decode_video(self._gl(indices), modality, num_frames, resize)

    def meta_batch(self, indices, keys):
        return self._inner.meta_batch(self._gl(indices), keys)

    # ---- python protocol + delegation ----
    def __len__(self):
        return len(self._indices) if self._indices is not None else len(self._inner)

    def __getitem__(self, i):
        return self._inner[self._g(i)]

    def __getattr__(self, name):
        # name/version/modalities/schema/manifest/verify/num_shards -> reader
        return getattr(self._inner, name)

    def __repr__(self):
        sub = "" if self._indices is None else f", subset of {len(self._inner)}"
        return f"Dataset(len={len(self)}{sub}, name={self._inner.name!r})"
