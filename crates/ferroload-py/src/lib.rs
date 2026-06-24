//! PyO3 bindings exposing the Ferroload format core to Python.
//!
//! Builds the CPython extension `ferroload._core`, wrapped by the pure-Python
//! `ferroload` package (`__init__.py`, `loader.py`, `cli.py`). It provides the
//! writer/reader, projection, **parallel in-Rust image/video decode** (GIL
//! released), metadata columns, and subsetting. The PyTorch glue
//! (`FerroTorchDataset`) lives in `ferroload.loader`. See `PYTHON_API.md`.

use ferroload_core::dataset::{Dataset, DatasetWriter};
use ferroload_core::manifest::Modality;
use pyo3::exceptions::{
    PyFileNotFoundError, PyIndexError, PyOSError, PyRuntimeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde_json::Value;
use std::collections::BTreeMap;

/// Generic Display -> ValueError (for non-core errors: numpy, parse, messages).
fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Map a core error to the most appropriate Python exception type.
fn core_err(e: ferroload_core::Error) -> PyErr {
    use ferroload_core::Error as E;
    match e {
        E::NotFound(s) => PyIndexError::new_err(s),
        E::ReaderTooOld { required, have } => {
            PyRuntimeError::new_err(format!("reader too old: requires {required}, have {have}"))
        }
        E::Io(io) if io.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(io.to_string())
        }
        E::Io(io) => PyOSError::new_err(io.to_string()),
        E::Format(s) => PyValueError::new_err(s),
        E::Json(j) => PyValueError::new_err(j.to_string()),
    }
}

/// Validate a `(height, width)` resize argument (both must be > 0).
fn check_resize(resize: Option<(usize, usize)>) -> PyResult<()> {
    if let Some((h, w)) = resize {
        if h == 0 || w == 0 {
            return Err(PyValueError::new_err(
                "resize must be (height, width) with both > 0",
            ));
        }
    }
    Ok(())
}

/// Recursively convert a JSON value (incl. nested dict/list) to a Python object.
fn value_to_py(py: Python<'_>, v: &Value) -> PyObject {
    match v {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_py(py),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py(py)
            } else {
                n.as_f64().unwrap_or(0.0).into_py(py)
            }
        }
        Value::String(s) => s.into_py(py),
        Value::Array(a) => {
            let l = PyList::empty_bound(py);
            for x in a {
                l.append(value_to_py(py, x)).unwrap();
            }
            l.into()
        }
        Value::Object(o) => {
            let d = PyDict::new_bound(py);
            for (k, x) in o {
                d.set_item(k, value_to_py(py, x)).unwrap();
            }
            d.into()
        }
    }
}

/// Parse a `{name: (ext,kind,codec) | (ext,codec) | ext}` modality dict (used by
/// the layer writer + commit) into core `Modality`s.
fn parse_modalities(modalities: Option<&Bound<'_, PyDict>>) -> PyResult<BTreeMap<String, Modality>> {
    let mut mods = BTreeMap::new();
    if let Some(modalities) = modalities {
        for (k, v) in modalities.iter() {
            let mname: String = k.extract()?;
            let m = if let Ok(s) = v.extract::<String>() {
                Modality::tensor(&s, "raw")
            } else if let Ok((ext, kind, codec)) = v.extract::<(String, String, String)>() {
                Modality { ext, kind, codec, attrs: Default::default() }
            } else if let Ok((ext, codec)) = v.extract::<(String, String)>() {
                Modality::tensor(&ext, &codec)
            } else {
                return Err(err(format!(
                    "modality '{mname}' must be ext:str or (ext, kind, codec) / (ext, codec)"
                )));
            };
            mods.insert(mname, m);
        }
    }
    Ok(mods)
}

/// Convert a Python scalar/object into a JSON metadata value.
fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::from(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::from(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::from(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::from(s));
    }
    // fallback: stringify (e.g. lists/dicts -> their repr)
    Ok(Value::from(obj.str()?.to_string()))
}

fn json_to_py(py: Python<'_>, v: &Value) -> PyObject {
    match v {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_py(py),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py(py)
            } else {
                n.as_f64().unwrap_or(0.0).into_py(py)
            }
        }
        Value::String(s) => s.into_py(py),
        other => other.to_string().into_py(py),
    }
}

/// Streaming writer for a self-contained dataset root.
/// Exposed to Python as `ferroload.Writer` (`FerroWriter` is a back-compat alias).
#[pyclass(name = "Writer")]
struct FerroWriter {
    inner: Option<DatasetWriter>,
}

#[pymethods]
impl FerroWriter {
    #[new]
    fn new(root: &str, name: &str) -> PyResult<Self> {
        Ok(FerroWriter {
            inner: Some(DatasetWriter::create(root, name).map_err(core_err)?),
        })
    }

    /// Declare a modality: name, file extension, kind (tensor|scalar|...), codec.
    #[pyo3(signature = (name, ext, kind="tensor", codec="raw"))]
    fn declare(&mut self, name: &str, ext: &str, kind: &str, codec: &str) -> PyResult<()> {
        let w = self.inner.take().ok_or_else(|| err("writer is closed"))?;
        let m = Modality {
            ext: ext.into(),
            kind: kind.into(),
            codec: codec.into(),
            attrs: Default::default(),
        };
        self.inner = Some(w.declare(name, m));
        Ok(())
    }

    /// Add a sample. `blobs`: dict[str, bytes]; `meta`: dict[str, scalar].
    #[pyo3(signature = (key, blobs, meta=None))]
    fn add(
        &mut self,
        key: &str,
        blobs: &Bound<'_, PyDict>,
        meta: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<u64> {
        let w = self.inner.as_mut().ok_or_else(|| err("writer is closed"))?;
        let mut b = BTreeMap::new();
        for (k, v) in blobs.iter() {
            let key: String = k.extract()?;
            let bytes: Vec<u8> = v.extract()?;
            b.insert(key, bytes);
        }
        let mut m = BTreeMap::new();
        if let Some(meta) = meta {
            for (k, v) in meta.iter() {
                m.insert(k.extract::<String>()?, py_to_json(&v)?);
            }
        }
        w.add(key, &b, &m).map_err(core_err)
    }

    /// Finalize: write index + manifest + version snapshot.
    fn close(&mut self) -> PyResult<()> {
        let w = self.inner.take().ok_or_else(|| err("writer already closed"))?;
        w.close().map_err(core_err)?;
        Ok(())
    }
}

/// Reader over a self-contained dataset root.
/// Exposed to Python as `ferroload.Dataset` (`FerroDataset` is a back-compat alias).
#[pyclass(name = "Dataset")]
struct FerroDataset {
    inner: Dataset,
}

#[pymethods]
impl FerroDataset {
    #[staticmethod]
    fn open(root: &str) -> PyResult<Self> {
        Ok(FerroDataset {
            inner: Dataset::open(root).map_err(core_err)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Map-style access: `ds[i]` == `ds.get(i)` (full sample dict).
    fn __getitem__(&self, py: Python<'_>, i: usize) -> PyResult<PyObject> {
        self.get(py, i, None)
    }

    fn num_shards(&self) -> u32 {
        self.inner.shard_count()
    }

    /// Filesystem root of this dataset (used by `map` to write enrichment layers).
    #[getter]
    fn root(&self) -> String {
        self.inner.root().to_string_lossy().into_owned()
    }

    /// Dataset name (from the manifest).
    #[getter]
    fn name(&self) -> String {
        self.inner.manifest().name.clone()
    }

    /// Current dataset version.
    #[getter]
    fn version(&self) -> u32 {
        self.inner.manifest().version
    }

    /// Declared modalities as `{name: {ext, kind, codec, ...}}`, including those
    /// contributed by enrichment layers (so `map` outputs are discoverable).
    fn modalities(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut merged = self.inner.manifest().modalities.clone();
        for layer in &self.inner.manifest().layers {
            for (k, v) in &layer.modalities {
                merged.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
        let v = serde_json::to_value(&merged).map_err(err)?;
        Ok(value_to_py(py, &v))
    }

    /// Index schema columns as a list of `{name, dtype, semantic?, attrs?}`.
    fn schema(&self, py: Python<'_>) -> PyResult<PyObject> {
        let v = serde_json::to_value(&self.inner.manifest().schema).map_err(err)?;
        Ok(value_to_py(py, &v))
    }

    /// The full manifest as a Python dict (incl. `extensions`).
    fn manifest(&self, py: Python<'_>) -> PyResult<PyObject> {
        let v = serde_json::to_value(self.inner.manifest()).map_err(err)?;
        Ok(value_to_py(py, &v))
    }

    /// Materialize sample `i`. `modalities=None` reads all; otherwise only those
    /// (projection). Returns a dict with bytes per present modality, a
    /// `<m>_present` flag per requested modality, `meta`, and `sample_id`.
    #[pyo3(signature = (i, modalities=None))]
    fn get(
        &self,
        py: Python<'_>,
        i: usize,
        modalities: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let sample = self
            .inner
            .get(i, modalities.as_deref())
            .map_err(core_err)?;
        let d = PyDict::new_bound(py);
        d.set_item("sample_id", sample.sample_id)?;
        d.set_item("basename", &sample.basename)?;
        for (m, bytes) in &sample.blobs {
            d.set_item(m, PyBytes::new_bound(py, bytes))?;
        }
        for (m, present) in &sample.present {
            d.set_item(format!("{m}_present"), *present)?;
        }
        let meta = PyDict::new_bound(py);
        for (k, v) in &sample.meta {
            meta.set_item(k, json_to_py(py, v))?;
        }
        d.set_item("meta", meta)?;
        Ok(d.into())
    }

    /// Minimal-overhead read of one modality's bytes for sample `i`
    /// (no dict/meta allocation). Returns `bytes` or `None` if absent.
    #[pyo3(signature = (i, modality="image"))]
    fn read(&self, py: Python<'_>, i: usize, modality: &str) -> PyResult<Option<Py<PyBytes>>> {
        match self.inner.read_blob(i, modality).map_err(core_err)? {
            Some(b) => Ok(Some(PyBytes::new_bound(py, &b).unbind())),
            None => Ok(None),
        }
    }

    /// Batched read of one modality for many indices. Performs all file I/O in
    /// Rust with the **GIL released**, then returns a list of bytes/None. This is
    /// the path a DataLoader worker should use — it amortizes the Python<->Rust
    /// boundary and overlaps with other Python threads.
    #[pyo3(signature = (indices, modality="image"))]
    fn read_many(&self, py: Python<'_>, indices: Vec<usize>, modality: &str) -> PyResult<PyObject> {
        let modality = modality.to_string();
        let blobs: Result<Vec<Option<Vec<u8>>>, _> =
            py.allow_threads(|| indices.iter().map(|&i| self.inner.read_blob(i, &modality)).collect());
        let blobs = blobs.map_err(core_err)?;
        let list = PyList::empty_bound(py);
        for b in blobs {
            match b {
                Some(b) => list.append(PyBytes::new_bound(py, &b))?,
                None => list.append(py.None())?,
            }
        }
        Ok(list.into())
    }

    /// Read one modality for many indices into a single contiguous buffer.
    /// Returns `(bytes, spans)` where `spans[k] == (offset, length)`. One
    /// allocation + one copy total — slice with `memoryview(buf)[o:o+l]`
    /// (zero-copy). Faster than `read_many` for large blobs (no per-item PyBytes).
    #[pyo3(signature = (indices, modality="image"))]
    fn read_batch(&self, py: Python<'_>, indices: Vec<usize>, modality: &str) -> PyResult<PyObject> {
        let modality = modality.to_string();
        let (buf, spans) = py
            .allow_threads(|| self.inner.read_blobs_contig(&indices, &modality))
            .map_err(core_err)?;
        let bytes = PyBytes::new_bound(py, &buf);
        let spans_py = PyList::empty_bound(py);
        for (o, l) in spans {
            spans_py.append((o, l))?;
        }
        Ok((bytes, spans_py).into_py(py))
    }

    /// Read **and decode** an image modality for many indices, in parallel in
    /// Rust with the **GIL released** (rayon across cores), returning a list of
    /// **zero-copy** NumPy `uint8` arrays shaped `[H, W, C]`. The decoded buffer
    /// is moved into NumPy (no serial copy), so this scales with cores and beats
    /// single-threaded Python/PIL — the real bottleneck for large images.
    #[pyo3(signature = (indices, modality="image", resize=None))]
    fn decode_many(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
        modality: &str,
        resize: Option<(usize, usize)>,
    ) -> PyResult<PyObject> {
        use ferroload_codec::{image_codec::ImageCodec, Codec, TensorData};
        use numpy::{IntoPyArray, PyArrayMethods};
        use rayon::prelude::*;

        check_resize(resize)?;
        let modality = modality.to_string();
        // absent modality -> None entry (tolerant, for sparse/flexible combos)
        let decoded: Result<Vec<Option<(usize, usize, usize, Vec<u8>)>>, String> =
            py.allow_threads(|| {
                indices
                    .par_iter()
                    .map(|&i| {
                        match self.inner.read_blob(i, &modality).map_err(|e| e.to_string())? {
                            Some(bytes) => {
                                let t = match resize {
                                    Some((h, w)) => ImageCodec.decode_resized(&bytes, h, w),
                                    None => ImageCodec.decode(&bytes),
                                }
                                .map_err(|e| e.to_string())?;
                                let (h, w, c) = (t.shape[0], t.shape[1], t.shape[2]);
                                match t.data {
                                    TensorData::U8(v) => Ok(Some((h, w, c, v))),
                                    _ => Err("expected u8 image".to_string()),
                                }
                            }
                            None => Ok(None),
                        }
                    })
                    .collect()
            });
        let decoded = decoded.map_err(err)?;
        let list = PyList::empty_bound(py);
        for opt in decoded {
            match opt {
                Some((h, w, c, data)) => {
                    // move the Vec into NumPy (no copy), then reshape (a view)
                    let arr = data.into_pyarray_bound(py).reshape((h, w, c)).map_err(err)?;
                    list.append(arr)?;
                }
                None => list.append(py.None())?,
            }
        }
        Ok(list.into())
    }

    /// Fetch metadata columns for a batch of indices **without any shard I/O**
    /// (metadata lives inline in the in-RAM index). Returns a dict
    /// `key -> [B] array` (typed when uniform) or a Python list (strings/nested).
    /// Batching/stacking is left to the DataLoader's `collate_fn` (idiomatic),
    /// not done here.
    #[pyo3(signature = (indices, keys))]
    fn meta_batch(&self, py: Python<'_>, indices: Vec<usize>, keys: Vec<String>) -> PyResult<PyObject> {
        let out = PyDict::new_bound(py);
        for key in keys {
            let vals: Vec<Option<Value>> = indices
                .iter()
                .map(|&i| self.inner.row(i).ok().and_then(|r| r.meta.get(&key).cloned()))
                .collect();
            out.set_item(&key, meta_column(py, &vals))?;
        }
        Ok(out.into())
    }

    /// Read **and decode video** for many indices: temporal-subsample `num_frames`
    /// frames per clip and decode them in parallel in Rust (ffmpeg, GIL released).
    /// Returns a list of `[T, H, W, 3]` uint8 NumPy arrays (None if absent).
    /// Only available when built with `--features video`.
    #[cfg(feature = "video")]
    #[pyo3(signature = (indices, modality="video", num_frames=16, resize=None))]
    fn decode_video(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
        modality: &str,
        num_frames: usize,
        resize: Option<(usize, usize)>,
    ) -> PyResult<PyObject> {
        use ferroload_codec::sampling::Sampling;
        use ferroload_codec::video::{VideoCodec, VideoConfig};
        use ferroload_codec::{Codec, TensorData};
        use numpy::{IntoPyArray, PyArrayMethods};
        use rayon::prelude::*;

        check_resize(resize)?;
        let modality = modality.to_string();
        let cfg = VideoConfig { num_frames, sampling: Sampling::Uniform, resize };
        let decoded: Result<Vec<Option<(usize, usize, usize, usize, Vec<u8>)>>, String> = py
            .allow_threads(|| {
                indices
                    .par_iter()
                    .map(|&i| {
                        match self.inner.read_blob(i, &modality).map_err(|e| e.to_string())? {
                            Some(bytes) => {
                                let t = VideoCodec::new(cfg).decode(&bytes).map_err(|e| e.to_string())?;
                                let (tt, h, w, c) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
                                match t.data {
                                    TensorData::U8(v) => Ok(Some((tt, h, w, c, v))),
                                    _ => Err("expected u8 video".to_string()),
                                }
                            }
                            None => Ok(None),
                        }
                    })
                    .collect()
            });
        let decoded = decoded.map_err(err)?;
        let list = PyList::empty_bound(py);
        for opt in decoded {
            match opt {
                Some((tt, h, w, c, data)) => {
                    let arr = data.into_pyarray_bound(py).reshape((tt, h, w, c)).map_err(err)?;
                    list.append(arr)?;
                }
                None => list.append(py.None())?,
            }
        }
        Ok(list.into())
    }

    /// Subset by a SQL-ish `WHERE` predicate over metadata; returns matching
    /// sample_ids (ascending). E.g. "duration_s < 16 AND lang = 'en'".
    fn subset(&self, where_sql: &str) -> PyResult<Vec<u64>> {
        self.inner.subset(where_sql).map_err(core_err)
    }

    /// Read **and decode audio** (WAV/PCM) for many indices, in parallel in Rust.
    /// Returns a list of `[channels, samples]` float32 NumPy arrays (None if absent).
    #[pyo3(signature = (indices, modality="audio"))]
    fn decode_audio(&self, py: Python<'_>, indices: Vec<usize>, modality: &str) -> PyResult<PyObject> {
        use ferroload_codec::{audio_wav::WavCodec, Codec, TensorData};
        use numpy::{IntoPyArray, PyArrayMethods};
        use rayon::prelude::*;

        let modality = modality.to_string();
        let decoded: Result<Vec<Option<(usize, usize, Vec<f32>)>>, String> = py.allow_threads(|| {
            indices
                .par_iter()
                .map(|&i| match self.inner.read_blob(i, &modality).map_err(|e| e.to_string())? {
                    Some(bytes) => {
                        let t = WavCodec.decode(&bytes).map_err(|e| e.to_string())?;
                        let (c, n) = (t.shape[0], t.shape[1]);
                        match t.data {
                            TensorData::F32(v) => Ok(Some((c, n, v))),
                            _ => Err("expected f32 audio".to_string()),
                        }
                    }
                    None => Ok(None),
                })
                .collect()
        });
        let decoded = decoded.map_err(err)?;
        let list = PyList::empty_bound(py);
        for opt in decoded {
            match opt {
                Some((c, n, data)) => {
                    let arr = data.into_pyarray_bound(py).reshape((c, n)).map_err(err)?;
                    list.append(arr)?;
                }
                None => list.append(py.None())?,
            }
        }
        Ok(list.into())
    }

    /// Verify all shards/members read back to declared lengths.
    fn verify(&self) -> PyResult<usize> {
        self.inner.verify().map_err(core_err)
    }
}

/// Deterministic distributed sampler over the `world_size x num_workers` grid.
/// Returns a disjoint, reproducible, block-shuffled slice of sample indices per
/// `(rank, worker)` from `(seed, epoch)` — the basis of `FerroSampler` and of
/// resumable training. Mirrors `ferroload_core::Sampler`.
#[pyclass]
struct Sampler {
    inner: ferroload_core::Sampler,
}

#[pymethods]
impl Sampler {
    #[new]
    #[pyo3(signature = (total, world_size=1, rank=0, num_workers=1, worker_id=0,
                        seed=0, shuffle=true, shuffle_block=1024))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        total: usize,
        world_size: u32,
        rank: u32,
        num_workers: u32,
        worker_id: u32,
        seed: u64,
        shuffle: bool,
        shuffle_block: usize,
    ) -> PyResult<Self> {
        let ws = world_size.max(1);
        let nw = num_workers.max(1);
        if rank >= ws {
            return Err(PyValueError::new_err("rank must be < world_size"));
        }
        if worker_id >= nw {
            return Err(PyValueError::new_err("worker_id must be < num_workers"));
        }
        let inner = ferroload_core::Sampler::new(total, world_size, rank, num_workers, worker_id)
            .seed(seed)
            .shuffle_block(shuffle_block)
            .with_shuffle(shuffle);
        Ok(Sampler { inner })
    }

    /// Indices for this `(rank, worker)` at `epoch`, skipping `resume_from`
    /// already-consumed items (for exact-resume).
    #[pyo3(signature = (epoch=0, resume_from=0))]
    fn indices(&self, epoch: u64, resume_from: usize) -> Vec<u32> {
        self.inner.indices(epoch, resume_from)
    }
}

/// Turn a column of optional JSON values into a typed NumPy array (when all
/// present values share a numeric/bool type) or a Python list otherwise.
fn meta_column(py: Python<'_>, vals: &[Option<Value>]) -> PyObject {
    use numpy::IntoPyArray;
    let all_present = vals.iter().all(|v| v.is_some());
    let all_int = all_present && vals.iter().all(|v| v.as_ref().unwrap().is_i64());
    let all_num = all_present && vals.iter().all(|v| v.as_ref().unwrap().is_number());
    let all_bool = all_present && vals.iter().all(|v| v.as_ref().unwrap().is_boolean());

    if all_bool {
        let col: Vec<bool> = vals.iter().map(|v| v.as_ref().unwrap().as_bool().unwrap()).collect();
        col.into_pyarray_bound(py).into()
    } else if all_int {
        let col: Vec<i64> = vals.iter().map(|v| v.as_ref().unwrap().as_i64().unwrap()).collect();
        col.into_pyarray_bound(py).into()
    } else if all_num {
        let col: Vec<f64> = vals.iter().map(|v| v.as_ref().unwrap().as_f64().unwrap()).collect();
        col.into_pyarray_bound(py).into()
    } else {
        let list = PyList::empty_bound(py);
        for v in vals {
            match v {
                Some(val) => list.append(json_to_py(py, val)).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list.into()
    }
}

/// Streaming writer for an enrichment **layer** — the low-level sink behind
/// `Dataset.map`. Writes new tensor modalities to `layers/<name>/shards/` and
/// scalar/annotation outputs into the layer's index, then registers the layer in
/// the manifest on `close()`. Re-opening an existing layer **appends** (the basis
/// of resumable maps): `existing_ids()` reports what's already done.
#[pyclass(name = "LayerWriter")]
struct PyLayerWriter {
    inner: Option<ferroload_core::dataset::LayerWriter>,
}

#[pymethods]
impl PyLayerWriter {
    /// `modalities`: dict `name -> (ext, kind, codec)` or `name -> ext` (str).
    /// Tensor outputs declared here are written to the layer's shards; annotation/
    /// scalar outputs need no declaration and go inline in the layer index.
    /// `partition`: if set, write a partition-local fragment (distributed map) —
    /// finish the job with `LayerWriter.commit(root, name, modalities)`.
    #[new]
    #[pyo3(signature = (root, name, modalities=None, partition=None))]
    fn new(
        root: &str,
        name: &str,
        modalities: Option<&Bound<'_, PyDict>>,
        partition: Option<u32>,
    ) -> PyResult<Self> {
        let mods = parse_modalities(modalities)?;
        let inner = match partition {
            None => ferroload_core::dataset::LayerWriter::create(root, name, mods),
            Some(p) => ferroload_core::dataset::LayerWriter::create_partition(root, name, mods, p),
        }
        .map_err(core_err)?;
        Ok(PyLayerWriter { inner: Some(inner) })
    }

    /// Merge all partition fragments into the layer and register it in the manifest
    /// (the single sync point of a distributed map). `modalities` must match what
    /// the partition writers declared.
    #[staticmethod]
    #[pyo3(signature = (root, name, modalities=None))]
    fn commit(root: &str, name: &str, modalities: Option<&Bound<'_, PyDict>>) -> PyResult<u64> {
        let mods = parse_modalities(modalities)?;
        ferroload_core::dataset::LayerWriter::commit(root, name, mods).map_err(core_err)
    }

    /// `sample_id`s already present in this layer (for resume — skip these).
    fn existing_ids(&self) -> PyResult<Vec<u64>> {
        let w = self.inner.as_ref().ok_or_else(|| err("layer writer is closed"))?;
        Ok(w.existing_ids())
    }

    /// Add one enriched sample. `blobs`: dict[str, bytes] for tensor modalities;
    /// `meta`: dict[str, scalar] for annotation/scalar outputs.
    #[pyo3(signature = (sample_id, blobs=None, meta=None))]
    fn add(
        &mut self,
        sample_id: u64,
        blobs: Option<&Bound<'_, PyDict>>,
        meta: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let w = self.inner.as_mut().ok_or_else(|| err("layer writer is closed"))?;
        let mut b = BTreeMap::new();
        if let Some(blobs) = blobs {
            for (k, v) in blobs.iter() {
                b.insert(k.extract::<String>()?, v.extract::<Vec<u8>>()?);
            }
        }
        let mut m = BTreeMap::new();
        if let Some(meta) = meta {
            for (k, v) in meta.iter() {
                m.insert(k.extract::<String>()?, py_to_json(&v)?);
            }
        }
        w.add(sample_id, &b, &m).map_err(core_err)
    }

    /// Finalize: write the layer index and register the layer in the manifest.
    fn close(&mut self) -> PyResult<()> {
        let w = self.inner.take().ok_or_else(|| err("layer writer already closed"))?;
        w.close().map_err(core_err)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FerroWriter>()?;
    m.add_class::<FerroDataset>()?;
    m.add_class::<PyLayerWriter>()?;
    m.add_class::<Sampler>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
