//! Parquet-backed [`IndexBackend`] (feature `parquet`).
//!
//! Flattens the dynamic index into a columnar schema for true projection /
//! predicate pushdown (and direct querying by DuckDB / Arrow / DataFusion):
//!   - `sample_id` (u64), `shard_id` (u32), `basename` (utf8), `shard` (utf8, null)
//!   - per tensor modality: `<m>_off`, `<m>_len` (i64, **null when absent**)
//!   - per meta key: a typed column (i64/f64/bool/utf8), null when absent —
//!     so captions, exif scalars, width/height, labels, etc. are real,
//!     queryable columns (nested values are JSON-encoded into a utf8 column).
//!
//! Sparse modalities are null offsets — exactly the "skip + mask, zero I/O"
//! contract. Compression is left at UNCOMPRESSED to avoid native codec deps.

use crate::error::{Error, Result};
use crate::index::{IndexBackend, IndexRow};
use crate::subset::{ColStat, Predicate, RowGroupStats};
use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Float64Array, Float64Builder, Int64Array,
    Int64Builder, StringArray, StringBuilder, UInt32Array, UInt32Builder, UInt64Array,
    UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::basic::{Compression, ZstdLevel};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::file::metadata::RowGroupMetaData;
use parquet::file::properties::WriterProperties;
use parquet::file::statistics::Statistics;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Rows per Parquet row group. Smaller groups give finer row-group pruning
/// granularity (min/max stats per group) at a little metadata cost.
const ROW_GROUP_SIZE: usize = 8192;

#[derive(Debug, Default, Clone, Copy)]
pub struct ParquetIndex;

#[derive(Clone, Copy, PartialEq)]
enum MetaKind {
    I64,
    F64,
    Bool,
    Str,
    Json,
}

fn infer_kind(rows: &[IndexRow], key: &str) -> MetaKind {
    let (mut all_bool, mut all_int, mut all_num, mut all_str, mut any) = (true, true, true, true, false);
    for r in rows {
        if let Some(v) = r.meta.get(key) {
            any = true;
            match v {
                Value::Bool(_) => {
                    all_int = false;
                    all_num = false;
                    all_str = false;
                }
                Value::Number(n) => {
                    all_bool = false;
                    all_str = false;
                    if n.as_i64().is_none() {
                        all_int = false;
                    }
                }
                Value::String(_) => {
                    all_bool = false;
                    all_int = false;
                    all_num = false;
                }
                _ => return MetaKind::Json,
            }
        }
    }
    if !any {
        return MetaKind::Json;
    }
    if all_bool {
        MetaKind::Bool
    } else if all_int {
        MetaKind::I64
    } else if all_num {
        MetaKind::F64
    } else if all_str {
        MetaKind::Str
    } else {
        MetaKind::Json
    }
}

impl IndexBackend for ParquetIndex {
    fn write(&self, path: &Path, rows: &[IndexRow]) -> Result<()> {
        // discover modality + meta columns
        let mut modalities: BTreeSet<String> = BTreeSet::new();
        let mut meta_keys: BTreeSet<String> = BTreeSet::new();
        for r in rows {
            modalities.extend(r.offsets.keys().cloned());
            meta_keys.extend(r.meta.keys().cloned());
        }
        let meta_kinds: BTreeMap<String, MetaKind> =
            meta_keys.iter().map(|k| (k.clone(), infer_kind(rows, k))).collect();

        // schema
        let mut fields = vec![
            Field::new("sample_id", DataType::UInt64, false),
            Field::new("shard_id", DataType::UInt32, false),
            Field::new("basename", DataType::Utf8, false),
            Field::new("shard", DataType::Utf8, true),
        ];
        for m in &modalities {
            fields.push(Field::new(format!("{m}_off"), DataType::Int64, true));
            fields.push(Field::new(format!("{m}_len"), DataType::Int64, true));
        }
        for k in &meta_keys {
            let dt = match meta_kinds[k] {
                MetaKind::I64 => DataType::Int64,
                MetaKind::F64 => DataType::Float64,
                MetaKind::Bool => DataType::Boolean,
                MetaKind::Str | MetaKind::Json => DataType::Utf8,
            };
            fields.push(Field::new(k.clone(), dt, true));
        }
        let schema = Arc::new(Schema::new(fields));

        // columns
        let mut arrays: Vec<ArrayRef> = Vec::new();
        {
            let mut b = UInt64Builder::new();
            for r in rows {
                b.append_value(r.sample_id);
            }
            arrays.push(Arc::new(b.finish()) as ArrayRef);
        }
        {
            let mut b = UInt32Builder::new();
            for r in rows {
                b.append_value(r.shard_id);
            }
            arrays.push(Arc::new(b.finish()) as ArrayRef);
        }
        {
            let mut b = StringBuilder::new();
            for r in rows {
                b.append_value(&r.basename);
            }
            arrays.push(Arc::new(b.finish()) as ArrayRef);
        }
        {
            let mut b = StringBuilder::new();
            for r in rows {
                match &r.shard {
                    Some(s) => b.append_value(s),
                    None => b.append_null(),
                }
            }
            arrays.push(Arc::new(b.finish()) as ArrayRef);
        }
        for m in &modalities {
            let mut off = Int64Builder::new();
            let mut len = Int64Builder::new();
            for r in rows {
                match r.offsets.get(m) {
                    Some([o, l]) => {
                        off.append_value(*o as i64);
                        len.append_value(*l as i64);
                    }
                    None => {
                        off.append_null();
                        len.append_null();
                    }
                }
            }
            arrays.push(Arc::new(off.finish()) as ArrayRef);
            arrays.push(Arc::new(len.finish()) as ArrayRef);
        }
        for k in &meta_keys {
            let kind = meta_kinds[k];
            let arr: ArrayRef = match kind {
                MetaKind::I64 => {
                    let mut b = Int64Builder::new();
                    for r in rows {
                        match r.meta.get(k).and_then(|v| v.as_i64()) {
                            Some(x) => b.append_value(x),
                            None => b.append_null(),
                        }
                    }
                    Arc::new(b.finish())
                }
                MetaKind::F64 => {
                    let mut b = Float64Builder::new();
                    for r in rows {
                        match r.meta.get(k).and_then(|v| v.as_f64()) {
                            Some(x) => b.append_value(x),
                            None => b.append_null(),
                        }
                    }
                    Arc::new(b.finish())
                }
                MetaKind::Bool => {
                    let mut b = BooleanBuilder::new();
                    for r in rows {
                        match r.meta.get(k).and_then(|v| v.as_bool()) {
                            Some(x) => b.append_value(x),
                            None => b.append_null(),
                        }
                    }
                    Arc::new(b.finish())
                }
                MetaKind::Str | MetaKind::Json => {
                    let mut b = StringBuilder::new();
                    for r in rows {
                        match r.meta.get(k) {
                            Some(Value::String(s)) => b.append_value(s),
                            Some(other) => b.append_value(other.to_string()), // json-encode
                            None => b.append_null(),
                        }
                    }
                    Arc::new(b.finish())
                }
            };
            arrays.push(arr);
        }

        let batch = RecordBatch::try_new(schema.clone(), arrays)
            .map_err(|e| Error::Format(format!("arrow batch: {e}")))?;
        let file = File::create(path)?;
        // Bounded row groups + per-column statistics (on by default) enable
        // row-group pruning at read time; zstd shrinks the index (esp. the
        // text-heavy caption/exif columns) for cheaper storage + network.
        let props = WriterProperties::builder()
            .set_max_row_group_size(ROW_GROUP_SIZE)
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .build();
        let mut w = ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|e| Error::Format(format!("parquet writer: {e}")))?;
        w.write(&batch).map_err(|e| Error::Format(format!("parquet write: {e}")))?;
        w.close().map_err(|e| Error::Format(format!("parquet close: {e}")))?;
        Ok(())
    }

    fn read(&self, path: &Path) -> Result<Vec<IndexRow>> {
        Self::read_chunks(File::open(path)?)
    }
}

impl ParquetIndex {
    /// Parse an index shard from an in-memory Parquet buffer (the remote loader
    /// has already fetched the bytes through the cache).
    pub fn read_bytes(bytes: impl Into<bytes::Bytes>) -> Result<Vec<IndexRow>> {
        Self::read_chunks(bytes.into())
    }

    fn read_chunks<R: parquet::file::reader::ChunkReader + 'static>(
        src: R,
    ) -> Result<Vec<IndexRow>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(src)
            .map_err(|e| Error::Format(format!("parquet open: {e}")))?;
        let reader = builder.build().map_err(|e| Error::Format(format!("parquet build: {e}")))?;
        Self::rows_from_reader(reader)
    }

    /// Subset a single shard with **column projection** (read only the predicate's
    /// columns plus the structural ids) and **row-group pruning** (skip groups whose
    /// min/max statistics can't match). Returns matching `sample_id`s; the caller
    /// sorts across shards.
    pub fn subset_shard(bytes: impl Into<bytes::Bytes>, pred: &Predicate) -> Result<Vec<u64>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.into())
            .map_err(|e| Error::Format(format!("parquet open: {e}")))?;

        let want = projection_want(pred);
        let pq = builder.parquet_schema();
        let mask = ProjectionMask::leaves(pq, leaf_indices(pq, &want));

        // row-group pruning from per-group min/max statistics
        let meta = builder.metadata().clone();
        let keep: Vec<usize> = (0..meta.num_row_groups())
            .filter(|&i| pred.might_match(&RgStats { rg: meta.row_group(i) }))
            .collect();

        let reader = builder
            .with_projection(mask)
            .with_row_groups(keep)
            .build()
            .map_err(|e| Error::Format(format!("parquet build: {e}")))?;
        let rows = Self::rows_from_reader(reader)?;
        Ok(rows.iter().filter(|r| pred.matches(r)).map(|r| r.sample_id).collect())
    }

    /// Subset a shard that lives on a remote object store, fetching only the
    /// footer + projected column chunks of matching row groups via **ranged GETs
    /// through the on-disk cache** — so a selective query pulls a few KB instead of
    /// the whole shard. Mirrors [`subset_shard`] but over the async Parquet reader.
    #[cfg(feature = "remote")]
    pub fn subset_shard_remote(
        store: &ferroload_io::CachedStorage,
        key: &str,
        pred: &Predicate,
    ) -> Result<Vec<u64>> {
        use futures::StreamExt;
        use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;

        let fmt = |e: parquet::errors::ParquetError| Error::Format(format!("remote parquet {key}: {e}"));
        ferroload_io::runtime().block_on(async move {
            let size = store
                .size(key)
                .await
                .map_err(|e| Error::Format(format!("head {key}: {e}")))?;
            let reader = CachedAsyncReader { store: store.clone(), key: key.to_string(), size };
            let builder = ParquetRecordBatchStreamBuilder::new(reader).await.map_err(&fmt)?;

            let want = projection_want(pred);
            let pq = builder.parquet_schema();
            let mask = ProjectionMask::leaves(pq, leaf_indices(pq, &want));

            let meta = builder.metadata().clone();
            let keep: Vec<usize> = (0..meta.num_row_groups())
                .filter(|&i| pred.might_match(&RgStats { rg: meta.row_group(i) }))
                .collect();

            let mut stream = builder
                .with_projection(mask)
                .with_row_groups(keep)
                .build()
                .map_err(&fmt)?;

            let mut rows: Vec<IndexRow> = Vec::new();
            while let Some(batch) = stream.next().await {
                rows_from_batch(&batch.map_err(&fmt)?, &mut rows);
            }
            Ok(rows.iter().filter(|r| pred.matches(r)).map(|r| r.sample_id).collect())
        })
    }

    fn rows_from_reader(reader: ParquetRecordBatchReader) -> Result<Vec<IndexRow>> {
        let mut rows: Vec<IndexRow> = Vec::new();
        for batch in reader {
            let batch = batch.map_err(|e| Error::Format(format!("parquet read: {e}")))?;
            rows_from_batch(&batch, &mut rows);
        }
        rows.sort_by_key(|r| r.sample_id);
        Ok(rows)
    }
}

/// Reconstruct [`IndexRow`]s from one (possibly projected) record batch. Requires
/// the `sample_id`/`shard_id`/`basename` columns; everything else is optional, so
/// a projected batch yields partial rows carrying only the read columns.
fn rows_from_batch(batch: &RecordBatch, rows: &mut Vec<IndexRow>) {
    let schema = batch.schema();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

    // identify modality pairs (both <m>_off and <m>_len present)
    let mut modalities: Vec<String> = Vec::new();
    for n in &names {
        if let Some(m) = n.strip_suffix("_off") {
            if names.iter().any(|x| x == &format!("{m}_len")) {
                modalities.push(m.to_string());
            }
        }
    }
    let fixed: BTreeSet<String> = ["sample_id", "shard_id", "basename", "shard"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mod_cols: BTreeSet<String> = modalities
        .iter()
        .flat_map(|m| [format!("{m}_off"), format!("{m}_len")])
        .collect();

    let nrows = batch.num_rows();
    let col = |name: &str| batch.column(schema.index_of(name).unwrap());

    let sid = col("sample_id").as_any().downcast_ref::<UInt64Array>().unwrap();
    let shid = col("shard_id").as_any().downcast_ref::<UInt32Array>().unwrap();
    let base = col("basename").as_any().downcast_ref::<StringArray>().unwrap();
    let shard_col = names
        .iter()
        .any(|x| x == "shard")
        .then(|| col("shard").as_any().downcast_ref::<StringArray>().unwrap());

    for i in 0..nrows {
        let mut offsets = BTreeMap::new();
        for m in &modalities {
            let o = col(&format!("{m}_off")).as_any().downcast_ref::<Int64Array>().unwrap();
            let l = col(&format!("{m}_len")).as_any().downcast_ref::<Int64Array>().unwrap();
            if o.is_valid(i) && l.is_valid(i) {
                offsets.insert(m.clone(), [o.value(i) as u64, l.value(i) as u64]);
            }
        }
        let mut meta = BTreeMap::new();
        for n in &names {
            if fixed.contains(n) || mod_cols.contains(n) {
                continue;
            }
            let c = col(n);
            if !c.is_valid(i) {
                continue;
            }
            meta.insert(n.clone(), arrow_value_to_json(c, i));
        }
        let shard = shard_col.and_then(|c| c.is_valid(i).then(|| c.value(i).to_string()));
        rows.push(IndexRow {
            sample_id: sid.value(i),
            shard_id: shid.value(i),
            basename: base.value(i).to_string(),
            offsets,
            meta,
            shard,
        });
    }
}

/// Columns to project for `pred`: structural ids + referenced columns (with
/// `<m>_off`/`<m>_len` pairs for any `<m>_present` flag). Excludes the fat,
/// unreferenced columns (captions, exif) so they're never decoded.
fn projection_want(pred: &Predicate) -> BTreeSet<String> {
    let mut want: BTreeSet<String> =
        ["sample_id", "shard_id", "basename"].iter().map(|s| s.to_string()).collect();
    for c in pred.referenced_columns() {
        match c.strip_suffix("_present") {
            Some(base) => {
                want.insert(format!("{base}_off"));
                want.insert(format!("{base}_len"));
            }
            None => {
                want.insert(c);
            }
        }
    }
    want
}

/// Leaf column indices in `pq` whose names are in `want` (a flat schema, so leaf
/// == column). Names not present in this shard are simply skipped.
fn leaf_indices(pq: &parquet::schema::types::SchemaDescriptor, want: &BTreeSet<String>) -> Vec<usize> {
    pq.columns()
        .iter()
        .enumerate()
        .filter(|(_, c)| want.contains(c.name()))
        .map(|(i, _)| i)
        .collect()
}

/// An [`AsyncFileReader`](parquet::arrow::async_reader::AsyncFileReader) that
/// serves ranged reads through the content-addressed [`CachedStorage`], so the
/// Parquet footer + projected column chunks are fetched (and cached) by byte
/// range rather than downloading the whole shard.
#[cfg(feature = "remote")]
struct CachedAsyncReader {
    store: ferroload_io::CachedStorage,
    key: String,
    size: u64,
}

#[cfg(feature = "remote")]
impl parquet::arrow::async_reader::AsyncFileReader for CachedAsyncReader {
    fn get_bytes(
        &mut self,
        range: std::ops::Range<usize>,
    ) -> futures::future::BoxFuture<'_, parquet::errors::Result<bytes::Bytes>> {
        use futures::FutureExt;
        let store = self.store.clone();
        let key = self.key.clone();
        async move {
            store
                .get_range(&key, range.start as u64..range.end as u64)
                .await
                .map_err(|e| parquet::errors::ParquetError::External(Box::new(e)))
        }
        .boxed()
    }

    fn get_metadata(
        &mut self,
    ) -> futures::future::BoxFuture<'_, parquet::errors::Result<Arc<parquet::file::metadata::ParquetMetaData>>>
    {
        use futures::FutureExt;
        use parquet::errors::ParquetError;
        use parquet::file::metadata::ParquetMetaDataReader;
        let store = self.store.clone();
        let key = self.key.clone();
        let size = self.size as usize;
        async move {
            let ext = |e| ParquetError::External(Box::new(e));
            let footer = store.get_range(&key, (size - 8) as u64..size as u64).await.map_err(ext)?;
            let footer8: [u8; 8] = footer
                .as_ref()
                .try_into()
                .map_err(|_| ParquetError::General("short parquet footer".into()))?;
            let metadata_len = ParquetMetaDataReader::decode_footer(&footer8)?;
            let start = (size - 8 - metadata_len) as u64;
            let meta_bytes = store.get_range(&key, start..(size - 8) as u64).await.map_err(ext)?;
            Ok(Arc::new(ParquetMetaDataReader::decode_metadata(&meta_bytes)?))
        }
        .boxed()
    }
}

/// Adapts a Parquet row group's column statistics to the predicate pruner.
struct RgStats<'a> {
    rg: &'a RowGroupMetaData,
}

impl RowGroupStats for RgStats<'_> {
    fn col(&self, name: &str) -> ColStat {
        for c in self.rg.columns() {
            if c.column_descr().name() == name {
                return c.statistics().map(stat_to_colstat).unwrap_or(ColStat::Unknown);
            }
        }
        ColStat::Unknown
    }
}

/// Convert a Parquet column-chunk `Statistics` into a min/max [`ColStat`].
fn stat_to_colstat(s: &Statistics) -> ColStat {
    fn num<T: Copy>(mn: Option<&T>, mx: Option<&T>, f: impl Fn(T) -> f64) -> ColStat {
        match (mn, mx) {
            (Some(a), Some(b)) => ColStat::Num { min: f(*a), max: f(*b) },
            _ => ColStat::Unknown,
        }
    }
    match s {
        Statistics::Int32(v) => num(v.min_opt(), v.max_opt(), |x| x as f64),
        Statistics::Int64(v) => num(v.min_opt(), v.max_opt(), |x| x as f64),
        Statistics::Float(v) => num(v.min_opt(), v.max_opt(), |x| x as f64),
        Statistics::Double(v) => num(v.min_opt(), v.max_opt(), |x| x),
        Statistics::Boolean(v) => match (v.min_opt(), v.max_opt()) {
            (Some(a), Some(b)) => ColStat::Bool { min: *a, max: *b },
            _ => ColStat::Unknown,
        },
        Statistics::ByteArray(v) => match (v.min_opt(), v.max_opt()) {
            (Some(a), Some(b)) => match (a.as_utf8(), b.as_utf8()) {
                (Ok(a), Ok(b)) => ColStat::Str { min: a.to_string(), max: b.to_string() },
                _ => ColStat::Unknown,
            },
            _ => ColStat::Unknown,
        },
        _ => ColStat::Unknown,
    }
}

fn arrow_value_to_json(c: &ArrayRef, i: usize) -> Value {
    match c.data_type() {
        DataType::Int64 => Value::from(c.as_any().downcast_ref::<Int64Array>().unwrap().value(i)),
        DataType::Float64 => Value::from(c.as_any().downcast_ref::<Float64Array>().unwrap().value(i)),
        DataType::Boolean => Value::from(c.as_any().downcast_ref::<BooleanArray>().unwrap().value(i)),
        DataType::Utf8 => {
            let s = c.as_any().downcast_ref::<StringArray>().unwrap().value(i);
            // decode json only for clearly-encoded structured values
            if s.starts_with('[') || s.starts_with('{') {
                serde_json::from_str(s).unwrap_or_else(|_| Value::from(s))
            } else {
                Value::from(s)
            }
        }
        _ => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ferroload_pq_{}_{}.parquet", std::process::id(), tag));
        p
    }

    fn rows() -> Vec<IndexRow> {
        let mk = |id: u64, depth: bool| {
            let mut offsets = BTreeMap::new();
            offsets.insert("image".to_string(), [512u64, 100u64]);
            if depth {
                offsets.insert("depth".to_string(), [2048, 200]);
            }
            let mut meta = BTreeMap::new();
            meta.insert("duration_s".to_string(), serde_json::json!(2 + id as i64));
            meta.insert("lang".to_string(), serde_json::json!("en"));
            meta.insert("has_audio".to_string(), serde_json::json!(id % 2 == 0));
            meta.insert("boxes".to_string(), serde_json::json!([[1, 2, 3, 4]]));
            IndexRow { sample_id: id, shard_id: 0, basename: format!("s{id}"), offsets, meta, shard: None }
        };
        vec![mk(0, true), mk(1, false), mk(2, true)]
    }

    #[test]
    fn parquet_roundtrip_with_sparse_and_types() {
        let p = tmp("rt");
        let rs = rows();
        ParquetIndex.write(&p, &rs).unwrap();
        let back = ParquetIndex.read(&p).unwrap();
        assert_eq!(back.len(), 3);

        // sparse depth: present on 0 and 2, null on 1
        assert!(back[0].offsets.contains_key("depth"));
        assert!(!back[1].offsets.contains_key("depth"));
        assert!(back[1].offsets.contains_key("image"));

        // typed meta round-trips
        assert_eq!(back[2].meta["duration_s"], serde_json::json!(4));
        assert_eq!(back[0].meta["lang"], serde_json::json!("en"));
        assert_eq!(back[0].meta["has_audio"], serde_json::json!(true));
        // nested annotation json-encoded + decoded back
        assert_eq!(back[0].meta["boxes"], serde_json::json!([[1, 2, 3, 4]]));

        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn subset_shard_projects_and_prunes_row_groups() {
        // > ROW_GROUP_SIZE rows, with width == sample_id, so each row group covers
        // a contiguous, non-overlapping width range — ideal for pruning.
        let p = tmp("prune");
        let n = 20_000u64;
        let rows: Vec<IndexRow> = (0..n)
            .map(|id| {
                let mut offsets = BTreeMap::new();
                offsets.insert("image".to_string(), [id, 10]);
                let mut meta = BTreeMap::new();
                meta.insert("width".to_string(), serde_json::json!(id as i64));
                meta.insert("caption".to_string(), serde_json::json!(format!("a caption {id}")));
                IndexRow { sample_id: id, shard_id: 0, basename: format!("s{id}"), offsets, meta, shard: None }
            })
            .collect();
        ParquetIndex.write(&p, &rows).unwrap();

        // the writer produced multiple row groups
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(&p).unwrap()).unwrap();
        let meta = builder.metadata().clone();
        let n_groups = meta.num_row_groups();
        assert!(n_groups >= 2, "expected >1 row group, got {n_groups}");

        // pruning: width = 100 lives only in the first row group
        let pred = Predicate::parse("width = 100").unwrap();
        let keep: Vec<usize> = (0..n_groups)
            .filter(|&i| pred.might_match(&RgStats { rg: meta.row_group(i) }))
            .collect();
        assert_eq!(keep, vec![0], "only the first row group can match");

        // end-to-end subset (projection + pruning) is correct
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(ParquetIndex::subset_shard(bytes.clone(), &pred).unwrap(), vec![100]);

        // a high range only touches the last group
        let pred2 = Predicate::parse("width >= 19990").unwrap();
        let keep2: Vec<usize> = (0..n_groups)
            .filter(|&i| pred2.might_match(&RgStats { rg: meta.row_group(i) }))
            .collect();
        assert_eq!(keep2, vec![n_groups - 1]);
        assert_eq!(
            ParquetIndex::subset_shard(bytes, &pred2).unwrap(),
            (19990..n).collect::<Vec<u64>>()
        );

        std::fs::remove_file(&p).ok();
    }
}
