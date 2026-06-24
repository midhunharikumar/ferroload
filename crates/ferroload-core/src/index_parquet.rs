//! Parquet-backed [`IndexBackend`] (feature `parquet`).
//!
//! Flattens the dynamic index into a columnar schema for true projection /
//! predicate pushdown:
//!   - `sample_id` (u64), `shard_id` (u32), `basename` (utf8)
//!   - per tensor modality: `<m>_off`, `<m>_len` (i64, **null when absent**)
//!   - per meta key: a typed column (i64/f64/bool/utf8), null when absent
//!
//! Sparse modalities are null offsets — exactly the "skip + mask, zero I/O"
//! contract. Compression is left at UNCOMPRESSED to avoid native codec deps.

use crate::error::{Error, Result};
use crate::index::{IndexBackend, IndexRow};
use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Float64Array, Float64Builder, Int64Array,
    Int64Builder, StringArray, StringBuilder, UInt32Array, UInt32Builder, UInt64Array,
    UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

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
        let mut w = ArrowWriter::try_new(file, schema, None)
            .map_err(|e| Error::Format(format!("parquet writer: {e}")))?;
        w.write(&batch).map_err(|e| Error::Format(format!("parquet write: {e}")))?;
        w.close().map_err(|e| Error::Format(format!("parquet close: {e}")))?;
        Ok(())
    }

    fn read(&self, path: &Path) -> Result<Vec<IndexRow>> {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Error::Format(format!("parquet open: {e}")))?;
        let reader = builder.build().map_err(|e| Error::Format(format!("parquet build: {e}")))?;

        let mut rows: Vec<IndexRow> = Vec::new();
        for batch in reader {
            let batch = batch.map_err(|e| Error::Format(format!("parquet read: {e}")))?;
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
            let fixed: BTreeSet<String> = ["sample_id", "shard_id", "basename"]
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
                    let v = arrow_value_to_json(c, i);
                    meta.insert(n.clone(), v);
                }
                rows.push(IndexRow {
                    sample_id: sid.value(i),
                    shard_id: shid.value(i),
                    basename: base.value(i).to_string(),
                    offsets,
                    meta,
                    shard: None,
                });
            }
        }
        rows.sort_by_key(|r| r.sample_id);
        Ok(rows)
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
}
