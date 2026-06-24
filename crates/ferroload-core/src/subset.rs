//! Lightweight `WHERE`-clause subsetting over the index's inline metadata.
//!
//! Supports comparisons (`= == != <> < <= > >=`), boolean columns, `AND`/`OR`/
//! `NOT`, parentheses, and numeric/string/bool literals — enough to express the
//! common subset predicates. In production this is replaced by DataFusion SQL
//! over the Parquet index (DESIGN §6); the surface (`subset_ids`) is the same:
//! a predicate in -> an ordered list of `sample_id`s out.

use crate::error::{Error, Result};
use crate::index::IndexRow;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64),
    Str(String),
    Bool(bool),
    Op(String),
    And,
    Or,
    Not,
    LParen,
    RParen,
}

fn lex(s: &str) -> Result<Vec<Tok>> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c == '(' {
            out.push(Tok::LParen);
            i += 1;
        } else if c == ')' {
            out.push(Tok::RParen);
            i += 1;
        } else if c == '\'' || c == '"' {
            let q = c;
            i += 1;
            let start = i;
            while i < b.len() && b[i] as char != q {
                i += 1;
            }
            if i >= b.len() {
                return Err(Error::Format("unterminated string".into()));
            }
            out.push(Tok::Str(s[start..i].to_string()));
            i += 1;
        } else if "=<>!".contains(c) {
            let start = i;
            i += 1;
            if i < b.len() && (b[i] as char == '=' || b[i] as char == '>') {
                i += 1;
            }
            out.push(Tok::Op(s[start..i].to_string()));
        } else if c.is_ascii_digit() || (c == '-' && i + 1 < b.len() && (b[i + 1] as char).is_ascii_digit()) {
            let start = i;
            i += 1;
            while i < b.len() && {
                let ch = b[i] as char;
                ch.is_ascii_digit() || ch == '.'
            } {
                i += 1;
            }
            let n: f64 = s[start..i]
                .parse()
                .map_err(|_| Error::Format(format!("bad number: {}", &s[start..i])))?;
            out.push(Tok::Num(n));
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && {
                let ch = b[i] as char;
                ch.is_alphanumeric() || ch == '_' || ch == '.'
            } {
                i += 1;
            }
            let w = &s[start..i];
            match w.to_ascii_uppercase().as_str() {
                "AND" => out.push(Tok::And),
                "OR" => out.push(Tok::Or),
                "NOT" => out.push(Tok::Not),
                "TRUE" => out.push(Tok::Bool(true)),
                "FALSE" => out.push(Tok::Bool(false)),
                _ => out.push(Tok::Ident(w.to_string())),
            }
        } else {
            return Err(Error::Format(format!("unexpected char '{c}'")));
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
enum Expr {
    Cmp { col: String, op: String, val: Value },
    Truthy(String),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn parse(&mut self) -> Result<Expr> {
        let e = self.or_expr()?;
        if self.pos != self.toks.len() {
            return Err(Error::Format("trailing tokens in predicate".into()));
        }
        Ok(e)
    }

    fn or_expr(&mut self) -> Result<Expr> {
        let mut left = self.and_expr()?;
        while matches!(self.peek(), Some(Tok::Or)) {
            self.next();
            let right = self.and_expr()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> Result<Expr> {
        let mut left = self.not_expr()?;
        while matches!(self.peek(), Some(Tok::And)) {
            self.next();
            let right = self.not_expr()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn not_expr(&mut self) -> Result<Expr> {
        if matches!(self.peek(), Some(Tok::Not)) {
            self.next();
            return Ok(Expr::Not(Box::new(self.not_expr()?)));
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<Expr> {
        match self.next() {
            Some(Tok::LParen) => {
                let e = self.or_expr()?;
                match self.next() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err(Error::Format("expected ')'".into())),
                }
            }
            Some(Tok::Ident(col)) => {
                if let Some(Tok::Op(op)) = self.peek().cloned() {
                    self.next();
                    let val = match self.next() {
                        Some(Tok::Num(n)) => Value::from(n),
                        Some(Tok::Str(s)) => Value::from(s),
                        Some(Tok::Bool(b)) => Value::from(b),
                        _ => return Err(Error::Format("expected literal after operator".into())),
                    };
                    Ok(Expr::Cmp { col, op, val })
                } else {
                    Ok(Expr::Truthy(col)) // bare boolean column
                }
            }
            Some(Tok::Bool(b)) => Ok(Expr::Cmp {
                col: String::new(),
                op: "const".into(),
                val: Value::from(b),
            }),
            other => Err(Error::Format(format!("unexpected token {other:?}"))),
        }
    }
}

fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
}

fn cmp_num(a: f64, op: &str, b: f64) -> bool {
    match op {
        "=" | "==" => a == b,
        "!=" | "<>" => a != b,
        "<" => a < b,
        "<=" => a <= b,
        ">" => a > b,
        ">=" => a >= b,
        _ => false,
    }
}

fn eval(e: &Expr, meta: &serde_json::Map<String, Value>) -> bool {
    match e {
        Expr::And(a, b) => eval(a, meta) && eval(b, meta),
        Expr::Or(a, b) => eval(a, meta) || eval(b, meta),
        Expr::Not(a) => !eval(a, meta),
        Expr::Truthy(col) => meta.get(col).and_then(|v| v.as_bool()).unwrap_or(false),
        Expr::Cmp { col, op, val } => {
            if op == "const" {
                return val.as_bool().unwrap_or(false);
            }
            let lhs = match meta.get(col) {
                Some(v) => v,
                None => return false, // null column never matches
            };
            match (as_f64(lhs), as_f64(val)) {
                (Some(x), Some(y)) => cmp_num(x, op, y),
                _ => {
                    // string / bool equality
                    match op.as_str() {
                        "=" | "==" => lhs == val,
                        "!=" | "<>" => lhs != val,
                        _ => false,
                    }
                }
            }
        }
    }
}

/// A compiled predicate that can be evaluated row-by-row.
pub struct Predicate {
    expr: Expr,
}

impl Predicate {
    pub fn parse(where_sql: &str) -> Result<Self> {
        let toks = lex(where_sql)?;
        if toks.is_empty() {
            return Err(Error::Format("empty predicate".into()));
        }
        let mut p = Parser { toks, pos: 0 };
        Ok(Predicate { expr: p.parse()? })
    }

    pub fn matches(&self, row: &IndexRow) -> bool {
        // build a meta map including derived presence flags `<modality>_present`
        let mut meta = serde_json::Map::new();
        for (k, v) in &row.meta {
            meta.insert(k.clone(), v.clone());
        }
        for m in row.offsets.keys() {
            meta.insert(format!("{m}_present"), Value::from(true));
        }
        eval(&self.expr, &meta)
    }
}

/// Filter rows by a `WHERE` predicate, returning `sample_id`s in ascending order
/// (deterministic ordering before sharding — see DESIGN §6).
pub fn subset_ids(rows: &[IndexRow], where_sql: &str) -> Result<Vec<u64>> {
    let pred = Predicate::parse(where_sql)?;
    let mut ids: Vec<u64> = rows
        .iter()
        .filter(|r| pred.matches(r))
        .map(|r| r.sample_id)
        .collect();
    ids.sort_unstable();
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn row(id: u64, dur: i64, lang: &str, has_audio: bool, depth: bool) -> IndexRow {
        let mut meta = BTreeMap::new();
        meta.insert("duration_s".into(), serde_json::json!(dur));
        meta.insert("lang".into(), serde_json::json!(lang));
        meta.insert("has_audio".into(), serde_json::json!(has_audio));
        let mut offsets = BTreeMap::new();
        offsets.insert("image".into(), [512u64, 10u64]);
        if depth {
            offsets.insert("depth".into(), [1024, 10]);
        }
        IndexRow { sample_id: id, shard_id: 0, basename: format!("s{id}"), offsets, meta, shard: None }
    }

    fn data() -> Vec<IndexRow> {
        vec![
            row(0, 3, "en", true, true),
            row(1, 20, "en", false, false),
            row(2, 8, "fr", true, false),
            row(3, 12, "en", true, true),
        ]
    }

    #[test]
    fn numeric_and_string() {
        let ids = subset_ids(&data(), "duration_s < 15 AND lang = 'en'").unwrap();
        assert_eq!(ids, vec![0, 3]);
    }

    #[test]
    fn boolean_column_and_or() {
        let ids = subset_ids(&data(), "has_audio AND (lang='fr' OR duration_s <= 3)").unwrap();
        assert_eq!(ids, vec![0, 2]);
    }

    #[test]
    fn not_and_neq() {
        let ids = subset_ids(&data(), "NOT lang = 'en'").unwrap();
        assert_eq!(ids, vec![2]);
        let ids = subset_ids(&data(), "lang <> 'en'").unwrap();
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn presence_flag_from_offsets() {
        // depth_present is derived from the offsets map, not stored in meta
        let ids = subset_ids(&data(), "depth_present").unwrap();
        assert_eq!(ids, vec![0, 3]);
    }

    #[test]
    fn parse_errors() {
        assert!(subset_ids(&data(), "duration_s <").is_err());
        assert!(subset_ids(&data(), "(lang='en'").is_err());
        assert!(subset_ids(&data(), "").is_err());
    }
}
