//! Fast OracleGeneral trace parser for FOO-JAX.
//!
//! Binary format (24 bytes per record, little-endian):
//!     - uint32 timestamp        (4 bytes)
//!     - uint64 obj_id           (8 bytes)
//!     - uint32 obj_size         (4 bytes)
//!     - int64  next_access_vtime (8 bytes, -1 if no next access)

use ahash::AHashMap;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::fs::File;
use std::io::{BufReader, Read};

const RECORD_SIZE: usize = 24;

/// Result of parsing a trace file
#[pyclass]
pub struct ParseResult {
    #[pyo3(get)]
    n_requests: usize,
    #[pyo3(get)]
    n_unique_objects: usize,
    timestamps: Vec<u32>,
    obj_ids: Vec<u64>,
    obj_sizes: Vec<u32>,
    next_access_idx: Vec<i32>,
    prev_access_idx: Vec<i32>,
}

#[pymethods]
impl ParseResult {
    /// Get timestamps as numpy array
    fn get_timestamps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        Ok(PyArray1::from_slice_bound(py, &self.timestamps))
    }

    /// Get object IDs as numpy array
    fn get_obj_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u64>>> {
        Ok(PyArray1::from_slice_bound(py, &self.obj_ids))
    }

    /// Get object sizes as numpy array
    fn get_obj_sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        Ok(PyArray1::from_slice_bound(py, &self.obj_sizes))
    }

    /// Get next_access_idx as numpy array
    fn get_next_access_idx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        Ok(PyArray1::from_slice_bound(py, &self.next_access_idx))
    }

    /// Get prev_access_idx as numpy array
    fn get_prev_access_idx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        Ok(PyArray1::from_slice_bound(py, &self.prev_access_idx))
    }
}

/// Parse a record from raw bytes
#[inline(always)]
fn parse_record(data: &[u8]) -> (u32, u64, u32, i64) {
    let timestamp = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let obj_id = u64::from_le_bytes([
        data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11],
    ]);
    let obj_size = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    let next_vtime = i64::from_le_bytes([
        data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
    ]);
    (timestamp, obj_id, obj_size, next_vtime)
}

/// Build access topology arrays
fn build_access_topology(
    obj_ids: &[u64],
    obj_sizes: &[u32],
) -> (Vec<i32>, Vec<i32>, usize) {
    let n = obj_ids.len();
    let mut next_access_idx = vec![-1i32; n];
    let mut prev_access_idx = vec![-1i32; n];

    // Map (obj_id, obj_size) -> last seen index
    // Pre-allocate with expected capacity
    let mut last_seen: AHashMap<(u64, u32), i32> = AHashMap::with_capacity(n / 4);

    for i in 0..n {
        let key = (obj_ids[i], obj_sizes[i]);

        if let Some(&prev_idx) = last_seen.get(&key) {
            prev_access_idx[i] = prev_idx;
            next_access_idx[prev_idx as usize] = i as i32;
        }

        last_seen.insert(key, i as i32);
    }

    let n_unique = last_seen.len();
    (next_access_idx, prev_access_idx, n_unique)
}

/// Parse trace file (supports .zst compression)
#[pyfunction]
#[pyo3(signature = (path, max_requests=None))]
fn parse_trace_fast(path: &str, max_requests: Option<usize>) -> PyResult<ParseResult> {
    let is_zst = path.ends_with(".zst");

    // Read and decompress file
    let data: Vec<u8> = if is_zst {
        let file = File::open(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to open file: {}", e))
        })?;
        let reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        let mut decoder = zstd::Decoder::new(reader).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to create zstd decoder: {}", e))
        })?;
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to decompress: {}", e))
        })?;
        buf
    } else {
        std::fs::read(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to read file: {}", e))
        })?
    };

    // Validate
    if data.len() % RECORD_SIZE != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid trace file: size {} is not multiple of {}",
            data.len(),
            RECORD_SIZE
        )));
    }

    let n_records = data.len() / RECORD_SIZE;
    let n_records = max_requests.map_or(n_records, |max| max.min(n_records));

    // Preallocate arrays
    let mut timestamps = Vec::with_capacity(n_records);
    let mut obj_ids = Vec::with_capacity(n_records);
    let mut obj_sizes = Vec::with_capacity(n_records);

    // Parse records (filter zero-size objects)
    for i in 0..n_records {
        let offset = i * RECORD_SIZE;
        let (ts, oid, size, _) = parse_record(&data[offset..offset + RECORD_SIZE]);

        if size > 0 {
            timestamps.push(ts);
            obj_ids.push(oid);
            obj_sizes.push(size);
        }
    }

    let n_requests = timestamps.len();

    // Build topology
    let (next_access_idx, prev_access_idx, n_unique) =
        build_access_topology(&obj_ids, &obj_sizes);

    Ok(ParseResult {
        n_requests,
        n_unique_objects: n_unique,
        timestamps,
        obj_ids,
        obj_sizes,
        next_access_idx,
        prev_access_idx,
    })
}

/// Parse trace from bytes (for pre-loaded data)
#[pyfunction]
#[pyo3(signature = (data, max_requests=None))]
fn parse_trace_from_bytes(data: &Bound<'_, PyBytes>, max_requests: Option<usize>) -> PyResult<ParseResult> {
    let bytes = data.as_bytes();

    if bytes.len() % RECORD_SIZE != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid data: size {} is not multiple of {}",
            bytes.len(),
            RECORD_SIZE
        )));
    }

    let n_records = bytes.len() / RECORD_SIZE;
    let n_records = max_requests.map_or(n_records, |max| max.min(n_records));

    let mut timestamps = Vec::with_capacity(n_records);
    let mut obj_ids = Vec::with_capacity(n_records);
    let mut obj_sizes = Vec::with_capacity(n_records);

    for i in 0..n_records {
        let offset = i * RECORD_SIZE;
        let (ts, oid, size, _) = parse_record(&bytes[offset..offset + RECORD_SIZE]);

        if size > 0 {
            timestamps.push(ts);
            obj_ids.push(oid);
            obj_sizes.push(size);
        }
    }

    let n_requests = timestamps.len();
    let (next_access_idx, prev_access_idx, n_unique) =
        build_access_topology(&obj_ids, &obj_sizes);

    Ok(ParseResult {
        n_requests,
        n_unique_objects: n_unique,
        timestamps,
        obj_ids,
        obj_sizes,
        next_access_idx,
        prev_access_idx,
    })
}

/// Build topology from pre-parsed arrays (for pure topology building benchmark)
#[pyfunction]
fn build_topology_from_arrays<'py>(
    py: Python<'py>,
    obj_ids: PyReadonlyArray1<'py, u64>,
    obj_sizes: PyReadonlyArray1<'py, u32>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>, usize)> {
    let ids = obj_ids.as_slice()?;
    let sizes = obj_sizes.as_slice()?;

    let (next_idx, prev_idx, n_unique) = build_access_topology(ids, sizes);

    Ok((
        PyArray1::from_slice_bound(py, &next_idx),
        PyArray1::from_slice_bound(py, &prev_idx),
        n_unique,
    ))
}

/// Python module
#[pymodule]
fn foo_jax_fast_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ParseResult>()?;
    m.add_function(wrap_pyfunction!(parse_trace_fast, m)?)?;
    m.add_function(wrap_pyfunction!(parse_trace_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(build_topology_from_arrays, m)?)?;
    Ok(())
}
