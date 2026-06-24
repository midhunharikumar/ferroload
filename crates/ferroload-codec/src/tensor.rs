//! Minimal owned tensor returned by decoders. The PyO3 layer maps these to
//! torch tensors (zero-copy via DLPack) at the boundary.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    U8,
    F32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
}

impl Tensor {
    pub fn dtype(&self) -> Dtype {
        match self.data {
            TensorData::U8(_) => Dtype::U8,
            TensorData::F32(_) => Dtype::F32,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Validate that the element count matches the declared shape.
    pub fn check(&self) -> bool {
        let n = self.numel();
        match &self.data {
            TensorData::U8(v) => v.len() == n,
            TensorData::F32(v) => v.len() == n,
        }
    }
}
