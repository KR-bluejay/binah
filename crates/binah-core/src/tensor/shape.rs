#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub(crate) dims: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum BroadcastError {
    IncompatibleShapes(usize, usize),
    DimensionMismatch,
}

impl Shape {
    pub fn new<const D: usize>(dims: [usize; D]) -> Self {
        // For backward compat
        Self {
            dims: dims.to_vec(),
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty() || (self.dims.len() == 1 && self.dims[0] == 1)
    }

    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn contiguous_strides(&self) -> Vec<usize> {
        let n = self.dims.len();

        let mut strides = vec![0; n];
        let mut stride = 1;

        for i in (0..n).rev() {
            strides[i] = stride;
            stride *= self.dims[i];
        }

        strides
    }

    pub fn can_broadcast_with(&self, other: &Shape) -> bool {
        self.broadcast_with(other).is_ok()
    }

    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape, BroadcastError> {
        let self_dims = &self.dims;
        let other_dims = &other.dims;
        let max_ndim = self_dims.len().max(other_dims.len());
        
        let mut result_dims = Vec::with_capacity(max_ndim);
        
        for i in 0..max_ndim {
            let self_dim = if i < self_dims.len() {
                self_dims[self_dims.len() - 1 - i]
            } else {
                1
            };
            
            let other_dim = if i < other_dims.len() {
                other_dims[other_dims.len() - 1 - i]
            } else {
                1
            };
            
            let result_dim = match (self_dim, other_dim) {
                (a, b) if a == b => a,
                (1, b) => b,
                (a, 1) => a,
                (a, b) => return Err(BroadcastError::IncompatibleShapes(a, b)),
            };
            
            result_dims.push(result_dim);
        }
        
        result_dims.reverse();
        Ok(Shape { dims: result_dims })
    }

    pub fn compute_broadcast_strides(&self, target_shape: &Shape) -> Result<Vec<usize>, BroadcastError> {
        if !self.can_broadcast_with(target_shape) {
            return Err(BroadcastError::DimensionMismatch);
        }
        
        let self_strides = self.contiguous_strides();
        let mut broadcast_strides = vec![0; target_shape.dims.len()];
        
        let self_offset = target_shape.dims.len().saturating_sub(self.dims.len());
        
        for (i, &dim) in self.dims.iter().enumerate() {
            let target_idx = self_offset + i;
            if dim == 1 && target_shape.dims[target_idx] != 1 {
                broadcast_strides[target_idx] = 0;
            } else {
                broadcast_strides[target_idx] = self_strides[i];
            }
        }
        
        Ok(broadcast_strides)
    }
}

impl<const D: usize> From<[usize; D]> for Shape {
    fn from(dims: [usize; D]) -> Self {
        Shape::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape { dims: dims.into() }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self { dims: shape }
    }
}

impl From<&Vec<usize>> for Shape {
    fn from(shape: &Vec<usize>) -> Self {
        Self {
            dims: shape.clone(),
        }
    }
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.dims
    }
}
