use num_traits::{One, Zero};
use shape::Shape;
use storage::{IntoStorage, TensorStorage};

pub mod shape;

pub(crate) mod storage;

#[derive(Debug, Clone)]
pub struct Tensor<T>
where
    T: IntoStorage,
{
    pub data: Vec<T>,
    pub shape: Shape,
}

impl<T> Tensor<T>
where
    T: IntoStorage + Zero + One,
{
    pub fn with_capacity(shape: Shape) -> Self {
        Self {
            data: Vec::with_capacity(shape.num_elements()),
            shape,
        }
    }

    pub fn zeros(shape: Shape) -> Self {
        Self {
            data: vec![T::zero(); shape.num_elements()],
            shape,
        }
    }

    pub fn ones(shape: Shape) -> Self {
        Self {
            data: vec![T::one(); shape.num_elements()],
            shape,
        }
    }

    pub fn from_data(data: Vec<T>, shape: Shape) -> Self {
        Self { data, shape }
    }

    pub fn into_storage(self) -> TensorStorage {
        T::into_storage(self.data, self.shape.into())
    }
}
