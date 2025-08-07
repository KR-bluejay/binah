use std::{cell::RefCell, rc::Rc};

use inner::GraphInner;
use num_traits::{One, Zero};

use crate::{
    op::Operation,
    tensor::{Tensor, shape::Shape, storage::IntoStorage},
};
pub mod execute;
pub(crate) mod inner;
pub mod tensor;

pub use execute::{ExecutionError, GraphExecutable};
pub use tensor::GraphTensor;

#[derive(Clone, Debug)]
pub struct Graph {
    inner: Rc<RefCell<GraphInner>>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(GraphInner::new())),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Rc::new(RefCell::new(GraphInner::with_capacity(capacity))),
        }
    }

    pub fn constant<T>(&mut self, data: Vec<T>, shape: Shape) -> GraphTensor
    where
        T: IntoStorage + Zero + One,
    {
        let node_id = self.inner.borrow_mut().add_op(Operation::Constant);
        let tensor = Tensor::from_data(data, shape.clone());

        self.inner
            .borrow_mut()
            .add_storage(node_id, tensor.into_storage());

        GraphTensor::new(self.inner.clone(), node_id, shape)
    }

    pub fn variable<T>(&mut self, data: Vec<T>, shape: Shape) -> GraphTensor
    where
        T: IntoStorage + Zero + One,
    {
        let node_id = self.inner.borrow_mut().add_op(Operation::Variable);
        let tensor = Tensor::from_data(data, shape.clone());

        self.inner
            .borrow_mut()
            .add_storage(node_id, tensor.into_storage());

        GraphTensor::new(self.inner.clone(), node_id, shape)
    }

    pub fn placeholder(&mut self, shape: Shape) -> GraphTensor {
        let node_id = self.inner.borrow_mut().add_op(Operation::Placeholder);

        GraphTensor::new(self.inner.clone(), node_id, shape)
    }

    pub fn compile(
        &mut self,
        target_tensors: &[&GraphTensor],
    ) -> Result<GraphExecutable, ExecutionError> {
        let graph_inner = self.inner.borrow();

        GraphExecutable::new(
            graph_inner.graph(),
            graph_inner.tensor_storage().clone(),
            target_tensors,
        )
    }
}
