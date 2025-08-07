use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use petgraph::graph::NodeIndex;

use crate::tensor::shape::Shape;

use super::GraphInner;

#[derive(Debug)]
pub struct GraphTensor {
    graph: Weak<RefCell<GraphInner>>,
    node_id: NodeIndex,
    shape: Shape,
}

impl GraphTensor {
    pub(crate) fn new(
        graph: Rc<RefCell<GraphInner>>,
        node_id: NodeIndex,
        shape: Shape,
    ) -> Self {
        Self {
            graph: Rc::downgrade(&graph),
            node_id,
            shape,
        }
    }

    pub fn graph(&self) -> Rc<RefCell<GraphInner>> {
        self.graph.upgrade().expect("Graph dropped")
    }

    pub fn node_id(&self) -> NodeIndex {
        self.node_id
    }

    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }
}
