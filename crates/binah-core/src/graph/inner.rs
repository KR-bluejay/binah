use std::collections::HashMap;

use petgraph::{graph::NodeIndex, prelude::StableGraph};

use crate::{op::Operation, tensor::storage::TensorStorage};

#[derive(Clone, Debug)]
pub(crate) struct GraphInner {
    graph: StableGraph<Operation, ()>,
    tensor_map: HashMap<NodeIndex, TensorStorage>,
}

impl GraphInner {
    pub(crate) fn new() -> Self {
        Self {
            graph: StableGraph::new(),
            tensor_map: HashMap::new(),
        }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        // TODO: StableGraph Edge capacity
        Self {
            graph: StableGraph::with_capacity(capacity, 0),
            tensor_map: HashMap::with_capacity(capacity),
        }
    }

    pub(crate) fn graph(&self) -> &StableGraph<Operation, ()> {
        &self.graph
    }

    pub(crate) fn tensor_storage(&self) -> &HashMap<NodeIndex, TensorStorage> {
        &self.tensor_map
    }

    pub fn add_op(&mut self, op: Operation) -> NodeIndex {
        self.graph.add_node(op)
    }

    pub fn add_binary_op(
        &mut self,
        lhs: NodeIndex,
        rhs: NodeIndex,
        op: Operation,
    ) -> NodeIndex {
        let node_id = self.add_op(op);

        self.graph.add_edge(lhs, node_id, ());
        self.graph.add_edge(rhs, node_id, ());

        node_id
    }

    pub(crate) fn add_storage(
        &mut self,
        node_id: NodeIndex,
        storage: TensorStorage,
    ) {
        self.tensor_map.insert(node_id, storage);
    }
}
