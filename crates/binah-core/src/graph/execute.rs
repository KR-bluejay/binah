use crate::{
    cpu::{cpu_add, cpu_sub, cpu_mul, cpu_div},
    op::Operation, 
    tensor::{shape::Shape, storage::TensorStorage}
};
use petgraph::{
    Direction, algo::toposort, graph::NodeIndex, prelude::StableGraph,
    visit::EdgeRef,
};
use std::collections::{HashMap, HashSet};

use super::tensor::GraphTensor;

#[derive(Debug)]
pub struct GraphExecutable {
    graph: StableGraph<Operation, ()>,
    execution_plan: Vec<NodeIndex>,
    tensor_storage: HashMap<NodeIndex, TensorStorage>,
    inputs: Vec<NodeIndex>,
    outputs: Vec<NodeIndex>,
}

#[derive(Debug)]
pub enum ExecutionError {
    MissingInput(NodeIndex),
    InvalidOperation,
    CyclicGraph,
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::MissingInput(node) => write!(f, "Missing input for node {:?}", node),
            ExecutionError::InvalidOperation => write!(f, "Invalid operation"),
            ExecutionError::CyclicGraph => write!(f, "Graph contains cycles"),
        }
    }
}

impl std::error::Error for ExecutionError {}

impl GraphExecutable {
    pub fn new(
        graph: &StableGraph<Operation, ()>,
        tensor_storage: HashMap<NodeIndex, TensorStorage>,
        target_tensors: &[&GraphTensor],
    ) -> Result<Self, ExecutionError> {
        let required_nodes = if target_tensors.is_empty() {
            graph.node_indices().collect()
        } else {
            Self::find_required_nodes(graph, target_tensors)
        };

        let full_execution_plan =
            toposort(graph, None).map_err(|_| ExecutionError::CyclicGraph)?;
        let execution_plan: Vec<NodeIndex> = full_execution_plan
            .into_iter()
            .filter(|node| required_nodes.contains(node))
            .collect();

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for &node_idx in &required_nodes {
            if let Some(op) = graph.node_weight(node_idx) {
                match op {
                    Operation::Placeholder => inputs.push(node_idx),
                    _ => {}
                }
            }

            let has_outgoing = graph
                .edges_directed(node_idx, Direction::Outgoing)
                .any(|edge| required_nodes.contains(&edge.target()));
            if !has_outgoing {
                outputs.push(node_idx);
            }
        }

        // Filter tensor storage to only include required nodes
        let pruned_tensor_storage: HashMap<NodeIndex, TensorStorage> =
            tensor_storage
                .into_iter()
                .filter(|(node, _)| required_nodes.contains(node))
                .collect();

        Ok(Self {
            graph: graph.clone(),
            execution_plan,
            tensor_storage: pruned_tensor_storage,
            inputs,
            outputs,
        })
    }

    fn find_required_nodes(
        graph: &StableGraph<Operation, ()>,
        target_tensors: &[&GraphTensor],
    ) -> HashSet<NodeIndex> {
        let mut required = HashSet::new();
        let mut stack = Vec::new();
        
        // Start from target nodes
        for tensor in target_tensors {
            let node_id = tensor.node_id();
            if !required.contains(&node_id) {
                stack.push(node_id);
                required.insert(node_id);
            }
        }
        
        // Backward traversal to find all dependencies
        while let Some(node_id) = stack.pop() {
            for edge in graph.edges_directed(node_id, Direction::Incoming) {
                let source = edge.source();
                if !required.contains(&source) {
                    required.insert(source);
                    stack.push(source);
                }
            }
        }
        
        required
    }

    pub fn execute(
        &mut self,
        input_data: HashMap<NodeIndex, TensorStorage>,
    ) -> Result<HashMap<NodeIndex, TensorStorage>, ExecutionError> {
        // Set input data
        for &node_idx in &self.inputs {
            if let Some(data) = input_data.get(&node_idx) {
                self.tensor_storage.insert(node_idx, data.clone());
            } else {
                return Err(ExecutionError::MissingInput(node_idx));
            }
        }

        // Execute operations in topological order
        let execution_plan = self.execution_plan.clone();
        for node_idx in execution_plan {
            if let Some(operation) = self.graph.node_weight(node_idx) {
                match operation {
                    Operation::Constant | Operation::Variable | Operation::Placeholder => {
                        // These already have their data in tensor_storage
                        continue;
                    }
                    Operation::Add => {
                        self.execute_binary_op(node_idx, cpu_add)?;
                    }
                    Operation::Sub => {
                        self.execute_binary_op(node_idx, cpu_sub)?;
                    }
                    Operation::Mul => {
                        self.execute_binary_op(node_idx, cpu_mul)?;
                    }
                    Operation::Div => {
                        self.execute_binary_op(node_idx, cpu_div)?;
                    }
                }
            }
        }

        // Collect outputs
        let mut results = HashMap::new();
        for &output_idx in &self.outputs {
            if let Some(data) = self.tensor_storage.get(&output_idx) {
                results.insert(output_idx, data.clone());
            }
        }

        Ok(results)
    }
    
    fn execute_binary_op(
        &mut self,
        node_idx: NodeIndex,
        op_fn: fn(&TensorStorage, &TensorStorage, &Shape) -> TensorStorage,
    ) -> Result<(), ExecutionError> {
        // Get input nodes (assumes binary operation has exactly 2 inputs)
        let inputs: Vec<NodeIndex> = self.graph
            .edges_directed(node_idx, Direction::Incoming)
            .map(|edge| edge.source())
            .collect();
        
        if inputs.len() != 2 {
            return Err(ExecutionError::InvalidOperation);
        }
        
        let lhs_data = self.tensor_storage.get(&inputs[0])
            .ok_or(ExecutionError::InvalidOperation)?;
        let rhs_data = self.tensor_storage.get(&inputs[1])
            .ok_or(ExecutionError::InvalidOperation)?;
        
        // Compute output shape (broadcast)
        let lhs_shape = Shape::from(lhs_data.shape().to_vec());
        let rhs_shape = Shape::from(rhs_data.shape().to_vec());
        let output_shape = lhs_shape.broadcast_with(&rhs_shape)
            .map_err(|_| ExecutionError::InvalidOperation)?;
        
        // Execute operation
        let result = op_fn(lhs_data, rhs_data, &output_shape);
        
        // Store result
        self.tensor_storage.insert(node_idx, result);
        
        Ok(())
    }

    pub fn inputs(&self) -> &[NodeIndex] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[NodeIndex] {
        &self.outputs
    }
}
