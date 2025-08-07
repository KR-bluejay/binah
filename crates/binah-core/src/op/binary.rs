use crate::{graph::tensor::GraphTensor, op::Operation};
use std::ops::{Add, Div, Mul, Sub};

impl Add for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: Self) -> Self::Output {
        let graph_rc = self.graph();
        
        let result_shape = self.shape()
            .broadcast_with(&rhs.shape())
            .expect("Incompatible shapes for addition");

        let node_id = graph_rc.borrow_mut().add_binary_op(
            self.node_id(),
            rhs.node_id(),
            Operation::Add,
        );

        GraphTensor::new(graph_rc, node_id, result_shape)
    }
}

impl Sub for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: Self) -> Self::Output {
        let graph_rc = self.graph();
        
        let result_shape = self.shape()
            .broadcast_with(&rhs.shape())
            .expect("Incompatible shapes for subtraction");

        let node_id = graph_rc.borrow_mut().add_binary_op(
            self.node_id(),
            rhs.node_id(),
            Operation::Sub,
        );

        GraphTensor::new(graph_rc, node_id, result_shape)
    }
}

impl Mul for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: Self) -> Self::Output {
        let graph_rc = self.graph();
        
        let result_shape = self.shape()
            .broadcast_with(&rhs.shape())
            .expect("Incompatible shapes for multiplication");

        let node_id = graph_rc.borrow_mut().add_binary_op(
            self.node_id(),
            rhs.node_id(),
            Operation::Mul,
        );

        GraphTensor::new(graph_rc, node_id, result_shape)
    }
}

impl Div for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: Self) -> Self::Output {
        let graph_rc = self.graph();
        
        let result_shape = self.shape()
            .broadcast_with(&rhs.shape())
            .expect("Incompatible shapes for division");

        let node_id = graph_rc.borrow_mut().add_binary_op(
            self.node_id(),
            rhs.node_id(),
            Operation::Div,
        );

        GraphTensor::new(graph_rc, node_id, result_shape)
    }
}
