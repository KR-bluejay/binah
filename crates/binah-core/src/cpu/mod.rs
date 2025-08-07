use crate::tensor::{shape::Shape, storage::TensorStorage};

pub fn cpu_add(lhs: &TensorStorage, rhs: &TensorStorage, output_shape: &Shape) -> TensorStorage {
    match (lhs, rhs) {
        (TensorStorage::F32 { data: lhs_data, shape: lhs_shape }, 
         TensorStorage::F32 { data: rhs_data, shape: rhs_shape }) => {
            let lhs_shape = Shape::from(lhs_shape.clone());
            let rhs_shape = Shape::from(rhs_shape.clone());
            
            let lhs_strides = lhs_shape.compute_broadcast_strides(output_shape).unwrap();
            let rhs_strides = rhs_shape.compute_broadcast_strides(output_shape).unwrap();
            
            let output_size = output_shape.num_elements();
            let mut result = vec![0.0f32; output_size];
            
            for i in 0..output_size {
                let lhs_idx = compute_index(i, &lhs_strides, &output_shape.dims());
                let rhs_idx = compute_index(i, &rhs_strides, &output_shape.dims());
                
                result[i] = lhs_data[lhs_idx] + rhs_data[rhs_idx];
            }
            
            TensorStorage::F32 { 
                data: result, 
                shape: output_shape.dims().to_vec() 
            }
        }
        _ => panic!("Unsupported tensor types for addition"),
    }
}

pub fn cpu_sub(lhs: &TensorStorage, rhs: &TensorStorage, output_shape: &Shape) -> TensorStorage {
    match (lhs, rhs) {
        (TensorStorage::F32 { data: lhs_data, shape: lhs_shape }, 
         TensorStorage::F32 { data: rhs_data, shape: rhs_shape }) => {
            let lhs_shape = Shape::from(lhs_shape.clone());
            let rhs_shape = Shape::from(rhs_shape.clone());
            
            let lhs_strides = lhs_shape.compute_broadcast_strides(output_shape).unwrap();
            let rhs_strides = rhs_shape.compute_broadcast_strides(output_shape).unwrap();
            
            let output_size = output_shape.num_elements();
            let mut result = vec![0.0f32; output_size];
            
            for i in 0..output_size {
                let lhs_idx = compute_index(i, &lhs_strides, &output_shape.dims());
                let rhs_idx = compute_index(i, &rhs_strides, &output_shape.dims());
                
                result[i] = lhs_data[lhs_idx] - rhs_data[rhs_idx];
            }
            
            TensorStorage::F32 { 
                data: result, 
                shape: output_shape.dims().to_vec() 
            }
        }
        _ => panic!("Unsupported tensor types for subtraction"),
    }
}

pub fn cpu_mul(lhs: &TensorStorage, rhs: &TensorStorage, output_shape: &Shape) -> TensorStorage {
    match (lhs, rhs) {
        (TensorStorage::F32 { data: lhs_data, shape: lhs_shape }, 
         TensorStorage::F32 { data: rhs_data, shape: rhs_shape }) => {
            let lhs_shape = Shape::from(lhs_shape.clone());
            let rhs_shape = Shape::from(rhs_shape.clone());
            
            let lhs_strides = lhs_shape.compute_broadcast_strides(output_shape).unwrap();
            let rhs_strides = rhs_shape.compute_broadcast_strides(output_shape).unwrap();
            
            let output_size = output_shape.num_elements();
            let mut result = vec![0.0f32; output_size];
            
            for i in 0..output_size {
                let lhs_idx = compute_index(i, &lhs_strides, &output_shape.dims());
                let rhs_idx = compute_index(i, &rhs_strides, &output_shape.dims());
                
                result[i] = lhs_data[lhs_idx] * rhs_data[rhs_idx];
            }
            
            TensorStorage::F32 { 
                data: result, 
                shape: output_shape.dims().to_vec() 
            }
        }
        _ => panic!("Unsupported tensor types for multiplication"),
    }
}

pub fn cpu_div(lhs: &TensorStorage, rhs: &TensorStorage, output_shape: &Shape) -> TensorStorage {
    match (lhs, rhs) {
        (TensorStorage::F32 { data: lhs_data, shape: lhs_shape }, 
         TensorStorage::F32 { data: rhs_data, shape: rhs_shape }) => {
            let lhs_shape = Shape::from(lhs_shape.clone());
            let rhs_shape = Shape::from(rhs_shape.clone());
            
            let lhs_strides = lhs_shape.compute_broadcast_strides(output_shape).unwrap();
            let rhs_strides = rhs_shape.compute_broadcast_strides(output_shape).unwrap();
            
            let output_size = output_shape.num_elements();
            let mut result = vec![0.0f32; output_size];
            
            for i in 0..output_size {
                let lhs_idx = compute_index(i, &lhs_strides, &output_shape.dims());
                let rhs_idx = compute_index(i, &rhs_strides, &output_shape.dims());
                
                result[i] = lhs_data[lhs_idx] / rhs_data[rhs_idx];
            }
            
            TensorStorage::F32 { 
                data: result, 
                shape: output_shape.dims().to_vec() 
            }
        }
        _ => panic!("Unsupported tensor types for division"),
    }
}

fn compute_index(linear_idx: usize, strides: &[usize], shape: &[usize]) -> usize {
    let mut index = 0;
    let mut remaining = linear_idx;
    
    for i in 0..shape.len() {
        let dim_size = shape[i];
        let stride = strides[i];
        
        // Calculate size of remaining dimensions
        let next_size: usize = shape[i+1..].iter().product();
        
        let coord = (remaining / next_size) % dim_size;
        index += coord * stride;
        remaining %= next_size;
    }
    
    index
}