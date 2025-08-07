use binah_core::{Graph, Shape};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Broadcast Test ===");
    
    let mut graph = Graph::new();
    
    // Test 1: [3, 1] + [2] -> [3, 2]
    println!("Test 1: [3, 1] + [2] -> [3, 2]");
    let a = graph.constant(vec![1.0f32, 2.0f32, 3.0f32], Shape::from([3, 1]));
    let b = graph.constant(vec![10.0f32, 20.0f32], Shape::from([2]));
    let c = a + b;
    
    let mut executable = graph.compile(&[&c])?;
    let results = executable.execute(HashMap::new())?;
    println!("Result: {:?}", results);
    
    // Test 2: Scalar + Vector
    println!("\nTest 2: [1] + [3] -> [3]");
    let mut graph2 = Graph::new();
    let d = graph2.constant(vec![5.0f32], Shape::from([1]));
    let e = graph2.constant(vec![1.0f32, 2.0f32, 3.0f32], Shape::from([3]));
    let f = d + e;
    
    let mut executable2 = graph2.compile(&[&f])?;
    let results2 = executable2.execute(HashMap::new())?;
    println!("Result: {:?}", results2);
    
    Ok(())
}