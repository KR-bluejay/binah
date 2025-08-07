use binah_core::{Graph, Shape};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new computational graph
    let mut graph = Graph::new();
    
    // Create constants
    let a = graph.constant(vec![1.0f32, 2.0f32], Shape::from([2]));
    let b = graph.constant(vec![3.0f32, 4.0f32], Shape::from([2]));
    
    // Perform addition
    let c = a + b;
    
    // Compile the graph
    let mut executable = graph.compile(&[&c])?;
    
    // Execute (no inputs needed for this example since we use constants)
    let results = executable.execute(HashMap::new())?;
    
    println!("Results: {:?}", results);
    
    Ok(())
}