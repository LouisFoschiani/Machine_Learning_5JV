#![allow(non_snake_case)]
mod linear_model;

fn main() {
    match linear_model::run() {
        Ok(()) => println!("Linear model executed successfully."),
        Err(e) => println!("Error executing linear model: {}", e),
    }
}