#![allow(non_snake_case)]
mod linear_model;
mod mlp_model;
mod scraping;
mod rbf_model;

fn main() {
    let algorithm_choice = "rbf_model";

    match algorithm_choice {
        "linear_model" => {
            println!("Running Linear Model Algorithm...");
            linear_model::main().unwrap();
        }
        "mlp_model" => {
             println!("Running Multi-Layer Perceptron Algorithm...");
            mlp_model::main();
         }
        "rbf_model" => {
            println!("Running Radial Basis Function Network Algorithm...");
            rbf_model::main().unwrap();
        }
        _ => {
            println!("Unknown algorithm choice.");
        }
    }
}