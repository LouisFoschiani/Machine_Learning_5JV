#![allow(non_snake_case)]
mod linear_model;
mod mlp_model;
//mod rbf;

fn main() {
    let algorithm_choice = "mlp_model";

    match algorithm_choice {
        "linear_model" => {
            println!("Running Linear Model Algorithm...");
            linear_model::main().unwrap();
        }
        "mlp" => {
             println!("Running Multi-Layer Perceptron Algorithm...");
            mlp_model::main();
         }
        /*"rbf" => {
            println!("Running Radial Basis Function Network Algorithm...");
            rbf::run_rbf_algorithm();
        }*/
        _ => {
            println!("Unknown algorithm choice.");
        }
    }
}