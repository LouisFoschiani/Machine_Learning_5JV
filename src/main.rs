#![allow(non_snake_case)]
mod linear_model;
//mod mlp;
//mod rbf;

fn main() {
    let algorithm_choice = "linear_model";

    match algorithm_choice {
        "linear_model" => {
            println!("Running Linear Model Algorithm...");
            linear_model::main().unwrap(); // Remarquez le changement de nom de la fonction
        }
        /* "mlp" => {
             println!("Running Multi-Layer Perceptron Algorithm...");
             mlp::run_mlp_algorithm(); // Changez le nom de la fonction en fonction de l'algorithme
         }
         "rbf" => {
             println!("Running Radial Basis Function Network Algorithm...");
             rbf::run_rbf_algorithm(); // Changez le nom de la fonction en fonction de l'algorithme
         }*/
        _ => {
            println!("Unknown algorithm choice.");
        }
    }
}
