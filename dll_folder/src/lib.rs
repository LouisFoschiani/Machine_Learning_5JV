include!("models/linear_model.rs");
include!("models/mlp_model.rs");

/**
cd ..; cargo build --release; if ($?) { cd src; python app.py }
*/

use serde::{Deserialize};

#[derive(Deserialize)]
struct Config {
    model: String,
    mode: String,
    category: usize,
}

#[no_mangle]
pub extern "C" fn run_algo() {
    let config_contents = fs::read_to_string("config.json").expect("Failed to read config file");
    let config: Config = serde_json::from_str(&config_contents).expect("Failed to parse config");
    println!("Model: {}, Mode: {}, Category: {}", config.model, config.mode, config.category);
    let algorithm_choice = config.model;

    match algorithm_choice.as_str() {
        "linear_model" => {
            println!("Running Linear Model Algorithm...");
            match run_linear_model(config.mode.as_str(),config.category) {
                Ok(_) => println!("Linear Model completed successfully."),
                Err(e) => println!("Linear Model encountered an error: {}", e),
            }
        },
        "mlp_model" => {
            println!("Running MLP Model Algorithm...");
            match run_mlp_model(config.mode.as_str()) {
                Ok(_) => println!("MLP Model completed successfully."),
                Err(e) => println!("MLP Model encountered an error: {}", e),
            }
        }
        // ...
        _ => {
            println!("Unknown algorithm choice.");
        }
    }
}
