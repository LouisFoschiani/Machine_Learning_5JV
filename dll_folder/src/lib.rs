
include!("models/linear_model.rs");
include!("models/mlp_model.rs");
extern crate image;
extern crate rand;

use std::collections::HashMap;
use image::{DynamicImage, GenericImageView, ImageError};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use plotters::prelude::*;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::element::PathElement;
use plotters::prelude::*;
use std::{fs, iter, path::Path};
use std::f32::consts::E;
use std::fs::File;
use std::io::{self, Read, Write};
use serde_json;
use plotters::prelude::*;
use std::io::BufReader;
use std::io::BufRead;

use serde::{Deserialize};
/**
cd ..; cargo build --release; if ($?) { cd src; python app.py }
*/




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
        },
        // ...
        _ => {
            println!("Unknown algorithm choice.");
        }
    }
}
