use std::collections::HashMap;
use std::fs::{File, read_dir};
use std::path::Path;
use std::vec::Vec;
use std::ops::Index;
use std::iter::Iterator;
use std::{f32, io};
use std::io::{BufRead, BufReader};

use image;
use image::DynamicImage;
use image::Rgb;
use image::io::Reader;
use rand::distributions::Uniform;
use rand::prelude::ThreadRng;

use rand::Rng;

fn destroy_array<T>(array: Vec<T>) {
    drop(array); // Cela libère la mémoire, mais est généralement inutile en Rust
}
fn print_float_array(array: &[f32], index: usize) -> Option<f32> {
    array.get(index).copied() // Renvoie None si l'index est hors limites, sinon renvoie la valeur
}
fn save_model_linear(model_weight: &[f32], file_path: &Path, efficiency: f64) -> Result<()> {
    let mut file = File::create(file_path)?;
    writeln!(file, "-- Efficiency --\n{}", efficiency)?;
    writeln!(file, "-- W --")?;
    for &weight in model_weight {
        writeln!(file, "{{{}}}", weight)?;
    }
    Ok(())
}

fn load_model_linear(file_path: &Path) -> io::Result<Vec<f32>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut weights = Vec::new();
    let mut read_weights = false;

    for line in reader.lines() {
        let line = line?;
        if line.trim() == "-- W --" {
            read_weights = true;
            continue;
        }
        if read_weights {
            if let Some(weight) = line.trim().strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
                if let Ok(weight) = weight.parse::<f32>() {
                    weights.push(weight);
                }
            }
        }
    }

    Ok(weights)
}

fn init_model_weights(cols_x_len: usize, rows_w_len: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    init_weights_with_random(&mut rng, cols_x_len, rows_w_len)
}

fn init_weights_with_random(rng: &mut ThreadRng, cols_x_len: usize, rows_w_len: usize) -> Vec<f32> {
    let mut between = Uniform::from(-1.0..1.0);
    let mut weights = vec![0.0; rows_w_len];
    for weight in weights.iter_mut().take(cols_x_len + 1) {
        *weight = between.sample(rng);
    }
    weights
}

fn predict_linear_model_classification_float(model_weights: &[f32], inputs: &[f32]) -> i32 {
    if model_weights.len() != inputs.len() + 1 {
        panic!("Les longueurs des poids et des entrées ne correspondent pas.");
    }

    let mut res = 0.0;
    for (&weight, &input) in model_weights.iter().skip(1).zip(inputs.iter()) {
        res += weight * input;
    }
    let total_sum = model_weights[0] + res;

    if total_sum >= 0.0 { 1 } else { -1 }
}

fn predict_linear_model_classification_int(model_weights: &[f32], inputs: &[i32]) -> i32 {
    if model_weights.len() != inputs.len() + 1 {
        panic!("Les longueurs des poids et des entrées ne correspondent pas.");
    }

    let mut res = 0.0;
    for (&weight, &input) in model_weights.iter().skip(1).zip(inputs.iter()) {
        res += weight * input as f32; // Conversion des entrées en f32 pour la multiplication
    }
    let total_sum = model_weights[0] + res;

    if total_sum >= 0.0 { 1 } else { -1 }
}

fn train_linear_float(x: &[Vec<f32>], y: &[i32], w: &mut [f32], rows_x_len: usize, rows_w_len: usize, iter: usize) {
    let learning_rate = 0.001;

    for _ in 0..iter {
        let k = rand::thread_rng().gen_range(0..rows_x_len);
        let gxk = predict_linear_model_classification_float(w, &x[k]);
        let yk = y[k];
        let diff = yk as f32 - gxk as f32;

        w[0] = w[0] + learning_rate * diff * 1.0;
        for j in 1..rows_w_len {
            w[j] = w[j] + learning_rate * diff * x[k][j - 1];
        }
    }
}

fn train_linear_int(x: &[Vec<i32>], y: &[i32], w: &mut [f32], rows_x_len: usize, rows_w_len: usize, iter: usize) {
    let learning_rate = 0.001;

    for _ in 0..iter {
        let k = rand::thread_rng().gen_range(0..rows_x_len);
        let gxk = predict_linear_model_classification_int(w, &x[k]);
        let yk = y[k];
        let diff = yk - gxk;
        w[0] = w[0] + learning_rate * diff as f32;

        for j in 1..rows_w_len {
            w[j] = w[j] + learning_rate * diff as f32 * x[k][j - 1] as f32;
        }
    }
}

fn load_data(path: &str) -> io::Result<(Vec<Vec<f32>>, Vec<i32>)> {
    let file = File::open(Path::new(path))?;
    let reader = BufReader::new(file);

    let mut features: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<i32> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let values: Vec<&str> = line.split(',').collect();

        if let Some((last, elements)) = values.split_last() {
            let label = last.parse::<i32>().unwrap_or_else(|_| panic!("Erreur de conversion du label"));
            let feature = elements.iter()
                .map(|&x| x.parse::<f32>().unwrap_or_else(|_| panic!("Erreur de conversion de la caractéristique")))
                .collect();

            features.push(feature);
            labels.push(label);
        }
    }

    Ok((features, labels))
}

fn main() {
    // Charger les données d'entraînement et de test
    let (x_train, y_train) = load_data("chemin_vers_données_entrainement");
    let (x_test, y_test) = load_data("chemin_vers_données_test");

    let rows_x_len = x_train.len();
    let rows_w_len = x_train[0].len(); // Assurez-vous que c'est correct pour votre modèle

    // Initialisation des poids du modèle
    let mut weights = init_model_weights(rows_x_len, rows_w_len);

    // Entraînement du modèle
    train_linear_float(&x_train, &y_train, &mut weights, rows_x_len, rows_w_len, N_ITER);

    // Test du modèle et calcul de la performance
    let mut final_result = 0;
    for (x, &y) in x_test.iter().zip(y_test.iter()) {
        let result = predict_linear_model_classification_float(&weights, x);
        if result == y {
            final_result += 1;
        }
    }
    let performance = final_result as f32 / y_test.len() as f32 * 100.0;
    println!("Performance: {}%", performance);

    // Sauvegarde des poids du modèle si la performance est satisfaisante
    if performance > MAX_POURCENTAGE {
        save_model_linear(&weights, Path::new("./save/best.txt"), performance as f64).unwrap();
    }
}