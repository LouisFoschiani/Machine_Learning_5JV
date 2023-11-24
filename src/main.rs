#[warn(non_snake_case)]

use image::{self, GenericImageView};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path};

fn load_image_data(base_path: &Path, categories: &[&str]) -> io::Result<(Vec<Vec<f32>>, Vec<i32>)> {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for (label, &category) in categories.iter().enumerate() {
        let category_path = base_path.join(category);
        for entry in fs::read_dir(category_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let image_features = process_image(&path)?;
                features.push(image_features);
                labels.push(label as i32);
            }
        }
    }

    Ok((features, labels))
}

fn process_image(path: &Path) -> io::Result<Vec<f32>> {
    let img = image::open(path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let features = img.pixels()
        .flat_map(|(_, _, pixel)| pixel.0.to_vec())
        .map(|value| value as f32 / 255.0)
        .collect();
    Ok(features)
}

fn init_model_weights(cols_x_len: usize, rows_w_len: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let between = Uniform::from(-1.0..1.0);
    let mut weights = vec![0.0; rows_w_len];
    for weight in weights.iter_mut().take(cols_x_len + 1) {
        *weight = between.sample(&mut rng);
    }
    weights
}

fn predict_linear_model_classification(model_weights: &[f32], inputs: &[f32]) -> i32 {
    let mut res = 0.0;
    for (&weight, &input) in model_weights.iter().skip(1).zip(inputs.iter()) {
        res += weight * input;
    }
    let total_sum = model_weights[0] + res;

    match total_sum {
        sum if sum > 0.5 => 1,
        sum if sum > -0.5 => 0,
        _ => 2,
    }
}

fn train_linear_model(x: &[Vec<f32>], y: &[i32], w: &mut [f32], rows_x_len: usize, rows_w_len: usize, iter: usize) -> f32 {
    let learning_rate = 0.001;
    let mut last_cost = f32::MAX;

    for _ in 0..iter {
        let k = rand::thread_rng().gen_range(0..rows_x_len);
        let gxk = predict_linear_model_classification(w, &x[k]);
        let yk = y[k];
        let diff = yk as f32 - gxk as f32;

        w[0] = w[0] + learning_rate * diff;
        for j in 1..rows_w_len {
            w[j] = w[j] + learning_rate * diff * x[k][j - 1];
        }

        // Calcul de l'erreur quadratique moyenne
        let cost = x.iter().zip(y.iter())
            .map(|(features, &label)| {
                let prediction = predict_linear_model_classification(w, features);
                (label as f32 - prediction as f32).powi(2)
            })
            .sum::<f32>() / x.len() as f32;

        // Vérifier la convergence
        if (last_cost - cost).abs() < 1e-5 {
            break;
        }
        last_cost = cost;
    }

    last_cost
}

fn save_model_linear(model_weight: &[f32], file_path: &Path, efficiency: f64) -> Result<(), io::Error> {
    let mut file = File::create(file_path)?;
    writeln!(file, "-- Efficiency --\n{}", efficiency)?;
    writeln!(file, "-- W --")?;
    for &weight in model_weight {
        writeln!(file, "{{{}}}", weight)?;
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let base_training_path = Path::new("images/Unknown");
    let base_testing_path = Path::new("images/Test");
    let categories = ["Avocado", "Tomato", "Banana"];

    let (train_features, train_labels) = load_image_data(base_training_path, &categories)?;
    let (test_features, test_labels) = load_image_data(base_testing_path, &categories)?;

    let rows_x_len = train_features.len();
    let rows_w_len = train_features[0].len();
    let mut weights = init_model_weights(rows_x_len, rows_w_len);

    // Entraînement du modèle avec contrôle de convergence
    let final_cost = train_linear_model(&train_features, &train_labels, &mut weights, rows_x_len, rows_w_len, 1000);
    println!("Erreur finale : {}", final_cost);

    // Test du modèle et calcul de la performance
    let mut final_result = 0;
    for (x, &y) in test_features.iter().zip(test_labels.iter()) {
        let result = predict_linear_model_classification(&weights, x);
        if result == y {
            final_result += 1;
        }
    }
    let performance = final_result as f32 / test_labels.len() as f32 * 100.0;
    println!("Performance: {}%", performance);

    // Sauvegarde des poids du modèle si la performance est satisfaisante
    let file_path = Path::new("model_weights.txt");
    save_model_linear(&weights, file_path, performance as f64)?;

    // Ajout de la prédiction pour une image spécifique
    let image_path = Path::new("images/banane.jpg");
    let image_features = process_image(image_path)?;
    let prediction = predict_linear_model_classification(&weights, &image_features);

    // Afficher la catégorie prédite
    let category = match prediction {
        1 => "Avocado",
        2 => "Tomato",
        3 => "Banana",
        _ => "Inconnu",
    };
    println!("Catégorie prédite: {}", category);

    Ok(())
}