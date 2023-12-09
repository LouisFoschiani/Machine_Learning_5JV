#[warn(non_snake_case)]

use image::{self, GenericImageView};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path};

fn load_image_data(base_path: &Path, target_category: &str, non_target_category: &str) -> io::Result<(Vec<Vec<f32>>, Vec<i32>)> {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    // Load target category images
    let target_path = base_path.join(target_category);
    for entry in fs::read_dir(target_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let image_features = process_image(&path)?;
            features.push(image_features);
            labels.push(1); // Label 1 for target category
        }
    }

    // Load non-target category images
    let non_target_path = base_path.join(non_target_category);
    for entry in fs::read_dir(non_target_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let image_features = process_image(&path)?;
            features.push(image_features);
            labels.push(0); // Label 0 for non-target category
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

fn predict_linear_model_score(model_weights: &[f32], inputs: &[f32]) -> f32 {
    let mut res = 0.0;
    for (&weight, &input) in model_weights.iter().skip(1).zip(inputs.iter()) {
        res += weight * input;
    }
    model_weights[0] + res
}

fn train_linear_model(x: &[Vec<f32>], y: &[i32], w: &mut [f32], rows_x_len: usize, rows_w_len: usize, iter: usize) -> f32 {
    let learning_rate = 0.001;
    let mut last_cost = f32::MAX;

    for _ in 0..iter {
        let k = rand::thread_rng().gen_range(0..rows_x_len);
        let gxk = predict_linear_model_score(w, &x[k]);
        let yk = y[k];
        let diff = yk as f32 - gxk;

        w[0] = w[0] + learning_rate * diff;
        for j in 1..rows_w_len {
            w[j] = w[j] + learning_rate * diff * x[k][j - 1];
        }

        // Calculating mean squared error
        let cost = x.iter().zip(y.iter())
            .map(|(features, &label)| {
                let prediction_score = predict_linear_model_score(w, features);
                (label as f32 - prediction_score).powi(2)
            })
            .sum::<f32>() / x.len() as f32;

        // Check for convergence
        if cost > last_cost || (last_cost - cost).abs() < 1e-5 {
            break;
        }
        last_cost = cost;
    }

    last_cost
}

fn save_model_linear(model_weight: &[f32], file_path: &Path, efficiency: f64) -> Result<(), io::Error> {
    let mut file = File::create(file_path)?;
    writeln!(file, "-- Efficiency --\n{}", efficiency)?;
    writeln!(file, "-- Weights --")?;
    for &weight in model_weight {
        writeln!(file, "{{{}}}", weight)?;
    }
    Ok(())
}

fn load_model_weights(file_path: &Path) -> io::Result<(Vec<f32>, f64)> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut weights = Vec::new();
    let mut efficiency = 0.0;
    let mut read_efficiency = false;

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("-- Efficiency --") {
            read_efficiency = true;
            continue;
        }

        if read_efficiency {
            efficiency = line.parse::<f64>().unwrap_or_else(|_| 0.0);
            read_efficiency = false;
            continue;
        }

        if line.starts_with("{") && line.ends_with("}") {
            let weight_str = &line[1..line.len()-1];
            match weight_str.trim().parse::<f32>() {
                Ok(weight) => weights.push(weight),
                Err(_) => continue,
            }
        }
    }

    Ok((weights, efficiency))
}

fn main() -> io::Result<()> {
    let iterations = 10;
    let weights_file_path = Path::new("G:\\Machine_Learning_5JV\\model_weights_Banana.txt");

    let mut last_efficiency = if weights_file_path.exists() {
        let (_, efficiency) = load_model_weights(weights_file_path)?;
        efficiency
    } else {
        0.0
    };

    for iter in 0..iterations {
        println!("----------------------");
        println!("Iteration: {}", iter + 1);
        let base_training_path = Path::new("images/Training");
        let base_testing_path = Path::new("images/Test");
        let target_category = "Banana"; // Specify your target category
        let non_target_category = "NonBanana"; // Specify your non-target category

        let (train_features, train_labels) = load_image_data(base_training_path, target_category, non_target_category)?;
        let (test_features, test_labels) = load_image_data(base_testing_path, target_category, non_target_category)?;

        let rows_x_len = train_features.len();
        let rows_w_len = train_features[0].len();
        let mut weights = init_model_weights(rows_x_len, rows_w_len);

        let final_cost = train_linear_model(&train_features, &train_labels, &mut weights, rows_x_len, rows_w_len, 1000);
        println!("Final Cost: {}", final_cost);

        // Evaluate model using Mean Squared Error
        let total_error: f32 = test_features.iter().zip(test_labels.iter())
            .map(|(features, &label)| {
                let prediction_score = predict_linear_model_score(&weights, features);
                (label as f32 - prediction_score).powi(2)
            })
            .sum();
        let mean_squared_error = total_error / test_features.len() as f32;
        println!("Mean Squared Error: {}", mean_squared_error);

        if mean_squared_error < last_efficiency as f32 {
            save_model_linear(&weights, weights_file_path, mean_squared_error as f64)?;
            println!("Weights saved in model_weights_Banana.txt");
            last_efficiency = mean_squared_error as f64;
        }else{
            println!("NO SAVE {} : {}", mean_squared_error, last_efficiency);
        }
    }
    Ok(())
}
