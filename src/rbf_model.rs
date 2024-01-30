use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use image::{self, GenericImageView};
use rand::distributions::{Distribution, Uniform};
use rand::{seq::SliceRandom, thread_rng, Rng};
use plotters::prelude::*;

const IMAGE_SIZE: usize = 16 * 16; // Taille d'image ajustée
fn plot_errors(train_errors: &Vec<f32>, test_errors: &Vec<f32>, index: i32, target: String, nonTarget: String) -> Result<(), Box<dyn std::error::Error>> {

    let name = format!("index-{}.png", index);
    let title = format!("Training and Test Errors Over Iteration: {}/{}", target, nonTarget);
    let root = BitMapBackend::new(&name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..train_errors.len(), 0f32..1f32)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        train_errors.iter().enumerate().map(|(i, &err)| (i, err)),
        &RED,
    ))?.label("Training Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.draw_series(LineSeries::new(
        test_errors.iter().enumerate().map(|(i, &err)| (i, err)),
        &BLUE,
    ))?.label("Test Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels().border_style(&BLACK).draw()?;
    Ok(())
}


struct RBFN {
    centers: Vec<Vec<f32>>,
    weights: Vec<f32>,
    beta: f32,
}
impl RBFN {
    fn new(num_centers: usize, input_dimension: usize, beta: f32) -> Self {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(0.0..1.0);
        let centers = (0..num_centers).map(|_| (0..input_dimension).map(|_| between.sample(&mut rng)).collect()).collect();

        let weights = vec![0.0; num_centers + 1]; // +1 for bias

        RBFN { centers, weights, beta }
    }

    fn train(&mut self, x: &[Vec<f32>], y: &[i32], epochs: usize, learning_rate: f32) {
        let mut rng = thread_rng();
        for _ in 0..epochs {
            x.iter().zip(y.iter()).for_each(|(input, &label)| {
                let output = self.predict(input);
                let error = label as f32 - output;
                self.weights[0] += learning_rate * error; // Update bias weight

                for i in 0..self.centers.len() {
                    let rbf_output = self.radial_basis_function(input, &self.centers[i]);
                    self.weights[i + 1] += learning_rate * error * rbf_output; // Update RBF weights
                }
            });

            self.centers.iter_mut().for_each(|center| {
                let random_input = x.choose(&mut rng).expect("Non-empty input list");
                for (c, &input) in center.iter_mut().zip(random_input.iter()) {
                    *c += learning_rate * (input - *c); // Update center towards a random input sample
                }
            });
        }
    }

    fn predict(&self, input: &[f32]) -> f32 {
        let mut output = self.weights[0]; // Start with bias
        for (i, center) in self.centers.iter().enumerate() {
            let rbf_output = self.radial_basis_function(input, center);
            output += self.weights[i + 1] * rbf_output;
        }
        output
    }

    fn radial_basis_function(&self, input: &[f32], center: &[f32]) -> f32 {
        let squared_sum = input.iter().zip(center.iter()).map(|(&x, &c)| (x - c).powi(2)).sum::<f32>();
        (-self.beta * squared_sum).exp()
    }



    fn load_image_data(&self, base_path: &Path, target_category: &str, non_target_category: &str) -> io::Result<(Vec<Vec<f32>>, Vec<i32>)> {
        let mut features = Vec::new();
        let mut labels = Vec::new();

        // Load target category images
        let target_path = base_path.join(target_category);
        for entry in fs::read_dir(target_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let image_features = RBFN::process_image(&path)?;
                features.push(image_features);
                labels.push(1);
            }
        }

        // Load non-target category images
        let non_target_path = base_path.join(non_target_category);
        for entry in fs::read_dir(non_target_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let image_features = RBFN::process_image(&path)?;
                features.push(image_features);
                labels.push(-1);
            }
        }

        Ok((features, labels))
    }

    fn process_image(path: &Path) -> io::Result<Vec<f32>> {
        let img = image::open(path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        // Assurez-vous que l'image est redimensionnée à 16x16 si ce n'est pas déjà le cas.
        let resized_img = img.resize_exact(16, 16, image::imageops::FilterType::Nearest);
        let features = resized_img.pixels()
            .flat_map(|(_, _, pixel)| pixel.0.to_vec())
            .map(|value| value as f32 / 255.0)
            .collect();
        Ok(features)
    }


    fn init_model_weights(&self,cols_x_len: usize, rows_w_len: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(-1.0..1.0);
        let mut weights = vec![0.0; rows_w_len];
        for weight in weights.iter_mut().take(cols_x_len + 1) {
            *weight = between.sample(&mut rng);
        }
        weights
    }



    fn save_model_rbfn(model_weight: &[f32], file_path: &Path, efficiency: f32) -> Result<(), io::Error> {
        let mut file = File::create(file_path)?;
        writeln!(file, "-- Efficiency --\n{}", efficiency)?;
        writeln!(file, "-- Weights --")?;
        for &weight in model_weight {
            writeln!(file, "{{{}}}", weight)?;
        }
        Ok(())
    }
    fn evaluate_model(rbfn: &RBFN, features: &[Vec<f32>], labels: &[i32]) -> f32 {
        let mut correct_predictions = 0;
        for (feature, &label) in features.iter().zip(labels.iter()) {
            let prediction = if rbfn.predict(feature) >= 0.0 { 1 } else { -1 };
            if prediction == label {
                correct_predictions += 1;
            }
        }
        correct_predictions as f32 / labels.len() as f32
    }
    fn load_model_weights(&self,file_path: &Path, colX: i32, rowW: i32) -> io::Result<(Vec<f32>, f32)> {

        let mut weights = Vec::new();
        let mut efficiency = 0.0;

        if fs::metadata(file_path).is_ok() {

            let file = File::open(file_path)?;
            let reader = BufReader::new(file);
            let mut read_efficiency = false;

            for line in reader.lines() {
                let line = line?;
                if line.starts_with("-- Efficiency --") {
                    read_efficiency = true;
                    continue;
                }

                if read_efficiency {
                    efficiency = line.parse::<f32>().unwrap_or_else(|_| 0.0);
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
        } else {
            weights = self.init_model_weights(colX as usize, rowW as usize)
        }



        Ok((weights, efficiency))
    }

    fn set_var(x: &Vec<Vec<f32>>) -> (i32, i32, i32) {
        let rows_x_len = x.len();
        let cols_x_len = if !x.is_empty() { x[0].len() } else { 0 };
        let rows_w_len = cols_x_len + 1;
        (rows_x_len as i32, cols_x_len as i32, rows_w_len as i32)
    }
}



pub(crate) fn main() -> io::Result<()> {

    let mode = "test";

    match mode {
        "train" => {
            train_model()?;
        },
        "test" => {
            test_classification()?;
        },
        _ => {
            println!("Invalid mode. Please choose 'train' or 'test'.");
        }
    }

    Ok(())
}

fn train_model() -> io::Result<()> {
    println!("Training mode selected...");

    let iteration = 50;
    let base_training_path = Path::new("images_16/Training");
    let base_test_path = Path::new("images_16/Test");
    let target_list = vec!["Tomato", "Orange", "Aubergine"];
    let non_target_list = vec!["Orange", "Aubergine", "Tomato"];
    let num_centers = 5;
    let input_dimension = 256;
    let beta = 1.0;
    let num_epochs = 10;
    let learning_rate = 0.01;
    let mut rbfn = RBFN::new(num_centers, input_dimension, beta);

    for iter in 0..iteration { // Exemple avec 1 itération pour la simplicité
        println!("------- ITERATION {} -------", iter);
        for i in 0..target_list.len() {
            let (train_features, train_labels) = rbfn.load_image_data(&base_training_path, target_list[i], non_target_list[i])?;
            rbfn.train(&train_features, &train_labels, num_epochs, learning_rate);

            let training_accuracy = RBFN::evaluate_model(&rbfn, &train_features, &train_labels);
            println!("Training Accuracy for {}: {}%", target_list[i], training_accuracy * 100.0);

            let (test_features, test_labels) = rbfn.load_image_data(&base_test_path, target_list[i], non_target_list[i])?;

            let test_accuracy = RBFN::evaluate_model(&rbfn, &test_features, &test_labels);
            println!("Test Accuracy for {}: {}%", target_list[i], test_accuracy * 100.0);
        }
    }

    // Enregistrement du modèle
    let model_file_path = Path::new("rbfn_model_weights.txt");
    RBFN::save_model_rbfn(&rbfn.weights, model_file_path, 1.0);

    Ok(())
}

fn test_classification() -> io::Result<()> {
    println!("Test mode selected...");

    // Supposition : `load_model_weights` est correctement implémentée pour configurer `rbfn` avec les poids sauvegardés
    let model_file_path = Path::new("rbfn_model_weights.txt");
    let mut rbfn = RBFN::new(5, 256, 1.0); // Assurez-vous que ces valeurs correspondent à celles utilisées pour l'entraînement
    let (weights, efficiency) = rbfn.load_model_weights(model_file_path, 256, 6)?;
    rbfn.weights = weights; // Assurez-vous que votre structure RBFN peut accepter des poids chargés comme ceci

    // Chargement et traitement d'une image de test
    let test_image_path = Path::new("images_16/CHECK/Orange/orange.jpg"); // Mettez à jour le chemin selon besoin
    let test_features = RBFN::process_image(test_image_path)?;

    // Prédiction
    let prediction = rbfn.predict(&test_features);
    println!("Prediction for test image: {}", prediction);

    Ok(())
}
