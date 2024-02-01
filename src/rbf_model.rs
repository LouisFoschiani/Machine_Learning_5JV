use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use image::{self, GenericImageView};
use rand::distributions::{Distribution, Uniform};
use rand::{seq::SliceRandom, thread_rng, Rng};
use plotters::prelude::*;

const IMAGE_SIZE: usize = 16 * 16;

struct RBFN {
    centers: Vec<Vec<f32>>,
    weights: Vec<f32>,
    beta: f32,
}

impl RBFN {
    fn new(num_centers: usize, input_dimension: usize) -> Self {
        let mut rng = thread_rng();
        let centers: Vec<Vec<f32>> = vec![vec![0.0; input_dimension]; num_centers];
        let weights: Vec<f32> = vec![0.0; num_centers + 1]; // +1 for bias
        let beta = 1.0 / input_dimension as f32; // Un exemple simple pour initialiser beta

        RBFN { centers, weights, beta }
    }
    fn copy_weights_from(&mut self, source_weights: &[f32]) {
        self.weights.copy_from_slice(source_weights);
    }

    fn copy_weights_from_rbfn(&mut self, source_rbfn: &RBFN) {
        self.weights.copy_from_slice(&source_rbfn.weights);
    }



    fn train(&mut self, x: &[Vec<f32>], y: &[i32], epochs: usize, learning_rate: f32) {
        let mut rng = thread_rng();

        // Initialisation des centres à des échantillons aléatoires
        let mut sample_indices: Vec<usize> = (0..x.len()).collect();
        sample_indices.shuffle(&mut rng);
        for i in 0..self.centers.len() {
            self.centers[i] = x[sample_indices[i]].clone();
        }

        for epoch in 0..epochs {
            for (input, &label) in x.iter().zip(y.iter()) {
                let output = self.predict(input);
                let error = label as f32 - output;

                // Mise à jour du poids de biais
                self.weights[0] += learning_rate * error;

                // Mise à jour des poids des RBF
                for i in 0..self.centers.len() {
                    let rbf_output = self.radial_basis_function(input, &self.centers[i]);
                    self.weights[i + 1] += learning_rate * error * rbf_output;
                }
            }
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


    fn load_image_data(
        &self,
        base_path: &Path,
        target_category: &str,
        non_target_category: &Vec<&str>,
    ) -> io::Result<(Vec<Vec<f32>>, Vec<i32>)> {
        let mut features = Vec::new();
        let mut labels = Vec::new();

        // Load target category images
        let target_path = base_path.join(target_category);
        match fs::read_dir(&target_path) {
            Ok(entries) => {
                for entry in entries {
                    let entry = match entry {
                        Ok(e) => e,
                        Err(e) => {
                            continue;
                        }
                    };
                    let path = entry.path();
                    if path.is_file() {
                        match RBFN::process_image(&path) {
                            Ok(image_features) => {
                                features.push(image_features);
                                labels.push(1);
                            }
                            Err(_) => {
                                continue;
                            }
                        }
                    }
                }
            }
            Err(e) => {
                return Err(e);
            }
        }

        // Load non-target category images
        for category in non_target_category.iter() {
            let non_target_path = base_path.join(category);
            match fs::read_dir(&non_target_path) {
                Ok(entries) => {
                    for entry in entries {
                        let entry = match entry {
                            Ok(e) => e,
                            Err(e) => {
                                eprintln!("Erreur lors de la lecture d'une entrée dans le répertoire non cible : {}", e);
                                continue;
                            }
                        };
                        let path = entry.path();
                        if path.is_file() {
                            match RBFN::process_image(&path) {
                                Ok(image_features) => {
                                    features.push(image_features);
                                    labels.push(-1);
                                }
                                Err(e) => {
                                    eprintln!("Erreur lors du traitement de l'image : {:?}, erreur : {}", path, e);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Erreur lors de la lecture du répertoire non cible : {}", e);
                    return Err(e);
                }
            }
        }

        Ok((features, labels))
    }

    fn process_image(path: &Path) -> io::Result<Vec<f32>> {
        let img = image::open(path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let resized_img = img.resize_exact(16, 16, image::imageops::FilterType::Nearest);
        let features = resized_img
            .pixels()
            .flat_map(|(_, _, pixel)| pixel.0.to_vec())
            .map(|value| value as f32 / 255.0)
            .collect();
        Ok(features)
    }

    fn init_model_weights(&self, cols_x_len: usize, rows_w_len: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(-1.0..1.0);
        let mut weights = vec![0.0; rows_w_len];
        for weight in weights.iter_mut().take(cols_x_len + 1) {
            *weight = between.sample(&mut rng);
        }
        weights
    }

    fn save_model(&self, file_path: &Path) -> io::Result<()> {
        let mut file = File::create(file_path)?;

        writeln!(file, "-- Centers --")?;
        for center in &self.centers {
            let center_str = center.iter().map(|c| c.to_string()).collect::<Vec<String>>().join(",");
            writeln!(file, "{}", center_str)?;
        }

        writeln!(file, "-- Weights --")?;
        for weight in &self.weights {
            writeln!(file, "{}", weight)?;
        }

        Ok(())
    }

    fn load_model(&mut self, file_path: &Path) -> io::Result<()> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut line_iter = reader.lines().filter_map(|line| line.ok());

        while let Some(line) = line_iter.next() {
            if line.starts_with("-- Centers --") {
                for center in self.centers.iter_mut() {
                    if let Some(center_line) = line_iter.next() {
                        let values: Vec<f32> = center_line.split(',')
                            .map(|s| s.trim().parse().unwrap_or(0.0))
                            .collect();
                        *center = values;
                    }
                }
            } else if line.starts_with("-- Weights --") {
                for weight in self.weights.iter_mut() {
                    if let Some(weight_line) = line_iter.next() {
                        *weight = weight_line.trim().parse().unwrap_or(0.0);
                    }
                }
            }
        }

        Ok(())
    }

    fn evaluate(&self, features: &[Vec<f32>], labels: &[i32]) -> f32 {
        let mut correct_predictions = 0;
        for (feature, &label) in features.iter().zip(labels.iter()) {
            let prediction = if self.predict(feature) >= 0.0 { 1 } else { -1 };
            if prediction == label {
                correct_predictions += 1;
            }
        }
        correct_predictions as f32 / labels.len() as f32
    }
}

    fn set_var(x: &Vec<Vec<f32>>) -> (i32, i32, i32) {
        let rows_x_len = x.len();
        let cols_x_len = if !x.is_empty() { x[0].len() } else { 0 };
        let rows_w_len = cols_x_len + 1;
        (rows_x_len as i32, cols_x_len as i32, rows_w_len as i32)
    }

pub(crate) fn main() -> io::Result<()> {
    let mode = "train"; // Changez "train" ou "test" selon le mode souhaité

    match mode {
        "train" => {
            train_model()?;
        }
        "test" => {
            test_model()?
        }
        _ => {
            println!("Invalid mode. Please choose 'train' or 'test'.");
        }
    }

    Ok(())
}

fn train_model() -> io::Result<()> {
    let base_training_path = Path::new("images_16/Training");
    let target_categories = vec!["Tomato", "Orange", "Aubergine"];
    let num_centers = 5;
    let input_dimension = IMAGE_SIZE; // Assurez-vous que cela correspond à la taille de l'image après traitement
    let beta = 1; // Ajustez si nécessaire
    let num_epochs = 200;
    let learning_rate = 0.001;

    for target_category in &target_categories {
        let mut rbfn = RBFN::new(num_centers, input_dimension);
        // Utilisez directement `x` et `target_category` sans déréférencement supplémentaire
        let non_target_categories: Vec<&str> = target_categories.iter().filter(|&x| x != target_category).cloned().collect();

        println!("Training for category: {}", target_category);
        let (features, labels) = rbfn.load_image_data(&base_training_path, target_category, &non_target_categories)?;

        if features.is_empty() {
            println!("No training data found for {}", target_category);
            continue;
        }

        rbfn.train(&features, &labels, num_epochs, learning_rate);

        // Évaluez le modèle sur les données d'entraînement
        let accuracy = rbfn.evaluate(&features, &labels);
        println!("Training accuracy for {}: {:.2}%", target_category, accuracy * 100.0);

        let model_file_path = format!("rbfn_model_weights_{}.txt", target_category);
        rbfn.save_model(Path::new(&model_file_path))?;
    }

    Ok(())
}


fn test_model() -> io::Result<()> {
    let base_test_path = Path::new("images_16\\Test");
    let target_categories = vec!["Tomato", "Orange", "Aubergine"];
    let input_dimension = IMAGE_SIZE; // Assurez-vous que cela correspond à la taille de l'image après traitement

    for target_category in &target_categories {
        let model_file_path = format!("rbfn_model_weights_{}.txt", target_category);
        let mut rbfn = RBFN::new(0, input_dimension); // Temporairement initialisé avec 0 centres, sera chargé du fichier
        rbfn.load_model(Path::new(&model_file_path))?;

        let non_target_categories: Vec<&str> = target_categories.iter().filter(|&&x| x != *target_category).map(|&x| x).collect();

        let (test_features, test_labels) = rbfn.load_image_data(&base_test_path, target_category, &non_target_categories)?;
        if test_features.is_empty() {
            println!("No test data found for {}", target_category);
            continue;
        }

        // Evaluate model on test data
        let accuracy = rbfn.evaluate(&test_features, &test_labels);
        println!("Test accuracy for {}: {:.2}%", target_category, accuracy * 100.0);
    }

    Ok(())
}
