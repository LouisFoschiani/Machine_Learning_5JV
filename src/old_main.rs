#![allow(non_snake_case)]
#[warn(unused_variables)]

extern crate serde;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;
use image::{DynamicImage, ImageError, open};
use rand::prelude::SliceRandom;
use rand::Rng;

// Utilisez un enum pour représenter les classes
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)] // Ajoutez PartialEq ici
pub enum ImageClass {
    Avocado,
    Tomato,
    Banana,
}

// Définir une structure pour le classificateur d'images.
#[derive(Debug, Deserialize, Serialize)]
pub struct ImageClassifier {
    input_size: u32,
    output_size: u32,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    learning_rate: f32,
}

impl ImageClassifier {
    // Méthode pour créer une nouvelle instance du classificateur d'images.
    pub fn new(input_size: u32, output_size: u32, learning_rate: f32) -> Self {
        // Génération de poids initiaux aléatoires et initialisation des autres valeurs.
        let mut rng = rand::thread_rng();
        let weights = vec![vec![rng.gen::<f32>(); input_size as usize]; output_size as usize];
        let biases = vec![0.0; output_size as usize];

        Self {
            input_size,
            output_size,
            weights,
            biases,
            learning_rate,
        }
    }

    // Méthode pour sauvegarder les poids du modèle dans un fichier.
    pub fn save_weights(&self, filename: &str) -> Result<(), std::io::Error> {
        // Utilisez le format JSON pour la sérialisation.
        let serialized = serde_json::to_string(self).unwrap();
        fs::write(filename, &serialized)
    }

    // Méthode pour charger les poids du modèle depuis un fichier.
    pub fn load_weights(filename: &str) -> Result<Self, std::io::Error> {
        let serialized = fs::read_to_string(filename)?;
        let classifier: ImageClassifier = serde_json::from_str(&serialized)?;

        Ok(classifier)
    }

    // Méthode pour réinitialiser les gradients accumulés.
    pub fn reset_gradients(&mut self) {
        self.biases.iter_mut().for_each(|bias| *bias = 0.0);
        self.weights
            .iter_mut()
            .for_each(|row| row.iter_mut().for_each(|w| *w = 0.0));
    }

    // Fonction pour calculer la perte d'entropie croisée.
    fn cross_entropy_loss(prediction: &[f32], label: &ImageClass) -> f32 {
        let true_label_prob = match label {
            ImageClass::Avocado => prediction[0],
            ImageClass::Tomato => prediction[1],
            ImageClass::Banana => prediction[2],
            // Ajoutez d'autres classes au besoin
        };
        -true_label_prob.ln()
    }

    // Méthode pour effectuer la propagation avant (forward pass) à travers le modèle.
    pub fn forward(&self, image: &DynamicImage) -> Vec<f32> {
        let pixels = to_vec(image.clone());
        let mut output = vec![0.0; self.output_size as usize];
        for i in 0..self.output_size as usize {
            output[i] = self.biases[i]
                + self.weights[i]
                .iter()
                .zip(pixels.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
        }
        output
    }

    // Méthode pour effectuer la rétropropagation (backpropagation) et mettre à jour les gradients accumulés.
    pub fn backpropagate(&mut self, prediction: &[f32], label: &ImageClass, pixels: &[f32]) {
        let loss_gradients: Vec<f32> = prediction
            .iter()
            .enumerate()
            .map(|(i, &prob)| {
                if i == *label as usize {
                    prob - 1.0
                } else {
                    prob
                }
            })
            .collect();

        for i in 0..self.output_size as usize {
            self.biases[i] += loss_gradients[i];
            for j in 0..self.input_size as usize {
                self.weights[i][j] += loss_gradients[i] * pixels[j];
            }
        }
    }

    // Méthode pour mettre à jour les poids et les biais du modèle.
    pub fn update_weights_biases(&mut self, batch_size: usize) {
        let learning_rate = self.learning_rate / batch_size as f32;
        for i in 0..self.output_size as usize {
            self.biases[i] -= learning_rate * self.biases[i];
            for j in 0..self.input_size as usize {
                self.weights[i][j] -= learning_rate * self.weights[i][j];
            }
        }
    }
}

// Fonction pour convertir une image en un vecteur de pixels normalisés.
fn to_vec(image: DynamicImage) -> Vec<f32> {
    let image = image.to_rgba8();
    image
        .pixels()
        .flat_map(|pixel| {
            let pixel_value = pixel[0] as f32 / 255.0;
            vec![pixel_value, pixel_value, pixel_value, pixel_value]
        })
        .collect()
}

// Fonction pour charger des images à partir d'un répertoire.
fn load_images_from_directory(directory_path: &str) -> Result<Vec<(DynamicImage, ImageClass)>, ImageError> {
    let mut images = Vec::new();
    let paths = fs::read_dir(directory_path)?;

    for path in paths {
        let entry = path?;
        let file_path = entry.path();
        if file_path.is_file() && is_valid_jpeg(&file_path) {
            let image = open(file_path.clone())?;
            let image_class = match file_path.parent() {
                Some(parent) => match parent.file_name().and_then(|s| s.to_str()) {
                    Some("Avocado") => ImageClass::Avocado,
                    Some("Tomato") => ImageClass::Tomato,
                    Some("Banana") => ImageClass::Banana,
                    // Ajoutez d'autres classes au besoin
                    _ => ImageClass::Avocado, // Par défaut
                },
                _ => ImageClass::Avocado, // Par défaut
            };
            images.push((image, image_class));
        }
    }

    Ok(images)
}

// Fonction pour vérifier si un fichier est un JPEG valide.
fn is_valid_jpeg(image_path: &Path) -> bool {
    use image::io::Reader;

    if let Ok(reader) = Reader::open(image_path) {
        if let Ok(_) = reader.with_guessed_format() {
            return true;
        }
    }
    false
}

// Fonction pour classifier une nouvelle image.
fn classify_new_image(classifier: &ImageClassifier, image: &DynamicImage) -> Result<ImageClass, ImageError> {
    let output = classifier.forward(image);
    let predicted_class = match output.iter().enumerate().max_by(|a, b| {
        a.1.partial_cmp(b.1)
            .unwrap_or(Ordering::Equal)
    }) {
        Some((idx, _)) => match idx {
            0 => ImageClass::Avocado,
            1 => ImageClass::Tomato,
            2 => ImageClass::Banana,
            // Ajoutez d'autres classes au besoin
            _ => ImageClass::Avocado, // Par défaut
        },
        None => ImageClass::Avocado, // Par défaut
    };

    Ok(predicted_class)
}


// Fonction pour classifier des images inconnues
fn classify_unknown_images(classifier: &ImageClassifier, directory_path: &str) -> Result<(), ImageError> {
    let unknown_images = load_images_from_directory(directory_path)?;

    for (image, _) in unknown_images.iter() {
        match classify_new_image(&classifier, &image) {
            Ok(predicted_class) => {
                println!("Predicted class: {:?}", predicted_class);
            }
            Err(e) => {
                eprintln!("Failed to classify image: {}", e);
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), ImageError> {
    let mut classifier = ImageClassifier::new(28 * 28 * 4, 3, 0.01);

    let training_data = load_images_from_directory("images\\Training")?;
    let batch_size = 32;

    for _epoch in 0..300 {
        let mut current_batch = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();

        for (image, label) in training_data.iter() {
            let image = image.clone(); // Clonez chaque image
            let pixels = to_vec(image.clone());
            let prediction = classifier.forward(&image);
            classifier.backpropagate(&prediction, label, &pixels);
            current_batch.push((image, label));

            if current_batch.len() == batch_size {
                classifier.update_weights_biases(batch_size);
                classifier.reset_gradients();
                current_batch.clear();
            }
        }
    }

    let test_data = load_images_from_directory("images\\Test")?;
    let mut correct_predictions = 0;

    for (image, label) in test_data.iter() {
        match classify_new_image(&classifier, &image) {
            Ok(predicted_class) if predicted_class == *label => correct_predictions += 1,
            _ => continue,
        }
    }

    let accuracy = correct_predictions as f32 / test_data.len() as f32;
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    classifier.save_weights("model_weights.json").unwrap();

    // Classification des images inconnues depuis le répertoire "images\\Unknown"
    classify_unknown_images(&classifier, "images\\Unknown").unwrap();

    Ok(())
}
