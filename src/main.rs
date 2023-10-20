#![allow(non_snake_case)]
#[warn(unused_variables)]

extern crate serde;

// Importer les bibliothèques nécessaires.
use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;
use image::{DynamicImage, ImageError, open};
use rand::Rng;

// Définir une structure pour le classificateur d'images.
#[derive(Debug, Deserialize, Serialize)]
pub struct ImageClassifier {
    input_size: u32,
    output_size: u32,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    acc_gradients: Vec<Vec<f32>>,  // Gradients accumulés pour les poids.
    acc_biases: Vec<f32>,  // Gradients accumulés pour les biais.
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
            acc_gradients: vec![vec![0.0; input_size as usize]; output_size as usize],
            acc_biases: vec![0.0; output_size as usize],
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
    pub fn load_weights(&mut self, filename: &str) -> Result<(), std::io::Error> {
        let serialized = fs::read_to_string(filename)?;
        *self = serde_json::from_str(&serialized).unwrap();
        Ok(())
    }

    // Méthode pour réinitialiser les gradients accumulés.
    pub fn reset_gradients(&mut self) {
        for i in 0..self.output_size as usize {
            self.acc_biases[i] = 0.0;
            for j in 0..self.input_size as usize {
                self.acc_gradients[i][j] = 0.0;
            }
        }
    }

    // Fonction pour calculer la perte d'entropie croisée.
    fn cross_entropy_loss(prediction: &Vec<f32>, label: u8) -> f32 {
        let true_label_prob = prediction[label as usize];
        -true_label_prob.ln()
    }

    // Méthode pour effectuer la propagation avant (forward pass) à travers le modèle.
    pub fn forward(&mut self, image: &DynamicImage) -> Vec<f32> {
        let pixels = to_vec(image.clone());
        let mut output = vec![0.0; self.output_size as usize];
        for i in 0..self.output_size as usize {
            for k in 0..self.input_size as usize {
                output[i] += self.weights[i][k] * pixels[k];
            }
            output[i] += self.biases[i];
        }
        output
    }

    // Méthode pour effectuer la rétropropagation (backpropagation) et mettre à jour les gradients accumulés.
    pub fn backpropagate(&mut self, prediction: Vec<f32>, label: u8, pixels: &[f32]) {
        let _loss = Self::cross_entropy_loss(&prediction, label);
        let mut loss_gradients: Vec<f32> = vec![0.0; self.output_size as usize];

        for i in 0..self.output_size as usize {
            let gradient = if i == label as usize {
                prediction[i] - 1.0
            } else {
                prediction[i]
            };
            loss_gradients.push(gradient);
        }

        for i in 0..self.output_size as usize {
            self.acc_biases[i] += loss_gradients[i];
            for j in 0..self.input_size as usize {
                self.acc_gradients[i][j] += loss_gradients[i] * pixels[j];
            }
        }
    }

    // Méthode pour mettre à jour les poids et les biais du modèle.
    pub fn update_weights_biases(&mut self) {
        for i in 0..self.output_size as usize {
            self.biases[i] -= self.learning_rate * self.acc_biases[i];
            for j in 0..self.input_size as usize {
                self.weights[i][j] -= self.learning_rate * self.acc_gradients[i][j];
            }
        }
    }
}

// Fonction pour convertir une image en un vecteur de pixels normalisés.
fn to_vec(image: DynamicImage) -> Vec<f32> {
    let mut pixels = Vec::new();
    let image = image.to_rgba8();
    for pixel in image.pixels() {
        pixels.push(pixel[0] as f32 / 255.0);
        pixels.push(pixel[1] as f32 / 255.0);
        pixels.push(pixel[2] as f32 / 255.0);
        pixels.push(pixel[3] as f32 / 255.0);
    }
    pixels
}

// Fonction pour charger des images à partir d'un répertoire.
fn load_images_from_directory(directory_path: &str) -> Result<Vec<DynamicImage>, ImageError> {
    let mut images = Vec::new();
    let paths = fs::read_dir(directory_path)?;

    for path in paths {
        let entry = path?;
        let file_path = entry.path();
        if file_path.is_file() && is_valid_jpeg(&file_path) {
            let image = open(file_path)?;
            images.push(image);
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
fn classify_new_image(classifier: &mut ImageClassifier, image_path: &str) -> Result<u8, ImageError> {
    let image = open(image_path)?;
    let output = classifier.forward(&image);
    let predicted_label = output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u8;
    Ok(predicted_label)
}

fn main() -> Result<(), ImageError> {
    let mut classifier = ImageClassifier::new(28 * 28, 3, 0.01); // L'input_size devrait correspondre à la taille de l'image.

    // Sauvegardez les poids du modèle.
    classifier.save_weights("model_weights.json").unwrap();

    // Chargez les poids du modèle depuis le fichier JSON.
    classifier.load_weights("model_weights.json").unwrap();

    // Chargez les données d'entraînement à partir des répertoires.
    let data_avocat = load_images_from_directory("images\\Avocado")?;
    let data_blueberry = load_images_from_directory("images\\Blueberry")?;
    let data_banane = load_images_from_directory("images\\Banana")?;

    let batch_size = 3;
    let mut current_batch = Vec::with_capacity(batch_size);

    for _epoch in 0..50 {
        for image in data_avocat.iter() {
            let pixels = to_vec(image.clone());
            let prediction = classifier.forward(image);
            classifier.backpropagate(prediction, 0, &pixels);
            current_batch.push(image);

            if current_batch.len() == batch_size {
                classifier.update_weights_biases();
                classifier.reset_gradients();
                current_batch.clear();
            }
        }

        for image in data_blueberry.iter() {
            let pixels = to_vec(image.clone());
            let prediction = classifier.forward(image);
            classifier.backpropagate(prediction, 1, &pixels);
            current_batch.push(image);

            if current_batch.len() == batch_size {
                classifier.update_weights_biases();
                classifier.reset_gradients();
                current_batch.clear();
            }
        }

        for image in data_banane.iter() {
            let pixels = to_vec(image.clone());
            let prediction = classifier.forward(image);
            classifier.backpropagate(prediction, 2, &pixels);
            current_batch.push(image);

            if current_batch.len() == batch_size {
                classifier.update_weights_biases();
                classifier.reset_gradients();
                current_batch.clear();
            }
        }
    }

    // Classifier une nouvelle image et afficher l'étiquette prédite.
    let image_path = "images\\avocat.jpg";
    match classify_new_image(&mut classifier, image_path) {
        Ok(label) => println!("Predicted label: {}", label),
        Err(e) => eprintln!("Failed to classify image: {}", e),
    }

    Ok(())
}
