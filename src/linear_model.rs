extern crate ndarray;
extern crate image;

use ndarray::{Array, Array1, Array2};
use image::{DynamicImage, GenericImageView, ImageError, open};
use std::fs;
use std::path::Path;
use rand::Rng;
use std::fmt;

// Structure du modèle linéaire
pub struct LinearModel {
    weights: Array1<f32>,
}


#[derive(Debug, PartialEq)]
pub enum ImageClass {
    Avocado,
    Tomato,
    Banana,
}


impl fmt::Display for ImageClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ImageClass::Avocado => write!(f, "Avocado"),
            ImageClass::Tomato => write!(f, "Tomato"),
            ImageClass::Banana => write!(f, "Banana"),
        }
    }
}


impl LinearModel {
    pub fn new(input_size: usize) -> Self {
        // Initialisez les poids aléatoirement
        let mut rng = rand::thread_rng();
        let weights = Array::from_shape_fn(input_size, |_| rng.gen());

        Self { weights }
    }

    pub fn train(&mut self, features: &Array2<f32>, labels: &Array1<f32>, learning_rate: f32, num_iterations: usize) {
        for _ in 0..num_iterations {
            let predictions = self.predict(features);
            let errors = labels - &predictions;

            let gradient = -&features.t().dot(&errors);

            for (weight, grad) in self.weights.iter_mut().zip(gradient.iter()) {
                *weight -= learning_rate * grad;
            }
        }
    }


    pub fn predict(&self, features: &Array2<f32>) -> Array1<f32> {
        features.dot(&self.weights)
    }
}

pub fn main() -> Result<(), ImageError> {
    let training_data = load_images_from_directory("images/Training")?;
    let test_data = load_images_from_directory("images/Test")?;
    let unknown_data = load_images_from_directory("images/Unknown")?;

    // Définissez les caractéristiques (features) et les étiquettes (labels) à partir des données d'entraînement.
    let mut features = Array::zeros((training_data.len(), 3)); // 3 features (R, G, B)
    let mut labels = Array1::zeros(training_data.len());

    for (i, (image, label)) in training_data.iter().enumerate() {
        let pixels = to_normalized_vec(image.clone());
        features.row_mut(i).assign(&Array1::from(pixels));
        labels[i] = match label {
            ImageClass::Avocado => 1.0,
            ImageClass::Tomato => 2.0,
            ImageClass::Banana => 3.0,
            // Ajoutez d'autres classes au besoin
        };
    }

    // Créez et entraînez le modèle linéaire.
    let input_size = features.shape()[1];
    let mut model = LinearModel::new(input_size);
    model.train(&features, &labels, 0.01, 1000);

    // Testez le modèle sur des données de test.
    let mut correct_predictions = 0;
    let mut problematic_images = Vec::new();
    for (image, label) in test_data.iter() {
        let pixels = to_normalized_vec(image.clone());
        if pixels.is_empty() {
            problematic_images.push(label.to_string());
            continue;  // Ignorez cette image problématique
        }
        let features = Array2::from_shape_vec((1, pixels.len()), pixels).unwrap();
        let prediction = model.predict(&features);
        let predicted_class_label = if prediction[0] > 0.5 { ImageClass::Avocado } else { ImageClass::Banana }; // Binaire

        if predicted_class_label == *label {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f32 / test_data.len() as f32;
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    if !problematic_images.is_empty() {
        println!("Problematic images: {:?}", problematic_images);
    }

    println!("Accuracy: {:.2}%", accuracy * 100.0);

    // Classification des images inconnues depuis le répertoire "images/Unknown"
    for (image, label) in unknown_data.iter() {
        let pixels = to_normalized_vec(image.clone());
        if pixels.is_empty() {
            problematic_images.push(label.to_string());
            continue;  // Ignorez cette image problématique
        }
        let features = Array2::from_shape_vec((1, pixels.len()), pixels).unwrap();
        let prediction = model.predict(&features);
        let predicted_class_label = if prediction[0] > 0.5 { ImageClass::Avocado } else { ImageClass::Banana }; // Binaire

        let predicted_class_str = match predicted_class_label {
            ImageClass::Avocado => "Avocado",
            ImageClass::Tomato => "Tomato",
            ImageClass::Banana => "Banana",
            // Ajoutez d'autres classes au besoin
        };

        println!("Predicted class: {} (True class: {:?})", predicted_class_str, label);
    }

    Ok(())
}
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

fn is_valid_jpeg(image_path: &Path) -> bool {
    use image::io::Reader;
    if let Ok(reader) = Reader::open(image_path) {
        if let Ok(_) = reader.with_guessed_format() {
            return true;
        }
    }
    false
}

fn to_normalized_vec(image: DynamicImage) -> Vec<f32> {
    image.pixels().flat_map(|pixel| {
        let pixel = pixel.2; // Extraction du pixel Rgba<u8>
        let (r, g, b) = (
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        );
        vec![r, g, b]
    }).collect()
}
