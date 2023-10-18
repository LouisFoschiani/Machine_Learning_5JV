#![allow(non_snake_case)]

use std::fs;
use std::path::Path;
use image::{DynamicImage, ImageError, open};
use rand::Rng;

#[derive(Debug)]
pub struct ImageClassifier {
    input_size: u32,
    output_size: u32,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    acc_gradients: Vec<Vec<f32>>,  // gradients accumulés pour les poids
    acc_biases: Vec<f32>,  // gradients accumulés pour les biais
    learning_rate: f32,

}

impl ImageClassifier {
    pub fn new(input_size: u32, output_size: u32, learning_rate: f32) -> Self {
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

    pub fn reset_gradients(&mut self) {
        for i in 0..self.output_size as usize {
            self.acc_biases[i] = 0.0;
            for j in 0..self.input_size as usize {
                self.acc_gradients[i][j] = 0.0;
            }
        }
    }

    fn cross_entropy_loss(prediction: &Vec<f32>, label: u8) -> f32 {
        let true_label_prob = prediction[label as usize];
        -true_label_prob.ln()
    }

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

    pub fn backpropagate(&mut self, prediction: Vec<f32>, label: u8, pixels: &[f32]) {
        let loss = Self::cross_entropy_loss(&prediction, label);
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

    pub fn update_weights_biases(&mut self) {
        for i in 0..self.output_size as usize {
            self.biases[i] -= self.learning_rate * self.acc_biases[i];
            for j in 0..self.input_size as usize {
                self.weights[i][j] -= self.learning_rate * self.acc_gradients[i][j];
            }
        }
    }
}

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

fn load_images_from_directory(directory_path: &str) -> Result<Vec<DynamicImage>, ImageError> {
    let mut images = Vec::new();
    let paths = fs::read_dir(directory_path)?;

    for path in paths {
        let entry = path?;
        let file_path = entry.path();
        if file_path.is_file() {
            let image = open(file_path)?;
            images.push(image);
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

fn classify_new_image(classifier: &mut ImageClassifier, image_path: &str) -> Result<u8, ImageError> {
    let image = open(image_path)?;
    let output = classifier.forward(&image);
    let predicted_label = output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u8;
    Ok(predicted_label)
}
fn main() -> Result<(), ImageError> {
    let mut classifier = ImageClassifier::new(28, 3, 0.01);

    let data_football = load_images_from_directory("C:\\Users\\Louis\\Documents\\GitHub\\machine_Learning_5JV\\images\\Avocado")?;
    let data_volleyball = load_images_from_directory("C:\\Users\\Louis\\Documents\\GitHub\\machine_Learning_5JV\\images\\Blueberry")?;
    let data_football_americain = load_images_from_directory("C:\\Users\\Louis\\Documents\\GitHub\\machine_Learning_5JV\\images\\Banana")?;


    let batch_size = 3;
    let mut current_batch = Vec::with_capacity(batch_size);

    for epoch in 0..100{

        // ... exemple d'entraînement avec les données de football ...
        for image in data_football.iter() {
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

        for image in data_volleyball.iter() {
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

        for image in data_football_americain.iter() {
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


    let image_path = "C:\\Users\\Louis\\Documents\\GitHub\\machine_Learning_5JV\\images\\avocat.jpg";
    match classify_new_image(&mut classifier, image_path) {
        Ok(label) => println!("Predicted label: {}", label),
        Err(e) => eprintln!("Failed to classify image: {}", e),
    }

    Ok(())
}
