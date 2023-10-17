#![allow(non_snake_case)]

use std::fs::File;
use std::path::Path;
use image::{DynamicImage, GenericImageView, ImageBuffer, open, Pixel, Rgb};
use rand::Rng;


#[derive(Debug)]
pub struct ImageClassifier {
    input_size: u32,
    output_size: u32,
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<f32>>,
    output: Vec<f32>,
    learning_rate: f32,
}

impl ImageClassifier {
    pub fn new(input_size: u32, output_size: u32, learning_rate: f32) -> Self {
        let mut weights = vec![vec![vec![]; input_size as usize]; output_size as usize];
        let mut biases = vec![vec![]; input_size as usize];

        for i in 0..output_size {
            for j in 0..input_size {
                biases[j as usize].push(0.0);
                for _ in 0..input_size {
                    weights[i as usize][j as usize].push(rand::thread_rng().gen::<f32>());
                }
            }
        }

        Self {
            input_size,
            output_size,
            weights,
            biases,
            output: vec![0.0; output_size as usize],
            learning_rate,
        }
    }

    pub fn forward(&mut self, image: &DynamicImage) -> Vec<f32> {
        // Convertir l'image en DynamicImage
        let dynamic_image: image::DynamicImage = image.clone().into();

        // Le reste du code reste le même
        let pixels = dynamic_image.to_rgb8().to_vec().iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let mut output = self.output.clone();

        for i in 0..self.output_size as usize {
            for j in 0..self.input_size as usize {
                let mut sum = 0.0;
                for k in 0..self.input_size as usize {
                    sum += self.weights[i][j][k] * pixels[k];
                }
                output[i] += sum + self.biases[j][i];
            }
        }

        output
    }

    pub fn backpropagate(&mut self, prediction: Vec<f32>, label: u8, _pixels: &[f32]) {
        // Calcul de l'erreur (loss) et des gradients
        let mut loss_gradients: Vec<f32> = Vec::with_capacity(self.output_size as usize);

        for i in 0..self.output_size as usize {
            let gradient = if i == label as usize {
                prediction[i] - 1.0
            } else {
                prediction[i]
            };

            loss_gradients.push(gradient);
        }

        // Mise à jour des poids et des biais
        for i in 0..self.output_size as usize {
            for j in 0..self.input_size as usize {
                for k in 0..self.input_size as usize {
                    self.weights[i][j][k] -= self.learning_rate * loss_gradients[i] * _pixels[k];
                }
                self.biases[j][i] -= self.learning_rate * loss_gradients[i];
            }
        }
    }
}

fn to_vec(mut image: DynamicImage) -> Vec<f32> {
    let mut pixels = Vec::new();

    // Convertir l'image en Rgba
    let image = image.to_rgba8();

    // Parcourir les pixels de l'image et les ajouter au vecteur
    for pixel in image.pixels() {
        pixels.push(pixel[0] as f32);
        pixels.push(pixel[1] as f32);
        pixels.push(pixel[2] as f32);
        pixels.push(pixel[3] as f32);
    }

    pixels
}

fn main() {
    use image::open;

    let mut classifier = ImageClassifier::new(28, 3, 0.01); // Remplacez 0.01 par la valeur souhaitée pour learning_rate



    // Charger les ensembles de données
    let data_football = open(Path::new("C:/Users/Louis/Documents/GitHub/machine_Learning_5JV/images/football/*.jpg")).unwrap();
    let data_volleyball = open(Path::new("C:/Users/Louis/Documents/GitHub/machine_Learning_5JV/images/volley/*.jpg")).unwrap();
    let data_football_americain = open(Path::new("C:/Users/Louis/Documents/GitHub/machine_Learning_5JV/images/american_football/*.jpg")).unwrap();

    // Charger l'image de test
    let image = open(Path::new("C:/Users/Louis/Documents/GitHub/machine_Learning_5JV/images/ballon.jpg")).unwrap();

    let prediction = classifier.forward(&image);
    let max_value = prediction.iter().cloned().fold(f32::MIN, f32::max);
    let max_index = prediction.iter().position(|&x| x == max_value).unwrap();

    // Backpropagation
    let pixels = to_vec(image);

    classifier.backpropagate(prediction, 0, &pixels);



    println!("La classe prédite est {}", max_index);
}