extern crate image;
extern crate rand;

use image::GrayImage;
use rand::Rng;
use std::fs::File;
use std::io::{self, Write};
use std::f64::consts::E;
use std::fmt::Error;
use std::{fs, iter};
use std::path::Path;
use serde_json; // Assurez-vous que serde_json est inclus dans vos dépendances

struct MLP {
    layers: usize,
    neurons_per_layer: Vec<usize>,
    weights: Vec<Vec<Vec<f64>>>,
    outputs: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
}

impl MLP {
    fn new(neurons_per_layer: Vec<usize>) -> MLP {
        let layers = neurons_per_layer.len() - 1;
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        let mut outputs = Vec::new();
        let mut deltas = Vec::new();

        for l in 0..layers {
            let mut layer_weights = Vec::new();
            let num_neurons = neurons_per_layer[l + 1];
            let num_inputs = neurons_per_layer[l] + 1; // +1 for bias

            for _ in 0..num_inputs {
                let neuron_weights: Vec<f64> = (0..num_neurons).map(|_| rng.gen_range(-1.0..1.0)).collect();
                layer_weights.push(neuron_weights);
            }

            weights.push(layer_weights);
            outputs.push(vec![0.0; num_neurons]);
            deltas.push(vec![0.0; num_neurons]);
        }

        MLP {
            layers,
            neurons_per_layer,
            weights,
            outputs,
            deltas,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }

    fn softmax(layer_outputs: &[f64]) -> Vec<f64> {
        let exp_sum: f64 = layer_outputs.iter().map(|&x| E.powf(x)).sum();
        layer_outputs.iter().map(|&x| E.powf(x) / exp_sum).collect()
    }

    fn cross_entropy_error(output: &[f64], expected: &[f64]) -> f64 {
        expected.iter().zip(output.iter()).map(|(&e, &o)| {
            if e == 1.0 { -o.ln() } else { -(1.0 - o).ln() }
        }).sum()
    }

    fn forward_propagate(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut activations = inputs;

        for l in 0..self.layers {
            let mut layer_inputs = vec![1.0]; // Bias neuron
            layer_inputs.extend(activations);

            if l == self.layers - 1 {
                activations = MLP::softmax(&layer_inputs.iter().zip(&self.weights[l]).map(|(i, weights)| {
                    weights.iter().map(|&w| i * w).sum::<f64>()
                }).collect::<Vec<f64>>());
            } else {
                activations = self.weights[l].iter().map(|weights| {
                    MLP::sigmoid(layer_inputs.iter().zip(weights.iter()).map(|(i, w)| i * w).sum())
                }).collect();
            }

            self.outputs[l] = activations.clone();
        }

        activations
    }

    fn backward_propagate_error(&mut self, expected: Vec<f64>) {
        for l in (0..self.layers).rev() {
            let errors: Vec<f64>;
            if l == self.layers - 1 {
                errors = expected.iter().zip(self.outputs[l].iter()).map(|(e, o)| e - o).collect();
            } else {
                errors = self.weights[l + 1].iter().zip(self.deltas[l + 1].iter()).map(|(weights, &delta)| {
                    weights.iter().map(|&w| w * delta).sum()
                }).collect();
            }

            self.deltas[l] = errors.iter().zip(self.outputs[l].iter()).map(|(error, output)| {
                error * MLP::sigmoid_derivative(*output)
            }).collect();
        }
    }

    fn update_weights(&mut self, inputs: Vec<f64>, learning_rate: f64) {
        let mut inputs = inputs;

        for (l, (layer_weights, layer_deltas)) in self.weights.iter_mut().zip(self.deltas.iter()).enumerate() {
            let inputs_bias = iter::once(&1.0).chain(inputs.iter()).copied().collect::<Vec<f64>>();

            for (neuron_weights, delta) in layer_weights.iter_mut().zip(layer_deltas.iter()) {
                for (weight, input) in neuron_weights.iter_mut().zip(inputs_bias.iter()) {
                    *weight += learning_rate * delta * input;
                }
            }

            if l < self.layers - 1 {
                inputs = self.outputs[l].clone();
            }
        }
    }

    fn train(&mut self, training_data: Vec<(Vec<f64>, Vec<f64>)>, learning_rate: f64, n_iterations: usize) {
        for epoch in 0..n_iterations {
            let mut total_error = 0.0;
            for (inputs, expected) in &training_data {
                let outputs = self.forward_propagate(inputs.clone());
                total_error += MLP::cross_entropy_error(&outputs, expected);
                self.backward_propagate_error(expected.clone());
                self.update_weights(inputs.clone(), learning_rate);
            }
            println!("Époque {}: Erreur moyenne = {}", epoch + 1, total_error / training_data.len() as f64);
        }
    }
    fn save_weights(&self, file_path: &str) -> io::Result<()> {
        let serialized_weights = serde_json::to_string(&self.weights).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let mut file = File::create(file_path)?;
        writeln!(file, "{}", serialized_weights)?;
        Ok(())
    }
    fn predict(&mut self, inputs: Vec<f64>) -> usize {
        let outputs = self.forward_propagate(inputs);
        outputs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap()
    }
}

fn load_images(folder_path: &str) -> Result<Vec<(Vec<f64>, Vec<f64>)>, String> {
    let mut data = Vec::new();
    let categories = ["Banana", "Avocado", "Tomato"];
    for (label, category) in categories.iter().enumerate() {
        let category_path = Path::new(folder_path).join(category);
        for entry in fs::read_dir(category_path).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let img_path = entry.path();
            let img = image::open(img_path).map_err(|e| e.to_string())?.to_luma8();
            let pixels: Vec<f64> = img.pixels().map(|p| p[0] as f64 / 255.0).collect();
            let mut label_vec = vec![0.0; categories.len()];
            label_vec[label] = 1.0;
            data.push((pixels, label_vec));
        }
    }
    Ok(data)
}

fn evaluate_model(mlp: &mut MLP, test_data: Vec<(Vec<f64>, Vec<f64>)>) {
    let mut correct_predictions = 0;
    for (pixels, expected) in test_data {
        let predicted = mlp.predict(pixels);
        if predicted == expected.iter().position(|&r| r == 1.0).unwrap() {
            correct_predictions += 1;
        }
    }
    let accuracy = correct_predictions as f64 / test_data.len() as f64;
    println!("Précision : {}", accuracy);
}

pub(crate) fn main() {
    let training_data = load_images("images/Training").expect("Erreur lors du chargement des images d'entraînement");
    let test_data = load_images("images/Test").expect("Erreur lors du chargement des images de test");

    let taille_image = training_data[0].0.len();
    let mut mlp = MLP::new(vec![taille_image, 128, 64, 3]); // Exemple de configuration du réseau

    mlp.train(training_data, 0.01, 1000);

    evaluate_model(&mut mlp, test_data);
}
