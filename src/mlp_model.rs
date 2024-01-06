extern crate image;
extern crate rand;

use image::{DynamicImage, GenericImageView, ImageError};
use rand::Rng;
use std::{fs, iter, path::Path};
use std::f64::consts::E;
use std::fs::File;
use std::io::{self, Read, Write};
use serde_json;
struct MLP {
    layers: usize,
    neurons_per_layer: Vec<usize>,
    weights: Vec<Vec<Vec<f64>>>,
    outputs: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
}


// Implémentation des méthodes pour MLP
impl MLP {
    // Constructeur pour initialiser un MLP
    fn new(neurons_per_layer: Vec<usize>) -> MLP {

        let layers = neurons_per_layer.len() - 1;
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        let mut outputs = Vec::new();
        let mut deltas = Vec::new();

        // Initialisation des poids et des structures pour chaque couche
        for l in 0..layers {
            let mut layer_weights = Vec::new();
            let num_neurons = neurons_per_layer[l + 1];
            let num_inputs = neurons_per_layer[l] + 1; // +1 for bias

            // Initialisation des poids aléatoires pour chaque connexion
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
    // Fonction d'activation sigmoidale
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    // Dérivée de la fonction sigmoidale
    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }

    // Fonction softmax pour la couche de sortie
    fn softmax(layer_outputs: &[f64]) -> Vec<f64> {
        let exp_sum: f64 = layer_outputs.iter().map(|&x| E.powf(x)).sum();
        layer_outputs.iter().map(|&x| E.powf(x) / exp_sum).collect()
    }

    // Calcul de l'erreur d'entropie croisée
    fn cross_entropy_error(output: &Vec<f64>, expected: &[f64]) -> f64 {
        expected.iter().zip(output.iter()).map(|(&e, &o)| {
            if e == 1.0 { -o.ln() } else { -(1.0 - o).ln() }
        }).sum()
    }

    // Propagation avant pour calculer les sorties du réseau
    fn forward_propagate(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut activations = inputs;

        // Calcul des activations pour chaque couche
        for l in 0..self.layers {
            let mut layer_inputs = vec![1.0]; // Ajout du biais
            layer_inputs.extend(activations);

            // Utilisation de softmax pour la dernière couche, sinon sigmoid
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

    // Rétropropagation pour calculer les deltas
    fn backward_propagate_error(&mut self, expected: Vec<f64>) {
        for l in (0..self.layers).rev() {
            let errors: Vec<f64>;

            // Calcul des erreurs pour chaque couche
            if l == self.layers - 1 {
                // Calcul des erreurs pour la couche de sortie
                errors = expected.iter().zip(self.outputs[l].iter()).map(|(e, o)| e - o).collect();
            } else {
                // Calcul des erreurs pour les couches cachées
                errors = self.weights[l + 1].iter().zip(self.deltas[l + 1].iter()).map(|(weights, &delta)| {
                    weights.iter().map(|&w| w * delta).sum()
                }).collect();
            }

            // Calcul des deltas en utilisant la dérivée de la fonction d'activation
            self.deltas[l] = errors.iter().zip(self.outputs[l].iter()).map(|(error, output)| {
                error * MLP::sigmoid_derivative(*output)
            }).collect();
        }
    }

    // Mise à jour des poids en utilisant les deltas et le taux d'apprentissage
    fn update_weights(&mut self, inputs: Vec<f64>, learning_rate: f64) {
        let mut inputs = inputs;

        for (l, (layer_weights, layer_deltas)) in self.weights.iter_mut().zip(self.deltas.iter()).enumerate() {
            let inputs_bias = iter::once(&1.0).chain(inputs.iter()).copied().collect::<Vec<f64>>();

            for (neuron_weights, delta) in layer_weights.iter_mut().zip(layer_deltas.iter()) {
                for (weight, input) in neuron_weights.iter_mut().zip(inputs_bias.iter()) {
                    *weight += learning_rate * delta * input;
                }
            }

            // Préparation des entrées pour la prochaine couche
            if l < self.layers - 1 {
                inputs = self.outputs[l].clone();
            }
        }
    }

    // Entraînement du réseau avec un ensemble de données, un taux d'apprentissage et un nombre d'itérations
    fn train(&mut self, training_data: Vec<(Vec<f64>, Vec<f64>)>, learning_rate: f64, n_iterations: usize) {
        for epoch in 0..n_iterations {
            let mut total_error = 0.0;

            // Propagation et rétropropagation pour chaque exemple d'entraînement
            for (inputs, expected) in &training_data {
                let outputs = self.forward_propagate(inputs.clone());
                total_error += MLP::cross_entropy_error(&outputs, expected);
                self.backward_propagate_error(expected.clone());
                self.update_weights(inputs.clone(), learning_rate);
            }

            // Affichage de l'erreur moyenne après chaque époque
            println!("Époque {}: Erreur moyenne = {}", epoch + 1, total_error / training_data.len() as f64);
        }
    }

    // Sauvegarde des poids du réseau dans un fichier
    pub fn save_weights(&self, file_path: &str) -> io::Result<()> {
        let serialized_weights = serde_json::to_string(&self.weights).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let mut file = File::create(file_path)?;
        writeln!(file, "{}", serialized_weights)?;
        Ok(())
    }

    // Chargement des poids du réseau à partir d'un fichier
    pub fn load_weights(&mut self, file_path: &str) -> io::Result<()> {
        let mut file = File::open(file_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        self.weights = serde_json::from_str(&contents)?;
        Ok(())
    }


    // Prédiction de l'indice de la catégorie d'une entrée donnée
    fn predict(&mut self, inputs: Vec<f64>) -> usize {
        let outputs = self.forward_propagate(inputs);
        outputs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap()
    }

    // Prédiction de la catégorie d'une image donnée
    pub fn predict_image(&mut self, img_path: &str, categories: &[&str]) -> Result<String, String> {
        let img = Self::open_and_process_image(img_path)
            .map_err(|e| e.to_string())?;
        let pixels: Vec<f64> = Self::image_to_pixels(&img);
        let predicted_index = self.predict(pixels);
        categories.get(predicted_index)
            .map(|&category| category.to_string())
            .ok_or("Catégorie non trouvée".to_string())
    }

    // Ouverture et traitement d'une image (conversion en niveaux de gris)
    fn open_and_process_image(img_path: &str) -> Result<DynamicImage, ImageError> {
        let img = image::open(img_path)?;
        Ok(img) // Convertir en RGB si nécessaire, ou ajouter d'autres traitements ici
    }

    // Conversion d'une image en vecteur de pixels normalisés
    fn image_to_pixels(img: &DynamicImage) -> Vec<f64> {
        img.pixels().flat_map(|(_, _, p)| {
            vec![
                p[0] as f64 / 255.0,
                p[1] as f64 / 255.0,
                p[2] as f64 / 255.0
            ]
        }).collect()
    }
}


// Chargement des images d'entraînement et de test en vecteurs normalisés
fn load_images(folder_path: &str) -> Result<Vec<(Vec<f64>, Vec<f64>)>, String> {
    let mut data = Vec::new();
    let categories = ["Aubergine", "Orange", "Tomato"];
    for (label, category) in categories.iter().enumerate() {
        let category_path = Path::new(folder_path).join(category);
        for entry in fs::read_dir(category_path).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let img_path = entry.path();
            let img = MLP::open_and_process_image(img_path.to_str().ok_or("Erreur de chemin d'image")?)
                .map_err(|e| e.to_string())?;
            let pixels = MLP::image_to_pixels(&img);
            let mut label_vec = vec![0.0; categories.len()];
            label_vec[label] = 1.0;
            data.push((pixels, label_vec));
        }
    }
    Ok(data)
}
// Évaluation de la précision du modèle sur les données de test
fn evaluate_model(mlp: &mut MLP, test_data: Vec<(Vec<f64>, Vec<f64>)>) {
    let mut correct_predictions = 0;
    for (pixels, expected) in &test_data { // Utiliser une référence ici
        let predicted = mlp.predict(pixels.clone()); // Clone pixels car mlp.predict prend la possession
        if predicted == expected.iter().position(|&r| r == 1.0).unwrap() {
            correct_predictions += 1;
        }
    }
    let accuracy = correct_predictions as f64 / test_data.len() as f64;
    println!("Précision : {}", accuracy);
}


// Fonction principale pour exécuter les opérations du MLP

pub fn main() {
    let should_train = false; // Mettez à true pour entraîner, false pour charger les poids et prédire

    let training_data = load_images("images/Training").expect("Erreur lors du chargement des images d'entraînement");
    let test_data = load_images("images/Test").expect("Erreur lors du chargement des images de test");
    let taille_image = training_data[0].0.len();

    let mut mlp = MLP::new(vec![taille_image, 128, 64, 3]);

    if should_train {
        mlp.train(training_data, 0.01, 50);
        evaluate_model(&mut mlp, test_data);
        mlp.save_weights("model_weights_mlp.json").expect("Erreur lors de l'enregistrement des poids");
    } else {

        mlp.load_weights("model_weights_mlp.json").expect("Erreur lors du chargement des poids");
        evaluate_model(&mut mlp, test_data);
    }

    // Testez avec un chemin d'image valide

    let categories = ["Aubergine", "Orange", "Tomato"];
    let img_path = "images/CHECK/orange/orange2.jpg"; // Mettez ici le chemin de votre image de test
    let result = mlp.predict_image(img_path, &categories);
    match result {
        Ok(category) => println!("Catégorie prédite : {}", category),
        Err(error) => println!("Erreur : {}", error),
    }
}