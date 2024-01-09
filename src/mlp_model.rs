extern crate image;
extern crate rand;

use image::{DynamicImage, GenericImageView, ImageError};
use rand::Rng;
use std::{fs, iter, path::Path};
use std::f32::consts::E;
use std::fs::File;
use std::io::{self, Read, Write};
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::element::PathElement;
use plotters::prelude::{BLACK, BLUE, IntoFont, LineSeries, RED, WHITE};
use serde_json;
use plotters::prelude::*;

struct MLP {
    layers: usize,
    neurons_per_layer: Vec<usize>,
    weights: Vec<Vec<Vec<f32>>>,
    outputs: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
    train_errors: Vec<f32>,
    test_errors: Vec<f32>,
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
        let mut train_errors = Vec::new();
        let mut test_errors = Vec::new();

        // Initialisation des poids et des structures pour chaque couche
        for l in 0..layers {
            let mut layer_weights = Vec::new();
            let num_neurons = neurons_per_layer[l + 1];
            let num_inputs = neurons_per_layer[l] + 1; // +1 for bias

            // Initialisation des poids aléatoires pour chaque connexion
            for _ in 0..num_inputs {
                let neuron_weights: Vec<f32> = (0..num_neurons).map(|_| rng.gen_range(-1.0..1.0)).collect();
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
            train_errors,
            test_errors,
        }
    }
    // Fonction d'activation sigmoidale
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + E.powf(-x))
    }

    // Dérivée de la fonction sigmoidale
    fn sigmoid_derivative(x: f32) -> f32 {
        x * (1.0 - x)
    }

    // Fonction softmax pour la couche de sortie
    fn softmax(layer_outputs: &[f32]) -> Vec<f32> {
        let exp_sum: f32 = layer_outputs.iter().map(|&x| E.powf(x)).sum();
        layer_outputs.iter().map(|&x| E.powf(x) / exp_sum).collect()
    }

    // Calcul de l'erreur d'entropie croisée
    fn cross_entropy_error(output: &Vec<f32>, expected: &[f32]) -> f32 {
        let epsilon = 1e-10;
        expected.iter().zip(output.iter()).map(|(&e, &o)| {
            let o = o.clamp(epsilon, 1.0 - epsilon); // Assure que o est dans l'intervalle [epsilon, 1-epsilon]
            if e == 1.0 { -o.ln() } else { -(1.0 - o).ln() }
        }).sum()
    }

    // Propagation avant pour calculer les sorties du réseau
    fn forward_propagate(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        let mut activations = inputs;

        // Calcul des activations pour chaque couche
        for l in 0..self.layers {
            let mut layer_inputs = vec![1.0]; // Ajout du biais
            layer_inputs.extend(activations);

            // Utilisation de softmax pour la dernière couche, sinon sigmoid
            if l == self.layers - 1 {
                activations = MLP::softmax(&layer_inputs.iter().zip(&self.weights[l]).map(|(i, weights)| {
                    weights.iter().map(|&w| i * w).sum::<f32>()
                }).collect::<Vec<f32>>());
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
    fn backward_propagate_error(&mut self, expected: Vec<f32>) {
        for l in (0..self.layers).rev() {
            let errors: Vec<f32>;

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
    fn update_weights(&mut self, inputs: Vec<f32>, learning_rate: f32) {
        let mut inputs = inputs;

        for (l, (layer_weights, layer_deltas)) in self.weights.iter_mut().zip(self.deltas.iter()).enumerate() {
            let inputs_bias = iter::once(&1.0).chain(inputs.iter()).copied().collect::<Vec<f32>>();

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
    fn train(&mut self, training_data: &Vec<(Vec<f32>, Vec<f32>)>, learning_rate: f32, iter_num: usize) {
        let mut total_error = 0.0;

        // Propagation et rétropropagation pour chaque exemple d'entraînement
        for (inputs, expected) in training_data {
            let outputs = self.forward_propagate(inputs.clone());
            total_error += MLP::cross_entropy_error(&outputs, expected);
            self.backward_propagate_error(expected.clone());
            self.update_weights(inputs.clone(), learning_rate);
        }

        // Affichage de l'erreur moyenne après chaque époque
        println!("Époque {}: Erreur moyenne = {}", iter_num, total_error / training_data.len() as f32);
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
    fn predict(&mut self, inputs: Vec<f32>) -> usize {
        let outputs = self.forward_propagate(inputs);
        outputs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap()
    }

    // Prédiction de la catégorie d'une image donnée
    pub fn predict_image(&mut self, img_path: &str, categories: &[&str]) -> Result<String, String> {
        let img = Self::open_and_process_image(img_path)
            .map_err(|e| e.to_string())?;
        let pixels: Vec<f32> = Self::image_to_pixels(&img);
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
    fn image_to_pixels(img: &DynamicImage) -> Vec<f32> {
        img.pixels().flat_map(|(_, _, p)| {
            vec![
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0
            ]
        }).collect()
    }
}

fn plot_errors(train_errors: &Vec<f32>, test_errors: &Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {

    let name = format!("Stats MLP.png");
    let title = "Training and Test Errors Over Iteration";
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



// Chargement des images d'entraînement et de test en vecteurs normalisés
fn load_images(folder_path: &str) -> Result<Vec<(Vec<f32>, Vec<f32>)>, String> {
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
fn evaluate_model(mlp: &mut MLP, test_data: &Vec<(Vec<f32>, Vec<f32>)>, isTest : bool)-> f32 {
    let mut correct_predictions = 0;
    for (pixels, expected) in test_data { // Utiliser une référence ici
        let predicted = mlp.predict(pixels.clone()); // Clone pixels car mlp.predict prend la possession
        if predicted == expected.iter().position(|&r| r == 1.0).unwrap() {
            correct_predictions += 1;
        }
    }
    let accuracy = correct_predictions as f32 / test_data.len() as f32;
    if(isTest){
        mlp.test_errors.push(1.0-accuracy);
    }else{
        mlp.train_errors.push(1.0-accuracy);
    }
    println!("Précision : {}", accuracy);

    return accuracy;
}


// Fonction principale pour exécuter les opérations du MLP
pub fn main() {
    let should_train = false; // Mettez à true pour entraîner, false pour charger les poids et prédire

    let training_data = load_images("images_16/Training").expect("Erreur lors du chargement des images d'entraînement");
    let test_data = load_images("images_16/Test").expect("Erreur lors du chargement des images de test");
    let taille_image = training_data[0].0.len();

    let mut mlp = MLP::new(vec![taille_image, 75, 15, 3]);

    let mut lastPerf = 0.0;

    if should_train {
        for iter in 0..250{

            mlp.train(&training_data, 0.001, iter);
            let result = evaluate_model(&mut mlp, &test_data, true);
            if(result > lastPerf){
                lastPerf = result;
                mlp.save_weights("model_weights_mlp.json").expect("Cannot save weights");
            }
            evaluate_model(&mut mlp, &training_data, false);

            if should_train {

            }
        }
    }else {
        mlp.load_weights("model_weights_mlp.json").expect("Erreur lors du chargement des poids");
        let categories = ["Aubergine", "Orange", "Tomato"];
        let img_path = "images_16/CHECK/Orange/orange3.jpg"; // Mettez ici le chemin de votre image de test
        let result = mlp.predict_image(img_path, &categories);
        match result {
            Ok(category) => println!("Catégorie prédite : {}", category),
            Err(error) => println!("Erreur : {}", error),
        }
    }

    plot_errors(&mlp.train_errors, &mlp.test_errors).expect("Error generating image");


}