use std::collections::HashMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use image::{self, GenericImageView};
use ndarray::Array2;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::element::Rectangle;
use plotters::prelude::{BLUE, Color, IntoFont, WHITE};
use rand::{seq::SliceRandom, thread_rng, Rng};


const IMAGE_SIZE: usize = 16 * 16;
const NUM_CENTERS: usize = 10; // Nombre de centres
const INPUT_DIMENSION: usize = IMAGE_SIZE; // Assurez-vous que cela correspond à la taille de l'image après traitement
const BETA: f32 = 1.0; // Valeur de beta
const NUM_EPOCHS: usize = 50;
const LEARNING_RATE: f32 = 0.01;

struct RBFN {
    centers: Vec<Vec<f32>>,
    weights: Vec<f32>,
    beta: f32,
}fn load_centers_from_file(file_path: &Path) -> Vec<Vec<f32>> {
    let file = File::open(file_path).expect("Erreur lors de l'ouverture du fichier");
    let reader = BufReader::new(file);
    let mut centers = Vec::new();

    for line in reader.lines() {
        if let Ok(line) = line {
            let values: Vec<f32> = line
                .split(',')
                .map(|s| s.trim().parse().expect("Erreur lors de la conversion en f32"))
                .collect();
            centers.push(values);
        }
    }

    centers
}

impl RBFN {
    fn new(num_centers: usize, input_dimension: usize) -> Self {
        let mut rng = thread_rng();
        let centers: Vec<Vec<f32>> = vec![vec![0.0; input_dimension]; num_centers];
        let weights: Vec<f32> = vec![0.0; num_centers + 1]; // +1 pour le biais
        let beta = 1.0 / input_dimension as f32; // Initialisation simple de beta

        RBFN { centers, weights, beta }
    }

    fn load_image_data(
        base_path: &Path,
        target_category: &str,
        non_target_categories: &[&str],
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
        for category in non_target_categories {
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

    fn evaluate(&self, features: &[Vec<f32>], labels: &[i32]) -> f32 {
        let mut correct_predictions = 0;
        for (feature, &label) in features.iter().zip(labels.iter()) {
            let prediction = if self.predict(feature) >= 0.0 { 1 } else { -1 };
            if prediction == label {
                correct_predictions += 1;
            }
        }
        let accuracy = correct_predictions as f32 / labels.len() as f32;
        accuracy // Retourne l'accuracy
    }
    fn load_model(&mut self, file_path: &Path) -> io::Result<()> {
        println!("Chargement du modèle à partir de : {:?}", file_path);

        let file = match File::open(file_path) {
            Ok(file) => file,
            Err(e) => {
                eprintln!("Erreur lors de l'ouverture du fichier : {:?}", e);
                return Err(e);
            },
        };

        let reader = BufReader::new(file);
        let mut line_iter = reader.lines().filter_map(|line| line.ok());

        let mut found_centers = false;
        let mut found_weights = false;

        while let Some(line) = line_iter.next() {
            if line.starts_with("-- Centers --") {
                println!("Chargement des centres...");
                found_centers = true;
            } else if line.starts_with("-- Weights --") {
                println!("Chargement des poids...");
                found_weights = true;
            }
        }

        if found_centers && found_weights {
            println!("Modèle chargé avec succès.");
        } else {
            println!("Le modèle n'a pas été chargé correctement. Vérifiez le format du fichier.");
        }

        Ok(())
    }

}

fn train_model() -> Result<(), Box<dyn Error>> {

    let mut model_accuracies: HashMap<String, f32> = HashMap::new();
    let base_training_path = Path::new("images_16\\Training");
    let target_categories = vec!["Tomato", "Orange", "Aubergine"];

    for target_category in &target_categories {
        let mut rbfn = RBFN::new(NUM_CENTERS, INPUT_DIMENSION);
        let non_target_categories: Vec<&str> = target_categories.iter().filter(|&&x| x != *target_category).map(|&x| x).collect();

        println!("Entraînement pour la catégorie : {}", target_category);
        let (features, labels) = RBFN::load_image_data(&base_training_path, target_category, &non_target_categories)?;

        if features.is_empty() {
            println!("Aucune donnée d'entraînement trouvée pour {}", target_category);
            continue;
        }

        rbfn.train(&features, &labels, NUM_EPOCHS, LEARNING_RATE);

        // Évaluez le modèle sur les données d'entraînement
        let accuracy = rbfn.evaluate(&features, &labels);

        println!("Précision d'entraînement pour {}: {:.2}%", target_category, accuracy * 100.0);
        model_accuracies.insert(target_category.to_string(), accuracy);

        let model_file_path = format!("rbfn_model_weights_{}.txt", target_category);
        rbfn.save_model(Path::new(&model_file_path))?;
    }
    generate_accuracy_comparison_chart(&model_accuracies, "model_accuracies.png")?;


    Ok(())
}

fn test_model() -> Result<(), Box<dyn Error>> {

    let mut model_accuracies: HashMap<String, f32> = HashMap::new();
    let base_test_path = Path::new("images_16/Test");
    let target_categories = vec!["Tomato", "Orange", "Aubergine"];

    for target_category in &target_categories {
        let model_file_path = format!("rbfn_model_weights_{}.txt", target_category);
        let mut rbfn = RBFN::new(0, INPUT_DIMENSION); // Temporairement initialisé avec 0 centres, sera chargé du fichier
        rbfn.load_model(Path::new(&model_file_path))?;

        let non_target_categories: Vec<&str> = target_categories.iter().filter(|&&x| x != *target_category).map(|&x| x).collect();

        let (test_features, test_labels) = RBFN::load_image_data(&base_test_path, target_category, &non_target_categories)?;
        if test_features.is_empty() {
            println!("Aucune donnée de test trouvée pour {}", target_category);
            continue;
        }

        // Évaluez le modèle sur les données de test
        let test_accuracy = rbfn.evaluate(&test_features, &test_labels);
        println!("Précision de test pour {}: {:.2}%", target_category, test_accuracy * 100.0);
        model_accuracies.insert(target_category.to_string(), test_accuracy);
    }

    Ok(())
}

fn predict_category_for_image(model_paths: &[&Path], image_path: &Path) -> io::Result<String> {
    let mut best_category = String::from("Inconnu");
    let mut best_score = f32::NEG_INFINITY;

    for model_path in model_paths {
        let model_name = model_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();

        // Extraire la catégorie du nom du modèle
        if let Some(category) = model_name.strip_prefix("rbfn_model_weights_").and_then(|s| s.strip_suffix(".txt")) {
            let mut rbfn = RBFN::new(0, INPUT_DIMENSION);
            rbfn.load_model(model_path)?;

            let image_features = RBFN::process_image(image_path)?;
            let score = rbfn.predict(&image_features);

            if score > best_score {
                best_score = score;
                best_category = category.to_string(); // Utilisez le nom de la catégorie
            }
        }
    }

    Ok(best_category)
}


pub(crate) fn main() -> Result<(), Box<dyn Error>> {
    let mode = "prediction"; // Changez "train" ou "test" selon le mode souhaité

    match mode {
        "train" => {
            train_model()?;
        }
        "test" => {
            test_model()?
        }
        "prediction" => {
            let image_path = Path::new("images_16\\CHECK\\Orange\\orange2.jpg");

            // Exemple de chemins vers les modèles entraînés
            let model_paths = vec![
                Path::new("rbfn_model_weights_Tomato.txt"),
                Path::new("rbfn_model_weights_Aubergine.txt"),
                Path::new("rbfn_model_weights_Orange.txt"),
            ];

            // Vérification des chemins des modèles avant de procéder à la prédiction
            for model_path in &model_paths {
                if !model_path.exists() {
                    println!("Le fichier modèle n'existe pas : {:?}", model_path);
                    return Ok(()); // Ou gérer l'erreur comme souhaité
                }
            }

            // Procéder à la prédiction si tous les modèles existent
            let category = predict_category_for_image(&model_paths, &image_path)?;
            println!("L'image appartient à la catégorie : {}", category);
        }
        _ => {
        println!("Invalid mode. Please choose 'train' or 'test'.");
        }
    }
    Ok(())
}
fn generate_accuracy_comparison_chart(
    model_accuracies: &HashMap<String, f32>,
    file_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_accuracy = model_accuracies
        .values()
        .cloned() // Clone the iterator's items, so we have f32 instead of &f32
        .fold(0.0_f32, |a, b| a.max(b)); // Use a closure to call f32::max


    let categories: Vec<String> = model_accuracies.keys().cloned().collect();
    let values: Vec<f32> = model_accuracies.values().cloned().collect();

    let mut chart = ChartBuilder::on(&root)
        .caption("Model Accuracy Comparison", ("sans-serif", 40).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..categories.len(), 0.0..(max_accuracy + 10.0))?;

    chart
        .configure_mesh()
        .x_labels(categories.len())
        .y_labels(10)
        .x_label_formatter(&|x| categories[*x].clone())
        .draw()?;

    chart.draw_series(
        values
            .iter()
            .enumerate()
            .map(|(idx, &val)| {
                Rectangle::new(
                    [
                        (idx, 0.0),
                        (idx + 1, val),
                    ],
                    BLUE.mix(0.5).filled(),
                )
            }),
    )?;

    root.present()?;

    Ok(())
}
