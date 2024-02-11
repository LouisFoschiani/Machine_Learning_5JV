use std::error::Error;
use std::io;

const IMAGE_SIZE: usize = 16 * 16; // Taille de l'image après aplatissement
const NUM_CENTERS: usize = 10; // Nombre de centres pour RBF
const INPUT_DIMENSION: usize = IMAGE_SIZE; // Dimension d'entrée basée sur la taille de l'image
const BETA: f32 = 1.0; // Paramètre beta pour la fonction de base
const NUM_EPOCHS: usize = 1000; // Nombre d'itérations d'entraînement
const LEARNING_RATE: f32 = 0.01; // Taux d'apprentissage pour l'ajustement des poids


struct RBFN {
    centers: Vec<Vec<f32>>,
    weights: Vec<f32>,
    beta: f32,
}

fn load_centers_from_file(file_path: &Path) -> Vec<Vec<f32>> {
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
        let centers: Vec<Vec<f32>> = vec![vec![0.0; input_dimension]; num_centers];
        let weights: Vec<f32> = vec![0.0; num_centers + 1]; // Inclut le poids pour le biais
        RBFN { centers, weights, beta: BETA }
    }
    pub fn print_weights(&self) {
        println!("Poids des connexions :");
        for (index, weight) in self.weights.iter().enumerate() {
            println!("Poids {}: {}", index, weight);
        }
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
        let img = image::open(path).map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let resized_img = img.resize_exact(16, 16, image::imageops::FilterType::Nearest);
        let features: Vec<f32> = resized_img.to_luma8().pixels().map(|p| p.0[0] as f32 / 255.0).collect();
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

        for epoch in 0..NUM_EPOCHS {
            println!("Époque {} ", epoch);
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
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        let mut section = ""; // Pour suivre la section actuelle du fichier (centres ou poids)
        for line in reader.lines() {
            let line = line?;
            if line == "-- Centers --" {
                section = "centers";
                self.centers.clear(); // Préparez pour le chargement des nouveaux centres
            } else if line == "-- Weights --" {
                section = "weights";
                self.weights.clear(); // Préparez pour le chargement des nouveaux poids
            } else {
                match section {
                    "centers" => {
                        let values: Vec<f32> = line.split(',')
                            .map(|s| s.trim().parse().unwrap())
                            .collect();
                        self.centers.push(values);
                    },
                    "weights" => {
                        let weight: f32 = line.parse().unwrap();
                        self.weights.push(weight);
                    },
                    _ => {}
                }
            }
        }

        Ok(())
    }
}

fn test_model() -> Result<(), Box<dyn Error>> {

    let mut model_accuracies: HashMap<String, f32> = HashMap::new();
    let base_test_path = Path::new("..\\images_16\\Test");
    let target_categories = vec!["Tomato", "Orange", "Aubergine"];

    for target_category in &target_categories {
        let model_file_path = format!("rbfn_model_weights_{}.txt", target_category);
        println!("Chargement du chemin :  {}", model_file_path);
        let mut rbfn = RBFN::new(0, INPUT_DIMENSION); // Temporairement initialisé avec 0 centres, sera chargé du fichier
        rbfn.load_model(Path::new(&model_file_path))?;
        rbfn.print_weights();

        let non_target_categories: Vec<&str> = target_categories.iter().filter(|&&x| x != *target_category).map(|&x| x).collect();

        let (test_features, test_labels) = RBFN::load_image_data(&base_test_path, target_category, &non_target_categories)?;
        if test_features.is_empty() {
            println!("Aucune donnée de test trouvée pour {}", target_category);
            continue;
        }

        // Évaluez le modèle sur les données de test
        let test_accuracy = rbfn.evaluate(&test_features, &test_labels);
        println!("Précision d'entraînement pour {}: {:.2}%", target_category, test_accuracy * 100.0);

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


pub fn run_rbfn_model(mode: &str) -> io::Result<()> {

    // Définissez le chemin de base pour vos données d'images
    let base_path = PathBuf::from("..\\images_16");

    // Mode d'exécution : train ou predict
    match mode {
        "train" => {
            // Chemins pour les données d'entraînement
            let training_path = base_path.join("Training");

            // Catégories cibles pour l'entraînement
            let categories = vec![ "Orange", "Tomato", "Aubergine"];

            for category in categories.iter() {
                println!("Entraînement du modèle pour la catégorie : {}", category);

                // Créer une nouvelle instance de RBFN pour cette catégorie
                let mut rbfn = RBFN::new(NUM_CENTERS, INPUT_DIMENSION);

                // Supposons que vous avez une fonction pour charger les données d'entraînement
                let (features, labels) = RBFN::load_image_data(&training_path, category, &categories)?;

                // Entraînement du modèle
                rbfn.train(&features, &labels, NUM_EPOCHS, LEARNING_RATE);
                test_model();
                // Sauvegarde du modèle entraîné
                let model_file_path = format!("rbfn_model_weights_{}.txt", category);
                rbfn.save_model(Path::new(&model_file_path))?;
            }
        },
        "predict" => {
            // Chemin pour une image spécifique à classer
            let image_path = base_path.join("CHECK\\Aubergine\\aubergine1.jpg");

            // Chemins vers les modèles entraînés
            let model_paths: Vec<&Path> = vec![
                Path::new("rbfn_model_weights_Orange.txt"),
                Path::new("rbfn_model_weights_Tomato.txt"),
                Path::new("rbfn_model_weights_Aubergine.txt"),
            ];

            // Prédiction de la catégorie pour l'image donnée
            let category = predict_category_for_image(&model_paths, &image_path)?;
            println!("L'image appartient à la catégorie : {}", category);
        },
        _ => println!("Mode non reconnu. Utilisez 'train' ou 'predict'."),
    }

    Ok(())
}
