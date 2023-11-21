extern crate ndarray;
extern crate image;

use ndarray::{Array1, Array2};
use image::{GenericImageView, ImageError, io::Reader as ImageReader};
use std::fs::{self, File};
use std::io::{Write, BufWriter};
use rand::Rng;

struct LinearModel {
    weights: Array1<f32>,
}

impl LinearModel {
    fn new(input_size: usize) -> LinearModel {
        let weights = Array1::from(vec![0.1; input_size]);
        LinearModel { weights }
    }


    fn train(&mut self, inputs: &Array2<f32>, labels: &Array1<i32>, iterations: usize, learning_rate: f32) {
        let labels_f32 = labels.mapv(|l| l as f32);
        for _ in 0..iterations {
            let predictions = inputs.dot(&self.weights);
            let errors = &labels_f32 - &predictions;
            self.weights += &(inputs.t().dot(&errors) * learning_rate);

            if self.weights.iter().any(|&w| w.is_infinite()) {
                eprintln!("Poids infinis détectés.");
                break;
            }
        }

    }

    fn predict(&self, input: &Array1<f32>) -> i32 {
        if input.dot(&self.weights) > 0.5 { 1 } else { 0 }
    }

    fn save_weights(&self, file_path: &str) -> std::io::Result<()> {
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);

        for weight in self.weights.iter() {
            writeln!(writer, "{}", weight)?;
        }

        Ok(())
    }

    fn predict_category(&self, input: &Array1<f32>) -> String {
        match self.predict(input) {
            0 => "Banane".to_string(),
            1 => "Avocat".to_string(),
            2 => "Tomate".to_string(),
            _ => "Inconnu".to_string(),
        }
    }
}

fn load_image(path: &str) -> Result<Vec<f32>, ImageError> {
    let img = ImageReader::open(path)?.decode()?;
    Ok(img.pixels().map(|p| p.2[0] as f32 / 255.0).collect())
}

fn load_images_from_folder(folder: &str) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut images = Vec::new();
    let read_dir = fs::read_dir(folder).map_err(|e| format!("Erreur lors de la lecture du dossier '{}': {}", folder, e))?;

    for entry in read_dir {
        let path = entry.map_err(|e| format!("Erreur lors de l'accès à une entrée de dossier: {}", e))?.path();
        if path.is_file() {
            images.push(load_image(path.to_str().ok_or_else(|| "Chemin non valide".to_string())?)?);
        }
    }
    Ok(images)
}

fn create_labels(num_images: usize, label: i32) -> Vec<i32> {
    vec![label; num_images]
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let banana_images = load_images_from_folder("images/Unknown/Banana")?;
    let avocado_images = load_images_from_folder("images/Unknown/Avocado")?;
    let tomato_images = load_images_from_folder("images/Unknown/Tomato")?;

    let banana_labels = create_labels(banana_images.len(), 0);
    let avocado_labels = create_labels(avocado_images.len(), 1);
    let tomato_labels = create_labels(tomato_images.len(), 2);

    let all_images = [banana_images, avocado_images, tomato_images].concat();
    let all_labels = [banana_labels, avocado_labels, tomato_labels].concat();

    let flattened_images: Vec<f32> = all_images.into_iter().flatten().collect();
    let inputs = Array2::from_shape_vec((all_labels.len(), flattened_images.len() / all_labels.len()), flattened_images)?;
    let labels = Array1::from(all_labels);

    let mut model = LinearModel::new(inputs.ncols());
    model.train(&inputs, &labels, 1000, 0.0001); // Essayez une valeur encore plus petite



    // Sauvegarde des poids du modèle
    model.save_weights("model_weights.txt")?;

    // Exemple de prédiction (à adapter selon vos besoins)
    // Supposons que vous ayez une image "path/to/new_image.jpg"
    //let new_image = load_image("images/avocat.jpg")?;
    //let new_image_vector = Array1::from(new_image);
    //let category = model.predict_category(&new_image_vector);
    //println!("Catégorie prédite: {}", category);

    Ok(())
}