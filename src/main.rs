#![allow(non_snake_case)]

extern crate image;
extern crate ndarray;
extern crate csv;
extern crate indicatif;

use image::{DynamicImage, GenericImageView};
use ndarray::{Array, Array2};
use std::fs;
use std::path::Path;
use std::error::Error;
use std::fs::File;
use indicatif::ProgressBar;

fn load_image(path: &str, target_width: u32, target_height: u32) -> Result<Array2<f32>, String> {
    let img = match image::io::Reader::open(path) {
        Ok(img) => img.decode().map_err(|err| format!("Erreur lors de la lecture de l'image : {}", err))?,
        Err(err) => return Err(format!("Erreur lors de la lecture de l'image : {}", err)),
    };

    let resized_img = img.resize_exact(target_width, target_height, image::imageops::FilterType::Lanczos3);

    let mut pixel_data: Vec<f32> = Vec::new();
    for pixel in resized_img.pixels() {
        let (r, g, b) = match pixel.2 {
            image::Rgba(color) => (color[0], color[1], color[2]),
            _ => (0, 0, 0), // Handle non-RGB images by setting to 0
        };
        pixel_data.push(r as f32 / 255.0);
        pixel_data.push(g as f32 / 255.0);
        pixel_data.push(b as f32 / 255.0);
    }

    let image_data = Array::from_shape_vec((target_height as usize, target_width as usize * 3), pixel_data)
        .expect("Failed to create image array");

    Ok(image_data)
}

fn load_images_from_directory(directory_path: &str, target_width: u32, target_height: u32) -> Result<Vec<Array2<f32>>, String> {
    let mut images: Vec<Array2<f32>> = Vec::new();

    let directory = match fs::read_dir(directory_path) {
        Ok(directory) => directory,
        Err(err) => return Err(format!("Erreur lors de l'accès au répertoire : {}", err)),
    };

    let progress = ProgressBar::new_spinner();

    for entry in directory {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.is_file() {
                let file_path = path.to_str().unwrap();
                match load_image(file_path, target_width, target_height) {
                    Ok(image_data) => {
                        images.push(image_data);
                        progress.inc(1);
                    }
                    Err(err) => eprintln!("Erreur : {} (Fichier : {})", err, file_path),
                }
            }
        }
    }

    progress.finish_and_clear();

    println!("Nombre d'images collectées depuis le répertoire '{}': {}", directory_path, images.len()); // Ajout d'une instruction de débogage

    Ok(images)
}

fn save_data_to_csv(data: &[Array2<f32>], file_path: &str) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_path(file_path)?;

    for image_data in data {
        let flattened_data: Vec<String> = image_data.iter().map(|&x| x.to_string()).collect();
        let flattened_data_bytes: Vec<&[u8]> = flattened_data.iter().map(|s| s.as_bytes()).collect();
        writer.write_record(&flattened_data_bytes)?;
    }

    writer.flush()?;
    Ok(())
}

fn main() {
    let data_dir = "D:/GitHub/Machine_Learning_5JV/images/";
    let target_width = 64;
    let target_height = 64;

    let class_directories = ["football_1", "volley_1", "american_football_1"];
    let mut all_images: Vec<Array2<f32>> = Vec::new();

    for class in &class_directories {
        let class_dir_path = Path::new(data_dir).join(class);
        match load_images_from_directory(class_dir_path.to_str().unwrap(), target_width, target_height) {
            Ok(mut images) => {
                all_images.append(&mut images);
                println!("Nombre d'images collectées pour la classe '{}': {}", class, images.len()); // Ajout d'une instruction de débogage
            }
            Err(err) => eprintln!("Erreur : {}", err),
        }
    }

    println!("Nombre total d'images collectées : {}", all_images.len()); // Ajout d'une instruction de débogage

    if let Err(err) = save_data_to_csv(&all_images, "D:/GitHub/Machine_Learning_5JV/images/data.csv") {
        eprintln!("Erreur lors de la sauvegarde des données : {}", err);
    }
}
