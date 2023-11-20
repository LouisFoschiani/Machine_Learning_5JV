extern crate ndarray;
extern crate image;

use ndarray::{Array, Array1, Array2};
use image::{DynamicImage, GenericImageView, ImageError, open};
use std::fs;
use std::path::Path;
use rand::Rng;
use std::fmt;
use image::io::Reader as ImageReader;

// Structure du modèle linéaire

struct LinearModel {
    weights: Array1<f32>,
}

impl LinearModel {
    fn new(input_size: usize) -> LinearModel {
        let mut rng = rand::thread_rng();
        let weights = Array1::from((0..input_size).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>());
        LinearModel { weights }
    }

    fn train(&mut self, inputs: &Array2<f32>, labels: &Array1<i32>, iterations: usize, learning_rate: f32) {
        for _ in 0..iterations {
            let predictions = inputs.dot(&self.weights);
            let errors = &labels - &predictions;
            self.weights += &(inputs.t().dot(&errors) * learning_rate);
        }
    }

    fn predict(&self, input: &Array1<f32>) -> i32 {
        if input.dot(&self.weights) > 0.5 { 1 } else { 0 }
    }
}

fn load_image(path: &str) -> Vec<f32> {
    let img = ImageReader::open(path).unwrap().decode().unwrap();
    img.pixels()
        .map(|p| p.2[0] as f32 / 255.0)
        .collect::<Vec<f32>>()
}

fn load_images_from_folder(folder: &str) -> Vec<Vec<f32>> {
    fs::read_dir(folder).unwrap()
        .filter_map(|entry| {
            entry.ok().and_then(|e| {
                let path = e.path();
                if path.is_file() {
                    Some(load_image(path.to_str().unwrap()))
                } else {
                    None
                }
            })
        })
        .collect()
}

fn create_labels(num_images: usize, label: i32) -> Vec<i32> {
    vec![label; num_images]
}

fn main() {
    let banana_images = load_images_from_folder("Unknown/Banana");
    let avocado_images = load_images_from_folder("Unknown/Avocado");
    let tomato_images = load_images_from_folder("Unknown/Tomato");

    let banana_labels = create_labels(banana_images.len(), 0);
    let avocado_labels = create_labels(avocado_images.len(), 1);
    let tomato_labels = create_labels(tomato_images.len(), 2);

    let all_images = [banana_images, avocado_images, tomato_images].concat();
    let all_labels = [banana_labels, avocado_labels, tomato_labels].concat();

    let flattened_images: Vec<f32> = all_images.into_iter().flatten().collect();
    let inputs = Array2::from_shape_vec((all_labels.len(), flattened_images.len() / all_labels.len()), flattened_images).unwrap();
    let labels = Array1::from(all_labels);

    let mut model = LinearModel::new(inputs.ncols());
    model.train(&inputs, &labels, 1000, 0.01);
}