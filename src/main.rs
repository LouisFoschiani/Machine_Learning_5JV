#![allow(non_snake_case)]

extern crate ndarray;
use ndarray::{Array, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use image::GenericImageView;

const INPUT_SIZE: usize = 64 * 64 * 1;
const NUM_CLASSES: usize = 3;
const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 1000;
const BATCH_SIZE: usize = 32;

fn load_and_preprocess_image(image_path: &str) -> Array<f64, ndarray::Ix2> {
    let img = image::open(image_path)
        .unwrap_or_else(|e| panic!("Failed to load image from '{}': {}", image_path, e));

    let img = img.resize_exact(64, 64, image::imageops::FilterType::Nearest).grayscale();

    let mut img_array = Array::from_elem((64, 64, 1), 0.0);
    for x in 0..64 {
        for y in 0..64 {
            let pixel = img.get_pixel(x as u32, y as u32);
            img_array[[x as usize, y as usize, 0]] = (pixel[0] as f64 / 255.0);
        }
    }

    img_array.into_shape((INPUT_SIZE, 1)).expect("Failed to reshape image")
}

fn initialize_weights(input_size: usize, num_classes: usize) -> (Array<f64, ndarray::Ix2>, Array<f64, ndarray::Ix1>) {
    let weights = Array::random((num_classes, input_size), Normal::new(0.0, 0.1).unwrap());
    let biases = Array::zeros(num_classes);
    (weights, biases)
}

fn softmax(x: &Array<f64, ndarray::Ix1>) -> Array<f64, ndarray::Ix1> {
    let exp_x = x.map(|val| val.exp());
    let exp_x_clone = exp_x.clone();
    exp_x / exp_x_clone.sum()
}

fn forward(input: &Array<f64, ndarray::Ix2>, weights: &Array<f64, ndarray::Ix2>, biases: &Array<f64, ndarray::Ix1>) -> Array<f64, ndarray::Ix1> {
    let logits = weights.dot(input) + biases;
    let logits_1d = logits.into_shape((NUM_CLASSES,)).expect("Failed to reshape logits");
    softmax(&logits_1d)
}

fn train_model(images: &Vec<Array<f64, ndarray::Ix2>>, labels: &Array<f64, ndarray::Ix2>, num_epochs: usize) -> (Array<f64, ndarray::Ix2>, Array<f64, ndarray::Ix1>) {
    let input_size = INPUT_SIZE;
    let num_classes = NUM_CLASSES;
    let learning_rate = LEARNING_RATE;

    let (mut weights, mut biases) = initialize_weights(input_size, num_classes);

    let num_samples = images.len();

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;

        for batch_start in (0..num_samples).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(num_samples);
            let mut batch_input = Array::zeros((input_size, batch_end - batch_start));
            let batch_label = labels.slice(s![.., batch_start..batch_end]);

            for (i, j) in (0..(batch_end - batch_start)).enumerate() {
                batch_input.slice_mut(s![.., i]).assign(&images[j]);
            }

            let batch_output = forward(&batch_input, &weights, &biases);
            let loss = -(&batch_label * &batch_output.map(|x| x.ln())).sum();
            total_loss += loss;
            let gradient = &batch_output - &batch_label;
            let batch_gradient = gradient.dot(&batch_input.t());
            weights -= &(learning_rate * &batch_gradient);
            biases -= &(learning_rate * &gradient.sum_axis(Axis(1)));
        }

        println!("Epoch {}: Loss: {}", epoch, total_loss);
    }

    (weights, biases)
}

fn main() {
    let mut soccer_ball_images: Vec<Array<f64, ndarray::Ix2>> = Vec::new();
    let mut american_football_images: Vec<Array<f64, ndarray::Ix2>> = Vec::new();
    let mut volleyball_images: Vec<Array<f64, ndarray::Ix2>> = Vec::new();

    for i in 0..400 {
        let image_path = format!("D:/GitHub/Machine_Learning_5JV/images/soccer_ball/soccer_ball_{}.jpg", i);
        let image = load_and_preprocess_image(&image_path);
        soccer_ball_images.push(image);
    }

    for i in 0..400 {
        let image_path = format!("D:/GitHub/Machine_Learning_5JV/images/american_ball/american_ball_{}.jpg", i);
        let image = load_and_preprocess_image(&image_path);
        american_football_images.push(image);
    }

    for i in 0..400 {
        let image_path = format!("D:/GitHub/Machine_Learning_5JV/images/volley_ball/volleyball_{}.jpg", i);
        let image = load_and_preprocess_image(&image_path);
        volleyball_images.push(image);
    }

    let num_samples = soccer_ball_images.len() + american_football_images.len() + volleyball_images.len();

    let mut soccer_ball_labels = Array::zeros((NUM_CLASSES, 400));
    soccer_ball_labels.row_mut(0).fill(1.0);

    let mut american_football_labels = Array::zeros((NUM_CLASSES, 400));
    american_football_labels.row_mut(1).fill(1.0);

    let mut volleyball_labels = Array::zeros((NUM_CLASSES, 400));
    volleyball_labels.row_mut(2).fill(1.0);

    let (soccer_ball_weights, soccer_ball_biases) = train_model(&soccer_ball_images, &soccer_ball_labels, EPOCHS);
    let (american_football_weights, american_football_biases) = train_model(&american_football_images, &american_football_labels, EPOCHS);
    let (volleyball_weights, volleyball_biases) = train_model(&volleyball_images, &volleyball_labels, EPOCHS);
}
