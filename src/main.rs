use std::collections::HashMap;
use std::fs::read_dir;
use std::path::Path;
use std::vec::Vec;
use std::ops::Index;
use std::iter::Iterator;
use std::f32;

use image;
use image::DynamicImage;
use image::Rgb;
use image::io::Reader;

use rand::Rng;

fn initialise_parameters(layer_dims: Vec<usize>) -> HashMap<String, Vec<f32>> {
    let mut parameters = HashMap::new();
    let L = layer_dims.len();

    for i in 1..L {
        let key_w = format!("W{}", i);
        let key_b = format!("B{}", i);
        let size_w = layer_dims[i];
        let size_prev = layer_dims[i - 1];

        let w = (0..size_w)
            .map(|_| (0..size_prev).map(|_| rand::thread_rng().gen::<f32>()).collect())
            .collect();

        let b = vec![0.0; size_w];

        parameters.insert(key_w, w);
        parameters.insert(key_b, b);
    }

    parameters
}

fn initialise_past(parameters: &HashMap<String, Vec<f32>>) -> HashMap<String, Vec<f32>> {
    let mut past = HashMap::new();
    for (key, value) in parameters.iter() {
        past.insert(key.clone(), vec![0.0; value.len()]);
    }
    past
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

fn forward_activation(
    a_prev: &Vec<f32>,
    w: &Vec<f32>,
    b: &Vec<f32>,
    activation: &str,
) -> (Vec<f32>, ((&Vec<f32>, &Vec<f32>, &Vec<f32>), f32)) {
    let z: Vec<f32> = w.iter()
        .zip(a_prev.iter())
        .map(|(w_val, a_val)| w_val * a_val)
        .zip(b.iter())
        .map(|(wa, b_val)| wa + b_val)
        .collect();
    let linear_cache = (a_prev, w, b);
    let a: Vec<f32>;
    let activation_cache: f32;
    if activation == "sigmoid" {
        a = z.iter().map(|z_val| sigmoid(*z_val)).collect();
        activation_cache = z[0]; // Note: For the first element, you can take any element since it's the same for all.
    } else {
        a = z.iter().map(|z_val| f32::max(0.0, *z_val)).collect();
        activation_cache = z[0];
    }
    (a, (linear_cache, activation_cache))
}

fn forward_propagate(x: &Vec<f32>, parameters: &HashMap<String, Vec<f32>>) -> (Vec<f32>, Vec<((&Vec<f32>, &Vec<f32>, &Vec<f32>), f32)>) {
    let mut caches = Vec::new();
    let mut a: &Vec<f32> = x;
    let l = parameters.len() / 2;
    for i in 1..l {
        let (a_next, cache) = forward_activation(
            &a,
            parameters.index(&format!("W{}", i)),
            parameters.index(&format!("B{}", i)),
            "relu",
        );
        caches.push(cache);
        a = &a_next;
    }
    let (a_last, cache) = forward_activation(
        &a,
        parameters.index(&format!("W{}", l)),
        parameters.index(&format!("B{}", l)),
        "sigmoid",
    );
    caches.push(cache);
    (a_last, caches)
}

fn compute_regularised_cost(
    a_last: &Vec<f32>,
    y: &Vec<f32>,
    caches: &Vec<(((&Vec<f32>, &Vec<f32>, &Vec<f32>), f32)>,
    lambd: f32,
) -> f32 {
    let m = y.len();
    let l = caches.len();
    let mut total: f32 = 0.0;
    for i in 0..l {
        let (cache, _) = &caches[i];
        let (a_prev, w, _) = cache;
        let summing: f32 = w.iter().map(|w_val| w_val.powi(2)).sum();
        total += summing;
    }
    let cost: f32 = -(1.0 / m as f32)
        * (y.iter().zip(a_last.iter()).map(|(y_val, a_val)| y_val * a_val.ln()).sum::<f32>()
        + (1.0 - y.iter().zip(a_last.iter()).map(|(y_val, a_val)| (1.0 - a_val).ln()).sum::<f32>()));
    let cost: f32 = cost.squeeze(); // Ensure cost array of size (1,1) becomes just a singular number
    let reg_cost = (lambd / (2.0 * m as f32)) * total;
    let regularised_cost = cost + reg_cost;
    regularised_cost
}

fn backward_linear(
    dz: &Vec<f32>,
    cache: &(&Vec<f32>, &Vec<f32>, &Vec<f32>),
    lambd: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (a_prev, w, b) = &cache.0;
    let m = a_prev.len();
    let d_w: Vec<f32> = w.iter().zip(dz.iter().map(|dz_val| dz_val / m as f32)).map(|(w_val, dz_val)| w_val * dz_val).collect();
    let d_b: Vec<f32> = dz.iter().map(|dz_val| dz_val / m as f32).collect();
    let d_a_prev: Vec<f32> = w.iter().map(|w_val| w_val * dz.iter().map(|dz_val| dz_val / m as f32).sum()).collect();
    (d_a_prev, d_w, d_b)
}

fn backward_activation(
    da: &Vec<f32>,
    cache: &((&Vec<f32>, &Vec<f32>, &Vec<f32>), f32),
    activation: &str,
    lambd: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (linear_cache, activation_cache) = &cache;
    let z = *activation_cache;
    let dz: Vec<f32>;
    let (da_prev, dw, db): (Vec<f32>, Vec<f32>, Vec<f32>);
    if activation == "relu" {
        dz = da.iter().map(|da_val| if z <= 0.0 { 0.0 } else { *da_val }).collect();
        let result = backward_linear(&dz, linear_cache, lambd);
        da_prev = result.0;
        dw = result.1;
        db = result.2;
    } else {
        dz = da.iter().zip(z).map(|(da_val, z_val)| da_val * sigmoid(z_val) * (1.0 - sigmoid(z_val))).collect();
        let result = backward_linear(&dz, linear_cache, lambd);
        da_prev = result.0;
        dw = result.1;
        db = result.2;
    }
    (da_prev, dw, db)
}

fn backward_propagate(
    a_last: &Vec<f32>,
    y: &Vec<f32>,
    caches: &Vec<(((&Vec<f32>, &Vec<f32>, &Vec<f32>), f32)>,
    lambd: f32,
) -> HashMap<String, Vec<f32>> {
    let mut gradients: HashMap<String, Vec<f32>> = HashMap::new();
    let l = caches.len();
    let y_vec = &y[0..a_last.len()];
    let d_a_last: Vec<f32> = a_last.iter().zip(y_vec).map(|(a_last_val, y_val)| -(y_val / a_last_val) + (1.0 - y_val) / (1.0 - a_last_val)).collect();
    let l_cache = &caches[l - 1];
    let (d_a, d_w, d_b) = backward_activation(&d_a_last, &l_cache, "sigmoid", lambd);
    gradients.insert("dA".to_string() + &(l - 1).to_string(), d_a);
    gradients.insert("dW".to_string() + &(l).to_string(), d_w);
    gradients.insert("dB".to_string() + &(l).to_string(), d_b);
    for i in (0..l - 1).rev() {
        let l_cache = &caches[i];
        let (d_a, d_w, d_b) = backward_activation(&gradients[&("dA".to_string() + &(i + 1).to_string())], l_cache, "relu", lambd);
        gradients.insert("dA".to_string() + &(i).to_string(), d_a);
        gradients.insert("dW".to_string() + &(i + 1).to_string(), d_w);
        gradients.insert("dB".to_string() + &(i + 1).to_string(), d_b);
    }
    gradients
}

fn gradient_descent_momentum(
    parameters: &HashMap<String, Vec<f32>>,
    gradients: &HashMap<String, Vec<f32>,
        learning_rate: f32,
        beta: f32,
        past: &HashMap<String, Vec<f32>,
) -> HashMap<String, Vec<f32>> {
    let l = parameters.len() / 2;
    for i in 1..=l {
        let key_w = "W".to_string() + &(i).to_string();
        let key_b = "B".to_string() + &(i).to_string();
        let beta_d_w = beta * past[&("dW".to_string() + &(i).to_string())]
            + (1.0 - beta) * &gradients[&("dW".to_string() + &(i).to_string())];
        let beta_d_b = beta * past[&("dB".to_string() + &(i).to_string())]
            + (1.0 - beta) * &gradients[&("dB".to_string() + &(i).to_string())];
        parameters.insert(
            key_w.clone(),
            parameters[&key_w]
                .iter()
                .zip(&beta_d_w)
                .map(|(w_val, beta_d_w_val)| w_val - learning_rate * beta_d_w_val)
                .collect(),
        );
        parameters.insert(
            key_b.clone(),
            parameters[&key_b]
                .iter()
                .zip(&beta_d_b)
                .map(|(b_val, beta_d_b_val)| b_val - learning_rate * beta_d_b_val)
                .collect(),
        );
    }
    parameters
}

fn batch_minibatch(
    x: &Vec<f32>,
    y: &Vec<f32>,
    minibatch_size: usize,
) -> Vec<(&Vec<f32>, &Vec<f32>)> {
    let m = y.len();
    let mut minibatches: Vec<(&Vec<f32>, &Vec<f32>)> = Vec::new();
    let num_minibatches = m / minibatch_size;
    for i in 0..num_minibatches {
        let start = i * minibatch_size;
        let end = (i + 1) * minibatch_size;
        let x_minibatch = &x[start..end];
        let y_minibatch = &y[start..end];
        minibatches.push((x_minibatch, y_minibatch));
    }
    if m % minibatch_size != 0 {
        let x_minibatch = &x[(num_minibatches * minibatch_size)..m];
        let y_minibatch = &y[(num_minibatches * minibatch_size)..m];
        minibatches.push((x_minibatch, y_minibatch));
    }
    minibatches
}

fn nn_model(
    x: &Vec<f32>,
    y: &Vec<f32>,
    layers_dims: &Vec<usize>,
    learning_rate: f32,
    beta: f32,
    lambd: f32,
    epochs: usize,
    minibatch_size: usize,
) -> HashMap<String, Vec<f32>> {
    let mut costs = Vec::new();
    let mut parameters = initialise_parameters(layers_dims.clone());
    let mut past = initialise_past(&parameters);
    let m = y.len();
    for epoch in 0..epochs {
        let minibatches = batch_minibatch(&x, &y, minibatch_size);
        for minibatch in minibatches {
            let (minibatch_x, minibatch_y) = minibatch;
            let (a_last, caches) = forward_propagate(minibatch_x, &parameters);
            let cost = compute_regularised_cost(&a_last, minibatch_y, &caches, lambd);
            let gradients = backward_propagate(&a_last, minibatch_y, &caches, lambd);
            parameters = gradient_descent_momentum(&parameters, &gradients, learning_rate, beta, &past);
        }
        if epoch % 50 == 0 {
            println!("Cost after epoch {}: {}", epoch, cost);
            costs.push(cost);
        }
    }
    parameters
}

fn predict(x: &Vec<f32>, parameters: &HashMap<String, Vec<f32>>) -> Vec<f32> {
    let a_last = forward_propagate(x, parameters).0;
    a_last.iter()
        .map(|a_last_val| if *a_last_val > 0.5 { 1.0 } else { 0.0 })
        .collect()
}

fn load_images_from_directory(path: &str) -> Result<(Vec<Vec<f32>>, Vec<f32>), image::ImageError> {
    let paths = fs::read_dir(path)?
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    let mut x_data = Vec::new();
    let mut y_labels = Vec::new();
    for path in paths {
        let x = image::open(&path)?.to_luma8();
        let x = x.to_bytes();
        x_data.push(x.iter().map(|x| *x as f32 / 255.0).collect());
        if path.to_str().unwrap().contains("ckt_sets") {
            y_labels.push(1.0);
        } else {
            y_labels.push(0.0);
        }
    }
    Ok((x_data, y_labels))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (x_data, y_labels) = load_images_from_directory("C:/Users/Tim/Documents/Python/Neural_Network/resized_ckt_sets")?;
    let (x_data2, mut y_labels2) = load_images_from_directory("C:/Users/Tim/Documents/Python/Neural_Network/resized_non_ckt")?;
    let mut x_data = x_data;
    x_data.append(&mut x_data2);
    y_labels.append(&mut y_labels2);
    let layers_dims = vec![x_data[0].len(), 16, 1];
    let train_data = train_test_split(&x_data, &y_labels, 0.85);
    let (train_x, test_x, train_y, test_y) = train_data;
    let parameters = nn_model(
        train_x,
        train_y,
        &layers_dims,
        0.0090,
        0.90,
        6.3,
        450,
        64,
    );
    let predict_train = predict(&train_x, &parameters);
    let sample_train = predict_train.len();
    let accuracy_train = (predict_train
        .iter()
        .zip(train_y.iter())
        .map(|(predict_train_val, train_y_val)| (predict_train_val == train_y_val) as u32)
        .sum::<u32>() as f32)
        / sample_train as f32;
    println!("Accuracy of training sets: {}", accuracy_train);
    let predict_test = predict(&test_x, &parameters);
    let sample_test = predict_test.len();
    let accuracy_test = (predict_test
        .iter()
        .zip(test_y.iter())
        .map(|(predict_test_val, test_y_val)| (predict_test_val == test_y_val) as u32)
        .sum::<u32>() as f32)
        / sample_test as f32;
    println!("Accuracy of testing sets: {}", accuracy_test);
    Ok(())
}
