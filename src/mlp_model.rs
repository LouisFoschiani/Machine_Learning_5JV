use rand::Rng;
use std::f64::consts::E;
use std::iter;

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
                errors = self.weights[l + 1].iter().zip(self.deltas[l + 1].iter()).map(|(weights, delta)| {
                    weights.iter().zip(delta.iter()).map(|(w, d)| w * d).sum()
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
        for _ in 0..n_iterations {
            let mut total_error = 0.0;
            for (inputs, expected) in &training_data {
                let outputs = self.forward_propagate(inputs.clone());
                total_error += MLP::cross_entropy_error(&outputs, expected);
                self.backward_propagate_error(expected.clone());
                self.update_weights(inputs.clone(), learning_rate);
            }
            println!("Erreur moyenne : {}", total_error / training_data.len() as f64);
        }
    }
}