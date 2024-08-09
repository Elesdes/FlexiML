use crate::utils::tasks::Task;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct MLP {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
    pub task: Task,
    pub rng: StdRng,
}

pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub activation: Activation,
    pub weight_momentum: Array2<f64>,
    pub bias_momentum: Array1<f64>,
}

#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    Sigmoid,
    TanH,
    Linear,
}

impl MLP {
    pub fn new(
        layer_sizes: &[usize],
        activations: &[Activation],
        learning_rate: f64,
        task: Task,
        seed: u64,
    ) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "At least input and output layers are required"
        );
        assert_eq!(
            layer_sizes.len() - 1,
            activations.len(),
            "Number of activations must match number of layers - 1"
        );

        let mut rng = StdRng::seed_from_u64(seed);

        let layers: Vec<Layer> = layer_sizes
            .windows(2)
            .zip(activations.iter())
            .map(|(sizes, &activation)| {
                let (n_in, n_out) = (sizes[0], sizes[1]);
                let weights = Array2::from_shape_fn((n_out, n_in), |_| {
                    rng.gen_range(-1.0..1.0) * (2.0 / n_in as f64).sqrt()
                });
                let biases = Array1::zeros(n_out);
                Layer {
                    weights,
                    biases,
                    activation,
                    weight_momentum: Array2::zeros((n_out, n_in)),
                    bias_momentum: Array1::zeros(n_out),
                }
            })
            .collect();

        MLP {
            layers,
            learning_rate,
            task,
            rng,
        }
    }

    pub fn predict(&self, x: &Array1<f64>) -> Array1<f64> {
        let num_layers = self.layers.len();
        self.layers
            .iter()
            .enumerate()
            .fold(x.clone(), |input, (i, layer)| {
                let output = layer.weights.dot(&input) + &layer.biases;
                Self::activate(&output, layer.activation, i == num_layers - 1, self.task)
            })
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>, epochs: usize) {
        let num_layers = self.layers.len();
        for _ in 0..epochs {
            for (input, target) in x.outer_iter().zip(y.outer_iter()) {
                let mut activations = vec![input.to_owned()];

                // Forward pass
                for (i, layer) in self.layers.iter().enumerate() {
                    let output =
                        layer.weights.dot(&activations.last().unwrap().view()) + &layer.biases;
                    activations.push(Self::activate(
                        &output,
                        layer.activation,
                        i == num_layers - 1,
                        self.task,
                    ));
                }

                // Backward pass
                let mut delta = activations.last().unwrap() - &target;

                for (i, (layer, activation)) in self
                    .layers
                    .iter_mut()
                    .rev()
                    .zip(activations.iter().rev().skip(1))
                    .enumerate()
                {
                    let gradient = delta.clone();
                    if i < num_layers - 1 {
                        delta = layer.weights.t().dot(&delta)
                            * Self::activate_derivative(
                                activation,
                                layer.activation,
                                i == 0, // is_output_layer is true for the last layer in backpropagation
                                self.task,
                            );
                    }

                    let weight_update = gradient
                        .clone()
                        .into_shape((gradient.len(), 1))
                        .unwrap()
                        .dot(&activation.view().into_shape((1, activation.len())).unwrap());

                    layer.weights -= &(self.learning_rate * &weight_update);
                    layer.biases -= &(self.learning_rate * &gradient);
                }
            }
        }
    }

    fn activate(
        x: &Array1<f64>,
        activation: Activation,
        is_output_layer: bool,
        task: Task,
    ) -> Array1<f64> {
        if is_output_layer && task == Task::Regression {
            return x.to_owned();
        }
        match activation {
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::TanH => x.mapv(|v| v.tanh()),
            Activation::Linear => x.to_owned(),
        }
    }

    fn activate_derivative(
        x: &Array1<f64>,
        activation: Activation,
        is_output_layer: bool,
        task: Task,
    ) -> Array1<f64> {
        if is_output_layer && task == Task::Regression {
            return Array1::ones(x.len());
        }
        match activation {
            Activation::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => {
                let s = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                s.clone() * (1.0 - s)
            }
            Activation::TanH => x.mapv(|v| 1.0 - v.tanh().powi(2)),
            Activation::Linear => Array1::ones(x.len()),
        }
    }
}
