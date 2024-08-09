use crate::utils::tasks::Task;
use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct LinearModel {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub learning_rate: f64,
    pub task: Task,
    pub rng: StdRng,
}

impl LinearModel {
    pub fn new(
        num_features: usize,
        num_classes: usize,
        learning_rate: f64,
        task: Task,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let num_outputs = match task {
            Task::BinaryClassification => 1,
            Task::Regression => 1,
            Task::MultiClassClassification => num_classes,
        };

        LinearModel {
            weights: Array2::from_shape_fn((num_features, num_outputs), |_| {
                rng.gen_range(-0.5..0.5)
            }),
            bias: Array1::zeros(num_outputs),
            learning_rate,
            task,
            rng,
        }
    }

    pub fn predict(&self, x: &Array1<f64>) -> Array1<f64> {
        let linear_output = self.weights.t().dot(x) + &self.bias;
        match self.task {
            Task::BinaryClassification => Array1::from(vec![self.sigmoid(linear_output[0])]),
            Task::Regression => linear_output,
            Task::MultiClassClassification => {
                let exp_output = linear_output.mapv(|x| x.exp());
                let sum = exp_output.sum();
                exp_output / sum
            }
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array2<f64>, epochs: usize) {
        for _ in 0..epochs {
            let predictions = x.dot(&self.weights) + &self.bias;
            let errors = match self.task {
                Task::BinaryClassification => predictions.mapv(|p| self.sigmoid(p)) - y,
                Task::Regression => predictions - y,
                Task::MultiClassClassification => {
                    let softmax = predictions.mapv(|p| p.exp())
                        / predictions
                            .mapv(|p| p.exp())
                            .sum_axis(Axis(1))
                            .insert_axis(Axis(1));
                    softmax - y
                }
            };

            let gradient = x.t().dot(&errors);
            self.weights -= &(self.learning_rate * gradient / x.nrows() as f64);
            self.bias -= &(self.learning_rate * errors.sum_axis(Axis(0)) / x.nrows() as f64);
        }
    }

    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}
