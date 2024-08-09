use ndarray::{Array1, Array2};

pub struct LinearModel {
    pub weights: Array1<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub task: Task,
}

#[derive(Debug, PartialEq)]
pub enum Task {
    BinaryClassification,
    Regression,
}

impl LinearModel {
    pub fn new(num_features: usize, learning_rate: f64, task: Task) -> Self {
        LinearModel {
            weights: Array1::zeros(num_features),
            bias: 0.0,
            learning_rate,
            task,
        }
    }

    pub fn predict(&self, x: &Array1<f64>) -> f64 {
        let linear_output = x.dot(&self.weights) + self.bias;
        match self.task {
            Task::BinaryClassification => self.sigmoid(linear_output),
            Task::Regression => linear_output,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, epochs: usize) {
        for _ in 0..epochs {
            let predictions = x.dot(&self.weights) + self.bias;
            let errors = match self.task {
                Task::BinaryClassification => predictions.map(|p| self.sigmoid(*p)) - y,
                Task::Regression => predictions - y,
            };

            let gradient = x.t().dot(&errors) / x.nrows() as f64;
            self.weights -= &(gradient * self.learning_rate);
            self.bias -= errors.mean().unwrap_or(0.0) * self.learning_rate;
        }
    }

    pub fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}
