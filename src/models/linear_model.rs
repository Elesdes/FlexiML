use pyo3::prelude::*;
use rand::Rng;

#[pyclass]
pub struct LinearModel {
    #[pyo3(get, set)]
    pub slope: f64,
    #[pyo3(get, set)]
    pub intercept: f64,
}

pub trait LinearModelMethods {
    fn predict(&self, x: &[f64]) -> Vec<f64>;
    fn train(
        &mut self,
        x: &[f64],
        y: &[f64],
        learning_rate: f64,
        epochs: usize,
    ) -> Result<(), String>;
    fn mse(&self, x: &[f64], y: &[f64]) -> Result<f64, String>;
}

// Rust implementation
impl LinearModelMethods for LinearModel {
    fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&xi| self.slope * xi + self.intercept)
            .collect()
    }

    fn train(
        &mut self,
        x: &[f64],
        y: &[f64],
        learning_rate: f64,
        epochs: usize,
    ) -> Result<(), String> {
        if x.len() != y.len() {
            return Err("Input and output vectors must have the same length".to_string());
        }

        for _ in 0..epochs {
            for (&xi, &yi) in x.iter().zip(y.iter()) {
                let prediction = self.slope * xi + self.intercept;
                let error = prediction - yi;

                self.slope -= learning_rate * error * xi;
                self.intercept -= learning_rate * error;
            }
        }
        Ok(())
    }

    fn mse(&self, x: &[f64], y: &[f64]) -> Result<f64, String> {
        if x.len() != y.len() {
            return Err("Input and output vectors must have the same length".to_string());
        }

        let predictions = LinearModelMethods::predict(self, x);
        Ok(predictions
            .iter()
            .zip(y.iter())
            .map(|(&pred, &actual)| (pred - actual).powi(2))
            .sum::<f64>()
            / y.len() as f64)
    }
}

// Python implementation
#[pymethods]
impl LinearModel {
    #[new]
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        LinearModel {
            slope: rng.gen_range(-1.0..1.0),
            intercept: rng.gen_range(-1.0..1.0),
        }
    }

    pub fn predict(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(LinearModelMethods::predict(self, &x))
    }

    pub fn train(
        &mut self,
        x: Vec<f64>,
        y: Vec<f64>,
        learning_rate: Option<f64>,
        epochs: Option<usize>,
    ) -> PyResult<()> {
        LinearModelMethods::train(
            self,
            &x,
            &y,
            learning_rate.unwrap_or(0.01),
            epochs.unwrap_or(1000),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    pub fn mse(&self, x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
        LinearModelMethods::mse(self, &x, &y)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }
}

impl LinearModel {
    pub fn with_parameters(slope: f64, intercept: f64) -> Self {
        LinearModel { slope, intercept }
    }
}
