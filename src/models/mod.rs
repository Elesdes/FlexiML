pub mod linear_model;

pub use linear_model::{LinearModel, LinearModelMethods};

pub trait Model {
    fn predict(&self, x: &[f64]) -> Vec<f64>;
    fn train(&mut self, x: &[f64], y: &[f64]) -> Result<(), String>;
}
