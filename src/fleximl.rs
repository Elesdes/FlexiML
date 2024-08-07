use crate::models::linear_model::LinearModel;
use pyo3::prelude::*;

#[pymodule]
fn fleximl(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LinearModel>()?;
    Ok(())
}
