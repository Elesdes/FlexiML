use approx::assert_relative_eq;
use fleximl::models::{LinearModel, LinearModelMethods};

#[test]
fn test_new() {
    let model = LinearModel::new();
    assert!(-1.0 <= model.slope && model.slope <= 1.0);
    assert!(-1.0 <= model.intercept && model.intercept <= 1.0);
}

#[test]
fn test_with_parameters() {
    let model = LinearModel::with_parameters(2.0, 1.0);
    assert_eq!(model.slope, 2.0);
    assert_eq!(model.intercept, 1.0);
}

#[test]
fn test_predict() {
    let model = LinearModel::with_parameters(2.0, 1.0);
    let predictions = LinearModelMethods::predict(&model, &[2.0, 0.0, -1.0]);
    assert_relative_eq!(predictions[0], 5.0, epsilon = 1e-10);
    assert_relative_eq!(predictions[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(predictions[2], -1.0, epsilon = 1e-10);
}

#[test]
fn test_train() {
    let mut model = LinearModel::new();
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    LinearModelMethods::train(&mut model, &x, &y, 0.01, 1000).unwrap();

    assert_relative_eq!(model.slope, 2.0, epsilon = 0.1);
    assert_relative_eq!(model.intercept, 0.0, epsilon = 0.1);
}

#[test]
fn test_mse() {
    let model = LinearModel::with_parameters(2.0, 0.0);
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    assert_relative_eq!(
        LinearModelMethods::mse(&model, &x, &y).unwrap(),
        0.0,
        epsilon = 1e-10
    );
}

#[test]
fn test_train_input_validation() {
    let mut model = LinearModel::new();
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0]; // Mismatched length
    assert!(LinearModelMethods::train(&mut model, &x, &y, 0.01, 1000).is_err());
}

#[test]
fn test_mse_input_validation() {
    let model = LinearModel::with_parameters(2.0, 0.0);
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0]; // Mismatched length
    assert!(LinearModelMethods::mse(&model, &x, &y).is_err());
}
