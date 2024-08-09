use approx::assert_relative_eq;
use fleximl_models::ml::mlp::{Activation, MLP};
use fleximl_models::utils::tasks::Task;
use ndarray::{arr1, arr2, Array1};

#[test]
fn test_mlp_creation() {
    let mlp = MLP::new(
        &[2, 3, 1],
        &[Activation::ReLU, Activation::Sigmoid],
        0.01,
        Task::BinaryClassification,
        42,
    );

    assert_eq!(mlp.layers.len(), 2);
    assert_eq!(mlp.layers[0].weights.shape(), &[3, 2]);
    assert_eq!(mlp.layers[1].weights.shape(), &[1, 3]);
    assert_eq!(mlp.learning_rate, 0.01);
    assert_eq!(mlp.task, Task::BinaryClassification);
}

#[test]
fn test_mlp_prediction_binary_classification() {
    let mut mlp = MLP::new(
        &[2, 2, 1],
        &[Activation::ReLU, Activation::Sigmoid],
        0.01,
        Task::BinaryClassification,
        42,
    );

    mlp.layers[0].weights = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    mlp.layers[0].biases = arr1(&[0.0, 0.0]);
    mlp.layers[1].weights = arr2(&[[1.0, 1.0]]);
    mlp.layers[1].biases = arr1(&[0.0]);

    let input = arr1(&[1.0, 0.0]);
    let prediction = mlp.predict(&input);

    assert_relative_eq!(prediction[0], 0.7310585786300049, epsilon = 1e-8);
}

#[test]
fn test_mlp_prediction_regression() {
    let mut mlp = MLP::new(
        &[2, 2, 1],
        &[Activation::ReLU, Activation::Linear],
        0.01,
        Task::Regression,
        42,
    );

    mlp.layers[0].weights = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    mlp.layers[0].biases = arr1(&[0.0, 0.0]);
    mlp.layers[1].weights = arr2(&[[1.0, 1.0]]);
    mlp.layers[1].biases = arr1(&[0.0]);

    let input = arr1(&[1.0, 0.0]);
    let prediction = mlp.predict(&input);

    assert_relative_eq!(prediction[0], 1.0, epsilon = 1e-8);
}

#[test]
fn test_mlp_fit_binary_classification() {
    let mut mlp = MLP::new(
        &[2, 8, 4, 1],
        &[Activation::ReLU, Activation::ReLU, Activation::Sigmoid],
        0.005,
        Task::BinaryClassification,
        42,
    );

    let x = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
    let y = arr2(&[[0.0], [1.0], [1.0], [0.0]]);

    mlp.fit(&x, &y, 20000);

    let predictions: Vec<f64> = x
        .outer_iter()
        .map(|input| mlp.predict(&Array1::from(input.to_vec()))[0])
        .collect();

    assert!(predictions[0] < 0.1);
    assert!(predictions[1] > 0.9);
    assert!(predictions[2] > 0.9);
    assert!(predictions[3] < 0.1);
}

#[test]
fn test_mlp_fit_regression() {
    let mut mlp = MLP::new(
        &[1, 16, 16, 1],
        &[Activation::ReLU, Activation::ReLU, Activation::Linear],
        0.001,
        Task::Regression,
        42,
    );

    let x = arr2(&[[0.0], [0.25], [0.5], [0.75], [1.0]]);
    let y = arr2(&[[0.0], [0.5], [1.0], [1.5], [2.0]]);

    mlp.fit(&x, &y, 10000);

    let predictions: Vec<f64> = x
        .outer_iter()
        .map(|input| mlp.predict(&Array1::from(input.to_vec()))[0])
        .collect();

    for (pred, target) in predictions.iter().zip(y.column(0).iter()) {
        println!("Prediction: {:.4}, Target: {:.4}", pred, target);
        assert_relative_eq!(pred, target, epsilon = 0.2, max_relative = 0.1);
    }
}

#[test]
fn test_mlp_multi_class_classification() {
    let mut mlp = MLP::new(
        &[2, 4, 3],
        &[Activation::ReLU, Activation::Linear],
        0.01,
        Task::MultiClassClassification,
        42,
    );

    let x = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
    let y = arr2(&[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]);

    mlp.fit(&x, &y, 5000);

    for (input, target) in x.outer_iter().zip(y.outer_iter()) {
        let prediction = mlp.predict(&Array1::from(input.to_vec()));
        let predicted_class = prediction
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let target_class = target
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(predicted_class, target_class);
    }
}
