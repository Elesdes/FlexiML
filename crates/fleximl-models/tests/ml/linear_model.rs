use approx::assert_relative_eq;
use fleximl_models::ml::{LinearModel, Task};
use ndarray::{arr1, arr2};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_model_creation() {
        let model = LinearModel::new(3, 0.01, Task::BinaryClassification);
        assert_eq!(model.weights.len(), 3);
        assert_eq!(model.bias, 0.0);
        assert_eq!(model.learning_rate, 0.01);
        assert_eq!(model.task, Task::BinaryClassification);
    }

    #[test]
    fn test_binary_classification_prediction() {
        let mut model = LinearModel::new(2, 0.01, Task::BinaryClassification);
        model.weights = arr1(&[1.0, -1.0]);
        model.bias = 0.5;

        let x = arr1(&[2.0, 1.0]);
        let prediction = model.predict(&x);
        assert_relative_eq!(prediction, 0.8175744761936437, epsilon = 1e-8);
    }

    #[test]
    fn test_regression_prediction() {
        let mut model = LinearModel::new(2, 0.01, Task::Regression);
        model.weights = arr1(&[1.5, -0.5]);
        model.bias = 1.0;

        let x = arr1(&[2.0, 3.0]);
        let prediction = model.predict(&x);
        assert_eq!(prediction, 2.5);
    }

    #[test]
    fn test_binary_classification_fit() {
        let mut model = LinearModel::new(2, 0.1, Task::BinaryClassification);
        let x = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]);
        let y = arr1(&[0.0, 0.0, 1.0, 1.0]);

        model.fit(&x, &y, 1000);

        let prediction1 = model.predict(&arr1(&[1.0, 2.0]));
        let prediction2 = model.predict(&arr1(&[4.0, 5.0]));

        assert!(prediction1 < 0.1);
        assert!(prediction2 > 0.9);
    }

    #[test]
    fn test_regression_fit() {
        let mut model = LinearModel::new(1, 0.01, Task::Regression);
        let x = arr2(&[[1.0], [2.0], [3.0], [4.0]]);
        let y = arr1(&[2.0, 4.0, 6.0, 8.0]);

        model.fit(&x, &y, 1000);

        let prediction = model.predict(&arr1(&[5.0]));
        assert_relative_eq!(prediction, 10.0, epsilon = 0.1);
    }
}
