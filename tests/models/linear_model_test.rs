use fleximl::models::LinearModel;

#[cfg(test)]
mod linear_model_tests {
    use super::*;

    #[test]
    fn test_linear_model_predict() {
        let model = LinearModel { slope: 2.0, intercept: 1.0 };
        assert_eq!(model.predict(2.0), 5.0);
        assert_eq!(model.predict(0.0), 1.0);
        assert_eq!(model.predict(-1.0), -1.0);
    }

    #[test]
    fn test_linear_model_train() {
        let mut model = LinearModel::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        model.train(&x, &y, 0.01, 1000);

        // After training, the model should approximate y = 2x
        assert!((model.slope - 2.0).abs() < 0.1);
        assert!(model.intercept.abs() < 0.1);
    }

    #[test]
    fn test_linear_model_new() {
        let model = LinearModel::new();
        assert!(model.slope >= -1.0 && model.slope <= 1.0);
        assert!(model.intercept >= -1.0 && model.intercept <= 1.0);
    }
}