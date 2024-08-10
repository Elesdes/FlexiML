use approx::assert_abs_diff_eq;
use fleximl_models::ml::rbf_kmeans::RBFKMeans;
use ndarray::Array2;
use rand::prelude::*;

#[test]
fn test_initialization() {
    let model = RBFKMeans::new(3, 100, 1e-4, Some(42));
    assert_eq!(model.n_clusters, 3);
    assert_eq!(model.max_iter, 100);
    assert_abs_diff_eq!(model.tol, 1e-4);
    assert!(model.centers.is_none());
}

#[test]
fn test_fit_predict() {
    let mut rng = StdRng::seed_from_u64(42);
    let x = Array2::from_shape_fn((100, 2), |_| rng.gen_range(0.0..10.0));

    let mut model = RBFKMeans::new(3, 100, 1e-4, Some(42));
    model.fit(&x);

    assert!(
        model.centers.is_some(),
        "Centers should be Some after fitting"
    );
    assert_eq!(
        model.centers.as_ref().unwrap().shape(),
        &[3, 2],
        "Centers should have shape [3, 2]"
    );

    let labels = model.predict(&x);

    assert_eq!(labels.len(), 100, "Should have 100 labels");
    assert!(
        labels.iter().all(|&l| l < 3),
        "All labels should be less than 3"
    );
}

#[test]
fn test_transform() {
    let mut rng = StdRng::seed_from_u64(42);
    let x = Array2::from_shape_fn((100, 2), |_| rng.gen_range(0.0..10.0));

    let mut model = RBFKMeans::new(3, 100, 1e-4, Some(42));
    model.fit(&x);

    let transformed = model.transform(&x);
    assert_eq!(transformed.shape(), &[100, 3]);
    assert!(transformed.iter().all(|&v| v >= 0.0 && v <= 1.0));
}

#[test]
fn test_reproducibility() {
    let mut rng = StdRng::seed_from_u64(42);
    let x = Array2::from_shape_fn((100, 2), |_| rng.gen_range(0.0..10.0));

    let mut model1 = RBFKMeans::new(3, 100, 1e-4, Some(42));
    model1.fit(&x);
    let labels1 = model1.predict(&x);

    let mut model2 = RBFKMeans::new(3, 100, 1e-4, Some(42));
    model2.fit(&x);
    let labels2 = model2.predict(&x);

    assert_eq!(labels1, labels2);
}

#[test]
fn test_different_seeds() {
    let mut data_rng = StdRng::seed_from_u64(100);
    let x = Array2::from_shape_fn((100, 2), |_| data_rng.gen_range(0.0..10.0));

    let mut model1 = RBFKMeans::new(3, 100, 1e-4, Some(42));
    model1.fit(&x);
    let labels1 = model1.predict(&x);

    let mut model2 = RBFKMeans::new(3, 100, 1e-4, Some(24));
    model2.fit(&x);
    let labels2 = model2.predict(&x);

    assert_ne!(
        labels1, labels2,
        "Labels should be different for different seeds"
    );

    assert!(
        labels1.iter().all(|&l| l < 3),
        "All labels should be less than 3 for seed 42"
    );
    assert!(
        labels2.iter().all(|&l| l < 3),
        "All labels should be less than 3 for seed 24"
    );

    let count1 = labels1.iter().fold(vec![0; 3], |mut acc, &l| {
        acc[l] += 1;
        acc
    });
    let count2 = labels2.iter().fold(vec![0; 3], |mut acc, &l| {
        acc[l] += 1;
        acc
    });
    assert_ne!(
        count1, count2,
        "Label distributions should be different for different seeds"
    );

    let centers1 = model1.centers.unwrap();
    let centers2 = model2.centers.unwrap();
    assert_ne!(
        centers1, centers2,
        "Centers should be different for different seeds"
    );
}
