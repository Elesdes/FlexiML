use ndarray::Array2;
use rand::prelude::*;
use std::ops::AddAssign;

pub struct RBFKMeans {
    pub n_clusters: usize,
    pub max_iter: usize,
    pub tol: f64,
    rng: StdRng,
    pub centers: Option<Array2<f64>>,
}

impl RBFKMeans {
    pub fn new(n_clusters: usize, max_iter: usize, tol: f64, random_state: Option<u64>) -> Self {
        let rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        RBFKMeans {
            n_clusters,
            max_iter,
            tol,
            rng,
            centers: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) {
        let (n_samples, n_features) = x.dim();

        // Initialize centers randomly
        let mut centers = Array2::zeros((self.n_clusters, n_features));
        let mut used_indices = vec![false; n_samples];
        for i in 0..self.n_clusters {
            loop {
                let idx = self.rng.gen_range(0..n_samples);
                if !used_indices[idx] {
                    centers.row_mut(i).assign(&x.row(idx));
                    used_indices[idx] = true;
                    break;
                }
            }
        }

        for _ in 0..self.max_iter {
            let labels = self.assign_labels(x, &centers);
            let new_centers = self.update_centers(x, &labels, n_features);

            let diff = (&new_centers - &centers).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                break;
            }

            centers = new_centers;
        }

        self.centers = Some(centers);
    }

    fn assign_labels(&self, x: &Array2<f64>, centers: &Array2<f64>) -> Vec<usize> {
        x.outer_iter()
            .map(|sample| {
                centers
                    .outer_iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let dist_a = (&sample - a).dot(&(&sample - a));
                        let dist_b = (&sample - b).dot(&(&sample - b));
                        dist_a.partial_cmp(&dist_b).unwrap()
                    })
                    .unwrap()
                    .0
            })
            .collect()
    }

    fn update_centers(
        &mut self,
        x: &Array2<f64>,
        labels: &[usize],
        n_features: usize,
    ) -> Array2<f64> {
        let mut new_centers = Array2::zeros((self.n_clusters, n_features));
        let mut counts = vec![0; self.n_clusters];

        for (sample, &label) in x.outer_iter().zip(labels.iter()) {
            new_centers.row_mut(label).add_assign(&sample);
            counts[label] += 1;
        }

        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                new_centers.row_mut(i).mapv_inplace(|x| x / *count as f64);
            } else {
                let idx = self.rng.gen_range(0..x.nrows());
                new_centers.row_mut(i).assign(&x.row(idx));
            }
        }

        new_centers
    }

    pub fn predict(&self, x: &Array2<f64>) -> Vec<usize> {
        match &self.centers {
            Some(centers) => self.assign_labels(x, centers),
            None => (0..x.nrows())
                .map(|_| self.rng.clone().gen_range(0..self.n_clusters))
                .collect(),
        }
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let centers = self.centers.as_ref().expect("Model not fitted yet");
        let n_samples = x.nrows();
        let mut result = Array2::zeros((n_samples, self.n_clusters));

        for (i, sample) in x.outer_iter().enumerate() {
            for (j, center) in centers.outer_iter().enumerate() {
                let dist = (&sample - &center).dot(&(&sample - &center));
                result[[i, j]] = (-dist).exp();
            }
        }

        result
    }
}
