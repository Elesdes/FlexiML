use rand::Rng;

pub struct LinearModel {
    pub slope: f64,
    pub intercept: f64,
}

impl LinearModel {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        LinearModel {
            slope: rng.gen_range(-1.0..1.0),
            intercept: rng.gen_range(-1.0..1.0),
        }
    }

    pub fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    pub fn train(&mut self, x: &[f64], y: &[f64], learning_rate: f64, epochs: usize) {
        for _ in 0..epochs {
            for (&xi, &yi) in x.iter().zip(y.iter()) {
                let prediction = self.predict(xi);
                let error = prediction - yi;

                self.slope -= learning_rate * error * xi;
                self.intercept -= learning_rate * error;
            }
        }
    }
}
