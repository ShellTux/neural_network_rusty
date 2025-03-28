use crate::matrix::Matrix;
use rand::{prelude::SliceRandom, rng, Rng};
use std::fmt::Debug;

#[derive(Debug)]
pub struct Perceptron {
    weights: Matrix<f32>,
    bias: f32,
    learning_rate: f32,
}

impl Perceptron {
    pub fn new(num_inputs: usize, learning_rate: f32) -> Self {
        let mut rng = rng();
        let weights = Matrix::from_vec(
            (0..num_inputs)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect(),
            1,
            num_inputs,
        )
        .unwrap();

        let bias = rng.random_range(-1.0..1.0);

        Perceptron {
            weights,
            bias,
            learning_rate,
        }
    }

    fn activation(&self, sum: f32) -> i32 {
        if sum >= 0.0 {
            1
        } else {
            0
        }
    }

    pub fn predict(&self, inputs: &Matrix<f32>) -> i32 {
        let weighted_sum = inputs.multiply(&self.weights.transpose()).unwrap().sum();
        self.activation(weighted_sum)
    }

    pub fn train(&mut self, inputs: &Vec<Matrix<f32>>, labels: &[i32], epochs: usize) {
        assert_eq!(inputs.len(), labels.len());

        for _ in 0..epochs {
            let mut rng = rng();
            let mut indices: Vec<usize> = (0..inputs.len()).collect();
            indices.shuffle(&mut rng);

            indices.iter().for_each(|&i| {
                let input = &inputs[i];
                let label = labels[i];

                let prediction = self.predict(input);
                let error = label - prediction;

                // Update weights and bias
                self.weights = self
                    .weights
                    .map(|x| x + self.learning_rate * error as f32 * x);
                self.bias += self.learning_rate * error as f32;
            });
        }
    }
}
