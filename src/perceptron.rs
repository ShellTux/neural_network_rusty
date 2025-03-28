use crate::matrix::Matrix;
use rand::{prelude::SliceRandom, rng, Rng};
use raylib::{
    color::Color,
    math::Vector2,
    prelude::{RaylibDraw, RaylibDrawHandle},
};
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

    pub fn visualize(&self, rdh: &mut RaylibDrawHandle<'_>, input_data: &Matrix<f32>) {
        let width = rdh.get_render_width();
        let height = rdh.get_render_height();

        let perceptron_pos = Vector2::new(width as f32 / 2., height as f32 / 2.);
        let perceptron_radius = 30.0; // Radius of the perceptron circle

        // Draw the perceptron circle with its value
        let perceptron_value = self.predict(&input_data);
        let perceptron_color = if perceptron_value == 1 {
            Color::GREEN
        } else {
            Color::RED
        };

        self.weights.foreachi(|weight, _, j| {
            let input = input_data.get(0, j).unwrap();

            let input_pos = Vector2::new(
                (perceptron_pos.x - 100.) as f32,
                (perceptron_pos.y as usize + j * 100 - 50) as f32,
            );

            let weight_thickness = (weight.abs() * 20.) as i32;

            rdh.draw_line_ex(
                input_pos,
                perceptron_pos,
                weight_thickness as f32,
                Color::BLUE,
            );

            rdh.draw_circle_v(input_pos, 20., Color::RED);
            rdh.draw_text(
                input.to_string().as_ref(),
                input_pos.x as i32 - 10,
                input_pos.y as i32 - 10,
                20,
                Color::WHITE,
            );
        });

        rdh.draw_circle_v(perceptron_pos, perceptron_radius, perceptron_color);
        rdh.draw_text(
            &perceptron_value.to_string(),
            perceptron_pos.x as i32 - 10,
            perceptron_pos.y as i32 - 10,
            20,
            Color::WHITE,
        );
    }
}
