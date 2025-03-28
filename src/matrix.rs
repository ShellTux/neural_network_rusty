use rand::distr::uniform::SampleUniform;
use rand::Rng;
use raylib::color::Color;
use raylib::prelude::{RaylibDraw, RaylibDrawHandle};

use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug)]
pub struct Matrix<T> {
    data: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

pub trait One {
    fn one() -> Self;
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }
}

impl One for f64 {
    fn one() -> Self {
        1.0
    }
}

impl One for i32 {
    fn one() -> Self {
        1
    }
}

impl One for usize {
    fn one() -> Self {
        1
    }
}

#[allow(dead_code)]
impl<T> Matrix<T>
where
    T: Copy
        + Default
        + Debug
        + Sum
        + PartialOrd
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + One,
{
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![vec![T::default(); cols]; rows];
        Matrix { data, rows, cols }
    }

    pub fn identity(size: usize) -> Self {
        let mut identity_matrix = Matrix::new(size, size);
        for i in 0..size {
            identity_matrix.set(i, i, T::one());
        }
        identity_matrix
    }

    pub fn invert(&self) -> Result<Matrix<T>, String> {
        if self.rows != self.cols {
            return Err("Matrix must be square for inversion".to_string());
        }

        let size = self.rows;
        let mut augmented = Matrix::new(size, size * 2);

        // Create the augmented matrix [A | I]
        for i in 0..size {
            for j in 0..size {
                augmented.set(i, j, self.data[i][j]);
                if i == j {
                    augmented.set(i, j + size, T::one());
                } else {
                    augmented.set(i, j + size, T::default());
                }
            }
        }

        // Perform Gauss-Jordan elimination
        for i in 0..size {
            // Make the diagonal contain only 1's
            let divisor = augmented.get(i, i).unwrap();
            if divisor == T::default() {
                return Err("Matrix is singular and cannot be inverted".to_string());
            }

            for j in 0..size * 2 {
                let value = augmented.get(i, j).unwrap();
                augmented.set(i, j, value / divisor);
            }

            // Use this row to eliminate other rows
            for k in 0..size {
                if k != i {
                    let factor = augmented.get(k, i).unwrap();
                    for j in 0..size * 2 {
                        let value = augmented.get(k, j).unwrap();
                        augmented.set(k, j, value - factor * augmented.get(i, j).unwrap());
                    }
                }
            }
        }

        // Extract the inverted matrix from the augmented matrix
        let mut inverted_matrix = Matrix::new(size, size);
        for i in 0..size {
            for j in 0..size {
                inverted_matrix.set(i, j, augmented.get(i, j + size).unwrap().clone());
            }
        }

        Ok(inverted_matrix)
    }

    pub fn random(rows: usize, cols: usize, value_range: (T, T)) -> Self
    where
        T: SampleUniform + Default,
    {
        let mut rng = rand::rng();
        let data = (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| rng.random_range(value_range.0..=value_range.1))
                    .collect()
            })
            .collect();

        Matrix { data, rows, cols }
    }

    pub fn get(&self, row: usize, col: usize) -> Result<T, String> {
        if row < self.rows && col < self.cols {
            Ok(self.data[row][col])
        } else {
            Err("Out of bounds access".to_string())
        }
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) {
        if row < self.rows && col < self.cols {
            self.data[row][col] = value;
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn multiply(&self, other: &Matrix<T>) -> Result<Matrix<T>, String> {
        if self.cols != other.rows {
            return Err("Dimensions don't match: cols != rows".to_string());
        }

        let mut result = Matrix::new(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum + self.data[i][k] * other.data[k][j];
                }

                result.set(i, j, sum);
            }
        }

        Ok(result)
    }

    pub fn add(&self, other: &Matrix<T>) -> Result<Matrix<T>, String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Dimensions don't match".to_string());
        }

        let mut result = Matrix::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.data[i][j] + other.data[i][j]);
            }
        }

        Ok(result)
    }

    pub fn transpose(&self) -> Matrix<T> {
        let mut transposed = Matrix::new(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed.set(j, i, self.data[i][j]);
            }
        }

        transposed
    }

    pub fn map<U, F>(&self, f: F) -> Matrix<U>
    where
        F: Fn(T) -> U,
        U: Default
            + Copy
            + Debug
            + Add<Output = U>
            + Sub<Output = U>
            + Mul<Output = U>
            + Div<Output = U>
            + PartialOrd
            + Sum
            + One,
    {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = f(self.data[i][j]);
                result.set(i, j, value);
            }
        }
        result
    }

    pub fn print(&self) {
        for row in &self.data {
            for value in row {
                print!("{:#?} ", value);
            }
            println!();
        }
    }

    #[allow(dead_code)]
    pub fn visualize(&self, rdh: &mut RaylibDrawHandle<'_>, min_value: f32, max_value: f32)
    where
        f32: From<T>,
    {
        rdh.draw_text("Matrix visualization", 12, 12, 20, Color::WHITE);

        let cell_width: i32 = 50;
        let cell_height: i32 = 50;

        let start_x = (rdh.get_render_width() - (cell_width * self.cols() as i32)) / 2;
        let start_y = (rdh.get_render_height() - (cell_height * self.rows() as i32)) / 2;

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let value: f32 = self.get(i, j).unwrap().try_into().unwrap();
                let normalized = (value - min_value) / (max_value - min_value);

                // Use raylib's lerp function to interpolate between red and green
                let color = Color::RED.lerp(Color::LIME, normalized);

                let x = start_x + j as i32 * cell_width;
                let y = start_y + i as i32 * cell_height;
                rdh.draw_rectangle(x, y, cell_width, cell_height, color);
                rdh.draw_text(&value.to_string(), x + 10, y + 15, 20, Color::BLACK);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let matrix: Matrix<i32> = Matrix::new(3, 3);
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.data, vec![vec![0; 3]; 3]);
    }

    #[test]
    fn test_matrix_addition() {
        let mut matrix_a = Matrix::new(2, 2);
        matrix_a.set(0, 0, 1);
        matrix_a.set(0, 1, 2);
        matrix_a.set(1, 0, 3);
        matrix_a.set(1, 1, 4);

        let mut matrix_b = Matrix::new(2, 2);
        matrix_b.set(0, 0, 5);
        matrix_b.set(0, 1, 6);
        matrix_b.set(1, 0, 7);
        matrix_b.set(1, 1, 8);

        let result = matrix_a.add(&matrix_b).unwrap();
        assert_eq!(result.data, vec![vec![6, 8], vec![10, 12]]);
    }

    #[test]
    fn test_matrix_multiplication() {
        let mut matrix_a = Matrix::new(2, 3);
        matrix_a.set(0, 0, 1);
        matrix_a.set(0, 1, 2);
        matrix_a.set(0, 2, 3);
        matrix_a.set(1, 0, 4);
        matrix_a.set(1, 1, 5);
        matrix_a.set(1, 2, 6);

        let mut matrix_b = Matrix::new(3, 2);
        matrix_b.set(0, 0, 7);
        matrix_b.set(0, 1, 8);
        matrix_b.set(1, 0, 9);
        matrix_b.set(1, 1, 10);
        matrix_b.set(2, 0, 11);
        matrix_b.set(2, 1, 12);

        let result = matrix_a.multiply(&matrix_b).unwrap();
        assert_eq!(result.data, vec![vec![58, 64], vec![139, 154]]);
    }

    #[test]
    fn test_matrix_transpose() {
        let mut matrix = Matrix::new(2, 3);
        matrix.set(0, 0, 1);
        matrix.set(0, 1, 2);
        matrix.set(0, 2, 3);
        matrix.set(1, 0, 4);
        matrix.set(1, 1, 5);
        matrix.set(1, 2, 6);

        let transposed = matrix.transpose();
        assert_eq!(transposed.rows(), 3);
        assert_eq!(transposed.cols(), 2);
        assert_eq!(transposed.data, vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
    }

    #[test]
    fn test_matrix_map() {
        let mut matrix = Matrix::new(2, 2);
        matrix.set(0, 0, 1);
        matrix.set(0, 1, 2);
        matrix.set(1, 0, 3);
        matrix.set(1, 1, 4);

        // Define a simple function that squares each element
        let squared_matrix = matrix.map(|x| x * x);

        assert_eq!(squared_matrix.data, vec![vec![1, 4], vec![9, 16]]);
    }

    #[test]
    fn test_matrix_identity() {
        let identity_matrix = Matrix::<i32>::identity(3);

        assert_eq!(
            identity_matrix.data,
            vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]
        );
    }

    #[test]
    fn test_matrix_inversion() {
        let mut matrix: Matrix<f32> = Matrix::new(2, 2);
        matrix.set(0, 0, 4.);
        matrix.set(0, 1, 7.);
        matrix.set(1, 0, 2.);
        matrix.set(1, 1, 6.);

        let inverted = matrix.invert().unwrap();
        assert_eq!(inverted.data, vec![vec![0.6, -0.7], vec![-0.2, 0.4]]);
    }

    #[test]
    fn test_matrix_inversion_singular() {
        let mut matrix = Matrix::new(2, 2);
        matrix.set(0, 0, 1);
        matrix.set(0, 1, 2);
        matrix.set(1, 0, 2);
        matrix.set(1, 1, 4);

        let result = matrix.invert();
        assert!(result.is_err());
    }

    #[test]
    fn test_random_matrix_generation() {
        let rows: usize = 2;
        let cols: usize = 2;
        let min: usize = 1;
        let max: usize = 10;

        let random_matrix = Matrix::random(rows, cols, (min, max));
        assert_eq!(random_matrix.rows(), rows);
        assert_eq!(random_matrix.cols(), cols);
        assert!(random_matrix
            .data
            .iter()
            .all(|row| row.iter().all(|&v| min <= v && v <= max)));
    }

    #[test]
    fn test_matrix_get_set() {
        let mut matrix = Matrix::new(2, 2);
        matrix.set(0, 0, 1);
        assert_eq!(matrix.get(0, 0).unwrap(), 1);

        // Test out of bounds
        assert!(matrix.get(2, 2).is_err());
    }
}
