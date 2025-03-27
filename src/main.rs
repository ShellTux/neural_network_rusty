mod matrix;

use matrix::Matrix;
use raylib::prelude::*;

#[allow(unused_variables)]
fn main() {
    let a: Matrix<usize> = Matrix::random(1, 2, (1, 10));
    let b: Matrix<usize> = Matrix::random(2, 3, (1, 10));

    println!("A:");
    a.print();
    println!("B:");
    b.print();
    println!();
    a.multiply(&b).unwrap().print();

    let (mut rl, thread) = raylib::init()
        .size(800, 600)
        .title("Matrix Visualization")
        .build();

    let min_value: f32 = 0.;
    let max_value: f32 = 100.;
    let matrix = Matrix::random(5, 5, (min_value, max_value));

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        matrix.visualize(&mut d, min_value, max_value);
    }
}
