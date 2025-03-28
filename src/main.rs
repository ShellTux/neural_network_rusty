mod matrix;

use std::f32::consts::PI;

use matrix::Matrix;
use raylib::prelude::*;

#[allow(unused_variables)]
fn main() {
    let (mut rl, thread) = raylib::init()
        .size(800, 600)
        .title("Matrix Visualization")
        .log_level(TraceLogLevel::LOG_NONE)
        .build();

    rl.set_target_fps(60);

    let min_value: f32 = -1.;
    let max_value: f32 = 1.;
    let matrix = Matrix::random(5, 5, (min_value, max_value));

    let mut phi: f32 = 0.;

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        matrix
            .map(|x| x * (phi * PI).sin())
            .visualize(&mut d, min_value, max_value);

        phi += 0.01;
    }
}
