//! Show how to use an `FFT` object from multiple threads

extern crate rustfft;

use std::thread;
use std::sync::Arc;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

fn main() {
    let inverse = false;
    let mut planner = FftPlanner::new(inverse);
    let fft = planner.plan_fft(100);

    let threads: Vec<thread::JoinHandle<_>> = (0..2).map(|_| {
        let fft_copy = Arc::clone(&fft);
        thread::spawn(move || {
            let mut signal = vec![Complex32::new(0.0, 0.0); 100];
            let mut spectrum = vec![Complex32::new(0.0, 0.0); 100];
            fft_copy.process(&mut signal, &mut spectrum);
        })
    }).collect();

    for thread in threads {
        thread.join().unwrap();
    }
}
