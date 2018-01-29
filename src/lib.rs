#![cfg_attr(all(feature = "bench", test), feature(test))]

//! RustFFT allows users to compute arbitrary-sized FFTs in O(nlogn) time.
//!
//! The recommended way to use RustFFT is to create a [`FftPlanner`](struct.FftPlanner.html) instance and then call its
//! `plan_fft` method. This method will automatically choose which FFT algorithms are best
//! for a given size and initialize the required buffers and precomputed data.
//!
//! ```
//! // Perform a forward FFT of size 1234
//! use std::sync::Arc;
//! use rustfft::FftPlanner;
//! use rustfft::num_complex::Complex;
//! use rustfft::num_traits::Zero;
//!
//! let mut input:  Vec<Complex<f32>> = vec![Complex::zero(); 1234];
//! let mut output: Vec<Complex<f32>> = vec![Complex::zero(); 1234];
//!
//! let mut planner = FftPlanner::new(false);
//! let fft = planner.plan_fft(1234);
//! fft.process(&mut input, &mut output);
//!
//! // The fft instance returned by the planner is stored behind an `Arc`, so it's cheap to clone
//! let fft_clone = Arc::clone(&fft);
//! ```
//! The planner returns trait objects of the [`Fft`](trait.Fft.html) trait, allowing for FFT sizes that aren't known
//! until runtime.
//!
//! RustFFT also exposes individual FFT algorithms. If you know beforehand that you need a power-of-two FFT, you can
//! avoid the overhead of the planner and trait object by directly creating instances of the Radix4 algorithm:
//!
//! ```
//! // Computes a forward FFT of size 4096
//! use rustfft::algorithm::Radix4;
//! use rustfft::Fft;
//! use rustfft::num_complex::Complex;
//! use rustfft::num_traits::Zero;
//!
//! let mut input:  Vec<Complex<f32>> = vec![Complex::zero(); 4096];
//! let mut output: Vec<Complex<f32>> = vec![Complex::zero(); 4096];
//!
//! let fft = Radix4::new(4096, false);
//! fft.process(&mut input, &mut output);
//! ```
//!
//! For the vast majority of situations, simply using the [`FftPlanner`](struct.FftPlanner.html) will be enough, but
//! advanced users may have better insight than the planner into which algorithms are best for a specific size. See the
//! [`algorithm`](algorithm/index.html) module for a complete list of algorithms implemented by RustFFT.

pub extern crate num_complex;
pub extern crate num_traits;
extern crate num_integer;
extern crate strength_reduce;
extern crate transpose;



/// Individual FFT algorithms
pub mod algorithm;
mod math_utils;
mod array_utils;
mod plan;
mod twiddles;
mod common;

use num_complex::Complex;

pub use plan::FftPlanner;
pub use common::FftNum;



/// A trait that allows FFT algorithms to report their expected input/output size
pub trait Length {
    /// The FFT size that this algorithm can process
    fn len(&self) -> usize;
}

/// A trait that allows FFT algorithms to report whether they compute forward FFTs or inverse FFTs
pub trait IsInverse {
    /// Returns false if this instance computes forward FFTs, true for inverse FFTs
    fn is_inverse(&self) -> bool;
}

/// An umbrella trait for all available FFT algorithms
pub trait Fft<T: FftNum>: Length + IsInverse + Sync + Send {
    /// Computes an FFT on the `input` buffer and places the result in the `output` buffer.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]);

    /// Divides the `input` and `output` buffers into chunks of length self.len(), then computes an FFT on each chunk.
    ///
    /// This method uses the `input` buffer as scratch space, so the contents of `input` should be considered garbage
    /// after calling
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]);
}

#[cfg(test)]
extern crate rand;
#[cfg(test)]
mod test_utils;
