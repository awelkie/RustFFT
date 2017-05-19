use std::sync::Arc;

use num_complex::Complex;
use common::{FFTnum, verify_length, verify_length_divisible};

use math_utils;
use array_utils;

use ::{Length, IsInverse, FFT};

/// Implementation of the [Good-Thomas Algorithm (AKA Prime Factor Algorithm)](https://en.wikipedia.org/wiki/Prime-factor_FFT_algorithm)
///
/// This algorithm factors a size n FFT into n1 * n2, where GCD(n1, n2) == 1
///
/// Conceptually, this algorithm is very similar to the Mixed-Radix FFT, except because GCD(n1, n2) == 1 we can do some
/// number theory trickery to reduce the number of floating-point multiplications and additions
///
/// ~~~
/// // Computes a forward FFT of size 1200, using the Good-Thomas Algorithm
/// use rustfft::algorithm::GoodThomasAlgorithm;
/// use rustfft::{FFT, FFTplanner};
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1200];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1200];
///
/// // we need to find an n1 and n2 such that n1 * n2 == 1200 and GCD(n1, n2) == 1
/// // n1 = 48 and n2 = 25 satisfies this
/// let mut planner = FFTplanner::new(false);
/// let inner_fft_n1 = planner.plan_fft(48);
/// let inner_fft_n2 = planner.plan_fft(25);
///
/// // the good-thomas FFT length will be inner_fft_n1.len() * inner_fft_n2.len() = 1200
/// let fft = GoodThomasAlgorithm::new(inner_fft_n1, inner_fft_n2);
/// fft.process(&mut input, &mut output);
/// ~~~
pub struct GoodThomasAlgorithm<T> {
    width: usize,
    // width_inverse: usize,
    width_size_fft: Arc<FFT<T>>,

    height: usize,
    // height_inverse: usize,
    height_size_fft: Arc<FFT<T>>,

    input_map: Box<[usize]>,
    output_map: Box<[usize]>,

    inverse: bool,
}

impl<T: FFTnum> GoodThomasAlgorithm<T> {
    /// Creates a FFT instance which will process inputs/outputs of size `n1_fft.len() * n2_fft.len()`
    ///
    /// GCD(n1.len(), n2.len()) must be equal to 1
    pub fn new(n1_fft: Arc<FFT<T>>, n2_fft: Arc<FFT<T>>) -> Self {
        assert_eq!(
            n1_fft.is_inverse(), n2_fft.is_inverse(), 
            "n1_fft and n2_fft must both be inverse, or neither. got n1 inverse={}, n2 inverse={}",
            n1_fft.is_inverse(), n2_fft.is_inverse());

        let n1 = n1_fft.len();
        let n2 = n2_fft.len();

        // compute the nultiplicative inverse of n1 mod n2 and vice versa
        let (gcd, mut n1_inverse, mut n2_inverse) =
            math_utils::extended_euclidean_algorithm(n1 as i64, n2 as i64);
        assert!(gcd == 1,
                "Invalid input n1 and n2 to Good-Thomas Algorithm: ({},{}): Inputs must be coprime",
                n1,
                n2);

        // n1_inverse or n2_inverse might be negative, make it positive
        if n1_inverse < 0 {
            n1_inverse += n2 as i64;
        }
        if n2_inverse < 0 {
            n2_inverse += n1 as i64;
        }

        // NOTE: we are precomputing the input and output reordering indexes
        // benchmarking shows that it's 20-30% faster
        // If we wanted to optimize for memory use or setup time instead of multiple-FFT speed,
        // these can be computed at runtime
        let input_map: Vec<usize> = 
            (0..n1 * n2)
                .map(|i| (i % n1, i / n1))
                .map(|(x, y)| (x * n2 + y * n1) % (n1 * n2))
                .collect();
        let output_map: Vec<usize> = 
            (0..n1 * n2)
                .map(|i| (i % n2, i / n2))
                .map(|(y, x)| {
                    (x * n2 * n2_inverse as usize + y * n1 * n1_inverse as usize) % (n1 * n2)
                })
                .collect();

        GoodThomasAlgorithm {
            inverse: n1_fft.is_inverse(),

            width: n1,
            width_size_fft: n1_fft,

            height: n2,
            height_size_fft: n2_fft,
            
            input_map: input_map.into_boxed_slice(),
            output_map: output_map.into_boxed_slice(),
        }
    }

    fn perform_fft(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {

        // copy the input using our reordering mapping
        for (output_element, &input_index) in output.iter_mut().zip(self.input_map.iter()) {
            *output_element = input[input_index];
        }

        // run FFTs of size `width`
        self.width_size_fft.process_multi(output, input);

        // transpose
        array_utils::transpose(self.width, self.height, input, output);

        // run FFTs of size 'height'
        self.height_size_fft.process_multi(output, input);

        // copy to the output, using our output redordeing mapping
        for (input_element, &output_index) in input.iter().zip(self.output_map.iter()) {
            output[output_index] = *input_element;
        }
    }
}

impl<T: FFTnum> FFT<T> for GoodThomasAlgorithm<T> {
    fn process(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length(input, output, self.len());

        self.perform_fft(input, output);
    }
    fn process_multi(&self, input: &mut [Complex<T>], output: &mut [Complex<T>]) {
        verify_length_divisible(input, output, self.len());

        for (in_chunk, out_chunk) in input.chunks_mut(self.len()).zip(output.chunks_mut(self.len())) {
            self.perform_fft(in_chunk, out_chunk);
        }
    }
}
impl<T> Length for GoodThomasAlgorithm<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.input_map.len()
    }
}
impl<T> IsInverse for GoodThomasAlgorithm<T> {
    #[inline(always)]
    fn is_inverse(&self) -> bool {
        self.inverse
    }
}


#[cfg(test)]
mod unit_tests {
    use super::*;
    use std::sync::Arc;
    use test_utils::check_fft_algorithm;
    use algorithm::DFT;

    #[test]
    fn test_good_thomas() {

        //gcd(n, n+1) is guaranteed to be 1, so we can generate some test sizes by just passing in n, n + 1
        for width in 2..20 {
            test_good_thomas_with_lengths(width, width - 1);
            test_good_thomas_with_lengths(width, width + 1);
        }

        //verify that it works correctly when width and/or height are 1
        test_good_thomas_with_lengths(1, 10);
        test_good_thomas_with_lengths(10, 1);
        test_good_thomas_with_lengths(1, 1);
    }

    fn test_good_thomas_with_lengths(width: usize, height: usize) {
        let width_fft = Arc::new(DFT::new(width, false)) as Arc<FFT<f32>>;
        let height_fft = Arc::new(DFT::new(height, false)) as Arc<FFT<f32>>;

        let fft = GoodThomasAlgorithm::new(width_fft, height_fft);

        check_fft_algorithm(&fft, width * height, false);
    }
}
