#include "cuda_utils.cuh"


namespace cuda_utils {

    // Call cudaMemset on padded before inserting
    __global__ void pad_array(
        cufftDoubleComplex* __restrict padded_array,
        const cufftDoubleComplex* __restrict unpadded_array,
        unsigned int N_padded, unsigned int N
    ) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < N * N) {
            unsigned int x = (tid & (N_padded - 1));
            unsigned int y = tid / N_padded;
            unsigned int x_unpadded, y_unpadded;

            unsigned int first_nonzero_index = (N_padded - N) / 2;

            // Type conversions necessary to allow for negative indices
            bool center_site = (first_nonzero_index <= x)
                && (x < first_nonzero_index + N)
                && (first_nonzero_index <= y)
                && (y < first_nonzero_index + N);

            if (center_site) {
                x_unpadded = x - first_nonzero_index;
                y_unpadded = y - first_nonzero_index;
                padded_array[y * N_padded + x] = unpadded_array[y_unpadded * N + x_unpadded];
            }
        }
    }

    __global__ void unpad_array(
        const cufftDoubleComplex* __restrict padded_array,
        cufftDoubleComplex* __restrict unpadded_array,
        unsigned int N_padded, unsigned int N
    ) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        unsigned int x = (tid & (N_padded - 1));
        unsigned int y = tid / N_padded;
        unsigned int x_unpadded, y_unpadded;

        unsigned int first_nonzero_index = (N_padded - N) / 2;


        bool center_site = (first_nonzero_index <= x)
                        && (x < first_nonzero_index + N)
                        && (first_nonzero_index <= y)
                        && (y < first_nonzero_index + N);
                            

        if (center_site) {
            x_unpadded = x - first_nonzero_index;
            y_unpadded = y - first_nonzero_index;
            unpadded_array[y_unpadded * N + x_unpadded] = padded_array[y * N_padded + x];
        }
    }

    // Shift the FFT output such that both axis go from
    // -|k_max_i| to k_max_i for i in x, y
    __global__ void fft_shift(
        cufftDoubleComplex* arr_shifted,
        const cufftDoubleComplex* arr_unshifted,
        unsigned int N_x, unsigned int N_y
    ) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        unsigned int x = tid % N_x;
        unsigned int y = tid / N_x;

        unsigned int x_new = (x + N_x / 2) % N_x;
        unsigned int y_new = (y + N_y / 2) % N_y;

        arr_shifted[y_new * N_x + x_new] = arr_unshifted[y * N_x + x];
    }

    //__global__ void fresnel_propagator_rad(
    //    cufftDoubleComplex* dst,
    //    double delta_z_px,
    //    unsigned int number_of_pixels_padded,
    //    unsigned int number_of_pixels_unpadded,
    //    double focal_length_px, 
    //    double wavelength_px
    //) {
    //    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    //    
    //    const int x = tid % number_of_pixels_padded;
    //    const int y = tid / number_of_pixels_padded;
    //    
    //    const int x_rel = int(x - number_of_pixels_padded / 2);
    //    const int y_rel = int(y - number_of_pixels_padded / 2);


    //    if (math_utils::is_in_circle(x_rel, y_rel, 0, 0, number_of_pixels_unpadded / 2)) {

    //        // lens(-z) is not - lens(z) therefore the phase is substracted if delta_z < 0

    //        double sign = (delta_z_px < 0) ? -1.0: 1.0;
    //        delta_z_px = sign * delta_z_px;

    //        //double phase_rad =
    //        //    math_utils::PI() / wavelength_px * (1.0 / (focal_length_px + delta_z_px) - 1.0 / focal_length_px)
    //        //    * (pow(double(x_rel), 2.0) + pow(double(y_rel), 2.0));
    //            
    //        double phase_rad =
    //            math_utils::PI() / wavelength_px * (delta_z_px / pow(focal_length_px, 2.0))
    //            * (pow(double(x_rel), 2.0) + pow(double(y_rel), 2.0));

    //        // phase_rad = fmod(phase_rad, 2 * math_utils::PI());

    //        const double re = 1.0 / sqrt(2.0) * cos(sign * phase_rad);
    //        const double im = 1.0 / sqrt(2.0) * sin(sign * phase_rad);
    //                    
    //        const double re_prev = dst[tid].x;
    //        const double im_prev = dst[tid].y;

    //        dst[tid].x = (re * re_prev - im * im_prev);
    //        dst[tid].y = (re * im_prev + im * re_prev);
    //    }
    //}

    __global__ void multiply_by_quadratic_phase_factor(
        cufftDoubleComplex* dst,
        unsigned int number_of_pixels_padded,
        double c
    ) {
        const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
            
        const int x = tid % number_of_pixels_padded;
        const int y = tid / number_of_pixels_padded;
            
        const auto x_rel = x - int(number_of_pixels_padded) / 2;
        const auto y_rel = y - int(number_of_pixels_padded) / 2;

        const double phase_rad = c * (x_rel * x_rel + y_rel * y_rel);


        const double re = 1.0 / sqrt(2.0) * cos(phase_rad);
        const double im = 1.0 / sqrt(2.0) * sin(phase_rad);
                            
        const double re_prev = dst[tid].x;
        const double im_prev = dst[tid].y;

        dst[tid].x = (re * re_prev - im * im_prev);
        dst[tid].y = (re * im_prev + im * re_prev);
    }

    __global__ void shifted_intensity_distribution(
        double* __restrict dst,
        const cufftDoubleComplex* __restrict src,
        unsigned int N_x, unsigned int N_y
    ) {

        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < N_x * N_y) {
            unsigned int x = tid % N_x;
            unsigned int y = tid / N_x;

            unsigned int x_new = (x + N_x / 2) % N_x;
            unsigned int y_new = (y + N_y / 2) % N_y;

            dst[y_new * N_x + x_new] = math_utils::intensity(src[y * N_x + x]);
        }
    }

    __global__ void set_phase_only_array(
        cufftDoubleComplex* __restrict dst,
        const double* __restrict phasemap
    ) {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        dst[tid].x = 1.0 * cos(phasemap[tid]);
        dst[tid].y = 1.0 * sin(phasemap[tid]);
    }

    __global__ void extract_phasemap(
        double* __restrict dst,
        const cufftDoubleComplex* __restrict src,
        bool overwrite
    ) {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (overwrite) {
            dst[tid] = math_utils::phase(src[tid]);
        }
        else {
            dst[tid] += math_utils::phase(src[tid]);
        }
    }

    __global__ void reset_fft_array(
        cufftDoubleComplex* dst,
        double val
    ) {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        dst[tid].x = val;
        dst[tid].y = val;
    }

    __global__ void scale_array(
        double* dst,
        double scale_factor
    ) {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        dst[tid] *= scale_factor;
    }

    __global__ void scale_array(
        cufftDoubleComplex* dst,
        double scale_factor
    ) {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        dst[tid].x *= scale_factor;
        dst[tid].y *= scale_factor;
    }

    __global__ void add_phase(
        cufftDoubleComplex* __restrict dst,
        const double* __restrict src,
        unsigned int number_of_pixels_padded
    ) {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        const int x = tid % number_of_pixels_padded;
        const int y = tid / number_of_pixels_padded;

        const double phase_rad = src[tid];

        const double re = 1.0 / sqrt(2.0) * cos(phase_rad);
        const double im = 1.0 / sqrt(2.0) * sin(phase_rad);

        const double re_prev = dst[tid].x;
        const double im_prev = dst[tid].y;

        dst[tid].x = (re * re_prev - im * im_prev);
        dst[tid].y = (re * im_prev + im * re_prev);


    }

    double get_norm(
        const cufftDoubleComplex* src,
        unsigned int size
    ) {
        double total = 0.0;
        std::for_each(src, src + size, [&total](const auto& c) {
            total += math_utils::intensity(c);
        });
        return total;
    }

    __global__ void simulate_two_FFTs_in_a_row(
        cufftDoubleComplex* __restrict dst,
        const cufftDoubleComplex* __restrict src,
        unsigned int width, unsigned int height
    ) {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

        const unsigned int x_old = tid % width;
        const unsigned int y_old = tid / width;
        // To flip an array an index i gets mapped to length - i - 1 because indices run from 0 to length - 1.
        // However, applying an FFT twice results in the array flipped AND shifted by 1, so -1 + 1 = 0
        const unsigned int x_new = math_utils::mod(width - x_old, width);
        const unsigned int y_new = math_utils::mod(height - y_old, height);

        dst[y_new * width + x_new] = src[y_old * width + x_old];
    }

    void save_phasemap(
        const std::string& filename,
        const cufftDoubleComplex* arr,
        unsigned int width, unsigned int height
    ) {
        const auto extract_phase = [pi = math_utils::PI()](const cufftDoubleComplex& c) {
            return byte(255.0 * math_utils::phase(c) / (2 * pi));
        };
        basic_fileIO::save_as_bmp<cufftDoubleComplex>(filename, arr, width, height, extract_phase);
    }
}