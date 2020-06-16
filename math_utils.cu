#include "math_utils.cuh"

namespace math_utils {
    //void gaussian_filter_parallel(
    //    double* arr2d,
    //    size_t N_x, size_t N_y,
    //    size_t gauss_kernel_size,
    //    double sigma
    //) {
    //    if ((gauss_kernel_size & 1) == 0) {
    //        throw std::runtime_error("Kernel size should be odd");
    //    }
    //    cudaError_t err;
    //    double* result, *gauss_kernel;
    //    size_t block_size = 256;
    //    size_t num_blocks = (N_x * N_y + block_size - 1) / block_size;

    //    err = cudaMallocManaged(&result, N_x * N_y * sizeof(double));

    //    if (err != cudaSuccess) {
    //        throw std::runtime_error("gaussian_filter_parallel: could not allocate memory");
    //    }
    //    err = cudaMallocManaged(&gauss_kernel, gauss_kernel_size * gauss_kernel_size * sizeof(double));
    //    if (err != cudaSuccess) {
    //        throw std::runtime_error("gaussian_filter_parallel: could not allocate memory");
    //    }

    //    err = cudaDeviceSynchronize();
    //    if (err != cudaSuccess) {
    //        throw std::runtime_error("gaussian_filter_parallel: could not synchronize");
    //    }

    //     Fill kernel
    //    double sum = 0.0;
    //    for (int i = 0; i < gauss_kernel_size; i++) {
    //        for (int j = 0; j < gauss_kernel_size; j++) {
    //            gauss_kernel[i * gauss_kernel_size + j] = math_utils::gaussian2d(i, j, (gauss_kernel_size - 1) / 2, (gauss_kernel_size - 1) / 2, sigma, sigma);
    //            sum += gauss_kernel[i * gauss_kernel_size + j];
    //        }
    //    }

    //     Normalize
    //    for (int i = 0; i < gauss_kernel_size * gauss_kernel_size; i++) {
    //        gauss_kernel[i] /= sum;
    //    }

    //    gaussian_filter_parallel_kernel << <num_blocks, block_size >> > (result, arr2d, gauss_kernel, N_x, N_y, gauss_kernel_size);
    //    cudaDeviceSynchronize();
    //    memcpy(arr2d, result, N_x * N_y * sizeof(double));

    //    cudaFree(result);
    //}
   
    //__global__ void gaussian_filter_parallel_kernel(
    //    double* dst, const double* src,
    //    double* gaussian_kernel,
    //    size_t N_x, size_t N_y,
    //    int gaussian_kernel_size
    //) {
    //    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    //    if(tid < N_x * N_y) {
    //        int x_min, y_min, x_max, y_max;
    //        double sum = 0.0;

    //        int y = tid / N_x;
    //        int x = tid % N_x;

    //        y_min = max(-(int)gaussian_kernel_size / 2, 0 - y);
    //        x_min = max(-(int)gaussian_kernel_size / 2, 0 - x);

    //        y_max = min((int)gaussian_kernel_size / 2 + 1, (int)N_y - y);
    //        x_max = min((int)gaussian_kernel_size / 2 + 1, (int)N_x - x);

    //        sum = 0.0;

    //        for (int i_rel = y_min; i_rel < y_max; i_rel++) {
    //            for (int j_rel = x_min; j_rel < x_max; j_rel++) {
    //                sum += gaussian_kernel[(i_rel - y_min)
    //                    * gaussian_kernel_size + j_rel - x_min]
    //                    * src[(y + i_rel) * (int)N_x + (x + j_rel)];
    //            }
    //        }
    //        dst[y * N_x + x] = sum;
    //    }
    //}
}
