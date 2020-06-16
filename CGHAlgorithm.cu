#include "CGHAlgorithm.cuh"
#include <chrono>
#include <sstream>
#include "errorMessage.h"
//created by Felix Ronchen


static __global__ void update_slm_plane_array(
    cufftDoubleComplex* __restrict padded_array,
    const cufftDoubleComplex* __restrict unpadded_array,
    size_t number_of_pixels_padded, size_t NUMBER_OF_PIXELS_UNPADDED
);


static_assert(
    math_utils::is_power_of_two(NUMBER_OF_PIXELS_PADDED),
    "Padded 1D size must be a power of 2"
    );


CGHAlgorithm::CGHAlgorithm(
    const Parameters& params, statusBox* box
) :
    slm_pixel_size_mm((params.get_pixel_size_x_mm() + params.get_pixel_size_y_mm()) / 2.0),
    focal_length_px(params.get_focal_length_mm() / slm_pixel_size_mm),
    wavelength_px(params.get_wavelength_um() / (1000.0 * slm_pixel_size_mm)),
    max_iterations(params.get_max_iterations()),
    max_nonuniformity(params.get_max_nonuniformity_percent() / 100.0),
    fixed_phase_limit_iterations(params.get_fixed_phase_limit_iterations()),
    fixed_phase_limit_nonuniformity(params.get_fixed_phase_limit_nonuniformity_percent() / 100.0),
    weighting_parameter(params.get_weighting_parameter()),
    first_nonzero_index((NUMBER_OF_PIXELS_PADDED - NUMBER_OF_PIXELS_UNPADDED) / 2)
{
    init_cuda();
    editA = box;
    const auto beam_waist_x_mm = params.get_beam_waist_x_mm();
    const auto beam_waist_y_mm = params.get_beam_waist_y_mm();

    generate_beam_amplitude_array(beam_waist_x_mm, beam_waist_y_mm);
}


void CGHAlgorithm::init_cuda(
    void
) {
    // Not sure if it needs to be synchronized after every allocation
    //My experience is that once after all allocatons is fine ^^ - Max
    // because of e.g. overlapping memory region
    if (cudaSuccess != cudaMallocManaged(&beam_amplitudes, NUMBER_OF_PIXELS_UNPADDED * NUMBER_OF_PIXELS_UNPADDED * sizeof(cufftDoubleComplex))) {
        errBox("init_cuda: Could not allocate input_array", __FILE__, __LINE__);
        throw std::runtime_error("init_cuda: Could not allocate input_array");
    }
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    if (cudaSuccess != cudaMallocManaged(&slm_plane, NUMBER_OF_PIXELS_PADDED * NUMBER_OF_PIXELS_PADDED * sizeof(cufftDoubleComplex))) {
        errBox("init_cuda: Could not allocate in", __FILE__, __LINE__);
        throw std::runtime_error("init_cuda: Could not allocate in");
    }
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    if (cudaSuccess != cudaMallocManaged(&image_plane, NUMBER_OF_PIXELS_PADDED * NUMBER_OF_PIXELS_PADDED * sizeof(cufftDoubleComplex))) {
        errBox("init_cuda: Could not allocate out array", __FILE__, __LINE__);
        throw std::runtime_error("init_cuda: Could not allocate out array");
    }
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    if (CUFFT_SUCCESS != cufftPlan2d(&fft_plan, (int)NUMBER_OF_PIXELS_PADDED, (int)NUMBER_OF_PIXELS_PADDED, CUFFT_Z2Z)) {
        errBox("CGHAlgorithm: Could not setup FFT plan", __FILE__, __LINE__);
        throw std::runtime_error("CGHAlgorithm: Could not setup FFT plan");
    }
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);
}


CGHAlgorithm::~CGHAlgorithm(
    void
) noexcept {

    cudaFree(beam_amplitudes);
    cudaFree(slm_plane);
    cudaFree(image_plane);

    cufftDestroy(fft_plan);

    cudaDeviceSynchronize();
}


void CGHAlgorithm::generate_beam_amplitude_array(
    double sigma_x_mm, double sigma_y_mm
) noexcept {
    // Divide by sqrt(2) to match difference in gaussian beam definition
    // compared to standard gaussian (normal) distribution definition
    const double sigma_x = sigma_x_mm / slm_pixel_size_mm / sqrt(2.);
    const double sigma_y = sigma_y_mm / slm_pixel_size_mm / sqrt(2.);

    constexpr double phase = math_utils::PI() / 4.0;
    double total = 0.0;
    double amp;
    long long center = NUMBER_OF_PIXELS_UNPADDED / 2;
    for (int i = 0; i < (int)NUMBER_OF_PIXELS_UNPADDED; i++) {
        for (int j = 0; j < (int)NUMBER_OF_PIXELS_UNPADDED; j++) {
            //the radius has length=center so beam spans whole unpadded size
            if (math_utils::is_in_circle(j, i, center, center, center)) {
                amp = math_utils::gaussian2d(
                    // Explicit conversion prevents compiler warnings
                    (double)j, (double)i,
                    (double)NUMBER_OF_PIXELS_UNPADDED / 2.0, (double)NUMBER_OF_PIXELS_UNPADDED / 2.0,
                    sigma_x, sigma_y
                );
                beam_amplitudes[i * NUMBER_OF_PIXELS_UNPADDED + j].x = amp * cos(phase);
                beam_amplitudes[i * NUMBER_OF_PIXELS_UNPADDED + j].y = amp * sin(phase);
            }
            else {
                beam_amplitudes[i * NUMBER_OF_PIXELS_UNPADDED + j].x = 0.;
                beam_amplitudes[i * NUMBER_OF_PIXELS_UNPADDED + j].y = 0.;
            }
            total += math_utils::intensity(beam_amplitudes[i * NUMBER_OF_PIXELS_UNPADDED + j]);
        }
    }
    // Normalize again
    long end = NUMBER_OF_PIXELS_UNPADDED * NUMBER_OF_PIXELS_UNPADDED;
    for (size_t i = 0; i < end; i++) {
        beam_amplitudes[i].x /= sqrt(total);
        beam_amplitudes[i].y /= sqrt(total);
    }
}


std::vector<double> CGHAlgorithm::AWGS2D_loop(
    TweezerArray& tweezer_array,
    byte* phasemap_out
) {
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    const auto random_vec = generate_random_phase_distribution(seed);
    //I changed to time seed. -Max (old seed was 1).

    std::vector<double> non_uniformity_vec;//stores deviation from average trap intensity

    bool fix_phase = false; // since we aren't doing camera feedback from Englund paper yet

    set_initial_phase_distribution(random_vec.data());
    //writes upadded SLM_plane with random phase and 1/sqrt(2) amplitude

    size_t iteration = 0;
    for (;;) {

        // 1. Replace with input field (add gaussian beam)
        update_slm_plane_array << <NUM_BLOCKS_PADDED, BLOCK_SIZE >> >
            (slm_plane, beam_amplitudes, NUMBER_OF_PIXELS_PADDED, NUMBER_OF_PIXELS_UNPADDED);
        cuda_utils::cuda_synchronize(__FILE__, __LINE__);

        // 2. Execute FFT
        if (CUFFT_SUCCESS != cufftExecZ2Z(fft_plan, slm_plane, image_plane, CUFFT_FORWARD)) {
            errBox("AWGS2D_loop: Could not perform forward FFT.", __FILE__, __LINE__);
            throw std::runtime_error("AWGS2D_loop: Could not perform forward FFT.");
        }
        cuda_utils::cuda_synchronize(__FILE__, __LINE__);

        // 3. Extracted phases and amplitudes
        tweezer_array.update_current_intensities_and_phases(image_plane, fix_phase);


        // 3.1 Determine non-uniformity
        const double delta = tweezer_array.get_nonuniformity();
        const double mean_intensity = tweezer_array.get_mean_intensity();
        non_uniformity_vec.push_back(delta);

        // Check if goal/bounds are reached
        if ((delta < max_nonuniformity) && (iteration > 3)) {

            editA->appendMessage("Reached required unifprmity");
            std::cout << "\nReached required uniformity\n";
            break;
        }
        if (iteration == max_iterations) {
            editA->appendMessage("Reached limit of max iterations");
            std::cout << "\nReached limit of max iterations\n";
            break;
        }

        // 3.2 perform weighting
        // Not sure if this if is needed but sometimes the first iteration is very uniform
        // but uniformly bad
        const auto intensities = tweezer_array.get_intensities();
        const auto weights = calculate_weights(intensities, mean_intensity);

        tweezer_array.update_target_intensities(weights);


        // 3.3 Reset whole out array (all zeros)
        if (cudaSuccess != cudaMemset(image_plane, 0,
            NUMBER_OF_PIXELS_PADDED * NUMBER_OF_PIXELS_PADDED * sizeof(cufftDoubleComplex))) {
            errBox("AWGS2D_loop: Could not set out array to 0", __FILE__, __LINE__);
            throw std::runtime_error("AWGS2D_loop: Could not set out array to 0");
        }
        cuda_utils::cuda_synchronize(__FILE__, __LINE__);


        // 3.4 Fill values at tweezer sites
        tweezer_array.update_fft_array(image_plane);

        // 4. Execute iFFT

        if (CUFFT_SUCCESS != cufftExecZ2Z(fft_plan, image_plane, slm_plane, CUFFT_INVERSE)) {
            errBox("AWGS2D_loop: Could not perform inverse FFT.", __FILE__, __LINE__);
            throw std::runtime_error("AWGS2D_loop: Could not perform inverse FFT.");
        }
        cuda_utils::cuda_synchronize(__FILE__, __LINE__);

        // 4.1 Update iteration, display it, determine if phase should be fixed
        iteration++;
        std::stringstream stream;
        stream << std::setfill('0') << std::setw((long long)(log10(max_iterations) + 1))
            << iteration << "/" << max_iterations << "; ";
    
        std::cout << stream.str();
        editA->appendMessage(stream.str().c_str());

        stream.str(std::string());
        stream << "Non-uniformity: " << std::setfill('0') << std::setprecision(4)
            << 100 * delta << "%\n";

        std::cout << stream.str();

        editA->appendMessage(stream.str().c_str());

        if (!fix_phase && (iteration > fixed_phase_limit_iterations || delta < fixed_phase_limit_nonuniformity)) {
            std::cout << "\nFixed phase\n\n";
            editA->appendMessage("Fixed phase\n");
            fix_phase = true;
        }
    }

    // Fill result in phasemap array so it can be processed in main
    extract_final_phasemap(phasemap_out);

    return non_uniformity_vec;
}


double CGHAlgorithm::AWGS2D_camera_feedback_iteration(
    TweezerArray& tweezer_array,
    const byte* camera_image,
    byte* phasemap_out
) {
    set_initial_phase_distribution(phasemap_out);

    // 1. Replace with input field
    update_slm_plane_array << <NUM_BLOCKS_PADDED, BLOCK_SIZE >> >
        (slm_plane, beam_amplitudes, NUMBER_OF_PIXELS_PADDED, NUMBER_OF_PIXELS_UNPADDED);
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    // 2. Execute FFT
    if (CUFFT_SUCCESS != cufftExecZ2Z(fft_plan, slm_plane, image_plane, CUFFT_FORWARD)) {
        errBox("AWGS2D_camera_feedback_iteration: Could not perform forward FFT.", __FILE__, __LINE__);
        throw std::runtime_error("AWGS2D_camera_feedback_iteration: Could not perform forward FFT.");
    }
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    // 3. Extracted phases and amplitudes (amplitudes only for camera feedback
    tweezer_array.update_current_intensities_from_camera_image(camera_image);


    // 3.1 Determine non-uniformity

    const auto non_uniformity = tweezer_array.get_nonuniformity();
    const auto mean_intensity = tweezer_array.get_mean_intensity();


    // 3.2 perform weighting

    const auto intensities = tweezer_array.get_intensities();
    const auto weights = calculate_weights(intensities, mean_intensity);//sticking with kim weight?

    tweezer_array.update_target_intensities(weights);


    // 3.3 Reset whole out array
    if (cudaSuccess != cudaMemset(image_plane, 0,
        NUMBER_OF_PIXELS_PADDED * NUMBER_OF_PIXELS_PADDED * sizeof(cufftDoubleComplex))) {
        errBox("AWGS2D_camera_feedback_iteration: Could not set out array to 0", __FILE__, __LINE__);
        throw std::runtime_error("AWGS2D_camera_feedback_iteration: Could not set out array to 0");
    }
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    // 3.4 Fill values at tweezer sites
    tweezer_array.update_fft_array(image_plane);

    // 4. Execute iFFT
    cufftExecZ2Z(fft_plan, image_plane, slm_plane, CUFFT_INVERSE);
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    extract_final_phasemap(phasemap_out);

    return non_uniformity;
}


// Private stuff directly related to the main iterative loop:
// Note that update_fft_in_array runs on the device and is declared later
//why is random phase generated by grayscale first??
void CGHAlgorithm::set_initial_phase_distribution(
    const byte* initial_guess
) noexcept {
    double phase;
    int end_active_area = first_nonzero_index + NUMBER_OF_PIXELS_UNPADDED;
    //Max added end_active_area;
    for (size_t i = first_nonzero_index; i < end_active_area; i++) {
        for (size_t j = first_nonzero_index; j < end_active_area; j++) {
            phase = math_utils::grayscale_to_rad(
                initial_guess[(i - first_nonzero_index) * NUMBER_OF_PIXELS_UNPADDED + (j - first_nonzero_index)]
            );
            slm_plane[i * NUMBER_OF_PIXELS_PADDED + j].x = 1. / sqrt(2.0) * cos(phase);//constant 1/root2 amplitude?
            slm_plane[i * NUMBER_OF_PIXELS_PADDED + j].y = 1. / sqrt(2.0) * sin(phase);
        }
    }
}


std::vector<byte> CGHAlgorithm::generate_random_phase_distribution(
    int seed
) const noexcept {
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, 255);

    std::vector<byte> result(NUMBER_OF_PIXELS_UNPADDED * NUMBER_OF_PIXELS_UNPADDED);

    std::generate(result.begin(), result.end(), [&] {return (byte)distribution(generator); });

    editA->appendMessage("Generated random initial phase distribution");
    std::cout << "Generated random initial phase distribution\n";
    return result;
}


std::vector<double> CGHAlgorithm::calculate_weights(
    std::vector<double> values, double mean
) const noexcept {

    std::vector<double> weights(values.size());

    // Because weighting_parameter is not defined in the current scope
    // we either have to pass "this"(that means everything) or reassign the parameter
    std::transform(
        weights.begin(), weights.end(), values.cbegin(), weights.begin(),
        [weighting_parameter = weighting_parameter, mean](double w, const double v) {
            return 1.0 / (1.0 - weighting_parameter * (1.0 - v / mean));
        }
    );
    return weights;
}


void CGHAlgorithm::extract_final_phasemap(
    byte* phasemap
) const noexcept {

    //why don't you have to shift the slm plane?
    // Shorter names for better readability of the loop

    const size_t center = first_nonzero_index + NUMBER_OF_PIXELS_UNPADDED / 2;
    const size_t radius = NUMBER_OF_PIXELS_UNPADDED / 2;
    const size_t i1 = first_nonzero_index;
    const size_t N = NUMBER_OF_PIXELS_UNPADDED;
    const size_t N_padded = NUMBER_OF_PIXELS_PADDED;

    double phase;
    for (size_t i = i1; i < i1 + N; i++) {
        for (size_t j = i1; j < i1 + N; j++) {
            if (math_utils::is_in_circle(j, i, center, center, radius)) {
                phase = math_utils::phase(slm_plane[i * N_padded + j]);
                phasemap[(i - i1) * N + (j - i1)]
                    = math_utils::rad_to_grayscale(phase);
            }
            else {
                phasemap[(i - i1) * N + (j - i1)] = 0;
            }
        }
    }
}


// Copy the unpadded input into the center of the padded input array
// Similar to pad_array() in cuda_utils, maybe these two can be combined
//This should probably be updated to use 2D thread using dim() type for kernel call - Max
static __global__ void update_slm_plane_array(
    cufftDoubleComplex* __restrict padded_array,
    const cufftDoubleComplex* __restrict unpadded_array,
    size_t number_of_pixels_padded, size_t NUMBER_OF_PIXELS_UNPADDED
) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    const unsigned int x = (tid & (number_of_pixels_padded - 1));
    const unsigned int y = tid / number_of_pixels_padded;

    const unsigned int first_nonzero_index = (number_of_pixels_padded - NUMBER_OF_PIXELS_UNPADDED) / 2;


    // Type conversions necessary to allow for negative indices
    const bool center_site = math_utils::is_in_circle(
        x, y,
        number_of_pixels_padded / 2, number_of_pixels_padded / 2,
        NUMBER_OF_PIXELS_UNPADDED / 2
    );//shouldn't it not matter if unpadded or padded since unpadded is in middle of padded? -Max

    if (center_site) {

        const unsigned int x_unpadded = x - first_nonzero_index;
        const unsigned int y_unpadded = y - first_nonzero_index;
        const double phase = math_utils::phase(padded_array[tid]);
        const double amp = math_utils::amplitude(unpadded_array[y_unpadded * NUMBER_OF_PIXELS_UNPADDED + x_unpadded]);


        padded_array[tid].x = amp * cos(phase);
        padded_array[tid].y = amp * sin(phase);

    }
    else {
        padded_array[tid].x = 0.0;
        padded_array[tid].y = 0.0;//are you sure? don't we loose info by doing this? - Max
    }
}


// FileIO. These could be moved into basic_fileIO but for debugging
// this is easier as they know all the member variables
void CGHAlgorithm::save_input_intensity_distribution(
    const std::string& filename
) const {
    double max = 0.0;
    for (size_t i = first_nonzero_index; i < first_nonzero_index + NUMBER_OF_PIXELS_UNPADDED; i++) {
        for (size_t j = first_nonzero_index; j < first_nonzero_index + NUMBER_OF_PIXELS_UNPADDED; j++) {
            if (math_utils::intensity(slm_plane[i * NUMBER_OF_PIXELS_PADDED + j]) > max) {
                max = math_utils::intensity(slm_plane[i * NUMBER_OF_PIXELS_PADDED + j]);
            }
        }
    }
    std::vector<byte> result;
    for (size_t i = first_nonzero_index; i < first_nonzero_index + NUMBER_OF_PIXELS_UNPADDED; i++) {
        for (size_t j = first_nonzero_index; j < first_nonzero_index + NUMBER_OF_PIXELS_UNPADDED; j++) {
            result.push_back((byte)(255.0 * math_utils::intensity(slm_plane[i * NUMBER_OF_PIXELS_PADDED + j]) / max));
        }
    }
    basic_fileIO::save_as_bmp(filename, result.data(), NUMBER_OF_PIXELS_UNPADDED, NUMBER_OF_PIXELS_UNPADDED);
    editA->appendMessage("Saved input intensity distribution");
    std::cout << "Saved input intensity distribution\n";
}


void CGHAlgorithm::save_input_phase_distribution(
    const std::string& filename
) const {
    const size_t center = first_nonzero_index + NUMBER_OF_PIXELS_UNPADDED / 2;
    const size_t radius = NUMBER_OF_PIXELS_UNPADDED / 2;
    const size_t i1 = first_nonzero_index;
    const size_t N = NUMBER_OF_PIXELS_UNPADDED;
    const size_t N_padded = NUMBER_OF_PIXELS_PADDED;

    std::vector<byte> result(NUMBER_OF_PIXELS_UNPADDED * NUMBER_OF_PIXELS_UNPADDED);
    auto result_it = result.begin();
    double phase;
    for (size_t i = i1; i < i1 + N; i++) {
        for (size_t j = i1; j < i1 + N; j++) {
            if (math_utils::is_in_circle(j, i, center, center, radius)) {
                phase = math_utils::phase(slm_plane[i * N_padded + j]);
                *result_it = math_utils::rad_to_grayscale(phase);
            }
            else {
                *result_it = 0.0;
            }
            std::advance(result_it, 1);
        }
    }
    basic_fileIO::save_as_bmp(filename, result.data(), NUMBER_OF_PIXELS_UNPADDED, NUMBER_OF_PIXELS_UNPADDED);
    editA->appendMessage("Saved input phase distribution");
    std::cout << "Saved input phase distribution\n";
}


void CGHAlgorithm::save_output_intensity_distribution(
    const std::string& filename
) const {
    cufftDoubleComplex* out_shifted;
    if (cudaSuccess != cudaMallocManaged(&out_shifted, NUMBER_OF_PIXELS_PADDED * NUMBER_OF_PIXELS_PADDED * sizeof(cufftDoubleComplex))) {
        errBox("save_output_intensity_distribution: Could not allocate out-shifted", __FILE__, __LINE__);
        throw std::runtime_error("save_output_intensity_distribution: Could not allocate out-shifted");
    }

    if (cudaSuccess != cudaDeviceSynchronize()) {
        errBox("save_output_intensity_distribution: Could not synchronize", __FILE__, __LINE__);
        throw std::runtime_error("save_output_intensity_distribution: Could not synchronize");
    }

    cuda_utils::fft_shift << <NUM_BLOCKS_PADDED, BLOCK_SIZE >> > (out_shifted, image_plane, NUMBER_OF_PIXELS_PADDED, NUMBER_OF_PIXELS_PADDED);
    if (cudaSuccess != cudaDeviceSynchronize()) {
        errBox("save_output_intensity_distribution: Could not synchronize", __FILE__, __LINE__);
        throw std::runtime_error("save_output_intensity_distribution: Could not synchronize");
    }
    double max = 0.0;
    double temp;
    for (size_t i = first_nonzero_index; i < first_nonzero_index + NUMBER_OF_PIXELS_UNPADDED; i++) {
        for (size_t j = first_nonzero_index; j < first_nonzero_index + NUMBER_OF_PIXELS_UNPADDED; j++) {
            max = (std::max)(max, math_utils::intensity(out_shifted[i * NUMBER_OF_PIXELS_PADDED + j]));
        }
    }
    //Above is necessary since hte GS loop doesn't shift each itteration

    // Transform double intensity distribution to grayscale
    std::vector<byte> result(NUMBER_OF_PIXELS_UNPADDED * NUMBER_OF_PIXELS_UNPADDED);

    auto result_it = result.begin();

    for (size_t i = first_nonzero_index; i < first_nonzero_index + NUMBER_OF_PIXELS_UNPADDED; i++) {
        for (size_t j = first_nonzero_index; j < first_nonzero_index + NUMBER_OF_PIXELS_UNPADDED; j++) {
            *result_it = (byte)(255.0 * math_utils::intensity(out_shifted[i * NUMBER_OF_PIXELS_PADDED + j]) / max);
            std::advance(result_it, 1);
        }
    }
    if (cudaSuccess != cudaFree(out_shifted)) {
        errBox("save_output_intensity_distribution: Could not free out_shifted", __FILE__, __LINE__);
        throw std::runtime_error("save_output_intensity_distribution: Could not free out_shifted");
    }
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    basic_fileIO::save_as_bmp(filename, result.data(), NUMBER_OF_PIXELS_UNPADDED, NUMBER_OF_PIXELS_UNPADDED);
    editA->appendMessage("Saved output intensity distribution");
    std::cout << "Saved output intensity distribution\n";
}
