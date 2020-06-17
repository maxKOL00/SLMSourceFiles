#include "CGHAlgorithm.cuh"
#include "errorMessage.h"
//created by Felix Ronchen

static __global__ void substitute_amplitudes_d(
    cufftDoubleComplex* __restrict padded_amps_dst,
    const double* __restrict unpadded_amps_src,
    unsigned int number_of_pixels_padded, unsigned int number_of_pixels_unpadded, unsigned int center_circle_radius
);

static __global__ void substitute_phases_d(
    cufftDoubleComplex* __restrict padded_phases_dst,
    const double* __restrict unpadded_phases_src,
    unsigned int number_of_pixels_padded, unsigned int number_of_pixels_unpadded, unsigned int center_circle_radius
);

static __global__ void extract_amplitudes_d(
    double* __restrict unpadded_amps_dst,
    const cufftDoubleComplex* __restrict padded_amps_src,
    unsigned int number_of_pixels_padded, unsigned int number_of_pixels_unpadded, unsigned int center_circle_radius
);

static __global__ void extract_phases_d(
    double* __restrict unpadded_phases_dst,
    const cufftDoubleComplex* __restrict padded_phases_src,
    unsigned int number_of_pixels_padded, unsigned int number_of_pixels_unpadded, unsigned int center_circle_radius, bool add
);

static __global__ void extract_phases_to_grayscale_d(
    byte* __restrict unpadded_phases_dst,
    const cufftDoubleComplex* __restrict padded_phases_src,
    unsigned int number_of_pixels_padded, unsigned int number_of_pixels_unpadded, unsigned int center_circle_radius
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
    random_seed(params.get_random_seed()),
    number_of_pixels_padded(params.get_number_of_pixels_padded()),
    number_of_pixels_unpadded(params.get_number_of_pixels_unpadded()),
    block_size(params.get_block_size()),
    num_blocks_padded(params.get_num_blocks_padded()),
    first_nonzero_index(params.get_first_nonzero_index())
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
    if (cudaSuccess != cudaMallocManaged(&beam_amplitudes, number_of_pixels_unpadded * number_of_pixels_unpadded * sizeof(double))) {
        errBox("init_cuda: Could not allocate input_array", __FILE__, __LINE__);
        throw std::runtime_error("init_cuda: Could not allocate input_array");
    }

    if (cudaSuccess != cudaMallocManaged(&slm_plane, number_of_pixels_padded * number_of_pixels_padded * sizeof(cufftDoubleComplex))) {
        errBox("init_cuda: Could not allocate in", __FILE__, __LINE__);
        throw std::runtime_error("init_cuda: Could not allocate in");
    }

    if (cudaSuccess != cudaMallocManaged(&image_plane, number_of_pixels_padded * number_of_pixels_padded * sizeof(cufftDoubleComplex))) {
        errBox("init_cuda: Could not allocate out array", __FILE__, __LINE__);
        throw std::runtime_error("init_cuda: Could not allocate out array");
    }


    if (CUFFT_SUCCESS != cufftPlan2d(&fft_plan, (int)number_of_pixels_padded, (int)number_of_pixels_padded, CUFFT_Z2Z)) {
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

    double total = 0.0;
    double amp;

    for (int i = 0; i < (int)number_of_pixels_unpadded; i++) {
        for (int j = 0; j < (int)number_of_pixels_unpadded; j++) {

            if (math_utils::is_in_circle(j, i, number_of_pixels_unpadded / 2, number_of_pixels_unpadded / 2, number_of_pixels_unpadded / 2)) {
                amp = math_utils::gaussian2d(
                    // Explicit conversion prevents compiler warnings
                    (double)j, (double)i,
                    (double)number_of_pixels_unpadded / 2.0, (double)number_of_pixels_unpadded / 2.0,
                    sigma_x, sigma_y
                );
                beam_amplitudes[i * number_of_pixels_unpadded + j] = amp;
            }
            else {
                beam_amplitudes[i * number_of_pixels_unpadded + j] = 0.;
            }
            total += math_utils::intensity(beam_amplitudes[i * number_of_pixels_unpadded + j]);
        }
    }
    // Normalize again
    for (size_t i = 0; i < number_of_pixels_unpadded * number_of_pixels_unpadded; i++) {
        beam_amplitudes[i] /= sqrt(total);
    }
}

std::vector<double> CGHAlgorithm::AWGS2D_loop(
    TweezerArray& tweezer_array,
    byte* phasemap_out
) {
    const auto random_vec = generate_random_phase_distribution(random_seed);

    std::vector<double> non_uniformity_vec;

    bool fix_phase = false;

    set_initial_phase_distribution(random_vec.data());
    //writes upadded SLM_plane with random phase and 1/sqrt(2) amplitude

    unsigned int iteration = 0;
    for (;;) {

        // 1. Replace with input field (add gaussian beam)
        substitute_amplitudes_d << <num_blocks_padded, block_size >> >
            (slm_plane, beam_amplitudes, number_of_pixels_padded, number_of_pixels_unpadded, number_of_pixels_unpadded / 2);
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
        non_uniformity_vec.push_back(delta);

        // Check if goal/bounds are reached
        if ((delta < max_nonuniformity) && (iteration > 3)) {

            editA->appendMessage("Reached required uniformity");
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
        const double mean_intensity = tweezer_array.get_mean_intensity();
        const auto weights = calculate_weights(intensities, mean_intensity);

        tweezer_array.update_target_intensities(weights);


        // 3.3 Reset whole out array (all zeros)
        if (cudaSuccess != cudaMemset(image_plane, 0,
            number_of_pixels_padded * number_of_pixels_padded * sizeof(cufftDoubleComplex))) {
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
    extract_phases_to_grayscale_d << <num_blocks_padded, block_size >> >
        (phasemap_out, slm_plane, number_of_pixels_padded, number_of_pixels_unpadded, number_of_pixels_unpadded / 2);
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    return non_uniformity_vec;
}


double CGHAlgorithm::AWGS2D_camera_feedback_iteration(
    TweezerArray& tweezer_array,
    const byte* camera_image,
    byte* phasemap_out
) {
    set_initial_phase_distribution(phasemap_out);

    // 1. Replace with input field
    substitute_amplitudes_d << <num_blocks_padded, block_size >> >
        (slm_plane, beam_amplitudes, number_of_pixels_padded, number_of_pixels_unpadded, number_of_pixels_unpadded / 2);
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
        number_of_pixels_padded * number_of_pixels_padded * sizeof(cufftDoubleComplex))) {
        errBox("AWGS2D_camera_feedback_iteration: Could not set out array to 0", __FILE__, __LINE__);
        throw std::runtime_error("AWGS2D_camera_feedback_iteration: Could not set out array to 0");
    }
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    // 3.4 Fill values at tweezer sites
    tweezer_array.update_fft_array(image_plane);

    // 4. Execute iFFT
    cufftExecZ2Z(fft_plan, image_plane, slm_plane, CUFFT_INVERSE);
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    extract_phases_to_grayscale_d << <num_blocks_padded, block_size >> >
        (phasemap_out, slm_plane, number_of_pixels_padded, number_of_pixels_unpadded, number_of_pixels_unpadded / 2);
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    return non_uniformity;
}

std::vector<double> CGHAlgorithm::AWGS3D_loop(
    std::vector<TweezerArray>& tweezer_array3D,
    double layer_separation_um,
    byte* phasemap_out
) {
    std::vector<double> test;
    const auto random_vec = generate_random_phase_distribution(random_seed);

    std::vector<double> non_uniformity_vec;

    set_initial_phase_distribution(random_vec.data());

    // Main weighted GS loop
    // One dark layer between the two bright layers
    const size_t number_of_layers = tweezer_array3D.size(); // 2 * tweezer_array3D.size() - 1;
    const double layer_separation_px = layer_separation_um / (1000.0 * slm_pixel_size_mm);


    cufftDoubleComplex* in_inverted;
    double* phasemap_internal;
    // Allocate
    {
        if (cudaSuccess != cudaMallocManaged(&in_inverted, number_of_pixels_padded * number_of_pixels_padded * sizeof(cufftDoubleComplex))) {
            throw std::runtime_error("init_cuda: Could not allocate in_copy array");
        }
        cuda_utils::cuda_synchronize(__FILE__, __LINE__);

        if (cudaSuccess != cudaMallocManaged(&phasemap_internal,
            number_of_pixels_padded * number_of_pixels_padded * sizeof(double))) {
            throw std::runtime_error("init_cuda: Could not allocate phasemap array");
        }
        cuda_utils::cuda_synchronize(__FILE__, __LINE__);
    }

    size_t iteration = 0;
    for (;;) {

        // 1. Replace with input field
        substitute_amplitudes_d << <num_blocks_padded, block_size >> >
            (slm_plane, beam_amplitudes, number_of_pixels_padded, number_of_pixels_unpadded, number_of_pixels_unpadded / 2);
        cuda_utils::cuda_synchronize(__FILE__, __LINE__);


        substitute_phases_d << <num_blocks_padded, block_size >> >
            (slm_plane, phasemap_internal, number_of_pixels_padded, number_of_pixels_unpadded, number_of_pixels_unpadded / 2);
        cuda_utils::cuda_synchronize(__FILE__, __LINE__);

        if (cudaSuccess != cudaMemset(phasemap_internal, 0,
            number_of_pixels_padded * number_of_pixels_padded * sizeof(double))) {
            throw std::runtime_error("AWGS2D_loop: Could not set phasemap_internal array to 0");
        }
        cuda_utils::cuda_synchronize(__FILE__, __LINE__);

        // we could also just iterate over the vector in the for loop
        // But I am not sure yet if dark layers are needed
        auto tweezer_array_vec_it = tweezer_array3D.begin();

        for (size_t layer = 0; layer < number_of_layers; layer++) {

            // 2.1 First FFT + FFT of Fresnel propagator
            cuda_utils::simulate_two_FFTs_in_a_row << <num_blocks_padded, block_size >> >
                (in_inverted, slm_plane, number_of_pixels_padded, number_of_pixels_padded);
            cuda_utils::cuda_synchronize(__FILE__, __LINE__);

            // 2.2 Multiply by quadratic phase
            // It could be that dividing by f^2 is not precise enough for large layer separations, see documentation
            const double square_phase_factor = math_utils::PI() / wavelength_px * layer_separation_px * layer / pow(focal_length_px, 2.0);

            cuda_utils::multiply_by_quadratic_phase_factor << <num_blocks_padded, block_size >> >
                (in_inverted, number_of_pixels_padded, square_phase_factor);
            cuda_utils::cuda_synchronize(__FILE__, __LINE__);

            // 2.3 Inverse FFT
            if (CUFFT_SUCCESS != cufftExecZ2Z(fft_plan, in_inverted, image_plane, CUFFT_INVERSE)) {
                throw std::runtime_error("AWGS2D_loop: Could not perform inverse FFT.");
            }
            cuda_utils::cuda_synchronize(__FILE__, __LINE__);

            std::vector<double> vec(number_of_pixels_padded * number_of_pixels_padded);
            std::transform(vec.begin(), vec.end(), image_plane, vec.begin(), [](const auto& v1, auto& v2) {return math_utils::amplitude(v2); });
            std::stringstream ss;
            ss << "data/intensity_" << iteration << "_" << layer << ".bmp";
            basic_fileIO::save_as_bmp(ss.str(), vec.data(), number_of_pixels_padded, number_of_pixels_padded);

            // 3.1
            tweezer_array_vec_it->update_current_intensities_and_phases(image_plane, false);

            // 3.2 Determine non-uniformity
            const double delta = tweezer_array_vec_it->get_nonuniformity();
            std::cout << "Layer " << layer + 1 << " non-uniformity " << 100.0 * delta << "%\n";

            const double mean_intensity = tweezer_array_vec_it->get_mean_intensity();
            non_uniformity_vec.push_back(delta);

            // 3.3 Weight
            if (iteration > 0) {
                const auto intensities = tweezer_array_vec_it->get_intensities();
                const auto mean_intensity = tweezer_array_vec_it->get_mean_intensity();
                const auto weights = calculate_weights(intensities, mean_intensity);
                tweezer_array_vec_it->update_target_intensities(weights);
            }

            // 3.4 Reset whole out array
            if (cudaSuccess != cudaMemset(image_plane, 0,
                number_of_pixels_padded * number_of_pixels_padded * sizeof(cufftDoubleComplex))) {
                throw std::runtime_error("AWGS2D_loop: Could not set image_plane array to 0");
            }
            cuda_utils::cuda_synchronize(__FILE__, __LINE__);


            // 3.5 Fill values at tweezer sites
            tweezer_array_vec_it->update_fft_array(image_plane);

            // 4.1 FFT of propagator
            if (CUFFT_SUCCESS != cufftExecZ2Z(fft_plan, image_plane, in_inverted, CUFFT_FORWARD)) {
                throw std::runtime_error("AWGS2D_loop: Could not perform FFT.");
            }
            cuda_utils::cuda_synchronize(__FILE__, __LINE__);

            // 4.2 Multiply by quadratic phase
            cuda_utils::multiply_by_quadratic_phase_factor << <num_blocks_padded, block_size >> >
                (in_inverted, number_of_pixels_padded, -square_phase_factor);
            cuda_utils::cuda_synchronize(__FILE__, __LINE__);

            // 4.3 Two inverse FFTs
            cuda_utils::simulate_two_FFTs_in_a_row << <num_blocks_padded, block_size >> >
                (slm_plane, in_inverted, number_of_pixels_padded, number_of_pixels_padded);
            cuda_utils::cuda_synchronize(__FILE__, __LINE__);

            extract_phases_d << <num_blocks_padded, block_size >> >
                (phasemap_internal, slm_plane, number_of_pixels_padded, number_of_pixels_unpadded, number_of_pixels_unpadded / 2, true);
            cuda_utils::cuda_synchronize(__FILE__, __LINE__);

            basic_fileIO::save_as_bmp("phasemap.bmp", phasemap_internal, number_of_pixels_padded, number_of_pixels_padded);

            std::advance(tweezer_array_vec_it, 1);
        }

        iteration++;
        std::cout << iteration << "/" << max_iterations << "\n";
        if (iteration == max_iterations) {
            break;
        }
    }

    // Fill result in phasemap array so it can be processed in main
    extract_phases_to_grayscale_d << <num_blocks_padded, block_size >> >
        (phasemap_out, slm_plane, number_of_pixels_padded, number_of_pixels_unpadded, number_of_pixels_unpadded / 2);
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    basic_fileIO::save_as_bmp("phasemap.bmp", phasemap_out, number_of_pixels_unpadded, number_of_pixels_unpadded);

    cudaFree(in_inverted);
    cudaFree(phasemap_internal);
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    return test;
}


// Private stuff directly related to the main iterative loop:
// Note that update_fft_in_array runs on the device and is declared later
void CGHAlgorithm::set_initial_phase_distribution(
    const byte* initial_guess
) noexcept {
    double phase;
    for (auto i = first_nonzero_index; i < first_nonzero_index + number_of_pixels_unpadded; i++) {
        for (auto j = first_nonzero_index; j < first_nonzero_index + number_of_pixels_unpadded; j++) {
            phase = math_utils::grayscale_to_rad(
                initial_guess[(i - first_nonzero_index) * number_of_pixels_unpadded + (j - first_nonzero_index)]
            );
            slm_plane[i * number_of_pixels_padded + j].x = 1. / sqrt(2.0) * cos(phase);
            slm_plane[i * number_of_pixels_padded + j].y = 1. / sqrt(2.0) * sin(phase);
        }
    }
}

std::vector<byte> CGHAlgorithm::generate_random_phase_distribution(
    int seed
) const noexcept {
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, 255);

    std::vector<byte> result(number_of_pixels_unpadded * number_of_pixels_unpadded);

    std::generate(result.begin(), result.end(), [&] {return (byte)distribution(generator); });

    editA->appendMessage("Generated random initial phase distribution");
    std::cout << "Generated random initial phase distribution\n";
    return result;
}

std::vector<double> CGHAlgorithm::calculate_weights(
    const std::vector<double>& values, double mean
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

// FileIO. These could be moved into basic_fileIO but for debugging
// this is easier as they know all the member variables
void CGHAlgorithm::save_input_intensity_distribution(
    const std::string& filename
) const {
    double max = 0.0;
    for (size_t i = first_nonzero_index; i < first_nonzero_index + number_of_pixels_unpadded; i++) {
        for (size_t j = first_nonzero_index; j < first_nonzero_index + number_of_pixels_unpadded; j++) {
            if (math_utils::intensity(slm_plane[i * number_of_pixels_padded + j]) > max) {
                max = math_utils::intensity(slm_plane[i * number_of_pixels_padded + j]);
            }
        }
    }
    std::vector<byte> result;
    for (size_t i = first_nonzero_index; i < first_nonzero_index + number_of_pixels_unpadded; i++) {
        for (size_t j = first_nonzero_index; j < first_nonzero_index + number_of_pixels_unpadded; j++) {
            result.push_back((byte)(255.0 * math_utils::intensity(slm_plane[i * number_of_pixels_padded + j]) / max));
        }
    }
    basic_fileIO::save_as_bmp(filename, result.data(), number_of_pixels_unpadded, number_of_pixels_unpadded);
    editA->appendMessage("Saved input intensity distribution");
    std::cout << "Saved input intensity distribution\n";
}


void CGHAlgorithm::save_input_phase_distribution(
    const std::string& filename
) const {
    const size_t center = first_nonzero_index + number_of_pixels_unpadded / 2;
    const size_t radius = number_of_pixels_unpadded / 2;
    const size_t i1 = first_nonzero_index;
    const size_t N = number_of_pixels_unpadded;
    const size_t N_padded = number_of_pixels_padded;

    std::vector<byte> result(number_of_pixels_unpadded * number_of_pixels_unpadded);
    auto result_it = result.begin();
    double phase;
    for (size_t i = i1; i < i1 + N; i++) {
        for (size_t j = i1; j < i1 + N; j++) {
            if (math_utils::is_in_circle(j, i, center, center, radius)) {
                phase = math_utils::phase(slm_plane[i * N_padded + j]);
                *result_it = math_utils::rad_to_grayscale(phase);
            }
            else {
                *result_it = byte(0);
            }
            std::advance(result_it, 1);
        }
    }
    basic_fileIO::save_as_bmp(filename, result.data(), number_of_pixels_unpadded, number_of_pixels_unpadded);
    std::cout << "Saved input phase distribution\n";
}


void CGHAlgorithm::save_output_intensity_distribution(
    const std::string& filename
) const {
    cufftDoubleComplex* out_shifted;
    if (cudaSuccess != cudaMallocManaged(&out_shifted, number_of_pixels_padded * number_of_pixels_padded * sizeof(cufftDoubleComplex))) {
        errBox("save_output_intensity_distribution: Could not allocate out-shifted", __FILE__, __LINE__);
        throw std::runtime_error("save_output_intensity_distribution: Could not allocate out-shifted");
    }

    if (cudaSuccess != cudaDeviceSynchronize()) {
        errBox("save_output_intensity_distribution: Could not synchronize", __FILE__, __LINE__);
        throw std::runtime_error("save_output_intensity_distribution: Could not synchronize");
    }

    cuda_utils::fft_shift << <num_blocks_padded, block_size >> > (out_shifted, image_plane, number_of_pixels_padded, number_of_pixels_padded);
    if (cudaSuccess != cudaDeviceSynchronize()) {
        errBox("save_output_intensity_distribution: Could not synchronize", __FILE__, __LINE__);
        throw std::runtime_error("save_output_intensity_distribution: Could not synchronize");
    }
    double max = 0.0;
    for (size_t i = first_nonzero_index; i < first_nonzero_index + number_of_pixels_unpadded; i++) {
        for (size_t j = first_nonzero_index; j < first_nonzero_index + number_of_pixels_unpadded; j++) {
            max = (std::max)(max, math_utils::intensity(out_shifted[i * number_of_pixels_padded + j]));
        }
    }

    // Transform double intensity distribution to grayscale
    std::vector<byte> result(number_of_pixels_unpadded * number_of_pixels_unpadded);

    auto result_it = result.begin();

    for (size_t i = first_nonzero_index; i < first_nonzero_index + number_of_pixels_unpadded; i++) {
        for (size_t j = first_nonzero_index; j < first_nonzero_index + number_of_pixels_unpadded; j++) {
            *result_it = (byte)(255.0 * math_utils::intensity(out_shifted[i * number_of_pixels_padded + j]) / max);
            std::advance(result_it, 1);
        }
    }
    if (cudaSuccess != cudaFree(out_shifted)) {
        errBox("save_output_intensity_distribution: Could not free out_shifted", __FILE__, __LINE__);
        throw std::runtime_error("save_output_intensity_distribution: Could not free out_shifted");
    }
    cuda_utils::cuda_synchronize(__FILE__, __LINE__);

    basic_fileIO::save_as_bmp(filename, result.data(), number_of_pixels_unpadded, number_of_pixels_unpadded);
    editA->appendMessage("Saved output intensity distribution");
    std::cout << "Saved output intensity distribution\n";
}

static __global__ void substitute_amplitudes_d(
    cufftDoubleComplex* __restrict padded_amps_dst,
    const double* __restrict unpadded_amps_src,
    unsigned int number_of_pixels_padded, unsigned int number_of_pixels_unpadded, unsigned int center_circle_radius
) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    const unsigned int x = tid % number_of_pixels_padded;
    const unsigned int y = tid / number_of_pixels_padded;

    // Type conversions necessary to allow for negative indices
    const bool center_site = math_utils::is_in_circle(
        x, y,
        number_of_pixels_padded / 2, number_of_pixels_padded / 2,
        center_circle_radius
    );

    if (center_site) {

        const unsigned int first_nonzero_index = (number_of_pixels_padded - number_of_pixels_unpadded) / 2;

        const unsigned int x_unpadded = x - first_nonzero_index;
        const unsigned int y_unpadded = y - first_nonzero_index;
        const double phase = math_utils::phase(padded_amps_dst[tid]);
        const double amp = unpadded_amps_src[y_unpadded * number_of_pixels_unpadded + x_unpadded];

        padded_amps_dst[tid].x = amp * cos(phase);
        padded_amps_dst[tid].y = amp * sin(phase);

    }
    else {
        padded_amps_dst[tid].x = 0.0;
        padded_amps_dst[tid].y = 0.0;
    }
}

static __global__ void substitute_phases_d(
    cufftDoubleComplex* __restrict padded_phases_dst,
    const double* __restrict unpadded_phases_src,
    unsigned int number_of_pixels_padded, unsigned int number_of_pixels_unpadded, unsigned int center_circle_radius
) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    const unsigned int x = tid % number_of_pixels_padded;
    const unsigned int y = tid / number_of_pixels_padded;

    const unsigned int first_nonzero_index = (number_of_pixels_padded - number_of_pixels_unpadded) / 2;

    const bool center_site = math_utils::is_in_circle(
        x, y,
        number_of_pixels_padded / 2, number_of_pixels_padded / 2,
        center_circle_radius
    );

    if (center_site) {
        const auto x_unpadded = x - first_nonzero_index;
        const auto y_unpadded = y - first_nonzero_index;
        const double phase = unpadded_phases_src[y_unpadded * number_of_pixels_unpadded + x_unpadded];
        const double amp = math_utils::amplitude(padded_phases_dst[tid]);

        padded_phases_dst[tid].x = amp * cos(phase);
        padded_phases_dst[tid].y = amp * sin(phase);
    }
    else {
        padded_phases_dst[tid].x = 0.0;
        padded_phases_dst[tid].y = 0.0;
    }
}

[[maybe_unused]]
static __global__ void extract_amplitudes_d(
    double* __restrict unpadded_amps_dst,
    const cufftDoubleComplex* __restrict padded_amps_src,
    unsigned int number_of_pixels_padded, unsigned int number_of_pixels_unpadded, unsigned int center_circle_radius
) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    const unsigned int x = tid % number_of_pixels_padded;
    const unsigned int y = tid / number_of_pixels_padded;

    const unsigned int first_nonzero_index = (number_of_pixels_padded - number_of_pixels_unpadded) / 2;

    const bool center_site = math_utils::is_in_square(
        x, y,
        number_of_pixels_padded / 2, number_of_pixels_padded / 2,
        number_of_pixels_unpadded
    );

    if (center_site) {

        const unsigned int x_unpadded = x - first_nonzero_index;
        const unsigned int y_unpadded = y - first_nonzero_index;

        const bool center_circle_site = math_utils::is_in_circle(
            x, y,
            number_of_pixels_padded / 2, number_of_pixels_padded / 2,
            center_circle_radius
        );
        double val = 0.0;
        if (center_circle_site) {
            val = math_utils::phase(padded_amps_src[tid]);
        }
        unpadded_amps_dst[y_unpadded * number_of_pixels_unpadded + x_unpadded] = val;
    }
}

static __global__ void extract_phases_d(
    double* __restrict unpadded_phases_dst,
    const cufftDoubleComplex* __restrict padded_phases_src,
    unsigned int number_of_pixels_padded, unsigned int number_of_pixels_unpadded, unsigned int center_circle_radius, bool add
) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;

    const int x = tid % number_of_pixels_padded;
    const int y = tid / number_of_pixels_padded;

    const unsigned int first_nonzero_index = (number_of_pixels_padded - number_of_pixels_unpadded) / 2;

    const bool center_site = math_utils::is_in_square(
        x, y,
        number_of_pixels_padded / 2, number_of_pixels_padded / 2,
        number_of_pixels_unpadded
    );

    if (center_site) {

        const unsigned int x_unpadded = x - first_nonzero_index;
        const unsigned int y_unpadded = y - first_nonzero_index;

        const bool center_circle_site = math_utils::is_in_circle(
            x, y,
            number_of_pixels_padded / 2, number_of_pixels_padded / 2,
            center_circle_radius
        );
        if (center_circle_site) {
            double val = math_utils::phase(padded_phases_src[tid]);
            if (add) {
                unpadded_phases_dst[y_unpadded * number_of_pixels_unpadded + x_unpadded] += val;
            }
            else {
                unpadded_phases_dst[y_unpadded * number_of_pixels_unpadded + x_unpadded] = val;
            }
        }
        else {
            unpadded_phases_dst[y_unpadded * number_of_pixels_unpadded + x_unpadded] = 0;
        }
        // If add is false the value is overwritten otherwise its added
        // Without if:
        // unpadded_phases_dst[y_unpadded * number_of_pixels_unpadded + x_unpadded] = val + add * unpadded_phases_dst[y_unpadded * number_of_pixels_unpadded + x_unpadded];

    }
}

static __global__ void extract_phases_to_grayscale_d(
    byte* __restrict unpadded_phases_dst,
    const cufftDoubleComplex* __restrict padded_phases_src,
    unsigned int number_of_pixels_padded, unsigned int number_of_pixels_unpadded, unsigned int center_circle_radius
) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    const unsigned int x = tid % number_of_pixels_padded;
    const unsigned int y = tid / number_of_pixels_padded;

    const unsigned int first_nonzero_index = (number_of_pixels_padded - number_of_pixels_unpadded) / 2;

    const bool center_site = math_utils::is_in_square(
        x, y,
        number_of_pixels_padded / 2, number_of_pixels_padded / 2,
        number_of_pixels_unpadded
    );


    if (center_site) {

        const unsigned int x_unpadded = x - first_nonzero_index;
        const unsigned int y_unpadded = y - first_nonzero_index;

        const bool center_circle_site = math_utils::is_in_circle(
            x, y,
            number_of_pixels_padded / 2, number_of_pixels_padded / 2,
            center_circle_radius
        );
        byte val = 0;
        if (center_circle_site) {
            val = math_utils::rad_to_grayscale(math_utils::phase(padded_phases_src[tid]));
        }
        unpadded_phases_dst[y_unpadded * number_of_pixels_unpadded + x_unpadded] = val;
    }
}