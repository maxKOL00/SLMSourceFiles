#include "TweezerArray.h"


TweezerArray::TweezerArray(
    const Parameters& params, statusBox *box
) {
    editT = box;
    geometry = params.get_array_geometry();
    num_traps_x = params.get_num_traps_x();
    num_traps_y = params.get_num_traps_y();

    num_traps = num_traps_x * num_traps_y;

    spacing_x_um = params.get_spacing_x_um();
    spacing_y_um = params.get_spacing_y_um();

    number_of_pixels_unpadded = (std::min)(params.get_slm_px_x(), params.get_slm_px_y());

    focal_length_um = params.get_focal_length_mm() * 1000.0;
    wavelength_um = params.get_wavelength_um();

    // delta_k in um in the padded Fourier spaced. It is used to
    // transform the desired spacings in um to a number in px
    const double delta_k_padded_uminv =
        double(number_of_pixels_unpadded) / double(NUMBER_OF_PIXELS_PADDED)
        * 1.0 / (1000.0 * params.get_sensor_size_y_mm());

    delta_x_padded_um = delta_k_padded_uminv * wavelength_um * focal_length_um;

    waist_um = params.get_waist_um();

    waist_px_in_fft = waist_um / delta_x_padded_um;
    waist_px_in_camera_image = waist_um / params.get_camera_px_size_um();

    tweezer_vec.resize(num_traps);


    const enum Geometry { rectangular, triangular, honeycomb, kagome, test };
    const std::map<std::string, Geometry> geometry_map = {
            {"RECTANGULAR", rectangular},
            {"TRIANGULAR", triangular},
            {"HONEYCOMB", honeycomb},
            {"KAGOME", kagome},
            {"TEST", test}
    };

    // Initialize array to type specified in parameter file
    switch (geometry_map.at(geometry)) {
        case rectangular:
            generate_rectangular_lattice();
            break;

        case triangular:
            generate_triangular_lattice();
            break;

        case honeycomb:
            generate_honeycomb_lattice();
            break;

        case kagome:
            generate_kagome_lattice();
            break;

        case test:
            generate_test_array();
            break;

        default:
            throw std::runtime_error("You should not be able to read this");
            break;
    }

    camera_px_h = params.get_camera_px_x();
    camera_px_v = params.get_camera_px_y();

    position_in_camera_image_set = false;
}

void TweezerArray::update_position_in_camera_image(
    const std::vector<size_t>& sorted_flattened_peak_indices
) noexcept {
    std::transform(
        tweezer_vec.begin(), tweezer_vec.end(), sorted_flattened_peak_indices.cbegin(), tweezer_vec.begin(),
        [camera_px_h = camera_px_h](auto& tweezer, const auto& p) {
            tweezer.position_in_camera_image.x = p % camera_px_h;
            tweezer.position_in_camera_image.y = p / camera_px_h;
            return tweezer;
        }
    );

    position_in_camera_image_set = true;
}

void TweezerArray::update_current_intensities_and_phases(
    const cufftDoubleComplex* fft_output,
    bool fix_phase
) noexcept {
    
    double sum_intensity_squares = 0.0;
    double total_intensity = 0.0;
    double total_amplitude = 0.0;
    
    for (auto& tweezer: tweezer_vec) {
        const size_t x = tweezer.position_in_fft_array.x;
        const size_t y = tweezer.position_in_fft_array.y;
        tweezer.current_intensity = math_utils::intensity(fft_output[y * NUMBER_OF_PIXELS_PADDED + x]);

        sum_intensity_squares += std::pow(tweezer.current_intensity, 2.0);
        total_intensity += tweezer.current_intensity;
        total_amplitude += sqrt(tweezer.current_intensity);

        if (!fix_phase) {
            tweezer.current_phase = math_utils::phase(fft_output[y * NUMBER_OF_PIXELS_PADDED + x]);
        }
    }
    mean_intensity = total_intensity / num_traps;
    mean_amplitude = total_amplitude / num_traps;

    standard_deviation = std::sqrt(sum_intensity_squares / (num_traps - 1) - std::pow(total_intensity, 2.0) / (num_traps * (num_traps - 1)));
}


void TweezerArray::update_current_intensities_from_camera_image(
    const byte* camera_image
) {
    if (!position_in_camera_image_set) {
        editT->appendColorMessage("update_current_intensities_from_camera_image: Positions not set,\
                                  call update_position_in_camera_image first", "red");
        throw std::runtime_error("update_current_intensities_from_camera_image: Positions not set,\
                                  call update_position_in_camera_image first");
    }
    const int local_intensity_radius = int(waist_px_in_camera_image + 1 + 0.5);

    const double sigma = waist_px_in_camera_image;

    // Calculate gaussian weights
    const auto gaussian_weights = calculate_gaussian_weights(local_intensity_radius, sigma);

    double sum_squares = 0.0;
    double total_intensity = 0.0;
    double total_amplitude = 0.0;
    
    for (auto& tweezer : tweezer_vec) {
        const size_t x = tweezer.position_in_camera_image.x;
        const size_t y = tweezer.position_in_camera_image.y;

        tweezer.current_intensity = get_local_intensity(
            camera_image, (int)x, (int)y, camera_px_h, camera_px_v, local_intensity_radius, gaussian_weights
        );
        sum_squares += std::pow(tweezer.current_intensity, 2.0);

        total_intensity += tweezer.current_intensity;
        total_amplitude += sqrt(tweezer.current_intensity);
    }
    mean_intensity = total_intensity / num_traps;
    mean_amplitude = total_amplitude / num_traps;
    standard_deviation = std::sqrt(sum_squares / (num_traps + 1) - std::pow(total_intensity, 2.0) / (num_traps * (num_traps + 1)));
}


void TweezerArray::update_target_intensities(
    const std::vector<double>& weights
) {
    if (weights.size() != num_traps) {
        editT->appendColorMessage("update_target_amplitudes: Invalid vector size","red");
        throw std::length_error("update_target_amplitudes: Invalid vector size");
    }

    double total = 0.0;

    std::transform(
        tweezer_vec.begin(), tweezer_vec.end(), weights.cbegin(), tweezer_vec.begin(),
        [&total](auto& tweezer, const auto& weight) {
            tweezer.target_intensity *= weight;
            total += tweezer.target_intensity;

            return tweezer;
        }
    );

    // Normalize again
    for (auto& tweezer: tweezer_vec) {
        tweezer.target_intensity /= total;
    };
}


size_t TweezerArray::get_array_size(
    void
) const noexcept{
    return num_traps;
} 


double TweezerArray::get_mean_intensity(
    void
) const noexcept {
    return mean_intensity;
}


double TweezerArray::get_mean_amplitude(
    void
) const noexcept {
    return mean_amplitude;
}


double TweezerArray::get_nonuniformity(
    void
) const noexcept {
    return standard_deviation / mean_intensity;
}


std::vector<double> TweezerArray::get_intensities(
    void
) const noexcept {
    std::vector<double> intensities(num_traps);
    std::transform(
        intensities.begin(), intensities.end(), tweezer_vec.cbegin(), intensities.begin(),
        [](double i, const Tweezer& t) {
            return t.current_intensity;
        }
    );
    return intensities;
}

std::vector<double> TweezerArray::get_amplitudes(
    void
) const noexcept {
    std::vector<double> amplitudes(num_traps);
    std::transform(
        amplitudes.begin(), amplitudes.end(), tweezer_vec.cbegin(), amplitudes.begin(),
        [](double a, const Tweezer& t) {
            return sqrt(t.current_intensity);
        }
    );
    return amplitudes;
}


void TweezerArray::update_fft_array(
    cufftDoubleComplex* fft_array
) const noexcept {
    unsigned int x, y;
    for (const auto& tweezer: tweezer_vec) {
        x = tweezer.position_in_fft_array.x;
        y = tweezer.position_in_fft_array.y;
        fft_array[y * NUMBER_OF_PIXELS_PADDED + x].x = 
            sqrt(tweezer.target_intensity) * cos(tweezer.current_phase);
        fft_array[y * NUMBER_OF_PIXELS_PADDED + x].y = 
            sqrt(tweezer.target_intensity) * sin(tweezer.current_phase);
    }
}


std::vector<double> TweezerArray::calculate_gaussian_weights(
    int local_intensity_radius, double sigma
) const noexcept {
    std::vector<double> gaussian_weights;

    double gaussian_total = 0.0;
    double temp;
    for (int i = -local_intensity_radius; i < local_intensity_radius + 1; i++) {
        for (int j = -local_intensity_radius; j < local_intensity_radius + 1; j++) {
            temp = 0.0;
            if (math_utils::is_in_circle(i, j, 0, 0, local_intensity_radius)) {
                temp = math_utils::gaussian2d(i, j, 0, 0, sigma / 2.0, sigma / 2.0);
            }
            gaussian_weights.push_back(temp);
            gaussian_total += temp;
        }
    }

    for (auto& w : gaussian_weights) {
        w /= gaussian_total;
    }

    return gaussian_weights;
}


// Different lattice types
void TweezerArray::generate_rectangular_lattice(
    void
) noexcept {
    const size_t spacing_x_px = 
        size_t(spacing_x_um / delta_x_padded_um) + 0.5;
    const size_t spacing_y_px =
        size_t(spacing_y_um / delta_x_padded_um) + 0.5;

    const size_t center = NUMBER_OF_PIXELS_PADDED / 2;
    const size_t offset_x = ((num_traps_x & 1) == 0) ? spacing_x_px / 2 : 0;//even #of traps centers between rows
    const size_t offset_y = ((num_traps_y & 1) == 0) ? spacing_y_px / 2 : 0;//odd #of traps centers on trap row

    const double total = (double)num_traps_x * num_traps_y;

    const int y_lower = -(int)(num_traps_y) / 2;
    const int y_upper = (int)(num_traps_y) / 2 + (int)(num_traps_y % 2);
    const int x_lower = -(int)(num_traps_x) / 2;
    const int x_upper = (int)(num_traps_x) / 2 + (int)(num_traps_x % 2);

    auto tweezer_vec_it = tweezer_vec.begin();

    for (int i = y_lower; i < y_upper; i++) {
        for (int j = x_lower; j < x_upper; j++) {
            size_t index_x = center + j * spacing_x_px + offset_x;
            size_t index_y = center + i * spacing_y_px + offset_y;

          
            index_x = (index_x + NUMBER_OF_PIXELS_PADDED / 2) % NUMBER_OF_PIXELS_PADDED;
            index_y = (index_y + NUMBER_OF_PIXELS_PADDED / 2) % NUMBER_OF_PIXELS_PADDED;

            tweezer_vec_it->position_in_fft_array.x = index_x;
            tweezer_vec_it->position_in_fft_array.y = index_y;
            tweezer_vec_it->target_intensity = 1.0 / total;
            tweezer_vec_it->current_intensity = 1.0 / total; //why divided by total?
            tweezer_vec_it->current_phase = 1.0;

            std::advance(tweezer_vec_it, 1);
        }
    }
    editT->appendMessage("Initialized square lattice");
    std::cout << "Initialized square lattice\n";
}

void TweezerArray::generate_triangular_lattice(
    void
) noexcept {
    const double spacing_um = spacing_x_um;
    const size_t spacing_x_px =
        size_t(spacing_um / delta_x_padded_um + 0.5);

    const size_t spacing_y_px =
        size_t(sqrt(3.0) / 2.0 * spacing_um / delta_x_padded_um + 0.5);

    const size_t center = NUMBER_OF_PIXELS_PADDED / 2;
    const size_t offset_x = ((num_traps_x & 1) == 0) ? spacing_x_px / 2 : 0;
    const size_t offset_y = ((num_traps_y & 1) == 0) ? spacing_y_px / 2 : 0;

    const double total = double(num_traps_x * num_traps_y);

    const int y_lower = -(int)(num_traps_y) / 2;
    const int y_upper = (int)(num_traps_y) / 2 + (int)(num_traps_y % 2);
    const int x_lower = -(int)(num_traps_x) / 2;
    const int x_upper = (int)(num_traps_x) / 2 + (int)(num_traps_x % 2);

    auto tweezer_vec_it = tweezer_vec.begin();

    for (int i = y_lower; i < y_upper; i++) {
        for (int j = x_lower; j < x_upper; j++) {
            size_t index_x = center + j * spacing_x_px + offset_x;

            // Shift odd rows by half the spacing
            if (!math_utils::is_even(i)) {
                index_x += spacing_x_px / 2;
            }
            size_t index_y = center + i * spacing_y_px + offset_y;
            index_x = (index_x + NUMBER_OF_PIXELS_PADDED / 2) % NUMBER_OF_PIXELS_PADDED;
            index_y = (index_y + NUMBER_OF_PIXELS_PADDED / 2) % NUMBER_OF_PIXELS_PADDED;
            tweezer_vec_it->position_in_fft_array.x = index_x;
            tweezer_vec_it->position_in_fft_array.y = index_y;
            tweezer_vec_it->target_intensity = 1.0 / total;
            tweezer_vec_it->current_intensity = 0.0;
            tweezer_vec_it->current_phase = 0.0;

            std::advance(tweezer_vec_it, 1);
        }
    }
    editT->appendMessage("Initialized triangular lattice");
    std::cout << "Initialized triangular lattice\n";
}

void TweezerArray::generate_honeycomb_lattice(
    void
) noexcept {

    const double spacing_um = spacing_x_um;
    const int spacing_x_px =
        int(spacing_um / delta_x_padded_um + 0.5);
    const int spacing_y_px =
        int(sqrt(3.0) / 2.0 * spacing_um / delta_x_padded_um + 0.5);

    const int N_padded_int = int(NUMBER_OF_PIXELS_PADDED);

    // Most of these ints can be converted to unsigned/size_t
    const int center = N_padded_int / 2;
    const int offset_x = -int(num_traps_x / 2 * spacing_x_px);

    const int offset_y = -int(num_traps_y / 2 * spacing_y_px);

    const int sites_x_max = int(num_traps_x + num_traps_x / 2);
    const int sites_y_max = int(num_traps_y);


    const double total = double(num_traps_x * num_traps_y);


    bool even_condition, odd_condition;

    auto tweezer_vec_it = tweezer_vec.begin();
    
    for (int i = 0; i < sites_y_max; i++) {
        for (int j = 0; j < sites_x_max; j++) {

            // Skip every third site but with different offsets for
            // even and odd rows
            even_condition = (math_utils::is_even(i) && ((j % 3) != 0));
            odd_condition = (!math_utils::is_even(i) && (((j + 2) % 3) != 0));

            if (even_condition || odd_condition) {

                unsigned int index_x = center + j * spacing_x_px + offset_x;

                if (odd_condition) {
                    index_x += spacing_x_px / 2;
                }

                unsigned int index_y = center + i * spacing_y_px + offset_y;

                index_x = (index_x + NUMBER_OF_PIXELS_PADDED / 2) % NUMBER_OF_PIXELS_PADDED;
                index_y = (index_y + NUMBER_OF_PIXELS_PADDED / 2) % NUMBER_OF_PIXELS_PADDED;
                tweezer_vec_it->position_in_fft_array.x = index_x;
                tweezer_vec_it->position_in_fft_array.y = index_y;
                tweezer_vec_it->target_intensity = 1.0 / total;
                tweezer_vec_it->current_intensity = 0.0;
                tweezer_vec_it->current_phase = 0.0;

                std::advance(tweezer_vec_it, 1);
            }
        }
    }
    editT->appendMessage("Initialized honey comb lattice");
    std::cout << "Initialized honey comb lattice\n";
}

void TweezerArray::generate_test_array(
    void
) {
    double total = 193.0;
    if (double(num_traps) != total) {
        throw std::runtime_error("For test image num_traps_x * num_traps_y must be equal to 193");
    }

    size_t spacing_x_px = 
        size_t(spacing_x_um / delta_x_padded_um + 0.5);
    size_t spacing_y_px = 
        size_t(spacing_y_um / delta_x_padded_um + 0.5);


    size_t center = NUMBER_OF_PIXELS_PADDED / 2;

    int jila[16][34] = {
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0}, 
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0}, 
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1},
        {1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    auto tweezer_vec_it = tweezer_vec.begin();

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 34; j++) {
            if (jila[i][j] > 0) {
                unsigned int index_x = center + j * spacing_x_px - 17 * spacing_x_px;
                unsigned int index_y = center + i * spacing_y_px - 8 * spacing_y_px;

                index_x = (index_x + NUMBER_OF_PIXELS_PADDED / 2) % NUMBER_OF_PIXELS_PADDED;
                index_y = (index_y + NUMBER_OF_PIXELS_PADDED / 2) % NUMBER_OF_PIXELS_PADDED;
                tweezer_vec_it->position_in_fft_array.x = index_x;
                tweezer_vec_it->position_in_fft_array.y = index_y;
                tweezer_vec_it->target_intensity = 1.0 / total;
                tweezer_vec_it->current_intensity = 0.0;
                tweezer_vec_it->current_phase = 0.0;
                std::advance(tweezer_vec_it, 1);
            }
        }
    }
    std::cout << "Initialized test image\n";
}

void TweezerArray::generate_kagome_lattice(
    void
) {
    editT->appendColorMessage("Kagome lattice not implemented", "red");
    throw std::runtime_error("Kagome lattice not implemented");
}
