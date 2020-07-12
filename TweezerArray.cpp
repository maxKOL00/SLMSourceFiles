#include "TweezerArray.h"
#include "errorMessage.h"


TweezerArray::TweezerArray(
    const Parameters& params, statusBox *box
):
number_of_pixels_padded(params.get_number_of_pixels_padded()),
number_of_pixels_unpadded(params.get_number_of_pixels_unpadded())
{
    editT = box;
    geometry = params.get_array_geometry();
    num_traps_x = params.get_num_traps_x();
    num_traps_y = params.get_num_traps_y();

    num_traps = num_traps_x * num_traps_y;

    spacing_x_um = params.get_spacing_x_um();
    spacing_y_um = params.get_spacing_y_um();

    focal_length_um = params.get_focal_length_mm() * 1000.0;
    wavelength_um = params.get_wavelength_um();

    // delta_k in um in the padded Fourier spaced. It is used to
    // transform the desired spacings in um to a number in px
    const double delta_k_padded_uminv =
        double(number_of_pixels_unpadded) / double(number_of_pixels_padded)
        * 1.0 / (1000.0 * params.get_sensor_size_y_mm());

    delta_x_padded_um = delta_k_padded_uminv * wavelength_um * focal_length_um;

    waist_um = params.get_waist_um();

    waist_px_in_fft = waist_um / delta_x_padded_um;
    waist_px_in_camera_image = waist_um / params.get_camera_px_size_um();

    tweezer_vec.resize(num_traps);

    // C++ doesn't have string matching in switch statements so this is a little workaround
    const enum class Geometry { rectangular, triangular, honeycomb, kagome, test };
    const std::map<std::string, Geometry> geometry_map = {
            {"RECTANGULAR", Geometry::rectangular},
            {"TRIANGULAR", Geometry::triangular},
            {"HONEYCOMB", Geometry::honeycomb},
            {"KAGOME", Geometry::kagome},
            {"TEST", Geometry::test}
    };

    // Initialize array to type specified in parameter file
    switch (geometry_map.at(geometry)) {
        case Geometry::rectangular:
            generate_rectangular_lattice();
            break;

        case Geometry::triangular:
            generate_triangular_lattice();
            break;

        case Geometry::honeycomb:
            generate_honeycomb_lattice();
            break;

        case Geometry::kagome:
            generate_kagome_lattice();
            break;

        case Geometry::test:
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
    const std::vector<unsigned int>& sorted_flattened_peak_indices
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

    unsigned int x, y;

    for (auto& tweezer : tweezer_vec) {
        x = tweezer.position_in_fft_array.x;
        y = tweezer.position_in_fft_array.y;
        tweezer.current_intensity = math_utils::intensity(fft_output[y * number_of_pixels_padded + x]);

        sum_intensity_squares += std::pow(tweezer.current_intensity, 2.0);
        total_intensity += tweezer.current_intensity;
        total_amplitude += sqrt(tweezer.current_intensity);

        if (!fix_phase) {
            tweezer.current_phase = math_utils::phase(fft_output[y * number_of_pixels_padded + x]);
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
        errBox("update_current_intensities_from_camera_image: Positions not set,\
                                  call update_position_in_camera_image first", __FILE__, __LINE__);
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

    unsigned int x, y;

    for (auto& tweezer : tweezer_vec) {
        x = tweezer.position_in_camera_image.x;
        y = tweezer.position_in_camera_image.y;

        tweezer.current_intensity = get_local_intensity(
            camera_image, (int)x, (int)y, camera_px_h, camera_px_v, local_intensity_radius, gaussian_weights
        );
        sum_squares += std::pow(tweezer.current_intensity, 2.0);

        total_intensity += tweezer.current_intensity;
        total_amplitude += sqrt(tweezer.current_intensity);
    }
    mean_intensity = total_intensity / num_traps;
    mean_amplitude = total_amplitude / num_traps;
    standard_deviation = std::sqrt(sum_squares / (num_traps + 1u) - std::pow(total_intensity, 2.0) / (num_traps * (num_traps + 1u)));
}


void TweezerArray::update_target_intensities(
    const std::vector<double>& weights
) {
    if (weights.size() != num_traps) {
        errBox("update_target_intensities: Invalid vector size", __FILE__, __LINE__);
        throw std::length_error("update_target_intensities: Invalid vector size");
    }

    double total_intensity = 0.0;

    std::transform(
        tweezer_vec.begin(), tweezer_vec.end(), weights.cbegin(), tweezer_vec.begin(),
        [&total_intensity](auto& tweezer, const auto& weight) {
            tweezer.target_intensity *= weight;
            total_intensity += tweezer.target_intensity;

            return tweezer;
        }
    );

    // Normalize again
    std::transform(
        tweezer_vec.begin(), tweezer_vec.end(), tweezer_vec.begin(),
        [&total_intensity](auto& tweezer) {
            tweezer.target_intensity /= total_intensity;
            return tweezer;
        }
    );

    //for (auto& tweezer: tweezer_vec) {
    //    tweezer.target_intensity /= total;
    //};
}


unsigned int TweezerArray::get_array_size(
    void
) const noexcept {
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

    std::for_each(
        // std::execution::par,
        tweezer_vec.cbegin(), tweezer_vec.cend(),
        [&](const auto& tweezer) {
            const auto x = tweezer.position_in_fft_array.x;
            const auto y = tweezer.position_in_fft_array.y;
            fft_array[y * number_of_pixels_padded + x].x =
                sqrt(tweezer.target_intensity) * cos(tweezer.current_phase);
            fft_array[y * number_of_pixels_padded + x].y =
                sqrt(tweezer.target_intensity) * sin(tweezer.current_phase);
        }
    );
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

    std::transform(
        gaussian_weights.begin(), gaussian_weights.end(), gaussian_weights.begin(),
        [&gaussian_total](auto& weight) {
            return weight /= gaussian_total;
        }
    );

    //for (auto& w : gaussian_weights) {
    //    w /= gaussian_total;
    //}

    return gaussian_weights;
}

void TweezerArray::update_plot_positions(int center, int spacing_x_px, int offset_x,
                                         int spacing_y_px, int offset_y) {
    if (num_traps_x & 1) {
        x_start_plot = center - 20 - (((num_traps_x - 1) / 2) * spacing_x_px);
        x_stop_plot = center + 20 + (((num_traps_x - 1) / 2) * spacing_x_px);
    }
    else{ x_start_plot = center - 20 - (((num_traps_x - 1) / 2) * spacing_x_px) - (spacing_x_px / 2);
        x_stop_plot = center + 20 + (((num_traps_x - 1) / 2) * spacing_x_px) + (spacing_x_px / 2);
    }
    if (num_traps_y & 1) {
        y_start_plot = center - 20 - (((num_traps_y - 1) / 2) * spacing_y_px);
        y_stop_plot = center + 20 + (((num_traps_y - 1) / 2) * spacing_y_px);
    }
    else { y_start_plot = center - 20 - (((num_traps_y - 1) / 2) * spacing_y_px) - (spacing_y_px / 2);
        y_stop_plot = center + 20 + (((num_traps_y - 1) / 2) * spacing_y_px) + (spacing_y_px / 2);
    }
    if (y_start_plot < 0) { y_start_plot = 0; }
    if (x_start_plot < 0) { x_start_plot = 0; }
    if (x_stop_plot > ((number_of_pixels_unpadded/2)+center)) { x_stop_plot = number_of_pixels_unpadded; }
    if (y_stop_plot > ((number_of_pixels_unpadded / 2) + center)) { y_stop_plot = number_of_pixels_unpadded; }

}

// Different lattice types
void TweezerArray::generate_rectangular_lattice(
    void
) noexcept {
    const auto spacing_x_px = (unsigned int)(spacing_x_um / delta_x_padded_um + 0.5);
    const auto spacing_y_px = (unsigned int)(spacing_y_um / delta_x_padded_um + 0.5);

    const auto center = number_of_pixels_padded / 2;
    const auto offset_x = ((num_traps_x & 1) == 0) ? spacing_x_px / 2 : 0;
    const auto offset_y = ((num_traps_y & 1) == 0) ? spacing_y_px / 2 : 0;

    const double total = (double)num_traps_x * num_traps_y;

    const int y_lower = -(int)(num_traps_y) / 2;
    const int y_upper = (int)(num_traps_y) / 2 + (int)(num_traps_y % 2);
    const int x_lower = -(int)(num_traps_x) / 2;
    const int x_upper = (int)(num_traps_x) / 2 + (int)(num_traps_x % 2);

    /// <summary>
    update_plot_positions(center, spacing_x_px, offset_x,spacing_y_px, offset_y);

    ///

    auto tweezer_vec_it = tweezer_vec.begin();
    int index_x, index_y;
    for (int i = y_lower; i < y_upper; i++) {
        for (int j = x_lower; j < x_upper; j++) {

            index_x = center + j * spacing_x_px + offset_x;
            index_y = center + i * spacing_y_px + offset_y;


            index_x = (index_x + number_of_pixels_padded / 2) % number_of_pixels_padded;
            index_y = (index_y + number_of_pixels_padded / 2) % number_of_pixels_padded;

            tweezer_vec_it->position_in_fft_array.x = index_x;
            tweezer_vec_it->position_in_fft_array.y = index_y;
            tweezer_vec_it->target_intensity = 1.0 / total;
            tweezer_vec_it->current_intensity = 1.0 / total;
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
    const auto spacing_x_px = (unsigned int)(spacing_um / delta_x_padded_um + 0.5);

    const auto spacing_y_px = (unsigned int)(sqrt(3.0) / 2.0 * spacing_um / delta_x_padded_um + 0.5);

    const auto center = number_of_pixels_padded / 2;
    const auto offset_x = ((num_traps_x & 1) == 0) ? spacing_x_px / 2 : 0;
    const auto offset_y = ((num_traps_y & 1) == 0) ? spacing_y_px / 2 : 0;

    const double total = (double)num_traps_x * num_traps_y;

    const int y_lower = -(int)(num_traps_y) / 2;
    const int y_upper = (int)(num_traps_y) / 2 + (int)(num_traps_y % 2);
    const int x_lower = -(int)(num_traps_x) / 2;
    const int x_upper = (int)(num_traps_x) / 2 + (int)(num_traps_x % 2);

    auto tweezer_vec_it = tweezer_vec.begin();
    int index_x, index_y;
    for (int i = y_lower; i < y_upper; i++) {
        for (int j = x_lower; j < x_upper; j++) {
            index_x = center + j * spacing_x_px + offset_x;

            // Shift odd rows by half the spacing
            if (!math_utils::is_even(i)) {
                index_x += spacing_x_px / 2;
            }
            index_y = center + i * spacing_y_px + offset_y;
            index_x = (index_x + number_of_pixels_padded / 2) % number_of_pixels_padded;
            index_y = (index_y + number_of_pixels_padded / 2) % number_of_pixels_padded;
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
    const auto spacing_x_px = (unsigned int)(spacing_um / delta_x_padded_um + 0.5);
    const auto spacing_y_px = int(sqrt(3.0) / 2.0 * spacing_um / delta_x_padded_um + 0.5);

    const int N_padded_int = int(number_of_pixels_padded);

    // Most of these ints can be converted to unsigned/size_t
    const auto center = N_padded_int / 2;
    const auto offset_x = -int(num_traps_x / 2 * spacing_x_px);

    const auto offset_y = -int(num_traps_y / 2 * spacing_y_px);

    const auto sites_x_max = int(num_traps_x + num_traps_x / 2);
    const auto sites_y_max = int(num_traps_y);


    const double total = (double)num_traps_x * num_traps_y;


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

                index_x = (index_x + number_of_pixels_padded / 2) % number_of_pixels_padded;
                index_y = (index_y + number_of_pixels_padded / 2) % number_of_pixels_padded;
                tweezer_vec_it->position_in_fft_array.x = index_x;
                tweezer_vec_it->position_in_fft_array.y = index_y;
                tweezer_vec_it->target_intensity = 1.0 / total;
                tweezer_vec_it->current_intensity = 0.0;
                tweezer_vec_it->current_phase = 0.0;

                std::advance(tweezer_vec_it, 1);
            }
        }
    }
    editT->appendMessage("Initialized triangular lattice");
    std::cout << "Initialized honey comb lattice\n";
}

void TweezerArray::generate_test_array(
    void
) {
    double total = 193.0;
    if (double(num_traps) != total) {
        errBox("For test image num_traps_x * num_traps_y must be equal to 193", __FILE__, __LINE__);
        throw std::runtime_error("For test image num_traps_x * num_traps_y must be equal to 193");
    }

    const auto spacing_x_px = (unsigned int)(spacing_x_um / delta_x_padded_um + 0.5);
    const auto spacing_y_px = (unsigned int)(spacing_y_um / delta_x_padded_um + 0.5);


    const auto center = number_of_pixels_padded / 2;

    const auto N_x = 34;
    const auto N_y = 16;

    int jila[N_y][N_x] = {
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
    int index_x, index_y;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 34; j++) {
            if (jila[i][j] > 0) {
                index_x = center + j * spacing_x_px - (N_x / 2) * spacing_x_px;
                index_y = center + i * spacing_y_px - (N_y / 2) * spacing_y_px;

                index_x = (index_x + number_of_pixels_padded / 2) % number_of_pixels_padded;
                index_y = (index_y + number_of_pixels_padded / 2) % number_of_pixels_padded;
                tweezer_vec_it->position_in_fft_array.x = index_x;
                tweezer_vec_it->position_in_fft_array.y = index_y;
                tweezer_vec_it->target_intensity = 1.0 / total;
                tweezer_vec_it->current_intensity = 0.0;
                tweezer_vec_it->current_phase = 0.0;
                std::advance(tweezer_vec_it, 1);
            }
        }
    }
    editT->appendMessage("Initialized test image");
    std::cout << "Initialized test image\n";
}

void TweezerArray::generate_kagome_lattice(
    void
) {
    errBox("Kagome lattice not implemented", __FILE__, __LINE__);
    throw std::runtime_error("Kagome lattice not implemented");
}
