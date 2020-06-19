#include "Parameters.h"

namespace fs = std::filesystem;

Parameters::Parameters() {
    std::ifstream config_fstream("config.json");
    if (!config_fstream.is_open()) {
        throw std::runtime_error("Could not open config.json");
    }
    config_fstream >> config;
    config_fstream.close();

}

unsigned int Parameters::get_slm_px_x() const {
    return config.at("SLM").at("SLM_PX_X");
}

unsigned int Parameters::get_slm_px_y() const {
    return config.at("SLM").at("SLM_PX_Y");
}

double Parameters::get_sensor_size_x_mm() const {
    return config.at("SLM").at("SENSOR_SIZE_X_MM");
}

double Parameters::get_sensor_size_y_mm() const {
    return config.at("SLM").at("SENSOR_SIZE_Y_MM");
}

double Parameters::get_pixel_size_x_mm() const {
    return get_sensor_size_x_mm() / get_slm_px_x();
}

double Parameters::get_pixel_size_y_mm() const {
    return get_sensor_size_y_mm() / get_slm_px_y();
}

unsigned int Parameters::get_frame_rate() const {
    return config.at("CALIBRATION").at("FRAME_RATE");
}

unsigned int Parameters::get_patch_size_x_px() const {
    return config.at("CALIBRATION").at("PATCH_SIZE_X_PX");
}

unsigned int Parameters::get_patch_size_y_px() const {
    return config.at("CALIBRATION").at("PATCH_SIZE_Y_PX");
}

unsigned int Parameters::get_lut_patch_size_x_px() const {
    return config.at("CALIBRATION").at("LUT_PATCH_SIZE_X_PX");
}

unsigned int Parameters::get_lut_patch_size_y_px() const {
    return config.at("CALIBRATION").at("LUT_PATCH_SIZE_Y_PX");
}

unsigned int Parameters::get_number_of_lut_patches_x() const {

    return get_slm_px_x() / get_lut_patch_size_x_px();
}

unsigned int Parameters::get_number_of_lut_patches_y() const {
    return get_slm_px_y() / get_lut_patch_size_y_px();
}

unsigned int Parameters::get_horizontal_offset() const {
    return config.at("CALIBRATION").at("HORIZONTAL_OFFSET");
}

unsigned int Parameters::get_grating_period_px() const {
    return config.at("CALIBRATION").at("GRATING_PERIOD_PX");
}

byte Parameters::get_blazed_grating_max() const {
    return config.at("CALIBRATION").at("GRATING_MAX");
}


std::string Parameters::get_camera_image_folder() const {
    const auto camera_image_folder = config.at("CALIBRATION").at("IMAGE_FOLDER").get<std::string>();
    if (!fs::exists(camera_image_folder)) {
        fs::create_directories(camera_image_folder);
    }
    if (camera_image_folder.find_last_of(std::string("\\/")) != camera_image_folder.size() - 1) {
        return camera_image_folder + "/";
    }
    return camera_image_folder;
}

std::string Parameters::get_pd_readout_folder() const {
    const auto pd_readout_folder = config.at("CALIBRATION").at("PD_READOUT_FOLDER").get<std::string>();
    if (!fs::exists(pd_readout_folder)) {
        fs::create_directories(pd_readout_folder);
    }
    if (pd_readout_folder.find_last_of(std::string("\\/")) != pd_readout_folder.size() - 1) {
        return pd_readout_folder + "/";
    }
    return pd_readout_folder;
}

double Parameters::get_axial_scan_range_lower_um() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("AXIAL_SCAN").at("AXIAL_SCAN_RANGE_LOWER_UM");
}
double Parameters::get_axial_scan_range_upper_um() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("AXIAL_SCAN").at("AXIAL_SCAN_RANGE_UPPER_UM");
}
double Parameters::get_axial_scan_stepsize_um() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("AXIAL_SCAN").at("AXIAL_SCAN_STEPSIZE_UM");
}

std::string Parameters::get_camera_id() const {
    return config.at("CAMERA").at("CAMERA_ID").get<std::string>();
}

unsigned int Parameters::get_camera_max_frame_rate() const {
    return config.at("CAMERA").at("CAMERA_MAX_FRAME_RATE");
}

unsigned int Parameters::get_camera_px_x() const {
    return config.at("CAMERA").at("CAMERA_PX_X");
}

unsigned int Parameters::get_camera_px_y() const {
    return config.at("CAMERA").at("CAMERA_PX_Y");
}

double Parameters::get_camera_px_size_um() const {
    return config.at("CAMERA").at("CAMERA_PX_SIZE_UM");
}

std::string Parameters::get_exposure_mode() const {
    return config.at("CAMERA").at("EXPOSURE_MODE").get<std::string>();
}

double Parameters::get_exposure_time_us() const {
    const double res = config.at("CAMERA").at("EXPOSURE_TIME_US");
    if (res < 0) {
        throw std::runtime_error("Parameters::get_exposure_time_us(): Exposure time must be larger than 0");
    }
    return res;
}


std::string Parameters::get_serial_port_name() const {
    return config.at("SERIAL").at("PORT_NAME").get<std::string>();
}

double Parameters::get_focal_length_mm() const {
    return config.at("OPTICAL_SYSTEM").at("FOCAL_LENGTH_MM");
}

double Parameters::get_wavelength_um() const {
    return config.at("OPTICAL_SYSTEM").at("WAVELENGTH_UM");
}

double Parameters::get_waist_um() const {
    return config.at("OPTICAL_SYSTEM").at("WAIST_UM");
}

double Parameters::get_beam_waist_x_mm() const {
    const double res = config.at("OPTICAL_SYSTEM").at("BEAM_WAIST_X_MM");

    // Check if zero
    // Although a negative value is a questionable input
    // the absolute value can be returned in this case
    if (res == 0.0) {
        throw ParameterError("get_beam_waist_x_mm: Must not be 0");
    }
    return res > 0 ? res : -res;
}

double Parameters::get_beam_waist_y_mm() const {
    const double res = config.at("OPTICAL_SYSTEM").at("BEAM_WAIST_Y_MM");

    // Same argument as above
    if (res == 0.0) {
        throw ParameterError("get_beam_waist_y_mm: Must not be 0");
    }
    return res > 0 ? res : -res;
}

// TWEEZER_ARRAY_GENERATION

unsigned int Parameters::get_number_of_pixels_unpadded() const {
    return (std::min)(get_slm_px_x(), get_slm_px_y());
}
unsigned int Parameters::get_number_of_pixels_padded() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("NUMBER_OF_PIXELS_PADDED");
}
unsigned int Parameters::get_block_size() const {
    return 256;
}
unsigned int Parameters::get_num_blocks_padded() const {
    return (get_number_of_pixels_padded() * get_number_of_pixels_padded() + get_block_size() - 1) / get_block_size();
}
unsigned int Parameters::get_num_blocks_unpadded() const {
    return (get_number_of_pixels_unpadded() * get_number_of_pixels_unpadded() + get_block_size() - 1) / get_block_size();
}
unsigned int Parameters::get_num_blocks_slm() const {
    return (get_slm_px_x() * get_slm_px_y() + get_block_size() - 1) / get_block_size();
}
unsigned int Parameters::get_first_nonzero_index() const {
    return (get_number_of_pixels_padded() - get_number_of_pixels_unpadded()) / 2;
}


int Parameters::get_random_seed() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("RANDOM_SEED");
}


std::string Parameters::get_array_geometry() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("ARRAY_GEOMETRY").get<std::string>();
}


unsigned int Parameters::get_num_traps_x() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("NUM_TRAPS_X");
}

unsigned int Parameters::get_num_traps_y() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("NUM_TRAPS_Y");
}

double Parameters::get_spacing_x_um() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("SPACING_X_UM");
}

double Parameters::get_spacing_y_um() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("SPACING_Y_UM");
}

double Parameters::get_radial_shift_x_um() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("RADIAL_SHIFT_X_UM");
}

double Parameters::get_radial_shift_y_um() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("RADIAL_SHIFT_Y_UM");
}

double Parameters::get_axial_shift_um() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("AXIAL_SHIFT_UM");
}

bool Parameters::get_camera_feedback_enabled() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("CAMERA_FEEDBACK_ENABLED");
}

//
//unsigned int Parameters::get_number_of_pixels_padded() const {
//    return config.at("TWEEZER_ARRAY_GENERATION").at("NUMBER_OF_PIXELS_PADDED");
//}

unsigned int Parameters::get_max_iterations() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("MAX_ITERATIONS");
}

unsigned int Parameters::get_max_iterations_camera_feedback() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("MAX_ITERATIONS_CAMERA_FEEDBACK");
}

unsigned int Parameters::get_fixed_phase_limit_iterations() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("FIXED_PHASE_LIMIT_ITERATIONS");
}

double Parameters::get_fixed_phase_limit_nonuniformity_percent() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("FIXED_PHASE_LIMIT_NONUNIFORMITY_PERCENT");
}

double Parameters::get_max_nonuniformity_percent() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("MAX_NONUNIFORMITY_PERCENT");
}

double Parameters::get_max_nonuniformity_camera_feedback_percent() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("MAX_NONUNIFORMITY_CAMERA_FEEDBACK_PERCENT");
}

double Parameters::get_weighting_parameter() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("WEIGHTING_PARAMETER");
}

double Parameters::get_layer_separation_um() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("LAYER_SEPARATION_UM");
}

std::string Parameters::get_output_folder() const {
    const auto output_folder = config.at("TWEEZER_ARRAY_GENERATION").at("OUTPUT_FOLDER").get<std::string>();
    if (!fs::exists(output_folder)) {
        fs::create_directories(output_folder);
    }
    if (output_folder.find_last_of(std::string("\\/")) != output_folder.size() - 1) {
        return output_folder + "/";
    }
    return output_folder;
}

bool Parameters::get_save_data() const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("SAVE_DATA");
}
