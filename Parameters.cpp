#include "Parameters.h"
#include "basic_fileIO.h"

Parameters::Parameters(void) {
    std::ifstream config_fstream("C:\\Users\\maxko\\SLM-codebase-dev\\src\\config.json");
    if (!config_fstream.is_open()) {
        throw std::runtime_error("Could not open config.json");
    }
    config_fstream >> config;
    config_fstream.close();

}

size_t Parameters::get_slm_px_x(void) const {
    return config.at("SLM").at("SLM_PX_X");
}

size_t Parameters::get_slm_px_y(void) const {
    return config.at("SLM").at("SLM_PX_Y");
}

double Parameters::get_sensor_size_x_mm(void) const {
    return config.at("SLM").at("SENSOR_SIZE_X_MM");
}

double Parameters::get_sensor_size_y_mm(void) const {
    return config.at("SLM").at("SENSOR_SIZE_Y_MM");
}

double Parameters::get_pixel_size_x_mm(void) const {
    return get_sensor_size_x_mm() / get_slm_px_x();
}

double Parameters::get_pixel_size_y_mm(void) const {
    return get_sensor_size_y_mm() / get_slm_px_y();
}

size_t Parameters::get_frame_rate(void) const {
    return config.at("MEASUREMENT").at("FRAME_RATE");
}

size_t Parameters::get_patch_size_x_px(void) const {
    return config.at("MEASUREMENT").at("PATCH_SIZE_X_PX");
}

size_t Parameters::get_patch_size_y_px(void) const {
    return config.at("MEASUREMENT").at("PATCH_SIZE_Y_PX");
}

size_t Parameters::get_lut_patch_size_x_px(void) const {
    return config.at("CALIBRATION").at("LUT_PATCH_SIZE_X_PX");
}

size_t Parameters::get_lut_patch_size_y_px(void) const {
    return config.at("CALIBRATION").at("LUT_PATCH_SIZE_Y_PX");
}

size_t Parameters::get_number_of_lut_patches_x(void) const {
    
    return get_slm_px_x() / get_lut_patch_size_x_px();
}

size_t Parameters::get_number_of_lut_patches_y(void) const {
    return get_slm_px_y() / get_lut_patch_size_y_px();
}

size_t Parameters::get_horizontal_offset(void) const {
    return config.at("CALIBRATION").at("HORIZONTAL_OFFSET");
}

size_t Parameters::get_grating_period_px(void) const {
    return config.at("MEASUREMENT").at("GRATING_PERIOD_PX");
}

size_t Parameters::get_binary_grating_width_px(void) const {
    return config.at("MEASUREMENT").at("GRATING_WIDTH_PX");
}

byte Parameters::get_blazed_grating_max(void) const {
    return config.at("MEASUREMENT").at("GRATING_MAX");
}


std::string Parameters::get_camera_image_folder(void) const {
    const auto camera_image_folder =  config.at("MEASUREMENT").at("IMAGE_FOLDER").get<std::string>();
    basic_fileIO::create_nested_directory(camera_image_folder);
    if (camera_image_folder.find_last_of(std::string("\\/")) != camera_image_folder.size() - 1) {
        return camera_image_folder + "/";
    }
    return camera_image_folder;
}

std::string Parameters::get_pd_readout_folder(void) const {
    const auto pd_readout_folder = config.at("MEASUREMENT").at("PD_READOUT_FOLDER").get<std::string>();
    basic_fileIO::create_nested_directory(pd_readout_folder);
    if (pd_readout_folder.find_last_of(std::string("\\/")) != pd_readout_folder.size() - 1) {
        return pd_readout_folder + "/";
    }
    return pd_readout_folder;
}

double Parameters::get_axial_scan_range_lower_um(void) const {
    return config.at("MEASUREMENT").at("AXIAL_SCAN_RANGE_LOWER_UM");
}
double Parameters::get_axial_scan_range_upper_um(void) const {
    return config.at("MEASUREMENT").at("AXIAL_SCAN_RANGE_UPPER_UM");
}
double Parameters::get_axial_scan_stepsize_um(void) const {
    return config.at("MEASUREMENT").at("AXIAL_SCAN_STEPSIZE_UM");
}

std::string Parameters::get_camera_id(void) const {
    return config.at("CAMERA").at("CAMERA_ID").get<std::string>();
}

size_t Parameters::get_camera_max_frame_rate(void) const {
    return config.at("CAMERA").at("CAMERA_MAX_FRAME_RATE");
}

size_t Parameters::get_camera_px_x(void) const {
    return config.at("CAMERA").at("CAMERA_PX_X");
}

size_t Parameters::get_camera_px_y(void) const {
    return config.at("CAMERA").at("CAMERA_PX_Y");
}

double Parameters::get_camera_px_size_um(void) const {
    return config.at("CAMERA").at("CAMERA_PX_SIZE_UM");
}

std::string Parameters::get_exposure_mode(void) const {
    return config.at("CAMERA").at("EXPOSURE_MODE").get<std::string>();
}

double Parameters::get_exposure_time_us(void) const {
    return config.at("CAMERA").at("EXPOSURE_TIME_US");
}


std::string Parameters::get_serial_port_name(void) const {
    return config.at("SERIAL").at("PORT_NAME").get<std::string>();
}

double Parameters::get_focal_length_mm(void) const {
    return config.at("OPTICAL_SYSTEM").at("FOCAL_LENGTH_MM");
}

double Parameters::get_wavelength_um(void) const {
    return config.at("OPTICAL_SYSTEM").at("WAVELENGTH_UM");
}

double Parameters::get_waist_um(void) const {
    return config.at("OPTICAL_SYSTEM").at("WAIST_UM");
}

double Parameters::get_beam_waist_x_mm(void) const {
    const double res = config.at("OPTICAL_SYSTEM").at("BEAM_WAIST_X_MM");

    // Check if zero
    // Although a negative value is a questionable input
    // the absolute value can be returned in this case
    if (res == 0.0) {
        throw ParameterError("get_beam_waist_x_mm: Must not be 0");
    }
    return res > 0 ? res: -res;
}

double Parameters::get_beam_waist_y_mm(void) const {
    const double res = config.at("OPTICAL_SYSTEM").at("BEAM_WAIST_Y_MM");

    // Same argument as above
    if (res == 0.0) {
        throw ParameterError("get_beam_waist_y_mm: Must not be 0");
    }
    return res > 0 ? res : -res;
}

// TWEEZER_ARRAY_GENERATION

std::string Parameters::get_array_geometry(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("ARRAY_GEOMETRY").get<std::string>();
}


size_t Parameters::get_num_traps_x(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("NUM_TRAPS_X");
}
 
size_t Parameters::get_num_traps_y(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("NUM_TRAPS_Y");
}

double Parameters::get_spacing_x_um(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("SPACING_X_UM");
}

double Parameters::get_spacing_y_um(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("SPACING_Y_UM");
}

double Parameters::get_radial_shift_x_um(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("RADIAL_SHIFT_X_UM");
}

double Parameters::get_radial_shift_y_um(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("RADIAL_SHIFT_Y_UM");
}

double Parameters::get_axial_shift_um(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("AXIAL_SHIFT_UM");
}

bool Parameters::get_camera_feedback_enabled(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("CAMERA_FEEDBACK_ENABLED");
}


size_t Parameters::get_number_of_pixels_padded(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("NUMBER_OF_PIXELS_PADDED");
}

size_t Parameters::get_max_iterations(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("MAX_ITERATIONS");
}

size_t Parameters::get_max_iterations_camera_feedback(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("MAX_ITERATIONS_CAMERA_FEEDBACK");
}

size_t Parameters::get_fixed_phase_limit_iterations(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("FIXED_PHASE_LIMIT_ITERATIONS");
}

double Parameters::get_fixed_phase_limit_nonuniformity_percent(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("FIXED_PHASE_LIMIT_NONUNIFORMITY_PERCENT");
}

double Parameters::get_max_nonuniformity_percent(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("MAX_NONUNIFORMITY_PERCENT");
}

double Parameters::get_max_nonuniformity_camera_feedback_percent(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("MAX_NONUNIFORMITY_CAMERA_FEEDBACK_PERCENT");
}

double Parameters::get_weighting_parameter(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("WEIGHTING_PARAMETER");
}

double Parameters::get_layer_separation_um(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("LAYER_SEPARATION_UM");
}

std::string Parameters::get_output_folder(void) const {
    const auto output_folder = config.at("TWEEZER_ARRAY_GENERATION").at("OUTPUT_FOLDER").get<std::string>();
    basic_fileIO::create_nested_directory(output_folder);
    if (output_folder.find_last_of(std::string("\\/")) != output_folder.size() - 1) {
        return output_folder + "/";
    }
    return output_folder;
}

bool Parameters::get_save_data(void) const {
    return config.at("TWEEZER_ARRAY_GENERATION").at("SAVE_DATA");
}
