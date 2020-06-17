#include "ImageProcessing.cuh"
#include "errorMessage.h"


// _d suffix for device code

static __global__ void insert_in_padded_d(
	cufftDoubleComplex* __restrict padded_array,
	const byte* __restrict unpadded_array,
	unsigned int N_padded, unsigned int N_x, unsigned int N_y, double norm
);

static __global__ void add_arrays_d(
	byte* __restrict p1, const byte* __restrict p2
);

// Generate a blazed grating
static __global__ void      blazed_grating_d(
	byte* result,
	unsigned int N_x, unsigned int N_y,
	unsigned int from_x,
	unsigned int period, byte phase_max
);

// Apply LUT and phase correction to the image
static __global__ void      correct_image_d(
	byte* __restrict result,
	unsigned int N_x, unsigned int N_y,
	unsigned int lut_patch_size_x,
	unsigned int lut_patch_size_y,
	const byte* __restrict phase_correction_ptr,
	const byte* __restrict lut_ptr
);


static __global__ void shift_fourier_image_d(
	byte* phasemap,
	unsigned int N,
	unsigned int shift_x_px, unsigned int shift_y_px
);


static __global__ void expand_to_sensor_size_d(
	byte* __restrict dst,
	const byte* __restrict src,
	unsigned int sensor_width_px,
	unsigned int number_of_pixels_unpadded, unsigned int horizontal_offset
);


ImageProcessing::ImageProcessing(const Parameters& params) {
	
	slm_px_x = params.get_slm_px_x();
	slm_px_y = params.get_slm_px_y();

	pixel_size_um = 1000.0 * (params.get_pixel_size_x_mm() + params.get_pixel_size_y_mm()) / 2.0;
	wavelength_um = params.get_wavelength_um();
	focal_length_um = 1000.0 * params.get_focal_length_mm();

	number_of_pixels_unpadded = slm_px_y;
	
	lut_patch_size_x = params.get_lut_patch_size_x_px();
	lut_patch_size_y = params.get_lut_patch_size_y_px();
	lut_patch_num_x = params.get_number_of_lut_patches_x();
	lut_patch_num_y = params.get_number_of_lut_patches_y();

	horizontal_offset = params.get_horizontal_offset();

	blazed_grating_period_px = params.get_grating_period_px();
	blazed_grating_max = params.get_blazed_grating_max();
	
	block_size = 256;
	num_blocks_slm = (unsigned int)(slm_px_x * slm_px_y + block_size - 1) / block_size;

}

std::vector<unsigned int> ImageProcessing::create_mask(
	const byte* image_data,
	unsigned int width, unsigned int height,
	unsigned int num_peaks_x, unsigned int num_peaks_y
) const {
	const auto num_peaks_tot = num_peaks_x * num_peaks_y;

	std::vector<double> image_data_smoothed(width * height);

	// Move hardcoded values to ctor or so
	math_utils::gaussian_filter(
		image_data_smoothed.data(), image_data, int(width), int(height),
		11, 2
	);

	const double max_value = *std::max_element(image_data_smoothed.cbegin(), image_data_smoothed.cend());

	auto peaks = math_utils::find_peaks2d(
		image_data_smoothed.cbegin(), image_data_smoothed.cend(),
		width,
		max_value / 2
	);


	if (peaks.size() != num_peaks_tot) {
		std::stringstream error_message;
		error_message << "Wrong number of peaks detected, ";
		error_message << peaks.size() << " instead of " << num_peaks_tot;
		errBox(error_message.str().c_str(), __FILE__, __LINE__);
		throw std::length_error(error_message.str());
	}

	math_utils::sort_row_wise_by_x_coordinate(
		peaks.begin(), peaks.end(),
		width, num_peaks_x
	);

	return peaks;
};

void ImageProcessing::shift_fourier_image(
	byte* phasemap, 
	double shift_x_um, double shift_y_um
) const noexcept{
	const size_t a_x_px = 255 * shift_x_um * pixel_size_um / (focal_length_um * wavelength_um);
	const size_t a_y_px = 255 * shift_y_um * pixel_size_um / (focal_length_um * wavelength_um);

	const auto num_blocks_square = (unsigned int)(slm_px_y * slm_px_y + block_size - 1) / block_size;

	shift_fourier_image_d<<<num_blocks_square, block_size>>>
		(phasemap, slm_px_y, a_x_px, a_y_px);
	cuda_utils::cuda_synchronize(__FILE__, __LINE__);
}


// Error checking needed
void ImageProcessing::expand_to_sensor_size(
	byte* dst, const byte* src
) const {
	expand_to_sensor_size_d<<<num_blocks_slm, block_size>>>
		(dst, src, slm_px_x, number_of_pixels_unpadded, horizontal_offset);
	cuda_utils::cuda_synchronize(__FILE__, __LINE__);
}


std::tuple<std::vector<byte>, size_t, size_t> ImageProcessing::crop_tweezer_array_image(
	const byte* image_data,
	size_t width, size_t height,
	size_t num_peaks_x, size_t num_peaks_y
) const {
	const auto mask = create_mask(image_data, width, height, num_peaks_x, num_peaks_y);
	

	const size_t offset_x = 60;
	const size_t offset_y = 60;

	// Get x,y coordinates of first and last peak
	// x is not const as it might need some padding because a bitmap width
	// must be divible by 4

	const size_t x_start = (std::max)(int(mask.at(0) % width - offset_x), 0);
	size_t x_end = (std::max)(int(mask.at(mask.size() - 1) % width + offset_x), 0);

	x_end += math_utils::mod(4 - math_utils::mod((x_end - x_start), 4), 4);

	const size_t y_start = (std::max)(int(mask.at(0) / width - offset_y), 0);
	const size_t y_end = (std::max)(int(mask.at(mask.size() - 1) / width + offset_y), 0);
	
	const size_t cropped_width = x_end - x_start;
	const size_t cropped_height = y_end - y_start;

	std::vector<byte> result(cropped_width * cropped_height);

	auto result_it = result.begin();
	
	for (size_t y = y_start; y < y_end; y ++) {
		for (size_t x = x_start; x < x_end; x++) {
			*result_it = image_data[y * width + x];
			std::advance(result_it, 1);
		}
	}

	return std::make_tuple(result, cropped_width, cropped_height);
}

// Error checking needed
void ImageProcessing::add_blazed_grating(
	byte* dst
) const {
	blazed_grating_d << <num_blocks_slm, block_size >> >
		(dst, slm_px_x, slm_px_y, horizontal_offset,
			blazed_grating_period_px, blazed_grating_max
			);
	cuda_utils::cuda_synchronize(__FILE__, __LINE__);
}


void ImageProcessing::fresnel_lens(
	byte* arr, size_t width, size_t height, double delta_z_um
) const {
	int h_int = int(height);
	int w_int = int(width);
		
	double temp;
	for (int i = 0; i < h_int; i++) {
		for (int j = 0; j < w_int; j++) {
			int y_rel = int(i - h_int / 2);
			int x_rel = int(j - w_int / 2);

			if (pow(x_rel, 2.0) + pow(y_rel, 2.0) <= pow(h_int / 2, 2.0)) {
				temp = 255.0 / (2 * wavelength_um) * delta_z_um / pow(focal_length_um, 2.0)
						* pow(pixel_size_um, 2.0) * (pow(x_rel, 2.0) + pow(y_rel, 2.0));

				arr[i * width + j] += byte(temp);

			}
		}
	}
}


// Error checking needed
void ImageProcessing::correct_image(
	byte* slm_image_ptr,
	const byte* phase_correction_ptr,
	const byte* lut_ptr
) const {
	correct_image_d <<<num_blocks_slm, block_size>>>
		(slm_image_ptr, slm_px_x, slm_px_y,
			lut_patch_size_x, lut_patch_size_y,
			phase_correction_ptr, lut_ptr);
	cuda_utils::cuda_synchronize(__FILE__, __LINE__);
}


// Copy the unpadded input into the center of the padded input array
static __global__ void insert_in_padded_d(
	cufftDoubleComplex* __restrict padded_array,
    const byte* __restrict unpadded_array,
    unsigned int number_of_pixels_padded, unsigned int N_x, unsigned int N_y, double norm
) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;

	const auto x = (tid & (number_of_pixels_padded - 1));
	const auto y = tid / number_of_pixels_padded;

	const auto first_nonzero_index_x = (number_of_pixels_padded - N_x) / 2;
	const auto first_nonzero_index_y = (number_of_pixels_padded - N_y) / 2;

	const bool center_site = (first_nonzero_index_x <= x) &&
					   (x < first_nonzero_index_x + N_x) &&
					   (first_nonzero_index_y <= y) &&
					   (y < first_nonzero_index_y + N_y);

    if (center_site) {
		const auto x_unpadded = x - first_nonzero_index_x;
		const auto y_unpadded = y - first_nonzero_index_y;

		padded_array[tid].x = sqrt(unpadded_array[y_unpadded * N_x + x_unpadded] / norm);
        padded_array[tid].y = 0;
    }
    else {
		padded_array[tid].x = 0;
		padded_array[tid].y = 0;
    }
}


// Add to phasemaps of the same dimensions
// IMPORTANT: the number of threads must equal the size of the arrays!
static __global__ void add_arrays_d(
	byte* p1, const byte* p2
) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	p1[tid] += p2[tid];
}


// Generate a blazed grating. Because the phasemap is a square the x starting point
// must be specified to which the height is added
static __global__ void blazed_grating_d(
	byte* result,
	unsigned int N_x, unsigned int N_y,
	unsigned int from_x,
	unsigned int period, byte phase_max
) {

	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;

	// This is ugly but works for now
	const auto val_x = tid % N_x;
	const auto val_y = tid / N_x;

	const auto slope_px = phase_max / (period - 1);

	const bool center_site = math_utils::is_in_circle(
		val_x, val_y,
		(from_x + N_y / 2), N_y / 2,
		N_y / 2
	);

	if (center_site) {
		result[tid] += slope_px * (val_x & (period - 1));
	}
}


// Apply phase_correction and lut
static __global__ void correct_image_d(
	byte* __restrict result, unsigned int N_x, unsigned int N_y,
	unsigned int lut_patch_size_x, unsigned int lut_patch_size_y,
	const byte* __restrict phase_correction_ptr, const byte* __restrict lut_ptr
) {

	const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t val_x = tid % N_x;
	const size_t val_y = tid / N_x;

	const size_t x_index = val_x / lut_patch_size_x;
	const size_t y_index = val_y / lut_patch_size_y;

	// Can be made byte but while debugging I wanted to check the value without it wrapping
	const size_t temp_byte = ((size_t)result[tid] + (size_t)phase_correction_ptr[tid]) % 256;
	const size_t lut_index = x_index * 256 + y_index * (N_x / lut_patch_size_x) * 256 + temp_byte;

	result[tid] = lut_ptr[lut_index];
}


static __global__ void shift_fourier_image_d(
	byte* phasemap,
	unsigned int number_of_pixels_unpadded,
	unsigned int shift_x_px, unsigned int shift_y_px
) {
	long long center_radius_in_circle = number_of_pixels_unpadded / 2;
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < number_of_pixels_unpadded * number_of_pixels_unpadded) {
		const auto x = tid % number_of_pixels_unpadded;
		const auto y = tid / number_of_pixels_unpadded;

		const bool center_site = math_utils::is_in_circle(x, y, center_radius_in_circle, center_radius_in_circle, center_radius_in_circle);
		if (center_site) {
			phasemap[tid] += shift_x_px * x + shift_y_px * y;
		}
	}
}


static __global__ void expand_to_sensor_size_d(
	byte* __restrict dst,
	const byte* __restrict src,
	unsigned int sensor_width_px,
	unsigned int number_of_pixels_unpadded, unsigned int horizontal_offset
) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	const auto x = tid % sensor_width_px;
	const auto y = tid / sensor_width_px;

	const bool site_is_contained = (horizontal_offset <= x) && (x < horizontal_offset + number_of_pixels_unpadded);

	if (site_is_contained) {
		dst[tid] = src[y * number_of_pixels_unpadded + (x - horizontal_offset)];
	}
	else {
		dst[tid] = 0.0;
	}
}
