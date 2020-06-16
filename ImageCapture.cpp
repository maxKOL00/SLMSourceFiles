#include "ImageCapture.h"
#include "statusBox.h"
#include <sstream>
#include "errorMessage.h"
#include <sstream>

ImageCapture::ImageCapture(
	const Parameters& params
) {
	ID = params.get_camera_id();
	camera_px_h = params.get_camera_px_x();
	camera_px_v = params.get_camera_px_y();
	
	buf = new byte[camera_px_h * camera_px_v];

	if (VmbErrorSuccess != system.Startup()) {
		errBox("Failed to start up image capture system", __FILE__, __LINE__);
		throw ImageCaptureException("Failed to start up image capture system");
	}

	if (VmbErrorSuccess != system.OpenCameraByID(ID.c_str(), VmbAccessModeFull, camera_ptr)) {
		errBox("Failed to open camera", __FILE__, __LINE__);
		throw ImageCaptureException("Failed to open camera");
	}
	

	if (VmbErrorSuccess != camera_ptr->GetFeatureByName("ExposureTime", exposure_time_ptr)) {
		errBox("Could not get feature by name", __FILE__, __LINE__);
		throw ImageCaptureException("Could not get feature by name");
	}

	double max_frame_rate = double(params.get_camera_max_frame_rate());

	min_timeout_from_frame_rate_ms = 1000.0 / max_frame_rate+ 0.0;

	// Automatic exposure time adjustment can only be done 
	// if an image is shown
	std::string exposure_mode = params.get_exposure_mode();
	if (exposure_mode == "manual") {
		double t = params.get_exposure_time_us();
		if (VmbErrorSuccess != exposure_time_ptr->SetValue(t)) {
			errBox("Could not set  exposure time", __FILE__, __LINE__);
			throw ImageCaptureException("Could not set exposure time");
		}
	}

	else if (exposure_mode == "preset") {}

	else {
		errBox("Please specify a valid exposure mode", __FILE__, __LINE__);
		throw ImageCaptureException("Please specify a valid exposure mode");
	}

	std::cout << "Exposure time set to: " << get_exposure_time_us() << "us\n";
}


ImageCapture::~ImageCapture(
) noexcept {

	camera_ptr->Close();

	system.Shutdown();

	std::cout << "\nCamera shutdown complete\n\n";
}


void ImageCapture::capture_image(
	byte* pixel_data, size_t width, size_t height
) const {
	if (width * height != camera_px_h * camera_px_v) {
		errBox("capture_image: invalid size", __FILE__, __LINE__);
		throw ImageCaptureException("capture_image: invalid size");
	}

	double exposure_time_ms = get_exposure_time_us() / 1000.0;
	// For obvious reason acquiring a frame takes longer than the exposure time
	// so the timeout should be set dynamically
	auto timeout_ms = unsigned int(exposure_time_ms + min_timeout_from_frame_rate_ms + 5.0 + 0.5);

	if (VmbErrorSuccess != camera_ptr->AcquireSingleImage(frame_ptr, timeout_ms)) {
		errBox("capture_image: Could not aquire frame", __FILE__, __LINE__);
		throw ImageCaptureException("capture_image: Could not acquire frame");
	}
	if (VmbErrorSuccess != frame_ptr->GetImage(buf)) {
		errBox("capture_iamge: Could not get image", __FILE__, __LINE__);
		throw ImageCaptureException("capture_image: Could not get image");
	}
	
	std::copy(buf, buf + width * height, pixel_data);
}


void ImageCapture::adjust_exposure_time_automatically(
	byte pixel_value = 230, size_t max_counts = 10, statusBox *box = NULL
) {
	double lower = 50;
	double upper = 40000; // The reason for hardcoding is that I haven't found a good way to determine this yet
	std::vector<byte> image_data(camera_px_h * camera_px_v);

	double mean;
	size_t counts;

	set_exposure_time(upper);

	// Binary search
	while (lower < upper) {
		mean = (lower + upper) / 2;
		set_exposure_time(mean);

		capture_image(image_data.data(), camera_px_h, camera_px_v);
		
		counts = number_of_pixels_with_value_or_more_in_image(
			image_data.data(), std::next(image_data.data(), image_data.size()), 
			pixel_value
		);
		
		if (counts < max_counts) {
			lower = mean + 1;
		}
		else if (counts > max_counts) {
			upper = mean - 1;
		}
		else {
			break;
		}
	}
	set_exposure_time(mean);
	std::stringstream stream;
	stream << "Set exposure time to " << mean << "us\n";
	box->appendMessage(stream.str().c_str());
	std::cout << "Set exposure time to " << mean << "us\n";
}


void ImageCapture::set_exposure_time(
	double time_us
) {
	if (VmbErrorSuccess != camera_ptr->GetFeatureByName("ExposureTime", exposure_time_ptr)) {
		errBox("set_exposure_time: Could not get feature by name", __FILE__, __LINE__);
		throw std::runtime_error("set_exposure_time: could not get feature by name");
	}
	if (VmbErrorSuccess != exposure_time_ptr->SetValue(time_us)) {
		errBox("set_exposure_time: Could not set exposure time", __FILE__, __LINE__);
		throw ImageCaptureException("Could not set exposure time");
	}
}


double ImageCapture::get_exposure_time_us(
	void
) const {
	double exposure_time;
	if (VmbErrorSuccess != camera_ptr->GetFeatureByName("ExposureTime", exposure_time_ptr)) {
		errBox("Could not get feature by name", __FILE__, __LINE__);
		throw ImageCaptureException("Could not get feature by name");
	}
	
	if (VmbErrorSuccess != exposure_time_ptr->GetValue(exposure_time)) {
		std::stringstream stream;
		stream << "Error type: " << exposure_time_ptr->GetValue(exposure_time) << "\n";
		errBox(stream.str(), __FILE__, __LINE__);
		throw ImageCaptureException("Could not read exposure time");
	}

	return exposure_time;
}


size_t ImageCapture::number_of_pixels_with_value_or_more_in_image(
	const byte* first, const byte* last, byte value
) const noexcept {
	size_t result = 0;
	std::for_each(
		first, last,
		[&result, value](byte px_val) {
			if (px_val >= value) {
				result++;
			}
		}
	);
	return result;
}
