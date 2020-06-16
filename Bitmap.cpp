//#include "pch.h"
#include "Bitmap.h"


Bitmap::Bitmap(const std::vector<byte>& pixel_data, size_t width, size_t height) {
    if (pixel_data.size() != width * height) {
        throw std::runtime_error("Bitmap: Invalid dimensions given");
    }
    this->height = height;
    this->width = width;
    this->pixel_data = pixel_data;
}

Bitmap::Bitmap(const byte* pixel_data, size_t width, size_t height) {
    this->height = height;
    this->width = width;

    this->pixel_data.reserve(width * height);

    std::copy(pixel_data, pixel_data + width * height, std::back_inserter(this->pixel_data));
}

Bitmap::Bitmap(const std::string& filename) {
    throw std::runtime_error("Bitmap construction from filename not implemented yet");
}

//Bitmap::Bitmap(size_t width, size_t height) {
//    this->height = height;
//    this->width = width;
//    this->pixel_data = std::vector<byte>(this->width * this->height);
//}

void Bitmap::save(const std::string& filename) {
    basic_fileIO::save_as_bmp(filename, this->pixel_data, this->width, this->height);
}
