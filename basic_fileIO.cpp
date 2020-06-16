#include "basic_fileIO.h"
#include "errorMessage.h"
#include <filesystem>

namespace basic_fileIO {

	void save_as_bmp(
		const std::string& filename, const byte* src,
		size_t width, size_t height
	) {
		if (((width * (sizeof(RGBQUAD) - 1) % 4) != 0)) {
			errBox("Manual padding required, width must be divisible by 4", __FILE__, __LINE__);
			throw std::runtime_error("Manual padding required, width must be divisible by 4");
		}

		BITMAPFILEHEADER bmfh;
		BITMAPINFOHEADER bmih;
		RGBQUAD          colors[256];

		for (size_t i = 0; i < 256; i++) {
			colors[i].rgbRed = byte(i);
			colors[i].rgbGreen = byte(i);
			colors[i].rgbBlue = byte(i);
			colors[i].rgbReserved = 0;
		}

		bmfh.bfType = 0x4d42;// ('B' + ('M' << 8));
		bmfh.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + (DWORD)(width * height);
		bmfh.bfReserved1 = 0;
		bmfh.bfReserved2 = 0;
		bmfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD); // 14 + 40 + 

		bmih.biSize = sizeof(BITMAPINFOHEADER);
		bmih.biWidth = (LONG)width;
		bmih.biHeight = (LONG)height; // - sign to flip image vertically
		bmih.biPlanes = 1;
		bmih.biBitCount = 8;
		bmih.biCompression = 0;
		bmih.biSizeImage = sizeof(RGBQUAD) * bmih.biWidth * bmih.biHeight;
		bmih.biXPelsPerMeter = 0;
		bmih.biYPelsPerMeter = 0;
		bmih.biClrUsed = 0;
		bmih.biClrImportant = 0;

		std::ofstream out(filename, std::iostream::binary);
		if (!out.is_open()) {
			errBox("ave_as_bmp: Error opening file", __FILE__, __LINE__);
			throw std::runtime_error("save_as_bmp: Error opening file");
		}

		out.write((const char*)&bmfh, sizeof(BITMAPFILEHEADER));
		out.write((const char*)&bmih, sizeof(BITMAPINFOHEADER));
		out.write((const char*)&colors, 256 * sizeof(RGBQUAD));
		out.write((const char*)src, height * width);

		out.close();
		if (!out) {
			errBox("save_as_bmp: File still open", __FILE__, __LINE__);
			throw std::runtime_error("save_as_bmp: File still open");
		}

	}

	void save_as_bmp(
		const std::string& filename, const double* src,
		size_t width, size_t height
	) {
		const auto first = src;
		const auto last = std::next(src, width * height);

		const double max = *std::max_element(first, last);
		std::vector<byte> result(width * height);
		std::transform(first, last, result.begin(),
			[max](const double& v) {return byte(255.0 * v / max); });
		save_as_bmp(filename, result.data(), width, height);
	}

	void read_from_bmp(
		const std::string& filename, byte* dst,
		size_t width, size_t height
	
	) {
		
		if (INVALID_FILE_ATTRIBUTES == GetFileAttributesA(filename.c_str()) && GetLastError() == ERROR_FILE_NOT_FOUND) {
			std::string message = "read_from_bmp: Cannot open file \"" + filename + "\"";
			errBox(message, __FILE__, __LINE__);
			throw std::runtime_error(message);
		}

		std::ifstream ifile(filename, std::ios::binary);

		size_t offbits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD);
		// char* header = (char*)malloc(sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD));

		if (!ifile.ignore(offbits)) {
			errBox("read_from_bmp: Error skipping header", __FILE__, __LINE__);
			throw std::runtime_error("read_from_bmp: Error skipping header");
		}

		if (!ifile.read((char*)dst, width * height)) {
			errBox("read_from_bmp: Error reading file", __FILE__, __LINE__);
			throw std::runtime_error("read_from_bmp: Error reading file");
		};

		ifile.close();
	}

	std::string create_filepath(
		const std::string& filename, const std::string& folder
	) {
		std::string filepath;
		if (folder.empty()) {
			filepath = filename;
		}
		else {
			std::stringstream ss;
			if (folder.substr(folder.size() - 1) == "\\") {
				ss << folder << filename;
			}
			else {
				ss << folder << "\\" << filename;
			}
			filepath = ss.str();
		}
		if (CreateDirectoryA(folder.c_str(), NULL) ||
			ERROR_ALREADY_EXISTS == GetLastError()) {
			return filepath;
		}
		errBox("create_filepath: Could not create directory", __FILE__, __LINE__);
		throw std::runtime_error("create_filepath: Could not create directory");
	}

	void create_nested_directory(
		const std::string& filepath
	) {
		const std::string separators = "\\/";

		std::vector<std::string> substrings;
		size_t pos = 0;

		// Detect all subpaths
		for (;;) {
			pos = filepath.find_first_of(separators, pos + 1);
			if (pos == std::string::npos) {
				break;
			}
			substrings.push_back(filepath.substr(0, pos));
		}
		// If the folder is not terminated by a separator we have to append 
		// the whole path as well otherwise most nested directory is not created
		if (filepath.find_last_of(separators) != filepath.size() - 1) {
			substrings.push_back(filepath);
		}
		for (const auto& substr : substrings) {
			if (!(CreateDirectoryA(substr.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())) {
				std::string messafe = "Could not create directory: " + filepath;
				errBox(messafe, __FILE__, __LINE__);
				throw std::runtime_error(messafe);
			}
		}
	}


	bool load_LUT(
		byte* lut_ptr,
		size_t lut_patch_num_x, size_t lut_patch_num_y
	) {

		bool errStatus = false;
		std::ifstream lut_file("correction_files/LUT.txt");

		// Look if file could be found ...
		// use default vals otherwise
		if (!lut_file.is_open()) {
			printf("Could not find LUT file, using default value\n");
			errStatus = true;
			for (size_t i = 0; i < 256 * lut_patch_num_x * lut_patch_num_y; i++) {
				lut_ptr[i] = i & 255;
			}
		}

		else {
			int temp_int;
			size_t cnt = 0;

			while (lut_file >> temp_int) {
				lut_ptr[cnt++] = (byte)temp_int;
			}

			if (cnt != 256 * lut_patch_num_x * lut_patch_num_y) {
				errBox("load_LUT: Wrong file size", __FILE__,__LINE__);
				throw std::runtime_error("load_LUT: Wrong file size");
			}
		}
		lut_file.close();
		return errStatus;
	}


	bool load_phase_correction(byte* phase_correction_ptr,
		unsigned int width, unsigned int height
	) {
		bool errStatus = false;
		try {
			read_from_bmp("correction_files/phase_correction.bmp", phase_correction_ptr, width, height);
		}
		catch (const std::runtime_error&) {
			errStatus = true;
			printf("Could not find phase correction file, using default value of 0.\n");
			for (auto i = 0u; i < width * height; i++) {
				phase_correction_ptr[i] = 0;
			}
		}
		return errStatus;
	}

}
