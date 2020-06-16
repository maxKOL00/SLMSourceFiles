#include "Patch.cuh"


Patch::Patch() {
    x_0 = 0;
    y_0 = 0;
    patch_size_x = 0;
    patch_size_y = 0;
    // init_edges();
}
Patch::Patch(size_t x, size_t y) {
    x_0 = x;
    y_0 = y;
    patch_size_x = 0;
    patch_size_y = 0;
    // init_edges();
}
Patch::Patch(size_t x, size_t y, size_t patch_size) {
    /*if (patch_size < 0) {
        throw std::runtime_error("Patch size must be larger than 0");
    }*/
    x_0 = x;
    y_0 = y;
    this->patch_size_x = patch_size;
    this->patch_size_y = patch_size;
    // init_edges();
}
Patch::Patch(size_t x, size_t y, size_t patch_size_x, size_t patch_size_y) {
    /*if ((patch_size_x < 0) || (patch_size_y < 0)) {
        throw std::runtime_error("Patch size must be larger than 0");
    }*/
    x_0 = x;
    y_0 = y;
    this->patch_size_x = patch_size_x;
    this->patch_size_y = patch_size_y;
    // init_edges();
}

void Patch::move_to(size_t x_upper_left_new, size_t y_upper_left_new) {
    x_0 = x_upper_left_new;
    y_0 = y_upper_left_new;
}

void Patch::move_by(size_t x_shift, size_t y_shift) {
    x_0 += x_shift;
    y_0 += y_shift;
}


//void Patch::init_edges(void) {
//    x_min = x_0;
//    y_min = y_0;
//    x_max = x_0 + patch_size_x;
//    y_max = y_0 + patch_size_y;
//}