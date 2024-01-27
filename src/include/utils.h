#pragma once

#include <vector>

namespace vector_index {
    struct Utils {
        static double l2_distance(std::vector<float> &a, std::vector<float> &b);

        static double cosine_distance(std::vector<float> &a, std::vector<float> &b);

        static float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out);

        static int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out);

        static double rand_double();

        static int rand_int(int min, int max);
    };
} // namespace vector_index
