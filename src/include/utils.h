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

        static int open_file(const char* fname, int flags, int mode);

        static int close(int fd);

        static int read(int fd, void* buf, size_t numBytes, off_t offset);
    };
} // namespace vector_index
