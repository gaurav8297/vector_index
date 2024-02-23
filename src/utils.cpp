#include <sys/stat.h>
#include <cmath>
#include <random>
#include <cassert>
#include "include/utils.h"
#include <cstring>
#include <sys/fcntl.h>
#include <unistd.h>

namespace vector_index {
    int* Utils::ivecs_read(const char *fname, size_t *d_out, size_t *n_out) {
        return (int*)fvecs_read(fname, d_out, n_out);
    }

    double Utils::l2_distance(std::vector<float> &a, std::vector<float> &b) {
        double distance = 0;
        for (int i = 0; i < a.size(); i++) {
            distance += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sqrt(distance);
    }

    double Utils::cosine_distance(std::vector<float> &a, std::vector<float> &b) {
        double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
        for (int i = 0; i < a.size(); i++) {
            dot += a[i] * b[i] ;
            denom_a += a[i] * a[i] ;
            denom_b += b[i] * b[i] ;
        }
        return dot / (sqrt(denom_a) * sqrt(denom_b)) ;
    }

    float* Utils::fvecs_read(const char *fname, size_t *d_out, size_t *n_out) {
        FILE* f = fopen(fname, "r");
        if (!f) {
            fprintf(stderr, "could not open %s\n", fname);
            perror("");
            abort();
        }
        int d;
        fread(&d, 1, sizeof(int), f);
        assert((d > 0 && d < 1000000) || !"unreasonable dimension");
        fseek(f, 0, SEEK_SET);
        struct stat st;
        fstat(fileno(f), &st);
        size_t sz = st.st_size;
        assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
        size_t n = sz / ((d + 1) * 4);

        *d_out = d;
        *n_out = n;
        float* x = new float[n * (d + 1)];
        size_t nr = fread(x, sizeof(float), n * (d + 1), f);
        assert(nr == n * (d + 1) || !"could not read whole file");

        // shift array to remove row headers
        for (size_t i = 0; i < n; i++)
            memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

        fclose(f);
        return x;
    }

    double Utils::rand_double() {
        std::random_device r;
        std::default_random_engine e1(r());
        std::uniform_real_distribution<double> uniform_dist(0,1);
        return uniform_dist(e1);
    }

    int Utils::rand_int(int min, int max) {
        std::random_device r;
        std::default_random_engine e1(r());
        std::uniform_int_distribution<int> uniform_dist(min,max);
        return uniform_dist(e1);
    }

    int Utils::open_file(const char *fname, int flags, int mode) {
        return open(fname, flags, mode);
    }

    int Utils::read(int fd, void *buf, size_t numBytes, off_t offset) {
        return pread(fd, buf, numBytes, offset);
    }

    int Utils::close(int fd) {
        return close(fd);
    }
} // namespace vector_index
