#include <iostream>
#include <vector>
#include "faiss/IndexHNSW.h"
#include <algorithm>
#include <string>
#include <sstream>
#include <unistd.h>
#include <sys/fcntl.h>
#include <thread>
#include <uv.h>
#include <filesystem>
#include <functional>

using namespace std;

//using namespace vector_index;


class InputParser {
public:
    InputParser(int &argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            this->tokens.emplace_back(argv[i]);
        }
    }

    const std::string &getCmdOption(const std::string &option) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
            return *itr;
        }
        static const std::string emptyString;
        return emptyString;
    }

private:
    std::vector<std::string> tokens;
};

class Barrier {
private:
    char padding0[64];
    const int releaseValue;
    char padding1[64];
    std::atomic<int> v;
    char padding2[64];
public:
    Barrier(int _releaseValue) : releaseValue(_releaseValue), v(0) {}
    void wait() {
        v++;
        while (v < releaseValue) {}
    }
};

static int64_t ONE_MB = 1 * 1024 * 1024;

int open_file(const char* filePath, int withKernelCache) {
#ifdef TARGET_OS_MAC
    int fd = open(filePath, O_RDONLY, 0633);
    if (!withKernelCache) {
        fcntl(fd, F_NOCACHE, 1);
    }
#else
    int fd;
    if (withKernelCache) {
        fd = open(filePath, O_DIRECT | O_RDONLY, 0633);
    } else {
        fd = open(filePath, O_RDONLY, 0633);
    }
#endif
    if (fd < 0) {
        std::cerr << "Error opening file" << std::endl;
        throw std::runtime_error("Error opening file");
    }
    return fd;
}

void read_from_file(int i, int fd, int64_t buffer_size, int64_t batch_size, int withKernelCache, Barrier *barrier, std::uintmax_t fileSize, int64_t gap) {
#ifdef TARGET_OS_MAC
    fcntl(fd, F_NOCACHE, withKernelCache);
#endif
    char* buffer = new char[buffer_size];
    barrier->wait();
    for (int j = 0; j < batch_size; j++) {
        // pread from random offset
        auto offset = (((i + j) * buffer_size) + gap) % fileSize;
        auto read_bytes = pread(fd, buffer, buffer_size, offset);
        assert(read_bytes == buffer_size);
    }
}

int64_t run_on_multiple_threads(const char* filePath, int numThreads, int64_t numRandomOperations, int64_t chunkSize, int withKernelCache, std::uintmax_t fileSize, int64_t gap) {
    Barrier barrier( numThreads + 1);
    std::vector<thread *> threads;
    int fd = open_file(filePath, withKernelCache);
    auto batch_size = numRandomOperations / numThreads;
    for (int i = 0; i < numThreads; i++) {
        threads.push_back(new thread(read_from_file, i, fd, chunkSize, batch_size, withKernelCache, &barrier, fileSize, gap));
    }

    barrier.wait();
    auto start_time = std::chrono::high_resolution_clock::now();
    for (auto &t : threads) {
        t->join();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    close(fd);
    return duration;
}


inline void on_read(uv_fs_t *req) {
    assert(req->result == *(int64_t*)req->data);
}

int64_t run_exp_with_uv(const char* filePath, int64_t numRandomOperations, int64_t chunkSize, int withKernelCache, std::uintmax_t fileSize, int64_t gap) {
    int fd = open_file(filePath, withKernelCache);
    uv_loop_t *loop = uv_loop_new();
    uv_loop_init(loop);
    auto read_req = new uv_fs_t[numRandomOperations];
    for (int i = 0; i < numRandomOperations; i++) {
        read_req[i].data = &chunkSize;
    }
    auto iovs = new uv_buf_t[numRandomOperations];
    auto paddedChunkSize = chunkSize + 64 + 64;
    char* buffers = new char[numRandomOperations * paddedChunkSize];
    for (int64_t i = 0; i < numRandomOperations; i++) {
        iovs[i] = uv_buf_init(buffers + (i * paddedChunkSize) + 64, chunkSize);
    }

//    uv_loop_init(loop);
//    int fd = uv_fs_open(loop, &open_req, "/Users/gauravsehgal/work/vector_index/data/gist/base.fvecs", O_RDONLY, 0, NULL);
//    if (fd < 0) {
//        std::cerr << "Error opening file" << std::endl;
//        return;
//    }
//    fcntl(fd, F_NOCACHE, 1);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numRandomOperations; i++) {
        uint64_t offset = (i * chunkSize + gap) % fileSize;
        uv_fs_read(loop, &read_req[i], fd, &iovs[i], 1, offset, on_read);
    }
    uv_run(loop, UV_RUN_DEFAULT);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    uv_loop_close(loop);
    close(fd);
    delete[] buffers;
    return duration;
}

int64_t read_random_on_single_thread(const char* filePath, int64_t numRandomOperations, int64_t chunkSize, int withKernelCache, std::uintmax_t fileSize, int64_t gap) {
    char* buffer = new char[chunkSize];
    int fd = open_file(filePath, withKernelCache);
    // create random vector of size numRandomOperations with random indexes
    vector<uint64_t> randomOffsets;
    for (int i = 0; i < numRandomOperations; i++) {
        randomOffsets.push_back((i * chunkSize + gap) % fileSize);
    }
    auto rd = std::random_device {};
    auto rng = std::default_random_engine { rd() };
    std::shuffle(std::begin(randomOffsets), std::end(randomOffsets), rng);
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numRandomOperations; i++) {
        auto offset = randomOffsets[i];
        auto read_bytes = pread(fd, buffer, chunkSize, offset);
        assert(read_bytes == chunkSize);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    close(fd);
    delete[] buffer;
    return duration;
}

int64_t read_sorted_random_on_single_thread(const char* filePath, int64_t numRandomOperations, int64_t chunkSize, int withKernelCache, std::uintmax_t fileSize, int64_t gap) {
    char* buffer = new char[chunkSize];
    int fd = open_file(filePath, withKernelCache);
    // create random vector of size numRandomOperations with random indexes
    vector<uint64_t> randomOffsets;
    for (int i = 0; i < numRandomOperations; i++) {
        randomOffsets.push_back((i * chunkSize + gap) % fileSize);
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numRandomOperations; i++) {
        auto offset = randomOffsets[i];
        auto read_bytes = pread(fd, buffer, chunkSize, offset);
        assert(read_bytes == chunkSize);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    close(fd);
    delete[] buffer;
    return duration;
}

int64_t read_sequential(const char* filePath, int64_t numRandomOperations, int64_t chunkSize, int withKernelCache, std::uintmax_t fileSize, int64_t gap) {
    char* buffer = new char[chunkSize * numRandomOperations];
    int fd = open_file(filePath, withKernelCache);
    auto start_time = std::chrono::high_resolution_clock::now();
    auto read_bytes = pread(fd, buffer, chunkSize * numRandomOperations, gap);
    assert(read_bytes == chunkSize * numRandomOperations);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    close(fd);
    delete[] buffer;
    return duration;
}

constexpr int64_t WARMUP_COUNT = 4;
constexpr int64_t BENCHMARK_COUNT = 20;

void run_benchmark(std::function<int64_t()> func, const char* name) {
    // warm up
    std::cout << "================================================" << std::endl;
    std::cout << "Benchmark name " << name << std::endl;
    for (int i = 0; i < WARMUP_COUNT; i++) {
        auto duration = func();
        std::cout << "Warm up run " << i << " took " << duration << " microseconds" << std::endl;
    }

    // actual execution
    std::vector<int64_t> durations;
    for (int i = 0; i < BENCHMARK_COUNT; i++) {
        auto duration = func();
        durations.push_back(duration);
        std::cout << "Run " << i << " took " << duration << " microseconds" << std::endl;
    }
    // Take average
    int64_t sum = 0;
    for (auto &d: durations) {
        sum += d;
    }
    std::cout << "Average duration for " << name << " is " << sum / BENCHMARK_COUNT << " microseconds" << std::endl;
    std::cout << "================================================" << std::endl;
}


int main(int argc, char **argv) {
    InputParser input(argc, argv);
    const std::string &filePath = input.getCmdOption("-f");
    auto numThreads = stoi(input.getCmdOption("-t"));
    int64_t numRandomOperations = stoi(input.getCmdOption("-l"));
    int64_t chunkSize = stoi(input.getCmdOption("-c"));
    auto withKernelCache = stoi(input.getCmdOption("-k"));
    auto gap_in_mbs = stoi(input.getCmdOption("-g"));
    auto gap = gap_in_mbs * ONE_MB;

    std::uintmax_t fileSize = std::filesystem::file_size(filePath.data());

    printf("File path: %s\n", filePath.data());
    printf("File size: %zu\n", fileSize);
    printf("Number of threads: %d\n", numThreads);
    printf("Number of random operations: %ld\n", numRandomOperations);
    printf("Chunk size: %ld\n", chunkSize);
    printf("With kernel cache: %d\n", withKernelCache);

    run_benchmark([&](){return read_sequential(filePath.data(), numRandomOperations, chunkSize, withKernelCache, fileSize, gap);}, "read_sequential");
    run_benchmark([&](){return run_on_multiple_threads(filePath.data(), numThreads, numRandomOperations, chunkSize, withKernelCache, fileSize, gap);}, "run_on_multiple_threads");
    run_benchmark([&](){return run_exp_with_uv(filePath.data(), numRandomOperations, chunkSize, withKernelCache, fileSize, gap);}, "run_exp_with_uv");
    run_benchmark([&](){return read_sorted_random_on_single_thread(filePath.data(), numRandomOperations, chunkSize, withKernelCache, fileSize, gap);}, "read_sorted_random_on_single_thread");
    run_benchmark([&](){return read_random_on_single_thread(filePath.data(), numRandomOperations, chunkSize, withKernelCache, fileSize, gap);}, "read_random_on_single_thread");
    return 0;
}
