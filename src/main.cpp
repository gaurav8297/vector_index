#include <iostream>
#include <vector>
#include "spdlog/fmt/fmt.h"
#include "faiss/IndexHNSW.h"
#include <algorithm>
#include <string>
#include <sstream>
#include <unistd.h>
#include <ios>
#include <fstream>
#include <chrono>

#include "utils.h"

using namespace vector_index;


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

void splitCommaSeperatedString(const std::string &s, std::vector<int> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        elems.push_back(stoi(item));
    }
}

void process_mem_usage(double& vm_usage, double& resident_set)
{
    using std::ios_base;
    using std::ifstream;
    using std::string;

    vm_usage     = 0.0;
    resident_set = 0.0;

    // 'file' stat seems to give the most reliable results
    //
    ifstream stat_stream("/proc/self/stat",ios_base::in);

    // dummy vars for leading entries in stat that we don't care about
    //
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;

    // the two fields we want
    //
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
                >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
                >> utime >> stime >> cutime >> cstime >> priority >> nice
                >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

    stat_stream.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage     = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}


int main(int argc, char **argv) {
//    InputParser input(argc, argv);
//    const std::string &basePath = input.getCmdOption("-f");
//    auto k = stoi(input.getCmdOption("-k"));
//    auto indexType = input.getCmdOption("-t");
//    auto efConstruction = stoi(input.getCmdOption("-efConstruction"));
//    auto m = stoi(input.getCmdOption("-m"));
//    auto pq_m = stoi(input.getCmdOption("-pq_m"));
//    auto pq_bits = stoi(input.getCmdOption("-pq_bits"));
//    auto efSearch = stoi(input.getCmdOption("-efSearch"));
//    auto nIndexingThreads = stoi(input.getCmdOption("-nIndexingThreads"));
//    std::vector<int> nSearchThreads;
//    splitCommaSeperatedString(input.getCmdOption("-nSearchThreads"), nSearchThreads);

    auto basePath = "/home/gaurav/vector_index_experiments/gist";

    auto baseVectorPath = fmt::format("{}/gist_base.fvecs", basePath);

//    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
//    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);
//    auto gtVectorPath = fmt::format("{}/groundtruth.ivecs", basePath);

    size_t baseDimension, baseNumVectors;
    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);
//    size_t queryDimension, queryNumVectors;
//    float *queryVecs = Utils::fvecs_read(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
//    size_t gtDimension, gtNumVectors;
//    int *gtVecs = Utils::ivecs_read(gtVectorPath.c_str(), &gtDimension, &gtNumVectors);

    double vm, rss;
//    process_mem_usage(vm, rss);
    std::cout << "Before training VM: " << vm << "; RSS: " << rss << std::endl;

    omp_set_num_threads(32);
    std::cout << "Base dimension: " << baseDimension << std::endl;
    std::cout << "Base num vectors: " << baseNumVectors << std::endl;
//    std::cout << "Query dimension: " << queryDimension << std::endl;
//    std::cout << "Query num vectors: " << queryNumVectors << std::endl;
//    std::cout << "Ground truth dimension: " << gtDimension << std::endl;
//    std::cout << "Ground truth num vectors: " << gtNumVectors << std::endl;
//    std::cout << "\nStarted build index: " << indexType << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    faiss::IndexHNSW* hnsw = nullptr;
    hnsw = new faiss::IndexHNSWFlat(baseDimension, 64);
//    if (indexType == "hnsw") {
//    } else if (indexType == "hnsw_pq") {
//        hnsw = new faiss::IndexHNSWPQ(baseDimension, pq_m, m, pq_bits);
//        hnsw->train(baseNumVectors, baseVecs);
//        end = std::chrono::high_resolution_clock::now();
//        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//        std::cout << "Training time: " << duration << " ms" << std::endl;
//        start = std::chrono::high_resolution_clock::now();
//    } else {
//        std::cout << "Invalid index type: " << indexType << std::endl;
//        return 1;
//    }
//    process_mem_usage(vm, rss);
    std::cout << "After training VM: " << vm << "; RSS: " << rss << std::endl;
    hnsw->hnsw.efConstruction = 120;
    hnsw->hnsw.efSearch = 120;
    hnsw->add(baseNumVectors, baseVecs);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Indexing time: " << duration << " ms" << std::endl;
//    int64_t* I = new int64_t[k * queryNumVectors];
//    float* D = new float[k * queryNumVectors];

//    process_mem_usage(vm, rss);
    std::cout << "After index build: VM: " << vm << "; RSS: " << rss << std::endl;

    std::cout << "Print Neighbor Stats: " << std::endl;
    hnsw->hnsw.print_neighbor_stats(0);

//    for (auto nSearchThread: nSearchThreads) {
//        // Find query per seconds
//        omp_set_num_threads(nSearchThread);
//        start = std::chrono::high_resolution_clock::now();
//        hnsw->search(queryNumVectors, queryVecs, k, D, I);
////        process_mem_usage(vm, rss);
//        std::cout << "After search VM: " << vm << "; RSS: " << rss << std::endl;
//        end = std::chrono::high_resolution_clock::now();
//        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//        auto queriesPerSecond = (queryNumVectors * 1000.0) / duration;
//        std::cout << "\nNumber of Search threads: " << nSearchThread << std::endl;
//        std::cout << "Search time: " << duration << " ms" << std::endl;
//        std::cout << "Queries per second: " << queriesPerSecond << std::endl;
//
//        double avgRecall = 0;
//        double avgRecallAt1 = 0;
//        for (int i = 0; i < queryNumVectors; i++) {
//            std::vector<int> gt;
//            for (int j = i * gtDimension; j < (i + 1) * gtDimension; j++) {
//                if (I[j] == gtVecs[j]) {
//                    avgRecall++;
//                }
//                gt.push_back(gtVecs[j]);
//            }
//            for (int j = i * gtDimension; j < (i + 1) * gtDimension; j++) {
//                if (std::find(gt.begin(), gt.end(), I[j]) != gt.end()) {
//                    avgRecallAt1++;
//                }
//            }
//        }
//        std::cout << "Average recall (exact pos): " << avgRecall / queryNumVectors << std::endl;
//        std::cout << "Average recall: " << avgRecallAt1 / queryNumVectors << std::endl;
//    }
}
