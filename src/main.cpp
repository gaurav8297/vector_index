#include <iostream>
#include <vector>
#include "spdlog/fmt/fmt.h"
#include "faiss/IndexHNSW.h"
#include <chrono>
#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>

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


int main(int argc, char **argv) {
    InputParser input(argc, argv);
    const std::string &basePath = input.getCmdOption("-f");
    auto k = stoi(input.getCmdOption("-k"));
    auto indexType = input.getCmdOption("-t");
    auto efConstruction = stoi(input.getCmdOption("-efConstruction"));
    auto m = stoi(input.getCmdOption("-m"));
    auto pq_m = stoi(input.getCmdOption("-pq_m"));
    auto efSearch = stoi(input.getCmdOption("-efSearch"));
    auto nIndexingThreads = stoi(input.getCmdOption("-nIndexingThreads"));
    std::vector<int> nSearchThreads;
    splitCommaSeperatedString(input.getCmdOption("-nSearchThreads"), nSearchThreads);

    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);
    auto gtVectorPath = fmt::format("{}/groundtruth.ivecs", basePath);

    size_t baseDimension, baseNumVectors;
    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = Utils::fvecs_read(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    size_t gtDimension, gtNumVectors;
    int *gtVecs = Utils::ivecs_read(gtVectorPath.c_str(), &gtDimension, &gtNumVectors);

    omp_set_num_threads(nIndexingThreads);
    std::cout << "Base dimension: " << baseDimension << std::endl;
    std::cout << "Base num vectors: " << baseNumVectors << std::endl;
    std::cout << "Query dimension: " << queryDimension << std::endl;
    std::cout << "Query num vectors: " << queryNumVectors << std::endl;
    std::cout << "Ground truth dimension: " << gtDimension << std::endl;
    std::cout << "Ground truth num vectors: " << gtNumVectors << std::endl;
    std::cout << "\nStarted build index: " << indexType << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSW* hnsw = nullptr;
    if (indexType == "hnsw") {
        hnsw = new faiss::IndexHNSWFlat(baseDimension, m);
    } else if (indexType == "hnsw_pq") {
        hnsw = new faiss::IndexHNSWPQ(baseDimension, pq_m, m);
        hnsw->train(baseNumVectors, baseVecs);
    } else {
        std::cout << "Invalid index type: " << indexType << std::endl;
        return 1;
    }
    hnsw->hnsw.efConstruction = efConstruction;
    hnsw->hnsw.efSearch = efSearch;
    hnsw->add(baseNumVectors, baseVecs);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Indexing time: " << duration << " ms" << std::endl;
    int64_t* I = new int64_t[k * queryNumVectors];
    float* D = new float[k * queryNumVectors];

    for (auto nSearchThread: nSearchThreads) {
        // Find query per seconds
        omp_set_num_threads(nSearchThread);
        start = std::chrono::high_resolution_clock::now();
        hnsw->search(queryNumVectors, queryVecs, k, D, I);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto queriesPerSecond = (queryNumVectors * 1000.0) / duration;
        std::cout << "\nNumber of Search threads: " << nSearchThread << std::endl;
        std::cout << "Search time: " << duration << " ms" << std::endl;
        std::cout << "Queries per second: " << queriesPerSecond << std::endl;

        double avgRecall = 0;
        double avgRecallAt1 = 0;
        for (int i = 0; i < queryNumVectors; i++) {
            std::vector<int> gt;
            for (int j = i * gtDimension; j < (i + 1) * gtDimension; j++) {
                if (I[j] == gtVecs[j]) {
                    avgRecall++;
                }
                gt.push_back(gtVecs[j]);
            }
            for (int j = i * gtDimension; j < (i + 1) * gtDimension; j++) {
                if (std::find(gt.begin(), gt.end(), I[j]) != gt.end()) {
                    avgRecallAt1++;
                }
            }
        }
        std::cout << "Average recall (exact pos): " << avgRecall / queryNumVectors << std::endl;
        std::cout << "Average recall: " << avgRecallAt1 / queryNumVectors << std::endl;
    }
}
