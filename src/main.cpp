#include <iostream>
#include <vector>
#include "spdlog/fmt/fmt.h"
#include "faiss/IndexHNSW.h"

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

int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}


int getValue(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
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
    auto nThreads = stoi(input.getCmdOption("-nThreads"));

    auto baseVectorPath = fmt::format("{}/base.fvecs", basePath);
    auto queryVectorPath = fmt::format("{}/query.fvecs", basePath);
    auto gtVectorPath = fmt::format("{}/groundtruth.ivecs", basePath);
    omp_set_num_threads(nThreads);

    size_t baseDimension, baseNumVectors;
    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);
    size_t queryDimension, queryNumVectors;
    float *queryVecs = Utils::fvecs_read(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    size_t gtDimension, gtNumVectors;
    int *gtVecs = Utils::ivecs_read(gtVectorPath.c_str(), &gtDimension, &gtNumVectors);

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
    int64_t* I = new int64_t[k * queryNumVectors];
    float* D = new float[k * queryNumVectors];
    hnsw->search(queryNumVectors, queryVecs, k, D, I);

    int avgRecall = 0;
    for (int i = 0; i < queryNumVectors; i++) {
        for (int j = i * gtDimension; j < (i + 1) * gtDimension; j++) {
            if (queryVecs[j] == gtVecs[j]) {
                avgRecall++;
            }
        }
    }

    std::cout << "Average recall: " << avgRecall / queryNumVectors << std::endl;
}

