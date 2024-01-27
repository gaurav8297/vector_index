#include "spdlog/fmt/fmt.h"
#include "gtest/gtest.h"

#include <faiss/IndexHNSW.h>
#include "utils.h"

using namespace vector_index;

TEST(HNSWPQTest, Benchmark) {
    int k = 100;
    size_t baseDimension, baseNumVectors;
    auto basePath = "/Users/gauravsehgal/work/vector_index/data";
    auto benchmarkType = "gist";
    auto baseVectorPath = fmt::format("{}/{}/base.fvecs", basePath, benchmarkType, benchmarkType);
    auto queryVectorPath = fmt::format("{}/{}/query.fvecs", basePath, benchmarkType, benchmarkType);
    auto gtVectorPath = fmt::format("{}/{}/groundtruth.ivecs", basePath, benchmarkType, benchmarkType);
    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);

    faiss::IndexHNSWFlat hnsw(baseDimension, 64);
    hnsw.train(baseNumVectors, baseVecs);
    hnsw.add(baseNumVectors, baseVecs);

    size_t queryDimension, queryNumVectors;
    float *queryVecs = Utils::fvecs_read(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    int64_t* I = new int64_t[k * queryNumVectors];
    float* D = new float[k * queryNumVectors];
    hnsw.search(queryNumVectors, queryVecs, k, D, I);

    std::vector<std::vector<int>> truth;
    for (int i = 0; i < queryNumVectors; i++) {
        truth.push_back(std::vector<int>());
        for (int j = 0; j < k; j++)
            truth[i].push_back(I[i * k + j]);
    }

    size_t gtDimension, gtNumVectors;
    int *gtVecs = Utils::ivecs_read(gtVectorPath.c_str(), &gtDimension, &gtNumVectors);
    std::vector<std::vector<int>> groundTruth;
    for (int i = 0; i < gtNumVectors; i++) {
        std::vector<int> gt;
        for (int j = i * gtDimension; j < (i + 1) * gtDimension; j++) {
            gt.push_back(gtVecs[j]);
        }
        groundTruth.push_back(gt);
    }

//    size_t avgNodesVisited = 0;
//    size_t avgDepth = 0;
//    size_t maxDepth = 0;
//    size_t avgHops = 0;
//    double avgSearchTime = 0.0;
    int avgRecall = 0;
    for (int i = 0; i < queryNumVectors; i++) {
        auto t = truth[i];
        auto gt = groundTruth[i];
        for (auto id: t) {
            if (std::find(gt.begin(), gt.end(), id) != gt.end()) {
                avgRecall++;
            }
        }
    }

    printf("Avg recall: %zu/%d\n", avgRecall / queryNumVectors, k);
}