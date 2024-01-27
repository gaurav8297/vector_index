#include "spdlog/fmt/fmt.h"
#include "gtest/gtest.h"
#include "hnsw.h"
#include "utils.h"

using namespace vector_index;
using namespace vector_index::hnsw;

TEST(HNSWTest, Benchmark) {
    size_t baseDimension, baseNumVectors;
    auto basePath = "/Users/gauravsehgal/work/vector_index/data";
    auto benchmarkType = "siftsmall";
    auto baseVectorPath = fmt::format("{}/{}/base.fvecs", basePath, benchmarkType, benchmarkType);
    auto queryVectorPath = fmt::format("{}/{}/query.fvecs", basePath, benchmarkType, benchmarkType);
    auto gtVectorPath = fmt::format("{}/{}/groundtruth.ivecs", basePath, benchmarkType, benchmarkType);

    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);

    auto efConstructions = 128;
    auto efSearch = 100;
    auto m = 32;
    auto m0 = 64;
    auto hnsw = HNSW(baseVecs, baseDimension, baseNumVectors, efConstructions, m, m0);
    auto k = 100;

    size_t queryDimension, queryNumVectors;
    float *queryVecs = Utils::fvecs_read(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    std::vector<std::vector<float>> queryEmbeddings;
    for (int i = 0; i < queryNumVectors; i++) {
        std::vector<float> embedding;
        for (int j = i * queryDimension; j < (i + 1) * queryDimension; j++) {
            embedding.push_back(queryVecs[j]);
        }
        queryEmbeddings.push_back(embedding);
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

    size_t avgNodesVisited = 0;
    size_t avgDepth = 0;
    size_t maxDepth = 0;
    size_t avgHops = 0;
    double avgSearchTime = 0.0;
    int avgRecall = 0;
    for (int i = 0; i < queryNumVectors; i++) {
        auto query = queryEmbeddings[i];
        auto gt = groundTruth[i];
        auto res = hnsw.knnSearch(query, k, efSearch);
        for (auto nodeWithDistance: res.nodes) {
            if (std::find(gt.begin(), gt.end(), nodeWithDistance.item->id) != gt.end()) {
                avgRecall++;
            }
        }
        avgNodesVisited += res.nodesVisited;
        avgHops += res.hops;
        avgDepth += res.depth;
        maxDepth = std::max(maxDepth, res.depth);
        avgSearchTime += res.searchTime.count();
    }
    printf("\n=====================================\n");
    printf("efConstruction: %d\n", efConstructions);
    printf("m: %d\n", m);
    printf("m0: %d\n", m0);

//    printf("Build time: %f s\n\n", hnsw.buildTime.count());
//    size_t avgDegree, maxDegree, minDegree;
//    hnsw.getGraphStats(avgDegree, maxDegree, minDegree);
//    printf("Avg degree: %zu\n", avgDegree);
//    printf("Max degree: %zu\n", maxDegree);
//    printf("Min degree: %zu\n\n", minDegree);

    printf("efSearch: %d\n", efSearch);
    printf("kSearch: %d\n", k);
    printf("Avg search time: %f s\n", avgSearchTime / (double) queryNumVectors);
    printf("Avg recall: %zu/%d\n", avgRecall / queryNumVectors, k);
    printf("Avg nodes visited: %zu/%zu\n", avgNodesVisited / queryNumVectors, baseNumVectors);
    printf("Avg hops: %zu\n", avgHops / queryNumVectors);
    printf("Avg depth: %zu\n", avgDepth / queryNumVectors);
    printf("Max depth: %zu\n", maxDepth);
}
