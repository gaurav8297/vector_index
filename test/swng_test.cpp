#include "spdlog/fmt/fmt.h"
#include "gtest/gtest.h"
#include "small_world.h"
#include "utils.h"

using namespace vector_index;

TEST(SWGTest, Benchmark) {
    size_t baseDimension, baseNumVectors;
    auto basePath = "/Users/gauravsehgal/work/vector_index/data";
    auto benchmarkType = "gist";
    auto baseVectorPath = fmt::format("{}/{}/{}_base.fvecs", basePath, benchmarkType, benchmarkType);
    auto queryVectorPath = fmt::format("{}/{}/{}_query.fvecs", basePath, benchmarkType, benchmarkType);
    auto gtVectorPath = fmt::format("{}/{}/{}_groundtruth.ivecs", basePath, benchmarkType, benchmarkType);

    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);

    int mCreate = 100;
    int kCreate = 100;

    int mSearch = 10000;
    int kSearch = 10;

    auto swng = SmallWorldNG(baseVecs, baseDimension, baseNumVectors, mCreate, kCreate);
    printf("Build time: %f s\n", swng.buildTime.count());

    size_t queryDimension, queryNumVectors;
    float* queryVecs = Utils::fvecs_read(queryVectorPath.c_str(), &queryDimension, &queryNumVectors);
    std::vector<std::vector<float>> queryEmbeddings;
    for (int i = 0; i < queryNumVectors; i++) {
        std::vector<float> embedding;
        for (int j = i * queryDimension; j < (i+1) * queryDimension; j++) {
            embedding.push_back(queryVecs[j]);
        }
        queryEmbeddings.push_back(embedding);
    }

    size_t gtDimension, gtNumVectors;
    int* gtVecs = Utils::ivecs_read(gtVectorPath.c_str(), &gtDimension, &gtNumVectors);
    std::vector<std::vector<int>> groundTruth;
    for (int i = 0; i < gtNumVectors; i++) {
        std::vector<int> gt;
        for (int j = i * gtDimension; j < (i+1) * gtDimension; j++) {
            gt.push_back(gtVecs[j]);
        }
        groundTruth.push_back(gt);
    }

    size_t vectorsToQuery = 10;
    auto avgNodesVisited = 0;
    double avgSearchTime = 0.0;
    int avgRecall = 0;
    size_t maxHops = 0;
    size_t avgHops = 0;
    for (int i = 0; i < vectorsToQuery; i++) {
        auto query = queryEmbeddings[i];
        auto gt = groundTruth[i];
        auto res = swng.knnSearch(query, mSearch, kSearch);
        int j = 0;
        for (auto nodeWithDistance : res.nodes) {
            if (j >= kSearch) {
                break;
            }
            if (std::find(gt.begin(), gt.end(), nodeWithDistance.node->id) != gt.end()) {
                avgRecall++;
            }
            j++;
        }
        avgNodesVisited += res.nodesVisited;
        avgSearchTime += res.searchTime.count();
        maxHops = std::max(maxHops, res.maxHops);
        avgHops += res.avgHops;
    }
    printf("\n=====================================\n");
    printf("Build time: %f s\n", swng.buildTime.count());
    printf("Avg nodes visited: %zu/%zu\n", avgNodesVisited / vectorsToQuery, baseNumVectors);
    printf("Avg search time: %f s\n", avgSearchTime / (double) vectorsToQuery);
    printf("Avg recall: %zu/%d\n", avgRecall / vectorsToQuery, kSearch);
    printf("Max hops: %zu\n", maxHops);
    printf("Avg hops: %zu\n", avgHops / vectorsToQuery);

    size_t avgDegree, maxDegree, minDegree;
    swng.getGraphStats(avgDegree, maxDegree, minDegree);
    printf("Avg degree: %zu\n", avgDegree);
    printf("Max degree: %zu\n", maxDegree);
    printf("Min degree: %zu\n", minDegree);
}
