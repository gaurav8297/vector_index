#include "spdlog/fmt/fmt.h"
#include "gtest/gtest.h"
#include "utils.h"
#include "sa_tree.h"

using namespace vector_index;
using namespace vector_index::sa_tree;

TEST(SATreeTest, Benchmark1) {
    size_t baseDimension, baseNumVectors;
    auto basePath = "/Users/gauravsehgal/work/vector_index/data";
    auto benchmarkType = "siftsmall";
    auto baseVectorPath = fmt::format("{}/{}/base.fvecs", basePath, benchmarkType, benchmarkType);
    auto queryVectorPath = fmt::format("{}/{}/query.fvecs", basePath, benchmarkType, benchmarkType);
    auto gtVectorPath = fmt::format("{}/{}/groundtruth.ivecs", basePath, benchmarkType, benchmarkType);

    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);

    int kSearch = 100;
    auto bSearches = {16, 32, 64, 100, 128, 256, 512, 1024};

    auto saTree = SATree(baseVecs, baseDimension, baseNumVectors);

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

    for (auto bSearch: bSearches) {
        auto avgNodesVisited = 0;
        double avgSearchTime = 0.0;
        int avgRecall = 0;
        auto avgDepth = 0;
        size_t maxDepth = 0;
        for (int i = 0; i < queryEmbeddings.size(); i++) {
            auto query = queryEmbeddings[i];
            auto gt = groundTruth[i];
            auto res = saTree.beamKnnSearch2(query, bSearch, kSearch);
            auto recall = 0;
            int j = 0;
            for (auto nodeWithDistance: res.nodes) {
                if (j++ >= kSearch) {
                    break;
                }
                if (std::find(gt.begin(), gt.end(), nodeWithDistance.node->id) != gt.end()) {
                    avgRecall++;
                    recall++;
                }
            }
            avgNodesVisited += res.nodesVisited;
            avgSearchTime += res.searchTime.count();
            avgDepth += res.maxDepth;
            maxDepth = std::max(maxDepth, res.maxDepth);
        }
        printf("\n=====================================\n");
        printf("bSearch: %d\n", bSearch);
        printf("Build time: %f s\n", saTree.buildTime.count());

        size_t avgDegree, maxDegree, minDegree;
        saTree.getGraphStats(avgDegree, maxDegree, minDegree);
        printf("Avg degree: %zu\n", avgDegree);
        printf("Max degree: %zu\n", maxDegree);
        printf("Min degree: %zu\n\n", minDegree);

        printf("Avg nodes visited: %zu/%zu\n", avgNodesVisited / queryNumVectors, baseNumVectors);
        printf("Avg search time: %f s\n", avgSearchTime / (double) queryNumVectors);
        printf("Avg recall: %zu/%d\n", avgRecall / queryNumVectors, kSearch);
        printf("Avg depth: %zu\n", avgDepth / queryNumVectors);
        printf("Max depth: %zu\n", maxDepth);
    }
}
