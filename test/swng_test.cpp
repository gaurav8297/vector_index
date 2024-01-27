#include "spdlog/fmt/fmt.h"
#include "gtest/gtest.h"
#include "small_world.h"
#include "utils.h"

using namespace vector_index;
using namespace vector_index::small_world;

TEST(SWGTest, Benchmark) {
    size_t baseDimension, baseNumVectors;
    auto basePath = "/Users/gauravsehgal/work/vector_index/data";
    auto benchmarkType = "siftsmall";
    auto baseVectorPath = fmt::format("{}/{}/base.fvecs", basePath, benchmarkType, benchmarkType);
    auto queryVectorPath = fmt::format("{}/{}/query.fvecs", basePath, benchmarkType, benchmarkType);
    auto gtVectorPath = fmt::format("{}/{}/groundtruth.ivecs", basePath, benchmarkType, benchmarkType);

    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);

    auto mCreates = {10};
    auto kCreates = {10};
    auto mSearches = {2};
    int kSearch = 100;

    for (auto mCreate : mCreates) {
        for (auto kCreate: kCreates) {
            auto swng = SmallWorldNG(baseVecs, baseDimension, baseNumVectors, mCreate, kCreate);
            printf("Build time: %f s\n", swng.buildTime.count());

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

            for (auto mSearch : mSearches) {
                size_t avgNodesVisited = 0;
                size_t avgDepth = 0;
                size_t maxDepth = 0;
                size_t avgHops = 0;
                double avgSearchTime = 0.0;
                int avgRecall = 0;
                for (int i = 0; i < queryNumVectors; i++) {
                    auto query = queryEmbeddings[i];
                    auto gt = groundTruth[i];
                    auto res = swng.greedyKnnSearch(query, mSearch, kSearch);
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
                printf("mCreate: %d\n", mCreate);
                printf("kCreate: %d\n", kCreate);
                printf("Build time: %f s\n\n", swng.buildTime.count());

                size_t avgDegree, maxDegree, minDegree;
                swng.getGraphStats(avgDegree, maxDegree, minDegree);
                printf("Avg degree: %zu\n", avgDegree);
                printf("Max degree: %zu\n", maxDegree);
                printf("Min degree: %zu\n\n", minDegree);

                printf("mSearch: %d\n", mSearch);
                printf("kSearch: %d\n", kSearch);
                printf("Avg search time: %f s\n", avgSearchTime / (double) queryNumVectors);
                printf("Avg recall: %zu/%d\n", avgRecall / queryNumVectors, kSearch);
                printf("Avg nodes visited: %zu/%zu\n", avgNodesVisited / queryNumVectors, baseNumVectors);
                printf("Avg hops: %zu\n", avgHops / queryNumVectors);
                printf("Avg depth: %zu\n", avgDepth / queryNumVectors);
                printf("Max depth: %zu\n", maxDepth);
            }
        }
    }
}
