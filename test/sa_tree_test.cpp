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
    auto baseVectorPath = fmt::format("{}/{}/{}_base.fvecs", basePath, benchmarkType, benchmarkType);
    auto queryVectorPath = fmt::format("{}/{}/{}_query.fvecs", basePath, benchmarkType, benchmarkType);
    auto gtVectorPath = fmt::format("{}/{}/{}_groundtruth.ivecs", basePath, benchmarkType, benchmarkType);

    float* baseVecs = Utils::fvecs_read(baseVectorPath.c_str(),&baseDimension,&baseNumVectors);

    auto saTree = SATree(baseVecs, baseDimension, baseNumVectors);
    printf("Build time: %f s\n", saTree.buildTime.count());

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

    auto avgNodesVisited = 0;
    double avgSearchTime = 0.0;

    for (int i = 0; i < queryEmbeddings.size(); i++) {
        auto query = queryEmbeddings[i];
        auto gt = groundTruth[i];
        auto res = saTree.knnSearch(query, 100);
        for (auto nodeWithDistance : res.nodes) {
            ASSERT_TRUE(std::find(gt.begin(), gt.end(), nodeWithDistance.node->id) != gt.end());
        }
        avgNodesVisited += res.nodesVisited;
        avgSearchTime += res.searchTime.count();
        printf("Nodes visited: %zu\n", res.nodesVisited);
        printf("Search time: %f s\n", res.searchTime.count());
    }
    printf("\n=====================================\n");
    printf("Build time: %f s\n", saTree.buildTime.count());
    printf("Avg nodes visited: %zu/%zu\n", avgNodesVisited / queryNumVectors, baseNumVectors);
    printf("Avg search time: %f s\n", avgSearchTime / (double) queryNumVectors);
}
