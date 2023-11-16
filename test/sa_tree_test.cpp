#include "gtest/gtest.h"
#include "utils.h"
#include "sa_tree.h"

using namespace vector_index;

TEST(SATreeTest, Benchmark) {
    size_t baseDimension, baseNumVectors;
    float* baseVecs = Utils::fvecs_read("/Users/gauravsehgal/work/vector_index/data/gist/gist_base.fvecs", &baseDimension, &baseNumVectors);

    auto saTree = SATree(baseVecs, baseDimension, baseNumVectors);
    printf("Build time: %f s\n", saTree.buildTime.count());

    size_t queryDimension, queryNumVectors;
    float* queryVecs = Utils::fvecs_read("/Users/gauravsehgal/work/vector_index/data/gist/gist_query.fvecs", &queryDimension, &queryNumVectors);
    std::vector<std::vector<float>> queryEmbeddings;
    for (int i = 0; i < queryNumVectors; i++) {
        std::vector<float> embedding;
        for (int j = i * queryDimension; j < (i+1) * queryDimension; j++) {
            embedding.push_back(queryVecs[j]);
        }
        queryEmbeddings.push_back(embedding);
    }

    size_t gtDimension, gtNumVectors;
    int* gtVecs = Utils::ivecs_read("/Users/gauravsehgal/work/vector_index/data/gist/gist_groundtruth.ivecs", &gtDimension, &gtNumVectors);
    std::vector<std::vector<int>> groundTruth;
    for (int i = 0; i < gtNumVectors; i++) {
        std::vector<int> gt;
        for (int j = i * gtDimension; j < (i+1) * gtDimension; j++) {
            gt.push_back(gtVecs[j]);
        }
        groundTruth.push_back(gt);
    }

    for (int i = 0; i < queryEmbeddings.size(); i++) {
        auto query = queryEmbeddings[i];
        auto gt = groundTruth[i];
        auto res = saTree.knnSearch(query, 100);
        printf("Nodes visited: %zu\n", res.nodesVisited);
        printf("Search time: %f s\n", res.searchTime.count());
    }
}

