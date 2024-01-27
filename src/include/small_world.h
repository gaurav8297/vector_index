#pragma once

#include <min_queue.h>

#include <vector>
#include <unordered_set>
#include <set>
#include <chrono>

namespace vector_index::small_world {
    struct Node {
        int id;
        std::vector<float> embedding;
        std::unordered_set<Node *> children;
    };

    struct Result {
        std::set<Record<Node*>> nodes;
        std::chrono::duration<double> searchTime;
        size_t nodesVisited;
        size_t hops;
        size_t depth;
    };

    class SmallWorldNG {
    public:
        SmallWorldNG(float *data, size_t dimension, size_t numVectors, int f, int w);

        void insert(std::vector<float> nodeEmbedding, int f, int w);

        Result trueKnnSearch(std::vector<float> &query, int k);

        Result beamKnnSearch(std::vector<float> &query, int b, int k);

        Result beamKnnSearch2(std::vector<float> &query, int b, int k);

        Result someOtherKnnSearch(std::vector<float> &query, int b, int k);

        Result greedyKnnSearch(std::vector<float> &query, int m, int k);

        void getGraphStats(size_t &avgDegree, size_t &maxDegree, size_t &minDegree);

        std::chrono::duration<double> buildTime;
    private:
        int id;
        std::vector<std::unique_ptr<Node>> nodes;
    };
} // namespace vector_index::small_world
