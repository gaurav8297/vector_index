#pragma once

#include <min_queue.h>

#include <vector>
#include <unordered_set>
#include <set>
#include <chrono>

namespace vector_index::hnsw {
    struct Node {
        int id;
        std::vector<float> embedding;
        std::vector<MinQueue<Node*>> children;
    };

    struct Result {
        std::set<Record<Node*>> nodes;
        std::chrono::duration<double> searchTime;
        size_t nodesVisited;
        size_t hops;
        size_t depth;
    };

    class HNSW {
    public:
        HNSW(float *data, size_t dimension, size_t numVectors, int efConstruction, int m, int m0);

        void insert(std::vector<float> &embedding, int efConstruction);

        MinQueue<Node *> searchLayer(std::vector<float> &query, MinQueue<Node*> entrypoints, int efSearch, int layer);

        std::set<Record<Node*>> searchNeighborsSimple(MinQueue<Node *> &elements, int m);

        std::set<Record<Node*>> selectNeighborsHeuristic(std::vector<float> &query, MinQueue<Node *> &elements, int m, int layer, bool extendCandidates, bool keepPrunedConnections);

        Result knnSearch(std::vector<float> &query, int k, int efSearch);

    private:
        int id;
        std::vector<std::unique_ptr<Node>> nodes;
        Node* entrypoint;
        int m;
        int m0;
        double mL;
        size_t nodesVisited;
    };
} // namespace vector_index::hnsw