#pragma once

#include <vector>
#include <set>

namespace vector_index {
    struct SWNGNode {
        int id;
        std::vector<float> embedding;
        std::vector<SWNGNode*> children;
    };

    struct SWNGNodeWithDistance {
        SWNGNode* node;
        double distance;
    };

    struct SWNGResultObject {
        std::set<SWNGNodeWithDistance> nodes;
        std::chrono::duration<double> searchTime;
        size_t nodesVisited;
        size_t maxHops;
        size_t avgHops;
    };

    class SmallWorldNG {
    public:
        SmallWorldNG(float* data, size_t dimension, size_t numVectors, int f, int w);
        void insert(std::vector<float> nodeEmbedding, int f, int w);
        SWNGResultObject knnSearch(std::vector<float> &query, int m, int k);
        void getGraphStats(size_t &avgDegree, size_t &maxDegree, size_t &minDegree);

        std::chrono::duration<double> buildTime;
    private:
        int id;
        std::vector<std::shared_ptr<SWNGNode>> nodes;

    };
}