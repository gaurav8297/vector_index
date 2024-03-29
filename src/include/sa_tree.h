#pragma once

#include <vector>
#include <set>
#include <chrono>

namespace vector_index::sa_tree {
    struct Node {
        int id;
        std::vector<float> embedding;
        // The maximum distance from this node to any of its children.
        double radius;
        std::vector<std::unique_ptr<Node>> children;
    };

    struct QueueObject {
        Node* node;
        double weight;
        double digression;
        double distance;
    };

    struct NodeWithDistance {
        Node* node;
        double distance;
    };

    struct ResultObject {
        std::multiset<NodeWithDistance> nodes;
        std::chrono::duration<double> searchTime;
        size_t nodesVisited;
        size_t maxDepth;
    };

    class SATree {
    public:
        SATree(float* data, size_t dimension, size_t numVectors);
        ResultObject rangeSearch(std::vector<float> &query, double r, double digression);
        ResultObject knnSearch(std::vector<float> &query, int k);
        ResultObject beamKnnSearch2(std::vector<float> &query, int b, int k);
        ResultObject beamKnnSearch(std::vector<float> &query, int b, int k);
        ResultObject greedyKnnSearch(std::vector<float> &query, int m, int b, int k);
        void getGraphStats(size_t &avgDegree, size_t &maxDegree, size_t &minDegree);

    private:
        // TODO - implement incremental insert
        static void buildTree(Node* root, std::vector<std::unique_ptr<Node>> &availableNodes);
        void rangeSearch(Node* node, std::vector<float> &query, double distance, double r, double digression, std::multiset<NodeWithDistance> &result);

    private:
        std::unique_ptr<Node> root;
    public:
        size_t dimension;
        size_t numVectors;

        // Stats
        std::chrono::duration<double> buildTime;
        size_t nodesVisited;
    };
} // namespace vector_index::sa_tree
