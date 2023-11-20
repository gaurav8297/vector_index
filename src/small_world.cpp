#include <unordered_set>
#include <queue>
#include "include/small_world.h"
#include "include/utils.h"

namespace vector_index {
    SmallWorldNG::SmallWorldNG(float *data, size_t dimension, size_t numVectors, int m, int k) {
        id = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto i = 0;
        while (i < numVectors) {
            std::vector<float> embedding;
            for (int j = i * dimension; j < (i+1) * dimension; j++) {
                embedding.push_back(data[j]);
            }
            insert(embedding, m, k);
            i++;
            if (i % 100 == 0) {
                printf("Inserted %d nodes\n", i);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        buildTime = end - start;
    }

    SWNGResultObject SmallWorldNG::knnSearch(std::vector<float> &query, int m, int k) {
        std::priority_queue<SWNGNodeWithDistance> candidates;
        std::set<SWNGNodeWithDistance> tmpResult;
        std::unordered_set<int> visited;
        size_t visitedCount = 0;
        size_t maxHops = 0;
        size_t avgHops = 0;
        size_t l = 0;
        auto start = std::chrono::high_resolution_clock::now();
        int oldRand = -1;
        for (int i = 0; i < m; i++) {
            int rand = Utils::rand_int(0, nodes.size() - 1);
            if (rand == oldRand) {
                continue;
            }
            oldRand = rand;
            auto entrypoint = nodes.at(rand).get();
            auto startNode = SWNGNodeWithDistance{entrypoint, Utils::l2_distance(entrypoint->embedding, query)};
            candidates.push(startNode);
            tmpResult.insert(startNode);
            visited.insert(entrypoint->id);
            visitedCount++;
            size_t hops = 0;
            while (!candidates.empty()) {
                auto closest = candidates.top();
                candidates.pop();
                if (tmpResult.size() >= k && std::next(tmpResult.begin(), (k - 1))->distance < closest.distance) {
                    break;
                }
                for (auto childNode : closest.node->children) {
                    if (visited.find(childNode->id) != visited.end()) {
                        continue;
                    }
                    auto child = SWNGNodeWithDistance{childNode, Utils::l2_distance(childNode->embedding, query)};
                    visited.insert(childNode->id);
                    candidates.push(child);
                    tmpResult.insert(child);
                    visitedCount++;
                }
                hops++;
            }
            maxHops = std::max(maxHops, hops);
            avgHops += hops;
            l++;
        }
        auto end = std::chrono::high_resolution_clock::now();
        return SWNGResultObject{tmpResult, end - start, visitedCount, maxHops, avgHops / l};
    }

    void SmallWorldNG::insert(std::vector<float> nodeEmbedding, int m, int k) {
        auto node = std::make_shared<SWNGNode>();
        node->embedding = nodeEmbedding;
        node->id = id++;
        node->children = std::vector<SWNGNode*>();
        if (nodes.empty()) {
            nodes.push_back(node);
            return;
        }

        auto result = knnSearch(nodeEmbedding, m, k);
        int j = 0;
        for (auto res : result.nodes) {
            if (j >= k) {
                break;
            }
            node->children.push_back(res.node);
            res.node->children.push_back(node.get());
            j++;
        }
        nodes.push_back(node);
    }

    void SmallWorldNG::getGraphStats(size_t &avgDegree, size_t &maxDegree, size_t &minDegree) {
        avgDegree = 0.0;
        maxDegree = 0.0;
        minDegree = INFINITY;
        for (const auto& node : nodes) {
            auto degree = node->children.size();
            avgDegree += degree;
            if (degree > maxDegree) {
                maxDegree = degree;
            }
            if (degree < minDegree) {
                minDegree = degree;
            }
        }
        avgDegree /= nodes.size();
    }

    bool operator<(const SWNGNodeWithDistance &x, const SWNGNodeWithDistance &y) {
        return x.distance < y.distance;
    }

    bool operator==(const SWNGNodeWithDistance &x, const SWNGNodeWithDistance &y) {
        return x.node->id == y.node->id;
    }
}
