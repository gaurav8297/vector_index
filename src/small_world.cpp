#include <unordered_set>
#include <queue>
#include "include/small_world.h"
#include "include/utils.h"
#include <random>

namespace vector_index {
    SmallWorldNG::SmallWorldNG(float *data, size_t dimension, size_t numVectors, int m, int k) {
        id = 0;
        auto i = 0;
        std::vector<std::vector<float>> embeddings;
        while (i < numVectors) {
            std::vector<float> embedding;
            for (int j = i * dimension; j < (i+1) * dimension; j++) {
                embedding.push_back(data[j]);
            }
            embeddings.push_back(embedding);
            i++;
        }

        auto start = std::chrono::high_resolution_clock::now();
        i = 0;
        for (auto embedding : embeddings) {
            insert(embedding, m, k);
            if (i % 10000 == 0) {
                printf("Inserted %d nodes\n", i);
            }
            i++;
        }

        auto end = std::chrono::high_resolution_clock::now();
        buildTime = end - start;
    }

    SWNGResultObject SmallWorldNG::knnSearch(std::vector<float> &query, int m, int k) {
        std::unordered_set<int> visited;
        std::set<SWNGNodeWithDistance> result;
        size_t hops = 0;
        size_t maxDepth = 0;
        size_t nodesVisited = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < m; i++) {
            std::set<SWNGNodeWithDistance> tmpResult;
            std::priority_queue<SWNGNodeWithDistance> candidates;
            int rand = Utils::rand_int(0, nodes.size() - 1);
            if (visited.size() >= nodes.size()) {
                break;
            }
            while (visited.contains(rand)) {
                rand = Utils::rand_int(0, nodes.size() - 1);
            }
            auto entrypoint = nodes.at(rand).get();
            auto startNode = SWNGNodeWithDistance{entrypoint, Utils::l2_distance(entrypoint->embedding, query)};
            candidates.push(startNode);
            nodesVisited++;
            size_t depth = 0;
            while (!candidates.empty()) {
                auto closest = candidates.top();
                candidates.pop();
                if (tmpResult.size() >= k && std::next(tmpResult.begin(), (k - 1))->distance < closest.distance) {
                    break;
                }
                if (result.size() >= k && std::next(result.begin(), (k - 1))->distance < closest.distance) {
                    break;
                }
                auto countDepth = false;
                for (auto childNode : closest.node->children) {
                    if (visited.contains(childNode->id)) {
                        continue;
                    }
                    auto child = SWNGNodeWithDistance{childNode, Utils::l2_distance(childNode->embedding, query)};
                    visited.insert(childNode->id);
                    candidates.push(child);
                    tmpResult.insert(child);
                    hops++;
                    countDepth = true;
                    nodesVisited++;
                }
                if (countDepth) {
                    depth++;
                }
            }
            maxDepth = std::max(maxDepth, depth);

            // push top k nodes from tmpResult to result
            int j = 0;
            for (auto nodeWithDistance : tmpResult) {
                if (j >= k) {
                    break;
                }
                result.insert(nodeWithDistance);
                j++;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return SWNGResultObject{result, end - start, nodesVisited, hops / m, maxDepth};
    }

    void SmallWorldNG::insert(std::vector<float> nodeEmbedding, int m, int k) {
        auto node = std::make_shared<SWNGNode>();
        node->embedding = nodeEmbedding;
        node->id = id++;
        node->children = std::unordered_set<SWNGNode*>();
        if (nodes.size() < k) {
            for (auto &n : nodes) {
                n->children.insert(node.get());
                node->children.insert(n.get());
            }
            nodes.push_back(node);
            return;
        }

        auto result = knnSearch(nodeEmbedding, m, k);
        int j = 0;
        for (auto res : result.nodes) {
            if (j >= k) {
                break;
            }
            node->children.insert(res.node);
            res.node->children.insert(node.get());
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
