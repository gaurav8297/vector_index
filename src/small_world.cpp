#include <unordered_set>
#include <queue>
#include <memory>
#include "include/small_world.h"
#include "include/utils.h"
#include "include/min_queue.h"
#include <random>
#include <chrono>

namespace vector_index::small_world {
    SmallWorldNG::SmallWorldNG(float *data, size_t dimension, size_t numVectors, int m, int k) {
        id = 0;
        auto i = 0;
        std::vector<std::vector<float>> embeddings;
        while (i < numVectors) {
            std::vector<float> embedding;
            for (int j = i * dimension; j < (i + 1) * dimension; j++) {
                embedding.push_back(data[j]);
            }
            embeddings.push_back(embedding);
            i++;
        }

        auto start = std::chrono::high_resolution_clock::now();
        i = 0;
        for (auto embedding: embeddings) {
            insert(embedding, m, k);
            if (i % 10000 == 0) {
                printf("Inserted %d nodes\n", i);
            }
            i++;
        }

        auto end = std::chrono::high_resolution_clock::now();
        buildTime = end - start;
    }

    void SmallWorldNG::insert(std::vector<float> nodeEmbedding, int m, int k) {
        auto node = std::make_unique<Node>();
        node->embedding = nodeEmbedding;
        node->id = id++;
        node->children = std::unordered_set<Node*>();
        if (nodes.size() < k) {
            for (auto &n: nodes) {
                n->children.insert(node.get());
                node->children.insert(n.get());
            }
            nodes.push_back(std::move(node));
            return;
        }

        auto result = greedyKnnSearch(nodeEmbedding, m, k);
        for (auto record: result.nodes) {
            node->children.insert(record.item);
            record.item->children.insert(node.get());
        }
        nodes.push_back(std::move(node));
    }

    Result SmallWorldNG::trueKnnSearch(std::vector<float> &query, int k) {
        MinQueue<Node *> result(k);
        auto start = std::chrono::high_resolution_clock::now();
        for (auto &node: nodes) {
            auto dist = Utils::l2_distance(node->embedding, query);
            result.insert({node.get(), dist});
        }
        auto end = std::chrono::high_resolution_clock::now();
        return Result{result.getRecords(), end - start, nodes.size(), 0, 0};
    }

    Result SmallWorldNG::beamKnnSearch(std::vector<float> &query, int b, int k) {
        MinQueue<Node *> beam(b);
        std::unordered_set<int> visited;
        size_t nodesVisited = 0;
        auto start = std::chrono::high_resolution_clock::now();
        size_t maxDepth = 0;
        for (int i = 0; i < b; i++) {
            if (visited.size() >= nodes.size()) {
                break;
            }
            int entryPointIdx = Utils::rand_int(0, nodes.size() - 1);
            while (visited.contains(entryPointIdx)) {
                entryPointIdx = Utils::rand_int(0, nodes.size() - 1);
            }
            auto entryPoint = Record<Node*>{nodes.at(entryPointIdx).get(), Utils::l2_distance(nodes.at(entryPointIdx).get()->embedding, query)};
            nodesVisited++;
            beam.insert(entryPoint);
            visited.insert(entryPoint.item->id);
        }

        while (true) {
            auto closestDistance = beam.last().distance;
            MinQueue<Node *> newBeam(b);
            auto flag = false;
            for (auto record: beam.getRecords()) {
                for (auto childNode: record.item->children) {
                    if (visited.contains(childNode->id)) {
                        continue;
                    }
                    auto child = Record<Node *>{childNode, Utils::l2_distance(childNode->embedding, query)};
                    nodesVisited++;
                    visited.insert(childNode->id);
                    newBeam.insert(child);
                    flag = true;
                }
            }
            if (flag) {
                maxDepth++;
            }
            for (auto record: newBeam.getRecords()) {
                beam.insert(record);
            }
            if (beam.last().distance >= closestDistance) {
                break;
            }
        }

        // copy beam to result
        MinQueue<Node *> result(k);
        for (auto nodeWithDistance: beam.getRecords()) {
            result.insert(nodeWithDistance);
        }

        return Result{result.getRecords(), std::chrono::high_resolution_clock::now() - start, nodesVisited, 0, maxDepth};
    }

    Result SmallWorldNG::beamKnnSearch2(std::vector<float> &query, int b, int k) {
        MinQueue<Node *> beam(b);
        MinQueue<Node *> result(k);
        std::unordered_set<int> visited;
        size_t nodesVisited = 0;
        auto start = std::chrono::high_resolution_clock::now();
        size_t maxDepth = 0;
        for (int i = 0; i < b; i++) {
            if (visited.size() >= nodes.size()) {
                break;
            }
            int entryPointIdx = Utils::rand_int(0, nodes.size() - 1);
            while (visited.contains(entryPointIdx)) {
                entryPointIdx = Utils::rand_int(0, nodes.size() - 1);
            }
            auto entryPoint = Record<Node*>{nodes.at(entryPointIdx).get(), Utils::l2_distance(nodes.at(entryPointIdx).get()->embedding, query)};
            nodesVisited++;
            beam.insert(entryPoint);
            result.insert(entryPoint);
            visited.insert(entryPoint.item->id);
        }

        while (true) {
            double closestDistance = INFINITY;
            if (result.size() >= k) {
                closestDistance = result.last().distance;
            }
            MinQueue<Node *> newBeam(b);
            auto flag = false;
            for (auto record: beam.getRecords()) {
                for (auto childNode: record.item->children) {
                    if (visited.contains(childNode->id)) {
                        continue;
                    }
                    auto child = Record<Node *>{childNode, Utils::l2_distance(childNode->embedding, query)};
                    nodesVisited++;
                    visited.insert(childNode->id);
                    newBeam.insert(child);
                    result.insert(child);
                    flag = true;
                }
            }
            if (flag) {
                maxDepth++;
            }
            for (auto record: newBeam.getRecords()) {
                beam.insert(record);
            }
            if (result.last().distance >= closestDistance) {
                break;
            }
        }

        return Result{result.getRecords(), std::chrono::high_resolution_clock::now() - start, nodesVisited, 0, maxDepth};
    }

    Result SmallWorldNG::someOtherKnnSearch(std::vector<float> &query, int b, int k) {
        MinQueue<Node *> beam(b);
        std::priority_queue<Record<Node*>> candidates;
        std::unordered_set<int> visited;
        size_t hops = 0;
        size_t maxDepth = 0;
        size_t nodesVisited = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < b; i++) {
            if (visited.size() >= nodes.size()) {
                break;
            }
            int entryPointIdx = Utils::rand_int(0, nodes.size() - 1);
            while (visited.contains(entryPointIdx)) {
                entryPointIdx = Utils::rand_int(0, nodes.size() - 1);
            }
            auto entryPoint = Record<Node*>{nodes.at(entryPointIdx).get(), Utils::l2_distance(nodes.at(entryPointIdx).get()->embedding, query)};
            nodesVisited++;
            beam.insert(entryPoint);
            visited.insert(entryPoint.item->id);
            candidates.push(entryPoint);
        }

        while (!candidates.empty()) {
            auto closest = candidates.top();
            candidates.pop();
            if (beam.size() >= b && beam.last().distance < closest.distance) {
                break;
            }

            for (auto childNode: closest.item->children) {
                if (visited.contains(childNode->id)) {
                    continue;
                }
                auto child = Record<Node *>{childNode, Utils::l2_distance(childNode->embedding, query)};
                nodesVisited++;
                visited.insert(childNode->id);
                candidates.push(child);
                beam.insert(child);
            }
        }

        // copy beam to result
        MinQueue<Node *> result(k);
        for (auto nodeWithDistance: beam.getRecords()) {
            result.insert(nodeWithDistance);
        }

        auto end = std::chrono::high_resolution_clock::now();
        return Result{result.getRecords(), end - start, nodesVisited, 0, 0};
    }

    Result SmallWorldNG::greedyKnnSearch(std::vector<float> &query, int m, int k) {
        std::unordered_set<int> visited;
        MinQueue<Node *> result(k);
        size_t hops = 0;
        size_t maxDepth = 0;
        size_t nodesVisited = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < m; i++) {
            MinQueue<Node *> tmpResult(k);
            MinQueue<Node *> candidates(k + 1);
            int rand = Utils::rand_int(0, nodes.size() - 1);
            if (visited.size() >= nodes.size()) {
                break;
            }
//            while (visited.contains(rand)) {
//                rand = Utils::rand_int(0, nodes.size() - 1);
//            }
            auto entrypoint = nodes.at(rand).get();
            candidates.insert({entrypoint, Utils::l2_distance(entrypoint->embedding, query)});
            nodesVisited++;
            size_t depth = 0;
            while (candidates.size() != 0) {
                auto closest = candidates.top();
                if (tmpResult.size() >= k && tmpResult.last().distance < closest.distance) {
                    break;
                }
//                if (result.size() >= k && result.last().distance < closest.distance) {
//                    break;
//                }
                auto countDepth = false;
                for (auto childNode: closest.item->children) {
                    if (visited.contains(childNode->id)) {
                        continue;
                    }
                    auto child = Record<Node*>{childNode, Utils::l2_distance(childNode->embedding, query)};
                    visited.insert(childNode->id);
                    candidates.insert(child);
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
            for (auto nodeWithDistance: tmpResult.getRecords()) {
                result.insert(nodeWithDistance);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return Result{result.getRecords(), end - start, nodesVisited, hops / m, maxDepth};
    }

    void SmallWorldNG::getGraphStats(size_t &avgDegree, size_t &maxDegree, size_t &minDegree) {
        avgDegree = 0.0;
        maxDegree = 0.0;
        minDegree = INFINITY;
        for (const auto &node: nodes) {
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
} // namespace vector_index::small_world
