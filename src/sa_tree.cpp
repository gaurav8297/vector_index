#include <cmath>
#include <queue>
#include <unordered_set>
#include "include/sa_tree.h"
#include "include/utils.h"
#include "include/min_queue.h"

namespace vector_index::sa_tree {
    SATree::SATree(float *data, size_t dimension, size_t numVectors) {
        std::vector<std::unique_ptr<Node>> nodes;
        auto i = 0;
        while (i < numVectors) {
            std::vector<float> embedding;
            for (int j = i * dimension; j < (i+1) * dimension; j++) {
                embedding.push_back(data[j]);
            }
            auto node = std::make_unique<Node>();
            node->id = i;
            node->embedding = embedding;
            node->radius = 0;
            nodes.push_back(std::move(node));
            i++;
        }

        // Chose a random node as the root.
        this->root = std::move(nodes.back());
        nodes.pop_back();

        // Build the index
        auto start = std::chrono::high_resolution_clock::now();
        buildTree(this->root.get(), nodes);
        buildTime = std::chrono::high_resolution_clock::now() - start;

        this->dimension = dimension;
        this->numVectors = numVectors;
    }

    // 1. Random node is selected as root
    // 2. Sort the available nodes by distance from the root.
    // 3. For each node in the sorted list, if the node is closer to a root than any off its child, then add it to the child.
    // 4. If a node is closer to a child than add it to its respective available nodes list.
    // 5. Recursively build the tree for each child.
    void SATree::buildTree(Node* root, std::vector<std::unique_ptr<Node>> &availableNodes) {
        root->children.clear();
        root->radius = 0;
        // Sort the available nodes by distance from the root.
        std::sort(availableNodes.begin(), availableNodes.end(), [&root](std::unique_ptr<Node> &a, std::unique_ptr<Node> &b) {
            return Utils::l2_distance(a.get()->embedding, root->embedding) < Utils::l2_distance(b.get()->embedding, root->embedding);
        });

        std::vector<std::unique_ptr<Node>> nonChildrenNodes;
        for (auto &availableNode : availableNodes) {
            auto node = availableNode.get();
            auto dist = Utils::l2_distance(node->embedding, root->embedding);
            root->radius = std::max(root->radius, dist);
            auto flag = true;
            for (const auto &j : root->children) {
                auto child = j.get();
                auto child_dist = Utils::l2_distance(node->embedding, child->embedding);
                // If a node is closer to a child than to the root, then set the flag and break.
                if (child_dist <= dist) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                root->children.push_back(std::move(availableNode));
            } else {
                nonChildrenNodes.push_back(std::move(availableNode));
            }
        }
        auto len = root->children.size();
        std::vector<std::unique_ptr<Node>> childrenAvailableNodes[len];

        for (auto &nonChildrenNode : nonChildrenNodes) {
            auto node = nonChildrenNode.get();
            auto close_idx = 0;
            double min_dist = INFINITY;
            for (int i = 0; i < len; i++) {
                auto child = root->children[i].get();
                auto dist = Utils::l2_distance(node->embedding, child->embedding);
                if (dist < min_dist) {
                    min_dist = dist;
                    close_idx = i;
                }
            }
            childrenAvailableNodes[close_idx].push_back(std::move(nonChildrenNode));
        }

        for (int i = 0; i < len; i++) {
            buildTree(root->children[i].get(), childrenAvailableNodes[i]);
        }
    }

    // 1. Range search based on given query and radius.
    // 2. Only consider neighbours based on this triangle inequality.
    //    d(q, b) <= d(q, c) + 2r where b & c are a neighbour of some root. And c is closest to q. (Not q')
    // 2. digression of a node b the maximum d(q, b) - d(q, a) value for any a ancestor of b in the path from the root to b.
    // 3. Using MaxSuff we find the digression of the node.
    ResultObject SATree::rangeSearch(std::vector<float> &query, double r, double digression) {
        auto start = std::chrono::high_resolution_clock::now();
        this->nodesVisited = 1;
        std::multiset<NodeWithDistance> result;
        auto distance = Utils::l2_distance(root->embedding, query);
        rangeSearch(root.get(), query, distance, r, digression, result);
        auto end = std::chrono::high_resolution_clock::now();
        return {result, end - start, nodesVisited};
    }

    void SATree::rangeSearch(Node* node, std::vector<float> &query, double distance, double r, double digression, std::multiset<NodeWithDistance> &result) {
        // Digression should be less than 2 * r.
        // Distance should be less than the cover radius of the node + r.
        if (digression <= 2 * r && distance <= node->radius + r) {
            if (distance <= r) {
                result.insert({node, distance});
            }

            std::vector<double> childDistances;
            double min_dist = distance;
            for (const auto & i : node->children) {
                auto child = i.get();
                auto dist = Utils::l2_distance(child->embedding, query);
                this->nodesVisited += 1;
                childDistances.push_back(dist);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }

            for (int i = 0; i < node->children.size(); i++) {
                auto child = node->children[i].get();
                auto childDistance = childDistances.at(i);
                if (childDistance <= min_dist + (2 * r)) {
                    return rangeSearch(child, query, r, childDistance, std::max(digression, (childDistance - distance)), result);
                }
            }
        }
    }

    ResultObject SATree::knnSearch(std::vector<float> &query, int k) {
        auto start = std::chrono::high_resolution_clock::now();
        auto distance = Utils::l2_distance(root->embedding, query);
        this->nodesVisited = 1;
        std::priority_queue<QueueObject> queue;
        queue.push({root.get(), std::max(0.0, (distance - root->radius)), 0, distance});
        std::multiset<NodeWithDistance> result;
        double rad = INFINITY;
        while (!queue.empty()) {
            auto element = queue.top();
            queue.pop();
            if (element.weight > rad) {
                break;
            }
            auto elementDistance = element.distance;
            result.insert({element.node, elementDistance});
            if (result.size() >= k + 1) {
                result.erase(--result.end());
            }
            if (result.size() == k) {
                rad = (--result.end())->distance;
            }

            auto closest = NodeWithDistance{element.node, elementDistance};
            std::vector<double> childDistances;
            for (const auto &i : element.node->children) {
                auto child = i.get();
                auto childDistance = Utils::l2_distance(child->embedding, query);
                this->nodesVisited += 1;
                childDistances.push_back(childDistance);
                if (childDistance < closest.distance) {
                    closest = NodeWithDistance{child, childDistance};
                }
            }
            for (int i = 0; i < element.node->children.size(); i++) {
                auto child = element.node->children[i].get();
                auto childDistance = childDistances.at(i);
                auto dig = std::max(0.0, (element.digression + (childDistance - elementDistance)));
                auto weight = std::max(element.weight, std::max(dig, (childDistance - closest.distance) / 2));
                queue.push({child, std::max(weight, (childDistance - child->radius)), dig, childDistance});
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        return {result, end - start, nodesVisited};
    }

    ResultObject SATree::beamKnnSearch2(std::vector<float> &query, int b, int k) {
        MinQueue<Node *> beam(b);
        MinQueue<Node *> result(k);
        size_t nodesVisited = 0;
        size_t maxDepth = 0;
        std::unordered_set<int> visited;
        auto start = std::chrono::high_resolution_clock::now();
        beam.insert({root.get(), Utils::l2_distance(root->embedding, query)});
        result.insert({root.get(), Utils::l2_distance(root->embedding, query)});
        while (true) {
            double closestDistance = INFINITY;
            if (result.size() >= k) {
                closestDistance = result.last().distance;
            }
            MinQueue<Node *> newBeam(b);
            auto flag = false;
            for (auto record: beam.getRecords()) {
                for (const auto &childNode: record.item->children) {
                    if (visited.contains(childNode->id)) {
                        continue;
                    }
                    auto child = Record<Node *>{childNode.get(), Utils::l2_distance(childNode->embedding, query)};
                    nodesVisited++;
                    newBeam.insert(child);
                    visited.insert(childNode->id);
                    flag = true;
                    result.insert(child);
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

        // copy beam to result
        std::multiset<NodeWithDistance> newResult;
        for (auto nodeWithDistance: result.getRecords()) {
            newResult.insert({nodeWithDistance.item, nodeWithDistance.distance});
        }

        return {newResult, std::chrono::high_resolution_clock::now() - start, nodesVisited, maxDepth};
    }

    ResultObject SATree::beamKnnSearch(std::vector<float> &query, int b, int k) {
        MinQueue<Node *> beam(b);
        size_t nodesVisited = 0;
        size_t maxDepth = 0;
        std::unordered_set<int> visited;
        auto start = std::chrono::high_resolution_clock::now();
        beam.insert({root.get(), Utils::l2_distance(root->embedding, query)});

        while (true) {
            double closestDistance = INFINITY;
            if (beam.size() >= b) {
                closestDistance = beam.last().distance;
            }
            MinQueue<Node *> newBeam(b);
            auto flag = false;
            for (auto record: beam.getRecords()) {
                for (const auto &childNode: record.item->children) {
                    if (visited.contains(childNode->id)) {
                        continue;
                    }
                    auto child = Record<Node *>{childNode.get(), Utils::l2_distance(childNode->embedding, query)};
                    nodesVisited++;
                    newBeam.insert(child);
                    visited.insert(childNode->id);
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
        auto j = 0;
        std::multiset<NodeWithDistance> result;
        for (auto nodeWithDistance: beam.getRecords()) {
            if (j >= k) {
                break;
            }
            result.insert({nodeWithDistance.item, nodeWithDistance.distance});
            j++;
        }

        return {result, std::chrono::high_resolution_clock::now() - start, nodesVisited, maxDepth};
    }

    ResultObject SATree::greedyKnnSearch(std::vector<float> &query, int m, int b, int k) {
        auto start = std::chrono::high_resolution_clock::now();
        std::multiset<NodeWithDistance> result;
        size_t nodesVisited = 0;
        for (int i = 0; i < m; i++) {
            MinQueue<Node *> tmpResult(b);
            std::unordered_set<int> visited;
            // add results to visited
            auto p = 0;
            for (auto nodeWithDistance: result) {
                if (p >= b) {
                    break;
                }
                visited.insert(nodeWithDistance.node->id);
                p++;
            }
            std::priority_queue<Record<Node *>> candidates;
            candidates.push({root.get(), Utils::l2_distance(root->embedding, query)});
            nodesVisited++;
            while (!candidates.empty()) {
                auto closest = candidates.top();
                candidates.pop();
                if (tmpResult.size() >= b && tmpResult.last().distance < closest.distance) {
                    break;
                }

                for (const auto &childNode: closest.item->children) {
                    if (visited.contains(childNode->id)) {
                        continue;
                    }
                    auto child = Record<Node *>{childNode.get(), Utils::l2_distance(childNode->embedding, query)};
                    visited.insert(childNode->id);
                    candidates.push(child);
                    tmpResult.insert(child);
                    nodesVisited++;
                }
            }
            auto j = 0;
            for (auto nodeWithDistance: tmpResult.getRecords()) {
                if (j >= b) {
                    break;
                }
                result.insert({nodeWithDistance.item, nodeWithDistance.distance});
                j++;
            }
        }
        return {result, std::chrono::high_resolution_clock::now() - start, nodesVisited};
    }

    void SATree::getGraphStats(size_t &avgDegree, size_t &maxDegree, size_t &minDegree) {
        std::queue<Node *> queue;
        queue.push(root.get());
        size_t sumDegree = 0;
        maxDegree = 0;
        minDegree = INFINITY;
        size_t count = 0;
        while (!queue.empty()) {
            auto node = queue.front();
            queue.pop();
            if (node->children.empty()) {
                continue;
            }

            sumDegree += node->children.size();
            maxDegree = std::max(maxDegree, node->children.size());
            minDegree = std::min(minDegree, node->children.size());
            count++;
            for (const auto &childNode: node->children) {
                queue.push(childNode.get());
            }
        }
        avgDegree = sumDegree / count;
    }

    bool operator<(const QueueObject &x, const QueueObject &y) {
        return x.weight > y.weight;
    }

    bool operator<(const NodeWithDistance &x, const NodeWithDistance &y) {
        return x.distance < y.distance;
    }
} // namespace vector_index::sa_tree
