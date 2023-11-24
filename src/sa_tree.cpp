#include <cmath>
#include <queue>
#include "include/sa_tree.h"
#include "utils.h"

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

    bool operator<(const QueueObject &x, const QueueObject &y) {
        return x.weight > y.weight;
    }

    bool operator<(const NodeWithDistance &x, const NodeWithDistance &y) {
        return x.distance < y.distance;
    }
} // namespace vector_index::sa_tree
