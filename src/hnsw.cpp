#include "include/hnsw.h"
#include "include/utils.h"

namespace vector_index::hnsw {
    HNSW::HNSW(float *data, size_t dimension, size_t numVectors, int efConstruction, int m, int m0): m(m), m0(m0), entrypoint(nullptr), id(0), nodesVisited(0) {
        mL = 1.0 / log(m);
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

        i = 0;
        for (auto embedding: embeddings) {
            insert(embedding, efConstruction);
            if (i % 10000 == 0) {
                printf("Inserted %d nodes\n", i);
            }
            i++;
        }
    }

    void HNSW::insert(std::vector<float> &embedding, int efConstruction) {
        auto layer = size_t(-log(Utils::rand_double()) * mL);
        if (entrypoint == nullptr) {
            // TODO - insert first node
            auto node = std::make_unique<Node>();
            node->id = id++;
            node->embedding = embedding;
            node->children = std::vector<MinQueue<Node*>>(layer + 1);
            for (int i = 0; i <= layer; i++) {
                if (i == 0) {
                    node->children[i] = MinQueue<Node*>(m0);
                } else {
                    node->children[i] = MinQueue<Node*>(m);
                }
            }
            entrypoint = node.get();
            nodes.push_back(std::move(node));
            return;
        }

        // initialize node
        auto node = std::make_unique<Node>();
        node->id = id++;
        node->embedding = embedding;
        node->children = std::vector<MinQueue<Node*>>(layer + 1);
        for (int i = 0; i <= layer; i++) {
            if (i == 0) {
                node->children[i] = MinQueue<Node*>(m0);
            } else {
                node->children[i] = MinQueue<Node*>(m);
            }
        }


        auto ep = MinQueue<Node*>(1);
        ep.insert(Record<Node*>{entrypoint, Utils::l2_distance(entrypoint->embedding, embedding)});
        auto maxLayer = entrypoint->children.size() - 1;

        for (int i = maxLayer; i > layer; i--) {
            ep = searchLayer(embedding, ep, 1, i);
        }

        auto mMax = m;
        auto startLayer = std::min(layer, maxLayer);
        for (int i = startLayer; i >= 0; i--) {
            if (i == 0) {
                mMax = m0;
            }
            ep = searchLayer(embedding, ep, efConstruction, i);
            auto mNeighbors = searchNeighborsSimple(ep, mMax);
            for (auto neighbor: mNeighbors) {
                node->children[i].insert(neighbor);
                neighbor.item->children[i].insert(Record<Node*>{node.get(), neighbor.distance});
            }
        }

        if (layer > maxLayer) {
            entrypoint = node.get();
        }
        nodes.push_back(std::move(node));
    }

    MinQueue<Node*> HNSW::searchLayer(std::vector<float> &query, MinQueue<Node*> entrypoints, int efSearch, int layer) {
        auto mNeighbors = MinQueue<Node*>(efSearch);
        std::unordered_set<int> visited;
        auto candidates = MinQueue<Node*>(SIZE_MAX);

        for (auto ep: entrypoints.getRecords()) {
            mNeighbors.insert(ep);
            candidates.insert(ep);
            visited.insert(ep.item->id);
        }

        while (candidates.size() > 0) {
            auto closest = candidates.top();
            auto furthest = mNeighbors.last();
            if (mNeighbors.size() >= efSearch && furthest.distance < closest.distance) {
                break;
            }
            for (auto neighbor: closest.item->children[layer].getRecords()) {
                if (visited.contains(neighbor.item->id)) {
                    continue;
                }
                auto child = Record<Node*>{neighbor.item, Utils::l2_distance(neighbor.item->embedding, query)};
                nodesVisited++;
                visited.insert(neighbor.item->id);
                if (mNeighbors.size() < efSearch || furthest.distance > child.distance) {
                    mNeighbors.insert(child);
                    candidates.insert(child);
                }
            }
        }
        return mNeighbors;
    }

    std::set<Record<Node*>> HNSW::searchNeighborsSimple(MinQueue<Node*> &elements, int mMax) {
        std::set<Record<Node*>> neighbors;
        auto i = 0;
        for (auto element: elements.getRecords()) {
            if (i++ >= mMax) {
                break;
            }
            neighbors.insert(element);
        }
        return neighbors;
    }

    Result HNSW::knnSearch(std::vector<float> &query, int k, int efSearch) {
        auto start = std::chrono::high_resolution_clock::now();
        nodesVisited = 0;
        size_t maxLayer = entrypoint->children.size() - 1;
        auto ep = MinQueue<Node*>(1);
        ep.insert(Record<Node*>{entrypoint, Utils::l2_distance(entrypoint->embedding, query)});
        for (int i = maxLayer; i >= 1; i--) {
            ep = searchLayer(query, ep, 1, i);
        }

        ep = searchLayer(query, ep, efSearch, 0);
        return Result{searchNeighborsSimple(ep, k), std::chrono::high_resolution_clock::now() - start, nodesVisited, 0, 0};
    }
} // namespace vector_index::hnsw
