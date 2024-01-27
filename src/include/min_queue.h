#pragma once

#include <vector>
#include <set>

namespace vector_index {
    template <typename T>
    struct Record {
        T item;
        double distance;
    };

    template <typename T>
    bool operator<(const Record<T> &x, const Record<T> &y) {
        if (x.distance == y.distance && x.item == y.item) {
            return false;
        } else if (x.distance == y.distance) {
            return true;
        }
        return x.distance < y.distance;
    }

    template <typename T>
    class MinQueue {
    public:
            explicit MinQueue(): maxSize{0} {};
            explicit MinQueue(size_t maxSize): maxSize{maxSize} {};
            inline void insert(Record<T> record) {
                if (records.size() < maxSize) {
                    records.insert(record);
                } else {
                    auto maxRecord = last();
                    if (record.distance < maxRecord.distance) {
                        records.erase(maxRecord);
                        records.insert(record);
                    }
                }
            }
            inline Record<T> pop() {
                auto last = records.last();
                records.erase(--records.end());
                return last;
            }

            inline size_t size() {
                return records.size();
            }

            inline Record<T> last() {
                return *(--records.end());
            }

            inline Record<T> top() {
                auto top = *(records.begin());
                records.erase(records.begin());
                return top;
            }

            // return records
            inline std::set<Record<T>> getRecords() const {
                return records;
            }
    private:
        size_t maxSize;
        std::set<Record<T>> records;
    };
} // namespace vector_index
