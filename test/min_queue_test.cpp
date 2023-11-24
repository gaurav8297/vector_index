#include "gtest/gtest.h"
#include "min_queue.h"

using namespace vector_index;
struct MM {
    int id;
    float distance;
};

TEST(MinQueueTest, Benchmark3) {
    MinQueue<MM*> queue(100);
    for (int i = 0; i < 1000000; i++) {
        auto item = new MM();
        item->id = i;
        item->distance = 100 + i;
        queue.insert({item, item->distance});
    }
    auto x = queue.last();
    printf("s");
}

