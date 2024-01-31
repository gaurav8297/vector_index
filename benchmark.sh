#!/bin/bash

# Specify the output file
output_file="output_2.txt"

echo "Running benchmark for hnsw_pq with pq_m = 4, pq_bits = 16"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw_pq -efConstruction 100 -m 64 -pq_m 16 -pq_bits 16 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"

echo "Running benchmark for hnsw_pq with pq_m = 16, pq_bits = 16"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw_pq -efConstruction 100 -m 64 -pq_m 60 -pq_bits 16 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"

echo "Running benchmark for hnsw_pq with pq_m = 4, pq_bits = 32"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw_pq -efConstruction 100 -m 64 -pq_m 16 -pq_bits 32 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"

echo "Running benchmark for hnsw_pq with pq_m = 16, pq_bits = 32"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw_pq -efConstruction 100 -m 64 -pq_m 60 -pq_bits 32 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"

echo "Running benchmark for hnsw_pq with pq_m = 4, pq_bits = 64"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw_pq -efConstruction 100 -m 64 -pq_m 16 -pq_bits 64 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"

echo "Running benchmark for hnsw_pq with pq_m = 8, pq_bits = 64"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw_pq -efConstruction 100 -m 64 -pq_m 60 -pq_bits 64 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"
