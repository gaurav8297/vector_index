#!/bin/bash

# Specify the output file
output_file="output.txt"

echo "Running benchmark for hnsw with M = 64"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw -efConstruction 100 -m 64 -pq_m 16 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"

echo "Running benchmark for hnsw_pq with pq_m = 16"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw_pq -efConstruction 100 -m 64 -pq_m 16 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"

echo "Running benchmark for hnsw_pq with pq_m = 60"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw_pq -efConstruction 100 -m 64 -pq_m 60 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"

echo "Running benchmark for hnsw_pq with pq_m = 120"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw_pq -efConstruction 100 -m 64 -pq_m 120 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"

echo "Running benchmark for hnsw_pq with pq_m = 480"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw_pq -efConstruction 100 -m 64 -pq_m 480 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"

echo "Running benchmark for hnsw_pq with pq_m = 480"
./build/release/bin/vector_index_main -f /home/gaurav/vector_index_experiments/gist -k 100 -t hnsw_pq -efConstruction 150 -m 128 -pq_m 480 -efSearch 128 -nIndexingThreads 16 -nSearchThreads 2,4,8,16 >> "$output_file"
