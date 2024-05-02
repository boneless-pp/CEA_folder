#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define MAX_ITER 1000
#define DAMPING_FACTOR 0.85
#define CONVERGENCE_THRESHOLD 0.000001

typedef struct {
    int from;
    int to;
} Edge;

typedef struct {
    double score;
    int node;
} NodeScore;

int compareNodeScore(const void* a, const void* b) {
    NodeScore na = *(NodeScore*)a;
    NodeScore nb = *(NodeScore*)b;
    if (na.score > nb.score) return -1;
    if (na.score < nb.score) return 1;
    return 0;
}

void powerMethod(double* pagerank, Edge* edges, long long edgeCount, int nodes) {
    double* tempRank = calloc(nodes, sizeof(double));
    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (long long i = 0; i < edgeCount; i++) {
            tempRank[edges[i].to] += pagerank[edges[i].from] / (double)nodes; // Simplification
        }
        double diff = 0.0;
        for (int i = 0; i < nodes; i++) {
            tempRank[i] = DAMPING_FACTOR * tempRank[i] + (1.0 - DAMPING_FACTOR) / nodes;
            diff += fabs(tempRank[i] - pagerank[i]);
            pagerank[i] = tempRank[i];
            tempRank[i] = 0.0; // Reset for next iteration
        }
        if (diff < CONVERGENCE_THRESHOLD) break;
    }
    free(tempRank);
}

// New function that wraps the PageRank calculation
void calculate_pagerank(Edge* edges, int edgeCount, int nodes, double* pagerank) {
    // Initialize PageRank
    for (int i = 0; i < nodes; i++) pagerank[i] = 1.0 / nodes;

    // Calculate PageRank
    powerMethod(pagerank, edges, edgeCount, nodes);
}
