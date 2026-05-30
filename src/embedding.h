#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stddef.h>

#define EMBEDDING_DIM 8

typedef struct {
    double data[EMBEDDING_DIM];
} Embedding;

typedef struct {
    double timestamp;
    int sensor_id;
    Embedding embedding;
} Tick;

typedef struct {
    Embedding position;
    Embedding velocity;
    Embedding acceleration;
    double strength;
} Vibe;

typedef struct {
    Embedding value;
    double strength;
    double timestamp;
} TaggedEmbedding;

typedef struct {
    char id[64];
    TaggedEmbedding *perception_db;
    size_t perception_count;
    size_t perception_cap;
    TaggedEmbedding *prediction_db;
    size_t prediction_count;
    size_t prediction_cap;
    Vibe vibe;
    Embedding prev_position;
    Embedding prev_velocity;
} Room;

typedef struct {
    int merged;
    int decayed;
    int pruned;
} GCReport;

typedef struct {
    char from_id[64];
    char to_id[64];
    int algorithm; // 0 = standard, 1 = weighted
} Edge;

typedef struct {
    Room *rooms;
    size_t room_count;
    size_t room_cap;
    Edge *edges;
    size_t edge_count;
    size_t edge_cap;
} CellularGraph;

// Embedding operations
Embedding embedding_zero(void);
Embedding embedding_add(Embedding a, Embedding b);
Embedding embedding_sub(Embedding a, Embedding b);
Embedding embedding_scale(Embedding e, double s);
double embedding_dot(Embedding a, Embedding b);
double embedding_norm(Embedding e);
double embedding_cosine_similarity(Embedding a, Embedding b);
double embedding_euclidean_dist(Embedding a, Embedding b);
Embedding embedding_centroid(TaggedEmbedding *db, size_t count);

// Room operations
Room room_create(const char *id);
void room_free(Room *room);
double tick(Room *room, double timestamp, int sensor_id, Embedding perception);
Embedding predict(Room *room);
int balance_check(Room *room);
void compute_vibe(Room *room);

// GC
GCReport gc(Room *room, double merge_threshold, double decay_rate, double min_strength);

// Graph
CellularGraph graph_create(void);
void graph_free(CellularGraph *g);
void graph_add_room(CellularGraph *g, Room room);
void graph_add_edge(CellularGraph *g, const char *from, const char *to, int algorithm);
Room *graph_find_room(CellularGraph *g, const char *id);
void murmur(Room *from, Room *to, Embedding *out_vibe_position);
double cross_room_correlation(Room *a, Room *b);
void tick_through_graph(CellularGraph *g, double timestamp, int sensor_id, Embedding perception);

#endif
