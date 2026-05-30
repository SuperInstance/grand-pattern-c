#ifndef GRAND_PATTERN_H
#define GRAND_PATTERN_H

#include <stddef.h>
#include <stdbool.h>

#define VIBE_DIMS 16

typedef struct { double dims[VIBE_DIMS]; } gp_vibe_t;

typedef struct {
    double *data;
    size_t dim;
    unsigned long timestamp;
    double surprise;
} gp_entry_t;

typedef struct {
    gp_entry_t *entries;
    size_t count;
    size_t capacity;
} gp_vector_db_t;

typedef struct {
    gp_vector_db_t db_in;
    gp_vector_db_t db_out;
    size_t window;
} gp_jepa_t;

typedef struct {
    char source[64];
    gp_vibe_t vibe_snapshot;
    double surprise_avg;
    unsigned long tick;
    unsigned int ttl;
    int level; // 0=neighbor, 1=zone, 2=fleet
} gp_murmur_t;

typedef struct {
    char name[64];
    gp_vibe_t vibe;
    gp_jepa_t jepa;
    double last_surprise;
    unsigned long tick_count;
} gp_room_t;

typedef struct {
    gp_room_t *rooms;
    size_t room_count;
    struct { size_t from; size_t to; double weight; } *edges;
    size_t edge_count;
    double bpm;
    unsigned long tick_count;
} gp_graph_t;

// Vibe
gp_vibe_t gp_vibe_new(void);
gp_vibe_t gp_vibe_blend(gp_vibe_t a, gp_vibe_t b, double ratio);
double gp_vibe_distance(gp_vibe_t a, gp_vibe_t b);
gp_vibe_t gp_vibe_diffuse(gp_vibe_t v, gp_vibe_t *neighbors, double *weights, size_t count, double coeff);
double gp_vibe_energy(gp_vibe_t v);

// JEPA
void gp_jepa_init(gp_jepa_t *j, size_t window);
void gp_jepa_perceive(gp_jepa_t *j, double *data, size_t dim, unsigned long ts);
double *gp_jepa_predict(gp_jepa_t *j, size_t dim);
double gp_surprise(double *a, double *b, size_t dim);
bool gp_jepa_check_conservation(gp_jepa_t *j, size_t tolerance);
void gp_jepa_gc(gp_jepa_t *j, double threshold);

// Murmur
gp_murmur_t gp_murmur_create(const char *source, gp_vibe_t vibe, double surprise, unsigned long tick);
void gp_murmur_decay(gp_murmur_t *m);
bool gp_murmur_is_expired(gp_murmur_t *m);

// Graph
gp_graph_t *gp_graph_create(double bpm);
void gp_graph_destroy(gp_graph_t *g);
size_t gp_graph_add_room(gp_graph_t *g, const char *name);
void gp_graph_add_edge(gp_graph_t *g, size_t from, size_t to);
void gp_graph_tick(gp_graph_t *g);
void gp_graph_gossip(gp_graph_t *g);
gp_vibe_t gp_graph_fleet_vibe(gp_graph_t *g);
double gp_graph_fleet_surprise(gp_graph_t *g);

#endif
