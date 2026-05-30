#include "embedding.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Embedding embedding_zero(void) {
    Embedding e;
    memset(e.data, 0, sizeof(e.data));
    return e;
}

Embedding embedding_add(Embedding a, Embedding b) {
    Embedding r;
    for (int i = 0; i < EMBEDDING_DIM; i++) r.data[i] = a.data[i] + b.data[i];
    return r;
}

Embedding embedding_sub(Embedding a, Embedding b) {
    Embedding r;
    for (int i = 0; i < EMBEDDING_DIM; i++) r.data[i] = a.data[i] - b.data[i];
    return r;
}

Embedding embedding_scale(Embedding e, double s) {
    Embedding r;
    for (int i = 0; i < EMBEDDING_DIM; i++) r.data[i] = e.data[i] * s;
    return r;
}

double embedding_dot(Embedding a, Embedding b) {
    double s = 0;
    for (int i = 0; i < EMBEDDING_DIM; i++) s += a.data[i] * b.data[i];
    return s;
}

double embedding_norm(Embedding e) {
    return sqrt(embedding_dot(e, e));
}

double embedding_cosine_similarity(Embedding a, Embedding b) {
    double d = embedding_norm(a) * embedding_norm(b);
    if (d < 1e-12) return 0.0;
    return embedding_dot(a, b) / d;
}

double embedding_euclidean_dist(Embedding a, Embedding b) {
    double s = 0;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        double d = a.data[i] - b.data[i];
        s += d * d;
    }
    return sqrt(s);
}

Embedding embedding_centroid(TaggedEmbedding *db, size_t count) {
    Embedding c = embedding_zero();
    if (count == 0) return c;
    for (size_t i = 0; i < count; i++) {
        c = embedding_add(c, db[i].value);
    }
    return embedding_scale(c, 1.0 / count);
}

Room room_create(const char *id) {
    Room r;
    strncpy(r.id, id, sizeof(r.id) - 1);
    r.id[sizeof(r.id) - 1] = '\0';
    r.perception_db = NULL;
    r.perception_count = 0;
    r.perception_cap = 0;
    r.prediction_db = NULL;
    r.prediction_count = 0;
    r.prediction_cap = 0;
    r.vibe.position = embedding_zero();
    r.vibe.velocity = embedding_zero();
    r.vibe.acceleration = embedding_zero();
    r.vibe.strength = 0.0;
    r.prev_position = embedding_zero();
    r.prev_velocity = embedding_zero();
    return r;
}

void room_free(Room *room) {
    free(room->perception_db);
    free(room->prediction_db);
    room->perception_db = NULL;
    room->prediction_db = NULL;
    room->perception_count = 0;
    room->prediction_count = 0;
}

static void db_push(TaggedEmbedding **db, size_t *count, size_t *cap, TaggedEmbedding te) {
    if (*count >= *cap) {
        *cap = (*cap == 0) ? 16 : (*cap * 2);
        *db = realloc(*db, *cap * sizeof(TaggedEmbedding));
    }
    (*db)[(*count)++] = te;
}

double tick(Room *room, double timestamp, int sensor_id, Embedding perception) {
    // Store perception
    TaggedEmbedding pe;
    pe.value = perception;
    pe.strength = 1.0;
    pe.timestamp = timestamp;
    db_push(&room->perception_db, &room->perception_count, &room->perception_cap, pe);

    // Generate prediction and store
    Embedding pred = predict(room);
    TaggedEmbedding pr;
    pr.value = pred;
    pr.strength = 1.0;
    pr.timestamp = timestamp;
    db_push(&room->prediction_db, &room->prediction_count, &room->prediction_cap, pr);

    // Compute prediction error (surprise)
    double error = embedding_euclidean_dist(perception, pred);

    // Update vibe
    compute_vibe(room);

    return error;
}

Embedding predict(Room *room) {
    // Simple prediction: position + velocity (momentum-based)
    Embedding pred = embedding_add(room->vibe.position, room->vibe.velocity);
    return pred;
}

int balance_check(Room *room) {
    return room->perception_count == room->prediction_count;
}

void compute_vibe(Room *room) {
    Embedding old_vel = room->vibe.velocity;

    // Position = centroid of perception_db
    room->vibe.position = embedding_centroid(room->perception_db, room->perception_count);

    // Velocity = position - prev_position
    room->vibe.velocity = embedding_sub(room->vibe.position, room->prev_position);

    // Acceleration = velocity - prev_velocity
    room->vibe.acceleration = embedding_sub(room->vibe.velocity, room->prev_velocity);

    // Strength = number of perceptions
    room->vibe.strength = (double)room->perception_count;

    room->prev_position = room->vibe.position;
    room->prev_velocity = old_vel;
}

static int merge_similar(Room *room, double threshold) {
    int merged = 0;
    for (size_t i = 0; i < room->perception_count; i++) {
        for (size_t j = i + 1; j < room->perception_count; j++) {
            if (room->perception_db[i].strength > 0 && room->perception_db[j].strength > 0) {
                double dist = embedding_euclidean_dist(room->perception_db[i].value, room->perception_db[j].value);
                if (dist < threshold) {
                    // Merge j into i (average)
                    room->perception_db[i].value = embedding_scale(
                        embedding_add(room->perception_db[i].value, room->perception_db[j].value), 0.5);
                    room->perception_db[i].strength += room->perception_db[j].strength;
                    room->perception_db[j].strength = 0; // mark for removal
                    merged++;
                }
            }
        }
    }
    // Also merge prediction_db
    for (size_t i = 0; i < room->prediction_count; i++) {
        for (size_t j = i + 1; j < room->prediction_count; j++) {
            if (room->prediction_db[i].strength > 0 && room->prediction_db[j].strength > 0) {
                double dist = embedding_euclidean_dist(room->prediction_db[i].value, room->prediction_db[j].value);
                if (dist < threshold) {
                    room->prediction_db[i].value = embedding_scale(
                        embedding_add(room->prediction_db[i].value, room->prediction_db[j].value), 0.5);
                    room->prediction_db[i].strength += room->prediction_db[j].strength;
                    room->prediction_db[j].strength = 0;
                    merged++;
                }
            }
        }
    }
    return merged;
}

static int decay(Room *room, double rate) {
    int decayed = 0;
    for (size_t i = 0; i < room->perception_count; i++) {
        room->perception_db[i].strength *= rate;
        decayed++;
    }
    for (size_t i = 0; i < room->prediction_count; i++) {
        room->prediction_db[i].strength *= rate;
        decayed++;
    }
    return decayed;
}

static int prune(Room *room, double min_strength) {
    int pruned = 0;
    // Prune perception_db
    size_t write = 0;
    for (size_t i = 0; i < room->perception_count; i++) {
        if (room->perception_db[i].strength >= min_strength) {
            room->perception_db[write++] = room->perception_db[i];
        } else {
            pruned++;
        }
    }
    room->perception_count = write;

    // Prune prediction_db
    write = 0;
    for (size_t i = 0; i < room->prediction_count; i++) {
        if (room->prediction_db[i].strength >= min_strength) {
            room->prediction_db[write++] = room->prediction_db[i];
        } else {
            pruned++;
        }
    }
    room->prediction_count = write;
    return pruned;
}

GCReport gc(Room *room, double merge_threshold, double decay_rate, double min_strength) {
    GCReport report;
    report.merged = merge_similar(room, merge_threshold);
    report.decayed = decay(room, decay_rate);
    report.pruned = prune(room, min_strength);
    return report;
}

CellularGraph graph_create(void) {
    CellularGraph g;
    g.rooms = NULL;
    g.room_count = 0;
    g.room_cap = 0;
    g.edges = NULL;
    g.edge_count = 0;
    g.edge_cap = 0;
    return g;
}

void graph_free(CellularGraph *g) {
    for (size_t i = 0; i < g->room_count; i++) {
        room_free(&g->rooms[i]);
    }
    free(g->rooms);
    free(g->edges);
    g->rooms = NULL;
    g->edges = NULL;
    g->room_count = 0;
    g->edge_count = 0;
}

void graph_add_room(CellularGraph *g, Room room) {
    if (g->room_count >= g->room_cap) {
        g->room_cap = (g->room_cap == 0) ? 16 : (g->room_cap * 2);
        g->rooms = realloc(g->rooms, g->room_cap * sizeof(Room));
    }
    g->rooms[g->room_count++] = room;
}

void graph_add_edge(CellularGraph *g, const char *from, const char *to, int algorithm) {
    if (g->edge_count >= g->edge_cap) {
        g->edge_cap = (g->edge_cap == 0) ? 16 : (g->edge_cap * 2);
        g->edges = realloc(g->edges, g->edge_cap * sizeof(Edge));
    }
    Edge e;
    strncpy(e.from_id, from, sizeof(e.from_id) - 1);
    e.from_id[sizeof(e.from_id) - 1] = '\0';
    strncpy(e.to_id, to, sizeof(e.to_id) - 1);
    e.to_id[sizeof(e.to_id) - 1] = '\0';
    e.algorithm = algorithm;
    g->edges[g->edge_count++] = e;
}

Room *graph_find_room(CellularGraph *g, const char *id) {
    for (size_t i = 0; i < g->room_count; i++) {
        if (strcmp(g->rooms[i].id, id) == 0) return &g->rooms[i];
    }
    return NULL;
}

void murmur(Room *from, Room *to, Embedding *out_vibe_position) {
    *out_vibe_position = from->vibe.position;
    // Incorporate into target's perception as a gossip signal
    TaggedEmbedding te;
    te.value = from->vibe.position;
    te.strength = 0.5; // gossip has lower strength
    te.timestamp = 0.0;
    if (to->perception_count >= to->perception_cap) {
        to->perception_cap = (to->perception_cap == 0) ? 16 : (to->perception_cap * 2);
        to->perception_db = realloc(to->perception_db, to->perception_cap * sizeof(TaggedEmbedding));
    }
    to->perception_db[to->perception_count++] = te;
}

double cross_room_correlation(Room *a, Room *b) {
    return embedding_cosine_similarity(a->vibe.position, b->vibe.position);
}

void tick_through_graph(CellularGraph *g, double timestamp, int sensor_id, Embedding perception) {
    // Tick all rooms
    for (size_t i = 0; i < g->room_count; i++) {
        tick(&g->rooms[i], timestamp, sensor_id, perception);
    }
    // Propagate via edges (murmur)
    for (size_t i = 0; i < g->edge_count; i++) {
        Room *from = graph_find_room(g, g->edges[i].from_id);
        Room *to = graph_find_room(g, g->edges[i].to_id);
        if (from && to) {
            Embedding vibe_pos;
            murmur(from, to, &vibe_pos);
        }
    }
}
