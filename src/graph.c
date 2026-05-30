#include "grand_pattern.h"
#include <stdlib.h>
#include <string.h>

gp_graph_t *gp_graph_create(double bpm) {
    gp_graph_t *g = calloc(1, sizeof(gp_graph_t));
    g->bpm = bpm;
    g->tick_count = 0;
    return g;
}

void gp_graph_destroy(gp_graph_t *g) {
    if (!g) return;
    for (size_t i = 0; i < g->room_count; i++) {
        // Free JEPA internals
        for (size_t j = 0; j < g->rooms[i].jepa.db_in.count; j++)
            free(g->rooms[i].jepa.db_in.entries[j].data);
        free(g->rooms[i].jepa.db_in.entries);
        for (size_t j = 0; j < g->rooms[i].jepa.db_out.count; j++)
            free(g->rooms[i].jepa.db_out.entries[j].data);
        free(g->rooms[i].jepa.db_out.entries);
    }
    free(g->rooms);
    free(g->edges);
    free(g);
}

size_t gp_graph_add_room(gp_graph_t *g, const char *name) {
    size_t new_cap = g->room_count + 1;
    g->rooms = realloc(g->rooms, new_cap * sizeof(gp_room_t));
    gp_room_t *r = &g->rooms[g->room_count];
    memset(r->name, 0, sizeof(r->name));
    strncpy(r->name, name, sizeof(r->name) - 1);
    r->vibe = gp_vibe_new();
    r->last_surprise = 0.0;
    r->tick_count = 0;
    gp_jepa_init(&r->jepa, 4);
    g->room_count = new_cap;
    return g->room_count - 1;
}

void gp_graph_add_edge(gp_graph_t *g, size_t from, size_t to) {
    if (from >= g->room_count || to >= g->room_count) return;
    size_t new_cap = g->edge_count + 1;
    g->edges = realloc(g->edges, new_cap * sizeof(g->edges[0]));
    g->edges[g->edge_count].from = from;
    g->edges[g->edge_count].to = to;
    g->edges[g->edge_count].weight = 1.0;
    g->edge_count = new_cap;
}

void gp_graph_tick(gp_graph_t *g) {
    g->tick_count++;
    for (size_t i = 0; i < g->room_count; i++) {
        gp_room_t *r = &g->rooms[i];
        r->tick_count++;

        // Perceive current vibe into JEPA
        gp_jepa_perceive(&r->jepa, r->vibe.dims, VIBE_DIMS, r->tick_count);

        // Calculate surprise
        double *pred = gp_jepa_predict(&r->jepa, VIBE_DIMS);
        r->last_surprise = gp_surprise(r->vibe.dims, pred, VIBE_DIMS);
        free(pred);
    }

    // Diffuse vibes along edges
    gp_vibe_t *new_vibes = calloc(g->room_count, sizeof(gp_vibe_t));
    for (size_t i = 0; i < g->room_count; i++) new_vibes[i] = g->rooms[i].vibe;

    for (size_t e = 0; e < g->edge_count; e++) {
        size_t fi = g->edges[e].from;
        size_t ti = g->edges[e].to;
        double w = g->edges[e].weight;
        for (int d = 0; d < VIBE_DIMS; d++) {
            double diff = g->rooms[fi].vibe.dims[d] - g->rooms[ti].vibe.dims[d];
            new_vibes[ti].dims[d] += 0.1 * w * diff;
            new_vibes[fi].dims[d] -= 0.1 * w * diff;
        }
    }

    for (size_t i = 0; i < g->room_count; i++) g->rooms[i].vibe = new_vibes[i];
    free(new_vibes);
}

void gp_graph_gossip(gp_graph_t *g) {
    // Exchange murmurs along edges — simplified: propagate surprise info
    for (size_t e = 0; e < g->edge_count; e++) {
        size_t fi = g->edges[e].from;
        size_t ti = g->edges[e].to;
        // Blend vibes slightly between connected rooms
        gp_vibe_t blended = gp_vibe_blend(g->rooms[fi].vibe, g->rooms[ti].vibe, 0.05);
        g->rooms[ti].vibe = blended;
    }
}

gp_vibe_t gp_graph_fleet_vibe(gp_graph_t *g) {
    gp_vibe_t fleet = gp_vibe_new();
    if (g->room_count == 0) return fleet;
    for (int i = 0; i < VIBE_DIMS; i++) {
        double sum = 0.0;
        for (size_t r = 0; r < g->room_count; r++) {
            sum += g->rooms[r].vibe.dims[i];
        }
        fleet.dims[i] = sum / (double)g->room_count;
    }
    return fleet;
}

double gp_graph_fleet_surprise(gp_graph_t *g) {
    if (g->room_count == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < g->room_count; i++) {
        sum += g->rooms[i].last_surprise;
    }
    return sum / (double)g->room_count;
}
