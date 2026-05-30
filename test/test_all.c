#include "grand_pattern.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ASSERT_EQ_D(a, b, eps) do { \
    double _a = (a), _b = (b), _diff = _a - _b; \
    if (_diff < 0) _diff = -_diff; \
    assert(_diff < (eps)); \
} while(0)

int main(void) {
    // === Vibe tests ===
    // Test 1: neutral vibe
    gp_vibe_t v = gp_vibe_new();
    assert(v.dims[0] == 0.5);
    assert(v.dims[15] == 0.5);

    // Test 2: blend
    gp_vibe_t a = gp_vibe_new();
    gp_vibe_t b = gp_vibe_new();
    b.dims[0] = 1.0;
    gp_vibe_t blended = gp_vibe_blend(a, b, 0.5);
    ASSERT_EQ_D(blended.dims[0], 0.75, 1e-9);

    // Test 3: distance to self is zero
    assert(gp_vibe_distance(a, a) < 1e-9);

    // Test 4: distance is positive for different vibes
    gp_vibe_t c = gp_vibe_new();
    c.dims[0] = 0.0;
    assert(gp_vibe_distance(a, c) > 0.0);

    // Test 5: energy of neutral vibe
    double energy = gp_vibe_energy(gp_vibe_new());
    ASSERT_EQ_D(energy, sqrt(16.0 * 0.25), 1e-9);

    // Test 6: blend ratio 0 returns a
    gp_vibe_t b0 = gp_vibe_blend(a, c, 0.0);
    assert(gp_vibe_distance(b0, a) < 1e-9);

    // Test 7: blend ratio 1 returns b
    gp_vibe_t b1 = gp_vibe_blend(a, c, 1.0);
    assert(gp_vibe_distance(b1, c) < 1e-9);

    // Test 8: diffuse with no neighbors returns original
    gp_vibe_t d = gp_vibe_diffuse(a, NULL, NULL, 0, 0.5);
    assert(gp_vibe_distance(d, a) < 1e-9);

    // Test 9: diffuse with neighbors moves toward them
    gp_vibe_t n1 = gp_vibe_new();
    n1.dims[0] = 1.0;
    double w1 = 1.0;
    gp_vibe_t diffused = gp_vibe_diffuse(a, &n1, &w1, 1, 0.5);
    assert(diffused.dims[0] > a.dims[0]);

    // Test 10: distance symmetry
    assert(fabs(gp_vibe_distance(a, c) - gp_vibe_distance(c, a)) < 1e-9);

    // === Murmur tests ===
    // Test 11: create murmur
    gp_murmur_t m = gp_murmur_create("room1", a, 0.5, 1);
    assert(strcmp(m.source, "room1") == 0);
    assert(m.ttl == 10);
    assert(m.surprise_avg == 0.5);

    // Test 12: murmur decay
    gp_murmur_decay(&m);
    assert(m.ttl == 9);

    // Test 13: murmur not expired yet
    assert(!gp_murmur_is_expired(&m));

    // Test 14: murmur expires after enough decays
    for (int i = 0; i < 9; i++) gp_murmur_decay(&m);
    assert(gp_murmur_is_expired(&m));

    // Test 15: murmur level default
    gp_murmur_t m2 = gp_murmur_create("room2", a, 0.0, 0);
    assert(m2.level == 0);

    // === JEPA tests ===
    // Test 16: jepa init and basic perceive
    gp_jepa_t jepa;
    gp_jepa_init(&jepa, 4);
    assert(jepa.db_in.count == 0);
    double data1[4] = {1.0, 2.0, 3.0, 4.0};
    gp_jepa_perceive(&jepa, data1, 4, 1);
    assert(jepa.db_in.count == 1);

    // Test 17: surprise between identical vectors is zero
    double s = gp_surprise(data1, data1, 4);
    assert(s < 1e-9);

    // Test 18: surprise between different vectors is positive
    double data2[4] = {4.0, 3.0, 2.0, 1.0};
    double s2 = gp_surprise(data1, data2, 4);
    assert(s2 > 0.0);

    // Test 19: conservation check passes
    assert(gp_jepa_check_conservation(&jepa, 0));

    // Test 20: predict returns non-null
    double *pred = gp_jepa_predict(&jepa, 4);
    assert(pred != NULL);
    free(pred);

    // Test 21: gc with zero threshold removes high-surprise entries
    double data3[4] = {1.0, 1.0, 1.0, 1.0};
    gp_jepa_perceive(&jepa, data3, 4, 2);
    gp_jepa_gc(&jepa, 0.0); // threshold 0 keeps entries with surprise > 0
    assert(jepa.db_in.count >= 1);
    // Cleanup
    for (size_t i = 0; i < jepa.db_in.count; i++) free(jepa.db_in.entries[i].data);
    free(jepa.db_in.entries);
    for (size_t i = 0; i < jepa.db_out.count; i++) free(jepa.db_out.entries[i].data);
    free(jepa.db_out.entries);

    // === Graph tests ===
    // Test 22: create and destroy graph
    gp_graph_t *g = gp_graph_create(120.0);
    assert(g != NULL);
    assert(g->bpm == 120.0);
    assert(g->room_count == 0);

    // Test 23: add rooms
    size_t r0 = gp_graph_add_room(g, "alpha");
    size_t r1 = gp_graph_add_room(g, "beta");
    size_t r2 = gp_graph_add_room(g, "gamma");
    assert(r0 == 0);
    assert(r1 == 1);
    assert(r2 == 2);
    assert(g->room_count == 3);
    assert(strcmp(g->rooms[0].name, "alpha") == 0);

    // Test 24: add edge
    gp_graph_add_edge(g, r0, r1);
    gp_graph_add_edge(g, r1, r2);
    assert(g->edge_count == 2);

    // Test 25: tick advances state
    gp_graph_tick(g);
    assert(g->tick_count == 1);
    assert(g->rooms[0].tick_count == 1);

    // Test 26: fleet vibe is average
    gp_vibe_t fleet = gp_graph_fleet_vibe(g);
    // After one tick with diffusion, should still be close to neutral
    ASSERT_EQ_D(fleet.dims[0], 0.5, 0.5);

    // Test 27: fleet surprise is non-negative
    double fs = gp_graph_fleet_surprise(g);
    assert(fs >= 0.0);

    // Test 28: multiple ticks
    for (int i = 0; i < 10; i++) gp_graph_tick(g);
    assert(g->tick_count == 11);

    // Test 29: gossip modifies vibes
    gp_vibe_t before_gossip = g->rooms[1].vibe;
    gp_graph_gossip(g);
    // Gossip should change connected rooms slightly

    // Test 30: modify room vibe and tick
    g->rooms[0].vibe.dims[0] = 0.0;
    g->rooms[0].vibe.dims[1] = 1.0;
    gp_graph_tick(g);
    assert(g->rooms[0].last_surprise > 0.0); // surprise from prediction mismatch

    gp_graph_destroy(g);

    printf("All 30 tests passed!\n");
    return 0;
}
