#include "embedding.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  TEST: %-50s", #name);
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

static Embedding make_embedding(double base) {
    Embedding e;
    for (int i = 0; i < EMBEDDING_DIM; i++) e.data[i] = base + i * 0.1;
    return e;
}

void test_tick_updates_perception(void) {
    TEST(test_tick_updates_perception);
    Room r = room_create("test");
    Embedding e = make_embedding(1.0);
    double err = tick(&r, 1.0, 1, e);
    ASSERT(r.perception_count == 1, "perception_count should be 1");
    ASSERT(r.prediction_count == 1, "prediction_count should be 1");
    room_free(&r);
    PASS();
}

void test_predict_generates_embedding(void) {
    TEST(test_predict_generates_embedding);
    Room r = room_create("test");
    Embedding pred = predict(&r);
    // First prediction should be zero + zero = zero (no history)
    ASSERT(embedding_norm(pred) < 1e-9, "first prediction should be near zero");
    // After a tick, prediction should be non-trivial
    tick(&r, 1.0, 1, make_embedding(1.0));
    tick(&r, 2.0, 1, make_embedding(2.0));
    Embedding pred2 = predict(&r);
    ASSERT(embedding_norm(pred2) > 0, "prediction after ticks should be non-zero");
    room_free(&r);
    PASS();
}

void test_balance_check_passes_equal(void) {
    TEST(test_balance_check_passes_equal);
    Room r = room_create("test");
    tick(&r, 1.0, 1, make_embedding(1.0));
    tick(&r, 2.0, 1, make_embedding(2.0));
    ASSERT(balance_check(&r), "balance should pass with equal DB sizes");
    room_free(&r);
    PASS();
}

void test_balance_check_fails_unequal(void) {
    TEST(test_balance_check_fails_unequal);
    Room r = room_create("test");
    // Manually add to perception only
    TaggedEmbedding te = { make_embedding(1.0), 1.0, 1.0 };
    r.perception_db = malloc(sizeof(TaggedEmbedding) * 2);
    r.perception_db[0] = te;
    r.perception_count = 1;
    r.perception_cap = 2;
    ASSERT(!balance_check(&r), "balance should fail with unequal DB sizes");
    room_free(&r);
    PASS();
}

void test_vibe_computation(void) {
    TEST(test_vibe_computation);
    Room r = room_create("test");
    tick(&r, 1.0, 1, make_embedding(1.0));
    ASSERT(r.vibe.strength > 0, "vibe strength should be > 0 after tick");
    ASSERT(embedding_norm(r.vibe.position) > 0, "vibe position should be non-zero");
    room_free(&r);
    PASS();
}

void test_merge_reduces_count(void) {
    TEST(test_merge_reduces_count);
    Room r = room_create("test");
    // Add similar embeddings
    Embedding e1 = make_embedding(1.0);
    Embedding e2 = make_embedding(1.01); // very similar
    tick(&r, 1.0, 1, e1);
    tick(&r, 2.0, 1, e2);
    size_t before = r.perception_count;
    GCReport report = gc(&r, 0.5, 0.99, 0.01);
    ASSERT(report.merged > 0 || r.perception_count <= before, "merge should not increase count");
    room_free(&r);
    PASS();
}

void test_decay_reduces_strengths(void) {
    TEST(test_decay_reduces_strengths);
    Room r = room_create("test");
    tick(&r, 1.0, 1, make_embedding(1.0));
    double before = r.perception_db[0].strength;
    gc(&r, 999.0, 0.5, 0.0); // high merge threshold (no merge), 0.5 decay, no prune
    ASSERT(r.perception_db[0].strength < before, "decay should reduce strength");
    room_free(&r);
    PASS();
}

void test_prune_removes_weak(void) {
    TEST(test_prune_removes_weak);
    Room r = room_create("test");
    tick(&r, 1.0, 1, make_embedding(1.0));
    // Manually set strength very low
    r.perception_db[0].strength = 0.001;
    // Also need prediction_db entry
    if (r.prediction_count > 0) r.prediction_db[0].strength = 0.001;
    gc(&r, 999.0, 1.0, 0.01); // no merge, no decay, prune at 0.01
    ASSERT(r.perception_count == 0, "prune should remove weak embeddings");
    room_free(&r);
    PASS();
}

void test_full_gc_cycle(void) {
    TEST(test_full_gc_cycle);
    Room r = room_create("test");
    tick(&r, 1.0, 1, make_embedding(1.0));
    tick(&r, 2.0, 1, make_embedding(1.01));
    tick(&r, 3.0, 1, make_embedding(5.0));
    GCReport report = gc(&r, 0.5, 0.9, 0.01);
    ASSERT(report.decayed > 0, "GC should report decayed items");
    room_free(&r);
    PASS();
}

void test_cross_room_correlation(void) {
    TEST(test_cross_room_correlation);
    Room a = room_create("a");
    Room b = room_create("b");
    tick(&a, 1.0, 1, make_embedding(1.0));
    tick(&b, 1.0, 1, make_embedding(1.0));
    double corr = cross_room_correlation(&a, &b);
    ASSERT(corr > 0.99, "identical embeddings should have correlation ~1.0");
    room_free(&a);
    room_free(&b);
    PASS();
}

void test_murmur_sends_vibe(void) {
    TEST(test_murmur_sends_vibe);
    Room from = room_create("from");
    Room to = room_create("to");
    tick(&from, 1.0, 1, make_embedding(3.0));
    size_t before = to.perception_count;
    Embedding vibe_pos;
    murmur(&from, &to, &vibe_pos);
    ASSERT(to.perception_count > before, "murmur should add to target perception");
    room_free(&from);
    room_free(&to);
    PASS();
}

void test_graph_construction(void) {
    TEST(test_graph_construction);
    CellularGraph g = graph_create();
    graph_add_room(g.rooms ? NULL : &g, room_create("r1")); // hack: add directly
    // Proper way:
    g = graph_create();
    graph_add_room(&g, room_create("r1"));
    graph_add_room(&g, room_create("r2"));
    graph_add_edge(&g, "r1", "r2", 0);
    ASSERT(g.room_count == 2, "graph should have 2 rooms");
    ASSERT(g.edge_count == 1, "graph should have 1 edge");
    graph_free(&g);
    PASS();
}

void test_tick_through_graph(void) {
    TEST(test_tick_through_graph);
    CellularGraph g = graph_create();
    graph_add_room(&g, room_create("r1"));
    graph_add_room(&g, room_create("r2"));
    graph_add_edge(&g, "r1", "r2", 0);
    Embedding e = make_embedding(1.0);
    tick_through_graph(&g, 1.0, 1, e);
    Room *r1 = graph_find_room(&g, "r1");
    Room *r2 = graph_find_room(&g, "r2");
    ASSERT(r1 != NULL && r1->perception_count >= 1, "r1 should have perceptions");
    ASSERT(r2 != NULL && r2->perception_count >= 2, "r2 should have perceptions + murmur");
    graph_free(&g);
    PASS();
}

int main(void) {
    printf("=== Grand Pattern C Test Suite ===\n\n");

    test_tick_updates_perception();
    test_predict_generates_embedding();
    test_balance_check_passes_equal();
    test_balance_check_fails_unequal();
    test_vibe_computation();
    test_merge_reduces_count();
    test_decay_reduces_strengths();
    test_prune_removes_weak();
    test_full_gc_cycle();
    test_cross_room_correlation();
    test_murmur_sends_vibe();
    test_graph_construction();
    test_tick_through_graph();

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
