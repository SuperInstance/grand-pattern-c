#include "grand_pattern.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

static void db_init(gp_vector_db_t *db) {
    db->entries = NULL;
    db->count = 0;
    db->capacity = 0;
}

static void db_push(gp_vector_db_t *db, double *data, size_t dim, unsigned long ts, double surprise) {
    if (db->count >= db->capacity) {
        size_t new_cap = db->capacity == 0 ? 8 : db->capacity * 2;
        db->entries = realloc(db->entries, new_cap * sizeof(gp_entry_t));
        db->capacity = new_cap;
    }
    gp_entry_t *e = &db->entries[db->count++];
    e->data = malloc(dim * sizeof(double));
    memcpy(e->data, data, dim * sizeof(double));
    e->dim = dim;
    e->timestamp = ts;
    e->surprise = surprise;
}

static void db_free(gp_vector_db_t *db) {
    for (size_t i = 0; i < db->count; i++) free(db->entries[i].data);
    free(db->entries);
    db->entries = NULL;
    db->count = 0;
    db->capacity = 0;
}

void gp_jepa_init(gp_jepa_t *j, size_t window) {
    db_init(&j->db_in);
    db_init(&j->db_out);
    j->window = window;
}

void gp_jepa_perceive(gp_jepa_t *j, double *data, size_t dim, unsigned long ts) {
    double surprise = 0.0;
    if (j->db_in.count > 0) {
        double *pred = gp_jepa_predict(j, dim);
        surprise = gp_surprise(data, pred, dim);
        free(pred);
    }
    db_push(&j->db_in, data, dim, ts, surprise);

    // Simple prediction: store same data as output (identity predictor)
    db_push(&j->db_out, data, dim, ts, surprise);
}

double *gp_jepa_predict(gp_jepa_t *j, size_t dim) {
    double *pred = calloc(dim, sizeof(double));
    if (j->db_in.count == 0) return pred;

    // Use weighted average of recent entries within window
    size_t start = 0;
    if (j->db_in.count > j->window) start = j->db_in.count - j->window;
    size_t n = j->db_in.count - start;
    for (size_t i = start; i < j->db_in.count; i++) {
        size_t d = j->db_in.entries[i].dim;
        size_t use = d < dim ? d : dim;
        for (size_t k = 0; k < use; k++) {
            pred[k] += j->db_in.entries[i].data[k];
        }
    }
    for (size_t k = 0; k < dim; k++) pred[k] /= (double)n;
    return pred;
}

double gp_surprise(double *a, double *b, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum / (double)dim);
}

bool gp_jepa_check_conservation(gp_jepa_t *j, size_t tolerance) {
    // Check that input and output DB sizes don't diverge beyond tolerance
    long diff = (long)j->db_in.count - (long)j->db_out.count;
    if (diff < 0) diff = -diff;
    return (size_t)diff <= tolerance;
}

void gp_jepa_gc(gp_jepa_t *j, double threshold) {
    // Remove low-surprise entries from both DBs to reclaim memory
    for (int db_idx = 0; db_idx < 2; db_idx++) {
        gp_vector_db_t *db = db_idx == 0 ? &j->db_in : &j->db_out;
        size_t write = 0;
        for (size_t i = 0; i < db->count; i++) {
            if (db->entries[i].surprise >= threshold || i >= db->count - j->window) {
                if (write != i) db->entries[write] = db->entries[i];
                write++;
            } else {
                free(db->entries[i].data);
            }
        }
        db->count = write;
    }
}
