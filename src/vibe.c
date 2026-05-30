#include "grand_pattern.h"
#include <math.h>
#include <string.h>

gp_vibe_t gp_vibe_new(void) {
    gp_vibe_t v;
    for (int i = 0; i < VIBE_DIMS; i++) v.dims[i] = 0.5;
    return v;
}

gp_vibe_t gp_vibe_blend(gp_vibe_t a, gp_vibe_t b, double ratio) {
    gp_vibe_t result;
    for (int i = 0; i < VIBE_DIMS; i++) {
        result.dims[i] = a.dims[i] * (1.0 - ratio) + b.dims[i] * ratio;
    }
    return result;
}

double gp_vibe_distance(gp_vibe_t a, gp_vibe_t b) {
    double sum = 0.0;
    for (int i = 0; i < VIBE_DIMS; i++) {
        double d = a.dims[i] - b.dims[i];
        sum += d * d;
    }
    return sqrt(sum);
}

gp_vibe_t gp_vibe_diffuse(gp_vibe_t v, gp_vibe_t *neighbors, double *weights, size_t count, double coeff) {
    gp_vibe_t result = v;
    if (count == 0) return result;
    for (size_t n = 0; n < count; n++) {
        for (int i = 0; i < VIBE_DIMS; i++) {
            result.dims[i] += coeff * weights[n] * (neighbors[n].dims[i] - v.dims[i]);
        }
    }
    return result;
}

double gp_vibe_energy(gp_vibe_t v) {
    double sum = 0.0;
    for (int i = 0; i < VIBE_DIMS; i++) {
        sum += v.dims[i] * v.dims[i];
    }
    return sqrt(sum);
}
