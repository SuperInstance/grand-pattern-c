#include "grand_pattern.h"
#include <string.h>

gp_murmur_t gp_murmur_create(const char *source, gp_vibe_t vibe, double surprise, unsigned long tick) {
    gp_murmur_t m;
    memset(m.source, 0, sizeof(m.source));
    strncpy(m.source, source, sizeof(m.source) - 1);
    m.vibe_snapshot = vibe;
    m.surprise_avg = surprise;
    m.tick = tick;
    m.ttl = 10;
    m.level = 0;
    return m;
}

void gp_murmur_decay(gp_murmur_t *m) {
    if (m->ttl > 0) m->ttl--;
}

bool gp_murmur_is_expired(gp_murmur_t *m) {
    return m->ttl == 0;
}
