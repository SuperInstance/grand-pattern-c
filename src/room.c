#include "grand_pattern.h"
#include <stdlib.h>
#include <string.h>

static void room_jepa_init(gp_room_t *r, size_t window) {
    gp_jepa_init(&r->jepa, window);
}

static gp_room_t gp_room_create(const char *name) {
    gp_room_t r;
    memset(r.name, 0, sizeof(r.name));
    strncpy(r.name, name, sizeof(r.name) - 1);
    r.vibe = gp_vibe_new();
    r.last_surprise = 0.0;
    r.tick_count = 0;
    room_jepa_init(&r, 4);
    return r;
}
