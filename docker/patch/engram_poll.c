/* Engram decode-stage miss path: all tables in one pass, misses polled (DSV41_ENGRAM_NATIVE_POLL=1).
 *
 * engram_native.c's eng_gather_bf16 runs per table: an RWF_NOWAIT probe per
 * row, then posix_fadvise(WILLNEED) for every miss, then a blocking pread per
 * missed row in row order, sleeping whenever that row's page is not there yet.
 * On this kernel (6.17) the RWF_NOWAIT probe of a missing page already starts
 * its read (it returns EAGAIN with the page under I/O), so the WILLNEED pass
 * submits nothing new, and every sleep in the pread loop pays the CPU's idle
 * exit on wakeup (LPI-2/3: 231/433 us here). B0c census at c~4: ~11 us per
 * miss on spark1, ~37 on spark2, whose 7.4 ms stage sets the step.
 *
 * eng_gather_tables_bf16 takes every table of the step in one call:
 *   1. the same probe per row for every table, so all missing pages of the step
 *      are in flight together;
 *   2. no WILLNEED;
 *   3. each missed row re-probed (RWF_NOWAIT) until its bytes are there, with
 *      a cpu yield between tries, for up to spin_ns from the end of pass 1;
 *      after that, WILLNEED for every row still missing and a blocking pread
 *      per row (a kernel whose probe does not start the read, or a probe that
 *      could not, gets today's sequence);
 *   4. engram_native.c's dequant (included below), unchanged.
 * The bytes come from the same buffered reads as before, so the page cache
 * ends up holding the same pages.
 *
 * This file builds into its own .so, only when the knob is on: the default
 * decode path keeps loading engram_native.c alone.
 */
#include "engram_native.c"

#include <time.h>

#define ENG_POLL_ABI_VERSION 1
/* per table: w_fd w_off s_fd s_off dim sb vocab_start vocab_end hash_col */
#define ENG_TAB 9

int eng_poll_abi_version(void) { return ENG_POLL_ABI_VERSION; }

static int64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__)
    __asm__ __volatile__("pause" ::: "memory");
#endif
}

/* The deadline passed with rows still missing: WILLNEED all of them before the blocking
 * reads, so a kernel whose RWF_NOWAIT probe does not start the read gets today's batch. */
static void willneed_rest(const int64_t *tab, const int32_t *hash, int64_t tok_stride, int64_t rows_t,
                          int local_heads, int dim, int sb, const uint8_t *state, int64_t from, int64_t rows,
                          int64_t *calls) {
    for (int64_t g = from; g < rows; g++) {
        if (state[g] <= 1) continue;
        const int64_t *T = tab + (g / rows_t) * ENG_TAB;
        const int64_t r = g % rows_t;
        const int64_t id = (int64_t)hash[(r / local_heads) * tok_stride + T[8] + r % local_heads];
        if (state[g] & 2) advise((int)T[0], T[1] + id * dim, dim, POSIX_FADV_WILLNEED, calls);
        if (state[g] & 4) advise((int)T[2], T[3] + id * sb, sb, POSIX_FADV_WILLNEED, calls);
    }
}

/* Re-probe until the bytes are there (1) or the deadline passes (0); <0: -errno. */
static int poll_read(int fd, uint8_t *buf, size_t len, int64_t off, int64_t deadline, int64_t *calls) {
    for (;;) {
        int a = read_nowait(fd, buf, len, off, calls);
        if (a != 0) return a;
        if (now_ns() >= deadline) return 0;
        cpu_relax();
    }
}

/*
 * tab:   n_tables x ENG_TAB int64 (see above); every table has the same dim / sb.
 * hash:  [n_tok rows] x tok_stride int32; table t's ids start at column hash_col.
 * outs:  per table, [n_tok, local_heads, dim] bf16 bits (heads >= heads_valid and
 *        unowned ids come out zero, as eng_gather_bf16).
 * sw/ss: scratch, n_tables * n_tok * local_heads * dim and * sb bytes.
 * stats: ENG_NSTATS (rows, owned, miss, syscalls, nowait), overwritten.
 * Returns 0, or -errno: -EOPNOTSUPP etc. when RWF_NOWAIT is not available
 * (the caller then gathers per table with eng_gather_bf16, which handles that).
 */
int eng_gather_tables_bf16(int n_tables, const int64_t *tab, const int32_t *hash, int64_t tok_stride,
                           int n_tok, int heads_valid, int local_heads, const float *lut,
                           uint16_t *const *outs, uint8_t *sw, uint8_t *ss, int64_t spin_ns,
                           int64_t *stats) {
    int64_t st[ENG_NSTATS] = {0};
    if (n_tables <= 0 || n_tok < 0 || local_heads <= 0 || heads_valid < 0 || heads_valid > local_heads)
        return -EINVAL;
    const int dim = (int)tab[4], sb = (int)tab[5];
    if (dim <= 0 || sb <= 0 || dim % sb != 0) return -EINVAL;
    for (int t = 1; t < n_tables; t++)
        if (tab[t * ENG_TAB + 4] != dim || tab[t * ENG_TAB + 5] != sb) return -EINVAL;
    const int64_t rows_t = (int64_t)n_tok * local_heads, rows = rows_t * n_tables;
    const int qb = dim / sb;
    uint8_t *state = NULL; /* 0 unowned, 1 read done, bit 1: w pending, bit 2: s pending */
    int rc = 0;
    st[ENG_ST_ROWS] = rows;
    st[ENG_ST_NOWAIT] = 1;
    if (rows == 0) goto out;
    state = (uint8_t *)malloc((size_t)rows);
    if (!state) return -ENOMEM;

    /* 1. probe every row of every table: a missing page's read starts here */
    for (int64_t g = 0; g < rows; g++) {
        const int64_t *T = tab + (g / rows_t) * ENG_TAB;
        const int64_t r = g % rows_t;
        const int tok = (int)(r / local_heads), hh = (int)(r % local_heads);
        const int64_t id = hh < heads_valid ? (int64_t)hash[(int64_t)tok * tok_stride + T[8] + hh] : -1;
        if (id < T[6] || id >= T[7]) {
            state[g] = 0;
            continue;
        }
        st[ENG_ST_OWNED] += 1;
        int a = read_nowait((int)T[0], sw + g * dim, (size_t)dim, T[1] + id * dim, &st[ENG_ST_SYSCALLS]);
        int b = a < 0 ? a : read_nowait((int)T[2], ss + g * sb, (size_t)sb, T[3] + id * sb, &st[ENG_ST_SYSCALLS]);
        if (a < 0 || b < 0) {
            rc = a < 0 ? a : b;
            goto out;
        }
        state[g] = (uint8_t)(1 | (a == 0 ? 2 : 0) | (b == 0 ? 4 : 0));
        if (state[g] != 1) st[ENG_ST_MISS] += 1;
    }

    /* 3. wait for the misses without sleeping, then block for whatever is left */
    const int64_t deadline = now_ns() + spin_ns;
    int late = 0;
    for (int64_t g = 0; g < rows; g++) {
        if (state[g] <= 1) continue;
        const int64_t *T = tab + (g / rows_t) * ENG_TAB;
        const int64_t r = g % rows_t;
        const int tok = (int)(r / local_heads), hh = (int)(r % local_heads);
        const int64_t id = (int64_t)hash[(int64_t)tok * tok_stride + T[8] + hh];
        for (int k = 0; k < 2; k++) {
            if (!(state[g] & (k ? 4 : 2))) continue;
            const int fd = (int)T[k ? 2 : 0];
            const size_t len = (size_t)(k ? sb : dim);
            const int64_t off = T[k ? 3 : 1] + id * (int64_t)len;
            uint8_t *dst = k ? ss + g * sb : sw + g * dim;
            int a = poll_read(fd, dst, len, off, deadline, &st[ENG_ST_SYSCALLS]);
            if (a < 0) {
                rc = a;
                goto out;
            }
            if (a == 1) continue;
            if (!late) { /* past the deadline: WILLNEED every row still missing, then block */
                willneed_rest(tab, hash, tok_stride, rows_t, local_heads, dim, sb, state, g, rows, &st[ENG_ST_SYSCALLS]);
                late = 1;
            }
            if ((rc = read_full(fd, dst, len, off, &st[ENG_ST_SYSCALLS]))) goto out;
        }
        state[g] = 1;
    }

    /* 4. engram_native.c's dequant */
    for (int64_t g = 0; g < rows; g++) {
        uint16_t *o = outs[g / rows_t] + (g % rows_t) * dim;
        if (!state[g]) {
            memset(o, 0, (size_t)dim * sizeof *o);
            continue;
        }
        const uint8_t *w = sw + g * dim, *s = ss + g * sb;
        for (int b = 0; b < sb; b++) {
            const float scale = bits_f32((uint32_t)s[b] << 23);
            for (int j = 0; j < qb; j++) o[b * qb + j] = f32_to_bf16_rne(lut[w[b * qb + j]] * scale);
        }
    }
out:
    free(state);
    if (stats) memcpy(stats, st, sizeof st);
    return rc;
}
