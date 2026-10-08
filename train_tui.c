/*
 * train-tui: a lightweight, dependency-free terminal UI for monitoring
 * (almost) any deep-learning training run. Pure C, no ncurses, no Python.
 *
 * Profiles describe how to read a run:
 *   - which substrings mark epoch, iter, losses and validation metrics
 *   - where total_iter comes from (config file key, a log line, or -t)
 *   - how checkpoints are named (filename prefix + suffix + iter extraction)
 *
 * Built-in profiles:
 *   basicsr   BasicSR / Real-ESRGAN / HAT / SwinIR text logs
 *   lightning PyTorch Lightning progress-bar logs
 *   hf        Hugging Face Trainer dict-style logs
 *   jsonl     one JSON object per line ({"step": 10, "loss": 2.1, ...})
 *   custom    a profile file (-c), optionally built on one of the above
 *
 * GPUs (auto-detected, any number of cards):
 *   amd       amdgpu sysfs: load, VRAM, edge/hotspot/memory temperatures
 *             against the driver's critical limits, power vs cap, fan, and
 *             which processes use each card (via /sys/class/kfd)
 *   nvidia    nvidia-smi --query-gpu / --query-compute-apps per refresh
 *   none      CPU-only training
 * CPU package and NVMe temperatures come from hwmon when present.
 *
 * Read-only: never writes to the training tree, never signals the process
 * (kill(pid, 0) only checks that it exists), never opens files for writing.
 *
 * Build:  make            Test:  make test
 * Run:    ./train_tui -p basicsr /path/to/project exp_name config.yml pid [log]
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/select.h>
#include <sys/time.h>
#include <dirent.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define TRAIN_TUI_VERSION "1.1.0"

/* ----------------------------- tuning ------------------------------ */

#define REFRESH_MS         1000
#define STALE_SECONDS      600    /* must be > longest gap between log lines
                                   * (BasicSR logs every 100 iters ~190s,
                                   * validation blocks can take 15+ min) */
#define LOG_BACKFILL       (8L << 20)  /* on start, read at most the last 8 MiB */
#define CKPT_RESCAN_S      30

#define MAX_LOSS_FIELDS    16   /* max tracked loss/metric fields per profile */
#define MAX_LOSS_LABEL     32
#define MAX_LOSS_MARK      64   /* max length of the "mark" substring to find */
#define MAX_GPUS           16
#define MAX_NVME           8
#define RATE_SAMPLES       64   /* (time, iter) pairs kept for the ETA */

/* ----------------------------- ANSI ------------------------------- */

#define ESC   "\033"
#define CLEAR ESC "[2J"
#define HOME  ESC "[H"
#define CLRLN ESC "[2K"
#define HIDE  ESC "[?25l"
#define SHOW  ESC "[?25h"
#define ALT   ESC "[?1049h"
#define RALT  ESC "[?1049l"
#define ED    ESC "[J"   /* erase to end of screen */

#define RST   ESC "[0m"
#define BOLD  ESC "[1m"
#define DIM   ESC "[2m"

#define FG_RED ESC "[31m"
#define FG_GRN ESC "[32m"
#define FG_YEL ESC "[33m"
#define FG_WHT ESC "[37m"

#define FG_BRED ESC "[91m"
#define FG_BGRN ESC "[92m"
#define FG_BYEL ESC "[93m"
#define FG_BBLU ESC "[94m"
#define FG_BMAG ESC "[95m"
#define FG_BCYN ESC "[96m"
#define FG_BWHT ESC "[97m"

#define FG_ORANGE ESC "[38;5;208m"
#define FG_AMBER  ESC "[38;5;214m"
#define FG_SLATE  ESC "[38;5;102m"
#define FG_STEEL  ESC "[38;5;110m"
#define FG_DKGRAY ESC "[38;5;240m"
#define FG_LTGRAY ESC "[38;5;250m"

#define DEG "\xC2\xB0"   /* ° */

/* ----------------------------- profiles ---------------------------- */

typedef enum {
    PROF_BASICSR = 0,
    PROF_LIGHTNING,
    PROF_HF,
    PROF_JSONL,
    PROF_CUSTOM,
    PROF_COUNT,
} profile_id_t;

/* How the "best" value of a validation metric is found. */
typedef enum {
    BEST_NONE = 0,
    BEST_LOG,      /* read "Best: <val> @ <iter>" from the log (BasicSR) */
    BEST_MIN,      /* lowest value seen (losses) */
    BEST_MAX,      /* highest value seen (psnr, accuracy) */
} best_t;

typedef struct {
    const char *name;        /* "basicsr", "lightning", ... */

    /* log markers: substring to find, then parse the number after it.
     * NULL means "not used by this profile". */
    const char *epoch_mark;  /* e.g. "epoch:" */
    const char *iter_mark;   /* e.g. "iter:"  */
    const char *lr_mark;     /* e.g. "lr:("   */
    const char *eta_mark;    /* e.g. "eta:"   */
    const char *time_mark;   /* e.g. "time (data):"  (per-iter time) */
    const char *save_mark;   /* e.g. "Saving models and training states." */
    const char *val_mark;    /* start of a validation block; NULL = validation
                              * values sit on ordinary log lines (jsonl) */
    const char *total_mark;  /* e.g. "\"total_steps\":" (total iterations) */
    const char *stamp_mark;  /* e.g. "\"time\":" (Unix time of the line) */

    /* iter format: whether the iter number has thousands commas like 50,100 */
    int iter_has_commas;

    /* how many numeric "loss" fields to extract, each with a mark + label */
    int num_loss_fields;
    struct {
        char mark[MAX_LOSS_MARK];
        char label[MAX_LOSS_LABEL];
    } loss_fields[MAX_LOSS_FIELDS];

    /* validation metric extraction (psnr/ssim etc.) */
    int num_val_metrics;
    struct {
        char mark[MAX_LOSS_MARK];   /* "psnr:" */
        char label[MAX_LOSS_LABEL]; /* "psnr"  */
        int best;                   /* best_t */
    } val_metrics[MAX_LOSS_FIELDS];

    /* checkpoint filename pattern */
    const char *ckpt_prefix;   /* "net_g_" */
    const char *ckpt_suffix;   /* ".pth"; NULL = checkpoints are directories */

    /* config keys for total_iter etc. (section + key in a simple YAML) */
    const char *total_iter_key;
    const char *total_iter_section;
    const char *val_freq_key;
    const char *val_freq_section;
    const char *ckpt_freq_key;
    const char *ckpt_freq_section;

    /* 1 = read the YAML config named on the command line */
    int has_config;
} profile_t;

/* ----- built-in profile definitions ----- */

static const profile_t profiles[] = {
[PROF_BASICSR] = {
    .name = "basicsr",
    .epoch_mark = "epoch:",
    .iter_mark  = "iter:",
    .lr_mark    = "lr:(",
    .eta_mark   = "eta:",
    .time_mark  = "time (data):",
    .save_mark  = "Saving models and training states.",
    .val_mark   = "Validation",
    .iter_has_commas = 1,
    .num_loss_fields = 8,
    .loss_fields = {
        {"l_g_pix:",    "l_g_pix"},
        {"l_pix:",      "l_pix"},
        {"l_g_percep:", "l_g_percep"},
        {"l_g_gan:",    "l_g_gan"},
        {"l_d_real:",   "l_d_real"},
        {"l_d_fake:",   "l_d_fake"},
        {"out_d_real:", "out_d_r"},
        {"out_d_fake:", "out_d_f"},
    },
    .num_val_metrics = 2,
    .val_metrics = {
        {"psnr:", "psnr", BEST_LOG},
        {"ssim:", "ssim", BEST_LOG},
    },
    .ckpt_prefix = "net_g_",
    .ckpt_suffix = ".pth",
    .total_iter_key = "total_iter",
    .total_iter_section = "train",
    .val_freq_key = "val_freq",
    .val_freq_section = "val",
    .ckpt_freq_key = "save_checkpoint_freq",
    .ckpt_freq_section = "logger",
    .has_config = 1,
},
[PROF_LIGHTNING] = {
    .name = "lightning",
    /* Lightning logs like:
     *   "Epoch 3:  45%|████▌      | 450/1000 [00:12<00:15, 36.2 it/s, loss=0.234, v_num=1]"
     * We parse "Epoch ", "lr=..." and "loss=..." */
    .epoch_mark = "Epoch ",
    .lr_mark    = "lr=",
    .val_mark   = "Validation",
    .num_loss_fields = 4,
    .loss_fields = {
        {"loss=",     "loss"},
        {"v_num=",    "v_num"},
        {"val_loss=", "val_loss"},
        {"val_acc=",  "val_acc"},
    },
    .ckpt_prefix = "epoch=",
    .ckpt_suffix = ".ckpt",
    .has_config = 0,  /* Lightning uses argparse, not a yml we can parse */
},
[PROF_HF] = {
    .name = "hf",
    /* HuggingFace Trainer logs like:
     *   "  Total optimization steps = 1,000"
     *   "{'loss': 0.234, 'learning_rate': 5e-05, 'epoch': 1.23, 'step': 450}"
     *   "{'eval_loss': 0.456, 'eval_runtime': 12.3, 'epoch': 1.23, 'step': 450}"
     */
    .epoch_mark = "'epoch':",
    .iter_mark  = "'step':",
    .lr_mark    = "'learning_rate':",
    .save_mark  = "Saving model checkpoint",
    .val_mark   = "'eval_loss'",
    .total_mark = "Total optimization steps =",
    .num_loss_fields = 2,
    .loss_fields = {
        {"'loss':",      "loss"},
        {"'grad_norm':", "grad_norm"},
    },
    .num_val_metrics = 3,
    .val_metrics = {
        {"'eval_loss':",     "eval_loss", BEST_MIN},
        {"'eval_accuracy':", "eval_acc",  BEST_MAX},
        {"'eval_f1':",       "eval_f1",   BEST_MAX},
    },
    .ckpt_prefix = "checkpoint-",
    .ckpt_suffix = NULL,  /* HF uses directories named checkpoint-N */
    .has_config = 0,      /* HF config is python, not parseable here */
},
[PROF_JSONL] = {
    .name = "jsonl",
    /* One JSON object per line, as many hand-written training loops log:
     *   {"event": "start", "step": 0, "total_steps": 17509}
     *   {"step": 3000, "loss": 2.026, "lr": 0.0006, "val_loss": 2.066, "time": 1791442036.2}
     * Validation values sit on the same line as the step. */
    .epoch_mark = "\"epoch\":",
    .iter_mark  = "\"step\":",
    .lr_mark    = "\"lr\":",
    .total_mark = "\"total_steps\":",
    .stamp_mark = "\"time\":",
    .num_loss_fields = 3,
    .loss_fields = {
        {"\"loss\":",      "loss"},
        {"\"grad_norm\":", "grad_norm"},
        {"\"gnorm\":",     "gnorm"},
    },
    .num_val_metrics = 3,
    .val_metrics = {
        {"\"val_loss\":",  "val_loss",  BEST_MIN},
        {"\"val\":",       "val",       BEST_MIN},
        {"\"eval_loss\":", "eval_loss", BEST_MIN},
    },
    .has_config = 0,
},
};

/* ----------------------------- GPU model --------------------------- */

typedef enum {
    GPU_AUTO = 0,
    GPU_AMD,
    GPU_NVIDIA,
    GPU_NONE,
} gpu_backend_t;

enum { T_EDGE = 0, T_HOT, T_MEM, T_COUNT };

typedef struct {
    int  nvidia;              /* 0 = amdgpu sysfs, 1 = nvidia-smi */
    char name[48];            /* "RX 9050 / 9060 XT", "RTX 4090" */
    char dev[PATH_MAX];       /* AMD: resolved PCI device directory */
    char hwmon[PATH_MAX];     /* AMD: its amdgpu hwmon directory ("" = none) */
    int  t_idx[T_COUNT];      /* AMD: hwmon tempN of edge/junction/mem, 0 = none */
    int  kfd_id;              /* AMD: KFD gpu_id, 0 = unknown */
    char uuid[64];            /* NVIDIA */

    int  busy;                /* load %, -1 = unknown */
    long long vram_used, vram_total;   /* bytes */
    int  temp[T_COUNT];       /* °C, -1 = not reported */
    int  crit[T_COUNT];       /* driver's critical limit in °C, -1 = unknown */
    int  power_w, cap_w;      /* -1 = unknown */
    int  fan_rpm, fan_pct;    /* -1 = unknown */
    int  run_procs;           /* processes of the watched run on this card */
    int  other_procs;         /* other compute processes */
} gpu_t;

/* ----------------------------- state ------------------------------- */

typedef enum {
    ST_TRAINING = 0,
    ST_VALIDATING,
    ST_SAVING,
    ST_IDLE,
    ST_CRASHED,
    ST_DONE,
    ST_UNKNOWN,
} run_state_t;

static const char *state_label[] = {
    "TRAINING", "VALIDATING", "SAVING", "IDLE", "CRASHED", "FINISHED", "UNKNOWN",
};
static const char *state_glyph[] = {
    FG_BGRN "\xE2\x97\x8F" RST,   /* ● */
    FG_BYEL "\xE2\x97\x8F" RST,
    FG_BBLU "\xE2\x97\x8F" RST,
    FG_SLATE "\xE2\x97\x8B" RST,  /* ○ */
    FG_BRED "\xE2\x9C\x96" RST,   /* ✖ */
    FG_BGRN "\xE2\x9C\x94" RST,   /* ✔ */
    FG_YEL  "?" RST,
};
static const char *state_tag_col[] = {
    FG_BGRN, FG_BYEL, FG_BBLU, FG_SLATE, FG_BRED, FG_BGRN, FG_YEL,
};

/* generic metric value (loss or validation metric) */
typedef struct {
    int have;
    double value;
    int have_best;
    double best;
    long best_iter;
} metric_t;

typedef struct {
    double t;
    long iter;
} sample_t;

typedef struct {
    char path[PATH_MAX];      /* tempN_input */
    int crit;                 /* °C, -1 = unknown */
} temp_src_t;

typedef struct {
    /* user config */
    char project_root[1024];
    char exp_name[256];
    char cfg_name[1024];
    char log_path[PATH_MAX];
    char cfg_path[PATH_MAX];
    char ckpt_dir[PATH_MAX];
    pid_t pid;
    long total_iter_override;   /* >0 if set via CLI */
    gpu_backend_t gpu_backend;

    /* profile */
    profile_id_t profile_id;
    profile_t profile;         /* copy (so custom can mutate) */

    /* derived */
    long total_iter;           /* from the config */
    long total_from_log;       /* from profile.total_mark */
    long val_freq;
    long ckpt_freq;

    /* latest parsed log */
    long cur_iter;
    double epoch;
    int have_epoch;
    double lr;
    int have_lr;
    char eta[64];
    double t_iter, t_data;
    int have_titer;
    metric_t losses[MAX_LOSS_FIELDS];
    int have_losses;

    /* validation */
    int have_val;              /* a validation block is being written now */
    int in_validation;
    int have_val_last;
    long val_iter;             /* iter of the last validation result */
    metric_t val_last[MAX_LOSS_FIELDS];

    /* progress rate (for an ETA when the log has none) */
    sample_t samples[RATE_SAMPLES];
    int nsamples;

    /* checkpoint */
    int have_ckpt;
    char last_ckpt_name[256];
    time_t last_ckpt_mtime;
    long last_ckpt_iter;
    long last_ckpt_scan_save_iter; /* save_at_iter when we last scanned */
    time_t last_ckpt_scan;

    /* GPUs */
    gpu_t gpus[MAX_GPUS];
    int ngpus;
    int n_amd;                 /* gpus[0..n_amd) are amdgpu cards */
    int use_nvidia;
    int procs_known;           /* per-GPU process counts are meaningful */

    /* host temperatures */
    temp_src_t cpu_src;
    temp_src_t nvme_src[MAX_NVME];
    int n_nvme;
    int cpu_temp, cpu_crit;
    int nvme_temp, nvme_crit;

    /* state machine */
    run_state_t state;
    time_t last_log_mtime;
    long save_at_iter;
    int proc_alive;
    int log_exists;

    /* incremental log reading */
    off_t log_offset;
    ino_t log_ino;
    int log_initialized;
    int skip_partial;          /* first line after a mid-file seek is cut */
    int backfilling;           /* reading old lines (no wall-clock samples) */

    time_t now;
} ctx_t;

/* ----------------------------- output ------------------------------ */

static FILE *g_out;            /* the frame being built (open_memstream) */
static int g_live = 1;         /* full-screen refresh loop (not --once) */
static int g_refresh_ms = REFRESH_MS;
static volatile sig_atomic_t g_resized;
static int g_strip_all;        /* --once into a pipe: no escape codes */
static int g_no_color;         /* NO_COLOR: keep cursor codes, drop colors */
static char g_sysroot[PATH_MAX];   /* prefix for /sys paths (tests) */
static char g_nvsmi[PATH_MAX];     /* nvidia-smi to run, "" = none */

__attribute__((format(printf, 1, 2)))
static void pf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_out, fmt, ap);
    va_end(ap);
}

static void ps(const char *s) { fputs(s, g_out); }

/* pf() that returns the visible width of what it printed (escape codes
 * excluded; the text must be ASCII). */
__attribute__((format(printf, 1, 2)))
static int pf_len(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fputs(buf, g_out);
    int w = 0;
    for (const char *p = buf; *p; p++) {
        if (*p == '\033') {
            while (*p && !(*p >= 0x40 && *p <= 0x7e && *p != '[')) p++;
            if (!*p) break;
            continue;
        }
        w++;
    }
    return w;
}

static void pad(int n) { while (n-- > 0) fputc(' ', g_out); }

/* Write a finished frame, dropping escape sequences when asked to. */
static void emit_frame(const char *buf, size_t len) {
    if (!g_strip_all && !g_no_color) {
        fwrite(buf, 1, len, stdout);
        fflush(stdout);
        return;
    }
    size_t i = 0;
    while (i < len) {
        if (buf[i] == '\033' && i + 1 < len && buf[i + 1] == '[') {
            size_t j = i + 2;
            while (j < len && !((unsigned char)buf[j] >= 0x40 &&
                                (unsigned char)buf[j] <= 0x7e))
                j++;
            if (j >= len) break;
            if (!g_strip_all && buf[j] != 'm')
                fwrite(buf + i, 1, j + 1 - i, stdout);
            i = j + 1;
            continue;
        }
        fputc(buf[i++], stdout);
    }
    fflush(stdout);
}

/* ----------------------------- helpers ---------------------------- */

static void die(const char *msg) {
    fprintf(stderr, "train-tui: %s\n", msg);
    exit(1);
}

__attribute__((format(printf, 3, 4)))
static int pathf(char *buf, size_t n, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vsnprintf(buf, n, fmt, ap);
    va_end(ap);
    return r;
}

/* Like pathf, but prefixed with the sysfs root (TRAIN_TUI_SYSFS in tests). */
__attribute__((format(printf, 3, 4)))
static int sysp(char *buf, size_t n, const char *fmt, ...) {
    int k = snprintf(buf, n, "%s", g_sysroot);
    if (k < 0 || (size_t)k >= n) return -1;
    va_list ap;
    va_start(ap, fmt);
    int r = vsnprintf(buf + k, n - (size_t)k, fmt, ap);
    va_end(ap);
    return r;
}

static long long read_ll(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    char buf[64];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = 0;
    char *end;
    long long v = strtoll(buf, &end, 10);
    if (end == buf) return -1;
    return v;
}

/* First line of a small text file, trailing whitespace removed. */
static int read_text(const char *path, char *dst, size_t n) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t r = read(fd, dst, n - 1);
    close(fd);
    if (r <= 0) return -1;
    dst[r] = 0;
    dst[strcspn(dst, "\n")] = 0;
    size_t L = strlen(dst);
    while (L && isspace((unsigned char)dst[L - 1])) dst[--L] = 0;
    return 0;
}

/* Bounded copy (snprintf "%s" trips gcc's truncation warnings). */
static void copy_str(char *dst, size_t n, const char *src) {
    size_t L = strlen(src);
    if (L >= n) L = n - 1;
    memcpy(dst, src, L);
    dst[L] = 0;
}

static int milli_c(long long v) { return v < 0 ? -1 : (int)((v + 500) / 1000); }

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    size_t L = strlen(s);
    while (L && isspace((unsigned char)s[L - 1])) s[--L] = 0;
    return s;
}

/* ----------------------------- GPU names -------------------------- */

/* "Navi 44 [Radeon RX 9050 / 9060 XT]" -> "RX 9050 / 9060 XT",
 * "NVIDIA GeForce RTX 4090" -> "RTX 4090", "Vega 20 [Radeon VII]" -> "Radeon VII" */
static void tidy_gpu_name(const char *raw, char *dst, size_t n) {
    char tmp[128];
    const char *lb = strchr(raw, '[');
    const char *rb = lb ? strchr(lb, ']') : NULL;
    if (lb && rb && rb > lb + 1)
        snprintf(tmp, sizeof(tmp), "%.*s", (int)(rb - lb - 1), lb + 1);
    else
        snprintf(tmp, sizeof(tmp), "%s", raw);
    const char *s = tmp;
    static const char *drop[] = { "NVIDIA ", "GeForce ", "AMD " };
    for (size_t i = 0; i < sizeof(drop) / sizeof(drop[0]); i++)
        if (strncmp(s, drop[i], strlen(drop[i])) == 0) s += strlen(drop[i]);
    if (strncmp(s, "Radeon RX ", 10) == 0 || strncmp(s, "Radeon Pro ", 11) == 0)
        s += 7;
    copy_str(dst, n, s);
}

/* Look a PCI device up in the system's pci.ids (hwdata). */
static int pci_ids_lookup(unsigned ven, unsigned dev, char *dst, size_t n) {
    const char *paths[] = {
        getenv("TRAIN_TUI_PCI_IDS"),
        "/usr/share/hwdata/pci.ids",
        "/usr/share/misc/pci.ids",
        "/usr/share/pci.ids",
    };
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        if (!paths[i] || !*paths[i]) continue;
        FILE *f = fopen(paths[i], "r");
        if (!f) continue;
        char line[512];
        int in_vendor = 0, found = 0;
        while (fgets(line, sizeof(line), f)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            if (line[0] != '\t') {
                unsigned v;
                if (in_vendor) break;           /* past our vendor */
                in_vendor = sscanf(line, "%4x", &v) == 1 && v == ven;
                continue;
            }
            if (!in_vendor || line[1] == '\t') continue;
            unsigned d;
            if (sscanf(line + 1, "%4x", &d) == 1 && d == dev) {
                char *name = line + 5;
                while (*name == ' ') name++;
                name[strcspn(name, "\n")] = 0;
                copy_str(dst, n, name);
                found = 1;
                break;
            }
        }
        fclose(f);
        if (found) return 0;
    }
    return -1;
}

/* ----------------------------- GPU: AMD sysfs --------------------- */

/* "card0" is a GPU; "card0-HDMI-A-1" is one of its connectors. */
static int is_card_name(const char *s, int *num) {
    if (strncmp(s, "card", 4) != 0 || !isdigit((unsigned char)s[4])) return 0;
    char *end;
    long v = strtol(s + 4, &end, 10);
    if (*end) return 0;
    *num = (int)v;
    return 1;
}

static void amd_find_hwmon(gpu_t *g) {
    char dir[PATH_MAX];
    pathf(dir, sizeof(dir), "%s/hwmon", g->dev);
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, "hwmon", 5) != 0) continue;
        char p[PATH_MAX], name[64];
        pathf(p, sizeof(p), "%s/%s/name", dir, e->d_name);
        if (read_text(p, name, sizeof(name)) == 0 && strcmp(name, "amdgpu") == 0) {
            pathf(g->hwmon, sizeof(g->hwmon), "%s/%s", dir, e->d_name);
            break;
        }
    }
    closedir(d);
    if (!g->hwmon[0]) return;

    /* Labelled sensors: edge, junction (hotspot), mem. Older kernels only
     * have an unlabelled temp1, which is the edge temperature. */
    for (int i = 1; i <= 8; i++) {
        char p[PATH_MAX], label[32];
        pathf(p, sizeof(p), "%s/temp%d_label", g->hwmon, i);
        if (read_text(p, label, sizeof(label)) != 0) continue;
        if (strcmp(label, "edge") == 0) g->t_idx[T_EDGE] = i;
        else if (strcmp(label, "junction") == 0) g->t_idx[T_HOT] = i;
        else if (strcmp(label, "mem") == 0) g->t_idx[T_MEM] = i;
    }
    if (!g->t_idx[T_EDGE] && !g->t_idx[T_HOT] && !g->t_idx[T_MEM])
        g->t_idx[T_EDGE] = 1;
    for (int k = 0; k < T_COUNT; k++) {
        g->crit[k] = -1;
        if (!g->t_idx[k]) continue;
        char p[PATH_MAX];
        pathf(p, sizeof(p), "%s/temp%d_crit", g->hwmon, g->t_idx[k]);
        g->crit[k] = milli_c(read_ll(p));
    }
}

static void amd_name(gpu_t *g) {
    char p[PATH_MAX], buf[256];
    pathf(p, sizeof(p), "%s/product_name", g->dev);
    if (read_text(p, buf, sizeof(buf)) == 0 && buf[0]) {
        tidy_gpu_name(buf, g->name, sizeof(g->name));
        return;
    }
    unsigned ven = 0, dev = 0;
    pathf(p, sizeof(p), "%s/uevent", g->dev);
    FILE *f = fopen(p, "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f))
            if (sscanf(line, "PCI_ID=%x:%x", &ven, &dev) == 2) break;
        fclose(f);
    }
    if (dev && pci_ids_lookup(ven, dev, buf, sizeof(buf)) == 0)
        tidy_gpu_name(buf, g->name, sizeof(g->name));
    else
        snprintf(g->name, sizeof(g->name), "AMD %04x:%04x", ven, dev);
}

/* Match KFD topology nodes (what /sys/class/kfd/kfd/proc refers to) to
 * the cards through their render node. */
static void amd_map_kfd(ctx_t *c) {
    char nodes[PATH_MAX];
    sysp(nodes, sizeof(nodes), "/sys/class/kfd/kfd/topology/nodes");
    DIR *d = opendir(nodes);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char p[PATH_MAX];
        pathf(p, sizeof(p), "%s/%s/gpu_id", nodes, e->d_name);
        long long id = read_ll(p);
        if (id <= 0) continue;                 /* CPU node */
        pathf(p, sizeof(p), "%s/%s/properties", nodes, e->d_name);
        FILE *f = fopen(p, "r");
        if (!f) continue;
        int minor = -1;
        char line[256];
        while (fgets(line, sizeof(line), f))
            if (sscanf(line, "drm_render_minor %d", &minor) == 1) break;
        fclose(f);
        if (minor < 0) continue;
        char r[PATH_MAX], real[PATH_MAX];
        sysp(r, sizeof(r), "/sys/class/drm/renderD%d/device", minor);
        if (!realpath(r, real)) continue;
        for (int i = 0; i < c->n_amd; i++)
            if (strcmp(c->gpus[i].dev, real) == 0) c->gpus[i].kfd_id = (int)id;
    }
    closedir(d);
}

static void amd_discover(ctx_t *c) {
    char base[PATH_MAX];
    sysp(base, sizeof(base), "/sys/class/drm");
    DIR *d = opendir(base);
    if (!d) return;
    int nums[MAX_GPUS], n = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL && n < MAX_GPUS) {
        int num;
        if (!is_card_name(e->d_name, &num)) continue;
        char p[PATH_MAX];
        pathf(p, sizeof(p), "%s/%s/device/gpu_busy_percent", base, e->d_name);
        if (access(p, R_OK) == 0) nums[n++] = num;
    }
    closedir(d);
    qsort(nums, (size_t)n, sizeof(int), cmp_int);

    for (int i = 0; i < n && c->ngpus < MAX_GPUS; i++) {
        gpu_t *g = &c->gpus[c->ngpus];
        memset(g, 0, sizeof(*g));
        char p[PATH_MAX];
        pathf(p, sizeof(p), "%s/card%d/device", base, nums[i]);
        if (!realpath(p, g->dev)) continue;
        amd_find_hwmon(g);
        amd_name(g);
        c->ngpus++;
    }
    c->n_amd = c->ngpus;
    amd_map_kfd(c);
}

static void amd_read(gpu_t *g) {
    char p[PATH_MAX];
    pathf(p, sizeof(p), "%s/gpu_busy_percent", g->dev);
    g->busy = (int)read_ll(p);
    pathf(p, sizeof(p), "%s/mem_info_vram_used", g->dev);
    g->vram_used = read_ll(p);
    pathf(p, sizeof(p), "%s/mem_info_vram_total", g->dev);
    g->vram_total = read_ll(p);
    if (g->vram_used < 0) g->vram_used = 0;
    if (g->vram_total < 0) g->vram_total = 0;

    for (int k = 0; k < T_COUNT; k++) g->temp[k] = -1;
    g->power_w = g->cap_w = g->fan_rpm = g->fan_pct = -1;
    if (!g->hwmon[0]) return;

    for (int k = 0; k < T_COUNT; k++) {
        if (!g->t_idx[k]) continue;
        pathf(p, sizeof(p), "%s/temp%d_input", g->hwmon, g->t_idx[k]);
        g->temp[k] = milli_c(read_ll(p));
    }
    /* Older cards report power1_average, RDNA3+/some Vega power1_input. */
    pathf(p, sizeof(p), "%s/power1_average", g->hwmon);
    long long uw = read_ll(p);
    if (uw < 0) {
        pathf(p, sizeof(p), "%s/power1_input", g->hwmon);
        uw = read_ll(p);
    }
    if (uw >= 0) g->power_w = (int)((uw + 500000) / 1000000);
    pathf(p, sizeof(p), "%s/power1_cap", g->hwmon);
    uw = read_ll(p);
    if (uw > 0) g->cap_w = (int)((uw + 500000) / 1000000);
    pathf(p, sizeof(p), "%s/fan1_input", g->hwmon);
    g->fan_rpm = (int)read_ll(p);
}

/* Count compute processes per card from /sys/class/kfd/kfd/proc/<pid>. */
static void amd_procs(ctx_t *c) {
    int any = 0;
    for (int i = 0; i < c->n_amd; i++) any |= c->gpus[i].kfd_id != 0;
    if (!any) return;
    char dir[PATH_MAX];
    sysp(dir, sizeof(dir), "/sys/class/kfd/kfd/proc");
    DIR *d = opendir(dir);
    if (!d) return;
    c->procs_known = 1;
    pid_t run_pg = c->pid > 0 ? getpgid(c->pid) : -1;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!isdigit((unsigned char)e->d_name[0])) continue;
        pid_t p = (pid_t)atoi(e->d_name);
        int mine = p == c->pid || (run_pg > 0 && getpgid(p) == run_pg);
        for (int i = 0; i < c->n_amd; i++) {
            gpu_t *g = &c->gpus[i];
            if (!g->kfd_id) continue;
            char f[PATH_MAX];
            pathf(f, sizeof(f), "%s/%s/vram_%d", dir, e->d_name, g->kfd_id);
            if (read_ll(f) > 0) {
                if (mine) g->run_procs++;
                else g->other_procs++;
            }
        }
    }
    closedir(d);
}

/* ----------------------------- GPU: NVIDIA nvidia-smi -------------- */

static int find_nvidia_smi(void) {
    const char *env = getenv("TRAIN_TUI_NVIDIA_SMI");
    if (env && *env) {
        if (access(env, X_OK) != 0) return 0;
        snprintf(g_nvsmi, sizeof(g_nvsmi), "%s", env);
        return 1;
    }
    const char *path = getenv("PATH");
    char *copy = strdup(path && *path ? path : "/usr/bin:/bin");
    if (!copy) return 0;
    int found = 0;
    char *save = NULL;
    for (char *dir = strtok_r(copy, ":", &save); dir; dir = strtok_r(NULL, ":", &save)) {
        char cand[PATH_MAX];
        pathf(cand, sizeof(cand), "%s/nvidia-smi", *dir ? dir : ".");
        if (access(cand, X_OK) == 0) {
            snprintf(g_nvsmi, sizeof(g_nvsmi), "%s", cand);
            found = 1;
            break;
        }
    }
    free(copy);
    return found;
}

/* nvidia-smi prints "N/A", "[N/A]" or "[Not Supported]" for missing values. */
static int nv_int(const char *s) {
    char *end;
    double v = strtod(s, &end);
    if (end == s) return -1;
    return (int)(v + 0.5);
}

static void nv_read(ctx_t *c) {
    c->ngpus = c->n_amd;
    char cmd[PATH_MAX + 512];
    pathf(cmd, sizeof(cmd),
          "'%s' --query-gpu=index,uuid,name,utilization.gpu,memory.used,"
          "memory.total,temperature.gpu,temperature.memory,power.draw,"
          "power.limit,fan.speed --format=csv,noheader,nounits 2>/dev/null",
          g_nvsmi);
    FILE *fp = popen(cmd, "r");
    if (!fp) return;
    char line[1024];
    while (fgets(line, sizeof(line), fp) && c->ngpus < MAX_GPUS) {
        char *f[11], *s = line, *tok;
        int nf = 0;
        while (nf < 11 && (tok = strsep(&s, ",")) != NULL) f[nf++] = trim(tok);
        if (nf < 11) continue;
        gpu_t *g = &c->gpus[c->ngpus];
        memset(g, 0, sizeof(*g));
        g->nvidia = 1;
        snprintf(g->uuid, sizeof(g->uuid), "%s", f[1]);
        tidy_gpu_name(f[2], g->name, sizeof(g->name));
        g->busy = nv_int(f[3]);
        long long used = nv_int(f[4]), total = nv_int(f[5]);   /* MiB */
        g->vram_used = used > 0 ? used << 20 : 0;
        g->vram_total = total > 0 ? total << 20 : 0;
        g->temp[T_EDGE] = nv_int(f[6]);
        g->temp[T_HOT] = -1;
        g->temp[T_MEM] = nv_int(f[7]);
        for (int k = 0; k < T_COUNT; k++) g->crit[k] = -1;
        g->power_w = nv_int(f[8]);
        g->cap_w = nv_int(f[9]);
        g->fan_pct = nv_int(f[10]);
        g->fan_rpm = -1;
        c->ngpus++;
    }
    pclose(fp);
    if (c->ngpus == c->n_amd) return;

    pathf(cmd, sizeof(cmd),
          "'%s' --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null",
          g_nvsmi);
    fp = popen(cmd, "r");
    if (!fp) return;
    c->procs_known = 1;
    pid_t run_pg = c->pid > 0 ? getpgid(c->pid) : -1;
    while (fgets(line, sizeof(line), fp)) {
        char *s = line, *pid_s = strsep(&s, ",");
        if (!s) continue;
        pid_t p = (pid_t)atoi(trim(pid_s));
        char *uuid = trim(s);
        int mine = p == c->pid || (run_pg > 0 && getpgid(p) == run_pg);
        for (int i = c->n_amd; i < c->ngpus; i++) {
            if (strcmp(c->gpus[i].uuid, uuid) != 0) continue;
            if (mine) c->gpus[i].run_procs++;
            else c->gpus[i].other_procs++;
        }
    }
    pclose(fp);
}

/* ----------------------------- GPU dispatch ----------------------- */

static void gpu_discover(ctx_t *c) {
    if (c->gpu_backend == GPU_NONE) return;
    if (c->gpu_backend == GPU_AUTO || c->gpu_backend == GPU_AMD)
        amd_discover(c);
    if (c->gpu_backend == GPU_AUTO || c->gpu_backend == GPU_NVIDIA)
        c->use_nvidia = find_nvidia_smi();
}

static void read_gpus(ctx_t *c) {
    c->procs_known = 0;
    for (int i = 0; i < c->n_amd; i++) {
        amd_read(&c->gpus[i]);
        c->gpus[i].run_procs = c->gpus[i].other_procs = 0;
    }
    amd_procs(c);
    if (c->use_nvidia) nv_read(c);
}

/* Highest load among the cards this run uses (all cards if unknown). */
static int run_gpu_busy(const ctx_t *c) {
    int any_run = 0, best = -1;
    for (int i = 0; i < c->ngpus; i++) any_run |= c->gpus[i].run_procs > 0;
    for (int i = 0; i < c->ngpus; i++) {
        const gpu_t *g = &c->gpus[i];
        if (c->procs_known && any_run && !g->run_procs) continue;
        if (g->busy > best) best = g->busy;
    }
    return best;
}

/* ----------------------------- host temperatures ------------------ */

static void host_discover(ctx_t *c) {
    char base[PATH_MAX];
    sysp(base, sizeof(base), "/sys/class/hwmon");
    DIR *d = opendir(base);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, "hwmon", 5) != 0) continue;
        char p[PATH_MAX], name[64];
        pathf(p, sizeof(p), "%s/%s/name", base, e->d_name);
        if (read_text(p, name, sizeof(name)) != 0) continue;

        if (!c->cpu_src.path[0] &&
            (!strcmp(name, "coretemp") || !strcmp(name, "k10temp") ||
             !strcmp(name, "zenpower"))) {
            /* Prefer the package / die sensor over single cores. */
            static const char *want[] = { "Package id 0", "Tdie", "Tctl" };
            int idx = 0, rank = 99;
            for (int i = 1; i <= 64; i++) {
                char label[64];
                pathf(p, sizeof(p), "%s/%s/temp%d_label", base, e->d_name, i);
                if (read_text(p, label, sizeof(label)) != 0) continue;
                for (int w = 0; w < 3; w++)
                    if (!strcmp(label, want[w]) && w < rank) { rank = w; idx = i; }
            }
            if (!idx) idx = 1;
            pathf(c->cpu_src.path, sizeof(c->cpu_src.path),
                  "%s/%s/temp%d_input", base, e->d_name, idx);
            pathf(p, sizeof(p), "%s/%s/temp%d_crit", base, e->d_name, idx);
            c->cpu_src.crit = milli_c(read_ll(p));
            if (access(c->cpu_src.path, R_OK) != 0) c->cpu_src.path[0] = 0;
        } else if (!strcmp(name, "nvme") && c->n_nvme < MAX_NVME) {
            temp_src_t *t = &c->nvme_src[c->n_nvme];
            pathf(t->path, sizeof(t->path), "%s/%s/temp1_input", base, e->d_name);
            pathf(p, sizeof(p), "%s/%s/temp1_crit", base, e->d_name);
            t->crit = milli_c(read_ll(p));
            if (access(t->path, R_OK) == 0) c->n_nvme++;
        }
    }
    closedir(d);
}

static void read_host(ctx_t *c) {
    c->cpu_temp = c->cpu_src.path[0] ? milli_c(read_ll(c->cpu_src.path)) : -1;
    c->cpu_crit = c->cpu_src.crit;
    c->nvme_temp = -1;
    c->nvme_crit = -1;
    for (int i = 0; i < c->n_nvme; i++) {     /* show the hottest drive */
        int t = milli_c(read_ll(c->nvme_src[i].path));
        if (t > c->nvme_temp) {
            c->nvme_temp = t;
            c->nvme_crit = c->nvme_src[i].crit;
        }
    }
}

/* ----------------------------- process ---------------------------- */

static void check_proc(ctx_t *c) {
    if (c->pid <= 0) { c->proc_alive = 0; return; }
    /* EPERM: it exists but belongs to another user */
    c->proc_alive = kill(c->pid, 0) == 0 || errno == EPERM;
}

/* ----------------------------- log parsing ----------------------- */

/* Find the numeric value after a mark substring. Returns 1 if found. */
static int parse_num_after(const char *line, const char *mark, double *out) {
    const char *p = strstr(line, mark);
    if (!p) return 0;
    p += strlen(mark);
    while (*p && (isspace((unsigned char)*p) || *p == '\'' || *p == ':'))
        p++;
    char *end;
    double v = strtod(p, &end);
    if (end == p) return 0;
    *out = v;
    return 1;
}

/* Parse an integer after mark, optionally with thousands commas. */
static long parse_iter_field(const char *line, const char *mark, int commas) {
    const char *p = strstr(line, mark);
    if (!p) return -1;
    p += strlen(mark);
    while (*p && isspace((unsigned char)*p)) p++;
    if (!commas) return strtol(p, NULL, 10);
    char numbuf[32];
    size_t j = 0;
    while (*p && (isdigit((unsigned char)*p) || *p == ',') && j < sizeof(numbuf) - 1) {
        if (*p != ',') numbuf[j++] = *p;
        p++;
    }
    numbuf[j] = 0;
    return strtol(numbuf, NULL, 10);
}

static void add_sample(ctx_t *c, double t, long iter) {
    if (c->nsamples > 0) {
        const sample_t *last = &c->samples[c->nsamples - 1];
        if (iter < last->iter) c->nsamples = 0;          /* rollback / restart */
        else if (iter == last->iter || t <= last->t) return;
    }
    if (c->nsamples == RATE_SAMPLES) {
        memmove(c->samples, c->samples + 1, sizeof(sample_t) * (RATE_SAMPLES - 1));
        c->nsamples--;
    }
    c->samples[c->nsamples].t = t;
    c->samples[c->nsamples].iter = iter;
    c->nsamples++;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Iterations per second: the median of the recent step-to-step rates, so a
 * pause (a restart, someone borrowing the GPU) does not drag the ETA. */
static double iter_rate(const ctx_t *c) {
    double r[RATE_SAMPLES];
    int n = 0;
    for (int i = 1; i < c->nsamples; i++) {
        double dt = c->samples[i].t - c->samples[i - 1].t;
        if (dt > 0) r[n++] = (c->samples[i].iter - c->samples[i - 1].iter) / dt;
    }
    if (n == 0) return 0;
    qsort(r, (size_t)n, sizeof(double), cmp_double);
    return n % 2 ? r[n / 2] : (r[n / 2 - 1] + r[n / 2]) / 2;
}

static void record_val(ctx_t *c, int i, double v, const char *line, int running) {
    metric_t *m = &c->val_last[i];
    m->have = 1;
    m->value = v;
    switch (c->profile.val_metrics[i].best) {
    case BEST_LOG: {
        const char *b = strstr(line, "Best:");
        if (b) {
            m->best = strtod(b + 5, NULL);
            m->have_best = 1;
            const char *at = strchr(b, '@');
            m->best_iter = at ? strtol(at + 1, NULL, 10) : 0;
        }
        break;
    }
    case BEST_MIN:
        if (!m->have_best || v < m->best) {
            m->best = v; m->best_iter = c->cur_iter; m->have_best = 1;
        }
        break;
    case BEST_MAX:
        if (!m->have_best || v > m->best) {
            m->best = v; m->best_iter = c->cur_iter; m->have_best = 1;
        }
        break;
    default:
        break;
    }
    c->have_val_last = 1;
    c->val_iter = c->cur_iter;
    if (running) c->have_val = 1;
}

static void parse_val_metrics(ctx_t *c, const char *line, int running) {
    const profile_t *pr = &c->profile;
    for (int i = 0; i < pr->num_val_metrics; i++) {
        double v;
        if (parse_num_after(line, pr->val_metrics[i].mark, &v))
            record_val(c, i, v, line, running);
    }
}

static void parse_log_line(ctx_t *c, const char *line) {
    const profile_t *pr = &c->profile;
    double stamp = 0;

    if (pr->save_mark && strstr(line, pr->save_mark)) {
        c->save_at_iter = c->cur_iter;
        return;
    }

    /* line timestamp: "YYYY-MM-DD HH:MM:SS,mmm" (BasicSR) or a stamp mark */
    {
        int y, mo, d, h, mi, s, ms;
        if (sscanf(line, "%d-%d-%d %d:%d:%d,%d", &y, &mo, &d, &h, &mi, &s, &ms) == 7) {
            struct tm tm = {0};
            tm.tm_year = y - 1900; tm.tm_mon = mo - 1; tm.tm_mday = d;
            tm.tm_hour = h; tm.tm_min = mi; tm.tm_sec = s;
            tm.tm_isdst = -1;
            stamp = (double)mktime(&tm);
        }
    }
    if (pr->stamp_mark) {
        double v;
        if (parse_num_after(line, pr->stamp_mark, &v) && v > 1e9) stamp = v;
    }
    if (pr->total_mark) {
        long t = parse_iter_field(line, pr->total_mark, 1);
        if (t > 0) c->total_from_log = t;
    }

    int has_iter = (pr->iter_mark && strstr(line, pr->iter_mark)) ||
                   (pr->epoch_mark && strstr(line, pr->epoch_mark));
    int val_start = pr->val_mark && strstr(line, pr->val_mark);

    /* BasicSR-style block: a "Validation" line, then one metric per line */
    if (val_start && !has_iter) {
        c->in_validation = 1;
        return;
    }
    if (c->in_validation && !has_iter) {
        parse_val_metrics(c, line, 1);
        return;
    }
    if (has_iter) {
        c->in_validation = 0;
        c->have_val = 0;
    }

    if (pr->epoch_mark) {
        double v;
        if (parse_num_after(line, pr->epoch_mark, &v)) {
            c->epoch = v;
            c->have_epoch = 1;
        }
    }

    if (pr->iter_mark) {
        long it = parse_iter_field(line, pr->iter_mark, pr->iter_has_commas);
        if (it > 0) {
            c->cur_iter = it;
            if (stamp == 0 && !c->backfilling) stamp = (double)c->now;
            if (stamp > 0) add_sample(c, stamp, it);
        }
    }

    if (pr->lr_mark) {
        double v;
        if (parse_num_after(line, pr->lr_mark, &v)) {
            c->lr = v;
            c->have_lr = 1;
        }
    }

    if (pr->eta_mark) {
        const char *ep = strstr(line, pr->eta_mark);
        if (ep) {
            const char *end = strchr(ep, ']');
            const char *src = ep + strlen(pr->eta_mark);
            size_t n = end ? (size_t)(end - src) : strlen(src);
            if (n >= sizeof(c->eta)) n = sizeof(c->eta) - 1;
            memcpy(c->eta, src, n);
            c->eta[n] = 0;
            char *ct = strstr(c->eta, ", time");
            if (ct) *ct = 0;
            char *t = trim(c->eta);
            memmove(c->eta, t, strlen(t) + 1);
            n = strlen(c->eta);
            while (n > 0 && c->eta[n - 1] == ',') c->eta[--n] = 0;
        }
    }

    if (pr->time_mark) {
        const char *tp = strstr(line, pr->time_mark);
        if (tp) {
            double ti = 0, td = 0;
            if (sscanf(tp + strlen(pr->time_mark), " %lf (%lf)", &ti, &td) == 2) {
                c->t_iter = ti;
                c->t_data = td;
                c->have_titer = 1;
            }
        }
    }

    /* jsonl-style: validation values on the step line; HF: the eval line */
    if (val_start || (!pr->val_mark && pr->num_val_metrics > 0))
        parse_val_metrics(c, line, 0);

    for (int i = 0; i < pr->num_loss_fields; i++) {
        double v;
        if (parse_num_after(line, pr->loss_fields[i].mark, &v)) {
            c->losses[i].have = 1;
            c->losses[i].value = v;
            c->have_losses = 1;
        }
    }
}

/* ----------------------------- log tail --------------------------- */

/* Read only what was appended since the last refresh. The first call reads
 * up to LOG_BACKFILL from the end, so "best" values and the ETA have history.
 * A replaced or truncated file is read again from the start. */
static void tail_log(ctx_t *c) {
    int fd = open(c->log_path, O_RDONLY);
    if (fd < 0) { c->log_exists = 0; return; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); c->log_exists = 0; return; }
    c->log_exists = 1;
    c->last_log_mtime = st.st_mtime;

    if (!c->log_initialized || st.st_ino != c->log_ino || st.st_size < c->log_offset) {
        c->log_ino = st.st_ino;
        c->log_offset = st.st_size > LOG_BACKFILL ? st.st_size - LOG_BACKFILL : 0;
        c->skip_partial = c->log_offset > 0;
        c->nsamples = 0;
        c->in_validation = 0;
        c->have_val = 0;
        c->log_initialized = 1;
        c->backfilling = 1;
    }

    static char buf[65536 + 1];
    while (c->log_offset < st.st_size) {
        ssize_t n = pread(fd, buf, sizeof(buf) - 1, c->log_offset);
        if (n <= 0) break;
        buf[n] = 0;
        char *line = buf, *nl;
        while ((nl = memchr(line, '\n', (size_t)(buf + n - line))) != NULL) {
            *nl = 0;
            if (c->skip_partial) c->skip_partial = 0;
            else parse_log_line(c, line);
            line = nl + 1;
        }
        size_t used = (size_t)(line - buf);
        if (used == 0) {
            if ((size_t)n < sizeof(buf) - 1) break;   /* wait for the line's end */
            parse_log_line(c, buf);                   /* absurdly long line */
            used = (size_t)n;
        }
        c->log_offset += (off_t)used;
    }
    c->backfilling = 0;
    close(fd);
}

/* ----------------------------- config parsing --------------------- */

static void parse_config(ctx_t *c) {
    if (!c->profile.has_config || !c->cfg_path[0]) return;
    FILE *f = fopen(c->cfg_path, "r");
    if (!f) return;
    char line[512];
    char section[64] = {0};
    while (fgets(line, sizeof(line), f)) {
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == ' ' || line[L-1] == '\r'))
            line[--L] = 0;
        if (L == 0 || line[0] == '#') continue;
        if (!isspace((unsigned char)line[0])) {
            char *colon = strchr(line, ':');
            if (colon) {
                /* if there's a value after the colon, it's a top-level key
                 * (like "name: foo"), not a section header */
                char *after = colon + 1;
                while (*after && isspace((unsigned char)*after)) after++;
                if (*after) {
                    *colon = 0;
                    if (strcmp(line, "name") == 0 && c->exp_name[0] == 0)
                        snprintf(c->exp_name, sizeof(c->exp_name), "%s", after);
                    continue;
                }
                /* no value after colon = section header */
                size_t n = (size_t)(colon - line);
                if (n < sizeof(section)) {
                    memcpy(section, line, n);
                    section[n] = 0;
                }
            }
            continue;
        }
        char *k = line;
        while (*k && isspace((unsigned char)*k)) k++;
        char *colon = strchr(k, ':');
        if (!colon) continue;
        *colon = 0;
        char *v = colon + 1;
        while (*v && isspace((unsigned char)*v)) v++;
        if (strncmp(v, "!!float", 7) == 0) v += 7;
        while (*v && isspace((unsigned char)*v)) v++;

        const profile_t *pr = &c->profile;
        if (pr->total_iter_key && strcmp(k, pr->total_iter_key) == 0 &&
            pr->total_iter_section && strcmp(section, pr->total_iter_section) == 0) {
            c->total_iter = (long)strtod(v, NULL);
        } else if (pr->val_freq_key && strcmp(k, pr->val_freq_key) == 0 &&
                   pr->val_freq_section && strcmp(section, pr->val_freq_section) == 0) {
            c->val_freq = (long)strtod(v, NULL);
        } else if (pr->ckpt_freq_key && strcmp(k, pr->ckpt_freq_key) == 0 &&
                   pr->ckpt_freq_section && strcmp(section, pr->ckpt_freq_section) == 0) {
            c->ckpt_freq = (long)strtod(v, NULL);
        }
    }
    fclose(f);
}

static long total_iters(const ctx_t *c) {
    if (c->total_iter_override > 0) return c->total_iter_override;
    if (c->total_iter > 0) return c->total_iter;
    return c->total_from_log;
}

/* ----------------------------- checkpoint scan -------------------- */

static void scan_checkpoints(ctx_t *c) {
    /* Re-scan when a save was logged, and every CKPT_RESCAN_S otherwise. */
    if (c->last_ckpt_scan && c->last_ckpt_scan_save_iter == c->save_at_iter &&
        c->now - c->last_ckpt_scan < CKPT_RESCAN_S)
        return;
    c->last_ckpt_scan = c->now;
    c->last_ckpt_scan_save_iter = c->save_at_iter;

    const char *prefix = c->profile.ckpt_prefix;
    const char *suffix = c->profile.ckpt_suffix;
    if (!prefix || !c->ckpt_dir[0]) return;
    size_t plen = strlen(prefix);

    DIR *d = opendir(c->ckpt_dir);
    if (!d) return;
    struct dirent *e;
    time_t newest = 0;
    char newest_name[256] = {0};
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        if (strncmp(e->d_name, prefix, plen) != 0) continue;
        size_t ln = strlen(e->d_name);
        char full[PATH_MAX];
        pathf(full, sizeof(full), "%s/%s", c->ckpt_dir, e->d_name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
        if (suffix) {
            size_t slen = strlen(suffix);
            if (ln < slen || strcmp(e->d_name + ln - slen, suffix) != 0) continue;
        } else if (!S_ISDIR(st.st_mode)) {
            continue;   /* no suffix: checkpoints are directories (HF) */
        }
        if (st.st_mtime > newest) {
            newest = st.st_mtime;
            snprintf(newest_name, sizeof(newest_name), "%s", e->d_name);
        }
    }
    closedir(d);
    if (newest_name[0]) {
        c->have_ckpt = 1;
        c->last_ckpt_mtime = newest;
        snprintf(c->last_ckpt_name, sizeof(c->last_ckpt_name), "%s", newest_name);
        /* iter from the name: prefix<iter>.suffix  or  prefix<iter> (dir) */
        c->last_ckpt_iter = strtol(newest_name + plen, NULL, 10);
    }
}

/* ----------------------------- state machine ---------------------- */

static void update_state(ctx_t *c) {
    long total = total_iters(c);
    if (!c->proc_alive) {
        c->state = (total > 0 && c->cur_iter >= total) ? ST_DONE : ST_CRASHED;
        return;
    }
    if (!c->log_exists) { c->state = ST_UNKNOWN; return; }
    if (c->have_val) { c->state = ST_VALIDATING; return; }
    if (c->save_at_iter > 0 && c->cur_iter <= c->save_at_iter) {
        c->state = ST_SAVING; return;
    }
    /* Quiet log but busy GPUs (a long validation, a slow first step) still
     * counts as training. */
    if (c->now - c->last_log_mtime < STALE_SECONDS || run_gpu_busy(c) > 5) {
        c->state = ST_TRAINING; return;
    }
    c->state = ST_IDLE;
}

/* ----------------------------- rendering -------------------------- */

static struct termios orig_term;
static int term_inited = 0;

static void term_enter(void) {
    if (term_inited) return;
    if (tcgetattr(0, &orig_term) == 0) {
        struct termios raw = orig_term;
        raw.c_lflag &= ~(tcflag_t)(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(0, TCSANOW, &raw);
    }
    term_inited = 1;
    fputs(ALT CLEAR HOME HIDE, stdout);
    fflush(stdout);
}

static void term_exit(void) {
    if (!term_inited) return;
    tcsetattr(0, TCSANOW, &orig_term);
    fputs(SHOW RALT, stdout);
    fflush(stdout);
    term_inited = 0;
}

static void on_sigint(int sig) {
    (void)sig;
    term_exit();
    _exit(0);
}

static void on_winch(int sig) {
    (void)sig;
    g_resized = 1;
}

static int win_cols(void) {
    struct winsize w;
    if (ioctl(1, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) return w.ws_col;
    const char *env = getenv("COLUMNS");     /* --once into a pipe or file */
    if (env && atoi(env) > 0) return atoi(env);
    return 80;
}

static void bar(double pct, int w) {
    if (pct < 0) pct = 0;
    if (pct > 1) pct = 1;
    double cells = pct * w;
    int full = (int)cells;
    double frac = cells - full;
    if (full > w) full = w;

    const char *fill_col, *lead_col;
    if (pct >= 0.85)      { fill_col = FG_BGRN; lead_col = FG_BCYN; }
    else if (pct >= 0.5)  { fill_col = FG_BGRN; lead_col = FG_BGRN; }
    else if (pct >= 0.2)  { fill_col = FG_BYEL; lead_col = FG_BYEL; }
    else                  { fill_col = FG_ORANGE; lead_col = FG_BYEL; }

    ps(fill_col);
    for (int i = 0; i < full; i++) ps("\xE2\x96\x88");
    ps(RST);

    if (full < w && frac > 0.0) {
        ps(lead_col);
        int idx = (int)(frac * 8 + 0.5);
        if (idx < 1) idx = 1;
        if (idx > 7) idx = 7;
        static const char *blocks[7] = {
            "\xE2\x96\x8F", "\xE2\x96\x8E", "\xE2\x96\x8D", "\xE2\x96\x8C",
            "\xE2\x96\x8B", "\xE2\x96\x8A", "\xE2\x96\x89",
        };
        ps(blocks[idx - 1]);
        ps(RST);
        full++;
    }
    ps(FG_DKGRAY);
    for (int i = full; i < w; i++) ps("\xE2\x96\x91");
    ps(RST);
}

/* Small bar. A full GPU is what training wants (green); full VRAM is a
 * warning (yellow, then red). */
enum { BAR_LOAD, BAR_FILL };

static void minbar(double pct, int w, int kind) {
    if (pct < 0) pct = 0;
    if (pct > 1) pct = 1;
    int cells = (int)(pct * w + 0.5);
    if (cells > w) cells = w;
    const char *col;
    if (kind == BAR_LOAD)
        col = pct >= 0.7 ? FG_BGRN : pct >= 0.3 ? FG_BYEL : FG_ORANGE;
    else
        col = pct >= 0.9 ? FG_BRED : pct >= 0.75 ? FG_BYEL : FG_BGRN;
    ps(col);
    for (int i = 0; i < cells; i++) ps("\xE2\x96\x93");
    ps(RST FG_DKGRAY);
    for (int i = cells; i < w; i++) ps("\xE2\x96\x91");
    ps(RST);
}

static void human_bytes(long long b, char *dst, size_t n) {
    if (b >= (1LL << 30)) snprintf(dst, n, "%.1fG", b / (double)(1LL << 30));
    else if (b >= (1LL << 20)) snprintf(dst, n, "%.0fM", b / (double)(1LL << 20));
    else if (b >= (1LL << 10)) snprintf(dst, n, "%.0fK", b / (double)(1LL << 10));
    else snprintf(dst, n, "%dB", b > 0 ? (int)b : 0);
}

/* 45s, 12m 03s, 4h 07m, 2d 3h 15m */
static void fmt_dur(long s, char *dst, size_t n) {
    if (s < 0) s = 0;
    if (s < 60) snprintf(dst, n, "%lds", s);
    else if (s < 3600) snprintf(dst, n, "%ldm %02lds", s / 60, s % 60);
    else if (s < 86400) snprintf(dst, n, "%ldh %02ldm", s / 3600, s / 60 % 60);
    else snprintf(dst, n, "%ldd %ldh %02ldm", s / 86400, s / 3600 % 24, s / 60 % 60);
}

static void hr(int cols) {
    ps(FG_DKGRAY);
    for (int i = 0; i < cols; i++) ps("\xE2\x94\x80");
    ps(RST "\n");
}

/* format a metric value with appropriate precision */
static void fmt_metric(double v, char *dst, size_t n) {
    double av = v < 0 ? -v : v;
    if (v == 0) { snprintf(dst, n, "0"); return; }
    if (av != av) { snprintf(dst, n, "nan"); return; }
    if (av >= 10000)      { snprintf(dst, n, "%.0f", v); return; }
    if (av < 0.001)       { snprintf(dst, n, "%.2e", v); return; }
    snprintf(dst, n, av >= 100 ? "%.2f" : av >= 1 ? "%.4f" : "%.6f", v);
    size_t L = strlen(dst);                  /* 0.358000 -> 0.358, 2.0000 -> 2.0 */
    while (L > 3 && dst[L - 1] == '0' && dst[L - 2] != '.') dst[--L] = 0;
}

/* Normal, then amber within 15 °C of the critical limit, red within 5. */
static const char *temp_col(int t, int crit, int fallback) {
    int cr = crit > 0 ? crit : fallback;
    if (t >= cr - 5) return BOLD FG_BRED;
    if (t >= cr - 15) return FG_AMBER;
    return FG_BWHT;
}

/* "label 66°C"; returns the columns it took. */
static int put_temp(const char *label, int t, int crit, int fallback) {
    pf(FG_DKGRAY "%s " RST "%s%d" DEG "C" RST, label, temp_col(t, crit, fallback), t);
    return (int)strlen(label) + 1 + snprintf(NULL, 0, "%d", t) + 2;
}

/* Print s in exactly w columns (ASCII), cut with an ellipsis if longer. */
static void put_fixed(const char *s, int w) {
    int L = (int)strlen(s);
    if (L <= w) { ps(s); pad(w - L); return; }
    pf("%.*s\xE2\x80\xA6", w - 1, s);
}

/* Items printed left to right, wrapped at the screen width. */
typedef struct {
    int col, width, indent;
} flow_t;

static void flow_item(flow_t *f, int w) {
    if (f->col > f->indent && f->col + w > f->width) {
        ps("\n" CLRLN);
        pad(f->indent);
        f->col = f->indent;
    }
    f->col += w;
}

static void render_gpus(const ctx_t *c, int W) {
    if (c->ngpus == 0) {
        pf(CLRLN "  " FG_DKGRAY "(no GPU stats)" RST "\n");
        return;
    }
    /* "  ● gpu0 " + name + load bar + vram bar, both bars share what is left
     * (58 fixed columns, 3 spare for long VRAM sizes like 141.0G/192.0G) */
    const int NAMEW = 18;
    int bars = W - 61;
    if (bars < 12) bars = 12;
    int bl = bars / 2, bv = bars - bl;

    for (int i = 0; i < c->ngpus; i++) {
        const gpu_t *g = &c->gpus[i];
        const char *mark = g->run_procs ? FG_BCYN "\xE2\x97\x8F"
                         : g->other_procs ? FG_AMBER "\xE2\x97\x8F"
                         : FG_DKGRAY "\xE2\x97\x8B";
        pf(CLRLN "  %s" RST " " FG_LTGRAY "gpu%-2d" RST FG_BWHT, mark, i);
        put_fixed(g->name, NAMEW);
        ps(RST "  ");

        ps(FG_DKGRAY "load " RST);
        if (g->busy >= 0) {
            minbar(g->busy / 100.0, bl, BAR_LOAD);
            pf(" " FG_WHT "%3d%%" RST "  ", g->busy);
        } else {
            pad(bl);
            ps(FG_DKGRAY "  n/a" RST "  ");
        }
        char used[16], total[16];
        human_bytes(g->vram_used, used, sizeof(used));
        human_bytes(g->vram_total, total, sizeof(total));
        ps(FG_DKGRAY "vram " RST);
        minbar(g->vram_total > 0 ? (double)g->vram_used / g->vram_total : 0, bv, BAR_FILL);
        pf(" " FG_WHT "%s" FG_DKGRAY "/" FG_SLATE "%s" RST "\n", used, total);

        /* second line: temperatures, power, fan, who uses the card */
        ps(CLRLN);
        pad(9);
        static const char *amd_lbl[T_COUNT] = { "edge", "hotspot", "mem" };
        static const char *nv_lbl[T_COUNT] = { "core", "hotspot", "mem" };
        static const int field[T_COUNT] = { 11, 15, 10 };
        static const int fallback[T_COUNT] = { 100, 110, 100 };
        for (int k = 0; k < T_COUNT; k++) {
            if (g->temp[k] < 0) continue;
            int fb = g->nvidia ? (k == T_MEM ? 95 : 92) : fallback[k];
            int w = put_temp(g->nvidia ? nv_lbl[k] : amd_lbl[k], g->temp[k], g->crit[k], fb);
            pad(field[k] - w);
        }
        if (g->power_w >= 0) {
            char pw[32];
            int w;
            if (g->cap_w > 0) {
                const char *col = g->power_w >= g->cap_w ? FG_AMBER : FG_WHT;
                w = snprintf(pw, sizeof(pw), "%d/%d W", g->power_w, g->cap_w);
                pf("%s%d" FG_DKGRAY "/%d W" RST, col, g->power_w, g->cap_w);
            } else {
                w = snprintf(pw, sizeof(pw), "%d W", g->power_w);
                pf(FG_WHT "%d" FG_DKGRAY " W" RST, g->power_w);
            }
            pad(11 - w);
        }
        if (g->fan_rpm >= 0) {
            int w = snprintf(NULL, 0, "%d rpm", g->fan_rpm);
            pf(FG_WHT "%d" FG_DKGRAY " rpm" RST, g->fan_rpm);
            pad(10 - w);
        } else if (g->fan_pct >= 0) {
            int w = snprintf(NULL, 0, "fan %d%%", g->fan_pct);
            pf(FG_DKGRAY "fan " FG_WHT "%d%%" RST, g->fan_pct);
            pad(10 - w);
        }
        if (c->procs_known) {
            if (g->run_procs && g->other_procs)        /* +N: other jobs too */
                pf(FG_BCYN "this run" FG_AMBER " +%d" RST, g->other_procs);
            else if (g->run_procs)
                ps(FG_BCYN "this run" RST);
            else if (g->other_procs == 1)
                ps(FG_AMBER "other job" RST);
            else if (g->other_procs > 1)
                pf(FG_AMBER "%d other jobs" RST, g->other_procs);
            else
                ps(FG_DKGRAY "free" RST);
        }
        ps("\n");
    }
}

static void render(ctx_t *c) {
    int cols = win_cols();
    int W = cols < 60 ? 60 : cols > 100 ? 100 : cols;
    long total = total_iters(c);
    double rate = iter_rate(c);

    if (g_live) ps(HOME);

    /* Title */
    pf(CLRLN " " FG_BCYN BOLD "\xE2\x96\x88\xE2\x96\x88" RST " "
       BOLD FG_BWHT "train-tui" RST FG_SLATE "  %s" RST
       FG_DKGRAY "  pid %d  profile %s" RST "\n",
       c->exp_name, (int)c->pid, c->profile.name);

    /* Progress */
    double pct = (total > 0 && c->cur_iter > 0) ? (double)c->cur_iter / total : 0;
    if (pct > 1) pct = 1;
    char tot[24];
    if (total > 0) snprintf(tot, sizeof(tot), "%ld", total);
    else snprintf(tot, sizeof(tot), "?");
    int itw = snprintf(NULL, 0, "%ld/%s", c->cur_iter, tot);
    int bar_w = W - 7 - itw - 2 - 8;
    if (bar_w < 10) bar_w = 10;
    pf(CLRLN "  " FG_LTGRAY "iter" RST " " BOLD FG_BWHT "%ld" RST
       FG_DKGRAY "/" RST FG_WHT "%s" RST "  ", c->cur_iter, tot);
    bar(pct, bar_w);
    if (total > 0) pf("  " BOLD FG_BWHT "%5.1f%%" RST "\n", pct * 100.0);
    else ps("\n");

    /* epoch / lr / speed: only what the log actually reports */
    ps(CLRLN " ");
    if (c->have_epoch)
        pf(" " FG_LTGRAY "epoch" RST " " FG_WHT "%g" RST "  ", c->epoch);
    if (c->have_lr)
        pf(" " FG_LTGRAY "lr" RST " " FG_STEEL "%.3e" RST "  ", c->lr);
    if (c->have_titer)
        pf(" " FG_LTGRAY "t/it" RST " " FG_WHT "%.2fs" RST
           FG_DKGRAY " (data %.3fs)" RST "  ", c->t_iter, c->t_data);
    else if (rate >= 1)
        pf(" " FG_LTGRAY "speed" RST " " FG_WHT "%.1f it/s" RST "  ", rate);
    else if (rate > 0)
        pf(" " FG_LTGRAY "speed" RST " " FG_WHT "%.2f s/it" RST "  ", 1.0 / rate);
    ps("\n");

    /* ETA: the log's own, else from the measured rate */
    ps(CLRLN);
    if (c->eta[0]) {
        pf("  " FG_LTGRAY "eta" RST " " BOLD FG_AMBER "%s" RST, c->eta);
    } else if (rate > 0 && total > c->cur_iter && c->state != ST_CRASHED) {
        long secs = (long)((total - c->cur_iter) / rate);
        char d[32], when[32] = {0};
        fmt_dur(secs, d, sizeof(d));
        time_t fin = c->now + secs;
        struct tm tmv;
        if (localtime_r(&fin, &tmv)) strftime(when, sizeof(when), "%a %H:%M", &tmv);
        pf("  " FG_LTGRAY "eta" RST " " BOLD FG_AMBER "%s" RST
           FG_DKGRAY "  (%s)" RST, d, when);
    }
    ps("\n");

    hr(W);

    /* Status + host temperatures */
    pf(CLRLN "  %s  " FG_DKGRAY "[" RST "%s%s" RST FG_DKGRAY "]" RST,
       state_glyph[c->state], state_tag_col[c->state], state_label[c->state]);
    if (c->cpu_temp >= 0) { pad(3); put_temp("cpu", c->cpu_temp, c->cpu_crit, 100); }
    if (c->nvme_temp >= 0) { pad(3); put_temp("nvme", c->nvme_temp, c->nvme_crit, 80); }
    if (c->ngpus > 1) pf(FG_DKGRAY "   %d GPUs" RST, c->ngpus);
    ps("\n");

    render_gpus(c, W);

    hr(W);

    /* Losses / metrics */
    const profile_t *pr = &c->profile;
    if (c->have_losses) {
        ps(CLRLN "  " BOLD FG_BMAG "metrics" RST "  ");
        flow_t f = { 11, W, 11 };
        for (int i = 0; i < pr->num_loss_fields; i++) {
            if (!c->losses[i].have) continue;
            char vs[32];
            fmt_metric(c->losses[i].value, vs, sizeof(vs));
            const char *label = pr->loss_fields[i].label;
            flow_item(&f, (int)(strlen(label) + 1 + strlen(vs) + 2));
            pf(FG_LTGRAY "%s" RST " " FG_WHT "%s" RST "  ", label, vs);
        }
        ps("\n");
    } else {
        ps(CLRLN "  " BOLD FG_BMAG "metrics" RST "  "
           FG_DKGRAY "(waiting for first log line)" RST "\n");
    }

    hr(W);

    /* Validation */
    if (c->have_val || c->have_val_last) {
        pf(CLRLN "  " BOLD FG_BCYN "validation" RST "  ");
        flow_t f = { 14, W, 14 };
        if (c->have_val) {
            ps(FG_BYEL BOLD "RUNNING" RST "  ");
            f.col += 9;
        } else {
            f.col += pf_len(FG_DKGRAY "last @ %ld" RST "  ", c->val_iter);
        }
        for (int i = 0; i < pr->num_val_metrics; i++) {
            const metric_t *m = &c->val_last[i];
            if (!m->have) continue;
            char vs[32], bs[32] = "", best[64] = "";
            fmt_metric(m->value, vs, sizeof(vs));
            if (m->have_best) {
                fmt_metric(m->best, bs, sizeof(bs));
                snprintf(best, sizeof(best), " (best %s @ %ld)", bs, m->best_iter);
            }
            const char *label = pr->val_metrics[i].label;
            flow_item(&f, (int)(strlen(label) + 1 + strlen(vs) + strlen(best) + 2));
            pf(FG_LTGRAY "%s" RST " " BOLD FG_WHT "%s" RST FG_DKGRAY "%s" RST "  ",
               label, vs, best);
        }
        if (c->val_freq > 0) {
            char every[32];
            int w = snprintf(every, sizeof(every), "(every %ld)", c->val_freq);
            flow_item(&f, w);
            pf(FG_DKGRAY "%s" RST, every);
        }
        ps("\n");
    } else {
        ps(CLRLN "  " BOLD FG_BCYN "validation" RST "  " FG_DKGRAY "(none yet)" RST "\n");
    }

    /* Checkpoint */
    if (c->have_ckpt) {
        char when[32] = {0}, ago[32];
        struct tm tmv;
        if (localtime_r(&c->last_ckpt_mtime, &tmv))
            strftime(when, sizeof(when),
                     c->now - c->last_ckpt_mtime > 20 * 3600 ? "%b %d %H:%M" : "%H:%M", &tmv);
        fmt_dur((long)(c->now - c->last_ckpt_mtime), ago, sizeof(ago));
        pf(CLRLN "  " BOLD FG_BMAG "checkpoint" RST "  " FG_WHT "%s" RST, c->last_ckpt_name);
        if (c->last_ckpt_iter > 0)
            pf("  " FG_DKGRAY "iter" RST " " BOLD FG_WHT "%ld" RST, c->last_ckpt_iter);
        pf("  " FG_SLATE "%s" RST FG_DKGRAY " (%s ago)" RST, when, ago);
        if (c->ckpt_freq > 0) pf(FG_DKGRAY "  every %ld" RST, c->ckpt_freq);
        ps("\n");
    } else {
        ps(CLRLN "  " BOLD FG_BMAG "checkpoint" RST "  " FG_DKGRAY "(none found)" RST "\n");
    }

    hr(W);

    /* Footer */
    char mtime[32] = {0}, ago[32];
    struct tm tmv;
    if (c->last_log_mtime && localtime_r(&c->last_log_mtime, &tmv))
        strftime(mtime, sizeof(mtime), "%H:%M:%S", &tmv);
    fmt_dur((long)(c->now - c->last_log_mtime), ago, sizeof(ago));
    pf(CLRLN "  " FG_DKGRAY "proc" RST " %s  " FG_DKGRAY "log" RST " %s",
       c->proc_alive ? FG_BGRN "alive" RST
                     : c->state == ST_DONE ? FG_BGRN "exited" RST : FG_BRED "DEAD" RST,
       c->log_exists ? FG_BGRN "ok" RST : FG_BRED "missing" RST);
    if (mtime[0])
        pf("  " FG_DKGRAY "last write" RST " %s" FG_DKGRAY " (%s ago)" RST, mtime, ago);
    ps("\n");

    if (g_live)
        pf(CLRLN "  " FG_DKGRAY "q quit  r refresh  (auto %dms)" RST "\n" ED, g_refresh_ms);
}

static void draw(ctx_t *c) {
    char *buf = NULL;
    size_t len = 0;
    g_out = open_memstream(&buf, &len);
    if (!g_out) die("out of memory");
    render(c);
    fclose(g_out);
    emit_frame(buf, len);
    free(buf);
}

/* ----------------------------- find log --------------------------- */

/* BasicSR names its log experiments/<exp>/train_<exp>_<timestamp>.log */
static int find_log(const ctx_t *c, char *dst, size_t n) {
    char dir[PATH_MAX], prefix[300];
    pathf(dir, sizeof(dir), "%s/experiments/%s", c->project_root, c->exp_name);
    pathf(prefix, sizeof(prefix), "train_%s_", c->exp_name);
    size_t plen = strlen(prefix);

    DIR *d = opendir(dir);
    if (!d) return -1;
    time_t best = 0;
    char best_full[PATH_MAX] = {0};
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, prefix, plen) != 0) continue;
        size_t ln = strlen(e->d_name);
        if (ln < 5 || strcmp(e->d_name + ln - 4, ".log") != 0) continue;
        char full[PATH_MAX];
        pathf(full, sizeof(full), "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
        if (st.st_mtime > best) {
            best = st.st_mtime;
            pathf(best_full, sizeof(best_full), "%s", full);
        }
    }
    closedir(d);
    if (!best_full[0]) return -1;
    pathf(dst, n, "%s", best_full);
    return 0;
}

/* ----------------------------- custom profile loader -------------- */

static int profile_by_name(const char *s) {
    for (int i = 0; i < PROF_CUSTOM; i++)
        if (strcmp(profiles[i].name, s) == 0) return i;
    return -1;
}

/* Label from a mark: "\"loss\":" -> "loss", "l_g_pix:" -> "l_g_pix". */
static void default_label(const char *mark, char *dst, size_t n) {
    size_t j = 0;
    for (const char *p = mark; *p && j + 1 < n; p++)
        if (!strchr("\"':=(", *p) && !isspace((unsigned char)*p)) dst[j++] = *p;
    dst[j] = 0;
}

static int split_list(char *v, char items[][MAX_LOSS_MARK], int max) {
    int n = 0;
    char *save = NULL;
    for (char *tok = strtok_r(v, ",", &save); tok && n < max; tok = strtok_r(NULL, ",", &save))
        copy_str(items[n++], MAX_LOSS_MARK, trim(tok));
    return n;
}

/* Load a profile file. Format: key = value lines, '#' comments.
 * Keys: base (must come first), epoch_mark, iter_mark, lr_mark, eta_mark,
 *       time_mark, save_mark, val_mark, total_mark, stamp_mark,
 *       iter_commas (0/1), ckpt_prefix, ckpt_suffix, has_config,
 *       total_iter_key/_section, val_freq_key/_section, ckpt_freq_key/_section,
 *       loss_fields, loss_labels, val_metrics, val_metric_labels,
 *       val_metric_track_best (per metric: 0, 1/log, min, max). */
static int load_custom_profile(ctx_t *c, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    profile_t *pr = &c->profile;
    memset(pr, 0, sizeof(*pr));
    pr->name = "custom";

    char line[1024];
    int lineno = 0, keys = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        line[strcspn(line, "\r\n")] = 0;
        char *s = trim(line);
        if (!*s || *s == '#') continue;
        char *eq = strchr(s, '=');
        if (!eq) continue;
        *eq = 0;
        char *key = trim(s);
        char *v = trim(eq + 1);
        char items[MAX_LOSS_FIELDS][MAX_LOSS_MARK];

        if (strcmp(key, "base") == 0) {
            int id = profile_by_name(v);
            if (id < 0 || keys > 0) {
                fprintf(stderr, "train-tui: %s:%d: %s\n", path, lineno,
                        id < 0 ? "unknown base profile" : "base must be the first key");
                fclose(f);
                return -1;
            }
            *pr = profiles[id];
            char *name = malloc(strlen(v) + 10);
            if (name) { sprintf(name, "custom/%s", v); pr->name = name; }
            keys++;
            continue;
        }
        keys++;
        if (strcmp(key, "epoch_mark") == 0) pr->epoch_mark = strdup(v);
        else if (strcmp(key, "iter_mark") == 0) pr->iter_mark = strdup(v);
        else if (strcmp(key, "lr_mark") == 0) pr->lr_mark = strdup(v);
        else if (strcmp(key, "eta_mark") == 0) pr->eta_mark = strdup(v);
        else if (strcmp(key, "time_mark") == 0) pr->time_mark = strdup(v);
        else if (strcmp(key, "save_mark") == 0) pr->save_mark = strdup(v);
        else if (strcmp(key, "val_mark") == 0) pr->val_mark = strdup(v);
        else if (strcmp(key, "total_mark") == 0) pr->total_mark = strdup(v);
        else if (strcmp(key, "stamp_mark") == 0) pr->stamp_mark = strdup(v);
        else if (strcmp(key, "iter_commas") == 0) pr->iter_has_commas = atoi(v);
        else if (strcmp(key, "ckpt_prefix") == 0) pr->ckpt_prefix = strdup(v);
        else if (strcmp(key, "ckpt_suffix") == 0) pr->ckpt_suffix = strdup(v);
        else if (strcmp(key, "total_iter_key") == 0) pr->total_iter_key = strdup(v);
        else if (strcmp(key, "total_iter_section") == 0) pr->total_iter_section = strdup(v);
        else if (strcmp(key, "val_freq_key") == 0) pr->val_freq_key = strdup(v);
        else if (strcmp(key, "val_freq_section") == 0) pr->val_freq_section = strdup(v);
        else if (strcmp(key, "ckpt_freq_key") == 0) pr->ckpt_freq_key = strdup(v);
        else if (strcmp(key, "ckpt_freq_section") == 0) pr->ckpt_freq_section = strdup(v);
        else if (strcmp(key, "has_config") == 0) pr->has_config = atoi(v);
        else if (strcmp(key, "loss_fields") == 0) {
            int n = split_list(v, items, MAX_LOSS_FIELDS);
            memset(pr->loss_fields, 0, sizeof(pr->loss_fields));
            for (int i = 0; i < n; i++)
                copy_str(pr->loss_fields[i].mark, MAX_LOSS_MARK, items[i]);
            pr->num_loss_fields = n;
        } else if (strcmp(key, "loss_labels") == 0) {
            int n = split_list(v, items, MAX_LOSS_FIELDS);
            for (int i = 0; i < n; i++)
                copy_str(pr->loss_fields[i].label, MAX_LOSS_LABEL, items[i]);
        } else if (strcmp(key, "val_metrics") == 0) {
            int n = split_list(v, items, MAX_LOSS_FIELDS);
            memset(pr->val_metrics, 0, sizeof(pr->val_metrics));
            for (int i = 0; i < n; i++)
                copy_str(pr->val_metrics[i].mark, MAX_LOSS_MARK, items[i]);
            pr->num_val_metrics = n;
        } else if (strcmp(key, "val_metric_labels") == 0) {
            int n = split_list(v, items, MAX_LOSS_FIELDS);
            for (int i = 0; i < n; i++)
                copy_str(pr->val_metrics[i].label, MAX_LOSS_LABEL, items[i]);
        } else if (strcmp(key, "val_metric_track_best") == 0) {
            int n = split_list(v, items, MAX_LOSS_FIELDS);
            for (int i = 0; i < n; i++) {
                const char *b = items[i];
                pr->val_metrics[i].best =
                    !strcmp(b, "min") ? BEST_MIN : !strcmp(b, "max") ? BEST_MAX
                    : (!strcmp(b, "1") || !strcmp(b, "log")) ? BEST_LOG : BEST_NONE;
            }
        } else {
            fprintf(stderr, "train-tui: %s:%d: unknown key '%s' (ignored)\n",
                    path, lineno, key);
        }
    }
    fclose(f);
    for (int i = 0; i < pr->num_loss_fields; i++)
        if (!pr->loss_fields[i].label[0])
            default_label(pr->loss_fields[i].mark, pr->loss_fields[i].label, MAX_LOSS_LABEL);
    for (int i = 0; i < pr->num_val_metrics; i++)
        if (!pr->val_metrics[i].label[0])
            default_label(pr->val_metrics[i].mark, pr->val_metrics[i].label, MAX_LOSS_LABEL);
    c->profile_id = PROF_CUSTOM;
    return 0;
}

/* ----------------------------- usage ------------------------------ */

static void usage(const char *prog) {
    fprintf(stderr,
        "train-tui %s: lightweight terminal UI for monitoring AI training\n"
        "\n"
        "usage: %s [options] project_root exp_name config_name pid [log_file]\n"
        "       %s -a [options]\n"
        "\n"
        "  (config_name may be '-' if your project has no parseable config;\n"
        "   use -t to set total_iter in that case)\n"
        "\n"
        "options:\n"
        "  -a              auto-detect a running training process (scan /proc)\n"
        "  -p <profile>    basicsr (default), lightning, hf or jsonl\n"
        "  -c <file>       profile file (see sample.profile)\n"
        "  -t <total>      total iterations (overrides config and log)\n"
        "  -k <dir>        checkpoint directory (default: next to the log)\n"
        "  -g <backend>    GPUs: auto (every card found), amd, nvidia, none\n"
        "  -i <seconds>    refresh interval (default 1; raise it on a slow link)\n"
        "  --once          print one frame and exit (no escape codes when piped)\n"
        "  -V, --version   print the version\n"
        "  -h, --help      this help\n"
        "\n"
        "positional args:\n"
        "  project_root    path to the training project root\n"
        "  exp_name        experiment name (subdir of experiments/ for BasicSR)\n"
        "  config_name     YAML config: absolute, relative to project_root, or in\n"
        "                  project_root/options/train/  ('-' = none)\n"
        "  pid             PID of the training process (its process group counts as\n"
        "                  the run when matching GPUs to processes)\n"
        "  log_file        log to follow (BasicSR: found in experiments/<exp>/)\n"
        "\n"
        "environment:\n"
        "  NO_COLOR        no colors\n"
        "  COLUMNS         width for --once when the output is not a terminal\n"
        "\n",
        TRAIN_TUI_VERSION, prog, prog);
    exit(2);
}

/* ----------------------------- auto-detect ------------------------ */

/* Scan /proc for a python training process (train.py, trainer, etc.).
 * Returns the PID, fills cwd and the --config / -opt argument. */
static pid_t autodetect_train(char *project_root, size_t pr_n,
                              char *cfg_name, size_t cfg_n) {
    DIR *d = opendir("/proc");
    if (!d) return -1;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!isdigit((unsigned char)e->d_name[0])) continue;
        pid_t pid = atoi(e->d_name);
        if (pid <= 0 || pid == getpid()) continue;

        char cmdpath[64];
        snprintf(cmdpath, sizeof(cmdpath), "/proc/%d/cmdline", (int)pid);
        int fd = open(cmdpath, O_RDONLY);
        if (fd < 0) continue;
        char cmdline[2048] = {0};
        ssize_t n = read(fd, cmdline, sizeof(cmdline) - 1);
        close(fd);
        if (n <= 0) continue;
        cmdline[n] = 0;
        /* null separators to spaces so strstr works across args */
        for (ssize_t i = 0; i < n; i++)
            if (cmdline[i] == '\0') cmdline[i] = ' ';

        if (!strstr(cmdline, "python")) continue;
        if (!strstr(cmdline, "train.py") && !strstr(cmdline, "trainer") &&
            !strstr(cmdline, "train_"))
            continue;

        char cwdlink[64];
        snprintf(cwdlink, sizeof(cwdlink), "/proc/%d/cwd", (int)pid);
        char cwd[1024] = {0};
        ssize_t cl = readlink(cwdlink, cwd, sizeof(cwd) - 1);
        if (cl <= 0) continue;
        cwd[cl] = 0;

        /* config: "-opt <file>" (BasicSR) or "--config <file>" */
        char cfg[512] = {0};
        char *opt = strstr(cmdline, "-opt ");
        size_t skip = 5;
        if (!opt) { opt = strstr(cmdline, "--config "); skip = 9; }
        if (opt) {
            char *p = opt + skip;
            while (*p == ' ') p++;
            size_t len = strcspn(p, " ");
            if (len >= sizeof(cfg)) len = sizeof(cfg) - 1;
            memcpy(cfg, p, len);
            cfg[len] = 0;
        }

        snprintf(project_root, pr_n, "%s", cwd);
        if (cfg[0]) snprintf(cfg_name, cfg_n, "%s", cfg);
        closedir(d);
        return pid;
    }
    closedir(d);
    return -1;
}

static void resolve_cfg(ctx_t *c) {
    c->cfg_path[0] = 0;
    if (!c->cfg_name[0] || strcmp(c->cfg_name, "-") == 0) return;
    if (c->cfg_name[0] == '/') {
        pathf(c->cfg_path, sizeof(c->cfg_path), "%s", c->cfg_name);
        return;
    }
    char p[PATH_MAX];
    pathf(p, sizeof(p), "%s/%s", c->project_root, c->cfg_name);
    if (access(p, R_OK) != 0)
        pathf(p, sizeof(p), "%s/options/train/%s", c->project_root, c->cfg_name);
    pathf(c->cfg_path, sizeof(c->cfg_path), "%s", p);
}

static int is_dir(const char *p) {
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

/* ----------------------------- main ------------------------------- */

int main(int argc, char **argv) {
    static ctx_t c;
    c.profile_id = PROF_BASICSR;
    c.gpu_backend = GPU_AUTO;

    const char *custom_profile_path = NULL;
    const char *ckpt_dir_opt = NULL;
    int auto_detect = 0, once = 0;

    int argi = 1;
    while (argi < argc && argv[argi][0] == '-' && argv[argi][1]) {
        const char *a = argv[argi];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage(argv[0]);
        } else if (strcmp(a, "-V") == 0 || strcmp(a, "--version") == 0) {
            printf("train-tui %s\n", TRAIN_TUI_VERSION);
            return 0;
        } else if (strcmp(a, "-a") == 0 || strcmp(a, "--auto") == 0) {
            auto_detect = 1;
        } else if (strcmp(a, "--once") == 0) {
            once = 1;
        } else if (strcmp(a, "-p") == 0 && argi + 1 < argc) {
            int id = profile_by_name(argv[++argi]);
            if (id < 0) { fprintf(stderr, "unknown profile: %s\n", argv[argi]); return 2; }
            c.profile_id = (profile_id_t)id;
        } else if (strcmp(a, "-c") == 0 && argi + 1 < argc) {
            custom_profile_path = argv[++argi];
        } else if (strcmp(a, "-t") == 0 && argi + 1 < argc) {
            c.total_iter_override = atol(argv[++argi]);
        } else if (strcmp(a, "-i") == 0 && argi + 1 < argc) {
            double sec = atof(argv[++argi]);
            if (sec < 0.2) { fprintf(stderr, "train-tui: -i needs at least 0.2 seconds\n"); return 2; }
            g_refresh_ms = (int)(sec * 1000 + 0.5);
        } else if (strcmp(a, "-k") == 0 && argi + 1 < argc) {
            ckpt_dir_opt = argv[++argi];
        } else if (strcmp(a, "-g") == 0 && argi + 1 < argc) {
            const char *g = argv[++argi];
            if (strcmp(g, "amd") == 0) c.gpu_backend = GPU_AMD;
            else if (strcmp(g, "nvidia") == 0) c.gpu_backend = GPU_NVIDIA;
            else if (strcmp(g, "none") == 0) c.gpu_backend = GPU_NONE;
            else if (strcmp(g, "auto") == 0) c.gpu_backend = GPU_AUTO;
            else { fprintf(stderr, "unknown GPU backend: %s\n", g); return 2; }
        } else if (strcmp(a, "--") == 0) {
            argi++;
            break;
        } else {
            fprintf(stderr, "unknown option: %s (try -h)\n", a);
            return 2;
        }
        argi++;
    }

    /* positional args */
    for (int pos = 0; argi < argc; argi++, pos++) {
        switch (pos) {
        case 0: snprintf(c.project_root, sizeof(c.project_root), "%s", argv[argi]); break;
        case 1: snprintf(c.exp_name, sizeof(c.exp_name), "%s", argv[argi]); break;
        case 2: snprintf(c.cfg_name, sizeof(c.cfg_name), "%s", argv[argi]); break;
        case 3: c.pid = (pid_t)atoi(argv[argi]); break;
        case 4: snprintf(c.log_path, sizeof(c.log_path), "%s", argv[argi]); break;
        default: fprintf(stderr, "train-tui: too many arguments (try -h)\n"); return 2;
        }
    }

    if (auto_detect || (c.project_root[0] == 0 && c.pid <= 0)) {
        pid_t found = autodetect_train(c.project_root, sizeof(c.project_root),
                                       c.cfg_name, sizeof(c.cfg_name));
        if (found <= 0) {
            fprintf(stderr,
                "train-tui: no training process found.\n\n"
                "  Use -a to auto-detect, or provide args explicitly:\n"
                "    %s -p basicsr /path/to/project experiment config.yml 12345\n\n"
                "  Run '%s -h' for full help.\n",
                argv[0], argv[0]);
            return 2;
        }
        c.pid = found;
        fprintf(stderr, "train-tui: auto-detected pid %d in %s\n",
                (int)found, c.project_root);
    }

    if (custom_profile_path) {
        if (load_custom_profile(&c, custom_profile_path) != 0)
            die("could not load the profile file");
    } else {
        c.profile = profiles[c.profile_id];
    }

    /* config: total_iter etc. (and the experiment name, if not given) */
    resolve_cfg(&c);
    parse_config(&c);

    if (c.project_root[0] == 0 || c.exp_name[0] == 0 || c.pid <= 0) {
        fprintf(stderr,
            "train-tui: missing required arguments.\n\n"
            "  Use -a to auto-detect a running training process, or provide:\n"
            "    %s -p basicsr /path/to/project my_experiment config.yml 12345\n\n"
            "  Run '%s -h' for full help.\n",
            argv[0], argv[0]);
        return 2;
    }

    if (c.log_path[0] == 0 && find_log(&c, c.log_path, sizeof(c.log_path)) != 0) {
        fprintf(stderr, "train-tui: no log file found for experiment '%s'\n", c.exp_name);
        return 1;
    }

    /* checkpoints: -k, else <log dir>/models (BasicSR), else the log's dir */
    if (ckpt_dir_opt) {
        pathf(c.ckpt_dir, sizeof(c.ckpt_dir), "%s", ckpt_dir_opt);
    } else {
        char log_dir[PATH_MAX];
        pathf(log_dir, sizeof(log_dir), "%s", c.log_path);
        char *slash = strrchr(log_dir, '/');
        if (slash) *slash = 0;
        else pathf(log_dir, sizeof(log_dir), ".");
        pathf(c.ckpt_dir, sizeof(c.ckpt_dir), "%s/models", log_dir);
        if (!is_dir(c.ckpt_dir)) pathf(c.ckpt_dir, sizeof(c.ckpt_dir), "%s", log_dir);
    }

    const char *root = getenv("TRAIN_TUI_SYSFS");
    if (root) snprintf(g_sysroot, sizeof(g_sysroot), "%s", root);
    g_no_color = getenv("NO_COLOR") && *getenv("NO_COLOR");
    gpu_discover(&c);
    host_discover(&c);

    /* "ssh host train-tui ..." without -t: one plain frame, not an endless
     * stream of screen updates into a pipe. */
    if (!once && !isatty(1)) {
        fprintf(stderr, "train-tui: not a terminal, printing one frame "
                        "(use ssh -t for the live view)\n");
        once = 1;
    }

    if (once) {
        g_live = 0;
        g_strip_all = !isatty(1);
        c.now = time(NULL);
        check_proc(&c);
        read_gpus(&c);
        read_host(&c);
        tail_log(&c);
        scan_checkpoints(&c);
        update_state(&c);
        draw(&c);
        return 0;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_sigint;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);
    sa.sa_handler = on_winch;      /* no SA_RESTART: select() wakes up */
    sigaction(SIGWINCH, &sa, NULL);

    term_enter();
    atexit(term_exit);

    int quit = 0, keys_open = 1;
    while (!quit) {
        c.now = time(NULL);
        check_proc(&c);
        read_gpus(&c);
        read_host(&c);
        tail_log(&c);
        scan_checkpoints(&c);
        update_state(&c);
        if (g_resized) {               /* a resized window: start clean */
            g_resized = 0;
            fputs(CLEAR, stdout);
        }
        draw(&c);

        struct timeval tv;
        tv.tv_sec = g_refresh_ms / 1000;
        tv.tv_usec = (g_refresh_ms % 1000) * 1000;
        fd_set fds;
        FD_ZERO(&fds);
        if (keys_open) FD_SET(0, &fds);
        if (select(keys_open ? 1 : 0, &fds, NULL, NULL, &tv) > 0 && FD_ISSET(0, &fds)) {
            char keys[32];
            ssize_t n = read(0, keys, sizeof(keys));
            if (n <= 0) {
                keys_open = 0;         /* stdin closed: keep drawing, stop reading */
            } else if (keys[0] == 27) {
                quit = n == 1;         /* a lone Esc; arrow keys are Esc sequences */
            } else {
                for (ssize_t i = 0; i < n; i++)
                    if (keys[i] == 'q' || keys[i] == 'Q' || keys[i] == 3) quit = 1;
            }
        }
    }

    term_exit();
    return 0;
}
