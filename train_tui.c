/*
 * train-tui: a lightweight, dependency-free terminal UI for monitoring
 * (almost) any deep-learning training run. Pure C, no ncurses, no Python.
 *
 * It supports pluggable "profiles" that describe:
 *   - how to parse the log (which substrings mark epoch, iter, losses, etc.)
 *   - how to find total_iter (config file key, or CLI override)
 *   - how checkpoints are named (filename prefix + suffix + iter extraction)
 *
 * Built-in profiles:
 *   basicsr  - BasicSR / Real-ESRGAN / HAT / SwinIR (GAN SR)
 *   lightning- PyTorch Lightning progress bar logs
 *   hf       - HuggingFace Trainer logs
 *   custom   - driven by a .train-tui.profile file
 *
 * GPU backends (auto-detected):
 *   amd      - sysfs /sys/class/drm/card0/device/... (amdgpu)
 *   nvidia   - nvidia-smi --query-gpu=... subprocess
 *   none     - no GPU stats (CPU-only training)
 *
 * Read-only: never writes to the training tree, never signals the process,
 * never opens model files for writing.
 *
 * Build:  make  (or: cc -O2 -Wall -o train_tui train_tui.c)
 * Run:    ./train_tui -p basicsr /path/to/project exp_name config.yml pid [log]
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <dirent.h>

/* ----------------------------- tuning ------------------------------ */

#define REFRESH_MS         1000
#define STALE_SECONDS      600    /* must be > longest gap between log lines
                                    * (BasicSR logs every 100 iters ~190s,
                                    * validation blocks can take 15+ min) */

#define MAX_LOSS_FIELDS    16   /* max tracked loss/metric fields per profile */
#define MAX_LOSS_LABEL     32
#define MAX_LOSS_MARK      64   /* max length of the "mark" substring to find */

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

#define FG_BLK ESC "[30m"
#define FG_RED ESC "[31m"
#define FG_GRN ESC "[32m"
#define FG_YEL ESC "[33m"
#define FG_BLU ESC "[34m"
#define FG_MAG ESC "[35m"
#define FG_CYN ESC "[36m"
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

/* ----------------------------- profiles ---------------------------- */

typedef enum {
    PROF_BASICSR = 0,
    PROF_LIGHTNING,
    PROF_HF,
    PROF_CUSTOM,
    PROF_COUNT,
} profile_id_t;

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
    const char *val_mark;    /* e.g. "Validation"  (start of validation block) */

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
        int track_best;             /* if 1, look for "Best: <val> @ <iter>" */
    } val_metrics[MAX_LOSS_FIELDS];

    /* checkpoint filename pattern */
    const char *ckpt_prefix;   /* "net_g_" */
    const char *ckpt_suffix;   /* ".pth"   */

    /* config keys for total_iter (section:key, or just key) */
    const char *total_iter_key;
    const char *total_iter_section;
    const char *val_freq_key;
    const char *val_freq_section;
    const char *ckpt_freq_key;
    const char *ckpt_freq_section;

    /* config format: 0 = simple YAML (section: key: value), 1 = none/CLI */
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
        {"psnr:", "psnr", 1},
        {"ssim:", "ssim", 1},
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
     * We parse "Epoch ", the tqdm "X/Y", "it/s" for speed, "loss=..." */
    .epoch_mark = "Epoch ",
    .iter_mark  = NULL,   /* iter is embedded in the tqdm bar; skip */
    .lr_mark    = "lr=",
    .eta_mark   = NULL,
    .time_mark  = NULL,
    .save_mark  = NULL,
    .val_mark   = "Validation",
    .iter_has_commas = 0,
    .num_loss_fields = 4,
    .loss_fields = {
        {"loss=",    "loss"},
        {"v_num=",   "v_num"},
        {"val_loss=", "val_loss"},
        {"val_acc=",  "val_acc"},
    },
    .num_val_metrics = 0,
    .ckpt_prefix = "epoch=",
    .ckpt_suffix = ".ckpt",
    .has_config = 0,  /* Lightning uses argparse, not a yml we can parse */
},
[PROF_HF] = {
    .name = "hf",
    /* HuggingFace Trainer logs like:
     *   "{'loss': 0.234, 'learning_rate': 5e-05, 'epoch': 1.23, 'step': 450}"
     *   "{'eval_loss': 0.456, 'eval_runtime': 12.3, 'epoch': 1.23, 'step': 450}"
     */
    .epoch_mark = "'epoch':",
    .iter_mark  = "'step':",
    .lr_mark    = "'learning_rate':",
    .eta_mark   = NULL,
    .time_mark  = NULL,
    .save_mark  = "Saving model checkpoint",
    .val_mark   = "'eval_loss'",
    .iter_has_commas = 0,
    .num_loss_fields = 2,
    .loss_fields = {
        {"'loss':", "loss"},
        {"'grad_norm':", "grad_norm"},
    },
    .num_val_metrics = 3,
    .val_metrics = {
        {"'eval_loss':", "eval_loss", 0},
        {"'eval_accuracy':", "eval_acc", 0},
        {"'eval_f1':", "eval_f1", 0},
    },
    .ckpt_prefix = "checkpoint-",
    .ckpt_suffix = NULL,  /* HF uses directories named checkpoint-N */
    .has_config = 0,  /* HF config is python, not parseable here */
},
};

/* ----------------------------- GPU backend ------------------------- */

typedef enum {
    GPU_AUTO = 0,
    GPU_AMD,
    GPU_NVIDIA,
    GPU_NONE,
} gpu_backend_t;

/* ----------------------------- state ------------------------------- */

typedef enum {
    ST_TRAINING = 0,
    ST_VALIDATING,
    ST_SAVING,
    ST_IDLE,
    ST_CRASHED,
    ST_UNKNOWN,
} gpu_state_t;

static const char *state_label[] = {
    "TRAINING", "VALIDATING", "SAVING", "IDLE", "CRASHED", "UNKNOWN",
};
static const char *state_glyph[] = {
    FG_BGRN "\xE2\x97\x8F" RST,
    FG_BYEL "\xE2\x97\x8F" RST,
    FG_BBLU "\xE2\x97\x8F" RST,
    FG_SLATE "\xE2\x97\x8B" RST,
    FG_BRED "\xE2\x9C\x96" RST,
    FG_YEL  "?" RST,
};
static const char *state_tag_col[] = {
    FG_BGRN, FG_BYEL, FG_BBLU, FG_SLATE, FG_BRED, FG_YEL,
};

/* generic metric value (loss or validation metric) */
typedef struct {
    int have;
    double value;
    double best;       /* if track_best */
    long best_iter;
} metric_t;

typedef struct {
    /* user config */
    char project_root[1024];
    char exp_name[256];
    char cfg_name[256];
    char log_path[2048];
    char cfg_path[1600];
    char ckpt_dir[1792];
    pid_t pid;
    long total_iter_override;   /* >0 if set via CLI */
    gpu_backend_t gpu_backend;

    /* profile */
    profile_id_t profile_id;
    profile_t profile;         /* copy (so custom can mutate) */

    /* derived */
    long total_iter;
    long val_freq;
    long ckpt_freq;

    /* latest parsed log */
    long cur_iter;
    long cur_epoch;
    double lr;
    char eta[64];
    double t_iter, t_data;
    metric_t losses[MAX_LOSS_FIELDS];
    int have_losses;

    /* validation (current refresh) */
    int have_val;
    int in_validation;
    metric_t val_results[MAX_LOSS_FIELDS];

    /* last known validation (persistent) */
    int have_val_last;
    metric_t val_last[MAX_LOSS_FIELDS];

    /* checkpoint */
    int have_ckpt;
    char last_ckpt_name[256];
    time_t last_ckpt_mtime;
    long last_ckpt_iter;
    long last_ckpt_scan_save_iter; /* save_at_iter when we last scanned */

    /* GPU */
    int gpu_busy, mem_busy;
    long vram_used, vram_total;
    int gpu_temp, fan_rpm, power_watts;
    int have_hwmon;
    char hwmon_path[1024];

    /* state machine */
    gpu_state_t state;
    time_t last_log_mtime;
    time_t last_log_line_ts;
    time_t last_seen_iter_change;
    long prev_iter_for_stale;
    long save_at_iter;
    int proc_alive;
    int log_exists;

    /* incremental log reading: track file offset so we only read new bytes */
    off_t log_offset;
    int log_initialized;

    time_t now;
} ctx_t;

/* ----------------------------- helpers ---------------------------- */

static void die(const char *msg) {
    fprintf(stderr, "train-tui: %s\n", msg);
    exit(1);
}

static long read_int_file(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    char buf[64];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = 0;
    return strtol(buf, NULL, 10);
}

static int read_file_str(const char *path, char *out, size_t n) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t r = read(fd, out, n - 1);
    close(fd);
    if (r <= 0) return -1;
    out[r] = 0;
    for (char *p = out; *p; p++) {
        if (isspace((unsigned char)*p)) { *p = 0; break; }
    }
    return 0;
}

/* ----------------------------- GPU: AMD sysfs --------------------- */

static char g_amd_card_path[256] = {0};

static void amd_discover_card(void) {
    if (g_amd_card_path[0]) return;
    for (int i = 0; i < 8; i++) {
        char p[256];
        snprintf(p, sizeof(p), "/sys/class/drm/card%d/device/gpu_busy_percent", i);
        if (access(p, R_OK) == 0) {
            snprintf(g_amd_card_path, sizeof(g_amd_card_path), "/sys/class/drm/card%d/device", i);
            return;
        }
    }
    snprintf(g_amd_card_path, sizeof(g_amd_card_path), "/sys/class/drm/card0/device");
}

static int amd_discover_hwmon(ctx_t *c) {
    amd_discover_card();
    char base[512];
    snprintf(base, sizeof(base), "%s/hwmon", g_amd_card_path);
    for (int i = 0; i < 8; i++) {
        char p[1024];
        snprintf(p, sizeof(p), "%s/hwmon%d/name", base, i);
        char name[64] = {0};
        if (read_file_str(p, name, sizeof(name)) == 0) {
            if (strcmp(name, "amdgpu") == 0) {
                snprintf(c->hwmon_path, sizeof(c->hwmon_path),
                         "%s/hwmon%d", base, i);
                c->have_hwmon = 1;
                return 0;
            }
        }
    }
    for (int i = 0; i < 8; i++) {
        char p[1024];
        snprintf(p, sizeof(p), "%s/hwmon%d/temp1_input", base, i);
        if (access(p, R_OK) == 0) {
            snprintf(c->hwmon_path, sizeof(c->hwmon_path),
                     "%s/hwmon%d", base, i);
            c->have_hwmon = 1;
            return 0;
        }
    }
    c->have_hwmon = 0;
    return -1;
}

static void read_gpu_amd(ctx_t *c) {
    amd_discover_card();
    char p[512];
    snprintf(p, sizeof(p), "%s/gpu_busy_percent", g_amd_card_path);
    c->gpu_busy  = (int)read_int_file(p);
    snprintf(p, sizeof(p), "%s/mem_busy_percent", g_amd_card_path);
    c->mem_busy  = (int)read_int_file(p);
    snprintf(p, sizeof(p), "%s/mem_info_vram_used", g_amd_card_path);
    c->vram_used = read_int_file(p);
    snprintf(p, sizeof(p), "%s/mem_info_vram_total", g_amd_card_path);
    c->vram_total = read_int_file(p);
    if (c->vram_used < 0) c->vram_used = 0;
    if (c->vram_total < 0) c->vram_total = 0;
    if (!c->have_hwmon) amd_discover_hwmon(c);
    if (c->have_hwmon) {
        char hp[1280];
        snprintf(hp, sizeof(hp), "%s/temp1_input", c->hwmon_path);
        long t = read_int_file(hp);
        c->gpu_temp = (t > 0) ? (int)(t / 1000) : -1;
        snprintf(hp, sizeof(hp), "%s/fan1_input", c->hwmon_path);
        c->fan_rpm = (int)read_int_file(hp);
        snprintf(hp, sizeof(hp), "%s/power1_average", c->hwmon_path);
        long pw = read_int_file(hp);
        c->power_watts = (pw > 0) ? (int)(pw / 1000000) : -1;
    } else {
        c->gpu_temp = -1; c->fan_rpm = -1; c->power_watts = -1;
    }
}

/* ----------------------------- GPU: NVIDIA nvidia-smi -------------- */

/* Run nvidia-smi once per refresh, parse the CSV output. We use a single
 * --query-gpu call that fetches all fields at once. */
static void read_gpu_nvidia(ctx_t *c) {
    /* nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,
     * temperature.gpu,power.draw,fan.speed --format=csv,noheader,nounits */
    char cmd[] = "nvidia-smi --query-gpu=utilization.gpu,memory.used,"
                 "memory.total,temperature.gpu,power.draw,fan.speed"
                 " --format=csv,noheader,nounits 2>/dev/null";
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        c->gpu_busy = -1; c->mem_busy = -1;
        c->vram_used = 0; c->vram_total = 0;
        c->gpu_temp = -1; c->fan_rpm = -1; c->power_watts = -1;
        return;
    }
    char line[512];
    if (!fgets(line, sizeof(line), fp)) {
        pclose(fp);
        c->gpu_busy = -1;
        return;
    }
    pclose(fp);
    /* parse CSV: "100, 7885, 17095, 62, 150.5, 720" */
    int util, temp, fan;
    long mem_used, mem_total;
    double power;
    /* nvidia-smi memory is in MiB; convert to bytes for human_bytes() */
    if (sscanf(line, "%d, %ld, %ld, %d, %lf, %d",
               &util, &mem_used, &mem_total, &temp, &power, &fan) >= 5) {
        c->gpu_busy = util;
        c->mem_busy = (mem_total > 0) ? (int)(mem_used * 100 / mem_total) : 0;
        c->vram_used = mem_used * (1L << 20);     /* MiB -> bytes */
        c->vram_total = mem_total * (1L << 20);
        c->gpu_temp = temp;
        c->power_watts = (int)power;
        c->fan_rpm = (fan >= 0) ? fan : -1;
    } else {
        c->gpu_busy = -1; c->mem_busy = -1;
        c->vram_used = 0; c->vram_total = 0;
        c->gpu_temp = -1; c->fan_rpm = -1; c->power_watts = -1;
    }
}

/* ----------------------------- GPU dispatch ----------------------- */

static void read_gpu(ctx_t *c) {
    if (c->gpu_backend == GPU_AMD) {
        read_gpu_amd(c);
    } else if (c->gpu_backend == GPU_NVIDIA) {
        read_gpu_nvidia(c);
    } else if (c->gpu_backend == GPU_NONE) {
        c->gpu_busy = -1; c->mem_busy = -1;
        c->vram_used = 0; c->vram_total = 0;
        c->gpu_temp = -1; c->fan_rpm = -1; c->power_watts = -1;
    } else {
        /* auto-detect: try AMD sysfs, then NVIDIA */
        if (access("/sys/class/drm/card0/device/gpu_busy_percent", R_OK) == 0) {
            c->gpu_backend = GPU_AMD;
            read_gpu_amd(c);
        } else if (access("/usr/bin/nvidia-smi", X_OK) == 0 ||
                   access("/usr/local/bin/nvidia-smi", X_OK) == 0) {
            c->gpu_backend = GPU_NVIDIA;
            read_gpu_nvidia(c);
        } else {
            c->gpu_backend = GPU_NONE;
            read_gpu(c);  /* will hit GPU_NONE branch */
        }
    }
}

static void check_proc(ctx_t *c) {
    if (c->pid <= 0) { c->proc_alive = 0; return; }
    c->proc_alive = (kill(c->pid, 0) == 0);
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

/* Parse iter value, handling optional thousands commas. */
static long parse_iter_field(const char *line, const char *mark, int commas) {
    const char *p = strstr(line, mark);
    if (!p) return -1;
    p += strlen(mark);
    while (*p && isspace((unsigned char)*p)) p++;
    if (!commas) return strtol(p, NULL, 10);
    /* strip commas */
    char numbuf[32];
    size_t j = 0;
    while (*p && (isdigit((unsigned char)*p) || *p == ',') && j < sizeof(numbuf) - 1) {
        if (*p != ',') numbuf[j++] = *p;
        p++;
    }
    numbuf[j] = 0;
    return strtol(numbuf, NULL, 10);
}

static int parse_log_line(ctx_t *c, char *line) {
    const profile_t *pr = &c->profile;
    int did_something = 0;

    /* Check for save marker (may or may not have INFO: prefix) */
    if ((pr->save_mark && strstr(line, pr->save_mark)) || strstr(line, "Saving guarded checkpoint")) {
        c->save_at_iter = c->cur_iter;
        return 0;
    }
    /* Check for validation block start */
    if (pr->val_mark && strstr(line, pr->val_mark)) {
        c->in_validation = 1;
        return 0;
    }

    /* parse timestamp: YYYY-MM-DD HH:MM:SS,mmm  (BasicSR) */
    {
        int y, mo, d, h, mi, s, ms;
        if (sscanf(line, "%d-%d-%d %d:%d:%d,%d",
                   &y, &mo, &d, &h, &mi, &s, &ms) == 7) {
            struct tm tm = {0};
            tm.tm_year = y - 1900; tm.tm_mon = mo - 1; tm.tm_mday = d;
            tm.tm_hour = h; tm.tm_min = mi; tm.tm_sec = s;
            c->last_log_line_ts = mktime(&tm);
        }
    }

    /* If we're in a validation block and this line has no epoch/iter mark,
     * try to parse val metrics. */
    int has_iter = (pr->iter_mark && strstr(line, pr->iter_mark)) ||
                   (pr->epoch_mark && strstr(line, pr->epoch_mark));

    if (c->in_validation && !has_iter) {
        for (int i = 0; i < pr->num_val_metrics; i++) {
            double v;
            if (parse_num_after(line, pr->val_metrics[i].mark, &v)) {
                c->have_val = 1;
                c->val_results[i].have = 1;
                c->val_results[i].value = v;
                /* look for "Best: <val> @ <iter>" */
                if (pr->val_metrics[i].track_best) {
                    const char *best = strstr(line, "Best:");
                    if (best) {
                        double bv = strtod(best + 5, NULL);
                        c->val_results[i].best = bv;
                        const char *at = strstr(best, "@");
                        if (at) c->val_results[i].best_iter = strtol(at + 1, NULL, 10);
                    }
                }
                /* persist to last known */
                c->have_val_last = 1;
                c->val_last[i] = c->val_results[i];
                did_something = 1;
            }
        }
        return did_something;
    }

    /* If we see an iter/epoch line, we're no longer validating. */
    if (has_iter) {
        c->in_validation = 0;
        c->have_val = 0;
    }

    /* parse epoch */
    if (pr->epoch_mark) {
        const char *p = strstr(line, pr->epoch_mark);
        if (p) {
            double v;
            if (parse_num_after(line, pr->epoch_mark, &v)) {
                c->cur_epoch = (long)v;
                did_something = 1;
            }
        }
    }

    /* parse iter */
    if (pr->iter_mark) {
        long it = parse_iter_field(line, pr->iter_mark, pr->iter_has_commas);
        if (it > 0) {
            c->cur_iter = it;
            did_something = 1;
        }
    }

    /* parse lr */
    if (pr->lr_mark) {
        double v;
        if (parse_num_after(line, pr->lr_mark, &v)) c->lr = v;
    }

    /* parse eta */
    if (pr->eta_mark) {
        const char *ep = strstr(line, pr->eta_mark);
        if (ep) {
            const char *end = strchr(ep, ']');
            const char *src = ep + strlen(pr->eta_mark);
            size_t n = end ? (size_t)(end - src) : strlen(src);
            if (n >= sizeof(c->eta)) n = sizeof(c->eta) - 1;
            memcpy(c->eta, src, n); c->eta[n] = 0;
            char *ct = strstr(c->eta, ", time");
            if (ct) *ct = 0;
            n = strlen(c->eta);
            while (n > 0 && (c->eta[n-1] == ' ' || c->eta[n-1] == ','))
                c->eta[--n] = 0;
        }
    }

    /* parse per-iter time */
    if (pr->time_mark) {
        const char *tp = strstr(line, pr->time_mark);
        if (tp) {
            double ti = 0, td = 0;
            if (sscanf(tp, "time (data): %lf (%lf)", &ti, &td) == 2) {
                c->t_iter = ti; c->t_data = td;
            }
        }
    }

    /* parse loss fields */
    int any_loss = 0;
    for (int i = 0; i < pr->num_loss_fields; i++) {
        double v;
        if (parse_num_after(line, pr->loss_fields[i].mark, &v)) {
            c->losses[i].have = 1;
            c->losses[i].value = v;
            any_loss = 1;
        }
    }
    if (any_loss) c->have_losses = 1;

    return did_something;
}

/* ----------------------------- log tail --------------------------- */

static void tail_log(ctx_t *c) {
    int fd = open(c->log_path, O_RDONLY);
    if (fd < 0) { c->log_exists = 0; return; }
    c->log_exists = 1;

    struct stat st;
    if (fstat(fd, &st) == 0) c->last_log_mtime = st.st_mtime;

    off_t sz = st.st_size;
    off_t off = sz > 65536 ? sz - 65536 : 0;
    if (lseek(fd, off, SEEK_SET) < 0) { close(fd); return; }

    char buf[65536 + 16];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return;
    buf[n] = 0;

    c->have_val = 0;
    c->in_validation = 0;
    int saw_iter = 0;

    char *line = buf;
    char *nl;
    while ((nl = strchr(line, '\n')) != NULL) {
        *nl = 0;
        if (parse_log_line(c, line)) saw_iter = 1;
        line = nl + 1;
    }
    if (*line) parse_log_line(c, line);

    if (saw_iter && c->cur_iter != c->prev_iter_for_stale) {
        c->last_seen_iter_change = c->now;
        c->prev_iter_for_stale = c->cur_iter;
    }

    /* Any new log data = training is alive */
    if (n > 0) {
        c->last_seen_iter_change = c->now;
    }
}

/* ----------------------------- config parsing --------------------- */

static void parse_config(ctx_t *c) {
    if (!c->profile.has_config) return;
    FILE *f = fopen(c->cfg_path, "r");
    if (!f) return;
    char line[512];
    char section[64] = {0};
    while (fgets(line, sizeof(line), f)) {
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == ' ' || line[L-1] == '\r'))
            line[--L] = 0;
        if (L == 0) continue;
        if (!isspace((unsigned char)line[0])) {
            char *colon = strchr(line, ':');
            if (colon) {
                /* if there's a value after the colon, it's a top-level key
                 * (like "name: foo"), not a section header */
                char *after = colon + 1;
                while (*after && isspace((unsigned char)*after)) after++;
                if (*after) {
                    /* top-level key: value — parse it */
                    *colon = 0;
                    char *v = after;
                    if (strncmp(v, "!!float", 7) == 0) v += 7;
                    while (*v && isspace((unsigned char)*v)) v++;
                    if (strcmp(line, "name") == 0 && c->exp_name[0] == 0) {
                        snprintf(c->exp_name, sizeof(c->exp_name), "%s", v);
                    }
                    continue;
                }
                /* no value after colon = section header */
                size_t n = (size_t)(colon - line);
                if (n < sizeof(section)) {
                    memcpy(section, line, n); section[n] = 0;
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
        if (strcmp(k, "name") == 0 && c->exp_name[0] == 0) {
            /* top-level "name:" field gives the experiment name */
            snprintf(c->exp_name, sizeof(c->exp_name), "%s", v);
        } else if (pr->total_iter_key && strcmp(k, pr->total_iter_key) == 0 &&
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

/* ----------------------------- checkpoint scan -------------------- */

static void scan_checkpoints(ctx_t *c) {
    /* Only re-scan the directory when a new save has been detected
     * (save_at_iter changed since last scan). Avoids opendir+stat storm
     * every second when checkpoints only change every N thousand iters. */
    if (c->last_ckpt_scan_save_iter == c->save_at_iter && c->have_ckpt)
        return;

    c->last_ckpt_scan_save_iter = c->save_at_iter;
    c->have_ckpt = 0;
    c->last_ckpt_mtime = 0;
    c->last_ckpt_iter = 0;
    c->last_ckpt_name[0] = 0;

    const char *prefix = c->profile.ckpt_prefix;
    const char *suffix = c->profile.ckpt_suffix;
    if (!prefix) return;
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
        if (suffix) {
            size_t slen = strlen(suffix);
            if (ln < slen || strcmp(e->d_name + ln - slen, suffix) != 0)
                continue;
        } else {
            /* no suffix: match directories (HF checkpoint-N) */
            struct stat st;
            char full[2560];
            snprintf(full, sizeof(full), "%s/%s", c->ckpt_dir, e->d_name);
            if (stat(full, &st) != 0) continue;
            if (!S_ISDIR(st.st_mode)) continue;
        }
        char full[2560];
        snprintf(full, sizeof(full), "%s/%s", c->ckpt_dir, e->d_name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
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
        /* parse iter from name: prefix<iter>.suffix  or  prefix<iter> (dir) */
        char *num = newest_name + plen;
        c->last_ckpt_iter = strtol(num, NULL, 10);
    }
}

/* ----------------------------- state machine ---------------------- */

static void update_state(ctx_t *c) {
    if (!c->proc_alive) { c->state = ST_CRASHED; return; }
    if (!c->log_exists) { c->state = ST_UNKNOWN; return; }
    if (c->save_at_iter > 0 && c->cur_iter <= c->save_at_iter) {
        c->state = ST_SAVING; return;
    }
    if (c->have_val) { c->state = ST_VALIDATING; return; }
    time_t since_change = c->now - c->last_seen_iter_change;
    /* GPU busy + log is recent enough = training. Use the full stale window
     * here, not 90s, because BasicSR logs every ~190s (100 iters * 1.9s). */
    if (c->gpu_busy > 5 && since_change < STALE_SECONDS) {
        c->state = ST_TRAINING; return;
    }
    if (since_change > STALE_SECONDS) { c->state = ST_IDLE; return; }
    c->state = ST_TRAINING;
}

/* ----------------------------- rendering -------------------------- */

static struct termios orig_term;
static int term_inited = 0;

static void term_enter(void) {
    if (term_inited) return;
    tcgetattr(0, &orig_term);
    struct termios raw = orig_term;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(0, TCSANOW, &raw);
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
    exit(0);
}

static int win_cols(void) {
    struct winsize w;
    if (ioctl(1, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) return w.ws_col;
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

    fputs(fill_col, stdout);
    for (int i = 0; i < full; i++) fputs("\xE2\x96\x88", stdout);
    fputs(RST, stdout);

    if (full < w && frac > 0.0) {
        fputs(lead_col, stdout);
        int idx = (int)(frac * 8 + 0.5);
        if (idx < 1) idx = 1;
        if (idx > 7) idx = 7;
        static const char *blocks[7] = {
            "\xE2\x96\x8F", "\xE2\x96\x8E", "\xE2\x96\x8D", "\xE2\x96\x8C",
            "\xE2\x96\x8B", "\xE2\x96\x8A", "\xE2\x96\x89",
        };
        fputs(blocks[idx - 1], stdout);
        fputs(RST, stdout);
        full++;
    }
    fputs(FG_DKGRAY, stdout);
    for (int i = full; i < w; i++) fputs("\xE2\x96\x91", stdout);
    fputs(RST, stdout);
}

static void minbar(double pct, int w) {
    if (pct < 0) pct = 0;
    if (pct > 1) pct = 1;
    int cells = (int)(pct * w);
    if (cells > w) cells = w;
    const char *col = FG_BGRN;
    if (pct >= 0.9) col = FG_BRED;
    else if (pct >= 0.75) col = FG_BYEL;
    fputs(col, stdout);
    for (int i = 0; i < cells; i++) fputs("\xE2\x96\x93", stdout);
    fputs(RST, stdout);
    fputs(FG_DKGRAY, stdout);
    for (int i = cells; i < w; i++) fputs("\xE2\x96\x91", stdout);
    fputs(RST, stdout);
}

static void human_bytes(long b, char *out, size_t n) {
    if (b >= (1L<<30)) snprintf(out, n, "%.1fG", b / (double)(1L<<30));
    else if (b >= (1L<<20)) snprintf(out, n, "%.0fM", b / (double)(1L<<20));
    else if (b >= (1L<<10)) snprintf(out, n, "%.0fK", b / (double)(1L<<10));
    else snprintf(out, n, "%ldB", b);
}

static void hr(int cols) {
    fputs(FG_DKGRAY, stdout);
    for (int i = 0; i < cols; i++) fputs("\xE2\x94\x80", stdout);
    fputs(RST, stdout);
}

/* format a metric value with appropriate precision */
static void fmt_metric(const metric_t *m, char *out, size_t n) {
    /* heuristic: small magnitude -> more decimals; large -> exponential */
    double v = m->value;
    if (v == 0) { snprintf(out, n, "0"); return; }
    double av = v < 0 ? -v : v;
    if (av >= 100)      snprintf(out, n, "%.2f", v);
    else if (av >= 1)   snprintf(out, n, "%.4f", v);
    else if (av >= 0.001) snprintf(out, n, "%.6f", v);
    else                 snprintf(out, n, "%.2e", v);
}

static void render(ctx_t *c) {
    int cols = win_cols();
    if (cols < 60) cols = 60;
    int inner = cols > 100 ? 96 : cols - 2;
    if (inner < 58) inner = 58;
    int bar_w = inner - 30;
    if (bar_w < 20) bar_w = 20;

    fputs(HOME, stdout);

    /* Title */
    printf(CLRLN " " FG_BCYN BOLD "\xE2\x96\x88\xE2\x96\x88" RST " "
           BOLD FG_BWHT "train-tui" RST FG_SLATE "  %s" RST
           FG_DKGRAY "  pid %d  " RST FG_DKGRAY "profile: %s" RST "\n",
           c->exp_name, (int)c->pid, c->profile.name);

    /* Progress */
    double pct = (c->total_iter > 0 && c->cur_iter > 0)
                 ? (double)c->cur_iter / c->total_iter : 0;
    printf(CLRLN "  " FG_LTGRAY "iter" RST " " BOLD FG_BWHT "%5ld" RST
           FG_DKGRAY "/" RST FG_WHT "%-5ld" RST "  ", c->cur_iter, c->total_iter);
    bar(pct, bar_w);
    printf("  " BOLD FG_BWHT "%5.1f%%" RST "\n", pct * 100.0);

    printf(CLRLN "  " FG_LTGRAY "epoch" RST " " FG_WHT "%-3ld" RST
           "  " FG_LTGRAY "lr" RST " " FG_STEEL "%.3e" RST
           "  " FG_LTGRAY "t/it" RST " " FG_WHT "%.2fs" RST
           FG_DKGRAY " (data %.3fs)" RST "\n",
           c->cur_epoch, c->lr, c->t_iter, c->t_data);

    if (c->eta[0]) {
        printf(CLRLN "  " FG_LTGRAY "eta" RST " " BOLD FG_AMBER "%s" RST "\n",
               c->eta);
    } else {
        printf(CLRLN "\n");
    }

    hr(cols); fputs("\n", stdout);

    /* GPU panel */
    const char *g = state_glyph[c->state];
    const char *lbl = state_label[c->state];
    const char *tag_col = state_tag_col[c->state];

    char vram[32], vramt[32];
    human_bytes(c->vram_used, vram, sizeof(vram));
    human_bytes(c->vram_total, vramt, sizeof(vramt));

    printf(CLRLN "  %s  %s[%s%s%s]%s  " RST,
           g, FG_DKGRAY, tag_col, lbl, FG_DKGRAY, RST);

    if (c->gpu_busy >= 0) {
        printf(FG_LTGRAY "GPU" RST " " FG_WHT "%3d%%" RST "  "
               FG_LTGRAY "MEM" RST " " FG_WHT "%3d%%" RST "  "
               FG_LTGRAY "VRAM" RST " " FG_WHT "%s" FG_DKGRAY "/" RST
               FG_SLATE "%s" RST "  "
               FG_LTGRAY "%d" FG_DKGRAY "C" RST "  "
               FG_LTGRAY "%d" FG_DKGRAY "W" RST "  "
               FG_LTGRAY "%d" FG_DKGRAY "rpm" RST "\n",
               c->gpu_busy, c->mem_busy, vram, vramt,
               c->gpu_temp, c->power_watts, c->fan_rpm);
    } else {
        printf(FG_DKGRAY "(no GPU stats)\n" RST);
    }

    if (c->gpu_busy >= 0) {
        int mbw = inner - 28;
        if (mbw < 16) mbw = 16;
        double gpu_pct = c->gpu_busy / 100.0;
        double mem_pct = c->mem_busy / 100.0;
        double vram_pct = (c->vram_total > 0)
                          ? (double)c->vram_used / c->vram_total : 0;
        printf(CLRLN "  " FG_DKGRAY "gpu " RST); minbar(gpu_pct, mbw);
        printf("  " FG_WHT "%3d%%\n", c->gpu_busy);
        printf(CLRLN "  " FG_DKGRAY "mem " RST); minbar(mem_pct, mbw);
        printf("  " FG_WHT "%3d%%\n", c->mem_busy);
        printf(CLRLN "  " FG_DKGRAY "vram" RST); minbar(vram_pct, mbw);
        printf("  " FG_WHT "%3.0f%%\n", vram_pct * 100.0);
    }

    hr(cols); fputs("\n", stdout);

    /* Losses / metrics */
    if (c->have_losses) {
        const profile_t *pr = &c->profile;
        /* print up to 3 metrics per line */
        int printed = 0;
        for (int i = 0; i < pr->num_loss_fields; i++) {
            if (!c->losses[i].have) continue;
            if (printed % 3 == 0) {
                if (printed == 0) {
                    printf(CLRLN "  " BOLD FG_BMAG "metrics" RST "  ");
                } else {
                    printf(CLRLN "  " FG_DKGRAY "        " RST);
                }
            }
            char valstr[32];
            fmt_metric(&c->losses[i], valstr, sizeof(valstr));
            printf(FG_LTGRAY "%s" RST " " FG_WHT "%s" RST "  ",
                   pr->loss_fields[i].label, valstr);
            printed++;
            if (printed % 3 == 0) printf("\n");
        }
        if (printed % 3 != 0) printf("\n");
    } else {
        printf(CLRLN "  " BOLD FG_BMAG "metrics" RST "  "
               FG_DKGRAY "(waiting for first log line)\n" RST);
    }

    hr(cols); fputs("\n", stdout);

    /* Validation */
    const profile_t *pr = &c->profile;
    if (c->have_val) {
        printf(CLRLN "  " BOLD FG_BCYN "validation" RST "  "
               FG_BYEL BOLD "RUNNING" RST "  ");
        for (int i = 0; i < pr->num_val_metrics; i++) {
            if (!c->val_results[i].have) continue;
            char vs[32];
            fmt_metric(&c->val_results[i], vs, sizeof(vs));
            printf(FG_LTGRAY "%s" RST " " BOLD FG_WHT "%s" RST,
                   pr->val_metrics[i].label, vs);
            if (pr->val_metrics[i].track_best && c->val_results[i].best > 0) {
                printf(FG_DKGRAY " (best %.4f @ %ld)" RST,
                       c->val_results[i].best, c->val_results[i].best_iter);
            }
            printf("  ");
        }
        printf("\n");
    } else if (c->have_val_last) {
        printf(CLRLN "  " BOLD FG_BCYN "validation" RST "  "
               FG_DKGRAY "last" RST "   ");
        for (int i = 0; i < pr->num_val_metrics; i++) {
            if (!c->val_last[i].have) continue;
            char vs[32];
            fmt_metric(&c->val_last[i], vs, sizeof(vs));
            printf(FG_LTGRAY "%s" RST " " FG_WHT "%s" RST,
                   pr->val_metrics[i].label, vs);
            if (pr->val_metrics[i].track_best && c->val_last[i].best > 0) {
                printf(FG_DKGRAY " (best %.4f @ %ld)" RST,
                       c->val_last[i].best, c->val_last[i].best_iter);
            }
            printf("  ");
        }
        printf("\n");
    } else {
        printf(CLRLN "  " BOLD FG_BCYN "validation" RST
               "  " FG_DKGRAY "(none yet)\n" RST);
    }

    /* Checkpoint */
    if (c->have_ckpt) {
        char when[32] = {0};
        struct tm tmv;
        if (localtime_r(&c->last_ckpt_mtime, &tmv))
            strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", &tmv);
        printf(CLRLN "  " BOLD FG_BMAG "checkpoint" RST "  "
               FG_WHT "%s" RST "  " FG_DKGRAY "iter" RST " "
               BOLD FG_WHT "%5ld" RST "  " FG_SLATE "%s" RST,
               c->last_ckpt_name, c->last_ckpt_iter, when);
        if (c->ckpt_freq > 0)
            printf("  " FG_DKGRAY "(every %ld)\n" RST, c->ckpt_freq);
        else
            printf("\n");
    } else {
        printf(CLRLN "  " BOLD FG_BMAG "checkpoint" RST
               "  " FG_DKGRAY "(none found)\n" RST);
    }

    hr(cols); fputs("\n", stdout);

    /* Footer */
    char mtime[32] = {0};
    struct tm tmv;
    if (c->last_log_mtime && localtime_r(&c->last_log_mtime, &tmv))
        strftime(mtime, sizeof(mtime), "%H:%M:%S", &tmv);
    printf(CLRLN "  " FG_DKGRAY "proc" RST " %s  "
               FG_DKGRAY "log" RST " %s  "
               FG_DKGRAY "mtime" RST " %s\n",
           c->proc_alive ? FG_BGRN "alive" RST : FG_BRED "DEAD" RST,
           c->log_exists ? FG_BGRN "ok" RST : FG_BRED "missing" RST,
           mtime[0] ? mtime : FG_BRED "?" RST);

    printf(CLRLN "  " FG_DKGRAY "q quit  " RST
           FG_DKGRAY "r refresh  " RST
           FG_DKGRAY "(auto %dms)" RST "\n" ED, REFRESH_MS);

    fflush(stdout);
}

/* ----------------------------- find log --------------------------- */

static int find_log(const ctx_t *c, char *out, size_t n) {
    char prefix[512];
    snprintf(prefix, sizeof(prefix), "train_%s_", c->exp_name);
    size_t plen = strlen(prefix);

    char d0[1600], d1[1600], d2[1600], d3[1600];
    snprintf(d0, sizeof(d0), "%s/experiments/%s", c->project_root, c->exp_name);
    snprintf(d1, sizeof(d1), "%s/experiments_hat_l/%s", c->project_root, c->exp_name);
    snprintf(d2, sizeof(d2), "/home/hermes/hat-face-training/experiments_hat_l/%s", c->exp_name);
    snprintf(d3, sizeof(d3), "%s/../experiments_hat_l/%s", c->project_root, c->exp_name);
    const char *search_dirs[4] = { d0, d1, d2, d3 };

    time_t best = 0;
    char best_full[2048] = {0};

    for (int idx = 0; idx < 4; idx++) {
        DIR *d = opendir(search_dirs[idx]);
        if (!d) continue;
        struct dirent *e;
        while ((e = readdir(d)) != NULL) {
            if (strncmp(e->d_name, prefix, plen) != 0) continue;
            size_t ln = strlen(e->d_name);
            if (ln < 5 || strcmp(e->d_name + ln - 4, ".log") != 0) continue;
            char full[2048];
            snprintf(full, sizeof(full), "%s/%s", search_dirs[idx], e->d_name);
            struct stat st;
            if (stat(full, &st) != 0) continue;
            if (st.st_mtime > best) {
                best = st.st_mtime;
                snprintf(best_full, sizeof(best_full), "%s", full);
            }
        }
        closedir(d);
    }
    if (!best_full[0]) return -1;
    snprintf(out, n, "%s", best_full);
    return 0;
}

/* ----------------------------- custom profile loader -------------- */

/* Load a .train-tui.profile file. Format: key=value lines.
 * Keys: epoch_mark, iter_mark, lr_mark, eta_mark, time_mark, save_mark,
 *        val_mark, iter_commas (0/1), ckpt_prefix, ckpt_suffix,
 *        total_iter_key, total_iter_section, val_freq_key, ...,
 *        loss_fields (comma-separated marks), loss_labels, val_metrics,
 *        val_metric_labels, val_metric_track_best, has_config */
static int load_custom_profile(ctx_t *c, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    profile_t *pr = &c->profile;
    memset(pr, 0, sizeof(*pr));
    pr->name = "custom";

    char line[1024];
    char key[128], val[512];
    while (fgets(line, sizeof(line), f)) {
        /* strip newline */
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (L == 0 || line[0] == '#') continue;
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        snprintf(key, sizeof(key), "%s", line);
        snprintf(val, sizeof(val), "%s", eq + 1);
        /* trim trailing whitespace from key */
        size_t klen = strlen(key);
        while (klen > 0 && isspace((unsigned char)key[klen-1])) key[--klen] = 0;
        char *v = val; while (*v && isspace((unsigned char)*v)) v++;

        if (strcmp(key, "epoch_mark") == 0) pr->epoch_mark = strdup(v);
        else if (strcmp(key, "iter_mark") == 0) pr->iter_mark = strdup(v);
        else if (strcmp(key, "lr_mark") == 0) pr->lr_mark = strdup(v);
        else if (strcmp(key, "eta_mark") == 0) pr->eta_mark = strdup(v);
        else if (strcmp(key, "time_mark") == 0) pr->time_mark = strdup(v);
        else if (strcmp(key, "save_mark") == 0) pr->save_mark = strdup(v);
        else if (strcmp(key, "val_mark") == 0) pr->val_mark = strdup(v);
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
            /* comma-separated list of marks */
            int n = 0;
            char *tok = strtok(v, ",");
            while (tok && n < MAX_LOSS_FIELDS) {
                while (*tok && isspace((unsigned char)*tok)) tok++;
                snprintf(pr->loss_fields[n].mark, MAX_LOSS_MARK, "%s", tok);
                tok = strtok(NULL, ",");
                n++;
            }
            pr->num_loss_fields = n;
        } else if (strcmp(key, "loss_labels") == 0) {
            int n = 0;
            char *tok = strtok(v, ",");
            while (tok && n < MAX_LOSS_FIELDS) {
                while (*tok && isspace((unsigned char)*tok)) tok++;
                snprintf(pr->loss_fields[n].label, MAX_LOSS_LABEL, "%s", tok);
                tok = strtok(NULL, ",");
                n++;
            }
        } else if (strcmp(key, "val_metrics") == 0) {
            int n = 0;
            char *tok = strtok(v, ",");
            while (tok && n < MAX_LOSS_FIELDS) {
                while (*tok && isspace((unsigned char)*tok)) tok++;
                snprintf(pr->val_metrics[n].mark, MAX_LOSS_MARK, "%s", tok);
                tok = strtok(NULL, ",");
                n++;
            }
            pr->num_val_metrics = n;
        } else if (strcmp(key, "val_metric_labels") == 0) {
            int n = 0;
            char *tok = strtok(v, ",");
            while (tok && n < MAX_LOSS_FIELDS) {
                while (*tok && isspace((unsigned char)*tok)) tok++;
                snprintf(pr->val_metrics[n].label, MAX_LOSS_LABEL, "%s", tok);
                tok = strtok(NULL, ",");
                n++;
            }
        } else if (strcmp(key, "val_metric_track_best") == 0) {
            int n = 0;
            char *tok = strtok(v, ",");
            while (tok && n < MAX_LOSS_FIELDS) {
                while (*tok && isspace((unsigned char)*tok)) tok++;
                pr->val_metrics[n].track_best = atoi(tok);
                tok = strtok(NULL, ",");
                n++;
            }
        }
    }
    fclose(f);
    c->profile_id = PROF_CUSTOM;
    return 0;
}

/* ----------------------------- usage ------------------------------ */

static void usage(const char *prog) {
    fprintf(stderr,
        "train-tui: lightweight terminal UI for monitoring AI training\n"
        "\n"
        "usage: %s [options] <project_root> <exp_name> <config_name> <pid> [log_file]\n"
        "\n"
        "  (config_name may be '-' if your project has no parseable config;\n"
        "   use -t to set total_iter in that case)\n"
        "\n"
        "options:\n"
        "  -a              auto-detect a running training process (scan /proc)\n"
        "  -p <profile>    built-in profile: basicsr, lightning, hf  (default: basicsr)\n"
        "  -c <file>       custom profile file (.train-tui.profile format)\n"
        "  -t <total>      total iterations (overrides config parsing)\n"
        "  -g <backend>    GPU backend: amd, nvidia, none, auto  (default: auto)\n"
        "  -h              this help\n"
        "\n"
        "positional args (required unless noted):\n"
        "  project_root    path to the training project root      (required)\n"
        "  exp_name        experiment name (subdir of experiments/) (required)\n"
        "  config_name     config file name (in options/train/)    (or '-')\n"
        "  pid             PID of the training process to monitor  (required)\n"
        "  log_file        explicit path to the log file           (auto-detected)\n"
        "\n"
        "built-in profiles:\n"
        "  basicsr    BasicSR / Real-ESRGAN / HAT / SwinIR  (GAN SR)\n"
        "  lightning  PyTorch Lightning progress logs\n"
        "  hf         HuggingFace Trainer dict-style logs\n"
        "  custom     load a .train-tui.profile file via -c\n"
        "\n"
        "GPU backends (auto-detected if -g not given):\n"
        "  amd        sysfs /sys/class/drm/card0/device/...\n"
        "  nvidia     nvidia-smi subprocess\n"
        "  none       no GPU stats (CPU training)\n"
        "\n",
        prog);
    exit(2);
}

/* ----------------------------- auto-detect ------------------------ */

/* Scan /proc for a python training process (train.py, trainer, etc.).
 * Returns the PID, fills cwd, cmdline config name. */
static pid_t autodetect_train(char *project_root, size_t pr_n,
                              char *cfg_name, size_t cfg_n) {
    DIR *d = opendir("/proc");
    if (!d) return -1;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        /* only numeric dirs (PIDs) */
        if (!isdigit((unsigned char)e->d_name[0])) continue;
        pid_t pid = atoi(e->d_name);
        if (pid <= 0) continue;

        /* read cmdline */
        char cmdpath[64];
        snprintf(cmdpath, sizeof(cmdpath), "/proc/%d/cmdline", pid);
        int fd = open(cmdpath, O_RDONLY);
        if (fd < 0) continue;
        char cmdline[2048] = {0};
        ssize_t n = read(fd, cmdline, sizeof(cmdline) - 1);
        close(fd);
        if (n <= 0) continue;
        cmdline[n] = 0;
        /* convert null separators to spaces so strstr works across args */
        for (ssize_t i = 0; i < n; i++) {
            if (cmdline[i] == '\0') cmdline[i] = ' ';
        }

        /* look for training python processes (train.py, train_*.py, trainer, safe_launch, etc.) */
        if (!strstr(cmdline, "python")) continue;
        if (!strstr(cmdline, "train.py") && !strstr(cmdline, "trainer") &&
            !strstr(cmdline, "train_") && !strstr(cmdline, "safe_launch"))
            continue;

        /* read cwd */
        char cwdlink[64];
        snprintf(cwdlink, sizeof(cwdlink), "/proc/%d/cwd", pid);
        char cwd[1024] = {0};
        ssize_t cl = readlink(cwdlink, cwd, sizeof(cwd) - 1);
        if (cl <= 0) continue;
        cwd[cl] = 0;

        /* extract config: look for "-opt" or "--config" followed by a .yml path */
        char cfg[512] = {0};
        char *opt = strstr(cmdline, "-opt ");
        if (!opt) opt = strstr(cmdline, "--config ");
        if (opt) {
            char *p = opt + (opt[1] == '-' ? 9 : 5);
            while (*p == ' ') p++;
            if (*p) {
                char path_buf[512] = {0};
                snprintf(path_buf, sizeof(path_buf), "%s", p);
                char *end = strchr(path_buf, ' ');
                if (end) *end = 0;
                snprintf(cfg, sizeof(cfg), "%s", path_buf);
            }
        }

        snprintf(project_root, pr_n, "%s", cwd);
        if (cfg[0]) snprintf(cfg_name, cfg_n, "%s", cfg);
        closedir(d);
        return pid;
    }
    closedir(d);
    return -1;
}

/* ----------------------------- main ------------------------------- */

int main(int argc, char **argv) {
    ctx_t c = {0};
    c.profile_id = PROF_BASICSR;
    c.gpu_backend = GPU_AUTO;

    /* parse flags */
    int argi = 1;
    const char *custom_profile_path = NULL;
    int auto_detect = 0;
    while (argi < argc && argv[argi][0] == '-') {
        if (strcmp(argv[argi], "-h") == 0 || strcmp(argv[argi], "--help") == 0)
            usage(argv[0]);
        else if (strcmp(argv[argi], "-a") == 0 || strcmp(argv[argi], "--auto") == 0) {
            auto_detect = 1;
        } else if (strcmp(argv[argi], "-p") == 0 && argi + 1 < argc) {
            const char *p = argv[++argi];
            if (strcmp(p, "basicsr") == 0) c.profile_id = PROF_BASICSR;
            else if (strcmp(p, "lightning") == 0) c.profile_id = PROF_LIGHTNING;
            else if (strcmp(p, "hf") == 0) c.profile_id = PROF_HF;
            else { fprintf(stderr, "unknown profile: %s\n", p); return 2; }
        } else if (strcmp(argv[argi], "-c") == 0 && argi + 1 < argc) {
            custom_profile_path = argv[++argi];
        } else if (strcmp(argv[argi], "-t") == 0 && argi + 1 < argc) {
            c.total_iter_override = atol(argv[++argi]);
        } else if (strcmp(argv[argi], "-g") == 0 && argi + 1 < argc) {
            const char *g = argv[++argi];
            if (strcmp(g, "amd") == 0) c.gpu_backend = GPU_AMD;
            else if (strcmp(g, "nvidia") == 0) c.gpu_backend = GPU_NVIDIA;
            else if (strcmp(g, "none") == 0) c.gpu_backend = GPU_NONE;
            else if (strcmp(g, "auto") == 0) c.gpu_backend = GPU_AUTO;
            else { fprintf(stderr, "unknown GPU backend: %s\n", g); return 2; }
        } else {
            break;
        }
        argi++;
    }

    /* positional args */
    int pos = 0;
    for (; argi < argc; argi++) {
        switch (pos) {
        case 0: snprintf(c.project_root, sizeof(c.project_root), "%s", argv[argi]); break;
        case 1: snprintf(c.exp_name, sizeof(c.exp_name), "%s", argv[argi]); break;
        case 2: snprintf(c.cfg_name, sizeof(c.cfg_name), "%s", argv[argi]); break;
        case 3: c.pid = (pid_t)atoi(argv[argi]); break;
        case 4: snprintf(c.log_path, sizeof(c.log_path), "%s", argv[argi]); break;
        }
        pos++;
    }

    /* validate required args */
    if (auto_detect || (c.project_root[0] == 0 && c.pid <= 0)) {
        /* auto-detect: scan /proc for a python training process */
        pid_t found = autodetect_train(c.project_root, sizeof(c.project_root),
                                        c.cfg_name, sizeof(c.cfg_name));
        if (found > 0) {
            c.pid = found;
            fprintf(stderr, "train-tui: auto-detected pid %d in %s\n",
                    (int)found, c.project_root);
        } else {
            fprintf(stderr,
                "train-tui: no training process found.\n\n"
                "  Use -a to auto-detect, or provide args explicitly:\n"
                "    %s -p basicsr /path/to/project experiment config.yml 12345\n\n"
                "  Run '%s -h' for full help.\n",
                argv[0], argv[0]);
            return 2;
        }

        /* after auto-detect, we have project_root + cfg_name + pid but
         * not exp_name. Load the profile and parse the config to get it. */
        if (custom_profile_path) {
            if (load_custom_profile(&c, custom_profile_path) != 0)
                die("could not load custom profile");
        } else {
            c.profile = profiles[c.profile_id];
        }
        if (c.cfg_name[0]) {
            if (c.cfg_name[0] == '/' && access(c.cfg_name, R_OK) == 0) {
                snprintf(c.cfg_path, sizeof(c.cfg_path), "%s", c.cfg_name);
            } else {
                snprintf(c.cfg_path, sizeof(c.cfg_path), "%s/%s", c.project_root, c.cfg_name);
                if (access(c.cfg_path, R_OK) != 0) {
                    snprintf(c.cfg_path, sizeof(c.cfg_path), "%s/options/train/%s",
                             c.project_root, c.cfg_name);
                }
                if (access(c.cfg_path, R_OK) != 0) {
                    snprintf(c.cfg_path, sizeof(c.cfg_path), "%s/.generated/options/%s",
                             c.project_root, c.cfg_name);
                }
            }
            parse_config(&c);
        }
    }

    if (c.project_root[0] == 0 || c.exp_name[0] == 0 || c.pid <= 0) {
        fprintf(stderr,
            "train-tui: missing required arguments.\n\n"
            "  Use -a to auto-detect a running training process, or provide:\n"
            "    %s -p basicsr /path/to/project my_experiment config.yml 12345\n\n"
            "  Run '%s -h' for full help.\n",
            argv[0], argv[0]);
        return 2;
    }

    /* load profile (if not already loaded by auto-detect) */
    if (!auto_detect) {
        if (custom_profile_path) {
            if (load_custom_profile(&c, custom_profile_path) != 0)
                die("could not load custom profile");
        } else {
            c.profile = profiles[c.profile_id];
        }
    }

    /* build paths (if not already built by auto-detect) */
    if (c.cfg_path[0] == 0 && c.cfg_name[0]) {
        snprintf(c.cfg_path, sizeof(c.cfg_path), "%s/options/train/%s",
                 c.project_root, c.cfg_name);
    }
    /* find log if not given */
    if (c.log_path[0] == 0) {
        if (find_log(&c, c.log_path, sizeof(c.log_path)) != 0) {
            fprintf(stderr, "train-tui: no log file found for experiment '%s'\n", c.exp_name);
            return 1;
        }
    }

    /* resolve checkpoint directory adjacent to the log file or in standard location */
    if (c.log_path[0]) {
        char log_dir[2048] = {0};
        snprintf(log_dir, sizeof(log_dir), "%s", c.log_path);
        char *last_slash = strrchr(log_dir, '/');
        if (last_slash) *last_slash = 0;
        snprintf(c.ckpt_dir, sizeof(c.ckpt_dir), "%s/models", log_dir);
    } else {
        snprintf(c.ckpt_dir, sizeof(c.ckpt_dir), "%s/experiments/%s/models",
                 c.project_root, c.exp_name);
    }

    /* parse config for total_iter */
    parse_config(&c);
    if (c.total_iter_override > 0) c.total_iter = c.total_iter_override;
    if (c.total_iter <= 0) {
        /* If no config-based total, and no override, we can still run
         * but the progress bar will show 0%. Warn but continue. */
        fprintf(stderr, "train-tui: warning: total_iter unknown "
                "(use -t to set it). Progress bar will be inactive.\n");
        c.total_iter = 0;
    }
    if (c.val_freq <= 0) c.val_freq = 5000;
    if (c.ckpt_freq <= 0) c.ckpt_freq = 5000;

    /* signal handling */
    struct sigaction sa = {0};
    sa.sa_handler = on_sigint;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    term_enter();
    atexit(term_exit);

    int quit = 0;
    while (!quit) {
        c.now = time(NULL);
        check_proc(&c);
        read_gpu(&c);
        tail_log(&c);
        scan_checkpoints(&c);
        update_state(&c);
        render(&c);

        struct timeval tv;
        tv.tv_sec = REFRESH_MS / 1000;
        tv.tv_usec = (REFRESH_MS % 1000) * 1000;
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(0, &fds);
        int sr = select(1, &fds, NULL, NULL, &tv);
        if (sr > 0 && FD_ISSET(0, &fds)) {
            char ch = 0;
            if (read(0, &ch, 1) == 1) {
                if (ch == 'q' || ch == 'Q' || ch == 3 || ch == 27)
                    quit = 1;
            }
        }
    }

    term_exit();
    return 0;
}