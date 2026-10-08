# Sample profile for train-tui. Copy it, edit it to match your log, then run:
#   ./train_tui -c my.profile -t 100000 /path/to/project exp - <pid> <log>
#
# Format: key = value, one per line; '#' starts a comment. All keys are
# optional. Marks are substrings; the number right after a mark is read.

# Start from a built-in profile (basicsr, lightning, hf, jsonl) and change
# only what differs. Must be the first key.
# base = jsonl

# --- Log markers ---
epoch_mark = epoch:
iter_mark  = iter:
lr_mark    = lr:(
eta_mark   = eta:
time_mark  = time (data):
save_mark  = Saving models and training states.

# Start of a validation block (metrics on the following lines). Leave it out
# when validation values sit on the training lines, as in JSON-lines logs.
val_mark   = Validation

# Optional: total iterations from a log line, and a Unix timestamp per line
# (used for the ETA when the log has no eta of its own).
# total_mark = "total_steps":
# stamp_mark = "time":

# Whether the iter number has thousands commas like 50,100 (1) or not (0)
iter_commas = 1

# --- Checkpoint filename pattern ---
# Without a suffix, checkpoints are directories (Hugging Face checkpoint-N).
ckpt_prefix = net_g_
ckpt_suffix = .pth

# --- Config file keys (for total_iter etc.) ---
# Set has_config = 0 if your project has no parseable YAML; use -t instead.
has_config = 1
total_iter_key    = total_iter
total_iter_section = train
val_freq_key      = val_freq
val_freq_section  = val
ckpt_freq_key     = save_checkpoint_freq
ckpt_freq_section = logger

# --- Loss / metric fields ---
# Comma-separated. loss_labels are the display names, in the same order;
# put them after loss_fields. Without labels the mark itself is shown.
loss_fields = l_g_pix:, l_g_percep:, l_g_gan:, l_d_real:, l_d_fake:, out_d_real:, out_d_fake:
loss_labels = l_g_pix, l_g_percep, l_g_gan, l_d_real, l_d_fake, out_d_r, out_d_f

# --- Validation metrics ---
val_metrics = psnr:, ssim:
val_metric_labels = psnr, ssim
# How to find the best value, one per metric:
#   log = read "Best: <val> @ <iter>" from the log (BasicSR), also written 1
#   min = lowest value seen (losses)
#   max = highest value seen (psnr, accuracy)
#   0   = don't track
val_metric_track_best = log, log
