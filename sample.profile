# Sample custom profile for train-tui.
# Copy this file, edit to match your training log format, then run:
#   ./train_tui -c my_profile.txt -t 100000 /path/to/project exp cfg pid log
#
# All fields are optional; omit ones your log doesn't have.

# --- Log markers: substring to find, number parsed after it ---
epoch_mark = epoch:
iter_mark  = iter:
lr_mark    = lr:(
eta_mark   = eta:
time_mark  = time (data):
save_mark  = Saving models and training states.
val_mark   = Validation

# Whether the iter number has thousands commas like 50,100 (1) or not (0)
iter_commas = 1

# --- Checkpoint filename pattern ---
ckpt_prefix = net_g_
ckpt_suffix = .pth

# --- Config file keys (for auto-detecting total_iter etc.) ---
# Set has_config = 0 if your project has no parseable config file; use -t instead.
has_config = 1
total_iter_key    = total_iter
total_iter_section = train
val_freq_key      = val_freq
val_freq_section  = val
ckpt_freq_key     = save_checkpoint_freq
ckpt_freq_section = logger

# --- Loss / metric fields ---
# Comma-separated lists. loss_fields are the substrings to find in each
# training log line; loss_labels are the display names (shown in the TUI).
# The Nth label corresponds to the Nth field.
loss_fields = l_g_pix:, l_g_percep:, l_g_gan:, l_d_real:, l_d_fake:, out_d_real:, out_d_fake:
loss_labels = l_g_pix, l_g_percep, l_g_gan, l_d_real, l_d_fake, out_d_r, out_d_f

# --- Validation metrics ---
# Parsed from the validation block in the log.
val_metrics = psnr:, ssim:
val_metric_labels = psnr, ssim
# Which metrics track a "Best: <val> @ <iter>" annotation.
# 1 = track best, 0 = don't. Comma-separated, one per metric.
val_metric_track_best = 1, 1