# train-tui

A lightweight terminal dashboard for watching AI training runs. Pure C, no dependencies, read-only.

**Live demo:** https://herrei.github.io/train-tui/

```
 ██ train-tui  main_rwkv94  pid 284926  profile custom/jsonl
  iter 3430/17509  ██████████▍░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   19.6%
  lr 6.000e-04   speed 9.05 s/it
  eta 1d 11h 23m  (Fri 21:15)
────────────────────────────────────────────────────────────────────────────────
  ●  [TRAINING]   cpu 61°C   nvme 67°C   2 GPUs
  ● gpu0 RX 9050 / 9060 XT   load ▓▓▓▓▓▓▓▓▓ 100%  vram ▓▓▓▓▓▓▓░░░ 10.5G/15.9G
         edge 66°C  hotspot 79°C   mem 81°C  132/165 W  1356 rpm  this run
  ● gpu1 Radeon VII          load ▓▓▓▓▓▓▓▓▓ 100%  vram ▓▓▓▓▓▓░░░░ 10.2G/16.0G
         edge 84°C  hotspot 103°C  mem 86°C  143/150 W  881 rpm   this run
────────────────────────────────────────────────────────────────────────────────
  metrics  loss 1.9857  gnorm 0.502  tok/s 28988
────────────────────────────────────────────────────────────────────────────────
  validation  last @ 3000  val 2.0662 (best 2.0662 @ 3000)
  checkpoint  ckpt_step2000.pt  iter 2000  04:44 (5h 08m ago)
────────────────────────────────────────────────────────────────────────────────
  proc alive  log ok  last write 09:52:05 (19s ago)
```

## What it shows

- **Progress:** step, total, ETA (from the log, or measured from the step rate), learning rate, speed.
- **Every GPU:** load, VRAM, power against its cap and fan speed, plus which job uses the card.
- **GPU temperatures:** edge, hotspot and memory, each colored against the driver's own critical limit.
- **Host:** CPU package temperature and the hottest NVMe drive.
- **Metrics:** losses and other metrics, the last validation with best values, and the newest checkpoint.
- **State:** `TRAINING`, `VALIDATING`, `SAVING`, `IDLE` (log quiet for 10 minutes and the GPUs idle), `CRASHED` or `FINISHED`.

Temperatures are white, amber within 15 °C of the sensor's critical limit, and bold red within 5 °C. The limits come from the driver. Where the driver gives none, as on NVIDIA, cautious defaults are used.

## Quick start

```bash
git clone https://github.com/HerRei/train-tui.git
cd train-tui
make
./train_tui -p basicsr /path/to/project my_experiment config.yml 12345
```

Press `q` to quit. `-a` finds a running `python … train*.py` by itself. `make install` installs it as `/usr/local/bin/train-tui`.

### Arguments

```
./train_tui [options] project_root exp_name config_name pid [log_file]

  -a              auto-detect a running training process (scan /proc)
  -p <profile>    basicsr (default), lightning, hf or jsonl
  -c <file>       profile file (see sample.profile)
  -t <total>      total iterations (overrides config and log)
  -k <dir>        checkpoint directory (default: next to the log)
  -g <backend>    GPUs: auto (every card found), amd, nvidia, none
  --once          print one frame and exit (no escape codes when piped)
  -V, --version   print the version
```

Use `-` for `config_name` if your framework has no config file. `NO_COLOR=1` turns colors off.

## Frameworks

| Profile | For | Total iterations from |
|---------|-----|-----------------------|
| `basicsr` | BasicSR, Real-ESRGAN, HAT, SwinIR | `train: total_iter` in the YAML |
| `lightning` | PyTorch Lightning progress logs | `-t` |
| `hf` | Hugging Face Trainer | the `Total optimization steps = …` line |
| `jsonl` | One JSON object per line | a `"total_steps"` field |
| custom | Anything else: a `key = value` file (`-c`), see [`sample.profile`](sample.profile) | your choice |

A custom profile can start from a built-in one with `base = jsonl` as its first line and change only what differs.

### JSON lines

Most hand-written training loops can feed the `jsonl` profile with one line per logging step:

```python
log.write(json.dumps({"step": step, "loss": loss, "lr": lr, "time": time.time()}) + "\n")
log.write(json.dumps({"step": step, "loss": loss, "val_loss": val_loss, "time": time.time()}) + "\n")
```

```bash
./train_tui -p jsonl . my_run - $PID runs/my_run/log.jsonl
```

It reads `step`, `loss`, `lr`, `grad_norm`/`gnorm`, `epoch`, and `val_loss`/`val`/`eval_loss`, keeping the best (lowest) value. It takes `total_steps` from any line and uses `time` (Unix seconds) for the ETA.

## GPUs

| Backend | Source |
|---------|--------|
| `amd` | sysfs (amdgpu): every card under `/sys/class/drm`, names from `pci.ids`, processes from `/sys/class/kfd` |
| `nvidia` | `nvidia-smi --query-gpu` and `--query-compute-apps` |
| `none` | CPU-only training |

`auto` uses every card it finds from both vendors. The second line of each card says who is using it:

- **this run:** a process in the same process group as the PID you passed. DDP ranks started by `torchrun` or by one shell script count.
- **+N:** N processes from other jobs on the same card.
- **other job / N other jobs:** the card is busy with something else.
- **free:** no compute process.

## Watching a remote machine

```bash
ssh -t gpu-box 'train-tui -p jsonl ~/proj run1 - 12345 ~/proj/runs/run1/log.jsonl'
ssh gpu-box 'train-tui --once -a'     # one plain-text frame, e.g. for scripts
```

## Safety

train-tui is read-only. It never signals, writes to or opens for writing anything in the training run; `kill(pid, 0)` only checks that the process exists. It reads the log, the config, GPU and temperature sensors, and the checkpoint listing. The log is read incrementally: on start, the last 8 MiB; after that, only new lines.

## Development

`make test` builds with `-Werror` and runs [`tests/run.sh`](tests/run.sh). The tests use fixture logs for each profile, a fake sysfs with two AMD cards and a stand-in `nvidia-smi`. CI runs them with gcc and clang.

## Requirements

- C compiler (gcc or clang)
- Linux (reads `/proc` and `/sys`)

## License

MIT
