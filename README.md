# train-tui

A lightweight terminal dashboard for monitoring AI training runs. Pure C, no dependencies.

**🎬 Live demo:** https://herrei.github.io/train-tui/

---

## Quick start

```bash
git clone https://github.com/HerRei/train-tui.git
cd train-tui
make
./train_tui -p basicsr /path/to/project my_experiment config.yml 12345
```

Press `q` to quit.

### Arguments

```
./train_tui [options] <project_root> <exp_name> <config_name> <pid> [log_file]

  -p <profile>    basicsr, lightning, or hf  (default: basicsr)
  -c <file>       custom profile file
  -t <total>      total iterations (if no parseable config)
  -g <backend>    amd, nvidia, none, or auto  (default: auto)
```

Use `-` for `config_name` if your framework has no config file.

---

## Frameworks

| Profile | For |
|---------|-----|
| `basicsr` | BasicSR, Real-ESRGAN, HAT, SwinIR |
| `lightning` | PyTorch Lightning |
| `hf` | HuggingFace Trainer |
| `custom` | Anything else — write a `key=value` profile file (`-c`), see [`sample.profile`](sample.profile) |

## GPU backends

| Backend | Source |
|---------|--------|
| `amd` | sysfs (amdgpu) |
| `nvidia` | nvidia-smi |
| `none` | CPU-only training |

Auto-detected. Force with `-g`.

## Safety

Read-only. Never signals, writes to, or opens the training process/files for writing. Only reads the log, config, GPU stats, and checkpoint listing.

## Requirements

- C compiler (gcc or clang)
- Linux

## License

MIT