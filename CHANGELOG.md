# Changelog

## 1.1.1 (2026-10-08)

Mostly about watching a run from another machine, over SSH or Tailscale.

- Without a terminal (`ssh host train-tui …` without `-t`) it now prints one
  plain frame and a hint instead of filling the pipe with screen updates.
- Resizing the window redraws it cleanly.
- Arrow keys no longer quit. Esc on its own, `q` and Ctrl-C still do.
- `-i` sets the refresh interval for slow links, and `COLUMNS` sets the
  width of `--once` output.
- More tests: one AMD card on an older kernel, one NVIDIA card, and the
  no-terminal path.

## 1.1.0 (2026-10-08)

The big one: every GPU in the box, and temperatures you can read at a glance.

- All cards are shown, not just the first: every amdgpu card, every card
  `nvidia-smi` lists, or a mix of both. Cards are named from `pci.ids`.
- Each card shows edge, hotspot and memory temperature, colored against the
  limits its driver reports, plus power against its cap. The status line
  adds the CPU package and the hottest NVMe drive.
- Each card says who is using it: this run (the watched process and its
  process group, so DDP ranks count), other jobs, or nobody.
- A `jsonl` profile for logs written one JSON object per line. Profile files
  can start from a built-in profile with `base = …`, and gained
  `total_mark`, `stamp_mark` and `min`/`max` best values.
- An ETA measured from the step rate when the log has none.
- Hugging Face logs now show their validation values and total steps.
- The log is read incrementally, so best values have history and `IDLE`
  finally works. `FINISHED` appears when the process exits after the last
  step.
- `--once`, `-k`, `-V` and `NO_COLOR`. A process owned by another user is no
  longer reported dead, and each frame is drawn in one write.
- Everything fits in 80 columns; long lists wrap.
- Tests and CI (gcc and clang), and `make install`.
- Paths that only existed on my machine are gone.

## 1.0.0 (2026-08-20)

- First release: BasicSR, Lightning, Hugging Face and custom profiles; AMD
  sysfs and NVIDIA backends for one GPU.
