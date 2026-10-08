# Changelog

## 1.1.0 (2026-10-08)

- Every GPU in the machine, not just the first: AMD (all amdgpu cards) and
  NVIDIA (every line of `nvidia-smi`), also mixed. Cards get their marketing
  name from `pci.ids`.
- Temperatures: edge, hotspot and memory per card, colored against the
  driver's critical limits; CPU package and NVMe temperature in the status
  line. Power is shown against the card's cap.
- Which job uses each card ("this run", "+N" other jobs, "free"), from
  `/sys/class/kfd` or `nvidia-smi --query-compute-apps`. DDP ranks in the
  watched process group count as this run.
- New `jsonl` profile for JSON-lines logs; profile files can build on a
  built-in profile with `base = ...`; new keys `total_mark`, `stamp_mark`,
  `val_metric_track_best = min|max`.
- ETA and speed measured from the step rate when the log has no ETA.
- Hugging Face: validation values and the total from the log now show.
- The log is read incrementally (last 8 MiB at start, then new lines only),
  so best values have history and IDLE detection works.
- `--once` prints a single frame (plain text when piped); `-k` sets the
  checkpoint directory; `-V`; `NO_COLOR`; a process owned by another user is
  no longer reported dead; the screen is drawn in one write (no flicker).
- `FINISHED` state when the process exits after the last step.
- Lines stay within 80 columns; lists wrap.
- Tests (`make test`) and CI with gcc and clang; `make install`.
- Removed paths that only existed on the author's machine.

## 1.0.0 (2026-08-20)

- First release: BasicSR, Lightning, Hugging Face and custom profiles; AMD
  sysfs and NVIDIA backends for one GPU.
