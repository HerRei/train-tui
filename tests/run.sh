#!/bin/sh
# End-to-end tests: render single frames (--once) against fixture logs, a fake
# sysfs with two AMD cards and a stand-in nvidia-smi, and check the text.
# Usage: sh tests/run.sh ./train_tui
set -u

BIN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
FIX=$HERE/fixtures
T=$(mktemp -d)
RUN=
SPAWNED=
# KEEP=1 keeps the outputs for a look afterwards
if [ -n "${KEEP:-}" ]; then
    trap 'kill "$RUN" $SPAWNED 2>/dev/null; echo "outputs kept in $T"' EXIT INT TERM
else
    trap 'kill "$RUN" $SPAWNED 2>/dev/null; rm -rf "$T"' EXIT INT TERM
fi

export TRAIN_TUI_PCI_IDS="$FIX/pci.ids"
# no real nvidia-smi: the AMD cases must not pick up a host's NVIDIA cards
NO_SMI=/nonexistent/nvidia-smi
export TRAIN_TUI_NVIDIA_SMI=$NO_SMI
unset NO_COLOR

# The watched "training process" and its process group (this shell's).
sleep 300 &
RUN=$!
export TT_RUN_PID=$RUN
# A process of some other job: its own session where setsid exists (PID 1
# can be in our group inside a container), else PID 1.
OTHER=1
SPAWNED=
if command -v setsid > /dev/null 2>&1; then
    setsid sleep 300 &
    OTHER=$!
    SPAWNED=$OTHER
fi
export TT_OTHER_PID=$OTHER

fails=0
pass=0

# expect NAME OUTPUT_FILE PATTERN...  (fixed strings)
expect() {
    name=$1 file=$2
    shift 2
    ok=1
    for pat in "$@"; do
        if ! grep -qF -- "$pat" "$file"; then
            echo "FAIL $name: missing '$pat'"
            ok=0
        fi
    done
    if [ "$ok" = 1 ]; then
        pass=$((pass + 1))
    else
        fails=$((fails + 1))
        sed 's/^/    | /' "$file"
    fi
}

# reject NAME OUTPUT_FILE PATTERN...
reject() {
    name=$1 file=$2
    shift 2
    for pat in "$@"; do
        if grep -qF -- "$pat" "$file"; then
            echo "FAIL $name: unexpected '$pat'"
            fails=$((fails + 1))
            return
        fi
    done
    pass=$((pass + 1))
}

mk() {
    mkdir -p "$(dirname "$1")"
    printf '%s\n' "$2" > "$1"
}

# --- fake sysfs: RX 9060 XT (card0), Intel iGPU (card1), Radeon VII (card2) ---
S=$T/root
dev=$S/sys/devices
a=$dev/pci0000:00/0000:04:00.0
b=$dev/pci0000:80/0000:83:00.0
i=$dev/pci0000:00/0000:00:02.0
mk "$a/gpu_busy_percent" 100
mk "$a/mem_info_vram_used" 11246424064
mk "$a/mem_info_vram_total" 17095983104
mk "$a/uevent" "DRIVER=amdgpu
PCI_ID=1002:7590"
h=$a/hwmon/hwmon4
mk "$h/name" amdgpu
mk "$h/temp1_label" edge;     mk "$h/temp1_input" 66000;  mk "$h/temp1_crit" 100000
mk "$h/temp2_label" junction; mk "$h/temp2_input" 83000;  mk "$h/temp2_crit" 110000
mk "$h/temp3_label" mem;      mk "$h/temp3_input" 80000;  mk "$h/temp3_crit" 105000
mk "$h/power1_average" 103000000
mk "$h/power1_cap" 165000000
mk "$h/fan1_input" 1259
mk "$b/gpu_busy_percent" 99
mk "$b/mem_info_vram_used" 10944200704
mk "$b/mem_info_vram_total" 17163091968
mk "$b/uevent" "DRIVER=amdgpu
PCI_ID=1002:66AF"
h=$b/hwmon/hwmon5
mk "$h/name" amdgpu
mk "$h/temp1_label" edge;     mk "$h/temp1_input" 83000;  mk "$h/temp1_crit" 100000
mk "$h/temp2_label" junction; mk "$h/temp2_input" 103000; mk "$h/temp2_crit" 110000
mk "$h/temp3_label" mem;      mk "$h/temp3_input" 86000;  mk "$h/temp3_crit" 94000
mk "$h/power1_input" 169000000
mk "$h/power1_cap" 150000000
mk "$h/fan1_input" 881
mk "$i/uevent" "DRIVER=i915"
drm=$S/sys/class/drm
mkdir -p "$drm/card0" "$drm/card1" "$drm/card2" "$drm/card0-HDMI-A-1" \
         "$drm/renderD128" "$drm/renderD129" "$drm/renderD130"
ln -s ../../../devices/pci0000:00/0000:04:00.0 "$drm/card0/device"
ln -s ../../../devices/pci0000:00/0000:00:02.0 "$drm/card1/device"
ln -s ../../../devices/pci0000:80/0000:83:00.0 "$drm/card2/device"
ln -s ../../../devices/pci0000:00/0000:00:02.0 "$drm/renderD128/device"
ln -s ../../../devices/pci0000:00/0000:04:00.0 "$drm/renderD129/device"
ln -s ../../../devices/pci0000:80/0000:83:00.0 "$drm/renderD130/device"
mk "$drm/card0-HDMI-A-1/status" connected
kfd=$S/sys/class/kfd/kfd
mk "$kfd/topology/nodes/0/gpu_id" 0
mk "$kfd/topology/nodes/1/gpu_id" 15209
mk "$kfd/topology/nodes/1/properties" "cpu_cores_count 0
drm_render_minor 129"
mk "$kfd/topology/nodes/2/gpu_id" 17126
mk "$kfd/topology/nodes/2/properties" "cpu_cores_count 0
drm_render_minor 130"
# the run: one process per card (same process group); another job uses card0
mk "$kfd/proc/$RUN/vram_15209" 11183026176
mk "$kfd/proc/$RUN/vram_17126" 0
mk "$kfd/proc/$$/vram_17126" 10929975296
mk "$kfd/proc/$OTHER/vram_15209" 524288000
hw=$S/sys/class/hwmon
mk "$hw/hwmon10/name" coretemp
mk "$hw/hwmon10/temp1_label" "Package id 0"; mk "$hw/hwmon10/temp1_input" 63000
mk "$hw/hwmon10/temp1_crit" 105000
mk "$hw/hwmon10/temp2_label" "Core 0";       mk "$hw/hwmon10/temp2_input" 71000
mk "$hw/hwmon2/name" nvme; mk "$hw/hwmon2/temp1_input" 66850; mk "$hw/hwmon2/temp1_crit" 84850
mk "$hw/hwmon3/name" nvme; mk "$hw/hwmon3/temp1_input" 56850; mk "$hw/hwmon3/temp1_crit" 87850
mk "$hw/hwmon4/name" amdgpu; mk "$hw/hwmon4/temp1_input" 99000

# --- a second fake sysfs with one card: an older kernel (one unlabelled
# temperature) and no fan reading, as on many passively cooled cards ---
S1=$T/root1
one=$S1/sys/devices/pci0000:00/0000:03:00.0
mk "$one/gpu_busy_percent" 87
mk "$one/mem_info_vram_used" 6442450944
mk "$one/mem_info_vram_total" 17163091968
mk "$one/uevent" "PCI_ID=1002:66AF"
mk "$one/hwmon/hwmon1/name" amdgpu
mk "$one/hwmon/hwmon1/temp1_input" 58000
mk "$one/hwmon/hwmon1/power1_average" 212000000
mkdir -p "$S1/sys/class/drm/card0"
ln -s ../../../devices/pci0000:00/0000:03:00.0 "$S1/sys/class/drm/card0/device"

empty=$T/empty
mkdir -p "$empty"

# --- logs written now, so the run looks alive ---
J=$T/run
mkdir -p "$J"
awk 'BEGIN {
    print "{\"event\": \"start\", \"step\": 0, \"total_steps\": 17509, \"tok_per_step\": 262144}"
    t = 1791400000
    for (s = 2400; s <= 3280; s += 10) {
        t += 90.4
        loss = 2.10 - (s - 2400) * 0.0001
        line = sprintf("{\"step\": %d, \"loss\": %.4f, \"lr\": 0.0006, \"gnorm\": 0.36, \"tok_s\": 28996, \"time\": %.1f", s, loss, t)
        if (s == 2500) line = line ", \"val\": 2.1263"
        if (s == 3000) line = line ", \"val\": 2.0662"
        print line "}"
    }
}' > "$J/log.jsonl"
: > "$J/ckpt.pt"
: > "$J/ckpt_step2000.pt"

B=$T/basicsr
cp -R "$FIX/basicsr" "$B"
touch -t 202001010000 "$B/experiments/demo/models/net_g_45000.pth"
touch "$B/experiments/demo/models/net_g_50000.pth"
blog=$(ls "$B"/experiments/demo/train_demo_*.log)

H=$T/hf
mkdir -p "$H/checkpoint-100" "$H/checkpoint-200"
cp "$FIX/hf.log" "$H/train.log"
touch -t 202001010000 "$H/checkpoint-100"

run() {   # run NAME ARGS...  -> $T/NAME.out
    name=$1
    shift
    TRAIN_TUI_SYSFS="$S" "$BIN" --once "$@" > "$T/$name.out" 2> "$T/$name.err" || {
        echo "FAIL $name: exit status $?"
        cat "$T/$name.err"
        fails=$((fails + 1))
    }
}

# 1. JSON lines + two AMD cards + host temperatures
run jsonl -p jsonl "$J" pianorun - "$RUN" "$J/log.jsonl"
expect jsonl "$T/jsonl.out" \
    "pianorun" "profile jsonl" "[TRAINING]" \
    "3280/17509" "18.7%" "speed 9.04 s/it" "eta 1d 11h" \
    "cpu 63°C" "nvme 67°C" "2 GPUs" \
    "gpu0 RX 9050 / 9060 XT" "load" "100%" "10.5G/15.9G" \
    "edge 66°C" "hotspot 83°C" "mem 80°C" "103/165 W" "1259 rpm" "this run +1" \
    "gpu1 Radeon VII" "hotspot 103°C" "mem 86°C" "169/150 W" "881 rpm" \
    "loss 2.012" "gnorm 0.36" \
    "validation  last @ 3000" "val 2.0662 (best 2.0662 @ 3000)" \
    "proc alive" "log ok"
reject jsonl-igpu "$T/jsonl.out" "gpu2" "card0-HDMI" "99°C"

# 2. a profile file built on jsonl
run custom -c "$FIX/custom.profile" "$J" pianorun - "$RUN" "$J/log.jsonl"
expect custom "$T/custom.out" "profile custom/jsonl" "tok/s 28996" \
    "checkpoint  ckpt_step2000.pt  iter 2000" "val 2.0662"

# 3. BasicSR: config, log discovery, validation block, checkpoints
run basicsr -p basicsr "$B" demo demo.yml "$RUN"
expect basicsr "$T/basicsr.out" \
    "50100/250000" "epoch 1" "lr 1.000e-04" "t/it 1.91s (data 0.002s)" \
    "eta 4 days, 19:24:47" "l_g_pix 0.040011" "l_g_percep 8.9233" "out_d_f -1.0288" \
    "validation  last @ 50000" "psnr 31.0703 (best 31.0892 @ 40000)" \
    "ssim 0.8519 (best 0.852 @ 30000)" "(every 5000)" \
    "checkpoint  net_g_50000.pth  iter 50000" "every 5000"

# 3a. sample.profile describes the same BasicSR log
run sample -c "$HERE/../sample.profile" "$B" demo demo.yml "$RUN"
expect sample "$T/sample.out" "profile custom" "50100/250000" \
    "psnr 31.0703 (best 31.0892 @ 40000)" "checkpoint  net_g_50000.pth"

# 3b. the same log cut inside the validation block
head -n 6 "$blog" > "$T/basicsr-val.log"
run basicsr_val -p basicsr "$B" demo demo.yml "$RUN" "$T/basicsr-val.log"
expect basicsr_val "$T/basicsr_val.out" "[VALIDATING]" "validation  RUNNING" "psnr 31.0703"

# 4. Hugging Face: total from the log, eval lines, checkpoint directories
run hf -p hf "$H" hfrun - "$RUN" "$H/train.log"
expect hf "$T/hf.out" "200/1250" "16.0%" "epoch 0.8" "[SAVING]" \
    "eval_loss 1.98 (best 1.98 @ 200)" "eval_acc 0.61 (best 0.61 @ 200)" \
    "checkpoint  checkpoint-200  iter 200"

# 5. NVIDIA through nvidia-smi
export TRAIN_TUI_NVIDIA_SMI="$FIX/nvidia-smi"
run nvidia -g nvidia -p jsonl "$J" pianorun - "$RUN" "$J/log.jsonl"
expect nvidia "$T/nvidia.out" "gpu0 RTX 4090" "core 71°C" "413/450 W" "fan 63%" \
    "20.0G/24.0G" "this run" "gpu1 RTX A6000" "core 44°C" "other job"
reject nvidia-mem "$T/nvidia.out" "hotspot" "mem 0°C"
export TRAIN_TUI_NVIDIA_SMI=$NO_SMI

# 5b. a single card, AMD and NVIDIA
TRAIN_TUI_SYSFS="$S1" "$BIN" --once -p jsonl "$J" pianorun - "$RUN" "$J/log.jsonl" > "$T/single.out" 2>&1
expect single "$T/single.out" "[TRAINING]" "gpu0 Radeon VII" "load" " 87%" "6.0G/16.0G" \
    "edge 58°C" "212 W"
reject single-extra "$T/single.out" "gpu1" "GPUs" "hotspot" "rpm" "cpu " "this run"
export TRAIN_TUI_NVIDIA_SMI="$FIX/nvidia-smi" TT_NV_ONE=1
run nvidia_single -g nvidia -p jsonl "$J" pianorun - "$RUN" "$J/log.jsonl"
expect nvidia_single "$T/nvidia_single.out" "gpu0 RTX 4090" "core 71°C" "this run"
reject nvidia_single-extra "$T/nvidia_single.out" "gpu1" "GPUs"
export TRAIN_TUI_NVIDIA_SMI=$NO_SMI
unset TT_NV_ONE

# 6. no GPUs, no sensors
TRAIN_TUI_SYSFS="$empty" "$BIN" --once -g none -p jsonl "$J" pianorun - "$RUN" "$J/log.jsonl" > "$T/none.out" 2>&1
expect none "$T/none.out" "(no GPU stats)"
reject none-temps "$T/none.out" "cpu " "nvme "

# 7. the process is gone: crashed, or finished when the last step was reached
sh -c 'exit 0' &
DEAD=$!
wait "$DEAD"
run dead -p jsonl "$J" pianorun - "$DEAD" "$J/log.jsonl"
expect dead "$T/dead.out" "[CRASHED]" "proc DEAD"
run done -t 3280 -p jsonl "$J" pianorun - "$DEAD" "$J/log.jsonl"
expect done "$T/done.out" "[FINISHED]" "proc exited" "100.0%"

# 8. every frame fits an 80-column terminal (characters = UTF-8 lead bytes)
for f in "$T"/*.out; do
    w=$(LC_ALL=C awk '{ n = gsub(/[^\200-\277]/, "&"); if (n > m) m = n } END { print m + 0 }' "$f")
    if [ "$w" -gt 80 ]; then
        echo "FAIL width: $(basename "$f") has a line of $w columns"
        fails=$((fails + 1))
    fi
done
pass=$((pass + 1))

# 8b. COLUMNS sets the width when the output is not a terminal
COLUMNS=96 TRAIN_TUI_SYSFS="$S" "$BIN" --once -p jsonl "$J" pianorun - "$RUN" "$J/log.jsonl" > "$T/wide.txt"
w=$(LC_ALL=C awk '{ n = gsub(/[^\200-\277]/, "&"); if (n > m) m = n } END { print m + 0 }' "$T/wide.txt")
if [ "$w" = 96 ]; then pass=$((pass + 1)); else echo "FAIL COLUMNS=96 gave $w columns"; fails=$((fails + 1)); fi

# 8c. the live view without a terminal (ssh without -t) prints one frame and exits
TRAIN_TUI_SYSFS="$S" "$BIN" -p jsonl "$J" pianorun - "$RUN" "$J/log.jsonl" > "$T/notty.txt" 2> "$T/notty.err" &
live=$!
i=0
while kill -0 "$live" 2> /dev/null && [ "$i" -lt 50 ]; do sleep 0.1; i=$((i + 1)); done
if kill -0 "$live" 2> /dev/null; then
    kill "$live"
    echo "FAIL the live view kept running without a terminal"
    fails=$((fails + 1))
else
    expect notty "$T/notty.txt" "[TRAINING]" "gpu1 Radeon VII"
    expect notty-hint "$T/notty.err" "use ssh -t"
    reject notty-escapes "$T/notty.txt" "$(printf '\033')"
fi

# 9. command line
"$BIN" --version | grep -q "^train-tui [0-9]" && pass=$((pass + 1)) || { echo "FAIL --version"; fails=$((fails + 1)); }
"$BIN" -h 2> "$T/help.out"; expect help "$T/help.out" "--once" "jsonl" "-k <dir>" "-i <seconds>" "COLUMNS"
if "$BIN" --bogus > /dev/null 2>&1; then echo "FAIL unknown option accepted"; fails=$((fails + 1)); else pass=$((pass + 1)); fi
printf 'loss_fields = "loss":\nbase = jsonl\n' > "$T/late-base.profile"
if "$BIN" --once -c "$T/late-base.profile" "$J" x - "$RUN" "$J/log.jsonl" > /dev/null 2>&1; then
    echo "FAIL a late 'base' key was accepted"; fails=$((fails + 1))
else
    pass=$((pass + 1))
fi

echo "$pass passed, $fails failed"
[ "$fails" = 0 ]
