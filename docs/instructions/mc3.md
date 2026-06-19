# MC3 — build and run ParaDiS (step by step)

Copy-paste each step in order. Run all commands from a login node unless a
step says to use a compute/GPU node.

Replace `/path/to/ParaDiS.llnl.git` with your checkout path (e.g.
`~/codes/ParaDiS.llnl.git`).

---

## Step 0 — Clone (first time only)

```bash
cd ~/codes
git clone git@github.com:hanfengzhai/ParaDiS.git ParaDiS.llnl.git
cd ParaDiS.llnl.git
```

---

## Step 1 — Install git hooks (first time only)

Blocks AI-agent `Co-authored-by` trailers in commits.

```bash
cd /path/to/ParaDiS.llnl.git
scripts/install-git-hooks.sh
```

---

## Step 2 — Load modules

`gnu12` must be loaded before `openmpi4`.

```bash
module purge
module load gnu12/12.3.0
module load openmpi4/4.1.6
```

Verify:

```bash
which gcc mpicxx
gcc --version | head -1
mpicxx --version | head -1
```

---

## Step 3 — Check for stuck builds

If a previous `make clean` hung, it can delete `obj/` while you compile.
Kill stray jobs before building:

```bash
ps aux | grep '[m]ake'
```

If you see stuck `make clean` processes:

```bash
pkill -9 -f "make clean"
```

---

## Step 4 — Create build directories

Required before the first build (avoids `can't create ../obj/p/*.o` errors
with parallel `make -j16`):

```bash
cd /path/to/ParaDiS.llnl.git
mkdir -p obj/p obj/s bin
```

---

## Step 5 — Build (CPU, on login node)

Use `SYS=linux` (not `SYS=gcc` — `gcc` links against `-lmpich`, which MC3
does not provide).

```bash
cd /path/to/ParaDiS.llnl.git
make SYS=linux
```

Build takes ~1–2 minutes. Expect many `mpicxx ../src/...` lines, ending with
`creating application ../bin/paradis`.

---

## Step 6 — Verify the build

```bash
ls -l bin/paradis
```

You should see an executable ~30 MB. A quick sanity check:

```bash
ls bin/ | head
```

Expected entries include `paradis`, `paradisgen`, `ctablegen`.

---

## Step 7 — Run a test simulation (CPU partition)

ParaDiS must be launched from the **repository root** so `inputs/` paths
resolve.

```bash
cd /path/to/ParaDiS.llnl.git
srun -N 1 -n 1 -p cpu --pty \
  ./bin/paradis tests/frank_read_src.ctrl
```

For an 8-domain parallel run:

```bash
srun -N 1 -n 8 -p cpu \
  ./bin/paradis -d tests/frank_read_src.data tests/frank_read_src.ctrl
```

(`numXdoms * numYdoms * numZdoms` in the `.ctrl` file must equal `-n`.)

---

## Step 8 — Run the Frank-Read GPU example (optional)

The example under `examples/2_frank_read/` uses elastic FMM forces and
`BCC_glide` mobility. Submit from the example directory:

```bash
cd /path/to/ParaDiS.llnl.git/examples/2_frank_read
sbatch submit_gpu.sh    # GPU (gpu-ampere)
# sbatch submit_cpu.sh  # CPU partition
```

Monitor:

```bash
squeue -u $USER
tail -f frank_read.<jobid>.out
```

The submit script builds with `GPU_ENABLED=ON` on the GPU node if `bin/paradis`
is missing. **Submit from `examples/2_frank_read/`** so Slurm sets
`SLURM_SUBMIT_DIR` correctly (Slurm copies the script to `/var/spool/`; using
`$0` to find the repo will fail).

---

## GPU build (optional, separate workflow)

CUDA (`nvcc`) is **not** available on login nodes. Request a GPU node first:

```bash
salloc -p gpu-ampere --gres=gpu:1 -N 1 -t 00:30:00
```

Inside the allocation:

```bash
module purge
module load gnu12/12.3.0 openmpi4/4.1.6 cuda/12.5

cd /path/to/ParaDiS.llnl.git
mkdir -p obj/p obj/s bin

export CUDA_PATH=/usr/local/cuda-12.5
export CUDA_LIBS=/usr/local/cuda-12.5/lib64
export NVCC="${CUDA_PATH}/bin/nvcc"
export NVCC_FLAGS="-O3 -g -rdc=true -Wno-deprecated-gpu-targets -gencode arch=compute_80,code=sm_80"

make SYS=linux GPU_ENABLED=ON
```

Verify CUDA sees the GPU:

```bash
nvidia-smi -L
ls -l bin/paradis
```

---

## Clean rebuild

Only run when you need a full wipe. **Wait for it to finish** — a hung
`make clean` can loop on `python/` and delete `obj/` in the background.

```bash
cd /path/to/ParaDiS.llnl.git
make SYS=linux clean
mkdir -p obj/p obj/s bin
make SYS=linux
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `can't create ../obj/p/*.o: No such file or directory` | Run Step 3 (kill stuck `make clean`), then Step 4 (`mkdir -p obj/p obj/s bin`), then rebuild |
| `target system has not been identified` | Add `SYS=linux` to every `make` command |
| `cannot find -lmpich` | You used `SYS=gcc`; use `SYS=linux` |
| `openmpi4` module won't load | Load `gnu12/12.3.0` first |
| `NVCC compiler not found` | Build on a GPU compute node with `cuda/12.5` loaded |
| FMM table not found at runtime | Launch `paradis` from the repository root |
| MPI task count error | Match `srun -n` to `numXdoms * numYdoms * numZdoms` in the `.ctrl` file |
| `Repo: /var/spool` in job output | Re-submit from `examples/2_frank_read/` with `submit_gpu.sh` or `submit_cpu.sh` (uses `SLURM_SUBMIT_DIR`) |

After the run, visualize results:

```bash
cd examples/2_frank_read
python ../utils/visualize.py
```

Outputs are written to `examples/2_frank_read/output/` (PNGs, MP4/MOV animation, properties plots).

The same script works for other examples under `examples/`; it auto-detects the `.ctrl` basename
and defaults to `<name>_results` and `output/` under the example directory.

## makefile reference

| Setting | MC3 value |
|---------|-----------|
| `SYS` | `linux` (pass on command line) |
| `MODE` | `PARALLEL` (default) |
| `GPU_ENABLED` | `OFF` for CPU build; `ON` for GPU build |
| `XLIB_MODE` | `OFF` |
