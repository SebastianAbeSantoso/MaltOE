# Task 1 hardware spikes

These are the steps of the Model Feasibility Round that need a GPU. Kiro wrote and tested the scripts (see "What was checked" at the bottom). You run them on your machines and paste the printed tables back into the chat. Kiro then fills in `docs/feasibility-report.md`.

Nothing here is part of the product. Spike code is not imported by production code (see `structure.md`).

## Before every run: power settings

A laptop on battery runs both its CPU and GPU at lower power, so a run on battery is slower and not comparable with one plugged in. Before any step below:

1. Plug in the charger, and use the charger that came with the laptop.
2. Set the Windows power mode to **Best performance** (Settings > System > Power & battery), and keep the same mode for every run. Use the laptop vendor's performance profile too if it has one.
3. Run `nvidia-smi` and note the power limit (`Pwr:Usage/Cap`). Your readings so far: 45 W on battery, 140 W plugged in. Use the plugged-in state for every run and paste a fresh `nvidia-smi` back with the other output.
4. Leave other GPU programs closed.

`python -m harness.manifest` records whether the laptop is on battery and the GPU power limit, and warns when either lowers performance. The scripts in this folder do not check this for you, so do it by hand.

## Run these in order

The first three are short and answer the engine and budget gates. Start the long one (G3) afterwards, ideally overnight.

| Step | Task | Machine | What it answers | Time (my estimate, not measured) |
| --- | --- | --- | --- | --- |
| 0 | prerequisites | laptop | build llama.cpp at the pinned commit, download the model | 1 h, mostly the 18.6 GB download |
| A | 1.6 | laptop | does a chunked expert run correctly and at what cost on CUDA? | 20 min |
| B | 1.7 | laptop | does llama.cpp's own MoE cache already do the paging, and how well? | 1 h |
| C | 1.8 | laptop, then PC | do the VRAM and RAM fillers create real pressure on Windows? | 1 h |
| D | 1.5 | laptop | does a width-reduced Qwen3-30B-A3B stay useful? | 3 to 6 h |
| E | 1.9 | laptop | does Edge0 run on this machine? | 1 to 2 h, mostly 42 GB of downloads |

Run every script from a PowerShell prompt (not `pwsh -File`), so that array arguments such as `-BallastGiB 0,2,4` work.

## Step 0: prerequisites and the pinned llama.cpp

Install once: Visual Studio 2022 Build Tools with the C++ workload, CMake, Ninja, Git, the CUDA Toolkit, [uv](https://docs.astral.sh/uv/).

**Which CUDA Toolkit:** 12.4 or newer, 12.9 and 13.x included. llama.cpp's own Windows release builds use 12.4 and 13.5, and its CMake has code paths for 12.8, 12.9 and 13. The RTX 4060 (Ada, architecture 89) works with all of them, and nothing in llama.cpp's CUDA backend at the pinned commit uses a feature that only a newer toolkit offers on this card. The safe choice is a toolkit no newer than the "CUDA Version" in the top-right of the `nvidia-smi` output (the newest runtime your driver was built for). NVIDIA's compatibility rules also let a build from the same major release (13.x) run on any driver 580 or newer, with a limited feature set, so a slightly newer 13.x probably works, but that is the less tested path. The laptop's driver 610.62 reports CUDA 13.3, so 12.9 and 13.0 to 13.3 are all fine. Kiro has not built with any of them on Windows, so the first build is the test.

The "CUDA Version" in `nvidia-smi` is what your *driver* supports, not an installed toolkit. Check what is installed with `nvcc --version` (in the x64 Native Tools prompt). If it is not found, install a toolkit from NVIDIA. In the installer choose **Custom** and untick the display driver if it is offered, so that it does not replace your current driver; every measurement should run on one driver version.

**Host compiler (important).** CUDA 12.9 accepts MSVC compilers whose `_MSC_VER` is 1930 to 1939 (NVIDIA's "MSVC Version 193x" row). The compiler reports that number as `cl` version 19.3x. The newest v143 toolset reports 19.44, which is outside that range, so CUDA 12.9 refuses it. You do not need to uninstall anything. Install an older v143 toolset next to the new one:

1. Open **Visual Studio Installer** > **Modify** on your VS 2022 Build Tools.
2. Go to **Individual components** and search for `14.39` (or `v143 ... 17.9`). Tick **MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.39)** and install.
3. Open the **x64 Native Tools Command Prompt for VS 2022** and select that toolset for the session:

```powershell
vcvars64.bat -vcvars_ver=14.39
```

Do not use `-allow-unsupported-compiler`. Before the long build, check the pairing in that same prompt:

```powershell
cl                       # expect a compiler version starting with 19.39 (not 19.44)
nvcc --version
"__global__ void k(){} int main(){k<<<1,1>>>(); return cudaDeviceSynchronize();}" | Out-File -Encoding ascii hello.cu
nvcc hello.cu -o hello.exe
.\hello.exe; echo "exit code: $LASTEXITCODE"      # 0 means toolkit, compiler and driver work together
```

The pinned commit is `a946eb541a6d3051496bdcc451a0fa85772b8dfd` (2026-11-16). The spike scripts were checked against its source.

```powershell
git clone https://github.com/ggml-org/llama.cpp C:\src\llama.cpp
cd C:\src\llama.cpp
git checkout a946eb541a6d3051496bdcc451a0fa85772b8dfd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
build\bin\llama-bench.exe --help | Select-Object -First 3
```

`89` is the RTX 4060 (Ada). Binaries land in `C:\src\llama.cpp\build\bin`, which is the default `-LlamaBin` in the scripts.

Download the model (official Qwen GGUF, Apache-2.0; 18.56 GB; the command resumes if interrupted):

```powershell
mkdir D:\models
curl.exe -L -C - -o D:\models\Qwen3-30B-A3B-Q4_K_M.gguf https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
```

Paste back: the output of `nvidia-smi`, and `systeminfo | findstr /C:"Total Physical Memory" /C:"Available Physical Memory"`.

## Step A: chunked experts on CUDA (Task 1.6)

```powershell
cd <repo>\spikes\chunked_moe
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_DIR=C:\src\llama.cpp -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
.\build\chunked_moe.exe --list-devices
.\build\chunked_moe.exe --backend CUDA0 --type q4_K --n-expert 128 --n-used 8 --n-tokens 1   --iters 200
.\build\chunked_moe.exe --backend CUDA0 --type q4_K --n-expert 128 --n-used 8 --n-tokens 512 --iters 20
.\build\chunked_moe.exe --backend CUDA0 --type q8_0 --n-expert 128 --n-used 8 --n-tokens 1   --iters 200 --chunks 4
```

This builds one Qwen3-30B-A3B MoE layer (2048 x 768, 128 experts, top-8). It checks that three chunks summed equal the fused expert, that width k equals a plain f32 reference using the first k chunks, and times fused against chunked k = 1..3. The last command tries four chunks (192 neurons each), which only works with 32-element block types such as q8_0.

Paste back: all three outputs in full.

## Step B: llama.cpp's MoE cache (Task 1.7)

At the pinned commit, llama.cpp has an experimental `--moe-cache-mib` option that keeps frequently used experts of host-resident layers in VRAM. Molt would build on it, so we measure it before designing anything.

```powershell
cd <repo>\spikes\moe_cache_probe
.\run_moe_cache.ps1 -LlamaBin C:\src\llama.cpp\build\bin -Model D:\models\Qwen3-30B-A3B-Q4_K_M.gguf
```

This runs 7 configurations twice each: three static `-ncmoe` values, and four cache sizes with all experts on the host. Close other GPU programs first. The laptop has less RAM (16 GB) than the model file (18.6 GB), so the SSD is used, and that is intended.

If a static configuration fails with out-of-memory, that is a result; the script records it and moves on.

Paste back: the printed table, and `nvidia-smi` once while idle.

## Step C: can we hold memory on Windows? (Task 1.8)

```powershell
cd <repo>\spikes\ballast_probe
nvcc -O2 -o ballast_probe.exe ballast_probe.cu
.\ballast_probe.exe 2 10        # holds 2 GiB for 10 seconds; check nvidia-smi in another window

# 1. with NVIDIA Control Panel > Manage 3D settings > "CUDA - Sysmem Fallback Policy" = Driver Default
.\run_g4.ps1 -LlamaBin C:\src\llama.cpp\build\bin -Model D:\models\Qwen3-30B-A3B-Q4_K_M.gguf -Label default
# 2. change the setting to "Prefer No Sysmem Fallback", then:
.\run_g4.ps1 -LlamaBin C:\src\llama.cpp\build\bin -Model D:\models\Qwen3-30B-A3B-Q4_K_M.gguf -Label no-fallback
```

Each run uses ballast sizes 0, 1, 2, 3, and 4 GiB. With no ballast the benchmark should fit in 8 GB; the larger ballasts push it over. The question is whether it then fails or silently slows down.

RAM filler, on **both** machines (on the 32 GB PC this is what forces the SSD tier into use):

```powershell
uv run --with psutil python ram_probe.py 12 --seconds 20
```

Paste back: the `ballast_probe` output, both `run_g4` tables, the driver version, and the `ram_probe` output from each machine.

## Step D: width quality (Task 1.5, gate G3)

This is the long one. It makes four versions of the model: full width, two thirds width, one third width, and one third width with random neurons kept (the control that shows whether importance ordering matters). It measures perplexity and a 5 + 5 problem smoke test for each.

```powershell
cd <repo>\spikes\width_emulation
.\run_g3.ps1 -LlamaBin C:\src\llama.cpp\build\bin -LlamaCppSrc C:\src\llama.cpp -BaseGguf D:\models\Qwen3-30B-A3B-Q4_K_M.gguf -WorkDir D:\molt-work
```

Needs about 45 GB free on the work drive. It deletes each variant file after measuring it. To try one variant first: `-Variants k1`. If it is interrupted, re-run with `-SkipImatrix` to reuse the importance statistics.

HumanEval scoring runs model-written Python on your machine, with a timeout. It is not a sandbox.

Paste back: the printed table, `work\calibration.json`, and the first 30 lines of `work\results\imatrix.log`.

## Step E: Edge0 reference run (Task 1.9)

See [`edge0/README.md`](edge0/README.md).

## What was checked

Kiro could not run anything on a GPU. These parts were verified in the Linux sandbox:

- `chunked_moe`: built against the pinned ggml on CPU and run for f32, f16, q8_0, and q4_K. Chunk sum equals the fused expert to about 1e-7, width k matches the f32 prefix reference, and the build is clean under AddressSanitizer. The CUDA path uses the same code with a different backend; its first run is yours.
- `make_variant.py`: 7 tests on a synthetic GGUF (selection matches an independent computation, kept rows unchanged, dropped rows all zero, other tensors untouched, FFN output equals the prefix-width reference, Q4_K row arithmetic, missing calibration statistics).
- `smoke_eval.py`: tested against a mock server answering with the datasets' own gold solutions (all solved) and with wrong answers (none solved).
- `run_g3.ps1`, `run_g4.ps1`, `run_moe_cache.ps1`: run end to end under PowerShell 7 on Linux with fake `llama-*` binaries. Windows PowerShell 5.1 and the real CUDA tools were not available, so expect to fix small things on the first real run. Send the error text and Kiro will fix the script.
- `ballast_probe.cu`: syntax-checked against a stub CUDA header. Not compiled with nvcc.
- `ram_probe.py`: only its non-Windows fallback ran.
