# Step E: Edge0 reference run (Task 1.9)

Goal: find out whether `edge0-infer` runs on the RTX 4060 laptop, and with which backend. This becomes baseline BL-3 later (Task 4). Edge0 runs its own checkpoint (a 4-bit Qwen3.6-35B-A3B with trained adapters), so the result is a reference point, not a same-model comparison.

Edge0 is Apache-2.0. These steps follow its own [Windows instructions](https://github.com/Edge0-AI/Edge0/blob/main/edge0-infer/docs/getting-started.md). If a command here disagrees with theirs, theirs wins; paste the difference back.

## Needs

- About **42 GB** free disk: 19.7 GB of original weights and 22 GB of converted assets.
- Visual Studio 2022 Build Tools (C++), CMake, Ninja, Git, uv, and the **Vulkan SDK** (for the Vulkan build).
- Run in the **x64 Native Tools Command Prompt for VS 2022**, then start `powershell` inside it.

## Build

```powershell
git clone https://github.com/Edge0-AI/Edge0.git C:\src\Edge0
cd C:\src\Edge0\edge0-infer
git rev-parse HEAD                      # paste this back
cmake --preset windows-vulkan
cmake --build --preset windows-vulkan
build\windows-vulkan\bin\edge0-infer.exe info --probe     # expect a line: model.0: edge0_35b, and a Vulkan device
```

Paste back: the `info --probe` output.

## Get the model

```powershell
$env:MOE_SOURCE_35B = "D:\edge0\source35b"      # must not exist yet
$env:MOE_ASSETS_35B = "D:\edge0\assets35b"      # must not exist yet
uv sync --project tools/assets
function assets_py { uv run --project tools/assets --offline --no-sync python -I @args }
assets_py tools/assets/edge0_35b/fetch_source.py probe
assets_py tools/assets/edge0_35b/fetch_source.py download --out $env:MOE_SOURCE_35B
assets_py tools/assets/edge0_35b/make_assets.py --source $env:MOE_SOURCE_35B --out $env:MOE_ASSETS_35B
assets_py tools/assets/check_assets.py check $env:MOE_ASSETS_35B
```

## Run

```powershell
# 1. both phases on the GPU, expert slots in VRAM (32 slots used 5.35 GiB of GPU memory on their RTX 4070 SUPER)
build\windows-vulkan\bin\edge0-infer.exe bench --model $env:MOE_ASSETS_35B --device vulkan --model-option expert_slots=32 --user-file tests/fixtures/bench/library-report-prompt-head.txt --warmup 10 --ntok 100 --runs 1

# 2. fewer slots, which is how a smaller VRAM budget is expressed in Edge0
build\windows-vulkan\bin\edge0-infer.exe bench --model $env:MOE_ASSETS_35B --device vulkan --model-option expert_slots=16 --user-file tests/fixtures/bench/library-report-prompt-head.txt --warmup 10 --ntok 100 --runs 1
build\windows-vulkan\bin\edge0-infer.exe bench --model $env:MOE_ASSETS_35B --device vulkan --model-option expert_slots=8  --user-file tests/fixtures/bench/library-report-prompt-head.txt --warmup 10 --ntok 100 --runs 1

# 3. CPU decode with Vulkan prefill, which needs the least VRAM
build\windows-vulkan\bin\edge0-infer.exe bench --model $env:MOE_ASSETS_35B --device cpu --prefill-device vulkan --user-file tests/fixtures/bench/library-report-prompt-head.txt --warmup 10 --ntok 100 --runs 1
```

If a run fails with out-of-memory, that is a result too; paste the error. While each run is going, note `nvidia-smi` memory used.

Paste back: the printed result of each `bench` run (time to first token, prompt tokens per second, decode tokens per second, memory), plus any error text.

Also say whether Edge0 offers any way to change width or precision at run time. Per its README it does not, so this is a check of what you see, not an expectation.
