// Task 1.8 (gate G4): hold N GiB of GPU memory so other programs feel real pressure.
//
// This is the seed of maltoe-ballast --vram (Task 9). It answers, on Windows with WDDM:
//   * can a process pin a chosen amount of dedicated VRAM, and does the driver report it as used?
//   * if the request exceeds free VRAM, does cudaMalloc fail, or does the driver quietly spill into
//     shared system memory ("CUDA - Sysmem Fallback Policy")?
//   * while it runs, what does free VRAM look like (printed every 2 s)?
//
// Usage:  ballast_probe.exe <GiB> [hold_seconds]
//   hold_seconds omitted or 0: hold until a line is typed (or stdin closes).
// Exit code: 0 target reached, 3 allocation stopped short, 1 CUDA error.
//
// Build (x64 Native Tools prompt, CUDA toolkit installed):
//   nvcc -O2 -o ballast_probe.exe ballast_probe.cu

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

static const size_t MiB = 1024ull * 1024ull;

static void print_mem(const char * tag) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        std::printf("%s: cudaMemGetInfo failed\n", tag);
        return;
    }
    std::printf("%s: free %zu MiB / total %zu MiB (used %zu MiB)\n", tag, free_b / MiB, total_b / MiB, (total_b - free_b) / MiB);
    std::fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <GiB> [hold_seconds]\n", argv[0]);
        return 1;
    }
    const double gib = std::atof(argv[1]);
    const int hold_s = argc > 2 ? std::atoi(argv[2]) : 0;
    if (gib <= 0) {
        std::fprintf(stderr, "GiB must be > 0\n");
        return 1;
    }
    const size_t target = (size_t) (gib * 1024.0) * MiB;
    const size_t block  = 256 * MiB;

    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    print_mem("before");

    std::vector<void *> blocks;
    size_t got = 0;
    while (got < target) {
        const size_t n = std::min(block, target - got);
        void * p = nullptr;
        const cudaError_t e = cudaMalloc(&p, n);
        if (e != cudaSuccess) {
            std::printf("ALLOC_FAILED after %zu MiB: %s\n", got / MiB, cudaGetErrorString(e));
            break;
        }
        cudaMemset(p, 0xA5, n); // touch the memory so it is really committed
        blocks.push_back(p);
        got += n;
    }
    cudaDeviceSynchronize();
    std::printf("READY allocated %zu MiB of requested %zu MiB\n", got / MiB, target / MiB);
    print_mem("after");

    std::atomic<bool> stop{false};
    std::thread reporter([&] {
        while (!stop) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            if (!stop) print_mem("holding");
        }
    });

    if (hold_s > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(hold_s));
    } else {
        std::string line;
        std::getline(std::cin, line);
    }
    stop = true;
    reporter.join();

    for (void * p : blocks) cudaFree(p);
    return got >= target ? 0 : 3;
}
