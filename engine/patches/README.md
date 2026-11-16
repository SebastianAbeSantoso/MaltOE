# llama.cpp patches

A small, numbered patch series applied on top of the llama.cpp commit in `../LLAMA_CPP_COMMIT`. One concern per patch, named by hook point, for example `0001-loader-molt-chunks.patch`.

`../build.py` applies every `*.patch` in name order to a clean checkout before building. Logic lives in `../molt/`; patches only add the hook points.

The first patches are written in Tasks 6 and 7, after the Feasibility Report approves the route.
