# Contributing

## Build and test

CUDA 12.x or 13.x, CMake ≥ 3.28 and Ninja; on Linux GCC ≥ 12 and GoogleTest, on Windows a Visual Studio 2022 x64
developer prompt.

```sh
cmake --preset dev && cmake --build --preset dev
CUDA_VISIBLE_DEVICES="" ctest --preset dev -L '^cpu$'     # what CI runs
ctest --preset dev                                         # adds the GPU tests on your GPU
```

The `dev` preset builds for compute capability 8.9; pass `-DCMAKE_CUDA_ARCHITECTURES=<cc>` for another GPU. Build
trees go to `build/<preset>`. Tests that need model files skip when the files are absent.

CI builds every release package (Linux and Windows, CUDA 12 and 13, the `-sm75` and `-sm70` builds, Vulkan and HIP)
and runs the CPU tests and the package checks. It has no GPU: a change to a kernel, the planner or a pipeline path
needs a GPU run, named in the pull request.

## Branches

Branch from `main` and name the branch for the kind of change: `feat/`, `fix/`, `perf/`, `refactor/`, `docs/`, `ci/`,
`test/`, and `release/X.Y.Z` for a version bump. Open a draft pull request to run CI on work in progress.

## Commits

Subject line: `area: what changes`, in the imperative, without a final period. Areas: `cuda`, `kernels`, `pipeline`,
`planner`, `weights`, `quantize`, `capi`, `server`, `cli`, `sm75`, `sm70`, `vulkan`, `hip`, `build`, `ci`, `release`,
`docs`, `tests`.

```
vulkan: count the DiT's weights in --vram-budget with every block streamed
capi: report the preload's plan before its first phase
```

The body says why, and what the change costs in speed or memory if anything.

## Pull requests

`main` takes changes through pull requests only. Small changes are squash-merged with the pull request's title as the
commit subject; a branch whose commits each build and tell a story (a new backend, a series of kernels) is merged with
a merge commit. A change on a hot path reports s/step and peak VRAM before and after, on the same GPU, model and
settings: no feature may cost speed.

## Releases

Maintainers release from `main`. A `release/X.Y.Z` pull request bumps `MI_VERSION_*` in
`include/mossinfer/mossinfer.h` and the version strings in the docs. After it merges and CI passes on `main`, a signed
`vX.Y.Z` tag on the merge commit starts `.github/workflows/release.yml`, which builds, checks and publishes the
packages. `vX.Y.Z-rcN` tags publish pre-releases. Release tags cannot be moved or deleted.
