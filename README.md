# GPU Gems 39 — Volume Rendering Techniques

A modern **Vulkan (C++)** implementation of the techniques in
[GPU Gems, Chapter 39 — *Volume Rendering Techniques*](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-39-volume-rendering-techniques).

Each distinct topic from the chapter is a **separate application** built on a
shared, reusable rendering engine. The project follows a small game-engine
architecture (layered `core` / `gfx` / `scene` / `ui` / `volume` modules,
SOLID responsibilities) and pulls its dependencies with CMake `FetchContent`.

| App | Chapter section | Technique |
|-----|-----------------|-----------|
| `01_basic_emissive`     | §39.2–39.3 | 3D texture upload, **view-aligned slicing** (Algorithm 39-2), emissive optical model (Listing 39-1), back-to-front "over" compositing |
| `02_transfer_functions` | §39.4.3    | **1D & 2D transfer functions**, interactive control-point editor, colormap presets, **opacity correction** (Eq. 3) |
| `03_illumination`       | §39.4.3    | **Blinn-Phong** local illumination (Eq. 4) with gradient normals via **central differences** (Eq. 2), evaluated per-fragment |
| `04_volumetric_shadows` | §39.5.1    | **Half-angle slicing** two-pass light/eye buffers (Algorithm 39-3), volumetric shadows |
| `05_translucency`       | §39.5.1    | Per-channel light **absorption** + **blur propagation** (Eq. 9) for forward-scattered translucency |
| `06_procedural`         | §39.5.2    | **Noise-volume perturbation** — coordinate warping (Fig. 39-14) and optical-property modulation, animated |

All six apps share an orbit camera (left-drag to rotate, scroll to zoom) and a
Dear ImGui control panel.

## Architecture

```
GPUGemsVolumeRendering/
├── engine/                     # VolumeEngine — reusable, technique-agnostic
│   ├── include/vve/ , src/
│   │   ├── core/   Window (GLFW), Application (template-method loop), Log
│   │   ├── gfx/    Context, Swapchain, Renderer, Buffer, Texture,
│   │   │           ShaderModule, PipelineBuilder   (Vulkan-Hpp RAII + VMA)
│   │   ├── scene/  Camera, ArcballController
│   │   ├── ui/     ImGuiLayer  (dynamic-rendering backend)
│   │   └── volume/ VolumeData, GradientComputer, TransferFunction,
│   │               SliceProxyGeometry, SliceGeometryBuffers
├── apps/                       # one executable per chapter topic
│   ├── 01_basic_emissive/ … 06_procedural/   (main.cpp + shaders/)
├── cmake/                      # Dependencies.cmake, CompileShaders.cmake
└── CMakeLists.txt
```

## Requirements

- **Vulkan SDK 1.3+** (provides headers, the loader, validation layers and
  `glslc`). Set the `VULKAN_SDK` environment variable (the SDK installer does
  this). A GPU/driver supporting Vulkan 1.3 dynamic rendering.
- **CMake ≥ 3.24** and a **C++20** compiler (MSVC 2022+, Clang, or GCC).
- Internet access on first configure (FetchContent clones GLFW, GLM, VMA and
  Dear ImGui).

## Building

```sh
cmake -S . -B build -G Ninja          # or the Visual Studio generator
cmake --build build
```

On Windows with the Visual Studio generator:

```sh
cmake -S . -B build
cmake --build build --config RelWithDebInfo
```

Executables land in `build/bin/`, and each app's SPIR-V shaders are compiled to
`build/bin/shaders/<app>/`.

## Running

Executables live in `build/bin/`. They locate their shaders relative to the
executable, so they can be launched from any working directory (double-click,
IDE, or command line):

```sh
./build/bin/01_basic_emissive
```

Validation layers are enabled by default in the app configs; disable them by
setting `Config::enableValidation = false`.

`vve::volume::VolumeData::loadRaw(path, nx, ny, nz, bytesPerVoxel)`
loads any raw volume directly.

## Notes & limitations

- Apps 04 and 05 implement the `dot(view, light) >= 0` branch of Algorithm 39-3;
  a runtime **"flip slice order"** toggle accommodates the opposite arrangement.
  They push two `mat4`s per draw and expect `maxPushConstantsSize ≥ 144`
  (true on all modern desktop GPUs).
- App 05's translucency is the chapter's model in spirit (per-channel absorption
  + light-buffer blur) rather than a full two-buffer scattering simulation.
