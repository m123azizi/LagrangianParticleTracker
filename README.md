# Real-Time Volumetric Particle Tracking (VPTV)

This repository hosts the code for a real-time, high-fidelity, volumetric 3D particle tracking system designed for capturing Lagrangian trajectories in turbulent or volumetric flows. The system integrates GPU acceleration (CUDA), a Qt-based GUI, and a synchronized multi-camera imaging setup to reconstruct particle positions in three dimensions with extensive visualization and data handling features.

---

## Dependencies

This project relies on several third-party libraries and frameworks. Please refer to the **Installation Guide PDF** (linked below) for version details and step-by-step setup instructions.

| Dependency               | Version (as tested)   | Description                                      |
|--------------------------|------------------------|--------------------------------------------------|
| **Qt**                   | 5.9.9                  | GUI framework                                    |
| **CUDA Toolkit**         | 11.8                   | GPU-accelerated particle tracking                |
| **CMake**                | ≥ 3.27.6               | Build system configuration                       |
| **Visual Studio**        | 2022 (MSVC 14.3)       | C++ development environment                      |
| **Boost**                | 1.83.0                 | Threading, chrono, random utilities              |
| **VTK**                  | 8.2.0                  | Visualization of particle tracks and meshes      |
| **OpenCV**               | 4.8.0                  | Image processing and camera input                |
| **YAML-CPP**             | 0.6                    | Output file serialization in YAML                |
| **LibTorch (PyTorch C++)** | 2.1.0+cu118         | Backend acceleration and tensor ops              |
| **OptiTrack Camera SDK** | 2.3.1 or 3.1.0         | For OptiTrack camera support                     |

📄 **Full details, screenshots, and configuration tips are available in the PDF guide below.**

📖 [Installation_Guide.pdf](./PTV%20software%20installation%20steps.pdf)

---

## Build Instructions (Short Summary)

1. Install all dependencies listed above and configure their environment variables.
2. Clone this repository and modify `CMakeLists.txt` as per your installed versions.
3. Use **CMake GUI** to configure the project and generate the build files.
4. Open the solution (`.sln`) in **Visual Studio 2022**, build in `Release` mode.
5. Copy all required `.dll` files from external libraries into the `build/bin/Release` folder.

For exact CMake flags, variable paths, and dependency linking, refer to the full PDF guide.

---

## Running the Software

Once built, the executable files can be found in:

```
build/bin/RelWithDebInfo/
```

- Run `app-LPT_Optitrack.exe` to interface with real cameras.
- Run `app-LPT_Virtual.exe` to test in virtual demonstration mode.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for full terms.

---

## Citation

If you use this project in academic research, please cite:

> to be added

---
