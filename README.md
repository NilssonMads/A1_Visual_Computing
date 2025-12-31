# Panorama Stitcher (OpenCV C++)

Feature-based panorama stitching in C++ with OpenCV. Supports SIFT, ORB, and AKAZE; ratio test filtering; configurable scale and RANSAC threshold; Voronoi seam finding; multi-band blending; and automatic export of both the panorama and a statistics image with match visualization.

## How It Works
1. **Feature Detection**: SIFT / ORB / AKAZE (factory-selected). Optional downscale factor to speed up processing.
2. **Descriptor Matching**: kNN (k=2) with detector-appropriate matcher (FLANN for SIFT, BF-Hamming for ORB/AKAZE), followed by Lowe's ratio test.
3. **Homography + RANSAC**: Robust estimation with configurable reprojection threshold; inliers reported.
4. **Warp + Seam Finding**: Images warped onto a common canvas; Voronoi seam finder chooses low-error seams.
5. **Multi-band Blending**: Smooths exposure/color transitions.
6. **Outputs**: Saves the stitched panorama and a statistics image (matches + key numbers) to `images/stored_images/`.

## Requirements
- C++17
- OpenCV 4.x (built with nonfree for SIFT)
- CMake 3.15+
- A C++ toolchain (MSVC, clang, or gcc)

## Build (Windows / CMake)
```powershell
# From repo root
mkdir build
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
The executable is produced at `build/Release/PanoramaStitcher.exe`.

## Run
From the build output directory:
```powershell
cd build/Release
./PanoramaStitcher.exe
```
You will be prompted for:
1) Left image name (without extension)
2) Right image name (without extension)
3) Detector: `SIFT` | `ORB` | `AKAZE` (default SIFT)
4) Scale factor (e.g., `0.5` for half-size; default 0.25)
5) Use ratio test? (`1` yes / `0` no; default 1)
6) Lowe's ratio threshold (default 0.75)
7) RANSAC reprojection threshold in pixels (default 3.0)

## Image Paths
- The program expects images in `../images` relative to the executable, i.e., `images/` at the repo root.
- Outputs are saved to `images/stored_images/` with auto-incremented names:
  - Panorama: `<left>_<right>_det-<DET>.png`
  - Statistics: `<left>_<right>_det-<DET>_stats.png`

## Notes
- If you add new input images, place the `.jpg` files in `images/`.
- SIFT offers the highest robustness; ORB is fastest; AKAZE is a balanced middle ground.
- The statistics image includes: keypoints per image, matches, good matches, RANSAC inliers (%), detector, ratio threshold, RANSAC threshold, and scale.# Image Stitching with OpenCV

This project implements a simple image stitching pipeline using **OpenCV**. Given two images of the same scene, it detects features, matches them, estimates a homography, and creates a panorama.

It supports multiple feature detectors (SIFT, ORB, AKAZE) and allows optional ratio test filtering and image scaling for faster processing.

---

## Features

- Detect keypoints and descriptors using **SIFT**, **ORB**, or **AKAZE**
- Match descriptors with **BFMatcher** or **FlannBasedMatcher**
- Optional **Lowe�s ratio test** to filter matches
- Estimate homography with **RANSAC**
- Warp and blend images to generate a panorama
- Display matches and panorama interactively
- Save output panorama to disk

---

## Requirements

- **C++17** or later
- **OpenCV 4.x** (compiled with features for SIFT/ORB/AKAZE)
- C++ compiler (MSVC, GCC, or Clang)
- Optional: Visual Studio for Windows

---

## File Structure

