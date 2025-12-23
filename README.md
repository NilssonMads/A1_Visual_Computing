# Image Stitching with OpenCV

This project implements a simple image stitching pipeline using **OpenCV**. Given two images of the same scene, it detects features, matches them, estimates a homography, and creates a panorama.

It supports multiple feature detectors (SIFT, ORB, AKAZE) and allows optional ratio test filtering and image scaling for faster processing.

---

## Features

- Detect keypoints and descriptors using **SIFT**, **ORB**, or **AKAZE**
- Match descriptors with **BFMatcher** or **FlannBasedMatcher**
- Optional **Lowe’s ratio test** to filter matches
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

