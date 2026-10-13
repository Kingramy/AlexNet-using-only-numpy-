# 🧠 AlexNet from Scratch — Pure NumPy & Numba CUDA

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Numba%20Accelerated-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![NumPy](https://img.shields.io/badge/Numpy-from--scratch-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-ImageNet10%20(Imagenette)-orange.svg)](https://www.kaggle.com/datasets/liusha249/imagenet10)

**An end-to-end, high-performance implementation of the classic AlexNet architecture written from first principles.**  
*Zero reliance on PyTorch, TensorFlow, or JAX for core tensor ops, autodiff, or training.*

[Features](#-key-features) •
[Architecture](#-network-architecture) •
[CUDA Kernels](#-custom-cuda-kernels) •
[Quickstart](#-quickstart--usage) •
[Pretrained Weights](#-pretrained-weights) •
[Memory Optimizations](#-memory--speed-optimizations) •
[License](#-license)

</div>

---

## 🌟 Overview

This repository demonstrates that high-performance deep learning models can be built, trained, and GPU-accelerated **entirely from scratch**. Using only **NumPy** for data orchestration and **Numba CUDA** for handcrafted GPU kernels, this project implements:

- Complete **forward pass**, **analytical backward pass** (manual backpropagation), and parameter updates.
- Custom GPU kernels for 2D convolutions, max-pooling with coordinate argmax masks, dense layers, activation functions, and the **Adam optimizer**.
- Low-overhead system optimizations including asynchronous multithreaded prefetching, RAM-sparing `uint8` dataset storage, and desktop-friendly GPU duty-cycle throttling.

---

## 🚀 Key Features

- **⚡ 100% Framework-Free Autograd & Training**: Every mathematical operation—from 2D convolution gradients to cross-entropy loss and Adam momentum updates—is derived and computed explicitly.
- **🏎️ Handcrafted CUDA Kernels**:
  - `_conv2d_forward_kernel` & `_conv2d_dx_kernel` / `_conv2d_dW_kernel` / `_conv2d_db_kernel`
  - `_maxpool_forward_kernel` & `_maxpool_backward_kernel` (retaining spatial max indices)
  - `_dense_forward_kernel` & `_dense_dx_kernel` / `_dense_dW_kernel` / `_dense_db_kernel`
  - `_relu_forward_kernel` & `_relu_backward_kernel`
  - `_bn_scale_shift_kernel` (Batch Normalization)
  - `_adam_update_kernel` (Adam optimizer executed directly on GPU device memory)
- **💾 75% RAM Reduction**: Dataset loader caches samples in `uint8` format (~1.6 GB RAM instead of ~6.4 GB for `float32`), converting to normalized `float32` on-the-fly during batch iteration.
- **🔄 Asynchronous Multi-threaded Prefetching**: Overlaps CPU data augmentation with GPU execution using Python queues and background worker threads.
- **🎨 Pure NumPy Augmentation Pipeline**: Random horizontal flipping, random crops with reflection padding, brightness jitter, contrast scaling, saturation modulation, and Cutout regularization.
- **🖥️ Desktop Usability & Process Throttling**: Includes Windows process priority management (`BELOW_NORMAL_PRIORITY_CLASS`) and customizable GPU duty-cycle idle sleep (`CPU_THROTTLE_RATIO`) so that training does not freeze your workstation.
- **🛠️ Zero-Config CUDA DLL Hook**: Automatically locates and hooks `nvvm` and `cudart` from pip-installed NVIDIA packages (`nvidia-cuda-nvcc-cu12`), removing the requirement for a heavy system-level CUDA Toolkit install.
- **💻 Automatic CPU Fallback**: Automatically switches to vectorised NumPy CPU implementations if an NVIDIA GPU is not detected.

---

## 📐 Network Architecture

The network adheres to the **AlexNet** (Krizhevsky et al., 2012) standard input dimensions ($227 \times 227 \times 3$) and incorporates modern Batch Normalization:

```
Input: (N, 3, 227, 227)
   │
   ├─► [Conv2D]        3  -> 96   (Kernel: 11x11, Stride: 4, Pad: 0)  ──► (N, 96, 55, 55)
   ├─► [BatchNorm2D]   96 channels
   ├─► [ReLU]
   ├─► [MaxPool2D]     Kernel: 3x3, Stride: 2                          ──► (N, 96, 27, 27)
   │
   ├─► [Conv2D]        96 -> 256  (Kernel: 5x5,   Stride: 1, Pad: 2)  ──► (N, 256, 27, 27)
   ├─► [BatchNorm2D]   256 channels
   ├─► [ReLU]
   ├─► [MaxPool2D]     Kernel: 3x3, Stride: 2                          ──► (N, 256, 13, 13)
   │
   ├─► [Conv2D]        256 -> 384 (Kernel: 3x3,   Stride: 1, Pad: 1)  ──► (N, 384, 13, 13)
   ├─► [ReLU]
   │
   ├─► [Conv2D]        384 -> 384 (Kernel: 3x3,   Stride: 1, Pad: 1)  ──► (N, 384, 13, 13)
   ├─► [ReLU]
   │
   ├─► [Conv2D]        384 -> 256 (Kernel: 3x3,   Stride: 1, Pad: 1)  ──► (N, 256, 13, 13)
   ├─► [ReLU]
   ├─► [MaxPool2D]     Kernel: 3x3, Stride: 2                          ──► (N, 256, 6, 6)
   │
   ├─► [Flatten]       256 * 6 * 6                                     ──► (N, 9216)
   │
   ├─► [Dense]         9216 -> 4096
   ├─► [ReLU]
   ├─► [Dropout]       p = 0.5
   │
   ├─► [Dense]         4096 -> 4096
   ├─► [ReLU]
   ├─► [Dropout]       p = 0.5
   │
   ├─► [Dense]         4096 -> 10 (or num_classes)                     ──► (N, 10)
   └─► [SoftmaxCrossEntropy]
```

### Layer Dimension Summary

| Layer | Type | Input Shape | Filter / Units | Stride | Pad | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **01** | `Conv2D` + `BN` + `ReLU` | $(N, 3, 227, 227)$ | $96 \times 3 \times 11 \times 11$ | 4 | 0 | $(N, 96, 55, 55)$ | $\sim 35\text{K}$ |
| **02** | `MaxPool2D` | $(N, 96, 55, 55)$ | $3 \times 3$ | 2 | 0 | $(N, 96, 27, 27)$ | — |
| **03** | `Conv2D` + `BN` + `ReLU` | $(N, 96, 27, 27)$ | $256 \times 96 \times 5 \times 5$ | 1 | 2 | $(N, 256, 27, 27)$ | $\sim 614\text{K}$ |
| **04** | `MaxPool2D` | $(N, 256, 27, 27)$ | $3 \times 3$ | 2 | 0 | $(N, 256, 13, 13)$ | — |
| **05** | `Conv2D` + `ReLU` | $(N, 256, 13, 13)$ | $384 \times 256 \times 3 \times 3$ | 1 | 1 | $(N, 384, 13, 13)$ | $\sim 885\text{K}$ |
| **06** | `Conv2D` + `ReLU` | $(N, 384, 13, 13)$ | $384 \times 384 \times 3 \times 3$ | 1 | 1 | $(N, 384, 13, 13)$ | $\sim 1.3\text{M}$ |
| **07** | `Conv2D` + `ReLU` | $(N, 384, 13, 13)$ | $256 \times 384 \times 3 \times 3$ | 1 | 1 | $(N, 256, 13, 13)$ | $\sim 885\text{K}$ |
| **08** | `MaxPool2D` | $(N, 256, 13, 13)$ | $3 \times 3$ | 2 | 0 | $(N, 256, 6, 6)$ | — |
| **09** | `Flatten` | $(N, 256, 6, 6)$ | — | — | — | $(N, 9216)$ | — |
| **10** | `Dense` + `ReLU` + `Dropout` | $(N, 9216)$ | $9216 \to 4096$ | — | — | $(N, 4096)$ | $\sim 37.7\text{M}$ |
| **11** | `Dense` + `ReLU` + `Dropout` | $(N, 4096)$ | $4096 \to 4096$ | — | — | $(N, 4096)$ | $\sim 16.8\text{M}$ |
| **12** | `Dense` (Logits) | $(N, 4096)$ | $4096 \to 10$ | — | — | $(N, 10)$ | $\sim 41\text{K}$ |

---

## 📂 Repository Structure

```
├── alexnet_numpy_cuda.py   # Flagship script: CUDA kernels, throttling, prefetcher, CPU/GPU training
├── alexnet_numpy_t4.py     # CUDA implementation configured specifically for T4 cloud GPUs (Colab/Kaggle)
├── alexnet_pytorch_T4.py   # Full reference architecture and benchmarking counterpart
├── alexnet_npz.txt         # Pretrained weights (.npz) download URL
├── README.md               # Project documentation
└── LICENSE                 # MIT License
```

---

## ⚡ Custom CUDA Kernels

All forward and backward passes are dispatched directly onto the GPU through JIT-compiled Numba CUDA kernels.

### 1. Convolution 2D (`_conv2d_*`)
- **Forward**: Maps output coordinates $(n, c_{\text{out}}, h_{\text{out}}, w_{\text{out}})$ to a 3D thread grid. Computes the receptive field dot product against input channels and kernels directly on device memory.
- **Backward**: Derives exact analytical gradients:
  - $\frac{\partial L}{\partial W}$: Outer products accumulated over mini-batches and receptive field coordinates.
  - $\frac{\partial L}{\partial b}$: Spatial sum reduction across batch and feature map dimensions.
  - $\frac{\partial L}{\partial X}$: Full transposed convolution mapping back through kernel weights and padding.

### 2. Max Pooling 2D (`_maxpool_*`)
- Forward kernel tracks the exact coordinate `(r_max, c_max)` for each pooling window in an argmax mask buffer.
- Backward kernel uses this mask to route incoming gradients exclusively to the winning coordinate in $\mathcal{O}(1)$ time.

### 3. Adam Optimizer on Device Memory (`_adam_update_kernel`)
Parameters are updated in-place on the GPU without round-tripping to host RAM:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 📊 Dataset: ImageNet10

The project uses the **ImageNet10** dataset (a 10-class curated subset of ImageNet comparable to Imagenette).

- **Automated Download**: Fetched seamlessly via `kagglehub` on the first execution:
  ```python
  import kagglehub
  path = kagglehub.dataset_download("liusha249/imagenet10")
  ```
- **Classes**: 10 ImageNet categories with resolution resized to $227 \times 227 \times 3$.
- **No PyTorch Dependency**: Native Pillow + NumPy pipeline handles parsing, resizing, and train/val splitting.

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Kingramy/AlexNet-using-only-numpy-.git
cd AlexNet-using-only-numpy-
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install numpy numba pillow kagglehub
```

### 4. GPU Acceleration Setup (Zero-Hassle via Pip)
You **do not** need a full manual CUDA Toolkit installation. Simply install the official NVIDIA wheels via pip:
```bash
pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12
```
`alexnet_numpy_cuda.py` includes an automatic DLL discovery routine (`_register_cuda_dlls`) that hooks these packages into Numba automatically.

---

## 🎯 Quickstart & Usage

### Train with Desktop Usability Throttling (Recommended for Local PCs)
Runs on GPU if available (otherwise CPU), keeping your desktop responsive:
```bash
python alexnet_numpy_cuda.py
```

### Train on Cloud GPU (e.g. Google Colab / Kaggle T4)
Optimized for high GPU occupancy and throughput:
```bash
python alexnet_numpy_t4.py
```

### Monitor Output
```text
============================================================
  CUDA AlexNet -- Imagenette 227x227 (10 classes) [GPU]
============================================================
[Dataset] Downloading liusha249/imagenet10 from Kaggle...
[ImageNet10] Loading 'train' split (8320 images)...
[ImageNet10] RAM saved: uint8 uses 1.29 GB (vs 5.15 GB for float32)
[GPU] CUDA device detected: NVIDIA GeForce RTX ...
[Model] Building CudaAlexNet (227x227) on GPU ...

  Epoch 01  batch   50  loss=2.1405  acc=24.3%
  Epoch 01  batch  100  loss=1.8920  acc=35.1%

Epoch 01/10  train_loss=1.8214  train_acc=38.4%  val_acc=41.2%
[Model] GPU weights saved to './alexnet_weights.npz'
```

---

## 💾 Pretrained Weights

Pre-trained weights are provided in standard NumPy `.npz` format:

🔗 **[Download AlexNet Pretrained Weights (.npz)](https://drive.google.com/file/d/11OtlQiJhVwnEc2krIPLz4r2EoTopAc33/view?usp=drive_link)** *(Link also available in [`alexnet_npz.txt`](file:///alexnet_npz.txt))*

### Loading Weights in Python
```python
import numpy as np
from alexnet_numpy_cuda import CudaAlexNet

model = CudaAlexNet(num_classes=10)
model.load_weights("alexnet_weights.npz")
print("Weights loaded successfully!")
```

---

## 🛠️ Memory & Speed Optimizations

| Technique | Implementation Details | Impact |
| :--- | :--- | :--- |
| **`uint8` RAM Caching** | Images stored as unsigned 8-bit ints ($0\text{–}255$) in host RAM. Converted to normalised `float32` only per mini-batch. | **75% reduction** in memory footprint ($\sim 1.6\text{ GB}$ vs $\sim 6.4\text{ GB}$). |
| **Asynchronous Prefetching** | Dedicated background `threading.Thread` prepares, augments, and normalizes batch $t+1$ while GPU trains on batch $t$. | Eliminates CPU wait stalls during GPU kernel execution. |
| **In-place Device Operations** | Adam momentum ($m$) and velocity ($v$) buffers reside persistently on GPU VRAM. | Avoids high-latency PCIe host-to-device memory copies. |
| **OS Thread Priority Management** | Sets Windows execution class to `BELOW_NORMAL_PRIORITY_CLASS`. | Prevents system UI micro-stutters and GPU driver display timeouts (TDR). |
| **Process Throttling** | Optional duty cycle idle delay (`CPU_THROTTLE_RATIO = 0.20`). | Maintains stable $\sim 80\text{–}85\%$ GPU utilization without thermal throttling. |

---

## 🖼️ NumPy Data Augmentation Pipeline

Training batches are dynamically augmented without external computer vision frameworks:

- **Random Horizontal Flip**: Mirroring with $p = 0.5$.
- **Random Crop**: Reflection padding followed by spatial crop to target dimensions.
- **Brightness & Contrast**: Random additive and multiplicative photometric shifts.
- **Saturation Scaling**: RGB to Luminance projection followed by random chroma scaling.
- **Cutout Regularization**: Spatial zero-masking to enforce distributed representations.

---

## 📜 Citation & References

```bibtex
@inproceedings{krizhevsky2012imagenet,
  title     = {ImageNet Classification with Deep Convolutional Neural Networks},
  author    = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {25},
  year      = {2012}
}
```

---

## 📄 License

This project is open-source and released under the **[MIT License](file:///LICENSE)**.

```text
Copyright (c) 2026 Ouail Chenafi (Kingramy)
```

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies. See [LICENSE](file:///LICENSE) for the complete text.
