import numpy as np
import os
import math
import site

# ==========================================
# NVVM DLL AUTO-DISCOVERY
# Finds nvvm.dll from pip-installed nvidia-cuda-nvcc-cu12 package
# so Numba can compile CUDA kernels without a full CUDA Toolkit install.
# ==========================================
def _register_nvvm_dll():
    candidates = []
    # Check all site-packages for the pip-installed CUDA nvcc package
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for subpath in [
            os.path.join(sp, 'nvidia', 'cuda_nvcc', 'nvvm', 'bin'),
            os.path.join(sp, 'nvidia', 'cuda_nvcc', 'bin'),
        ]:
            if os.path.isdir(subpath):
                candidates.append(subpath)

    for path in candidates:
        if any(f.lower().startswith('nvvm') and f.lower().endswith('.dll')
               for f in os.listdir(path)):
            os.add_dll_directory(path)
            print(f"[GPU] Registered NVVM DLL directory: {path}")
            return True
    return False

_register_nvvm_dll()

# ==========================================
# GPU SETUP  (Numba CUDA)
# ==========================================
try:
    from numba import cuda
    import numba
    # Attempt to detect a CUDA device
    _GPU_AVAILABLE = len(cuda.gpus) > 0
except Exception:
    _GPU_AVAILABLE = False

if _GPU_AVAILABLE:
    print(f"[GPU] CUDA device detected: {cuda.gpus[0].name.decode()}")
    print(f"[GPU] Numba version: {numba.__version__}")
else:
    print("[GPU] No CUDA device found – running on CPU (NumPy fallback).")


# ==========================================
# 0. DATASET LOADER  (Imagenette via torchvision, 227x227)
# ==========================================
class KaggleImageNet10Loader:
    """
    Downloads and loads the ImageNet10 dataset using kagglehub.
    Stores images as UINT8 (values 0-255) to cap RAM at ~2 GB instead of ~8 GB.
    Float32 normalisation is deferred to batch_iter so only ~5 MB is converted
    at a time during training.  NO PyTorch dependency.
    """
    def __init__(self, img_size=227, val_fraction=0.2, max_samples=None):
        import kagglehub
        self.img_size = img_size
        self.val_fraction = val_fraction
        self.max_samples = max_samples
        print("[Dataset] Downloading liusha249/imagenet10 from Kaggle...")
        self.data_dir = kagglehub.dataset_download("liusha249/imagenet10")
        print(f"[Dataset] Downloaded to {self.data_dir}")

    def _find_class_dir(self):
        """Walk self.data_dir and return the directory that directly holds class sub-dirs."""
        import os
        for root, dirs, _ in os.walk(self.data_dir):
            sub = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
            if len(sub) >= 2:
                for s in sub[:2]:
                    sp = os.path.join(root, s)
                    try:
                        fs = os.listdir(sp)
                        if any(f.lower().endswith(('.png','.jpg','.jpeg','.bmp')) for f in fs):
                            return root
                    except Exception:
                        pass
        return self.data_dir

    def _collect_tasks(self, class_dir: str):
        """Return [(img_path, label_idx), ...] for every image under class_dir."""
        import os
        classes = sorted([d for d in os.listdir(class_dir)
                          if os.path.isdir(os.path.join(class_dir, d))])
        class_to_idx = {c: i for i, c in enumerate(classes)}
        tasks = []
        for cls_name in classes:
            cls_path = os.path.join(class_dir, cls_name)
            for fn in os.listdir(cls_path):
                if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    tasks.append((os.path.join(cls_path, fn), class_to_idx[cls_name]))
        return tasks

    def _load_tasks(self, tasks: list, split_name: str):
        """Load tasks into a pre-allocated UINT8 array (1/4 the RAM of float32)."""
        import numpy as np
        import gc
        from PIL import Image

        n = len(tasks)
        print(f"[ImageNet10] Loading '{split_name}' split ({n} images)...")
        # uint8: 10400 * 3 * 227 * 227 * 1 byte = 1.61 GB  (vs 6.43 GB for float32)
        X = np.empty((n, 3, self.img_size, self.img_size), dtype=np.uint8)
        y = np.empty((n,), dtype=np.int64)

        count = 0
        for img_path, label in tasks:
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
                    # Transpose HWC -> CHW, store as uint8 (no float conversion yet)
                    X[count] = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
                    y[count] = label
                    count += 1
                    if count % 1000 == 0:
                        print(f"  ... loaded {count}/{n}", flush=True)
            except Exception:
                pass

        X = X[:count]
        y = y[:count]
        gc.collect()
        print(f"[ImageNet10] '{split_name}' done -> X={X.shape} dtype={X.dtype}  y={y.shape}")
        return X, y

    def load(self):
        import random

        class_dir  = self._find_class_dir()
        all_tasks  = self._collect_tasks(class_dir)

        if self.max_samples and self.max_samples < len(all_tasks):
            random.seed(42)
            random.shuffle(all_tasks)
            all_tasks = all_tasks[:self.max_samples]

        # Split the PATH LIST before loading – peak RAM = size of train split only
        random.seed(42)
        random.shuffle(all_tasks)
        n_val   = max(1, int(len(all_tasks) * self.val_fraction))
        n_train = len(all_tasks) - n_val

        X_train, y_train = self._load_tasks(all_tasks[:n_train], 'train')
        X_test,  y_test  = self._load_tasks(all_tasks[n_train:], 'val')

        print(f"[ImageNet10] Load complete -> X_train={X_train.shape}  X_test={X_test.shape}")
        print(f"[ImageNet10] RAM saved: uint8 uses {X_train.nbytes/1e9:.2f} GB "
              f"(vs {X_train.nbytes*4/1e9:.2f} GB for float32)")
        return X_train, y_train, X_test, y_test
# ==========================================
# 1. IMAGE PREPROCESSING
# ==========================================
class ImagePreprocessor:
    """All methods expect images in shape (N, C, H, W) float32 [0,1]."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    @staticmethod
    def bilinear_resize_single(img, new_h, new_w):
        """Resize a single (C, H, W) image to (C, new_h, new_w) using bilinear interpolation."""
        C, H, W = img.shape
        row_scale = H / new_h
        col_scale = W / new_w
        r_src = (np.arange(new_h) + 0.5) * row_scale - 0.5
        c_src = (np.arange(new_w) + 0.5) * col_scale - 0.5
        r0 = np.clip(np.floor(r_src).astype(np.int32), 0, H - 2)
        c0 = np.clip(np.floor(c_src).astype(np.int32), 0, W - 2)
        r1 = r0 + 1
        c1 = c0 + 1
        dr = (r_src - r0).reshape(-1, 1)
        dc = (c_src - c0).reshape(1, -1)
        out = np.empty((C, new_h, new_w), dtype=np.float32)
        for ch in range(C):
            tl = img[ch][r0[:, None], c0[None, :]]
            tr = img[ch][r0[:, None], c1[None, :]]
            bl = img[ch][r1[:, None], c0[None, :]]
            br = img[ch][r1[:, None], c1[None, :]]
            out[ch] = (tl * (1 - dr) * (1 - dc) +
                       tr * (1 - dr) * dc +
                       bl * dr       * (1 - dc) +
                       br * dr       * dc)
        return out

    @classmethod
    def resize_batch(cls, X, new_h, new_w):
        """Resize a batch (N, C, H, W) -> (N, C, new_h, new_w)."""
        N = X.shape[0]
        out = np.empty((N, X.shape[1], new_h, new_w), dtype=np.float32)
        for i in range(N):
            out[i] = cls.bilinear_resize_single(X[i], new_h, new_w)
        return out

    @classmethod
    def normalize(cls, X):
        """Per-channel mean / std normalisation."""
        return (X - cls.MEAN) / cls.STD

    @classmethod
    def denormalize(cls, X):
        """Invert normalisation (for visualisation)."""
        return np.clip(X * cls.STD + cls.MEAN, 0, 1)

    @classmethod
    def preprocess(cls, X, target_h=227, target_w=227):
        """Full preprocessing pipeline: float32 [0,1] -> resize -> normalize."""
        if X.shape[2] != target_h or X.shape[3] != target_w:
            print(f"[Preprocess] Resizing {X.shape[2]}x{X.shape[3]} -> {target_h}x{target_w} ...")
            X = cls.resize_batch(X, target_h, target_w)
        X = cls.normalize(X)
        return X


# ==========================================
# 2. IMAGE AUGMENTATION  (pure NumPy)
# ==========================================
class ImageAugmentor:
    """Random augmentations applied only during training. All methods accept/return (N, C, H, W) float32."""

    @staticmethod
    def random_horizontal_flip(X, prob=0.5):
        """Flip each image left-right with probability prob."""
        mask = np.random.rand(X.shape[0]) < prob
        out  = X.copy()
        out[mask] = X[mask, :, :, ::-1]
        return out

    @staticmethod
    def random_crop(X, crop_h, crop_w, pad=4):
        """Pad by pad pixels on each side then take a random crop of (crop_h, crop_w)."""
        N, C, H, W = X.shape
        Xp = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='reflect')
        _, _, pH, pW = Xp.shape
        out = np.empty((N, C, crop_h, crop_w), dtype=X.dtype)
        for i in range(N):
            top  = np.random.randint(0, pH - crop_h + 1)
            left = np.random.randint(0, pW - crop_w + 1)
            out[i] = Xp[i, :, top:top+crop_h, left:left+crop_w]
        return out

    @staticmethod
    def random_brightness(X, delta=0.2):
        """Add a random brightness offset in [-delta, +delta]."""
        shifts = np.random.uniform(-delta, delta, (X.shape[0], 1, 1, 1)).astype(np.float32)
        return np.clip(X + shifts, -3.0, 3.0)

    @staticmethod
    def random_contrast(X, lo=0.8, hi=1.2):
        """Scale contrast by a random factor in [lo, hi]."""
        factors = np.random.uniform(lo, hi, (X.shape[0], 1, 1, 1)).astype(np.float32)
        means   = X.mean(axis=(2, 3), keepdims=True)
        return np.clip((X - means) * factors + means, -3.0, 3.0)

    @staticmethod
    def random_saturation(X, lo=0.8, hi=1.2):
        """Scale saturation by a random factor (per image)."""
        grey = (0.299 * X[:, 0:1] + 0.587 * X[:, 1:2] + 0.114 * X[:, 2:3])
        factors = np.random.uniform(lo, hi, (X.shape[0], 1, 1, 1)).astype(np.float32)
        return np.clip(grey + factors * (X - grey), -3.0, 3.0)

    @staticmethod
    def random_rotation(X, max_angle=15):
        """Rotate each image by a random angle in [-max_angle, +max_angle] degrees."""
        N, C, H, W = X.shape
        out = np.zeros_like(X)
        cy, cx = H / 2.0, W / 2.0
        angles = np.random.uniform(-max_angle, max_angle, N)
        for i, angle in enumerate(angles):
            theta = np.radians(angle)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            gy, gx = np.mgrid[0:H, 0:W]
            ry = (gy - cy) * cos_t + (gx - cx) * sin_t + cy
            rx = -(gy - cy) * sin_t + (gx - cx) * cos_t + cx
            ry = np.clip(np.round(ry).astype(np.int32), 0, H - 1)
            rx = np.clip(np.round(rx).astype(np.int32), 0, W - 1)
            out[i] = X[i][:, ry, rx]
        return out

    @staticmethod
    def cutout(X, size=32):
        """Zero out a random (size x size) square patch in each image."""
        N, C, H, W = X.shape
        out = X.copy()
        for i in range(N):
            cy = np.random.randint(0, H)
            cx = np.random.randint(0, W)
            y1 = max(0, cy - size // 2)
            y2 = min(H, cy + size // 2)
            x1 = max(0, cx - size // 2)
            x2 = min(W, cx + size // 2)
            out[i, :, y1:y2, x1:x2] = 0.0
        return out

    @classmethod
    def augment(cls, X, crop_h=None, crop_w=None, pad=4,
                flip_prob=0.5, brightness=0.2, contrast_range=(0.8, 1.2),
                saturation_range=(0.8, 1.2), max_rotation=15,
                cutout_size=32, use_cutout=True):
        """Apply a random combination of augmentations to a batch. Call only on training batches."""
        H, W = X.shape[2], X.shape[3]
        crop_h = crop_h or H
        crop_w = crop_w or W
        X = cls.random_horizontal_flip(X, prob=flip_prob)
        X = cls.random_crop(X, crop_h, crop_w, pad=pad)
        X = cls.random_brightness(X, delta=brightness)
        X = cls.random_contrast(X, lo=contrast_range[0], hi=contrast_range[1])
        X = cls.random_saturation(X, lo=saturation_range[0], hi=saturation_range[1])
        X = cls.random_rotation(X, max_angle=max_rotation)
        if use_cutout:
            X = cls.cutout(X, size=cutout_size)
        return X


# ==========================================
# NUMBA CUDA KERNELS
# ==========================================

if _GPU_AVAILABLE:

    @cuda.jit
    def _relu_forward_kernel(x, out):
        """Element-wise ReLU: out[i] = max(0, x[i])."""
        idx = cuda.grid(1)
        if idx < x.size:
            val = x.flat[idx]
            out.flat[idx] = val if val > 0.0 else 0.0

    @cuda.jit
    def _relu_backward_kernel(dout, x, dx):
        """Element-wise ReLU gradient."""
        idx = cuda.grid(1)
        if idx < dout.size:
            dx.flat[idx] = dout.flat[idx] if x.flat[idx] > 0.0 else 0.0

    @cuda.jit
    def _conv2d_forward_kernel(x_pad, W, b, out, K, S, H_out, W_out):
        """
        Grid shape: (batch, out_channels, H_out * W_out)
        Each thread computes one output element out[b, c_out, h, w].
        """
        b_idx  = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        c_out  = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        hw_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

        N     = out.shape[0]
        C_out = out.shape[1]
        C_in  = x_pad.shape[1]

        if b_idx >= N or c_out >= C_out or hw_idx >= H_out * W_out:
            return

        h  = hw_idx // W_out
        w  = hw_idx  % W_out
        hs = h * S
        ws = w * S

        acc = b[c_out]
        for ci in range(C_in):
            for kh in range(K):
                for kw in range(K):
                    acc += x_pad[b_idx, ci, hs + kh, ws + kw] * W[c_out, ci, kh, kw]
        out[b_idx, c_out, h, w] = acc

    @cuda.jit
    def _conv2d_dW_kernel(x_pad, dout, dW, K, S):
        """Accumulate gradient w.r.t. W. Grid: (c_out, c_in, K*K)."""
        c_out  = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        c_in   = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        kk_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

        C_out = dW.shape[0]
        C_in  = dW.shape[1]
        N     = dout.shape[0]
        H_out = dout.shape[2]
        W_out = dout.shape[3]

        if c_out >= C_out or c_in >= C_in or kk_idx >= K * K:
            return

        kh = kk_idx // K
        kw = kk_idx  % K

        acc = 0.0
        for b_idx in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    acc += x_pad[b_idx, c_in, h * S + kh, w * S + kw] * dout[b_idx, c_out, h, w]
        dW[c_out, c_in, kh, kw] = acc

    @cuda.jit
    def _conv2d_db_kernel(dout, db):
        """Accumulate gradient w.r.t. b. Grid: (c_out,)."""
        c_out = cuda.grid(1)
        C_out = db.shape[0]
        if c_out >= C_out:
            return
        N     = dout.shape[0]
        H_out = dout.shape[2]
        W_out = dout.shape[3]
        acc = 0.0
        for b_idx in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    acc += dout[b_idx, c_out, h, w]
        db[c_out] = acc

    @cuda.jit
    def _conv2d_dx_kernel(dout, W, dx_pad, K, S):
        """Compute gradient w.r.t. padded input. Grid: (batch, c_in, pH*pW)."""
        b_idx  = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        c_in   = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        hw_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

        N     = dx_pad.shape[0]
        C_in  = dx_pad.shape[1]
        pH    = dx_pad.shape[2]
        pW    = dx_pad.shape[3]
        C_out = dout.shape[1]
        H_out = dout.shape[2]
        W_out = dout.shape[3]

        if b_idx >= N or c_in >= C_in or hw_idx >= pH * pW:
            return

        ph = hw_idx // pW
        pw = hw_idx  % pW

        acc = 0.0
        for c_out in range(C_out):
            for kh in range(K):
                for kw in range(K):
                    oh = ph - kh
                    ow = pw - kw
                    if oh >= 0 and ow >= 0 and oh % S == 0 and ow % S == 0:
                        ho = oh // S
                        wo = ow // S
                        if ho < H_out and wo < W_out:
                            acc += W[c_out, c_in, kh, kw] * dout[b_idx, c_out, ho, wo]
        dx_pad[b_idx, c_in, ph, pw] = acc

    @cuda.jit
    def _maxpool_forward_kernel(x, out, mask, K, S):
        """
        Grid: (batch, channels, H_out * W_out).
        mask stores the flat window index of the max element.
        """
        b_idx  = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        c_idx  = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        hw_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

        N     = out.shape[0]
        C     = out.shape[1]
        H_out = out.shape[2]
        W_out = out.shape[3]

        if b_idx >= N or c_idx >= C or hw_idx >= H_out * W_out:
            return

        h  = hw_idx // W_out
        w  = hw_idx  % W_out
        hs = h * S
        ws = w * S

        max_val = x[b_idx, c_idx, hs, ws]
        max_kh  = 0
        max_kw  = 0
        for kh in range(K):
            for kw in range(K):
                val = x[b_idx, c_idx, hs + kh, ws + kw]
                if val > max_val:
                    max_val = val
                    max_kh  = kh
                    max_kw  = kw
        out [b_idx, c_idx, h, w] = max_val
        mask[b_idx, c_idx, h, w] = max_kh * K + max_kw

    @cuda.jit
    def _maxpool_backward_kernel(dout, mask, dx, K, S):
        """Scatter gradients back using the saved mask."""
        b_idx  = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        c_idx  = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        hw_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

        N     = dout.shape[0]
        C     = dout.shape[1]
        H_out = dout.shape[2]
        W_out = dout.shape[3]

        if b_idx >= N or c_idx >= C or hw_idx >= H_out * W_out:
            return

        h      = hw_idx // W_out
        w      = hw_idx  % W_out
        flat_k = mask[b_idx, c_idx, h, w]
        kh     = flat_k // K
        kw     = flat_k  % K
        cuda.atomic.add(dx, (b_idx, c_idx, h * S + kh, w * S + kw),
                        dout[b_idx, c_idx, h, w])

    @cuda.jit
    def _dense_forward_kernel(x, W, b, out):
        """
        x: (N, in_f)  W: (in_f, out_f)  b: (1, out_f)  out: (N, out_f)
        Thread: (row=sample, col=out_feature).
        """
        row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

        N     = x.shape[0]
        in_f  = x.shape[1]
        out_f = W.shape[1]

        if row >= N or col >= out_f:
            return

        acc = 0.0
        for k in range(in_f):
            acc += x[row, k] * W[k, col]
        out[row, col] = acc + b[0, col]

    @cuda.jit
    def _dense_dW_kernel(x, dout, dW):
        """dW = x^T dout. Thread: (in_f, out_f)."""
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        in_f  = x.shape[1]
        out_f = dout.shape[1]
        N     = x.shape[0]
        if i >= in_f or j >= out_f:
            return
        acc = 0.0
        for n in range(N):
            acc += x[n, i] * dout[n, j]
        dW[i, j] = acc

    @cuda.jit
    def _dense_db_kernel(dout, db):
        """db = sum(dout, axis=0). Thread: (out_f,)."""
        j = cuda.grid(1)
        out_f = dout.shape[1]
        N     = dout.shape[0]
        if j >= out_f:
            return
        acc = 0.0
        for n in range(N):
            acc += dout[n, j]
        db[0, j] = acc

    @cuda.jit
    def _dense_dx_kernel(dout, W, dx):
        """dx = dout W^T. Thread: (N, in_f)."""
        row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        N    = dout.shape[0]
        in_f = W.shape[0]
        out_f= W.shape[1]
        if row >= N or col >= in_f:
            return
        acc = 0.0
        for k in range(out_f):
            acc += dout[row, k] * W[col, k]
        dx[row, col] = acc

    @cuda.jit
    def _bn_scale_shift_kernel(x_norm, gamma, beta, out):
        """out = gamma * x_norm + beta. Grid: flat over all elements."""
        idx = cuda.grid(1)
        if idx >= out.size:
            return
        C  = out.shape[1]
        HW = out.shape[2] * out.shape[3]
        c  = (idx // HW) % C
        out.flat[idx] = gamma[0, c, 0, 0] * x_norm.flat[idx] + beta[0, c, 0, 0]

    @cuda.jit
    def _adam_update_kernel(param, grad, m, v, lr, beta1, beta2, eps, t):
        """In-place Adam update for a flat parameter array."""
        idx = cuda.grid(1)
        if idx >= param.size:
            return
        g   = grad.flat[idx]
        mi  = beta1 * m.flat[idx] + (1.0 - beta1) * g
        vi  = beta2 * v.flat[idx] + (1.0 - beta2) * g * g
        m.flat[idx] = mi
        v.flat[idx] = vi
        m_hat = mi / (1.0 - beta1 ** t)
        v_hat = vi / (1.0 - beta2 ** t)
        param.flat[idx] -= lr * m_hat / (math.sqrt(v_hat) + eps)


# ==========================================
# HELPER: kernel launch grid
# ==========================================
def _grid1d(n, threads=256):
    return (math.ceil(n / threads),), (threads,)

def _grid3d(d0, d1, d2, tx=4, ty=4, tz=8):
    bx = math.ceil(d0 / tx)
    by = math.ceil(d1 / ty)
    bz = math.ceil(d2 / tz)
    return (bx, by, bz), (tx, ty, tz)


# ==========================================
# CPU LAYERS  (kept as fallback)
# ==========================================
class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx


class Dropout:
    """Inverted dropout for use during training."""
    def __init__(self, rate=0.5):
        self.rate  = rate
        self.mask  = None
        self.train = True

    def forward(self, x):
        if self.train:
            self.mask = (np.random.rand(*x.shape) > self.rate).astype(np.float32)
            return x * self.mask / (1.0 - self.rate)
        return x

    def backward(self, dout):
        return dout * self.mask / (1.0 - self.rate)


class SoftmaxCrossEntropy:
    def forward(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps    = np.exp(shifted)
        self.probs  = exps / np.sum(exps, axis=1, keepdims=True)
        self.y_true = y_true
        N    = logits.shape[0]
        loss = -np.sum(y_true * np.log(self.probs + 1e-9)) / N
        return loss

    def backward(self):
        N = self.y_true.shape[0]
        return (self.probs - self.y_true) / N


class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)


class Dense:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.params = {'W': self.W, 'b': self.b}
        self.grads  = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}

    def forward(self, x):
        self.x = x
        return np.dot(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        self.grads['W'] = np.dot(self.x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.params['W'].T)


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.K = kernel_size
        self.S = stride
        self.P = padding
        self.W = np.random.randn(out_channels, in_channels, self.K, self.K) \
                 * np.sqrt(2.0 / (in_channels * self.K * self.K))
        self.b = np.zeros(out_channels)
        self.params = {'W': self.W, 'b': self.b}
        self.grads  = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}

    def forward(self, x):
        self.x_pad = np.pad(x, ((0,0),(0,0),(self.P,self.P),(self.P,self.P)), mode='constant')
        self.batch_size, _, self.H_in, self.W_in = x.shape
        H_out = (self.H_in - self.K + 2 * self.P) // self.S + 1
        W_out = (self.W_in - self.K + 2 * self.P) // self.S + 1
        out = np.zeros((self.batch_size, self.out_channels, H_out, W_out), dtype=np.float32)
        for b in range(self.batch_size):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        hs, ws = h * self.S, w * self.S
                        patch = self.x_pad[b, :, hs:hs+self.K, ws:ws+self.K]
                        out[b, c_out, h, w] = (np.sum(patch * self.params['W'][c_out])
                                               + self.params['b'][c_out])
        return out

    def backward(self, dout):
        batch_size, _, H_out, W_out = dout.shape
        dx_pad = np.zeros_like(self.x_pad)
        self.grads['W'].fill(0)
        self.grads['b'].fill(0)
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        hs, ws = h * self.S, w * self.S
                        patch = self.x_pad[b, :, hs:hs+self.K, ws:ws+self.K]
                        self.grads['W'][c_out] += patch * dout[b, c_out, h, w]
                        self.grads['b'][c_out] += dout[b, c_out, h, w]
                        dx_pad[b, :, hs:hs+self.K, ws:ws+self.K] += (
                            self.params['W'][c_out] * dout[b, c_out, h, w])
        if self.P > 0:
            return dx_pad[:, :, self.P:-self.P, self.P:-self.P]
        return dx_pad


class MaxPool2D:
    def __init__(self, kernel_size, stride):
        self.K = kernel_size
        self.S = stride

    def forward(self, x):
        self.x = x
        N, C, H_in, W_in = x.shape
        H_out = (H_in - self.K) // self.S + 1
        W_out = (W_in - self.K) // self.S + 1
        out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for b in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        hs, ws = h * self.S, w * self.S
                        out[b, c, h, w] = np.max(self.x[b, c, hs:hs+self.K, ws:ws+self.K])
        return out

    def backward(self, dout):
        N, C, H_out, W_out = dout.shape
        dx = np.zeros_like(self.x)
        for b in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        hs, ws = h * self.S, w * self.S
                        patch = self.x[b, c, hs:hs+self.K, ws:ws+self.K]
                        mask  = (patch == np.max(patch))
                        dx[b, c, hs:hs+self.K, ws:ws+self.K] += (mask * dout[b, c, h, w])
        return dx


class BatchNorm2D:
    def __init__(self, channels):
        self.gamma  = np.ones ((1, channels, 1, 1), dtype=np.float32)
        self.beta   = np.zeros((1, channels, 1, 1), dtype=np.float32)
        self.params = {'gamma': self.gamma, 'beta': self.beta}
        self.grads  = {'gamma': np.zeros_like(self.gamma), 'beta': np.zeros_like(self.beta)}
        self.eps    = 1e-5

    def forward(self, x):
        self.x    = x
        self.mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        self.var  = np.var (x, axis=(0, 2, 3), keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.params['gamma'] * self.x_norm + self.params['beta']

    def backward(self, dout):
        N = dout.shape[0] * dout.shape[2] * dout.shape[3]
        self.grads['gamma'] = np.sum(dout * self.x_norm, axis=(0,2,3), keepdims=True)
        self.grads['beta']  = np.sum(dout,               axis=(0,2,3), keepdims=True)
        dx_norm = dout * self.params['gamma']
        dvar    = np.sum(dx_norm * (self.x - self.mean) * -0.5
                         * (self.var + self.eps)**-1.5, axis=(0,2,3), keepdims=True)
        dmean   = (np.sum(dx_norm * -1.0 / np.sqrt(self.var + self.eps),
                          axis=(0,2,3), keepdims=True)
                   + dvar * np.sum(-2.0 * (self.x - self.mean),
                                   axis=(0,2,3), keepdims=True) / N)
        return (dx_norm / np.sqrt(self.var + self.eps)
                + dvar * 2.0 * (self.x - self.mean) / N
                + dmean / N)


class Adam:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999):
        self.layers = layers
        self.lr     = lr
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = 1e-8
        self.t      = 0
        self.m      = {}
        self.v      = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                self.m[i] = {k: np.zeros_like(v) for k, v in layer.params.items()}
                self.v[i] = {k: np.zeros_like(v) for k, v in layer.params.items()}

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for key in layer.params:
                    g = layer.grads[key]
                    self.m[i][key] = self.beta1 * self.m[i][key] + (1 - self.beta1) * g
                    self.v[i][key] = self.beta2 * self.v[i][key] + (1 - self.beta2) * g**2
                    m_hat = self.m[i][key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[i][key] / (1 - self.beta2 ** self.t)
                    layer.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ==========================================
# GPU LAYER CLASSES  (Numba CUDA)
# ==========================================

if _GPU_AVAILABLE:

    class CudaReLU:
        """ReLU activation using a Numba CUDA kernel."""
        def __init__(self):
            self._x_dev = None

        def forward(self, x_dev):
            out_dev = cuda.device_array_like(x_dev)
            n = x_dev.size
            grid, block = _grid1d(n)
            _relu_forward_kernel[grid, block](x_dev, out_dev)
            self._x_dev = x_dev
            return out_dev

        def backward(self, dout_dev):
            dx_dev = cuda.device_array_like(dout_dev)
            n = dout_dev.size
            grid, block = _grid1d(n)
            _relu_backward_kernel[grid, block](dout_dev, self._x_dev, dx_dev)
            return dx_dev

    class CudaDropout:
        """Inverted dropout using a NumPy mask (small; overhead is negligible)."""
        def __init__(self, rate=0.5):
            self.rate  = rate
            self.mask  = None
            self.train = True

        def forward(self, x_dev):
            if self.train:
                shape   = x_dev.shape
                mask_np = (np.random.rand(*shape) > self.rate).astype(np.float32)
                self.mask = mask_np
                x_np    = x_dev.copy_to_host()
                out_np  = x_np * mask_np / (1.0 - self.rate)
                return cuda.to_device(np.ascontiguousarray(out_np))
            return x_dev

        def backward(self, dout_dev):
            dout_np = dout_dev.copy_to_host()
            dx_np   = dout_np * self.mask / (1.0 - self.rate)
            return cuda.to_device(np.ascontiguousarray(dx_np))

    class CudaFlatten:
        """Flatten NCHW -> N*(C*H*W) operating on device arrays."""
        def __init__(self):
            self._input_shape = None

        def forward(self, x_dev):
            self._input_shape = x_dev.shape
            N    = x_dev.shape[0]
            flat = int(np.prod(x_dev.shape[1:]))
            x_np = x_dev.copy_to_host()
            return cuda.to_device(np.ascontiguousarray(x_np.reshape(N, flat)))

        def backward(self, dout_dev):
            dout_np = dout_dev.copy_to_host()
            return cuda.to_device(np.ascontiguousarray(dout_np.reshape(self._input_shape)))

    class CudaDense:
        """Fully-connected layer with CUDA kernels for forward and backward."""
        def __init__(self, in_features, out_features):
            W_np = (np.random.randn(in_features, out_features)
                    * np.sqrt(2.0 / in_features)).astype(np.float32)
            b_np = np.zeros((1, out_features), dtype=np.float32)

            self._W_dev  = cuda.to_device(W_np)
            self._b_dev  = cuda.to_device(b_np)
            self._dW_dev = cuda.to_device(np.zeros_like(W_np))
            self._db_dev = cuda.to_device(np.zeros_like(b_np))

            self.params = {'W': W_np, 'b': b_np}
            self.grads  = {'W': np.zeros_like(W_np), 'b': np.zeros_like(b_np)}

        def _sync_params_to_device(self):
            self._W_dev.copy_to_device(np.ascontiguousarray(self.params['W']))
            self._b_dev.copy_to_device(np.ascontiguousarray(self.params['b']))

        def _sync_grads_to_host(self):
            self.grads['W'] = self._dW_dev.copy_to_host()
            self.grads['b'] = self._db_dev.copy_to_host()

        def forward(self, x_dev):
            self._x_dev = x_dev
            N     = x_dev.shape[0]
            out_f = self._W_dev.shape[1]
            out_dev = cuda.device_array((N, out_f), dtype=np.float32)
            bx = math.ceil(N     / 16)
            by = math.ceil(out_f / 16)
            _dense_forward_kernel[(bx, by), (16, 16)](
                x_dev, self._W_dev, self._b_dev, out_dev)
            return out_dev

        def backward(self, dout_dev):
            N     = self._x_dev.shape[0]
            in_f  = self._W_dev.shape[0]
            out_f = self._W_dev.shape[1]

            bx = math.ceil(in_f  / 16)
            by = math.ceil(out_f / 16)
            _dense_dW_kernel[(bx, by), (16, 16)](self._x_dev, dout_dev, self._dW_dev)

            g1d, b1d = _grid1d(out_f)
            _dense_db_kernel[g1d, b1d](dout_dev, self._db_dev)

            dx_dev = cuda.device_array((N, in_f), dtype=np.float32)
            bx2 = math.ceil(N    / 16)
            by2 = math.ceil(in_f / 16)
            _dense_dx_kernel[(bx2, by2), (16, 16)](dout_dev, self._W_dev, dx_dev)

            self._sync_grads_to_host()
            return dx_dev

    class CudaConv2D:
        """Conv2D layer with CUDA kernels for forward and backward passes."""
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
            self.in_channels  = in_channels
            self.out_channels = out_channels
            self.K = kernel_size
            self.S = stride
            self.P = padding

            W_np = (np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
                    * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))).astype(np.float32)
            b_np = np.zeros(out_channels, dtype=np.float32)

            self._W_dev  = cuda.to_device(W_np)
            self._b_dev  = cuda.to_device(b_np)
            self._dW_dev = cuda.to_device(np.zeros_like(W_np))
            self._db_dev = cuda.to_device(np.zeros_like(b_np))

            self.params = {'W': W_np, 'b': b_np}
            self.grads  = {'W': np.zeros_like(W_np), 'b': np.zeros_like(b_np)}

        def _sync_params_to_device(self):
            self._W_dev.copy_to_device(np.ascontiguousarray(self.params['W']))
            self._b_dev.copy_to_device(np.ascontiguousarray(self.params['b']))

        def _sync_grads_to_host(self):
            self.grads['W'] = self._dW_dev.copy_to_host()
            self.grads['b'] = self._db_dev.copy_to_host()

        def forward(self, x_dev):
            x_np = x_dev.copy_to_host()
            x_pad_np = np.pad(x_np,
                              ((0,0),(0,0),(self.P,self.P),(self.P,self.P)),
                              mode='constant').astype(np.float32)
            self._x_pad_dev = cuda.to_device(x_pad_np)

            N, _, H_in, W_in = x_np.shape
            H_out = (H_in - self.K + 2 * self.P) // self.S + 1
            W_out = (W_in - self.K + 2 * self.P) // self.S + 1
            self._H_out = H_out
            self._W_out = W_out

            out_dev = cuda.device_array(
                (N, self.out_channels, H_out, W_out), dtype=np.float32)
            grid, block = _grid3d(N, self.out_channels, H_out * W_out)
            _conv2d_forward_kernel[grid, block](
                self._x_pad_dev, self._W_dev, self._b_dev, out_dev,
                self.K, self.S, H_out, W_out)
            return out_dev

        def backward(self, dout_dev):
            N    = self._x_pad_dev.shape[0]
            C_in = self.in_channels
            K    = self.K

            grid_dW, block_dW = _grid3d(self.out_channels, C_in, K * K)
            _conv2d_dW_kernel[grid_dW, block_dW](
                self._x_pad_dev, dout_dev, self._dW_dev, K, self.S)

            g1, b1 = _grid1d(self.out_channels)
            _conv2d_db_kernel[g1, b1](dout_dev, self._db_dev)

            pH = self._x_pad_dev.shape[2]
            pW = self._x_pad_dev.shape[3]
            dx_pad_np = np.zeros(self._x_pad_dev.shape, dtype=np.float32)
            dx_pad_dev = cuda.to_device(dx_pad_np)
            grid_dx, block_dx = _grid3d(N, C_in, pH * pW)
            _conv2d_dx_kernel[grid_dx, block_dx](
                dout_dev, self._W_dev, dx_pad_dev, K, self.S)

            self._sync_grads_to_host()

            if self.P > 0:
                dx_full = dx_pad_dev.copy_to_host()
                return cuda.to_device(
                    np.ascontiguousarray(dx_full[:, :, self.P:-self.P, self.P:-self.P]))
            return dx_pad_dev

    class CudaMaxPool2D:
        """MaxPool2D with CUDA kernels."""
        def __init__(self, kernel_size, stride):
            self.K = kernel_size
            self.S = stride
            self._mask_dev = None
            self._x_shape  = None

        def forward(self, x_dev):
            self._x_shape = x_dev.shape
            N, C, H_in, W_in = x_dev.shape
            H_out = (H_in - self.K) // self.S + 1
            W_out = (W_in - self.K) // self.S + 1

            out_dev  = cuda.device_array((N, C, H_out, W_out), dtype=np.float32)
            mask_dev = cuda.device_array((N, C, H_out, W_out), dtype=np.int32)
            self._mask_dev = mask_dev

            grid, block = _grid3d(N, C, H_out * W_out)
            _maxpool_forward_kernel[grid, block](
                x_dev, out_dev, mask_dev, self.K, self.S)
            return out_dev

        def backward(self, dout_dev):
            N, C, H_in, W_in = self._x_shape
            dx_dev = cuda.to_device(np.zeros(self._x_shape, dtype=np.float32))
            H_out = dout_dev.shape[2]
            W_out = dout_dev.shape[3]
            grid, block = _grid3d(N, C, H_out * W_out)
            _maxpool_backward_kernel[grid, block](
                dout_dev, self._mask_dev, dx_dev, self.K, self.S)
            return dx_dev

    class CudaBatchNorm2D:
        """
        BatchNorm2D — statistics on CPU, scale/shift on GPU.
        Hybrid approach: keeps numerical accuracy while leveraging GPU for the
        per-element scale-shift that dominates cost at large spatial sizes.
        """
        def __init__(self, channels):
            gamma_np = np.ones ((1, channels, 1, 1), dtype=np.float32)
            beta_np  = np.zeros((1, channels, 1, 1), dtype=np.float32)

            self._gamma_dev = cuda.to_device(gamma_np)
            self._beta_dev  = cuda.to_device(beta_np)
            self.params = {'gamma': gamma_np, 'beta': beta_np}
            self.grads  = {'gamma': np.zeros_like(gamma_np),
                           'beta':  np.zeros_like(beta_np)}
            self.eps        = 1e-5
            self._x_norm_dev = None
            self._x_np       = None
            self._mean       = None
            self._var        = None

        def _sync_params_to_device(self):
            self._gamma_dev.copy_to_device(np.ascontiguousarray(self.params['gamma']))
            self._beta_dev.copy_to_device(np.ascontiguousarray(self.params['beta']))

        def forward(self, x_dev):
            x_np       = x_dev.copy_to_host()
            self._x_np = x_np
            self._mean = np.mean(x_np, axis=(0, 2, 3), keepdims=True)
            self._var  = np.var (x_np, axis=(0, 2, 3), keepdims=True)
            x_norm_np  = (x_np - self._mean) / np.sqrt(self._var + self.eps)
            self._x_norm_dev = cuda.to_device(np.ascontiguousarray(x_norm_np))

            out_dev = cuda.device_array_like(self._x_norm_dev)
            n = out_dev.size
            g, b = _grid1d(n)
            _bn_scale_shift_kernel[g, b](
                self._x_norm_dev, self._gamma_dev, self._beta_dev, out_dev)
            return out_dev

        def backward(self, dout_dev):
            dout_np   = dout_dev.copy_to_host()
            x_np      = self._x_np
            x_norm_np = self._x_norm_dev.copy_to_host()
            N_pixels  = dout_np.shape[0] * dout_np.shape[2] * dout_np.shape[3]

            self.grads['gamma'] = np.sum(dout_np * x_norm_np, axis=(0,2,3), keepdims=True)
            self.grads['beta']  = np.sum(dout_np,             axis=(0,2,3), keepdims=True)

            gamma_np = self.params['gamma']
            dx_norm  = dout_np * gamma_np
            dvar     = np.sum(dx_norm * (x_np - self._mean) * -0.5
                              * (self._var + self.eps)**-1.5,
                              axis=(0,2,3), keepdims=True)
            dmean    = (np.sum(dx_norm * -1.0 / np.sqrt(self._var + self.eps),
                               axis=(0,2,3), keepdims=True)
                        + dvar * np.sum(-2.0 * (x_np - self._mean),
                                        axis=(0,2,3), keepdims=True) / N_pixels)
            dx_np = (dx_norm / np.sqrt(self._var + self.eps)
                     + dvar * 2.0 * (x_np - self._mean) / N_pixels
                     + dmean / N_pixels)
            return cuda.to_device(np.ascontiguousarray(dx_np))

    class CudaSoftmaxCrossEntropy:
        """Softmax cross-entropy computed on the host (small N x C tensor)."""
        def forward(self, logits_dev, y_true_dev):
            logits = logits_dev.copy_to_host()
            y_true = y_true_dev.copy_to_host()
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exps    = np.exp(shifted)
            self.probs  = exps / np.sum(exps, axis=1, keepdims=True)
            self.y_true = y_true
            N    = logits.shape[0]
            loss = -np.sum(y_true * np.log(self.probs + 1e-9)) / N
            return float(loss)

        def backward(self):
            N = self.y_true.shape[0]
            grad_np = (self.probs - self.y_true) / N
            return cuda.to_device(np.ascontiguousarray(grad_np.astype(np.float32)))

    class CudaAdam:
        """Adam optimiser with per-parameter CUDA update kernels."""
        def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999):
            self.layers = layers
            self.lr     = lr
            self.beta1  = beta1
            self.beta2  = beta2
            self.eps    = 1e-8
            self.t      = 0
            self._m  = {}
            self._v  = {}
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'params'):
                    self._m[i] = {}
                    self._v[i] = {}
                    for k, v in layer.params.items():
                        self._m[i][k] = cuda.to_device(
                            np.zeros_like(v, dtype=np.float32))
                        self._v[i][k] = cuda.to_device(
                            np.zeros_like(v, dtype=np.float32))

        def step(self):
            self.t += 1
            for i, layer in enumerate(self.layers):
                if not hasattr(layer, 'params'):
                    continue
                for key in layer.params:
                    grad_dev  = cuda.to_device(
                        np.ascontiguousarray(layer.grads[key].astype(np.float32)))
                    param_dev = cuda.to_device(
                        np.ascontiguousarray(layer.params[key].astype(np.float32)))

                    n = param_dev.size
                    g, b = _grid1d(n)
                    _adam_update_kernel[g, b](
                        param_dev, grad_dev,
                        self._m[i][key], self._v[i][key],
                        self.lr, self.beta1, self.beta2, self.eps,
                        float(self.t)
                    )
                    layer.params[key][:] = param_dev.copy_to_host()

                    # Sync updated CPU params back to device buffers in Cuda layers
                    if hasattr(layer, '_sync_params_to_device'):
                        layer._sync_params_to_device()


# ==========================================
# ALEXNET MODELS
# ==========================================

class NumPyAlexNet:
    """
    Original AlexNet architecture for 227x227 RGB images.
    WARNING: Pure-NumPy convolutions on 227x227 are very slow.
             Use CudaAlexNet when a GPU is available.
    """
    def __init__(self, num_classes=10):
        self.dropout1 = Dropout(rate=0.5)
        self.dropout2 = Dropout(rate=0.5)
        self.layers = [
            Conv2D(3,  96, kernel_size=11, stride=4, padding=0),
            BatchNorm2D(96),  ReLU(),
            MaxPool2D(kernel_size=3, stride=2),
            Conv2D(96, 256, kernel_size=5, stride=1, padding=2),
            BatchNorm2D(256), ReLU(),
            MaxPool2D(kernel_size=3, stride=2),
            Conv2D(256, 384, kernel_size=3, stride=1, padding=1), ReLU(),
            Conv2D(384, 384, kernel_size=3, stride=1, padding=1), ReLU(),
            Conv2D(384, 256, kernel_size=3, stride=1, padding=1), ReLU(),
            MaxPool2D(kernel_size=3, stride=2),
            Flatten(),
            Dense(9216, 4096), ReLU(), self.dropout1,
            Dense(4096, 4096), ReLU(), self.dropout2,
            Dense(4096, num_classes),
        ]
        self.loss_fn = SoftmaxCrossEntropy()

    def set_train(self, mode=True):
        self.dropout1.train = mode
        self.dropout2.train = mode

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true):
        dout = self.loss_fn.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)


class SmallAlexNet:
    """AlexNet adapted for 96x96 RGB images (STL-10 native resolution)."""
    def __init__(self, num_classes=10):
        self.dropout1 = Dropout(rate=0.5)
        self.dropout2 = Dropout(rate=0.5)
        self.layers = [
            Conv2D(3,  96, kernel_size=7, stride=2, padding=0),
            BatchNorm2D(96),  ReLU(),
            MaxPool2D(kernel_size=3, stride=2),
            Conv2D(96, 256, kernel_size=5, stride=1, padding=2),
            BatchNorm2D(256), ReLU(),
            MaxPool2D(kernel_size=3, stride=2),
            Conv2D(256, 384, kernel_size=3, stride=1, padding=1), ReLU(),
            Conv2D(384, 384, kernel_size=3, stride=1, padding=1), ReLU(),
            Conv2D(384, 256, kernel_size=3, stride=1, padding=1), ReLU(),
            MaxPool2D(kernel_size=3, stride=2),
            Flatten(),
            Dense(6400, 4096), ReLU(), self.dropout1,
            Dense(4096, 4096), ReLU(), self.dropout2,
            Dense(4096, num_classes),
        ]
        self.loss_fn = SoftmaxCrossEntropy()

    def set_train(self, mode=True):
        self.dropout1.train = mode
        self.dropout2.train = mode

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true):
        dout = self.loss_fn.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def save_weights(self, path):
        """Save all learnable parameters to a .npz file."""
        arrays = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for k, v in layer.params.items():
                    arrays[f"layer{i}_{k}"] = v
        np.savez(path, **arrays)
        print(f"[Model] Weights saved to '{path}'")

    def load_weights(self, path):
        """Load parameters from a .npz file saved by save_weights()."""
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for k in layer.params:
                    key = f"layer{i}_{k}"
                    if key in data:
                        layer.params[k][:] = data[key]
        print(f"[Model] Weights loaded from '{path}'")


if _GPU_AVAILABLE:

    class CudaAlexNet:
        """
        AlexNet for 227x227 RGB input, fully accelerated with Numba CUDA.

        Architecture (identical to NumPyAlexNet):
            CudaConv(3->96, k=11, s=4) -> CudaBN -> CudaReLU -> CudaMaxPool(3,2)
            CudaConv(96->256, k=5, p=2) -> CudaBN -> CudaReLU -> CudaMaxPool(3,2)
            CudaConv(256->384, k=3, p=1) -> CudaReLU
            CudaConv(384->384, k=3, p=1) -> CudaReLU
            CudaConv(384->256, k=3, p=1) -> CudaReLU -> CudaMaxPool(3,2)
            CudaFlatten
            CudaDense(9216,4096) -> CudaReLU -> CudaDropout(0.5)
            CudaDense(4096,4096) -> CudaReLU -> CudaDropout(0.5)
            CudaDense(4096, num_classes)
        """

        def __init__(self, num_classes=10):
            self.dropout1 = CudaDropout(rate=0.5)
            self.dropout2 = CudaDropout(rate=0.5)
            self.layers = [
                # Block 1
                CudaConv2D(3,  96, kernel_size=11, stride=4, padding=0),
                CudaBatchNorm2D(96),  CudaReLU(),
                CudaMaxPool2D(kernel_size=3, stride=2),
                # Block 2
                CudaConv2D(96, 256, kernel_size=5, stride=1, padding=2),
                CudaBatchNorm2D(256), CudaReLU(),
                CudaMaxPool2D(kernel_size=3, stride=2),
                # Block 3-5
                CudaConv2D(256, 384, kernel_size=3, stride=1, padding=1), CudaReLU(),
                CudaConv2D(384, 384, kernel_size=3, stride=1, padding=1), CudaReLU(),
                CudaConv2D(384, 256, kernel_size=3, stride=1, padding=1), CudaReLU(),
                CudaMaxPool2D(kernel_size=3, stride=2),
                # FC layers  (256 * 6 * 6 = 9216)
                CudaFlatten(),
                CudaDense(9216, 4096), CudaReLU(), self.dropout1,
                CudaDense(4096, 4096), CudaReLU(), self.dropout2,
                CudaDense(4096, num_classes),
            ]
            self.loss_fn = CudaSoftmaxCrossEntropy()

        def set_train(self, mode=True):
            self.dropout1.train = mode
            self.dropout2.train = mode

        def forward(self, x_dev):
            """
            Parameters
            ----------
            x_dev : numba cuda device array  (N, 3, 227, 227) float32

            Returns
            -------
            logits_dev : cuda device array  (N, num_classes) float32
            """
            out = x_dev
            for layer in self.layers:
                out = layer.forward(out)
            return out

        def backward(self, y_true_dev):
            """
            y_true_dev : cuda device array  (N, num_classes) float32 one-hot
            """
            dout = self.loss_fn.backward()
            for layer in reversed(self.layers):
                dout = layer.backward(dout)

        def save_weights(self, path):
            """Checkpoint all learnable parameters to a .npz file."""
            arrays = {}
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'params'):
                    for k, v in layer.params.items():
                        arrays[f"layer{i}_{k}"] = v
            np.savez(path, **arrays)
            print(f"[Model] GPU weights saved to '{path}'")

        def load_weights(self, path):
            """Restore parameters from a .npz checkpoint."""
            data = np.load(path)
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'params'):
                    for k in layer.params:
                        key = f"layer{i}_{k}"
                        if key in data:
                            layer.params[k][:] = data[key]
                    if hasattr(layer, '_sync_params_to_device'):
                        layer._sync_params_to_device()
            print(f"[Model] GPU weights loaded from '{path}'")


# ==========================================
# HELPER UTILITIES
# ==========================================
def one_hot(labels, num_classes):
    """Convert integer labels (N,) -> one-hot (N, C)."""
    N = labels.shape[0]
    oh = np.zeros((N, num_classes), dtype=np.float32)
    oh[np.arange(N), labels] = 1.0
    return oh


def accuracy(logits, y_true_onehot):
    """Top-1 accuracy given logits and one-hot ground truth."""
    pred  = np.argmax(logits, axis=1)
    truth = np.argmax(y_true_onehot, axis=1)
    return np.mean(pred == truth)


# ImageNet normalisation constants (broadcast-ready)
_IMGNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMGNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def batch_iter(X, y, batch_size, shuffle=True):
    """Mini-batch generator.
    If X is uint8 (stored by KaggleImageNet10Loader), converts each mini-batch
    to normalised float32 on the fly so that only ~5 MB is live at once instead
    of the full dataset.
    """
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, N, batch_size):
        sl  = idx[start:start + batch_size]
        Xb  = X[sl]
        if Xb.dtype == np.uint8:
            Xb = Xb.astype(np.float32) / 255.0
            Xb = (Xb - _IMGNET_MEAN) / _IMGNET_STD
        yield Xb, y[sl]