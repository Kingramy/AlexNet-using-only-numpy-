import numpy as np
import os
import math
import site
import time
import threading
import queue
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

try:
    from numba import cuda
    import numba
    _GPU_AVAILABLE = len(cuda.gpus) > 0
except Exception:
    _GPU_AVAILABLE = False

if _GPU_AVAILABLE:
    print(f"[GPU] CUDA device detected: {cuda.gpus[0].name}")
    print(f"[GPU] Numba version: {numba.__version__}")
else:
    print("[GPU] No CUDA device found – running on CPU (NumPy fallback).")


class KaggleImageNet10Loader:
    def __init__(self, img_size=227, val_fraction=0.2, max_samples=None):
        import kagglehub
        self.img_size = img_size
        self.val_fraction = val_fraction
        self.max_samples = max_samples
        print("[Dataset] Downloading liusha249/imagenet10 from Kaggle...")
        self.data_dir = kagglehub.dataset_download("liusha249/imagenet10")
        print(f"[Dataset] Downloaded to {self.data_dir}")

    def _find_class_dir(self):
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
        import numpy as np
        import gc
        from PIL import Image

        n = len(tasks)
        print(f"[ImageNet10] Loading '{split_name}' split ({n} images)...")
        X = np.empty((n, 3, self.img_size, self.img_size), dtype=np.uint8)
        y = np.empty((n,), dtype=np.int64)

        count = 0
        for img_path, label in tasks:
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
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

        random.seed(42)
        random.shuffle(all_tasks)
        n_val   = max(1, int(len(all_tasks) * self.val_fraction))
        n_train = len(all_tasks) - n_val

        X_train, y_train = self._load_tasks(all_tasks[:n_train], 'train')
        X_test,  y_test  = self._load_tasks(all_tasks[n_train:], 'val')

        print(f"[ImageNet10] Load complete -> X_train={X_train.shape}  X_test={X_test.shape}")
        return X_train, y_train, X_test, y_test


class ImagePreprocessor:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    @staticmethod
    def bilinear_resize_single(img, new_h, new_w):
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
        N = X.shape[0]
        out = np.empty((N, X.shape[1], new_h, new_w), dtype=np.float32)
        for i in range(N):
            out[i] = cls.bilinear_resize_single(X[i], new_h, new_w)
        return out

    @classmethod
    def normalize(cls, X):
        return (X - cls.MEAN) / cls.STD

    @classmethod
    def denormalize(cls, X):
        return np.clip(X * cls.STD + cls.MEAN, 0, 1)

    @classmethod
    def preprocess(cls, X, target_h=227, target_w=227):
        if X.shape[2] != target_h or X.shape[3] != target_w:
            X = cls.resize_batch(X, target_h, target_w)
        X = cls.normalize(X)
        return X


class ImageAugmentor:
    @staticmethod
    def random_horizontal_flip(X, prob=0.5):
        mask = np.random.rand(X.shape[0]) < prob
        out  = X.copy()
        out[mask] = X[mask, :, :, ::-1]
        return out

    @staticmethod
    def random_crop(X, crop_h, crop_w, pad=4):
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
        shifts = np.random.uniform(-delta, delta, (X.shape[0], 1, 1, 1)).astype(np.float32)
        return np.clip(X + shifts, -3.0, 3.0)

    @staticmethod
    def random_contrast(X, lo=0.8, hi=1.2):
        factors = np.random.uniform(lo, hi, (X.shape[0], 1, 1, 1)).astype(np.float32)
        means   = X.mean(axis=(2, 3), keepdims=True)
        return np.clip((X - means) * factors + means, -3.0, 3.0)

    @staticmethod
    def random_saturation(X, lo=0.8, hi=1.2):
        grey = (0.299 * X[:, 0:1] + 0.587 * X[:, 1:2] + 0.114 * X[:, 2:3])
        factors = np.random.uniform(lo, hi, (X.shape[0], 1, 1, 1)).astype(np.float32)
        return np.clip(grey + factors * (X - grey), -3.0, 3.0)

    @staticmethod
    def random_rotation(X, max_angle=15):
        if max_angle == 0:
            return X
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


if _GPU_AVAILABLE:

    @cuda.jit
    def _relu_forward_kernel(x, out):
        idx = cuda.grid(1)
        if idx < x.size:
            val = x.flat[idx]
            out.flat[idx] = val if val > 0.0 else 0.0

    @cuda.jit
    def _relu_backward_kernel(dout, x, dx):
        idx = cuda.grid(1)
        if idx < dout.size:
            dx.flat[idx] = dout.flat[idx] if x.flat[idx] > 0.0 else 0.0

    @cuda.jit
    def _conv2d_forward_kernel(x_pad, W, b, out, K, S, H_out, W_out):
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
        b_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        c_out = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        c_in  = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

        N     = x_pad.shape[0]
        C_out_ = dW.shape[0]
        C_in_  = dW.shape[1]
        H_out = dout.shape[2]
        W_out = dout.shape[3]

        if b_idx >= N or c_out >= C_out_ or c_in >= C_in_:
            return

        for kh in range(K):
            for kw in range(K):
                acc = 0.0
                for h in range(H_out):
                    for w in range(W_out):
                        acc += x_pad[b_idx, c_in, h * S + kh, w * S + kw] * dout[b_idx, c_out, h, w]
                cuda.atomic.add(dW, (c_out, c_in, kh, kw), acc)

    @cuda.jit
    def _conv2d_db_kernel(dout, db):
        b_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        c_out = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        N = dout.shape[0]
        C_out_ = db.shape[0]
        H_out = dout.shape[2]
        W_out = dout.shape[3]

        if b_idx >= N or c_out >= C_out_:
            return

        acc = 0.0
        for h in range(H_out):
            for w in range(W_out):
                acc += dout[b_idx, c_out, h, w]
        cuda.atomic.add(db, c_out, acc)

    @cuda.jit
    def _conv2d_dx_kernel(dout, W, dx_pad, K, S):
        b_idx  = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        c_in   = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        hw_idx = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

        N     = dx_pad.shape[0]
        C_in_ = dx_pad.shape[1]
        pH    = dx_pad.shape[2]
        pW    = dx_pad.shape[3]
        C_out = dout.shape[1]
        H_out = dout.shape[2]
        W_out = dout.shape[3]

        if b_idx >= N or c_in >= C_in_ or hw_idx >= pH * pW:
            return

        ph = hw_idx // pW
        pw = hw_idx  % pW

        ho_min = 0
        if ph >= K:
            ho_min = (ph - K + S) // S
        ho_max = H_out - 1
        if ho_max > ph // S:
            ho_max = ph // S

        wo_min = 0
        if pw >= K:
            wo_min = (pw - K + S) // S
        wo_max = W_out - 1
        if wo_max > pw // S:
            wo_max = pw // S

        acc = 0.0
        for c_out in range(C_out):
            for ho in range(ho_min, ho_max + 1):
                for wo in range(wo_min, wo_max + 1):
                    kh = ph - ho * S
                    kw = pw - wo * S
                    acc += W[c_out, c_in, kh, kw] * dout[b_idx, c_out, ho, wo]
        dx_pad[b_idx, c_in, ph, pw] = acc

    @cuda.jit
    def _maxpool_forward_kernel(x, out, mask, K, S):
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
        idx = cuda.grid(1)
        if idx >= out.size:
            return
        C  = out.shape[1]
        HW = out.shape[2] * out.shape[3]
        c  = (idx // HW) % C
        out.flat[idx] = gamma[0, c, 0, 0] * x_norm.flat[idx] + beta[0, c, 0, 0]

    @cuda.jit
    def _adam_update_kernel(param, grad, m, v, lr, beta1, beta2, eps, t):
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

def _grid1d(n, threads=256):
    return (math.ceil(n / threads),), (threads,)

def _grid3d(d0, d1, d2, tx=4, ty=4, tz=8):
    bx = math.ceil(d0 / tx)
    by = math.ceil(d1 / ty)
    bz = math.ceil(d2 / tz)
    return (bx, by, bz), (tx, ty, tz)


class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx


class Dropout:
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


if _GPU_AVAILABLE:

    class CudaReLU:
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

            self._params_dev = {'W': self._W_dev, 'b': self._b_dev}
            self._grads_dev  = {'W': self._dW_dev, 'b': self._db_dev}

        def forward(self, x_dev):
            self._x_dev = x_dev
            N     = x_dev.shape[0]
            in_f  = self._W_dev.shape[0]
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

            self._dW_dev = cuda.to_device(np.zeros_like(self.params['W']))
            self._db_dev = cuda.to_device(np.zeros_like(self.params['b']))
            self._grads_dev['W'] = self._dW_dev
            self._grads_dev['b'] = self._db_dev

            bx = math.ceil(in_f  / 16)
            by = math.ceil(out_f / 16)
            _dense_dW_kernel[(bx, by), (16, 16)](self._x_dev, dout_dev, self._dW_dev)

            g1d, b1d = _grid1d(out_f)
            _dense_db_kernel[g1d, b1d](dout_dev, self._db_dev)

            dx_dev = cuda.device_array((N, in_f), dtype=np.float32)
            bx2 = math.ceil(N    / 16)
            by2 = math.ceil(in_f / 16)
            _dense_dx_kernel[(bx2, by2), (16, 16)](dout_dev, self._W_dev, dx_dev)

            return dx_dev

    class CudaConv2D:
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
            self.in_channels  = in_channels
            self.out_channels = out_channels
            self.K = kernel_size
            self.S = stride
            self.P = padding

            W_np = (np.random.randn(out_channels, in_channels, self.K, self.K)
                    * np.sqrt(2.0 / (in_channels * self.K * self.K))).astype(np.float32)
            b_np = np.zeros(out_channels, dtype=np.float32)

            self._W_dev  = cuda.to_device(W_np)
            self._b_dev  = cuda.to_device(b_np)
            self._dW_dev = cuda.to_device(np.zeros_like(W_np))
            self._db_dev = cuda.to_device(np.zeros_like(b_np))

            self.params = {'W': W_np, 'b': b_np}
            self.grads  = {'W': np.zeros_like(W_np), 'b': np.zeros_like(b_np)}

            self._params_dev = {'W': self._W_dev, 'b': self._b_dev}
            self._grads_dev  = {'W': self._dW_dev, 'b': self._db_dev}

        def forward(self, x_dev):
            x_np = x_dev.copy_to_host()
            x_pad_np = np.pad(x_np, ((0,0),(0,0),(self.P,self.P),(self.P,self.P)), mode='constant')
            self._x_pad_dev = cuda.to_device(np.ascontiguousarray(x_pad_np))

            N = x_np.shape[0]
            H_in = x_np.shape[2]
            W_in = x_np.shape[3]
            H_out = (H_in - self.K + 2 * self.P) // self.S + 1
            W_out = (W_in - self.K + 2 * self.P) // self.S + 1

            out_dev = cuda.device_array((N, self.out_channels, H_out, W_out), dtype=np.float32)
            grid, block = _grid3d(N, self.out_channels, H_out * W_out)
            _conv2d_forward_kernel[grid, block](
                self._x_pad_dev, self._W_dev, self._b_dev,
                out_dev, self.K, self.S, H_out, W_out)
            return out_dev

        def backward(self, dout_dev):
            N    = self._x_pad_dev.shape[0]
            C_in = self.in_channels
            K    = self.K

            self._dW_dev = cuda.to_device(np.zeros_like(self.params['W']))
            self._db_dev = cuda.to_device(np.zeros_like(self.params['b']))
            self._grads_dev['W'] = self._dW_dev
            self._grads_dev['b'] = self._db_dev

            grid_dW, block_dW = _grid3d(N, self.out_channels, C_in)
            _conv2d_dW_kernel[grid_dW, block_dW](
                self._x_pad_dev, dout_dev, self._dW_dev, K, self.S)

            bx = math.ceil(N / 16)
            by = math.ceil(self.out_channels / 16)
            _conv2d_db_kernel[(bx, by), (16, 16)](dout_dev, self._db_dev)

            pH = self._x_pad_dev.shape[2]
            pW = self._x_pad_dev.shape[3]
            dx_pad_dev = cuda.device_array(self._x_pad_dev.shape, dtype=np.float32)
            grid_dx, block_dx = _grid3d(N, C_in, pH * pW)
            _conv2d_dx_kernel[grid_dx, block_dx](
                dout_dev, self._W_dev, dx_pad_dev, K, self.S)

            if self.P > 0:
                dx_full = dx_pad_dev.copy_to_host()
                return cuda.to_device(
                    np.ascontiguousarray(dx_full[:, :, self.P:-self.P, self.P:-self.P]))
            return dx_pad_dev

    class CudaMaxPool2D:
        def __init__(self, kernel_size, stride):
            self.K = kernel_size
            self.S = stride
            self._x_shape  = None
            self._mask_dev = None

        def forward(self, x_dev):
            self._x_shape = x_dev.shape
            N, C, H_in, W_in = self._x_shape
            H_out = (H_in - self.K) // self.S + 1
            W_out = (W_in - self.K) // self.S + 1

            out_dev = cuda.device_array((N, C, H_out, W_out), dtype=np.float32)
            self._mask_dev = cuda.device_array((N, C, H_out, W_out), dtype=np.int32)

            grid, block = _grid3d(N, C, H_out * W_out)
            _maxpool_forward_kernel[grid, block](
                x_dev, out_dev, self._mask_dev, self.K, self.S)
            return out_dev

        def backward(self, dout_dev):
            dx_dev = cuda.to_device(np.zeros(self._x_shape, dtype=np.float32))
            N, C, H_out, W_out = dout_dev.shape
            grid, block = _grid3d(N, C, H_out * W_out)
            _maxpool_backward_kernel[grid, block](
                dout_dev, self._mask_dev, dx_dev, self.K, self.S)
            return dx_dev

    class CudaBatchNorm2D:
        def __init__(self, channels):
            gamma_np = np.ones ((1, channels, 1, 1), dtype=np.float32)
            beta_np  = np.zeros((1, channels, 1, 1), dtype=np.float32)

            self._gamma_dev = cuda.to_device(gamma_np)
            self._beta_dev  = cuda.to_device(beta_np)

            self.params = {'gamma': gamma_np, 'beta': beta_np}
            self.grads  = {'gamma': np.zeros_like(gamma_np), 'beta': np.zeros_like(beta_np)}

            self._params_dev = {'gamma': self._gamma_dev, 'beta': self._beta_dev}
            self._grads_dev  = {'gamma': cuda.to_device(np.zeros_like(gamma_np)),
                                'beta':  cuda.to_device(np.zeros_like(beta_np))}
            self.eps = 1e-5

        def forward(self, x_dev):
            x_np = x_dev.copy_to_host()
            self._x_np = x_np
            self._mean = np.mean(x_np, axis=(0, 2, 3), keepdims=True)
            self._var  = np.var (x_np, axis=(0, 2, 3), keepdims=True)
            x_norm_np  = (x_np - self._mean) / np.sqrt(self._var + self.eps)
            self._x_norm_dev = cuda.to_device(np.ascontiguousarray(x_norm_np))

            out_dev = cuda.device_array_like(self._x_norm_dev)
            grid, block = _grid1d(out_dev.size)
            _bn_scale_shift_kernel[grid, block](
                self._x_norm_dev, self._gamma_dev, self._beta_dev, out_dev)
            return out_dev

        def backward(self, dout_dev):
            dout_np   = dout_dev.copy_to_host()
            x_np      = self._x_np
            x_norm_np = self._x_norm_dev.copy_to_host()
            N_pixels  = dout_np.shape[0] * dout_np.shape[2] * dout_np.shape[3]

            dgamma = np.sum(dout_np * x_norm_np, axis=(0,2,3), keepdims=True)
            dbeta  = np.sum(dout_np,             axis=(0,2,3), keepdims=True)

            self._grads_dev['gamma'] = cuda.to_device(np.ascontiguousarray(dgamma))
            self._grads_dev['beta']  = cuda.to_device(np.ascontiguousarray(dbeta))

            dx_norm = dout_np * self.params['gamma']
            dvar    = np.sum(dx_norm * (x_np - self._mean) * -0.5
                             * (self._var + self.eps)**-1.5, axis=(0,2,3), keepdims=True)
            dmean   = (np.sum(dx_norm * -1.0 / np.sqrt(self._var + self.eps),
                              axis=(0,2,3), keepdims=True)
                       + dvar * np.sum(-2.0 * (x_np - self._mean),
                                       axis=(0,2,3), keepdims=True) / N_pixels)

            dx_np = (dx_norm / np.sqrt(self._var + self.eps)
                     + dvar * 2.0 * (x_np - self._mean) / N_pixels
                     + dmean / N_pixels)

            return cuda.to_device(np.ascontiguousarray(dx_np))

    class CudaAdam:
        def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999):
            self.layers = layers
            self.lr     = lr
            self.beta1  = beta1
            self.beta2  = beta2
            self.eps    = 1e-8
            self.t      = 0

            self._m = {}
            self._v = {}
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'params'):
                    self._m[i] = {}
                    self._v[i] = {}
                    for k, v in layer.params.items():
                        self._m[i][k] = cuda.to_device(np.zeros_like(v, dtype=np.float32))
                        self._v[i][k] = cuda.to_device(np.zeros_like(v, dtype=np.float32))

        def step(self):
            self.t += 1
            for i, layer in enumerate(self.layers):
                if not hasattr(layer, 'params'):
                    continue
                for key in layer.params:
                    if hasattr(layer, '_grads_dev') and hasattr(layer, '_params_dev'):
                        grad_dev  = layer._grads_dev[key]
                        param_dev = layer._params_dev[key]
                    else:
                        grad_dev  = cuda.to_device(np.ascontiguousarray(layer.grads[key].astype(np.float32)))
                        param_dev = cuda.to_device(np.ascontiguousarray(layer.params[key].astype(np.float32)))

                    n = param_dev.size
                    g, b = _grid1d(n)
                    _adam_update_kernel[g, b](
                        param_dev, grad_dev,
                        self._m[i][key], self._v[i][key],
                        self.lr, self.beta1, self.beta2, self.eps,
                        float(self.t)
                    )

                    if not hasattr(layer, '_params_dev'):
                        layer.params[key][:] = param_dev.copy_to_host()


class CudaAlexNet:
    def __init__(self, num_classes=10):
        if not _GPU_AVAILABLE:
            raise RuntimeError("CUDA is required for CudaAlexNet.")
        
        self.layers = [
            CudaConv2D(3, 96, kernel_size=11, stride=4, padding=0),
            CudaBatchNorm2D(96),
            CudaReLU(),
            CudaMaxPool2D(kernel_size=3, stride=2),

            CudaConv2D(96, 256, kernel_size=5, stride=1, padding=2),
            CudaBatchNorm2D(256),
            CudaReLU(),
            CudaMaxPool2D(kernel_size=3, stride=2),

            CudaConv2D(256, 384, kernel_size=3, stride=1, padding=1),
            CudaReLU(),

            CudaConv2D(384, 384, kernel_size=3, stride=1, padding=1),
            CudaReLU(),

            CudaConv2D(384, 256, kernel_size=3, stride=1, padding=1),
            CudaReLU(),
            CudaMaxPool2D(kernel_size=3, stride=2),

            CudaFlatten(),
            CudaDense(256 * 6 * 6, 4096),
            CudaReLU(),
            CudaDropout(0.5),

            CudaDense(4096, 4096),
            CudaReLU(),
            CudaDropout(0.5),

            CudaDense(4096, num_classes)
        ]
        self.loss_fn = SoftmaxCrossEntropy()

    def set_train(self, train: bool):
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train = train

    def forward(self, x_dev):
        out = x_dev
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        dout = self.loss_fn.backward()
        dout_dev = cuda.to_device(np.ascontiguousarray(dout.astype(np.float32)))
        for layer in reversed(self.layers):
            dout_dev = layer.backward(dout_dev)

    def save_weights(self, path):
        arrays = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                if hasattr(layer, '_params_dev'):
                    for k, v_dev in layer._params_dev.items():
                        layer.params[k][:] = v_dev.copy_to_host()
                for k, v in layer.params.items():
                    arrays[f"layer{i}_{k}"] = v
        np.savez(path, **arrays)
        print(f"Weights saved to {path}")

    def load_weights(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for k in layer.params:
                    key = f"layer{i}_{k}"
                    if key in data:
                        layer.params[k][:] = data[key]
                        if hasattr(layer, '_params_dev'):
                            layer._params_dev[k].copy_to_device(np.ascontiguousarray(data[key]))
        print(f"Weights loaded from {path}")


def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

_IMGNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_IMGNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

def batch_iter(X, y, batch_size, shuffle=True):
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, n, batch_size):
        sl = indices[i:i+batch_size]
        Xb = X[sl]
        if Xb.dtype == np.uint8:
            Xb = Xb.astype(np.float32) / 255.0
            Xb = (Xb - _IMGNET_MEAN) / _IMGNET_STD
        yield Xb, y[sl]


class Prefetcher:
    def __init__(self, data_iter, augmentor, aug_kwargs, queue_size=3):
        self.data_iter = data_iter
        self.augmentor = augmentor
        self.aug_kwargs = aug_kwargs
        self.q = queue.Queue(maxsize=queue_size)
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    def _worker(self):
        try:
            for Xb, yb in self.data_iter:
                Xb_aug = self.augmentor.augment(Xb, **self.aug_kwargs)
                self.q.put((Xb_aug, yb))
        finally:
            self.q.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.q.get()
        if item is None:
            raise StopIteration
        return item


if __name__ == "__main__":
    loader = KaggleImageNet10Loader(img_size=227, val_fraction=0.2, max_samples=None)
    X_train, y_train, X_val, y_val = loader.load()

    y_train_oh = one_hot(y_train, 10)
    y_val_oh   = one_hot(y_val,   10)

    EPOCHS = 10
    BATCH_SIZE = 32 if _GPU_AVAILABLE else 16

    total_train_samples = X_train.shape[0]
    total_train_batches = math.ceil(total_train_samples / BATCH_SIZE)

    print("\nInitializing CudaAlexNet model...")
    model = CudaAlexNet(num_classes=10)
    optimizer = CudaAdam(model.layers, lr=0.0005)

    AUG_KWARGS = {
        'crop_h': 227, 'crop_w': 227, 'pad': 8,
        'flip_prob': 0.5, 'brightness': 0.2,
        'contrast_range': (0.8, 1.2), 'saturation_range': (0.8, 1.2),
        'max_rotation': 0, 
        'use_cutout': True, 'cutout_size': 32
    }
    augmentor = ImageAugmentor()

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        start_t = time.time()
        model.set_train(True)
        train_loss = 0.0
        train_acc  = 0.0
        n_batches  = 0
        processed_images = 0

        train_iter = batch_iter(X_train, y_train_oh, BATCH_SIZE)
        prefetcher = Prefetcher(train_iter, augmentor, AUG_KWARGS)

        for Xb_np, yb_np in prefetcher:
            if n_batches == 0 and epoch == 0:
                print("  [JIT] Compiling Numba CUDA kernels on first batch...")

            Xb_dev = cuda.to_device(np.ascontiguousarray(Xb_np))
            logits_dev = model.forward(Xb_dev)
            logits_np  = logits_dev.copy_to_host()

            loss = model.loss_fn.forward(logits_np, yb_np)
            train_loss += loss
            train_acc  += np.mean(np.argmax(logits_np, axis=1) == np.argmax(yb_np, axis=1))

            model.backward()
            optimizer.step()
            n_batches += 1

            # Progress tracking metrics
            batch_sz = Xb_np.shape[0]
            processed_images += batch_sz
            remaining_images = total_train_samples - processed_images

            # Log status every 10 batches or at the end of the epoch
            if n_batches % 10 == 0 or processed_images == total_train_samples:
                pct = (processed_images / total_train_samples) * 100
                print(f"    Batch {n_batches:02d}/{total_train_batches:02d} | "
                      f"Processed: {processed_images}/{total_train_samples} ({pct:.1f}%) | "
                      f"Remaining: {remaining_images} imgs | Loss: {loss:.4f}")

        train_loss /= n_batches
        train_acc  /= n_batches

        model.set_train(False)
        val_loss = 0.0
        val_acc  = 0.0
        n_val_batches = 0

        for Xb_np, yb_np in batch_iter(X_val, y_val_oh, BATCH_SIZE, shuffle=False):
            Xb_dev = cuda.to_device(np.ascontiguousarray(Xb_np))
            logits_dev = model.forward(Xb_dev)
            logits_np  = logits_dev.copy_to_host()
            val_loss += model.loss_fn.forward(logits_np, yb_np)
            val_acc  += np.mean(np.argmax(logits_np, axis=1) == np.argmax(yb_np, axis=1))
            n_val_batches += 1

        val_loss /= max(1, n_val_batches)
        val_acc  /= max(1, n_val_batches)
        elapsed   = time.time() - start_t

        print(f"Epoch {epoch+1:02d}/{EPOCHS} [{elapsed:.1f}s] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n")

    print("Training Complete.")
    model.save_weights("alexnet_imagenet10.npz")