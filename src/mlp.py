"""
mlp.py – Mạng Neuron nhiều lớp (MLP) tự cài đặt bằng NumPy, huấn luyện bằng lan truyền ngược.

Kiến trúc: Input -> [Linear -> ReLU] x (L-1) -> Linear -> Softmax
Hàm mất mát: Cross-entropy + L2 (weight decay)  J = -1/n Σ Σ y_ik log p_ik + λ/2 Σ ||W_l||²
Tối ưu: mini-batch SGD + momentum, dừng sớm (early stopping) theo accuracy trên validation.

Dùng với data.py:
    from src.data import MNISTPreprocessor
    from src.mlp import MLP

    ds = MNISTPreprocessor(val_size=10000, seed=42).run()
    X_tr, y_tr = ds.get("train", "minmax")
    X_val, y_val = ds.get("val", "minmax")
    X_te, y_te = ds.get("test", "minmax")

    model = MLP(sizes=(784, 128, 10), lr=0.05, epochs=15).fit(X_tr, y_tr, X_val, y_val)
    print(model.score(X_te, y_te))

Chạy thử trực tiếp (đứng ở thư mục mnist-project):
    python src/mlp.py
"""
import os
import pickle
import time

import numpy as np

try:                        
    from .data import OneHotEncoder
except ImportError:         
    from data import OneHotEncoder


def softmax(Z):
    """Softmax ổn định số: trừ max mỗi hàng trước khi lấy exp để tránh tràn số."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def relu(Z):
    return np.maximum(Z, 0)


class MLP:
    """Multi-Layer Perceptron cho phân lớp nhiều lớp.

    Tham số:
        sizes       số nơ-ron từng lớp, VD (784, 128, 10) nghĩa là input layer có 784 chiều
                    1 lớp ẩn 128 nơ-ron và output layer có 10 nơ ron t/ứng 10 nhãn
        lr          learning rate
        momentum    hệ số quán tính momentum μ (0 = SGD thường), giúp tăng tốc quá trình hội tụ và vượt qua 
                    các điểm cực tiểu địa phương bằng cách tích lũy vận tốc từ các bước cập nhật trước
        lam         hệ số L2 λ (weight decay), chỉ áp lên W, giúp phạt các trọng số có giá trị lớn, ngăn chặn overfitting
        epochs      số epoch tối đa
        batch_size  kích thước mini-batch
        patience    dừng sớm nếu val_acc không tăng sau `patience` epoch (None = không dừng sớm)
        seed        hạt giống ngẫu nhiên (khởi tạo trọng số + xáo trộn dữ liệu)
        dtype       kiểu số thực; float32 nhanh hơn, float64 dùng khi kiểm tra gradient
        verbose     in tiến trình mỗi epoch

    Sau khi fit:
        history     dict: train_loss, train_acc, val_loss, val_acc, epoch_time (mỗi phần tử = 1 epoch)
        best_epoch_ epoch (đếm từ 1) có val_acc cao nhất, trọng số được khôi phục về epoch này
        train_time_ tổng thời gian huấn luyện (giây)
    """

    def __init__(self, sizes=(784, 128, 10), lr=0.05, momentum=0.9, lam=0.0,
                 epochs=15, batch_size=128, patience=3, seed=0,
                 dtype=np.float32, verbose=True):
        self.sizes = tuple(sizes)
        self.lr, self.mu, self.lam = lr, momentum, lam
        self.epochs, self.bs, self.patience, self.seed = epochs, batch_size, patience, seed
        self.dtype, self.verbose = dtype, verbose
        self.n_classes = self.sizes[-1]
        self._onehot = OneHotEncoder(self.n_classes)

    def _init(self, rng):
        """Khởi tạo He: W ~ N(0, 2/fan_in), phù hợp với ReLU; b = 0."""
        self.W = [rng.normal(0, np.sqrt(2 / m), (m, k)).astype(self.dtype)
                  for m, k in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.zeros(k, dtype=self.dtype) for k in self.sizes[1:]]
        self.vW = [np.zeros_like(W) for W in self.W]    # vận tốc cho momentum
        self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, X, cache=True):
        """Z_l = A_{l-1} W_l + b_l;  A_l = ReLU(Z_l) (lớp ẩn) hoặc softmax(Z_l) (lớp ra).
        cache=True lưu A, Z để backward; cache=False khi chỉ dự đoán (tiết kiệm bộ nhớ)."""
        A = np.asarray(X, dtype=self.dtype)
        As, Zs = [A], []
        L = len(self.W)
        for l in range(L):
            Z = A @ self.W[l] + self.b[l]
            A = softmax(Z) if l == L - 1 else relu(Z)
            if cache:
                Zs.append(Z)
                As.append(A)
        if cache:
            self.A, self.Z = As, Zs
        return A

    def backward(self, Y):
        """Lan truyền ngược với softmax + cross-entropy:
            δ_L = (P - Y) / n
            ∂J/∂W_l = A_{l-1}ᵀ δ_l + λ W_l ,   ∂J/∂b_l = Σ_i δ_l
            δ_{l-1} = (δ_l W_lᵀ) ⊙ 1[Z_{l-1} > 0]
        """
        delta = (self.A[-1] - Y) / len(Y)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.A[l].T @ delta + self.lam * self.W[l]
            gb[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * (self.Z[l - 1] > 0)
        return gW, gb

    def _step(self, gW, gb):
        """SGD + momentum: v ← μ v − η g ;  θ ← θ + v."""
        for l in range(len(self.W)):
            self.vW[l] = self.mu * self.vW[l] - self.lr * gW[l]
            self.vb[l] = self.mu * self.vb[l] - self.lr * gb[l]
            self.W[l] += self.vW[l]
            self.b[l] += self.vb[l]

    def loss(self, X, y):
        """Cross-entropy trung bình + L2 trên toàn bộ X."""
        P = self.predict_proba(X)
        ce = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean()
        return float(ce + 0.5 * self.lam * sum((W.astype(np.float64) ** 2).sum() for W in self.W))

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=self.dtype)
        y = np.asarray(y, dtype=np.int64)
        if X.shape[1] != self.sizes[0]:
            raise ValueError(f"X có {X.shape[1]} đặc trưng nhưng sizes[0] = {self.sizes[0]}")
        has_val = X_val is not None and y_val is not None

        rng = np.random.default_rng(self.seed)
        self._init(rng)
        Y = self._onehot.transform(y).astype(self.dtype)
        best, wait = -1.0, 0
        self.best_epoch_ = None
        self.history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time")}
        t_start = time.time()

        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            idx = rng.permutation(len(X)) #Xáo trộn DL: Mỗi epoch, dữ liệu được xáo trộn ngẫu nhiên nhằm phá vỡ trật tự ban đầu, giúp mô hình không bị học thuộc thứ tự mẫu và giảm phương sai của gradient
            tot_loss, tot_correct = 0.0, 0
            for s in range(0, len(X), self.bs):
                bi = idx[s:s + self.bs]
                P = self.forward(X[bi]) #Lan truyền tiến để tính toán đầu ra cho mini-batch.
                # thống kê train lấy luôn từ forward của mini-batch (không tốn thêm một lượt duyệt)
                #tích lũy lỗi
                tot_loss += -np.log(np.clip(P[np.arange(len(bi)), y[bi]], 1e-12, None)).sum()
                #tích lũy số mẫu đoán đúng
                tot_correct += (P.argmax(axis=1) == y[bi]).sum()
                gW, gb = self.backward(Y[bi])
                self._step(gW, gb)

            l2 = 0.5 * self.lam * sum((W.astype(np.float64) ** 2).sum() for W in self.W)
            #Lưu lại loss và accuracy của train, validation sau mỗi epoch để phân tích learning curves
            self.history["train_loss"].append(tot_loss / len(X) + l2)   # cùng công thức với val_loss
            self.history["train_acc"].append(tot_correct / len(X))
            self.history["epoch_time"].append(time.time() - t0)

            if has_val:
                val_loss, val_acc = self.loss(X_val, y_val), self.score(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                if val_acc > best: #độ chính xác trên tập val cải thiện
                    best, wait, self.best_epoch_ = val_acc, 0, ep
                    self._best = ([W.copy() for W in self.W], [b.copy() for b in self.b])
                else:
                    wait += 1
            if not np.isfinite(self.history["train_loss"][-1]):
                print("Loss = NaN/inf: learning rate quá lớn, dừng huấn luyện.")
                break

            if self.verbose:
                msg = (f"Epoch {ep:>2}/{self.epochs} | loss {self.history['train_loss'][-1]:.4f} "
                       f"| train_acc {self.history['train_acc'][-1]:.4f}")
                if has_val:
                    msg += f" | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
                print(msg + f" | {self.history['epoch_time'][-1]:.1f}s")

            if has_val and self.patience is not None and wait >= self.patience:
                if self.verbose:
                    print(f"Dừng sớm: val_acc không tăng sau {self.patience} epoch.")
                break

        if has_val and self.best_epoch_ is not None:   # khôi phục trọng số tốt nhất
            self.W, self.b = self._best
            if self.verbose:
                print(f"Khôi phục trọng số epoch {self.best_epoch_} (val_acc = {best:.4f})")
        self.train_time_ = time.time() - t_start
        self.A = self.Z = None   # giải phóng cache
        return self

    def predict_proba(self, X, batch_size=4096):
        """Xác suất từng lớp (N, n_classes); chạy theo lô để không tốn RAM."""
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim == 1:
            X = X[None]
        #np.vstack(...): Ghép nối các kết quả xác suất của từng lô lại thành một mảng hoàn chỉnh có kích thước (N, n_classes)
        return np.vstack([self.forward(X[s:s + batch_size], cache=False) #hạy lan truyền tiến (forward pass) cho từng lô. Tham số cache=False có nghĩa là không lưu lại các trung gian activation
                          for s in range(0, len(X), batch_size)]) #Chia DL thành các batches

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        """Accuracy."""
        return float((self.predict(X) == np.asarray(y)).mean())

    @property
    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    def gradient_check(self, X, y, eps=1e-5, n_checks=10, seed=0):
        """So sánh gradient từ backward với sai phân hữu hạn trung tâm:
            g ≈ [J(θ + ε) − J(θ − ε)] / 2ε
        Trả về sai số tương đối lớn nhất; < 1e-6 nghĩa là backprop đúng.
        Nên dùng mạng nhỏ, vài mẫu, dtype=np.float64."""
        rng = np.random.default_rng(seed)
        if not hasattr(self, "W"):
            self._init(rng)
        y = np.asarray(y)
        Y = self._onehot.transform(y).astype(self.dtype)
        self.forward(X)
        gW, gb = self.backward(Y)

        max_err = 0.0
        for params, grads in ((self.W, gW), (self.b, gb)):
            for l in range(len(params)):
                for _ in range(n_checks):
                    i = tuple(rng.integers(0, s) for s in params[l].shape)
                    old = params[l][i]
                    params[l][i] = old + eps
                    jp = self.loss(X, y)
                    params[l][i] = old - eps
                    jm = self.loss(X, y)
                    params[l][i] = old
                    num = (jp - jm) / (2 * eps)
                    err = abs(num - grads[l][i]) / max(1e-8, abs(num) + abs(grads[l][i]))
                    max_err = max(max_err, err)
        return max_err

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {"params": {k: getattr(self, k) for k in ("sizes", "lr", "mu", "lam", "epochs", "bs",
                                                         "patience", "seed", "dtype")},
                 "W": self.W, "b": self.b, "history": getattr(self, "history", None),
                 "best_epoch": getattr(self, "best_epoch_", None)}
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Đã lưu mô hình vào {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            st = pickle.load(f)
        p = st["params"]
        m = cls(sizes=p["sizes"], lr=p["lr"], momentum=p["mu"], lam=p["lam"], epochs=p["epochs"],
                batch_size=p["bs"], patience=p["patience"], seed=p["seed"], dtype=p["dtype"], verbose=False)
        m.W, m.b, m.history, m.best_epoch_ = st["W"], st["b"], st["history"], st["best_epoch"]
        return m
