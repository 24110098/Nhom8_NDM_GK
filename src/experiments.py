import numpy as np
import pandas as pd
from scipy import ndimage

def augment_dataset(X, copies=1, max_deg=15, max_shift=2, sigma=0.1, seed=0):
    """Trả về X gốc + `copies` bản biến đổi ngẫu nhiên (xoay, dịch, nhiễu)."""
    rng = np.random.default_rng(seed)
    out = [X]
    for _ in range(copies):
        imgs = X.reshape(-1, 28, 28).copy()
        for i in range(len(imgs)):
            im = ndimage.rotate(imgs[i], rng.uniform(-max_deg, max_deg), reshape=False, order=1)
            imgs[i] = ndimage.shift(im, rng.integers(-max_shift, max_shift + 1, size=2), order=0)
        imgs = np.clip(imgs + rng.normal(0, sigma, imgs.shape), 0, 1)
        out.append(imgs.reshape(len(X), 784).astype(np.float32))
    return np.concatenate(out)

def writer_split(writers, test_ratio=0.3, seed=0):
    """Chia theo NGƯỜI VIẾT: mỗi người chỉ nằm ở một phía."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.unique(writers))
    test_ids = set(ids[:max(1, round(len(ids) * test_ratio))])
    te = np.array([w in test_ids for w in writers])
    return ~te, te

def evaluate(model, X, y):
    pred = model.predict(X)
    C = np.zeros((10, 10), int); np.add.at(C, (y, pred), 1)
    tp = np.diag(C).astype(float)
    p = np.divide(tp, C.sum(0), out=np.zeros(10), where=C.sum(0) > 0)
    r = np.divide(tp, C.sum(1), out=np.zeros(10), where=C.sum(1) > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros(10), where=(p + r) > 0)
    return {"acc": (pred == y).mean(), "f1": f1.mean(), "recall_per_class": r}


def perturb(X, kind, level, seed=0):
    """Tạo phiên bản bị biến đổi của cả tập: xoay `level` độ, dịch `level` pixel, hoặc nhiễu sigma = `level`."""
    imgs = X.reshape(-1, 28, 28)
    if kind == "rotate":
        out = ndimage.rotate(imgs, level, axes=(2, 1), reshape=False, order=1)
    elif kind == "shift":
        out = ndimage.shift(imgs, (0, level, level), order=0)          # dịch chéo xuống-phải
    elif kind == "noise":
        out = np.clip(imgs + np.random.default_rng(seed).normal(0, level, imgs.shape), 0, 1)
    else:
        raise ValueError(kind)
    return out.reshape(len(X), 784).astype(np.float32)


LEVELS = {"rotate": [0, 10, 20, 30, 45], "shift": [0, 1, 2, 3, 4], "noise": [0, 0.1, 0.2, 0.3, 0.5]}


def robustness(models, X, y, levels=LEVELS):
    """models: dict tên -> mô hình đã huấn luyện. Trả về bảng accuracy theo loại và mức biến đổi."""
    rows = []
    for kind, lv_list in levels.items():
        for lv in lv_list:
            Xp = perturb(X, kind, lv)
            for name, m in models.items():
                rows.append({"model": name, "kind": kind, "level": lv, "acc": (m.predict(Xp) == y).mean()})
    return pd.DataFrame(rows)
