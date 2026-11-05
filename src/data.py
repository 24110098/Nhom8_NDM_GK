import argparse
import gzip             
import os               
import struct           
import time
import urllib.request
from pathlib import Path

import numpy as np

IMG_SIZE = 28
N_FEATURES = IMG_SIZE * IMG_SIZE
N_CLASSES = 10

# Thư mục dự án = thư mục cha của thư mục chứa file này (VD: mnist-project/src/data.py -> mnist-project)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "src" else _THIS_DIR
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Đường dẫn dạng Path (main.ipynb dùng VN_FILE.exists())
ROOT = Path(PROJECT_DIR)
MNIST_FILE = Path(DATA_DIR) / "mnist.npz"            # 4 mảng X_train, y_train, X_test, y_test (pixel 0..255)
VN_FILE = Path(DATA_DIR) / "vn_digits.npz"          
VN_EXCLUDE = Path(DATA_DIR) / "vn_exclude.csv"       

VAL_RATIO = 0.1     

# Các bản sao (mirror) của 4 file gốc MNIST, thử theo thứ tự
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


# dùng để kiểm tra file tải về có đủ dữ liệu không.
MNIST_FILES = {
    "X_train": ("train-images-idx3-ubyte.gz", 60000),
    "y_train": ("train-labels-idx1-ubyte.gz", 60000),
    "X_test": ("t10k-images-idx3-ubyte.gz", 10000),
    "y_test": ("t10k-labels-idx1-ubyte.gz", 10000),
}


# Tải và nạp dữ liệu
class MNISTLoader:
    """Tải MNIST từ Internet và lưu thành <data_dir>/mnist.npz; nạp lại khi cần.

    Ví dụ:
        loader = MNISTLoader()                       # lưu vào mnist-project/data/
        loader = MNISTLoader("D:/datasets/mnist")    # hoặc thư mục tuỳ chọn
        loader.download()                            # tải nếu chưa có
        loader.download(force=True)                  # xoá bản cũ, tải lại
        X_train, y_train, X_test, y_test = loader.load()
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = os.path.abspath(data_dir)
        self.raw_dir = os.path.join(self.data_dir, "mnist_raw")     #4 file .gz gốc
        self.npz_path = os.path.join(self.data_dir, "mnist.npz")    #file gộp sau khi tải

    def is_downloaded(self):
        return os.path.exists(self.npz_path) 

    # tải 
    def download(self, force=False, verbose=True):
        """Tải 4 file gốc (thử lần lượt các mirror, rồi đến OpenML), kiểm tra, lưu mnist.npz."""
        if self.is_downloaded() and not force:
            if verbose:
                print(f"Đã có {self.npz_path}, bỏ qua. Dùng force=True để tải lại.")
            return self
        os.makedirs(self.raw_dir, exist_ok=True)
        if force:
            for f in os.listdir(self.raw_dir):
                os.remove(os.path.join(self.raw_dir, f))

        try:
            data = self._from_mirrors(verbose)
        except Exception as e:
            print(f"Các mirror đều lỗi: {e}\nThử nguồn dự phòng OpenML ...")
            try:
                data = self._from_openml()
            except ImportError:
                raise RuntimeError("Không tải được. Kiểm tra mạng, hoặc cài scikit-learn để dùng nguồn OpenML.")
            except Exception as e2:
                raise RuntimeError(f"Nguồn OpenML cũng lỗi: {e2}\nKiểm tra kết nối mạng rồi chạy lại.")

        self._sanity_check(data) #Kiểm tra DL
        np.savez_compressed(self.npz_path, **data) #Gộp 4 mảng vào 1 file .npz nén
        if verbose:
            print(f"Đã lưu {self.npz_path} ({os.path.getsize(self.npz_path) / 1e6:.1f} MB)")
        return self

    def _download_file(self, filename, verbose, retries=2):
        path = os.path.join(self.raw_dir, filename)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path #Nếu file đã có thì dùng luôn, không tải lại
        errors = []
        for base in MIRRORS:
            for attempt in range(retries): #mỗi mirror được thử retries lần
                try:
                    if verbose:
                        print(f"  Tải {base}{filename} ...", end=" ", flush=True)
                    req = urllib.request.Request(base + filename, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as f:
                        f.write(r.read())
                    os.replace(path + ".part", path) #DL được ghi vào file tạm .part
                    if verbose:
                        print("xong")
                    return path
                except Exception as e:  # mạng lỗi, 403, timeout...
                    if verbose:
                        print(f"lỗi ({e})")
                    errors.append(f"{base}: {e}")
                    time.sleep(1 + attempt)
        raise RuntimeError(f"Không tải được {filename}:\n  " + "\n  ".join(errors))

    @staticmethod
    def _read_idx(path, expected_n, is_images):
        """Đọc file IDX (.gz) và kiểm tra magic number, số mẫu, kích thước."""
        with gzip.open(path, "rb") as f:
            data = f.read()
        name = os.path.basename(path)
        if is_images:
            magic, n, rows, cols = struct.unpack(">IIII", data[:16])
            if magic != 2051 or n != expected_n or (rows, cols) != (IMG_SIZE, IMG_SIZE):
                raise ValueError(f"{name} sai cấu trúc (magic={magic}, n={n}, {rows}x{cols})")
            arr = np.frombuffer(data, dtype=np.uint8, offset=16)
            if arr.size != n * N_FEATURES: #kiểm tra để phát hiện file tải thiếu
                raise ValueError(f"{name} bị thiếu dữ liệu (file tải dở?)")
            return arr.reshape(n, IMG_SIZE, IMG_SIZE)
        magic, n = struct.unpack(">II", data[:8])
        if magic != 2049 or n != expected_n:
            raise ValueError(f"{name} sai cấu trúc (magic={magic}, n={n})")
        arr = np.frombuffer(data, dtype=np.uint8, offset=8)
        if arr.size != n or arr.max() > 9: #Nhãn nằm ngoài 0-9 là bất thường
            raise ValueError(f"{name} có nhãn không hợp lệ")
        return arr

    #tải và đọc 4 file
    #Nếu đọc lỗi (file hỏng), nó xoá file và tải lại một lần nữa. 
    #Đến lần thứ hai vẫn lỗi thì báo lỗi ra ngoài để download() chuyển sang OpenML.
    def _from_mirrors(self, verbose):
        out = {}
        for key, (filename, n) in MNIST_FILES.items():
            for attempt in range(2):
                path = self._download_file(filename, verbose)
                try:
                    out[key] = self._read_idx(path, n, is_images=key.startswith("X"))
                    break
                except (ValueError, OSError, EOFError) as e:
                    os.remove(path)  # file hỏng hoặc tải dở: xoá rồi tải lại một lần
                    print(f"  {filename} hỏng ({e}), tải lại ...")
                    if attempt == 1:
                        raise
        return out

    @staticmethod
    #Nguồn dự phòng
    def _from_openml():
        """Nguồn dự phòng: OpenML qua scikit-learn (chỉ dùng để tải, không phải phần tính toán lõi)."""
        from sklearn.datasets import fetch_openml
        print("  Tải từ OpenML (scikit-learn), có thể mất vài phút ...")
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True, parser="liac-arff")
        X = X.astype(np.uint8).reshape(-1, IMG_SIZE, IMG_SIZE)
        y = y.astype(np.uint8)
        return {"X_train": X[:60000], "y_train": y[:60000], "X_test": X[60000:], "y_test": y[60000:]}

    @staticmethod #1 decorator định nghĩa 1 phương thức tĩnh -> không cần self
    #Kiểm tra lần cuối
    def _sanity_check(d):
        #assert dừng ctr nếu ĐK sai
        assert d["X_train"].shape == (60000, IMG_SIZE, IMG_SIZE) and d["X_test"].shape == (10000, IMG_SIZE, IMG_SIZE)
        assert d["y_train"].shape == (60000,) and d["y_test"].shape == (10000,)
        for k in ("y_train", "y_test"):
            #bincount đếm số mẫu của từng nhãn 0-9-> Lớp nào = 0 -> Dl bị thiếu lớp
            assert np.bincount(d[k], minlength=N_CLASSES).min() > 0, f"{k} thiếu lớp"

    def load(self, auto_download=True):
        """Trả về X_train (60000,28,28) uint8, y_train (60000,) int64, X_test, y_test.
        Nhãn được ép sang int64 để tính toán an toàn và dùng được với PyTorch."""
        if not self.is_downloaded():
            if not auto_download:
                raise FileNotFoundError(f"Chưa có {self.npz_path}. Gọi MNISTLoader().download() trước.")
            self.download()
        with np.load(self.npz_path) as d:
            return (d["X_train"], d["y_train"].astype(np.int64),
                    d["X_test"], d["y_test"].astype(np.int64))


# Làm phẳng ảnh 2D <-> vector 1D
class Flattener:
    """Ảnh (N, 28, 28) <-> vector (N, 784), thứ tự sắp xếp các pixel tuân theo quy tắc hàng, vị trí pixel (i, j) nằm ở vị trí tuyến tính i*28 + j.

    Ép sang float32 để tránh tràn số khi tính toán trên uint8 (3 - 5 = 254 thay vì -2 -> underflow).
    """
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def transform(self, images):
        images = np.asarray(images)               
        if images.ndim == 2:                      
            images = images[None]                 # thêm một chiều batch để biến nó thành (1, 28, 28)
        if images.shape[1:] != (IMG_SIZE, IMG_SIZE): #Kiểm tra kích thước, đảm bảo tất cả các chiều từ chiều thứ 2 trở đi đúng bằng (IMG_SIZE, IMG_SIZE)
            raise ValueError(f"Cần ảnh 28x28, nhận được {images.shape}")
        return images.reshape(len(images), -1).astype(self.dtype) 
        #Trải phẳng và ép kiểu, tự động gom 2 chiều cuối (28, 28) thành 784
        #-1 báo cho NumPy tự tính toán kích thước chiều này dựa trên tổng số phần tử, ép kiểu DL sang float32


    def inverse_transform(self, X):
        """Vector -> ảnh, dùng để vẽ ảnh"""
        return np.asarray(X).reshape(-1, IMG_SIZE, IMG_SIZE)



# Chia tập
class StratifiedSplitter:
    """Chia train/validation theo pp phân tầng (giữ tỉ lệ lớp).
    -> Đảm bảo tỷ lệ các nhãn ở train và val xấp xỉ bằng nhau và bằng với tập dữ liệu gốc
    
    val_size: tỉ lệ phần trăm (0 < val_size < 1) hoặc số lượng mẫu cụ thể (>= 1).
    Sau khi split, train_idx_ và val_idx_ lưu chỉ số gốc để tái lập.
    """

    def __init__(self, val_size=VAL_RATIO, seed=42):
        self.val_size = val_size
        self.seed = seed
        self.train_idx_ = None
        self.val_idx_ = None

    def split(self, X, y):
        #Tính toán tổng số lượng mẫu cho tập Val
        y = np.asarray(y)
        n = len(y)
        #Quy đổi val_size thành tổng số lượng mẫu cần đưa vào tập val
        n_val_total = int(round(n * self.val_size)) if self.val_size < 1 else int(self.val_size)
        if not 0 < n_val_total < n:
            raise ValueError(f"val_size không hợp lệ: {self.val_size}")

        #Phân bổ số lượng mẫu val cho từng lớp (Giữ tỷ lệ)
        rng = np.random.default_rng(self.seed)
        classes, counts = np.unique(y, return_counts=True)
        # Số mẫu val mỗi lớp: làm tròn xuống, phần dư chia cho các lớp có phần lẻ lớn nhất
        #dùng phương pháp "phần dư lớn nhất" để tổng số mẫu ra đúng n_val_total (VD: 6.000 = 10% của 60.000)
        raw = counts * n_val_total / n      #Tính toán lý thuyết số lượng mẫu val cần có cho từng lớp.
        n_val = np.floor(raw).astype(int)   #Làm tròn xuống số lượng mẫu cho mỗi lớp
        #xử lý phần dư (do làm tròn xuống có thể khiến tổng số mẫu val bị hụt một vài đơn vị so với n_val_total
        n_val[np.argsort(-(raw - n_val))[:n_val_total - n_val.sum()]] += 1

        #Với mỗi lớp c, code tìm các vị trí xuất hiện (np.where(y == c)[0]), 
        # xáo trộn ngẫu nhiên chúng lên bằng rng.permutation, và cắt lấy đúng k mẫu cần thiết. 
        # Sau đó nối lại thành một mảng val_idx chung.
        val_idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:k]
                                  for c, k in zip(classes, n_val)])
        mask = np.ones(n, dtype=bool)   #ban đầu tất cả là True . 
        mask[val_idx] = False           #chuyển các vị trí thuộc val_idx thành False, Các mẫu còn True trở thành tập train
        self.train_idx_ = rng.permutation(np.where(mask)[0]) #permutation: xáo trộn ngẫu nhiên một lần nữa để tránh việc các mẫu trong cùng một lớp đứng liền kề nhau
        self.val_idx_ = rng.permutation(val_idx)
        return X[self.train_idx_], X[self.val_idx_], y[self.train_idx_], y[self.val_idx_]



# Chuẩn hoá
class MinMaxScaler:
    """x' = (x - min) / (max - min). Mặc định min = 0, max = 255 (giá trị pixel cố định),
    nên không cần fit. Đặt fit_from_data=True để học min/max từng pixel từ tập train."""

    def __init__(self, min_value=0.0, max_value=255.0, fit_from_data=False):
        self.min_ = min_value
        self.max_ = max_value
        self.fit_from_data = fit_from_data

    #Nếu fit_from_data=True, tính giá trị min_ và max_ theo từng cột/pixel (axis=0)
    def fit(self, X):
        if self.fit_from_data:
            self.min_ = X.min(axis=0)
            self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        #Nếu khoảng chênh lệch bằng 0, nó sẽ tạm thời thay bằng 1.0
        rng = np.where(np.asarray(self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_)
        return ((X - self.min_) / rng).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    #đưa dữ liệu về lại thang 0–255.
    def inverse_transform(self, X):
        return X * (self.max_ - self.min_) + self.min_

class StandardScaler:
    """Z-score theo từng pixel: x' = (x - mu) / sqrt(var + eps).

    Chỉ fit trên tập train rồi transform val/test để tránh rò rỉ dữ liệu.
    eps = 1e-2 vì nhiều pixel viền có phương sai 0 hoặc rất nhỏ; nếu không làm trơn,
    một nét bút hiếm ở viền ảnh test sẽ cho giá trị z-score hàng nghìn.
    """

    def __init__(self, eps=1e-2):
        self.eps = eps
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.sqrt(X.var(axis=0) + self.eps)
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("StandardScaler chưa được fit. Gọi fit(X_train) trước.")
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    #Khôi phục lại giá trị ban đầu
    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_


class OneHotEncoder:
    """Nhãn (N,) <-> ma trận one-hot (N, n_classes)."""

    def __init__(self, n_classes=N_CLASSES):
        self.n_classes = n_classes


    def transform(self, y):
        y = np.asarray(y, dtype=np.int64)
        Y = np.zeros((len(y), self.n_classes), dtype=np.float32)
        #gán giá trị 1.0 tại các vị trí cột tương ứng với nhãn của từng dòng mà không cần dùng vòng lặp
        Y[np.arange(len(y)), y] = 1.0
        return Y

    #Chuyển đổi ngược từ ma trận one-hot hoặc ma trận xác suất dự đoán (output của mạng Softmax) về lại nhãn số nguyên.
    def inverse_transform(self, Y):
        """One-hot hoặc ma trận xác suất -> nhãn (lấy argmax)."""
        return np.asarray(Y).argmax(axis=1)


class MNISTDataset:
    """Chứa dữ liệu sau xử lý.

    data[(split, scaling)] với split ∈ {"train", "val", "test"},
    scaling ∈ {"raw", "minmax", "standard"}; nhãn ở labels[split].
    """

    SPLITS = ("train", "val", "test")
    SCALINGS = ("raw", "minmax", "standard")

    def __init__(self, data, labels, standard_scaler=None):
        self.data = data
        self.labels = labels
        self.standard_scaler = standard_scaler
        self._onehot = OneHotEncoder()

    #Nếu one_hot=True, tự động chuyển nhãn y sang dạng one-hot vector
    def get(self, split="train", scaling="minmax", one_hot=False):
        """Trả về (X, y). one_hot=True để nhận y dạng one-hot (cho Softmax/MLP)."""
        if split not in self.SPLITS:
            raise ValueError(f"split phải thuộc {self.SPLITS}")
        if scaling not in self.SCALINGS:
            raise ValueError(f"scaling phải thuộc {self.SCALINGS}")
        y = self.labels[split]
        return self.data[(split, scaling)], (self._onehot.transform(y) if one_hot else y)

    #Trả về đồng thời cả 6 mảng dữ liệu
    def get_all(self, scaling="minmax", one_hot=False):
        """Trả về X_train, y_train, X_val, y_val, X_test, y_test."""
        out = []
        for s in self.SPLITS:
            out.extend(self.get(s, scaling, one_hot))
        return tuple(out)

    #Chuyển đổi dữ liệu dạng vector phẳng (N, 784) trở lại định dạng ảnh chuẩn (N, 1, 28, 28) (thêm kênh màu 1 cho ảnh xám)
    def get_images(self, split="train", scaling="minmax"):
        """Dữ liệu dạng ảnh (N, 1, 28, 28), dùng cho CNN (phần nâng cao)."""
        X, y = self.get(split, scaling)
        return X.reshape(-1, 1, IMG_SIZE, IMG_SIZE), y

    #In ra bảng tổng kết chi tiết gồm: Số lượng mẫu, số đặc trưng, tỷ lệ phân bố của từng lớp (tính theo %) trên mỗi tập, 
    # và các chỉ số thống kê (min, max, mean, std) của dữ liệu sau khi chuẩn hóa
    def summary(self):
        lines = [f"{'Tập':<6}{'Số mẫu':>8}{'Đặc trưng':>11}   Tỉ lệ lớp (%)"]
        for s in self.SPLITS:
            X, y = self.get(s, "raw")
            p = np.round(np.bincount(y, minlength=N_CLASSES) / len(y) * 100, 1)
            lines.append(f"{s:<6}{len(y):>8}{X.shape[1]:>11}   {p.tolist()}")
        for sc in ("minmax", "standard"):
            X = self.data[("train", sc)]
            lines.append(f"train/{sc}: min={X.min():.3f} max={X.max():.3f} "
                         f"mean={X.mean():.4f} std={X.std():.4f}")
        text = "\n".join(lines)
        print(text)
        return text

    #Lưu toàn bộ các mảng dữ liệu đã xử lý cùng thông số của StandardScaler (mu, sigma, eps) vào một tệp nén
    def save(self, path=os.path.join(DATA_DIR, "mnist_preprocessed.npz")):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrays = {f"X_{s}_{sc}": X for (s, sc), X in self.data.items() if sc != "raw"}
        arrays.update({f"y_{s}": y for s, y in self.labels.items()})
        if self.standard_scaler is not None:
            arrays["mu"] = self.standard_scaler.mean_
            arrays["sigma"] = self.standard_scaler.scale_
            arrays["eps"] = np.array(self.standard_scaler.eps)
        np.savez_compressed(path, **arrays)
        print(f"Đã lưu {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
        return path

    @classmethod
    #Nạp ngược lại tệp .npz đã lưu trước đó để khôi phục nhanh chóng đối tượng MNISTDataset đầy đủ 
    # mà không cần huấn luyện hay xử lý lại từ đầu.
    def load(cls, path=os.path.join(DATA_DIR, "mnist_preprocessed.npz")):
        """Nạp lại file đã save(), không cần chạy lại pipeline."""
        with np.load(path) as d:
            data, labels = {}, {}
            for s in cls.SPLITS:
                labels[s] = d[f"y_{s}"]
                for sc in ("minmax", "standard"):
                    data[(s, sc)] = d[f"X_{s}_{sc}"]
                data[(s, "raw")] = (data[(s, "minmax")] * 255).round().astype(np.float32)
            scaler = None
            if "mu" in d:
                scaler = StandardScaler(eps=float(d["eps"]))
                scaler.mean_, scaler.scale_ = d["mu"], d["sigma"]
        return cls(data, labels, scaler)


class MNISTPreprocessor:
    """Tự động hóa các bước: Tải → nạp → làm phẳng → chia train/val → chuẩn hoá (min-max và z-score).

    Ví dụ:
        prep = MNISTPreprocessor(val_size=0.1, seed=42)
        ds = prep.run()                         # MNISTDataset
        X_tr, y_tr = ds.get("train", "minmax")
        x = prep.transform_new(img_28x28)       # xử lý ảnh mới (demo vẽ chữ số)
    """

    def __init__(self, val_size=VAL_RATIO, seed=42, eps=1e-2, data_dir=DATA_DIR):
        self.loader = MNISTLoader(data_dir)
        self.flattener = Flattener()
        self.splitter = StratifiedSplitter(val_size, seed)
        self.minmax = MinMaxScaler()
        self.standard = StandardScaler(eps)
        self.dataset_ = None

    def run(self, force_download=False, verbose=True):
        self.loader.download(force=force_download, verbose=verbose)
        X_train_img, y_train_full, X_test_img, y_test = self.loader.load()

        # 1. Làm phẳng
        X_train_full = self.flattener.transform(X_train_img)
        X_test = self.flattener.transform(X_test_img)

        # 2. Chia train / validation
        X_train, X_val, y_train, y_val = self.splitter.split(X_train_full, y_train_full)

        # 3. Chuẩn hoá: min-max cố định 0-255; z-score fit trên train (đã min-max)
        raw = {"train": X_train, "val": X_val, "test": X_test}
        data = {}
        for s, X in raw.items():
            data[(s, "raw")] = X
            data[(s, "minmax")] = self.minmax.transform(X)
        self.standard.fit(data[("train", "minmax")])
        for s in raw:
            data[(s, "standard")] = self.standard.transform(data[(s, "minmax")])

        labels = {"train": y_train, "val": y_val, "test": y_test}
        self.dataset_ = MNISTDataset(data, labels, self.standard)
        if verbose:
            self.dataset_.summary()
        return self.dataset_

    def transform_new(self, images, scaling="minmax"):
        """Xử lý ảnh mới (VD: chữ số người dùng vẽ trong demo) giống hệt dữ liệu train.
        images: (28, 28) hoặc (N, 28, 28), giá trị 0-255, nền đen chữ trắng như MNIST."""
        X = self.minmax.transform(self.flattener.transform(images))
        if scaling == "minmax":
            return X
        if scaling == "standard":
            return self.standard.transform(X)
        raise ValueError("scaling phải là 'minmax' hoặc 'standard'")


def get_mnist(scaling="minmax", one_hot=False, val_size=VAL_RATIO, seed=42):
    """Hàm tắt: X_train, y_train, X_val, y_val, X_test, y_test = get_mnist()"""
    ds = MNISTPreprocessor(val_size, seed).run(verbose=False)
    return ds.get_all(scaling, one_hot)

# API dạng hàm – main.ipynb dùng phần này. Mọi hàm đều gọi lại các class ở trên.
def load_mnist_raw(auto_download=True, data_dir=DATA_DIR):
    """Ảnh gốc chưa xử lý: X_train (60000, 28, 28) uint8, y_train int64, X_test, y_test.
    Dùng cho EDA (vẽ ảnh, histogram giá trị 0..255). Tự tải nếu chưa có mnist.npz."""
    return MNISTLoader(data_dir).load(auto_download=auto_download)


def preprocess(X):
    """Làm phẳng 28x28 -> 784 và chuẩn hoá về [0, 1] (chia 255).
    = Flattener + MinMaxScaler mặc định; 255 là hằng số biết trước nên không rò rỉ dữ liệu."""
    return MinMaxScaler().transform(Flattener().transform(X))


def stratified_split(y, val_ratio=VAL_RATIO, seed=42):
    """Chia CHỈ SỐ thành (train_idx, val_idx), giữ tỷ lệ mỗi lớp (dùng StratifiedSplitter)."""
    splitter = StratifiedSplitter(val_ratio, seed)
    y = np.asarray(y)
    splitter.split(np.arange(len(y)), y)
    return splitter.train_idx_, splitter.val_idx_


def stratified_subsample(X, y, n, seed=0):
    """Lấy ĐÚNG n mẫu, giữ tỷ lệ lớp. Dùng cho chạy thử nhanh (QUICK) hoặc Random Forest trên tập con."""
    if n is None or n >= len(y):
        return X, y
    # Chia phần "validation" có len(y) - n mẫu rồi bỏ đi; phần "train" còn lại đúng n mẫu
    keep_idx, _ = stratified_split(y, val_ratio=len(y) - n, seed=seed)
    return X[keep_idx], y[keep_idx]


def load_and_prepare(val_ratio=VAL_RATIO, seed=42, n_train=None, data_dir=DATA_DIR):
    """HÀM CHÍNH của notebook: trả về X_tr, y_tr, X_va, y_va, X_te, y_te.

    Đi đúng pipeline của MNISTPreprocessor (tải -> nạp -> làm phẳng -> chia phân tầng -> min-max):
      val_ratio: tỉ lệ (< 1) hoặc số mẫu (>= 1) của validation. Mặc định 0.1 -> 54.000 / 6.000.
      n_train  : nếu đặt (VD 5000) thì chỉ giữ n_train ảnh train để chạy thử nhanh;
                 validation và test luôn giữ đủ.
    """
    X_full, y_full, X_test, y_test = load_mnist_raw(data_dir=data_dir)
    X_full, X_test = preprocess(X_full), preprocess(X_test)
    X_tr, X_va, y_tr, y_va = StratifiedSplitter(val_ratio, seed).split(X_full, y_full)
    X_tr, y_tr = stratified_subsample(X_tr, y_tr, n_train, seed)
    return X_tr, y_tr, X_va, y_va, X_test, y_test


def load_vn(path=VN_FILE, exclude=VN_EXCLUDE):
    """Tập chữ số người Việt: trả về X_vn (n, 784) trong [0, 1], y_vn, w_vn (mã người viết).
    Tự bỏ các ô liệt kê trong vn_exclude.csv (cột: writer,row,col,reason)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Chưa có {path}. Tạo bằng vn_digits.build_dataset(...)")
    d = np.load(path)
    keep = np.ones(len(d["y"]), dtype=bool)
    if Path(exclude).exists():
        bad = set()
        for line in Path(exclude).read_text(encoding="utf-8").splitlines()[1:]:   # bỏ dòng tiêu đề
            if line.strip():
                w, r, c = line.split(",")[:3]
                bad.add((w.strip(), int(r), int(c)))
        keep = np.array([(w, int(r), int(c)) not in bad for w, (r, c) in zip(d["writer"], d["pos"])])
        print(f"Tập Việt: bỏ {(~keep).sum()} ô theo {Path(exclude).name}")
    return d["X"][keep].astype(np.float32), d["y"][keep].astype(np.int64), d["writer"][keep]


def summary(X_tr, y_tr, X_va, y_va, X_te, y_te):
    """In nhanh kích thước, kiểu, khoảng giá trị và số ảnh mỗi lớp của 3 tập."""
    for name, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
        print(f"{name:5s}: X {X.shape} {X.dtype} [{X.min():.1f}, {X.max():.1f}]  "
              f"mỗi lớp: {np.bincount(y, minlength=N_CLASSES).tolist()}")

