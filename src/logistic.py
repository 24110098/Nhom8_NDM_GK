"""
src/logistic.py – Logistic Regression đa lớp (softmax) huấn luyện bằng mini-batch Gradient Descent.

TÓM TẮT Ý TƯỞNG
---------------
Mỗi lớp k (chữ số 0..9) có một "tấm khuôn" trọng số w_k gồm 784 số, cùng kích thước ảnh.
Điểm của lớp k cho ảnh x:      z_k = x · w_k + b_k
  - pixel sáng nằm ở chỗ khuôn có trọng số dương  -> tăng điểm lớp k
  - pixel sáng nằm ở chỗ khuôn có trọng số âm     -> giảm điểm lớp k
Softmax biến 10 điểm thành 10 xác suất (dương, tổng bằng 1). Lớp có xác suất cao nhất là dự đoán.

Huấn luyện = tìm W, b sao cho xác suất của lớp ĐÚNG càng cao càng tốt,
tức là làm nhỏ hàm mất mát cross-entropy, bằng cách đi ngược hướng gradient.

KÝ HIỆU KÍCH THƯỚC (dùng trong các chú thích bên dưới)
  n : số ảnh          D : số pixel = 784          K : số lớp = 10
  X : (n, D)   W : (D, K)   b : (K,)   Z = XW + b : (n, K)   P = softmax(Z) : (n, K)
"""
import numpy as np


def softmax(Z):
    """Biến mỗi HÀNG điểm số của Z (n, K) thành phân phối xác suất trên K lớp.

    Công thức: P[i, k] = exp(Z[i, k]) / sum_l exp(Z[i, l])

    Vì sao trừ giá trị lớn nhất trước khi lấy exp?
      exp(1000) vượt quá giới hạn số thực -> inf, rồi inf/inf = nan.
      softmax(z) = softmax(z - c) với mọi hằng số c (tử và mẫu cùng nhân e^-c),
      nên trừ max của từng hàng không đổi kết quả mà mọi số mũ đều <= 0, không tràn.
    """
    Z = Z - Z.max(axis=1, keepdims=True)          # keepdims=True: giữ shape (n, 1) để trừ theo từng hàng
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)       # chia cho tổng của hàng -> mỗi hàng cộng lại bằng 1


def one_hot(y, K):
    """Đổi nhãn số thành vector one-hot.
    Ví dụ K = 4, y = [2, 0]  ->  [[0, 0, 1, 0],
                                  [1, 0, 0, 0]]
    np.eye(K) là ma trận đơn vị K x K; lấy hàng thứ y[i] chính là one-hot của y[i].
    """
    return np.eye(K)[y]


class LogisticRegressionGD:
    def __init__(self, lr=0.1, lam=1e-4, epochs=20, batch_size=128, seed=0):
        """Chỉ lưu siêu tham số; trọng số được tạo trong fit().

        lr         : learning rate (eta) – độ dài mỗi bước đi xuống dốc.
                     Quá lớn: loss dao động hoặc ra nan. Quá nhỏ: học rất chậm.
        lam        : hệ số L2 (lambda) – phạt trọng số lớn để giảm overfitting.
        epochs     : số lượt duyệt hết tập train.
        batch_size : số ảnh dùng cho MỘT lần cập nhật (mini-batch).
        seed       : cố định ngẫu nhiên (khởi tạo W, thứ tự ảnh) để chạy lại ra cùng kết quả.
        """
        self.lr, self.lam, self.epochs, self.bs, self.seed = lr, lam, epochs, batch_size, seed

    # ------------------------------------------------------------------
    def loss(self, X, y):
        """Hàm mất mát cần làm nhỏ:

            L = -(1/n) * sum_i log P[i, y_i]   +   (lam/2) * ||W||^2
                \\_____ cross-entropy _____/       \\__ phạt L2 __/

        Cross-entropy của một ảnh = -log(xác suất mô hình gán cho lớp ĐÚNG):
          xác suất lớp đúng 0.9  -> loss 0.105 (nhỏ, tốt)
          xác suất lớp đúng 0.1  -> loss 2.303 (lớn, bị phạt nặng)
        Không phạt b vì b chỉ dịch điểm số, không làm mô hình "phức tạp" hơn.
        """
        P = softmax(X @ self.W + self.b)                          # (n, K)
        # P[np.arange(n), y] lấy đúng 1 phần tử mỗi hàng: xác suất của lớp đúng y_i
        # + 1e-12 để tránh log(0) = -inf khi mô hình gán xác suất 0 cho lớp đúng
        ce = -np.log(P[np.arange(len(y)), y] + 1e-12).mean()
        return ce + 0.5 * self.lam * (self.W ** 2).sum()

    def gradient(self, X, y):
        """Gradient của loss theo W và b (dùng để gradient check).

        Kết quả dẫn xuất (xem Chương 5 của tài liệu hướng dẫn):
            dL/dZ = (P - Y) / n                    "dự đoán trừ đáp án", shape (n, K)
            dL/dW = X^T (P - Y) / n  +  lam * W     shape (D, K)
            dL/db = tổng theo các ảnh của (P - Y) / n   shape (K,)
        Phải dùng ĐÚNG công thức như trong fit(), để kiểm tra đúng code đang huấn luyện.
        """
        P = softmax(X @ self.W + self.b)
        G = (P - one_hot(y, self.W.shape[1])) / len(y)
        return X.T @ G + self.lam * self.W, G.sum(axis=0)
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Huấn luyện bằng mini-batch Gradient Descent.

        X_val, y_val (không bắt buộc): nếu có, ghi lại accuracy validation sau mỗi epoch
        để vẽ learning curve và chọn siêu tham số.
        """
        rng = np.random.default_rng(self.seed)
        n, D = X.shape
        K = int(y.max()) + 1                       # số lớp lấy từ nhãn lớn nhất (MNIST: 9 + 1 = 10)

        # Khởi tạo: W là số ngẫu nhiên rất nhỏ quanh 0, b = 0.
        # Lúc đầu mọi điểm số gần 0 nên softmax cho xác suất gần đều 1/K: mô hình "chưa biết gì".
        self.W = rng.normal(0, 0.01, (D, K))       # (D, K)
        self.b = np.zeros(K)                       # (K,)
        Y = one_hot(y, K)                          # (n, K), làm một lần trước vòng lặp cho nhanh

        self.history = {"train_loss": [], "val_acc": []}
        for _ in range(self.epochs):
            # Mỗi epoch xáo trộn thứ tự ảnh, để mỗi mini-batch là một mẫu ngẫu nhiên
            idx = rng.permutation(n)

            # Duyệt từng mini-batch: s = 0, 128, 256, ...  (batch cuối có thể ít hơn 128 ảnh)
            for s in range(0, n, self.bs):
                bi = idx[s:s + self.bs]            # chỉ số các ảnh trong batch này

                # (1) Lan truyền thuận: điểm số -> xác suất
                P = softmax(X[bi] @ self.W + self.b)          # (B, K)

                # (2) Sai số "dự đoán trừ đáp án", chia cho số ảnh trong batch để lấy trung bình.
                G = (P - Y[bi]) / len(bi)                     # (B, K)

                # (3) Cập nhật: W <- W - lr * gradient
                #     X[bi].T @ G  : (D, B) @ (B, K) = (D, K), gradient phần cross-entropy
                #     self.lam * W : gradient của (lam/2)||W||^2
                #     cột của lớp đúng được CỘNG thêm "một phần ảnh x",
                #     cột của lớp sai bị TRỪ đi -> dần dần mỗi cột giống hình chữ số của lớp đó.
                self.W -= self.lr * (X[bi].T @ G + self.lam * self.W)
                self.b -= self.lr * G.sum(axis=0)             # gradient theo b: cộng G theo các ảnh

            # Sau mỗi epoch: ghi lại loss trên TOÀN BỘ tập train (để vẽ learning curve)
            self.history["train_loss"].append(self.loss(X, y))
            if X_val is not None:
                self.history["val_acc"].append((self.predict(X_val) == y_val).mean())
        return self                                

    def predict_proba(self, X):
        """Xác suất của từng lớp, shape (n, K). Dùng cho demo hoặc phân tích độ tự tin."""
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        """Lớp có xác suất cao nhất. argmax của softmax = argmax của điểm số,
        nên thực ra không cần softmax ở bước này; giữ lại cho dễ đọc."""
        return self.predict_proba(X).argmax(axis=1)

def grad_check(loss_fn, W, dW, n_checks=10, h=1e-5, seed=0):
    """Kiểm tra gradient giải tích dW có đúng không, bằng gradient số (sai phân trung tâm).

    Ý tưởng: đạo hàm = độ dốc. Nhích một phần tử W[j, k] lên h rồi xuống h,
    xem loss thay đổi bao nhiêu:
        g_num = (L(W + h) - L(W - h)) / (2h)
    Nếu công thức gradient đúng thì g_num ≈ dW[j, k].

    loss_fn : hàm KHÔNG tham số, tính loss với W hiện tại, ví dụ lambda: model.loss(Xs, ys).
              W được sửa TRỰC TIẾP (cùng một mảng với model.W) nên loss_fn thấy ngay thay đổi.
    W, dW   : ma trận trọng số và gradient giải tích tương ứng (cùng shape).
    Trả về  : sai số tương đối lớn nhất. Đạt nếu < 1e-6 (thường cỡ 1e-8 hoặc nhỏ hơn).

    Lưu ý khi dùng: dùng dữ liệu float64 và vài chục ảnh; đặt lam > 0 để kiểm tra luôn phần L2.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_checks):
        idx = tuple(rng.integers(s) for s in W.shape)   # chọn ngẫu nhiên một vị trí (j, k)
        old = W[idx]
        W[idx] = old + h; Lp = loss_fn()                # loss khi nhích lên
        W[idx] = old - h; Lm = loss_fn()                # loss khi nhích xuống
        W[idx] = old                                    # TRẢ LẠI giá trị cũ, không làm hỏng mô hình
        g_num = (Lp - Lm) / (2 * h)
        # Sai số TƯƠNG ĐỐI (không dùng sai số tuyệt đối vì gradient có thể rất nhỏ hoặc rất lớn);
        # + 1e-12 để tránh chia cho 0 khi cả hai gradient đều bằng 0.
        rel = abs(g_num - dW[idx]) / (abs(g_num) + abs(dW[idx]) + 1e-12)
        worst = max(worst, rel)
        print(idx, f"số: {g_num:.6e}  giải tích: {dW[idx]:.6e}  sai số tương đối: {rel:.1e}")
    return worst
