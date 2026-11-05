"""
Random Forest cho bài toán PHÂN LOẠI, cài đặt từ đầu (không dùng scikit-learn).
Tiêu chí tách: Gini. Chỉ dùng numpy cho các phép tính trên mảng.

Cấu trúc:
    1. Node            - một nút của cây
    2. DecisionTree    - cây quyết định CART dùng Gini
    3. RandomForest    - bootstrap + chọn ngẫu nhiên đặc trưng tại mỗi nút + bỏ phiếu
    4. Hàm tiện ích    - chia train/test, tính accuracy
    5. Demo
"""

import numpy as np


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    # vị trí đặc trưng để tách, ngưỡng giá trị, nút trái, nút phải, giá trị nếu là lá

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value          # lá: vector tỉ lệ các lớp, ví dụ [0, 0.25, 0.75]

    def is_leaf(self):              # kiểm tra đây có phải lá hay không
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, rng=None):
        self.max_depth = max_depth                  # độ sâu tối đa của cây
        self.min_samples_split = min_samples_split  # số mẫu tối thiểu để được tách, ít hơn -> lá
        self.min_samples_leaf = min_samples_leaf    # số mẫu tối thiểu của 1 lá con sau khi tách
        self.max_features = max_features            # số đặc trưng ngẫu nhiên được xét tại MỖI nút
        self.rng = rng if rng is not None else np.random.default_rng()  # bộ sinh số ngẫu nhiên

    def fit(self, X, y, n_classes=None):
        self.n_features = X.shape[1]                # tổng số đặc trưng
        if self.max_features is None:
            self.max_features = self.n_features
        self.n_classes = n_classes if n_classes is not None else int(y.max()) + 1
        self.feature_importances_ = np.zeros(self.n_features)

        self.root = self._grow(X, y, depth=0)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _gini(self, y):
        """Độ vẩn đục Gini của một tập nhãn: G = 1 - sum(p_k^2)."""
        p = np.bincount(y, minlength=self.n_classes) / len(y)
        return 1.0 - np.sum(p ** 2)

    def _leaf_value(self, y):
        """Lá lưu vector tỉ lệ các lớp."""
        return np.bincount(y, minlength=self.n_classes) / len(y)

    def _grow(self, X, y, depth):
        n = len(y)

        # điều kiện dừng -> tạo lá 
        if (self.max_depth is not None and depth >= self.max_depth) \
                or n < self.min_samples_split \
                or self._gini(y) == 0:
            return Node(value=self._leaf_value(y))

        # ĐIỂM NGẪU NHIÊN THỨ 2 CỦA RANDOM FOREST
        # chỉ xét một tập con ngẫu nhiên các đặc trưng tại nút này
        feats = self.rng.choice(self.n_features, self.max_features, replace=False)

        best = self._best_split(X, y, feats)
        if best is None:                            # không tìm được cách tách có lợi
            return Node(value=self._leaf_value(y))

        f, thr, gain = best
        # độ quan trọng đặc trưng = tổng (số mẫu tại nút * mức giảm Gini)
        self.feature_importances_[f] += n * gain

        mask = X[:, f] <= thr
        left = self._grow(X[mask], y[mask], depth + 1)
        right = self._grow(X[~mask], y[~mask], depth + 1)
        return Node(feature=f, threshold=thr, left=left, right=right)

    def _best_split(self, X, y, feats):
        """
        Duyệt từng đặc trưng trong feats, thử MỌI ngưỡng tách và chọn cách tách
        làm giảm Gini nhiều nhất:  Δ = G_cha - (n_L/n) G_L - (n_R/n) G_R

        Mẹo tăng tốc: sắp xếp theo giá trị đặc trưng (O(n log n)), rồi dùng tổng
        cộng dồn (cumsum) để tính Gini của mọi vị trí tách trong O(n).
        """
        n = len(y)                                  # số mẫu tại nút
        parent_gini = self._gini(y)                 # G cha
        n_left = np.arange(1, n)                    # vị trí j: trái có j+1 mẫu
        n_right = n - n_left                        # phải có n-(j+1) mẫu; n mẫu -> n-1 vị trí tách
        m = self.min_samples_leaf                   # số mẫu tối thiểu 1 lá

        best_gain, best = 0.0, None                 # chưa tìm được cách tách nào
        for f in feats:                             # feats là tập con, f là một đặc trưng
            # sắp xếp các mẫu theo giá trị đặc trưng f tăng dần
            order = np.argsort(X[:, f], kind="mergesort")
            xs, ys = X[order, f], y[order]  #ys: nhãn các mẫu nút hiện tại đc sx theo gia trị đặc trưng

            # tính Gini cho mọi vị trí tách cùng lúc
            onehot = np.eye(self.n_classes)[ys]                 # (n, K), tạo ma trận vuông cỡ nxn, với mỗi ys, ta đánh số 1 vào vị trí tương ứng
            left_cnt = np.cumsum(onehot, axis=0)[:-1]           # số mẫu mỗi lớp bên trái
            right_cnt = onehot.sum(axis=0) - left_cnt           # số mẫu mỗi lớp bên phải
            gini_l = 1 - np.sum((left_cnt / n_left[:, None]) ** 2, axis=1)  #gini tại bên trái mỗi vị trí tách
            gini_r = 1 - np.sum((right_cnt / n_right[:, None]) ** 2, axis=1) #gini tại bên phải mỗi vị trí tách
            child_gini = (n_left * gini_l + n_right * gini_r) / n   # Gini có trọng số

            # chỉ tách giữa hai giá trị KHÁC nhau và mỗi bên đủ min_samples_leaf
            #mọi pt trừ phần tử cuối < mọi phần tử trừ pt đầu và mỗi nhánh con phải có nhiều hơn số đặc trưng tối thiểu 
            valid = (xs[:-1] < xs[1:]) & (n_left >= m) & (n_right >= m)
            if not valid.any(): #trả về true nếu có 1 true
                continue    #chuyển qua đặc trưng kế nếu kh có true
            #vô hiệu vị trí kh hợp lệ
            child_gini = np.where(valid, child_gini, np.inf) #với valid chỗ nào hợp lệ thì giữ nguyên gini, kh hợp lệ là inf
            #chọn vị trí tốt nhất
            j = int(np.argmin(child_gini)) #trả về vị trí pt nhỏ nhất
            gain = parent_gini - child_gini[j]      # Δ: ct giảm gini
            #so sánh với đặc trưng đã xét trước đó
            if gain > best_gain:
                best_gain = gain
                best = (f, (xs[j] + xs[j + 1]) / 2.0, gain)    # ngưỡng = trung điểm
        return best

    # dự đoán
    def _traverse(self, x, node):  #đưa 1 mẫu (X) từ gốc xuống lá
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, X): #dự đoán cho nhiều mẫu
        """Trả về xác suất các lớp, kích thước (số mẫu, K)."""
        return np.array([self._traverse(x, self.root) for x in X])

    def predict(self, X):
        """Trả về chỉ số lớp (0..K-1) cho từng mẫu."""
        return self.predict_proba(X).argmax(axis=1)

#n_estimator: số cây, bootstrap: quyết định lấy mẫu có hoàn lại hay kh, tắt đi thì cây học trên toàn bộ dữ liệu
#ôb_score: quyết định có tính độ chính xác out-of-bag hay kh
#voting: chọn cách gộp: hard: đếm phiếu; soft: trung bình xác suất, randon_state: hạt giống ngẫu nhiên
class RandomForest:
    def __init__(self, n_estimators=100 , max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", bootstrap=True,
                 oob_score=False, voting="soft", random_state=None):
        self.n_estimators = n_estimators            # số cây
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features            # "sqrt", "log2", số nguyên, số thực hoặc None
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.voting = voting                        # "hard" = đếm phiếu, "soft" = trung bình xác suất
        self.random_state = random_state

    def _resolve_max_features(self, p): #p: tổng đặc trưng
        """Đổi max_features về số nguyên trước khi truyền cho cây."""
        mf = self.max_features
        if mf is None:
            return p
        if mf == "sqrt":
            return max(1, int(np.sqrt(p)))
        if mf == "log2":
            return max(1, int(np.log2(p)))
        if isinstance(mf, float):
            return max(1, int(mf * p))
        return min(int(mf), p)

    #lấy ý kiến của 1 cây
    def _tree_output(self, tree, X):
        out = tree.predict_proba(X) #lấy vecto xác suất mà cây đưa ra cho mỗi mẫu
        if self.voting == "hard":
            out = np.eye(self.n_classes_)[out.argmax(axis=1)]   # mỗi cây = 1 phiếu giống onehot
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, p = X.shape

        # mã hoá nhãn về 0..K-1 (ví dụ 0,1,7 -> 0,1,2)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        #bộ sinh số ngẫu nhiên
        k = self._resolve_max_features(p)
        rng = np.random.default_rng(self.random_state)

        if self.oob_score: #thu thập dự đoán out_of_bag
            oob_sum = np.zeros((n, self.n_classes_))  #có 1 hàng cho mỗi mẫu hl, cộng dồn ý kiến các cây kh có mẫu đó
            oob_cnt = np.zeros(n)       #đếm mỗi mẫu đã đc bao nhiêu cây dự đoán như vậy

        #vòng lặp tạo từng cây
        self.trees_ = []
        for _ in range(self.n_estimators):
            tree_rng = np.random.default_rng(rng.integers(2 ** 31))

            # ĐIỂM NGẪU NHIÊN THỨ 1: BOOTSTRAP (lấy mẫu CÓ HOÀN LẠI) 
            idx = tree_rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            #tạo 1 cây với cấu hình rừng, max_feature đc đổi thành số nguyên
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                max_features=k, rng=tree_rng)
            tree.fit(X[idx], y_enc[idx], n_classes=self.n_classes_) #train theo decision tree
            self.trees_.append(tree)

            # OUT-OF-BAG: dự đoán các mẫu mà cây này KHÔNG được thấy 
            if self.oob_score and self.bootstrap:
                oob = np.ones(n, dtype=bool)
                oob[idx] = False
                if oob.any():
                    oob_sum[oob] += self._tree_output(tree, X[oob])
                    oob_cnt[oob] += 1

        # độ quan trọng đặc trưng = trung bình qua các cây
        #lấy trung bình các vecto đặc trung quan trọng rồi chuẩn hóa
        imp = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        #tìm điểm OOB
        if self.oob_score and self.bootstrap:
            seen = oob_cnt > 0 #chỉ đánh giá khi ít nhất 1 cây ôb dự đoán
            pred = oob_sum[seen].argmax(axis=1) #lấy lớp có tổng điểm cao nhất mỗi mẫu
            self.oob_score_ = float(np.mean(pred == y_enc[seen])) #tính tỷ lệ t/f
        return self

    # TỔNG HỢP KẾT QUẢ CÁC CÂY 
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.mean([self._tree_output(t, X) for t in self.trees_], axis=0)

    def predict(self, X):
        """Trả về nhãn gốc (ví dụ chữ số 0..9 với MNIST)."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

def train_test_split(X, y, test_size=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_size)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))
