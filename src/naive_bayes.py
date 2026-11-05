
import numpy as np

def softmax(Z):
    Z = Z- Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)



class MultinomialNB:
    def __init__(self, alpha=1.0):  #tham số làm trơn Laplace, giúp tránh xác suất bằng 0
        self.alpha = alpha

    #hàm huấn luyện mô hình
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64) #ma trận đặc trưng của các ảnh
        y = np.asarray(y)                   #nhãn đúng tương ứng với từng ảnh

        self.classes = np.unique(y)         #số lớp cần phân loại
        n_samples, n_features = X.shape     #số mẫu cần huấn luyện, số đặc trưng mỗi ảnh
        n_classes = len(self.classes)

        self.log_prior = np.zeros(n_classes) #lưu log xác suất tiên nghiệm của từng số
        self.log_feature_prob = np.zeros((n_classes, n_features)) #lưu log xác suất đặc trưng từng lớp

        #học lớp chữ số
        for i, c in enumerate(self.classes):    #i: biến lặp , c: nhãn số thực tế
            X_c = X[y == c]                     #lấy những ảnh có nhãn là c và lưu vào xc

            # Xác suất tiên nghiệm của lớp
            self.log_prior[i] = np.log(len(X_c) / n_samples)

            # Tổng giá trị từng pixel trong lớp
            count = X_c.sum(axis=0)         

            # Làm trơn Laplace
            feature_prob = (
                (count + self.alpha)
                / (count.sum() + self.alpha * n_features)
            )

            self.log_feature_prob[i] = np.log(feature_prob)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)

        # Tính điểm cho mọi ảnh và mọi lớp
        score = X @ self.log_feature_prob.T + self.log_prior

        # Lấy lớp có điểm cao nhất
        return self.classes[np.argmax(score, axis=1)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)

        # Điểm log: log P(c) + sum(x_j * log P(j|c)) cho mọi ảnh và mọi lớp
        score = X @ self.log_feature_prob.T + self.log_prior

        # Chuẩn hóa thành xác suất hậu nghiệm P(c|x), mỗi hàng có tổng bằng 1
        return softmax(score)