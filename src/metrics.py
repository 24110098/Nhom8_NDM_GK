"""
src/metrics.py – độ đo đánh giá phân lớp, tự cài bằng NumPy.

Dùng trong notebook:
    from metrics import confusion_matrix, prf_macro
    C = confusion_matrix(y_true, y_pred, K=10)
    precision, recall, f1, accuracy = prf_macro(C)

QUY ƯỚC MA TRẬN NHẦM LẪN
    C[i, j] = số ảnh có NHÃN THẬT i (hàng) bị DỰ ĐOÁN là j (cột)
    - đường chéo C[k, k]  : đoán đúng lớp k
    - tổng HÀNG k         : số ảnh thật sự thuộc lớp k
    - tổng CỘT k          : số ảnh mô hình bảo là lớp k

    Precision_k = C[k, k] / tổng cột k   "mô hình bảo là k thì đúng bao nhiêu phần?"
    Recall_k    = C[k, k] / tổng hàng k  "ảnh thật là k thì bắt được bao nhiêu phần?"
    F1_k        = 2 P_k R_k / (P_k + R_k)  (trung bình điều hoà của P và R)

    Macro = trung bình cộng đều trên các lớp (mỗi lớp quan trọng như nhau).
    Với bài toán đơn nhãn, micro-average đúng bằng accuracy.
"""
import numpy as np


def confusion_matrix(y_true, y_pred, K=10):
    """Ma trận nhầm lẫn K x K (kiểu int).

    np.add.at(C, (y_true, y_pred), 1) cộng 1 vào ô C[y_true[i], y_pred[i]] cho MỌI i,
    kể cả khi một ô xuất hiện nhiều lần (C[y_true, y_pred] += 1 thì chỉ cộng 1 lần, SAI).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} và y_pred {y_pred.shape} phải cùng kích thước")
    C = np.zeros((K, K), dtype=np.int64)
    np.add.at(C, (y_true, y_pred), 1)
    return C


def per_class(C):
    """Precision, recall, F1 của TỪNG lớp, mỗi cái là mảng (K,).

    Lớp không có ảnh nào được đoán (tổng cột = 0) hoặc không có ảnh thật (tổng hàng = 0)
    sẽ chia cho 0; quy ước gán 0 (giống sklearn với zero_division=0) thay vì ra nan.
    """
    C = np.asarray(C, dtype=np.float64)
    tp = np.diag(C)
    col, row = C.sum(axis=0), C.sum(axis=1)
    prec = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    rec = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def prf_macro(C):
    """Trả về (precision_macro, recall_macro, f1_macro, accuracy) từ ma trận nhầm lẫn."""
    prec, rec, f1 = per_class(C)
    accuracy = np.trace(C) / np.sum(C)
    return prec.mean(), rec.mean(), f1.mean(), accuracy


def classification_report(y_true, y_pred, K=10, digits=4):
    """Bảng giống sklearn.metrics.classification_report, trả về chuỗi để in."""
    C = confusion_matrix(y_true, y_pred, K)
    prec, rec, f1 = per_class(C)
    support = C.sum(axis=1)
    lines = [f"{'lớp':>6} {'precision':>10} {'recall':>10} {'f1':>10} {'số ảnh':>8}"]
    for k in range(K):
        lines.append(f"{k:>6} {prec[k]:>10.{digits}f} {rec[k]:>10.{digits}f} {f1[k]:>10.{digits}f} {support[k]:>8d}")
    p, r, f, acc = prf_macro(C)
    lines.append("")
    lines.append(f"{'macro':>6} {p:>10.{digits}f} {r:>10.{digits}f} {f:>10.{digits}f} {support.sum():>8d}")
    lines.append(f"{'acc':>6} {acc:>10.{digits}f}")
    return "\n".join(lines)


def top_confusions(C, n=5):
    """n cặp (nhãn thật, dự đoán) bị nhầm nhiều nhất, bỏ đường chéo. Dùng cho phần phân tích lỗi."""
    off = np.array(C, copy=True)
    np.fill_diagonal(off, 0)
    idx = np.argsort(off, axis=None)[::-1][:n]
    return [(int(i), int(j), int(off[i, j])) for i, j in zip(*np.unravel_index(idx, off.shape)) if off[i, j] > 0]


