# Đồ án 1 – Nhận dạng chữ số viết tay bằng các phương pháp phân lớp (MNIST)

Nhóm tự cài đặt (bằng NumPy) và so sánh bốn bộ phân lớp trên MNIST:

| Yêu cầu của đề | Phương pháp nhóm chọn | File |
|---|---|---|
| Naïve Bayes | Multinomial Naïve Bayes, làm trơn Laplace | `src/naive_bayes.py` |
| Bộ phân lớp tuyến tính | Logistic Regression đa lớp (softmax), mini-batch Gradient Descent | `src/logistic.py` |
| Mạng nơ-ron nhiều lớp | MLP, lan truyền ngược tự viết, SGD + momentum, early stopping | `src/mlp.py` |
| Ensemble | Random Forest (CART, Gini, bootstrap, OOB score) | `src/random_forest.py` |

Phần nâng cao:

- **Bộ chữ số viết tay người Việt** do nhóm tự thu thập, dùng để đo hiện tượng lệch miền (domain shift).
- **Tăng cường dữ liệu** (xoay, dịch, nhiễu) và thí nghiệm độ bền.
- **Demo** vẽ chữ số bằng Gradio.

scikit-learn **chỉ** được dùng để đối chiếu kết quả, không dùng cho phần tính toán lõi.

## Thành viên

| Họ tên | MSSV | Phụ trách |
|---|---|---|
| Nguyễn Thị Huệ Giang | 24110014 | Multinomial NB, Random Forest, thí nghiệm độ bền |
| Lâm Hà Quỳnh Như | 24110043 | Dữ liệu, MLP, thí nghiệm tập Việt |
| Hồ Việt Thuận Phú | 24110098 | Logistic Regression, `metrics.py`, xử lý phiếu chữ viết, demo |

Tỷ lệ đóng góp chi tiết nằm trong báo cáo.

## Cấu trúc thư mục

```text
mnist-project/
├── app.py                  # demo vẽ chữ số (Gradio)
├── data/
│   ├── download_mnist.py   # (tuỳ chọn) script tải MNIST riêng
│   ├── mnist.npz           # MNIST đã tải (tự tạo, KHÔNG commit)
│   ├── vn_raw/             # ảnh chụp phiếu chữ viết (KHÔNG commit: có chữ ký)
│   ├── vn_digits.npz       # tập chữ số người Việt đã cắt
│   ├── vn_exclude.csv      # các ô bị loại sau khi kiểm tra bằng mắt
│   └── writers.csv         # sổ theo dõi người viết (chỉ mã, không có họ tên)
├── src/
│   ├── data.py             # tải, đọc, tiền xử lý, chia tập
│   ├── naive_bayes.py      # Multinomial NB
│   ├── logistic.py         # Logistic Regression + gradient check
│   ├── mlp.py              # MLP + backpropagation
│   ├── random_forest.py    # Random Forest
│   ├── metrics.py          # confusion matrix, precision/recall/F1 macro
│   ├── vn_digits.py        # nắn phẳng phiếu, cắt ô, tạo tập chữ số Việt
│   └── experiments.py      # augmentation, thí nghiệm độ bền, chia theo người viết
├── notebooks/
│   └── main.ipynb          # notebook chính, tái tạo mọi bảng và hình
├── results/                # bảng (.csv), hình (.png), mô hình đã lưu (models/*.pkl)
├── report/                 # báo cáo PDF, slide
├── requirements.txt
└── README.md
```

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```bash
# 1. Tạo môi trường ảo (khuyên dùng)
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Cài thư viện
pip install -r requirements.txt
```

`requirements.txt` gồm:

```text
numpy
scipy
pandas
matplotlib
scikit-learn      # chỉ để đối chiếu; khi import dùng tên "sklearn"
opencv-python     # xử lý ảnh phiếu chữ viết
Pillow
gradio            # demo
torch             # tuỳ chọn: chỉ dùng ở ô đối chiếu MLP bằng PyTorch (mục 5.4)
```

## Chạy

### 1. Tải dữ liệu MNIST

```bash
python src/data.py
```

Lệnh này tải MNIST về `data/mnist.npz`. Nếu bỏ qua bước này, notebook cũng tự tải ở lần chạy đầu tiên (cần mạng).

Quy ước dữ liệu dùng chung cho cả bốn mô hình:

- Tập train gốc được chia phân tầng thành **54.000 train / 6.000 validation**; tập test giữ nguyên **10.000** ảnh.
- Ảnh được làm phẳng thành vector 784 chiều và chuẩn hoá min-max (chia 255) về [0, 1].

### 2. Chạy notebook

Mở `notebooks/main.ipynb` rồi chạy từ trên xuống. Hai công tắc trong ô thiết lập (mục 0):

| Biến | Ý nghĩa |
|---|---|
| `QUICK = True` | Chạy thử nhanh trên 5.000 ảnh train, ít epoch, ít cây (vài phút) |
| `QUICK = False` | Chạy đầy đủ để lấy số liệu cho báo cáo |
| `RUN_TEST = True` | Đánh giá trên tập test. **Chỉ bật ở lần chạy cuối**, sau khi đã chốt siêu tham số trên validation |

Để tái tạo toàn bộ kết quả trong báo cáo: đặt `QUICK = False`, `RUN_TEST = True`, rồi chọn **Restart & Run All**.

Các mục trong notebook:

| Mục | Nội dung |
|---|---|
| 0 | Thiết lập |
| 1 | Dữ liệu và EDA |
| 2 | Công cụ đánh giá chung |
| 3–6 | Multinomial NB, Logistic Regression, MLP, Random Forest (huấn luyện, đối chiếu sklearn, siêu tham số) |
| 7 | Tổng hợp trên validation |
| 8 | Đánh giá cuối trên test |
| 9 | Thí nghiệm mở rộng: chữ số người Việt |
| 10 | Tăng cường dữ liệu và thí nghiệm độ bền |
| 11 | Lưu mô hình cho demo |

Mọi bảng và hình được lưu vào `results/`.

### 3. Tạo tập chữ số người Việt (phần nâng cao)

1. In `Phieu_thu_thap_chu_so.pdf` **đúng 100% kích thước** (tắt "Fit to page").
2. Mời người viết điền phiếu. Người viết phải đánh dấu ô đồng ý.
3. Chụp hoặc scan phiếu, thấy rõ 4 ô vuông đen ở góc. Lưu vào `data/vn_raw/` với **tên file là mã người viết**, ví dụ `W01.jpg`, `W02.jpg`.
4. Chạy mục **9.0** trong notebook:
   - kiểm tra căn chỉnh (khung đỏ phải nằm gọn trong từng ô);
   - xem các ô đã cắt;
   - tạo `data/vn_digits.npz`.
5. Các ô viết nhầm, bị gạch xoá hay bẩn: ghi vào `data/vn_exclude.csv` với các cột `writer,row,col,reason`.

### 4. Chạy demo

Cần chạy mục 11 của notebook trước để lưu mô hình vào `results/models/`. Sau đó, đứng ở thư mục gốc repo:

```bash
python app.py
```

Mở địa chỉ hiện trong terminal (mặc định `http://127.0.0.1:7860`), vẽ một chữ số và xem bốn mô hình dự đoán. Giao diện hiện cả ảnh 28×28 sau tiền xử lý, tức đúng ảnh mà mô hình nhận vào.

## Kết quả

*(Điền sau khi chạy với `QUICK = False`, `RUN_TEST = True`; số liệu lấy từ `results/summary_final.csv`.)*

| Mô hình | Accuracy test | F1 macro test | Thời gian huấn luyện | Dự đoán (ms/ảnh) |
|---|---|---|---|---|
| Multinomial NB | | | | |
| Logistic Regression | | | | |
| MLP | | | | |
| Random Forest | | | | |

## Kiểm chứng cài đặt

- **Multinomial NB:** tham số và xác suất khớp `sklearn.naive_bayes.MultinomialNB` đến sai số làm tròn.
- **Logistic Regression:** gradient check, sai số tương đối nhỏ hơn 10⁻⁶; accuracy được so với `sklearn.linear_model.LogisticRegression`.
- **MLP:** unit test với ví dụ tính tay mạng 2–2–2; accuracy được so với bản PyTorch cùng kiến trúc.
- **Random Forest:** accuracy được so với `sklearn.ensemble.RandomForestClassifier` cùng tham số; có OOB score.
- **metrics.py:** tự kiểm tra bằng `python src/metrics.py`, rồi đối chiếu với `sklearn.metrics`.

## Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | `pip install scikit-learn` rồi **Restart kernel** |
| `ModuleNotFoundError: No module named 'data'` | Mở notebook từ trong thư mục repo; ô thiết lập tự tìm `src/data.py` |
| `'NoneType' object has no attribute 'ndim'` khi đọc ảnh phiếu | Thường gặp khi đường dẫn có dấu tiếng Việt trên Windows. Dùng bản `src/vn_digits.py` mới (đã xử lý), hoặc đổi tên thư mục không dấu |
| "Chỉ tìm thấy k ô đánh dấu" | Chụp lại cho thấy rõ cả 4 ô đen ở góc phiếu |
| Demo báo "Chưa có mô hình nào" | Chạy mục 11 của notebook để lưu `results/models/*.pkl` |

## Không commit lên Git

Thêm vào `.gitignore`:

```text
data/mnist.npz
data/mnist_raw/
data/mnist_preprocessed.npz
data/vn_raw/
results/models/
__pycache__/
.venv/
.ipynb_checkpoints/
```

Ảnh phiếu gốc (`data/vn_raw/`) có chữ ký người viết nên **không được công bố**. Chỉ commit tập đã cắt `vn_digits.npz` và sổ `writers.csv`; sổ này chỉ chứa mã người viết, không có họ tên.

## Khai báo sử dụng AI

Nhóm có sử dụng trợ lý AI để hỗ trợ: *(mô tả cụ thể phần nào, ví dụ: giải thích lý thuyết, gợi ý cấu trúc code, sửa lỗi)*. Mọi mã nguồn nộp đều đã được nhóm đọc hiểu, chạy kiểm thử và chịu trách nhiệm. Chi tiết xem mục Phân công trong báo cáo.

## Tài liệu tham khảo

- C. M. Bishop. *Pattern Recognition and Machine Learning*. Springer, 2006. Chương 4 (mô hình tuyến tính, softmax).
- T. Hastie, R. Tibshirani, J. Friedman. *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.
- I. Goodfellow, Y. Bengio, A. Courville. *Deep Learning*. MIT Press, 2016.
