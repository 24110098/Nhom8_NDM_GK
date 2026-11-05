"""
app.py – Demo vẽ chữ số (Gradio). Đặt ở thư mục gốc repo, chạy:
    pip install gradio
    python app.py
Cần có các mô hình đã lưu trong results/models/*.pkl (lưu từ notebook).
"""
import pickle
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))      # để pickle tìm thấy các lớp mô hình trong src/
from vn_digits import clean_cell, to_mnist  # noqa: E402  (dùng CHUNG tiền xử lý với tập Việt)

MODEL_DIR = ROOT / "results" / "models"
NAMES = {"naive_bayes": "Multinomial NB", "logistic": "Logistic Regression",
         "mlp": "MLP", "random_forest": "Random Forest"}
MODELS = {}
for fname, title in NAMES.items():
    path = MODEL_DIR / f"{fname}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            MODELS[title] = pickle.load(f)
if not MODELS:
    sys.exit(f"Chưa có mô hình nào trong {MODEL_DIR}. Lưu mô hình từ notebook trước.")


def sketch_to_gray(sketch):
    """Ảnh từ gr.Sketchpad -> mảng xám (H, W), nền trắng, nét tối.
    Gradio 4/5 trả về dict, ảnh nằm ở khóa "composite" (thường RGBA nền trong suốt)."""
    img = sketch["composite"] if isinstance(sketch, dict) else sketch
    if img is None:
        return None
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] == 4:                    # ghép lên nền trắng theo kênh alpha
        alpha = a[..., 3:] / 255.0
        a = a[..., :3] * alpha + 255.0 * (1 - alpha)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return a


def canvas_to_mnist(sketch, min_contrast=40):
    """Vector (784,) trong [0, 1] kiểu MNIST, hoặc None nếu chưa vẽ gì."""
    gray = sketch_to_gray(sketch)
    if gray is None:
        return None
    a = clean_cell(gray, min_contrast=min_contrast)
    return None if a is None else to_mnist(a).reshape(784)


def predict(sketch):
    x = canvas_to_mnist(sketch)
    if x is None:
        return [{} for _ in MODELS] + [None, "Hãy vẽ một chữ số."]
    outs, notes, votes = [], [], []
    for title, m in MODELS.items():
        t = time.perf_counter()
        p = m.predict_proba(x[None, :])[0]
        notes.append(f"{title}: {(time.perf_counter() - t) * 1000:.1f} ms")
        outs.append({str(k): float(p[k]) for k in range(10)})
        votes.append(int(p.argmax()))
    agree = "Các mô hình **đồng thuận**" if len(set(votes)) == 1 else "Các mô hình **bất đồng**"
    preview = (x.reshape(28, 28) * 255).astype(np.uint8)   # ảnh 28x28 mà mô hình thật sự nhận
    return outs + [preview, agree + "  |  " + "  ·  ".join(notes)]


with gr.Blocks(title="DigitLab") as demo:
    gr.Markdown("# DigitLab – vẽ một chữ số, xem các mô hình đoán")
    with gr.Row():
        pad = gr.Sketchpad(label="Vẽ ở đây")
        preview = gr.Image(label="Ảnh 28×28 sau tiền xử lý", image_mode="L", height=200)
    with gr.Row():
        labels = [gr.Label(num_top_classes=3, label=title) for title in MODELS]
    info = gr.Markdown()
    pad.change(predict, inputs=pad, outputs=labels + [preview, info])

if __name__ == "__main__":
    demo.launch()
