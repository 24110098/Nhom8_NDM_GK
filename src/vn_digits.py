import glob, os
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

# Hình học phiếu (mm), khớp với file Phieu_thu_thap_chu_so.pdf
PX_PER_MM = 10
MARK_CENTERS_MM = [(15, 15), (195, 15), (15, 282), (195, 282)]   # TL, TR, BL, BR
GRID_X, GRID_Y, CELL, N = 40, 85, 15, 10                           # lưới 10x10, ô 15 mm
INSET = 1.5                                                         # bỏ viền ô (mm)
W_PX, H_PX = 210 * PX_PER_MM, 297 * PX_PER_MM

def find_markers(gray):
    """Tìm tâm 4 ô vuông đen ở góc phiếu, trả về theo thứ tự TL, TR, BL, BR."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if not 1e-4 * gray.size < area < 2e-2 * gray.size:   # ô đánh dấu chiếm ~0.05–1% ảnh
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not 0.7 < w / h < 1.4:                         # phải gần vuông
            continue
        if (bw[y:y + h, x:x + w] > 0).mean() < 0.85:      # phải đặc: đo tỷ lệ pixel đen thật sự
            continue                                      # (khung lưới, chữ số đều rỗng)
        M = cv2.moments(c)
        cands.append((area, M["m10"] / M["m00"], M["m01"] / M["m00"]))
    if len(cands) < 4:
        raise ValueError(f"Chỉ tìm thấy {len(cands)} ô đánh dấu, cần 4. Chụp lại cho rõ 4 góc.")
    pts = np.array([(x, y) for _, x, y in sorted(cands, reverse=True)[:4]], np.float32)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return np.array([pts[s.argmin()], pts[d.argmax()], pts[d.argmin()], pts[s.argmax()]], np.float32)

def warp_sheet(img_bgr):
    """Nắn phẳng ảnh chụp phiếu về ảnh chuẩn 2100x2970 px (10 px/mm)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    src = find_markers(gray)
    dst = np.array(MARK_CENTERS_MM, np.float32) * PX_PER_MM
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (W_PX, H_PX), borderValue=255)

def cut_cells(sheet):
    """Cắt 100 ô; nhãn của ô = số thứ tự hàng."""
    cells = []
    for r in range(N):
        for c in range(N):
            x0 = int((GRID_X + c * CELL + INSET) * PX_PER_MM)
            y0 = int((GRID_Y + r * CELL + INSET) * PX_PER_MM)
            size = int((CELL - 2 * INSET) * PX_PER_MM)
            cells.append((r, c, sheet[y0:y0 + size, x0:x0 + size]))
    return cells

def clean_cell(cell, min_contrast=60):
    """Ô xám (nền giấy sáng, mực tối) -> mực trong [0, 1], nền về 0. Ô trống -> None."""
    a = 255.0 - cell.astype(np.float32)                   # mực thành giá trị lớn
    a = a - np.median(a)                                  # trừ nền giấy (đa số pixel là nền)
    peak = np.percentile(a, 99.5)
    if peak < min_contrast:                               # không có nét nào đậm hơn nền rõ rệt
        return None
    a = np.clip(a / peak, 0, 1)
    a[a < 0.25] = 0                                       # bỏ bóng, nhiễu nhẹ
    a[:3, :] = a[-3:, :] = 0; a[:, :3] = a[:, -3:] = 0    # xoá vệt viền ô còn sót
    return a

def to_mnist(a):
    """a: mực trong [0, 1] (nền 0). Trả về (28, 28) kiểu MNIST: khung 20x20, trọng tâm ở giữa."""
    ys, xs = np.nonzero(a > 0.1)
    if len(ys) == 0:
        return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)) / 255.0
    out = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = a
    cy, cx = ndimage.center_of_mass(out)
    return np.clip(ndimage.shift(out, (14 - cy, 14 - cx), order=1), 0, 1)

def process_sheet(path):
    """Một ảnh phiếu -> (X (k, 784), y (k,), vị trí ô (k, 2), số ô trống)."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh {path}")
    sheet = warp_sheet(img)
    X, y, pos, empty = [], [], [], 0
    for r, c, cell in cut_cells(sheet):
        a = clean_cell(cell)
        if a is None:                                     # ô trống
            empty += 1
            continue
        X.append(to_mnist(a).reshape(784)); y.append(r); pos.append((r, c))
    return np.array(X, np.float32), np.array(y), np.array(pos), empty

def build_dataset(folder, out="data/vn_digits.npz"):
    """Thư mục chứa ảnh tên W01.jpg, W02.png, ... (tên file = mã người viết)."""
    Xs, ys, ws, ps = [], [], [], []
    for path in sorted(glob.glob(os.path.join(folder, "*.*"))):
        wid = os.path.splitext(os.path.basename(path))[0]
        try:
            X, y, pos, empty = process_sheet(path)
        except ValueError as e:
            print(f"[BỎ QUA] {wid}: {e}")
            continue
        print(f"{wid}: {len(y)} ảnh, {empty} ô trống")
        Xs.append(X); ys.append(y); ps.append(pos); ws += [wid] * len(y)
    if not Xs:
        raise ValueError("Không xử lý được phiếu nào.")
    np.savez_compressed(out, X=np.concatenate(Xs), y=np.concatenate(ys),
                        writer=np.array(ws), pos=np.concatenate(ps))
    print("Đã lưu", out)

def review_sheet(path, out_png):
    """Vẽ khung cắt lên phiếu đã nắn phẳng để kiểm tra căn chỉnh bằng mắt."""
    sheet = cv2.cvtColor(warp_sheet(cv2.imread(path)), cv2.COLOR_GRAY2BGR)
    for r in range(N):
        for c in range(N):
            x0 = int((GRID_X + c * CELL + INSET) * PX_PER_MM)
            y0 = int((GRID_Y + r * CELL + INSET) * PX_PER_MM)
            size = int((CELL - 2 * INSET) * PX_PER_MM)
            cv2.rectangle(sheet, (x0, y0), (x0 + size, y0 + size), (0, 0, 255), 3)
    cv2.imwrite(out_png, sheet)
