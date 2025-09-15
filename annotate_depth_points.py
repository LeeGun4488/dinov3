# annotate_depth_points.py
import argparse, os, sys
import numpy as np
import cv2
from pathlib import Path

# ── GUI 파일 선택을 위해 tkinter 사용 ──
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception as e:
    tk = None  # 헤드리스 환경 대비

def load_depth(path):
    d = np.load(path).astype(np.float32)  # (H,W) meters
    return d

def median_at(depth_m, x, y, k=5):
    h, w = depth_m.shape
    r = k // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    mask = np.isfinite(patch) & (patch > 0)
    if not mask.any():
        return np.nan
    return float(np.median(patch[mask]))

def median_rect(depth_m, x0, y0, x1, y1):
    """사각 패치(폐구간)의 median(m)."""
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(int(x1), depth_m.shape[1]); y1 = min(int(y1), depth_m.shape[0])
    if x1 <= x0 or y1 <= y0:
        return np.nan
    patch = depth_m[y0:y1, x0:x1]
    mask = np.isfinite(patch) & (patch > 0)
    if not mask.any():
        return np.nan
    return float(np.median(patch[mask]))

def put_label(img, xy, text, color=(255,255,255)):
    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, th = 0.6, 2
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    cv2.rectangle(img, (x+8, y-20-tht), (x+8+tw+8, y-8), (0,0,0), -1)
    cv2.putText(img, text, (x+12, y-12), font, scale, color, th, cv2.LINE_AA)

def infer_npy_from_image(img_path: Path) -> Path | None:
    """
    이미지 이름에서 대응되는 .npy를 다음 우선순위로 탐색:
      1) *_metric_vis → *_metric_m.npy, *_metric_map.npy
      2) *_vis        → *_m.npy, *_map.npy, *.npy
      3) 일반 후보    → <stem>_m.npy, <stem>_map.npy, <stem>.npy
    """
    parent = img_path.parent
    stem   = img_path.stem
    cands  = []

    if stem.endswith("_metric_vis"):
        base = stem[: -len("_metric_vis")]
        cands += [parent / f"{base}_metric_m.npy",
                  parent / f"{base}_metric_map.npy"]

    if stem.endswith("_vis"):
        base = stem[:-4]
        cands += [parent / f"{base}_m.npy",
                  parent / f"{base}_map.npy",
                  parent / f"{base}.npy"]

    cands += [
        parent / f"{stem}_m.npy",
        parent / f"{stem}_map.npy",
        parent / f"{stem.replace('_vis','')}_m.npy",
        parent / f"{stem.replace('_vis','')}_map.npy",
        parent / f"{stem.replace('_vis','')}.npy",
        parent / f"{stem}.npy",
    ]
    for c in cands:
        if c.exists():
            return c
    return None

def pick_file_dialog(title="파일 선택", types=None):
    if tk is None:
        return None
    root = tk.Tk()
    root.withdraw()
    root.update()
    if types is None:
        types = [
            ("이미지 파일", "*.png *.jpg *.jpeg *.bmp *.webp *.PNG *.JPG *.JPEG *.BMP *.WEBP"),
            ("모든 파일", "*.*"),
        ]
    path = filedialog.askopenfilename(title=title, filetypes=types)
    root.destroy()
    return path if path else None

def alert(msg: str):
    if tk is not None:
        try:
            root = tk.Tk(); root.withdraw(); messagebox.showinfo("안내", msg); root.destroy()
            return
        except Exception:
            pass
    print(msg)

def auto_grid_points(depth_m, rows=4, cols=4, margin_ratio=0.0):
    """
    (rows x cols) 그리드로 나누고 각 셀 중앙을 기준으로
    셀 전체 영역의 median(m)을 계산해 라벨링 포인트 리스트 반환.
    margin_ratio>0이면 셀 경계에서 margin_ratio만큼 내부로 여유(클린 영역).
    반환: [(x, y, label_str), ...]
    """
    H, W = depth_m.shape
    points = []
    cell_w = W / float(cols)
    cell_h = H / float(rows)
    mx = cell_w * margin_ratio
    my = cell_h * margin_ratio

    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_w + mx
            x1 = (c + 1) * cell_w - mx
            y0 = r * cell_h + my
            y1 = (r + 1) * cell_h - my
            z = median_rect(depth_m, x0, y0, x1, y1)
            cx = int(round((x0 + x1) / 2.0))
            cy = int(round((y0 + y1) / 2.0))
            label = f"{z:.2f} m" if np.isfinite(z) else "NaN"
            points.append((cx, cy, label))
    return points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="배경 이미지(PNG/JPG) (미지정 시 파일 선택창)")
    ap.add_argument("--depth_npy", default=None, help="metric depth .npy (미지정 시 PNG에서 자동 추론, 없으면 파일 선택창)")
    ap.add_argument("--out", default=None, help="저장 파일명(미지정 시 annotated_<원본이름>.png)")
    ap.add_argument("--patch", type=int, default=5, help="클릭 시 median 패치 크기(홀수 권장)")
    # 자동 주석 관련 옵션
    ap.add_argument("--grid", type=int, default=4, help="그리드 크기(N) → N×N 개 자동 점(기본 4→16점)")
    ap.add_argument("--margin", type=float, default=0.0, help="셀 내부 마진 비율(0~0.4 권장). 0.1이면 10%% 안쪽에서 median")
    args = ap.parse_args()

    # 1) 이미지 선택
    img_path = Path(args.image) if args.image else None
    if img_path is None:
        sel = pick_file_dialog("주석을 달 이미지를 선택하세요")
        if not sel:
            print("[ERR] 이미지를 선택하지 않았습니다."); sys.exit(1)
        img_path = Path(sel)
    if not img_path.exists():
        print(f"[ERR] 이미지를 찾을 수 없습니다: {img_path}"); sys.exit(1)

    # 2) depth npy 자동/선택
    if args.depth_npy:
        depth_path = Path(args.depth_npy)
        if not depth_path.exists():
            print(f"[ERR] depth_npy를 찾을 수 없습니다: {depth_path}"); sys.exit(1)
    else:
        depth_path = infer_npy_from_image(img_path)
        if depth_path is None or not depth_path.exists():
            alert("이미지와 같은 이름의 .npy를 찾을 수 없습니다. 직접 선택해 주세요.")
            sel_npy = pick_file_dialog("NPY(depth, meters) 선택", types=[("NumPy 파일","*.npy"), ("모든 파일","*.*")])
            if not sel_npy:
                print("[ERR] NPY를 선택하지 않았습니다."); sys.exit(1)
            depth_path = Path(sel_npy)

    # 3) 출력 경로
    out_path = Path(args.out) if args.out else (img_path.parent / f"annotated_{img_path.stem}.png")

    # 4) 로드
    depth_m = load_depth(str(depth_path))      # (H,W)
    bg = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # (H,W,3) BGR
    if bg is None:
        print(f"[ERR] 이미지를 열 수 없습니다: {img_path}"); sys.exit(1)

    # 크기 정합
    h, w = depth_m.shape
    if (bg.shape[0], bg.shape[1]) != (h, w):
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_NEAREST)

    canvas = bg.copy()
    points = []

    # ── 자동 16점(기본 4x4) 생성 & 그리기 ──
    grid_n = max(1, int(args.grid))
    auto_pts = auto_grid_points(depth_m, rows=grid_n, cols=grid_n, margin_ratio=max(0.0, float(args.margin)))
    points.extend(auto_pts)
    for px, py, lb in points:
        cv2.circle(canvas, (px, py), 5, (0,0,255), 2)
        put_label(canvas, (px, py), lb)

    win = "depth-annotator (Left: add, u: undo, s: save, q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # 클릭 추가용 패치(클릭 시에는 주변 작은 패치 median)
    patch_k = max(3, args.patch | 1)

    def on_mouse(event, x, y, flags, param):
        nonlocal canvas, points
        if event == cv2.EVENT_LBUTTONDOWN:
            z = median_at(depth_m, x, y, k=patch_k)
            label = f"{z:.2f} m" if np.isfinite(z) else "NaN"
            points.append((x, y, label))
            canvas = bg.copy()
            for px, py, lb in points:
                cv2.circle(canvas, (px, py), 5, (0,0,255), 2)
                put_label(canvas, (px, py), lb)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, canvas)
        key = cv2.waitKey(10) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('u') and points:
            points.pop()
            canvas = bg.copy()
            for px, py, lb in points:
                cv2.circle(canvas, (px, py), 5, (0,0,255), 2)
                put_label(canvas, (px, py), lb)
        elif key == ord('s'):
            cv2.imwrite(str(out_path), canvas)
            print(f"Saved: {out_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
