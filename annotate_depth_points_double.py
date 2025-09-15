# annotate_depth_points_double.py
import argparse, os, sys
import numpy as np
import cv2
from pathlib import Path

# ── GUI 파일 선택을 위해 tkinter 사용 ──
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None  # 헤드리스 환경 대비

def load_depth(path: str) -> np.ndarray:
    d = np.load(path).astype(np.float32)  # (H, W) meters
    return d

def median_at(depth_m: np.ndarray, x: int, y: int, k: int = 5) -> float:
    h, w = depth_m.shape
    r = k // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    mask = np.isfinite(patch) & (patch > 0)
    if not mask.any():
        return np.nan
    return float(np.median(patch[mask]))

def median_rect(depth_m: np.ndarray, x0, y0, x1, y1) -> float:
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
        parent / f"{stem}_depth_gt.npy",
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

def auto_grid_points(depth_m: np.ndarray, rows=4, cols=4, margin_ratio=0.0):
    """
    (rows x cols) 그리드로 나누고 각 셀 중앙 위치에
    셀 영역의 median(m)을 라벨로 붙인 포인트들을 반환.
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

def compose_side_by_side(bgL: np.ndarray, bgR: np.ndarray):
    """
    좌우 이미지를 세로로 맞춰 하나의 캔버스로 합성.
    상단 정렬, 폭은 Wl+Wr, 높이는 max(Hl,Hr).
    """
    hL, wL = bgL.shape[:2]
    hR, wR = bgR.shape[:2]
    H = max(hL, hR)
    W = wL + wR
    canvas = np.zeros((H, W, 3), dtype=bgL.dtype)
    # 상단 정렬(위쪽부터 쌓기)
    canvas[:hL, :wL] = bgL
    canvas[:hR, wL:wL+wR] = bgR
    return canvas, (hL, wL), (hR, wR)

def main():
    ap = argparse.ArgumentParser()
    # 좌/우 이미지 & NPY 경로(미지정 시 파일 선택창)
    ap.add_argument("--left-image", default=None, help="왼쪽 이미지 파일(PNG/JPG)")
    ap.add_argument("--left-depth-npy", default=None, help="왼쪽 depth .npy (미지정 시 자동 추론)")
    ap.add_argument("--right-image", default=None, help="오른쪽 이미지 파일(PNG/JPG)")
    ap.add_argument("--right-depth-npy", default=None, help="오른쪽 depth .npy (미지정 시 자동 추론)")
    ap.add_argument("--out", default=None, help="저장 파일명(미지정 시 annotated_<L>__<R>.png)")
    # 자동 주석 옵션
    ap.add_argument("--grid", type=int, default=4, help="그리드 크기 N → 각 이미지에 N×N 점(기본 4)")
    ap.add_argument("--margin", type=float, default=0.0, help="셀 내부 마진 비율(0~0.4 권장)")
    # 클릭 시 median 패치 크기
    ap.add_argument("--patch", type=int, default=5, help="클릭 시 median 패치(홀수 권장)")
    args = ap.parse_args()

    # 1) 좌/우 이미지 선택
    L_path = Path(args.left_image) if args.left_image else None
    if L_path is None:
        sel = pick_file_dialog("왼쪽 이미지를 선택하세요")
        if not sel: print("[ERR] 왼쪽 이미지를 선택하지 않았습니다."); sys.exit(1)
        L_path = Path(sel)
    if not L_path.exists():
        print(f"[ERR] 왼쪽 이미지를 찾을 수 없습니다: {L_path}"); sys.exit(1)

    R_path = Path(args.right_image) if args.right_image else None
    if R_path is None:
        sel = pick_file_dialog("오른쪽 이미지를 선택하세요")
        if not sel: print("[ERR] 오른쪽 이미지를 선택하지 않았습니다."); sys.exit(1)
        R_path = Path(sel)
    if not R_path.exists():
        print(f"[ERR] 오른쪽 이미지를 찾을 수 없습니다: {R_path}"); sys.exit(1)

    # 2) 좌/우 depth npy 자동/선택
    if args.left_depth_npy:
        L_npy = Path(args.left_depth_npy)
        if not L_npy.exists(): print(f"[ERR] 왼쪽 NPY 없음: {L_npy}"); sys.exit(1)
    else:
        L_npy = infer_npy_from_image(L_path)
        if L_npy is None or not L_npy.exists():
            alert("왼쪽 이미지에 대응하는 .npy를 찾을 수 없습니다. 선택해 주세요.")
            sel = pick_file_dialog("왼쪽 NPY 선택", types=[("NumPy 파일","*.npy"), ("모든 파일","*.*")])
            if not sel: print("[ERR] 왼쪽 NPY를 선택하지 않았습니다."); sys.exit(1)
            L_npy = Path(sel)

    if args.right_depth_npy:
        R_npy = Path(args.right_depth_npy)
        if not R_npy.exists(): print(f"[ERR] 오른쪽 NPY 없음: {R_npy}"); sys.exit(1)
    else:
        R_npy = infer_npy_from_image(R_path)
        if R_npy is None or not R_npy.exists():
            alert("오른쪽 이미지에 대응하는 .npy를 찾을 수 없습니다. 선택해 주세요.")
            sel = pick_file_dialog("오른쪽 NPY 선택", types=[("NumPy 파일","*.npy"), ("모든 파일","*.*")])
            if not sel: print("[ERR] 오른쪽 NPY를 선택하지 않았습니다."); sys.exit(1)
            R_npy = Path(sel)

    # 3) 출력 경로
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = L_path.parent / f"annotated_{L_path.stem}__{R_path.stem}.png"

    # 4) 로드 & 크기 정합(각자)
    dL = load_depth(str(L_npy))
    dR = load_depth(str(R_npy))
    bgL = cv2.imread(str(L_path), cv2.IMREAD_COLOR)
    bgR = cv2.imread(str(R_path), cv2.IMREAD_COLOR)
    if bgL is None: print(f"[ERR] 왼쪽 이미지를 열 수 없습니다: {L_path}"); sys.exit(1)
    if bgR is None: print(f"[ERR] 오른쪽 이미지를 열 수 없습니다: {R_path}"); sys.exit(1)

    hL, wL = dL.shape
    hR, wR = dR.shape
    if (bgL.shape[0], bgL.shape[1]) != (hL, wL):
        bgL = cv2.resize(bgL, (wL, hL), interpolation=cv2.INTER_NEAREST)
    if (bgR.shape[0], bgR.shape[1]) != (hR, wR):
        bgR = cv2.resize(bgR, (wR, hR), interpolation=cv2.INTER_NEAREST)

    # 5) 합성 캔버스 만들기
    base_canvas, (hL, wL), (hR, wR) = compose_side_by_side(bgL, bgR)
    canvas = base_canvas.copy()
    x_off_R = wL  # 오른쪽 패널의 x 오프셋

    # 6) 자동 그리드 포인트 생성 (각 이미지별)
    grid_n = max(1, int(args.grid))
    margin_ratio = max(0.0, float(args.margin))
    autoL = auto_grid_points(dL, rows=grid_n, cols=grid_n, margin_ratio=margin_ratio)
    autoR = auto_grid_points(dR, rows=grid_n, cols=grid_n, margin_ratio=margin_ratio)

    # points: (side, x_local, y_local, label)
    points = []
    for x, y, lb in autoL:
        points.append(("L", x, y, lb))
    for x, y, lb in autoR:
        points.append(("R", x, y, lb))

    # 초기 렌더
    for side, x, y, lb in points:
        if side == "L":
            cv2.circle(canvas, (x, y), 5, (0,0,255), 2)
            put_label(canvas, (x, y), lb)
        else:
            xx = x + x_off_R
            cv2.circle(canvas, (xx, y), 5, (0,0,255), 2)
            put_label(canvas, (xx, y), lb)

    # 7) 인터랙션
    win = "depth-annotator (Left: add, u: undo, s: save, q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    patch_k = max(3, args.patch | 1)

    def redraw():
        nonlocal canvas
        canvas = base_canvas.copy()
        for side, x, y, lb in points:
            if side == "L":
                cv2.circle(canvas, (x, y), 5, (0,0,255), 2)
                put_label(canvas, (x, y), lb)
            else:
                xx = x + x_off_R
                cv2.circle(canvas, (xx, y), 5, (0,0,255), 2)
                put_label(canvas, (xx, y), lb)

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            # 좌/우 판별
            if x < wL and y < hL:
                # Left
                z = median_at(dL, x, y, k=patch_k)
                lb = f"{z:.2f} m" if np.isfinite(z) else "NaN"
                points.append(("L", x, y, lb))
            elif x_off_R <= x < x_off_R + wR and y < hR:
                # Right
                xr = x - x_off_R
                z = median_at(dR, xr, y, k=patch_k)
                lb = f"{z:.2f} m" if np.isfinite(z) else "NaN"
                points.append(("R", xr, y, lb))
            else:
                # 빈 영역(패딩)에 클릭
                return
            redraw()

    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, canvas)
        key = cv2.waitKey(10) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('u') and points:
            points.pop()
            redraw()
        elif key == ord('s'):
            cv2.imwrite(str(out_path), canvas)
            print(f"[INFO] Saved: {out_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
