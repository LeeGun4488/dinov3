#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

# 파일 선택 창
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_OK = True
except Exception:
    TK_OK = False

# ─────────────────────────────────────────────────────────────
# 타깃 클래스(0~29), 255=unknown
# ─────────────────────────────────────────────────────────────
TARGET30 = [
    "obstacle","bench","bicycle","boardwalk","bollard","box","bus_stop","bush","car",
    "construction_fence","crosswalk","dirt_road","elevator","escalator","fire_hydrant",
    "ground_transformer","mailbox","motorcycle","person","railing","road","stair",
    "standing_signboard","stone","street_light","traffic_cone","traffic_light","trash_bin",
    "tree","utility_pole",
]
IGNORE_ID = 255

def colorize_mask(mask_hw: np.ndarray, num_classes: int) -> np.ndarray:
    """고정 랜덤 팔레트로 색칠 (0..num_classes-1), 그 외 값은 회색."""
    rng = np.random.default_rng(42)
    palette = (rng.integers(0, 256, size=(num_classes, 3))).astype(np.uint8)
    h, w = mask_hw.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask_hw >= 0) & (mask_hw < num_classes)
    out[valid] = palette[mask_hw[valid]]
    out[~valid] = (200, 200, 200)  # unknown 등
    return out

def overlay(img_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if img_rgb.shape[:2] != mask_rgb.shape[:2]:
        mask_rgb = cv2.resize(mask_rgb, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = cv2.addWeighted(img_rgb.astype(np.float32), 1 - alpha,
                          mask_rgb.astype(np.float32), alpha, 0.0)
    return out.clip(0, 255).astype(np.uint8)

def pick_file_dialog(title: str, filetypes=(("이미지 파일", "*.png *.jpg *.jpeg *.bmp *.webp"), ("모든 파일", "*.*"))):
    if not TK_OK:
        return None
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.update()
    root.destroy()
    return path if path else None

def read_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_label_gray(path: Path) -> np.ndarray:
    """라벨 PNG(0..29, 255)를 8비트 단일 채널로 읽기"""
    lbl = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if lbl is None:
        raise FileNotFoundError(f"라벨을 열 수 없습니다: {path}")
    if lbl.ndim == 3:
        lbl = lbl[:, :, 0]
    lbl = lbl.astype(np.uint8)
    return lbl

def put_text_panel(img_bgr, lines, topleft=(8, 8)):
    """반투명 패널에 텍스트"""
    x, y = topleft
    pad = 6
    line_h = 20
    w = max([cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for s in lines] + [1]) + 2*pad
    h = line_h*len(lines) + 2*pad
    overlay_img = img_bgr.copy()
    cv2.rectangle(overlay_img, (x, y), (x+w, y+h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay_img, 0.35, img_bgr, 0.65, 0)
    yy = y + pad + 14
    for s in lines:
        cv2.putText(img, s, (x+pad, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        yy += line_h
    return img

def majority_label(lbl, x, y, k=1):
    """(x,y) 주변 (2k+1)^2 영역 다수결 (경계 자동 클램프)"""
    h, w = lbl.shape
    x0, x1 = max(0, x-k), min(w-1, x+k)
    y0, y1 = max(0, y-k), min(h-1, y+k)
    patch = lbl[y0:y1+1, x0:x1+1].reshape(-1)
    vals, cnts = np.unique(patch, return_counts=True)
    order = np.argsort(-cnts)
    for idx in order:
        v = vals[idx]
        if v != IGNORE_ID:
            return int(v)
    return int(IGNORE_ID)

def find_label_from_image(img_path: Path) -> Path | None:
    """이미지 경로를 기준으로 *_ids.png(또는 *_ids16.png) 라벨 파일 자동 탐색"""
    cand = [
        img_path.with_name(img_path.stem + "_ids.png"),
        img_path.with_name(img_path.stem + "_ids16.png"),
    ]
    for c in cand:
        if c.exists():
            return c
    return None

def main():
    ap = argparse.ArgumentParser(description="클릭한 픽셀의 세그먼트 라벨 보기 (0..29, 255=unknown)")
    ap.add_argument("--image", type=str, default="", help="표시할 원본 이미지 경로 (선택)")
    ap.add_argument("--label", type=str, default="", help="라벨 PNG 경로(단일채널 0..29, 255)")
    ap.add_argument("--sample-k", type=int, default=1, help="클릭 주변 다수결 반경 k(1이면 3x3)")
    ap.add_argument("--alpha", type=float, default=0.5, help="오버레이 알파값")
    args = ap.parse_args()

    img_path = Path(args.image) if args.image else None
    lbl_path = Path(args.label) if args.label else None

    # 이미지만 주어졌다면, 같은 폴더의 <이미지이름>_ids.png(또는 _ids16.png)를 자동으로 찾음
    if img_path and img_path.exists() and (lbl_path is None or not lbl_path.exists()):
        auto_lbl = find_label_from_image(img_path)
        if auto_lbl is not None:
            lbl_path = auto_lbl

    # 파일 선택창
    if img_path is None or not img_path.exists():
        p = pick_file_dialog("원본 이미지 선택")
        if p:
            img_path = Path(p)
            # 이미지 선택 후 자동 라벨 탐색
            auto_lbl = find_label_from_image(img_path)
            if auto_lbl is not None:
                lbl_path = auto_lbl

    if lbl_path is None or not lbl_path.exists():
        p = pick_file_dialog("라벨 PNG 선택(0..29, 255)")
        if p:
            lbl_path = Path(p)

    if lbl_path is None or not lbl_path.exists():
        print("[ERR] 라벨 PNG를 지정해야 합니다 (--label 또는 *_ids.png 자동탐색/파일 선택).")
        sys.exit(1)

    # 라벨만 있어도 동작 가능(색칠 화면에 클릭)
    lbl = read_label_gray(lbl_path)
    h_lbl, w_lbl = lbl.shape[:2]

    if img_path and img_path.exists():
        img_rgb = read_image_rgb(img_path)
    else:
        img_rgb = colorize_mask(lbl, len(TARGET30))
    h_img, w_img = img_rgb.shape[:2]

    # 보기용 라벨 리사이즈
    if (h_img, w_img) != (h_lbl, w_lbl):
        lbl_for_vis = cv2.resize(lbl, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        use_resized_label = True
    else:
        lbl_for_vis = lbl
        use_resized_label = False

    mask_rgb = colorize_mask(lbl_for_vis.clip(0, len(TARGET30)-1), len(TARGET30))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay_on = True
    help_on = True

    def compose_frame():
        base = overlay(img_rgb, mask_rgb, args.alpha) if overlay_on else img_rgb.copy()
        out = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        if help_on:
            lines = [
                "[조작] 좌클릭: 픽셀 라벨 보기 / h: 도움말 토글 / o: 오버레이 토글 / q or ESC: 종료",
                f"이미지 크기: {w_img}x{h_img}, 라벨 크기: {w_lbl}x{h_lbl}, overlay={overlay_on}"
            ]
            return put_text_panel(out, lines, (8, 8))
        return out

    win = "Seg Click Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    frame = compose_frame()
    cv2.imshow(win, frame)

    def on_mouse(event, x, y, flags, userdata):
        nonlocal frame
        if event == cv2.EVENT_LBUTTONDOWN:
            if use_resized_label:
                x_l = int(round(x * (w_lbl / w_img)))
                y_l = int(round(y * (h_lbl / h_img)))
                x_l = np.clip(x_l, 0, w_lbl-1)
                y_l = np.clip(y_l, 0, h_lbl-1)
            else:
                x_l, y_l = x, y
            lbl_id = majority_label(lbl, x_l, y_l, k=args.sample_k)

            if lbl_id == IGNORE_ID:
                name = "UNKNOWN/IGNORE"
            elif 0 <= lbl_id < len(TARGET30):
                name = TARGET30[lbl_id]
            else:
                name = f"INVALID({lbl_id})"

            vis = compose_frame()
            txt = f"(x={x}, y={y}) → label={lbl_id} [{name}]"
            vis = put_text_panel(vis, [txt], (8, 60))
            cv2.imshow(win, vis)
            frame = vis
            print(txt)

    cv2.setMouseCallback(win, on_mouse)

    try:
        while True:
            # 창이 닫히면 자동 종료
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q'), ord('Q')):  # ESC or q
                break
            elif key in (ord('o'), ord('O')):
                overlay_on = not overlay_on
                frame = compose_frame()
                cv2.imshow(win, frame)
            elif key in (ord('h'), ord('H')):
                help_on = not help_on
                frame = compose_frame()
                cv2.imshow(win, frame)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
