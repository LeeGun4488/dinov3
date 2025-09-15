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
# ADE 라벨 텍스트 (세미콜론으로 동의어 연결, 줄 = 클래스 ID)
# ─────────────────────────────────────────────────────────────
ADE_NAMES_TXT = """wall
building;edifice
sky
floor;flooring
tree
ceiling
road;route
bed
windowpane;window
grass
cabinet
sidewalk;pavement
person;individual;someone;somebody;mortal;soul
earth;ground
door;double;door
table
mountain;mount
plant;flora;plant;life
curtain;drape;drapery;mantle;pall
chair
car;auto;automobile;machine;motorcar
water
painting;picture
sofa;couch;lounge
shelf
house
sea
mirror
rug;carpet;carpeting
field
armchair
seat
fence;fencing
desk
rock;stone
wardrobe;closet;press
lamp
bathtub;bathing;tub;bath;tub
railing;rail
cushion
base;pedestal;stand
box
column;pillar
signboard;sign
chest;of;drawers;chest;bureau;dresser
counter
sand
sink
skyscraper
fireplace;hearth;open;fireplace
refrigerator;icebox
grandstand;covered;stand
path
stairs;steps
runway
case;display;case;showcase;vitrine
pool;table;billiard;table;snooker;table
pillow
screen;door;screen
stairway;staircase
river
bridge;span
bookcase
blind;screen
coffee;table;cocktail;table
toilet;can;commode;crapper;pot;potty;stool;throne
flower
book
hill
bench
countertop
stove;kitchen;stove;range;kitchen;range;cooking;stove
palm;palm;tree
kitchen;island
computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system
swivel;chair
boat
bar
arcade;machine
hovel;hut;hutch;shack;shanty
bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle
towel
light;light;source
truck;motortruck
tower
chandelier;pendant;pendent
awning;sunshade;sunblind
streetlight;street;lamp
booth;cubicle;stall;kiosk
television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box
airplane;aeroplane;plane
dirt;track
apparel;wearing;apparel;dress;clothes
pole
land;ground;soil
bannister;banister;balustrade;balusters;handrail
escalator;moving;staircase;moving;stairway
ottoman;pouf;pouffe;puff;hassock
bottle
buffet;counter;sideboard
poster;posting;placard;notice;bill;card
stage
van
ship
fountain
conveyer;belt;conveyor;belt;conveyer;conveyor;transporter
canopy
washer;automatic;washer;washing;machine
plaything;toy
swimming;pool;swimming;bath;natatorium
stool
barrel;cask
basket;handbasket
waterfall;falls
tent;collapsible;shelter
bag
minibike;motorbike
cradle
oven
ball
food;solid;food
step;stair
tank;storage;tank
trade;name;brand;name;brand;marque
microwave;microwave;oven
pot;flowerpot
animal;animate;being;beast;brute;creature;fauna
bicycle;bike;wheel;cycle
lake
dishwasher;dish;washer;dishwashing;machine
screen;silver;screen;projection;screen
blanket;cover
sculpture
hood;exhaust;hood
sconce
vase
traffic;light;traffic;signal;stoplight
tray
ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin
fan
pier;wharf;wharfage;dock
crt;screen
plate
monitor;monitoring;device
bulletin;board;notice;board
shower
radiator
glass;drinking;glass
clock
flag
"""

# 255=unknown 그대로 유지
IGNORE_ID = 255

def parse_ade_names(txt: str):
    names_full = [ln.strip() for ln in txt.strip().splitlines() if ln.strip()]
    names_short = [ln.split(";")[0].strip() for ln in names_full]
    return names_full, names_short

ADE_NAMES_FULL, ADE_NAMES = parse_ade_names(ADE_NAMES_TXT)  # FULL=동의어 포함, ADE_NAMES=대표명(세미콜론 앞)

NUM_CLASSES = len(ADE_NAMES)

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────
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
    """라벨 PNG(0..NUM_CLASSES-1, 255)를 8비트 단일 채널로 읽기"""
    lbl = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if lbl is None:
        raise FileNotFoundError(f"라벨을 열 수 없습니다: {path}")
    if lbl.ndim == 3:
        # 팔레트 PNG라도 인덱스가 첫 채널에 들어있을 수 있음(데이터에 따라 상이)
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
    # ignore 제외 다수결 → 전부 255면 255
    vals, cnts = np.unique(patch, return_counts=True)
    order = np.argsort(-cnts)
    for idx in order:
        v = vals[idx]
        if v != IGNORE_ID:
            return int(v)
    return int(IGNORE_ID)

def main():
    ap = argparse.ArgumentParser(description=f"클릭한 픽셀의 세그먼트 라벨 보기 (0..{NUM_CLASSES-1}, 255=unknown)")
    ap.add_argument("--image", type=str, default="", help="표시할 원본 이미지 경로 (선택)")
    ap.add_argument("--label", type=str, default="", help=f"라벨 PNG 경로(단일채널 0..{NUM_CLASSES-1}, 255)")
    ap.add_argument("--sample-k", type=int, default=1, help="클릭 주변 다수결 반경 k(1이면 3x3)")
    ap.add_argument("--alpha", type=float, default=0.5, help="오버레이 알파값")
    args = ap.parse_args()

    img_path = Path(args.image) if args.image else None
    lbl_path = Path(args.label) if args.label else None

    # 파일 선택창
    if img_path is None or not img_path.exists():
        p = pick_file_dialog("원본 이미지 선택")
        if p:
            img_path = Path(p)

    if lbl_path is None or not lbl_path.exists():
        p = pick_file_dialog(f"라벨 PNG 선택(0..{NUM_CLASSES-1}, 255)")
        if p:
            lbl_path = Path(p)

    if lbl_path is None or not lbl_path.exists():
        print("[ERR] 라벨 PNG를 지정해야 합니다 (--label 또는 파일 선택).")
        sys.exit(1)

    # 라벨만 있어도 동작 가능(색칠 화면에 클릭)
    lbl = read_label_gray(lbl_path)
    h_lbl, w_lbl = lbl.shape[:2]

    if img_path and img_path.exists():
        img_rgb = read_image_rgb(img_path)
    else:
        # 원본 이미지가 없으면, 라벨을 색칠해서 베이스 이미지로 사용
        img_rgb = colorize_mask(lbl, NUM_CLASSES)
    h_img, w_img = img_rgb.shape[:2]

    # 라벨/이미지 해상도가 다르면 시각화용 라벨만 리사이즈
    if (h_img, w_img) != (h_lbl, w_lbl):
        lbl_for_vis = cv2.resize(lbl, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        use_resized_label = True
    else:
        lbl_for_vis = lbl
        use_resized_label = False

    # 컬러 마스크 & 오버레이 초기화 (유효 범위밖은 회색 처리되도록 colorize_mask가 처리)
    mask_rgb = colorize_mask(lbl_for_vis, NUM_CLASSES)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay_on = True
    help_on = True

    def compose_frame():
        base_rgb = overlay(img_rgb, mask_rgb, args.alpha) if overlay_on else img_rgb.copy()
        out = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
        if help_on:
            lines = [
                "[조작] 좌클릭: 픽셀 라벨 보기 / h: 도움말 토글 / o: 오버레이 토글 / q or ESC: 종료",
                f"이미지 크기: {w_img}x{h_img}, 라벨 크기: {w_lbl}x{h_lbl}, classes={NUM_CLASSES}, overlay={overlay_on}"
            ]
            return put_text_panel(out, lines, (8, 8))
        return out

    win = "Seg Click Viewer (ADE)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    frame = compose_frame()
    cv2.imshow(win, frame)

    def on_mouse(event, x, y, flags, userdata):
        nonlocal frame
        if event == cv2.EVENT_LBUTTONDOWN:
            # (x,y)는 이미지 좌표. 라벨 좌표로 매핑
            if use_resized_label:
                x_l = int(round(x * (w_lbl / w_img)))
                y_l = int(round(y * (h_lbl / h_img)))
                x_l = np.clip(x_l, 0, w_lbl-1)
                y_l = np.clip(y_l, 0, h_lbl-1)
            else:
                x_l, y_l = x, y

            # 다수결 샘플
            lbl_id = majority_label(lbl, x_l, y_l, k=args.sample_k)

            if lbl_id == IGNORE_ID:
                name = "UNKNOWN/IGNORE"
            elif 0 <= lbl_id < NUM_CLASSES:
                # 대표명 사용(세미콜론 앞), 필요 시 ADE_NAMES_FULL[lbl_id]로 전체 동의어 출력 가능
                name = ADE_NAMES[lbl_id]
            else:
                name = f"INVALID({lbl_id})"

            # 정보 패널 업데이트
            vis = compose_frame()
            txt = f"(x={x}, y={y}) → label={lbl_id} [{name}]"
            vis = put_text_panel(vis, [txt], (8, 60))
            cv2.imshow(win, vis)
            frame = vis
            print(txt)

    cv2.setMouseCallback(win, on_mouse)

    while True:
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

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
