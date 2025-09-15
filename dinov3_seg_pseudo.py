#!/usr/bin/env python3
import os, io, argparse, requests, sys, json
from functools import partial
from pathlib import Path
import re

import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import torch.distributed as dist

def get_dist_env():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank

def maybe_init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

def shard_list(items, world_size, rank):
    # rank가 world_size로 나눈 슬라이싱을 담당
    return items[rank::world_size]

# ──────────────────────────────────────────────────────────────────────────────
# Configs: TARGET 30 클래스(이름 고정) + ADE20K 이름 목록(네가 준 순서, idx는 1부터)
# ──────────────────────────────────────────────────────────────────────────────
TARGET30 = [
    "obstacle","bench","bicycle","boardwalk","bollard","box","bus_stop","bush","car",
    "construction_fence","crosswalk","dirt_road","elevator","escalator","fire_hydrant",
    "ground_transformer","mailbox","motorcycle","person","railing","road","stair",
    "standing_signboard","stone","street_light","traffic_cone","traffic_light","trash_bin",
    "tree","utility_pole",
]
TIDX = {n:i for i,n in enumerate(TARGET30)}

# ADE20K 이름 원문(세미콜론 동의어 포함). "Name" 헤더 제외, 1-based → 0-based로 사용
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
""".strip().splitlines()

def _prep_ade_names(ade_names_txt=None):
    if ade_names_txt is None:
        ade_names_txt = ADE_NAMES_TXT
    # 정리: 소문자/공백 정규화
    return [re.sub(r"\s+", " ", line.strip().lower()) for line in ade_names_txt]

# 타깃 → ADE 키워드(부분문자열) 집합 (없으면 빈 집합 → 비감독)
TARGET2ADE_KEYS = {
    # exact/proxy
    "bench": {"bench"},
    "bicycle": {"bicycle","bike","wheel","cycle"},
    "box": {"box"},
    "car": {"car","auto","automobile","motorcar"},
    "elevator": {"elevator"},
    "escalator": {"escalator","moving staircase","moving stairway"},
    "fire_hydrant": {"fire hydrant"},
    "mailbox": {"mailbox"},
    "person": {"person","individual","someone","somebody","mortal","soul"},
    "railing": {"railing","rail","bannister","banister","balustrade","balusters","handrail"},
    "road": {"road","route"},
    "stair": {"stairs","steps","stairway","staircase","step"},
    "stone": {"rock","stone"},
    "street_light": {"streetlight","street lamp"},
    "traffic_light": {"traffic light","traffic signal","stoplight"},
    "trash_bin": {"ashcan","trash can","garbage can","wastebin","ash bin","ash-bin","ashbin","dustbin","trash barrel","trash bin"},
    "tree": {"tree","palm tree"},

    # proxy (주의: 보수적 threshold 권장)
    "boardwalk": {"sidewalk","pavement","path"},
    "bush": {"plant","flora","plant life","grass"},
    "construction_fence": {"fence","fencing"},
    "crosswalk": {"path","sidewalk","pavement"},  # 대리
    "dirt_road": {"dirt track","land","ground","soil"},
    "bus_stop": {"booth","stall","kiosk","awning","sunshade","sunblind","canopy","grandstand","covered stand"},
    "standing_signboard": {"signboard","sign","poster","placard","notice","bill","card","bulletin board","notice board"},
    "bollard": {"pole"},
    "motorcycle": {"minibike","motorbike"},
    "ground_transformer": {"case", "display case", "showcase", "vitrine"},
    "obstacle": {
        # 가로환경에서 길을 막을 수 있는 구조물/사물 (기존 키와 중복 낮춤)
        "column", "pillar",                    # column;pillar
        "base", "pedestal", "stand",           # base;pedestal;stand
        "barrel", "cask",                      # barrel;cask
        "basket", "bag",                       # basket;handbasket / bag
        "stage",                                # stage
        "fountain",                             # fountain
        "sculpture",                            # sculpture
        "vase",                                 # vase
    },

    # ADE에 소스가 사실상 없는 항목들(비워둠 → pseudo ignore)
    "utility_pole": set(),
    "traffic_cone": set(),
}

# 클래스별 threshold (픽셀을 라벨링할 최소 확률) — 필요시 CLI로 조정
DEFAULT_TAU = {t:0.68 for t in TARGET30}
DEFAULT_TAU.update({
    "traffic_cone":0.85, "fire_hydrant":0.80, "trash_bin":0.78, "bollard":0.85,
    "mailbox":0.78, "street_light":0.74, "standing_signboard":0.74,
    "bench":0.74, "utility_pole":0.72, "bush":0.70, "tree":0.70, "person":0.72,
    "motorcycle":0.72, "bus_stop":0.80, "crosswalk":0.86, "dirt_road":0.76,
})


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────
def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def load_image_from_path(p: str | Path) -> Image.Image:
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(f"[ERR] image file not found: {p}")
    return Image.open(p).convert("RGB")

def make_transform(resize_size: int | None = 224):
    """
    resize_size > 0 : (resize_size, resize_size)로 리사이즈
    resize_size <=0 또는 None : 리사이즈 없이 원본 크기 유지
    """
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    if resize_size is None or (isinstance(resize_size, int) and resize_size <= 0):
        # ✅ 원본 유지 (리사이즈 없음)
        return transforms.Compose([to_tensor, normalize])
    else:
        resize = transforms.Resize((resize_size, resize_size), antialias=True)
        return transforms.Compose([resize, to_tensor, normalize])

def colorize_mask(mask_hw: np.ndarray, num_classes: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    palette = (rng.integers(0, 256, size=(num_classes, 3))).astype(np.uint8)
    return palette[mask_hw % num_classes]

def overlay(image_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if image_rgb.shape[:2] != mask_rgb.shape[:2]:
        mask_rgb = cv2.resize(mask_rgb, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    if mask_rgb.ndim == 2:
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_GRAY2RGB)
    out = cv2.addWeighted(image_rgb.astype(np.float32), 1 - alpha, mask_rgb.astype(np.float32), alpha, 0.0)
    return out.clip(0, 255).astype(np.uint8)

def walk_images(input_dir: Path, exts={".jpg",".jpeg",".png",".bmp",".webp"}):
    for p in input_dir.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            yield p


# ──────────────────────────────────────────────────────────────────────────────
# ADE → TARGET30 그룹핑 (부분문자열 매칭)
# ──────────────────────────────────────────────────────────────────────────────
def build_groups_from_names(ade_names_lines, target2keys):
    src = _prep_ade_names(ade_names_lines)  # lower/space normalized lines
    groups = [[] for _ in range(len(TARGET30))]
    for tname, keys in target2keys.items():
        ti = TIDX[tname]
        if not keys:
            continue
        for j, line in enumerate(src):
            for key in keys:
                k = re.sub(r"\s+", " ", key.strip().lower())
                if k and k in line:
                    groups[ti].append(j)  # 0-based ADE index
                    break
    return groups  # list(len=30) of list(ADE idx)

def ade_logits_to_target30_logits(logits_ade: torch.Tensor, groups):
    """
    logits_ade: [B,150,H,W] (LOGITS)  # 중요: softmax 적용 전
    groups    : list of 30 lists of ADE indices(0-based)
    returns   : [B,30,H,W] logits via logsumexp over grouped channels
    """
    B,K,H,W = logits_ade.shape
    outs = []
    neglarge = torch.finfo(logits_ade.dtype).min/4  # very negative
    for idxs in groups:
        if len(idxs) == 0:
            outs.append(torch.full((B,1,H,W), neglarge, device=logits_ade.device, dtype=logits_ade.dtype))
        else:
            outs.append(torch.logsumexp(logits_ade[:, idxs, :, :], dim=1, keepdim=True))
    return torch.cat(outs, dim=1)

def refine_small_blobs(lbl: np.ndarray, min_pixels=32):
    """아주 작은 블롭 제거 (클래스별로 세밀화하고 싶으면 개선 가능)"""
    if min_pixels <= 0:
        return lbl
    H,W = lbl.shape
    out = lbl.copy()
    for c in range(len(TARGET30)):
        m = (out == c).astype(np.uint8)
        if m.sum() == 0:
            continue
        num, comps, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < min_pixels:
                out[comps == i] = 255  # ignore
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    # 입력 폴더 (재귀)
    ap.add_argument("--input-dir", required=True, type=str, help="이미지 루트 폴더(재귀 탐색)")
    ap.add_argument("--image-exts", type=str, default=".jpg,.jpeg,.png,.bmp,.webp")

    # 저장 경로들
    ap.add_argument("--out-label-dir", required=True, type=str, help="PNG pseudo-label(0~29, 255) 저장 루트")
    ap.add_argument("--out-conf-dir", type=str, default="", help="픽셀별 confidence(.npy) 저장 루트(옵션)")
    ap.add_argument("--out-overlay-dir", type=str, default="", help="시각화 저장 루트(옵션)")

    # 모델/리포지토리
    ap.add_argument("--repo-dir", default="./dinov3", type=str,
                    help="클론한 facebookresearch/dinov3 경로 (torch.hub source='local')")
    ap.add_argument("--weights-dir", default="./weights", type=str)
    ap.add_argument("--backbone-ckpt", default="dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth", type=str)
    ap.add_argument("--seg-ckpt", default="dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth", type=str)
    ap.add_argument("--hub-entry", default="dinov3_vit7b16_ms", type=str)

    # 해상도/인퍼런스 모드
    ap.add_argument("--img-size", default=896, type=int,
                    help=">0이면 정사각 리사이즈, <=0이면 원본 해상도 그대로")
    ap.add_argument("--inference-mode", default="slide", choices=["slide", "whole"],
                    help="whole: 한 번에(메모리 충분), slide: 슬라이딩 윈도우")
    ap.add_argument("--crop-size", default=896, type=int, help="슬라이딩 타일 크기")
    ap.add_argument("--stride", default=896, type=int, help="슬라이딩 스트라이드")

    # 라벨링 정책
    ap.add_argument("--filter-by-ade-top1", action="store_true",
                    help="ADE Top-1이 TARGET30로 직접 매핑될 때만 라벨, 아니면 255(ignore)")
    ap.add_argument("--ignore-index", type=int, default=255)
    ap.add_argument("--min-blob", type=int, default=32, help="min connected pixels to keep (<=0 끄기)")
    ap.add_argument("--tau-global", type=float, default=-1.0, help="(filter-by-ade-top1 미사용 시) 전체 클래스 공통 임계값")

    # 기타
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--save-idx-map", action="store_true", help="클래스 인덱스 매핑 JSON 저장")

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    assert input_dir.is_dir(), f"[ERR] input-dir not found: {input_dir}"

    # 출력 폴더 준비
    out_label_root = Path(args.out_label_dir); out_label_root.mkdir(parents=True, exist_ok=True)
    out_conf_root = Path(args.out_conf_dir) if args.out_conf_dir else None
    out_overlay_root = Path(args.out_overlay_dir) if args.out_overlay_dir else None
    if out_conf_root: out_conf_root.mkdir(parents=True, exist_ok=True)
    if out_overlay_root: out_overlay_root.mkdir(parents=True, exist_ok=True)

    # dinov3 import
    repo_dir = Path(args.repo_dir)
    assert repo_dir.exists(), f"[ERR] repo-dir not found: {repo_dir}"
    sys.path.append(str(repo_dir.resolve()))
    try:
        from dinov3.eval.segmentation.inference import make_inference
    except Exception as e:
        raise RuntimeError(
            f"[ERR] cannot import dinov3 inference utils from {repo_dir}. "
            f"'pip install -e {repo_dir}' 혹은 PYTHONPATH 설정을 확인하세요."
        ) from e

    # 체크포인트
    backbone_ckpt_path = str(Path(args.weights_dir) / args.backbone_ckpt)
    seg_ckpt_path = str(Path(args.weights_dir) / args.seg_ckpt)
    if not os.path.isfile(backbone_ckpt_path): raise FileNotFoundError(f"[ERR] backbone ckpt not found: {backbone_ckpt_path}")
    if not os.path.isfile(seg_ckpt_path):      raise FileNotFoundError(f"[ERR] segmentor ckpt not found: {seg_ckpt_path}")

    # ───────────────────── 분산 초기화 & 디바이스 선택 ─────────────────────
    maybe_init_distributed()
    WORLD_SIZE, RANK, LOCAL_RANK = get_dist_env()

    if torch.cuda.is_available():
        device = f"cuda:{LOCAL_RANK}"
        torch.cuda.set_device(LOCAL_RANK)
    else:
        device = "cpu"
    to_device = device  # args.device 대신 사용

    # 모델 로드
    if RANK == 0:
        print(f"[INFO] Loading hub entry '{args.hub_entry}' ...")
    segmentor = torch.hub.load(
        str(repo_dir),
        args.hub_entry,
        source="local",
        weights=seg_ckpt_path,
        backbone_weights=backbone_ckpt_path,
    ).to(to_device).eval()

    # 변환 (img-size<=0이면 원본 유지)
    tfm = make_transform(args.img_size if args.img_size > 0 else None)

    # 그룹핑 빌드(ADE→TARGET30) 및 매핑 테이블
    ade_names_lines = ADE_NAMES_TXT
    groups = build_groups_from_names(ade_names_lines, TARGET2ADE_KEYS)
    if RANK == 0:
        for ti, idxs in enumerate(groups):
            print(f"[MAP] {TARGET30[ti]:<20s} ← {len(idxs):3d} ADE chans")

    # ADE(0..149) → TARGET30(0..29) 직접 매핑(-1 = 매핑없음)
    ade2t = np.full(150, -1, dtype=np.int32)
    for t_idx, idxs in enumerate(groups):
        for j in idxs:
            ade2t[j] = t_idx  # 중복 시 마지막 wins (필요하면 우선순위 정책으로 조정)

    # 클래스별 임계값 (filter-by-ade-top1 안 쓰는 경우)
    if args.tau_global > 0:
        tau = np.array([args.tau_global]*len(TARGET30), dtype=np.float32)
    else:
        tau = np.array([DEFAULT_TAU[t] for t in TARGET30], dtype=np.float32)

    # 인덱스 맵 저장(옵션, rank 0만)
    if args.save_idx_map and RANK == 0:
        (out_label_root / "_target30.json").write_text(json.dumps({"index_to_class": TARGET30}, indent=2), encoding="utf-8")
        aux = {"ade2target": {int(k): int(v) for k, v in enumerate(ade2t)}}
        (out_label_root / "_map_aux.json").write_text(json.dumps(aux, indent=2), encoding="utf-8")

    # ──────────────── 전체 이미지 리스트 작성 & 랭크별 샤딩 ────────────────
    exts = set([e.strip().lower() for e in args.image_exts.split(",") if e.strip()])
    all_paths = list(walk_images(input_dir, exts=exts))
    my_paths  = shard_list(all_paths, WORLD_SIZE, RANK)

    if RANK == 0:
        print(f"[DIST] WORLD_SIZE={WORLD_SIZE}  total_images={len(all_paths)}")
    print(f"[RANK {RANK}] shard_size={len(my_paths)}")

    # 추론 루프
    IGN = int(args.ignore_index)
    cnt = 0
    for img_path in my_paths:
        rel = img_path.relative_to(input_dir)
        stem = rel.stem
        subdir = rel.parent

        # 출력 경로(구조 보존)
        out_lbl_path = out_label_root / subdir / f"{stem}_seg.png"
        out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
        out_conf_path = out_conf_root / subdir / f"{stem}_conf.npy" if out_conf_root else None
        out_vis_path  = out_overlay_root / subdir / f"{stem}_vis.png" if out_overlay_root else None
        if out_conf_path: out_conf_path.parent.mkdir(parents=True, exist_ok=True)
        if out_vis_path:  out_vis_path.parent.mkdir(parents=True, exist_ok=True)

        # 원본 크기
        pil = load_image_from_path(img_path)
        W0, H0 = pil.size

        # ✅ img-size<=0이면 원본 유지
        batch = tfm(pil)[None, ...]  # [1,3,H0,W0] 또는 리사이즈

        with torch.inference_mode():
            batch = batch.to(to_device)
            amp_dtype = torch.bfloat16 if str(to_device).startswith("cuda") else torch.float32
            with torch.autocast(device_type="cuda" if str(to_device).startswith("cuda") else "cpu",
                                dtype=amp_dtype):

                # 슬라이딩 파라미터 결정
                if args.inference_mode == "whole":
                    crop_size = None
                    stride = None
                else:
                    crop_size = (args.crop_size, args.crop_size)
                    stride = (args.stride, args.stride)

                logits_ade = make_inference(
                    batch,
                    segmentor,
                    inference_mode=args.inference_mode,   # "slide" or "whole"
                    decoder_head_type="m2f",
                    rescale_to=(H0, W0),                  # ✅ (H, W) 순서
                    n_output_channels=150,                # ADE20K
                    crop_size=crop_size,
                    stride=stride,
                    output_activation=None,               # logits
                )  # [1,150,H0,W0]

        # ─────────────── 라벨 생성 정책 분기 ───────────────
        if args.filter_by_ade_top1:
            # ADE Top-1 → TARGET30 매핑, 없으면 255
            ade_top = logits_ade.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)  # [H,W], 0..149
            mapped  = ade2t[ade_top]                                                       # [H,W], -1 or 0..29

            label = np.full_like(ade_top, IGN, dtype=np.uint8)
            m = mapped >= 0
            label[m] = mapped[m].astype(np.uint8)

            # (옵션) conf 저장: ADE softmax 최댓값
            if out_conf_path:
                with torch.no_grad():
                    prob_ade = torch.softmax(logits_ade, dim=1)
                    conf_np = prob_ade.max(dim=1)[0][0].detach().cpu().numpy().astype(np.float32)
                np.save(out_conf_path, conf_np.astype(np.float16))

        else:
            # 그룹 로그-합으로 30클래스 확률 → (필요시) tau 임계값 적용
            with torch.no_grad():
                logits_30 = ade_logits_to_target30_logits(logits_ade, groups)   # [1,30,H0,W0]
                prob_30   = torch.softmax(logits_30, dim=1)
                conf, pred= prob_30.max(dim=1)                                  # [1,H0,W0], [1,H0,W0]

            pred_np = pred[0].detach().cpu().numpy().astype(np.int32)
            conf_np = conf[0].detach().cpu().numpy().astype(np.float32)

            # tau 적용(전역 또는 기본 per-class)
            if args.tau_global > 0:
                tau_vec = np.array([args.tau_global]*len(TARGET30), dtype=np.float32)
            else:
                tau_vec = np.array([DEFAULT_TAU[t] for t in TARGET30], dtype=np.float32)

            tau_pix = np.take(tau_vec, pred_np, mode="clip")
            keep = conf_np >= tau_pix

            label = np.full_like(pred_np, IGN, dtype=np.uint8)
            label[keep] = pred_np[keep].astype(np.uint8)

            if out_conf_path:
                np.save(out_conf_path, conf_np.astype(np.float16))

        # 소블롭 제거
        if args.min_blob > 0:
            label = refine_small_blobs(label, min_pixels=args.min_blob)

        # 저장
        Image.fromarray(label).save(out_lbl_path)

        # 시각화: ignore는 그대로 두고 라벨 있는 픽셀만 혼합
        if out_vis_path:
            rgb = np.array(pil, dtype=np.uint8)
            mask_rgb = colorize_mask(label.clip(0, len(TARGET30)-1), len(TARGET30))
            vis = rgb.copy()
            mm = (label != IGN)
            vis[mm] = (0.5 * rgb[mm].astype(np.float32) + 0.5 * mask_rgb[mm].astype(np.float32)).astype(np.uint8)
            cv2.imwrite(str(out_vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        cnt += 1
        if cnt % 50 == 0:
            print(f"[RANK {RANK}] processed {cnt}/{len(my_paths)} ...")

    # 동기화 & 요약
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if RANK == 0:
        print(f"[DONE] total images: {len(all_paths)}")
        print(f"[OUT] labels  → {out_label_root}")
        if out_conf_root:   print(f"[OUT] confs   → {out_conf_root}")
        if out_overlay_root:print(f"[OUT] overlays→ {out_overlay_root}")


if __name__ == "__main__":
    main()
