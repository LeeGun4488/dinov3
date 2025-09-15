# dinov3_depth_batch_fp32.py
import os, io, argparse, requests, sys
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import multiprocessing as mp

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ------------------------- IO utils -------------------------
def load_image(src: str) -> Image.Image:
    if os.path.isfile(src):
        return Image.open(src).convert("RGB")
    p = urlparse(src)
    if p.scheme in ("http", "https"):
        r = requests.get(src, timeout=20); r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    if p.scheme == "file":
        return Image.open(p.path).convert("RGB")
    raise ValueError(f"Unsupported image source: {src}")

def make_transform_no_resize():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

def pad_to_multiple(t: torch.Tensor, multiple: int):
    """t: [B,C,H,W] → 오른쪽/아래 0패딩으로 H,W를 multiple의 배수로."""
    if multiple <= 1:
        return t, (0, 0)
    H, W = t.shape[-2:]
    Ht = (H + multiple - 1) // multiple * multiple
    Wt = (W + multiple - 1) // multiple * multiple
    pad_h, pad_w = Ht - H, Wt - W
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h))
    return t, (pad_h, pad_w)

def depth_to_colormap(depth_hw: np.ndarray, invert: bool = True,
                      min_val: float | None = None, max_val: float | None = None) -> np.ndarray:
    """
    depth_hw: 2D float array (보통 '미터' 단위)
    min_val/max_val가 주어지면 그 구간으로 정규화(절대 스케일).
    invert=True면 가까움(작은 값)이 밝게 보이도록 반전.
    """
    d = depth_hw.astype(np.float32)
    if min_val is not None and max_val is not None and max_val > min_val:
        d = (d - float(min_val)) / (float(max_val) - float(min_val) + 1e-8)
        d = np.clip(d, 0.0, 1.0)
    else:
        # (fallback) 이미지별 min/max 정규화
        if np.isfinite(d).any():
            d = np.nan_to_num(d, nan=np.nanmin(d))
            mn, mx = float(np.min(d)), float(np.max(d))
            d = (d - mn) / (mx - mn + 1e-8)
        else:
            d = np.zeros_like(d, dtype=np.float32)

    if invert:
        d = 1.0 - d  # 가까움 밝게, 멀어짐 어둡게

    d8 = np.clip(d * 255.0, 0, 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

def make_out_paths(inp_path: Path, in_root: Path, out_root: Path, want_npy: bool):
    """입력 경로의 상대경로 유지 + 파일명 'leftImg8bit' → 'depth_map' 치환."""
    rel = inp_path.relative_to(in_root)
    stem = rel.stem
    new_stem = stem.replace("leftImg8bit", "depth_map") if "leftImg8bit" in stem else f"{stem}_depth_map"
    out_dir = out_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / (new_stem + ".png")
    out_npy = (out_dir / (new_stem + ".npy")) if want_npy else None
    return out_png, out_npy

# ------------------------- Dataset -------------------------
class CSImageDataset(Dataset):
    def __init__(self, paths, tfm):
        self.paths = list(paths)
        self.tfm = tfm
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        pil = Image.open(p).convert("RGB")
        tensor = self.tfm(pil)               # [3,H,W] (NCHW 기대)
        H0, W0 = pil.size[1], pil.size[0]    # H,W
        return tensor, str(p), (H0, W0)

# ------------------------- Model -------------------------
def build_model(repo_dir: Path, hub_entry: str, backbone_path: str, depth_path: str, device: str):
    sys.path.append(str(repo_dir))
    torch.backends.cuda.matmul.allow_tf32 = True   # FP32 matmul → TF32 텐서코어 가속
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    depther = torch.hub.load(
        str(repo_dir),
        hub_entry,
        source="local",
        weights=depth_path if os.path.isfile(depth_path) else None,
        backbone_weights=backbone_path if os.path.isfile(backbone_path) else None,
    ).to(device).eval()
    # NHWC(channels_last) 비활성화 → NCHW로 고정 (INT_MAX 업샘플 이슈 회피)
    return depther
    
# ─────────────────────────────────────────────
# (NEW) 비디오 프레임 범위 분할
# ─────────────────────────────────────────────
def split_frame_ranges(total_frames: int, gpus: int, start_f: int, end_f: int):
    """[start_f, end_f] 구간을 gpus개로 나눠 (s,e) 리스트 반환"""
    end_f = min(end_f, total_frames - 1)
    if end_f < start_f:
        return []
    n = end_f - start_f + 1
    base = n // gpus
    rem  = n % gpus
    ranges = []
    cur = start_f
    for r in range(gpus):
        take = base + (1 if r < rem else 0)
        if take <= 0:
            ranges.append((0, -1))  # 빈 구간
            continue
        s = cur
        e = cur + take - 1
        ranges.append((s, e))
        cur = e + 1
    return ranges

# ─────────────────────────────────────────────
# (NEW) 비디오 워커: 프레임 구간만 처리(비디오 파일은 저장 안 함)
# ─────────────────────────────────────────────
def video_worker(rank: int, args, frame_range: tuple[int,int]):
    s_frame, e_frame = frame_range
    if e_frame < s_frame:
        print(f"[GPU{rank}] empty range, skip")
        return

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # 모델 로드
    repo_dir = Path(args.repo_dir).resolve()
    backbone_path = str(Path(args.weights_dir) / args.backbone_ckpt)
    depth_path    = str(Path(args.weights_dir) / args.depth_ckpt)
    depther = build_model(repo_dir, args.hub_entry, backbone_path, depth_path, device)
    tfm = make_transform_no_resize()

    # 비디오 열기 & 시킹
    vpath = Path(args.video_path)
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {vpath}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)

    # 출력 디렉토리
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    frames_dir = out_root / (vpath.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    with torch.inference_mode():
        gidx = s_frame  # 전역 프레임 인덱스
        while gidx <= e_frame:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # 프레임 텐서화
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame_rgb)
            x = tfm(im).unsqueeze(0).to(device).float()  # [1,3,H,W]
            x, (pad_h, pad_w) = pad_to_multiple(x, max(1, args.pad_to_multiple))

            # FP32 추론
            out = depther(x)
            if isinstance(out, (list, tuple)): out = out[0]
            if isinstance(out, dict): out = next(t for t in out.values() if torch.is_tensor(t))
            assert out.ndim == 4 and out.shape[1] == 1
            if pad_h or pad_w:
                out = out[:, :, :out.shape[-2]-pad_h or None, :out.shape[-1]-pad_w or None]

            depth_raw = out[0,0].detach().float().cpu().numpy()  # (H,W)

            # raw → meters 매핑
            if args.abs_mode == "per_image":
                rmin = float(np.nanmin(depth_raw))
                rmax = float(np.nanmax(depth_raw))
            else:
                rmin, rmax = float(args.raw_min), float(args.raw_max)

            S = float(args.meter_scale)
            maxm = float(args.max_meters) * S

            if not np.isfinite(rmin) or not np.isfinite(rmax) or rmax <= rmin:
                meters = np.zeros_like(depth_raw, dtype=np.float32)
            else:
                meters = (depth_raw - rmin) / (rmax - rmin + 1e-8) * maxm
                meters = np.clip(meters, 0.0, maxm).astype(np.float32)

            # 저장: NPY (+ 옵션 PNG)
            np.save(str(frames_dir / f"frame_{gidx:06d}.npy"), meters)

            if args.save_frame_png:
                depth_rgb = depth_to_colormap(meters, invert=True, min_val=0.0, max_val=maxm)
                cv2.imwrite(str(frames_dir / f"frame_{gidx:06d}.png"),
                            cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))

            processed += 1
            if (processed % 50 == 0) or (total > 0 and gidx == e_frame):
                print(f"[GPU{rank}] frames {gidx}/{e_frame} (saved {processed})")

            if torch.cuda.is_available() and (processed % 64 == 0):
                torch.cuda.empty_cache()

            gidx += 1

    cap.release()
    print(f"[GPU{rank}] done range [{s_frame}, {e_frame}] (saved {processed})")

def process_video(args):
    """동영상을 읽어 depth 컬러 비디오를 저장하고, 프레임별 미터 단위 depth를 .npy로 저장."""
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"

    # 모델 로드
    repo_dir = Path(args.repo_dir).resolve()
    backbone_path = str(Path(args.weights_dir) / args.backbone_ckpt)
    depth_path    = str(Path(args.weights_dir) / args.depth_ckpt)
    depther = build_model(repo_dir, args.hub_entry, backbone_path, depth_path, device)
    tfm = make_transform_no_resize()

    # 비디오 IO
    vpath = Path(args.video_path)
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {vpath}")

    in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    # 시간 범위 설정
    if args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)
    end_msec = (args.end_sec * 1000.0) if args.end_sec > 0 else None

    # 출력 준비
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    frames_dir = out_root / (vpath.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video_path = Path(args.out_video)
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (in_w, in_h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {out_video_path}")

    frame_idx = 0
    with torch.inference_mode():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            cur_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if end_msec is not None and cur_msec > end_msec:
                break

            # 프레임을 텐서로
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame_rgb)
            x = tfm(im).unsqueeze(0).to(device).float()  # [1,3,H,W]
            x, (pad_h, pad_w) = pad_to_multiple(x, max(1, args.pad_to_multiple))

            # FP32 추론
            out = depther(x)
            if isinstance(out, (list, tuple)): out = out[0]
            if isinstance(out, dict): out = next(t for t in out.values() if torch.is_tensor(t))
            assert out.ndim == 4 and out.shape[1] == 1
            if pad_h or pad_w:
                out = out[:, :, :out.shape[-2]-pad_h or None, :out.shape[-1]-pad_w or None]

            depth_raw = out[0,0].detach().float().cpu().numpy()  # (H,W)

            # raw → meters 매핑 (이미지 모드와 동일 로직 사용)
            if args.abs_mode == "per_image":
                rmin = float(np.nanmin(depth_raw))
                rmax = float(np.nanmax(depth_raw))
            else:
                rmin, rmax = float(args.raw_min), float(args.raw_max)

            S = float(args.meter_scale)
            maxm = float(args.max_meters) * S

            if not np.isfinite(rmin) or not np.isfinite(rmax) or rmax <= rmin:
                meters = np.zeros_like(depth_raw, dtype=np.float32)
            else:
                meters = (depth_raw - rmin) / (rmax - rmin + 1e-8) * maxm
                meters = np.clip(meters, 0.0, maxm).astype(np.float32)

            # 프레임별 .npy 저장
            npy_path = frames_dir / f"frame_{frame_idx:06d}.npy"
            np.save(str(npy_path), meters)

            # 컬러맵 (가까움 밝게) → 비디오 프레임
            depth_rgb = depth_to_colormap(meters, invert=True, min_val=0.0, max_val=maxm)
            if args.save_frame_png:
                png_path = frames_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(png_path), cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))

            if writer:
                writer.write(cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))

            frame_idx += 1
            if (frame_idx % 50 == 0) or (total > 0 and frame_idx == total):
                print(f"[video] {frame_idx}/{total if total>0 else '?'} frames")

            if torch.cuda.is_available() and (frame_idx % 64 == 0):
                torch.cuda.empty_cache()

    cap.release()
    if writer:
        writer.release()
    print(f"[DONE] video frames saved @ {frames_dir}")
    if args.out_video:
        print(f"[DONE] depth video saved @ {args.out_video}")

# ------------------------- Worker -------------------------
def worker(rank, world_size, args, img_paths):
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    repo_dir = Path(args.repo_dir).resolve()
    backbone_path = str(Path(args.weights_dir) / args.backbone_ckpt)
    depth_path    = str(Path(args.weights_dir) / args.depth_ckpt)
    depther = build_model(repo_dir, args.hub_entry, backbone_path, depth_path, device)
    tfm = make_transform_no_resize()

    dataset = CSImageDataset(img_paths, tfm)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,          # GPU당 배치 (기본 4)
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    in_root  = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    total = len(dataset)
    processed = 0

    with torch.inference_mode():
        for imgs, paths, sizes in loader:
            B = imgs.shape[0]
            # 마이크로배치로 쪼개 처리(기본 4 → 한 번에 처리)
            for s in range(0, B, args.micro_batch):
                e  = min(s + args.micro_batch, B)
                mb = imgs[s:e].to(device, non_blocking=True).float()  # FP32 고정
                mb, (pad_h, pad_w) = pad_to_multiple(mb, max(1, args.pad_to_multiple))

                # ★ FP32 추론 (autocast 없음)
                out = depther(mb)   # [mb,1,h,w] or dict/list
                if isinstance(out, (list, tuple)): out = out[0]
                if isinstance(out, dict): out = next(t for t in out.values() if torch.is_tensor(t))
                assert out.ndim == 4 and out.shape[1] == 1, f"Unexpected output {tuple(out.shape)}"

                # 패딩 제거
                if pad_h or pad_w:
                    out = out[:, :, :out.shape[-2]-pad_h or None, :out.shape[-1]-pad_w or None]

                # 원본 크기 보장 (시티스케이프는 동일하나 안전하게)
                origH, origW = mb.shape[-2], mb.shape[-1]
                if out.shape[-2:] != (origH, origW):
                    out = F.interpolate(out, size=(origH, origW), mode="bilinear", align_corners=False)

                depths = out[:, 0].detach().float().cpu().numpy()  # [mb,H,W] (raw)

                # 저장
                for i in range(depths.shape[0]):
                    p = Path(paths[s+i])
                    raw_depth = depths[i]  # 모델 raw

                    # --- raw -> meters 선형 매핑 ---
                    if args.abs_mode == "per_image":
                        rmin = float(np.nanmin(raw_depth))
                        rmax = float(np.nanmax(raw_depth))
                    else:  # fixed
                        rmin, rmax = float(args.raw_min), float(args.raw_max)

                    S = float(args.meter_scale)
                    maxm = float(args.max_meters) * S
                    
                    if not np.isfinite(rmin) or not np.isfinite(rmax) or rmax <= rmin:
                        meters = np.zeros_like(raw_depth, dtype=np.float32)
                    else:
                        meters = (raw_depth - rmin) / (rmax - rmin + 1e-8) * maxm
                        meters = np.clip(meters, 0.0, maxm).astype(np.float32)

                    # --- 출력 경로 ---
                    out_png, _ = make_out_paths(p, in_root, out_root, want_npy=False)
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    base = str(out_png)[:-4]  # ".png" 제거

                    # --- 시각화: 가까움 밝게 / 멀어짐 어둡게 ---
                    depth_rgb = depth_to_colormap(meters, invert=True,
                                                  min_val=0.0, max_val=maxm)
                    cv2.imwrite(str(out_png), cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))

                    # --- NPY 저장: meters(미터) ---
                    if args.save_npy:
                        np.save(base + ".npy", meters)

                    # --- raw NPY(선택) ---
                    if args.save_raw_npy:
                        np.save(base + "_raw.npy", raw_depth)

                    processed += 1

                if torch.cuda.is_available() and (processed % 64 == 0):
                    torch.cuda.empty_cache()

            print(f"[GPU{rank}] {processed}/{total} done")

    print(f"[GPU{rank}] finished shard.")

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    # 입력(폴더 모드 권장)
    ap.add_argument("--input-dir",  type=str, help="재귀적으로 이미지를 처리할 입력 폴더(root)")
    ap.add_argument("--image-path", type=str, help="단일 이미지")
    ap.add_argument("--image-url",  type=str, help="단일 URL")
    # 모델/가중치
    ap.add_argument("--repo-dir",   default=".", type=str)
    ap.add_argument("--weights-dir", default="./weights", type=str)
    ap.add_argument("--backbone-ckpt", default="dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth", type=str)
    ap.add_argument("--depth-ckpt",    default="dinov3_vit7b16_synthmix_dpt_head-02040be1.pth", type=str)
    ap.add_argument("--hub-entry",     default="dinov3_vit7b16_dd", type=str)
    # 출력
    ap.add_argument("--output-dir", default="./depth_out", type=str)
    ap.add_argument("--save-path",  default="./depth_full.png", type=str)  # 단일 모드
    ap.add_argument("--save-npy", action="store_true")
    # 하이퍼파라미터 / 병렬
    ap.add_argument("--pad-to-multiple", default=16, type=int)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--batch-size", default=4, type=int)      # ★ 기본 4
    ap.add_argument("--micro-batch", default=4, type=int)     # ★ 기본 4 (배치 = 마이크로배치)
    ap.add_argument("--num-workers", default=4, type=int)
    ap.add_argument("--gpus", default=1, type=int)
    
    ap.add_argument("--abs-mode", choices=["per_image", "fixed"], default="per_image",
                help="per_image: 각 이미지 min/max를 0~max_meters로 매핑, fixed: raw_min~raw_max를 0~max_meters로 매핑")
    ap.add_argument("--raw-min", default=30.0, type=float, help="--abs-mode fixed일 때 사용")
    ap.add_argument("--raw-max", default=50.0, type=float, help="--abs-mode fixed일 때 사용")
    ap.add_argument("--max-meters", default=80.0, type=float, help="가장 먼 곳을 몇 m로 볼지")
    ap.add_argument("--save-raw-npy", action="store_true", help="raw depth도 별도 *_raw.npy로 저장")
    
    ap.add_argument("--meter-scale", default=0.64, type=float,
                help="전역 스케일 S (meters에 곱함). 예: 0.5면 거리를 절반으로 축소")
                
    # ▶ 동영상 모드 추가
    ap.add_argument("--video-path", type=str, help="입력 동영상 경로")
    ap.add_argument("--out-video",  type=str, help="출력 depth 비디오 경로 (예: out.mp4)")
    ap.add_argument("--start-sec",  type=float, default=0.0, help="시작 시각(초)")
    ap.add_argument("--end-sec",    type=float, default=0.0, help="끝 시각(초, 0=영상 끝까지)")
    ap.add_argument("--save-frame-png", action="store_true",
                    help="프레임별 depth 컬러맵 PNG도 저장(기본은 비활성)")

    args = ap.parse_args()
    
    if args.video_path:
        vpath = Path(args.video_path)
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {vpath}")
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
        cap.release()
        if total <= 0:
            raise RuntimeError("총 프레임 수를 읽을 수 없습니다.")

        # 초 → 프레임
        start_f = int(max(0.0, args.start_sec) * fps)
        end_f   = int(args.end_sec * fps) - 1 if args.end_sec > 0 else (total - 1)
        end_f   = min(end_f, total - 1)

        # GPU 수
        gpus = min(args.gpus, torch.cuda.device_count() if torch.cuda.is_available() else 1)
        ranges = split_frame_ranges(total, gpus, start_f, end_f)
        print(f"[VIDEO] total_frames={total}  fps={fps:.2f}  range=[{start_f},{end_f}]  gpus={gpus}")
        for r, (s,e) in enumerate(ranges):
            print(f"  - shard{r}: [{s},{e}] (len={max(0,e-s+1)})")

        # 멀티프로세스 실행
        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(gpus):
            p = ctx.Process(target=video_worker, args=(rank, args, ranges[rank]))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        print(f"[DONE] per-frame .npy 저장 완료 → {Path(args.output_dir) / (vpath.stem + '_frames')}")
        return
    
    # 단일 이미지/URL 모드는 멀티GPU 배치 저장 지원 X
    if not args.input_dir and (args.image_path or args.image_url):
        raise SystemExit("[ERR] 멀티GPU 배치 저장 기능은 --input-dir 모드에서 사용하세요.]")

    if not args.input_dir:
        raise SystemExit("[ERR] --input-dir 이 필요합니다.]")
    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    if not in_root.exists():
        raise SystemExit(f"[ERR] input-dir not found: {in_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted([p for p in in_root.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not all_imgs:
        raise SystemExit(f"[ERR] No images found under: {in_root}")

    gpus = min(args.gpus, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    shards = [all_imgs[i::gpus] for i in range(gpus)]
    print(f"[INFO] total images: {len(all_imgs)} | gpus: {gpus} | per-shard {[len(s) for s in shards]}")

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(gpus):
        p = ctx.Process(target=worker, args=(rank, gpus, args, shards[rank]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    print(f"[DONE] outputs @ {out_root}")

if __name__ == "__main__":
    main()
