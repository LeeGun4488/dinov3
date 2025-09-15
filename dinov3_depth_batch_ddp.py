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

def depth_to_colormap(depth_hw: np.ndarray, invert: bool = False) -> np.ndarray:
    d = depth_hw.astype(np.float32)
    if np.isfinite(d).any():
        d = np.nan_to_num(d, nan=np.nanmin(d))
        mn, mx = float(np.min(d)), float(np.max(d))
        d = (d - mn) / (mx - mn + 1e-8)
    else:
        d = np.zeros_like(d, dtype=np.float32)
    if invert: d = 1.0 - d
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

                depths = out[:, 0].detach().float().cpu().numpy()  # [mb,H,W]
                # 저장
                for i in range(depths.shape[0]):
                    p = Path(paths[s+i])
                    depth = depths[i]
                    out_png, out_npy = make_out_paths(p, in_root, out_root, args.save_npy)
                    depth_rgb = depth_to_colormap(depth, invert=args.invert)
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_png), cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))
                    if out_npy is not None:
                        np.save(str(out_npy), depth)
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
    args = ap.parse_args()

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
