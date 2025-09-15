import os, io, argparse, requests, sys
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms


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
        t = F.pad(t, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-path", type=str, help="로컬 이미지 경로")
    ap.add_argument("--image-url",  type=str, help="원격 이미지 URL")
    ap.add_argument("--repo-dir",   default=".", type=str, help="dinov3 레포 루트(hubconf.py 위치)")
    ap.add_argument("--weights-dir", default="./weights", type=str)
    ap.add_argument("--backbone-ckpt", default="dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth", type=str)
    ap.add_argument("--depth-ckpt",    default="dinov3_vit7b16_synthmix_dpt_head-02040be1.pth", type=str)
    ap.add_argument("--hub-entry",     default="dinov3_vit7b16_dd", type=str)
    ap.add_argument("--pad-to-multiple", default=16, type=int, help="H,W를 N 배수로 패딩(0=비활성)")
    ap.add_argument("--save-path", default="./depth_full.png", type=str)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--save-npy", action="store_true")
    args = ap.parse_args()

    image_src = args.image_path or args.image_url
    if not image_src:
        raise SystemExit("[ERR] --image-path 또는 --image-url 중 하나는 필요합니다.")

    repo_dir = Path(args.repo_dir).resolve()
    assert repo_dir.exists(), f"[ERR] repo-dir not found: {repo_dir}"
    sys.path.append(str(repo_dir))

    # 이미지 로드(풀해상도)
    pil = load_image(image_src)
    W0, H0 = pil.size
    tfm = make_transform_no_resize()
    img = tfm(pil).unsqueeze(0)  # [1,3,H0,W0]

    # ViT 패치(16) 배수 보장 패딩
    img, (pad_h, pad_w) = pad_to_multiple(img, max(1, args.pad_to_multiple))

    # 허브 로드 (캐시에 동일 파일명이 있으면 네트워크 없이 사용)
    backbone_path = str(Path(args.weights_dir) / args.backbone_ckpt)
    depth_path    = str(Path(args.weights_dir) / args.depth_ckpt)
    print(f"[INFO] Loading hub entry '{args.hub_entry}' ...")

    # 약간의 성능 옵션
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass

    depther = torch.hub.load(
        str(repo_dir),
        args.hub_entry,
        source="local",
        weights=depth_path if os.path.isfile(depth_path) else None,
        backbone_weights=backbone_path if os.path.isfile(backbone_path) else None,
    ).to(args.device).eval()
    depther = depther.to(memory_format=torch.channels_last)

    # 단일 패스 추론
    with torch.inference_mode():
        x = img.to(args.device, memory_format=torch.channels_last, non_blocking=True)
        amp_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu",
                            dtype=amp_dtype):
            out = depther(x)  # 기대: [B,1,h,w] 또는 dict/list
            if isinstance(out, (list, tuple)): out = out[0]
            if isinstance(out, dict): out = next(t for t in out.values() if torch.is_tensor(t))
            if out.ndim != 4 or out.shape[1] != 1:
                raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)}")
            depth_t = out[0, 0]  # [h,w]

    # 패딩 제거
    if pad_h or pad_w:
        depth_t = depth_t[:depth_t.shape[0]-pad_h or None, :depth_t.shape[1]-pad_w or None]

    # 필요하면 원본 크기로 보간(일치하지 않을 때)
    h_out, w_out = depth_t.shape[-2], depth_t.shape[-1]
    if (h_out, w_out) != (H0, W0):
        depth_t = F.interpolate(depth_t[None, None, ...], size=(H0, W0),
                                mode="bilinear", align_corners=False)[0, 0]

    depth = depth_t.detach().float().cpu().numpy()

    if args.save_npy:
        np.save(Path(args.save_path).with_suffix(".npy"), depth)

    depth_rgb = depth_to_colormap(depth, invert=args.invert)
    cv2.imwrite(args.save_path, cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))
    print(f"[DONE] {args.save_path} | input {W0}x{H0}, output {depth.shape[1]}x{depth.shape[0]}")

if __name__ == "__main__":
    main()
