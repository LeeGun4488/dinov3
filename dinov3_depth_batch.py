# dinov3_depth_batch.py
import os, io, argparse, requests, sys
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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


def build_model(repo_dir: Path, hub_entry: str, backbone_path: str, depth_path: str, device: str):
    sys.path.append(str(repo_dir))
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass

    depther = torch.hub.load(
        str(repo_dir),
        hub_entry,
        source="local",
        weights=depth_path if os.path.isfile(depth_path) else None,
        backbone_weights=backbone_path if os.path.isfile(backbone_path) else None,
    ).to(device).eval()
    return depther.to(memory_format=torch.channels_last)


def infer_one(pil: Image.Image, depther, tfm, device: str, pad_mult: int):
    W0, H0 = pil.size
    img = tfm(pil).unsqueeze(0)  # [1,3,H0,W0]
    img, (pad_h, pad_w) = pad_to_multiple(img, max(1, pad_mult))

    with torch.inference_mode():
        x = img.to(device, memory_format=torch.channels_last, non_blocking=True)
        amp_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        ctx = torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                             dtype=amp_dtype)
        with ctx:
            out = depther(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, dict):
                out = next(t for t in out.values() if torch.is_tensor(t))
            if out.ndim != 4 or out.shape[1] != 1:
                raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)}")
            depth_t = out[0, 0]  # [h,w]

    # 패딩 제거
    if pad_h or pad_w:
        depth_t = depth_t[:depth_t.shape[0]-pad_h or None, :depth_t.shape[1]-pad_w or None]

    # 원본 크기로 보간
    if depth_t.shape[-2:] != (H0, W0):
        depth_t = F.interpolate(depth_t[None, None, ...], size=(H0, W0),
                                mode="bilinear", align_corners=False)[0, 0]
    return depth_t.detach().float().cpu().numpy()


def make_out_paths(inp_path: Path, in_root: Path, out_root: Path, want_npy: bool):
    """입력 경로의 상대경로를 유지하며 파일명에서 'leftImg8bit' → 'depth_map' 치환."""
    rel = inp_path.relative_to(in_root)
    stem = rel.stem  # e.g., aachen_000001_000019_leftImg8bit
    if "leftImg8bit" in stem:
        new_stem = stem.replace("leftImg8bit", "depth_map")
    else:
        new_stem = f"{stem}_depth_map"

    out_dir = out_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / (new_stem + ".png")
    out_npy = (out_dir / (new_stem + ".npy")) if want_npy else None
    return out_png, out_npy


def main():
    ap = argparse.ArgumentParser()
    # 입력
    ap.add_argument("--image-path", type=str, help="단일 로컬 이미지 경로")
    ap.add_argument("--image-url",  type=str, help="단일 원격 이미지 URL")
    ap.add_argument("--input-dir",  type=str, help="재귀적으로 이미지를 처리할 입력 폴더(root)")
    # 모델/가중치
    ap.add_argument("--repo-dir",   default=".", type=str, help="dinov3 레포 루트(hubconf.py 위치)")
    ap.add_argument("--weights-dir", default="./weights", type=str)
    ap.add_argument("--backbone-ckpt", default="dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth", type=str)
    ap.add_argument("--depth-ckpt",    default="dinov3_vit7b16_synthmix_dpt_head-02040be1.pth", type=str)
    ap.add_argument("--hub-entry",     default="dinov3_vit7b16_dd", type=str)
    # 출력
    ap.add_argument("--save-path", default="./depth_full.png", type=str, help="단일 이미지 저장 경로")
    ap.add_argument("--output-dir", default="./depth_out", type=str, help="폴더 처리 시 출력 루트")
    ap.add_argument("--save-npy", action="store_true")
    # 기타
    ap.add_argument("--pad-to-multiple", default=16, type=int)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = ap.parse_args()

    # 모드 판단
    folder_mode = args.input_dir is not None
    if not folder_mode and not (args.image_path or args.image_url):
        raise SystemExit("[ERR] --input-dir 또는 (--image-path | --image-url) 중 하나는 필요합니다.")

    repo_dir = Path(args.repo_dir).resolve()
    assert repo_dir.exists(), f"[ERR] repo-dir not found: {repo_dir}"

    backbone_path = str(Path(args.weights_dir) / args.backbone_ckpt)
    depth_path    = str(Path(args.weights_dir) / args.depth_ckpt)

    depther = build_model(repo_dir, args.hub_entry, backbone_path, depth_path, args.device)
    tfm = make_transform_no_resize()

    if folder_mode:
        in_root = Path(args.input_dir).resolve()
        out_root = Path(args.output_dir).resolve()
        if not in_root.exists():
            raise SystemExit(f"[ERR] input-dir not found: {in_root}")

        img_list = [p for p in in_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if not img_list:
            raise SystemExit(f"[ERR] No images found under: {in_root}")

        print(f"[INFO] Found {len(img_list)} images under {in_root}")
        for idx, p in enumerate(sorted(img_list), 1):
            try:
                pil = Image.open(p).convert("RGB")
                depth = infer_one(pil, depther, tfm, args.device, args.pad_to_multiple)

                out_png, out_npy = make_out_paths(p, in_root, out_root, args.save_npy)
                depth_rgb = depth_to_colormap(depth, invert=args.invert)
                out_png.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_png), cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))
                if out_npy is not None:
                    np.save(str(out_npy), depth)

                print(f"[{idx}/{len(img_list)}] saved: {out_png.name}")
            except Exception as e:
                print(f"[WARN] failed on {p}: {e}")

        print(f"[DONE] outputs @ {out_root}")
        return

    # ---- 단일 이미지/URL 모드 ----
    pil = load_image(args.image_path or args.image_url)
    depth = infer_one(pil, depther, tfm, args.device, args.pad_to_multiple)

    if args.save_npy:
        np.save(Path(args.save_path).with_suffix(".npy"), depth)

    depth_rgb = depth_to_colormap(depth, invert=args.invert)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.save_path, cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))
    H0, W0 = depth.shape
    print(f"[DONE] {args.save_path} | output {W0}x{H0}")


if __name__ == "__main__":
    main()
