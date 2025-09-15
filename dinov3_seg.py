import os, io, argparse, requests, sys
from functools import partial
from pathlib import Path

import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms


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

def make_transform(resize_size: int = 224):
    # README의 LVD-1689M용 표준 변환
    # mean/std = (0.485,0.456,0.406)/(0.229,0.224,0.225)
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([resize, to_tensor, normalize])

def colorize_mask(mask_hw: np.ndarray, num_classes: int) -> np.ndarray:
    # 고정 시드 팔레트 (재현성)
    rng = np.random.default_rng(42)
    palette = (rng.integers(0, 256, size=(num_classes, 3))).astype(np.uint8)
    return palette[mask_hw % num_classes]

def overlay(image_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    # 크기 다르면 mask를 원본에 맞춤
    if image_rgb.shape[:2] != mask_rgb.shape[:2]:
        mask_rgb = cv2.resize(
            mask_rgb, (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    # 채널 3개 보장
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    if mask_rgb.ndim == 2:
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_GRAY2RGB)

    out = cv2.addWeighted(
        image_rgb.astype(np.float32), 1 - alpha,
        mask_rgb.astype(np.float32), alpha, 0.0
    )
    return out.clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    # ✅ 로컬 파일 경로 우선
    ap.add_argument("--image-path", type=str, help="세그멘테이션할 로컬 이미지 경로 (우선 사용)")
    ap.add_argument("--image-url", type=str, help="이미지 URL (image-path 없을 때만 사용)")

    ap.add_argument("--repo-dir", default="./dinov3", type=str,
                    help="클론한 facebookresearch/dinov3 경로 (torch.hub source='local')")
    ap.add_argument("--weights-dir", default="./weights", type=str)

    # 체크포인트 파일명 (weights-dir 내부의 파일명)
    ap.add_argument("--backbone-ckpt", default="dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth", type=str,
                    help="예: dinov3_vit7b16_backbone.pth")
    ap.add_argument("--seg-ckpt", default="dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth", type=str,
                    help="예: dinov3_vit7b16_ms_ade20k_head.pth")

    # Hub entry: README 기준 세그멘테이터 예시는 'dinov3_vit7b16_ms'
    ap.add_argument("--hub-entry", default="dinov3_vit7b16_ms", type=str,
                    help="예: dinov3_vit7b16_ms (ADE20K), dinov3_vit7b16_dd (depth) 등")
    ap.add_argument("--img-size", default=896, type=int)
    ap.add_argument("--num-classes", default=150, type=int,  # ADE20K = 150
                    help="세그멘테이션 클래스 수")
    ap.add_argument("--save-path", default="./segmentation.png", type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir)
    assert repo_dir.exists(), f"[ERR] repo-dir not found: {repo_dir} (git clone facebookresearch/dinov3)"

    # dinov3 내부 유틸 import
    sys.path.append(str(repo_dir.resolve()))
    try:
        from dinov3.eval.segmentation.inference import make_inference
    except Exception as e:
        raise RuntimeError(
            f"[ERR] cannot import dinov3 inference utils from {repo_dir}. "
            f"'pip install -e {repo_dir}' 혹은 PYTHONPATH 설정을 확인하세요."
        ) from e

    # ✅ 이미지 로드: 파일 경로가 있으면 우선 사용
    if args.image_path:
        pil = load_image_from_path(args.image_path)
    elif args.image_url:
        pil = load_image_from_url(args.image_url)
    else:
        raise ValueError("[ERR] one of --image-path or --image-url must be provided")

    W0, H0 = pil.size

    # 변환
    tfm = make_transform(args.img_size)
    batch = tfm(pil)[None, ...]  # [1,3,H,W]

    # 체크포인트 경로
    backbone_ckpt_path = str(Path(args.weights_dir) / args.backbone_ckpt)
    seg_ckpt_path = str(Path(args.weights_dir) / args.seg_ckpt)

    if not os.path.isfile(backbone_ckpt_path):
        raise FileNotFoundError(f"[ERR] backbone ckpt not found: {backbone_ckpt_path}")
    if not os.path.isfile(seg_ckpt_path):
        raise FileNotFoundError(f"[ERR] segmentor ckpt not found: {seg_ckpt_path}")

    # 모델 로드 (PyTorch Hub / source='local')
    print(f"[INFO] Loading hub entry '{args.hub_entry}' ...")
    segmentor = torch.hub.load(
        str(repo_dir),
        args.hub_entry,
        source="local",
        weights=seg_ckpt_path,
        backbone_weights=backbone_ckpt_path,
    ).to(args.device).eval()

    # 인퍼런스
    with torch.inference_mode():
        batch = batch.to(args.device)
        amp_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu",
                            dtype=amp_dtype):
            seg_logits = make_inference(
                batch,
                segmentor,
                inference_mode="slide",
                decoder_head_type="m2f",  # README 예시 head
                rescale_to=(W0, H0),
                n_output_channels=args.num_classes,
                crop_size=(args.img_size, args.img_size),
                stride=(args.img_size, args.img_size),  # 필요시 (img_size//2, img_size//2)로 겹치기
                output_activation=partial(torch.nn.functional.softmax, dim=1),
            )  # [1, C, H0, W0]

        pred = seg_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)  # [H0,W0]

    # 시각화 & 저장
    rgb = np.array(pil, dtype=np.uint8)
    mask_rgb = colorize_mask(pred, args.num_classes)
    vis = overlay(rgb, mask_rgb, alpha=0.5)
    cv2.imwrite(args.save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[DONE] saved: {args.save_path}")


if __name__ == "__main__":
    main()
