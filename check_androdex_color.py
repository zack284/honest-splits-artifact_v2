#!/usr/bin/env python3
"""
check_androdex_color.py -- AndroDex 이미지의 컬러 규약 판별

세 경우를 구분한다:
  1. R=G=B         -> 회색조를 RGB 로 저장. convert("L") 무손실. 그대로 진행
  2. colormap      -> 고유 색이 256개 이하이고 R!=G!=B. convert("L") 은 파괴적
  3. 3 byte -> RGB -> 고유 색이 매우 많음. 채널마다 다른 byte. in_channels=3 필요

Usage:
    python check_androdex_color.py --dir /path/to/androdex/images --n 20
"""

import argparse, os, random
from collections import Counter

import numpy as np
from PIL import Image

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="AndroDex 이미지 최상위 폴더")
    ap.add_argument("--n", type=int, default=20, help="표본 개수")
    args = ap.parse_args()

    files = []
    for root, _, fs in os.walk(args.dir):
        for f in fs:
            if f.lower().endswith(EXTS):
                files.append(os.path.join(root, f))
    if not files:
        raise SystemExit(f"[FATAL] 이미지를 못 찾음: {args.dir}")
    print(f"발견: {len(files)}장 -> {min(args.n, len(files))}장 표본\n")

    random.Random(42).shuffle(files)
    sample = files[:args.n]

    modes, sizes, widths = Counter(), [], Counter()
    n_gray, n_color = 0, 0
    uniq_colors = []

    for p in sample:
        with Image.open(p) as im:
            modes[im.mode] += 1
            W, H = im.size
            sizes.append((W, H)); widths[W] += 1
            a = np.asarray(im)

        if a.ndim == 2:
            n_gray += 1
            continue
        if a.shape[2] >= 3:
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            if np.array_equal(r, g) and np.array_equal(g, b):
                n_gray += 1
            else:
                n_color += 1
                flat = a[..., :3].reshape(-1, 3)
                uniq_colors.append(len(np.unique(flat, axis=0)))

    print("=" * 60)
    print("기본 정보")
    print("=" * 60)
    print(f"  PIL mode : {dict(modes)}")
    print(f"  width    : {dict(sorted(widths.items()))}")
    print(f"  크기 예시: {sizes[:3]}")

    print("\n" + "=" * 60)
    print("판정")
    print("=" * 60)
    print(f"  R=G=B 인 이미지 : {n_gray}/{len(sample)}")
    print(f"  진짜 컬러       : {n_color}/{len(sample)}")

    if n_color == 0:
        print("\n  => [경우 1] 회색조입니다 (RGB 로 저장됐을 뿐).")
        print("     convert('L') 무손실. 지금 pix2pix 코드 그대로 사용 가능.")
    else:
        m = int(np.median(uniq_colors))
        print(f"  고유 색 개수(중앙값): {m}")
        if m <= 256:
            print("\n  => [경우 2] colormap 이 적용된 것으로 보입니다.")
            print("     convert('L') 은 세 채널을 가중평균해 byte 값을 파괴합니다.")
            print("     colormap 을 역산해 원래 byte 를 복원해야 합니다.")
            print("     AndroDex 변환 코드(figshare 23931477)에서 팔레트 확인 필요.")
        else:
            print("\n  => [경우 3] 3 byte -> 1 RGB 픽셀 구조로 보입니다.")
            print("     채널마다 서로 다른 dex byte 입니다.")
            print("     convert('L') 은 무관한 세 byte 를 섞습니다. 절대 쓰면 안 됩니다.")
            print("     pix2pix 를 in_channels=3 / Discriminator in_channels=6 으로 수정.")
            print("     공간 구조도 CICMalDroid(회색조)와 달라집니다.")

    print("\n" + "=" * 60)
    print("width 규칙 비교")
    print("=" * 60)
    nataraj = {32, 64, 128, 256, 384, 512, 768, 1024}
    seen = set(widths)
    print(f"  AndroDex width : {sorted(seen)}")
    print(f"  CICMalDroid    : {sorted(nataraj)}")
    print(f"  일치 여부      : {'예' if seen <= nataraj else '아니오 - 규칙이 다름'}")


if __name__ == "__main__":
    main()