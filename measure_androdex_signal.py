#!/usr/bin/env python3
"""
measure_androdex_signal.py -- AndroDex JPG 에서 난독화 신호가 살아있는가

배경:
  CICMalDroid(회색조 PNG, 1byte/pixel) 에서 SSIM(x, x') = 0.12 였다.
  즉 난독화가 이미지를 크게 바꾼다 = 신호가 강하다.

  AndroDex 는 RGB JPG (3byte -> 1pixel, 손실압축) 다.
  JPG 압축은 고주파를 버리는데, 난독화가 만드는 byte 패턴이 바로 고주파다.
  압축이 신호를 지웠다면 SSIM 이 크게 높아진다.

판정:
  SSIM ~ 0.15  -> 신호 살아있음. JPG 그대로 진행 가능
  SSIM ~ 0.50  -> 압축이 신호를 지움. binary 재렌더링 필요

같이 재는 것:
  - RGB 채널별 SSIM (채널마다 다른 dex byte 이므로 따로 봐야 함)
  - Grayscale 변환 후 SSIM (기존 코드가 하는 짓. 얼마나 손해인지)
  - 고유 색 개수 (JPG 압축이 값 분포를 얼마나 뭉갰는지)
  - 크기 비율 (CICMalDroid 는 median 1.000 이었음)

Usage:
    python measure_androdex_signal.py \
        --manifest ~Androdex/androdex_set2_manifest.csv --n 300
"""

import argparse, csv, random
import statistics as st

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

S = 256


def load(path, mode):
    with Image.open(path) as im:
        return np.asarray(im.convert(mode).resize((S, S), Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    random.Random(42).shuffle(rows)
    rows = rows[: args.n]
    print(f"표본 {len(rows)} 쌍\n")

    ss_rgb, ss_gray = [], []
    ss_ch = {0: [], 1: [], 2: []}
    n_uniq_o, n_uniq_b = [], []
    ratios = []
    by_cat = {}

    for i, r in enumerate(rows):
        if i % 50 == 0:
            print(f"  ... {i}/{len(rows)}")
        o = load(r["original_image"], "RGB")
        b = load(r["obfuscated_image"], "RGB")

        # 채널별 (각 채널이 서로 다른 dex byte)
        for c in range(3):
            ss_ch[c].append(float(ssim_fn(o[..., c], b[..., c], data_range=255)))

        # RGB 전체 (multichannel)
        ss_rgb.append(float(ssim_fn(o, b, data_range=255, channel_axis=2)))

        # Grayscale 변환 후 (기존 코드가 하는 짓)
        og = load(r["original_image"], "L")
        bg = load(r["obfuscated_image"], "L")
        ss_gray.append(float(ssim_fn(og, bg, data_range=255)))

        n_uniq_o.append(len(np.unique(o.reshape(-1, 3), axis=0)))
        n_uniq_b.append(len(np.unique(b.reshape(-1, 3), axis=0)))

        # 원본 해상도 크기비 (resize 전)
        with Image.open(r["original_image"]) as im1, \
             Image.open(r["obfuscated_image"]) as im2:
            a1, a2 = im1.size[0] * im1.size[1], im2.size[0] * im2.size[1]
        ratios.append(a2 / a1)
        by_cat.setdefault(r["category"], []).append(ss_rgb[-1])

    print("\n" + "=" * 62)
    print("1. SSIM(원본, 난독화) — 낮을수록 난독화 신호가 강함")
    print("=" * 62)
    print(f"{'측정':22s} {'중앙값':>9s} {'평균':>9s} {'25분위':>9s} {'75분위':>9s}")
    for name, v in (("RGB (3채널)", ss_rgb),
                    ("  R 채널", ss_ch[0]),
                    ("  G 채널", ss_ch[1]),
                    ("  B 채널", ss_ch[2]),
                    ("Grayscale 변환후", ss_gray)):
        print(f"{name:22s} {st.median(v):9.4f} {st.mean(v):9.4f} "
              f"{np.percentile(v,25):9.4f} {np.percentile(v,75):9.4f}")

    print(f"\n  CICMalDroid (회색조 PNG) 기준값: 0.1244")

    print("\n" + "=" * 62)
    print("2. 판정")
    print("=" * 62)
    m = st.median(ss_rgb)
    print(f"  AndroDex RGB SSIM 중앙값: {m:.4f}")
    if m < 0.25:
        print(f"  => 신호 살아있음. JPG 그대로 진행 가능.")
        print(f"     단, RGB 3채널을 유지해야 함 (Grayscale 쓰면 손해:")
        print(f"     {st.median(ss_gray):.4f} 로 올라감 = 그만큼 뭉개짐)")
    elif m > 0.45:
        print(f"  => JPG 압축이 난독화 신호를 상당히 지웠습니다.")
        print(f"     binary 를 받아 회색조로 재렌더링하는 것을 권합니다.")
        print(f"     (figshare 23931477)")
    else:
        print(f"  => 중간. 신호는 있으나 CICMalDroid 보다 약합니다.")
        print(f"     JPG 로 진행하되 augmentation 효과가 작게 나올 수 있음을")
        print(f"     감안하거나, binary 재렌더링을 고려하세요.")

    print(f"\n  Grayscale 변환 손해: {st.median(ss_gray) - m:+.4f}")
    print(f"    (양수면 Grayscale 이 두 이미지를 더 비슷하게 만듦 = 신호 파괴)")

    print("\n" + "=" * 62)
    print("3. 값 분포 / 크기")
    print("=" * 62)
    print(f"  고유 색 개수  원본 {int(st.median(n_uniq_o))}, "
          f"난독화 {int(st.median(n_uniq_b))}  (256x256=65536 중)")
    print(f"  크기비(난독화/원본) 중앙값 {st.median(ratios):.4f}  "
          f"최소 {min(ratios):.3f}  최대 {max(ratios):.3f}")
    print(f"    (CICMalDroid 는 1.0000 — 난독화가 크기를 보존)")

    print("\n" + "=" * 62)
    print("4. category 별 RGB SSIM")
    print("=" * 62)
    for c in sorted(by_cat):
        v = by_cat[c]
        print(f"  {c:10s} n={len(v):4d}  중앙값 {st.median(v):.4f}")


if __name__ == "__main__":
    main()