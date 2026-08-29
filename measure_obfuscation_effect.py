#!/usr/bin/env python3
"""
measure_obfuscation_effect.py

"난독화가 실제로 적용되긴 했나" 를 잰다.

핵심 아이디어: 이 이미지는 dex 바이트를 row-major 로 늘어놓은 것이므로,
이미지를 1D 로 펴면 그게 곧 dex 바이트 스트림이다.
따라서 resize 없이 바이트 단위로 직접 비교할 수 있다.

재는 것:
  1) byte-identical 쌍 개수   <- 0 이 아니면 난독화가 아무것도 안 한 것
  2) 공통 구간에서 다른 바이트 비율
  3) 첫 차이 위치 (dex 헤더는 112 바이트, checksum 포함 -> 반드시 바뀌어야 정상)
  4) 256x256 리사이즈 후 SSIM (pix2pix 가 보는 것과 동일)

test split 990 쌍만 본다.

Usage:
    python measure_obfuscation_effect.py \
        --manifest~image_pairing_manifest.csv \
        --splits ./splits_v2/splits.json
"""

import argparse, csv, json
import statistics as st
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False


def load_flat(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L")).reshape(-1)


def load_256(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((256, 256), Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default="./splits_v2/splits.json")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    sp = json.load(open(args.splits))
    target = set(sp[args.split])
    rows = [r for r in csv.DictReader(open(args.manifest, newline=""))
            if r["hash"].strip() in target]
    print(f"{args.split} split: {len(rows)} 쌍\n")

    n_ident = 0
    ident_list = []
    diffs, firsts, ssims = [], [], []
    by_cat = defaultdict(list)
    broken = []

    for i, r in enumerate(rows):
        if i % 200 == 0:
            print(f"  ... {i}/{len(rows)}")
        cat = r["category"].strip()
        a = load_flat(r["original_image"])
        b = load_flat(r["obfuscated_image"])

        if len(a) == len(b) and np.array_equal(a, b):
            n_ident += 1
            ident_list.append(r["hash"])
            diffs.append(0.0); firsts.append(-1)
            by_cat[cat].append(0.0)
            continue

        n = min(len(a), len(b))
        neq = a[:n] != b[:n]
        frac = float(neq.mean())
        diffs.append(frac)
        by_cat[cat].append(frac)
        firsts.append(int(np.argmax(neq)) if neq.any() else -1)

        ratio = len(b) / len(a)
        if ratio < 0.9:
            broken.append((r["hash"], cat, ratio, frac))

        if HAVE_SSIM:
            ssims.append(float(ssim_fn(load_256(r["original_image"]),
                                       load_256(r["obfuscated_image"]),
                                       data_range=255)))

    print("\n" + "=" * 64)
    print("1. 난독화가 아무것도 안 한 쌍")
    print("=" * 64)
    print(f"  byte-identical: {n_ident} / {len(rows)}   <- 0 이어야 정상")
    if n_ident:
        print(f"  예시: {ident_list[:5]}")
        print(f"  !! 이 샘플들은 Obfuscapk 가 성공을 반환했지만 dex 를 안 바꿨습니다.")

    print("\n" + "=" * 64)
    print("2. 바뀐 바이트 비율 (공통 구간)")
    print("=" * 64)
    nz = [d for d in diffs if d > 0]
    if nz:
        print(f"  중앙값 {100*st.median(nz):.1f}%   평균 {100*st.mean(nz):.1f}%")
        print(f"  최소   {100*min(nz):.2f}%   최대 {100*max(nz):.1f}%")
        for q in (10, 25, 50, 75, 90):
            print(f"    {q:2d}분위: {100*np.percentile(nz, q):.1f}%")

    print("\n" + "=" * 64)
    print("3. 첫 차이 위치 (dex 헤더 = 0~111, checksum 포함)")
    print("=" * 64)
    fv = [f for f in firsts if f >= 0]
    if fv:
        print(f"  헤더(<112) 안에서 시작: {sum(1 for f in fv if f < 112)}/{len(fv)}")
        print(f"  중앙값 위치: {int(st.median(fv))}")

    if HAVE_SSIM and ssims:
        print("\n" + "=" * 64)
        print("4. SSIM (256x256 nearest, pix2pix 가 보는 것)")
        print("=" * 64)
        print(f"  중앙값 {st.median(ssims):.4f}  평균 {st.mean(ssims):.4f}")
        print(f"  최소   {min(ssims):.4f}  최대 {max(ssims):.4f}")
    elif not HAVE_SSIM:
        print("\n  (SSIM 생략: pip install scikit-image)")

    print("\n" + "=" * 64)
    print("category 별 바뀐 바이트 비율")
    print("=" * 64)
    print(f"{'category':10s} {'n':>5s} {'중앙값':>9s} {'무변화':>8s}")
    for c in sorted(by_cat):
        v = by_cat[c]
        print(f"{c:10s} {len(v):5d} {100*st.median(v):8.1f}% "
              f"{sum(1 for x in v if x == 0):8d}")

    print("\n" + "=" * 64)
    print("깨진 난독화 (크기가 10% 이상 줄어든 것)")
    print("=" * 64)
    print(f"  {len(broken)}건")
    for h, c, ratio, frac in sorted(broken, key=lambda x: x[2])[:10]:
        print(f"    {c:9s} ratio={ratio:.3f}  바뀐비율={100*frac:.1f}%  {h[:40]}")


if __name__ == "__main__":
    main()