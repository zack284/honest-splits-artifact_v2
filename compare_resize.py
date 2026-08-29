#!/usr/bin/env python3
"""
compare_resize.py -- nearest vs bilinear: 난독화 신호가 살아남는가

주장: bilinear 은 다운샘플 시 이웃 바이트를 평균한다.
      1024x5699 -> 256x256 이면 출력 1픽셀이 원본 88바이트의 평균이다.
      임의 바이트 88개의 평균은 원본이든 난독화본이든 128 근처로 수렴한다.
      -> 평균이 차이를 지운다.

nearest 는 88개 중 하나를 뽑는다. 98.9% 를 버리지만
뽑힌 값은 실제 바이트이고 원본/난독화본에서 다르다.

예측: SSIM(x256, x'256) 이 bilinear 에서 크게 높다.
      = 두 이미지가 비슷해 보인다
      = 살리려는 신호가 지워졌다

또한 "없는 값" 을 정량화한다:
      출력 픽셀값 중 원본 dex 에 실제로 존재하는 값의 비율.
      nearest = 100% (부분집합). bilinear = ?

Usage:
    python compare_resize.py \
        --manifest ~mage_pairing_manifest.csv \
        --splits ./splits_final/splits.json --n 300
"""

import argparse, csv, json, random
import statistics as st

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

METHODS = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR}
S = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default="./splits_final/splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    target = set(json.load(open(args.splits))[args.split])
    rows = [r for r in csv.DictReader(open(args.manifest, newline=""))
            if r["hash"].strip() in target]
    random.Random(42).shuffle(rows)
    rows = rows[:args.n]
    print(f"{args.split} 에서 {len(rows)} 쌍 표본\n")

    res = {m: {"ssim": [], "exist": [], "nuniq": []} for m in METHODS}
    scales = []

    for i, r in enumerate(rows):
        if i % 50 == 0:
            print(f"  ... {i}/{len(rows)}")
        with Image.open(r["original_image"]) as im:
            a = im.convert("L")
            W, H = a.size
            src_vals = set(np.unique(np.asarray(a)).tolist())
            scales.append((W * H) / (S * S))
            with Image.open(r["obfuscated_image"]) as im2:
                b = im2.convert("L")
                for name, flt in METHODS.items():
                    x = np.asarray(a.resize((S, S), flt))
                    y = np.asarray(b.resize((S, S), flt))
                    res[name]["ssim"].append(
                        float(ssim_fn(x, y, data_range=255)))
                    out_vals = set(np.unique(x).tolist())
                    res[name]["exist"].append(
                        len(out_vals & src_vals) / max(1, len(out_vals)))
                    res[name]["nuniq"].append(len(out_vals))

    print("\n" + "=" * 64)
    print(f"압축률: 출력 1픽셀당 원본 {st.median(scales):.1f} 바이트 (중앙값)")
    print("=" * 64)

    print("\n" + "=" * 64)
    print("1. SSIM(원본256, 난독화256) — 낮을수록 난독화 신호가 살아있음")
    print("=" * 64)
    print(f"{'방법':10s} {'중앙값':>9s} {'평균':>9s} {'25분위':>9s} {'75분위':>9s}")
    for m in METHODS:
        v = res[m]["ssim"]
        print(f"{m:10s} {st.median(v):9.4f} {st.mean(v):9.4f} "
              f"{np.percentile(v,25):9.4f} {np.percentile(v,75):9.4f}")

    sn, sb = st.median(res["nearest"]["ssim"]), st.median(res["bilinear"]["ssim"])
    print(f"\n  bilinear 가 {sb - sn:+.4f} 높음")
    if sb > sn + 0.05:
        print(f"  => bilinear 는 원본과 난독화본을 더 비슷하게 만듭니다.")
        print(f"     평균화가 차이를 지웠습니다. nearest 가 맞습니다.")
    elif sn > sb + 0.05:
        print(f"  => 예상과 반대입니다. nearest 쪽이 더 비슷해졌습니다.")
    else:
        print(f"  => 차이가 미미합니다. 신호 보존 관점에선 동등.")

    print("\n" + "=" * 64)
    print("2. '없는 값' — 출력 픽셀값 중 원본 dex 에 실재하는 값의 비율")
    print("=" * 64)
    print(f"{'방법':10s} {'실재비율':>10s} {'고유값수':>10s}")
    for m in METHODS:
        print(f"{m:10s} {100*st.mean(res[m]['exist']):9.1f}% "
              f"{st.median(res[m]['nuniq']):10.0f}")
    print(f"\n  nearest 는 정의상 100% (부분집합).")
    print(f"  bilinear 가 100% 미만이면 dex 에 없던 바이트를 만들어낸 것.")


if __name__ == "__main__":
    main()