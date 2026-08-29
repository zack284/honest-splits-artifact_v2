#!/usr/bin/env python3
"""
audit_dims.py -- native 이미지 크기 분포와 RandomResizedCrop 손실률

PIL 은 헤더만 읽고 .size 를 주므로 전체 디코딩 없이 빠르다.

torchvision RandomResizedCrop(224, scale=(0.9,1.0), ratio=(3/4,4/3)) 는
10회 무작위 샘플에 실패하면 center crop 으로 fallback 한다.
길쭉한 이미지에서는 항상 fallback 이 걸리고, 대부분이 버려진다.
그 비율을 실측한다.

Usage:
    python audit_dims.py --splits ./splits_v2/splits.json \
                         --manifest image_pairing_manifest.csv
"""

import argparse, csv, json, math, statistics as st
from collections import Counter, defaultdict
from PIL import Image

SCALE = (0.9, 1.0)
RATIO = (3.0 / 4.0, 4.0 / 3.0)


def rrc_kept_fraction(W, H):
    """RandomResizedCrop 이 실제로 유지하는 픽셀 비율.

    무작위 시도가 한 번이라도 성공 가능한지 판정하고,
    불가능하면 torchvision 의 center-crop fallback 을 그대로 재현한다.
    """
    A = W * H
    # 최적 조건(면적 최소 0.9A, 비율은 범위 내 자유)에서도 들어가는지
    feasible = False
    for ar in (RATIO[0], 1.0, RATIO[1]):
        w = math.sqrt(SCALE[0] * A * ar)
        h = math.sqrt(SCALE[0] * A / ar)
        if w <= W and h <= H:
            feasible = True
            break
    if feasible:
        return 1.0, "random"          # 대략 0.9~1.0 유지

    in_ratio = W / H
    if in_ratio < RATIO[0]:
        w = W; h = int(round(w / RATIO[0]))
    elif in_ratio > RATIO[1]:
        h = H; w = int(round(h * RATIO[1]))
    else:
        w, h = W, H
    return (w * h) / A, "fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default="./splits_v2/splits.json")
    args = ap.parse_args()

    keep = set()
    for v in json.load(open(args.splits)).values():
        keep.update(v)
    print(f"대상: {len(keep)}장\n")

    rows = [r for r in csv.DictReader(open(args.manifest, newline=""))
            if r["hash"].strip() in keep]

    dims, kept, modes = [], [], Counter()
    by_cat = defaultdict(list)
    for i, r in enumerate(rows):
        if i % 2000 == 0:
            print(f"  ... {i}/{len(rows)}")
        with Image.open(r["original_image"]) as im:
            W, H = im.size
        f, mode = rrc_kept_fraction(W, H)
        dims.append((W, H)); kept.append(f); modes[mode] += 1
        by_cat[r["category"].strip()].append((W, H, f))

    Ws = [d[0] for d in dims]; Hs = [d[1] for d in dims]
    ars = [w / h for w, h in dims]

    print("\n" + "=" * 60)
    print("width 분포 (AndroDex 규칙)")
    print("=" * 60)
    for w, n in sorted(Counter(Ws).items()):
        print(f"  {w:5d} px : {n:6d}")

    print("\n" + "=" * 60)
    print("height / 종횡비")
    print("=" * 60)
    print(f"  height  min={min(Hs)}  median={int(st.median(Hs))}  max={max(Hs)}")
    print(f"  W/H     min={min(ars):.4f}  median={st.median(ars):.4f}  max={max(ars):.4f}")
    n_ok = sum(1 for a in ars if RATIO[0] <= a <= RATIO[1])
    print(f"  종횡비가 (0.75, 1.333) 안: {n_ok}/{len(ars)} ({100*n_ok/len(ars):.1f}%)")

    print("\n" + "=" * 60)
    print("RandomResizedCrop(224, scale=(0.9,1.0)) 실측")
    print("=" * 60)
    print(f"  무작위 성공: {modes['random']:6d}  ({100*modes['random']/len(rows):.1f}%)")
    print(f"  fallback   : {modes['fallback']:6d}  ({100*modes['fallback']/len(rows):.1f}%)")
    print(f"\n  평균 유지 픽셀 비율: {100*st.mean(kept):.1f}%")
    print(f"  중앙값             : {100*st.median(kept):.1f}%")
    fb = [k for k, m in zip(kept, [m for m in modes.elements()]) if True][:0]
    only_fb = [f for (w, h), f in zip(dims, kept) if f < 1.0]
    if only_fb:
        print(f"  fallback 건만: 평균 {100*st.mean(only_fb):.1f}%  "
              f"최악 {100*min(only_fb):.2f}%")

    print("\n" + "=" * 60)
    print("category 별")
    print("=" * 60)
    print(f"{'category':10s} {'n':>6s} {'median W':>9s} {'median H':>9s} {'유지율':>8s}")
    for c in sorted(by_cat):
        v = by_cat[c]
        print(f"{c:10s} {len(v):6d} {int(st.median([x[0] for x in v])):9d} "
              f"{int(st.median([x[1] for x in v])):9d} "
              f"{100*st.mean([x[2] for x in v]):7.1f}%")

    print(f"\n유지율이 낮으면: 현재 classifier 는 dex 대부분을 매번 버리고 있음.")
    print(f"256x256 전체 resize 는 손실이 아니라 개선.")


if __name__ == "__main__":
    main()