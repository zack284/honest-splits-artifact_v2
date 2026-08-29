#!/usr/bin/env python3
"""
check_obfuscation_determinism.py

핵심 질문: 난독화는 결정론적 함수인가, 확률적 사상인가?

같은 dex 를 Obfuscapk 로 여러 번 돌린 결과가 이미 데이터에 있다.
중복 그룹(같은 원본 dex, 각각 별도 난독화)이 그것이다.
그 그룹 안에서 난독화본끼리 비교하면 답이 나온다.

    SSIM(x'_i, x'_j)  = 같은 입력에 대한 서로 다른 두 난독화 출력의 유사도

이 값이 어떤 결정론적 generator 의 '천장' 이다.
정답들이 서로 안 닮았다면, G 가 그 중 무엇을 맞혀도 나머지와는 틀린다.
L1 loss 는 그럴 때 E[x'|x] 로 수렴한다 = 흐릿한 평균.

참고값 (CICMalDroid test, 256x256 nearest):
    SSIM(x, x')      = 0.1837   항등 baseline
    SSIM(G(x), x')   = 0.2037   학습된 generator
    SSIM(G(x), x)    = 0.1562

판정:
    SSIM(x'_i, x'_j) ~ 0.2  -> 천장이 0.2. G 는 이미 도달. 과제가 ill-posed
    SSIM(x'_i, x'_j) ~ 0.9  -> 난독화는 결정론적. G 가 못 배운 것. 구조/손실 문제

Usage:
    python check_obfuscation_determinism.py \
        --manifest ~image_pairing_manifest.csv \
        --report ./audit_report.json --max_groups 60
"""

import argparse, csv, json, random
import statistics as st
from itertools import combinations

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

S = 256


def load256(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((S, S), Image.NEAREST))


def flat(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L")).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", default="./audit_report.json")
    ap.add_argument("--max_groups", type=int, default=60)
    ap.add_argument("--max_pairs_per_group", type=int, default=6)
    args = ap.parse_args()

    rows = {}
    for r in csv.DictReader(open(args.manifest, newline="")):
        rows[r["hash"].strip()] = (r["original_image"], r["obfuscated_image"],
                                   r["category"].strip())

    dup = json.load(open(args.report))["dup_content"]
    groups = [(c, hs) for c, hs in dup.items() if len(hs) >= 2]
    groups.sort(key=lambda x: -len(x[1]))
    print(f"중복 그룹 {len(groups)}개 (최대 크기 {len(groups[0][1])})\n")

    rng = random.Random(42)
    sel = groups[:5] + rng.sample(groups[5:], min(args.max_groups - 5,
                                                  len(groups) - 5))

    ss_oo, ss_xo, sizes = [], [], []
    byte_oo = []
    for gi, (cid, hs) in enumerate(sel):
        print(f"  ... 그룹 {gi+1}/{len(sel)} (크기 {len(hs)})")
        hs = [h for h in hs if h in rows]
        if len(hs) < 2:
            continue
        sizes.append(len(hs))

        # 원본은 정의상 동일 -> 아무거나
        x = load256(rows[hs[0]][0])

        pairs = list(combinations(hs, 2))
        rng.shuffle(pairs)
        for h1, h2 in pairs[:args.max_pairs_per_group]:
            o1, o2 = load256(rows[h1][1]), load256(rows[h2][1])
            ss_oo.append(float(ssim_fn(o1, o2, data_range=255)))
            # 바이트 수준도
            f1, f2 = flat(rows[h1][1]), flat(rows[h2][1])
            n = min(len(f1), len(f2))
            byte_oo.append(float((f1[:n] != f2[:n]).mean()))
        for h in hs[:3]:
            ss_xo.append(float(ssim_fn(x, load256(rows[h][1]), data_range=255)))

    print("\n" + "=" * 66)
    print("같은 dex, 서로 다른 난독화 결과끼리")
    print("=" * 66)
    print(f"  비교한 쌍: {len(ss_oo)}   (그룹 {len(sizes)}개, 크기 중앙값 {int(st.median(sizes))})")
    print(f"\n  SSIM(x'_i, x'_j)  중앙값 {st.median(ss_oo):.4f}  평균 {st.mean(ss_oo):.4f}")
    print(f"                    25분위 {np.percentile(ss_oo,25):.4f}  "
          f"75분위 {np.percentile(ss_oo,75):.4f}")
    print(f"  바뀐 바이트 비율   중앙값 {100*st.median(byte_oo):.1f}%")
    print(f"\n  SSIM(x, x')       중앙값 {st.median(ss_xo):.4f}  (이 그룹 기준 항등)")

    print("\n" + "=" * 66)
    print("판정")
    print("=" * 66)
    m = st.median(ss_oo)
    print(f"  {'천장 SSIM(x_i, x_j)':28s} {m:.4f}")
    print(f"  {'학습된 G: SSIM(G(x), x)':28s} 0.2037")
    print(f"  {'항등:     SSIM(x, x)':28s} 0.1837")
    print()
    if m < 0.35:
        print("  => 난독화는 확률적입니다. 같은 입력에 정답이 여러 개고 서로 안 닮았습니다.")
        print("     어떤 결정론적 G 도 이 천장을 못 넘습니다.")
        print("     L1 은 E[x'|x] 로 수렴하고 그건 흐릿한 평균입니다.")
        print("     -> pix2pix 로 난독화를 학습한다는 전제 자체가 성립하지 않습니다.")
    elif m > 0.7:
        print("  => 난독화는 거의 결정론적입니다. 정답이 하나로 잘 정의됩니다.")
        print("     그런데 G 가 0.20 에 머물렀다면 학습 실패이지 과제 문제가 아닙니다.")
        print("     -> 구조/손실/해상도를 다시 봐야 합니다.")
    else:
        print("  => 중간입니다. 일부는 결정론적, 일부는 무작위.")
        print("     G 의 0.2037 이 천장 대비 어디인지로 판단하세요.")


if __name__ == "__main__":
    main()