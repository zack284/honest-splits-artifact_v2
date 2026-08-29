#!/usr/bin/env python3
"""
size_shift_under_obfuscation.py

크기만 쓰는 모델을 clean train 으로 학습한 뒤,
같은 990개 APK 의 clean test 와 obfuscated test 에 각각 걸어본다.

같은 앱이므로 정답은 동일하다. 달라지는 건 dex 크기뿐이다.
성능이 떨어진 만큼이 곧 "난독화가 크기 단서를 파괴한 양" 이다.

Usage:
    python size_shift_under_obfuscation.py \
        --manifest ~image_pairing_manifest.csv \
        --splits ./splits_v2/splits.json
"""

import argparse, csv, json, math
import statistics as st
from collections import defaultdict

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

LABELS = {"Adware": 0, "Banking": 1, "Benign": 2, "Riskware": 3, "SMS": 4}
NAMES = [k for k, _ in sorted(LABELS.items(), key=lambda x: x[1])]


def feat(W, H):
    return [math.log(W), math.log(H), math.log(W * H), W / H]


def dims(p):
    with Image.open(p) as im:
        return im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default="./splits_v2/splits.json")
    args = ap.parse_args()

    sp = json.load(open(args.splits))
    which = {h: k for k in sp for h in sp[k]}
    rows = [r for r in csv.DictReader(open(args.manifest, newline=""))
            if r["hash"].strip() in which]

    Xtr, ytr = [], []
    Xc, Xo, yte, cats = [], [], [], []
    areas = defaultdict(list)

    for i, r in enumerate(rows):
        if i % 2000 == 0:
            print(f"  ... {i}/{len(rows)}")
        k = which[r["hash"].strip()]
        lab = LABELS[r["category"].strip()]
        if k == "train":
            Xtr.append(feat(*dims(r["original_image"]))); ytr.append(lab)
        elif k == "test":
            Wc, Hc = dims(r["original_image"])
            Wo, Ho = dims(r["obfuscated_image"])
            Xc.append(feat(Wc, Hc)); Xo.append(feat(Wo, Ho))
            yte.append(lab); cats.append(r["category"].strip())
            areas[r["category"].strip()].append((Wc * Hc, Wo * Ho))

    Xtr, ytr = np.array(Xtr), np.array(ytr)
    Xc, Xo, yte = np.array(Xc), np.array(Xo), np.array(yte)
    print(f"\ntrain {len(ytr)} / test {len(yte)} (clean 과 obf 는 동일 APK)\n")

    m = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr)
    pc, po = m.predict(Xc), m.predict(Xo)

    ac, ao = accuracy_score(yte, pc), accuracy_score(yte, po)
    fc, fo = f1_score(yte, pc, average="macro"), f1_score(yte, po, average="macro")

    print("=" * 64)
    print("크기만 쓰는 모델 — clean 으로 학습, 두 test 에 적용")
    print("=" * 64)
    print(f"{'test':16s} {'정확도':>9s} {'macro F1':>10s}")
    print(f"{'clean':16s} {100*ac:8.1f}% {fc:10.3f}")
    print(f"{'obfuscated':16s} {100*ao:8.1f}% {fo:10.3f}")
    print(f"{'하락':16s} {100*(ac-ao):8.1f}p {fc-fo:10.3f}")

    print("\n" + "=" * 64)
    print("obfuscated test 상세")
    print("=" * 64)
    print(classification_report(yte, po, target_names=NAMES, digits=3, zero_division=0))

    print("=" * 64)
    print("난독화가 dex 크기를 얼마나 바꾸는가 (obf / clean 면적비)")
    print("=" * 64)
    print(f"{'category':10s} {'n':>5s} {'중앙값':>9s} {'최소':>8s} {'최대':>8s}")
    for c in sorted(areas):
        rr = [o / cl for cl, o in areas[c]]
        print(f"{c:10s} {len(rr):5d} {st.median(rr):9.3f} {min(rr):8.3f} {max(rr):8.3f}")

    allr = [o / cl for v in areas.values() for cl, o in v]
    print(f"\n전체 중앙값: {st.median(allr):.3f}  "
          f"(1.0 이면 크기 불변, >1 이면 커짐)")
    print(f"크기가 5% 이상 변한 샘플: "
          f"{sum(1 for x in allr if abs(x-1) > 0.05)}/{len(allr)}")


if __name__ == "__main__":
    main()