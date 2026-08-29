#!/usr/bin/env python3
"""
size_only_androdex.py -- 픽셀을 안 보고 크기만으로 분류 (AndroDex, 2-class)

CICMalDroid 에서 나온 결과:
    size-only   clean 74.6  ->  obf 68.1   (-5.7)   난독화가 크기 보존(1.000)
    ViT-ASPP    clean 75.6  ->  obf 33.4   (-42.2)
    => 이미지 모델이 크기 모델보다 8배 취약

AndroDex 는 난독화가 크기를 16% 늘린다(비율 1.164). 그런데도 ViT-ASPP 가
p=0 에서 78.9% 를 유지했다. 크기 신호가 살아남아 떠받친 것인지 확인한다.

핵심: 크기가 변하는 것 자체는 문제가 아니다. 대소 관계(순서)가 유지되면
      분류에는 여전히 쓸모가 있다.

clean 으로 학습해서 clean/obf 두 test 에 각각 평가한다. (동일 APK)

Usage:
    python size_only_androdex.py \
        --manifest ~Androdex/androdex_set2_manifest.csv \
        --splits ./splits_androdex/splits.json
"""

import argparse, csv, json, math
import statistics as st
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier

LABELS = {"benign": 0, "malware": 1}
NAMES = ["benign", "malware"]


def feat(W, H):
    return [math.log(W), math.log(H), math.log(W * H), W / H]


def dims(p):
    with Image.open(p) as im:
        return im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default="./splits_androdex/splits.json")
    args = ap.parse_args()

    sp = json.load(open(args.splits))
    which = {h: k for k in sp for h in sp[k]}
    print(f"train {len(sp['train'])} / val {len(sp['val'])} / test {len(sp['test'])}\n")

    rows = [r for r in csv.DictReader(open(args.manifest, newline=""))
            if r["hash"].strip() in which]

    Xtr, ytr = [], []
    Xc, Xo, yte = [], [], []
    areas = defaultdict(list)

    for i, r in enumerate(rows):
        if i % 1000 == 0:
            print(f"  ... {i}/{len(rows)}")
        k = which[r["hash"].strip()]
        lab = LABELS[r["category"].strip()]
        if k == "train":
            Xtr.append(feat(*dims(r["original_image"]))); ytr.append(lab)
        elif k == "test":
            Wc, Hc = dims(r["original_image"])
            Wo, Ho = dims(r["obfuscated_image"])
            Xc.append(feat(Wc, Hc)); Xo.append(feat(Wo, Ho)); yte.append(lab)
            areas[r["category"].strip()].append((Wc * Hc, Wo * Ho))

    Xtr, ytr = np.array(Xtr), np.array(ytr)
    Xc, Xo, yte = np.array(Xc), np.array(Xo), np.array(yte)
    print(f"\ntrain {len(ytr)} / test {len(yte)} (clean 과 obf 는 동일 APK)\n")

    models = [
        ("최빈 클래스 (하한)", DummyClassifier(strategy="most_frequent")),
        ("로지스틱 회귀", LogisticRegression(max_iter=2000)),
        ("결정트리 (depth 6)", DecisionTreeClassifier(max_depth=6, random_state=42)),
        ("랜덤포레스트", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ]

    print("=" * 68)
    print("크기(W, H)만 사용 — 픽셀 내용 0 바이트")
    print("=" * 68)
    print(f"{'모델':22s} {'clean acc':>10s} {'obf acc':>9s} {'하락':>8s} "
          f"{'clean F1':>9s} {'obf F1':>8s}")
    best = None
    for name, m in models:
        m.fit(Xtr, ytr)
        pc, po = m.predict(Xc), m.predict(Xo)
        ac, ao = accuracy_score(yte, pc), accuracy_score(yte, po)
        fc = f1_score(yte, pc, average="macro")
        fo = f1_score(yte, po, average="macro")
        print(f"{name:22s} {100*ac:9.1f}% {100*ao:8.1f}% {100*(ac-ao):7.1f}p "
              f"{fc:9.3f} {fo:8.3f}")
        if best is None or ac > best[1]:
            best = (name, ac, ao, pc, po)

    print("\n" + "=" * 68)
    print(f"최고 모델: {best[0]}   clean 상세")
    print("=" * 68)
    print(classification_report(yte, best[3], target_names=NAMES, digits=3,
                                zero_division=0))
    print(f"obf 상세")
    print(classification_report(yte, best[4], target_names=NAMES, digits=3,
                                zero_division=0))

    print("=" * 68)
    print("난독화가 크기를 얼마나 바꾸는가 (obf/clean 면적비)")
    print("=" * 68)
    print(f"{'category':10s} {'n':>5s} {'중앙값':>9s} {'최소':>8s} {'최대':>8s}")
    for c in sorted(areas):
        rr = [o / cl for cl, o in areas[c]]
        print(f"{c:10s} {len(rr):5d} {st.median(rr):9.3f} {min(rr):8.3f} {max(rr):8.3f}")

    # 대소 관계(순서)가 유지되는가 — 크기 신호가 살아남는 진짜 조건
    print("\n" + "=" * 68)
    print("클래스 간 크기 분리도 (중앙 면적)")
    print("=" * 68)
    for tag, idx in (("clean", 0), ("obf", 1)):
        med = {c: st.median([x[idx] for x in v]) for c, v in areas.items()}
        b, m = med.get("benign", 0), med.get("malware", 1)
        print(f"  {tag:5s}  benign {int(b):9d}   malware {int(m):9d}   "
              f"비율 {b/max(1,m):6.2f}x")
    print(f"\n  비율이 clean 과 obf 에서 비슷하면 크기 신호가 난독화를 통과한 것.")

    floor = max(Counter(yte).values()) / len(yte)
    print("\n" + "=" * 68)
    print("해석")
    print("=" * 68)
    print(f"  최빈 클래스 하한 : {100*floor:.1f}%")
    print(f"  크기만으로 clean : {100*best[1]:.1f}%")
    print(f"  크기만으로 obf   : {100*best[2]:.1f}%")
    print(f"\n  참고 CICMalDroid: clean 74.6 -> obf 68.1 (-5.7), 하한 34.5")
    print(f"  참고 AndroDex ViT-ASPP p=0: clean 87.6 -> obf 78.9 (-8.7)")


if __name__ == "__main__":
    main()