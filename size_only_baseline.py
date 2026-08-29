#!/usr/bin/env python3
"""
size_only_baseline.py -- 픽셀을 전혀 안 보고 크기만으로 5-class 분류

이미지의 (width, height) 두 숫자만 쓴다. 내용은 한 바이트도 안 본다.
area = W*H 는 사실상 dex 바이트 수다.

이 정확도가 높으면: 지금까지의 malware image classifier 성적 중
상당 부분이 "코드를 읽어서"가 아니라 "파일이 크니까 Benign"으로
맞춘 것일 수 있다. 리뷰어가 물어볼 질문이기도 하다.

splits_v2 의 train 으로 학습하고 test 로 평가한다 (ViT-ASPP 와 동일 조건).

Usage:
    python size_only_baseline.py --manifest ~/image_pairing_manifest.csv \
                                 --splits ./splits_v2/splits.json
"""

import argparse, csv, json, math
from collections import Counter

import numpy as np
from PIL import Image

try:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    raise SystemExit("sklearn 이 필요합니다:  pip install scikit-learn")

LABELS = {"Adware": 0, "Banking": 1, "Benign": 2, "Riskware": 3, "SMS": 4}
NAMES = [k for k, _ in sorted(LABELS.items(), key=lambda x: x[1])]


def featurize(W, H):
    A = W * H
    return [math.log(W), math.log(H), math.log(A), W / H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default="./splits_v2/splits.json")
    args = ap.parse_args()

    sp = json.load(open(args.splits))
    which = {h: k for k in sp for h in sp[k]}
    print(f"train {len(sp['train'])} / val {len(sp['val'])} / test {len(sp['test'])}\n")

    rows = [r for r in csv.DictReader(open(args.manifest, newline=""))
            if r["hash"].strip() in which]

    X = {"train": [], "val": [], "test": []}
    y = {"train": [], "val": [], "test": []}
    for i, r in enumerate(rows):
        if i % 2000 == 0:
            print(f"  ... {i}/{len(rows)}")
        with Image.open(r["original_image"]) as im:
            W, H = im.size
        k = which[r["hash"].strip()]
        X[k].append(featurize(W, H))
        y[k].append(LABELS[r["category"].strip()])

    for k in X:
        X[k] = np.array(X[k]); y[k] = np.array(y[k])

    sc = StandardScaler().fit(X["train"])
    Xtr, Xte = sc.transform(X["train"]), sc.transform(X["test"])

    models = [
        ("최빈 클래스 (하한)", DummyClassifier(strategy="most_frequent")),
        ("로지스틱 회귀",       LogisticRegression(max_iter=2000, multi_class="multinomial")),
        ("결정트리 (depth 6)",  DecisionTreeClassifier(max_depth=6, random_state=42)),
        ("랜덤포레스트",        RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ]

    print("\n" + "=" * 64)
    print("크기(W, H)만 사용 — 픽셀 내용 0 바이트")
    print("=" * 64)
    print(f"{'모델':22s} {'정확도':>9s} {'macro F1':>10s}")
    best = None
    for name, m in models:
        m.fit(Xtr, y["train"])
        p = m.predict(Xte)
        acc = accuracy_score(y["test"], p)
        f1 = f1_score(y["test"], p, average="macro")
        print(f"{name:22s} {100*acc:8.1f}% {f1:10.3f}")
        if best is None or acc > best[1]:
            best = (name, acc, p)

    print("\n" + "=" * 64)
    print(f"최고 모델 상세: {best[0]}")
    print("=" * 64)
    print(classification_report(y["test"], best[2], target_names=NAMES, digits=3))

    print("=" * 64)
    print("해석")
    print("=" * 64)
    floor = max(Counter(y["test"]).values()) / len(y["test"])
    print(f"  최빈 클래스 하한 : {100*floor:.1f}%")
    print(f"  크기만으로       : {100*best[1]:.1f}%")
    print(f"\n  이 수치가 높을수록, 이미지 분류기의 성적 중")
    print(f"  '코드를 읽어서'가 아니라 '크기를 보고' 맞춘 몫이 큽니다.")


if __name__ == "__main__":
    main()