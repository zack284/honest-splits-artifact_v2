#!/usr/bin/env python3
"""
make_ssl_splits.py -- splits_final -> FixMatch 학습 스크립트가 읽는 txt 목록

기존 코드가 요구하는 것:
    --labeled_id_path     labeled 학습 목록
    --unlabeled_id_path   unlabeled 학습 목록
    --val_id_path         checkpoint 선택용

추가로 필요한 것:
    test_clean.txt        clean test  (990)
    test_obf.txt          fully obfuscated test (990, 같은 990개 APK)

설계:
    labeled/unlabeled 는 train 11,955 를 category 비율 유지하며 나눔.
      -> 같은 dex 가 양쪽에 걸쳐도 무방. 둘 다 학습에 쓰이므로 누출이 아님.
    val 은 val_orig + val_obf 합본 (1,974).
      -> clean 과 obfuscated 를 함께 보고 고르는 기준. baseline 에도 동일 적용.

Usage:
    python make_ssl_splits.py --splits_dir ./splits_final --out_dir ./splits_ssl \
                              --labeled_ratio 0.5
"""

import argparse, os, random
from collections import Counter


def read(p):
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]


def write(lines, p):
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {os.path.basename(p):22s} {len(lines):6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="./splits_final")
    ap.add_argument("--out_dir", default="./splits_ssl")
    ap.add_argument("--labeled_ratio", type=float, default=0.5,
                    help="train 중 labeled 비율. 나머지는 unlabeled")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    s = lambda n: os.path.join(args.splits_dir, n)
    o = lambda n: os.path.join(args.out_dir, n)

    train = read(s("train_orig.txt"))
    val_c = read(s("val_orig.txt"))
    val_o = read(s("val_obf.txt"))
    test_c = read(s("test_orig.txt"))
    test_o = read(s("test_obf.txt"))
    print(f"입력: train {len(train)}, val {len(val_c)}+{len(val_o)}, "
          f"test {len(test_c)}+{len(test_o)}\n")

    # ---- labeled / unlabeled: category 비율 유지 ----
    by_lab = {}
    for line in train:
        by_lab.setdefault(line.rsplit(" ", 1)[1], []).append(line)

    rng = random.Random(args.seed)
    lab, unlab = [], []
    for k in sorted(by_lab):
        v = sorted(by_lab[k]); rng.shuffle(v)
        n = int(len(v) * args.labeled_ratio)
        lab += v[:n]; unlab += v[n:]
    rng.shuffle(lab); rng.shuffle(unlab)

    cnt = lambda L: dict(Counter(x.rsplit(" ", 1)[1] for x in L))
    print(f"labeled   {len(lab):6d}  {cnt(lab)}")
    print(f"unlabeled {len(unlab):6d}  {cnt(unlab)}")
    print(f"  (0=Adware 1=Banking 2=Benign 3=Riskware 4=SMS)\n")

    print(f"출력: {args.out_dir}")
    write(lab,          o("train_labeled.txt"))
    write(unlab,        o("train_unlabeled.txt"))
    write(val_c + val_o, o("val_combined.txt"))   # checkpoint 선택 기준
    write(val_c,        o("val_clean.txt"))       # 참고용
    write(val_o,        o("val_obf.txt"))
    write(test_c,       o("test_clean.txt"))      # 헤드라인 A
    write(test_o,       o("test_obf.txt"))        # 헤드라인 B




if __name__ == "__main__":
    main()