#!/usr/bin/env python3
"""
make_splits_random.py -- 관행대로의 random split (대조군)

목적: split 구성만 바꾸고 나머지를 전부 동일하게 두어,
      기존 CICMalDroid 논문들의 높은 수치가 누출에서 온 것임을 통제 측정한다.

splits_final 과의 유일한 차이:
    splits_final  : content group 단위로 나눔. val/test 의 dex 가 train 에 없음
    splits_random : 행 단위로 무작위. 같은 dex 가 train 과 test 에 흩어짐  <- 관행

같은 것:
    - 같은 manifest, 같은 이미지, 같은 라벨
    - 라벨 모순 8그룹(175행) 제거 (오류이지 관행이 아님)
    - 같은 seed, 같은 비율, 같은 category 층화

따라서 두 조건의 성능 차이는 오직 split 구성에서 온다.

출력은 splits_final 과 같은 형식이므로 make_ssl_splits.py 가 그대로 돈다.

Usage:
    python make_splits_random.py \
        --manifest ~image_pairing_manifest.csv \
        --report ./audit_report.json --out_dir ./splits_random
"""

import argparse, csv, json, os, random, sys
from collections import Counter, defaultdict

LABELS = {"Adware": 0, "Banking": 1, "Benign": 2, "Riskware": 3, "SMS": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", default="./audit_report.json")
    ap.add_argument("--out_dir", default="./splits_random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- manifest ----
    rows = {}
    for r in csv.DictReader(open(args.manifest, newline="")):
        rows[r["hash"].strip()] = {
            "hash": r["hash"].strip(), "category": r["category"].strip(),
            "orig": r["original_image"], "obf": r["obfuscated_image"]}
    print(f"manifest: {len(rows)} 행")

    # ---- content group (누출 측정용. split 에는 안 씀) ----
    dup = json.load(open(args.report))["dup_content"]
    gid = {}
    for cid, hs in dup.items():
        for h in hs:
            gid[h] = cid
    for h in rows:
        gid.setdefault(h, f"solo:{h}")

    groups = defaultdict(list)
    for h, g in gid.items():
        groups[g].append(h)

    # ---- 라벨 모순 제거 (오류이지 관행이 아니므로 양쪽 조건에서 동일하게 제외) ----
    conflict = {g for g, hs in groups.items()
                if len({rows[h]["category"] for h in hs}) > 1}
    drop = {h for g in conflict for h in groups[g]}
    pool = [rows[h] for h in sorted(rows) if h not in drop]
    print(f"라벨 모순 {len(conflict)}그룹 ({len(drop)}행) 제거 -> {len(pool)}행\n")

    # ---- 행 단위 무작위 층화 분할  <- 이게 '관행' ----
    by_cat = defaultdict(list)
    for r in pool:
        by_cat[r["category"]].append(r)

    rng = random.Random(args.seed)
    sp = {"train": [], "val": [], "test": []}
    for c in sorted(by_cat):
        v = sorted(by_cat[c], key=lambda r: r["hash"])
        rng.shuffle(v)
        n = len(v); a = int(n * args.ratios[0]); b = a + int(n * args.ratios[1])
        sp["train"] += v[:a]; sp["val"] += v[a:b]; sp["test"] += v[b:]

    print("=" * 66)
    print(f"{'split':7s} {'행':>7s}  category")
    print("=" * 66)
    for k in ("train", "val", "test"):
        print(f"{k:7s} {len(sp[k]):7d}  {dict(Counter(r['category'] for r in sp[k]))}")

    # ---- 누출 측정: 이게 핵심 숫자다 ----
    print("\n" + "=" * 66)
    print("누출 프로파일 — 이 split 이 얼마나 오염됐는가")
    print("=" * 66)
    train_g = {gid[r["hash"]] for r in sp["train"]}
    for k in ("val", "test"):
        leaky = [r for r in sp[k] if gid[r["hash"]] in train_g]
        pct = 100 * len(leaky) / len(sp[k])
        print(f"  {k:5s}: {len(leaky):5d}/{len(sp[k])} ({pct:5.1f}%) 가 train 에 "
              f"byte-identical 쌍둥이를 가짐")
        if leaky:
            print(f"         category: {dict(Counter(r['category'] for r in leaky))}")

    # 내부 중복도 (평가 표본이 실제로 몇 개인가)
    for k in ("val", "test"):
        uniq = len({gid[r["hash"]] for r in sp[k]})
        print(f"  {k:5s}: {len(sp[k])}행이지만 고유 dex 는 {uniq}개 "
              f"({100*uniq/len(sp[k]):.1f}%)")

    print(f"\n  splits_final 은 위 두 수치가 각각 0% 와 100% 입니다.")
    print(f"  이 차이가 곧 기존 논문들이 보고한 수치의 부풀림 요인입니다.")

    # ---- 출력 ----
    print("\n" + "=" * 66)
    print(f"출력: {args.out_dir}")
    o = lambda n: os.path.join(args.out_dir, n)
    json.dump({k: sorted(r["hash"] for r in sp[k]) for k in sp},
              open(o("splits.json"), "w"), indent=2)
    print(f"  splits.json")
    for name, key, fn in (("train", "orig", "train_orig.txt"),
                          ("val", "orig", "val_orig.txt"),
                          ("test", "orig", "test_orig.txt"),
                          ("val", "obf", "val_obf.txt"),
                          ("test", "obf", "test_obf.txt")):
        with open(o(fn), "w") as f:
            for r in sp[name]:
                f.write(f"{r[key]} {LABELS[r['category']]}\n")
        print(f"  {fn:16s} ({len(sp[name])} lines)")


if __name__ == "__main__":
    main()