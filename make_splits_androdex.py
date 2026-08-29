#!/usr/bin/env python3
"""
make_splits_androdex.py -- AndroDex Set2 split (2-class)

CICMalDroid 용 make_splits_final.py 와 두 곳이 다르다:
  1. LABELS   : benign/malware 2-class
  2. 실물검증 : JPG 이므로 파일 바이트가 아니라 디코딩된 픽셀을 해싱

전략은 동일:
  train   : 관행대로 중복 유지 (그룹 전체를 통째로)
  val/test: dex 단위로 유일 (그룹당 대표 1개)
  불변식  : val/test 의 이미지가 train 에 단 하나도 없을 것

선행: audit_androdex.py 로 androdex_audit_report.json 생성

Usage:
    python make_splits_androdex.py \
        --manifest ~/Androdex/androdex_set2_manifest.csv \
        --report ./androdex_audit_report.json \
        --out_dir ./splits_androdex
"""

import argparse, csv, hashlib, json, os, random, sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

LABELS = {"benign": 0, "malware": 1}


def content_md5(path):
    """JPG 메타데이터에 영향받지 않도록 디코딩된 픽셀을 해싱."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"))
    h = hashlib.md5()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", default="./androdex_audit_report.json")
    ap.add_argument("--out_dir", default="./splits_androdex")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--skip_verify", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- manifest ----
    rows = {}
    for r in csv.DictReader(open(args.manifest, newline="")):
        c = r["category"].strip()
        if c not in LABELS:
            sys.exit(f"[FATAL] 알 수 없는 category: {c} (허용: {sorted(LABELS)})")
        rows[r["hash"].strip()] = {
            "hash": r["hash"].strip(), "category": c,
            "orig": r["original_image"], "obf": r["obfuscated_image"]}
    print(f"manifest: {len(rows)} 쌍")

    # ---- content group ----
    rep = json.load(open(args.report))
    dup = rep["dup_content"]
    gid = {}
    for cid, hs in dup.items():
        for h in hs:
            gid[h] = cid
    for h in rows:
        gid.setdefault(h, f"solo:{h}")

    groups = defaultdict(list)
    for h, g in gid.items():
        if h in rows:
            groups[g].append(h)
    print(f"content group: {len(groups)}  (중복 {len(dup)}, 단독 {len(groups)-len(dup)})")

    # ---- 라벨 모순 제거 (AndroDex 는 0건이었지만 방어적으로) ----
    conflict = [g for g, hs in groups.items()
                if len({rows[h]["category"] for h in hs}) > 1]
    n_cr = sum(len(groups[g]) for g in conflict)
    for g in conflict:
        del groups[g]
    if conflict:
        print(f"라벨 모순 {len(conflict)}그룹 ({n_cr}행) 제거")
    else:
        print(f"라벨 모순 없음")
    print()

    # ---- 그룹 단위 stratified split ----
    by_cat = defaultdict(list)
    for g, hs in groups.items():
        by_cat[rows[hs[0]]["category"]].append(g)

    rng = random.Random(args.seed)
    gsp = {"train": [], "val": [], "test": []}
    for c in sorted(by_cat):
        gs = sorted(by_cat[c]); rng.shuffle(gs)
        n = len(gs); a = int(n * args.ratios[0]); b = a + int(n * args.ratios[1])
        gsp["train"] += gs[:a]; gsp["val"] += gs[a:b]; gsp["test"] += gs[b:]

    sp = {"train": [], "val": [], "test": []}
    for g in gsp["train"]:
        sp["train"] += [rows[h] for h in sorted(groups[g])]
    for k in ("val", "test"):
        sp[k] = [rows[sorted(groups[g])[0]] for g in gsp[k]]

    print("=" * 62)
    print(f"{'split':7s} {'그룹':>7s} {'샘플':>7s}  category")
    print("=" * 62)
    for k in ("train", "val", "test"):
        print(f"{k:7s} {len(gsp[k]):7d} {len(sp[k]):7d}  "
              f"{dict(Counter(r['category'] for r in sp[k]))}")
    dropped = sum(len(groups[g]) - 1 for g in gsp["val"] + gsp["test"])
    print(f"\nval/test 그룹의 중복 {dropped}행은 사용 안 함")
    print(f"train 중복 배수: {len(sp['train'])/max(1,len(gsp['train'])):.2f}x")

    # train 과 val/test 의 클래스 분포 차이 (중복 유지의 대가)
    for k in ("train", "val"):
        c = Counter(r["category"] for r in sp[k])
        tot = sum(c.values())
        print(f"  {k:5s} malware 비율 {100*c['malware']/tot:.1f}%")

    # ---- 불변식 ----
    print("\n" + "=" * 62)
    print("불변식: val/test 의 이미지가 train 에 없을 것")
    print("=" * 62)
    G = {k: set(gsp[k]) for k in gsp}
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(G[a_] & G[b_])
        print(f"[{'ok' if not n else 'FATAL'}] {a_} ∩ {b_} (content group) = {n}")
        if n:
            sys.exit(1)
    for k in ("val", "test"):
        hs = [r["hash"] for r in sp[k]]
        ok = len(hs) == len(set(hs))
        print(f"[{'ok' if ok else 'FATAL'}] {k} 내부 hash 유일: {len(set(hs))}/{len(hs)}")
        if not ok:
            sys.exit(1)

    # ---- 실물 검증 (픽셀 해시) ----
    if not args.skip_verify:
        print("\n" + "=" * 62)
        print("실물 검증: 디코딩된 픽셀로")
        print("=" * 62)
        allr = sp["train"] + sp["val"] + sp["test"]
        if not os.path.exists(allr[0]["orig"]):
            sys.exit(f"[FATAL] 파일 없음: {allr[0]['orig']}")
        cmd5, n_err = {}, 0
        for i, r in enumerate(allr):
            if i % 1000 == 0:
                print(f"  ... {i}/{len(allr)}")
            try:
                cmd5[r["hash"]] = content_md5(r["orig"])
            except Exception as e:
                n_err += 1; print(f"  !! {r['orig']} ({e})")
        if n_err:
            sys.exit(f"[FATAL] {n_err}건 읽기 실패. 부분 검사는 신뢰 불가.")
        print(f"  [ok] {len(allr)}장 전부 읽음")

        C = {k: {cmd5[r["hash"]] for r in sp[k]} for k in sp}
        for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
            n = len(C[a_] & C[b_])
            print(f"  [{'ok' if not n else 'FATAL'}] {a_} ∩ {b_} (실제 픽셀) = {n}")
            if n:
                sys.exit(1)
        for k in ("val", "test"):
            n = len(C[k])
            print(f"  [{'ok' if n==len(sp[k]) else 'FATAL'}] {k} 고유 이미지 "
                  f"{n} == 샘플 {len(sp[k])}")

    # ---- 출력 ----
    print("\n" + "=" * 62)
    o = lambda n: os.path.join(args.out_dir, n)
    json.dump({k: sorted(r["hash"] for r in sp[k]) for k in sp},
              open(o("splits.json"), "w"), indent=2)
    print(f"  splits.json  <- 정본")
    for name, key, fn in (("train", "orig", "train_orig.txt"),
                          ("val", "orig", "val_orig.txt"),
                          ("test", "orig", "test_orig.txt"),
                          ("val", "obf", "val_obf.txt"),
                          ("test", "obf", "test_obf.txt")):
        with open(o(fn), "w") as f:
            for r in sp[name]:
                f.write(f"{r[key]} {LABELS[r['category']]}\n")
        print(f"  {fn:16s} ({len(sp[name])} lines)")

    print(f"\n다음:")
    print(f"  python make_ssl_splits.py --splits_dir {args.out_dir} "
          f"--out_dir ./splits_ssl_androdex")
    print(f"  * --nclass 2 로 학습할 것 (AndroDex 는 2-class)")
    print(f"  * AndroDex 는 RGB JPG. transform 의 Grayscale 처리 확인 필요")


if __name__ == "__main__":
    main()