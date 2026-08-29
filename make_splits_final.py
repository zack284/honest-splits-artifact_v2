#!/usr/bin/env python3
"""
make_splits_final.py -- CICMalDroid 최종 split

전략:
  train  : 기존 관행대로 중복을 그대로 둔다 (그룹 전체를 통째로)
  val/test: dex 단위로 유일하게 (그룹당 대표 1개)
  불변식 : val/test 의 dex 가 train 에 단 하나도 없을 것

  누출은 train 과 test 가 '겹칠 때'만 생긴다. train 내부의 중복은
  해당 dex 를 가중 학습하는 것일 뿐 평가를 오염시키지 않는다.

  라벨 모순 그룹(같은 dex 에 다른 category)은 중복이 아니라 오류이므로 제거.

선행: audit_manifest.py (audit_report.json 생성)

Usage:
    python make_splits_final.py \
        --manifest ~image_pairing_manifest.csv \
        --report ./audit_report.json --out_dir ./splits_final
"""

import argparse, csv, hashlib, json, os, random, sys
from collections import Counter, defaultdict

LABELS = {"Adware": 0, "Banking": 1, "Benign": 2, "Riskware": 3, "SMS": 4}


def file_md5(path, chunk=1 << 20):
    m = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            m.update(b)
    return m.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", default="./audit_report.json")
    ap.add_argument("--out_dir", default="./splits_final")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--skip_verify", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1. manifest ----
    rows = {}
    for r in csv.DictReader(open(args.manifest, newline="")):
        rows[r["hash"].strip()] = {
            "hash": r["hash"].strip(), "category": r["category"].strip(),
            "orig": r["original_image"], "obf": r["obfuscated_image"]}
    print(f"manifest: {len(rows)} 행")

    # ---- 2. content group 구성 (중복 그룹 + 단독 샘플) ----
    dup = json.load(open(args.report))["dup_content"]
    gid = {}
    for cid, hs in dup.items():
        for h in hs:
            gid[h] = cid
    for h in rows:
        gid.setdefault(h, f"solo:{h}")     # 단독은 자기 자신이 그룹

    groups = defaultdict(list)
    for h, g in gid.items():
        groups[g].append(h)
    print(f"content group: {len(groups)}  (중복 그룹 {len(dup)}, 단독 {len(groups)-len(dup)})")

    # ---- 3. 라벨 모순 그룹 제거 ----
    conflict = [g for g, hs in groups.items()
                if len({rows[h]["category"] for h in hs}) > 1]
    n_cr = sum(len(groups[g]) for g in conflict)
    for g in conflict:
        del groups[g]
    print(f"라벨 모순 {len(conflict)}그룹 ({n_cr}행) 제거 -> {len(groups)}그룹\n")

    # ---- 4. 그룹 단위 stratified split ----
    by_cat = defaultdict(list)
    for g, hs in groups.items():
        by_cat[rows[hs[0]]["category"]].append(g)

    rng = random.Random(args.seed)
    gsp = {"train": [], "val": [], "test": []}
    for c in sorted(by_cat):
        gs = sorted(by_cat[c]); rng.shuffle(gs)
        n = len(gs); a = int(n * args.ratios[0]); b = a + int(n * args.ratios[1])
        gsp["train"] += gs[:a]; gsp["val"] += gs[a:b]; gsp["test"] += gs[b:]

    # train: 그룹 전체 / val,test: 대표 1개
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
    print(f"\nval/test 그룹의 중복 {dropped}행은 사용 안 함 (어디에도 안 넣음)")
    print(f"train 중복 배수: {len(sp['train'])/max(1,len(gsp['train'])):.2f}x")

    # ---- 5. 불변식 검증 ----
    print("\n" + "=" * 62)
    print("불변식: val/test 의 dex 가 train 에 없을 것")
    print("=" * 62)
    G = {k: set(gsp[k]) for k in gsp}
    for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(G[a_] & G[b_])
        print(f"[{'ok' if not n else 'FATAL'}] {a_} ∩ {b_} (content group) = {n}")
        if n:
            sys.exit(1)
    for k in ("val", "test"):
        hs = [r["hash"] for r in sp[k]]
        print(f"[{'ok' if len(hs)==len(set(hs)) else 'FATAL'}] {k} 내부 dex 유일: "
              f"{len(set(hs))}/{len(hs)}")

    # ---- 6. 실물 검증 ----
    if not args.skip_verify:
        print("\n" + "=" * 62)
        print("실물 검증: 이름이 아니라 파일 내용으로")
        print("=" * 62)
        allr = sp["train"] + sp["val"] + sp["test"]
        if not os.path.exists(allr[0]["orig"]):
            sys.exit(f"[FATAL] 파일 없음: {allr[0]['orig']}")
        cmd5, n_err = {}, 0
        for i, r in enumerate(allr):
            if i % 3000 == 0:
                print(f"  ... {i}/{len(allr)}")
            try:
                cmd5[r["hash"]] = file_md5(r["orig"])
            except Exception as e:
                n_err += 1; print(f"  !! {r['orig']} ({e})")
        if n_err:
            sys.exit(f"[FATAL] {n_err}건 읽기 실패. 부분 검사는 신뢰 불가.")
        print(f"  [ok] {len(allr)}장 전부 읽음")

        C = {k: {cmd5[r["hash"]] for r in sp[k]} for k in sp}
        for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
            n = len(C[a_] & C[b_])
            print(f"  [{'ok' if not n else 'FATAL'}] {a_} ∩ {b_} (실제 내용) = {n}")
            if n:
                sys.exit(1)
        for k in ("val", "test"):
            n = len(C[k])
            print(f"  [{'ok' if n==len(sp[k]) else 'FATAL'}] {k} 고유 이미지 "
                  f"{n} == 샘플 {len(sp[k])}")

    # ---- 7. 출력 ----
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


if __name__ == "__main__":
    main()