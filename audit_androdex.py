#!/usr/bin/env python3
"""
audit_androdex.py -- AndroDex manifest 감사 (중복 / 라벨모순)

CICMalDroid 용 audit_manifest.py 와 두 곳이 다르다:

  1. hash 규칙
     CICMalDroid: 97/68/36 자 세 형식 (sha256.md5 / sha256.apk / md5.apk)
     AndroDex   : 순수 64자 sha256

  2. 내용 해시
     CICMalDroid: PNG 무손실 -> 파일 바이트 해시로 충분
     AndroDex   : JPG. 메타데이터(타임스탬프 등)가 다르면
                  픽셀이 같아도 파일 해시가 달라진다 -> 중복을 놓친다.
                  따라서 디코딩된 픽셀 배열을 해싱한다.

배경: AndroDex 는 Drebin + Kronodroid + Androzoo 합본이다.
      Irolla & Dey (2018) 가 Drebin 에서 opcode 수준 중복 49.35% 를 보고했다.
      CICMalDroid 는 34.6% 였다. 여기는 더 심할 수 있다.

Usage:
    python audit_androdex.py \
        --manifest androdex_set2_manifest.csv
"""

import argparse, csv, hashlib, json, os, sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image


def content_md5(path):
    """디코딩된 픽셀을 해싱. JPG 메타데이터 차이에 영향받지 않는다."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"))
    h = hashlib.md5()
    h.update(str(a.shape).encode())      # 크기가 다르면 다른 이미지
    h.update(a.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="./androdex_audit_report.json")
    ap.add_argument("--skip_content", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    print(f"manifest: {len(rows)} 쌍\n")

    # ---------- 1. hash 형식 ----------
    print("=" * 62)
    print("1. hash 형식")
    print("=" * 62)
    HEX = set("0123456789abcdefABCDEF")
    lens = Counter(len(r["hash"]) for r in rows)
    bad = [r["hash"] for r in rows
           if len(r["hash"]) != 64 or not all(c in HEX for c in r["hash"])]
    print(f"  길이 분포: {dict(lens)}   (64 = sha256)")
    print(f"  sha256 아님: {len(bad)}")
    if bad:
        print(f"    예시: {bad[:3]}")

    uniq = len({r["hash"] for r in rows})
    print(f"  고유 hash: {uniq} / {len(rows)}")
    if uniq != len(rows):
        c = Counter(r["hash"] for r in rows)
        dups = [h for h, n in c.items() if n > 1]
        print(f"  [!] hash 중복 {len(dups)}건 — 카테고리를 넘나들 가능성")
        for h in dups[:3]:
            cats = [r["category"] for r in rows if r["hash"] == h]
            print(f"      {h[:20]}... -> {cats}")

    print(f"\n  category: {dict(Counter(r['category'] for r in rows))}")

    if args.skip_content:
        print("\n[--skip_content] 내용 검사 생략")
        return

    # ---------- 2. 내용 중복 (원본 기준) ----------
    print("\n" + "=" * 62)
    print("2. 내용 중복 — 픽셀 해시 (JPG 메타데이터 무관)")
    print("=" * 62)

    probe = rows[0]["original_image"]
    if not os.path.exists(probe):
        sys.exit(f"[FATAL] 첫 이미지 없음: {probe}")

    by_content = defaultdict(list)
    n_err = 0
    for i, r in enumerate(rows):
        if i % 500 == 0:
            print(f"  ... {i}/{len(rows)}")
        try:
            by_content[content_md5(r["original_image"])].append(r["hash"])
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f"  !! 읽기 실패: {r['original_image']} ({e})")

    if n_err:
        sys.exit(f"\n[FATAL] {n_err}/{len(rows)} 읽기 실패. "
                 f"부분 결과는 신뢰할 수 없으므로 중단합니다.")
    if not by_content:
        sys.exit("\n[FATAL] 읽은 이미지 0장. 검사가 수행되지 않았습니다.")
    print(f"  [ok] {len(rows)}장 전부 읽음 — 검사가 실제로 수행됨")

    dup = {k: v for k, v in by_content.items() if len(v) > 1}
    n_extra = sum(len(v) - 1 for v in dup.values())
    pct = 100 * n_extra / len(rows)

    print(f"\n  고유 이미지 : {len(by_content)} / {len(rows)}")
    print(f"  중복 그룹   : {len(dup)}")
    print(f"  초과 항목   : {n_extra}  ({pct:.1f}%)")
    print(f"\n  참고: CICMalDroid 34.6%,  Drebin(Irolla&Dey 2018) 49.35%")

    if dup:
        sizes = Counter(len(v) for v in dup.values())
        print(f"  그룹 크기 분포(상위): {dict(sorted(sizes.items())[:8])}"
              f"  최대 {max(len(v) for v in dup.values())}")

    # ---------- 3. 라벨 모순 ----------
    print("\n" + "=" * 62)
    print("3. 라벨 모순 — 같은 이미지가 benign 과 malware 양쪽에")
    print("=" * 62)
    cat = {r["hash"]: r["category"] for r in rows}
    conflict = {}
    for cmd5, hs in dup.items():
        cs = {cat[h] for h in hs if h in cat}
        if len(cs) > 1:
            conflict[cmd5] = hs

    n_conf_rows = sum(len(v) for v in conflict.values())
    print(f"  모순 그룹: {len(conflict)}  (해당 {n_conf_rows}행)")
    if conflict:
        print(f"  [!] 악성/정상 경계가 무너진 샘플입니다. 반드시 제거하세요.")
        for cmd5, hs in list(conflict.items())[:3]:
            print(f"    {[f'{cat[h]}:{h[:16]}...' for h in hs[:4]]}")
    else:
        print(f"  [ok] 없음. 악성/정상 경계는 온전합니다.")

    # ---------- 4. 정리 후 예상 ----------
    print("\n" + "=" * 62)
    print("4. 정리 후 예상 (모순 전체제거 + 중복 대표1개)")
    print("=" * 62)
    drop = set()
    for hs in conflict.values():
        drop.update(hs)
    for cmd5, hs in dup.items():
        if cmd5 in conflict:
            continue
        drop.update(sorted(hs)[1:])
    keep = [r for r in rows if r["hash"] not in drop]
    print(f"  {len(rows)} -> {len(keep)}  (제거 {len(drop)})")
    before = Counter(r["category"] for r in rows)
    after = Counter(r["category"] for r in keep)
    print(f"\n  {'category':10s} {'before':>8s} {'after':>8s} {'유지율':>8s}")
    for c in sorted(before):
        print(f"  {c:10s} {before[c]:8d} {after[c]:8d} "
              f"{100*after[c]/before[c]:7.1f}%")

    json.dump({"n_rows": len(rows),
               "dup_content": {k: v for k, v in dup.items()},
               "conflict": {k: v for k, v in conflict.items()}},
              open(args.out, "w"), indent=2)
    print(f"\n리포트 저장: {args.out}")
    print(f"다음: make_splits_final.py --manifest ... --report {args.out}")


if __name__ == "__main__":
    main()