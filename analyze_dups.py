#!/usr/bin/env python3
"""
analyze_dups.py -- 내용 중복 그룹의 라벨 모순 분석

audit_manifest.py 가 만든 audit_report.json 의 dup_content 를 읽어
각 그룹이 몇 개의 category 를 걸치는지 본다. 파일은 안 읽는다.

  단일 category 그룹 -> 그냥 중복. 대표 1개만 남기면 됨.
  복수 category 그룹 -> 라벨 모순. 같은 dex 가 서로 다른 정답을 가짐. 전부 제거.

Usage:
    python analyze_dups.py --manifest image_pairing_manifest.csv \
                           --report ./audit_report.json
"""

import argparse
import csv
import json
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", default="./audit_report.json")
    ap.add_argument("--out", default="./dedup_plan.json")
    args = ap.parse_args()

    # hash -> category
    cat = {}
    for r in csv.DictReader(open(args.manifest, newline="")):
        cat[r["hash"].strip()] = r["category"].strip()
    print(f"manifest: {len(cat)} rows")

    rep = json.load(open(args.report))
    groups = rep["dup_content"]          # content_md5 -> [hash, ...]
    print(f"중복 그룹: {len(groups)}\n")

    clean, conflict = {}, {}
    for cmd5, hs in groups.items():
        cats = {cat[h] for h in hs if h in cat}
        (conflict if len(cats) > 1 else clean)[cmd5] = hs

    n_rows_clean = sum(len(v) for v in clean.values())
    n_rows_conf = sum(len(v) for v in conflict.values())

    print("=" * 62)
    print("A. 단일 category 중복 — 대표 1개만 남기면 됨")
    print("=" * 62)
    print(f"그룹 {len(clean)}, 해당 행 {n_rows_clean}, 제거될 행 {n_rows_clean - len(clean)}")
    cc = Counter(cat[v[0]] for v in clean.values() if v[0] in cat)
    print(f"category 분포: {dict(cc)}")
    sizes = Counter(len(v) for v in clean.values())
    print(f"그룹 크기 분포(상위): {dict(sorted(sizes.items())[:8])} ... max={max(sizes) if sizes else 0}")

    print("\n" + "=" * 62)
    print("B. 복수 category — 라벨 모순. 전부 제거 대상")
    print("=" * 62)
    print(f"그룹 {len(conflict)}, 해당 행 {n_rows_conf}")
    if conflict:
        pair = Counter()
        for hs in conflict.values():
            pair["/".join(sorted({cat[h] for h in hs if h in cat}))] += 1
        print(f"\ncategory 조합별 그룹 수:")
        for k, v in pair.most_common():
            mark = "  <<< 악성/정상 경계" if "Benign" in k else ""
            print(f"  {k:32s} {v:4d}{mark}")
        print(f"\n예시:")
        for cmd5, hs in list(conflict.items())[:3]:
            print(f"  content {cmd5[:12]}...")
            for h in hs[:4]:
                print(f"    {cat.get(h,'?'):9s} {h[:44]}")

    # ---- 최종 생존 집계 ----
    drop = set()
    for hs in conflict.values():
        drop.update(hs)                       # 모순 -> 전부 제거
    for hs in clean.values():
        drop.update(sorted(hs)[1:])           # 중복 -> 첫 것만 유지(결정론적)

    keep = [h for h in cat if h not in drop]
    print("\n" + "=" * 62)
    print("C. 정리 후")
    print("=" * 62)
    print(f"{len(cat)} -> {len(keep)}  (제거 {len(drop)})")
    before = Counter(cat.values())
    after = Counter(cat[h] for h in keep)
    print(f"\n{'category':10s} {'before':>8s} {'after':>8s} {'유지율':>8s}")
    for c in sorted(before):
        pct = 100 * after[c] / before[c] if before[c] else 0
        print(f"{c:10s} {before[c]:8d} {after[c]:8d} {pct:7.1f}%")

    json.dump({"keep": sorted(keep), "drop": sorted(drop),
               "n_conflict_groups": len(conflict), "n_clean_groups": len(clean)},
              open(args.out, "w"), indent=2)
    print(f"\n저장: {args.out}  <- make_splits.py 가 이걸 참조")


if __name__ == "__main__":
    main()