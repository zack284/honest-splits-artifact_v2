#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_determinism.py   (Linux 서버에서 실행: APK 접근 필요)

Obfuscapk 를 재실행하지 않고 난독화의 결정론을 검증한다.

원리:
  이 코퍼스에는 원본 dex 가 byte-identical 인 샘플들이 다수 존재한다(중복 34.6%).
  같은 원본에서 나온 난독화 결과가 서로 byte-identical 이라면,
  그것이 곧 "같은 입력 -> 같은 출력"의 증거다.

절차:
  1) 각 쌍에서 원본 dex / 난독화 dex 를 추출해 sha256 계산
  2) 원본 해시로 그룹핑
  3) 원본이 같은 그룹 안에서 난독화 해시가 몇 종류인지 카운트
     - 1종류  -> 결정론적
     - 2종류+ -> 비결정 요소 존재

주의: APK 자체는 재패키징/서명 때문에 byte-identical 이 아닐 수 있으므로
      반드시 '추출한 dex 바이트' 기준으로 비교한다.

사용:
    python check_determinism.py --n 3000
    python check_determinism.py --n 0        # 전체
"""

import os
import csv
import zipfile
import hashlib
import random
import argparse
from collections import defaultdict

BASE = "~cicmaldroid"


def extract_dex(apk_path):
    try:
        with zipfile.ZipFile(apk_path) as z:
            dex = sorted([f for f in z.namelist()
                          if f.startswith("classes") and f.endswith(".dex")])
            if not dex:
                return None
            return b"".join(z.read(f) for f in dex)
    except Exception:
        return None


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--log", default=None)
    ap.add_argument("--n", type=int, default=3000, help="표본 수 (0=전체)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    base = args.base
    log = args.log or f"{base}/pairing_log_all.csv"

    pairs = []
    with open(log) as f:
        for r in csv.DictReader(f):
            if r["obf_success"] == "True":
                pairs.append((r["category"], r["hash"]))

    if args.n and args.n < len(pairs):
        pairs = random.Random(args.seed).sample(pairs, args.n)
    print(f"측정 표본: {len(pairs)} 쌍\n")

    # orig_hash -> set(obf_hash)
    groups = defaultdict(set)
    counts = defaultdict(int)
    skipped = 0

    for i, (cat, h) in enumerate(pairs):
        do = extract_dex(f"{base}/{cat}/{h}")
        df = extract_dex(f"{base}/obfuscated/{cat}/{h}_obf.apk")
        if do is None or df is None:
            skipped += 1
            continue
        oh = sha(do)
        groups[oh].add(sha(df))
        counts[oh] += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pairs)} 처리")

    # 원본이 중복된(즉 2개 이상 샘플이 같은 원본을 갖는) 그룹만 검증 대상
    multi = {k: v for k, v in groups.items() if counts[k] > 1}
    det = sum(1 for v in multi.values() if len(v) == 1)
    nondet = len(multi) - det
    covered = sum(counts[k] for k in multi)

    print("\n" + "=" * 62)
    print("난독화 결정론 검증 (재실행 없이, 중복 원본을 이용)")
    print("=" * 62)
    print(f"  추출 성공 쌍           : {sum(counts.values())} (실패 {skipped})")
    print(f"  서로 다른 원본 dex     : {len(groups)}")
    print(f"  원본이 2개 이상 중복된 그룹: {len(multi)}  (해당 샘플 {covered}개)")
    print()
    if multi:
        print(f"  난독화 결과가 완전히 동일한 그룹 : {det}/{len(multi)} "
              f"({100*det/len(multi):.2f}%)")
        print(f"  난독화 결과가 갈린 그룹          : {nondet}")
        if nondet:
            print("\n  [갈린 그룹 예시]")
            for k, v in list(multi.items()):
                if len(v) > 1:
                    print(f"    원본 {k[:16]}... -> 난독화 결과 {len(v)}종 "
                          f"(샘플 {counts[k]}개)")
                    if sum(1 for kk, vv in multi.items() if len(vv) > 1) > 5:
                        break
    else:
        print("  중복 원본 그룹이 표본에 없음. --n 을 늘려서 재시도.")
    print("=" * 62)


if __name__ == "__main__":
    main()