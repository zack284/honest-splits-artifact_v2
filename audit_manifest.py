#!/usr/bin/env python3
"""
audit_manifest.py -- split 만들기 전 manifest 감사

목적:
  hash 문자열이 unique해도, 같은 APK가 서로 다른 파일명 규칙으로
  중복 등재돼 있으면 train/test가 새어버린다.
  파일명 대신 '이미지 바이트 내용'으로 중복을 잡는다.

  97 = <sha256>.<md5>   (sha256, md5 둘 다 보유)
  68 = <sha256>.apk     (sha256만)
  36 = <md5>.apk        (md5만)
  -> 68 과 36 은 공통 키가 없어 파일명 대조가 불가능.
  -> 그래서 내용 해시가 유일한 보편 해법.

Usage:
    python audit_manifest.py --manifest ~/image_pairing_manifest.csv
"""

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

HEX = re.compile(r"^[0-9a-fA-F]+$")


def decode(h):
    """hash 문자열에서 (sha256, md5, 규칙이름) 추출. 실패 시 None."""
    if len(h) == 97 and h[64] == "." and HEX.match(h[:64]) and HEX.match(h[65:]):
        return h[:64], h[65:], "sha256.md5"
    if len(h) == 68 and h.endswith(".apk") and HEX.match(h[:64]):
        return h[:64], None, "sha256.apk"
    if len(h) == 36 and h.endswith(".apk") and HEX.match(h[:32]):
        return None, h[:32], "md5.apk"
    return None, None, f"UNKNOWN(len={len(h)})"


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
    ap.add_argument("--out", default="./audit_report.json")
    ap.add_argument("--skip_content", action="store_true",
                    help="이미지 내용 해시 건너뛰기 (파일명 검사만)")
    ap.add_argument("--path_from", default=None,
                    help="manifest 경로 접두사 
    ap.add_argument("--path_to", default=None,
                    help="치환할 로컬 접두사. 서버에서 돌리면 불필요.")
    args = ap.parse_args()

    def remap(p):
        if args.path_from and args.path_to and p.startswith(args.path_from):
            return args.path_to + p[len(args.path_from):]
        return p

    rows = []
    with open(args.manifest, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"manifest rows: {len(rows)}\n")

    # ---------- 1. 파일명 규칙 해독 ----------
    print("=" * 60)
    print("1. 파일명 규칙")
    print("=" * 60)
    scheme_of, sha_of, md5_of = {}, {}, {}
    for r in rows:
        h = r["hash"].strip()
        s, m, name = decode(h)
        scheme_of[h], sha_of[h], md5_of[h] = name, s, m

    print(f"규칙 분포: {dict(Counter(scheme_of.values()))}")
    unknown = [h for h, s in scheme_of.items() if s.startswith("UNKNOWN")]
    if unknown:
        print(f"  !! 해독 실패 {len(unknown)}건 — 예시: {unknown[:3]}")
    else:
        print("  [ok] 전부 세 규칙 중 하나로 해독됨 (가설 확인)")

    # 규칙 x 카테고리 교차표: 수집 출처가 갈리는지
    print(f"\n규칙 x category:")
    ct = defaultdict(Counter)
    for r in rows:
        ct[scheme_of[r["hash"].strip()]][r["category"].strip()] += 1
    cats = sorted({r["category"].strip() for r in rows})
    print(f"{'scheme':14s} " + " ".join(f"{c:>9s}" for c in cats))
    for s in sorted(ct):
        print(f"{s:14s} " + " ".join(f"{ct[s][c]:9d}" for c in cats))
    print("  (규칙이 category와 완전히 겹치면 수집 배치가 다른 것 — 기록해둘 것)")

    # ---------- 2. 파일명 키 기반 중복 ----------
    print("\n" + "=" * 60)
    print("2. 파일명 키 중복 (sha256 / md5)")
    print("=" * 60)
    by_sha, by_md5 = defaultdict(list), defaultdict(list)
    for h in scheme_of:
        if sha_of[h]:
            by_sha[sha_of[h]].append(h)
        if md5_of[h]:
            by_md5[md5_of[h]].append(h)

    dup_sha = {k: v for k, v in by_sha.items() if len(v) > 1}
    dup_md5 = {k: v for k, v in by_md5.items() if len(v) > 1}
    print(f"sha256 충돌: {len(dup_sha)}   (97 <-> 68 대조)")
    print(f"md5    충돌: {len(dup_md5)}   (97 <-> 36 대조)")
    for k, v in list(dup_sha.items())[:3]:
        print(f"  sha256 {k[:16]}... -> {[scheme_of[x] for x in v]}")
    for k, v in list(dup_md5.items())[:3]:
        print(f"  md5    {k[:16]}... -> {[scheme_of[x] for x in v]}")
    print("  주의: 68 <-> 36 은 공통 키가 없어 여기서 안 잡힘. 3번이 필요한 이유.")

    # ---------- 3. 이미지 내용 해시 (보편 검사) ----------
    content_groups = {}
    if not args.skip_content:
        print("\n" + "=" * 60)
        print("3. 이미지 내용 해시 — 이름 규칙과 무관한 보편 중복 검사")
        print("=" * 60)
        # 첫 파일부터 확인. 못 읽으면 여기서 죽는다 (경로 문제 조기 발견)
        probe = remap(rows[0]["original_image"])
        if not os.path.exists(probe):
            raise SystemExit(
                f"\n[FATAL] 첫 이미지를 못 찾습니다:\n  {probe}\n"
                f"(이전 버전은 여기서 전부 실패하고도 '중복 없음'을 출력했습니다.)")

        by_content = defaultdict(list)
        n_err = 0
        for i, r in enumerate(rows):
            if i % 2000 == 0:
                print(f"  ... {i}/{len(rows)}")
            try:
                by_content[file_md5(remap(r["original_image"]))].append(r["hash"].strip())
            except Exception as e:
                n_err += 1
                if n_err <= 3:
                    print(f"  !! 읽기 실패: {remap(r['original_image'])} ({e})")

        # --- false pass 차단: 검사 실패를 '통과'로 보고하지 않는다 ---
        n_read = len(rows) - n_err
        if n_err > 0:
            raise SystemExit(
                f"\n[FATAL] {n_err}/{len(rows)}건 읽기 실패. "
                f"부분 검사 결과는 신뢰할 수 없으므로 중단합니다.")
        if n_read == 0 or not by_content:
            raise SystemExit("\n[FATAL] 읽은 이미지가 0장. 검사가 수행되지 않았습니다.")
        print(f"  [ok] {n_read}장 전부 읽음 — 검사가 실제로 수행됨")

        dup_c = {k: v for k, v in by_content.items() if len(v) > 1}
        n_extra = sum(len(v) - 1 for v in dup_c.values())
        print(f"\n고유 이미지: {len(by_content)} / 등재 {len(rows) - n_err}")
        print(f"중복 그룹  : {len(dup_c)}   초과 항목: {n_extra}")

        if dup_c:
            print(f"\n  !! 중복 발견 — 같은 dex가 여러 이름으로 등재됨")
            xs = Counter()
            for v in dup_c.values():
                xs["+".join(sorted({scheme_of[h] for h in v}))] += 1
            print(f"  중복이 걸친 규칙 조합: {dict(xs)}")
            for v in list(dup_c.values())[:3]:
                print(f"    {[f'{h[:20]}...({scheme_of[h]})' for h in v]}")
            print(f"\n  -> split은 반드시 '내용 해시' 단위로 묶어야 합니다.")
            print(f"     같은 그룹은 통째로 같은 split에 넣거나, 대표 1개만 남기고 제거.")
        else:
            print(f"\n  [ok] {n_read}장 전수 검사 결과 내용 중복 없음. "
                  f"hash 단위 split이 안전합니다.")

        content_groups = {k: v for k, v in dup_c.items()}

    # ---------- 출력 ----------
    report = {
        "n_rows": len(rows),
        "schemes": dict(Counter(scheme_of.values())),
        "dup_sha256": {k: v for k, v in list(dup_sha.items())},
        "dup_md5": {k: v for k, v in list(dup_md5.items())},
        "dup_content": content_groups,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n리포트 저장: {args.out}")


if __name__ == "__main__":
    main()