#!/usr/bin/env python3
"""
make_androdex_manifest.py -- AndroDex Set2 -> CICMalDroid 형식 manifest

파일명 규칙:
    원본   : <sha256>~.jpg            (69자)
    난독화 : <sha256>_obfuscated.jpg  (79자)
앞 64자 sha256 으로 짝을 맞춘다.

출력 컬럼은 CICMalDroid manifest 와 동일:
    hash, category, original_image, obfuscated_image
-> audit_manifest.py, analyze_dups.py, make_splits_final.py 가 그대로 돈다.

주의: AndroDex 이미지는 RGB(3byte->1pixel) JPG 다.
      CICMalDroid(회색조 1byte/pixel PNG)와 표현이 다르므로
      Grayscale 변환을 그대로 쓰면 무관한 세 바이트를 평균하게 된다.
      SSIM 측정 후 binary 재렌더링 여부를 결정할 것.

Usage:
    python make_androdex_manifest.py \
        --root ~/Androdex/Images/Set2_ObfuscAPK \
        --out ~/Androdex/androdex_set2_manifest.csv
"""

import argparse, csv, os
from collections import Counter

# (category, 원본폴더, 난독화폴더)
PAIRS = [
    ("benign",  "benign_apk_binaries_results",  "obfuscated_benign_apk_binaries_results"),
    ("malware", "malware_apk_binaries_results", "obfuscated_malware_binaries_results"),
]

HEX = set("0123456789abcdefABCDEF")


def index_dir(path):
    """sha256 -> 파일명. 접미사 규칙을 가정하지 않고 앞 64자만 본다.

    폴더마다 명명 규칙이 다르다(`~.jpg`, `_obfuscated.jpg`, 그 외).
    sha256 은 항상 64자 hex 이고 파일명 맨 앞에 오므로 그것만 쓴다.
    """
    m, bad, dup = {}, 0, 0
    suffixes = Counter()
    for f in sorted(os.listdir(path)):
        if not os.path.isfile(os.path.join(path, f)):
            continue
        h = f[:64]
        if len(f) < 64 or not all(c in HEX for c in h):
            bad += 1
            continue
        suffixes[f[64:]] += 1
        if h in m:                      # 같은 sha256 이 여러 파일
            dup += 1
            continue
        m[h] = f
    return m, bad, dup, suffixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    print(f"root: {args.root}\n")

    for cat, od, bd in PAIRS:
        op, bp = os.path.join(args.root, od), os.path.join(args.root, bd)
        for p in (op, bp):
            if not os.path.isdir(p):
                raise SystemExit(f"[FATAL] 폴더 없음: {p}")

        orig, bo, do, so = index_dir(op)
        obf,  bb, db, sb = index_dir(bp)
        common = sorted(set(orig) & set(obf))

        print(f"[{cat}]")
        print(f"  원본   {len(orig):5d}  (sha256 아님 {bo}, 중복 {do})")
        print(f"         접미사: {dict(so.most_common(3))}")
        print(f"  난독화 {len(obf):5d}  (sha256 아님 {bb}, 중복 {db})")
        print(f"         접미사: {dict(sb.most_common(3))}")
        print(f"  쌍     {len(common):5d}")
        print(f"  짝없는 원본 {len(orig)-len(common)}, "
              f"짝없는 난독화 {len(obf)-len(common)}\n")

        if not common:
            print(f"  [!] {cat} 쌍이 0개입니다. 폴더 경로를 확인하세요.\n")

        for h in common:
            rows.append({
                "hash": h,
                "category": cat,
                "original_image": os.path.join(op, orig[h]),
                "obfuscated_image": os.path.join(bp, obf[h]),
            })

    # ---- 카테고리 간 hash 충돌 확인 ----
    # 같은 dex 가 benign 과 malware 양쪽에 있으면 라벨 모순이다.
    c = Counter(r["hash"] for r in rows)
    dup = [h for h, n in c.items() if n > 1]
    if dup:
        print(f"[!] 카테고리를 넘나드는 hash {len(dup)}건 — 라벨 모순")
        print(f"    예시: {dup[:3]}")
        print(f"    manifest 에는 남겨둠. analyze_dups.py 가 잡아냄.\n")

    # ---- 파일 존재 확인 ----
    miss = sum(1 for r in rows
               for k in ("original_image", "obfuscated_image")
               if not os.path.isfile(r[k]))
    if miss:
        raise SystemExit(f"[FATAL] {miss}개 파일이 실제로 없음")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["hash", "category",
                                          "original_image", "obfuscated_image"])
        w.writeheader()
        w.writerows(rows)

    print("=" * 58)
    print(f"총 {len(rows)} 쌍 -> {args.out}")
    print(f"category: {dict(Counter(r['category'] for r in rows))}")
    print("=" * 58)
    print(f"\n다음:")
    print(f"  1) 내용 중복 감사")
    print(f"     python audit_manifest.py --manifest {args.out}")
    print(f"     * JPG 는 메타데이터로 파일해시가 달라질 수 있음.")
    print(f"       file_md5 대신 디코딩된 픽셀 해시를 써야 정확함.")
    print(f"  2) SSIM(x, x') 측정 -> JPG 손실이 난독화 신호를 지웠는지 확인")
    print(f"     CICMalDroid 는 0.12. 0.5 근처면 binary 재렌더링 필요.")


if __name__ == "__main__":
    main()