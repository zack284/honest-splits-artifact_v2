#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_obfuscation.py   (Linux 서버에서 실행: APK 접근 필요)

논문 Methods 에 들어갈 세 가지 수치를 실측한다.

  (1) 바이트 변화율   : 원본 dex vs 난독화 dex 에서 같은 위치의 바이트가 다른 비율
  (2) 구조 필드 보존비: dex 헤더의 class_defs_size / method_ids_size / file_size 등의
                        (난독화 / 원본) 비율의 median
  (3) SSIM            : 원본 이미지 vs 난독화 이미지 (고정폭 렌더 기준)
                        + 난독화 결정론 확인은 별도 옵션

사용:
    python audit_obfuscation.py --n 500                 # 표본 500쌍
    python audit_obfuscation.py --n 0                   # 전체
    python audit_obfuscation.py --n 500 --images        # SSIM 도 같이 (이미지 필요)
"""

import os
import csv
import zipfile
import struct
import argparse
import random
import numpy as np

BASE = "~/cicmaldroid"
CATEGORIES = ["Adware", "Banking", "Benign", "Riskware", "SMS"]

# dex header (little-endian uint32)
FIELDS = {
    "file_size":       0x20,
    "header_size":     0x24,
    "string_ids_size": 0x38,
    "type_ids_size":   0x40,
    "proto_ids_size":  0x48,
    "field_ids_size":  0x50,
    "method_ids_size": 0x58,
    "class_defs_size": 0x60,
    "data_size":       0x68,
}


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


def parse_header(dex_bytes):
    """첫 dex 의 헤더 필드. (멀티덱스 병합본이면 첫 dex 헤더)"""
    if len(dex_bytes) < 0x70:
        return None
    if dex_bytes[:4] != b"dex\n":
        return None
    out = {}
    for k, off in FIELDS.items():
        out[k] = struct.unpack_from("<I", dex_bytes, off)[0]
    return out


def byte_change_rate(a, b):
    """같은 위치 바이트가 다른 비율. 길이가 다르면 짧은 쪽까지 비교하고,
    길이 차이는 '변화'로 계산에 포함한다."""
    n = min(len(a), len(b))
    m = max(len(a), len(b))
    if m == 0:
        return float("nan")
    xa = np.frombuffer(a[:n], dtype=np.uint8)
    xb = np.frombuffer(b[:n], dtype=np.uint8)
    diff = int((xa != xb).sum()) + (m - n)
    return diff / m


def ssim(img1, img2):
    """global SSIM (외부 의존 없이)."""
    a = img1.astype(np.float64); b = img2.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * 255) ** 2; c2 = (0.03 * 255) ** 2
    return ((2*mu_a*mu_b + c1) * (2*cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--log", default=None)
    ap.add_argument("--n", type=int, default=500, help="표본 수 (0=전체)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--images", action="store_true",
                    help="렌더된 고정폭 이미지로 SSIM 도 측정")
    ap.add_argument("--manifest", default=None,
                    help="이미지 SSIM 용 manifest (기본: base/image_pairing_manifest.csv)")
    args = ap.parse_args()
    base = args.base
    log = args.log or f"{base}/pairing_log_all.csv"

    # 성공 쌍 목록
    pairs = []
    with open(log) as f:
        for r in csv.DictReader(f):
            if r["obf_success"] == "True":
                pairs.append((r["category"], r["hash"]))
    print(f"성공 쌍: {len(pairs)}")

    rng = random.Random(args.seed)
    if args.n and args.n < len(pairs):
        pairs = rng.sample(pairs, args.n)
    print(f"측정 표본: {len(pairs)}\n")

    change_rates = []
    ratios = {k: [] for k in FIELDS}
    magic_ok = 0
    hdr_112 = 0
    parsed = 0
    skipped = 0

    for cat, h in pairs:
        do = extract_dex(f"{base}/{cat}/{h}")
        df = extract_dex(f"{base}/obfuscated/{cat}/{h}_obf.apk")
        if do is None or df is None:
            skipped += 1
            continue

        change_rates.append(byte_change_rate(do, df))

        ho, hf = parse_header(do), parse_header(df)
        if ho and hf:
            parsed += 1
            magic_ok += 1
            if ho["header_size"] == 112:
                hdr_112 += 1
            for k in FIELDS:
                if ho[k] > 0:
                    ratios[k].append(hf[k] / ho[k])

    print("=" * 62)
    print("(1) 바이트 변화율  (원본 dex vs 난독화 dex, 같은 위치 비교)")
    cr = np.array(change_rates)
    print(f"    n={len(cr)}  mean {cr.mean()*100:.1f}%   median {np.median(cr)*100:.1f}%")
    print(f"    (5th {np.percentile(cr,5)*100:.1f}% / 95th {np.percentile(cr,95)*100:.1f}%)")

    print("\n(2) dex 헤더 구조 필드 보존비  (난독화 / 원본, median)")
    for k in FIELDS:
        v = np.array(ratios[k])
        if len(v):
            print(f"    {k:18s} median {np.median(v):.4f}   "
                  f"(mean {v.mean():.4f}, n={len(v)})")

    print(f"\n    렌더/파싱 검증: magic 'dex\\n' {magic_ok}/{parsed}, "
          f"header_size==112 {hdr_112}/{parsed}, dex 추출 실패 {skipped}")

    # ---- (3) 이미지 SSIM ----
    if args.images:
        from PIL import Image
        man = args.manifest or f"{base}/image_pairing_manifest.csv"
        rows = []
        with open(man) as f:
            for r in csv.DictReader(f):
                rows.append(r)
        if args.n and args.n < len(rows):
            rows = random.Random(args.seed).sample(rows, args.n)

        vals, miss = [], 0
        for r in rows:
            try:
                a = np.array(Image.open(r["original_image"]).convert("L"))
                b = np.array(Image.open(r["obfuscated_image"]).convert("L"))
            except Exception:
                miss += 1
                continue
            # 고정폭 렌더는 크기가 다를 수 있으므로 평가와 동일하게 224x224 로 맞춘다
            if a.shape != b.shape:
                a = np.array(Image.fromarray(a).resize((224, 224), Image.NEAREST))
                b = np.array(Image.fromarray(b).resize((224, 224), Image.NEAREST))
            vals.append(ssim(a, b))

        v = np.array(vals)
        print("\n(3) SSIM  원본 이미지 vs 난독화 이미지")
        print(f"    n={len(v)} (누락 {miss})  mean {v.mean():.4f}   median {np.median(v):.4f}")

    print("=" * 62)


if __name__ == "__main__":
    main()