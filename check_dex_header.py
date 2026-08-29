#!/usr/bin/env python3
"""
check_dex_header.py -- 이미지에서 dex 헤더를 파싱해 원본/난독화본 구조 비교

이미지 픽셀 = dex 바이트이므로, 이미지를 1D 로 펴면 dex 헤더를 직접 읽을 수 있다.
APK 원본이 없어도 된다.

DEX 헤더 (112 bytes):
    0   magic[8]        "dex\n035\0"
    8   checksum        adler32
    12  signature[20]   SHA-1
    32  file_size       uint32 LE
    36  header_size     0x70
    40  endian_tag      0x12345678
    56  string_ids_size
    64  type_ids_size
    88  method_ids_size
    96  class_defs_size   <- 클래스 개수. 핵심.
    104 data_size

판정:
    class_defs 비율 ~ 1.0  -> 클래스 보존. 크기 감소는 디버그/데이터 섹션. 정상
    class_defs 비율 << 1.0 -> 클래스 소실. multidex 재배분 의심. 쌍이 깨진 것

Usage:
    python check_dex_header.py \
        --manifest ~image_pairing_manifest.csv \
        --splits ./splits_final/splits.json
"""

import argparse, csv, json, struct
import statistics as st
from collections import Counter

import numpy as np
from PIL import Image

MAGIC = b"dex\n"


def read_header(path):
    """이미지 -> 1D 바이트 -> dex 헤더 필드."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L")).reshape(-1)
    n_px = int(a.size)
    b = a[:112].tobytes()
    if len(b) < 112:
        return None
    u = lambda off: struct.unpack_from("<I", b, off)[0]
    return {
        "magic_ok": b[:4] == MAGIC,
        "version": b[4:7].decode("ascii", "replace"),
        "file_size": u(32),
        "header_size": u(36),
        "endian": u(40),
        "string_ids": u(56),
        "type_ids": u(64),
        "method_ids": u(88),
        "class_defs": u(96),
        "data_size": u(104),
        "n_px": n_px,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default="./splits_final/splits.json")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    target = set(json.load(open(args.splits))[args.split])
    rows = [r for r in csv.DictReader(open(args.manifest, newline=""))
            if r["hash"].strip() in target]
    print(f"{args.split}: {len(rows)} 쌍\n")

    recs = []
    for i, r in enumerate(rows):
        if i % 200 == 0:
            print(f"  ... {i}/{len(rows)}")
        ho = read_header(r["original_image"])
        hb = read_header(r["obfuscated_image"])
        if ho is None or hb is None:
            continue
        recs.append((r["hash"].strip(), r["category"].strip(), ho, hb))

    print("\n" + "=" * 66)
    print("1. 렌더링 정합성 — 이미지가 정말 dex 인가")
    print("=" * 66)
    bad_magic = [x for x in recs if not (x[2]["magic_ok"] and x[3]["magic_ok"])]
    print(f"  magic 'dex\\n' 확인: {len(recs)-len(bad_magic)}/{len(recs)}")
    if bad_magic:
        print(f"  !! magic 불일치 {len(bad_magic)}건 — 이미지가 dex 가 아니거나 렌더링 오류")
    hs = Counter(x[2]["header_size"] for x in recs)
    print(f"  header_size (0x70=112 여야 함): {dict(hs)}")
    vs = Counter(x[2]["version"] for x in recs)
    print(f"  dex version: {dict(vs)}")

    # file_size 가 픽셀 수를 넘으면 이미지가 dex 를 다 못 담은 것
    trunc = [x for x in recs if x[2]["file_size"] > x[2]["n_px"]]
    print(f"  file_size > 픽셀수 (이미지가 dex 를 잘라먹음): {len(trunc)}/{len(recs)}")
    if trunc:
        for h, c, o, b in trunc[:3]:
            print(f"    {c:9s} dex={o['file_size']} px={o['n_px']} "
                  f"({100*o['n_px']/o['file_size']:.1f}% 만 담김)  {h[:32]}")

    print("\n" + "=" * 66)
    print("2. 난독화 전후 구조 비율 (obf / orig)")
    print("=" * 66)
    print(f"{'필드':14s} {'중앙값':>9s} {'10분위':>9s} {'90분위':>9s} {'최소':>8s}")
    for k in ("class_defs", "method_ids", "string_ids", "type_ids",
              "file_size", "data_size"):
        rr = [x[3][k] / x[2][k] for x in recs if x[2][k] > 0]
        if not rr:
            continue
        print(f"{k:14s} {st.median(rr):9.3f} {np.percentile(rr,10):9.3f} "
              f"{np.percentile(rr,90):9.3f} {min(rr):8.3f}")

    print("\n" + "=" * 66)
    print("3. 판정 — 크기가 준 샘플에서 클래스도 줄었는가")
    print("=" * 66)
    shrunk = [x for x in recs
              if x[2]["file_size"] > 0 and x[3]["file_size"] / x[2]["file_size"] < 0.9]
    print(f"  file_size 가 10% 이상 준 샘플: {len(shrunk)}/{len(recs)}")
    if shrunk:
        cd = [x[3]["class_defs"] / max(1, x[2]["class_defs"]) for x in shrunk]
        print(f"  그 샘플들의 class_defs 비율: 중앙값 {st.median(cd):.3f}  "
              f"최소 {min(cd):.3f}  최대 {max(cd):.3f}")
        n_lost = sum(1 for r in cd if r < 0.9)
        print(f"  클래스도 10% 이상 잃은 것: {n_lost}/{len(shrunk)}")
        print()
        if n_lost > len(shrunk) * 0.5:
            print("  => 클래스가 사라졌습니다. multidex 재배분 의심.")
            print("     원본 classes.dex 와 난독화 classes.dex 가 서로 다른")
            print("     클래스 집합을 담고 있습니다. 쌍이 성립하지 않습니다.")
        else:
            print("  => 클래스는 보존됐습니다. 크기 감소는 디버그/데이터 섹션에서")
            print("     온 것으로 보입니다. apktool 재컴파일의 정상적 부작용.")

        print(f"\n  {'category':9s} {'sz비율':>7s} {'cls비율':>8s} {'orig cls':>9s} {'obf cls':>8s}")
        for h, c, o, b in sorted(shrunk,
                                 key=lambda x: x[3]["file_size"]/x[2]["file_size"])[:10]:
            print(f"  {c:9s} {b['file_size']/o['file_size']:7.3f} "
                  f"{b['class_defs']/max(1,o['class_defs']):8.3f} "
                  f"{o['class_defs']:9d} {b['class_defs']:8d}")

    print("\n" + "=" * 66)
    print("4. 전체에서 클래스를 잃은 샘플")
    print("=" * 66)
    lost = [x for x in recs
            if x[2]["class_defs"] > 0 and x[3]["class_defs"] / x[2]["class_defs"] < 0.9]
    print(f"  {len(lost)}/{len(recs)} ({100*len(lost)/len(recs):.1f}%)")
    if lost:
        print(f"  category: {dict(Counter(x[1] for x in lost))}")


if __name__ == "__main__":
    main()