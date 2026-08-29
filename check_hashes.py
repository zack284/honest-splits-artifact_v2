# check_hashes.py
import csv, os, sys
from collections import Counter

MANIFEST = os.path.expanduser("~/image_pairing_manifest.csv")
SPLITS = {
    "train": "train_original.txt",
    "test":  "test_obf.txt",
}

def h(path):
    """파일명에서 hash 추출. _obfus / _gan 접미사 제거."""
    b = os.path.splitext(os.path.basename(path))[0]
    for suf in ("_obfus", "_obfuscated", "_gan"):
        if b.endswith(suf):
            b = b[: -len(suf)]
    return b

# --- manifest ---
man_h, man_orig_h = set(), {}
with open(MANIFEST) as f:
    for row in csv.DictReader(f):
        man_h.add(row["hash"])
        man_orig_h[h(row["original_image"])] = row["hash"]

print(f"manifest hashes            : {len(man_h)}")
print(f"  sample                   : {list(man_h)[:2]}")
print(f"  hash len distribution    : {Counter(len(x) for x in man_h)}")
print(f"  basename(original) len   : {Counter(len(x) for x in man_orig_h)}")
print(f"  hash == basename(orig)?  : {len(man_h & set(man_orig_h))} / {len(man_h)}")
print()

# --- splits ---
sp = {}
for name, path in SPLITS.items():
    if not os.path.exists(path):
        print(f"!! {path} 없음 — 이게 comm이 0을 뱉은 이유일 수 있음")
        continue
    hs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                hs.append(h(line.split()[0]))
    sp[name] = set(hs)
    print(f"{name:6s}: lines={len(hs)}  unique={len(sp[name])}")
    print(f"        sample: {hs[:2]}")
    print(f"        len dist: {Counter(len(x) for x in hs)}")
    hit = len(sp[name] & man_h)
    print(f"        manifest 매칭: {hit}/{len(sp[name])}  <- 0이면 추출/형식 불일치")
    print()

# --- leak check ---
if "train" in sp and "test" in sp:
    inter = sp["train"] & sp["test"]
    print(f"train ∩ test = {len(inter)}   (0이어야 정상)")
    if inter:
        print(f"  누출 예시: {list(inter)[:5]}")