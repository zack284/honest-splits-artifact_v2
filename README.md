# honest-splits-artifact_v2
honest-splits-artifact


# Honest Splits Reveal the Obfuscation Fragility of Image-Based Malware Detectors

Artifact for the USENIX Security '27 submission *"Honest Splits Reveal
the Obfuscation Fragility of Image-Based Malware Detectors: Leakage,
Baselines, and a Real-Pair Remedy."*

This repository contains the code to reproduce the audit, the
leakage-free split protocol, the size-only baseline, and the
real-pair training experiments reported in the paper.

## 1. Overview

The artifact is organized around the paper's three contributions:

1. **Content-hash audit and leakage-free splits** — detect byte-identical
   duplication in the rendered corpus and construct train/val/test
   partitions over content groups so that no test sample is seen in
   training.
2. **Size-only baseline** — a two-feature (image width, height) random
   forest that measures how much reported accuracy is available from
   file size alone.
3. **Real-pair remedy** — stochastic substitution of each original with
   its real obfuscated rendering during a FixMatch-based
   semi-supervised training loop.

## 2. Datasets

We do **not** redistribute the malware corpora. Both are publicly
available from their original sources:

- **CICMalDroid 2020** — obtain from the dataset authors' official
  release.
- **AndroDex (Set2)** — obtain from the published figshare records
  accompanying the AndroDex paper (Scientific Data, 2024).

After obtaining the datasets, place the extracted APKs / distributed
images under a local data directory and set its path via the
`--data-root` argument (or the `DATA_ROOT` environment variable) used
by the scripts. No absolute paths are hardcoded.

Obfuscated pairs for CICMalDroid are generated with
[Obfuscapk](https://github.com/ClaudiuGeorgiu/Obfuscapk) using a fixed
chain of four transformations (class renaming, method renaming,
constant-string encryption, instruction reordering). AndroDex already
ships clean/obfuscated renderings and requires no obfuscation step.

## 3. Environment

- Python 3.10+
- PyTorch (CUDA-enabled build recommended for training)
- scikit-learn, numpy, pillow, scikit-image, timm

Install dependencies:

pip install -r requirements.txt


## 4. Pipeline

The scripts are intended to be run in the following order. Replace
`<DATA_ROOT>` with your local dataset path.

### 4.1 Rendering and pair construction
- Render dex byte streams to images and build the clean/obfuscated
  pair manifest.
- CICMalDroid manifest: `build_augmentation_splits.py`
- AndroDex manifest: `make_androdex_manifest.py`

### 4.2 Content-hash audit (Contribution 1)
- `audit_manifest.py` — hash decoded pixels, report duplicate groups
  and label-inconsistent groups (reproduces the 34.6% duplication and
  the eight removed label-conflict groups on CICMalDroid; 17.5% on
  AndroDex).
- `check_hashes.py`, `analyze_dups.py` — supporting duplicate analysis.
- `check_dex_header.py` — verify each rendered image parses as valid
  dex (magic, 112-byte header).
- `check_determinism.py`, `check_obfuscation_determinism.py` —
  verify obfuscation determinism (695/697 byte-identical groups on
  CICMalDroid).

### 4.3 Leakage-free splits (Contribution 1)
- `make_splits_final.py` — content-group 80/10/10 split, duplicates
  retained in training only, verified twice (identifiers + pixel
  re-hash).
- `make_ssl_splits.py` — halve the training partition into labeled /
  unlabeled.
- `make_splits_random.py` — conventional random split, for the
  leakage comparison (Section 4.1.5).
- AndroDex: `make_splits_androdex.py`.

### 4.4 Size-only baseline (Contribution 2)
- `size_only_baseline.py` (CICMalDroid), `size_only_androdex.py`
  (AndroDex) — width/height random forest on clean and obfuscated
  test sets.
- `size_shift_under_obfuscation.py` — confirm file size is preserved
  under obfuscation (median ratio 1.000).

### 4.5 Training and the real-pair remedy (Contribution 3)
- `research_maldroid_ViT_v7.py` (CICMalDroid),
  `research_androdex_ViT_v7.py` (AndroDex) — ViT+ASPP backbone,
  FixMatch consistency (τ = 0.60), with probability `p` of substituting
  each sample by its real obfuscated rendering.
- Key arguments: `--p` (substitution probability, main result uses
  0.5), `--batch-size` (6 for the main result; 24 for the batch
  ablation), `--seed`.

### 4.6 Measurement of the transformation
- `measure_obfuscation_effect.py`, `measure_androdex_signal.py` —
  byte-offset difference and SSIM between clean and obfuscated
  renderings (median SSIM 0.124 on CICMalDroid, 0.059 on AndroDex).
- `compare_resize.py` — nearest vs. bilinear resize effect on SSIM
  (0.12 → 0.31).

## 5. Reproducing the main tables

- **Table 6 / Table 11 (main results):** run 4.3–4.5 with `p ∈ {0, 0.5}`
  over three seeds on CICMalDroid and AndroDex respectively.
- **Table 8 (p sweep):** run 4.5 with `p ∈ {0, 0.25, 0.5, 0.75, 1.0}`.
- **Table 9 (batch size):** run 4.5 with `--batch-size ∈ {6, 24}`.
- **Table 10 (leakage comparison):** run 4.5 on both the leakage-free
  split (4.3) and the random split (`make_splits_random.py`).

Reported headline numbers are averaged over three seeds; individual
runs vary within the standard deviations given in the paper.

## 6. Notes

- All paths are supplied at runtime; no user-specific or absolute
  paths are embedded in the code.
- Training is stochastic across seeds; exact per-run values will differ
  slightly from the reported three-seed means.
- We release code and split assignments only; the underlying malware
  corpora must be obtained from their original sources, subject to the
  redistribution terms of those datasets.


