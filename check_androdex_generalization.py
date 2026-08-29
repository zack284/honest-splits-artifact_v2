"""
Cross-domain generalization check.

The generator was trained ONLY on CICMalDroid pairs. Here we apply it to
AndroDex pairs (a completely different dataset, never seen during training)
and measure whether it still moves generated images closer to the real
obfuscated image than the original input already is.

If this holds, it's evidence the generator learned something about
obfuscation transformations in general, not just CICMalDroid-specific
artifacts.

Usage:
    python check_androdex_generalization.py \
        --manifest ~/Androdex/Images/Set2_ObfuscAPK/androdex_pairing_manifest.csv \
        --checkpoint ~/checkpoints/generator_epoch50.pth \
        --n_samples 500
"""

import argparse
import csv
import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_pix2pix import GeneratorUNet


def tensor_to_numpy01(t):
    arr = t.squeeze(0).cpu().numpy()
    return (arr * 0.5 + 0.5).clip(0, 1)


def load_image(path, img_size):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    img = Image.open(path).convert("L")
    return tf(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_category", action="store_true",
                        help="Also report metrics broken down by category (malware/benign)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = GeneratorUNet(in_channels=1, out_channels=1).to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    generator.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # -------- Load manifest --------
    rows = []
    with open(args.manifest, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["hash"], row["category"],
                        row["original_image"], row["obfuscated_image"]))
    print(f"Manifest has {len(rows)} pairs total")

    random.seed(args.seed)
    n = min(args.n_samples, len(rows))
    sample_rows = random.sample(rows, n)

    results = {"all": {"l1_orig": [], "l1_fake": [], "ssim_orig": [], "ssim_fake": []}}

    with torch.no_grad():
        for i, (h, cat, orig_path, obf_path) in enumerate(sample_rows):
            try:
                real_A = load_image(orig_path, args.img_size)
                real_B = load_image(obf_path, args.img_size)
            except Exception as e:
                print(f"  skip {h[:12]}...: {e}")
                continue

            real_A_dev = real_A.unsqueeze(0).to(device)
            fake_B = generator(real_A_dev).squeeze(0).cpu()

            l1_o = torch.mean(torch.abs(real_A - real_B)).item()
            l1_f = torch.mean(torch.abs(fake_B - real_B)).item()

            a_np = tensor_to_numpy01(real_A)
            b_np = tensor_to_numpy01(real_B)
            f_np = tensor_to_numpy01(fake_B)
            ssim_o = ssim(a_np, b_np, data_range=1.0)
            ssim_f = ssim(f_np, b_np, data_range=1.0)

            results["all"]["l1_orig"].append(l1_o)
            results["all"]["l1_fake"].append(l1_f)
            results["all"]["ssim_orig"].append(ssim_o)
            results["all"]["ssim_fake"].append(ssim_f)

            if args.per_category:
                if cat not in results:
                    results[cat] = {"l1_orig": [], "l1_fake": [], "ssim_orig": [], "ssim_fake": []}
                results[cat]["l1_orig"].append(l1_o)
                results[cat]["l1_fake"].append(l1_f)
                results[cat]["ssim_orig"].append(ssim_o)
                results[cat]["ssim_fake"].append(ssim_f)

            if (i + 1) % 100 == 0:
                print(f"  processed {i+1}/{n}")

    def report(name, d):
        l1_o = np.array(d["l1_orig"]); l1_f = np.array(d["l1_fake"])
        s_o  = np.array(d["ssim_orig"]); s_f = np.array(d["ssim_fake"])
        n_eval = len(l1_o)
        if n_eval == 0:
            print(f"\n[{name}] no samples evaluated")
            return

        print(f"\n{'='*60}")
        print(f"[{name}]  n={n_eval}")
        print(f"{'='*60}")
        print(f"L1  Original->Real: {l1_o.mean():.4f} ± {l1_o.std():.4f}")
        print(f"L1  Generated->Real: {l1_f.mean():.4f} ± {l1_f.std():.4f}")
        imp_l1 = (l1_o.mean() - l1_f.mean()) / l1_o.mean() * 100
        print(f"L1 improvement: {imp_l1:+.1f}%")

        print(f"SSIM Original->Real: {s_o.mean():.4f} ± {s_o.std():.4f}")
        print(f"SSIM Generated->Real: {s_f.mean():.4f} ± {s_f.std():.4f}")
        imp_ssim = s_f.mean() - s_o.mean()
        print(f"SSIM improvement: {imp_ssim:+.4f}")

        win_l1 = np.sum(l1_f < l1_o) / n_eval * 100
        win_ssim = np.sum(s_f > s_o) / n_eval * 100
        print(f"Per-sample win rate: L1={win_l1:.1f}%  SSIM={win_ssim:.1f}%")

    report("ALL (AndroDex, cross-domain)", results["all"])
    if args.per_category:
        for cat in results:
            if cat != "all":
                report(f"category={cat}", results[cat])

    print(f"\n{'='*60}")
    print("Reference (in-domain, CICMalDroid val set, for comparison):")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()