"""
compute_fid.py — Tính relative FID giữa các sample directories.

Chỉ số Frechet Inception Distance (FID), hay viết tắt là FID, là một thước đo
tính toán khoảng cách giữa các vectơ đặc trưng được tính toán cho hình ảnh
thực và hình ảnh được tạo ra.

Dùng pytorch-fid (có epsilon regularization, chạy trên CPU).

Usage:
    cd <repo_root>
    python study-zone/scripts/compute_fid.py \
        --samples_dir study-zone/samples \
        --ref celeba_128step_n504

NOTE (TASK-04 · Option A):
  "celeba_fm_baseline_128step" = Shortcut model với OOD conditioning (d=0),
  KHÔNG phải FM model thực sự. Dùng như ablation.
"""
import os, json, argparse
import numpy as np
from pathlib import Path


# NFE mapping: dir name → (nfe, display label)
NFE_MAP = {
    'celeba_1step_n504':              (1,   'Shortcut (NFE=1)'),
    'celeba_2step_n504':              (2,   'Shortcut (NFE=2)'),
    'celeba_4step_n504':              (4,   'Shortcut (NFE=4)'),
    'celeba_8step_n504':              (8,   'Shortcut (NFE=8)'),
    'celeba_16step_n504':             (16,  'Shortcut (NFE=16)'),
    'celeba_32step_n504':             (32,  'Shortcut (NFE=32)'),
    'celeba_64step_n504':             (64,  'Shortcut (NFE=64)'),
    'celeba_128step_n504':            (128, 'Shortcut (NFE=128) [ref]'),
    'celeba_fm_baseline_128step':     (None, 'Shortcut, OOD cond. d=0 (NFE=128)'),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--samples_dir', default='study-zone/samples')
    p.add_argument('--ref', default='celeba_128step_n504')
    p.add_argument('--min_samples', type=int, default=100)
    p.add_argument('--out_json', default='study-zone/logs/fid_results.json')
    return p.parse_args()


def compute_fid_cpu(path1, path2, batch_size=32, dims=192):
    """Compute FID using pytorch-fid with CPU device and epsilon regularization."""
    import torch
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import get_activations, calculate_frechet_distance
    from torchvision import transforms
    from PIL import Image

    device = torch.device('cpu')
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device).eval()

    def get_acts(folder):
        files = sorted(list(Path(folder).glob('*.png')))
        imgs = []
        for f in files:
            img = Image.open(f).convert('RGB')
            img = transforms.ToTensor()(img)
            imgs.append(img)
        imgs = torch.stack(imgs)  # (N, 3, H, W)
        acts = []
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i+batch_size].to(device)
            with torch.no_grad():
                pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
            acts.append(pred.squeeze().cpu().numpy())
        return np.concatenate(acts, axis=0)

    print(f"  Computing activations for {path1}...")
    acts1 = get_acts(path1)
    print(f"  Computing activations for {path2}...")
    acts2 = get_acts(path2)

    mu1, sigma1 = acts1.mean(axis=0), np.cov(acts1, rowvar=False)
    mu2, sigma2 = acts2.mean(axis=0), np.cov(acts2, rowvar=False)

    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def main():
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    ref_dir_path = samples_dir / args.ref
    assert ref_dir_path.exists(), f"Ref dir not found: {ref_dir_path}"

    all_dirs = sorted([d.name for d in samples_dir.iterdir()
                       if d.is_dir() and d.name != args.ref])
    pairs = [(d, args.ref) for d in all_dirs]

    results = {}
    print(f"\n{'='*60}")
    print(f"Reference: {args.ref}")
    print(f"NOTE: OOD baseline = Shortcut model with d=0 (ablation, not FM)")
    print(f"{'='*60}")

    for gen_name, ref_name in pairs:
        gen_dir = str(samples_dir / gen_name)
        ref_dir_p = str(samples_dir / ref_name)
        if not Path(gen_dir).exists():
            print(f"  SKIP {gen_name} — dir not found")
            continue
        n_gen = len(list(Path(gen_dir).glob('*.png')))
        if n_gen < args.min_samples:
            print(f"  SKIP {gen_name} — only {n_gen} images (min={args.min_samples})")
            continue
        label = NFE_MAP.get(gen_name, (None, gen_name))[1]
        nfe = NFE_MAP.get(gen_name, (None, gen_name))[0]
        print(f"\nFID({label} | {ref_name})  [{n_gen} images]")
        try:
            score = compute_fid_cpu(gen_dir, ref_dir_p)
            results[gen_name] = {
                'fid': round(float(score), 4),
                'nfe': nfe,
                'label': label,
                'n_gen': n_gen,
                'ref': ref_name,
            }
            print(f"  → FID = {score:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print("Summary (sorted by NFE):")
    sorted_results = sorted(results.items(),
                            key=lambda x: (x[1]['nfe'] is None, x[1]['nfe'] or 999))
    for k, v in sorted_results:
        nfe_str = str(v['nfe']) if v['nfe'] else 'OOD'
        print(f"  NFE={nfe_str:>4}  FID={v['fid']:>8.4f}  {v['label']}")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to: {args.out_json}")


if __name__ == '__main__':
    main()
