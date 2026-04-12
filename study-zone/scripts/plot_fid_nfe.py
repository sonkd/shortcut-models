"""
plot_fid_nfe.py — Vẽ FID vs NFE curve từ kết quả compute_fid.py

Usage:
    cd <repo_root>
    python study-zone/scripts/plot_fid_nfe.py \
        --fid_json study-zone/logs/fid_results.json \
        --out study-zone/plots/fid_nfe_curve.png

Output: study-zone/plots/fid_nfe_curve.png (Figure 1 trong capstone)

NOTE (TASK-04 · Option A):
  OOD baseline (force_dt0) được hiển thị riêng dưới nhãn
  "Shortcut, OOD cond. d=0" — KHÔNG phải FM model thực sự.
  Thesis phải có disclaimer: "This is not a trained FM model;
  it is the shortcut model evaluated with out-of-distribution
  conditioning (d=0), serving as an ablation of shortcut conditioning."
"""
import json, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fid_json', default='study-zone/logs/fid_results.json')
    p.add_argument('--out', default='study-zone/plots/fid_nfe_curve.png')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.fid_json) as f:
        results = json.load(f)

    # Separate main curve vs OOD baseline
    main_points = []   # (nfe, fid)
    ood_points  = []   # (nfe, fid) — OOD conditioning ablation

    for key, val in results.items():
        nfe = val.get('nfe')
        fid = val.get('fid')
        label = val.get('label', key)
        if fid is None:
            # Skip aggregate stats entries (e.g. seed_stats without fid key)
            continue
        if nfe is None:
            # OOD baseline — plot at NFE=128 position
            ood_points.append((128, fid, label))
        else:
            main_points.append((nfe, fid))

    main_points.sort(key=lambda x: x[0])
    nfe_vals = [p[0] for p in main_points]
    fid_vals = [p[1] for p in main_points]

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(nfe_vals, fid_vals, 'o-', color='steelblue', linewidth=2,
            markersize=7, label='Shortcut Model (Euler, dt=log₂(NFE))')

    # Annotate each point
    for nfe, fid in main_points:
        ax.annotate(f'{fid:.1f}', (nfe, fid),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8, color='steelblue')

    # OOD baseline — FID >> 0.3 so outside axis; show as text note instead
    for nfe, fid, label in ood_points:
        ax.plot([], [], color='tomato', linestyle='--', linewidth=1.2,
                label=f'Shortcut, OOD cond. d=0 (NFE=128): FID={fid:.1f}\n[Ablation — not a trained FM model]')
        ax.annotate(f'OOD (d=0): FID={fid:.1f} (off-axis)', xy=(0.98, 0.95),
                    xycoords='axes fraction', ha='right', va='top',
                    fontsize=8, color='tomato',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='tomato', alpha=0.8))

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xticks(nfe_vals)

    ax.set_ylim(bottom=0, top=0.3)
    ax.set_xlabel('Number of Function Evaluations (NFE)', fontsize=11)
    ax.set_ylabel('FID ↓  (relative, vs. 128-step shortcut)', fontsize=11)
    ax.set_title('Sample Quality vs. Inference Cost\nCelebA 256×256, DiT-B, n=504 per config',
                 fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")

    # Print table
    print(f"\n{'NFE':>6}  {'FID':>8}  Label")
    print('-' * 50)
    for nfe, fid in main_points:
        print(f"{nfe:>6}  {fid:>8.4f}  Shortcut (NFE={nfe})")
    for nfe, fid, label in ood_points:
        print(f"{'OOD':>6}  {fid:>8.4f}  {label}")


if __name__ == '__main__':
    main()
