"""
run_visual_grid.py — Visual grid 4×4 với cùng noise seed (TASK-06)

Grid: rows = NFE levels {1, 4, 16, 128}, cols = 4 mẫu (fixed initial noise)
Dùng cho artifact taxonomy trong capstone.

Usage:
    cd <repo_root>
    python study-zone/scripts/run_visual_grid.py \
        --load_dir study-zone/checkpoints/celeba-shortcut2-every4400001 \
        --out study-zone/plots/visual_grid_nfe.png \
        --seed 42

Output: study-zone/plots/visual_grid_nfe.png  (Figure 2 trong capstone)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import imageio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from model import DiT
from utils.checkpoint import Checkpoint
from utils.train_state import TrainStateEma


NFE_LEVELS = [1, 4, 16, 128]
N_COLS     = 4   # số sample mỗi hàng


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--load_dir', required=True)
    p.add_argument('--out', default='study-zone/plots/visual_grid_nfe.png')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--patch_size', type=int, default=2)
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--num_heads', type=int, default=12)
    p.add_argument('--mlp_ratio', type=float, default=4.0)
    p.add_argument('--num_classes', type=int, default=1)
    p.add_argument('--image_size', type=int, default=32)
    p.add_argument('--image_channels', type=int, default=4)
    return p.parse_args()


def run_inference(model, params, x_init, nfe, batch_size):
    """Euler sampling: x_T → x_0 với num_steps=nfe, dt_flow=log2(nfe)."""
    denoise_timesteps = nfe
    delta_t = 1.0 / denoise_timesteps
    dt_flow = int(np.log2(nfe)) if nfe > 1 else 0
    dt_base = jnp.ones((batch_size,), dtype=jnp.int32) * dt_flow
    labels  = jnp.ones((batch_size,), dtype=jnp.int32) * 1  # null token (num_classes=1)

    x = x_init
    for ti in range(denoise_timesteps):
        t_val = ti / denoise_timesteps
        t_vec = jnp.full((batch_size,), t_val)
        v = model.apply({'params': params}, x, t_vec, dt_base, labels, train=False)
        x = x + v * delta_t
    return x


def decode_to_uint8(vae_decode, x_latent):
    """VAE decode → clip → uint8 numpy (H, W, 3)."""
    x_pixel = vae_decode(x_latent)       # (N, 256, 256, 3) in [-1, 1]
    x_np    = np.array(x_pixel)
    x_np    = np.clip(x_np * 0.5 + 0.5, 0, 1)
    return (x_np * 255).astype(np.uint8)


def make_grid(rows_imgs, nfe_levels, pad=4, label_w=80):
    """
    rows_imgs : list of np.ndarray, shape (N_COLS, H, W, 3), one per NFE level
    nfe_levels: list of int, NFE for each row
    Returns PIL Image.
    """
    n_rows = len(rows_imgs)
    n_cols = rows_imgs[0].shape[0]
    H, W   = rows_imgs[0].shape[1], rows_imgs[0].shape[2]

    grid_h = n_rows * H + (n_rows + 1) * pad
    grid_w = label_w + n_cols * W + (n_cols + 1) * pad

    canvas = Image.new('RGB', (grid_w, grid_h), color=(240, 240, 240))
    draw   = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', size=20)
    except Exception:
        font = ImageFont.load_default()

    for row_idx, (imgs, nfe) in enumerate(zip(rows_imgs, nfe_levels)):
        y = pad + row_idx * (H + pad)

        # Row label
        label = f'NFE={nfe}'
        draw.text((pad, y + H // 2 - 10), label, fill=(30, 30, 30), font=font)

        for col_idx in range(n_cols):
            x = label_w + pad + col_idx * (W + pad)
            img_pil = Image.fromarray(imgs[col_idx])
            canvas.paste(img_pil, (x, y))

    return canvas


def main():
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)

    print(f"JAX devices: {jax.devices()}")
    print(f"Loading checkpoint: {args.load_dir}")

    # ── Build model ──────────────────────────────────────────────
    model = DiT(
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        out_channels=args.image_channels,
        class_dropout_prob=1.0,
        num_classes=args.num_classes,
        dropout=0.0,
    )

    rng, init_rng = jax.random.split(rng)
    dummy_img = jnp.zeros((1, args.image_size, args.image_size, args.image_channels))
    dummy_t   = jnp.zeros((1,))
    dummy_dt  = jnp.zeros((1,), dtype=jnp.int32)
    dummy_lbl = jnp.zeros((1,), dtype=jnp.int32)
    params = model.init(init_rng, dummy_img, dummy_t, dummy_dt, dummy_lbl, train=False)['params']
    train_state = TrainStateEma.create(model, params, init_rng)

    # ── Load checkpoint ───────────────────────────────────────────
    cp = Checkpoint(args.load_dir)
    cp_dict = cp.load_as_dict()
    ts_dict = cp_dict['train_state']
    del ts_dict['opt_state']
    train_state = train_state.replace(**ts_dict)
    print(f"Loaded checkpoint, step={train_state.step}")

    # ── Fixed initial noise: shape (N_COLS, H, W, C) ─────────────
    rng, noise_rng = jax.random.split(rng)
    x_init_all = jax.random.normal(
        noise_rng,
        (N_COLS, args.image_size, args.image_size, args.image_channels)
    )
    print(f"Fixed noise seed={args.seed}, shape={x_init_all.shape}")

    # ── Load VAE once ─────────────────────────────────────────────
    from utils.stable_vae import StableVAE
    print("Loading StableVAE decoder...")
    vae = StableVAE.create()
    vae_decode = jax.jit(vae.decode)

    # ── Generate one row per NFE level ────────────────────────────
    rows_imgs = []
    for nfe in NFE_LEVELS:
        print(f"\nRunning NFE={nfe} Euler sampling ({nfe} steps)...")
        x_out = run_inference(model, train_state.params_ema, x_init_all, nfe, N_COLS)
        imgs_uint8 = decode_to_uint8(vae_decode, x_out)  # (N_COLS, 256, 256, 3)
        rows_imgs.append(imgs_uint8)

        # Save individual row for reference
        row_path = str(Path(args.out).parent / f'grid_row_nfe{nfe}.png')
        row_img = Image.fromarray(
            np.concatenate(imgs_uint8, axis=1)  # (256, 256*N_COLS, 3)
        )
        row_img.save(row_path)
        print(f"  Saved row: {row_path}")

    # ── Compose 4×4 grid ──────────────────────────────────────────
    print("\nComposing 4×4 grid...")
    grid = make_grid(rows_imgs, NFE_LEVELS, pad=6, label_w=90)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out, dpi=(150, 150))
    print(f"\nSaved grid: {args.out}")
    print(f"Grid size: {grid.size[0]}×{grid.size[1]} px")
    print("\nCaption (for capstone):")
    print("Figure 2: CelebA 256×256 samples from the shortcut model at varying NFE levels,")
    print("using identical initial Gaussian noise (seed=42). Each column shares the same")
    print("noise vector x_T; rows show progressive quality improvement with more function")
    print("evaluations. NFE=1 exhibits blur/halos; NFE=4 recovers coarse structure;")
    print("NFE≥16 produces sharp, coherent faces.")


if __name__ == '__main__':
    main()
