"""
run_inference_standalone.py
Chạy inference shortcut model mà KHÔNG cần TFDS dataset.
Output ảnh lưu vào study-zone/samples/

Usage:
    cd <repo_root>
    python study-zone/scripts/run_inference_standalone.py \
        --load_dir study-zone/checkpoints/celeba-shortcut2-every4400001 \
        --num_steps 1 \
        --batch_size 16 \
        --save_dir study-zone/samples/celeba_1step
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import imageio

from model import DiT
from utils.checkpoint import Checkpoint
from utils.train_state import TrainStateEma


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--load_dir', required=True)
    p.add_argument('--num_steps', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--save_dir', default='study-zone/samples/output')
    p.add_argument('--num_batches', type=int, default=1,
                   help='Number of batches to generate (total samples = batch_size × num_batches)')
    # CelebA DiT-B defaults — model runs in VAE latent space (32x32x4)
    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--patch_size', type=int, default=2)
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--num_heads', type=int, default=12)
    p.add_argument('--mlp_ratio', type=float, default=4.0)
    p.add_argument('--num_classes', type=int, default=1)
    p.add_argument('--image_size', type=int, default=32)    # latent space
    p.add_argument('--image_channels', type=int, default=4) # latent channels
    p.add_argument('--train_type', default='shortcut')      # short-cut model
    p.add_argument('--force_dt0', action='store_true', default=False,
                   help='Force dt=0 at every step (approximate Flow Matching baseline)')   # flow-matching mimic model
    p.add_argument('--decode_vae', action='store_true', default=True,
                   help='Decode latent to pixel image via StableVAE')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)

    print(f"JAX devices: {jax.devices()}")
    print(f"Loading checkpoint: {args.load_dir}")
    print(f"Steps: {args.num_steps}, Batch: {args.batch_size}")

    # ── Build model ──────────────────────────────────────────────
    model = DiT(
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        out_channels=args.image_channels,
        class_dropout_prob=1.0,   # unconditional (CelebA)
        num_classes=args.num_classes,
        dropout=0.0,
    )

    obs_shape = (1, args.image_size, args.image_size, args.image_channels)
    rng, init_rng = jax.random.split(rng)
    dummy_img = jnp.zeros(obs_shape)
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

    # ── Setup sampling constants ──────────────────────────────────
    def call_model(params, x, t, dt, labels):
        return model.apply({'params': params}, x, t, dt, labels, train=False)

    denoise_timesteps = args.num_steps
    delta_t = 1.0 / denoise_timesteps
    if args.force_dt0:
        dt_flow = 0
        print("NOTE: force_dt0=True → dt=0 (approximate Flow Matching baseline)")
    else:
        dt_flow = int(np.log2(denoise_timesteps)) if denoise_timesteps > 1 else 0
    dt_base = jnp.ones((args.batch_size,), dtype=jnp.int32) * dt_flow
    labels = jnp.ones((args.batch_size,), dtype=jnp.int32) * args.num_classes  # null token

    total_samples = args.batch_size * args.num_batches
    print(f"Running {denoise_timesteps}-step Euler sampling...")
    print(f"Batches: {args.num_batches} × batch_size {args.batch_size} = {total_samples} samples total")

    # ── Load VAE once before batch loop ──────────────────────────
    if args.decode_vae:
        from utils.stable_vae import StableVAE
        print("Loading StableVAE decoder...")
        vae = StableVAE.create()
        vae_decode = jax.jit(vae.decode)

    # ── Batch loop ────────────────────────────────────────────────
    global_idx = 0
    for batch_idx in range(args.num_batches):
        rng, noise_rng = jax.random.split(rng)
        x = jax.random.normal(noise_rng, (args.batch_size, args.image_size, args.image_size, args.image_channels))

        for ti in range(denoise_timesteps):
            t_val = ti / denoise_timesteps
            t_vec = jnp.full((args.batch_size,), t_val)
            v = call_model(train_state.params_ema, x, t_vec, dt_base, labels)
            x = x + v * delta_t

        # Decode
        if args.decode_vae:
            x_pixel = vae_decode(x)  # → (batch, 256, 256, 3) in [-1, 1]
            x_np = np.array(x_pixel)
        else:
            x_np = np.array(x)

        x_np = np.clip(x_np * 0.5 + 0.5, 0, 1)
        x_np = (x_np * 255).astype(np.uint8)

        for img in x_np:
            imageio.imwrite(os.path.join(args.save_dir, f'sample_{global_idx:05d}.png'), img)
            global_idx += 1

        print(f"  batch {batch_idx+1}/{args.num_batches} — saved {global_idx}/{total_samples} images")

    print(f"\nDone! Saved {global_idx} images to: {args.save_dir}")


if __name__ == '__main__':
    main()
