"""check_env.py — Phase 0 environment verification script.
Chạy: python study-zone/scripts/check_env.py
"""
import sys

def check(label, fn):
    try:
        result = fn()
        print(f"  [OK]  {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False

print(f"\n{'='*50}")
print("Shortcut Models — Environment Check")
print(f"{'='*50}")

print("\n[1] Python")
check("version", lambda: sys.version)

print("\n[2] JAX")
ok_jax = check("import jax", lambda: __import__("jax").__version__)
if ok_jax:
    import jax
    check("devices", lambda: jax.devices())
    check("simple op", lambda: jax.numpy.array([1,2,3]).sum())

print("\n[3] Flax / Optax / Orbax")
check("flax", lambda: __import__("flax").__version__)
check("optax", lambda: __import__("optax").__version__)
check("orbax", lambda: __import__("orbax.checkpoint", fromlist=["checkpoint"]) and "OK")

print("\n[4] Other ML deps")
check("einops", lambda: __import__("einops").__version__)
check("ml_collections", lambda: __import__("ml_collections").__version__)
check("diffusers", lambda: __import__("diffusers").__version__)
check("tensorflow", lambda: __import__("tensorflow").__version__)

print("\n[5] FID stats")
import os
repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
fid_celeba = os.path.join(repo_root, "data", "celeba256_fidstats_ours.npz")
fid_imgnet = os.path.join(repo_root, "data", "imagenet256_fidstats_ours.npz")
check("celeba256 FID stats", lambda: "found" if os.path.exists(fid_celeba) else (_ for _ in ()).throw(FileNotFoundError(fid_celeba)))
check("imagenet256 FID stats", lambda: "found" if os.path.exists(fid_imgnet) else (_ for _ in ()).throw(FileNotFoundError(fid_imgnet)))

print("\n[6] Checkpoints in study-zone")
ckpt_dir = os.path.join(repo_root, "study-zone", "checkpoints")
ckpts = [f for f in os.listdir(ckpt_dir) if not f.startswith(".")]
check("checkpoints dir", lambda: f"{len(ckpts)} item(s): {ckpts[:3]}" if ckpts else (_ for _ in ()).throw(FileNotFoundError("No checkpoints found — download from Google Drive")))

print(f"\n{'='*50}\n")
