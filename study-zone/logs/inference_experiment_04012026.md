# Inference Experiment Log — Shortcut Models (CelebA)

**Ngày thực hiện:** 2026-04-01 → 2026-04-02  
**Người thực hiện:** tieunui  
**Repo:** `/Users/tieunui/Data science/shortcut-models/`  
**Môi trường:** macOS ARM, CPU-only, Python 3.12.4

---

## 1. Mục tiêu

Chạy inference pretrained Shortcut Model trên CelebA để:

1. Quan sát chất lượng sinh ảnh tại các mức step khác nhau: **1-step, 4-step, 128-step**
2. So sánh với **approximate Flow Matching baseline** (forced `dt=0`) trên cùng checkpoint
3. Xác nhận đặc tính cốt lõi của Shortcut Models: *1 model duy nhất hỗ trợ nhiều mức step*

---

## 2. Cài đặt môi trường

### 2.1 Tạo virtualenv

```bash
cd "/Users/tieunui/Data science/shortcut-models"
python3 -m venv study-zone/.venv
source study-zone/.venv/bin/activate
```

> **Lưu ý:** Máy không có conda → dùng venv. Conda không nên cài thêm vì có thể conflict ARM path.

### 2.2 Package versions (pinned)

| Package | Version | Ghi chú |
|---------|---------|---------|
| jax | 0.4.23 | Không dùng 0.7+, sẽ break flax/optax |
| jaxlib | 0.4.23 | Phải match jax |
| flax | 0.7.4 | |
| optax | 0.1.7 | |
| orbax-checkpoint | 0.4.8 | Không dùng 0.11+, cần JAX ≥0.6.0 |
| tensorflow | 2.21.0 | `tensorflow-cpu` không có cho Python 3.12 ARM |
| diffusers | latest | Dùng để load StableVAE |
| typeguard | latest | Dependency của `stable_vae.py` |

```bash
pip install "jax[cpu]==0.4.23" "jaxlib==0.4.23"
pip install flax==0.7.4 optax==0.1.7 "orbax-checkpoint==0.4.8"
pip install tensorflow einops ml-collections diffusers transformers typeguard imageio
```

### 2.3 Checkpoint

Download từ Google Drive: `https://drive.google.com/drive/folders/1g665i0vMxm8qqqcp5mAiexnL919-gMwW`

Lưu tại: `study-zone/checkpoints/celeba-shortcut2-every4400001`  
Kích thước: ~2 GB, format pickle (không phải orbax).

---

## 3. Script inference

### 3.1 Lý do tạo standalone script

Repo gốc (`train.py --mode inference`) yêu cầu TFDS dataset `celebahq256` — đây là **custom builder** từ repo phụ `kvfrans/tfds_builders`, không có trong TFDS chuẩn. Thay vì cài thêm, tạo script độc lập sinh noise trực tiếp, bỏ qua dataset.

Script: `study-zone/scripts/run_inference_standalone.py`

### 3.2 Cấu hình model (DiT-B)

```python
model = DiT(
    patch_size=2,
    hidden_size=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    out_channels=4,          # latent channels, không phải RGB
    class_dropout_prob=1.0,  # unconditional (CelebA không có label)
    num_classes=1,
    dropout=0.0,
)
```

**Không gian làm việc:** latent space `32×32×4` (không phải pixel `256×256`). VAE decode ra ảnh cuối.

### 3.3 Euler sampling

```python
dt_flow = int(np.log2(denoise_timesteps)) if denoise_timesteps > 1 else 0
dt_base = jnp.ones((batch_size,), dtype=jnp.int32) * dt_flow

for ti in range(denoise_timesteps):
    t_val = ti / denoise_timesteps
    t_vec = jnp.full((batch_size,), t_val)
    v = model.apply({'params': params_ema}, x, t_vec, dt_base, labels, train=False)
    x = x + v * delta_t
```

`dt_flow = log2(steps)` là shortcut conditioning — cho model biết đang dùng bước nhảy lớn.

### 3.4 VAE decode

```python
vae = StableVAE.create()   # tải 'pcuenq/sd-vae-ft-mse-flax' từ HuggingFace
x_pixel = vae.decode(x)    # (batch, 256, 256, 3) trong [-1, 1]
x_np = np.clip(x_np * 0.5 + 0.5, 0, 1)  # → [0, 1]
```

---

## 4. Các sửa đổi và lưu ý trong quá trình inference

### 4.1 Lỗi JAX version conflict

**Error:** `AttributeError: module 'jax.core' has no attribute 'Shape'`  
**Nguyên nhân:** Cài JAX 0.7.x, không tương thích với flax 0.7.4 và optax 0.1.7.  
**Fix:** Downgrade `jax==0.4.23` + `jaxlib==0.4.23`.

### 4.2 Lỗi orbax-checkpoint version

**Error:** `ImportError: cannot import name 'DeviceLocalLayout'`  
**Nguyên nhân:** orbax-checkpoint 0.11.x yêu cầu JAX ≥0.6.0.  
**Fix:** Downgrade `orbax-checkpoint==0.4.8`.

### 4.3 TrainStateEma.create() sai signature

**Sai:** `TrainStateEma.create(model.apply, params, None, rng=init_rng)`  
**Đúng:** `TrainStateEma.create(model, params, init_rng)`  
**Lý do:** Signature là `(cls, model_def, params, rng, ...)` — truyền object `model`, không phải `model.apply`.

### 4.4 Out-of-memory khi dùng jax.jit trên CPU

**Error:** Process bị kill (exit 137) khi compile JIT DiT-B.  
**Fix:** Bỏ `@jax.jit`, chạy eager mode. Chậm hơn nhưng không OOM.

### 4.5 Grid save sai kích thước

**Error:** `ValueError: could not broadcast shape (256,256,3) into (32,32,3)`  
**Nguyên nhân:** Dùng `args.image_size=32` (latent dims) cho grid, nhưng sau VAE decode ảnh là 256×256.  
**Fix:**
```python
h, w = x_np.shape[1], x_np.shape[2]  # lấy actual pixel dims
grid = np.zeros((n * h, n * w, x_np.shape[3]), dtype=np.uint8)
```

### 4.6 tensorflow-cpu không có cho Python 3.12 ARM

**Fix:** Cài `tensorflow` (full package, 2.21.0) thay vì `tensorflow-cpu`.

### 4.7 latent space vs pixel

Key insight: model hoạt động ở `32×32×4` (latent), không phải `256×256×3` (pixel).  
Cần khởi tạo:
```python
obs_shape = (1, 32, 32, 4)   # latent
dummy_img = jnp.zeros(obs_shape)
```
Nếu dùng `(1, 256, 256, 3)` → model init sai, checkpoint không load được đúng.

---

## 5. Approximate Flow Matching baseline

Để so sánh mà không cần checkpoint FM riêng, dùng flag `--force_dt0`:

```python
# Trong run_inference_standalone.py
if args.force_dt0:
    dt_flow = 0   # thay vì log2(steps)
```

Khi `dt=0`, model bỏ qua shortcut conditioning → xấp xỉ hành vi Flow Matching tiêu chuẩn.  
**Lưu ý:** Đây là *approximation* vì model được train với shortcut loss, không phải FM loss thuần túy.

---

## 6. Tổng hợp kết quả

### 6.1 Các run đã thực hiện

| Run | Steps | dt_flow | Save dir |
|-----|-------|---------|----------|
| Shortcut 1-step | 1 | 0 | `study-zone/samples/celeba_1step/` |
| Shortcut 4-step | 4 | 2 | `study-zone/samples/celeba_4step/` |
| Shortcut 128-step | 128 | 7 | `study-zone/samples/celeba_128step/` |
| FM baseline (dt=0) | 128 | 0 (forced) | `study-zone/samples/celeba_fm_baseline_128step/` |

Mỗi run: `batch_size=4`, seed=42, checkpoint `celeba-shortcut2-every4400001` (step 400000).

### 6.2 Nhận xét chất lượng

- **1-step:** Ảnh mờ, artifact rõ, khuôn mặt nhận ra được nhưng không sắc nét (team tự test bằng trực giác/survey)
- **4-step:** Cải thiện đáng kể — texture da, tóc rõ hơn
- **128-step (shortcut):** Chất lượng tốt nhất, gương mặt sắc nét, màu sắc tự nhiên
- **128-step (FM baseline, dt=0):** Chất lượng thấp hơn 128-step shortcut — xác nhận shortcut conditioning giúp model sử dụng tốt hơn "bước nhảy lớn" trong trajectory

### 6.3 Kết luận

Kết quả nhất quán với paper: Shortcut model với `dt=log2(steps)` luôn tốt hơn hoặc bằng nhiều steps nhỏ hơn. Đặc biệt, 4-step shortcut đã đạt chất lượng gần 128-step, chứng minh hiệu quả của shortcut conditioning trong việc nén số bước suy diễn.

---

## 7. Lệnh chạy tham khảo

```bash
cd "/Users/tieunui/Data science/shortcut-models"
source study-zone/.venv/bin/activate

# Shortcut: 1, 4, 128 steps
python study-zone/scripts/run_inference_standalone.py \
    --load_dir study-zone/checkpoints/celeba-shortcut2-every4400001 \
    --num_steps 1 --batch_size 4 \
    --save_dir study-zone/samples/celeba_1step

# FM baseline (128 steps, dt=0 forced)
python study-zone/scripts/run_inference_standalone.py \
    --load_dir study-zone/checkpoints/celeba-shortcut2-every4400001 \
    --num_steps 128 --batch_size 4 \
    --save_dir study-zone/samples/celeba_fm_baseline_128step \
    --force_dt0
```
