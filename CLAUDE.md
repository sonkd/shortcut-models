# CLAUDE.md — Shortcut Models Project

## Quy tắc quan trọng

### study-zone isolation
**Tất cả output thí nghiệm PHẢI đặt trong thư mục `study-zone/`**, không được ghi vào repo gốc.

- Checkpoints, samples, FID results, logs → `study-zone/`
- Script tùy chỉnh hoặc notebook thí nghiệm → `study-zone/`
- Không sửa đổi code gốc (train.py, model.py, ...) trực tiếp — copy vào study-zone nếu cần thay đổi

Mục đích: giữ nguyên trạng repo gốc `kvfrans/shortcut-models` để có thể so sánh và pull update.

## Cấu trúc study-zone

```
study-zone/
├── checkpoints/      # pretrained + finetuned checkpoints
├── samples/          # generated image samples
├── fid_results/      # FID evaluation outputs
├── logs/             # training/eval logs
├── scripts/          # custom scripts (copies/modifications)
└── notebooks/        # experiment notebooks
```

## Môi trường

- **Venv** (thay conda — conda không có trên máy): `study-zone/.venv/`
- Activate: `source study-zone/.venv/bin/activate`
- Framework: JAX 0.4.23 + Flax 0.7.4 + Optax 0.1.7 + orbax-checkpoint 0.4.8
- Python: 3.12.4 (system `/usr/local/bin/python3`)
- Backend: CPU (macOS). Để dùng Metal GPU: `pip install jax-metal` (perlu test compatibility)
- Dataset: CelebA-HQ 256, Imagenet-256 (qua TFDS)

## Dataset & Checkpoints

- Pretrained checkpoints: [Google Drive](https://drive.google.com/drive/folders/1g665i0vMxm8qqqcp5mAiexnL919-gMwW?usp=sharing)
- FID stats: `data/celeba256_fidstats_ours.npz`, `data/imagenet256_fidstats_ours.npz`
- Đặt checkpoints vào `study-zone/checkpoints/`

## Sanity check FID targets (DiT-B)

| Dataset  | 128-step | 4-step | 1-step |
|----------|----------|--------|--------|
| CelebA   | 6.9      | 13.8   | 20.5   |
| ImageNet | 15.5     | 28.3   | 40.3   |
