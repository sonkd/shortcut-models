# Experiment Log — Shortcut Models
**Môn học:** Statistical Machine Learning
**Researcher:** Núi
**Paper:** Frans et al., ICLR 2025

> **Quy tắc ghi log:**
> - Mỗi run = 1 entry riêng biệt
> - Ghi TRƯỚC khi chạy: hypothesis + config
> - Ghi SAU khi chạy: result + observation
> - Không xóa entry cũ — chỉ thêm ghi chú "SUPERSEDED BY EXP-XXX"

---

## [TEMPLATE] — Copy mỗi khi tạo run mới

```
### EXP-XXX — [Tên ngắn gọn]
**Date:** YYYY-MM-DD HH:MM
**Phase:** [Setup / Baseline / Ablation / Inference]
**Status:** [ ] Running | [ ] Completed | [ ] Failed

#### Config
- Model: [DiT-B / DiT-XL]
- Dataset: [CelebA / ImageNet-256]
- train_type: [shortcut / naive]
- num_steps (inference): [1 / 4 / 16 / 128]
- Checkpoint: [path hoặc "pretrained"]
- Batch size:
- Random seed:
- Hardware: [GPU model, RAM]
- Commit hash (nếu có):

#### Hypothesis trước khi chạy
> "Tôi kỳ vọng... vì..."

#### Command chạy
\```bash
python ...
\```

#### Kết quả
| Metric | Value | Paper Target | Delta |
|--------|-------|--------------|-------|
| FID-50k | | | |
| IS | | | |
| Latency (ms/batch) | | | |

#### Observations
- [ ] Sample ảnh nhìn hợp lý?
- [ ] Training loss ổn định?
- [ ] Có artifacts rõ ràng không?

**Ghi chú định tính:**
>

#### Kết luận & Next step
>
```

---

## Runs

### EXP-001 — Environment Check & Inference Sanity Test
**Date:** ___________
**Phase:** Setup
**Status:** [ ] Running | [ ] Completed | [ ] Failed

#### Config
- Model: DiT-B
- Dataset: CelebA
- train_type: shortcut (pretrained)
- num_steps (inference): 128
- Checkpoint: pretrained (Google Drive)
- Hardware: ___________
- Random seed: 42

#### Hypothesis trước khi chạy
> "Với pretrained checkpoint, inference 128-step trên CelebA sẽ đạt FID ≈ 6.9 như trong paper."

#### Command chạy
```bash
# Kiểm tra JAX
python -c "import jax; print(jax.devices())"

# Inference + FID eval
python eval_fid.py \
  --load_dir <checkpoint_path> \
  --num_steps 128 \
  --fid_stats data/celeba256_fidstats_ours.npz \
  --seed 42
```

#### Kết quả
| Metric | Value | Paper Target | Delta |
|--------|-------|--------------|-------|
| FID-50k | | 6.9 | |
| Latency (ms/batch) | | — | |

#### Observations
- [ ] Sample ảnh nhìn hợp lý?
- [ ] JAX thiết bị được nhận diện đúng?

**Ghi chú:**
>

---

### EXP-002 — Reproduce 4-Step FID (CelebA)
**Date:** ___________
**Phase:** Baseline
**Status:** [ ] Running | [ ] Completed | [ ] Failed

#### Config
- Model: DiT-B / Dataset: CelebA
- num_steps: **4**
- Checkpoint: pretrained (same as EXP-001)
- Seed: 42

#### Hypothesis
> "4-step inference sẽ đạt FID ≈ 13.8, tức tệ hơn ~2× so với 128-step (6.9), thể hiện rằng shortcut models duy trì chất lượng tốt ở few-step."

#### Command
```bash
python eval_fid.py --load_dir <checkpoint_path> --num_steps 4 --seed 42
```

#### Kết quả
| Metric | Value | Paper Target | Delta |
|--------|-------|--------------|-------|
| FID-50k | | 13.8 | |

#### Observations
>

---

### EXP-003 — Reproduce 1-Step FID (CelebA)
**Date:** ___________
**Phase:** Baseline
**Status:** [ ] Running | [ ] Completed | [ ] Failed

#### Config
- num_steps: **1**
- Checkpoint: pretrained

#### Hypothesis
> "1-step inference sẽ đạt FID ≈ 20.5. Dù chất lượng giảm, vẫn tốt hơn nhiều so với naive 1-step diffusion (FID >> 100)."

#### Command
```bash
python eval_fid.py --load_dir <checkpoint_path> --num_steps 1 --seed 42
```

#### Kết quả
| Metric | Value | Paper Target | Delta |
|--------|-------|--------------|-------|
| FID-50k | | 20.5 | |

---

### EXP-004 — FID vs. NFE Sweep (Inference Curve)
**Date:** ___________
**Phase:** Inference Investigation
**Status:** [ ] Running | [ ] Completed | [ ] Failed

#### Config
- NFE sweep: [1, 2, 4, 8, 16, 32, 64, 128]
- Checkpoint: pretrained CelebA
- Seed: 42

#### Hypothesis
> "FID sẽ giảm theo log-scale khi tăng NFE. Gain lớn nhất xảy ra ở 1→4 bước, sau đó diminishing returns."

#### Command
```bash
for steps in 1 2 4 8 16 32 64 128; do
  python eval_fid.py --load_dir <ckpt> --num_steps $steps --seed 42 \
    | tee -a logs/fid_nfe_sweep.txt
done
```

#### Kết quả
| NFE | FID | Notes |
|-----|-----|-------|
| 1 | | |
| 2 | | |
| 4 | | |
| 8 | | |
| 16 | | |
| 32 | | |
| 64 | | |
| 128 | | |

#### Visualization
> Sau khi có số liệu, plot FID vs NFE (log scale x-axis) và lưu vào `plots/fid_nfe_curve.png`

---

### EXP-005 — Seed Variance Analysis
**Date:** ___________
**Phase:** Statistical Validation
**Status:** [ ] Running | [ ] Completed | [ ] Failed

#### Config
- num_steps: 4
- Seeds: [42, 123, 2025]

#### Hypothesis
> "FID variance qua seeds sẽ nhỏ (±0.5), cho thấy kết quả ổn định."

#### Kết quả
| Seed | FID |
|------|-----|
| 42 | |
| 123 | |
| 2025 | |
| **Mean ± Std** | |

---

### EXP-006 — Visual Quality Grid
**Date:** ___________
**Phase:** Inference Investigation
**Status:** [ ] Running | [ ] Completed | [ ] Failed

#### Config
- Generate 64 samples tại NFE ∈ {1, 4, 16, 128}
- Fixed seed để so sánh cùng noise

#### Command
```bash
python sample_grid.py --load_dir <ckpt> \
  --num_steps 1 --save_path plots/samples_nfe1.png

python sample_grid.py --load_dir <ckpt> \
  --num_steps 4 --save_path plots/samples_nfe4.png
# ...
```

#### Visual Observations (điền sau khi xem ảnh)

| NFE | Sharpness | Color | Artifacts | Overall |
|-----|-----------|-------|-----------|---------|
| 1 | | | | /5 |
| 4 | | | | /5 |
| 16 | | | | /5 |
| 128 | | | | /5 |

---

### EXP-007 — Ablation: Naive Flow vs. Shortcut (Optional, nếu có GPU đủ)
**Date:** ___________
**Phase:** Ablation
**Status:** [ ] Running | [ ] Completed | [ ] Failed

#### Config
- E1: train_type=`shortcut` (pretrained)
- E2: train_type=`naive` (train mới hoặc nếu có checkpoint)

#### Kết quả Comparison
| Model | NFE=1 | NFE=4 | NFE=128 |
|-------|-------|-------|---------|
| Shortcut | | | |
| Naive Flow | | | |
| Delta | | | |

---

## Summary Table (cập nhật liên tục)

| EXP | Date | Dataset | NFE | FID | Status |
|-----|------|---------|-----|-----|--------|
| 001 | | CelebA | 128 | | |
| 002 | | CelebA | 4 | | |
| 003 | | CelebA | 1 | | |
| 004 | | CelebA | sweep | | |
| 005 | | CelebA | 4 | mean±std | |
| 006 | | CelebA | visual | N/A | |
| 007 | | CelebA | 1,4,128 | | |

---

*Cập nhật lần cuối: 2026-04-01*
