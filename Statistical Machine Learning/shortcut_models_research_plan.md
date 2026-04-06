# Research Plan: One Step Diffusion via Shortcut Models
**Môn học:** Statistical Machine Learning
**Tác giả nghiên cứu:** Núi
**Paper gốc:** Frans et al., "One Step Diffusion via Shortcut Models", ICLR 2025
**Repo:** https://github.com/kvfrans/shortcut-models
**Ngày bắt đầu:** 2026-04-01

---

## 1. Câu hỏi nghiên cứu (Research Questions)

> "Shortcut models có thể thay thế các phương pháp tăng tốc sampling phức tạp (distillation, consistency models) bằng một quy trình đơn giản hơn mà không đánh đổi chất lượng ảnh?"

| # | Câu hỏi cụ thể | Loại câu hỏi |
|---|---|---|
| RQ1 | Conditioning trên $\Delta t$ (step size) giúp ích bao nhiêu so với chỉ conditioning trên $t$? | Ablation / Causal |
| RQ2 | FID của shortcut models tại 1-step vs. 4-step so với consistency models và reflow? | Comparative |
| RQ3 | Có trade-off gì giữa số bước inference và chất lượng ảnh (FID curve)? | Descriptive |
| RQ4 | Bootstrapping target ($F_\theta(x_t, t, d/2)$) ổn định như thế nào trong training? | Diagnostic |

**Assumption chính:**
- Giả sử phần cứng có GPU (không có TPU-v3 như paper gốc) — kết quả FID có thể lệch nhỏ do batch size / precision khác nhau.
- Dataset: bắt đầu với **CIFAR-10** hoặc **CelebA** trước, không bắt đầu từ ImageNet-256.

---

## 2. Sơ đồ tổng quan phương pháp

```
Training signal:
  ┌─────────────────────────────────────────────┐
  │  Small step (d ≈ 0):                         │
  │    F_θ(x_t, t, d) ← ODE / flow-matching      │
  │                                               │
  │  Large step (d > 0):                          │
  │    F_θ(x_t, t, d) ← F_θ(F_θ(x_t, t, d/2),   │
  │                          t+d/2, d/2)          │
  │    (self-consistency bootstrapping)           │
  └─────────────────────────────────────────────┘

Loss:
  L = E_{t, d, x_0, ε} [ || F_θ(x_{t+d}, t, d) - x_t ||² ]

Inference:
  x_0 ← F_θ(x_1, 1, Δt) → F_θ(x_{1-Δt}, 1-Δt, Δt) → ... (N steps)
  Đặc biệt: N=1 → F_θ(x_1, 1, 1) = x_0 trực tiếp
```

---

## 3. Kế hoạch thí nghiệm — 5 Phases

### Phase 0: Environment Setup (Ngày 1–2)

**Mục tiêu:** Môi trường hoạt động, dataset sẵn sàng.

- [ ] Clone repo: `git clone https://github.com/kvfrans/shortcut-models`
- [ ] Cài môi trường: `conda env create -f environment.yml && pip install -r requirements.txt`
- [ ] Kiểm tra JAX + GPU/TPU availability:
  ```python
  import jax
  print(jax.devices())  # Phải thấy GPU hoặc CPU
  ```
- [ ] Download pretrained checkpoints (Google Drive từ README)
- [ ] Download FID stats: `data/celeba256_fidstats_ours.npz`
- [ ] Verify inference chạy được với pretrained checkpoint:
  ```bash
  python sample.py --load_dir <checkpoint_path> --num_steps 1
  ```

**Checklist xác nhận:**
- [ ] Sample ảnh 1-step chạy được và nhìn hợp lý
- [ ] FID eval script chạy được với `--num_steps 128` → kết quả gần với 6.9 (CelebA)

---

### Phase 1: Baseline Reproduction (Ngày 3–7)

**Mục tiêu:** Tái hiện kết quả FID từ bảng trong README.

**Cấu hình chạy — CelebA (DiT-B):**
```bash
# Inference với pretrained checkpoint
python eval_fid.py \
  --load_dir <checkpoint_dir> \
  --num_steps 128 \
  --fid_stats data/celeba256_fidstats_ours.npz
# Lặp lại với --num_steps 4, 1
```

**Kết quả kỳ vọng (benchmark từ paper):**

| num_steps | FID kỳ vọng | FID thực đo | Delta | Ghi chú |
|-----------|-------------|-------------|-------|---------|
| 128       | 6.9         |             |       |         |
| 4         | 13.8        |             |       |         |
| 1         | 20.5        |             |       |         |

> **Tiêu chí chấp nhận (Definition of Done):** FID reproduce trong khoảng ±1.5 so với paper.
> Nếu lệch >2 → ghi nhận lý do (hardware, random seed, batch size).

---

### Phase 2: Ablation Study (Ngày 8–15)

**Mục tiêu:** Cô lập đóng góp của việc conditioning trên $\Delta t$.

**Biến thể cần chạy:**

| Experiment | train_type | Mô tả | Giả thuyết |
|---|---|---|---|
| E1 | `shortcut` | Full shortcut model (conditional on d) | Baseline shortcut |
| E2 | `naive` | Flow-matching không có shortcut | FID tốt ở nhiều bước, kém ở ít bước |
| E3 | `shortcut` (fixed d) | Shortcut train với d cố định | Kém linh hoạt hơn E1 |

```bash
# E2: Train naive flow model
python train.py --model.train_type naive \
  --dataset_name celebahq256 \
  --max_steps 410_000 \
  ...same arch params...

# E1 đã có từ pretrained — dùng checkpoint
```

**Metric đo tại mỗi step budget {1, 2, 4, 8, 16, 32, 64, 128}:**
- FID-50k (chính)
- Inception Score (bổ sung)
- Thời gian inference per batch (ms)

---

### Phase 3: Inference Investigation (Ngày 14–18)

**Mục tiêu:** Hiểu hành vi mô hình khi thay đổi số bước.

**3.1 FID vs. NFE (Number of Function Evaluations) curve:**
```python
nfe_list = [1, 2, 4, 8, 16, 32, 64, 128]
fid_results = {}
for nfe in nfe_list:
    fid = eval_fid(checkpoint, num_steps=nfe)
    fid_results[nfe] = fid
# Plot: x=NFE (log scale), y=FID
```

**3.2 Visual quality inspection:**
- Generate grid ảnh tại mỗi NFE: 1, 4, 16, 128
- So sánh visual artifacts (blurring, color shift, structure collapse)
- Log quan sát định tính vào experiment log

**3.3 Trajectory visualization:**
- Với 1 ảnh cố định, visualize intermediate states khi dùng 4-step inference
- Kiểm tra xem "shortcut" có thực sự nhảy qua denoising path không

**3.4 Sensitivity analysis:**
- Thay đổi random seed (3 seeds: 42, 123, 2025) → đo variance của FID
- Báo cáo: FID mean ± std

---

### Phase 4: Statistical Analysis & Thesis Writing (Ngày 19–25)

**Mục tiêu:** Chuyển kết quả thực nghiệm thành luận văn.

- [ ] Tổng hợp bảng kết quả đầy đủ
- [ ] Statistical testing: dùng bootstrap confidence interval cho FID differences
- [ ] Viết methodology section (dựa trên phần toán ở trên)
- [ ] Viết results section với phân tích
- [ ] Critical evaluation: limitations của shortcut models

---

## 4. Rủi ro & Phương án dự phòng

| Rủi ro | Xác suất | Mitigation |
|---|---|---|
| JAX không tương thích GPU cá nhân | Cao | Dùng Google Colab TPU (miễn phí) hoặc port sang PyTorch |
| Không đủ VRAM để train | Cao | Giảm batch size, dùng pretrained checkpoint thay vì train từ đầu |
| FID không reproduce được | Trung bình | Ghi nhận delta và giải thích (hardware, seed) |
| Training mất quá nhiều thời gian | Cao | Tập trung vào inference investigation với pretrained weights |

**Khuyến nghị thực tế:** Với resource hạn chế → ưu tiên **Phase 1 + 3** (inference investigation với pretrained model) thay vì train lại từ đầu.

---

## 5. Tài nguyên tính toán ước tính

| Task | GPU (V100) | Thời gian ước tính |
|---|---|---|
| Train CelebA DiT-B (410k steps) | 8× V100 | ~48 giờ |
| Eval FID-50k (1 lần) | 1× V100 | ~30 phút |
| Inference investigation (tất cả NFE) | 1× V100 | ~3 giờ |
| **Tổng (inference only)** | **1× V100** | **~4 giờ** |

---

*Last updated: 2026-04-01*
