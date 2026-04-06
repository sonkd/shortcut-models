# Thesis Notes — Shortcut Models
## Dữ kiện quan trọng cần ghi lại cho bài luận Thống kê Học Máy

**Paper:** "One Step Diffusion via Shortcut Models", Frans et al., ICLR 2025
**Loại bài toán:** Generative modeling — accelerated sampling
**Keyword:** Diffusion models, flow-matching, few-step generation, self-consistency

---

## Phần 1: Nền tảng lý thuyết cần trình bày

### 1.1 Ký hiệu & Không gian bài toán

```
x_0 ~ p_data          : dữ liệu thực (ảnh sạch)
x_t = α_t·x_0 + σ_t·ε : dữ liệu noisy tại thời điểm t ∈ [0,1]
ε ~ N(0,I)            : Gaussian noise
t ∈ [0,1]             : noise level (t=1: pure noise, t=0: clean)
d ∈ [0,1]             : step size (độ dài "nhảy cóc")
F_θ(x_t, t, d)        : shortcut network output tại (t, d)
```

> **⚠️ Phân biệt notation:** Một số paper dùng $t$ tăng dần (t=0 là clean), paper này dùng $t$ giảm dần trong inference. Cần nhất quán trong thesis.

### 1.2 Objective function — ghi rõ 2 thành phần

**Component 1 — Flow-matching loss (small d):**
$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, x_0, \varepsilon}\left[\| F_\theta(x_t, t, 0) - v_t \|^2\right]$$

Trong đó $v_t = \frac{d x_t}{dt}$ là velocity target theo Conditional Flow Matching (Lipman et al., 2022).

**Component 2 — Shortcut self-consistency loss (large d):**
$$\mathcal{L}_{\text{shortcut}} = \mathbb{E}_{t, d, x_0, \varepsilon}\left[\| F_\theta(x_t, t, d) - \text{sg}[F_\theta(F_\theta(x_t, t, d/2), t\!-\!d/2, d/2)] \|^2\right]$$

Trong đó $\text{sg}[\cdot]$ là **stop-gradient** (không backprop qua target).

**Combined loss:**
$$\mathcal{L} = \mathcal{L}_{\text{flow}} + \lambda \cdot \mathcal{L}_{\text{shortcut}}$$

> **Điểm cần lý giải trong thesis:**
> - Tại sao stop-gradient là cần thiết? (Ngăn bootstrapping collapse)
> - Liên hệ với TD-learning trong Reinforcement Learning (cùng cấu trúc)

### 1.3 Tính chất Self-Consistency — cần chứng minh/phát biểu

**Self-consistency property:**
$$F_\theta(x_t, t, d) \approx F_\theta\left(F_\theta(x_t, t, \tfrac{d}{2}),\ t - \tfrac{d}{2},\ \tfrac{d}{2}\right)$$

Nếu tính chất này thỏa, model có thể ghép bất kỳ số bước inference nào mà vẫn nhất quán.

> **Câu hỏi pedagogical:** Tính chất này được học hay được enforce? Ý nghĩa là gì nếu nó không hoàn toàn thỏa?

### 1.4 Liên hệ với các phương pháp liên quan

| Method | Số network | Số training phase | Flexible NFE | Cơ chế |
|--------|-----------|-------------------|--------------|--------|
| DDPM | 1 | 1 | ✓ (chậm) | Score matching |
| DDIM | 1 | 1 | ✓ | Deterministic ODE |
| Consistency Models | 1 | 1 (CT) hoặc 2 (CD) | Hạn chế | Self-consistency t→0 |
| Reflow / Rectified Flow | 1+ | 2+ | Hạn chế | Straightening trajectory |
| Distillation | 2 | 2 | Hạn chế | Teacher-student |
| **Shortcut Models** | **1** | **1** | **✓ (any d)** | **Conditional on d** |

> **Điểm novel cần nhấn mạnh:** Shortcut = Consistency Models nhưng không giới hạn target là $x_0$; thay vào đó target là $x_{t-d}$ với $d$ tùy ý.

---

## Phần 2: Kết quả thực nghiệm cần ghi lại

### 2.1 Target FID từ paper (dùng làm baseline so sánh)

| Model | 1-Step | 4-Step | 128-Step | Dataset |
|-------|--------|--------|----------|---------|
| CelebA (DiT-B) | 20.5 | 13.8 | 6.9 | CelebA-HQ 256 |
| ImageNet-256 (DiT-B) | 40.3 | 28.3 | 15.5 | ImageNet 256×256 |
| ImageNet-256 (DiT-XL) | 10.6 | 7.8 | 3.8 | ImageNet 256×256 |

> **Lưu ý quan trọng:** FID được đo trên **50,000 samples** (FID-50k). Đây là tiêu chuẩn cộng đồng — cần báo cáo đúng số này, không dùng FID-10k.

### 2.2 Kết quả của bạn (điền sau khi chạy experiment)

| EXP | NFE | FID (measured) | Delta vs. paper | Lý giải lệch |
|-----|-----|----------------|-----------------|--------------|
| 001 | 128 | | | |
| 002 | 4 | | | |
| 003 | 1 | | | |

### 2.3 FID vs. NFE curve (điền sau EXP-004)

| NFE | FID | % improvement từ NFE-1 |
|-----|-----|------------------------|
| 1 | | 0% |
| 2 | | |
| 4 | | |
| 8 | | |
| 16 | | |
| 32 | | |
| 64 | | |
| 128 | | |

### 2.4 Statistical reporting — bắt buộc cho thesis

Khi báo cáo FID, **luôn** kèm theo:
- **Số sample:** n = 50,000
- **Số seeds:** ≥ 3
- **Format:** FID = [mean] ± [std]
- **Hardware:** GPU/TPU type (ảnh hưởng đến numerical precision)

Ví dụ đúng: `FID-50k = 20.7 ± 0.3 (n=3 seeds, V100 GPU)`
Ví dụ sai: `FID = 20.5`

---

## Phần 3: Phân tích kỹ thuật cần viết trong thesis

### 3.1 Convergence & Training Stability

**Cần ghi lại trong training log (nếu train):**
- Loss curve (flow loss + shortcut loss riêng biệt)
- FID eval tại checkpoints trung gian (e.g., mỗi 50k steps)
- Gradient norm → phát hiện instability

**Quan sát cần lý giải:**
- Khi nào shortcut loss bắt đầu có signal hữu ích? (sớm hay muộn trong training?)
- EMA weight có quan trọng không? (paper dùng EMA cho target network)

### 3.2 Bootstrapping Target — Phân tích "Deadly Triad"

Stop-gradient trong shortcut loss tạo ra một dạng **off-policy bootstrapping** tương tự Q-learning trong RL. Cần lý giải:

1. **Điều kiện ổn định:** Target phải "lag" so với online network → EMA giúp ổn định
2. **Deadly triad risk:** Bootstrapping + Function approximation + Off-policy data = divergence risk
3. **Tại sao shortcut models tránh được?** Vì target luôn bounded (trong image space, không unbounded như Q-values)

> **Phân biệt:** Đây là *observation* (ghi nhận pattern), không phải *claim có chứng minh*. Ghi rõ trong thesis: "we observe... which is consistent with..."

### 3.3 Inference Quality Analysis

**Artifact taxonomy** — phân loại lỗi theo NFE:
- **NFE=1:** Color shift, global structure correct nhưng texture flat
- **NFE=2-4:** Texture improve, có thể thấy ghosting artifacts
- **NFE=8+:** Artifact giảm, gần như visual quality ≈ 128-step

> Ghi quan sát này khi chạy EXP-006, với reference ảnh cụ thể.

### 3.4 Comparison Framework — Cách viết so sánh công bằng

Khi so sánh với baseline trong thesis, phải nêu rõ:

| Điều kiện | Shortcut | Consistency Model | Reflow |
|-----------|----------|-------------------|--------|
| Training data | CelebA | CelebA | CelebA |
| Architecture | DiT-B | DiT-B | DiT-B |
| Training steps | 410k | 410k | ? |
| NFE | 1 | 1 | 1 |
| FID | 20.5 | ? | ? |

> **Warning:** So sánh không cùng architecture/training budget là không công bằng. Nếu không tìm được baseline dùng DiT-B trên CelebA, phải ghi rõ limitation này.

---

## Phần 4: Limitation phải khai báo trong thesis

### 4.1 Limitation của paper gốc (phân tích độc lập)

1. **No ablation on d-distribution:** Paper không báo cáo phân phối $d$ nào trong training là tối ưu (uniform? log-uniform?). Đây là missing experiment.

2. **Convergence bounds không tight:** Self-consistency property được chứng minh chỉ tại optimal (khi $F_\theta = F^*$). Không có bound về tốc độ hội tụ trong training.

3. **Dataset scope:** Kết quả chính chỉ trên CelebA và ImageNet-256. Chưa kiểm chứng trên text-to-image, video, hay 3D generation.

4. **Comparison gap:** Paper không so sánh trực tiếp với Consistency Models dùng cùng architecture trên cùng dataset.

### 4.2 Limitation của reproduction của bạn

- Hardware khác (V100 vs. TPU-v3) → precision khác → FID có thể lệch nhỏ
- Không train từ đầu (nếu chỉ dùng pretrained) → không thể kiểm chứng training stability
- Sample size cho seed variance: 3 seeds ít hơn lý tưởng (nên ≥ 5)

---

## Phần 5: Structure đề xuất cho bài luận

```
1. Introduction (1-1.5 trang)
   - Bài toán: sample quality vs. sampling speed trade-off
   - Prior work limitations (3 câu)
   - Contribution: shortcut models giải quyết bằng cách nào

2. Background & Related Work (1-2 trang)
   - Diffusion models + score matching
   - Flow Matching (Lipman 2022, Albergo 2023)
   - Consistency Models (Song 2023)
   - Bảng so sánh (dùng bảng 1.4 ở trên)

3. Methodology (2-3 trang)
   - Ký hiệu (Section 1.1)
   - Objective function + derivation (Section 1.2)
   - Self-consistency property + lý giải (Section 1.3)
   - Architecture: conditioning trên d như thế nào?

4. Experiments & Results (2-3 trang)
   - Setup: dataset, metric, hardware
   - Baseline reproduction (FID table)
   - FID vs. NFE curve + phân tích
   - Visual quality analysis
   - Statistical reporting (mean ± std)

5. Analysis & Discussion (1-2 trang)
   - Training stability observations
   - Bootstrapping target analysis
   - Artifact taxonomy
   - Comparison với baselines

6. Limitations & Future Work (0.5-1 trang)
   - Paper limitations
   - Reproduction limitations
   - Hướng mở rộng (conditional generation, video, text-to-image)

7. Conclusion (0.5 trang)
   - 3-4 câu summary
   - 1 câu về ý nghĩa rộng hơn
```

---

## Phần 6: Câu hỏi dẫn dắt (để tự kiểm tra hiểu biết)

1. Tại sao conditioning trên $d$ cần thiết? Tại sao không chỉ dùng multi-step DDIM?
2. Stop-gradient ngăn điều gì cụ thể? Điều gì xảy ra nếu bỏ stop-gradient?
3. Sự khác biệt giữa $F_\theta(x_t, t, 0)$ và score function $s_\theta(x_t, t)$ là gì?
4. Nếu FID của bạn lệch +3 so với paper, đó là do model hay do evaluation protocol?
5. Tại sao FID không đủ — cần thêm Precision/Recall để nói gì?

---

## Phần 7: References cần cite

```bibtex
@inproceedings{Frans2025,
  title={One Step Diffusion via Shortcut Models},
  author={Kevin Frans and Danijar Hafner and Sergey Levine and Pieter Abbeel},
  booktitle={ICLR},
  year={2025}
}

@article{Song2023,
  title={Consistency Models},
  author={Yang Song and Prafulla Dhariwal and Mark Chen and Ilya Sutskever},
  year={2023}
}

@article{Lipman2022,
  title={Flow Matching for Generative Modeling},
  author={Yaron Lipman et al.},
  year={2022}
}

@article{Liu2022,
  title={Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
  author={Xingchao Liu et al.},
  year={2022}
}

@article{Ho2020,
  title={Denoising Diffusion Probabilistic Models},
  author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
  year={2020}
}
```

---

*Last updated: 2026-04-01*
*Status: Template — cập nhật sau mỗi experiment session*
