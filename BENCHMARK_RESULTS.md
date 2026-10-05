# Camera Health Score từ mask bẩn lens: bằng chứng chung của nhóm 36Thieves

Track 4 · Day 04 · Sensor Reality Sprint · Chủ đề **T1: sức khỏe camera ADAS khi lens bẩn**

Tài liệu này chứa phần **dùng chung** của nhóm: code, lệnh chạy, số đo và plot. Mỗi thành viên viết báo cáo riêng trong [reports/](reports/) và trích số liệu từ đây. Danh sách thành viên ở [TEAMMATES.md](TEAMMATES.md).

---

## 1. Bài toán (Bước 1)

| Mục cần chốt | Nhóm chốt |
|---|---|
| Nền tảng, tính năng, sensor | Xe ADAS có hệ surround-view gồm 4 camera fisheye: FV (trước), RV (sau), MVL, MVR (gương trái/phải). Tính năng: giám sát sức khỏe camera, cho sensor fusion biết nên tin camera nào và tin bao nhiêu. |
| Failure case | Lens bị bẩn (bùn, nước, bụi). Mask chia vết bẩn thành 3 loại: transparent, semi-transparent và opaque, che một phần khung hình. |
| Claim ban đầu | Khi vết bẩn tăng coverage, đục hơn, hoặc nằm gần trung tâm/ROI, thì Camera Health Score giảm và state chuyển Healthy → Degraded → Unreliable. Score tính từ mask **dự đoán** của repo chỉ lệch vài điểm so với score tính từ mask **GT**. |
| Metric và đơn vị | Health Score (điểm, 0–100). State accuracy (%). **Missed Unreliable** (% frame GT là Unreliable nhưng hệ thống không báo Unreliable). Score MAE (điểm). Proxy ảnh: edge density trong ROI so với baseline (tỉ lệ). |
| Baseline và điều kiện lỗi | **Exp A:** score từ GT mask (baseline) so với score từ mask dự đoán của 4 model. **Exp B:** 32 frame ít bẩn nhất (baseline) so với chính các frame đó cộng thêm vết bẩn mô phỏng 10/25/40/60%. **Exp C:** pipeline v1 (chỉ dùng mask) so với v2 (mask + kiểm tra ảnh). |
| Input → output | Ảnh RGB + mask bẩn lens → Camera Health Score (0–100), Camera State (Healthy/Degraded/Unreliable), Camera Weight (1.0/0.5/0). |

## 2. Phương pháp (Bước 2)

### 2.1 Nguồn nhóm đã đọc và dùng

| Câu hỏi khi đọc nguồn | Ghi chép |
|---|---|
| Phương pháp nhận gì, tạo gì? | Repo nhận ảnh RGB WoodScape và dùng mạng segmentation (`segmentation_models_pytorch`: FPN, UNet, PAN… với encoder ResNet) để tạo mask 4 lớp 512×512: 0 clear, 1 transparent, 2 semi-transparent, 3 opaque. Xem [pytorch_l_base_pred.py](networks_run/pytorch_networks/predict/pytorch_l_base_pred.py). |
| Nguồn đo chất lượng bằng gì? | IoU, precision và recall theo pixel cho từng lớp ([evaluation.py](networks_run/evaluation/evaluation.py)), trên 39 cấu hình model. Repo **không** có bước nào đi từ mask đến quyết định về camera. |
| Chạy được ở lớp không? | Không train hay predict lại, vì cần GPU, dataset 5.5 GB và model 6.6 GB, trong khi máy chỉ còn khoảng 5 GB trống. Nhóm dùng **mask dự đoán mà repo đã lưu sẵn** (`model_outputs.zip`), tải từng phần bằng HTTP range request ([fetch_woodscape_subset.py](benchmark/fetch_woodscape_subset.py)). |
| Nhóm tái hiện phần nào? | Dùng nguyên mask dự đoán của repo. Nhóm tự phát triển phần mask → health score ([camera_health/](camera_health/)). Có 1 benchmark trên dữ liệu thật (A) và 2 benchmark mô phỏng (B, C). |
| Trích dẫn | **Paper:** F. Beránek, V. Diviš, I. Gruber, *Soiling detection for Advanced Driver Assistance Systems*, <https://arxiv.org/abs/2511.09740>. **Code gốc:** <https://github.com/filipberanek/woodscape_revision>, commit `e9dc138` (2024-08-07). Repo nhóm là bản fork: <https://github.com/vuthien3002-sys/K4-Track4-Day04-36Thieves-Sensor-Reality-Sprint>. Dataset `woodscape_input.zip` (Drive id `1WNlDBADwlaheMaVpIjEeAMklw7jle9Ja`) và model `model_outputs.zip` (id `13k17SjgQHZCO-1Ctr3DY_bW6DGvQZZie`), link lấy từ [README.md](README.md). |

Danh sách đủ 5 paper (P1 paper của repo; P2 SoilingNet; P3 TiledSoilingNet; P4 Let's Get Dirty; P5 WoodScape), kèm ghi chú đã đọc phần nào và bảng "paper cho biết / nhóm đo được", nằm trong [SOURCES.md](SOURCES.md).

### 2.2 Phần nhóm phát triển (tương ứng sơ đồ pipeline)

```
[Repo có sẵn]  WoodScape RGB ──► soiling segmentation ──► predicted mask (512×512, lớp 0..3)
                                                                │
[Nhóm làm]     5 đặc trưng: Coverage · Transparent ratio · Opaque ratio · Spatial position · Critical ROI coverage
                    ──► Soiling Severity ──► Camera Health Score 0–100 ──► Healthy / Degraded / Unreliable ──► Weight 1.0 / 0.5 / 0
```

Gọi o(c) là độ cản sáng của từng lớp: `[0, 0.33, 0.66, 1.0]`. V là vùng fisheye nhìn thấy được ([camera_health/assets/](camera_health/assets/)).

| Bước | Công thức (file) |
|---|---|
| Coverage | số pixel bẩn trong V / |V| ([features.py](camera_health/features.py)) |
| Transparent / Semi / Opaque ratio | số pixel của từng lớp / số pixel bẩn |
| Spatial position | `spatial_occlusion` = Σ G·o / Σ G, với G là Gaussian quanh tâm (σ = 0.5). Kèm theo `soiling_centroid_r` là bán kính tại trọng tâm vết bẩn. |
| Critical ROI coverage | Tính trong ROI riêng của từng camera ([config.yaml](camera_health/config.yaml)): `roi_coverage`, `roi_occlusion` = trung bình o, và `roi_opaque` |
| Soiling Severity | S = 0.3·E + 0.3·P + 0.4·R. E là trung bình o trên V, P là spatial occlusion, R là ROI occlusion ([scoring.py](camera_health/scoring.py)). |
| Health Score | 100 · (1 − S) |
| State | Healthy nếu score ≥ 80. Unreliable nếu score < 50 **hoặc** opaque chiếm ≥ 50% ROI (luật override an toàn). Còn lại là Degraded. |
| Weight | Healthy → 1.0, Degraded → 0.5, Unreliable → 0 |
| v2 (Exp C, mặc định tắt) | Thêm luật: chuyển thành Unreliable nếu edge density trong ROI < 0.2 × mức tham chiếu sạch của camera đó |

**Giả định thiết kế** (do nhóm đặt, không lấy từ chuẩn hay từ WoodScape): độ cản sáng từng lớp, σ, ROI, trọng số 0.3/0.3/0.4, ngưỡng 80/50, override 50%. Mức ảnh hưởng của ngưỡng được đo ở mục 3.1.

## 3. Benchmark (Bước 3–4)

| Điều kiện | Tham số thay đổi | Metric (đơn vị) | Bằng chứng lưu | Metric cho phép kết luận |
|---|---|---|---|---|
| A · Baseline | GT mask, 497 frame test | Score (điểm), state | [exp_a_real/log.txt](results/exp_a_real/log.txt) | Mốc tham chiếu |
| A · Lỗi | Mask dự đoán của 4 model (mIoU 0.44–0.62) | MAE, state acc, missed Unreliable | [summary_models.csv](results/exp_a_real/summary_models.csv), [scatter](results/exp_a_real/scatter_gt_vs_pred.png) | Lỗi segmentation làm sai quyết định bao nhiêu |
| B · Baseline | 32 frame sạch nhất (8 frame mỗi camera, GT coverage 3–11%) | Score, edge density ROI | [exp_b_controlled/log.txt](results/exp_b_controlled/log.txt) | Mốc của từng frame |
| B · Lỗi | Loại {transparent, opaque} × vị trí {centre, periphery} × coverage thêm {10, 25, 40, 60}%, seed 2026 | Score, state, edge density giữ lại (%) | [score_vs_coverage.png](results/exp_b_controlled/score_vs_coverage.png) | Xu hướng khi mức lỗi tăng |
| C | v1 so với v2 (kiểm tra ảnh), min_ratio ∈ {0.1…0.3} | % Unreliable, báo động sai | [image_check_tradeoff.png](results/exp_c_image_check/image_check_tradeoff.png) | Lợi ích so với chi phí của cải tiến |

### 3.1 Exp A: dữ liệu thật, GT mask so với mask dự đoán

497 frame test (FV 139, RV 129, MVR 116, MVL 113), chỉ có 399 GT mask khác nhau. State theo GT: Healthy 175, Degraded 179, Unreliable 143.

| Model của repo | mIoU | IoU transparent | Score MAE (điểm) | State acc | Missed Unreliable | GT Unreliable → Healthy | Báo động sai |
|---|---|---|---|---|---|---|---|
| FPN-R18 (correct_files) | 0.616 | 0.287 | 2.95 | 94.8% | **2.8%** (4/143) | 0% | 0% |
| FPN-R50 (all_files) | 0.617 | 0.284 | 2.72 | 94.2% | **9.1%** (13/143) | 0% | 0% |
| UNet-R18 (all_files) | 0.558 | 0.216 | 3.29 | 94.4% | 2.1% | 0% | 0% |
| PAN-R18 (strict) | 0.440 | 0.130 | 9.03 | 80.3% | 22.4% | **10.5%** (15/143) | 0% |

- **Xu hướng khi vết bẩn tăng** (FPN-R18): score MAE tăng từ 1.3 điểm (GT coverage < 10%) lên 1.7, 2.8 rồi 5.8 điểm (coverage > 60%). Score dự đoán thường thấp hơn GT (bias −1.1 điểm). Xem [summary_per_coverage.csv](results/exp_a_real/summary_per_coverage.csv).
- **Theo camera:** MVR kém nhất, với state acc 88.8% và missed Unreliable 8.6%. Xem [summary_per_camera.csv](results/exp_a_real/summary_per_camera.csv).
- **Confusion matrix** (hàng là GT, cột là dự đoán, thứ tự H/D/U): `[[174, 1, 0], [12, 158, 9], [0, 4, 139]]`. Xem [state_confusion.png](results/exp_a_real/state_confusion.png).
- **Ngưỡng:** nếu đổi ngưỡng Unreliable từ 50 lên 60 thì missed Unreliable giảm từ 2.8% xuống 0%, nhưng tỉ lệ frame bị loại tăng từ 28.8% lên 32.4%. Xem [threshold_sensitivity.csv](results/exp_a_real/threshold_sensitivity.csv).
- **Ablation:** chỉ dùng coverage cho state acc 94.2%, coverage × opacity cho 97.6%, đủ pipeline cho 94.8%. Đặc trưng ROI và spatial mang thêm thông tin, nhưng nhạy hơn với lỗi segmentation vì ROI nhỏ hơn cả khung hình. Xem [ablation_scores.csv](results/exp_a_real/ablation_scores.csv).

### 3.2 Exp B: vết bẩn mô phỏng có kiểm soát

Có 32 frame × 17 điều kiện = 544 lần đo. Bảng dưới là Health Score trung bình, kèm edge density còn giữ lại trong ROI so với baseline. Baseline đạt 92.5 ± 5.4 điểm, 100% Healthy.

| Coverage thêm | transparent · centre | transparent · periphery | opaque · centre | opaque · periphery |
|---|---|---|---|---|
| +10% | 86.6 (edge 82%) | 91.2 (100%) | 74.0 (83%) | 88.5 (100%) |
| +25% | 79.6 (53%) | 88.8 (97%) | 52.2 (54%) | 81.3 (97%) |
| +40% | 74.4 (30%) | 85.9 (90%) | 35.9 (32%) | 72.4 (91%) |
| +60% | **69.2 → Degraded** (**12%**) | 80.8 (73%) | **20.0 → Unreliable** (**12%**) | 56.5 (75%) |

Kiểm tra claim (chi tiết trong [log.txt](results/exp_b_controlled/log.txt)):

- **C1:** score không bao giờ tăng khi coverage tăng, đúng trên 100% frame ở cả 4 nhóm.
- **C2:** opaque luôn cho score thấp hơn transparent (100% frame). Ở vị trí centre, khoảng cách tăng từ 12.6 lên 49.3 điểm.
- **C3:** centre luôn cho score thấp hơn periphery (100% frame), chênh 4.6–36.5 điểm.
- **C4:** Spearman giữa score và edge density còn lại trong ROI là 0.68; với sharpness là 0.69 (n = 512, p < 1e-70).
- Ảnh minh họa: [example_strip.png](results/exp_b_controlled/example_strip.png).

### 3.3 Exp C: cải tiến bằng kiểm tra ảnh

Mức tham chiếu edge density của ROI được hiệu chuẩn từ 91 frame GT-Healthy có id < 2500: FV 0.101, MVL 0.087, MVR 0.100, RV 0.125. Phần đánh giá dùng 248 frame có id ≥ 2500, thuộc một phiên ghi hình khác.

| min_ratio = 0.2 | v1 (chỉ mask) | v2 (+ kiểm tra ảnh) |
|---|---|---|
| Exp B transparent · centre · 60% → Unreliable | 0% | **97%** |
| Exp B transparent · centre · 40% → Unreliable | 0% | 44% |
| Exp B baseline và 8 điều kiện periphery: số frame Unreliable tăng thêm | — | 0 frame (v2 giữ nguyên v1) |
| Frame thật GT Healthy bị gắn cờ (báo động sai) | 0% | **0%** (0/84) |
| Frame thật GT Degraded bị chuyển thành Unreliable | 0% | 13.8% (15/109, chỉ từ 3 cảnh) |
| Tỉ lệ frame thật bị loại (Unreliable) | 22.2% | 28.2% |

Chi tiết trong [real_frames_tradeoff.csv](results/exp_c_image_check/real_frames_tradeoff.csv) và [exp_b_conditions_v1_vs_v2.csv](results/exp_c_image_check/exp_b_conditions_v1_vs_v2.csv).

## 4. Failure case (Bước 5)

**F1: dữ liệu thật. Model bỏ sót vết bẩn transparent nên camera trông khỏe hơn thực tế.** Xem [failure_cases.png](results/exp_a_real/failure_cases.png).

- *Nhóm quan sát được:* có 16 frame được dự đoán lạc quan hơn GT. Ví dụ 4500_RV: GT cho 45.0 điểm, Unreliable (weight 0); FPN-R18 cho 52.9 điểm, Degraded (weight 0.5). Ở frame này, 10% khung hình là transparent theo GT nhưng bị dự đoán là clear. Frame 4518_MVR và 4522_MVR cũng vậy, lần lượt 15% và 16% khung hình.
- *Repo/paper cho biết:* IoU của lớp transparent chỉ đạt 0.10–0.30 trên mọi model, trong khi opaque khoảng 0.80 (file evaluations trong `model_outputs`). Paper P1 cũng thừa nhận nhãn lớp này mơ hồ: *"the differentiation between the Transparent and Semi-transparent classes is somewhat ambiguous"*.
- *Suy luận, chưa đo:* fusion sẽ cho camera này weight 0.5 dù theo GT nó phải bị loại. Nhóm chưa đo tác động lên detector.

**F2: thiết kế. Mask chỉ cho biết vết bẩn "transparent", không cho biết nó làm nhòe ảnh đến mức nào.** Xem [example_strip.png](results/exp_b_controlled/example_strip.png).

- *Nhóm quan sát được:* ở điều kiện transparent · centre · 60%, ROI chỉ còn 12% edge, ngang với opaque · centre · 60%. Tuy vậy score vẫn là 69 (Degraded, weight 0.5) so với 20 (Unreliable, weight 0).
- Đây là một trần cấu trúc: kể cả khi toàn bộ lens bị phủ transparent, score vẫn là 100 · (1 − 0.33) = 67, nên camera **không bao giờ** thành Unreliable.
- *Giới hạn của phép thử:* vết transparent mô phỏng dùng blur σ = 6 px, có thể nặng hơn vết thật.

## 5. Quyết định kỹ thuật và trade-off

1. **Chọn model theo metric quyết định, không theo mIoU.** FPN-R50 và FPN-R18 có mIoU bằng nhau (0.617 và 0.616), nhưng FPN-R50 bỏ sót Unreliable nhiều gấp khoảng 3 lần (9.1% so với 2.8%). Checkpoint FPN-R18 cũng nhỏ hơn (157 MB so với 315 MB). Đề xuất dùng **FPN-R18**.
2. **Giữ score từ mask làm tín hiệu chính, thêm kiểm tra ảnh (v2, min_ratio 0.2) như một luật chỉ được hạ state.**
   - Lợi ích: bắt được F2 (97%), không tăng báo động sai trên frame GT Healthy.
   - Chi phí: tỉ lệ frame thật bị loại tăng từ 22.2% lên 28.2%. Khi camera bị loại, hệ thống mất tính năng dựa vào camera đó, tức là giảm availability.
   - Nên chạy ở **chế độ shadow** (chỉ ghi log, không tác động) cho đến khi được kiểm tra trên dữ liệu đêm và cảnh ít chi tiết.
3. **Khi nào nên và không nên dùng.**
   - Hợp với hệ multi-camera, nơi weight 0 của một camera vẫn còn camera khác bù lại.
   - Với hệ chỉ có một camera, nên ưu tiên dùng Degraded kèm cảnh báo cho người lái thay vì loại camera ngay.
   - Không dùng các ngưỡng hiện tại cho dữ liệu đêm hoặc mưa nếu chưa hiệu chuẩn lại.
4. **Vòng thử tiếp theo và metric để kiểm chứng.**
   - (a) Thêm trọng số cho lớp transparent khi train lại, đo IoU transparent và missed Unreliable.
   - (b) Làm mượt theo thời gian kèm hysteresis trên chuỗi frame, đo số lần đổi state mỗi phút.
   - (c) Hiệu chuẩn ngưỡng 80/50 bằng hiệu năng của một detector thật, ví dụ mAP theo từng mức Health Score.

## 6. Giới hạn

- Không có nhãn chuẩn cho "sức khỏe camera". Mốc tham chiếu là score tính từ GT mask bằng chính công thức của nhóm, nên Exp A chỉ đo lỗi lan truyền từ segmentation sang quyết định, không chứng minh ngưỡng nào là "đúng".
- Các frame không độc lập: 497 frame chỉ có 399 mask khác nhau, vì frame cùng chuỗi dùng chung nhãn. Paper P1 đã chia lại train/test theo chuỗi để tránh rò rỉ; nhóm kiểm tra lại và thấy chỉ còn 8/497 frame test (1.6%) có mask trùng hệ với tập train/val ([results/tv1_data/log.txt](results/tv1_data/log.txt)).
- Nhóm không chạy inference, nên không đo latency. Tác động lên detector hay tính năng ADAS là suy luận, chưa đo.
- Edge density phụ thuộc nội dung cảnh. 15 frame bị v2 hạ state chỉ đến từ 3 cảnh, trong đó 2 cảnh là mặt đường MVR ít chi tiết.
- Ảnh được resize từ 1280×960 về 512×512 giống repo. Repo resize nhãn bằng bicubic; so với NEAREST, cách này đổi 0.20% pixel nên không đáng kể.

## 7. Tái hiện

```bash
pip install numpy pandas pillow scipy scikit-learn matplotlib pyyaml tqdm pytest   # Python 3.11, CPU
python benchmark/fetch_woodscape_subset.py --evals                                 # ~20 s
python benchmark/fetch_woodscape_subset.py --gt --rgb --models fpn_resnet18_torch_cross_entropy_correct_files fpn_resnet50_torch_cross_entropy_all_files unet_resnet18_torch_cross_entropy_all_files pan_resnet18_torch_cross_entropy_correct_clear_strict_files   # ~6 min, tải ~600 MB, lưu ~200 MB
python -m benchmark.build_valid_masks
python -m benchmark.tv1_data_checks         # -> results/tv1_data/   (kiểm tra dữ liệu)
python -m pytest tests -q                    # 9 tests
python -m benchmark.exp_a_real               # -> results/exp_a_real/
python -m benchmark.exp_b_controlled         # -> results/exp_b_controlled/   (seed 2026)
python -m benchmark.exp_c_image_check        # -> results/exp_c_image_check/
python -m camera_health --masks model_outputs/fpn_resnet18_torch_cross_entropy_correct_files/predictions --images woodscape_input/rgbImages_512 --image-check --out results/cli_demo_fpn_r18_v2.csv
```

## 8. Phân công (4 thành viên)

| Thành viên | Phụ trách | File chính | Phần pitch |
|---|---|---|---|
| TV1 · Dữ liệu và nguồn | Bước 1–2: đọc repo/paper, tải dữ liệu, mặt nạ fisheye, kiểm tra trùng nhãn/rò rỉ | `benchmark/fetch_woodscape_subset.py`, `benchmark/build_valid_masks.py` | Problem (≈40 s) |
| TV2 · Thuật toán | 5 đặc trưng → severity → score → state → weight; config; test; độ nhạy ngưỡng; ablation | `camera_health/`, `tests/` | Method (≈60 s) |
| TV3 · Benchmark thật | Exp A với 4 model; failure case F1; chọn model | `benchmark/exp_a_real.py`, `results/exp_a_real/` | Benchmark A + F1 (≈75 s) |
| TV4 · Benchmark mô phỏng và quyết định | Exp B, Exp C; failure case F2; engineering decision; ghép slide | `benchmark/synth_soiling.py`, `exp_b_controlled.py`, `exp_c_image_check.py` | Benchmark B/C + F2 + Decision (≈90 s) |

Báo cáo riêng: [reports/](reports/).
