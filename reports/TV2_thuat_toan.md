# Báo cáo cá nhân: Võ Minh Quân · 2A202602429 · TV2 Thuật toán health score

Nhóm 36Thieves · T1 Camera health khi lens bẩn · Repo chung: <https://github.com/vuthien3002-sys/K4-Track4-Day04-36Thieves-Sensor-Reality-Sprint> · Bằng chứng chung: [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md) · Nguồn: [SOURCES.md](../SOURCES.md)

Quy ước đánh dấu:
- **[Đo]**: số liệu nhóm tự đo, có link log hoặc plot.
- **[Nguồn]**: điều repo hoặc paper cho biết, có link.
- **[Giả thuyết]**: suy luận, chưa kiểm chứng.

## Phần tôi phụ trách

Tôi phụ trách module **`camera_health/`**. Đây là phần "nhóm phát triển" trong sơ đồ pipeline, biến mask bẩn lens thành Camera Health Score, State và Weight. Commit `c48f473`.

| File | Vai trò |
|---|---|
| [config.yaml](../camera_health/config.yaml) | Toàn bộ tham số thiết kế: opacity, σ, ROI, trọng số, ngưỡng, override, image check |
| [features.py](../camera_health/features.py) | Tính 5 đặc trưng từ mask |
| [scoring.py](../camera_health/scoring.py) | Severity → Score → State → Weight, luật override, image check v2 |
| [io_utils.py](../camera_health/io_utils.py) | Đọc config, mask (resize NEAREST), ảnh, mặt nạ fisheye |
| [quality.py](../camera_health/quality.py) | Proxy ảnh trong ROI: edge density, Laplacian variance, RMS contrast |
| [\_\_main\_\_.py](../camera_health/__main__.py) | CLI `python -m camera_health`, chấm điểm cả thư mục mask |
| [tests/test_camera_health.py](../tests/test_camera_health.py) | 9 unit test cho các tính chất bắt buộc của score |

Kết quả chạy lại [Đo]:

| Lệnh | Kết quả |
|---|---|
| `python -m pytest tests -q` | **9 passed** |
| `python -m camera_health --masks model_outputs/fpn_resnet18_torch_cross_entropy_correct_files/predictions --images woodscape_input/rgbImages_512 --image-check --out results/cli_demo_fpn_r18_v2.csv` | 497 frame: Healthy 186, Degraded 143, Unreliable 168 ([cli_demo_fpn_r18_v2.csv](../results/cli_demo_fpn_r18_v2.csv)) |

## 1. Problem

- **Bài toán:** model segmentation của repo cho ra một mask 512×512 cho mỗi frame. Khối sensor fusion không dùng trực tiếp được ma trận pixel. Nó cần **một con số và một trạng thái cho mỗi camera** để biết nên tin camera đó bao nhiêu.
- [Nguồn] TiledSoilingNet ([P3](https://arxiv.org/abs/2007.00801)) dùng coverage của vết bẩn để kích hoạt hệ thống rửa lens, và cho phép *"partial functionality in unsoiled areas while reducing confidence in soiled areas"*. Đây là cơ sở cho ý tưởng Camera Weight: camera bẩn thì giảm độ tin, không nhất thiết loại bỏ ngay.
- **Vì sao chỉ dùng coverage là chưa đủ:** cùng một diện tích bẩn, nhưng vết opaque ở giữa đường nguy hiểm hơn hẳn vết transparent ở rìa fisheye. Score phải phân biệt được loại vết bẩn và vị trí của nó.
- **Claim:** score phải không tăng khi coverage tăng. Ở cùng diện tích, vết opaque phải bị phạt nặng hơn transparent, và vết ở tâm/ROI nặng hơn vết ở rìa. Lens sạch cho 100 điểm, lens opaque toàn bộ cho 0 điểm.

## 2. Method

Pipeline trong [scoring.py](../camera_health/scoring.py) (hàm `assess`):

| Bước | Công thức | Lý do thiết kế |
|---|---|---|
| Độ cản sáng mỗi pixel | o = 0 / 0.33 / 0.66 / 1.0 cho lớp clear / transparent / semi / opaque | Tăng tuyến tính theo mức đục. Semi không có trong sơ đồ nhưng có trong mask, nên được giữ ở mức giữa. |
| Coverage, tỉ lệ từng loại | số pixel bẩn / số pixel vùng fisheye hợp lệ; tỉ lệ transparent / semi / opaque trong phần bẩn | Mô tả mask. Ngoài vòng fisheye không tính (vùng hợp lệ chiếm 88.9–95.6% ảnh) |
| E (effective occlusion) | trung bình o trên vùng hợp lệ | Gộp coverage và độ đục |
| P (spatial) | trung bình o có trọng số Gaussian quanh tâm, σ = 0.5 | Rìa fisheye méo mạnh và ít được perception dùng |
| R (critical ROI) | trung bình o trong ROI riêng của từng camera (FV/RV là đường trước/sau; MVL/MVR là làn bên cạnh) | Vùng perception thực sự cần |
| Severity | S = 0.3·E + 0.3·P + 0.4·R | ROI có trọng số cao nhất |
| Score | 100 × (1 − S) | |
| State | Healthy nếu score ≥ 80; Unreliable nếu score < 50 **hoặc** opaque phủ ≥ 50% ROI; còn lại Degraded | Override an toàn: vết opaque lớn ngay ROI nhưng phần còn lại của lens sạch vẫn có thể cho score trung bình |
| Weight | 1.0 / 0.5 / 0 | |
| v2 (mặc định tắt) | Unreliable nếu edge density trong ROI < 0.2 × mức tham chiếu sạch của camera | Xử lý failure case F2 (mục 4) |

**Ví dụ tính tay với frame thật 4500_RV trên GT mask [Đo]:**
- Đầu vào: E = 0.484, P = 0.539, R = 0.608.
- S = 0.3·0.484 + 0.3·0.539 + 0.4·0.608 = 0.145 + 0.162 + 0.243 = **0.550**.
- Score = 100 × (1 − 0.550) = **45.0** → Unreliable, weight 0.

Mọi tham số trong `config.yaml` đều là **giả định thiết kế của nhóm**, không lấy từ chuẩn hay từ WoodScape. Mức ảnh hưởng của chúng được đo ở mục 3.

## 3. Benchmark

### 3.1 Unit test: 9 tính chất bắt buộc [Đo]

| Tính chất | Test |
|---|---|
| Lens sạch → 100 điểm, Healthy, weight 1.0 | `test_clean_camera_is_healthy` |
| Lens opaque toàn bộ → 0 điểm, Unreliable, weight 0 | `test_fully_opaque_camera_is_unreliable` |
| Cùng diện tích: transparent > semi > opaque | `test_same_coverage_more_opaque_scores_lower` |
| Các tỉ lệ loại vết bẩn chia đúng phần bẩn | `test_ratios_split_the_soiled_area` |
| Cùng diện tích: vết ở tâm bị phạt nặng hơn vết ở góc | `test_centre_soiling_costs_more_than_corner_soiling` |
| Opaque phủ trên 50% ROI → Unreliable | `test_opaque_roi_forces_unreliable` |
| Pixel ngoài vòng fisheye không được tính | `test_pixels_outside_fisheye_are_ignored` |
| Đúng ngưỡng 80/50 | `test_state_thresholds` |
| Image check chỉ có hiệu lực khi được bật | `test_image_check_flags_a_blind_camera_only_when_enabled` |

### 3.2 Mask tổng hợp trên camera FV, có dùng mặt nạ fisheye [Đo]

| Trường hợp | coverage | E | P | R | ROI opaque | Score | State |
|---|---|---|---|---|---|---|---|
| Sạch | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 100.00 | Healthy |
| Toàn bộ transparent | 1.000 | 0.330 | 0.330 | 0.330 | 0.000 | **67.00** | Degraded |
| Toàn bộ semi | 1.000 | 0.660 | 0.660 | 0.660 | 0.000 | 34.00 | Unreliable |
| Toàn bộ opaque | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | Unreliable (override) |
| Dải opaque 80 px ở mép trái | 0.137 | 0.137 | 0.061 | 0.008 | 0.008 | 93.72 | Healthy |
| Ô opaque 112×112 ở tâm | 0.050 | 0.050 | 0.126 | 0.137 | 0.137 | 89.23 | Healthy |
| Nửa dưới transparent | 0.492 | 0.162 | 0.165 | 0.199 | 0.000 | 82.25 | Healthy |
| Opaque phủ nửa trên ROI | 0.183 | 0.183 | 0.340 | 0.500 | 0.500 | 64.31 | **Unreliable (override)** |
| Opaque phủ toàn bộ ROI | 0.366 | 0.366 | 0.622 | 1.000 | 1.000 | 30.35 | Unreliable (override) |

Nhận xét:
- **Vị trí quan trọng hơn diện tích.** Dải opaque ở mép phủ 13.7% ảnh mà vẫn được 93.72 điểm. Ô opaque ở tâm chỉ phủ 5% nhưng điểm thấp hơn, 89.23.
- **Override hoạt động đúng.** Trường hợp opaque phủ nửa ROI có score 64.31, nằm trong vùng Degraded, nhưng bị ép thành Unreliable.

### 3.3 Trên dữ liệu thật và dữ liệu mô phỏng

Phần này do TV3 và TV4 chạy; tôi đọc lại từ góc độ thuật toán [Đo].

- **Vết bẩn mô phỏng ([exp_b log](../results/exp_b_controlled/log.txt)):**
  - Score không bao giờ tăng khi coverage tăng: đúng trên 100% frame (C1).
  - Opaque luôn thấp điểm hơn transparent: 100% frame, ở vị trí tâm chênh 12.6–49.3 điểm (C2).
  - Tâm luôn thấp điểm hơn rìa: 100% frame (C3).
  - Claim ở mục 1 được xác nhận.
- **Độ nhạy của ngưỡng ([threshold_sensitivity.csv](../results/exp_a_real/threshold_sensitivity.csv)):**
  - Dời ngưỡng Unreliable từ 50 lên 60 thì tỉ lệ bỏ sót Unreliable giảm từ 2.8% xuống 0%.
  - Đổi lại, tỉ lệ frame bị loại tăng từ 28.8% lên 32.4%.
- **Ablation ([ablation_scores.csv](../results/exp_a_real/ablation_scores.csv)):** độ khớp state giữa mask dự đoán và GT là 94.2% nếu chỉ dùng coverage, 97.6% với coverage × opacity, và 94.8% với pipeline đầy đủ. Các đặc trưng spatial và ROI mang thêm thông tin về vị trí, nhưng **nhạy hơn với lỗi segmentation**, vì ROI nhỏ hơn cả khung hình. Đây là một trade-off của thiết kế.
- **v1 so với v2 trên mask dự đoán FPN-R18:**

  | | Healthy | Degraded | Unreliable |
  |---|---|---|---|
  | v1, chỉ dùng mask | 186 | 163 | 148 |
  | v2, thêm kiểm tra ảnh | 186 | 143 | 168 |

  v2 chuyển 20 frame từ Degraded sang Unreliable và không làm thay đổi frame Healthy nào.

## 4. Failure case

Tôi chọn **F2: mask chỉ cho biết vết bẩn là "transparent", không cho biết nó làm nhòe ảnh đến mức nào.**

- **Trần cấu trúc, chứng minh bằng công thức:**
  - Lens bị phủ transparent toàn bộ cho E = P = R = 0.33.
  - S = 0.33, nên score = **67**, luôn là Degraded.
  - Luật override chỉ xét opaque, nên **camera không bao giờ thành Unreliable**. [Đo] Bảng 3.2, dòng "Toàn bộ transparent".
- **Trên ảnh mô phỏng [Đo]** ([example_strip.png](../results/exp_b_controlled/example_strip.png)):
  - Ở điều kiện transparent · centre · 60%, ROI chỉ còn 12% edge, ngang với opaque · centre · 60%.
  - Tuy vậy score vẫn là 69, Degraded, weight 0.5. Trong khi opaque cùng điều kiện cho 20 điểm, Unreliable, weight 0.
- **Một trường hợp lạc quan nữa [Đo]:** nửa dưới ảnh FV bị transparent vẫn được 82.25 điểm, tức Healthy. Mà nửa dưới FV chính là mặt đường gần xe.
- **Giới hạn của kết luận [Giả thuyết]:** vết transparent mô phỏng dùng blur σ = 6 px, có thể nặng hơn vết thật. Nhóm chưa đo mức nhòe thật của nhãn transparent trên WoodScape.

## 5. Engineering decision

**Quyết định:**
- Giữ score từ mask (v1) làm tín hiệu chính.
- Image check (v2) chỉ được phép **hạ** state và chạy ở chế độ shadow trước.
- [Đo] v2 bắt được 97% trường hợp F2 mà không báo động sai trên frame GT Healthy. Đổi lại, tỉ lệ camera bị loại tăng từ 22.2% lên 28.2% ([exp_c log](../results/exp_c_image_check/log.txt)).

**Đề xuất cho vòng sau, mỗi đề xuất kèm metric để kiểm chứng:**

1. **Hệ số opacity của transparent phụ thuộc mức nhòe đo trên ảnh**, thay cho hằng số 0.33. Metric: Spearman giữa score và edge density còn lại (hiện là 0.68), cùng tỉ lệ trường hợp F2 bị xếp sai.
2. **Hiệu chuẩn ngưỡng 80/50 bằng hiệu năng của một detector thật**, ví dụ mAP theo từng mức score, thay cho giả định. Metric: độ dốc của mAP theo score.
3. **Làm mượt theo thời gian (EMA kèm hysteresis)** trước khi đổi state, vì hiện tại score tính riêng từng frame. Metric: số lần đổi state mỗi phút trên chuỗi video.

**Trade-off:**
- Pipeline nhiều đặc trưng biểu đạt rủi ro theo vị trí tốt hơn, nhưng nhạy với lỗi mask hơn: 94.8% so với 97.6% của coverage × opacity.
- Ngưỡng càng chặt thì càng an toàn, nhưng availability càng thấp.
- Hệ nhiều camera chịu được weight 0 cho một camera. Hệ một camera nên ưu tiên Degraded kèm cảnh báo người lái.

## Nguồn, version, lệnh chạy

- **Code:** module `camera_health/` trong commit `c48f473`; tham số trong [config.yaml](../camera_health/config.yaml).
- **Nguồn:**
  - [Nguồn] P3 TiledSoilingNet, <https://arxiv.org/abs/2007.00801> (chỉ đọc abstract): coverage dùng để giảm độ tin.
  - [Nguồn] P1, <https://arxiv.org/abs/2511.09740>: định nghĩa 4 lớp mask. Paper tự nhận nhãn transparent/semi *"somewhat ambiguous"*.
- **Lệnh chạy:**

```
python -m pytest tests -q
python -m camera_health --masks model_outputs/fpn_resnet18_torch_cross_entropy_correct_files/predictions --out results/health_v1.csv
python -m camera_health --masks model_outputs/fpn_resnet18_torch_cross_entropy_correct_files/predictions --images woodscape_input/rgbImages_512 --image-check --out results/cli_demo_fpn_r18_v2.csv
```
