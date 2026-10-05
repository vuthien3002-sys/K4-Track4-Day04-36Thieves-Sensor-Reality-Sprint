# Camera Health Score — Báo cáo Lab Day 04 (Sensor Reality Sprint)

**Họ tên:** Võ Minh Quân
**MSSV:** 2A202602429
**Module:** `camera_health/`

---

## 1. Mục tiêu

Biến **mask lens-soiling** (đầu ra của mạng segmentation WoodScape trong repo này) thành một chỉ số sức khỏe camera dùng được cho hệ thống ADAS:

```
Ảnh + Mask soiling (0..3)
  → Features (coverage, transparent / semi / opaque ratio, spatial position, critical-ROI coverage)
  → Soiling Severity [0, 1]
  → Camera Health Score 0–100
  → State: Healthy / Degraded / Unreliable
  → Camera Weight (1.0 / 0.5 / 0.0) cho khối sensor fusion
```

Lớp trong mask: `0` clear, `1` transparent, `2` semi-transparent, `3` opaque.

## 2. Cấu trúc module

| File | Vai trò |
|---|---|
| [config.yaml](camera_health/config.yaml) | Toàn bộ tham số thiết kế (opacity, ROI, trọng số, ngưỡng). |
| [io_utils.py](camera_health/io_utils.py) | Đọc config, mask (resize `NEAREST` để giữ class id), ảnh RGB, mask vùng fisheye hợp lệ. |
| [features.py](camera_health/features.py) | Tính các feature soiling từ mask. |
| [scoring.py](camera_health/scoring.py) | Severity → Score → State → Weight, safety override, image check tùy chọn. |
| [quality.py](camera_health/quality.py) | Proxy chất lượng ảnh trong ROI (Laplacian variance, edge density, RMS contrast). |
| [\_\_main\_\_.py](camera_health/__main__.py) | CLI chấm điểm cả thư mục mask, xuất CSV. |
| `assets/valid_{FV,RV,MVL,MVR}.png` | Mask vùng nhìn thấy của ống kính fisheye cho từng camera. |

Cách chạy:

```bash
# Chỉ dùng mask
python -m camera_health --masks model_outputs/<model>/predictions --out results/health.csv

# Mask + kiểm tra ảnh (edge density trong ROI)
python -m camera_health --masks <masks> --images woodscape_input/rgbImages_512 --image-check --out results/health_v2.csv
```

## 3. Thiết kế chi tiết

### 3.1 Tiền xử lý
- Mọi mask được đưa về `512×512` — cùng kích thước mà `pytorch_l_base_pred.py` dự đoán.
- Resize mask bằng `Image.NEAREST`: resize bicubic mặc định sẽ sinh ra nhãn trung gian không tồn tại.
- Pixel nằm ngoài vùng fisheye hợp lệ (`valid_<cam>.png`) bị bỏ qua. Tỉ lệ vùng hợp lệ đo được: FV 0.956, RV 0.889, MVL 0.945, MVR 0.955.

### 3.2 Features ([features.py](camera_health/features.py))

Mỗi lớp được gán độ chắn sáng `class_opacity = [0.0, 0.33, 0.66, 1.0]`.

| Feature | Định nghĩa |
|---|---|
| `coverage` | Tỉ lệ pixel bẩn (class > 0) trên vùng hợp lệ. |
| `transparent_ratio`, `semi_ratio`, `opaque_ratio` | Tỉ lệ từng lớp trong phần bị bẩn. |
| `effective_occlusion` | Trung bình opacity trên toàn vùng hợp lệ. |
| `spatial_occlusion` | Opacity có trọng số Gaussian quanh tâm ảnh (`σ = 0.5`, bán kính chuẩn hóa). Vùng biên fisheye méo mạnh nên ít ảnh hưởng hơn. |
| `soiling_centroid_r` | Bán kính chuẩn hóa của trọng tâm vùng bẩn (feature dùng để chẩn đoán). |
| `roi_coverage`, `roi_occlusion`, `roi_opaque` | Coverage / opacity / tỉ lệ opaque trong **critical ROI** của từng camera. |

Critical ROI (tọa độ chuẩn hóa `[x0, y0, x1, y1]`):

| Camera | ROI | Lý do |
|---|---|---|
| FV | [0.15, 0.30, 0.85, 0.80] | Mặt đường và đường chân trời phía trước, trên cản xe. |
| RV | [0.15, 0.25, 0.85, 0.75] | Tương tự, phía sau. |
| MVL / MVR | [0.20, 0.20, 0.80, 0.70] | Làn bên cạnh, tránh phần thân xe nhìn thấy trong ảnh. |

### 3.3 Severity, Score, State ([scoring.py](camera_health/scoring.py))

```
severity = (0.3·effective_occlusion + 0.3·spatial_occlusion + 0.4·roi_occlusion) / 1.0   ∈ [0, 1]
score    = 100 · (1 − severity)
state    = Healthy     nếu score ≥ 80
           Unreliable  nếu score < 50
           Degraded    còn lại
override : roi_opaque ≥ 0.5  →  Unreliable (bất kể score)
weight   : Healthy 1.0 · Degraded 0.5 · Unreliable 0.0
```

ROI nhận trọng số cao nhất (0.4) vì đó là vùng perception thực sự dùng. Safety override tồn tại vì một vết opaque lớn ở ngay ROI có thể vẫn cho score trung bình khi phần còn lại của ống kính sạch.

### 3.4 Cải tiến: Image check (tắt mặc định)

**Điểm yếu của pipeline chỉ dùng mask:** nhãn `transparent` không cho biết mức độ mờ thật. Một ống kính phủ kín bởi soiling transparent luôn có score `100·(1−0.33) = 67` → **không bao giờ đạt Unreliable**.

Khi bật `image_check`, module tính edge density (Sobel, ngưỡng 0.1) trong ROI và so với giá trị tham chiếu của frame sạch cùng camera. Nếu `edge_ratio < 0.2` → ép state về Unreliable. Giá trị tham chiếu trong config: FV 0.1006, MVL 0.0869, MVR 0.1003, RV 0.1248.

## 4. Kiểm thử trên mask tổng hợp

Tôi chạy `assess()` trên các mask 512×512 tự tạo, camera FV, có dùng `valid_FV.png`. Đây là các số đo thực tế từ code hiện tại:

| Trường hợp | coverage | eff. occ | spatial occ | ROI occ | ROI opaque | Score | State | Override |
|---|---|---|---|---|---|---|---|---|
| Sạch | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 100.00 | Healthy | – |
| Toàn bộ transparent | 1.000 | 0.330 | 0.330 | 0.330 | 0.000 | 67.00 | Degraded | – |
| Toàn bộ semi-transparent | 1.000 | 0.660 | 0.660 | 0.660 | 0.000 | 34.00 | Unreliable | – |
| Toàn bộ opaque | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | Unreliable | ✔ |
| Dải opaque 80 px ở mép trái | 0.137 | 0.137 | 0.061 | 0.008 | 0.008 | 93.72 | Healthy | – |
| Ô opaque 112×112 ở tâm | 0.050 | 0.050 | 0.126 | 0.137 | 0.137 | 89.23 | Healthy | – |
| Nửa dưới transparent | 0.492 | 0.162 | 0.165 | 0.199 | 0.000 | 82.25 | Healthy | – |
| Opaque phủ nửa trên ROI | 0.183 | – | – | – | 0.500 | 64.31 | **Unreliable** | ✔ |
| Opaque phủ toàn bộ ROI | 0.366 | 0.366 | 0.622 | 1.000 | 1.000 | 30.35 | Unreliable | ✔ |

**Nhận xét:**
1. **Vị trí quan trọng hơn diện tích.** Dải ở mép có coverage 13.7% nhưng score 93.72. Ô ở tâm chỉ có coverage 5% nhưng score thấp hơn (89.23). Đúng với chủ đích của spatial weight và ROI.
2. **Safety override hoạt động.** Trường hợp opaque phủ nửa ROI có score 64.31 (vùng Degraded), nhưng bị ép thành Unreliable vì `roi_opaque = 0.5`.
3. **Giới hạn của transparent đã được xác nhận.** Ống kính phủ kín transparent dừng ở 67 / Degraded, đúng như phân tích ở mục 3.4. Image check được thiết kế để xử lý trường hợp này.
4. Nửa dưới bị transparent vẫn được tính là Healthy (82.25). Điều này có thể hơi lạc quan với FV, vì nửa dưới chứa mặt đường gần xe.

## 5. Hạn chế

- Mọi tham số trong `config.yaml` (opacity, σ, ROI, trọng số, ngưỡng 80/50, override 0.5) là **giả định thiết kế của lab**, không lấy từ WoodScape hay một tiêu chuẩn nào. Chúng chưa được hiệu chỉnh trên dữ liệu thật trong báo cáo này.
- ROI là hình chữ nhật cố định. ROI thật phụ thuộc vào tác vụ (phát hiện làn, người đi bộ, đỗ xe) và tốc độ xe.
- Score được tính theo từng frame, chưa làm mượt theo thời gian (hysteresis). Trên video, state có thể nhảy qua lại quanh ngưỡng.
- Score phụ thuộc hoàn toàn vào chất lượng mask dự đoán. Lỗi segmentation sẽ truyền thẳng sang health score.
- Image check dùng edge density, nên có thể báo nhầm trong cảnh ít texture (đêm, đường trống, sương mù) dù ống kính sạch.

## 6. Hướng phát triển

1. Hiệu chỉnh trọng số và ngưỡng bằng nhãn state thật trên tập test, rồi báo cáo confusion matrix.
2. Thêm làm mượt theo thời gian (EMA + hysteresis) trước khi đổi state.
3. Dùng ROI theo tác vụ, hoặc lấy ROI từ heatmap attention của mạng perception.
4. Kết hợp image check với độ tin cậy của mạng segmentation (softmax entropy).

## 7. Cập nhật sau khi nhóm đã push đủ benchmark

- Lúc viết báo cáo này, `benchmark/` và `results/` chưa có trong repo. **Hiện cả hai đã có:** code benchmark ở commit `1671f3d`; log, CSV, plot và unit test ở commit `ede9490`.
- Số liệu trên ảnh WoodScape thật và trên mask dự đoán của mạng:
  - Exp A: [results/exp_a_real/log.txt](results/exp_a_real/log.txt)
  - Exp B: [results/exp_b_controlled/log.txt](results/exp_b_controlled/log.txt)
  - Exp C: [results/exp_c_image_check/log.txt](results/exp_c_image_check/log.txt)
  - Tổng hợp: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)
- Giá trị `reference_edge_density` trong config (FV 0.1006, MVL 0.0869, MVR 0.1003, RV 0.1248) được đo lại trong Exp C, xem [reference_edge_density.json](results/exp_c_image_check/reference_edge_density.json).
- Bảng kiểm thử ở mục 4 đã được chạy lại và cho đúng các số ở trên.
- Bản báo cáo đủ 5 mục theo đề (Problem, Method, Benchmark, Failure case, Engineering decision), có thêm số liệu benchmark thật: [reports/TV2_thuat_toan.md](reports/TV2_thuat_toan.md).
