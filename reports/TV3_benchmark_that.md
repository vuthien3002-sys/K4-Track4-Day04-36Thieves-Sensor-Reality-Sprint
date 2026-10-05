# Báo cáo cá nhân: Dương Văn Thành · 2A202602368 · TV3 Benchmark dữ liệu thật (Exp A)

**Nhóm 36Thieves** · **Chủ đề T1**: Giám sát sức khỏe camera fisheye khi lens bẩn (WoodScape)  
**Repository chung**: <https://github.com/vuthien3002-sys/K4-Track4-Day04-36Thieves-Sensor-Reality-Sprint>  
**Bằng chứng chung**: [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md) · Log của Exp A: [results/exp_a_real/log.txt](../results/exp_a_real/log.txt)

Quy ước trong báo cáo:
- **[Đo]**: số do nhóm chạy, có link CSV, log hoặc plot.
- **[Nguồn]**: điều repo hoặc paper cho biết.
- **[Giả thuyết]**: suy luận chưa được kiểm chứng.

---

## Phần tôi phụ trách

Tôi phụ trách **TV3: benchmark trên dữ liệu thật (Experiment A)**:

1. **Kịch bản đối sánh GT với dự đoán.** Health Score, State và Weight được tính hai lần trên cùng 497 frame test WoodScape: một lần từ **GT mask** (`gtLabels`), một lần từ **mask dự đoán** của các model segmentation trong repo. Code nằm ở [benchmark/exp_a_real.py](../benchmark/exp_a_real.py) và [benchmark/common.py](../benchmark/common.py), commit `1671f3d`.
2. **Đo cách lỗi pixel chuyển thành lỗi quyết định**, trên 4 model của repo: `FPN-R18`, `FPN-R50`, `UNet-R18`, `PAN-R18-strict`. Các metric gồm:
   - `score_MAE`, `score_bias`
   - `state_acc`, Cohen's Kappa
   - `unsafe_rate`, `missed_unreliable`, `false_alarm`, `weight_MAE`
3. **Phân tích theo camera, theo mức bẩn và ablation:**
   - 4 camera: FV, RV, MVL, MVR
   - 4 dải coverage của GT
   - 3 biến thể cách tính score
4. **Phân tích failure case:** tìm các frame có quyết định lệch nhiều nhất giữa GT và dự đoán.

Lệnh tôi chạy lại: `python -m benchmark.exp_a_real`. [Đo] Lần chạy lại cho ra 8/8 file CSV và log giống hệt bản đã lưu.

---

## 1. Problem (vấn đề và câu hỏi của Exp A)

Hệ surround-view của xe ADAS dùng 4 camera fisheye, nên camera tiếp xúc trực tiếp với bùn đất và nước.

- **Khoảng trống của công bố trước:**
  - [Nguồn] Paper của repo (P1: Beránek et al., [arXiv 2511.09740](https://arxiv.org/abs/2511.09740)) giải bài toán ở mức segmentation theo pixel, với 4 lớp Clear / Transparent / Semi-Transparent / Opaque, và đánh giá bằng accuracy theo pixel. FPN đạt cao nhất, 0.940.
  - Khối sensor fusion lại không dùng trực tiếp được mask pixel. Nó cần một tín hiệu cho mỗi camera: Health Score (0–100), State (Healthy / Degraded / Unreliable) và Weight (1.0 / 0.5 / 0).
- **Câu hỏi của Exp A:** lỗi của model segmentation làm sai quyết định về camera đến mức nào? Một model có mIoU khoảng 0.6 đã đủ để đưa ra quyết định an toàn chưa?
- **Claim ban đầu của tôi:**
  - Lớp opaque có độ tương phản cao nên dễ nhận diện. Vì vậy các camera hỏng nặng (Unreliable) sẽ được phát hiện đúng.
  - Lớp transparent có IoU thấp, nên [Giả thuyết] hệ thống sẽ cho **score cao hơn thực tế** (lạc quan) và bỏ sót trạng thái Unreliable.

Mục 3 cho thấy claim này **chỉ đúng một phần**.

---

## 2. Method

### 2.1. Dữ liệu
- **497 frame test**, đúng tập test mà P1 đã chia lại theo chuỗi để tránh rò rỉ (4503 frame train / 497 frame test).
  - Phân bố theo camera: FV 139, RV 129, MVL 113, MVR 116.
  - Mask đưa về 512×512 bằng NEAREST.
- **Mặt nạ fisheye:** chỉ tính các pixel nằm trong vòng tròn nhìn thấy được của từng camera (do TV1 dựng trong [camera_health/assets/](../camera_health/assets/)). Vùng này chiếm 88.9–95.6% khung hình.
- [Đo] Các frame không độc lập: 497 frame chỉ có 399 mask khác nhau, vì frame cùng chuỗi dùng chung nhãn. Đó là lý do biểu đồ scatter có những cột điểm thẳng đứng.

### 2.2. Baseline và điều kiện so sánh
- **Baseline:** score, state, weight tính từ **GT mask** (`woodscape_input/gtLabels/`).
- **Điều kiện so sánh:** cũng tính như vậy nhưng từ **mask dự đoán** của 4 model trong `model_outputs/`.
  1. `FPN-R18` (`fpn_resnet18_torch_cross_entropy_correct_files`): model chính. mIoU thuộc nhóm cao nhất, checkpoint nhỏ.
  2. `FPN-R50` (`fpn_resnet50_torch_cross_entropy_all_files`): encoder sâu hơn, mIoU tương đương FPN-R18.
  3. `UNet-R18` (`unet_resnet18_torch_cross_entropy_all_files`): model có mIoU ở mức trung bình.
  4. `PAN-R18-strict` (`pan_resnet18_torch_cross_entropy_correct_clear_strict_files`): model kém nhất trong số 39 model.
- Cả hai bên dùng **cùng frame và cùng công thức**. Chỉ có mask đầu vào là khác nhau.

### 2.3. Từ mask đến quyết định (thuật toán TV2, tóm tắt từ [config.yaml](../camera_health/config.yaml))
1. **Độ cản sáng của mỗi lớp:** clear 0, transparent **0.33**, semi **0.66**, opaque **1.0**.
2. **Ba thành phần che phủ:**
   - E là độ che phủ trung bình trên vùng fisheye;
   - P là độ che phủ có trọng số **Gaussian quanh tâm (σ = 0.5)**;
   - R là độ che phủ trong critical ROI của từng camera.
3. **Severity:** S = 0.3·E + 0.3·P + 0.4·R. **Score** = 100 × (1 − S).
4. **State:**
   - Healthy nếu score ≥ 80, weight 1.0;
   - Degraded nếu 50 ≤ score < 80, weight 0.5;
   - Unreliable nếu score < 50 **hoặc** opaque phủ **≥ 50%** ROI, weight 0.

### 2.4. Metric
Định nghĩa trong hàm `decision_metrics()`, [exp_a_real.py L54–L69](../benchmark/exp_a_real.py#L54-L69):
- `score_MAE` và `score_bias` là trung bình của |pred − GT| và của (pred − GT). Bias dương nghĩa là dự đoán **lạc quan hơn** GT.
- `state_acc` là tỉ lệ frame có state dự đoán trùng state GT. Kappa là Cohen's kappa có trọng số tuyến tính.
- Các metric an toàn:
  - `unsafe_rate`: trong các frame GT Unreliable, tỉ lệ bị dự đoán thành Healthy. Đây là lỗi nguy hiểm nhất.
  - `missed_unreliable`: trong các frame GT Unreliable, tỉ lệ dự đoán khác Unreliable.
  - `false_alarm`: trong các frame GT Healthy, tỉ lệ bị dự đoán thành Unreliable.
  - `weight_MAE`: sai số tuyệt đối trung bình của weight.

---

## 3. Benchmark (kết quả trên dữ liệu thật)

### 3.1. So sánh 4 model: metric pixel và metric quyết định

[Đo] Nguồn: [summary_models.csv](../results/exp_a_real/summary_models.csv). mIoU được nhóm tự tính lại trên vùng fisheye, resize NEAREST.

| Model | mIoU | IoU transp | IoU opaque | score_MAE | score_bias | state_acc | Kappa | missed_unrel | unsafe_rate | false_alarm |
|---|---|---|---|---|---|---|---|---|---|---|
| **FPN-R18 (chính)** | **0.616** | **0.287** | **0.828** | **2.95** | **−1.11** | **94.8%** | **0.940** | **2.8%** (4/143) | **0.0%** | **0.0%** |
| FPN-R50 | 0.617 | 0.284 | 0.818 | 2.72 | −0.80 | 94.2% | 0.933 | 9.1% (13/143) | 0.0% | 0.0% |
| UNet-R18 | 0.558 | 0.216 | 0.795 | 3.29 | −2.60 | 94.4% | 0.935 | 2.1% | 0.0% | 0.0% |
| PAN-R18-strict | 0.440 | 0.130 | 0.645 | 9.03 | +1.54 | 80.3% | 0.744 | 22.4% | **10.5%** (15/143) | 0.0% |

Theo GT, tập test có Healthy 175, Degraded 179, Unreliable 143.

Các điểm rút ra [Đo]:

1. **FPN-R18 đạt state_acc 94.8% và kappa 0.940** dù mIoU chỉ 0.616. Score lệch trung bình chưa đến 3 điểm. Đánh giá sức khỏe camera không cần mask hoàn hảo đến từng ranh giới pixel.
2. **Cùng mIoU nhưng quyết định khác nhau.**
   - FPN-R50 và FPN-R18 có mIoU gần như bằng nhau (0.617 và 0.616), nhưng FPN-R50 bỏ sót Unreliable nhiều gấp khoảng 3 lần (9.1% so với 2.8%).
   - Vì vậy **phải chọn model theo metric quyết định**, không chọn theo mIoU.
3. **Claim "lạc quan" của tôi không đúng ở mức trung bình.**
   - 3/4 model có bias **âm**, nghĩa là score dự đoán hơi **bi quan** hơn GT (FPN-R18: −1.11 điểm).
   - Chỉ model kém nhất (PAN) có bias dương.
   - Phần đúng của claim là lỗi lạc quan có xảy ra ở từng frame riêng lẻ (mục 4.1), và đó là loại lỗi nguy hiểm.
4. **Khi model quá kém, lỗi nguy hiểm xuất hiện.** PAN-R18-strict có mIoU 0.440, và 10.5% frame GT Unreliable bị nó chấm thành Healthy. Ba model tốt hơn đều có unsafe_rate bằng 0%.

### 3.2. Theo camera (FPN-R18)

[Đo] Nguồn: [summary_per_camera.csv](../results/exp_a_real/summary_per_camera.csv).

| Camera | Số frame | score_MAE | score_bias | state_acc | Kappa | missed_unrel | false_alarm |
|---|---|---|---|---|---|---|---|
| FV (trước) | 139 | 3.19 | −0.44 | 97.8% | 0.977 | 0.0% | 0.0% |
| RV (sau) | 129 | 3.17 | −1.64 | 96.1% | 0.952 | 4.0% | 0.0% |
| MVL (gương trái) | 113 | 1.57 | −0.25 | 95.6% | 0.943 | 0.0% | 0.0% |
| MVR (gương phải) | 116 | 3.78 | −2.17 | **88.8%** | 0.862 | **8.6%** | 0.0% |

- [Đo] **MVR là camera kém nhất**: state_acc 88.8%, bỏ sót Unreliable 8.6%. Hai trong ba failure case lạc quan nhất (mục 4.1) cũng thuộc MVR.
- [Giả thuyết] Các frame MVR trong tập test có nhiều vết transparent nằm ở rìa vết opaque. Model lại nhận diện lớp transparent kém. Nhóm chưa đo riêng điều này theo từng camera.

### 3.3. Theo mức độ che phủ (FPN-R18)

[Đo] Nguồn: [summary_per_coverage.csv](../results/exp_a_real/summary_per_coverage.csv).

| GT coverage | Số frame | Score GT TB | Score dự đoán TB | score_MAE | state_acc | missed_unrel |
|---|---|---|---|---|---|---|
| < 10% | 93 | 90.9 | 90.5 | 1.28 | 100% | 0.0% |
| 10–30% | 168 | 80.2 | 80.4 | 1.74 | 92.3% | 0.0% |
| 30–60% | 104 | 62.3 | 60.4 | 2.77 | 97.1% | **13.6%** |
| > 60% | 132 | 30.0 | 27.5 | 5.82 | 92.4% | 0.8% |

- [Đo] Khi vết bẩn tăng từ dưới 10% lên trên 60%, score_MAE tăng từ **1.3 lên 5.8 điểm**.
- [Đo] Dải nhạy nhất với lỗi bỏ sót là **30–60%**: 13.6% frame Unreliable bị bỏ sót. Ở dải này score dao động quanh ngưỡng 50, nên chỉ cần lệch vài điểm là state bị đổi.
- [Đo] Ở dải 10–30%, state_acc là 92.3% vì score dao động quanh ngưỡng 80, ranh giới Healthy/Degraded.
- [Đo] Ở dải > 60%, phần lớn frame đã nằm sâu dưới 50, nên sai số lớn vẫn hiếm khi làm đổi state. Kappa ở dải này thấp (0.259) vì gần như chỉ có một loại state.

### 3.4. Confusion matrix và ablation

[Đo] Confusion matrix của FPN-R18 ([state_confusion.png](../results/exp_a_real/state_confusion.png)). Hàng là GT, cột là dự đoán, thứ tự H / D / U:

```
            Healthy  Degraded  Unreliable
Healthy       174        1          0
Degraded       12      158          9
Unreliable      0        4        139
```

Không có frame Unreliable nào bị chấm thành Healthy. 4 frame Unreliable bị hạ xuống thành Degraded. 9 frame Degraded bị chấm nặng thành Unreliable.

[Đo] Ablation: 3 cách tính score, cùng ngưỡng ([ablation_scores.csv](../results/exp_a_real/ablation_scores.csv)):

| Biến thể | Công thức | score_MAE | state_acc | unsafe_rate |
|---|---|---|---|---|
| Chỉ coverage | 100 × (1 − coverage) | 3.60 | 94.2% | 0.0% |
| Coverage × opacity | 100 × (1 − effective occlusion) | **2.58** | **97.6%** | 0.0% |
| Pipeline đầy đủ của nhóm | thêm vị trí tâm, ROI và override | 2.95 | 94.8% | 0.0% |

- [Đo] Biến thể **coverage × opacity khớp với GT tốt nhất**, không phải pipeline đầy đủ.
- Lý do: đặc trưng ROI và spatial tính trên một vùng nhỏ hơn cả khung hình, nên nhạy hơn với lỗi segmentation.
- Pipeline đầy đủ đổi một phần độ ổn định lấy khả năng phân biệt **vị trí** vết bẩn. Đây là trade-off của thiết kế, không phải một cải thiện thuần túy.

### 3.5. Độ nhạy của ngưỡng

[Đo] Nguồn: [threshold_sensitivity.csv](../results/exp_a_real/threshold_sensitivity.csv).
- Dời ngưỡng Unreliable từ 50 lên 60 thì missed_unreliable giảm từ 2.8% xuống **0%**.
- Đổi lại, tỉ lệ frame mà GT xếp Unreliable tăng từ 28.8% lên 32.4%, tức là nhiều camera bị loại hơn.

---

## 4. Failure cases

Ảnh minh họa: [failure_cases.png](../results/exp_a_real/failure_cases.png). Số liệu: [failure_cases.csv](../results/exp_a_real/failure_cases.csv).

[Đo] Có **16 frame** được dự đoán lạc quan hơn GT và **10 frame** bi quan hơn GT.

### 4.1. F1: đánh giá lạc quan vì bỏ sót vết transparent

[Đo] Ba frame lạc quan nhất:

| Frame | Score GT | Score dự đoán | State GT → dự đoán | Weight | Lỗi pixel chính |
|---|---|---|---|---|---|
| **4500_RV** | 45.0 | 52.9 | Unreliable → Degraded | 0 → 0.5 | 10% khung hình transparent bị đoán là clear; 7% semi bị đoán là transparent |
| 4518_MVR | 48.2 | 53.4 | Unreliable → Degraded | 0 → 0.5 | 15% khung hình transparent bị đoán là clear |
| 4522_MVR | 48.2 | 53.1 | Unreliable → Degraded | 0 → 0.5 | 16% khung hình transparent bị đoán là clear |

- **Hiện tượng [Đo]:**
  - Ở 4500_RV, GT coverage là 69.4% còn dự đoán chỉ 58.5%. Model bỏ sót phần viền transparent quanh các vết bẩn.
  - Score vượt ngưỡng 50, nên camera đáng lẽ bị loại lại vẫn được fusion tin một nửa.
- **Nguyên nhân:**
  1. [Nguồn] Trên cả 39 model của repo, IoU lớp transparent chỉ 0.10–0.30, trong khi opaque đạt 0.64–0.82. Chính P1 cũng thừa nhận định nghĩa nhãn mơ hồ: *"the differentiation between the Transparent and Semi-transparent classes is somewhat ambiguous"*.
  2. [Giả thuyết] Vết transparent vẫn để lộ kết cấu cảnh phía sau, nên CNN dễ coi nó là mặt đường hay bầu trời bình thường. Pixel clear lại chiếm đa số (62% pixel GT trong tập test), nên khi không chắc, model nghiêng về lớp clear.
- **Rủi ro [Giả thuyết]:** camera bị phủ vết mờ mà vẫn có weight 0.5. Khi đó module phát hiện vật cản có thể nhận ảnh kém chất lượng mà không có cảnh báo. Nhóm **chưa đo** tác động này trên một detector thật.

### 4.2. F2: báo động quá tay vì dự đoán vết bẩn ở vùng sạch

- **Frame 0568_MVR [Đo]:**
  - GT cho 52.4 điểm, Degraded. Dự đoán cho **31.3** điểm, Unreliable.
  - Coverage dự đoán 97.4%, so với 77.7% theo GT.
  - 11% khung hình là clear theo GT nhưng bị đoán là transparent; 7% khác bị đoán là semi.
- **Hệ quả:** lỗi này làm giảm availability (camera bị loại oan), nhưng theo hướng an toàn.
- [Đo] Trên toàn tập test **không có frame GT Healthy nào** bị chấm thành Unreliable (false_alarm = 0%). Lỗi bi quan chỉ xảy ra ở các frame vốn đã Degraded.

### 4.3. Lỗi phụ: resize nhãn trong repo gốc

- [Đo] Repo gốc resize ảnh nhãn bằng `Image.resize()` mặc định của PIL. Với ảnh mode L, mặc định này là bicubic, nên sinh ra cả những nhãn trung gian không tồn tại.
- [Đo] So với NEAREST, cách này làm đổi **0.20%** pixel nhãn (hàm `label_resize_artefact`, [exp_a_real.py L199](../benchmark/exp_a_real.py#L199)), tức là không đáng kể.
- Pipeline của nhóm đọc mask bằng NEAREST ([io_utils.py L22](../camera_health/io_utils.py#L22)).

---

## 5. Engineering decisions và trade-off

### Quyết định 1: chọn model theo metric quyết định
- **Căn cứ [Đo]:** FPN-R18 và FPN-R50 có mIoU bằng nhau, nhưng tỉ lệ bỏ sót Unreliable là 2.8% so với 9.1%.
- **Quyết định:** dùng **FPN-R18** cho bộ giám sát sức khỏe camera. Checkpoint của nó cũng nhỏ hơn: 157 MB so với 314 MB.
- **Bài học:** khi đánh giá model cho tính năng an toàn, phải báo cáo cả missed_unreliable và unsafe_rate, không chỉ mIoU.

### Quyết định 2: giữ ngưỡng 80/50, nhưng nêu rõ trade-off
- **Căn cứ [Đo]:** dời ngưỡng Unreliable lên 60 thì hết bỏ sót (0%), nhưng tỉ lệ camera bị loại tăng từ 28.8% lên 32.4%.
- **Quyết định:**
  - Giữ 80/50 làm điểm xuất phát, vì đây là giả định thiết kế chưa được hiệu chuẩn.
  - Đề xuất hiệu chuẩn lại bằng hiệu năng detector thật ở vòng sau.
  - Đề xuất thêm **hysteresis** theo thời gian để state không nhảy qua lại quanh ngưỡng. **Nhóm chưa cài đặt phần này.**

### Quyết định 3: cần thêm một kênh kiểm tra trên ảnh
- **Kết luận từ Exp A:** chỉ dùng mask thì không xử lý dứt điểm được F1, vì nguồn lỗi nằm ở lớp transparent.
- **Đề xuất:** kiểm tra mật độ cạnh trực tiếp trên ảnh RGB. TV4 đã thử ý này ở Exp C.
  - [Đo] Phương án này bắt được 97% trường hợp transparent nặng ở tâm, không báo động sai trên frame GT Healthy.
  - Đổi lại, tỉ lệ camera bị loại tăng từ 22.2% lên 28.2%.
- **Vòng sau:** train lại với trọng số cao hơn cho lớp transparent. Metric kiểm chứng: IoU transparent và missed_unreliable.

---

## Nguồn, version và hướng dẫn tái hiện

### Nguồn (trích từ [SOURCES.md](../SOURCES.md))
- **P1 (paper của repo):** F. Beránek, V. Diviš, I. Gruber, *Soiling detection for Advanced Driver Assistance Systems*, <https://arxiv.org/abs/2511.09740>, đọc toàn văn. Lấy từ paper: cách chia 4503/497, 4 tập con đã lọc nhãn, accuracy của FPN là 0.940.
- **P3 (TiledSoilingNet, ITSC 2020):** <https://arxiv.org/abs/2007.00801>, chỉ đọc abstract. Lấy từ paper: dùng coverage để giảm độ tin ở vùng bẩn.

Lưu ý: P1 báo cáo accuracy theo pixel, còn Exp A đo các metric quyết định. Hai loại số này **không so sánh trực tiếp** với nhau.

### Môi trường đã chạy
Python 3.11.9 (CPU), numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, Pillow 12.3.0, matplotlib 3.11.2.

### Lệnh tái hiện Exp A
Chạy từ thư mục gốc của repo:

```bash
# 1. Tải đúng phần dữ liệu cần: GT, ảnh test, mask dự đoán của 4 model, file evaluation
python benchmark/fetch_woodscape_subset.py --evals
python benchmark/fetch_woodscape_subset.py --gt --rgb --models fpn_resnet18_torch_cross_entropy_correct_files fpn_resnet50_torch_cross_entropy_all_files unet_resnet18_torch_cross_entropy_all_files pan_resnet18_torch_cross_entropy_correct_clear_strict_files
# 2. Dựng mặt nạ fisheye
python -m benchmark.build_valid_masks
# 3. Chạy Exp A: CSV, log và plot được ghi vào results/exp_a_real/
python -m benchmark.exp_a_real
```

**Đầu ra:**
- [log.txt](../results/exp_a_real/log.txt)
- [summary_models.csv](../results/exp_a_real/summary_models.csv), [summary_per_camera.csv](../results/exp_a_real/summary_per_camera.csv), [summary_per_coverage.csv](../results/exp_a_real/summary_per_coverage.csv)
- [ablation_scores.csv](../results/exp_a_real/ablation_scores.csv), [threshold_sensitivity.csv](../results/exp_a_real/threshold_sensitivity.csv), [failure_cases.csv](../results/exp_a_real/failure_cases.csv)
- [scatter_gt_vs_pred.png](../results/exp_a_real/scatter_gt_vs_pred.png), [state_confusion.png](../results/exp_a_real/state_confusion.png), [models_pixel_vs_decision.png](../results/exp_a_real/models_pixel_vs_decision.png), [failure_cases.png](../results/exp_a_real/failure_cases.png)
