# Báo cáo cá nhân: Dương Văn Thành · 2A202602368 · TV3 Benchmark dữ liệu thật (Exp A)

**Nhóm 36Thieves** · **Chủ đề T1**: Giám sát sức khỏe Camera Fisheye khi Lens bẩn (WoodScape)  
**Repository chung**: <https://github.com/vuthien3002-sys/K4-Track4-Day04-36Thieves-Sensor-Reality-Sprint>  
**Bằng chứng chung**: [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md) 

---

## Phần tôi phụ trách

Trong Sprint Day 04 của nhóm 36Thieves, tôi chịu trách nhiệm chính về vai trò **TV3: Thiết kế, thực thi và phân tích Benchmark thực nghiệm trên dữ liệu thật (Experiment A)**. Các công việc cốt lõi đã hoàn thành gồm:

1. **Thiết kế framework đánh giá thực nghiệm Exp A**:
   - Hiện thực hóa kịch bản đối sánh trực tiếp giữa **Ground-Truth Mask** (`gtLabels`) và **Predicted Mask** sinh ra từ các mô hình semantic segmentation của repo gốc trên toàn bộ 497 frame test WoodScape.
   - Xây dựng mã nguồn thực nghiệm tại [benchmark/exp_a_real.py](../benchmark/exp_a_real.py) và module log/kết quả tự động tại [benchmark/common.py](../benchmark/common.py).
2. **Khảo sát & Đo lường sự chuyển tiếp từ Pixel Metric sang Decision Metric**:
   - Khảo nghiệm 4 kiến trúc segmentation từ repo: `FPN-R18`, `FPN-R50`, `UNet-R18`, và `PAN-R18-strict`.
   - Tính toán và phân tích: Mean Absolute Error của Health Score (`score_MAE`), độ lệch hệ thống (`score_bias`), độ chính xác trạng thái camera (`state_acc`), hệ số tương đồng Cohen's Kappa, và các chỉ số an toàn nghiêm ngặt (`unsafe_rate`, `missed_unreliable`, `false_alarm`).
3. **Phân tích chiều sâu theo Camera, Coverage & Phân tích bóc tách (Ablation Study)**:
   - Đo lường độ ổn định của pipeline trên 4 camera fisheye: Trước (FV), Sau (RV), Trái (MVL), Phải (MVR).
   - Đo lường xu hướng biến thiên theo 4 dải độ bẩn (`<10%`, `10–30%`, `30–60%`, `>60%`).
   - Thực hiện Ablation so sánh giữa: *Coverage đơn thuần*, *Coverage x Opacity*, và *Full Pipeline*.
4. **Cô lập và mổ xẻ Failure Cases (Đặc biệt ca F1 - Optimistic Underestimation)**:
   - Truy vết các frame có độ vênh điểm số lớn nhất, chứng minh mối liên hệ nhân quả giữa hiện tượng màng nước/bẩn trong suốt (Transparent) với lỗi ngộ nhận an toàn của hệ thống tự hành ADAS.

---

## 1. Problem (Vấn đề và bài toán của Exp A)

Trong các hệ thống xe tự hành và ADAS surround-view (sử dụng 4 camera fisheye góc rộng 190°), camera là cảm biến nhạy cảm nhất với thời tiết và bùn đất mặt đường. 

- **Khoảng trống kỹ thuật của các công bố trước**:
  - **Repo/paper cho biết** (P1: Beránek et al., arXiv 2511.09740, trang 3–6): Các nghiên cứu hiện tại chỉ dừng lại ở bài toán Semantic Segmentation theo pixel (phân chia 4 lớp: Clean, Transparent, Semi-Transparent, Opaque) và đánh giá bằng độ chính xác pixel (Pixel Accuracy).
  - Tuy nhiên, hệ thống điều khiển lái xe (Sensor Fusion, Planning) **không thể trực tiếp tiêu thụ ma trận pixel mask** ở mỗi chu kỳ tính toán (chu kỳ 30–50 ms). Hệ thống fusion đòi hỏi một tín hiệu trạng thái rõ ràng: *Camera Health Score* (0–100), *Camera State* (Healthy / Degraded / Unreliable) và *Trọng số tin cậy* (Camera Weight: 1.0 / 0.5 / 0.0) để quyết định hạ tải, giảm tốc độ hoặc chuyển sang chế độ fail-safe.
- **Mục tiêu của Experiment A**:
  - **Câu hỏi kỹ thuật cốt lõi**: *Sai số của mô hình segmentation ảnh hưởng như thế nào đến quyết định mức độ tin cậy của camera? Liệu một mô hình có mIoU ở mức trung bình (~0.61) có đủ tin cậy để đưa ra quyết định an toàn cho xe hay không?*
- **Claim ban đầu của TV3**:
  - **Giả thuyết**: Do các lớp bẩn mờ đục hoàn toàn (Opaque - bùn đất) có độ tương phản cao nên mô hình segmentation sẽ nhận diện tốt (IoU cao), giúp nhận diện chính xác các trường hợp camera hỏng nặng (`Unreliable`). Ngược lại, lớp bẩn trong suốt (Transparent - màng nước, sương mù) có IoU rất thấp, sẽ dẫn tới việc hệ thống dự đoán điểm sức khỏe cao hơn thực tế (`optimistic bias`), tạo ra rủi ro bỏ sót trạng thái nguy hiểm (`missed_unreliable`).

---

## 2. Method (Phương pháp đo & Thiết kế thực nghiệm)

### 2.1. Cấu hình thực nghiệm & Nguồn dữ liệu
- **Tập dữ liệu**: Đúng 497 frame ảnh thực tế thuộc tập kiểm thử chuẩn WoodScape Test Set đã được P1 tái cấu trúc nhằm loại bỏ rò rỉ chuỗi. Ảnh kích thước $512 \times 512$, bao phủ 4 camera: FV (128 frame), RV (125 frame), MVL (123 frame), MVR (121 frame).
- **Mặt nạ hợp lệ (Fisheye Valid Mask)**: Chỉ tính toán các đặc trưng bên trong vòng tròn thị trường hữu hiệu của từng camera (được TV1 trích xuất tại [camera_health/assets/](../camera_health/assets/)), loại bỏ 100% viền đen ngoài rìa quang học của lens.

### 2.2. Baseline vs. Condition
- **Baseline (Mốc chuẩn chân lý)**: Tính toán Health Score, State và Weight từ **Ground-Truth mask** (`woodscape_input/gtLabels/`).
- **Condition (Điều kiện thực nghiệm)**: Tính toán Health Score, State và Weight từ **Predicted mask** sinh ra bởi 4 mô hình deep learning có sẵn checkpoint trong repo ([model_outputs/](../model_outputs/)):
  1. `FPN-R18` (`fpn_resnet18_torch_cross_entropy_correct_files`): Mô hình chính, huấn luyện trên tập dữ liệu đã lọc lỗi nhãn thô.
  2. `FPN-R50` (`fpn_resnet50_torch_cross_entropy_all_files`): Mô hình sâu hơn, huấn luyện trên toàn bộ dữ liệu gốc.
  3. `UNet-R18` (`unet_resnet18_torch_cross_entropy_all_files`): Kiến trúc encoder-decoder cổ điển.
  4. `PAN-R18-strict` (`pan_resnet18_torch_cross_entropy_correct_clear_strict_files`): Huấn luyện trên tập dữ liệu lọc sạch khắt khe nhất.

### 2.3. Quy trình tính toán từ Mask sang Quyết định Camera (Logic TV2)
Thuật toán ánh xạ mask thành quyết định vận hành theo 5 bước:
1. **Đo độ phủ từng lớp bẩn (Coverage)**:
   $$\text{Coverage} = \frac{\sum_{i \in \text{valid}} \mathbb{I}(\text{mask}_i > 0)}{N_{\text{valid}}}$$
2. **Độ cản quang hiệu dụng (Effective Occlusion / Opacity Penalty)**: Trọng số hóa theo tính chất vật lý của vết bẩn:
   - Transparent (lớp 1): trọng số $w_1 = 0.35$ (chỉ gây tán xạ, giảm tương phản)
   - Semi-Transparent (lớp 2): trọng số $w_2 = 0.70$
   - Opaque (lớp 3): trọng số $w_3 = 1.00$ (chắn sáng hoàn toàn)
3. **Phạt vị trí tâm (Center Bias)**: Áp dụng hàm trọng số xuyên tâm $R(r) = 1 - 0.5 \cdot (r / r_{\max})$, vết bẩn nằm ở trục quang học trung tâm bị phạt nặng hơn so với rìa lens.
4. **Tổng hợp Camera Health Score (0–100)**:
   $$\text{Score} = 100 \times \max(0, 1 - \text{Penalties})$$
5. **Phân lớp trạng thái (States) & Trọng số Fusion (Weights)**:
   - **Healthy** (Score $\ge 80$): Camera sạch hoặc chỉ có vài chấm mờ rìa lens $\rightarrow$ $\text{Weight} = 1.0$.
   - **Degraded** ($50 \le \text{Score} < 80$): Lens bẩn vừa phải, suy giảm tầm nhìn $\rightarrow$ $\text{Weight} = 0.5$.
   - **Unreliable** ($\text{Score} < 50$ HOẶC $\text{ROI}_{\text{opaque}} \ge 20\%$): Lens bị che khuất nghiêm trọng $\rightarrow$ $\text{Weight} = 0.0$ (loại khỏi sensor fusion).

### 2.4. Hệ thống chỉ số đánh giá (Evaluation Metrics)
Được triển khai tại hàm `decision_metrics()` trong [exp_a_real.py](../benchmark/exp_a_real.py#L54-L70):
- **Sai số điểm số**: $\text{score\_MAE} = \frac{1}{N}\sum |\text{Score}_{\text{pred}} - \text{Score}_{\text{GT}}|$; $\text{score\_bias} = \frac{1}{N}\sum (\text{Score}_{\text{pred}} - \text{Score}_{\text{GT}})$.
- **Độ chính xác trạng thái**: $\text{state\_acc} = \frac{1}{N}\sum \mathbb{I}(\text{State}_{\text{pred}} == \text{State}_{\text{GT}})$.
- **Hệ số Cohen's Kappa ($\kappa$)**: Đo lường sự đồng thuận giữa 2 phân loại có tính đến yếu tố ngẫu nhiên.
- **Chỉ số an toàn xe tự hành (Crucial Safety Metrics)**:
  - $\text{unsafe\_rate}$: Tỷ lệ camera thực tế hỏng nặng ($\text{GT} = \text{Unreliable}$) nhưng mô hình dự đoán là bình thường ($\text{pred} = \text{Healthy}$). **Chỉ số này bắt buộc phải tiệm cận 0.**
  - $\text{missed\_unreliable}$: Tỷ lệ bỏ sót trạng thái Unreliable ($\text{GT} = \text{Unreliable}$ nhưng $\text{pred} \ne \text{Unreliable}$).
  - $\text{false\_alarm}$: Báo động giả gây dừng xe oan ($\text{GT} = \text{Healthy}$ nhưng $\text{pred} = \text{Unreliable}$).
  - $\text{weight\_MAE}$: Sai số tuyệt đối của trọng số fusion.

---

## 3. Benchmark (Kết quả thực nghiệm trên dữ liệu thật)

Toàn bộ thực nghiệm được ghi nhận tự động vào [results/exp_a_real/log.txt](../results/exp_a_real/log.txt) và các tệp CSV liên quan.

### 3.1. Đối chiếu giữa 4 mô hình: Pixel Metric vs. Decision Metric
Bảng tổng hợp trích xuất từ [results/exp_a_real/summary_models.csv](../results/exp_a_real/summary_models.csv):

| Model (Mô hình) | seg_mIoU | IoU_transp | IoU_opaque | score_MAE | score_bias | state_acc | Cohen Kappa | missed_unrel | unsafe_rate | false_alarm |
|---|---|---|---|---|---|---|---|---|---|---|
| **FPN-R18 (Main)** | **0.617** | **0.231** | **0.785** | **4.82** | **+2.68** | **94.8%** | **0.902** | **6.1%** | **0.0%** | **0.4%** |
| FPN-R50 | 0.622 | 0.245 | 0.801 | 4.65 | +2.31 | 95.2% | 0.908 | 5.3% | 0.0% | 0.4% |
| UNet-R18 | 0.584 | 0.182 | 0.724 | 6.12 | +3.45 | 92.4% | 0.854 | 9.8% | 0.8% | 0.8% |
| PAN-R18-strict | 0.598 | 0.210 | 0.751 | 5.34 | +2.92 | 93.6% | 0.878 | 7.9% | 0.0% | 0.4% |

**Nhận xét rút ra từ thực nghiệm**:
1. **Nhóm đo được** ([results/exp_a_real/summary_models.csv](../results/exp_a_real/summary_models.csv)): 
   - Mô hình chính `FPN-R18` đạt `state_acc` lên tới **94.8%** và Cohen's Kappa đạt **0.902** (mức gần như tương đồng tuyệt đối), mặc dù mIoU phân đoạn chỉ đạt **0.617**.
   - Điều này mang ý nghĩa kỹ thuật rất lớn: *Hệ thống giám sát sức khỏe cảm biến không đòi hỏi phân vùng pixel hoàn hảo từng ranh giới, mà phụ thuộc vào việc ước lượng đúng mật độ khối bẩn và vị trí trọng yếu.*
2. **Xu hướng ngộ nhận sạch (Optimistic Bias)**:
   - Tất cả các mô hình đều có `score_bias` dương ($+2.31$ đến $+3.45$). Nghĩa là mô hình mạng nơ-ron luôn có xu hướng "nhìn đời sáng hơn thực tế", dự đoán lens sạch hơn so với nhãn GT. Nguyên nhân cốt lõi là do `IoU_transparent` quá thấp (chỉ từ 0.18 – 0.24), mô hình bỏ sót phần lớn màng mờ mỏng.
3. **An toàn tuyệt đối đối với lỗi nguy hiểm**:
   - `unsafe_rate` của FPN-R18 đạt **0.0%** (không có frame nào GT Unreliable bị đoán nhảy vọt thành Healthy). Đây là thành công của quy tắc fail-safe override ở vùng ROI.

---

### 3.2. Hiệu năng theo từng Camera (Front, Rear, Left, Right)
Trích xuất từ [results/exp_a_real/summary_per_camera.csv](../results/exp_a_real/summary_per_camera.csv) trên mô hình chính FPN-R18:

| Camera | Số frame | score_MAE | score_bias | state_acc | Kappa | missed_unrel | false_alarm |
|---|---|---|---|---|---|---|---|
| **FV (Trước)** | 128 | 4.12 | +2.15 | 96.1% | 0.925 | 4.8% | 0.0% |
| **RV (Sau)** | 125 | 5.68 | +3.42 | 92.8% | 0.868 | 8.3% | 0.8% |
| **MVL (Gương trái)** | 123 | 4.65 | +2.48 | 95.1% | 0.909 | 5.6% | 0.4% |
| **MVR (Gương phải)**| 121 | 4.81 | +2.65 | 95.0% | 0.905 | 5.9% | 0.4% |

**Nhóm đo được** ([results/exp_a_real/summary_per_camera.csv](../results/exp_a_real/summary_per_camera.csv)):
- Camera sau (RV) có sai số lớn nhất (`score_MAE` = 5.68, `state_acc` = 92.8%). 
- **Giải thích vật lý**: Camera lùi đặt phía đuôi xe, chịu tác động trực tiếp của luồng gió quẩn cuộn bùn đất và nước bắn từ bánh xe sau, dẫn đến việc tích tụ đồng thời cả mảng bùn dày và màng nước văng loang lổ, tạo ra thách thức phân đoạn lớn nhất cho mô hình.

---

### 3.3. Hiệu năng theo Mức độ Che phủ (Coverage Trends)
Trích xuất từ [results/exp_a_real/summary_per_coverage.csv](../results/exp_a_real/summary_per_coverage.csv):

| Dải GT Coverage | Số frame | GT Score Mean | Pred Score Mean | score_MAE | state_acc | missed_unrel |
|---|---|---|---|---|---|---|
| **< 10%** (Rất ít bẩn) | 68 | 91.4 | 93.1 | 2.85 | 98.5% | 0.0% |
| **10 – 30%** (Bẩn nhẹ) | 215 | 78.2 | 81.6 | 4.21 | 95.8% | 3.2% |
| **30 – 60%** (Bẩn trung bình)| 162 | 56.4 | 60.2 | 5.94 | 92.6% | 7.8% |
| **> 60%** (Bẩn nghiêm trọng) | 52 | 31.8 | 34.1 | 6.42 | 94.2% | 8.1% |

**Nhóm đo được**: Khi độ bẩn tăng dần từ dưới 10% lên trên 60%, sai số `score_MAE` tăng từ 2.85 lên 6.42 điểm. Tuy nhiên, `state_acc` ở nhóm bẩn nghiêm trọng (>60%) vẫn giữ ở mức 94.2% vì khi độ bẩn đã vượt quá 60%, dù mô hình có dự đoán sai vài phần trăm thì điểm số vẫn nằm trọn trong vùng `< 50` (`Unreliable`). Vùng ranh giới nhạy cảm nhất là $30\% - 60\%$, nơi điểm số dao động quanh ngưỡng phân giới 50 điểm giữa `Degraded` và `Unreliable`.

---

### 3.4. Phân tích bóc tách thuật toán (Ablation Study)
Đối chiếu 3 biến thể tính điểm trên cùng một tập ngưỡng phân loại tại [results/exp_a_real/ablation_scores.csv](../results/exp_a_real/ablation_scores.csv):

| Biến thể thuật toán | Công thức tính điểm | score_MAE | state_acc | unsafe_rate |
|---|---|---|---|---|
| **Coverage only** | $100 \times (1 - \text{Coverage})$ | 6.74 | 94.2% | 1.8% |
| **Coverage $\times$ Opacity** | $100 \times (1 - \text{Effective Occlusion})$ | 5.42 | 94.4% | 1.2% |
| **Full Pipeline (Nhóm)** | Kết hợp Coverage, Opacity, Vị trí tâm + ROI override | **4.82** | **94.8%** | **0.0%** |

**Nhóm đo được**: 
- Chỉ dùng diện tích che phủ (`Coverage only` như ý tưởng ban đầu của P3 - TiledSoilingNet) cho kết quả không tệ (`state_acc` = 94.2%), nhưng lại có `unsafe_rate` = 1.8% (để lọt nguy cơ tai nạn).
- Pipeline toàn diện của nhóm không chỉ hạ thấp `score_MAE` xuống **4.82** mà quan trọng nhất là đưa **`unsafe_rate` về đúng 0.0%**. Đây là minh chứng cụ thể cho giá trị đóng góp của thuật toán đánh giá cảm biến mà nhóm đã xây dựng.

---

## 4. Failure Cases (Phân tích lỗi & Cơ chế gây sai lệch)

Trong khuôn khổ Exp A, tôi đã cô lập các trường hợp sai lệch trạng thái nghiêm trọng nhất giữa GT và Dự đoán (xem ảnh minh họa tại [results/exp_a_real/failure_cases.png](../results/exp_a_real/failure_cases.png) và dữ liệu chi tiết tại [results/exp_a_real/failure_cases.csv](../results/exp_a_real/failure_cases.csv)).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FAILURE CASE F1 (LỖI NGUY HIỂM NHẤT)                     │
│                                                                                 │
│  Ảnh gốc RGB               Ground-Truth Mask            FPN-R18 Predicted Mask  │
│  ┌───────────────┐         ┌───────────────┐            ┌───────────────┐       │
│  │    Màng nước  │         │   Amber / Blue│            │      Dark     │       │
│  │   loang mờ    │  ───>   │ (Semi/Transp) │   ───>     │ (Đoán là Clear│       │
│  │   trên lens   │         │               │            │    hoàn toàn) │       │
│  └───────────────┘         └───────────────┘            └───────────────┘       │
│                            GT Score: 41.2               Pred Score: 76.5        │
│                            State: UNRELIABLE            State: DEGRADED         │
│                                                         (Ngộ nhận an toàn!)     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1. Failure Case F1: Hiện tượng Ngộ nhận Sạch (Optimistic Underestimation)
- **Tệp mẫu điển hình**: Frame camera sau `0421_RV.png` (và `1892_FV.png`).
- **Hiện tượng**:
  - **GT Mask**: Toàn bộ góc phần tư phía trên và tâm lens bị phủ bởi một màng nước mờ lớn (Semi-transparent / Transparent coverage đạt 48.6%, trong đó ROI trung tâm bị che mờ đáng kể). Điểm chuẩn GT: **41.2 điểm** $\rightarrow$ Trạng thái: **Unreliable**.
  - **Mô hình FPN-R18 dự đoán**: Chỉ nhận diện được một vài đốm rìa là Transparent, 75% diện tích màng nước bị mô hình phân loại nhầm thành **Clear (lớp 0)**. Điểm dự đoán: **76.5 điểm** $\rightarrow$ Trạng thái: **Degraded** (tiệm cận Healthy).
- **Cơ chế gốc rễ (Root Cause)**:
  1. **Repo/paper cho biết** (P1): Ranh giới giữa Transparent và Clear rất mơ hồ (*"differentiation between Transparent and Semi-transparent is somewhat ambiguous"*). Lớp Transparent thực chất là lớp nước làm biến dạng khúc xạ ánh sáng nhẹ hoặc mờ sương, giữ nguyên kết cấu cảnh nền phía sau. Mạng CNN dựa trên trích xuất đặc trưng biên (edges) và texture dễ dàng coi đây là bề mặt mặt đường hoặc bầu trời bình thường.
  2. Độ nhạy của hàm mất mát Cross-Entropy trong bài toán mất cân bằng lớp: Số lượng pixel Clean áp đảo tuyệt đối, khiến mạng nơ-ron có thiên hướng ưu tiên phân loại về lớp đa số (Clean) khi không chắc chắn.
- **Mức độ rủi ro trong xe tự hành**: **Cực kỳ nguy hiểm (Catastrophic Failure)**. Khi camera bị màng nước làm mờ nhưng hệ thống vẫn gắn trọng số $\text{Weight} = 0.5$ hoặc $1.0$, module phát hiện vật cản (Obstacle Detection) sẽ bỏ sót chướng ngại vật mờ phía trước do ảnh đầu vào bị mất nét mà không hề phát cảnh báo hỏng cảm biến.

### 4.2. Failure Case F2: Báo động giả do Bóng râm & Biến dạng Quang học (Pessimistic False Alarm)
- **Tệp mẫu điển hình**: `3105_MVL.png`.
- **Hiện tượng**: Lens thực tế khá sạch (GT Score: **84.3** $\rightarrow$ Healthy). Tuy nhiên mô hình dự đoán score tụt xuống **48.7** $\rightarrow$ Unreliable.
- **Cơ chế**: Vùng gương chiếu hậu có một vệt bóng đổ gắt của thân xe kết hợp với mặt đường nhựa sẫm màu dưới ánh nắng gắt ban ngày. Mạng FPN nhận diện nhầm dải bóng đen này thành một vệt bùn đặc (`Opaque`).
- **Mức độ rủi ro**: Ảnh hưởng đến độ khả dụng (Availability), khiến xe có thể dừng khẩn cấp hoặc từ chối bật tính năng tự lái một cách không cần thiết, tuy nhiên an toàn tính mạng vẫn được bảo toàn (Fail-Safe).

### 4.3. Phát hiện lỗi phụ: Artefact từ phép nội suy Resize của Repo gốc
- **Nhóm đo được** ([exp_a_real.py L200–206](../benchmark/exp_a_real.py#L200-L206)): Khi kiểm tra hàm tiền xử lý của repo gốc, phát hiện tác giả sử dụng phép resize mặc định của thư viện PIL (`Image.resize()` mặc định Bicubic cho ảnh Mode L) để co ảnh nhãn từ full-size về $512 \times 512$. Phép nội suy làm mịn giá trị pixel này đã vô tình làm biến đổi và sai lệch nhãn danh mục của **1.82% tổng số pixel nhãn** so với phép nội suy lân cận gần nhất (`Image.NEAREST`). Nhóm đã khắc phục triệt để lỗi này trong pipeline đọc nhãn của Exp A.

---

## 5. Engineering Decisions & Trade-offs (Quyết định Kỹ thuật & Đánh đổi)

Dựa trên các bằng chứng định lượng thu được từ Exp A, tôi đã cùng nhóm đưa ra các quyết định kỹ thuật then chốt:

### Quyết định 1: Đánh đổi An toàn vs. Khả dụng qua Cơ chế "ROI Override"
- **Bối cảnh**: Nếu chỉ dựa vào điểm số Health Score trung bình toàn ảnh, một vết bùn nhỏ nhưng che đúng vật cản trọng yếu ngay trung tâm làn đường chỉ làm giảm điểm từ 100 xuống 85 (vẫn được coi là Healthy).
- **Quyết định**: Bổ sung điều kiện ghi đè an toàn (**Hard Override**):
  $$\text{Nếu } \text{ROI}_{\text{opaque}} \ge 20\% \implies \text{Ép trạng thái về } \mathbf{Unreliable} \ (\text{Weight} = 0.0)$$
- **Đánh đổi (Trade-off)**:
  - *Mất đi*: Tăng tỷ lệ False Alarm lên thêm $0.4\%$ (chấp nhận một số trường hợp xe báo rửa kính sớm hơn nhu cầu).
  - *Được*: Triệt tiêu hoàn toàn các ca nguy hiểm chết người, đưa `unsafe_rate` về mức **0.0%**. Trong tiêu chuẩn an toàn chức năng ô tô ISO 26262 (ASIL-D), an toàn tính mạng luôn được ưu tiên tuyệt đối trước sự tiện lợi.

### Quyết định 2: Chiến lược Ngưỡng phân lớp bất đối xứng (Asymmetric Thresholds)
- **Bối cảnh**: Khảo sát độ nhạy ngưỡng trong [results/exp_a_real/threshold_sensitivity.csv](../results/exp_a_real/threshold_sensitivity.csv) giữa các cặp $(H \ge 80, U < 50)$ so với $(H \ge 85, U < 40)$.
- **Quyết định**: Lựa chọn cận trên Healthy $\ge 80$ và cận dưới Unreliable $< 50$. Đồng thời ở tầng điều khiển thời gian thực, thiết lập khoảng đệm trễ (Hysteresis Band): Trạng thái chỉ chuyển từ Unreliable lên Degraded khi điểm số thực tế phục hồi ổn định trên 55 điểm liên tục trong 10 frame nhằm tránh hiện tượng dao động tín hiệu điều khiển (actuator chattering).

### Quyết định 3: Kiến nghị nâng cấp thuật toán v2 từ dữ liệu Exp A
- **Kết luận từ Exp A**: Segmentation mask đơn thuần không thể giải quyết dứt điểm các ca màng nước trong suốt (Failure F1).
- **Đề xuất kỹ thuật**: Cần một kênh kiểm tra bổ trợ trực tiếp từ ảnh thô RGB. Đây chính là tiền đề thôi thúc thành viên TV4 xây dựng thực nghiệm Exp C: Kết hợp mật độ biên Laplacian/Sobel của ảnh thực để bắt dính hiện tượng mất nét do màng nước trong suốt gây ra.

---

## Nguồn, Version và Hướng dẫn Tái hiện Thực nghiệm

### Nguồn tham khảo (Trích lục từ [SOURCES.md](../SOURCES.md))
- **P1 (Paper chính của Repo)**: F. Beránek, V. Diviš, I. Gruber, *Soiling detection for Advanced Driver Assistance Systems*, arXiv 2511.09740 (11/2025). Đã đọc toàn văn: kế thừa kiến trúc mô hình, tập chia train/test 4503/497 frame và bảng lọc nhãn `filter_of_files.csv`.
- **P3 (Cơ sở ý tưởng Coverage & Trọng số)**: A. Das et al., *TiledSoilingNet*, ITSC 2020. Nhóm đã chứng minh giải pháp pixel mask của nhóm vượt trội hơn cách tiếp cận tile coverage thô ở khả năng triệt tiêu lỗi nguy hiểm.

### Môi trường thực thi & Version
- Python 3.10+, PyTorch 2.2.2, torchvision 0.17.2, scikit-learn 1.4.1, pandas 2.2+, numpy 1.26+, Pillow 9.5+.
- Toàn bộ pipeline benchmark chạy được trên CPU tiêu chuẩn (đạt tốc độ ~45 fps khi xử lý mask offline).

### Lệnh chạy tái hiện 100% bằng chứng của Exp A
Để tái hiện toàn bộ kết quả, log và biểu đồ đã trình bày trong báo cáo này, người chấm chỉ cần thực hiện 2 lệnh sau từ thư mục gốc của repository:

```bash
# 1. Tải tập con test dữ liệu WoodScape và các mô hình segmentation đã huấn luyện
python benchmark/fetch_woodscape_subset.py --evals

# 2. Chạy toàn bộ thực nghiệm Exp A (tự động xuất CSV, log và biểu đồ vào results/exp_a_real/)
python -m benchmark.exp_a_real
```

**Danh mục minh chứng đầu ra tự động được sinh ra trong repo**:
- Log chi tiết: [results/exp_a_real/log.txt](../results/exp_a_real/log.txt)
- Bảng so sánh 4 mô hình: [results/exp_a_real/summary_models.csv](../results/exp_a_real/summary_models.csv)
- Bảng phân tích 4 camera: [results/exp_a_real/summary_per_camera.csv](../results/exp_a_real/summary_per_camera.csv)
- Bảng phân tích coverage: [results/exp_a_real/summary_per_coverage.csv](../results/exp_a_real/summary_per_coverage.csv)
- Bảng ablation study: [results/exp_a_real/ablation_scores.csv](../results/exp_a_real/ablation_scores.csv)
- Bảng khảo sát ngưỡng: [results/exp_a_real/threshold_sensitivity.csv](../results/exp_a_real/threshold_sensitivity.csv)
- Biểu đồ phân tán GT vs Pred: [results/exp_a_real/scatter_gt_vs_pred.png](../results/exp_a_real/scatter_gt_vs_pred.png)
- Ma trận nhầm lẫn trạng thái: [results/exp_a_real/state_confusion.png](../results/exp_a_real/state_confusion.png)
- Minh chứng Failure Cases: [results/exp_a_real/failure_cases.png](../results/exp_a_real/failure_cases.png)
