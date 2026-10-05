# Báo cáo cá nhân: Hồ Ngọc Mai · 2A202602509 · TV4 Benchmark mô phỏng, cải tiến và quyết định kỹ thuật

**Nhóm 36Thieves · Chủ đề T1: Camera Health khi lens bẩn**  
**Repository chung:** <https://github.com/vuthien3002-sys/K4-Track4-Day04-36Thieves-Sensor-Reality-Sprint>  
**Phần phụ trách:** Exp B (soiling mô phỏng), Exp C (image check) và khuyến nghị triển khai.

Quy ước trong báo cáo: **[Đo]** là số do nhóm chạy; **[Thiết kế]** là giả định/cấu hình của lab; **[Giới hạn]** là điều chưa được chứng minh trên xe thật.

## 1. Problem

Camera fisheye của ADAS có thể bị bùn đất hoặc màng nước che lens. Nhóm dùng Camera Health Score để chuyển mask soiling thành `Healthy / Degraded / Unreliable` và weight fusion `1.0 / 0.5 / 0.0`.

Phần tôi kiểm tra hai vấn đề không thể hiện rõ nếu chỉ chạy một mask cố định:

1. Khi tăng coverage, độ đục hoặc đưa vết bẩn vào tâm ảnh, Health Score có giảm đúng hướng không?
2. Mask transparent có thể đánh giá một ảnh mờ nặng là chỉ Degraded. Một kiểm tra trực tiếp từ ảnh có phát hiện được ca này không, và phải đánh đổi bao nhiêu báo động giả?

**Claim:** Trên cùng frame baseline, score không được tăng khi coverage thêm vào tăng; opaque và vết ở tâm phải giảm score mạnh hơn transparent/vết ở rìa. Image check có thể hạ các ca transparent nặng xuống Unreliable, nhưng chỉ nên dùng sau khi đo trade-off trên cảnh thật.

## 2. Method

### Exp B — benchmark mô phỏng có đối chứng

Mã chạy: [`benchmark/exp_b_controlled.py`](../benchmark/exp_b_controlled.py). Chọn 8 frame test ít bẩn nhất cho mỗi camera (32 frame, 4 camera) làm baseline; vì dữ liệu WoodScape không có frame sạch hoàn toàn, baseline vẫn còn 3.0–11.1% GT coverage ([base_frames.csv](../results/exp_b_controlled/base_frames.csv)). Trên **chính các frame đó**, tôi chỉ thay một nhân tố:

- loại soiling: transparent hoặc opaque;
- vị trí: centre hoặc periphery;
- coverage thêm vào vùng fisheye hợp lệ: 10%, 25%, 40%, 60%.

`synth_soiling.py` dùng seed cố định `2026`; vùng 10% nằm trong vùng 25% cùng điều kiện, và transparent/opaque dùng đúng cùng vùng. Vì vậy chênh lệch được quy cho loại, vị trí hoặc coverage thay đổi, không phải do đổi ảnh đầu vào. Tôi đo Health Score/state/weight và ba proxy trong critical ROI: edge density, Laplacian sharpness, RMS contrast. Các proxy **không** được đưa vào score v1.

### Exp C — image check cho failure transparent

Mã chạy: [`benchmark/exp_c_image_check.py`](../benchmark/exp_c_image_check.py). Bản v2 giữ nguyên score từ mask (v1), nhưng ép state thành Unreliable nếu:

`ROI edge density / clean-reference edge density < min_ratio`

Reference là median edge density của frame GT-Healthy, lấy từ phiên ghi hình có id `<2500`; đánh giá trên phiên id `≥2500`. Các ngưỡng thử là 0.10, 0.15, 0.20, 0.25, 0.30; lựa chọn thử nghiệm [Thiết kế] là `min_ratio=0.20`.

## 3. Benchmark và bằng chứng

Lệnh tái hiện khi đã tải WoodScape subset, RGB và prediction theo hướng dẫn repo:

```bash
python -m benchmark.exp_b_controlled
python -m benchmark.exp_c_image_check
```

Exp B lưu `per_frame.csv`, `summary_conditions.csv`, log và các plot vào `results/exp_b_controlled/`; Exp C lưu reference, bảng v1/v2, bảng trade-off và plot vào `results/exp_c_image_check/`.

**Kết quả chung đã ghi nhận [Đo chung]:** score không tăng khi coverage tăng trên **100%** frame/điều kiện kiểm tra; Spearman giữa score và edge density còn lại trong ROI là **0.68**. Với ca transparent/centre/60%, v2 image check bắt được **97%** ca; trên frame GT-Healthy của tập đánh giá không có frame nào bị hạ nhầm. Đổi lại, tỷ lệ camera bị xếp Unreliable tăng từ **22.2% lên 28.2%**. Các số này là kết quả benchmark nhóm, được TV1 tổng hợp; TV4 dùng chúng để đưa ra quyết định dưới đây, không gán chúng là kết quả từ paper.

Các kiểm tra cần đọc từ log Exp B:

- C1: score không tăng khi coverage tăng, xét riêng từng frame và từng điều kiện;
- C2: opaque phải có score thấp hơn transparent ở cùng vị trí/coverage;
- C3: centre phải có score thấp hơn periphery ở cùng loại/coverage;
- C4: tương quan Spearman giữa score và edge/sharpness/contrast còn lại trong ROI.

Kết quả không được diễn giải là mAP/recall của object detector, vì Exp B dùng soiling tổng hợp và các proxy ảnh. Nó chỉ kiểm tra tính nhạy đúng hướng của health-score pipeline dưới một thay đổi có kiểm soát.

## 4. Failure case và cải tiến

**Failure case:** màng bẩn transparent phủ mạnh ở vùng tâm. Theo mask-only pipeline, transparent có opacity 0.33 nên ngay cả khi phủ toàn lens score là 67, vẫn là Degraded; camera có thể thực tế đã thiếu edge/texture để perception đáng tin.

**Cải tiến v2:** render transparent trong Exp B làm mờ và làm nhạt ảnh, sau đó Exp C dùng ROI edge density để phát hiện thông tin ảnh mất đi. Thước đo chính là tỷ lệ frame transparent/centre/60% được hạ xuống Unreliable; chi phí là tỷ lệ GT-Healthy và GT-Degraded bị hạ nhầm trên phiên ghi hình khác.

**[Giới hạn]** Edge density thấp cũng có thể do đêm, sương mù, mặt đường trống hoặc cảnh ít texture dù lens sạch. Do đó v2 chỉ được phép hạ state (không nâng state), và không được coi là ground truth về soiling.

## 5. Engineering decision và trade-off

**Quyết định đề xuất:** triển khai v1 mask-only là tín hiệu cơ sở; chạy image check v2 ở chế độ shadow log trước. Chỉ bật quyền loại camera khi dữ liệu vận hành cho thấy false-alarm theo từng loại cảnh ở mức chấp nhận được.

| Quyết định | Lợi ích | Chi phí / metric cần theo dõi |
|---|---|---|
| ROI-weighted score + ROI opaque override | Ưu tiên vết che vùng tác vụ, không chỉ diện tích | Có thể giảm availability; theo dõi share Unreliable |
| v2 chỉ hạ state | Không tạo cảm giác camera khỏe hơn thực tế | Có báo động giả; đo GT-Healthy/Degraded flagged |
| Ngưỡng 0.20 ở chế độ shadow | Có điểm xuất phát để so sánh | Chưa hiệu chỉnh; sweep 0.10–0.30 trước khi chốt |
| Seed và cùng baseline frame | Kết quả Exp B tái hiện được, giảm nhiễu so sánh | Soiling tổng hợp chưa thay thế bùn/mưa thật |

Vòng tiếp theo nên thu chuỗi video có nhãn điều kiện ánh sáng/thời tiết, hiệu chỉnh ngưỡng theo camera, và thêm EMA + hysteresis để state không nhảy giữa các frame. Metrics cần báo cáo: missed-Unreliable, false-alarm, availability và thời gian giữ state theo chuỗi.

## Nguồn và phạm vi

Nguồn bài toán, repo và paper được ghi trong [`SOURCES.md`](../SOURCES.md). Các tham số opacity, ROI, trọng số severity, ngưỡng state và image-check là giả định thiết kế trong [`camera_health/config.yaml`](../camera_health/config.yaml), không phải một chuẩn an toàn xe. Kết quả benchmark của nhóm phải tách biệt với kết luận từ paper/repo.
