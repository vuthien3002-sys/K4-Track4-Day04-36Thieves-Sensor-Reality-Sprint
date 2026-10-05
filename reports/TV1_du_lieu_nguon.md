# Báo cáo cá nhân: Vũ Đức Thiện · 2A202602437 · TV1 Dữ liệu và nguồn

Nhóm 36Thieves · T1 Camera health khi lens bẩn · Repo chung: <https://github.com/vuthien3002-sys/K4-Track4-Day04-36Thieves-Sensor-Reality-Sprint> · Bằng chứng chung: [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md) · Nguồn: [SOURCES.md](../SOURCES.md)

Trong báo cáo này, mỗi kết luận được đánh dấu theo nguồn gốc:
- **[Đo]**: nhóm tự đo được, kèm link tới log hoặc plot.
- **[Nguồn]**: repo hoặc paper cho biết, kèm link.
- **[Giả thuyết]**: suy luận, chưa được kiểm chứng.

## Phần tôi phụ trách

Tôi phụ trách phần dữ liệu và nguồn tham khảo của nhóm:

- **Đọc nguồn:** đọc README, script dự đoán [pytorch_l_base_pred.py](../networks_run/pytorch_networks/predict/pytorch_l_base_pred.py) và script đánh giá [evaluation.py](../networks_run/evaluation/evaluation.py) của repo. Tìm và đọc paper gốc của repo (P1), cùng 4 paper liên quan. Danh sách tổng hợp ở [SOURCES.md](../SOURCES.md).
- **Tải dữ liệu:** viết [fetch_woodscape_subset.py](../benchmark/fetch_woodscape_subset.py), dùng HTTP range request để chỉ lấy đúng phần cần trong 2 file zip trên Google Drive (5.5 GB + 6.6 GB). Máy chỉ còn khoảng 5 GB trống, nên không tải được cả hai file. Tổng dung lượng thực tải khoảng 600 MB, lưu lại khoảng 200 MB.
- **Dựng mặt nạ fisheye:** viết [build_valid_masks.py](../benchmark/build_valid_masks.py) để xác định vùng nhìn thấy được của từng camera.
- **Kiểm tra dữ liệu:** viết [tv1_data_checks.py](../benchmark/tv1_data_checks.py), kết quả lưu ở [results/tv1_data/](../results/tv1_data/).
- **Quản lý repo:** chủ repo, quản lý `.gitignore`, `TEAMMATES.md`, và chia file cho 4 thành viên commit.

Kết quả chạy lại các lệnh của tôi:

| Lệnh | Kết quả |
|---|---|
| `python benchmark/fetch_woodscape_subset.py --evals` | Tải 156 file CSV evaluation của 39 model trong 33 giây. Cả 156 file giống hệt bản đã dùng trong benchmark (so bằng hash). |
| `python -m benchmark.build_valid_masks` | Dựng lại 4 mặt nạ fisheye, giống hệt bản đã commit. |
| `python -m benchmark.tv1_data_checks` | Ghi [log.txt](../results/tv1_data/log.txt), [coverage_hist.png](../results/tv1_data/coverage_hist.png), [valid_masks.png](../results/tv1_data/valid_masks.png) |

## 1. Problem

- **Nền tảng:** xe ADAS có hệ surround-view gồm 4 camera fisheye: FV (trước), RV (sau), MVL, MVR (gương trái/phải). Dữ liệu là bộ WoodScape soiling.
- **Tính năng bị ảnh hưởng:** sensor fusion cần biết nên tin từng camera bao nhiêu. Nhóm xây một bộ giám sát sức khỏe camera, trả ra Camera Health Score (0–100), state (Healthy / Degraded / Unreliable) và weight (1.0 / 0.5 / 0).
- **Failure case:** lens bị bẩn (bùn, nước, bụi). Mask chia vết bẩn thành 3 mức: transparent, semi-transparent, opaque.
  - [Nguồn] Theo SoilingNet ([P2](https://arxiv.org/abs/1905.01492)), camera surround-view tiếp xúc trực tiếp với môi trường, nên khi bẩn thì hiệu năng giảm mạnh hơn nhiều so với các sensor khác.
- **Claim ban đầu:** khi vết bẩn tăng coverage, đục hơn, hoặc nằm gần tâm/ROI, thì Health Score giảm và state chuyển Healthy → Degraded → Unreliable. Score tính từ mask *dự đoán* chỉ lệch vài điểm so với score tính từ mask *GT*.
- **Từ góc dữ liệu:** claim chỉ kiểm chứng được khi biết rõ đang đo trên dữ liệu gì.
  - [Đo] Không có frame test nào sạch hoàn toàn. Coverage thấp nhất là 3.0%.
  - Vì vậy nhóm không có baseline "lens sạch" thật. Exp B phải dùng các frame ít bẩn nhất làm baseline.

## 2. Method

| Câu hỏi khi đọc nguồn | Repo / paper (nguồn) | Nhóm |
|---|---|---|
| Nhận gì, tạo gì? | [Nguồn] Repo nhận ảnh RGB và tạo mask 4 lớp (0 clear, 1 transparent, 2 semi, 3 opaque) kích thước 512×512. [P1](https://arxiv.org/abs/2511.09740) so sánh 9 kiến trúc segmentation. | Nhận mask (kèm ảnh nếu có), tính 5 đặc trưng → severity → score → state → weight |
| Đo chất lượng bằng gì? | [Nguồn] P1 dùng accuracy theo pixel; FPN tốt nhất với 0.940. File evaluation của repo có thêm IoU theo từng lớp. | State accuracy, tỉ lệ bỏ sót Unreliable, sai số score (điểm) |
| Chạy được ở lớp không? | Train lại cần GPU và 12.1 GB dữ liệu cộng model | Không train lại. Dùng mask dự đoán mà repo đã lưu sẵn, tải từng phần bằng HTTP range. |
| Nhóm tái hiện phần nào? | Dữ liệu và mask dự đoán của 39 model | Phần mask → score do nhóm tự viết, nằm trong [camera_health/](../camera_health/) |

Tóm tắt công thức (TV2 trình bày chi tiết, xem [BENCHMARK_RESULTS §2.2](../BENCHMARK_RESULTS.md)):
- Mỗi pixel được đổi thành độ cản sáng: 0 / 0.33 / 0.66 / 1.0.
- Severity S = 0.3 × (độ che phủ toàn ảnh) + 0.3 × (độ che phủ có trọng số theo tâm) + 0.4 × (độ che phủ trong ROI).
- Score = 100 × (1 − S).
- State: Healthy nếu score ≥ 80; Unreliable nếu score < 50 hoặc opaque phủ ≥ 50% ROI; còn lại là Degraded.

**Mặt nạ fisheye (phần của tôi):**
- Mặt nạ dựng từ **trung vị độ sáng của nhiều frame** cho mỗi camera, thay vì đặt ngưỡng trên từng ảnh. Lý do: vết bùn đen nằm trong vòng tròn fisheye cũng tối như góc ảnh, nên ngưỡng từng ảnh sẽ loại nhầm chúng ra khỏi vùng hợp lệ.
- [Đo] Vùng nhìn thấy theo camera: FV 95.6%, RV 88.9%, MVL 94.5%, MVR 95.5% ([valid_masks.png](../results/tv1_data/valid_masks.png)).

## 3. Benchmark

### Dữ liệu dùng để đo

Toàn bộ số liệu dưới đây là [Đo], lấy từ [results/tv1_data/log.txt](../results/tv1_data/log.txt).

| Kiểm tra | Kết quả |
|---|---|
| Số frame | 5000 frame có GT; 497 frame test và 4503 frame train/val. [Nguồn] Khớp đúng cách chia 4503/497 của P1. |
| Frame test theo camera | FV 139, RV 129, MVL 113, MVR 116 |
| Định dạng mask | GT: 1280×960, mode L, 4 lớp 0–3. Mask dự đoán: 512×512. |
| Mức bẩn của test (GT) | Coverage thấp nhất 3.0%, trung vị 27.0%, cao nhất 100%. Có 154 frame bẩn hơn 50%. **0 frame sạch hoàn toàn.** Xem [coverage_hist.png](../results/tv1_data/coverage_hist.png). |
| Tỉ lệ pixel theo lớp (test) | clear 62.0%, transparent 6.0%, semi 8.7%, opaque 23.3% |
| Mask trùng nhau | 497 frame test chỉ có **399 mask khác nhau**. Nhóm lớn nhất gồm 8 frame liên tiếp, từ 0268_MVR đến 0275_MVR. |
| Rò rỉ train/test | **8/497 frame test (1.6%)** có mask trùng hệ với một frame train/val |
| Phiên ghi hình | id 0–1999 có 249 frame, id 4000–4999 có 248 frame. Đây là 2 khối tách biệt, Exp C dùng cách chia này. |
| Lỗi resize nhãn của repo | Resize bicubic so với NEAREST chỉ đổi 0.20% pixel, không đáng kể |

### Kết quả chính của nhóm

Các số dưới đây là [Đo], chi tiết ở [BENCHMARK_RESULTS §3](../BENCHMARK_RESULTS.md).

- **Exp A (dữ liệu thật):** với model FPN-R18, state đúng 94.8%, sai số score 2.95 điểm, bỏ sót Unreliable 2.8%. Model FPN-R50 có mIoU bằng FPN-R18 nhưng bỏ sót Unreliable 9.1%.
- **Exp B (vết bẩn mô phỏng):** score không bao giờ tăng khi coverage tăng, đúng trên 100% frame. Tương quan Spearman giữa score và lượng cạnh còn lại trong ROI là 0.68.
- **Exp C (kiểm tra ảnh):** v2 bắt được 97% trường hợp transparent · centre · 60%, không báo động sai trên frame GT Healthy. Đổi lại, tỉ lệ camera bị loại tăng từ 22.2% lên 28.2%.

## 4. Failure case

Tôi chọn **F1: model bỏ sót vết bẩn transparent, nên camera trông khỏe hơn thực tế**. Đây là failure case gắn trực tiếp với chất lượng nhãn dữ liệu. Hình minh họa ở [failure_cases.png](../results/exp_a_real/failure_cases.png).

- **Điều kiện đầu vào:** frame 4500_RV, camera sau bị phủ cả vết transparent và semi-transparent.
- **Tác động lên metric [Đo]:**
  - Tính từ GT: score 45.0 → Unreliable, weight 0.
  - Tính từ mask FPN-R18: score 52.9 → Degraded, weight 0.5.
  - 10% khung hình là transparent theo GT nhưng bị dự đoán thành clear.
  - Toàn tập test có 16 frame được đánh giá lạc quan hơn GT.
- **Repo và paper cho biết [Nguồn]:**
  - Trên cả 39 model, IoU lớp transparent chỉ 0.10–0.30, trong khi opaque đạt 0.64–0.82.
  - P1 tự nhận định nghĩa nhãn này mơ hồ: *"the differentiation between the Transparent and Semi-transparent classes is somewhat ambiguous"*.
  - P1 đếm được 519 ảnh có ranh giới Transparent/Semi không rõ.
- **Hệ quả [Giả thuyết]:**
  - Fusion sẽ tin camera này một nửa (weight 0.5), trong khi theo GT nó phải bị loại.
  - Một phần lỗi có thể đến từ chính nhãn mơ hồ: model học từ ranh giới không nhất quán.
  - Nhóm chưa đo tác động lên detector, và chưa tách được phần lỗi do nhãn với phần lỗi do model.

**Giới hạn của dữ liệu:**
- Các frame không độc lập. 497 frame chỉ có 399 mask khác nhau, nên số mẫu hiệu dụng khoảng 400. Mọi tỉ lệ % trong báo cáo đếm trùng một phần các cảnh lặp.
- Rò rỉ chỉ được kiểm tra ở mức mask trùng hệ (1.6%). Trường hợp frame gần giống nhau nhưng không trùng hệ thì chưa kiểm tra.
- Các ảnh mẫu nhóm đã xem đều là cảnh ban ngày. Nhóm chưa đo phân bố ngày/đêm hay mưa trong tập test.
- Ảnh bị resize từ 1280×960 về 512×512 giống repo, nên tỉ lệ khung hình bị méo.

## 5. Engineering decision

**Quyết định của nhóm:**
- Dùng model FPN-R18. Lý do: bỏ sót Unreliable 2.8% so với 9.1% của FPN-R50, dù mIoU gần như bằng nhau (0.616 và 0.617).
- Score từ mask là tín hiệu chính. Phần kiểm tra ảnh (v2) chỉ được phép hạ state và chạy ở chế độ shadow trước.

**Đề xuất từ góc dữ liệu**, kèm metric để kiểm chứng ở vòng sau:

1. **Đánh giá theo chuỗi, không theo frame.** Báo cáo metric trên 399 mask khác nhau, hoặc trên từng chuỗi frame, để các cảnh lặp không bị đếm nhiều lần. Kiểm tra thêm frame gần trùng bằng perceptual hash. Metric: tỉ lệ bỏ sót Unreliable tính theo chuỗi.
2. **Bổ sung dữ liệu đêm, mưa và cảnh ít chi tiết trước khi bật v2.** v2 dựa vào mật độ cạnh. Trong Exp C, 15 frame bị hạ state chỉ đến từ 3 cảnh, trong đó 2 cảnh là mặt đường MVR ít chi tiết. Metric: tỉ lệ báo động sai của v2 trên từng loại cảnh.
3. **Làm rõ định nghĩa nhãn transparent/semi rồi train lại** với trọng số cao hơn cho lớp transparent. Metric: IoU lớp transparent và tỉ lệ bỏ sót Unreliable.

**Trade-off:**
- Ngưỡng càng chặt (loại camera sớm hơn) thì càng an toàn, nhưng mất nhiều availability hơn. [Đo] Ngưỡng Unreliable 60 bắt hết frame Unreliable, nhưng tỉ lệ camera bị loại tăng từ 28.8% lên 32.4%.
- Với hệ nhiều camera, các camera còn lại có thể bù cho camera bị loại. Với hệ một camera, nên ưu tiên trạng thái Degraded kèm cảnh báo người lái.

## Nguồn, version, lệnh chạy

- **Repo nhóm:** <https://github.com/vuthien3002-sys/K4-Track4-Day04-36Thieves-Sensor-Reality-Sprint>, nhánh `main`. Commit của tôi:
  - `0142cb0`: công cụ tải dữ liệu, mặt nạ fisheye, `.gitignore`
  - `bf763e5`: `SOURCES.md`, `TEAMMATES.md`
  - `c1e08da`: kiểm tra dữ liệu `results/tv1_data/`, báo cáo TV1
  - `ede9490`: đưa log, plot của Exp A/B/C, unit test và `BENCHMARK_RESULTS.md` lên repo
  - `f17fc04`, `40838e1`: dọn repo và đồng bộ số liệu giữa các báo cáo
- **Code gốc:** <https://github.com/filipberanek/woodscape_revision>, commit `e9dc138`.
- **Dữ liệu (link lấy từ README):** `woodscape_input.zip` (Drive id `1WNlDBADwlaheMaVpIjEeAMklw7jle9Ja`, 5.5 GB) và `model_outputs.zip` (id `13k17SjgQHZCO-1Ctr3DY_bW6DGvQZZie`, 6.6 GB).
- **Paper đọc toàn văn:**
  - P1: Beránek, Diviš, Gruber, *Soiling detection for Advanced Driver Assistance Systems*, <https://arxiv.org/abs/2511.09740>.
- **Paper chỉ đọc abstract:**
  - P2 SoilingNet: <https://arxiv.org/abs/1905.01492>
  - P3 TiledSoilingNet: <https://arxiv.org/abs/2007.00801>
  - P4 Let's Get Dirty: <https://arxiv.org/abs/1912.02249>
  - P5 WoodScape: <https://arxiv.org/abs/1905.01489>
- **Lệnh chạy:**

```
python benchmark/fetch_woodscape_subset.py --evals
python benchmark/fetch_woodscape_subset.py --gt --rgb --models fpn_resnet18_torch_cross_entropy_correct_files fpn_resnet50_torch_cross_entropy_all_files unet_resnet18_torch_cross_entropy_all_files pan_resnet18_torch_cross_entropy_correct_clear_strict_files
python -m benchmark.build_valid_masks
python -m benchmark.tv1_data_checks
```
