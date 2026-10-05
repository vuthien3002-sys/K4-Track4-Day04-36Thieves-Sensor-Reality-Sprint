# Nguồn tham khảo (Bước 2): nhóm 36Thieves, chủ đề T1 lens bẩn

Truy cập ngày 2026-10-05. Cột "Đã đọc" ghi đúng phần nhóm thực sự đọc. Paper nào chỉ đọc abstract thì chỉ trích nội dung trong abstract.

## Danh sách

| # | Nguồn | Link | Đã đọc | Vai trò trong bài |
|---|---|---|---|---|
| P1 | F. Beránek, V. Diviš, I. Gruber, *Soiling detection for Advanced Driver Assistance Systems*, arXiv 2511.09740 (11/2025) | <https://arxiv.org/abs/2511.09740> · code: <https://github.com/filipberanek/woodscape_revision> | Bản HTML đầy đủ: dataset, kết quả, kết luận | **Paper của chính repo nhóm dùng.** Đây là nguồn của các mask dự đoán và của cách chia train/test. |
| P2 | M. Uřičář, P. Křížek, G. Sistu, S. Yogamani, *SoilingNet: Soiling Detection on Automotive Surround-View Cameras*, ITSC 2019 | <https://arxiv.org/abs/1905.01492> | Abstract | Nguồn gốc của bài toán soiling và hai loại vết bẩn opaque/transparent trong WoodScape |
| P3 | A. Das, P. Křížek, G. Sistu et al., *TiledSoilingNet: Tile-level Soiling Detection on Automotive Surround-view Cameras Using Coverage Metric*, ITSC 2020 | <https://arxiv.org/abs/2007.00801> | Abstract | Ý tưởng dùng **coverage** để ra quyết định: kích hoạt hệ thống rửa lens, giảm độ tin ở vùng bẩn |
| P4 | M. Uřičář, G. Sistu, H. Rashed et al., *Let's Get Dirty: GAN Based Data Augmentation for Camera Lens Soiling Detection in Autonomous Driving*, WACV 2021 | <https://arxiv.org/abs/1912.02249> | Abstract | Cơ sở cho việc **tạo vết bẩn nhân tạo** (Exp B) |
| P5 | S. Yogamani et al., *WoodScape: A Multi-Task, Multi-Camera Fisheye Dataset for Autonomous Driving*, ICCV 2019 | <https://arxiv.org/abs/1905.01489> · data: <https://github.com/valeoai/WoodScape> | Abstract | Nguồn của dataset: 4 camera fisheye surround-view, soiling là 1 trong 9 task |

## Từng nguồn: phương pháp nhận gì, tạo gì, đo bằng gì

**P1: Beránek et al. 2025 (paper của repo)**

- **Input → output:** ảnh camera WoodScape → segmentation theo pixel thành 4 lớp: Clear, Transparent, Semi-Transparent, Opaque.
- **Dataset:**
  - Paper chỉ ra lỗi rò rỉ dữ liệu trong cách chia gốc của WoodScape: *"These 7 or 8 images are split between the train and test folder, which means we would be training the network on the same scene as we are later concluding the test."*
  - Tác giả chia lại theo chuỗi: *"totalling 4503 images for training and 497 images for test."*
  - Họ đếm được 685 ảnh có nhãn bị cắt, 519 ảnh có ranh giới Transparent/Semi mơ hồ, và khoảng 1375 ảnh gán lớp không nhất quán giữa các cảnh giống nhau. Từ đó họ tạo 4 tập con: all, correct, correct clear, correct clear strict. Đây chính là cột `filter_of_files.csv` trong repo.
- **Model và metric:** 9 kiến trúc segmentation (UNet, UNet++, DeepLabV3/V3+, FPN, PSPNet, MaNet, LinkNet, PAN), đánh giá bằng accuracy theo pixel. FPN tốt nhất với 0.940. Mọi phương pháp segmentation đều vượt cách phân loại theo tile (TiledSoilingNet, 0.874): *"All the tested semantic segmentation methods surpass the original tile-level classification approach."*
- **Chính paper thừa nhận:** *"the differentiation between the Transparent and Semi-transparent classes is somewhat ambiguous according to this definition."* Hướng tiếp theo họ nêu: thêm ablation cho hyperparameter, thử transformer, và cần thêm dữ liệu.
- **Paper không làm:** không có bước nào đi từ mask đến quyết định về camera (score, state, weight). Đây là khoảng trống nhóm lấp vào.

**P2: SoilingNet (abstract)**

Camera surround-view tiếp xúc trực tiếp với môi trường, nên khi bẩn thì hiệu năng giảm mạnh hơn nhiều so với các sensor khác. Nhóm tác giả tạo dataset với hai loại vết bẩn opaque và transparent, sau đó phát hành nó như một phần của WoodScape. Phương pháp gồm CNN, kết hợp multi-task với object detection, và dùng GAN để tăng cường dữ liệu.

**P3: TiledSoilingNet (abstract)**

- Phương pháp hồi quy trực tiếp **diện tích của từng loại vết bẩn trong mỗi tile** (coverage metric), thay cho segmentation. Decoder nhanh hơn segmentation "an order of magnitude".
- Mục đích của coverage: kích hoạt hệ thống rửa lens, và cho phép *"partial functionality in unsoiled areas while reducing confidence in soiled areas"*.
- Đây chính là logic của Camera Weight: camera bẩn thì giảm độ tin chứ không phải lúc nào cũng loại bỏ.

**P4: Let's Get Dirty (abstract)**

- GAN sinh ra **ảnh bẩn kèm sẵn mask vết bẩn**, nên không tốn công gán nhãn tay.
- Khi dùng ảnh này để tăng cường dữ liệu train, độ chính xác soiling detection tăng **18%**.
- Paper cũng đánh giá mức suy giảm của semantic segmentation khi gặp dữ liệu bẩn.

**P5: WoodScape (abstract)**

Đây là dataset fisheye ô tô lớn đầu tiên: 4 camera surround-view, 9 task, trong đó có soiling detection.

## Nguồn cho biết gì, nhóm đo được gì

Các số dưới đây **không so sánh trực tiếp** với nhau, vì khác metric hoặc khác điều kiện đo.

| Chủ đề | Paper/repo cho biết | Nhóm đo được (link) |
|---|---|---|
| Rò rỉ train/test | P1: dataset gốc có 7–8 frame cùng cảnh bị chia cả vào train lẫn test. P1 đã chia lại thành 4503/497. | Trên đúng 497 frame test này: chỉ có 399 mask khác nhau, và 2% frame test có mask trùng hệ với một frame train/val ([BENCHMARK_RESULTS §6](BENCHMARK_RESULTS.md)) |
| Lớp transparent | P1: ranh giới Transparent/Semi "somewhat ambiguous" | IoU transparent 0.13–0.29 so với opaque 0.65–0.83. Đó là nguyên nhân của failure case F1 ([summary_models.csv](results/exp_a_real/summary_models.csv)) |
| Chất lượng segmentation | P1: FPN đạt accuracy 0.940 | File evaluation của repo cho FPN-R50 accuracy 0.945. Nhóm tự tính lại mIoU = 0.617 (resize NEAREST, chỉ trong vùng fisheye) |
| Từ coverage đến quyết định | P3: dùng coverage để kích hoạt rửa lens và giảm độ tin ở vùng bẩn | Nhóm tính coverage, opacity, vị trí và ROI rồi đưa ra score, state, weight. Ablation: chỉ dùng coverage cho state acc 94.2%, so với 94.8% của pipeline đầy đủ ([ablation_scores.csv](results/exp_a_real/ablation_scores.csv)) |
| Vết bẩn nhân tạo | P4: GAN sinh ảnh bẩn kèm mask, train thêm thì tăng 18% | Exp B dùng vết bẩn tạo theo thủ tục (nhiễu + blur/màu bùn, **không dùng GAN**), nên kém thực tế hơn. Nhóm chỉ dùng nó để kiểm tra xu hướng của score, không dùng để train ([exp_b log](results/exp_b_controlled/log.txt)) |

## Nguồn chưa đọc, không dùng làm bằng chứng

Trong danh sách gợi ý S1–S9 của đề lab, phần lớn không liên quan đến lens bẩn: Jetson, PX4, Basler, radar, LiDAR calibration. Nguồn S5 (Dong et al., CVPR 2023, benchmark độ bền với corruption) có liên quan nhưng nhóm **chưa đọc**.
