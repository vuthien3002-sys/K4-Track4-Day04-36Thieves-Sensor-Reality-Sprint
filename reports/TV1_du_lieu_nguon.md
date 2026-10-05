# Báo cáo cá nhân: _(Họ tên)_ · _(MSSV)_ · TV1 Dữ liệu và nguồn

Nhóm 36Thieves · T1 Camera health khi lens bẩn · Repo chung: <https://github.com/vuthien3002-sys/K4-Track4-Day04-36Thieves-Sensor-Reality-Sprint> · Bằng chứng chung: [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md)

> Hãy viết bằng lời của bạn và xóa các dòng gợi ý (bắt đầu bằng `>`) trước khi nộp. Mỗi kết luận ghi rõ loại câu: **"Nhóm đo được…"** (kèm link log/plot), **"Repo/paper cho biết…"** (kèm link), hoặc **"Giả thuyết…"**.

## Phần tôi phụ trách

> **Việc của bạn:**
> - Đọc README, [pytorch_l_base_pred.py](../networks_run/pytorch_networks/predict/pytorch_l_base_pred.py) và [evaluation.py](../networks_run/evaluation/evaluation.py).
> - Tải dữ liệu bằng [fetch_woodscape_subset.py](../benchmark/fetch_woodscape_subset.py).
> - Dựng mặt nạ fisheye bằng [build_valid_masks.py](../benchmark/build_valid_masks.py).
> - Kiểm tra trùng nhãn: 497 frame nhưng chỉ có 399 mask khác nhau.
> - Kiểm tra rò rỉ giữa train và test: chỉ 2% frame trùng mask.
> - Nếu kịp: đọc paper WoodScape (ICCV 2019) và SoilingNet (2019), ghi link chính xác.
>
> **Lệnh cần chạy lại:** `python benchmark/fetch_woodscape_subset.py --evals`, rồi `python -m benchmark.build_valid_masks`
>
> **Câu hỏi phải trả lời được khi bị hỏi:**
> - Vì sao không tải full dataset? (5.5 GB + 6.6 GB, máy còn 5 GB trống; nhóm dùng HTTP range)
> - Mask có những lớp nào?
> - Vì sao mặt nạ fisheye lấy từ trung vị của nhiều ảnh, không threshold từng ảnh? (Vì bùn đen nằm trong vòng tròn sẽ bị loại nhầm.)
> - Frame test có độc lập với nhau không?

## 1. Problem

> Phần trọng tâm của bạn. Nêu: nền tảng ADAS surround-view có 4 camera fisheye; tính năng giám sát sức khỏe camera cho fusion; failure là lens bẩn với 3 mức độ đục. Viết claim ban đầu theo dạng "thay đổi gì → metric dự kiến đổi ra sao".

## 2. Method

> Phần trọng tâm của bạn. Viết bảng "nguồn nhận gì, tạo gì, đo bằng gì, chạy được không, nhóm tái hiện phần nào". Phân biệt rõ phần repo có sẵn (RGB → mask) với phần nhóm tự làm (mask → score). Phần công thức chỉ cần tóm tắt 2–3 câu và dẫn sang TV2 hoặc mục 2.2 của BENCHMARK_RESULTS.

## 3. Benchmark

> Tóm tắt ngắn Exp A, B, C, mỗi phần 1–2 số chính. Trình bày kỹ phần dữ liệu:
> - 497 frame, phân bố theo 4 camera
> - GT coverage có trung vị 27%, không có frame nào sạch hoàn toàn (đó là lý do Exp B dùng các frame ít bẩn nhất)
> - 399 mask khác nhau

## 4. Failure case

> Chọn F1 hoặc F2 (mục 4 của BENCHMARK_RESULTS) và viết lại bằng lời của bạn. Thêm giới hạn của dữ liệu: frame không độc lập, chỉ có cảnh ban ngày, ảnh resize về 512×512.

## 5. Engineering decision

> 2–3 câu về quyết định của nhóm, kèm một đề xuất từ góc dữ liệu. Ví dụ: chia train/test theo chuỗi (sequence) thay vì theo frame; bổ sung dữ liệu đêm và mưa trước khi bật v2.

## Nguồn, version, lệnh chạy

> Ghi: link repo, commit, link Google Drive của dataset và model, các lệnh ở mục 7 của BENCHMARK_RESULTS. Paper nào chưa đọc thì không ghi như nguồn đã dùng.
