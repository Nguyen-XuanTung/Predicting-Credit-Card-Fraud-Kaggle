# Kaggle Playground Series S6E11 – Predicting Credit Card Fraud

Dự án nghiên cứu và xây dựng giải pháp học máy cho cuộc thi **Kaggle Playground Series Season 6 Episode 11: Predicting Credit Card Fraud**.

## 1. Thông tin Cuộc thi
- **Cuộc thi:** [Kaggle Playground Series s6e11](https://www.kaggle.com/competitions/playground-series-s6e11)
- **Mục tiêu:** Dự đoán xác suất giao dịch gian lận thẻ tín dụng (`is_fraud`).
- **Thước đo Đánh giá (Metric):** ROC AUC (Area Under the ROC Curve).
- **Giai đoạn Hiện tại:** Hoàn thành Phase 00 (Project Setup) & Phase 01 (Data Understanding & EDA). Chuẩn bị cho Phase 02 (Validation & Baseline).

## 2. Tài liệu Dự án
- **Kế hoạch & Tiến độ Tổng thể (Single Source of Truth):** [WORKFLOW.md](file:///c:/Users/LaptopK1/Downloads/Code/Predicting%20Credit%20Card%20Fraud/WORKFLOW.md)
- **Báo cáo Nghiên cứu:**
  - [Phase 01 Report: Data Understanding & EDA](file:///c:/Users/LaptopK1/Downloads/Code/Predicting%20Credit%20Card%20Fraud/reports/01_data_understanding.md)

## 3. Cấu trúc Thư mục Chính
- `01_eda_baseline.ipynb`: Notebook Jupyter chính chứa toàn bộ mã nguồn kiểm toán dữ liệu, phân tích khám phá (EDA), kiểm tra rò rỉ và mô hình cơ sở.
- `reports/`: Thư mục lưu trữ báo cáo Markdown cho từng phase hoàn thành.
- `.gitignore`: Cấu hình loại trừ các file dữ liệu nặng (`train.csv`, `test.csv`, `sample_submission.csv`), checkpoints và môi trường ảo.
