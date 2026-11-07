# WORKFLOW.md – KAGGLE PLAYGROUND SERIES S6E11: PREDICTING CREDIT CARD FRAUD

Tài liệu này là **Single Source of Truth** duy nhất theo dõi tiến độ, kiến trúc, các quyết định kỹ thuật và kết quả thực nghiệm của toàn bộ dự án qua từng phase.

---

## 1. Project Overview

- **Competition Name:** Kaggle Playground Series Season 6 Episode 11 – Predicting Credit Card Fraud
- **Competition URL:** https://www.kaggle.com/competitions/playground-series-s6e11
- **GitHub Repository:** https://github.com/Nguyen-XuanTung/Predicting-Credit-Card-Fraud-Kaggle.git
- **Evaluation Metric:** ROC AUC (`roc_auc_score` trên xác suất dự đoán `is_fraud`)
- **Main Objective:** Tối đa hóa hiệu năng tổng quát hóa (generalization performance) trên Public và Private Leaderboard thông qua kiểm soát chặt chẽ validation scheme, khai thác đặc trưng dữ liệu tổng hợp (synthetic artifacts) và kết hợp mô hình (ensembling).
- **Current Local Data Files:**
  - `train.csv` (699,960 rows, 26 columns, ~94.0 MB)
  - `test.csv` (299,984 rows, 25 columns, ~39.2 MB)
  - `sample_submission.csv` (299,984 rows, 2 columns, ~8.0 MB)

---

## 2. Global Development Rules (Part A Compliance)

1. **Notebook-first development:** Mọi mã nguồn Python bắt buộc phải nằm trong các cell của Jupyter Notebook (`.ipynb`). Nghiêm cấm tạo script `.py` phụ trợ, helper modules, hoặc các script tiền xử lý tách rời.
2. **Minimal project files:** Duy trì tối giản số lượng file. Chỉ giữ lại `README.md`, `WORKFLOW.md`, `.gitignore`, notebook chính (`01_eda_baseline.ipynb`) và các báo cáo Markdown theo từng phase trong thư mục `reports/`. Không tạo nhật ký làm việc thừa.
3. **Phase-based execution:** Triển khai nghiêm ngặt theo thứ tự từng phase trong Roadmap. Không nhảy cóc hoặc thực hiện nhiều phase cùng lúc.
4. **Preserve existing work:** Kiểm tra kỹ lưỡng hiện trạng workspace trước khi sửa đổi. Không ghi đè hoặc xóa bỏ các thực nghiệm có giá trị. Tuyệt đối không can thiệp vào các tệp dữ liệu gốc.
5. **Scientific integrity:** Phân biệt rõ ràng giữa sự thật đã kiểm chứng (confirmed facts), quan sát thực nghiệm (observations) và giả thuyết (hypotheses). Không bịa đặt chỉ số. Ngôn ngữ giải thích trong notebook và báo cáo bằng tiếng Việt; tên biến và thuật ngữ kỹ thuật bằng tiếng Anh.

---

## 3. Project Roadmap

| Phase | Name | Status | Main Deliverable |
|---|---|---|---|
| 00 | Project Setup & Workflow | Completed | WORKFLOW.md, README.md, .gitignore, Git remote setup |
| 01 | Data Understanding & EDA | Completed | `01_eda_baseline.ipynb`, `reports/01_data_understanding.md` |
| 02 | Validation & Baseline | Not Started | Standardized 5-fold Stratified CV scheme & validated baseline models |
| 03 | Feature Engineering | Not Started | Feature engineering experiment suite (digit artifacts, risk aggregations) |
| 04 | Model Optimization | Not Started | Hyperparameter tuning for LightGBM, XGBoost, CatBoost |
| 05 | OOF Stacking & Ensemble | Not Started | Validated blending/stacking pipeline & OOF calibration |
| 06 | Final Submission | Not Started | Final submission selection, verification & pipeline cleanup |

*Allowed statuses: Not Started | In Progress | Completed | Needs Review | Blocked*

---

## 4. Current Project State

- **Active Notebooks:**
  - `01_eda_baseline.ipynb`: Kiểm toán toàn diện chất lượng dữ liệu, phân tích đơn biến/đa biến, kiểm tra độ chính xác chữ số thập phân (synthetic artifacts), phân tích rò rỉ (leakage), kiểm chứng Adversarial Validation và chạy thử nghiệm baseline sơ bộ.
- **Completed Experiments:**
  - *Data Integrity Audit:* 699,960 dòng train, 299,984 dòng test. Tỷ lệ mất cân bằng target `is_fraud`: 1.67% (11,689 ca gian lận / 688,271 ca hợp lệ). Hoàn toàn không có giá trị khuyết thiếu (0 missing values), không có bản ghi trùng lặp (0 duplicate rows).
  - *Synthetic Artifact Verification:* Các biến số thực có độ làm tròn cố định (1 hoặc 2 chữ số thập phân). `velocity_score` chỉ có 666 giá trị duy nhất (bước nhảy 0.1), `merchant_risk_score` có 950 giá trị duy nhất (bước nhảy 0.1, ngoại trừ 1 ngoại lệ duy nhất tại dòng 662,731 là 19.12).
  - *Train-Test Distribution & Covariate Shift:* Kiểm định KS cho thấy các phân phối đồng nhất (KS stat < 0.0038). Adversarial Validation đạt ROC AUC = 0.5025, khẳng định tập Train và Test được chia I.I.D ngẫu nhiên 70/30, không có rò rỉ thời gian hay domain shift.
  - *Signal Strength Check:* Các biến có tương quan mạnh nhất đơn lẻ: `merchant_risk_score` (ROC AUC 0.7489), `velocity_score` (ROC AUC 0.7108), `cvv_retry_count` (ROC AUC 0.7081), `is_new_merchant` (ROC AUC 0.6422), `customer_age` (ROC AUC 0.6247).
  - *Initial Raw LightGBM Baseline:* Mô hình LightGBM thô (chưa tạo đặc trưng) đạt ROC AUC Fold 1 = 0.94304.
- **Unresolved Issues / Hypotheses for Next Phases:**
  - Khai thác đặc trưng chữ số thập phân (decimal digit features) và mã hóa tần suất giá trị rời rạc (per-value target encoding) để tăng độ phân giải cho cây quyết định.
  - Xây dựng chiến lược phân tầng bảo toàn tỷ lệ gian lận (5-Fold StratifiedKFold) thống nhất cho Phase 02.

---

## 5. Important Decisions

1. **Validation Strategy Selection (StratifiedKFold):** Dữ liệu không có cấu trúc chuỗi thời gian (ID ngẫu nhiên, không tương quan với target: ROC AUC = 0.4985). Do đó, Stratified 5-Fold CV cố định seed là giải pháp chuẩn xác nhất để phản ánh chính xác Public và Private Leaderboard.
2. **Notebook-First Implementation:** Đóng gói toàn bộ logic trong `01_eda_baseline.ipynb` theo quy tắc dự án, tránh phát sinh file script phân mảnh.
3. **No External Data Injection at EDA Stage:** Không vội vàng tải và gộp dữ liệu gốc/bên ngoài trước khi hiểu rõ cấu trúc dữ liệu tổng hợp của S6E11.

---

## 6. Next Action

**Phase 02: Validation & Baseline Pipeline**
- Thiết lập khung kiểm chứng chéo 5-Fold Stratified K-Fold cố định (reproducible seed).
- Xây dựng và đánh giá các mô hình cơ sở chuẩn (Dummy/Logistic Regression, LightGBM Baseline, XGBoost Baseline, CatBoost Baseline) trên cùng một split.
- Thiết lập bảng theo dõi OOF ROC-AUC chuẩn xác để làm mốc so sánh (benchmark) cho Phase 03.
