# BÁO CÁO NGHIÊN CỨU PHASE 01: DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS (EDA)

**Dự án:** Kaggle Playground Series Season 6 Episode 11 – Predicting Credit Card Fraud  
**Người thực hiện:** Senior Kaggle Data Scientist & ML Researcher  
**Ngày hoàn thành:** 07/11/2026  
**Thước đo Đánh giá:** ROC AUC  
**Notebook Thực thi:** [`01_eda_baseline.ipynb`](file:///c:/Users/LaptopK1/Downloads/Code/Predicting%20Credit%20Card%20Fraud/01_eda_baseline.ipynb)

---

## 1. Research Objective (Mục tiêu Nghiên cứu)

Mục tiêu cốt lõi của Phase 01 là thiết lập sự hiểu biết sâu sắc, toàn diện và có căn cứ toán học về tập dữ liệu Kaggle Playground Series S6E11:
- Kiểm toán toàn diện chất lượng dữ liệu thực tế (integrity, schema, missing values, duplicates).
- Xác định hành vi và mức độ mất cân bằng của biến mục tiêu `is_fraud`.
- Đo lường sức mạnh dự đoán độc lập và mối quan hệ phi tuyến tính của từng đặc trưng định lượng, danh mục và boolean.
- Khảo sát các tương tác đặc trưng phức hợp và hiệu ứng cộng hưởng rủi ro (risk synergy).
- Phát hiện các dấu vết lượng tử hóa và đặc tính sinh dữ liệu nhân tạo (synthetic artifacts).
- Đánh giá độ dịch chuyển phân phối (covariate shift) qua KS-test và Adversarial Validation.
- Kiểm tra các nguy cơ rò rỉ dữ liệu (target leakage, ID ordering).
- Thiết lập mốc benchmark sơ bộ từ mô hình cơ sở để định hướng các pha tiếp theo.

---

## 2. Dataset Overview (Tổng quan Tập dữ liệu)

Dữ liệu bao gồm 3 tệp CSV chuẩn của Kaggle:
- **`train.csv`**: 699,960 dòng, 26 cột (bao gồm `id` và nhãn mục tiêu `is_fraud`), dung lượng bộ nhớ xấp xỉ 94.0 MB.
- **`test.csv`**: 299,984 dòng, 25 cột (bao gồm `id` và 24 đặc trưng dự đoán), dung lượng bộ nhớ xấp xỉ 39.2 MB.
- **`sample_submission.csv`**: 299,984 dòng, 2 cột (`id`, `is_fraud`), với xác suất mặc định điền sẵn là 0.0167.

### Danh mục các đặc trưng:
- **Đặc trưng Định danh (Identifier):** `id` (nguyên tăng dần từ 0 đến 699,959 trong train; từ 699,960 đến 999,943 trong test).
- **Đặc trưng Định lượng Liên tục / Rời rạc (13 biến):** `amount_usd`, `hours_since_last_txn`, `txn_count_last_24h`, `distance_from_home_km`, `card_age_months`, `customer_age`, `account_balance_usd`, `cvv_retry_count`, `velocity_score`, `time_of_day_hour`, `day_of_week`, `merchant_risk_score`, `prior_disputes`.
- **Đặc trưng Phân loại Chuỗi (5 biến):** `merchant_category` (12 danh mục), `card_type` (5 danh mục), `auth_method` (5 danh mục), `channel` (5 danh mục), `device_type` (8 danh mục).
- **Đặc trưng Cờ Cảnh báo Rủi ro Boolean (6 biến):** `is_foreign_transaction`, `is_new_merchant`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch`, `is_ai_generated_scam_attempt`.
- **Biến Mục tiêu:** `is_fraud` (kiểu số thực: 0.0 hoặc 1.0).

---

## 3. Data Quality Findings (Chất lượng Dữ liệu Thực tế)

Tất cả các con số sau đây đã được kiểm chứng trực tiếp bằng mã thực thi trong notebook:
1. **Giá trị khuyết thiếu (Missing Values):** Cả hai tập Train và Test đều có **0 giá trị khuyết thiếu** trên toàn bộ 26 cột (100% complete data).
2. **Bản ghi trùng lặp (Duplicate Records):**
   - Không có bất kỳ dòng dữ liệu đặc trưng trùng lặp nội bộ nào trong Train (0 duplicates).
   - Không có dòng trùng lặp nội bộ nào trong Test (0 duplicates).
   - Không có bất kỳ dòng đặc trưng nào trùng lặp chéo giữa Train và Test (0 exact feature overlaps). Mọi bản ghi trong tập dữ liệu là độc nhất.
3. **Đồng bộ ID (ID Alignment):** Cột `id` của `test.csv` và `sample_submission.csv` khớp nhau 100% về cả số lượng, giá trị thứ tự và kiểu dữ liệu.

---

## 4. Target Distribution (Phân tích Biến Mục tiêu)

Thống kê thực tế của `is_fraud` trong `train.csv`:
- **Giao dịch hợp lệ (Lớp 0):** 688,271 giao dịch (**98.330%**)
- **Giao dịch gian lận (Lớp 1):** 11,689 giao dịch (**1.670%**)
- **Tỷ lệ mất cân bằng (Imbalance Ratio):** xấp xỉ **58.88 : 1** (cứ ~59 giao dịch thì có 1 giao dịch gian lận).
- **Prior Base Rate:** 0.0166995 (trùng khớp chính xác với giá trị mặc định trong `sample_submission.csv` là 0.0167).

> [!IMPORTANT]
> Do sự mất cân bằng lớp ở mức cao (1.67%), chiến lược xác thực bắt buộc phải là **Stratified K-Fold** để bảo toàn tỷ lệ gian lận đồng đều trên từng fold, tránh tình trạng variance lớn giữa các lần chia fold.

---

## 5. Numeric Feature Findings (Đặc trưng Định lượng)

Bảng phân tích thống kê mô tả và sức mạnh phân tách đơn biến (Single-feature ROC-AUC):

| Feature | Min | Median | Mean | Max | Skew | Nunique Train | Nunique Test | Overlap (%) | Eff. ROC-AUC | Chiều hướng |
|---|---|---|---|---|---|---|---|---|---|---|
| `merchant_risk_score` | 0.00 | 34.20 | 35.22 | 100.00 | 0.62 | 950 | 917 | 99.02% | **0.7489** | (+) Tăng theo rủi ro |
| `velocity_score` | 0.00 | 18.60 | 18.93 | 72.90 | 0.26 | 666 | 644 | 98.76% | **0.7108** | (+) Tăng theo nhịp độ |
| `cvv_retry_count` | 0 | 0 | 0.085 | 3 | 3.32 | 4 | 4 | 100.00% | **0.7081** | (+) Tăng theo số lần thử |
| `customer_age` | 18 | 52 | 51.43 | 81 | -0.14 | 64 | 64 | 100.00% | **0.6247** | (+) Tăng theo độ tuổi |
| `txn_count_last_24h` | 0 | 3 | 2.98 | 11 | 0.44 | 12 | 11 | 100.00% | **0.6203** | (+) Tăng theo tần suất |
| `hours_since_last_txn` | 0.01 | 4.73 | 6.63 | 51.98 | 1.82 | 3,633 | 3,331 | 96.34% | **0.5682** | (-) Giảm theo thời gian |
| `amount_usd` | 1.00 | 27.25 | 79.89 | 6,872.69 | 7.00 | 13,387 | 10,510 | 86.56% | **0.5448** | (+) Lệch phải mạnh |
| `prior_disputes` | 0 | 0 | 0.15 | 4 | 2.38 | 5 | 5 | 100.00% | **0.5322** | (+) Rời rạc |
| `day_of_week` | 0 | 3 | 2.98 | 6 | 0.02 | 7 | 7 | 100.00% | **0.5081** | Chu kỳ tuần |
| `account_balance_usd` | 58.25 | 1,382.74 | 2,065.00 | 99,701.39 | 14.44 | 43,317 | 31,803 | 81.74% | **0.5063** | (+) Lệch phải cực mạnh |
| `distance_from_home_km`| 0.00 | 10.22 | 13.59 | 57.93 | 1.06 | 4,365 | 4,108 | 97.57% | **0.5039** | Tín hiệu đơn lẻ yếu |
| `card_age_months` | 25 | 47 | 46.86 | 73 | 0.13 | 49 | 49 | 97.96% | **0.5019** | Phân phối chuẩn |
| `time_of_day_hour` | 0 | 12 | 11.76 | 23 | -0.06 | 24 | 24 | 100.00% | **0.5002** | Đồng đều qua các giờ |

### Phát hiện quan trọng từ biến số:
1. **Hành vi thử lại mã CVV (`cvv_retry_count`):**
   - 0 lần thử: Tỷ lệ gian lận 0.93% (642,088 giao dịch)
   - 1 lần thử: Tỷ lệ gian lận **8.75%** (55,942 giao dịch) -> gấp 9.4 lần
   - 2 lần thử: Tỷ lệ gian lận **41.73%** (1,905 giao dịch) -> gấp 44.7 lần
   - 3 lần thử: Tỷ lệ gian lận **36.00%** (25 giao dịch)
2. **Mức độ tổn thương theo Độ tuổi (`customer_age`):**
   - Tỷ lệ gian lận tăng đơn điệu gần như hoàn hảo theo độ tuổi khách hàng:
   - Tuổi 18–25: ~0.24% – 0.51%
   - Tuổi 45–55: ~1.68% – 1.73%
   - Tuổi 70–80: **2.59% – 2.89%** (khách hàng cao tuổi có nguy cơ bị gian lận cao gấp hơn 10 lần so với nhóm tuổi trẻ).
3. **Cụm giao dịch dồn dập (Burst Transactions):**
   - Khi khoảng thời gian cách giao dịch trước `< 0.85 giờ`, tỷ lệ gian lận đạt **2.98%**, trong khi ở khoảng thời gian dài hơn tỷ lệ này giảm xuống 1.34%.
   - Số lượng giao dịch trong 24 giờ (`txn_count_last_24h`) tăng từ 0 giao dịch (0.23% gian lận) lên 10 giao dịch (12.31% gian lận) và 11 giao dịch (22.22% gian lận).

---

## 6. Categorical Feature Findings (Đặc trưng Phân loại)

### 6.1. Phương thức Xác thực (`auth_method`):
- **`No Authentication`:** Tỷ lệ gian lận lên tới **6.44%** (4,322 ca gian lận trên 67,060 giao dịch, chiếm 37% tổng số ca gian lận trong tập train dù chỉ chiếm 9.6% số giao dịch).
- `3D Secure`: 1.47% gian lận (3,098 ca).
- `OTP`: 1.41% gian lận (2,464 ca).
- `PIN`: 0.87% gian lận (1,337 ca).
- `Biometric`: **0.50% gian lận** (468 ca) – Phương thức an toàn nhất, tỷ lệ gian lận thấp hơn 13 lần so với không xác thực.

### 6.2. Ngành hàng Người bán (`merchant_category`):
Các danh mục có rủi ro gian lận vượt trội:
- **`Gift Cards`:** Tỷ lệ gian lận **7.30%** (690 / 9,456)
- **`Crypto Exchange`:** Tỷ lệ gian lận **6.53%** (345 / 5,285)
- **`Gaming`:** Tỷ lệ gian lận **5.51%** (1,311 / 23,784)
- **`Electronics`:** Tỷ lệ gian lận **3.08%** (1,501 / 48,693)
- `Streaming`: 2.25% (1,061 / 47,196)
- Nhóm rủi ro thấp (1.11% – 1.86%): `Travel`, `Utilities`, `Healthcare`, `Fuel`, `Restaurants`, `Groceries`, `Online Retail`.

### 6.3. Các biến danh mục có phân phối rủi ro đồng đều:
- `card_type`: Dao động từ 1.46% (`Discover`) đến 1.90% (`Amex`), Visa (1.70%), Mastercard (1.60%).
- `channel`: Dao động từ 1.55% (`Contactless`) đến 1.74% (`Online`).
- `device_type`: Dao động từ 1.61% (`Android Phone`) đến 1.73% (`POS Terminal`).

---

## 7. Boolean Risk Flag Findings (Các Cờ Cảnh báo Rủi ro)

Cả 6 cờ boolean đều cho thấy sự phân kỳ tỷ lệ gian lận rõ rệt khi bật `True`:

| Boolean Flag | Tần suất True (Train) | Tần suất True (Test) | Gian lận khi False | Gian lận khi True | Hệ số Gia tăng (Multiplier) |
|---|---|---|---|---|---|
| `ip_country_mismatch` | 1.62% | 1.60% | 1.43% | **16.50%** | **11.5x** |
| `billing_shipping_mismatch` | 1.29% | 1.31% | 1.52% | **12.99%** | **8.5x** |
| `is_foreign_transaction` | 1.73% | 1.74% | 1.48% | **12.57%** | **8.5x** |
| `is_ai_generated_scam_attempt`| 0.33% | 0.34% | 1.63% | **15.22%** | **9.3x** |
| `used_vpn` | 3.38% | 3.30% | 1.50% | **6.55%** | **4.4x** |
| `is_new_merchant` | 15.25% | 15.28% | 1.12% | **4.73%** | **4.2x** |

### Hiệu ứng Hiệp đồng Cờ Rủi ro (`risk_flags_sum`):
Khi tính tổng số lượng cờ bật đồng thời trên cùng một giao dịch:
- **0 cờ bật (549,155 giao dịch):** Tỷ lệ gian lận = **0.70%**
- **1 cờ bật (137,966 giao dịch):** Tỷ lệ gian lận = **3.61%** (tăng 5.1x)
- **2 cờ bật (11,455 giao dịch):** Tỷ lệ gian lận = **18.18%** (tăng 26x)
- **3 cờ bật (1,268 giao dịch):** Tỷ lệ gian lận = **51.89%** (hơn một nửa là gian lận)
- **4 cờ bật (112 giao dịch):** Tỷ lệ gian lận = **83.93%**
- **5 cờ bật (4 giao dịch):** Tỷ lệ gian lận = **100.00%** (4/4 ca là gian lận tuyệt đối)

> [!NOTE]
> Đây là một phát hiện bản lề cho Phase 03: biến tổng `risk_flags_sum` và tích hợp tổ hợp cờ (bitmask / combination encoding) sẽ cung cấp một tín hiệu phi tuyến tính cực mạnh cho các mô hình boosting.

---

## 8. Train-Test Comparison (So sánh Phân phối Train vs Test)

1. **Kiểm định Kolmogorov-Smirnov (KS-test):**
   - Giá trị KS statistic trên toàn bộ 13 biến định lượng đều rất nhỏ (tất cả đều `< 0.0039`, ngoại trừ `day_of_week` = 0.00383).
   - Tần suất phần trăm của các danh mục chuỗi và cờ boolean hoàn toàn trùng khớp giữa Train và Test (độ lệch `< 0.08%`).
2. **Adversarial Validation:**
   - Huấn luyện mô hình LightGBM để phân biệt nguồn gốc dòng dữ liệu Train (0) hay Test (1) trên 100,000 mẫu ngẫu nhiên.
   - Kết quả: **ROC-AUC = 0.50255**.
   - **Kết luận:** Mô hình không thể phân biệt dữ liệu Train và Test (tương đương đoán ngẫu nhiên). Tập Train và Test tuân theo phân phối độc lập cùng phân phối (I.I.D) hoàn hảo, không có Covariate Shift, không có phân tách theo thời gian.

---

## 9. Synthetic Data Observations (Đặc tính Dữ liệu Tổng hợp & Lượng tử hóa)

Kiểm tra số học chi tiết về độ chính xác chữ số:
1. **Lượng tử hóa tuyệt đối (Exact Quantization):**
   - `velocity_score`: 100% các giá trị có đúng **1 chữ số thập phân** (bước nhảy chính xác 0.1). Toàn bộ 700k dòng Train chỉ có 666 giá trị duy nhất.
   - `merchant_risk_score`: 99.9999% các giá trị có đúng **1 chữ số thập phân** (chỉ có duy nhất 1 dòng ngoại lệ tại index 662,731 trong Train mang giá trị 19.12).
   - `hours_since_last_txn`, `distance_from_home_km`, `amount_usd`, `account_balance_usd`: 100% các giá trị tuân thủ quy tắc làm tròn **2 chữ số thập phân**.
2. **Dấu vết Chữ số Thập phân Đầu tiên (`dec1`):**
   - Khảo sát `dec1 = int(floor(val * 10)) % 10` cho thấy tỷ lệ gian lận có sự chênh lệch có hệ thống theo chữ số thập phân:
     - Với `velocity_score`: Chữ số 0 có tỷ lệ gian lận 1.10%, trong khi chữ số 4 và 9 có tỷ lệ gian lận lần lượt là 1.93% và 1.92%.
     - Với `hours_since_last_txn`: Chữ số 0 có tỷ lệ gian lận 2.20%, trong khi chữ số 6 chỉ có 1.27%.

*Phân loại:*
- **Confirmed Fact:** Dữ liệu bị lượng tử hóa cố định về số chữ số thập phân.
- **Hypothesis:** Trích xuất chữ số thập phân và mã hóa tần suất giá trị (frequency encoding / per-value target encoding) sẽ cung cấp thêm góc nhìn phân giải mịn cho cây quyết định.

---

## 10. Leakage Assessment (Đánh giá Nguy cơ Rò rỉ Dữ liệu)

1. **Kiểm tra Cột `id`:**
   - ROC-AUC của `id` đối với `is_fraud`: **0.49855** (hoàn toàn ngẫu nhiên).
   - Hệ số tương quan Pearson giữa `id` và tất cả các biến đặc trưng khác đều `< 0.0025`.
   - Tỷ lệ gian lận trên 10 phân vị đều nhau của `id` dao động ngẫu nhiên xung quanh 1.59% – 1.73%.
2. **Kiểm tra Target Leakage / Post-event Variables:**
   - Không có biến nào đạt ROC-AUC đơn lẻ bất thường (> 0.90). Biến mạnh nhất là `merchant_risk_score` (0.7489).
   - Không có dấu hiệu rò rỉ target gián tiếp. Cột `id` thuần túy là chỉ mục sắp xếp ngẫu nhiên.

---

## 11. Relevant Kaggle Approaches (Nghiên cứu Giải pháp Cộng đồng)

Dựa trên nghiên cứu các giải pháp công khai và thảo luận nổi bật tại Playground S6E11:
1. **Shallow Gradient Boosting Trees:** Do tập dữ liệu tổng hợp có lượng mẫu lớn (~700k) và các tương tác logic rõ ràng, các cây boosting nông (`max_depth`: 3–6, `num_leaves`: 15–63) giúp kiểm soát overfitting hiệu quả và đạt ROC-AUC cao trên cả CV lẫn Public LB.
2. **Decimal Digit & Quantized Feature Extraction:** Các thí sinh hàng đầu tách các chữ số thập phân (e.g., `amount_cents = round(amount * 100) % 100`, `velocity_dec = round(velocity * 10) % 10`) để giúp cây quyết định dễ dàng phân nhánh tại các ranh giới lượng tử hóa.
3. **Out-of-Fold Target Encoding (OOF TE):** Mã hóa tỷ lệ gian lận cho các biến danh mục có độ phân tách cao (`auth_method`, `merchant_category`) và các cụm kết hợp (ví dụ: `auth_method` + `is_new_merchant`) thông qua K-Fold out-of-fold nghiêm ngặt để tránh rò rỉ.
4. **Mô hình RealMLP / Tabular Neural Nets:** Thử nghiệm kết hợp LightGBM/CatBoost/XGBoost với mạng nơ-ron chuyên dụng cho tabular data (như RealMLP qua pytabkit) cho thấy tiềm năng đa dạng hóa (diversity) đáng kể trong bước ensembling cuối cùng.
5. **Cơ sở Benchmark ban đầu:** Mô hình LightGBM thô chưa qua kỹ thuật đặc trưng đạt ROC-AUC xấp xỉ **0.9430** (Fold 1). Các giải pháp hàng đầu có feature engineering đạt xấp xỉ **0.9509+**.

---

## 12. Main Conclusions (Kết luận Chính)

1. **Cấu trúc Dữ liệu:** 699,960 dòng train, 299,984 dòng test, 0 missing values, 0 bản ghi trùng lặp. Phân phối Train và Test đồng nhất 100% (I.I.D, Adversarial AUC = 0.5025).
2. **Tín hiệu Gian lận Cốt lõi:** Tín hiệu tập trung mạnh ở các chỉ số đánh giá rủi ro (`merchant_risk_score`, `velocity_score`), hành vi bất thường (`cvv_retry_count`), phương thức giao dịch không chứng thực (`auth_method = 'No Authentication'`), ngành hàng rủi ro (`Gift Cards`, `Crypto Exchange`), và các cờ cảnh báo bất thường (`ip_country_mismatch`, `billing_shipping_mismatch`).
3. **Cộng hưởng Rủi ro (Synergy):** Số lượng cờ rủi ro kích hoạt đồng thời là một siêu đặc trưng (super-feature), đẩy xác suất gian lận từ 0.70% lên 100%.
4. **Không có Rò rỉ:** ID hoàn toàn vô hại, không có target leakage.

---

## 13. Recommended Next Experiments (Đề xuất Thực nghiệm Tiếp theo)

### Dành cho Phase 02 (Validation & Baseline):
- Thiết lập quy trình 5-Fold Stratified K-Fold cố định seed (`random_state=42`) làm thước đo chuẩn duy nhất cho toàn bộ dự án.
- Huấn luyện và ghi nhận kết quả OOF ROC-AUC chuẩn xác của 3 họ mô hình GBDT cơ sở: **LightGBM**, **XGBoost**, **CatBoost** trên tập đặc trưng thô.

### Dành cho Phase 03 (Feature Engineering):
- Tạo đặc trưng tổng hợp cờ: `risk_flags_sum`, `has_any_mismatch` (`ip_country_mismatch | billing_shipping_mismatch`).
- Trích xuất đặc trưng thập phân: `velocity_dec`, `merchant_risk_dec`, `amount_cents`.
- Tạo đặc trưng tương tác tỷ lệ: `amount_to_balance_ratio = amount_usd / (account_balance_usd + 1)`, `risk_velocity_product = merchant_risk_score * velocity_score`.
- Áp dụng Out-of-Fold Target Encoding (kèm smoothing) cho `merchant_category`, `auth_method` và các cặp tương tác danh mục.

---

## 14. Sources and References (Nguồn Tham khảo)

1. Kaggle Competition: [Playground Series Season 6 Episode 11 – Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11)
2. Danil Malyshev: [Predicting Credit Card Fraud - Getting Started](https://www.kaggle.com/code/malyshevdanil/predicting-credit-card-fraud-getting-started)
3. Oguz Uzan: [Predicting Credit Card Fraud PS-S6E11 Baseline](https://www.kaggle.com/code/oguzuzan/predicting-credit-card-fraud-ps-s6e11)
4. Kaggle Discussions on S6E11: Shallow tree architectures, decimal digit extraction, and RealMLP ensembling strategies.
