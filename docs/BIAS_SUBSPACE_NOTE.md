# Nghiên cứu & Hướng dẫn Kỹ thuật: CLIP Bias PEFT, MIBS và MBBS

Tài liệu này tổng hợp toàn diện cơ sở lý thuyết, công thức toán học, giải thuật tối ưu hóa và hướng dẫn thực thi thực nghiệm cho phương pháp **Parameter-Efficient Fine-Tuning (PEFT)** trên CLIP ViT kết hợp với **Manipulation-Invariant / Manipulation-Balanced Bias Subspace (MIBS / MBBS)** trong bài toán phát hiện Deepfake.

---

## 1. Tổng quan & Động lực Nghiên cứu

### 1.1 Vấn đề của Fine-Tuning truyền thống
Khi áp dụng các mô hình thị giác nền tảng (Foundation Models) như **CLIP ViT-L/14** (303M tham số) vào bài toán phát hiện ảnh giả mạo:
- **Overfitting & Catastrophic Forgetting**: Fine-tune toàn bộ mô hình (100% tham số) trên một tập dữ liệu cụ thể (như FaceForensics++) khiến mô hình nhanh chóng học thuộc các artifacts đặc thù của bộ sinh, mất đi khả năng khái quát hóa (Generalization) sang các phương pháp giả mạo chưa từng thấy (như Celeb-DF-v2, DFDC).
- **Chi phí tính toán cao**: Cần lưu trữ và đồng bộ hàng trăm triệu gradient giữa các GPU.

### 1.2 Giải pháp: CLIP Bias PEFT + Subspace Regularization
1. **Bias Parameter Isolation**: Đóng băng 100% ma trận trọng số $W$, patch embeddings và scale LayerNorm $\gamma$. Chỉ cập nhật một phần cực nhỏ tham số bias $b$ trong backbone (từ **0.003% đến 0.09%** tham số) cùng phân loại nhị phân (Head).
2. **Manipulation Subspace Constraint**: Ép các cập nhật bias $\Delta b = b - b_0$ phải nằm gần một không gian con trực chuẩn chung $U \in \mathbb{R}^{P \times r}$ ($r \ll M$) được trích xuất từ các phương pháp giả mạo đã biết, ngăn chặn mô hình học các artifacts giả mạo ngẫu nhiên.

---

## 2. Cơ sở Toán học & Thuật toán

### 2.1 Trích xuất và Ánh xạ Tham số Bias ($P = 272,384$)
Trong backbone CLIP ViT-Large/14 (24 transformer blocks), tập hợp toàn bộ tham số bias (`all_bias`) gồm:
- Attention Query bias: $24 \times 1,024 = 24,576$
- Attention Key bias: $24 \times 1,024 = 24,576$
- Attention Value bias: $24 \times 1,024 = 24,576$
- Attention Output Projection bias: $24 \times 1,024 = 24,576$
- MLP fc1 bias: $24 \times 4,096 = 98,304$
- MLP fc2 bias: $24 \times 1,024 = 24,576$
- LayerNorm beta ($\beta$ trong 24 blocks + Pre/Post LN): $51,200$
- **Tổng số chiều bias $P = 272,384$** (chiếm 0.0905% tổng số tham số mô hình).

Mỗi tham số được chuẩn hóa thành một slice trên vector 1D contiguous:
$$\text{flatten}(b) \in \mathbb{R}^P \iff \text{restore}(v) \in \{\Theta_{bias}\}$$

### 2.2 Subspace Loss với độ phức tạp $O(Pr)$
Tại điểm bắt đầu huấn luyện, snapshot bất biến $b_0$ được tạo từ weights gốc của CLIP. Cập nhật bias hiện tại là:
$$\Delta b = b - b_0 \in \mathbb{R}^P$$

Chiếu $\Delta b$ lên không gian con trực chuẩn $U \in \mathbb{R}^{P \times r}$ ($U^T U = I_r$):
1. Tính tọa độ chiếu trong không gian con ($r$ chiều):
   $$\text{coeff} = U^T \Delta b \in \mathbb{R}^r$$
2. Tái tạo vector chiếu trong không gian tham số ($P$ chiều):
   $$\text{proj} = U \cdot \text{coeff} \in \mathbb{R}^P$$
3. Tính phần dư (thành phần nằm ngoài không gian con):
   $$\text{residual} = \Delta b - \text{proj} = (I - U U^T) \Delta b \in \mathbb{R}^P$$
4. Hàm mất mát chính quy hóa:
   $$\mathcal{L}_{subspace} = \frac{1}{P} \|\text{residual}\|_2^2 = \frac{1}{P} \sum_{i=1}^P (\Delta b_i - \text{proj}_i)^2$$

> **Đặc điểm tối ưu:** Thuật toán **không bao giờ khởi tạo ma trận $P \times P$** (sẽ tốn $272,384^2 \times 4 \approx 296\text{ GB}$ VRAM), mà luôn thực hiện qua 2 phép nhân ma trận - vector với chi phí chỉ $O(Pr)$ (dưới 1ms trên GPU).

Tổng hàm mục tiêu huấn luyện:
$$\mathcal{L}_{total} = \mathcal{L}_{CE}(y, \hat{y}) + \lambda \cdot \mathcal{L}_{subspace}$$

---

## 3. Ước lượng Không gian con: SVD thuần vs MBBS

### 3.1 Trích xuất Gradient phương pháp
Từ tập dữ liệu huấn luyện (FaceForensics++), hệ thống lấy mẫu $K$ balanced batches (sử dụng `pairDataset` ghép 1 Real : 1 Fake) cho từng phương pháp giả mạo $m \in \{\text{FF-DF, FF-F2F, FF-FS, FF-NT}\}$:
$$\tilde{g}_m = \frac{\frac{1}{K} \sum_{k=1}^K \hat{g}_{m,k}}{\|\frac{1}{K} \sum_{k=1}^K \hat{g}_{m,k}\|_2} \in \mathbb{R}^P$$

### 3.2 SVD thuần (`shared_svd`)
Ghép ma trận gradient $G = [\tilde{g}_1, \tilde{g}_2, \tilde{g}_3, \tilde{g}_4] \in \mathbb{R}^{P \times 4}$, thực hiện Thin SVD:
$$G = U \Sigma V^T$$
Lấy $r$ cột đầu tiên của $U$: $U_{SVD} = U[:, :r]$.

### 3.3 Hạn chế của SVD & Giải pháp MBBS (`balanced_subspace`)
- **Hạn chế của SVD**: SVD tối đa hóa tổng năng lượng captured $\sum_m \|U^T \tilde{g}_m\|^2$. Do đó, các phương pháp có gradient tương đồng hoặc chiếm ưu thế (như Deepfakes và FaceSwap) sẽ chi phối các hướng chính, khiến phương pháp khác (như Face2Face) bị bỏ rơi với năng lượng chiếu rất thấp.
- **Manipulation-Balanced Bias Subspace (MBBS)**: Khởi tạo từ $U_{SVD}$, sau đó tối ưu hóa xoay $U$ trên đa tạp Stiefel $\text{St}(r, P)$ bằng phép co QR (QR Retraction) để cân bằng độ bao phủ giữa tất cả các phương pháp.

Hàm mục tiêu cân bằng (`mean_variance`):
$$\max_{U \in \text{St}(r, P)} \left[ \text{mean}_m(R_m) - \beta \cdot \text{Var}_m(R_m) \right]$$
trong đó $R_m = \frac{\|U^T \tilde{g}_m\|^2}{\|\tilde{g}_m\|^2}$ là tỉ lệ năng lượng chiếu của phương pháp $m$.

---

## 4. Kết quả Thực nghiệm So sánh Trực tiếp

Kết quả chạy thực tế trên dữ liệu FaceForensics++ với $r = 2$:

### 4.1 Ma trận Tương đồng Cosine giữa các phương pháp
```text
Method         FF-DF    FF-F2F     FF-FS     FF-NT
--------------------------------------------------
FF-DF         1.0000    0.0712    0.1296    0.1953
FF-F2F        0.0712    1.0000    0.0177    0.0558
FF-FS         0.1296    0.0177    1.0000   -0.1280
FF-NT         0.1953    0.0558   -0.1280    1.0000
```

### 4.2 Bảng so sánh Năng lượng Chiếu $R_m$
```text
==============================================================
Phương pháp         SVD thuần (rank-2)     Balanced MBBS (rank-2)
--------------------------------------------------------------
FF-DF                    65.8 %                    55.3 %
FF-F2F                   17.6 % (Bị bỏ rơi)        51.0 % (Tăng gấp 3!)
FF-FS                    80.5 % (Chiếm ưu thế)     60.2 %
FF-NT                    70.4 %                    63.6 %
--------------------------------------------------------------
Trung bình (Mean)        58.5 %                    57.5 %
Độ phủ tối thiểu (Min)   17.6 %                    51.0 %
Độ lệch chuẩn (Std)      24.2 %                     4.8 % (Phân bố đều!)
==============================================================
```
**Nhận xét:** MBBS giúp kéo độ bao phủ của phương pháp yếu nhất (`FF-F2F`) từ **17.6% lên 51.0%**, đồng thời giảm độ phân tán (Std) từ **24.2% xuống chỉ còn 4.8%**, đảm bảo không gian con đại diện công bằng cho mọi dạng biến đổi giả mạo.

---

## 5. Hướng dẫn Thực thi với `make` & CLI

### 5.1 Chạy Ước lượng Subspace

#### Cách A: Chạy qua Makefile
```bash
# 1. Ước lượng SVD thuần từ dữ liệu:
make estimate-subspace RANK=2 BATCHES_METHOD=25 SUBSPACE_PATH=./bias_subspace_pair_rank2.pt

# 2. Ước lượng Balanced MBBS (tối ưu hóa nhanh từ gradient có sẵn):
make estimate-balanced-subspace \
  RANK=2 \
  SUBSPACE_PATH=./bias_subspace_pair_rank2.pt \
  BALANCED_OUTPUT=./bias_subspace_pair_balanced_rank2.pt
```

#### Cách B: Chạy trực tiếp qua Python CLI
```bash
# SVD thuần:
python3 training/estimate_bias_subspace.py \
  --detector_path ./training/config/detector/clip_bias.yaml \
  --tuning all_bias \
  --subspace_rank 2 \
  --subspace_method shared_svd \
  --batches_per_method 25 \
  --batch_size 8 \
  --dataset_type pair \
  --output ./bias_subspace_pair_rank2.pt

# Balanced MBBS:
python3 training/estimate_bias_subspace.py \
  --detector_path ./training/config/detector/clip_bias.yaml \
  --subspace_rank 2 \
  --subspace_method balanced_subspace \
  --balanced_objective mean_variance \
  --cached_gradients_artifact ./bias_subspace_pair_rank2.pt \
  --output ./bias_subspace_pair_balanced_rank2.pt
```

### 5.2 Huấn luyện với Subspace Regularization

```bash
# Sử dụng make (2 GPUs DDP):
make train-balanced-subspace RANK=2 LAMBDA=0.01

# Sử dụng maketrain.sh:
./maketrain.sh all_bias_subspace 2 \
  --subspace ./bias_subspace_pair_balanced_rank2.pt \
  --lambda 0.01

# Huấn luyện 1 GPU:
./maketrain.sh all_bias_subspace 1 \
  --subspace ./bias_subspace_pair_balanced_rank2.pt \
  --lambda 0.01
```

### 5.3 Bộ công cụ Kiểm thử & Đánh giá (Test Suites)

```bash
# Kiểm thử toàn bộ 11 tiêu chuẩn toán học của MIBS / MBBS:
make test-subspace
# hoặc:
python3 test_bias_subspace.py

# Kiểm thử cách ly gradient trên 19 chiến lược PEFT:
make verify
# hoặc:
python3 test_bias_tuning.py

# Thống kê phân bố nhãn & phương pháp huấn luyện:
make audit-distribution
# hoặc:
python3 training/audit_training_distribution.py
```
