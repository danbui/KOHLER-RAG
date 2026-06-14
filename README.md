# Hệ thống RAG Tìm Kiếm Nội Bộ (Rita Võ - Kohler & Karat)

Dự án này là hệ thống **RAG (Retrieval-Augmented Generation)** tìm kiếm nội bộ dành cho bảng báo giá và tài liệu kỹ thuật thiết bị vệ sinh **Kohler** và **Karat** của công ty **Rita Võ**. Hệ thống giúp nhân viên tra cứu nhanh giá cả, thông tin SKU, chương trình khuyến mãi, và các tài liệu liên quan với thời gian phản hồi nhanh và chính xác.

---

## ⚡ Các Tính Năng Nổi Bật
* **Tra cứu chính xác:** Kết hợp tìm kiếm vector dense (BGE-M3) trên Qdrant với khả năng tổng hợp câu trả lời thông minh của **Gemini API (Gemini 3.5 Flash / Gemini 1.5 Flash)**.
* **Mở rộng SKU thông minh:** Tự động chuẩn hóa và mở rộng SKU trong câu hỏi (ví dụ: `3722X-0` -> `3722X-0 X0`) giúp tăng độ phủ và độ chính xác của kết quả tìm kiếm.
* **Tối ưu hóa Latency vượt trội:**
  * Hỗ trợ bật/tắt **Rerank** linh hoạt thông qua biến môi trường để tiết kiệm từ 6 - 20 giây mỗi lượt truy vấn.
  * Hỗ trợ giới hạn **RAG_TOP_K** để giảm đến 50% context gửi đi, tăng tốc độ xử lý của Gemini thêm 20 - 40%.
* **Giao diện Web UI hiện đại:**
  * Trực quan, dễ sử dụng, thiết kế responsive trên mọi thiết bị di động.
  * Tích hợp **đồng hồ bấm giờ (live stopwatch)** chạy ngay khi gửi truy vấn và dừng khi nhận kết quả để đo kiểm latency.
  * Danh sách nguồn trích dẫn (**Sources Panel**) chi tiết, hỗ trợ click nhanh để sao chép đường dẫn file gốc.

---

## 📂 Cấu Trúc Thư Mục
```text
Xay dung RAG noi bo/
├── clean_data/               # Thư mục chứa các file Excel/PDF gốc (đã bỏ qua trong Git)
├── File_List.xlsx            # Danh sách quản lý file tài liệu (đã bỏ qua trong Git)
├── .gitignore                # Quản lý các file loại bỏ khi push lên Git
├── README.md                 # Tài liệu hướng dẫn sử dụng này
└── rag_system/               # Mã nguồn hệ thống RAG
    ├── server.py             # FastAPI backend server
    ├── search.py             # Pipeline tìm kiếm (Qdrant search + Rerank + SKU Expansion)
    ├── ai_models.py          # Quản lý kết nối HuggingFace Inference API (Embeddings & Reranker)
    ├── gemini_client.py      # Client kết nối Gemini API (Hỗ trợ retry & backoff)
    ├── config.py             # Quản lý cấu hình & biến môi trường
    ├── index.html            # Giao diện Web UI (HTML, CSS, JS)
    └── .env.example          # File mẫu cấu hình biến môi trường
```

---

## ⚙️ Cấu Hình Hệ Thống

Tạo file `.env` bên trong thư mục `rag_system/` (sử dụng mẫu từ `.env.example`) và điền các thông tin:

```env
# API Key của Gemini
GEMINI_API_KEY=AIzaSy...

# Cấu hình Qdrant Database (Local hoặc Cloud)
QDRANT_DB_PATH=C:/path/to/your/local/qdrant_db
QDRANT_URL=https://your-qdrant-cloud-url.io
QDRANT_API_KEY=your-qdrant-api-key

# HuggingFace Token (dùng cho BGE-M3 Embeddings & Reranker)
HF_TOKEN=hf_...

# Cấu hình Bật/Tắt Reranker (true: bật, false: tắt để giảm latency)
ENABLE_RERANKER=false
RERANK_TIMEOUT_SECONDS=10
QUERY_EMBEDDING_CACHE_SIZE=256

# Số lượng tài liệu gửi vào context cho Gemini (Mặc định là 3 để giảm 50% context)
RAG_TOP_K=3
MAX_CONTEXT_CHARS=6000
```

---

## 🚀 Hướng Dẫn Khởi Chạy

### 1. Cài đặt thư viện cần thiết
Mở terminal tại thư mục dự án và cài đặt các thư viện Python:
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Backend Server
Di chuyển vào thư mục `rag_system/` và chạy lệnh sau để khởi chạy API server:
```bash
cd rag_system
python server.py
```
Server sẽ chạy mặc định tại: **`http://127.0.0.1:8000`**

### 3. Trải nghiệm giao diện Web UI
Mở file `index.html` trong thư mục `rag_system/` bằng bất kỳ trình duyệt nào (Chrome, Edge, Firefox). Nhập câu hỏi và trải nghiệm tính năng tra cứu nội bộ siêu tốc.

---

## 🛠️ Quy Trình Tối Ưu Hóa Latency

Để đạt được tốc độ tra cứu **dưới 6 giây** (so với 30 - 50 giây ban đầu):
1. **Tắt Reranker:** Đặt `ENABLE_RERANKER=false` trong file `.env`. Việc này giúp loại bỏ bước gọi API Rerank tới HuggingFace, tiết kiệm **6s - 20s**. Hệ thống sẽ sử dụng trực tiếp điểm số độ tương đồng (score) từ Qdrant.
2. **Giới hạn số lượng tài liệu (`RAG_TOP_K=3`):** Giảm số lượng tài liệu đính kèm vào prompt gửi cho Gemini giúp giảm 50% kích thước context (chỉ còn khoảng ~1,600 tokens thay vì ~3,200 tokens), tiết kiệm thêm **2s - 5s** thời gian xử lý của Gemini mà vẫn đảm bảo độ chính xác cho các tra cứu sản phẩm cụ thể.
