# Hướng Dẫn Sử Dụng Google Colab Để Tạo Vector Database Siêu Tốc (GPU Accelerated)

Hiện tại, file dữ liệu thô **`extracted_chunks.jsonl`** đã được tạo thành công tại thư mục:
`d:\OneDrive\Desktop\Xay dung RAG noi bo\rag_system\extracted_chunks.jsonl`

Để tiến hành tạo cơ sở dữ liệu Vector Qdrant bằng GPU miễn phí trên Google Colab, anh/chị hãy làm theo các bước đơn giản sau:

---

## Bước 1: Mở Google Colab & Cấu hình GPU

1. Truy cập đường dẫn: **[Google Colab](https://colab.research.google.com/)** và chọn **New Notebook** (Sổ tay mới).
2. Đổi tên notebook (ví dụ: `Kohler_RAG_Indexer.ipynb`).
3. Kích hoạt card đồ họa GPU:
   * Trên thanh menu chọn: **Runtime** -> **Change runtime type** (Thay đổi loại thời gian chạy).
   * Tại mục **Hardware accelerator** (Bộ tăng tốc phần cứng), chọn **T4 GPU** (hoặc GPU bất kỳ có sẵn).
   * Nhấn **Save** (Lưu).

---

## Bước 2: Upload file dữ liệu thô lên Colab

1. Nhấp vào biểu tượng **Thư mục** ở thanh công cụ bên trái giao diện Google Colab.
2. Kéo thả file **`extracted_chunks.jsonl`** (dung lượng có thể thay đổi tùy theo số lượng tài liệu đã extract) từ máy tính vào danh sách file của Colab.
3. Đợi khoảng 1 - 2 phút để file upload hoàn tất (sẽ có vòng tròn tiến độ nhỏ ở góc dưới bên trái).

---

## Bước 3: Chạy code tạo Vector Database

Nhấp vào ô nhập code đầu tiên trên Colab, copy đoạn mã dưới đây dán vào và nhấn nút **Run** (hoặc phím tắt `Ctrl + Enter`):

```python
# 1. Cài đặt các thư viện cần thiết
!pip install sentence-transformers qdrant-client

# 2. Khởi chạy script build database Qdrant bằng GPU
import json
import time
import zipfile
import os
import gc
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from sentence_transformers import SentenceTransformer

# Cấu hình đường dẫn trên Colab
jsonl_path = "extracted_chunks.jsonl"  # File JSONL chứa dữ liệu chunks thô đã extract từ máy local
db_dir = "qdrant_db"                   # Thư mục lưu trữ database Qdrant sau khi tạo xong

print("=== GOOGLE COLAB RAG EMBEDDING GENERATOR (v2 – JSONL) ===")
if not os.path.exists(jsonl_path):
    raise FileNotFoundError(f"Không tìm thấy file '{jsonl_path}'. Hãy chắc chắn bạn đã upload file này lên Colab.")

# Đọc dữ liệu thô từ JSONL (mỗi dòng 1 JSON object)
print("Đang đọc dữ liệu chunks thô từ JSONL...")
chunks = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
total_chunks = len(chunks)
print(f"Đã nạp {total_chunks:,} chunks.")

# Khởi tạo Qdrant Client (chế độ local file trên Colab)
if os.path.exists(db_dir):
    import shutil
    shutil.rmtree(db_dir)
    print("Đã xóa thư mục qdrant_db cũ trên Colab.")

client = QdrantClient(path=db_dir)
collection_name = "kohler_rag"

print(f"Khởi tạo collection '{collection_name}' với vector BGE-M3 (1024 dimensions)...")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

# Tạo payload indexes để tăng tốc lọc khi truy vấn
print("Tạo payload indexes cho year, brand, doc_type...")
client.create_payload_index(collection_name, 'year', PayloadSchemaType.INTEGER)
client.create_payload_index(collection_name, 'brand', PayloadSchemaType.KEYWORD)
client.create_payload_index(collection_name, 'doc_type', PayloadSchemaType.KEYWORD)

# Tải mô hình BGE-M3 lên GPU
print("Đang tải mô hình BAAI/bge-m3 lên GPU (CUDA)...")
# device='cuda' giúp tận dụng sức mạnh xử lý song song của card đồ họa trên Colab
model = SentenceTransformer('BAAI/bge-m3', device='cuda')
print("Nạp mô hình thành công!")

# Tạo vector embedding và đẩy dữ liệu vào Qdrant theo lô (Batch)
batch_size = 256  # Lô lớn để tận dụng tối đa GPU T4
t_start = time.time()

print("\nBắt đầu tạo embeddings và đẩy dữ liệu vào Qdrant...")
for i in range(0, total_chunks, batch_size):
    batch = chunks[i:i+batch_size]
    texts = [item["content"] for item in batch]

    # Giải phóng bộ nhớ đệm CUDA để tránh OOM
    gc.collect()
    torch.cuda.empty_cache()

    # Tạo vector embedding bằng GPU với batch_size nội bộ nhỏ (32) để tránh tràn bộ nhớ
    with torch.no_grad():
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    points_to_upsert = []
    for j, item in enumerate(batch):
        point_id = i + j
        vector = embeddings[j].tolist()

        # Tạo cấu trúc điểm lưu vào Qdrant
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "content": item["content"],
                "source_file": item["source_file"],
                "location": item["location"],
                "year": int(item["year"]),
                "brand": item["brand"],
                "doc_type": item["doc_type"]
            }
        )
        points_to_upsert.append(point)

    # Đẩy lô dữ liệu lên cơ sở dữ liệu
    client.upsert(
        collection_name=collection_name,
        points=points_to_upsert
    )

    # In tiến độ sau mỗi 1024 chunks
    if (i + batch_size) % 1024 == 0 or (i + batch_size) >= total_chunks:
        pct = min(100.0, (i + batch_size) / total_chunks * 100.0)
        elapsed = time.time() - t_start
        chunks_done = min(total_chunks, i + batch_size)
        speed = chunks_done / elapsed
        eta = (total_chunks - chunks_done) / speed if chunks_done < total_chunks else 0
        print(f"Tiến độ: {chunks_done:,}/{total_chunks:,} ({pct:.1f}%) | "
              f"Thời gian đã chạy: {elapsed:.1f}s | "
              f"Tốc độ: {speed:.1f} chunks/s | "
              f"Còn lại dự kiến (ETA): {eta:.1f}s")

print(f"\n✅ Hoàn thành tạo database Qdrant trong {time.time() - t_start:.2f} giây!")

# Nén thư mục qdrant_db thành file zip để tải về máy local
print("Đang nén thư mục qdrant_db thành file zip...")
zip_path = "qdrant_db.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(db_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Giữ nguyên cấu trúc thư mục tương đối bên trong file zip
            zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(db_dir)))

print(f"✅ Nén thành công file: {zip_path}")
print("Bây giờ bạn hãy nhấp vào biểu tượng thư mục bên trái của Google Colab, tìm file 'qdrant_db.zip' và click chuột phải chọn Download để tải về máy tính.")
```

---

## Bước 4: Tải cơ sở dữ liệu về máy local

1. Khi code chạy xong (khoảng 8-10 phút), click vào nút **Refresh** ở thanh thư mục bên trái của Google Colab.
2. Tìm file **`qdrant_db.zip`**.
3. Click chuột phải vào file và chọn **Download** (Tải xuống) về máy tính.

---

## Bước 5: Giải nén vào dự án local

1. Trên máy tính của anh/chị, tạo thư mục mới tên là `qdrant_db` tại đường dẫn:
   `d:\OneDrive\Desktop\Xay dung RAG noi bo\rag_system\qdrant_db`
   *(Nếu thư mục `qdrant_db` đã tồn tại, hãy xóa toàn bộ file bên trong nó để tránh xung đột).*
2. Giải nén file **`qdrant_db.zip`** trực tiếp vào thư mục `qdrant_db` vừa tạo. Cấu trúc thư mục sau khi giải nén sẽ trông như thế này:
   ```text
   rag_system/
   ├── qdrant_db/
   │   ├── collection/
   │   │   └── kohler_rag/
   │   │       ├── storage.py
   │   │       └── ...
   │   └── meta.json
   ├── server.py
   └── index.html
   ```

---

## Bước 6: Khởi chạy máy chủ tìm kiếm RAG

1. Tắt máy chủ server cũ đang chạy (nếu có).
2. Mở cmd hoặc Powershell tại thư mục `d:\OneDrive\Desktop\Xay dung RAG noi bo\rag_system\` và chạy lệnh:
   ```bash
   python server.py
   ```
3. Mở trình duyệt và truy cập **`http://127.0.0.1:8000`** để sử dụng giao diện tìm kiếm mượt mà với đầy đủ bộ lọc!
