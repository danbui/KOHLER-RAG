# -*- coding: utf-8 -*-
"""
migrate_to_cloud.py - Script di chuyển dữ liệu từ Qdrant Local sang Qdrant Cloud
=============================================================================
Hướng dẫn sử dụng:
1. Nhập QDRANT_URL và QDRANT_API_KEY của Qdrant Cloud vào file .env
2. Chạy lệnh: python migrate_to_cloud.py
"""

import os
import sys
import time
import logging

# Thiết lập UTF-8 stdout cho Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("migrate")

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Đã load cấu hình từ .env")
except ImportError:
    logger.warning("⚠️ Chưa cài python-dotenv, sử dụng os.getenv")

# Lấy cấu hình
local_db_path = os.getenv("QDRANT_DB_PATH", os.path.join(os.path.dirname(__file__), "qdrant_db"))
cloud_url = os.getenv("QDRANT_URL", "")
cloud_api_key = os.getenv("QDRANT_API_KEY", "")

if not cloud_url:
    logger.error("❌ Lỗi: Chưa cấu hình QDRANT_URL trong file .env!")
    logger.error("Vui lòng truy cập Qdrant Cloud, tạo cluster và copy URL vào file .env")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    logger.error("❌ Lỗi: Thiếu thư viện qdrant-client. Vui lòng cài đặt bằng: pip install qdrant-client")
    sys.exit(1)

def migrate():
    # 1. Kết nối local
    logger.info("📂 Kết nối tới Qdrant Local tại: %s", local_db_path)
    logger.info("⏳ Đang tải cơ sở dữ liệu local vào RAM (quá trình này có thể mất vài phút)...")
    t_start_local = time.time()
    local_client = QdrantClient(path=local_db_path)
    
    # Lấy thông tin collection local
    collection_name = "kohler_rag"
    try:
        local_info = local_client.get_collection(collection_name)
        total_points = local_info.points_count
        logger.info("✅ Đã tải local DB thành công trong %.2fs!", time.time() - t_start_local)
        logger.info("📊 Tìm thấy collection '%s' có %d điểm vector.", collection_name, total_points)
    except Exception as e:
        logger.error("❌ Không tìm thấy collection '%s' ở Local DB: %s", collection_name, e)
        return

    # 2. Kết nối cloud
    logger.info("🌐 Kết nối tới Qdrant Cloud: %s", cloud_url)
    cloud_client = QdrantClient(url=cloud_url, api_key=cloud_api_key if cloud_api_key else None)
    
    # 3. Tạo collection trên Cloud nếu chưa tồn tại
    try:
        cloud_client.get_collection(collection_name)
        logger.info("ℹ️ Collection '%s' đã tồn tại trên Qdrant Cloud.", collection_name)
    except Exception:
        logger.info("🆕 Collection '%s' chưa có trên Cloud. Đang tiến hành tạo mới...", collection_name)
        # Lấy cấu hình vector từ local
        local_vector_config = local_info.config.params.vectors
        
        # Mặc định cấu hình nếu local config không đọc được trực tiếp
        vector_size = 1024  # BGE-M3 mặc định là 1024
        distance = Distance.COSINE
        
        if hasattr(local_vector_config, 'size'):
            vector_size = local_vector_config.size
            distance = local_vector_config.distance
        elif isinstance(local_vector_config, dict) and 'size' in local_vector_config:
            vector_size = local_vector_config['size']
            distance = local_vector_config.get('distance', Distance.COSINE)
        
        try:
            cloud_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            logger.info("✅ Đã tạo thành công collection '%s' với vector_size=%d", collection_name, vector_size)
        except Exception as e:
            logger.error("❌ Lỗi khi tạo collection '%s' trên Cloud: %s", collection_name, e)
            return

    # 4. Migrate dữ liệu theo Batch (Scroll & Upsert)
    logger.info("🚀 Bắt đầu quá trình tải dữ liệu lên Cloud...")
    batch_size = 500
    offset = None
    transferred = 0
    t_start_migration = time.time()

    while True:
        # Scroll lấy points từ local
        scroll_res = local_client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            with_vectors=True,
            with_payload=True,
            offset=offset
        )
        points, next_offset = scroll_res
        
        if not points:
            break
            
        # Convert Records to PointStructs
        points_to_upload = [
            PointStruct(
                id=record.id,
                vector=record.vector,
                payload=record.payload
            )
            for record in points
        ]
        
        # Tải lên Cloud
        cloud_client.upsert(
            collection_name=collection_name,
            points=points_to_upload
        )
        
        transferred += len(points)
        percentage = (transferred / total_points) * 100 if total_points > 0 else 100
        logger.info("📤 Đã tải lên: %d/%d points (%.1f%%)", transferred, total_points, percentage)
        
        if next_offset is None:
            break
        offset = next_offset

    duration = time.time() - t_start_migration
    logger.info("🎉 Hoàn thành di chuyển dữ liệu!")
    logger.info("⏱️ Tổng thời gian upload: %.2fs", duration)
    logger.info("✨ Vui lòng đổi QDRANT_URL trong file .env để kết nối trực tiếp đến Cloud.")

if __name__ == "__main__":
    migrate()
