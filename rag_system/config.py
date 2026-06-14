# -*- coding: utf-8 -*-
"""
config.py - Cấu hình trung tâm cho hệ thống RAG
==================================================
Module chứa toàn bộ cấu hình: biến môi trường, đường dẫn,
API keys, và kết nối Qdrant.
"""

import os
import sys
import logging

# ── Logger ────────────────────────────────────────────────────────────
logger = logging.getLogger("rag_system.config")

# ═══════════════════════════════════════════════════════════════
# 1. UTF-8 stdout (hỗ trợ tiếng Việt trên Windows)
# ═══════════════════════════════════════════════════════════════
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════════
# 2. Load biến môi trường từ file .env
# ═══════════════════════════════════════════════════════════════
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
    logger.info("✅ Loaded .env via python-dotenv")
except ImportError:
    logger.info("⚠️  python-dotenv không có, dùng os.getenv trực tiếp")

# ═══════════════════════════════════════════════════════════════
# 3. Đường dẫn gốc của project
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════
# 4. Cấu hình Gemini API
# ═══════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    logger.warning("❌ GEMINI_API_KEY chưa được thiết lập! Kiểm tra file .env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Số lượng candidates tối đa khi search Qdrant (trước khi rerank)
CANDIDATE_LIMIT = int(os.getenv("CANDIDATE_LIMIT", "15"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))


def get_gemini_url(action: str = "generateContent") -> str:
    """Tạo Gemini API URL on-demand — tránh lưu API key trong module-level constant.

    Args:
        action: API action (mặc định 'generateContent').

    Returns:
        str — URL đầy đủ kèm API key.
    """
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:{action}?key={GEMINI_API_KEY}"
    )

# ═══════════════════════════════════════════════════════════════
# 5. Cấu hình HuggingFace Inference API (BGE-M3 & Reranker)
# ═══════════════════════════════════════════════════════════════
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    logger.warning("⚠️ HF_TOKEN chưa được thiết lập — models sẽ không hoạt động")

HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-m3")
HF_RERANKER_MODEL = os.getenv("HF_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() in ("true", "1", "yes")

# ═══════════════════════════════════════════════════════════════
# 5. Cấu hình và kết nối Qdrant (Hỗ trợ Local & Cloud/Server)
# ═══════════════════════════════════════════════════════════════
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

qdrant_client = None  # Khởi tạo mặc định là None

try:
    from qdrant_client import QdrantClient
    if QDRANT_URL:
        logger.info("🌐 Đang kết nối tới Qdrant Cloud/Server: %s", QDRANT_URL)
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            timeout=30
        )
    else:
        qdrant_db_path = os.getenv("QDRANT_DB_PATH", os.path.join(BASE_DIR, "qdrant_db"))
        logger.info("📂 Đang kết nối tới Qdrant Local DB: %s", qdrant_db_path)
        qdrant_client = QdrantClient(path=qdrant_db_path)
        
    logger.info("✅ Kết nối Qdrant thành công")
except Exception as e:
    logger.error("❌ Lỗi kết nối Qdrant: %s", e)
