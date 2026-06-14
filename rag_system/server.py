# -*- coding: utf-8 -*-
"""
server.py - FastAPI Backend cho hệ thống RAG nội bộ Kohler/Karat (Rita Võ)
============================================================================
Phiên bản modular: server.py chỉ chứa route definitions.
Toàn bộ logic được tách vào các module riêng biệt:
  - config.py:        Cấu hình, .env, paths, Qdrant client
  - schemas.py:       Pydantic request/response models
  - ai_models.py:     Lazy loading BGE-M3 & Reranker
  - gemini_client.py: Gọi Gemini API (retry, streaming)
  - search.py:        Qdrant search + rerank pipeline
  - prompts.py:       System prompt & prompt builder
"""

import os
import sys
import logging
import time
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Import từ các module nội bộ ────────────────────────────────────────
from config import BASE_DIR, qdrant_client
from schemas import QueryRequest, HealthResponse
from ai_models import start_loading, is_ready, get_load_error
from gemini_client import query_gemini
from search import search_and_rerank, build_qdrant_filter
from prompts import SYSTEM_PROMPT, build_user_prompt

# ── Logging setup ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_system.server")


# ── Lifespan (thay thế @app.on_event deprecated) ─────────────────────
@asynccontextmanager
async def lifespan(app):
    """Khởi tạo model trong background thread khi app start."""
    start_loading()
    logger.info("🚀 Server started — models đang load trong background...")
    yield
    # Shutdown: không cần cleanup đặc biệt


# ── FastAPI App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Kohler Sanitary Ware Advanced RAG (Qdrant + BGE)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.get("/")
def get_index():
    """Serve frontend HTML."""
    index_html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_html_path):
        with open(index_html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>Frontend index.html not found!</h3>")


@app.get("/api/health")
def health_check():
    """Kiểm tra trạng thái hệ thống — models, Qdrant connection."""
    qdrant_connected = qdrant_client is not None
    collection_count = None

    if qdrant_connected:
        try:
            info = qdrant_client.get_collection("kohler_rag")
            collection_count = info.points_count
        except Exception:
            pass

    status = "ready" if is_ready() and qdrant_connected else "loading"
    load_error = get_load_error()
    if load_error:
        status = "error"

    return HealthResponse(
        status=status,
        models_ready=is_ready(),
        qdrant_connected=qdrant_connected,
        collection_count=collection_count,
    )


@app.post("/api/query")
async def handle_query(req: QueryRequest):
    """
    Endpoint chính — nhận câu hỏi, tìm kiếm Qdrant, rerank, gọi Gemini.
    Hỗ trợ optional conversation history.
    """
    # ── Kiểm tra model đã sẵn sàng chưa ──
    if not is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "answer": "Hệ thống đang khởi tạo mô hình AI, vui lòng thử lại sau 30 giây...",
                "sources": [],
            },
        )

    # ── Kiểm tra Qdrant client ──
    if qdrant_client is None:
        return JSONResponse(
            status_code=503,
            content={
                "answer": "Không thể kết nối cơ sở dữ liệu Qdrant. Vui lòng kiểm tra thư mục qdrant_db.",
                "sources": [],
            },
        )

    # Validate input đã được Pydantic Field(max_length=1000) xử lý
    query = req.query.strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "Query is empty"})

    year_filter = req.year
    brand_filter = req.brand
    doc_type_filter = req.doc_type
    history = req.history

    logger.info("📨 Processing query: %s", query)
    logger.info(
        "   Filters — Year: %s | Brand: %s | DocType: %s",
        year_filter, brand_filter, doc_type_filter,
    )
    if history:
        logger.info("   History: %d message(s)", len(history))

    # ── 1. Build Metadata Filters ──
    qdrant_filter = build_qdrant_filter(year_filter, brand_filter, doc_type_filter)

    # ── 2. Search + Rerank Pipeline (chạy trong thread pool để không block event loop) ──
    try:
        context_str, sources = await asyncio.to_thread(search_and_rerank, query, qdrant_filter)
    except RuntimeError as e:
        logger.error("Qdrant error: %s", e)
        return JSONResponse(
            status_code=503,
            content={
                "answer": f"Lỗi kết nối Qdrant: {e}",
                "sources": [],
            },
        )
    except Exception as e:
        logger.error("Search error: %s", e)
        error_msg = str(e)
        if "collection" in error_msg.lower() and "not found" in error_msg.lower():
            return JSONResponse(
                status_code=404,
                content={
                    "answer": "Cơ sở dữ liệu chưa có dữ liệu. Vui lòng chạy indexing trước.",
                    "sources": [],
                },
            )
        return JSONResponse(
            status_code=500,
            content={"error": f"Lỗi truy vấn Qdrant DB: {e}"},
        )

    if not context_str:
        return {
            "answer": "Không tìm thấy tài liệu phù hợp trong cơ sở dữ liệu với các bộ lọc đã chọn.",
            "sources": [],
        }

    # ── 3. Gọi Gemini (kèm history nếu có, chạy trong thread pool) ──
    user_prompt = build_user_prompt(query, context_str)

    t_start = time.time()
    answer = await asyncio.to_thread(query_gemini, user_prompt, SYSTEM_PROMPT, history)
    logger.info("💬 Answer generated in %.2fs", time.time() - t_start)

    return {
        "answer": answer,
        "sources": sources,
    }


@app.post("/api/reindex")
async def handle_reindex():
    """Chạy lại script build_rag_db.py để re-index dữ liệu."""
    logger.info("🔄 Re-indexing database...")

    # Tìm build script: ưu tiên trong rag_system/, fallback scripts/
    candidates = [
        os.path.join(BASE_DIR, "build_rag_db.py"),
        os.path.join(BASE_DIR, "..", "scripts", "build_rag_db.py"),
    ]
    build_script = None
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            build_script = abs_path
            break

    if build_script is None:
        tried_paths = [os.path.abspath(p) for p in candidates]
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Không tìm thấy script build tại: {', '.join(tried_paths)}. "
                         f"Vui lòng đặt file build_rag_db.py tại thư mục scripts/.",
            },
        )

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, build_script],
            capture_output=True, text=True, encoding="utf-8",
            timeout=600,  # timeout 10 phút
        )
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error("Reindex stderr: %s", result.stderr)
            return JSONResponse(
                status_code=500,
                content={"error": f"Script lỗi: {result.stderr[:500]}"},
            )
        return {"status": "success", "output": result.stdout[-500:] if result.stdout else ""}
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=504, content={"error": "Reindex timeout (>10 phút)"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
