# -*- coding: utf-8 -*-
"""
ai_models.py - Quản lý mô hình AI qua HuggingFace Inference API
=================================================================
Module này gọi BGE-M3 và BGE Reranker v2 thông qua HuggingFace
Inference API (sử dụng huggingface_hub.InferenceClient).

Ưu điểm:
  - RAM models = 0 MB (chạy trên server HuggingFace)
  - Thời gian khởi động = vài giây (warm-up API, không load model)
  - Vẫn dùng cùng model BGE-M3 → không cần re-index database

Sử dụng:
    from ai_models import start_loading, is_ready, encode_query

    start_loading()          # Warm up models trên HuggingFace
    ...
    if is_ready():
        vector = encode_query("câu hỏi của tôi")
"""

import time
import logging
import threading
import numpy as np
from functools import lru_cache
from typing import Optional

from huggingface_hub import InferenceClient

import config

# ── Logger ───────────────────────────────────────────────────────────
logger = logging.getLogger("rag_system.ai_models")

# ── HuggingFace Inference Client ────────────────────────────────────
_hf_client: Optional[InferenceClient] = None

# ── Trạng thái module-level (thread-safe qua _lock) ─────────────────
_lock = threading.Lock()
_models_ready = False
_loading_started = False
_load_error: Optional[str] = None


# =====================================================================
#  INTERNAL: Warmup HuggingFace models
# =====================================================================

def _warmup_models() -> None:
    """Warm up models trên HuggingFace — chạy trong daemon thread.

    Gửi test request để HuggingFace load models vào GPU/CPU của họ.
    Lần gọi đầu (cold start) có thể mất 20-60s, sau đó nhanh (~200ms).
    """
    global _models_ready, _load_error, _hf_client

    try:
        logger.info("🔄 [Background] Warming up models trên HuggingFace...")
        t0 = time.time()

        # Khởi tạo client
        _hf_client = InferenceClient(token=config.HF_TOKEN)

        # --- Warm up BGE-M3 Embedding ---
        logger.info("   ⏳ Warming up BGE-M3 trên HuggingFace...")
        result = _hf_client.feature_extraction(
            "warmup test query",
            model=config.HF_EMBEDDING_MODEL,
        )
        vec = np.array(result)
        logger.info("   ✅ BGE-M3 ready — vector shape: %s (%.1fs)", vec.shape, time.time() - t0)

        # --- Warm up BGE Reranker (chỉ khi bật để không block startup không cần thiết) ---
        if config.ENABLE_RERANKER:
            t1 = time.time()
            logger.info("   ⏳ Warming up BGE Reranker trên HuggingFace...")
            _hf_client.text_classification(
                "warmup query",
                model=config.HF_RERANKER_MODEL,
            )
            logger.info("   ✅ Reranker ready (%.1fs)", time.time() - t1)
        else:
            logger.info("   ⚡ Reranker OFF — bỏ qua warmup reranker")

        # Đánh dấu sẵn sàng
        with _lock:
            _models_ready = True
        logger.info("🎉 Tất cả models sẵn sàng trên HuggingFace! (tổng %.1fs)", time.time() - t0)

    except Exception as e:
        with _lock:
            _load_error = str(e)
        logger.error("❌ Lỗi khi warm up HF models: %s", e, exc_info=True)


# =====================================================================
#  PUBLIC API
# =====================================================================

def start_loading() -> Optional[threading.Thread]:
    """Warm up models trên HuggingFace trong background thread.

    Gửi test request để HuggingFace load models sẵn sàng.
    Nếu đã gọi rồi (hot-reload), sẽ bỏ qua.

    Returns:
        threading.Thread đang chạy (daemon), hoặc None nếu đã gọi rồi.
    """
    global _loading_started
    with _lock:
        if _loading_started:
            logger.info("⚠️ start_loading() đã được gọi trước đó, bỏ qua.")
            return None
        _loading_started = True

    if not config.HF_TOKEN:
        with _lock:
            _load_error = "HF_TOKEN chưa được thiết lập trong file .env"
        logger.error("❌ HF_TOKEN chưa được thiết lập!")
        return None

    thread = threading.Thread(
        target=_warmup_models,
        name="hf-model-warmup",
        daemon=True,
    )
    thread.start()
    logger.info("🚀 Đã khởi chạy thread warm up HF models (%s)", thread.name)
    return thread


def is_ready() -> bool:
    """Kiểm tra tất cả models đã sẵn sàng trên HuggingFace chưa."""
    with _lock:
        return _models_ready


def get_load_error() -> Optional[str]:
    """Trả về thông báo lỗi nếu warmup thất bại, None nếu chưa có lỗi."""
    with _lock:
        return _load_error


@lru_cache(maxsize=config.QUERY_EMBEDDING_CACHE_SIZE)
def _encode_query_cached(query: str) -> tuple[float, ...]:
    """Encode query và cache kết quả để giảm latency cho truy vấn lặp lại."""
    if _hf_client is None:
        logger.error("HF client chưa được khởi tạo")
        return ()

    try:
        result = _hf_client.feature_extraction(
            query,
            model=config.HF_EMBEDDING_MODEL,
        )
        vec = np.array(result).flatten()
        return tuple(float(x) for x in vec.tolist())

    except Exception as e:
        logger.error("HF Embedding API error: %s", e)
        return ()


def encode_query(query: str) -> list:
    """Encode câu truy vấn thành embedding vector qua HuggingFace API.

    Gọi BGE-M3 trên HuggingFace Inference API để tạo vector 1024d.

    Args:
        query: Câu truy vấn cần encode.

    Returns:
        list[float] — vector embedding 1024d, hoặc [] nếu có lỗi.
    """
    return list(_encode_query_cached(query))


def rerank_pairs(pairs: list[list[str]]) -> list[float]:
    """Rerank các cặp (query, document) qua 1 batch HuggingFace API call.

    Gửi tất cả pairs trong 1 request duy nhất thay vì gọi riêng lẻ,
    giảm overhead từ N*latency xuống ~1*latency.

    Args:
        pairs: Danh sách cặp [query, document] cần rerank.

    Returns:
        list[float] — điểm relevance tương ứng, hoặc [] nếu có lỗi.
    """
    if not pairs or _hf_client is None:
        return []

    try:
        # Kết hợp mỗi cặp thành 1 chuỗi cho text-classification batch
        inputs = [f"{pair[0]} [SEP] {pair[1]}" for pair in pairs]

        # 1 batch API call duy nhất (thay vì N calls riêng lẻ)
        import requests as req_lib
        t0 = time.time()
        resp = req_lib.post(
            f"https://router.huggingface.co/hf-inference/models/{config.HF_RERANKER_MODEL}",
            headers={
                "Authorization": f"Bearer {config.HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json={"inputs": inputs},
            timeout=config.RERANK_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()

        # Parse response
        results = resp.json()
        logger.info("Batch rerank API call: %.2fs (%d pairs)", time.time() - t0, len(pairs))

        # HF API trả format: [[{label, score}, {label, score}, ...]]
        # Tất cả scores nằm trong results[0], mỗi entry = 1 pair
        score_entries = results[0] if results and isinstance(results[0], list) else results
        scores = [entry.get("score", 0.0) for entry in score_entries]

        # Log top/bottom scores để debug
        if scores:
            logger.info(
                "Rerank scores: max=%.4f, min=%.4f, median=%.4f",
                max(scores), min(scores), sorted(scores)[len(scores) // 2],
            )

        return scores

    except Exception as e:
        logger.error("Batch rerank error: %s — falling back to no-rerank", e)
        return []


# ── Backward compatibility (không dùng nữa nhưng giữ để không lỗi import) ──

def get_embedding_model():
    """Deprecated — dùng encode_query() thay thế."""
    return None


def get_reranker():
    """Deprecated — dùng rerank_pairs() thay thế."""
    return None
