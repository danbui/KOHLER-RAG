# -*- coding: utf-8 -*-
"""
search.py - Pipeline tìm kiếm và rerank cho hệ thống RAG
==========================================================
Module chứa toàn bộ logic tìm kiếm: chuẩn hoá query,
xây dựng Qdrant filter, dense vector search + rerank,
và format kết quả thành context string.

Pipeline chính:
    query → normalize_sku_query()
          → build_qdrant_filter()
          → search Qdrant (BGE-M3 embedding)
          → rerank (BGE Reranker v2)
          → format_context_and_sources()

Sử dụng:
    from rag_system.search import search_and_rerank, build_qdrant_filter

    qdrant_filter = build_qdrant_filter("2025", "Kohler", "Tất cả")
    context_str, sources = search_and_rerank("Giá bồn cầu?", qdrant_filter)
"""

import os
import re
import time
import logging
from typing import Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

import config
import ai_models

# ── Logger ───────────────────────────────────────────────────────────
logger = logging.getLogger("rag_system.search")

# ── Regex pattern nhận diện mã SKU (VD: K-8658T-CP, K77017X-0) ──────
_SKU_PATTERN = re.compile(r"[A-Z]+-?\d+[A-Z]*-?\w*", re.IGNORECASE)


# =====================================================================
#  1. Chuẩn hoá query chứa mã SKU
# =====================================================================

def normalize_sku_query(query: str) -> str:
    """Phát hiện mã SKU trong query và tạo biến thể mở rộng.

    Ví dụ: "Giá K-8658T-CP" → "Giá K-8658T-CP K8658TCP"
    Việc thêm biến thể không dấu gạch giúp matching tốt hơn
    khi dữ liệu lưu SKU không nhất quán (có/không dấu '-').

    Args:
        query: Câu truy vấn gốc từ người dùng.

    Returns:
        str — query đã mở rộng thêm biến thể SKU (nếu tìm thấy),
              hoặc query gốc nếu không chứa SKU.
    """
    matches = _SKU_PATTERN.findall(query)
    if not matches:
        return query

    # Tạo biến thể không dấu gạch ngang cho mỗi SKU tìm được
    variants = []
    for sku in matches:
        no_dash = sku.replace("-", "")
        if no_dash.lower() != sku.lower():
            variants.append(no_dash)

    if variants:
        expanded = query + " " + " ".join(variants)
        logger.debug("SKU variants mở rộng: %s → %s", query, expanded)
        return expanded

    return query


# =====================================================================
#  2. Xây dựng Qdrant filter từ UI
# =====================================================================

def build_qdrant_filter(
    year_filter: Optional[str],
    brand_filter: Optional[str],
    doc_type_filter: Optional[str],
) -> Optional[Filter]:
    """Xây dựng Qdrant Filter từ các giá trị lọc trên giao diện.

    Giá trị "Tất cả" hoặc None/empty sẽ được bỏ qua (không lọc).

    Args:
        year_filter:     Năm cần lọc, VD: "2025", "Tất cả".
        brand_filter:    Thương hiệu, VD: "Kohler", "Karat", "Tất cả".
        doc_type_filter: Loại tài liệu, VD: "Bảng giá", "Catalogue", "Tất cả".

    Returns:
        Filter | None — đối tượng Qdrant Filter, hoặc None nếu không có điều kiện.
    """
    conditions: list[FieldCondition] = []

    if year_filter and year_filter != "Tất cả":
        try:
            conditions.append(
                FieldCondition(key="year", match=MatchValue(value=int(year_filter)))
            )
        except (ValueError, TypeError):
            logger.warning("year_filter không hợp lệ, bỏ qua: %s", year_filter)

    if brand_filter and brand_filter != "Tất cả":
        conditions.append(
            FieldCondition(key="brand", match=MatchValue(value=brand_filter))
        )
    if doc_type_filter and doc_type_filter != "Tất cả":
        conditions.append(
            FieldCondition(key="doc_type", match=MatchValue(value=doc_type_filter))
        )

    return Filter(must=conditions) if conditions else None


# =====================================================================
#  3. Format context và sources từ kết quả rerank
# =====================================================================

def format_context_and_sources(
    top_results: list[tuple],
) -> tuple[str, list[dict]]:
    """Format danh sách kết quả rerank thành context string và sources list.

    Args:
        top_results: Danh sách tuple (ScoredPoint, rerank_score)
                     đã được sort theo rerank score giảm dần.

    Returns:
        (context_str, sources_list):
            - context_str: Chuỗi chứa nội dung tài liệu để đưa vào prompt.
            - sources_list: Danh sách dict chứa thông tin nguồn trích dẫn.
    """
    context_str = ""
    sources: list[dict] = []

    for idx, (doc, score) in enumerate(top_results):
        payload = doc.payload

        # Xây dựng context block cho prompt (dùng .get() để tránh KeyError)
        src_file = payload.get("source_file", "Unknown")
        location = payload.get("location", "Unknown")
        content = payload.get("content", "")

        context_str += (
            f"[Tài liệu {idx + 1}]\n"
            f"File: {src_file}\n"
            f"Location: {location}\n"
            f"Content: {content}\n\n"
        )

        # Thông tin nguồn để hiển thị trên UI
        sources.append({
            "file_name": os.path.basename(src_file),
            "rel_path": src_file,
            "location": location,
            "content": content,
            "score": score,
        })

    return context_str, sources


# =====================================================================
#  4. Pipeline chính: Search + Rerank
# =====================================================================

def search_and_rerank(
    query: str,
    qdrant_filter: Optional[Filter],
    top_k: Optional[int] = None,
) -> tuple[str, list[dict]]:
    """Pipeline tìm kiếm: embed query → Qdrant search → rerank → format.

    Bước 1: Chuẩn hoá SKU trong query
    Bước 2: Encode query bằng BGE-M3 (từ ai_models)
    Bước 3: Dense search trên Qdrant (top 30 candidates)
    Bước 4: Rerank bằng BGE Reranker v2, lấy top_k kết quả
    Bước 5: Format thành context string và sources list

    Args:
        query:         Câu truy vấn gốc từ người dùng.
        qdrant_filter: Bộ lọc Qdrant (None = không lọc).
        top_k:         Số kết quả cuối cùng sau rerank (mặc định lấy từ config).

    Returns:
        (context_str, sources_list):
            - context_str: Chuỗi context cho prompt, hoặc "" nếu không có kết quả.
            - sources_list: Danh sách dict nguồn trích dẫn.

    Raises:
        RuntimeError: Nếu Qdrant client chưa kết nối.
        Exception:    Nếu Qdrant search gặp lỗi (collection chưa tồn tại, v.v.).
    """
    if top_k is None:
        top_k = config.RAG_TOP_K

    # ── Bước 1: Chuẩn hoá SKU ──
    expanded_query = normalize_sku_query(query)
    if expanded_query != query:
        logger.info("Query mở rộng SKU: '%s' → '%s'", query, expanded_query)

    # ── Bước 2: Encode query bằng BGE-M3 ──
    t0 = time.time()
    query_vector = ai_models.encode_query(expanded_query)
    if not query_vector:
        logger.error("Không thể encode query — model chưa sẵn sàng")
        return "", []

    # ── Bước 3: Dense search trên Qdrant ──
    qdrant = config.qdrant_client
    if qdrant is None:
        raise RuntimeError("Qdrant client chưa được kết nối")

    search_results = qdrant.query_points(
        collection_name="kohler_rag",
        query=query_vector,
        query_filter=qdrant_filter,
        limit=config.CANDIDATE_LIMIT,  # Lấy top candidates để rerank (cấu hình từ config)
    ).points
    logger.info(
        "🔍 Retrieved %d candidates từ Qdrant trong %.3fs",
        len(search_results),
        time.time() - t0,
    )

    if not search_results:
        return "", []

    # ── Bước 4: Rerank (nếu bật) hoặc dùng Qdrant score ──
    t1 = time.time()

    if config.ENABLE_RERANKER:
        pairs = [[query, res.payload.get("content", "")] for res in search_results]
        rerank_scores = ai_models.rerank_pairs(pairs)
    else:
        rerank_scores = []  # Skip reranker → dùng Qdrant score

    if rerank_scores:
        scored_results = list(zip(search_results, rerank_scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_results[:top_k]
        logger.info("🏆 Rerank hoàn thành trong %.3fs", time.time() - t1)
    else:
        # Dùng Qdrant similarity score (đã được sort sẵn)
        top_results = [(res, float(res.score if res.score is not None else 0.0)) for res in search_results[:top_k]]
        if config.ENABLE_RERANKER:
            logger.warning("⚠️ Rerank thất bại — fallback sang Qdrant score")
        else:
            logger.info("⚡ Reranker OFF — dùng Qdrant score (%.3fs)", time.time() - t1)

    # ── Bước 5: Format context và sources ──
    return format_context_and_sources(top_results)
