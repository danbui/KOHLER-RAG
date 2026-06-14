# -*- coding: utf-8 -*-
"""
schemas.py - Pydantic models cho hệ thống RAG
===============================================
Định nghĩa các schema dùng cho request/response
của API endpoints.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# Request Models
# ═══════════════════════════════════════════════════════════════

class HistoryMessage(BaseModel):
    """Một tin nhắn trong lịch sử hội thoại."""
    role: Literal["user", "model"]  # Chỉ chấp nhận 'user' hoặc 'model'
    content: str


class QueryRequest(BaseModel):
    """Request body cho endpoint /query - truy vấn RAG."""
    query: str = Field(..., min_length=1, max_length=1000, description="Câu hỏi tra cứu")
    year: Optional[str] = "Tất cả"
    brand: Optional[str] = "Tất cả"
    doc_type: Optional[str] = "Tất cả"
    history: Optional[List[HistoryMessage]] = None


# ═══════════════════════════════════════════════════════════════
# Response Models
# ═══════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Response body cho endpoint /health - kiểm tra trạng thái hệ thống."""
    status: str
    models_ready: bool
    qdrant_connected: bool
    collection_count: Optional[int] = None
