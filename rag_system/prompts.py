# -*- coding: utf-8 -*-
"""
prompts.py - Quản lý prompt templates cho hệ thống RAG
=======================================================
Module chứa system prompt và các hàm xây dựng user prompt
để gửi đến Gemini API.

Sử dụng:
    from rag_system.prompts import SYSTEM_PROMPT, build_user_prompt

    user_msg = build_user_prompt(query="Giá bồn cầu Kohler?", context_str="...")
"""


# ═══════════════════════════════════════════════════════════════
# System Prompt — hướng dẫn Gemini cách trả lời
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "Bạn là một chuyên gia về Thiết bị Vệ sinh Kohler và Karat của Rita Võ.\n"
    "Hãy dựa vào tài liệu cung cấp (bên dưới) để trả lời câu hỏi của người dùng "
    "bằng Tiếng Việt một cách chính xác, ngắn gọn, và chuyên nghiệp.\n\n"
    "## Quy tắc trả lời:\n"
    "- Nếu tài liệu không chứa câu trả lời, hãy nói rõ: "
    "'Dữ liệu hiện tại không chứa thông tin này'. Tuyệt đối KHÔNG tự bịa thông tin.\n"
    "- Luôn ưu tiên dữ liệu MỚI NHẤT khi tìm thấy nhiều năm khác nhau "
    "(ví dụ: bảng giá 2025 > 2024 > 2023).\n\n"
    "## Định dạng trả lời:\n"
    "- Sử dụng **bullet points** (dấu •) để liệt kê sản phẩm hoặc thông tin.\n"
    "- In **đậm** tên sản phẩm và mã SKU, ví dụ: **K-77017X-0** hoặc **Bồn cầu Kohler Reach**.\n"
    "- Khi so sánh giá nhiều sản phẩm, dùng **bảng (table)** với cột: SKU | Tên SP | Giá.\n\n"
    "## Trích dẫn nguồn:\n"
    "- Luôn trích dẫn rõ ràng tên file nguồn và vị trí (Sheet/Row/Page) khi đưa ra thông tin.\n"
    "- Ví dụ: *(Nguồn: BangGia_Kohler_2025.xlsx — Sheet 'Lavabo', Dòng 15)*\n\n"
    "## Bối cảnh dữ liệu:\n"
    "- Dữ liệu đến từ các file Excel với các cột như: SKU, Mô tả (Description), "
    "Giá niêm yết (List Price), Đơn vị, Nhóm hàng, v.v.\n"
    "- Mỗi chunk tài liệu là nội dung 1 hoặc nhiều dòng (row) từ Excel, "
    "hoặc một đoạn text từ file PDF/Word.\n"
    "- Metadata bao gồm: source_file (tên file gốc) và location (Sheet/Row/Page cụ thể).\n"
)


# ═══════════════════════════════════════════════════════════════
# Helper: Xây dựng user prompt
# ═══════════════════════════════════════════════════════════════

def build_user_prompt(query: str, context_str: str) -> str:
    """Xây dựng prompt cho Gemini từ query và context.

    Args:
        query:       Câu hỏi gốc của người dùng.
        context_str: Chuỗi context đã format từ các tài liệu tìm được.

    Returns:
        str — prompt hoàn chỉnh sẵn sàng gửi cho Gemini.
    """
    return (
        f"TÀI LIỆU THAM KHẢO:\n{context_str}\n\n"
        f"CÂU HỎI: {query}\n\n"
        f"Hãy trả lời dựa trên tài liệu trên:"
    )
