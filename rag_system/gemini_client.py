# -*- coding: utf-8 -*-
"""
gemini_client.py - Module gọi Gemini API cho hệ thống RAG
===========================================================
Chức năng:
  - Gửi prompt (kèm system instruction + history) đến Gemini API
  - Retry tự động khi gặp lỗi tạm thời (timeout, HTTP 5xx)
  - Hỗ trợ streaming (generator) cho tích hợp SSE trong tương lai

Sử dụng:
    from gemini_client import query_gemini, query_gemini_stream

    answer = query_gemini(user_prompt, system_prompt, history=messages)

    for chunk in query_gemini_stream(user_prompt, system_prompt):
        print(chunk, end="")
"""

import json
import logging
import time
import urllib.request
from typing import Generator, List, Optional
from urllib.error import HTTPError, URLError

from config import get_gemini_url

# ═══════════════════════════════════════════════════════════════
# Logger — sử dụng logging thay vì print()
# ═══════════════════════════════════════════════════════════════
logger = logging.getLogger("rag_system.gemini")

# ═══════════════════════════════════════════════════════════════
# Hằng số cấu hình
# ═══════════════════════════════════════════════════════════════

# Số lượt hội thoại gần nhất giữ lại khi gửi lên Gemini
MAX_HISTORY_TURNS = 3

# Số lần retry tối đa khi gặp lỗi tạm thời
_MAX_RETRIES = 2

# Hệ số backoff cơ sở (giây) — delay = base * 2^attempt → 1s, 2s
_BACKOFF_BASE = 1

# Timeout cho mỗi lần gọi API (giây)
_REQUEST_TIMEOUT = 30


# ═══════════════════════════════════════════════════════════════
# Hàm nội bộ
# ═══════════════════════════════════════════════════════════════

def _is_retryable(error: Exception) -> bool:
    """Kiểm tra xem lỗi có phải loại tạm thời (nên retry) không.

    Các lỗi được coi là tạm thời:
      - TimeoutError: hết thời gian chờ
      - URLError: lỗi mạng (DNS, kết nối bị từ chối, ...)
      - HTTPError với mã 5xx: lỗi phía server Gemini

    Returns:
        True nếu nên retry, False nếu không.
    """
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, URLError) and not isinstance(error, HTTPError):
        # Lỗi mạng thuần (không phải HTTP status code)
        return True
    if isinstance(error, HTTPError) and error.code >= 500:
        # Lỗi phía server (500, 502, 503, ...)
        return True
    return False


def _build_payload(
    user_prompt: str,
    system_prompt: str,
    history: Optional[List] = None,
) -> dict:
    """Xây dựng payload JSON để gửi đến Gemini API.

    Args:
        user_prompt:   Câu hỏi / prompt của người dùng.
        system_prompt: System instruction cho model.
        history:       Danh sách tin nhắn hội thoại trước đó
                       (mỗi phần tử cần có .role và .content).

    Returns:
        dict payload sẵn sàng json.dumps().
    """
    contents = []

    # Thêm tối đa MAX_HISTORY_TURNS tin nhắn gần nhất từ history
    if history:
        recent = history[-MAX_HISTORY_TURNS:]
        for msg in recent:
            role = msg.role if msg.role in ("user", "model") else "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}],
            })

    # Thêm câu hỏi hiện tại
    contents.append({
        "role": "user",
        "parts": [{"text": user_prompt}],
    })

    return {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "generationConfig": {
            "temperature": 0.2,
        },
    }


def _extract_text(response_json: dict) -> str:
    """Trích xuất văn bản trả lời từ JSON response của Gemini.

    Args:
        response_json: Dict đã parse từ response body.

    Returns:
        Chuỗi text trả lời, hoặc thông báo lỗi nếu rỗng.
    """
    candidates = response_json.get("candidates", [])
    if candidates:
        text = (
            candidates[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if text:
            return text
        logger.warning("Gemini trả về candidate nhưng text rỗng.")
        return "Lỗi: Gemini trả về phản hồi rỗng."
    return "Không nhận được phản hồi từ mô hình Gemini."


# ═══════════════════════════════════════════════════════════════
# Hàm public
# ═══════════════════════════════════════════════════════════════

def query_gemini(
    user_prompt: str,
    system_prompt: str,
    history: Optional[List] = None,
) -> str:
    """Gọi Gemini API và trả về câu trả lời dạng text.

    Hỗ trợ:
      - Multi-turn conversation thông qua ``history``
      - Retry tự động (tối đa 2 lần) với exponential backoff
        khi gặp lỗi timeout hoặc HTTP 5xx

    Args:
        user_prompt:   Prompt gửi đến model (bao gồm context tài liệu).
        system_prompt: System instruction mô tả vai trò của model.
        history:       Danh sách tin nhắn hội thoại trước đó (Optional).
                       Mỗi phần tử cần thuộc tính ``.role`` và ``.content``.

    Returns:
        Chuỗi text phản hồi từ Gemini, hoặc thông báo lỗi tiếng Việt
        nếu không thành công.

    Ví dụ::

        answer = query_gemini(
            user_prompt="Giá bồn cầu Kohler Reach?",
            system_prompt=SYSTEM_PROMPT,
            history=conversation_history,
        )
    """
    payload = _build_payload(user_prompt, system_prompt, history)
    data = json.dumps(payload).encode("utf-8")

    last_error: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                get_gemini_url(),
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
                res_json = json.loads(body)
                return _extract_text(res_json)

        except HTTPError as e:
            last_error = e
            if _is_retryable(e) and attempt < _MAX_RETRIES:
                delay = _BACKOFF_BASE * (2 ** attempt)  # 1s, 2s
                logger.warning(
                    "Gemini HTTP %d — retry %d/%d sau %ds",
                    e.code, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            # Lỗi HTTP không retry được (4xx, ...)
            logger.error("Gemini HTTP %d: %s", e.code, e.reason)
            return f"Lỗi API Gemini (HTTP {e.code}): {e.reason}"

        except URLError as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                delay = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Lỗi mạng khi gọi Gemini — retry %d/%d sau %ds: %s",
                    attempt + 1, _MAX_RETRIES, delay, e.reason,
                )
                time.sleep(delay)
                continue
            logger.error("Lỗi mạng gọi Gemini (hết retry): %s", e.reason)
            return f"Lỗi kết nối mạng đến Gemini: {e.reason}"

        except TimeoutError as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                delay = _BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Gemini timeout — retry %d/%d sau %ds",
                    attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            logger.error("Gemini timeout sau %d lần thử", _MAX_RETRIES + 1)
            return "Lỗi: Gemini API không phản hồi (timeout)."

        except Exception as e:
            # Lỗi không mong đợi — không retry
            logger.exception("Lỗi không xác định khi gọi Gemini API")
            return f"Lỗi gọi API Gemini: {e}"

    # Trường hợp fallback (không nên đến đây)
    logger.error("Hết retry gọi Gemini. Lỗi cuối: %s", last_error)
    return f"Lỗi gọi API Gemini sau nhiều lần thử: {last_error}"


def query_gemini_stream(
    user_prompt: str,
    system_prompt: str,
    history: Optional[List] = None,
) -> Generator[str, None, None]:
    """Generator trả về phản hồi Gemini theo từng chunk (SSE-ready).

    Hiện tại đây là wrapper đơn giản: gọi ``query_gemini()`` rồi
    yield toàn bộ kết quả thành một chunk duy nhất.

    Trong tương lai, hàm này sẽ sử dụng Gemini streaming endpoint
    (``streamGenerateContent``) và yield từng đoạn text nhỏ để hỗ trợ
    Server-Sent Events (SSE) trên frontend.

    Args:
        user_prompt:   Prompt gửi đến model.
        system_prompt: System instruction.
        history:       Danh sách tin nhắn hội thoại trước đó (Optional).

    Yields:
        Từng đoạn text (hiện tại yield một lần duy nhất).

    Ví dụ::

        for chunk in query_gemini_stream(prompt, sys_prompt):
            response.write(f"data: {chunk}\\n\\n")
    """
    full_response = query_gemini(user_prompt, system_prompt, history)
    yield full_response
