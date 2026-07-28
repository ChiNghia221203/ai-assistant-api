"""
LLM client — gọi OpenAI hoặc mock khi học.

NestJS tương đương: Injectable service gọi HTTP bên ngoài (axios/SDK).
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LlmClient:
    """Gói gọi chat completion — domain không cần biết chi tiết HTTP."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def complete(self, system: str, user: str) -> str:
        if self.settings.mock_llm or not self.settings.openai_api_key:
            return self._mock_complete(system, user)
        return await self._openai_complete(system, user)

    def _mock_complete(self, system: str, user: str) -> str:
        """Trả lời giả lập — để chạy app không cần API key."""
        logger.debug("MOCK_LLM: system=%s user=%s", system[:40], user[:80])
        return (
            "[MOCK LLM] Tôi đã nhận được tin nhắn của bạn.\n"
            f"→ Nội dung: {user}\n"
            "(Bật OPENAI_API_KEY và đặt MOCK_LLM=false để dùng model thật.)"
        )

    async def _openai_complete(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
        }
        url = f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"]


def get_llm_client() -> LlmClient:
    """Factory cho Depends() — ≈ @Injectable() provider."""
    return LlmClient()
