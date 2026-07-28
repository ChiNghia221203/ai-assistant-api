"""
DTO / validation cho Chat.

NestJS tương đương: create-chat.dto.ts + class-validator + @ApiProperty
Python: Pydantic BaseModel (FastAPI tự validate + docs).
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Tin nhắn người dùng gửi cho AI",
        examples=["Xin chào, FastAPI là gì?"],
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=2000,
        description="Prompt hệ thống (optional)",
    )


class ChatResponse(BaseModel):
    reply: str
    mock: bool = Field(description="true nếu đang dùng MOCK_LLM")
