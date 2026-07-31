"""
DB / domain model cho Chat (placeholder).

NestJS: Prisma model / Mongoose schema.
Mẫu học chưa persist — giữ file để bạn thấy chỗ đặt entity.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ChatMessage:
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
