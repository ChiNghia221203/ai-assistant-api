"""
Database stub (học cấu trúc).

NestJS tương đương: PrismaService / MongooseModule.
Mẫu này chưa kết nối DB thật — chỉ giữ chỗ để bạn mở rộng sau.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


class Database:
    """Giả lập connection pool — sau này thay bằng SQLAlchemy/asyncpg."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False


_db: Database | None = None


def get_database(url: str) -> Database:
    global _db
    if _db is None:
        _db = Database(url)
    return _db


@asynccontextmanager
async def db_session(url: str) -> AsyncGenerator[Database, None]:
    db = get_database(url)
    if not db.connected:
        await db.connect()
    yield db
