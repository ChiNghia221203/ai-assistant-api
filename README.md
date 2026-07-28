# AI Assistant

> Một trợ lý AI thông minh được xây dựng bằng **Python** và **FastAPI**, hỗ trợ [mô tả ngắn gọn mục đích chính — ví dụ: trả lời câu hỏi, xử lý tài liệu, trò chuyện thông minh...].

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

---

## Giới thiệu

`AI-Assistant` là một backend service cung cấp API cho các tính năng trợ lý AI, được xây dựng trên nền tảng **FastAPI** với hiệu năng cao và dễ mở rộng.

**Điểm nổi bật:**

- Hiệu năng cao nhờ FastAPI (async/await)
- Tích hợp LLM (OpenAI / Gemini / Local model...)
- Hỗ trợ RAG (Retrieval-Augmented Generation) [nếu có]
- Xác thực & phân quyền người dùng
- Tự động sinh tài liệu API (Swagger/OpenAPI)

---

## Cấu trúc dự án

```
ai-assistant/
├── app/
│   ├── api/                # Định nghĩa routes/endpoints
│   │   └── v1/
│   ├── core/                # Config, settings, security
│   ├── models/               # Pydantic schemas / ORM models
│   ├── services/             # Business logic, xử lý AI
│   ├── db/                   # Kết nối database
│   └── main.py                # Entry point FastAPI app
├── tests/                     # Unit tests
├── .env.example                # Mẫu biến môi trường
├── requirements.txt             # Thư viện Python
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Yêu cầu hệ thống

- Python >= 3.11
- pip / poetry
- (Tùy chọn) Docker & Docker Compose
- (Tùy chọn) PostgreSQL / MongoDB / Vector DB (pgvector, Qdrant...)

---

## Cài đặt

### 1. Clone dự án

```bash
git clone https://github.com/<username>/ai-assistant.git
cd ai-assistant
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Cấu hình biến môi trường

Sao chép file mẫu và điền thông tin:

```bash
cp .env.example .env
```

```env
APP_ENV=development
OPENAI_API_KEY=your_api_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/ai_assistant
```

### 5. Chạy ứng dụng

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Truy cập tài liệu API tại: `http://localhost:8000/docs`

---

## Chạy bằng Docker

```bash
docker-compose up --build
```

---

## API Endpoints (ví dụ)

| Method | Endpoint             | Mô tả                      |
| ------ | -------------------- | -------------------------- |
| POST   | `/api/v1/chat`       | Gửi tin nhắn tới trợ lý AI |
| GET    | `/api/v1/health`     | Kiểm tra trạng thái server |
| POST   | `/api/v1/auth/login` | Đăng nhập người dùng       |

> Xem chi tiết đầy đủ tại `/docs` (Swagger UI) khi server đang chạy.

---

## Chạy kiểm thử (Testing)

```bash
pytest -v
```

---

## Công nghệ sử dụng

- **Backend:** Python, FastAPI, Pydantic
- **AI/LLM:** [OpenAI API / LangChain / LangGraph...]
- **Database:** [PostgreSQL / MongoDB / pgvector...]
- **Khác:** Docker, Uvicorn, Poetry/Pip

---
