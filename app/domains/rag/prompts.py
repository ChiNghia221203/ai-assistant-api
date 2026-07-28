
RAG_SYSTEM_PROMPT = """Bạn là trợ lý trả lời dựa trên ngữ cảnh (RAG).
Chỉ dùng thông tin trong CONTEXT. Nếu thiếu thông tin, nói rõ là không biết.
Trả lời bằng tiếng Việt, ngắn gọn.
"""


def build_rag_user_prompt(question: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Hãy trả lời câu hỏi dựa trên CONTEXT ở trên."
    )
