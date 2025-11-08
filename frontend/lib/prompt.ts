import { Message } from "ai";

export function buildPrompt(messages: Message[], context: string, question: string): string {
  const history = messages
    .filter((msg) => msg.role !== "system")
    .map((msg) => `${msg.role.toUpperCase()}: ${msg.content}`)
    .join("\n");

  return `SYSTEM: Bạn là trợ lý tuyển sinh của trường Đại học. Chỉ trả lời dựa trên NGỮ CẢNH cung cấp. Nếu không tìm thấy thông tin hãy nói "Không tìm thấy trong tài liệu". Trả lời bằng tiếng Việt và dẫn nguồn theo định dạng (Tên tài liệu - Trang/Bảng).

HISTORY:
${history}

CONTEXT:
${context}

USER QUESTION: ${question}`;
}
