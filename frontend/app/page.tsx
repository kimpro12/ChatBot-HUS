"use client";

import { FormEvent, useMemo } from "react";
import { useChat } from "ai/react";

const CTA_MESSAGE = "Hỏi về điểm chuẩn, tổ hợp xét tuyển, hoặc thông tin tuyển sinh.";

export default function HomePage() {
  const { messages, input, handleInputChange, handleSubmit, isLoading, stop } = useChat({
    api: "/api/chat",
    initialMessages: [
      {
        id: "system-intro",
        role: "assistant",
        content: "Xin chào! Tôi là trợ lý tuyển sinh của trường. Bạn cần biết điều gì?",
      },
    ],
  });

  const lastAnswer = useMemo(() => messages.filter((msg) => msg.role === "assistant").at(-1), [messages]);

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    handleSubmit(event);
  };

  return (
    <main className="max-w-3xl w-full">
      <header className="text-center space-y-2">
        <h1 className="text-3xl font-bold">Chatbot Tuyển Sinh</h1>
        <p className="text-slate-300">{CTA_MESSAGE}</p>
      </header>

      <section className="flex flex-col gap-3 bg-slate-900/50 border border-slate-700 rounded-xl p-4">
        <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-2">
          {messages.map((msg) => (
            <article key={msg.id} className="bg-slate-900/70 border border-slate-800 rounded-lg p-3">
              <h3 className="font-semibold text-sky-300">{msg.role === "assistant" ? "Trợ lý" : "Bạn"}</h3>
              <p className="whitespace-pre-wrap text-slate-100">{msg.content}</p>
            </article>
          ))}
          {isLoading ? <p className="text-sm text-slate-400">Đang soạn câu trả lời...</p> : null}
        </div>
        <form onSubmit={onSubmit} className="flex flex-col gap-2">
          <textarea
            value={input}
            onChange={handleInputChange}
            placeholder="Ví dụ: Điểm chuẩn ngành Khoa học dữ liệu năm 2024?"
            className="w-full min-h-[120px] rounded-lg border border-slate-700 bg-slate-950/60 p-3 text-slate-50"
          />
          <div className="flex gap-2 justify-end">
            {isLoading ? (
              <button type="button" onClick={stop} className="px-4 py-2 rounded-md bg-slate-700 text-slate-100">
                Dừng
              </button>
            ) : null}
            <button type="submit" className="px-4 py-2 rounded-md bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold">
              Gửi câu hỏi
            </button>
          </div>
        </form>
      </section>

      {lastAnswer ? (
        <footer className="text-sm text-slate-400">
          <p>
            <strong>Gợi ý:</strong> Hãy hỏi rõ tên ngành, chương trình hoặc bảng nếu bạn cần thông tin chi tiết.
          </p>
        </footer>
      ) : null}
    </main>
  );
}
