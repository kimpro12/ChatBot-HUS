"use client";

import { FormEvent, useMemo, useRef, useState } from "react";

type ChatRole = "assistant" | "user";

type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
};

const CTA_MESSAGE = "Hỏi về điểm chuẩn, tổ hợp xét tuyển, hoặc thông tin tuyển sinh.";

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "system-intro",
      role: "assistant",
      content: "Xin chào! Tôi là trợ lý tuyển sinh của trường. Bạn cần biết điều gì?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const lastAnswer = useMemo(
    () => messages.filter((msg) => msg.role === "assistant").at(-1),
    [messages],
  );

  const stop = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
  };

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = input.trim();

    if (!trimmed || isLoading) {
      return;
    }

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
    };

    const nextMessages: ChatMessage[] = [...messages, userMessage];
    const placeholder: ChatMessage = {
      id: `assistant-${Date.now()}`,
      role: "assistant",
      content: "",
    };

    setMessages([...nextMessages, placeholder]);
    setInput("");
    setIsLoading(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ messages: nextMessages }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(await response.text());
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let assistantText = "";
      let done = false;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          assistantText += decoder.decode(value, { stream: !done });
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              content: assistantText,
            };
            return updated;
          });
        }
      }
    } catch (error) {
      if ((error as DOMException).name === "AbortError") {
        setMessages((prev) => prev.slice(0, -1));
      } else {
        const fallback =
          error instanceof Error && error.message
            ? error.message
            : "Đã có lỗi khi gọi mô hình.";
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: fallback,
          };
          return updated;
        });
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
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
            onChange={(event) => setInput(event.target.value)}
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
