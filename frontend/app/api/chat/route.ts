import { NextRequest } from "next/server";
import { streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

import { retrieveContext } from "@/lib/backend";
import { buildPrompt } from "@/lib/prompt";

const llm = createOpenAI({
  baseURL: process.env.LLM_BASE_URL ?? "http://localhost:8000/v1",
  apiKey: process.env.LLM_API_KEY ?? "token-abc123",
});

const DEFAULT_MODEL = process.env.LLM_MODEL ?? "Qwen/Qwen2.5-7B-Instruct-AWQ";

export async function POST(request: NextRequest) {
  const { messages } = await request.json();
  const question = [...messages].reverse().find((msg: any) => msg.role === "user")?.content ?? "";

  if (!question) {
    return new Response("Missing user question", { status: 400 });
  }

  let context = "";
  try {
    const response = await retrieveContext(question);
    context = response.answer_context;
  } catch (error) {
    console.error(error);
    context = "Không tìm thấy trong tài liệu.";
  }

  const prompt = buildPrompt(messages, context, question);

  try {
    const result = await streamText({
      model: llm(DEFAULT_MODEL),
      prompt,
    });

    return result.toAIStreamResponse();
  } catch (error) {
    console.error("Failed to call LLM", error);
    const message =
      error instanceof Error && error.message
        ? error.message
        : "Không thể kết nối tới máy chủ mô hình.";
    return new Response(message, { status: 502 });
  }
}
