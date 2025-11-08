import { NextRequest } from "next/server";
import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";

import { retrieveContext } from "@/lib/backend";
import { buildPrompt } from "@/lib/prompt";

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

  const result = await streamText({
    model: openai("gpt-4o-mini"),
    prompt,
  });

  return result.toAIStreamResponse();
}
