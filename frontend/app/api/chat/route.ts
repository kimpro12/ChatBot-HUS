import { NextRequest } from "next/server";
import { chatWithBackend } from "@/lib/backend";

export async function POST(request: NextRequest) {
  const { messages } = await request.json();

  if (!Array.isArray(messages) || messages.length === 0) {
    return new Response("Missing chat history", { status: 400 });
  }

  try {
    const data = await chatWithBackend(messages);
    return Response.json(data);
  } catch (error) {
    console.error("Failed to call backend chat", error);
    const message =
      error instanceof Error && error.message
        ? error.message
        : "Không thể kết nối tới backend.";
    return new Response(message, { status: 502 });
  }
}
