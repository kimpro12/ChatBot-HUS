const DEFAULT_BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export type RetrievedContext = {
  answer_context: string;
};

export type ChatMessage = {
  role: string;
  content: string;
};

export type ChatResponse = {
  answer: string;
  context: string;
};

export async function retrieveContext(question: string, k = 6): Promise<RetrievedContext> {
  const response = await fetch(`${DEFAULT_BACKEND_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question, k }),
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message = detail?.detail ?? response.statusText;
    throw new Error(`Backend query failed: ${message}`);
  }

  return (await response.json()) as RetrievedContext;
}

export async function chatWithBackend(messages: ChatMessage[], k = 6): Promise<ChatResponse> {
  const response = await fetch(`${DEFAULT_BACKEND_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ messages, k }),
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message = detail?.detail ?? response.statusText;
    throw new Error(`Backend chat failed: ${message}`);
  }

  return (await response.json()) as ChatResponse;
}
