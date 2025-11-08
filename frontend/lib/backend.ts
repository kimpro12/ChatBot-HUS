export type RetrievedContext = {
  answer_context: string;
};

const DEFAULT_BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

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
