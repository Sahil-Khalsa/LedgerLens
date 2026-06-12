const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "";

export interface Filing {
  accn: string;
  ticker: string;
  form: string;
  filing_date: string;
  report_date: string;
  page_count: number;
  is_indexed_visual: boolean;
  is_indexed_text: boolean;
  is_amendment: boolean;
}

export interface FilingsResponse {
  items: Filing[];
  total: number;
  page: number;
  page_size: number;
}

export interface SSEStatusEvent {
  stage: string;
  pipeline?: string;
  chunks_found?: number;
}

export interface SSETokenEvent {
  text: string;
}

export interface SSEResultEvent {
  query_id: string;
  answer: string | null;
  answer_status: "answered" | "low_confidence" | "insufficient_data" | "error";
  facts: Array<{
    text: string;
    value: number | null;
    concept: string | null;
    page_ref: string;
    verified: string;
    bbox: number[] | null;
  }>;
  route: string;
  retries: number;
  cost_usd: number;
  latency_ms: number;
  pipeline: string;
  thread_id: string | null;
}

function headers(): Record<string, string> {
  return { "X-API-Key": API_KEY, "Content-Type": "application/json" };
}

export async function listFilings(
  ticker?: string,
  page = 1,
  pageSize = 20,
): Promise<FilingsResponse> {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
  if (ticker) params.set("ticker", ticker);
  const res = await fetch(`${API_BASE}/filings?${params}`, { headers: headers() });
  if (!res.ok) throw new Error(`Failed to list filings: ${res.status}`);
  return res.json() as Promise<FilingsResponse>;
}

export type QueryCallbacks = {
  onStatus?: (e: SSEStatusEvent) => void;
  onToken?: (text: string) => void;
  onResult?: (e: SSEResultEvent) => void;
  onError?: (msg: string) => void;
  onDone?: () => void;
};

export async function streamQuery(
  question: string,
  filingAccn: string | null,
  pipeline: "baseline" | "visual" | "auto",
  threadId: string | null,
  callbacks: QueryCallbacks,
): Promise<void> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ question, filing_accn: filingAccn, pipeline, thread_id: threadId }),
  });

  if (!res.ok || !res.body) {
    callbacks.onError?.(`HTTP ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";

    for (const part of parts) {
      const lines = part.trim().split("\n");
      let eventName = "";
      let dataLine = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) eventName = line.slice(7);
        if (line.startsWith("data: ")) dataLine = line.slice(6);
      }
      if (!eventName || !dataLine) continue;

      try {
        const payload = JSON.parse(dataLine);
        if (eventName === "status") callbacks.onStatus?.(payload as SSEStatusEvent);
        else if (eventName === "token") callbacks.onToken?.((payload as SSETokenEvent).text);
        else if (eventName === "result") callbacks.onResult?.(payload as SSEResultEvent);
        else if (eventName === "done") callbacks.onDone?.();
      } catch {
        // malformed SSE chunk — ignore
      }
    }
  }
}

export function pageImageUrl(accn: string, pageIdx: number): string {
  return `${API_BASE}/pages/${accn}/${pageIdx}`;
}
