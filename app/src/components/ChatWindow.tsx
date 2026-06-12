"use client";

import { useRef, useState } from "react";
import { streamQuery, type SSEResultEvent } from "@/lib/api";
import { CitationChip } from "./CitationChip";

interface Message {
  role: "user" | "assistant";
  content: string;
  result?: SSEResultEvent;
  streaming?: boolean;
  stage?: string;
}

interface Props {
  filingAccn: string | null;
  pipeline: "baseline" | "visual" | "auto";
}

const SUGGESTIONS = [
  "What was NVIDIA's total revenue for fiscal year 2024?",
  "What was Microsoft's net income for fiscal year 2024?",
  "What was NVIDIA's gross profit for fiscal year 2024?",
  "What was NVIDIA's total revenue for fiscal year 2023?",
];

const STATUS_LABELS: Record<string, string> = {
  planner:     "Planning query…",
  retriever:   "Searching filings…",
  extractor:   "Extracting facts…",
  verifier:    "Verifying against XBRL…",
  synth:       "Synthesizing answer…",
  critic:      "Checking faithfulness…",
  synthesizer: "Synthesizing answer…",
};

function StatusRow({ stage }: { stage: string }) {
  return (
    <div className="flex items-center gap-2 text-gray-400 text-xs animate-fade-in py-1">
      <svg className="w-3 h-3 animate-spin text-brand-500" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
      </svg>
      {STATUS_LABELS[stage] ?? `${stage}…`}
    </div>
  );
}

function AnswerStatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    answered:          "bg-emerald-50 text-emerald-700 border-emerald-200",
    low_confidence:    "bg-amber-50  text-amber-700  border-amber-200",
    insufficient_data: "bg-gray-100  text-gray-500   border-gray-200",
    error:             "bg-red-50    text-red-600    border-red-200",
  };
  const labels: Record<string, string> = {
    answered:          "Verified",
    low_confidence:    "Low confidence",
    insufficient_data: "Not found",
    error:             "Error",
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium border ${map[status] ?? map.error}`}>
      {labels[status] ?? status}
    </span>
  );
}

export function ChatWindow({ filingAccn, pipeline }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const threadId = useRef<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  async function send(question?: string) {
    const q = (question ?? input).trim();
    if (!q || busy) return;
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "44px";
    setBusy(true);

    setMessages((prev) => [
      ...prev,
      { role: "user", content: q },
      { role: "assistant", content: "", streaming: true, stage: "thinking" },
    ]);
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);

    await streamQuery(q, filingAccn, pipeline, threadId.current, {
      onStatus: (e) => {
        setMessages((prev) => {
          const copy = [...prev];
          const last = copy[copy.length - 1];
          if (last?.role === "assistant") copy[copy.length - 1] = { ...last, stage: e.stage };
          return copy;
        });
      },
      onToken: (text) => {
        setMessages((prev) => {
          const copy = [...prev];
          const last = copy[copy.length - 1];
          if (last?.role === "assistant")
            copy[copy.length - 1] = { ...last, content: last.content + text, stage: "" };
          return copy;
        });
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      },
      onResult: (result) => {
        if (result.thread_id) threadId.current = result.thread_id;
        setMessages((prev) => {
          const copy = [...prev];
          const last = copy[copy.length - 1];
          if (last?.role === "assistant")
            copy[copy.length - 1] = {
              ...last,
              content: result.answer ?? "(no answer)",
              result,
              streaming: false,
              stage: "",
            };
          return copy;
        });
      },
      onError: (msg) => {
        setMessages((prev) => {
          const copy = [...prev];
          const last = copy[copy.length - 1];
          if (last?.role === "assistant")
            copy[copy.length - 1] = { ...last, content: `Error: ${msg}`, streaming: false, stage: "" };
          return copy;
        });
      },
      onDone: () => {
        setBusy(false);
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      },
    });
  }

  return (
    <div className="flex flex-col flex-1 overflow-hidden bg-gray-50">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full max-w-lg mx-auto gap-8 animate-fade-in">
            <div className="text-center">
              <div className="w-12 h-12 rounded-2xl bg-brand-600 flex items-center justify-center mx-auto mb-4 shadow-lg shadow-brand-200">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h2 className="text-lg font-semibold text-gray-900 mb-1">Ask about SEC filings</h2>
              <p className="text-sm text-gray-500">
                Every number is verified against XBRL and cited to the source page.
              </p>
            </div>
            <div className="w-full grid grid-cols-1 gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="text-left px-4 py-3 rounded-xl border border-gray-200 hover:border-brand-300 bg-white hover:bg-brand-50 text-sm text-gray-600 hover:text-gray-900 transition-all shadow-sm"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-2xl mx-auto space-y-5">
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-slide-up`}>
                {msg.role === "user" ? (
                  <div className="max-w-[80%] bg-brand-600 rounded-2xl rounded-tr-sm px-4 py-3 shadow-sm">
                    <p className="text-sm text-white">{msg.content}</p>
                  </div>
                ) : (
                  <div className="w-full">
                    {msg.streaming && msg.stage && <StatusRow stage={msg.stage} />}

                    {(msg.content || !msg.streaming) && (
                      <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm">
                        <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
                          {msg.content || " "}
                          {msg.streaming && (
                            <span className="inline-block w-0.5 h-4 bg-brand-500 ml-0.5 animate-blink" />
                          )}
                        </p>

                        {msg.result && msg.result.facts.length > 0 && (
                          <div className="mt-4 pt-4 border-t border-gray-100">
                            <p className="text-[10px] uppercase tracking-widest text-gray-400 font-semibold mb-2">
                              Verified facts
                            </p>
                            <div className="flex flex-col gap-2">
                              {msg.result.facts.map((f, fi) => (
                                <div key={fi} className="flex items-start gap-2 text-xs">
                                  <CitationChip pageRef={f.page_ref} verified={f.verified} factText={f.text} bbox={f.bbox} />
                                  <span className="text-gray-500 leading-relaxed">{f.text}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {msg.result && (
                          <div className="mt-3 flex items-center gap-2 flex-wrap">
                            <AnswerStatusBadge status={msg.result.answer_status} />
                            <span className="text-[10px] text-gray-400">{msg.result.pipeline}</span>
                            {msg.result.latency_ms > 0 && (
                              <span className="text-[10px] text-gray-400">{msg.result.latency_ms}ms</span>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input bar */}
      <div className="shrink-0 border-t border-gray-200 px-4 py-3 bg-white">
        <div className="max-w-2xl mx-auto flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            className="flex-1 bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-800 placeholder-gray-400 outline-none focus:border-brand-400 focus:ring-1 focus:ring-brand-400/30 resize-none transition-colors min-h-[44px] max-h-32"
            placeholder="Ask about revenue, net income, EPS, margins…"
            value={input}
            rows={1}
            onChange={(e) => {
              setInput(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = Math.min(e.target.scrollHeight, 128) + "px";
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }}
            disabled={busy}
          />
          <button
            className="shrink-0 w-10 h-10 bg-brand-600 hover:bg-brand-500 disabled:opacity-40 rounded-xl flex items-center justify-center transition-colors shadow-sm"
            onClick={() => send()}
            disabled={busy || !input.trim()}
          >
            {busy ? (
              <svg className="w-4 h-4 animate-spin text-white" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg>
            ) : (
              <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
              </svg>
            )}
          </button>
        </div>
        <p className="text-[10px] text-gray-400 text-center mt-2">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
