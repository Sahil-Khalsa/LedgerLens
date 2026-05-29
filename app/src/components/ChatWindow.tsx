"use client";

import { useRef, useState } from "react";
import { streamQuery, type SSEResultEvent } from "@/lib/api";
import { CitationChip } from "./CitationChip";

interface Message {
  role: "user" | "assistant";
  content: string;
  result?: SSEResultEvent;
  streaming?: boolean;
}

interface Props {
  filingAccn: string | null;
  pipeline: "baseline" | "visual" | "auto";
}

export function ChatWindow({ filingAccn, pipeline }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState("");
  const threadId = useRef<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  async function send() {
    const question = input.trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    setStage("thinking…");

    const userMsg: Message = { role: "user", content: question };
    const assistantMsg: Message = { role: "assistant", content: "", streaming: true };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    await streamQuery(question, filingAccn, pipeline, threadId.current, {
      onStatus: (e) => setStage(e.stage),
      onToken: (text) => {
        setMessages((prev) => {
          const copy = [...prev];
          const last = copy[copy.length - 1];
          if (last?.role === "assistant") {
            copy[copy.length - 1] = { ...last, content: last.content + text };
          }
          return copy;
        });
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      },
      onResult: (result) => {
        if (result.thread_id) threadId.current = result.thread_id;
        const answer = result.answer ?? "(no answer)";
        setMessages((prev) => {
          const copy = [...prev];
          const last = copy[copy.length - 1];
          if (last?.role === "assistant") {
            copy[copy.length - 1] = {
              ...last,
              content: answer,
              result,
              streaming: false,
            };
          }
          return copy;
        });
      },
      onError: (msg) => {
        setMessages((prev) => {
          const copy = [...prev];
          const last = copy[copy.length - 1];
          if (last?.role === "assistant") {
            copy[copy.length - 1] = { ...last, content: `Error: ${msg}`, streaming: false };
          }
          return copy;
        });
      },
      onDone: () => {
        setBusy(false);
        setStage("");
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      },
    });
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && (
          <p className="text-gray-500 text-sm text-center mt-16">
            Ask a question about the selected filing, or search across all indexed filings.
          </p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-2xl rounded-lg px-4 py-3 text-sm ${
                msg.role === "user"
                  ? "bg-brand-500/20 text-gray-100"
                  : "bg-gray-800 text-gray-100"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.streaming && <span className="inline-block w-1.5 h-4 bg-brand-400 animate-pulse ml-0.5" />}

              {/* Facts + citations */}
              {msg.result && msg.result.facts.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-700">
                  <p className="text-xs text-gray-500 mb-1.5">Verified facts</p>
                  <div className="flex flex-wrap gap-1.5">
                    {msg.result.facts.map((f, fi) => (
                      <div key={fi} className="flex items-center gap-1 text-xs">
                        <span className="text-gray-300">{f.text}</span>
                        <CitationChip pageRef={f.page_ref} verified={f.verified} />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Status badge */}
              {msg.result && (
                <div className="mt-2 flex gap-2 text-xs text-gray-500">
                  <span>{msg.result.pipeline}</span>
                  <span>·</span>
                  <span
                    className={
                      msg.result.answer_status === "answered"
                        ? "text-green-400"
                        : msg.result.answer_status === "low_confidence"
                          ? "text-yellow-400"
                          : "text-red-400"
                    }
                  >
                    {msg.result.answer_status}
                  </span>
                  {msg.result.cost_usd > 0 && (
                    <>
                      <span>·</span>
                      <span>${msg.result.cost_usd.toFixed(4)}</span>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Stage indicator */}
      {stage && (
        <div className="px-4 py-1 text-xs text-gray-500 border-t border-gray-800">
          {stage}…
        </div>
      )}

      {/* Input */}
      <div className="px-4 py-3 border-t border-gray-800 flex gap-2">
        <input
          className="flex-1 bg-gray-800 rounded-lg px-4 py-2 text-sm outline-none focus:ring-1 focus:ring-brand-500 disabled:opacity-50"
          placeholder="Ask about revenue, net income, EPS…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
          disabled={busy}
        />
        <button
          className="px-4 py-2 bg-brand-500 hover:bg-brand-600 rounded-lg text-sm font-medium disabled:opacity-50"
          onClick={send}
          disabled={busy || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}
