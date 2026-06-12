"use client";

import { useState } from "react";
import { ChatWindow } from "@/components/ChatWindow";
import { FilingSelector } from "@/components/FilingSelector";

type Pipeline = "baseline" | "visual" | "auto";

const PIPELINES: { id: Pipeline; label: string; desc: string }[] = [
  { id: "auto",     label: "Auto",     desc: "XBRL fast path or visual" },
  { id: "baseline", label: "Baseline", desc: "Text retrieval only"       },
  { id: "visual",   label: "Visual",   desc: "ColPali + XBRL verify"     },
];

export default function Home() {
  const [selectedAccn, setSelectedAccn] = useState<string | null>(null);
  const [selectedLabel, setSelectedLabel] = useState<string>("");
  const [pipeline, setPipeline] = useState<Pipeline>("auto");

  function handleSelect(accn: string | null, label?: string) {
    setSelectedAccn(accn);
    setSelectedLabel(label ?? "");
  }

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      <aside className="w-64 shrink-0 bg-white border-r border-gray-200 flex flex-col overflow-hidden">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-gray-100">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-brand-600 flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-900 tracking-tight">LedgerLens</p>
              <p className="text-[10px] text-gray-400 leading-none mt-0.5">SEC Filing Intelligence</p>
            </div>
          </div>
        </div>

        {/* Filings */}
        <div className="flex-1 overflow-y-auto px-3 py-4">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400 px-2 mb-3">
            Filings
          </p>
          <FilingSelector onSelect={handleSelect} selected={selectedAccn} />
        </div>

        {/* Pipeline */}
        <div className="px-3 py-4 border-t border-gray-100">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400 px-2 mb-2">
            Pipeline
          </p>
          <div className="flex flex-col gap-0.5">
            {PIPELINES.map((p) => (
              <button
                key={p.id}
                onClick={() => setPipeline(p.id)}
                className={`flex items-center gap-2.5 px-2 py-2 rounded-lg text-left transition-colors ${
                  pipeline === p.id
                    ? "bg-brand-50 text-brand-700"
                    : "text-gray-500 hover:bg-gray-50 hover:text-gray-800"
                }`}
              >
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${pipeline === p.id ? "bg-brand-500" : "bg-gray-300"}`} />
                <span className="text-sm font-medium">{p.label}</span>
              </button>
            ))}
          </div>
        </div>
      </aside>

      {/* ── Main ────────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="h-12 shrink-0 bg-white border-b border-gray-200 flex items-center px-5 gap-3">
          {selectedAccn ? (
            <>
              <span className="text-xs text-gray-400">Scoped to</span>
              <span className="text-xs font-mono bg-gray-100 text-gray-600 px-2 py-0.5 rounded">
                {selectedLabel || selectedAccn}
              </span>
              <button
                onClick={() => handleSelect(null)}
                className="text-xs text-gray-400 hover:text-gray-700 transition-colors"
              >
                × clear
              </button>
            </>
          ) : (
            <span className="text-xs text-gray-400">Searching all indexed filings</span>
          )}
          <div className="ml-auto flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            <span className="text-[10px] text-gray-400">{pipeline}</span>
          </div>
        </header>

        <ChatWindow filingAccn={selectedAccn} pipeline={pipeline} />
      </main>
    </div>
  );
}
