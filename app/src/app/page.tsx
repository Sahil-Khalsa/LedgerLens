"use client";

import { useState } from "react";
import { ChatWindow } from "@/components/ChatWindow";
import { FilingSelector } from "@/components/FilingSelector";

type Pipeline = "baseline" | "visual" | "auto";

export default function Home() {
  const [selectedAccn, setSelectedAccn] = useState<string | null>(null);
  const [pipeline, setPipeline] = useState<Pipeline>("auto");

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-72 bg-gray-900 border-r border-gray-800 flex flex-col p-4 gap-4 overflow-y-auto">
        <div>
          <h1 className="text-lg font-semibold text-white tracking-tight">LedgerLens</h1>
          <p className="text-xs text-gray-500 mt-0.5">Financial filing intelligence</p>
        </div>

        <div>
          <label className="block text-xs text-gray-400 mb-1.5 font-medium uppercase tracking-wider">
            Filing
          </label>
          <FilingSelector onSelect={setSelectedAccn} selected={selectedAccn} />
        </div>

        <div>
          <label className="block text-xs text-gray-400 mb-1.5 font-medium uppercase tracking-wider">
            Pipeline
          </label>
          <div className="flex flex-col gap-1">
            {(["auto", "baseline", "visual"] as Pipeline[]).map((p) => (
              <button
                key={p}
                onClick={() => setPipeline(p)}
                className={`text-left px-2 py-1.5 rounded text-sm capitalize ${
                  pipeline === p
                    ? "bg-brand-500/30 text-brand-300"
                    : "text-gray-400 hover:bg-gray-800"
                }`}
              >
                {p}
                {p === "auto" && (
                  <span className="ml-1 text-xs text-gray-600">
                    (XBRL fast path or visual)
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        {selectedAccn && (
          <div className="mt-auto text-xs text-gray-600 break-all">
            <p className="text-gray-500 mb-0.5">Scoped to</p>
            {selectedAccn}
          </div>
        )}
      </aside>

      {/* Chat */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <ChatWindow filingAccn={selectedAccn} pipeline={pipeline} />
      </main>
    </div>
  );
}
