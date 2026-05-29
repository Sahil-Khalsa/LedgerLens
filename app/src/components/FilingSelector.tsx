"use client";

import { useEffect, useState } from "react";
import { listFilings, type Filing } from "@/lib/api";

interface Props {
  onSelect: (accn: string | null) => void;
  selected: string | null;
}

export function FilingSelector({ onSelect, selected }: Props) {
  const [ticker, setTicker] = useState("");
  const [filings, setFilings] = useState<Filing[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function search() {
    if (!ticker.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await listFilings(ticker.trim().toUpperCase());
      setFilings(res.items);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-2">
        <input
          className="flex-1 bg-gray-800 rounded px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-brand-500"
          placeholder="Ticker (e.g. NVDA)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && search()}
        />
        <button
          className="px-3 py-1.5 bg-brand-500 hover:bg-brand-600 rounded text-sm font-medium disabled:opacity-50"
          onClick={search}
          disabled={loading}
        >
          {loading ? "..." : "Search"}
        </button>
      </div>

      {error && <p className="text-red-400 text-xs">{error}</p>}

      {filings.length > 0 && (
        <div className="flex flex-col gap-1 max-h-48 overflow-y-auto">
          <button
            className={`text-left px-2 py-1 rounded text-sm ${selected === null ? "bg-brand-500/30 text-brand-300" : "hover:bg-gray-800"}`}
            onClick={() => onSelect(null)}
          >
            All filings
          </button>
          {filings.map((f) => (
            <button
              key={f.accn}
              className={`text-left px-2 py-1 rounded text-sm ${selected === f.accn ? "bg-brand-500/30 text-brand-300" : "hover:bg-gray-800"}`}
              onClick={() => onSelect(f.accn)}
            >
              <span className="font-mono text-xs text-gray-400">{f.form}</span>{" "}
              {f.filing_date}
              {f.is_indexed_text && (
                <span className="ml-1 text-xs text-green-500" title="Text indexed">
                  T
                </span>
              )}
              {f.is_indexed_visual && (
                <span className="ml-1 text-xs text-blue-400" title="Visually indexed">
                  V
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
