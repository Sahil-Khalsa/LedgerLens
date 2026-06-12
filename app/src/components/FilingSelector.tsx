"use client";

import { useEffect, useState } from "react";
import { listFilings, type Filing } from "@/lib/api";

interface Props {
  onSelect: (accn: string | null, label?: string) => void;
  selected: string | null;
}

const DEFAULT_TICKERS = ["NVDA", "MSFT"];

function FormBadge({ form }: { form: string }) {
  const isAmend = form.includes("/A");
  return (
    <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-medium ${
      isAmend ? "bg-amber-100 text-amber-700" : "bg-brand-100 text-brand-700"
    }`}>
      {form}
    </span>
  );
}

export function FilingSelector({ onSelect, selected }: Props) {
  const [ticker, setTicker] = useState("");
  const [filings, setFilings] = useState<Filing[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [autoLoaded, setAutoLoaded] = useState(false);

  useEffect(() => {
    if (autoLoaded) return;
    setAutoLoaded(true);
    setLoading(true);
    Promise.all(DEFAULT_TICKERS.map((t) => listFilings(t)))
      .then((results) => setFilings(results.flatMap((r) => r.items)))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [autoLoaded]);

  async function search() {
    const t = ticker.trim().toUpperCase();
    if (!t) return;
    setLoading(true);
    setError("");
    try {
      const res = await listFilings(t);
      setFilings((prev) => {
        const existing = new Set(prev.map((f) => f.accn));
        return [...prev, ...res.items.filter((f) => !existing.has(f.accn))];
      });
      setTicker("");
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  const grouped: Record<string, Filing[]> = {};
  for (const f of filings) {
    if (!grouped[f.ticker]) grouped[f.ticker] = [];
    grouped[f.ticker].push(f);
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-1.5">
        <input
          className="flex-1 min-w-0 bg-gray-50 border border-gray-200 rounded-lg px-3 py-1.5 text-sm text-gray-800 placeholder-gray-400 outline-none focus:border-brand-400 focus:ring-1 focus:ring-brand-400/30 transition-colors"
          placeholder="Add ticker…"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          onKeyDown={(e) => e.key === "Enter" && search()}
        />
        <button
          className="px-3 py-1.5 bg-brand-600 hover:bg-brand-500 rounded-lg text-xs font-medium text-white disabled:opacity-40 transition-colors shrink-0"
          onClick={search}
          disabled={loading || !ticker.trim()}
        >
          {loading ? (
            <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
          ) : "Add"}
        </button>
      </div>

      {error && <p className="text-red-500 text-xs px-1">{error}</p>}

      {loading && filings.length === 0 && (
        <div className="flex items-center gap-2 px-2 py-3 text-gray-400">
          <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
          </svg>
          <span className="text-xs">Loading filings…</span>
        </div>
      )}

      {Object.entries(grouped).map(([tick, items]) => (
        <div key={tick}>
          <p className="text-[10px] font-bold text-gray-400 px-1 mb-1">{tick}</p>
          <div className="flex flex-col gap-0.5">
            {items.map((f) => {
              const label = `${f.ticker} ${f.form} ${f.filing_date}`;
              const active = selected === f.accn;
              return (
                <button
                  key={f.accn}
                  onClick={() => onSelect(active ? null : f.accn, label)}
                  className={`w-full text-left px-2 py-2 rounded-lg transition-colors group border ${
                    active
                      ? "bg-brand-50 border-brand-200"
                      : "hover:bg-gray-50 border-transparent"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <FormBadge form={f.form} />
                    <span className={`text-xs ${active ? "text-gray-700" : "text-gray-500 group-hover:text-gray-700"}`}>
                      {f.filing_date}
                    </span>
                    <div className="ml-auto flex gap-0.5">
                      {f.is_indexed_text && (
                        <span title="Text indexed" className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                      )}
                      {f.is_indexed_visual && (
                        <span title="Visually indexed" className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                      )}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
