"use client";

import { useState } from "react";
import { pageImageUrl } from "@/lib/api";

interface Props {
  pageRef: string;
  verified: string;
}

const STYLES: Record<string, { chip: string; dot: string }> = {
  match:        { chip: "bg-emerald-50  text-emerald-700 border-emerald-200", dot: "bg-emerald-500" },
  unverifiable: { chip: "bg-amber-50   text-amber-700  border-amber-200",    dot: "bg-amber-400"   },
  mismatch:     { chip: "bg-red-50     text-red-600    border-red-200",       dot: "bg-red-500"     },
  pending:      { chip: "bg-gray-100   text-gray-500   border-gray-200",      dot: "bg-gray-400"    },
};

export function CitationChip({ pageRef, verified }: Props) {
  const [open, setOpen] = useState(false);
  const [accn, rawIdx] = pageRef.split(":");
  const isXbrl = rawIdx === "xbrl";
  const pageIdx = isXbrl ? null : parseInt(rawIdx, 10);
  const style = STYLES[verified] ?? STYLES.pending;

  return (
    <>
      <button
        onClick={() => !isXbrl && pageIdx !== null && setOpen(true)}
        className={`inline-flex items-center gap-1 shrink-0 px-1.5 py-0.5 rounded-md border text-[10px] font-mono font-medium transition-opacity ${style.chip} ${!isXbrl ? "cursor-pointer hover:opacity-75" : "cursor-default"}`}
        title={isXbrl ? "Verified against XBRL structured data" : `Page ${pageIdx} — click to view source`}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
        {isXbrl ? "XBRL" : `p.${pageIdx}`}
      </button>

      {open && pageIdx !== null && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fade-in"
          onClick={() => setOpen(false)}
        >
          <div
            className="relative bg-white border border-gray-200 rounded-2xl shadow-2xl max-w-3xl w-full mx-4 p-5 animate-slide-up"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-4">
              <div>
                <p className="text-xs text-gray-400 font-mono">{accn}</p>
                <p className="text-sm font-medium text-gray-800 mt-0.5">Page {pageIdx}</p>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="w-8 h-8 rounded-lg bg-gray-100 hover:bg-gray-200 flex items-center justify-center text-gray-500 hover:text-gray-700 transition-colors"
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <img
              src={pageImageUrl(accn, pageIdx)}
              alt={`Filing page ${pageIdx}`}
              className="w-full rounded-lg border border-gray-200"
            />
          </div>
        </div>
      )}
    </>
  );
}
