"use client";

import { useState } from "react";
import { pageImageUrl } from "@/lib/api";

interface Props {
  pageRef: string; // "accn:page_idx" or "accn:xbrl"
  verified: string;
}

const verifiedColors: Record<string, string> = {
  match: "bg-green-600/20 text-green-400 border-green-700",
  mismatch: "bg-red-600/20 text-red-400 border-red-700",
  unverifiable: "bg-yellow-600/20 text-yellow-400 border-yellow-700",
  pending: "bg-gray-600/20 text-gray-400 border-gray-600",
};

export function CitationChip({ pageRef, verified }: Props) {
  const [open, setOpen] = useState(false);
  const [accn, rawIdx] = pageRef.split(":");
  const isXbrl = rawIdx === "xbrl";
  const pageIdx = isXbrl ? null : parseInt(rawIdx, 10);
  const color = verifiedColors[verified] ?? verifiedColors.pending;

  return (
    <>
      <button
        onClick={() => !isXbrl && setOpen(true)}
        className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-xs font-mono ${color} ${!isXbrl ? "cursor-pointer hover:opacity-80" : "cursor-default"}`}
        title={isXbrl ? "Verified against XBRL" : `Page ${pageIdx} — click to view`}
      >
        {isXbrl ? "XBRL" : `p${pageIdx}`}
        <span className="opacity-60">·{verified[0].toUpperCase()}</span>
      </button>

      {open && pageIdx !== null && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
          onClick={() => setOpen(false)}
        >
          <div
            className="relative bg-gray-900 rounded-lg shadow-2xl max-w-3xl w-full mx-4 p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-3">
              <span className="text-sm text-gray-400 font-mono">
                {accn} · page {pageIdx}
              </span>
              <button
                onClick={() => setOpen(false)}
                className="text-gray-500 hover:text-gray-300 text-xl leading-none"
              >
                ×
              </button>
            </div>
            <img
              src={pageImageUrl(accn, pageIdx)}
              alt={`Filing page ${pageIdx}`}
              className="w-full rounded border border-gray-700"
            />
          </div>
        </div>
      )}
    </>
  );
}
