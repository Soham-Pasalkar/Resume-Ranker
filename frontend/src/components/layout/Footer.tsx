"use client";

import React from "react";
import { Cpu, ArrowUpRight } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-zinc-200 bg-zinc-50 px-4 py-12 text-xs text-zinc-500 transition-colors sm:px-6 lg:px-8 dark:border-zinc-800/80 dark:bg-zinc-950 dark:text-zinc-400">
      <div className="mx-auto flex max-w-7xl flex-col justify-between gap-6 sm:flex-row sm:items-center">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded border border-blue-200 bg-blue-50 text-blue-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-blue-400">
            <Cpu className="h-3.5 w-3.5" />
          </div>
          <span className="font-semibold text-zinc-800 dark:text-zinc-200">People Lens</span>
          <span>• Grounded Candidate Intelligence Engine</span>
        </div>

        <div className="flex flex-wrap items-center gap-6">
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 transition-colors hover:text-zinc-800 dark:hover:text-zinc-200"
          >
            FastAPI Documentation
            <ArrowUpRight className="h-3 w-3" />
          </a>
          <span>Zero PII Retention Option</span>
          <span>Dual Vector &amp; BM25 RRF Engine</span>
        </div>
      </div>
    </footer>
  );
}
