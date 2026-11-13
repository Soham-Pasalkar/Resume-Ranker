"use client";

import React from "react";
import { Cpu, Database, Network, ShieldCheck, Zap } from "lucide-react";

export function ArchitectureSection() {
  return (
    <section
      id="architecture"
      className="border-t border-zinc-200 bg-zinc-50 px-4 py-20 transition-colors sm:px-6 lg:px-8 dark:border-zinc-800 dark:bg-zinc-950"
    >
      <div className="mx-auto max-w-7xl">
        <div className="max-w-2xl">
          <span className="font-mono text-xs font-semibold uppercase tracking-wider text-blue-600 dark:text-blue-400">
            System Architecture
          </span>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-zinc-900 sm:text-4xl dark:text-zinc-100">
            Engineered for Precision, Grounding &amp; Fairness.
          </h2>
          <p className="mt-4 text-sm leading-relaxed text-zinc-600 sm:text-base dark:text-zinc-400">
            Modern recruitment requires both contextual semantic comprehension and rigorous keyword accuracy. People Lens eliminates hallucinations by grounding every evaluation in direct textual proof from candidate resumes.
          </p>
        </div>

        <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-zinc-200 bg-white p-6 space-y-3 shadow-sm transition-colors dark:border-zinc-800 dark:bg-zinc-900/30">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-blue-200 bg-blue-50 text-blue-600 dark:border-zinc-700/60 dark:bg-zinc-900 dark:text-blue-400">
              <Network className="h-5 w-5" />
            </div>
            <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">
              Dual Vector + BM25 Fusion
            </h3>
            <p className="text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">
              Combines dense embedding vector similarity with sparse Okapi BM25 lexical indexing via Reciprocal Rank Fusion ($k=60$).
            </p>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-white p-6 space-y-3 shadow-sm transition-colors dark:border-zinc-800 dark:bg-zinc-900/30">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-emerald-200 bg-emerald-50 text-emerald-600 dark:border-zinc-700/60 dark:bg-zinc-900 dark:text-emerald-400">
              <Cpu className="h-5 w-5" />
            </div>
            <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">
              Section-Weighted Analysis
            </h3>
            <p className="text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">
              Gives 2x weight to hands-on responsibilities in Work Experience sections over passive skill keyword lists.
            </p>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-white p-6 space-y-3 shadow-sm transition-colors dark:border-zinc-800 dark:bg-zinc-900/30">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-purple-200 bg-purple-50 text-purple-600 dark:border-zinc-700/60 dark:bg-zinc-900 dark:text-purple-400">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">
              Hard Knockout Constraints
            </h3>
            <p className="text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">
              Enforces non-negotiable criteria (mandatory skills, minimum YOE, degree requirements) with automated demotion.
            </p>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-white p-6 space-y-3 shadow-sm transition-colors dark:border-zinc-800 dark:bg-zinc-900/30">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-amber-200 bg-amber-50 text-amber-600 dark:border-zinc-700/60 dark:bg-zinc-900 dark:text-amber-400">
              <Zap className="h-5 w-5" />
            </div>
            <h3 className="text-base font-bold text-zinc-900 dark:text-zinc-100">
              Interview Screening Prep
            </h3>
            <p className="text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">
              Synthesizes candidate-tailored technical screening questions targeted specifically at identified missing requirements.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
