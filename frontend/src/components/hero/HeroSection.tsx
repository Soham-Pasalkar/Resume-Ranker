"use client";

import React from "react";
import { CheckCircle2, ShieldAlert, Layers, Cpu, ArrowRight } from "lucide-react";

interface HeroSectionProps {
  onGetStarted: () => void;
  onLoadSample: () => void;
}

export function HeroSection({ onGetStarted, onLoadSample }: HeroSectionProps) {
  return (
    <section
      style={{ minHeight: "calc(100svh - var(--navbar-height, 4rem))" }}
      className="relative flex flex-col justify-center overflow-hidden border-b border-zinc-200/80 bg-zinc-50/50 px-4 py-12 transition-colors sm:px-6 sm:py-16 lg:px-8 lg:py-8 dark:border-zinc-800/80 dark:bg-zinc-950"
    >
      {/* Subtle architectural background grid */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.03] dark:opacity-[0.03]"
        style={{
          backgroundImage: `linear-gradient(#000 1px, transparent 1px), linear-gradient(90deg, #000 1px, transparent 1px)`,
          backgroundSize: "40px 40px",
        }}
      />

      <div className="relative mx-auto w-full max-w-7xl">
        <div className="grid grid-cols-1 items-center gap-12 lg:grid-cols-12">
          {/* Left Column: Core Value Proposition */}
          <div className="lg:col-span-7">
            <div className="inline-flex items-center gap-2 rounded-full border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/80 px-3 py-1 text-xs text-zinc-600 dark:text-zinc-400 shadow-sm">
              <span className="flex h-2 w-2 rounded-full bg-blue-600 dark:bg-blue-500 animate-pulse" />
              <span className="font-medium">Intelligent Recruitment &amp; Pipeline Engine</span>
            </div>

            <h1 className="mt-6 text-4xl font-extrabold tracking-tight text-zinc-900 dark:text-zinc-100 sm:text-5xl lg:text-6xl leading-[1.12]">
              See Beyond Resumes.{" "}
              <span className="text-zinc-500 dark:text-zinc-400 font-medium">
                Discover the Right People.
              </span>
            </h1>

            <p className="mt-6 text-base leading-relaxed text-zinc-600 dark:text-zinc-400 sm:text-lg max-w-2xl">
              People Lens helps recruiters identify the right candidates by intelligently evaluating their skills, experience, and alignment with job requirements. Make informed hiring decisions with clear, evidence-backed insights.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-4">
              <button
                onClick={onGetStarted}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition-all hover:bg-blue-500 active:scale-95"
              >
                Start Campaign Screening
                <ArrowRight className="h-4 w-4" />
              </button>
              <button
                onClick={onLoadSample}
                className="inline-flex items-center justify-center rounded-lg border border-zinc-300 dark:border-zinc-800 bg-white dark:bg-zinc-900/80 px-5 py-3 text-sm font-medium text-zinc-700 dark:text-zinc-300 shadow-sm transition-all hover:border-zinc-400 dark:hover:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-800/80 hover:text-zinc-950 dark:hover:text-white"
              >
                Load Sample Tech Campaign
              </button>
            </div>

            {/* Feature Pills */}
            <div className="mt-12 grid grid-cols-1 gap-4 border-t border-zinc-200 dark:border-zinc-800/80 pt-8 sm:grid-cols-3">
              <div className="flex items-start gap-3">
                <Layers className="mt-0.5 h-4 w-4 text-blue-600 dark:text-blue-500 shrink-0" />
                <div>
                  <h4 className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                    Hybrid Evaluation
                  </h4>
                  <p className="mt-0.5 text-[11px] text-zinc-500 dark:text-zinc-400">
                    Dual dense semantic matching &amp; exact technical keywords
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <CheckCircle2 className="mt-0.5 h-4 w-4 text-emerald-600 dark:text-emerald-500 shrink-0" />
                <div>
                  <h4 className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                    Evidence-Backed
                  </h4>
                  <p className="mt-0.5 text-[11px] text-zinc-500 dark:text-zinc-400">
                    Direct resume citations verify every competency score
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <ShieldAlert className="mt-0.5 h-4 w-4 text-zinc-500 dark:text-zinc-400 shrink-0" />
                <div>
                  <h4 className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                    Blind Screening
                  </h4>
                  <p className="mt-0.5 text-[11px] text-zinc-500 dark:text-zinc-400">
                    One-click PII masking ensures fair, objective evaluations
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Architectural Live Evidence Widget Preview */}
          <div className="lg:col-span-5">
            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 p-5 shadow-xl dark:shadow-2xl backdrop-blur-sm transition-colors">
              <div className="flex items-center justify-between border-b border-zinc-100 dark:border-zinc-800/80 pb-3">
                <div className="flex items-center gap-2">
                  <div className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
                  <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                    Candidate Evaluation Preview
                  </span>
                </div>
                <span className="rounded bg-zinc-100 dark:bg-zinc-800 px-2 py-0.5 text-[10px] font-mono text-zinc-600 dark:text-zinc-400">
                  RRF: 98.4
                </span>
              </div>

              {/* Candidate Quick Card Preview */}
              <div className="mt-4 rounded-lg border border-zinc-200 dark:border-zinc-800/90 bg-zinc-50 dark:bg-zinc-950/70 p-4 transition-colors">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                      Alex Rivera
                    </h3>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400">
                      Staff Backend Engineer • 7.2 yrs YOE • Master&apos;s Degree
                    </p>
                  </div>
                  <div className="flex h-11 w-11 items-center justify-center rounded-lg border border-emerald-500/30 bg-emerald-50 dark:bg-emerald-950/20 font-mono text-sm font-bold text-emerald-600 dark:text-emerald-400">
                    91
                  </div>
                </div>

                {/* Score breakdown mini-bars */}
                <div className="mt-4 space-y-2 text-xs">
                  <div>
                    <div className="flex justify-between text-[11px] text-zinc-500 dark:text-zinc-400">
                      <span>Semantic Similarity</span>
                      <span className="font-mono text-zinc-700 dark:text-zinc-300">94.0%</span>
                    </div>
                    <div className="mt-1 h-1.5 w-full rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
                      <div className="h-full bg-blue-600 dark:bg-blue-500 rounded-full w-[94%]" />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[11px] text-zinc-500 dark:text-zinc-400">
                      <span>Skill Competency Match</span>
                      <span className="font-mono text-zinc-700 dark:text-zinc-300">92.5%</span>
                    </div>
                    <div className="mt-1 h-1.5 w-full rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
                      <div className="h-full bg-emerald-600 dark:bg-emerald-500 rounded-full w-[92%]" />
                    </div>
                  </div>
                </div>

                {/* Verified Quote Snippet */}
                <div className="mt-4 rounded border border-zinc-200 dark:border-zinc-800/90 bg-white dark:bg-zinc-900/60 p-2.5 transition-colors">
                  <span className="text-[10px] font-semibold tracking-wider uppercase text-zinc-500 dark:text-zinc-400">
                    Verified Resume Citation
                  </span>
                  <p className="mt-1 font-mono text-[11px] leading-relaxed text-zinc-700 dark:text-zinc-300">
                    &ldquo;Architected distributed payment processing gateway in Python and Go processing 15,000 transactions/sec with 99.999% SLA.&rdquo;
                  </p>
                </div>
              </div>

              {/* Bottom Telemetry */}
              <div className="mt-4 flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400">
                <span className="flex items-center gap-1.5">
                  <Cpu className="h-3 w-3 text-zinc-400 dark:text-zinc-500" />
                  People Lens Core Active
                </span>
                <span>Latency: 18ms</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
