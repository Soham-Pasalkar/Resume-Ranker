"use client";

import React from "react";
import { X, CheckCircle2, XCircle } from "lucide-react";
import { RankedCandidate } from "@/types";

interface ComparisonModalProps {
  candidates: RankedCandidate[];
  onClose: () => void;
  blindMode?: boolean;
}

export function ComparisonModal({
  candidates,
  onClose,
  blindMode = false,
}: ComparisonModalProps) {
  if (candidates.length === 0) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm">
      <div className="relative flex max-h-[90vh] w-full max-w-5xl flex-col rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800/80 bg-zinc-900/40 p-6">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-zinc-100">
              Candidate Head-to-Head Comparison
            </h2>
            <p className="mt-1 text-xs text-zinc-400">
              Evaluating {candidates.length} candidates across multi-factor dimensions
            </p>
          </div>

          <button
            onClick={onClose}
            className="rounded-lg p-2 text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-100"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Comparison Grid */}
        <div className="flex-1 overflow-y-auto p-6">
          <div
            className={`grid gap-4 ${
              candidates.length === 2
                ? "grid-cols-2"
                : candidates.length === 3
                ? "grid-cols-3"
                : "grid-cols-4"
            }`}
          >
            {candidates.map((c) => {
              const displayName = blindMode
                ? `Candidate #${c.rank}`
                : c.candidate.name;

              return (
                <div
                  key={c.candidate.id}
                  className="flex flex-col rounded-xl border border-zinc-800 bg-zinc-900/40 p-5 space-y-5"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <span className="text-xs font-mono font-bold text-zinc-500">
                        RANK #{c.rank}
                      </span>
                      <h3 className="mt-1 text-base font-bold text-zinc-100">
                        {displayName}
                      </h3>
                      <p className="text-xs text-zinc-400">
                        {c.candidate.highest_degree || "Degree Not Detected"}
                      </p>
                    </div>

                    <div
                      className={`flex h-12 w-12 items-center justify-center rounded-xl border font-mono text-base font-bold ${
                        c.score.overall_score >= 65
                          ? "border-emerald-500/30 bg-emerald-950/20 text-emerald-400"
                          : c.score.overall_score >= 40
                          ? "border-amber-500/30 bg-amber-950/20 text-amber-400"
                          : "border-rose-500/30 bg-rose-950/20 text-rose-400"
                      }`}
                    >
                      {c.score.overall_score.toFixed(0)}
                    </div>
                  </div>

                  {/* Metrics */}
                  <div className="space-y-2 border-t border-zinc-800/80 pt-4 text-xs">
                    <div className="flex justify-between">
                      <span className="text-zinc-400">Experience:</span>
                      <span className="font-semibold text-zinc-200">
                        {c.candidate.detected_yoe.toFixed(1)} yrs
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-zinc-400">Semantic Fit:</span>
                      <span className="font-mono text-zinc-200">
                        {c.score.semantic_score.toFixed(1)}%
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-zinc-400">Skill Coverage:</span>
                      <span className="font-mono text-zinc-200">
                        {c.score.skill_score.toFixed(1)}%
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-zinc-400">Domain Match:</span>
                      <span className="font-mono text-zinc-200">
                        {c.score.domain_score.toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Matched skills */}
                  <div className="border-t border-zinc-800/80 pt-4">
                    <span className="text-[11px] font-semibold text-emerald-400">
                      Top Matched Skills
                    </span>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {c.evidence.matched_skills.slice(0, 5).map((s) => (
                        <span
                          key={s}
                          className="rounded bg-emerald-950/30 border border-emerald-900/50 px-2 py-0.5 text-[10px] text-emerald-300"
                        >
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Missing skills */}
                  <div className="border-t border-zinc-800/80 pt-4">
                    <span className="text-[11px] font-semibold text-rose-400">
                      Missing Skills
                    </span>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {c.evidence.missing_skills.length > 0 ? (
                        c.evidence.missing_skills.slice(0, 4).map((s) => (
                          <span
                            key={s}
                            className="rounded bg-rose-950/30 border border-rose-900/50 px-2 py-0.5 text-[10px] text-rose-300"
                          >
                            {s}
                          </span>
                        ))
                      ) : (
                        <span className="text-[11px] text-zinc-500">None</span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
