"use client";

import React from "react";
import { X, CheckCircle2, XCircle } from "lucide-react";
import { RankedCandidate, RecruitmentStage } from "@/types";

interface ComparisonModalProps {
  candidates: RankedCandidate[];
  onClose: () => void;
  blindMode?: boolean;
}

const getStageBadgeStyle = (stage: RecruitmentStage = "Applied") => {
  switch (stage) {
    case "Applied":
      return "bg-zinc-100 text-zinc-700 border-zinc-300 dark:bg-zinc-800/80 dark:text-zinc-300 dark:border-zinc-700";
    case "Shortlisted":
      return "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/50 dark:text-blue-300 dark:border-blue-800/60";
    case "Interview":
      return "bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-950/50 dark:text-purple-300 dark:border-purple-800/60";
    case "Selected":
      return "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/50 dark:text-emerald-300 dark:border-emerald-800/60";
    case "Rejected":
      return "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950/50 dark:text-rose-300 dark:border-rose-800/60";
  }
};

export function ComparisonModal({
  candidates,
  onClose,
  blindMode = false,
}: ComparisonModalProps) {
  if (candidates.length === 0) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <div className="relative flex max-h-[90vh] w-full max-w-5xl flex-col rounded-2xl border border-zinc-200 bg-white shadow-2xl overflow-hidden transition-colors dark:border-zinc-800 dark:bg-zinc-950">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-200 bg-zinc-50/80 p-6 transition-colors dark:border-zinc-800/80 dark:bg-zinc-900/40">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              Candidate Head-to-Head Comparison
            </h2>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
              Evaluating {candidates.length} candidates across multi-factor dimensions
            </p>
          </div>

          <button
            onClick={onClose}
            className="rounded-lg p-2 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Comparison Grid */}
        <div className="flex-1 overflow-y-auto p-6">
          <div
            className={`grid gap-4 ${
              candidates.length === 2
                ? "grid-cols-1 sm:grid-cols-2"
                : candidates.length === 3
                ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
                : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4"
            }`}
          >
            {candidates.map((c) => {
              const displayName = blindMode
                ? `Candidate #${c.rank}`
                : c.candidate.name;

              const stage = c.recruitment_status || c.candidate.recruitment_status || "Applied";

              return (
                <div
                  key={c.candidate.id}
                  className="flex flex-col rounded-xl border border-zinc-200 bg-zinc-50/50 p-5 space-y-5 transition-colors dark:border-zinc-800 dark:bg-zinc-900/40"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <span className="text-xs font-mono font-bold text-zinc-400 dark:text-zinc-500">
                        RANK #{c.rank}
                      </span>
                      <h3 className="mt-1 text-base font-bold text-zinc-900 dark:text-zinc-100">
                        {displayName}
                      </h3>
                      <p className="text-xs text-zinc-500 dark:text-zinc-400">
                        {c.candidate.highest_degree || "Degree Not Detected"}
                      </p>

                      <div className="mt-2">
                        <span
                          className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[10px] font-medium ${getStageBadgeStyle(
                            stage
                          )}`}
                        >
                          Stage: {stage}
                        </span>
                      </div>
                    </div>

                    <div
                      className={`flex h-12 w-12 items-center justify-center rounded-xl border font-mono text-base font-bold ${
                        c.score.overall_score >= 65
                          ? "border-emerald-500/30 bg-emerald-50 text-emerald-700 dark:bg-emerald-950/20 dark:text-emerald-400"
                          : c.score.overall_score >= 40
                          ? "border-amber-500/30 bg-amber-50 text-amber-700 dark:bg-amber-950/20 dark:text-amber-400"
                          : "border-rose-500/30 bg-rose-50 text-rose-700 dark:bg-rose-950/20 dark:text-rose-400"
                      }`}
                    >
                      {c.score.overall_score.toFixed(0)}
                    </div>
                  </div>

                  {/* Metrics */}
                  <div className="space-y-2 border-t border-zinc-200 pt-4 text-xs transition-colors dark:border-zinc-800/80">
                    <div className="flex justify-between">
                      <span className="text-zinc-500 dark:text-zinc-400">Experience:</span>
                      <span className="font-semibold text-zinc-800 dark:text-zinc-200">
                        {c.candidate.detected_yoe.toFixed(1)} yrs
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-zinc-500 dark:text-zinc-400">Semantic Fit:</span>
                      <span className="font-mono text-zinc-800 dark:text-zinc-200">
                        {c.score.semantic_score.toFixed(1)}%
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-zinc-500 dark:text-zinc-400">Skill Coverage:</span>
                      <span className="font-mono text-zinc-800 dark:text-zinc-200">
                        {c.score.skill_score.toFixed(1)}%
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-zinc-500 dark:text-zinc-400">Domain Match:</span>
                      <span className="font-mono text-zinc-800 dark:text-zinc-200">
                        {c.score.domain_score.toFixed(1)}%
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-zinc-500 dark:text-zinc-400">Knockout Status:</span>
                      <span className="font-semibold">
                        {c.knockout.is_knocked_out ? (
                          <span className="inline-flex items-center gap-1 text-rose-600 dark:text-rose-400">
                            <XCircle className="h-3 w-3" /> Knocked Out
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
                            <CheckCircle2 className="h-3 w-3" /> Passed
                          </span>
                        )}
                      </span>
                    </div>
                  </div>

                  {/* Matched skills */}
                  <div className="border-t border-zinc-200 pt-4 transition-colors dark:border-zinc-800/80">
                    <span className="text-[11px] font-semibold text-emerald-700 dark:text-emerald-400">
                      Top Matched Skills
                    </span>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {c.evidence.matched_skills.slice(0, 5).map((s) => (
                        <span
                          key={s}
                          className="rounded border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-300"
                        >
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Missing skills */}
                  <div className="border-t border-zinc-200 pt-4 transition-colors dark:border-zinc-800/80">
                    <span className="text-[11px] font-semibold text-rose-700 dark:text-rose-400">
                      Missing Skills
                    </span>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {c.evidence.missing_skills.length > 0 ? (
                        c.evidence.missing_skills.slice(0, 4).map((s) => (
                          <span
                            key={s}
                            className="rounded border border-rose-200 bg-rose-50 px-2 py-0.5 text-[10px] text-rose-800 dark:border-rose-900/50 dark:bg-rose-950/30 dark:text-rose-300"
                          >
                            {s}
                          </span>
                        ))
                      ) : (
                        <span className="text-[11px] text-zinc-400 dark:text-zinc-500">None</span>
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
