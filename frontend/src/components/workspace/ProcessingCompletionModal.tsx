"use client";

import React from "react";
import {
  CheckCircle2,
  Sparkles,
  LayoutList,
  Kanban,
  X,
  Award,
  Users,
  AlertTriangle,
  ArrowRight,
} from "lucide-react";
import { RankingResponse, RankedCandidate } from "@/types";

interface ProcessingCompletionModalProps {
  isOpen: boolean;
  onClose: () => void;
  rankingData: RankingResponse | null;
  blindMode?: boolean;
  onViewLeaderboard: () => void;
  onViewPipeline: () => void;
  processedCount?: number;
}

export function ProcessingCompletionModal({
  isOpen,
  onClose,
  rankingData,
  blindMode = false,
  onViewLeaderboard,
  onViewPipeline,
  processedCount,
}: ProcessingCompletionModalProps) {
  if (!isOpen || !rankingData) return null;

  const { summary, results } = rankingData;
  const topCandidate: RankedCandidate | undefined = results[0];
  const count = processedCount && processedCount > 0 ? processedCount : summary.total_candidates;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-lg rounded-2xl border border-zinc-200 bg-white p-6 shadow-2xl transition-all dark:border-zinc-800 dark:bg-zinc-950">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute right-4 top-4 rounded-lg p-1.5 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
          title="Close confirmation"
        >
          <X className="h-4 w-4" />
        </button>

        {/* Header with Icon */}
        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-emerald-100 text-emerald-600 dark:bg-emerald-950/80 dark:text-emerald-400 shadow-sm">
            <CheckCircle2 className="h-6 w-6" />
          </div>

          <div>
            <div className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-400">
              <Sparkles className="h-3 w-3" />
              Evaluation Complete
            </div>
            <h3 className="mt-1 text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
              Resumes Processed Successfully
            </h3>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
              Evaluated {count} candidate {count === 1 ? "resume" : "resumes"} using hybrid semantic and BM25 evidence scoring.
            </p>
          </div>
        </div>

        {/* Quick Results Summary Grid */}
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-xl border border-zinc-200 bg-zinc-50/70 p-3 text-center dark:border-zinc-800 dark:bg-zinc-900/50">
            <span className="text-[10px] font-medium text-zinc-500 dark:text-zinc-400">
              Total Evaluated
            </span>
            <p className="mt-1 font-mono text-xl font-bold text-zinc-900 dark:text-zinc-100">
              {summary.total_candidates}
            </p>
          </div>

          <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-3 text-center dark:border-emerald-900/40 dark:bg-emerald-950/30">
            <span className="text-[10px] font-medium text-emerald-700 dark:text-emerald-400">
              Strong Matches
            </span>
            <p className="mt-1 font-mono text-xl font-bold text-emerald-700 dark:text-emerald-400">
              {summary.strong_matches}
            </p>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-zinc-50/70 p-3 text-center dark:border-zinc-800 dark:bg-zinc-900/50">
            <span className="text-[10px] font-medium text-zinc-500 dark:text-zinc-400">
              Average Score
            </span>
            <p className="mt-1 font-mono text-xl font-bold text-zinc-900 dark:text-zinc-100">
              {summary.average_score.toFixed(0)}%
            </p>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-zinc-50/70 p-3 text-center dark:border-zinc-800 dark:bg-zinc-900/50">
            <span className="text-[10px] font-medium text-zinc-500 dark:text-zinc-400">
              Knockouts
            </span>
            <p className="mt-1 font-mono text-xl font-bold text-rose-600 dark:text-rose-400">
              {summary.knockouts_count}
            </p>
          </div>
        </div>

        {/* Top Candidate Spotlight */}
        {topCandidate && (
          <div className="mt-4 rounded-xl border border-blue-200/80 bg-blue-50/40 p-4 transition-colors dark:border-blue-900/40 dark:bg-blue-950/20">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-blue-700 dark:text-blue-400">
                <Award className="h-3.5 w-3.5" />
                Top Recommended Candidate
              </span>
              <span className="rounded bg-blue-100 px-2 py-0.5 font-mono text-xs font-bold text-blue-800 dark:bg-blue-900/60 dark:text-blue-300">
                Score: {topCandidate.score.overall_score.toFixed(0)}/100
              </span>
            </div>

            <div className="mt-2.5 flex items-start justify-between">
              <div>
                <h4 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                  {blindMode
                    ? `Candidate #${topCandidate.rank} [ANONYMIZED]`
                    : topCandidate.candidate.name}
                </h4>
                <p className="text-xs text-zinc-600 dark:text-zinc-400">
                  {topCandidate.candidate.detected_yoe.toFixed(1)} yrs experience •{" "}
                  {topCandidate.candidate.highest_degree || "Degree Verified"}
                </p>
              </div>

              <span className="rounded-md border border-emerald-300 bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
                {topCandidate.score.match_level}
              </span>
            </div>

            {/* Matched skills chips */}
            {topCandidate.evidence.matched_skills.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1">
                {topCandidate.evidence.matched_skills.slice(0, 4).map((skill) => (
                  <span
                    key={skill}
                    className="rounded border border-blue-200 bg-white px-2 py-0.5 text-[10px] text-blue-800 font-mono dark:border-blue-800/60 dark:bg-zinc-900 dark:text-blue-300"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Action Buttons */}
        <div className="mt-6 flex flex-col gap-2.5 sm:flex-row">
          <button
            onClick={onViewLeaderboard}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-blue-600 py-3 text-xs font-semibold text-white shadow-sm transition-all hover:bg-blue-500 active:scale-95"
          >
            <LayoutList className="h-4 w-4" />
            View Leaderboard
          </button>

          <button
            onClick={onViewPipeline}
            className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-zinc-200 bg-zinc-50 py-3 text-xs font-semibold text-zinc-700 shadow-sm transition-all hover:bg-zinc-100 active:scale-95 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
          >
            <Kanban className="h-4 w-4" />
            Open Pipeline Kanban
          </button>
        </div>

        <div className="mt-3 text-center">
          <button
            onClick={onClose}
            className="text-[11px] font-medium text-zinc-500 hover:text-zinc-800 transition-colors dark:text-zinc-400 dark:hover:text-zinc-200"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
