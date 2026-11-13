"use client";

import React, { useState } from "react";
import {
  X,
  GraduationCap,
  Briefcase,
  MapPin,
  CheckCircle2,
  XCircle,
  HelpCircle,
  ShieldCheck,
  Quote,
  Check,
} from "lucide-react";
import { RankedCandidate, RecruitmentStage } from "@/types";

const STAGES: RecruitmentStage[] = [
  "Applied",
  "Shortlisted",
  "Interview",
  "Selected",
  "Rejected",
];

interface CandidateModalProps {
  candidate: RankedCandidate | null;
  onClose: () => void;
  blindMode?: boolean;
  onUpdateStatus?: (candidateId: string, newStatus: RecruitmentStage) => void;
}

export function CandidateModal({
  candidate,
  onClose,
  blindMode = false,
  onUpdateStatus,
}: CandidateModalProps) {
  const [activeTab, setActiveTab] = useState<"insights" | "sections">("insights");
  const [statusToast, setStatusToast] = useState<string | null>(null);

  if (!candidate) return null;

  const { rank, candidate: cand, score, evidence, knockout } = candidate;
  const currentStatus = candidate.recruitment_status || cand.recruitment_status || "Applied";

  const displayName = blindMode
    ? `Candidate #${rank} [ANONYMIZED]`
    : cand.name;

  const handleStatusChange = (newStatus: RecruitmentStage) => {
    if (onUpdateStatus) {
      onUpdateStatus(cand.id, newStatus);
      setStatusToast(`Status updated to ${newStatus}`);
      setTimeout(() => setStatusToast(null), 2500);
    }
  };

  const getStatusBadge = (status: RecruitmentStage) => {
    switch (status) {
      case "Selected":
        return "bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-950/60 dark:text-emerald-300 dark:border-emerald-800/60";
      case "Interview":
        return "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-950/60 dark:text-amber-300 dark:border-amber-800/60";
      case "Shortlisted":
        return "bg-purple-100 text-purple-800 border-purple-300 dark:bg-purple-950/60 dark:text-purple-300 dark:border-purple-800/60";
      case "Rejected":
        return "bg-rose-100 text-rose-800 border-rose-300 dark:bg-rose-950/60 dark:text-rose-300 dark:border-rose-800/60";
      default:
        return "bg-blue-100 text-blue-800 border-blue-300 dark:bg-blue-950/60 dark:text-blue-300 dark:border-blue-800/60";
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 dark:bg-black/80 p-4 backdrop-blur-sm">
      <div className="relative flex max-h-[90vh] w-full max-w-4xl flex-col rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 shadow-2xl overflow-hidden transition-colors">
        {/* Toast confirmation */}
        {statusToast && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 rounded-lg bg-zinc-900 dark:bg-zinc-100 px-4 py-2 text-xs font-medium text-white dark:text-zinc-900 shadow-lg animate-in fade-in">
            <Check className="h-4 w-4 text-emerald-400 dark:text-emerald-600" />
            {statusToast}
          </div>
        )}

        {/* Header */}
        <div className="flex flex-col sm:flex-row items-start justify-between border-b border-zinc-200 dark:border-zinc-800/80 bg-zinc-50 dark:bg-zinc-900/40 p-6 gap-4">
          <div className="flex items-start gap-4">
            <div
              className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-xl border font-mono text-xl font-bold ${
                score.overall_score >= 65
                  ? "border-emerald-500/30 bg-emerald-50 text-emerald-700 dark:bg-emerald-950/20 dark:text-emerald-400"
                  : score.overall_score >= 40
                  ? "border-amber-500/30 bg-amber-50 text-amber-700 dark:bg-amber-950/20 dark:text-amber-400"
                  : "border-rose-500/30 bg-rose-50 text-rose-700 dark:bg-rose-950/20 dark:text-rose-400"
              }`}
            >
              {score.overall_score.toFixed(0)}
            </div>

            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-mono font-semibold text-zinc-500 dark:text-zinc-400">
                  RANK #{rank}
                </span>

                {/* Match Level */}
                <span
                  className={`rounded px-2 py-0.5 text-[11px] font-medium border ${
                    knockout.is_knocked_out
                      ? "bg-rose-100 text-rose-800 border-rose-300 dark:bg-rose-950/60 dark:text-rose-400 dark:border-rose-800/60"
                      : score.overall_score >= 65
                      ? "bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-950/60 dark:text-emerald-400 dark:border-emerald-800/60"
                      : "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-950/60 dark:text-amber-400 dark:border-amber-800/60"
                  }`}
                >
                  {score.match_level}
                </span>

                {blindMode && (
                  <span className="flex items-center gap-1 rounded bg-blue-100 dark:bg-blue-950/40 border border-blue-300 dark:border-blue-800/50 px-2 py-0.5 text-[11px] text-blue-700 dark:text-blue-400">
                    <ShieldCheck className="h-3 w-3" /> Blind Mode
                  </span>
                )}
              </div>

              <h2 className="mt-1 text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
                {displayName}
              </h2>

              <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-zinc-600 dark:text-zinc-400">
                <span className="flex items-center gap-1">
                  <Briefcase className="h-3.5 w-3.5 text-zinc-400 dark:text-zinc-500" />
                  {cand.detected_yoe.toFixed(1)} yrs experience
                </span>

                {cand.highest_degree && (
                  <span className="flex items-center gap-1">
                    <GraduationCap className="h-3.5 w-3.5 text-zinc-400 dark:text-zinc-500" />
                    {cand.highest_degree}
                    {!blindMode && cand.institution && ` • ${cand.institution}`}
                  </span>
                )}

                {!blindMode && cand.contact.location && (
                  <span className="flex items-center gap-1">
                    <MapPin className="h-3.5 w-3.5 text-zinc-400 dark:text-zinc-500" />
                    {cand.contact.location}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Right: Pipeline Status Selector & Close */}
          <div className="flex items-center gap-3 w-full sm:w-auto justify-between sm:justify-end">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                Pipeline Stage:
              </span>
              <select
                value={currentStatus}
                onChange={(e) => handleStatusChange(e.target.value as RecruitmentStage)}
                className={`rounded-lg border px-2.5 py-1.5 text-xs font-semibold focus:outline-none focus:ring-1 focus:ring-blue-500 ${getStatusBadge(
                  currentStatus as RecruitmentStage
                )}`}
              >
                {STAGES.map((stg) => (
                  <option key={stg} value={stg} className="bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100">
                    {stg}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={onClose}
              className="rounded-lg p-2 text-zinc-400 transition-colors hover:bg-zinc-200 dark:hover:bg-zinc-800 hover:text-zinc-900 dark:hover:text-zinc-100"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Tab Bar */}
        <div className="flex border-b border-zinc-200 dark:border-zinc-800/80 bg-zinc-100/50 dark:bg-zinc-900/20 px-6">
          <button
            onClick={() => setActiveTab("insights")}
            className={`border-b-2 py-3 text-xs font-medium transition-colors ${
              activeTab === "insights"
                ? "border-blue-600 dark:border-blue-500 text-blue-600 dark:text-blue-400"
                : "border-transparent text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            Match Insights &amp; Evidence
          </button>
          <button
            onClick={() => setActiveTab("sections")}
            className={`ml-6 border-b-2 py-3 text-xs font-medium transition-colors ${
              activeTab === "sections"
                ? "border-blue-600 dark:border-blue-500 text-blue-600 dark:text-blue-400"
                : "border-transparent text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200"
            }`}
          >
            Segmented Resume Sections
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {activeTab === "insights" ? (
            <>
              {/* Executive Rationale Box */}
              <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50 p-4">
                <span className="text-[11px] font-semibold tracking-wider uppercase text-zinc-500 dark:text-zinc-400">
                  Hiring Assessment Rationale
                </span>
                <p className="mt-2 text-sm leading-relaxed text-zinc-800 dark:text-zinc-300">
                  {evidence.rationale}
                </p>
                {knockout.is_knocked_out && (
                  <div className="mt-3 rounded-lg border border-rose-300 dark:border-rose-900/60 bg-rose-50 dark:bg-rose-950/30 p-3 text-xs text-rose-800 dark:text-rose-300">
                    <span className="font-semibold">Knockout Triggers:</span>
                    <ul className="mt-1 list-disc list-inside space-y-0.5">
                      {knockout.reasons.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Score Component Bars */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-4 space-y-3">
                  <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-300">
                    Multi-Factor Evaluation Breakdown
                  </span>

                  <div>
                    <div className="flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
                      <span>Semantic &amp; Lexical Relevance</span>
                      <span className="font-mono text-zinc-800 dark:text-zinc-200">
                        {score.semantic_score.toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-1 h-1.5 w-full rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
                      <div
                        className="h-full bg-blue-600 dark:bg-blue-500 rounded-full"
                        style={{ width: `${score.semantic_score}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
                      <span>Technical Skill Coverage</span>
                      <span className="font-mono text-zinc-800 dark:text-zinc-200">
                        {score.skill_score.toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-1 h-1.5 w-full rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
                      <div
                        className="h-full bg-emerald-600 dark:bg-emerald-500 rounded-full"
                        style={{ width: `${score.skill_score}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
                      <span>Experience Alignment</span>
                      <span className="font-mono text-zinc-800 dark:text-zinc-200">
                        {score.experience_score.toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-1 h-1.5 w-full rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
                      <div
                        className="h-full bg-purple-600 dark:bg-purple-500 rounded-full"
                        style={{ width: `${score.experience_score}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
                      <span>Domain &amp; Industry Fit</span>
                      <span className="font-mono text-zinc-800 dark:text-zinc-200">
                        {score.domain_score.toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-1 h-1.5 w-full rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
                      <div
                        className="h-full bg-amber-600 dark:bg-amber-500 rounded-full"
                        style={{ width: `${score.domain_score}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Skills Breakdown */}
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-4 space-y-4">
                  <div>
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                      <CheckCircle2 className="h-4 w-4" />
                      Matched Competencies ({evidence.matched_skills.length})
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {evidence.matched_skills.length > 0 ? (
                        evidence.matched_skills.map((s) => (
                          <span
                            key={s}
                            className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-[11px] font-medium text-emerald-700 dark:border-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-300"
                          >
                            {s}
                          </span>
                        ))
                      ) : (
                        <span className="text-xs text-zinc-400">
                          No direct skill matches detected
                        </span>
                      )}
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-rose-600 dark:text-rose-400">
                      <XCircle className="h-4 w-4" />
                      Missing Requirements ({evidence.missing_skills.length})
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {evidence.missing_skills.length > 0 ? (
                        evidence.missing_skills.map((s) => (
                          <span
                            key={s}
                            className="rounded-full border border-rose-200 bg-rose-50 px-2.5 py-0.5 text-[11px] font-medium text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/30 dark:text-rose-300"
                          >
                            {s}
                          </span>
                        ))
                      ) : (
                        <span className="text-xs text-emerald-600 dark:text-emerald-400">
                          Complete technical coverage! 🎉
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Verified Text Citations */}
              <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-4">
                <div className="flex items-center gap-2 text-xs font-semibold text-zinc-900 dark:text-zinc-200">
                  <Quote className="h-4 w-4 text-blue-600 dark:text-blue-500" />
                  Verified Resume Text Citations
                </div>
                <div className="mt-3 space-y-2">
                  {evidence.key_citations.length > 0 ? (
                    evidence.key_citations.map((cite, i) => (
                      <div
                        key={i}
                        className="rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950/60 p-3 font-mono text-xs leading-relaxed text-zinc-700 dark:text-zinc-300"
                      >
                        &ldquo;{cite}&rdquo;
                      </div>
                    ))
                  ) : (
                    <span className="text-xs text-zinc-400">
                      No direct sentence citations extracted
                    </span>
                  )}
                </div>
              </div>

              {/* Tailored Technical Interview Questions */}
              <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-4">
                <div className="flex items-center gap-2 text-xs font-semibold text-zinc-900 dark:text-zinc-200">
                  <HelpCircle className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                  Suggested Technical Screening Questions
                </div>
                <div className="mt-3 space-y-2">
                  {evidence.interview_questions.map((q, i) => (
                    <div
                      key={i}
                      className="rounded-lg border border-zinc-200 dark:border-zinc-800/80 bg-zinc-50 dark:bg-zinc-950/50 p-3 text-xs leading-relaxed text-zinc-700 dark:text-zinc-300"
                    >
                      <span className="font-semibold text-blue-600 dark:text-blue-400">
                        Q{i + 1}:{" "}
                      </span>
                      {q}
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="space-y-4">
              {Object.keys(cand.sections).length > 0 ? (
                Object.entries(cand.sections).map(([secName, secText]) => (
                  <div
                    key={secName}
                    className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-4"
                  >
                    <span className="text-xs font-semibold uppercase tracking-wider text-blue-600 dark:text-blue-400">
                      {secName}
                    </span>
                    <pre className="mt-2 whitespace-pre-wrap font-sans text-xs leading-relaxed text-zinc-700 dark:text-zinc-300">
                      {secText}
                    </pre>
                  </div>
                ))
              ) : (
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-4">
                  <pre className="whitespace-pre-wrap font-sans text-xs leading-relaxed text-zinc-700 dark:text-zinc-300">
                    {cand.raw_text}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
