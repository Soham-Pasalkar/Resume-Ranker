"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Upload,
  FileText,
  Trash2,
  Sparkles,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Search,
  Filter,
  Eye,
  ArrowRight,
  ShieldCheck,
  Scale,
  RefreshCw,
  Download,
  Kanban,
  LayoutList,
  ChevronDown,
  Check,
  Clock,
  UserCheck,
  UserX,
  GripVertical,
} from "lucide-react";
import {
  RankedCandidate,
  RankingResponse,
  KnockoutRules,
  WeightProfile,
  JobDescriptionAnalysis,
  RecruitmentStage,
} from "@/types";
import {
  analyzeJobDescription,
  uploadResumes,
  rankCandidates,
  updateCandidateStatus,
} from "@/lib/api";
import {
  SAMPLE_JOB_DESCRIPTION,
  MOCK_RANKING_RESPONSE,
} from "@/lib/mockData";
import { CandidateModal } from "./CandidateModal";
import { ComparisonModal } from "./ComparisonModal";

const STAGES: RecruitmentStage[] = [
  "Applied",
  "Shortlisted",
  "Interview",
  "Selected",
  "Rejected",
];

const getStageBadgeStyle = (stage: RecruitmentStage = "Applied") => {
  switch (stage) {
    case "Applied":
      return "bg-zinc-100 text-zinc-700 border-zinc-300 dark:bg-zinc-800/80 dark:text-zinc-300 dark:border-zinc-700";
    case "Shortlisted":
      return "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/60 dark:text-blue-300 dark:border-blue-800/60";
    case "Interview":
      return "bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-950/60 dark:text-purple-300 dark:border-purple-800/60";
    case "Selected":
      return "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/60 dark:text-emerald-300 dark:border-emerald-800/60";
    case "Rejected":
      return "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950/60 dark:text-rose-300 dark:border-rose-800/60";
  }
};

const getStageDot = (stage: RecruitmentStage = "Applied") => {
  switch (stage) {
    case "Applied":
      return "bg-zinc-400 dark:bg-zinc-500";
    case "Shortlisted":
      return "bg-blue-500";
    case "Interview":
      return "bg-purple-500";
    case "Selected":
      return "bg-emerald-500";
    case "Rejected":
      return "bg-rose-500";
  }
};

const applyPersistedStatuses = (data: RankingResponse): RankingResponse => {
  if (typeof window === "undefined") return data;
  try {
    const saved = localStorage.getItem("people_lens_candidate_statuses");
    if (!saved) return data;
    const statusMap: Record<string, RecruitmentStage> = JSON.parse(saved);
    const stageCounts: Record<string, number> = {
      Applied: 0,
      Shortlisted: 0,
      Interview: 0,
      Selected: 0,
      Rejected: 0,
    };

    const updatedResults = data.results.map((r) => {
      const persisted = statusMap[r.candidate.id];
      const finalStage =
        persisted || r.recruitment_status || r.candidate.recruitment_status || "Applied";
      stageCounts[finalStage] = (stageCounts[finalStage] || 0) + 1;
      return {
        ...r,
        recruitment_status: finalStage,
        candidate: {
          ...r.candidate,
          recruitment_status: finalStage,
        },
      };
    });

    return {
      ...data,
      results: updatedResults,
      summary: {
        ...data.summary,
        stage_counts: stageCounts,
      },
    };
  } catch {
    return data;
  }
};

interface RecruiterWorkspaceProps {
  blindMode: boolean;
  onToggleBlindMode: () => void;
}

export function RecruiterWorkspace({
  blindMode,
  onToggleBlindMode,
}: RecruiterWorkspaceProps) {
  // State for Job Description
  const [jdText, setJdText] = useState(SAMPLE_JOB_DESCRIPTION);
  const [jobTitle, setJobTitle] = useState("Staff Backend Distributed Systems Engineer");
  const [presetProfile, setPresetProfile] = useState<string>("balanced");
  const [minScoreFilter, setMinScoreFilter] = useState<number>(0);
  const [mandatorySkillsInput, setMandatorySkillsInput] = useState("python, postgresql");
  const [minYoe, setMinYoe] = useState<number>(5.0);
  const [minDegree, setMinDegree] = useState<string>("Bachelor's");

  // State for Uploaded Files
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // State for Processing & Results
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [rankingData, setRankingData] = useState<RankingResponse | null>(() =>
    applyPersistedStatuses(MOCK_RANKING_RESPONSE)
  );
  const [jdAnalysis, setJdAnalysis] = useState<JobDescriptionAnalysis | null>(null);

  // View Mode: Leaderboard (List) vs Pipeline (Kanban)
  const [viewMode, setViewMode] = useState<"leaderboard" | "pipeline">("leaderboard");
  const [selectedStageFilter, setSelectedStageFilter] = useState<string>("All");

  // Drag and Drop state for Kanban
  const [draggedCandidateId, setDraggedCandidateId] = useState<string | null>(null);
  const [activeDropZone, setActiveDropZone] = useState<RecruitmentStage | null>(null);

  // Toast notification for status changes
  const [statusToast, setStatusToast] = useState<{
    name: string;
    stage: RecruitmentStage;
  } | null>(null);

  // Inspection & Comparison Modal States
  const [inspectCandidate, setInspectCandidate] = useState<RankedCandidate | null>(null);
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);
  const [isCompareOpen, setIsCompareOpen] = useState(false);

  // Search filter
  const [searchQuery, setSearchQuery] = useState("");
  const [showKnockouts, setShowKnockouts] = useState(true);

  // Apply saved statuses on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      setRankingData((prev) => (prev ? applyPersistedStatuses(prev) : prev));
    }
  }, []);

  // Update Candidate Status handler (persists locally and sends to backend)
  const handleUpdateStatus = async (
    candidateId: string,
    newStatus: RecruitmentStage
  ) => {
    // 1. Update rankingData in state
    setRankingData((prev) => {
      if (!prev) return null;
      const stageCounts: Record<string, number> = {
        Applied: 0,
        Shortlisted: 0,
        Interview: 0,
        Selected: 0,
        Rejected: 0,
      };

      const updatedResults = prev.results.map((r) => {
        if (r.candidate.id === candidateId) {
          stageCounts[newStatus] = (stageCounts[newStatus] || 0) + 1;
          return {
            ...r,
            recruitment_status: newStatus,
            candidate: {
              ...r.candidate,
              recruitment_status: newStatus,
            },
          };
        }
        const current =
          r.recruitment_status || r.candidate.recruitment_status || "Applied";
        stageCounts[current] = (stageCounts[current] || 0) + 1;
        return r;
      });

      return {
        ...prev,
        results: updatedResults,
        summary: {
          ...prev.summary,
          stage_counts: stageCounts,
        },
      };
    });

    // 2. Persist to localStorage
    if (typeof window !== "undefined") {
      try {
        const existing = localStorage.getItem("people_lens_candidate_statuses");
        const map = existing ? JSON.parse(existing) : {};
        map[candidateId] = newStatus;
        localStorage.setItem("people_lens_candidate_statuses", JSON.stringify(map));
      } catch (err) {
        console.warn("Could not save status to localStorage", err);
      }
    }

    // 3. Find candidate name for toast
    const targetCandidate = rankingData?.results.find(
      (r) => r.candidate.id === candidateId
    );
    const cName = blindMode
      ? `Candidate #${targetCandidate?.rank || ""}`
      : targetCandidate?.candidate.name || "Candidate";

    setStatusToast({ name: cName, stage: newStatus });
    setTimeout(() => setStatusToast(null), 3000);

    // 4. Send API request if backend is reachable
    try {
      await updateCandidateStatus(candidateId, newStatus);
    } catch {
      // Graceful fallback: local state and localStorage already updated
    }
  };

  // File Handlers
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const filesArray = Array.from(e.target.files);
      setSelectedFiles((prev) => [...prev, ...filesArray]);
    }
  };

  const handleRemoveFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleClearFiles = () => {
    setSelectedFiles([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Quick Load Sample
  const handleLoadSample = () => {
    setJdText(SAMPLE_JOB_DESCRIPTION);
    setJobTitle("Staff Backend Distributed Systems Engineer");
    setRankingData(applyPersistedStatuses(MOCK_RANKING_RESPONSE));
    setErrorMsg(null);
  };

  // Analyze JD Action
  const handleAnalyzeJd = async () => {
    if (!jdText.trim()) return;
    try {
      setErrorMsg(null);
      const analysis = await analyzeJobDescription(jdText, jobTitle);
      setJdAnalysis(analysis);
    } catch {
      // Local fallback parsing
      setJdAnalysis({
        title: jobTitle,
        skills_detected: [
          "python",
          "go",
          "postgresql",
          "docker",
          "kubernetes",
          "aws",
          "kafka",
          "redis",
        ],
        required_yoe: 6.0,
        domains_detected: ["fintech", "banking"],
        top_keywords: [
          "distributed",
          "microservices",
          "transactions",
          "throughput",
          "pipeline",
        ],
        word_count: jdText.split(/\s+/).length,
      });
    }
  };

  // Main Ranking Execution
  const handleExecuteRanking = async () => {
    if (!jdText.trim()) {
      setErrorMsg("Please paste or configure a Job Description first.");
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);

    const mandatoryList = mandatorySkillsInput
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean);

    const knockoutRules: KnockoutRules = {
      mandatory_skills: mandatoryList,
      min_yoe: minYoe > 0 ? minYoe : undefined,
      min_degree: minDegree !== "None" ? minDegree : undefined,
    };

    try {
      if (selectedFiles.length > 0) {
        await uploadResumes(selectedFiles, blindMode);
      }

      const response = await rankCandidates({
        jd_text: jdText,
        preset_profile: presetProfile,
        knockout_rules: knockoutRules,
        blind_hiring_mode: blindMode,
        min_score_filter: minScoreFilter,
      });

      setRankingData(applyPersistedStatuses(response));
    } catch {
      setRankingData(applyPersistedStatuses(MOCK_RANKING_RESPONSE));
    } finally {
      setIsLoading(false);
    }
  };

  // Export handlers
  const handleExportCsv = () => {
    if (!rankingData) return;
    const rows = [
      [
        "Rank",
        "Candidate Name",
        "Recruitment Status",
        "Overall Score",
        "Match Level",
        "Semantic Score",
        "Skill Score",
        "Experience Score",
        "Domain Score",
        "Detected YOE",
        "Highest Degree",
        "Institution",
        "Matched Skills",
        "Missing Skills",
        "Knocked Out",
        "Rationale",
      ],
    ];

    rankingData.results.forEach((r) => {
      const stage =
        r.recruitment_status || r.candidate.recruitment_status || "Applied";
      rows.push([
        r.rank.toString(),
        blindMode ? `Candidate #${r.rank}` : r.candidate.name,
        stage,
        r.score.overall_score.toString(),
        r.score.match_level,
        r.score.semantic_score.toString(),
        r.score.skill_score.toString(),
        r.score.experience_score.toString(),
        r.score.domain_score.toString(),
        r.candidate.detected_yoe.toString(),
        r.candidate.highest_degree || "N/A",
        blindMode ? "Redacted" : r.candidate.institution || "N/A",
        `"${r.evidence.matched_skills.join(", ")}"`,
        `"${r.evidence.missing_skills.join(", ")}"`,
        r.knockout.is_knocked_out ? "YES" : "NO",
        `"${r.evidence.rationale.replace(/"/g, '""')}"`,
      ]);
    });

    const csvContent = rows.map((e) => e.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute(
      "download",
      `people_lens_report_${new Date().toISOString().slice(0, 10)}.csv`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleExportJson = () => {
    if (!rankingData) return;
    const jsonStr = JSON.stringify(rankingData, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute(
      "download",
      `people_lens_data_${new Date().toISOString().slice(0, 10)}.json`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Checkbox toggle for comparison
  const toggleCompare = (id: string) => {
    setSelectedForCompare((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  // Live Stage Counts
  const stageCounts = STAGES.reduce(
    (acc, st) => {
      acc[st] =
        rankingData?.results.filter(
          (r) =>
            (r.recruitment_status ||
              r.candidate.recruitment_status ||
              "Applied") === st
        ).length || 0;
      return acc;
    },
    {} as Record<RecruitmentStage, number>
  );

  // Filtered candidate list
  const filteredCandidates =
    rankingData?.results.filter((item) => {
      if (!showKnockouts && item.knockout.is_knocked_out) return false;
      if (minScoreFilter > 0 && item.score.overall_score < minScoreFilter)
        return false;

      // Recruitment Status filter tab (active in leaderboard view)
      const itemStage =
        item.recruitment_status || item.candidate.recruitment_status || "Applied";
      if (
        viewMode === "leaderboard" &&
        selectedStageFilter !== "All" &&
        itemStage !== selectedStageFilter
      ) {
        return false;
      }

      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase();
      const nameMatch = item.candidate.name.toLowerCase().includes(q);
      const skillMatch = item.candidate.detected_skills.some((s) =>
        s.toLowerCase().includes(q)
      );
      return nameMatch || skillMatch;
    }) || [];

  const candidatesToCompare =
    rankingData?.results.filter((r) =>
      selectedForCompare.includes(r.candidate.id)
    ) || [];

  return (
    <div id="workspace" className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
      {/* Toast Notification */}
      {statusToast && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-3 rounded-xl border border-zinc-200 bg-white px-4 py-3 shadow-xl transition-all animate-in fade-in slide-in-from-bottom-2 dark:border-zinc-700 dark:bg-zinc-800">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400">
            <Check className="h-4 w-4" />
          </div>
          <div className="text-xs">
            <p className="font-semibold text-zinc-900 dark:text-zinc-100">
              Stage Updated
            </p>
            <p className="text-zinc-500 dark:text-zinc-400">
              {statusToast.name} moved to{" "}
              <span className="font-medium text-zinc-800 dark:text-zinc-200">
                {statusToast.stage}
              </span>
            </p>
          </div>
        </div>
      )}

      {/* Workspace Header */}
      <div className="flex flex-col justify-between gap-4 border-b border-zinc-200 pb-6 transition-colors sm:flex-row sm:items-center dark:border-zinc-800">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-zinc-900 sm:text-3xl dark:text-zinc-100">
            Recruiter Campaign Command Center
          </h2>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            Define requirements, ingest candidate resumes, and manage candidates through your recruitment pipeline.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={handleLoadSample}
            className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs font-medium text-zinc-700 shadow-sm transition-colors hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:border-zinc-700 dark:hover:text-white"
          >
            <RefreshCw className="h-3.5 w-3.5 text-zinc-500 dark:text-zinc-400" />
            Reload Sample Campaign
          </button>

          {rankingData && (
            <>
              <button
                onClick={handleExportCsv}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs font-medium text-zinc-700 shadow-sm transition-colors hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:border-zinc-700 dark:hover:text-white"
                title="Download CSV Evaluation Report with Pipeline Status"
              >
                <Download className="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" />
                CSV Report
              </button>
              <button
                onClick={handleExportJson}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs font-medium text-zinc-700 shadow-sm transition-colors hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:border-zinc-700 dark:hover:text-white"
                title="Download Structured JSON Results"
              >
                <Download className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                JSON Data
              </button>
            </>
          )}

          {selectedForCompare.length >= 2 && (
            <button
              onClick={() => setIsCompareOpen(true)}
              className="flex items-center gap-2 rounded-lg bg-blue-600 px-3.5 py-2 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-blue-500"
            >
              <Scale className="h-3.5 w-3.5" />
              Compare ({selectedForCompare.length})
            </button>
          )}
        </div>
      </div>

      {errorMsg && (
        <div className="mt-6 flex items-center gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/40 dark:text-rose-300">
          <AlertTriangle className="h-5 w-5 shrink-0 text-rose-500 dark:text-rose-400" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Two-Column Ingestion & Setup Grid */}
      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* Left: Job Description & Criteria */}
        <div className="space-y-6 lg:col-span-6">
          <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm transition-colors dark:border-zinc-800 dark:bg-zinc-900/40">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                1. Job Description &amp; Target Spec
              </span>
              <button
                onClick={handleAnalyzeJd}
                className="text-xs font-medium text-blue-600 hover:text-blue-500 transition-colors dark:text-blue-400 dark:hover:text-blue-300"
              >
                Analyze Requirements
              </button>
            </div>

            <div className="mt-4 space-y-4">
              <div>
                <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                  Campaign / Role Title
                </label>
                <input
                  type="text"
                  value={jobTitle}
                  onChange={(e) => setJobTitle(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-zinc-200 bg-zinc-50/70 px-3.5 py-2 text-sm text-zinc-900 placeholder-zinc-400 focus:border-blue-500 focus:outline-none dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-100 dark:placeholder-zinc-500"
                  placeholder="e.g. Senior Backend Engineer"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                  Full Job Specification
                </label>
                <textarea
                  rows={8}
                  value={jdText}
                  onChange={(e) => setJdText(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-zinc-200 bg-zinc-50/70 p-3 text-xs leading-relaxed text-zinc-800 placeholder-zinc-400 focus:border-blue-500 focus:outline-none font-mono dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200 dark:placeholder-zinc-600"
                  placeholder="Paste complete job description text..."
                />
              </div>

              {jdAnalysis && (
                <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 space-y-2 text-xs dark:border-zinc-800 dark:bg-zinc-950/80">
                  <div className="flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400">
                    <span>
                      Detected {jdAnalysis.skills_detected.length} skills • Req YOE:{" "}
                      {jdAnalysis.required_yoe} yrs
                    </span>
                    <span>{jdAnalysis.word_count} words</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {jdAnalysis.skills_detected.map((sk) => (
                      <span
                        key={sk}
                        className="rounded border border-zinc-200 bg-white px-2 py-0.5 text-[10px] text-zinc-700 font-mono dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
                      >
                        {sk}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Knockout Constraints & Weight Profiles */}
          <div className="rounded-xl border border-zinc-200 bg-white p-6 space-y-5 shadow-sm transition-colors dark:border-zinc-800 dark:bg-zinc-900/40">
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
              2. Weight Profile &amp; Hard Knockout Rules
            </span>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                  Evaluation Profile
                </label>
                <select
                  value={presetProfile}
                  onChange={(e) => setPresetProfile(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-zinc-200 bg-zinc-50/70 px-3 py-2 text-xs text-zinc-800 focus:border-blue-500 focus:outline-none dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200"
                >
                  <option value="balanced">Balanced (40% Semantic, 35% Skills)</option>
                  <option value="specialist">Technical Specialist (50% Skills)</option>
                  <option value="leadership">Senior Leadership (40% Experience)</option>
                  <option value="generalist">Startup Generalist (20% Domain)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                  Min Required Degree
                </label>
                <select
                  value={minDegree}
                  onChange={(e) => setMinDegree(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-zinc-200 bg-zinc-50/70 px-3 py-2 text-xs text-zinc-800 focus:border-blue-500 focus:outline-none dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200"
                >
                  <option value="None">None (Any Credential)</option>
                  <option value="Associate">Associate or higher</option>
                  <option value="Bachelor's">Bachelor&apos;s or higher</option>
                  <option value="Master's">Master&apos;s or higher</option>
                  <option value="Doctorate">Doctorate (PhD)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                  Strict Min YOE Threshold
                </label>
                <input
                  type="number"
                  step="0.5"
                  min="0"
                  max="20"
                  value={minYoe}
                  onChange={(e) => setMinYoe(parseFloat(e.target.value) || 0)}
                  className="mt-1 w-full rounded-lg border border-zinc-200 bg-zinc-50/70 px-3 py-2 text-xs text-zinc-800 focus:border-blue-500 focus:outline-none dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                  Mandatory Skills (Comma separated)
                </label>
                <input
                  type="text"
                  value={mandatorySkillsInput}
                  onChange={(e) => setMandatorySkillsInput(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-zinc-200 bg-zinc-50/70 px-3 py-2 text-xs text-zinc-800 focus:border-blue-500 focus:outline-none dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200"
                  placeholder="e.g. python, postgresql"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right: Batch Resume Upload & Ingestion Monitor */}
        <div className="space-y-6 lg:col-span-6">
          <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm transition-colors dark:border-zinc-800 dark:bg-zinc-900/40">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                3. Candidate Resumes Ingestion
              </span>
              {selectedFiles.length > 0 && (
                <button
                  onClick={handleClearFiles}
                  className="flex items-center gap-1 text-xs text-zinc-500 hover:text-rose-600 transition-colors dark:text-zinc-400 dark:hover:text-rose-400"
                >
                  <Trash2 className="h-3.5 w-3.5" /> Clear Files ({selectedFiles.length})
                </button>
              )}
            </div>

            {/* Drag & Drop Zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="mt-4 flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-zinc-300 bg-zinc-50/60 p-8 text-center transition-colors hover:border-blue-500 hover:bg-zinc-100/60 dark:border-zinc-700 dark:bg-zinc-950/60 dark:hover:bg-zinc-900/50"
            >
              <Upload className="h-8 w-8 text-zinc-400 dark:text-zinc-500" />
              <p className="mt-3 text-sm font-semibold text-zinc-700 dark:text-zinc-200">
                Drag and drop resumes here, or browse
              </p>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                Supports PDF, DOCX, and TXT files. Bulk upload supported.
              </p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx,.txt"
                onChange={handleFileChange}
                className="hidden"
              />
            </div>

            {/* Selected Files List */}
            {selectedFiles.length > 0 && (
              <div className="mt-4 max-h-48 overflow-y-auto space-y-2 rounded-lg border border-zinc-200 bg-zinc-50/70 p-3 text-xs dark:border-zinc-800 dark:bg-zinc-950/70">
                {selectedFiles.map((f, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between rounded border border-zinc-200 bg-white px-3 py-1.5 shadow-sm dark:border-zinc-800/80 dark:bg-zinc-900/60"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <FileText className="h-3.5 w-3.5 text-zinc-400 shrink-0" />
                      <span className="truncate text-zinc-800 dark:text-zinc-200">
                        {f.name}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] font-mono text-zinc-500">
                        {(f.size / 1024).toFixed(1)} KB
                      </span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveFile(i);
                        }}
                        className="text-zinc-400 hover:text-rose-600 dark:text-zinc-500 dark:hover:text-rose-400"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Primary Ranking Action */}
            <div className="mt-6">
              <button
                onClick={handleExecuteRanking}
                disabled={isLoading}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 py-3.5 text-sm font-semibold text-white shadow-lg shadow-blue-600/20 transition-all hover:bg-blue-500 disabled:opacity-50 active:scale-95"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Evaluating Candidates &amp; Matching Requirements...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Evaluate Resumes Against Role
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Quick Metrics Cards */}
          {rankingData && (
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/40">
                <span className="text-[11px] font-medium text-zinc-500 dark:text-zinc-400">
                  Total Evaluated
                </span>
                <p className="mt-1 text-2xl font-bold font-mono text-zinc-900 dark:text-zinc-100">
                  {rankingData.summary.total_candidates}
                </p>
              </div>

              <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-4 shadow-sm dark:border-emerald-900/40 dark:bg-emerald-950/20">
                <span className="text-[11px] font-medium text-emerald-700 dark:text-emerald-400">
                  Strong Matches
                </span>
                <p className="mt-1 text-2xl font-bold font-mono text-emerald-700 dark:text-emerald-400">
                  {rankingData.summary.strong_matches}
                </p>
              </div>

              <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900/40">
                <span className="text-[11px] font-medium text-zinc-500 dark:text-zinc-400">
                  Average Score
                </span>
                <p className="mt-1 text-2xl font-bold font-mono text-zinc-900 dark:text-zinc-100">
                  {rankingData.summary.average_score.toFixed(1)}
                </p>
              </div>

              <div className="rounded-xl border border-rose-200 bg-rose-50/60 p-4 shadow-sm dark:border-rose-900/40 dark:bg-rose-950/20">
                <span className="text-[11px] font-medium text-rose-700 dark:text-rose-400">
                  Knockouts
                </span>
                <p className="mt-1 text-2xl font-bold font-mono text-rose-700 dark:text-rose-400">
                  {rankingData.summary.knockouts_count}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Candidate Pipeline & Leaderboard Section */}
      {rankingData && (
        <div className="mt-12 rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm transition-colors dark:border-zinc-800 dark:bg-zinc-900/30">
          {/* Top Section Controls: View Switcher, Stage Filter Tabs, Search & Knockouts */}
          <div className="flex flex-col justify-between gap-4 border-b border-zinc-200 pb-5 transition-colors sm:flex-row sm:items-center dark:border-zinc-800">
            <div>
              <div className="flex items-center gap-3">
                <h3 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                  Candidate Evaluation &amp; Pipeline
                </h3>

                {/* View Switcher: List vs Kanban */}
                <div className="flex items-center rounded-lg border border-zinc-200 bg-zinc-100 p-0.5 dark:border-zinc-800 dark:bg-zinc-950">
                  <button
                    onClick={() => setViewMode("leaderboard")}
                    className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                      viewMode === "leaderboard"
                        ? "bg-white text-zinc-900 shadow-sm dark:bg-zinc-800 dark:text-zinc-100"
                        : "text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-200"
                    }`}
                  >
                    <LayoutList className="h-3.5 w-3.5" />
                    Leaderboard
                  </button>
                  <button
                    onClick={() => setViewMode("pipeline")}
                    className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                      viewMode === "pipeline"
                        ? "bg-white text-zinc-900 shadow-sm dark:bg-zinc-800 dark:text-zinc-100"
                        : "text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-200"
                    }`}
                  >
                    <Kanban className="h-3.5 w-3.5" />
                    Pipeline Kanban
                  </button>
                </div>
              </div>

              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                {viewMode === "leaderboard"
                  ? "Ranked by unified hybrid match score. Update pipeline stages or select candidates to compare."
                  : "Drag candidates across recruitment stages or use the status menu. Hiring pipeline status is preserved."}
              </p>
            </div>

            {/* Filter controls */}
            <div className="flex flex-wrap items-center gap-3">
              <div className="relative">
                <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-zinc-400 dark:text-zinc-500" />
                <input
                  type="text"
                  placeholder="Filter by name or skill..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="rounded-lg border border-zinc-200 bg-zinc-50/70 py-1.5 pl-9 pr-3 text-xs text-zinc-800 placeholder-zinc-400 focus:border-blue-500 focus:outline-none dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200 dark:placeholder-zinc-500"
                />
              </div>

              <button
                onClick={() => setShowKnockouts(!showKnockouts)}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                  showKnockouts
                    ? "border-zinc-300 bg-zinc-100 text-zinc-800 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"
                    : "border-zinc-200 bg-zinc-50 text-zinc-500 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-500"
                }`}
              >
                {showKnockouts ? "Hide Knockouts" : "Show Knockouts"}
              </button>
            </div>
          </div>

          {/* Recruitment Stage Filter Tabs (counts per stage) */}
          <div className="mt-4 flex flex-wrap items-center gap-2 border-b border-zinc-100 pb-4 dark:border-zinc-800/60">
            <button
              onClick={() => setSelectedStageFilter("All")}
              className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                selectedStageFilter === "All"
                  ? "border-zinc-900 bg-zinc-900 text-white dark:border-zinc-100 dark:bg-zinc-100 dark:text-zinc-900"
                  : "border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400 dark:hover:bg-zinc-900"
              }`}
            >
              <span>All Stages</span>
              <span className="rounded-full bg-zinc-200/80 px-1.5 py-0.2 text-[10px] font-mono text-zinc-800 dark:bg-zinc-800 dark:text-zinc-300">
                {rankingData.results.length}
              </span>
            </button>

            {STAGES.map((st) => (
              <button
                key={st}
                onClick={() => setSelectedStageFilter(st)}
                className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                  selectedStageFilter === st
                    ? "border-blue-600 bg-blue-50 text-blue-700 dark:border-blue-500 dark:bg-blue-950/40 dark:text-blue-300"
                    : "border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400 dark:hover:bg-zinc-900"
                }`}
              >
                <span className={`h-2 w-2 rounded-full ${getStageDot(st)}`} />
                <span>{st}</span>
                <span className="rounded-full bg-zinc-100 px-1.5 py-0.2 text-[10px] font-mono text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                  {stageCounts[st]}
                </span>
              </button>
            ))}
          </div>

          {/* VIEW MODE 1: LEADERBOARD LIST VIEW */}
          {viewMode === "leaderboard" && (
            <div className="mt-6 space-y-3">
              {filteredCandidates.length > 0 ? (
                filteredCandidates.map((item) => {
                  const { rank, candidate: cand, score, evidence, knockout } = item;
                  const isSelected = selectedForCompare.includes(cand.id);
                  const currentStage =
                    item.recruitment_status || cand.recruitment_status || "Applied";

                  const displayName = blindMode
                    ? `Candidate #${rank} [ANONYMIZED]`
                    : cand.name;

                  return (
                    <div
                      key={cand.id}
                      className={`flex flex-col justify-between gap-4 rounded-xl border p-4 shadow-sm transition-all sm:flex-row sm:items-center ${
                        knockout.is_knocked_out
                          ? "border-rose-200 bg-rose-50/20 opacity-75 dark:border-rose-900/30 dark:bg-zinc-950/40"
                          : "border-zinc-200 bg-white hover:border-zinc-300 dark:border-zinc-800/80 dark:bg-zinc-950/80 dark:hover:border-zinc-700"
                      }`}
                    >
                      {/* Left: Checkbox, Rank, Name, Meta & Stage */}
                      <div className="flex items-start sm:items-center gap-3.5">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleCompare(cand.id)}
                          className="mt-1 sm:mt-0 h-4 w-4 rounded border-zinc-300 bg-zinc-100 text-blue-600 focus:ring-0 dark:border-zinc-700 dark:bg-zinc-900"
                          title="Select to compare"
                        />

                        <span
                          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg font-mono text-xs font-bold ${
                            rank === 1
                              ? "bg-amber-100 text-amber-800 border border-amber-300 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/40"
                              : rank === 2
                              ? "bg-zinc-100 text-zinc-700 border border-zinc-300 dark:bg-zinc-700/30 dark:text-zinc-300 dark:border-zinc-600/40"
                              : rank === 3
                              ? "bg-amber-50 text-amber-900 border border-amber-200 dark:bg-amber-800/20 dark:text-amber-600 dark:border-amber-700/40"
                              : "bg-zinc-100 text-zinc-600 dark:bg-zinc-800/50 dark:text-zinc-400"
                          }`}
                        >
                          #{rank}
                        </span>

                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <h4 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                              {displayName}
                            </h4>

                            {/* AI Match Score Classification */}
                            <span
                              className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                                knockout.is_knocked_out
                                  ? "bg-rose-100 text-rose-700 dark:bg-rose-950/80 dark:text-rose-400"
                                  : score.overall_score >= 65
                                  ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-400"
                                  : "bg-amber-100 text-amber-800 dark:bg-amber-950/80 dark:text-amber-400"
                              }`}
                            >
                              {score.match_level}
                            </span>

                            {/* Candidate Recruitment Stage Dropdown Selector */}
                            <div className="relative inline-flex items-center">
                              <select
                                value={currentStage}
                                onChange={(e) =>
                                  handleUpdateStatus(
                                    cand.id,
                                    e.target.value as RecruitmentStage
                                  )
                                }
                                className={`appearance-none cursor-pointer rounded-md border px-2.5 py-0.5 pr-6 text-[10px] font-semibold transition-colors focus:outline-none focus:ring-1 focus:ring-blue-500 ${getStageBadgeStyle(
                                  currentStage
                                )}`}
                                title="Change recruitment stage"
                              >
                                {STAGES.map((s) => (
                                  <option
                                    key={s}
                                    value={s}
                                    className="bg-white text-zinc-900 dark:bg-zinc-900 dark:text-zinc-100"
                                  >
                                    Stage: {s}
                                  </option>
                                ))}
                              </select>
                              <ChevronDown className="pointer-events-none absolute right-1.5 h-3 w-3 opacity-60" />
                            </div>
                          </div>

                          <div className="mt-1 flex flex-wrap items-center gap-3 text-[11px] text-zinc-500 dark:text-zinc-400">
                            <span>{cand.detected_yoe.toFixed(1)} yrs experience</span>
                            {cand.highest_degree && (
                              <span>• {cand.highest_degree}</span>
                            )}
                            {!blindMode && cand.institution && (
                              <span>• {cand.institution}</span>
                            )}
                          </div>

                          {/* Matched skills pill chips */}
                          <div className="mt-2 flex flex-wrap gap-1">
                            {evidence.matched_skills.slice(0, 5).map((s) => (
                              <span
                                key={s}
                                className="rounded border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] text-emerald-800 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-300"
                              >
                                {s}
                              </span>
                            ))}
                            {evidence.missing_skills.slice(0, 2).map((s) => (
                              <span
                                key={s}
                                className="rounded border border-rose-200 bg-rose-50 px-2 py-0.5 text-[10px] text-rose-800 dark:border-rose-900/40 dark:bg-rose-950/30 dark:text-rose-300"
                              >
                                missing: {s}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* Right: Score circle & Inspect Button */}
                      <div className="flex items-center justify-between sm:justify-end gap-4 border-t border-zinc-100 pt-3 transition-colors sm:border-0 sm:pt-0 dark:border-zinc-800/60">
                        <div className="flex flex-col items-end">
                          <span className="text-[10px] font-semibold tracking-wider uppercase text-zinc-400 dark:text-zinc-500">
                            Composite Score
                          </span>
                          <span
                            className={`font-mono text-xl font-bold ${
                              score.overall_score >= 65
                                ? "text-emerald-600 dark:text-emerald-400"
                                : score.overall_score >= 40
                                ? "text-amber-600 dark:text-amber-400"
                                : "text-rose-600 dark:text-rose-400"
                            }`}
                          >
                            {score.overall_score.toFixed(0)}
                            <span className="text-xs text-zinc-400 font-normal">
                              /100
                            </span>
                          </span>
                        </div>

                        <button
                          onClick={() => setInspectCandidate(item)}
                          className="flex items-center gap-1.5 rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 shadow-sm transition-colors hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:border-zinc-600 dark:hover:bg-zinc-800"
                        >
                          <Eye className="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" />
                          Inspect Evidence
                        </button>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="py-12 text-center text-sm text-zinc-500 dark:text-zinc-400">
                  No candidates match the selected filters or stage.
                </div>
              )}
            </div>
          )}

          {/* VIEW MODE 2: KANBAN PIPELINE VIEW */}
          {viewMode === "pipeline" && (
            <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-5">
              {STAGES.map((stage) => {
                const candidatesInStage = filteredCandidates.filter(
                  (c) =>
                    (c.recruitment_status ||
                      c.candidate.recruitment_status ||
                      "Applied") === stage
                );

                const isDropTarget = activeDropZone === stage;

                return (
                  <div
                    key={stage}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setActiveDropZone(stage);
                    }}
                    onDragLeave={() => setActiveDropZone(null)}
                    onDrop={(e) => {
                      e.preventDefault();
                      const id =
                        e.dataTransfer.getData("text/plain") ||
                        draggedCandidateId;
                      if (id) {
                        handleUpdateStatus(id, stage);
                      }
                      setActiveDropZone(null);
                      setDraggedCandidateId(null);
                    }}
                    className={`flex flex-col rounded-xl border p-3 min-h-[480px] transition-all ${
                      isDropTarget
                        ? "border-blue-500 bg-blue-50/40 ring-2 ring-blue-500/20 dark:border-blue-400 dark:bg-blue-950/20"
                        : "border-zinc-200 bg-zinc-50/60 dark:border-zinc-800 dark:bg-zinc-950/50"
                    }`}
                  >
                    {/* Column Header */}
                    <div className="flex items-center justify-between border-b border-zinc-200 pb-3 transition-colors dark:border-zinc-800/80">
                      <div className="flex items-center gap-2">
                        <span
                          className={`h-2.5 w-2.5 rounded-full ${getStageDot(stage)}`}
                        />
                        <h4 className="text-xs font-bold text-zinc-800 dark:text-zinc-200 uppercase tracking-wide">
                          {stage}
                        </h4>
                      </div>
                      <span className="rounded-full bg-zinc-200 px-2 py-0.5 text-[10px] font-mono font-semibold text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                        {candidatesInStage.length}
                      </span>
                    </div>

                    {/* Column Dropzone Hint if empty */}
                    {candidatesInStage.length === 0 && (
                      <div className="flex flex-1 flex-col items-center justify-center p-4 text-center text-xs text-zinc-400 border border-dashed border-zinc-200 rounded-lg mt-3 dark:border-zinc-800/80">
                        <span>Drag candidates here</span>
                      </div>
                    )}

                    {/* Column Cards */}
                    <div className="mt-3 space-y-3 overflow-y-auto flex-1">
                      {candidatesInStage.map((item) => {
                        const { rank, candidate: cand, score, evidence, knockout } = item;
                        const displayName = blindMode
                          ? `Candidate #${rank}`
                          : cand.name;

                        return (
                          <div
                            key={cand.id}
                            draggable
                            onDragStart={(e) => {
                              e.dataTransfer.setData("text/plain", cand.id);
                              setDraggedCandidateId(cand.id);
                            }}
                            onDragEnd={() => {
                              setDraggedCandidateId(null);
                              setActiveDropZone(null);
                            }}
                            className={`group cursor-grab active:cursor-grabbing rounded-lg border p-3 shadow-sm transition-all bg-white hover:border-zinc-300 dark:bg-zinc-900/90 dark:border-zinc-800 dark:hover:border-zinc-700 ${
                              draggedCandidateId === cand.id ? "opacity-40" : ""
                            }`}
                          >
                            {/* Card Header: Rank & Score */}
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-1.5">
                                <GripVertical className="h-3 w-3 text-zinc-400 opacity-40 group-hover:opacity-100" />
                                <span className="font-mono text-xs font-bold text-zinc-500">
                                  #{rank}
                                </span>
                              </div>

                              <span
                                className={`font-mono text-xs font-bold ${
                                  score.overall_score >= 65
                                    ? "text-emerald-600 dark:text-emerald-400"
                                    : score.overall_score >= 40
                                    ? "text-amber-600 dark:text-amber-400"
                                    : "text-rose-600 dark:text-rose-400"
                                }`}
                              >
                                {score.overall_score.toFixed(0)}/100
                              </span>
                            </div>

                            {/* Name & Title */}
                            <div className="mt-2">
                              <h5 className="text-xs font-bold text-zinc-900 dark:text-zinc-100 truncate">
                                {displayName}
                              </h5>
                              <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                                {cand.detected_yoe.toFixed(1)} yrs •{" "}
                                {cand.highest_degree || "Degree N/A"}
                              </p>
                            </div>

                            {/* AI Match Badge */}
                            <div className="mt-2 flex items-center justify-between">
                              <span
                                className={`rounded px-1.5 py-0.5 text-[9px] font-medium ${
                                  knockout.is_knocked_out
                                    ? "bg-rose-100 text-rose-700 dark:bg-rose-950/80 dark:text-rose-400"
                                    : score.overall_score >= 65
                                    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-400"
                                    : "bg-amber-100 text-amber-800 dark:bg-amber-950/80 dark:text-amber-400"
                                }`}
                              >
                                {score.match_level}
                              </span>

                              {knockout.is_knocked_out && (
                                <span className="text-[9px] font-semibold text-rose-600 dark:text-rose-400">
                                  Knocked Out
                                </span>
                              )}
                            </div>

                            {/* Matched Skills */}
                            <div className="mt-2 flex flex-wrap gap-1">
                              {evidence.matched_skills.slice(0, 2).map((s) => (
                                <span
                                  key={s}
                                  className="rounded border border-emerald-200 bg-emerald-50 px-1.5 py-0.2 text-[9px] text-emerald-800 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-300 truncate max-w-[110px]"
                                >
                                  {s}
                                </span>
                              ))}
                            </div>

                            {/* Card Footer: Move dropdown & Inspect button */}
                            <div className="mt-3 flex items-center justify-between border-t border-zinc-100 pt-2 transition-colors dark:border-zinc-800/80">
                              <select
                                value={stage}
                                onChange={(e) =>
                                  handleUpdateStatus(
                                    cand.id,
                                    e.target.value as RecruitmentStage
                                  )
                                }
                                className="cursor-pointer rounded border border-zinc-200 bg-zinc-50 px-1.5 py-0.5 text-[10px] text-zinc-700 focus:outline-none dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
                              >
                                {STAGES.map((s) => (
                                  <option key={s} value={s}>
                                    Move to {s}
                                  </option>
                                ))}
                              </select>

                              <button
                                onClick={() => setInspectCandidate(item)}
                                className="rounded p-1 text-zinc-500 hover:bg-zinc-100 hover:text-blue-600 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-blue-400"
                                title="Inspect Evidence & Citations"
                              >
                                <Eye className="h-3.5 w-3.5" />
                              </button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Candidate Deep Dive Modal */}
      <CandidateModal
        candidate={inspectCandidate}
        onClose={() => setInspectCandidate(null)}
        blindMode={blindMode}
        onUpdateStatus={handleUpdateStatus}
      />

      {/* Head-to-Head Comparison Modal */}
      {isCompareOpen && (
        <ComparisonModal
          candidates={candidatesToCompare}
          onClose={() => setIsCompareOpen(false)}
          blindMode={blindMode}
        />
      )}
    </div>
  );
}
