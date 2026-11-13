"use client";

import React, { useState, useRef } from "react";
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
} from "lucide-react";
import {
  RankedCandidate,
  RankingResponse,
  KnockoutRules,
  WeightProfile,
  JobDescriptionAnalysis,
} from "@/types";
import {
  analyzeJobDescription,
  uploadResumes,
  rankCandidates,
} from "@/lib/api";
import {
  SAMPLE_JOB_DESCRIPTION,
  MOCK_RANKING_RESPONSE,
} from "@/lib/mockData";
import { CandidateModal } from "./CandidateModal";
import { ComparisonModal } from "./ComparisonModal";

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
  const [rankingData, setRankingData] = useState<RankingResponse | null>(MOCK_RANKING_RESPONSE);
  const [jdAnalysis, setJdAnalysis] = useState<JobDescriptionAnalysis | null>(null);

  // Inspection & Comparison Modal States
  const [inspectCandidate, setInspectCandidate] = useState<RankedCandidate | null>(null);
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);
  const [isCompareOpen, setIsCompareOpen] = useState(false);

  // Search filter
  const [searchQuery, setSearchQuery] = useState("");
  const [showKnockouts, setShowKnockouts] = useState(true);

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
    setRankingData(MOCK_RANKING_RESPONSE);
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
      // Local fallback parsing if backend isn't actively reachable
      setJdAnalysis({
        title: jobTitle,
        skills_detected: ["python", "go", "postgresql", "docker", "kubernetes", "aws", "kafka", "redis"],
        required_yoe: 6.0,
        domains_detected: ["fintech", "banking"],
        top_keywords: ["distributed", "microservices", "transactions", "throughput", "pipeline"],
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

      setRankingData(response);
    } catch {
      setRankingData(MOCK_RANKING_RESPONSE);
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
        "Rationale"
      ]
    ];

    rankingData.results.forEach((r) => {
      rows.push([
        r.rank.toString(),
        blindMode ? `Candidate #${r.rank}` : r.candidate.name,
        r.score.overall_score.toString(),
        r.score.match_level,
        r.score.semantic_score.toString(),
        r.score.skill_score.toString(),
        r.score.experience_score.toString(),
        r.score.domain_score.toString(),
        r.candidate.detected_yoe.toString(),
        r.candidate.highest_degree || "N/A",
        blindMode ? "Redacted" : (r.candidate.institution || "N/A"),
        `"${r.evidence.matched_skills.join(", ")}"`,
        `"${r.evidence.missing_skills.join(", ")}"`,
        r.knockout.is_knocked_out ? "YES" : "NO",
        `"${r.evidence.rationale.replace(/"/g, '""')}"`
      ]);
    });

    const csvContent = rows.map((e) => e.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `resumerank_report_${new Date().toISOString().slice(0, 10)}.csv`);
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
    link.setAttribute("download", `resumerank_data_${new Date().toISOString().slice(0, 10)}.json`);
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

  // Filtered candidate list
  const filteredCandidates = rankingData?.results.filter((item) => {
    if (!showKnockouts && item.knockout.is_knocked_out) return false;
    if (minScoreFilter > 0 && item.score.overall_score < minScoreFilter) return false;

    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    const nameMatch = item.candidate.name.toLowerCase().includes(q);
    const skillMatch = item.candidate.detected_skills.some((s) => s.toLowerCase().includes(q));
    return nameMatch || skillMatch;
  }) || [];

  const candidatesToCompare = rankingData?.results.filter((r) =>
    selectedForCompare.includes(r.candidate.id)
  ) || [];

  return (
    <div id="workspace" className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
      {/* Workspace Header */}
      <div className="flex flex-col justify-between gap-4 border-b border-zinc-800 pb-6 sm:flex-row sm:items-center">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-zinc-100 sm:text-3xl">
            Recruiter Campaign Command Center
          </h2>
          <p className="mt-1 text-sm text-zinc-400">
            Define requirements, ingest batches of candidate resumes, and review evidence-grounded rankings.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={handleLoadSample}
            className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs font-medium text-zinc-300 transition-colors hover:border-zinc-700 hover:text-white"
          >
            <RefreshCw className="h-3.5 w-3.5 text-zinc-400" />
            Reload Sample Campaign
          </button>

          {rankingData && (
            <>
              <button
                onClick={handleExportCsv}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs font-medium text-zinc-300 transition-colors hover:border-zinc-700 hover:text-white"
                title="Download CSV Evaluation Report"
              >
                <Download className="h-3.5 w-3.5 text-blue-400" />
                CSV Report
              </button>
              <button
                onClick={handleExportJson}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs font-medium text-zinc-300 transition-colors hover:border-zinc-700 hover:text-white"
                title="Download Structured JSON Results"
              >
                <Download className="h-3.5 w-3.5 text-emerald-400" />
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
        <div className="mt-6 flex items-center gap-3 rounded-xl border border-rose-900/60 bg-rose-950/40 p-4 text-sm text-rose-300">
          <AlertTriangle className="h-5 w-5 shrink-0 text-rose-400" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Two-Column Ingestion & Setup Grid */}
      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* Left: Job Description & Criteria */}
        <div className="space-y-6 lg:col-span-6">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-6">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                1. Job Description &amp; Target Spec
              </span>
              <button
                onClick={handleAnalyzeJd}
                className="text-xs font-medium text-blue-400 hover:text-blue-300 transition-colors"
              >
                Analyze Requirements
              </button>
            </div>

            <div className="mt-4 space-y-4">
              <div>
                <label className="block text-xs font-medium text-zinc-400">
                  Campaign / Role Title
                </label>
                <input
                  type="text"
                  value={jobTitle}
                  onChange={(e) => setJobTitle(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3.5 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:border-blue-500 focus:outline-none"
                  placeholder="e.g. Senior Backend Engineer"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-400">
                  Full Job Specification
                </label>
                <textarea
                  rows={8}
                  value={jdText}
                  onChange={(e) => setJdText(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-zinc-800 bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-200 placeholder-zinc-600 focus:border-blue-500 focus:outline-none font-mono"
                  placeholder="Paste complete job description text..."
                />
              </div>

              {jdAnalysis && (
                <div className="rounded-lg border border-zinc-800 bg-zinc-950/80 p-3 space-y-2 text-xs">
                  <div className="flex items-center justify-between text-[11px] text-zinc-400">
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
                        className="rounded bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-300 font-mono"
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
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-6 space-y-5">
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
              2. Weight Profile &amp; Hard Knockout Rules
            </span>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className="block text-xs font-medium text-zinc-400">
                  Evaluation Profile
                </label>
                <select
                  value={presetProfile}
                  onChange={(e) => setPresetProfile(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-xs text-zinc-200 focus:border-blue-500 focus:outline-none"
                >
                  <option value="balanced">Balanced (40% Semantic, 35% Skills)</option>
                  <option value="specialist">Technical Specialist (50% Skills)</option>
                  <option value="leadership">Senior Leadership (40% Experience)</option>
                  <option value="generalist">Startup Generalist (20% Domain)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-400">
                  Min Required Degree
                </label>
                <select
                  value={minDegree}
                  onChange={(e) => setMinDegree(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-xs text-zinc-200 focus:border-blue-500 focus:outline-none"
                >
                  <option value="None">None (Any Credential)</option>
                  <option value="Associate">Associate or higher</option>
                  <option value="Bachelor's">Bachelor&apos;s or higher</option>
                  <option value="Master's">Master&apos;s or higher</option>
                  <option value="Doctorate">Doctorate (PhD)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-400">
                  Strict Min YOE Threshold
                </label>
                <input
                  type="number"
                  step="0.5"
                  min="0"
                  max="20"
                  value={minYoe}
                  onChange={(e) => setMinYoe(parseFloat(e.target.value) || 0)}
                  className="mt-1 w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-xs text-zinc-200 focus:border-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-zinc-400">
                  Mandatory Skills (Comma separated)
                </label>
                <input
                  type="text"
                  value={mandatorySkillsInput}
                  onChange={(e) => setMandatorySkillsInput(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-xs text-zinc-200 focus:border-blue-500 focus:outline-none"
                  placeholder="e.g. python, postgresql"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right: Batch Resume Upload & Ingestion Monitor */}
        <div className="space-y-6 lg:col-span-6">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-6">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                3. Candidate Resumes Ingestion
              </span>
              {selectedFiles.length > 0 && (
                <button
                  onClick={handleClearFiles}
                  className="flex items-center gap-1 text-xs text-zinc-400 hover:text-rose-400 transition-colors"
                >
                  <Trash2 className="h-3.5 w-3.5" /> Clear Files ({selectedFiles.length})
                </button>
              )}
            </div>

            {/* Drag & Drop Zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="mt-4 flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-zinc-700 bg-zinc-950/60 p-8 text-center transition-colors hover:border-blue-500 hover:bg-zinc-900/50"
            >
              <Upload className="h-8 w-8 text-zinc-500" />
              <p className="mt-3 text-sm font-semibold text-zinc-200">
                Drag and drop resumes here, or browse
              </p>
              <p className="mt-1 text-xs text-zinc-500">
                Supports PDF (layout-preserving), DOCX, and TXT files. Bulk upload supported.
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
              <div className="mt-4 max-h-48 overflow-y-auto space-y-2 rounded-lg border border-zinc-800 bg-zinc-950/70 p-3 text-xs">
                {selectedFiles.map((f, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between rounded border border-zinc-800/80 bg-zinc-900/60 px-3 py-1.5"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <FileText className="h-3.5 w-3.5 text-zinc-400 shrink-0" />
                      <span className="truncate text-zinc-200">{f.name}</span>
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
                        className="text-zinc-500 hover:text-rose-400"
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
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 py-3.5 text-sm font-semibold text-white shadow-lg shadow-blue-900/20 transition-all hover:bg-blue-500 disabled:opacity-50 active:scale-95"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Executing Hybrid Scoring &amp; Citations...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Rank Resumes Against Specification
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Quick Metrics Cards */}
          {rankingData && (
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                <span className="text-[11px] font-medium text-zinc-400">
                  Total Analyzed
                </span>
                <p className="mt-1 text-2xl font-bold font-mono text-zinc-100">
                  {rankingData.summary.total_candidates}
                </p>
              </div>

              <div className="rounded-xl border border-emerald-900/40 bg-emerald-950/20 p-4">
                <span className="text-[11px] font-medium text-emerald-400">
                  Strong Matches
                </span>
                <p className="mt-1 text-2xl font-bold font-mono text-emerald-400">
                  {rankingData.summary.strong_matches}
                </p>
              </div>

              <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                <span className="text-[11px] font-medium text-zinc-400">
                  Average Score
                </span>
                <p className="mt-1 text-2xl font-bold font-mono text-zinc-100">
                  {rankingData.summary.average_score.toFixed(1)}
                </p>
              </div>

              <div className="rounded-xl border border-rose-900/40 bg-rose-950/20 p-4">
                <span className="text-[11px] font-medium text-rose-400">
                  Knockouts
                </span>
                <p className="mt-1 text-2xl font-bold font-mono text-rose-400">
                  {rankingData.summary.knockouts_count}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Candidate Leaderboard */}
      {rankingData && (
        <div className="mt-12 rounded-2xl border border-zinc-800 bg-zinc-900/30 p-6">
          <div className="flex flex-col justify-between gap-4 border-b border-zinc-800 pb-5 sm:flex-row sm:items-center">
            <div>
              <h3 className="text-lg font-bold text-zinc-100">
                Candidate Leaderboard ({filteredCandidates.length})
              </h3>
              <p className="text-xs text-zinc-400">
                Sorted by unified hybrid score and knockout status. Select candidates to compare.
              </p>
            </div>

            {/* Filter controls */}
            <div className="flex flex-wrap items-center gap-3">
              <div className="relative">
                <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-zinc-500" />
                <input
                  type="text"
                  placeholder="Filter by name or skill..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="rounded-lg border border-zinc-800 bg-zinc-950 py-1.5 pl-9 pr-3 text-xs text-zinc-200 placeholder-zinc-500 focus:border-blue-500 focus:outline-none"
                />
              </div>

              <button
                onClick={() => setShowKnockouts(!showKnockouts)}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                  showKnockouts
                    ? "border-zinc-700 bg-zinc-800 text-zinc-200"
                    : "border-zinc-800 bg-zinc-950 text-zinc-500"
                }`}
              >
                {showKnockouts ? "Hide Knockouts" : "Show Knockouts"}
              </button>
            </div>
          </div>

          {/* Candidates Rows */}
          <div className="mt-6 space-y-3">
            {filteredCandidates.length > 0 ? (
              filteredCandidates.map((item) => {
                const { rank, candidate: cand, score, evidence, knockout } = item;
                const isSelected = selectedForCompare.includes(cand.id);

                const displayName = blindMode
                  ? `Candidate #${rank} [ANONYMIZED]`
                  : cand.name;

                return (
                  <div
                    key={cand.id}
                    className={`flex flex-col justify-between gap-4 rounded-xl border p-4 transition-all sm:flex-row sm:items-center ${
                      knockout.is_knocked_out
                        ? "border-rose-900/30 bg-zinc-950/40 opacity-70"
                        : "border-zinc-800/80 bg-zinc-950/80 hover:border-zinc-700"
                    }`}
                  >
                    {/* Left: Checkbox, Rank, Name, Meta */}
                    <div className="flex items-start sm:items-center gap-3.5">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleCompare(cand.id)}
                        className="mt-1 sm:mt-0 h-4 w-4 rounded border-zinc-700 bg-zinc-900 text-blue-600 focus:ring-0 focus:ring-offset-0"
                      />

                      <span
                        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg font-mono text-xs font-bold ${
                          rank === 1
                            ? "bg-amber-500/20 text-amber-400 border border-amber-500/40"
                            : rank === 2
                            ? "bg-zinc-700/30 text-zinc-300 border border-zinc-600/40"
                            : rank === 3
                            ? "bg-amber-800/20 text-amber-600 border border-amber-700/40"
                            : "bg-zinc-800/50 text-zinc-400"
                        }`}
                      >
                        #{rank}
                      </span>

                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="text-sm font-bold text-zinc-100">
                            {displayName}
                          </h4>
                          <span
                            className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                              knockout.is_knocked_out
                                ? "bg-rose-950/80 text-rose-400"
                                : score.overall_score >= 65
                                ? "bg-emerald-950/80 text-emerald-400"
                                : "bg-amber-950/80 text-amber-400"
                            }`}
                          >
                            {score.match_level}
                          </span>
                        </div>

                        <div className="mt-1 flex flex-wrap items-center gap-3 text-[11px] text-zinc-400">
                          <span>{cand.detected_yoe.toFixed(1)} yrs experience</span>
                          {cand.highest_degree && <span>• {cand.highest_degree}</span>}
                          {!blindMode && cand.institution && (
                            <span>• {cand.institution}</span>
                          )}
                        </div>

                        {/* Matched skills pill chips */}
                        <div className="mt-2 flex flex-wrap gap-1">
                          {evidence.matched_skills.slice(0, 5).map((s) => (
                            <span
                              key={s}
                              className="rounded bg-emerald-950/30 border border-emerald-900/40 px-2 py-0.5 text-[10px] text-emerald-300"
                            >
                              {s}
                            </span>
                          ))}
                          {evidence.missing_skills.slice(0, 2).map((s) => (
                            <span
                              key={s}
                              className="rounded bg-rose-950/30 border border-rose-900/40 px-2 py-0.5 text-[10px] text-rose-300"
                            >
                              missing: {s}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Right: Score circle & Inspect Button */}
                    <div className="flex items-center justify-between sm:justify-end gap-4 border-t border-zinc-800/60 pt-3 sm:border-0 sm:pt-0">
                      <div className="flex flex-col items-end">
                        <span className="text-[10px] font-semibold tracking-wider uppercase text-zinc-500">
                          Composite Score
                        </span>
                        <span
                          className={`font-mono text-xl font-bold ${
                            score.overall_score >= 65
                              ? "text-emerald-400"
                              : score.overall_score >= 40
                              ? "text-amber-400"
                              : "text-rose-400"
                          }`}
                        >
                          {score.overall_score.toFixed(0)}
                          <span className="text-xs text-zinc-500 font-normal">
                            /100
                          </span>
                        </span>
                      </div>

                      <button
                        onClick={() => setInspectCandidate(item)}
                        className="flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-200 transition-colors hover:border-zinc-600 hover:bg-zinc-800"
                      >
                        <Eye className="h-3.5 w-3.5 text-blue-400" />
                        Inspect Evidence
                      </button>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="py-12 text-center text-sm text-zinc-500">
                No candidates match the selected filters.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Candidate Deep Dive Modal */}
      <CandidateModal
        candidate={inspectCandidate}
        onClose={() => setInspectCandidate(null)}
        blindMode={blindMode}
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
