import {
  RankingResponse,
  CandidateProfile,
  JobDescriptionAnalysis,
  WeightProfile,
  KnockoutRules,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}

export async function analyzeJobDescription(
  text: string,
  title?: string,
  mandatorySkills?: string[]
): Promise<JobDescriptionAnalysis> {
  const res = await fetch(`${API_BASE}/api/v1/jobs/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      title: title || undefined,
      mandatory_skills: mandatorySkills || [],
    }),
  });
  if (!res.ok) {
    throw new Error(`Failed to analyze JD: ${res.statusText}`);
  }
  return res.json();
}

export async function uploadResumes(
  files: File[],
  anonymize: boolean = false
): Promise<CandidateProfile[]> {
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));

  const url = `${API_BASE}/api/v1/resumes/upload?anonymize=${anonymize}`;
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`Failed to upload resumes: ${res.statusText}`);
  }
  return res.json();
}

export async function rankCandidates(params: {
  jd_text: string;
  preset_profile?: string;
  weights?: WeightProfile;
  knockout_rules?: KnockoutRules;
  blind_hiring_mode?: boolean;
  min_score_filter?: number;
}): Promise<RankingResponse> {
  const res = await fetch(`${API_BASE}/api/v1/ranking/rank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`Failed to rank candidates: ${res.statusText}`);
  }
  return res.json();
}

export async function exportRankingCsv(params: {
  jd_text: string;
  preset_profile?: string;
  knockout_rules?: KnockoutRules;
  blind_hiring_mode?: boolean;
}): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/v1/ranking/export/csv`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    throw new Error(`Failed to export CSV: ${res.statusText}`);
  }
  return res.blob();
}
