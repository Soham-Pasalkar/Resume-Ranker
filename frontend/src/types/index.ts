export interface ContactInfo {
  email?: string | null;
  phone?: string | null;
  linkedin?: string | null;
  github?: string | null;
  location?: string | null;
}

export interface CandidateProfile {
  id: string;
  filename: string;
  name: string;
  contact: ContactInfo;
  detected_yoe: number;
  detected_skills: string[];
  detected_domains: string[];
  highest_degree?: string | null;
  institution?: string | null;
  graduation_year?: number | null;
  sections: Record<string, string>;
  raw_text: string;
  is_anonymized: boolean;
}

export interface WeightProfile {
  semantic: number;
  skills: number;
  experience: number;
  domain: number;
}

export interface KnockoutRules {
  mandatory_skills: string[];
  min_yoe?: number | null;
  min_degree?: string | null;
  apply_penalty_instead?: boolean;
}

export interface KnockoutStatus {
  is_knocked_out: boolean;
  reasons: string[];
}

export interface ScoreBreakdown {
  overall_score: number;
  semantic_score: number;
  skill_score: number;
  experience_score: number;
  domain_score: number;
  rrf_score?: number | null;
  match_level: string;
}

export interface CandidateEvidence {
  matched_skills: string[];
  missing_skills: string[];
  experience_note: string;
  domain_note: string;
  rationale: string;
  key_citations: string[];
  interview_questions: string[];
}

export interface RankedCandidate {
  rank: number;
  candidate: CandidateProfile;
  score: ScoreBreakdown;
  evidence: CandidateEvidence;
  knockout: KnockoutStatus;
}

export interface RankingSummary {
  total_candidates: number;
  strong_matches: number;
  average_score: number;
  top_score: number;
  knockouts_count: number;
}

export interface RankingResponse {
  summary: RankingSummary;
  results: RankedCandidate[];
}

export interface JobDescriptionAnalysis {
  title?: string | null;
  skills_detected: string[];
  required_yoe: number;
  domains_detected: string[];
  top_keywords: string[];
  word_count: number;
}
