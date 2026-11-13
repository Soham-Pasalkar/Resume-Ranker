import { RankedCandidate, RankingResponse } from "@/types";

export const SAMPLE_JOB_DESCRIPTION = `Staff Backend Distributed Systems Engineer
We are seeking an experienced Staff Backend Engineer to lead the architecture of our real-time transactional payment platform.
Requirements:
- 6+ years of professional engineering experience building high-throughput microservices.
- Mastery of Python, Go, and PostgreSQL with deep knowledge of ACID guarantees and replication.
- Extensive production hands-on experience with Docker, Kubernetes, and AWS (EKS, RDS, S3).
- Demonstrated experience in Redis caching, Apache Kafka event streaming, and CI/CD pipelines.
- Bachelor's or Master's degree in Computer Science or equivalent.
- Financial technology (Fintech) or high-scale payment gateway domain experience is highly preferred.`;

export const MOCK_RANKING_RESPONSE: RankingResponse = {
  summary: {
    total_candidates: 4,
    strong_matches: 2,
    average_score: 68.4,
    top_score: 91.2,
    knockouts_count: 1,
  },
  results: [
    {
      rank: 1,
      candidate: {
        id: "cand-alex-rivera",
        filename: "Alex_Rivera_Staff_Engineer.pdf",
        name: "Alex Rivera",
        contact: {
          email: "alex.rivera@example.com",
          phone: "+1 (415) 890-1234",
          linkedin: "linkedin.com/in/alexriveradev",
          github: "github.com/alexrivera",
          location: "San Francisco, CA",
        },
        detected_yoe: 7.2,
        detected_skills: [
          "python",
          "go",
          "postgresql",
          "docker",
          "kubernetes",
          "aws",
          "kafka",
          "redis",
          "ci/cd",
          "sql",
          "linux",
          "microservices",
        ],
        detected_domains: ["fintech", "banking", "finance"],
        highest_degree: "Master's",
        institution: "Stanford University",
        graduation_year: 2017,
        sections: {
          experience:
            "Staff Platform Engineer at FinTech Global (2020-Present). Architected distributed payment processing gateway in Python and Go processing 15,000 transactions/sec with 99.999% SLA. Migrated monolithic ledger to event-driven Apache Kafka and PostgreSQL architecture on AWS EKS.",
          skills:
            "Python, Go, PostgreSQL, Kafka, Kubernetes, Docker, AWS, Redis, Terraform, CI/CD",
          education:
            "M.S. in Computer Science | Stanford University (2017)\nB.S. in Software Engineering | UC Berkeley (2015)",
        },
        raw_text: "Alex Rivera resume text...",
        is_anonymized: false,
      },
      score: {
        overall_score: 91.2,
        semantic_score: 94.0,
        skill_score: 92.5,
        experience_score: 100.0,
        domain_score: 90.0,
        rrf_score: 98.4,
        match_level: "Strong Match",
      },
      evidence: {
        matched_skills: [
          "python",
          "go",
          "postgresql",
          "docker",
          "kubernetes",
          "aws",
          "kafka",
          "redis",
          "ci/cd",
        ],
        missing_skills: [],
        experience_note: "Exceeds requirement: 7.2 yrs vs 6.0 yrs required",
        domain_note: "Direct high-scale Fintech & Payment Gateway background verified",
        rationale:
          "Alex Rivera is an exceptional match (91.2/100) holding a Master's degree. Demonstrates 7+ years of senior backend experience with comprehensive alignment across Python, Go, distributed Kafka event-streams, and Kubernetes on AWS.",
        key_citations: [
          "Architected distributed payment processing gateway in Python and Go processing 15,000 transactions/sec.",
          "Migrated monolithic ledger to event-driven Apache Kafka and PostgreSQL architecture on AWS EKS.",
          "Containerized services with Docker and automated zero-downtime CI/CD workflows.",
        ],
        interview_questions: [
          "Can you walk through how you maintained data consistency and idempotency across distributed transactions with Kafka and PostgreSQL?",
          "How did you structure fault tolerance and dead-letter queues during peak transactional bursts?",
        ],
      },
      knockout: {
        is_knocked_out: false,
        reasons: [],
      },
    },
    {
      rank: 2,
      candidate: {
        id: "cand-elena-rostova",
        filename: "Elena_Rostova_Resume.pdf",
        name: "Elena Rostova",
        contact: {
          email: "elena.rostova@example.com",
          phone: "+1 (206) 456-7890",
          linkedin: "linkedin.com/in/elenarostova",
          github: "github.com/erostova",
          location: "Seattle, WA",
        },
        detected_yoe: 6.0,
        detected_skills: [
          "python",
          "postgresql",
          "aws",
          "docker",
          "redis",
          "sql",
          "microservices",
          "fastapi",
        ],
        detected_domains: ["saas", "ecommerce"],
        highest_degree: "Bachelor's",
        institution: "University of Washington",
        graduation_year: 2018,
        sections: {
          experience:
            "Senior Backend Engineer at CloudScale SaaS (2018-Present). Built REST and gRPC microservices in Python (FastAPI) and PostgreSQL backed by Redis caches and AWS ECS.",
          skills: "Python, FastAPI, PostgreSQL, Redis, Docker, AWS, SQL",
        },
        raw_text: "Elena Rostova resume text...",
        is_anonymized: false,
      },
      score: {
        overall_score: 72.8,
        semantic_score: 78.5,
        skill_score: 73.0,
        experience_score: 95.0,
        domain_score: 60.0,
        rrf_score: 74.2,
        match_level: "Strong Match",
      },
      evidence: {
        matched_skills: [
          "python",
          "postgresql",
          "aws",
          "docker",
          "redis",
          "sql",
        ],
        missing_skills: ["go", "kafka", "kubernetes"],
        experience_note: "Meets requirement: exactly 6.0 yrs professional experience",
        domain_note: "Cloud SaaS & E-commerce background; fintech transferrable",
        rationale:
          "Elena Rostova is a strong contender (72.8/100) with solid Python microservice foundations. Gaps in Go and Kafka can be evaluated during technical deep-dive.",
        key_citations: [
          "Built REST and gRPC microservices in Python (FastAPI) and PostgreSQL backed by Redis caches.",
          "Managed relational database query tuning and indexing in PostgreSQL on AWS RDS.",
        ],
        interview_questions: [
          "The role relies on Go and Kafka for event streaming; how quickly have you onboarded onto concurrent systems languages in the past?",
          "How do you approach latency optimization when caching relational database records in Redis?",
        ],
      },
      knockout: {
        is_knocked_out: false,
        reasons: [],
      },
    },
    {
      rank: 3,
      candidate: {
        id: "cand-marcus-chen",
        filename: "Marcus_Chen_CV.pdf",
        name: "Marcus Chen",
        contact: {
          email: "marcus.chen@example.com",
          phone: "+1 (650) 332-9011",
          linkedin: "linkedin.com/in/marcuschen",
          location: "San Jose, CA",
        },
        detected_yoe: 4.5,
        detected_skills: ["python", "docker", "sql", "git"],
        detected_domains: ["healthcare"],
        highest_degree: "Bachelor's",
        institution: "San Jose State University",
        graduation_year: 2020,
        sections: {
          experience:
            "Software Engineer at HealthPulse Systems (2020-Present). Maintained internal Python analytics portals and batch SQL reporting pipelines.",
        },
        raw_text: "Marcus Chen resume text...",
        is_anonymized: false,
      },
      score: {
        overall_score: 48.2,
        semantic_score: 52.0,
        skill_score: 45.0,
        experience_score: 65.0,
        domain_score: 40.0,
        rrf_score: 46.0,
        match_level: "Moderate Match",
      },
      evidence: {
        matched_skills: ["python", "docker", "sql"],
        missing_skills: [
          "go",
          "postgresql",
          "kubernetes",
          "aws",
          "kafka",
          "redis",
        ],
        experience_note: "Short by 1.5 yrs: has 4.5 yrs vs 6.0 yrs required",
        domain_note: "Healthcare analytics focus",
        rationale:
          "Marcus Chen displays moderate alignment (48.2/100). Background is skewed toward batch data reporting rather than high-concurrency distributed payment infrastructure.",
        key_citations: [
          "Maintained internal Python analytics portals and batch SQL reporting pipelines.",
        ],
        interview_questions: [
          "Have you architected services with strict high-availability requirements and zero downtime deployments?",
        ],
      },
      knockout: {
        is_knocked_out: false,
        reasons: [],
      },
    },
    {
      rank: 4,
      candidate: {
        id: "cand-jordan-taylor",
        filename: "Jordan_Taylor_Intern.pdf",
        name: "Jordan Taylor",
        contact: {
          email: "jordan.t@example.com",
          phone: "+1 (510) 902-1144",
          github: "github.com/jordantaylor",
        },
        detected_yoe: 1.2,
        detected_skills: ["python", "html", "css", "javascript"],
        detected_domains: ["media"],
        highest_degree: "Associate",
        institution: "City College",
        graduation_year: 2023,
        sections: {
          experience:
            "Junior Web Intern at Digital Press (2023-2024). Created HTML widgets and assisted with basic Python web scrapers.",
        },
        raw_text: "Jordan Taylor resume text...",
        is_anonymized: false,
      },
      score: {
        overall_score: 18.5,
        semantic_score: 22.0,
        skill_score: 15.0,
        experience_score: 18.0,
        domain_score: 20.0,
        rrf_score: 12.0,
        match_level: "Knocked Out",
      },
      evidence: {
        matched_skills: ["python"],
        missing_skills: [
          "go",
          "postgresql",
          "kubernetes",
          "aws",
          "kafka",
          "docker",
          "redis",
        ],
        experience_note: "Short by 4.8 yrs: has 1.2 yrs vs 6.0 yrs required",
        domain_note: "Media web development",
        rationale:
          "Jordan Taylor is knocked out due to substantial seniority and degree requirement gaps for this Staff-level position.",
        key_citations: [
          "Created HTML widgets and assisted with basic Python web scrapers.",
        ],
        interview_questions: [
          "This role targets Staff level engineering; consider candidate for Junior front-end or associate roles instead.",
        ],
      },
      knockout: {
        is_knocked_out: true,
        reasons: [
          "Does not meet minimum experience: has 1 yr, required 6 yrs",
          "Missing mandatory distributed systems competencies (Kafka, Kubernetes, Go)",
        ],
      },
    },
  ],
};
