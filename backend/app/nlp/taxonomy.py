import re
from typing import Set, List

# Comprehensive tech skill taxonomy categorized by domain
SKILL_TAXONOMY = {
    # Programming Languages
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "golang",
    "rust", "kotlin", "swift", "scala", "ruby", "php", "sql", "bash", "shell",
    "r", "dart", "elixir",

    # Frontend Frameworks & Libraries
    "react", "react.js", "next.js", "nextjs", "vue", "vue.js", "angular",
    "svelte", "sveltekit", "tailwind", "tailwindcss", "html", "html5",
    "css", "css3", "sass", "scss", "redux", "zustand", "graphql", "webpack", "vite",

    # Backend & Distributed Systems
    "node", "node.js", "express", "express.js", "django", "flask", "fastapi",
    "spring", "spring boot", "ruby on rails", "asp.net", "grpc", "rest", "restful api",
    "microservices", "rabbitmq", "celery", "event-driven architecture",

    # AI, Machine Learning & Data Science
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "artificial intelligence", "generative ai", "llm",
    "large language models", "rag", "langchain", "llamaindex", "pytorch",
    "tensorflow", "keras", "scikit-learn", "sklearn", "pandas", "numpy",
    "scipy", "huggingface", "transformers", "xgboost", "lightgbm",
    "data science", "data analysis", "statistics", "feature engineering",

    # Big Data & Data Engineering
    "spark", "apache spark", "pyspark", "hadoop", "kafka", "apache kafka",
    "airflow", "apache airflow", "dbt", "databricks", "snowflake", "bigquery",
    "flink", "etl", "elt", "data pipeline", "data warehouse", "lakehouse",

    # Databases & Caches
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "sqlite", "neo4j", "milvus", "qdrant", "pinecone",
    "chromadb", "supabase", "firebase",

    # Cloud, DevOps & Infrastructure
    "aws", "amazon web services", "gcp", "google cloud", "azure", "docker",
    "kubernetes", "k8s", "terraform", "ansible", "helm", "ci/cd",
    "github actions", "gitlab ci", "jenkins", "linux", "prometheus",
    "grafana", "datadog", "serverless", "cloudformation",

    # Methodologies & Tools
    "agile", "scrum", "git", "jira", "tdd", "unit testing", "system design",
    "object-oriented programming", "oop", "clean architecture"
}

# Industry Domains
DOMAIN_TAXONOMY = {
    "finance", "fintech", "banking", "healthcare", "healthtech", "medical",
    "e-commerce", "ecommerce", "retail", "saas", "cybersecurity", "telecom",
    "telecommunications", "automotive", "gaming", "media", "edtech", "education",
    "logistics", "supply chain", "insurance", "insurtech", "real estate",
    "proptech", "energy", "cleantech", "biotech", "adtech", "consulting"
}


def extract_skills_from_text(text: str) -> Set[str]:
    """Identify taxonomy skills in text using word-boundary matching."""
    text_lower = text.lower()
    matched = set()
    for skill in SKILL_TAXONOMY:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            matched.add(skill)
    return matched


def extract_domains_from_text(text: str) -> List[str]:
    """Identify industry domains in text."""
    text_lower = text.lower()
    matched = []
    for domain in DOMAIN_TAXONOMY:
        pattern = r"\b" + re.escape(domain) + r"\b"
        if re.search(pattern, text_lower):
            matched.append(domain)
    return matched
