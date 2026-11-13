import pytest

@pytest.fixture
def sample_jd_text() -> str:
    return """
    Senior Python Backend Engineer
    We are looking for a Senior Software Engineer with 5+ years of experience in Python,
    FastAPI, PostgreSQL, Docker, and Kubernetes. Experience in AWS and building high-scale
    distributed microservices is required. Financial services or fintech background is a plus.
    Must have strong knowledge of SQL, Redis, and CI/CD pipelines.
    """

@pytest.fixture
def sample_senior_resume_text() -> str:
    return """
    Alex Rivera
    alex.rivera@example.com | (555) 234-5678 | San Francisco, CA
    https://linkedin.com/in/alexriveradev | https://github.com/alexrivera

    Summary
    Senior Backend Software Engineer with 6 years of experience building resilient microservices
    using Python, FastAPI, and PostgreSQL across fintech and banking applications.

    Experience
    Senior Software Engineer | FinTech Cloud Corp
    Jan 2021 – Present
    - Architected distributed payment processing services in Python and FastAPI handling 10,000 req/sec.
    - Designed relational schemas and query optimizations in PostgreSQL and Redis cache layers.
    - Containerized microservices using Docker and deployed on Kubernetes clusters within AWS.
    - Established CI/CD automation pipelines via GitHub Actions reducing deployment failures by 40%.

    Backend Engineer | PayGlobal Solutions
    2018 – 2020
    - Built RESTful APIs using Python and Django.
    - Integrated third-party payment gateways and maintained database migrations.

    Education
    B.S. in Computer Science | University of California, Berkeley | 2018

    Skills
    Python, FastAPI, Django, PostgreSQL, Redis, Docker, Kubernetes, AWS, CI/CD, SQL, Git, Linux
    """

@pytest.fixture
def sample_junior_resume_text() -> str:
    return """
    Jordan Lee
    jordan.lee@example.com | 555-987-6543
    https://github.com/jordanlee

    Objective
    Junior web developer seeking an entry-level position.

    Experience
    Web Intern | Local Media Group
    2023 - 2024
    - Created simple HTML/CSS pages and modified basic WordPress templates.

    Skills
    HTML, CSS, JavaScript, Basic Python
    """
