from typing import List, Set


def generate_interview_questions(
    missing_skills: Set[str],
    matched_skills: Set[str],
    jd_yoe: float,
    resume_yoe: float
) -> List[str]:
    """
    Synthesize targeted technical screening questions addressing gaps
    and validating candidate competencies against the job description.
    """
    questions: List[str] = []

    # Questions based on key missing skills
    for skill in sorted(missing_skills)[:3]:
        s_cap = skill.title()
        questions.append(
            f"The role relies on {s_cap}, which wasn't prominently featured on your resume. Could you share your hands-on exposure or adjacent experience with {s_cap}?"
        )

    # Question on experience level
    if jd_yoe > 0 and resume_yoe < jd_yoe:
        diff = jd_yoe - resume_yoe
        questions.append(
            f"This role targets {jd_yoe:.0f} years of seniority, while your background reflects ~{resume_yoe:.0f} years. What high-impact projects demonstrated your ability to deliver at a senior level?"
        )
    elif matched_skills:
        top_skill = sorted(matched_skills)[0].title()
        questions.append(
            f"Can you walk through the architectural tradeoffs and challenges you encountered when scaling applications using {top_skill}?"
        )

    # General domain or system design question
    questions.append(
        "Describe a production incident or bug you diagnosed recently: how did you isolate the root cause and ensure long-term stability?"
    )

    return questions[:4]
