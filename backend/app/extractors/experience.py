import re
from datetime import datetime
from typing import List, Tuple

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12
}

CURRENT_YEAR = datetime.now().year

# Explicit pattern: "5+ years", "3-5 years experience", "4 years of experience"
EXPLICIT_YOE_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:–|-|to)\s*(\d+(?:\.\d+)?)\s*years?", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\+\s*years?", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*years?\s*(?:of\s*)?experience", re.IGNORECASE),
    re.compile(r"experience\s*:\s*(\d+(?:\.\d+)?)\s*years?", re.IGNORECASE),
]

# Date range: e.g. "Jan 2019 - Present", "2018 – 2022", "06/2017 to 08/2021"
DATE_RANGE_PATTERN = re.compile(
    r"(?P<start_month>[a-zA-Z]{3,9}\.?\s*|\d{1,2}/)?(?P<start_year>19\d{2}|20\d{2})"
    r"\s*(?:–|-|—|to)\s*"
    r"(?P<end_pres>present|current|now|ongoing|(?P<end_month>[a-zA-Z]{3,9}\.?\s*|\d{1,2}/)?(?P<end_year>19\d{2}|20\d{2}))",
    re.IGNORECASE
)


def extract_years_of_experience(text: str) -> float:
    """
    Compute total years of experience combining:
      1. Explicit statements ("X years of experience")
      2. Sum of parsed non-overlapping date ranges
    """
    found_explicit = []
    for pat in EXPLICIT_YOE_PATTERNS:
        for m in pat.finditer(text):
            groups = [float(g) for g in m.groups() if g is not None]
            if groups:
                found_explicit.append(sum(groups) / len(groups))

    # Parse date ranges
    intervals: List[Tuple[float, float]] = []
    for m in DATE_RANGE_PATTERN.finditer(text):
        start_yr = int(m.group("start_year"))
        if start_yr < 1980 or start_yr > CURRENT_YEAR + 1:
            continue

        end_pres = m.group("end_pres")
        if end_pres and re.match(r"present|current|now|ongoing", end_pres, re.IGNORECASE):
            end_yr = CURRENT_YEAR + (datetime.now().month / 12.0)
        else:
            end_yr_str = m.group("end_year")
            if end_yr_str:
                end_yr = int(end_yr_str)
            else:
                continue

        if end_yr >= start_yr and (end_yr - start_yr) <= 40:
            intervals.append((float(start_yr), float(end_yr)))

    # Merge overlapping intervals
    date_based_yoe = 0.0
    if intervals:
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for cur in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if cur[0] <= prev_end:
                merged[-1] = (prev_start, max(prev_end, cur[1]))
            else:
                merged.append(cur)
        date_based_yoe = sum(end - start for start, end in merged)

    candidates = [date_based_yoe] + found_explicit
    candidates = [c for c in candidates if 0.0 < c <= 45.0]

    if not candidates:
        return 0.0

    # Max credible YOE
    return round(max(candidates), 1)
