
import os
import json
import logging
from typing import Dict, Optional

from groq import Groq

logger = logging.getLogger("ats_resume_scorer")

GROQ_MODEL = "openai/gpt-oss-120b"

# Groq client
_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client

    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set."
            )

        _client = Groq(
            api_key=api_key,
            timeout=60.0,
            max_retries=1
        )

    return _client


# =========================================================
# RESUME PARSING PROMPTS
# =========================================================

RESUME_SYSTEM_PROMPT = (
    "You are a professional resume parser. "
    "Extract information from the resume and return "
    "ONLY a valid JSON object. "
    "Do not include explanations or markdown."
)


RESUME_USER_PROMPT = """Extract the following information from this resume.

Return the result as valid JSON using this exact structure:

{
  "name": "full name",
  "email": "email address",
  "phone": "phone number",
  "linkedin": "LinkedIn URL if present, otherwise null",
  "github": "GitHub URL if present, otherwise null",
  "professional_summary": "Full professional summary or empty string",
  "skills": ["list of skills"],
  "experience": [
    {
      "job_title": "",
      "company": "",
      "start_date": "",
      "end_date": "",
      "duration_months": 0,
      "description": ""
    }
  ],
  "education": [
    {
      "degree": "",
      "institution": "",
      "year": ""
    }
  ],
  "certifications": ["list of certifications"],
  "projects": [
    {
      "title": "project name",
      "description": "project description",
      "technologies": ["technologies used"]
    }
  ],
  "action_verbs": ["action verbs used in resume"],
  "keywords": ["important ATS keywords"]
}

Important instructions:

- Extract all technical and soft skills mentioned in the resume.
- Extract projects and their technologies.
- Extract education and certifications.
- Copy the professional summary accurately.
- Calculate experience duration in months when dates are available.
- Do not invent missing information.
- Use empty strings, empty lists, or null for missing information.
- Return ONLY valid JSON without markdown code fences.

Resume Text:

{raw_text}
"""


# =========================================================
# GROQ API CALL
# =========================================================

def _call_groq(
    client: Groq,
    system_prompt: str,
    user_prompt: str
) -> str:

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.0,
        max_tokens=4096
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Groq returned an empty response.")

    return content.strip()


# =========================================================
# JSON PARSING
# =========================================================

def _try_parse_json(text: str) -> Optional[dict]:

    if not text:
        return None

    cleaned = text.strip()

    # Remove markdown code fences
    if cleaned.startswith("```"):

        first_newline = cleaned.find("\n")

        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)

        if isinstance(result, dict):
            return result

        return None

    except (json.JSONDecodeError, TypeError):
        return None


# =========================================================
# RESUME RESULT VALIDATION
# =========================================================

def _validate_resume_result(result: dict) -> dict:

    if not isinstance(result, dict):
        raise ValueError(
            "Invalid resume parser result. Expected a JSON object."
        )

    defaults = {
        "name": "",
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "professional_summary": "",
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "action_verbs": [],
        "keywords": [],
    }

    for key, default in defaults.items():

        if key not in result or result[key] is None:
            result[key] = default

        if isinstance(default, list) and not isinstance(
            result[key], list
        ):
            result[key] = default

    # Validate experience entries
    validated_experience = []

    for exp in result.get("experience", []):

        if not isinstance(exp, dict):
            continue

        validated_exp = {
            "job_title": exp.get("job_title") or "",
            "company": exp.get("company") or "",
            "start_date": exp.get("start_date") or "",
            "end_date": exp.get("end_date") or "",
            "duration_months": exp.get("duration_months", 0),
            "description": exp.get("description") or "",
        }

        try:
            validated_exp["duration_months"] = int(
                validated_exp["duration_months"]
            )
        except (ValueError, TypeError):
            validated_exp["duration_months"] = 0

        validated_experience.append(validated_exp)

    result["experience"] = validated_experience

    # Validate education entries
    validated_education = []

    for education in result.get("education", []):

        if not isinstance(education, dict):
            continue

        validated_education.append({
            "degree": education.get("degree") or "",
            "institution": education.get("institution") or "",
            "year": education.get("year") or "",
        })

    result["education"] = validated_education

    # Validate project entries
    validated_projects = []

    for project in result.get("projects", []):

        if not isinstance(project, dict):
            continue

        technologies = project.get("technologies", [])

        if not isinstance(technologies, list):
            technologies = []

        validated_projects.append({
            "title": project.get("title") or "",
            "description": project.get("description") or "",
            "technologies": technologies,
        })

    result["projects"] = validated_projects

    return result


# =========================================================
# MAIN RESUME PARSER
# =========================================================

def parse_resume(raw_text: str) -> Dict:

    if not raw_text or not raw_text.strip():
        raise ValueError("Resume text is empty.")

    logger.info("Starting Groq resume parsing.")

    client = _get_client()

    prompt = RESUME_USER_PROMPT.format(
        raw_text=raw_text
    )

    # First attempt
    raw_response = _call_groq(
        client,
        RESUME_SYSTEM_PROMPT,
        prompt
    )

    result = _try_parse_json(raw_response)

    # Correct condition: valid JSON should be returned immediately
    if result is not None:

        logger.info("Resume parsed successfully on first attempt.")

        return _validate_resume_result(result)

    # Retry with stricter instructions
    logger.warning(
        "Groq returned invalid JSON. Retrying resume parsing."
    )

    strict_prompt = (
        "Your previous response was not valid JSON. "
        "Return ONLY one valid raw JSON object. "
        "Do not include markdown, explanations, or code fences.\n\n"
        + prompt
    )

    raw_response = _call_groq(
        client,
        RESUME_SYSTEM_PROMPT,
        strict_prompt
    )

    result = _try_parse_json(raw_response)

    if result is not None:

        logger.info("Resume parsed successfully on retry.")

        return _validate_resume_result(result)

    raise ValueError(
        "Groq returned invalid JSON after two attempts. "
        f"Response: {raw_response[:500]}"
    )


# =========================================================
# JOB DESCRIPTION PARSING PROMPTS
# =========================================================

JD_SYSTEM_PROMPT = (
    "You are a professional job description parser. "
    "Extract information and return ONLY a valid JSON object. "
    "Do not include explanations or markdown."
)


JD_USER_PROMPT = """Extract the following information from this job description.

Return valid JSON using this exact structure:

{
  "job_title": "",
  "required_skills": ["list of required skills"],
  "preferred_skills": ["list of preferred skills"],
  "experience_required": "",
  "education_required": "",
  "key_responsibilities": ["list of responsibilities"],
  "keywords": ["important ATS keywords"]
}

Important instructions:

- Identify required and preferred skills separately.
- Extract experience requirements.
- Extract education requirements.
- Extract key responsibilities.
- Extract all important ATS keywords.
- Do not invent missing information.
- Return ONLY valid JSON without markdown code fences.

Job Description Text:

{raw_text}
"""


# =========================================================
# JOB DESCRIPTION RESULT VALIDATION
# =========================================================

def _validate_jd_result(result: dict) -> dict:

    if not isinstance(result, dict):
        raise ValueError(
            "Invalid job description parser result."
        )

    defaults = {
        "job_title": "",
        "required_skills": [],
        "preferred_skills": [],
        "experience_required": "",
        "education_required": "",
        "key_responsibilities": [],
        "keywords": [],
    }

    for key, default in defaults.items():

        if key not in result or result[key] is None:
            result[key] = default

        if isinstance(default, list) and not isinstance(
            result[key], list
        ):
            result[key] = default

    return result


# =========================================================
# MAIN JOB DESCRIPTION PARSER
# =========================================================

def parse_job_description(raw_text: str) -> Dict:

    if not raw_text or not raw_text.strip():
        raise ValueError("Job description text is empty.")

    logger.info("Starting Groq job description parsing.")

    client = _get_client()

    prompt = JD_USER_PROMPT.format(
        raw_text=raw_text
    )

    # First attempt
    raw_response = _call_groq(
        client,
        JD_SYSTEM_PROMPT,
        prompt
    )

    result = _try_parse_json(raw_response)

    if result is not None:

        logger.info(
            "Job description parsed successfully on first attempt."
        )

        return _validate_jd_result(result)

    # Retry
    logger.warning(
        "Groq returned invalid JD JSON. Retrying."
    )

    strict_prompt = (
        "Return ONLY a valid raw JSON object. "
        "No markdown, no explanation, no code fences.\n\n"
        + prompt
    )

    raw_response = _call_groq(
        client,
        JD_SYSTEM_PROMPT,
        strict_prompt
    )

    result = _try_parse_json(raw_response)

    if result is not None:

        logger.info(
            "Job description parsed successfully on retry."
        )

        return _validate_jd_result(result)

    raise ValueError(
        "Groq returned invalid job description JSON after two attempts. "
        f"Response: {raw_response[:500]}"
    )
