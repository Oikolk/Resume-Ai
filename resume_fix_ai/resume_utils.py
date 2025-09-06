import io
import re
from typing import Dict

from PyPDF2 import PdfReader


def extract_text(uploaded_file) -> str:
    """Extract plain text from a Streamlit UploadedFile PDF.

    Returns an empty string if extraction fails.
    """
    if uploaded_file is None:
        return ""

    try:
        # Read bytes and keep the UploadedFile reusable
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        if not raw_bytes:
            return ""

        reader = PdfReader(io.BytesIO(raw_bytes))
        texts = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            texts.append(page_text)
        return "\n".join(texts).strip()
    except Exception:
        return ""


def _strip_fences(s: str) -> str:
    s = s.strip()
    # Remove markdown code fences/backticks if present
    s = re.sub(r"^```[a-zA-Z0-9]*", "", s)
    s = s.replace("```", "")
    s = s.strip("`")
    return s.strip()


def parse_gemini_csv_line(line: str) -> Dict[str, str]:
    """Parse a single-line Gemini output into a structured dict.

    Expected format (but we handle minor deviations):
    Name: John, Email: john@x.com, Skills: Python; SQL, Role: Data Analyst, Experience: 3 years
    """
    result = {
        "Name": "Not found",
        "Email": "Not found",
        "Skills": "Not found",
        "Role": "Not found",
        "Experience": "Not found",
    }

    if not line:
        return result

    s = _strip_fences(line)
    s = re.sub(r"[\r\n]+", " ", s).strip()

    # Primary label-based extraction (robust to ordering)
    patterns = {
        "Name": r"Name\s*:\s*(.*?)(?:,\s*[A-Z][a-zA-Z]+\s*:|$)",
        "Email": r"Email\s*:\s*([^,\n]+)",
        "Skills": r"Skills\s*:\s*(.*?)(?:,\s*[A-Z][a-zA-Z]+\s*:|$)",
        "Role": r"Role\s*:\s*(.*?)(?:,\s*[A-Z][a-zA-Z]+\s*:|$)",
        "Experience": r"Experience\s*:\s*([^,\n]+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            # Remove surrounding brackets if present
            value = value.strip("[]{}() ")
            if value:
                result[key] = value

    # Fallback email finder if the labeled parse failed
    if result["Email"] == "Not found":
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", s)
        if m:
            result["Email"] = m.group(0)

    # Normalize skills separator to semicolons
    if result["Skills"] != "Not found":
        skills = result["Skills"].strip()
        # If comma-separated and looks like a list, convert to semicolons
        if ";" not in skills and "," in skills and len(skills) < 200:
            parts = [p.strip() for p in skills.split(",") if p.strip()]
            if len(parts) >= 2:
                skills = "; ".join(parts)
        result["Skills"] = skills

    # Clean experience to preserve the original but trim whitespace
    if result["Experience"] != "Not found":
        result["Experience"] = result["Experience"].strip()

    # Final safety trims
    for k, v in list(result.items()):
        result[k] = (v or "Not found").strip()

    return result