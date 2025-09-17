import io
import re
from typing import Dict

from PyPDF2 import PdfReader
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    # Optional, more robust text extraction fallback
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None


def extract_text(uploaded_file) -> str:
    """Extract plain text from a Streamlit UploadedFile PDF.

    Uses PyPDF2 first; if that yields no text, falls back to PyMuPDF, then pdfminer.six when available.
    Returns an empty string if all methods fail.
    """
    if uploaded_file is None:
        return ""

    try:
        # Read bytes and keep the UploadedFile reusable
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)

        if not raw_bytes:
            return ""

        # Primary: PyPDF2 (fast)
        text_out = ""
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            texts = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                texts.append(page_text)
            text_out = "\n".join(texts).strip()
        except Exception:
            text_out = ""

        # Fallback 1: PyMuPDF
        if (not text_out) and fitz is not None:
            try:
                doc = fitz.open(stream=raw_bytes, filetype="pdf")
                texts2 = []
                for page in doc:
                    try:
                        ptxt = page.get_text("text") or ""
                    except Exception:
                        ptxt = ""
                    texts2.append(ptxt)
                text_out = "\n".join(texts2).strip()
            except Exception:
                pass

        # Fallback 2: pdfminer.six (robust on some PDFs and forms)
        if (not text_out) and pdfminer_extract_text is not None:
            try:
                text_out = (pdfminer_extract_text(io.BytesIO(raw_bytes)) or "").strip()
            except Exception:
                text_out = text_out  # keep previous value

        return text_out or ""
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


def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _extract_skills_block(text: str) -> str:
    # Try to capture text following a Skills heading
    m = re.search(r"(?is)\bskills\b[\s:\-]*([\s\S]{0,500})", text)
    if not m:
        return ""
    blob = m.group(1)
    # Stop at next common section heading to avoid overrun
    blob = re.split(r"(?i)\n\s*(education|experience|projects?|work history|summary|objective|profile|certifications?)\b", blob)[0]
    tokens = re.split(r"[\n,;\|\u2022\-]+", blob)
    skills = []
    seen = set()
    for t in tokens:
        t = _clean_spaces(t)
        if not t:
            continue
        if len(t) > 40:
            continue
        if t.lower() in ("skills", "technical skills"):
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            skills.append(t)
    return "; ".join(skills[:12])


def heuristic_extract(text: str) -> Dict[str, str]:
    """Lightweight, manual extractor using regex and simple heuristics.

    Returns a dict with keys: Name, Email, Skills, Role, Experience
    """
    if not text:
        return {
            "Name": "Not found",
            "Email": "Not found",
            "Skills": "Not found",
            "Role": "Not found",
            "Experience": "Not found",
        }

    # Email
    email = "Not found"
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if m:
        email = m.group(0)

    # Top lines for name/role
    lines = [ln.strip() for ln in re.split(r"\r?\n", text) if ln.strip()]
    top = lines[:15]

    # Name: first reasonable line that is not a heading
    name = "Not found"
    for ln in top:
        lcl = ln.lower()
        if any(b in lcl for b in ["resume", "curriculum", "vitae", "profile", "objective", "summary", "skills"]):
            continue
        if "@" in ln or any(ch.isdigit() for ch in ln):
            continue
        if len(ln) < 2 or len(ln) > 60:
            continue
        name = ln
        break

    # Role: look for typical titles in the top lines
    role = "Not found"
    title_pat = re.compile(r"(?i)\b(software|backend|front\s*end|full\s*stack|data|ml|ai|devops|cloud|mobile|android|ios|qa|test|automation|security|network|web)\b.*\b(engineer|developer|analyst|scientist|architect|manager|consultant|designer|administrator|lead|intern)\b")
    for ln in top:
        if title_pat.search(ln):
            role = ln
            break

    # Skills: from a skills block if present
    skills = _extract_skills_block(text) or "Not found"

    # Experience: pick the largest X years pattern
    exp_val = "Not found"
    years = re.findall(r"(?i)(\d{1,2}(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)\b", text)
    if years:
        try:
            mx = max(float(y) for y in years)
            if mx.is_integer():
                exp_val = f"{int(mx)} years"
            else:
                exp_val = f"{mx} years"
        except Exception:
            pass

    return {
        "Name": name or "Not found",
        "Email": email or "Not found",
        "Skills": skills or "Not found",
        "Role": role or "Not found",
        "Experience": exp_val or "Not found",
    }