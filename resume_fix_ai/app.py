import os
import io
from typing import List, Dict, Optional
from html import escape as html_escape
import re

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListFlowable, ListItem
from reportlab.lib import colors
from reportlab.lib.units import cm

from resume_utils import extract_text, parse_gemini_csv_line
import streamlit.components.v1 as components
try:
    from streamlit_html_editor import html_editor
except Exception:  # graceful fallback if package not installed
    html_editor = None


# ----- App / Model Setup -----
st.set_page_config(page_title="ResumeFix AI", page_icon="🧠", layout="wide")

APP_TITLE = "ResumeFix AI"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
BUILDER_STEPS = [
    "Personal Detail",
    "Summary",
    "Experience",
    "Projects",
    "Education",
    "Skills",
]


def _init_gemini_once() -> None:
    """Configure Gemini and cache the model or error in session state."""
    if "gemini_initialized" in st.session_state:
        return

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.session_state["gemini_error"] = (
            "Missing GEMINI_API_KEY. Create a .env with GEMINI_API_KEY=your_key_here"
        )
        st.session_state["gemini_initialized"] = True
        return

    try:
        genai.configure(api_key=api_key)
        st.session_state["gemini_model"] = genai.GenerativeModel(MODEL_NAME)
        st.session_state["gemini_error"] = None
    except Exception as ex:
        st.session_state["gemini_error"] = f"Failed to initialize Gemini: {ex}"
    finally:
        st.session_state["gemini_initialized"] = True


def _get_model() -> Optional[object]:
    return st.session_state.get("gemini_model")


def _generate_text(prompt: str) -> str:
    model = _get_model()
    if model is None:
        return ""
    try:
        response = model.generate_content(prompt)
        # google-generativeai returns response.text normally; fall back just in case
        if hasattr(response, "text") and isinstance(response.text, str):
            return response.text.strip()
        return str(response)
    except Exception as ex:
        st.error(f"Gemini API error: {ex}")
        return ""


# ----- Prompts -----
def build_chatbot_prompt(resume_text: str) -> str:
    truncated = (resume_text or "")[:5000]
    return (
        "You are a strict HR recruiter. Analyze this resume and reply in this exact format:\n\n"
        "💯 SCORE: [number]/100\n\n"
        "✅ STRENGTHS:\n\n...\n\n"
        "❌ WEAKNESSES:\n\n...\n\n"
        "💡 3 QUICK FIXES:\n\n...\n...\n...\n\n"
        "🚫 ATS FRIENDLY? Yes/No — [reason]\n\n"
        "Keep it under 150 words. Be direct. No fluff.\n\n"
        f"RESUME: {truncated}"
    )


def build_extractor_prompt(resume_text: str) -> str:
    truncated = (resume_text or "")[:4000]
    return (
        "Extract structured data from this resume. Output in this exact format — one line, comma separated:\n\n"
        "Name: [name], Email: [email], Skills: [skill1; skill2; skill3], Role: [role], Experience: [X] years\n\n"
        "If data not found, write 'Not found'.\n\n"
        f"RESUME TEXT: {truncated}"
    )


# ----- Tool 1: AI Resume Chatbot -----
def render_ai_resume_chatbot() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # List[Dict[str, str]] with keys: role, content

    cols = st.columns([1, 1])
    with cols[1]:
        if st.button("🗑️ Clear Chat"):
            st.session_state["messages"] = []
            st.session_state["processed_file_marker"] = None

    uploaded_file = st.file_uploader("➕ Upload Resume (PDF)", type=["pdf"], key="single_pdf")

    # Show existing messages (if any)
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"]) 

    if uploaded_file is None:
        return

    # Prevent reprocessing the same file on reruns
    file_marker = f"{uploaded_file.name}-{uploaded_file.size}"
    already_processed = st.session_state.get("processed_file_marker") == file_marker

    if not already_processed:
        text = extract_text(uploaded_file)
        if not text.strip():
            st.error("Could not extract text from PDF. Please try another file.")
            return

        # Append user message
        st.session_state["messages"].append({
            "role": "user",
            "content": f"📄 Uploaded: {uploaded_file.name}",
        })

        # Build and send prompt
        prompt = build_chatbot_prompt(text)

        if st.session_state.get("gemini_error"):
            st.error(st.session_state["gemini_error"])
            return

        with st.spinner("Analyzing resume..."):
            ai_text = _generate_text(prompt)

        if not ai_text:
            st.error("No response from AI. Please try again.")
            return

        st.session_state["messages"].append({
            "role": "assistant",
            "content": ai_text,
        })

        st.session_state["processed_file_marker"] = file_marker

        # Re-render the full chat in order after update
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"]) 


# ----- Tool 2: Bulk Resume → CSV -----
def render_bulk_extractor() -> None:
    uploaded_files = st.file_uploader(
        "📤 Upload Multiple Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="multi_pdf",
    )

    if st.button("🚀 Extract → CSV"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF resume.")
            return

        if st.session_state.get("gemini_error"):
            st.error(st.session_state["gemini_error"])
            return

        rows: List[Dict[str, str]] = []
        with st.spinner("Extracting data from resumes..."):
            for uf in uploaded_files:
                text = extract_text(uf)
                if not text.strip():
                    rows.append({
                        "Name": "Not found",
                        "Email": "Not found",
                        "Skills": "Not found",
                        "Role": "Not found",
                        "Experience": "Not found",
                    })
                    continue

                prompt = build_extractor_prompt(text)
                ai_line = _generate_text(prompt)
                parsed = parse_gemini_csv_line(ai_line)
                rows.append(parsed)

        df = pd.DataFrame(rows, columns=["Name", "Email", "Skills", "Role", "Experience"])
        st.subheader("Preview")
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv_bytes,
            file_name="resume_extract.csv",
            mime="text/csv",
        )


# ----- Tool 3: Resume Builder -----
def _ensure_builder_state() -> None:
    if "builder" not in st.session_state:
        st.session_state["builder"] = {
            "title": "",
            "theme_color": "#2ecc71",
            "personal": {
                "first_name": "",
                "last_name": "",
                "job_title": "",
                "address": "",
                "city": "",
                "state": "",
                "phone": "",
                "email": "",
            },
            "summary": "",
            "experiences": [],
            "projects": [],
            "education": [],
            "skills": [],
        }

    # AI suggestion scratch space
    if "builder_ai" not in st.session_state:
        st.session_state["builder_ai"] = {
            "summary": [],           # List[str]
            "exp": {},               # Dict[int, List[str]]
            "prj": {},               # Dict[int, List[str]]
        }


def _builder_theme_picker() -> None:
    # Popover palette like the screenshots (falls back to expander if popover missing)
    def palette():
        colors = [
            "#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#1abc9c", "#e84393", "#fd79a8",
        ]
        cols = st.columns(len(colors))
        for i, hex_c in enumerate(colors):
            with cols[i]:
                if st.button(" ", key=f"theme_{hex_c}", help=hex_c):
                    st.session_state["builder"]["theme_color"] = hex_c
                st.markdown(
                    f"<div style='width:22px;height:22px;border-radius:50%;background:{hex_c};margin-top:-36px;border:1px solid #ddd'></div>",
                    unsafe_allow_html=True,
                )

    if hasattr(st, "popover"):
        with st.popover("Theme"):
            palette()
    else:
        with st.expander("Theme"):
            palette()


def _ai_generate_summary(job_title: str, skills: List[str]) -> str:
    base = (
        "Write three alternative professional summaries (2–3 sentences each). "
        "Do not include labels like 'Variant' or numbering. Plain text only (no markdown/HTML). "
        "Separate each alternative by a single line containing only '---'. Avoid placeholders. Be ATS-friendly."
    )
    job_part = f" Job title: {job_title}." if job_title else ""
    skills_part = f" Key skills: {', '.join(skills[:10])}." if skills else ""
    prompt = base + job_part + skills_part
    return _generate_text(prompt)


def _ai_generate_experience_bullets(position: str, tech: List[str]) -> str:
    prompt = (
        "Create three alternative sets of 4–6 resume bullet points for this role. "
        "Do not use labels like 'Variant' or numbering. Plain text only (no markdown/HTML). "
        "Separate each alternative set with a line containing only '---'. Within each set, start each bullet with '-' or '•'."
        f" Role: {position or 'Software Engineer'}."
        f" Tech stack: {', '.join(tech[:12]) if tech else 'N/A'}."
    )
    return _generate_text(prompt)


def _ai_generate_project_bullets(name: str, tech: List[str]) -> str:
    prompt = (
        "Create three alternative sets of 3–5 concise bullets describing a personal project for a resume. "
        "Do not use labels like 'Variant' or numbering. Plain text only (no markdown/HTML). "
        "Separate each alternative set with a line containing only '---'. In each set, put each bullet on its own line beginning with '-' or '•'."
        f" Project: {name or 'Portfolio Website'}."
        f" Tech stack: {', '.join(tech[:12]) if tech else 'N/A'}."
    )
    return _generate_text(prompt)


def _strip_html(text: str) -> str:
    if not text:
        return ""
    # remove HTML tags and code fences
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    # collapse spaces
    return re.sub(r"\s+", " ", text).strip()


def _split_suggestions(raw: str) -> List[str]:
    if not raw:
        return []
    cleaned = raw.replace("\r", "\n").strip()
    # Split on '---' first
    parts = re.split(r"\n\s*---\s*\n", cleaned)
    if len(parts) == 1:
        # Fallback: split on double newlines
        parts = re.split(r"\n\n+", cleaned)
    # Sanitize each part and ensure non-empty
    out: List[str] = []
    for p in parts:
        candidate = _strip_html(p)
        if candidate:
            out.append(candidate.strip())
    # Limit to 5 suggestions
    return out[:5]


def _normalize_bullets(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    norm = []
    for ln in lines:
        ln = re.sub(r"^[•\-\*\u2022\s]+", "", ln)
        norm.append(f"• {ln}")
    return "\n".join(norm)


def _builder_preview_html(builder: Dict) -> str:
    color = builder.get("theme_color", "#2ecc71")
    personal = builder.get("personal", {})
    full_name = f"{personal.get('first_name', '').strip()} {personal.get('last_name', '').strip()}".strip()
    job_title = personal.get("job_title", "")
    city_state = ", ".join([v for v in [personal.get("city", ""), personal.get("state", "")] if v])
    contact_line = " | ".join([v for v in [personal.get("phone", ""), personal.get("email", "")] if v])
    summary = builder.get("summary", "")
    skills = builder.get("skills", [])
    experiences = builder.get("experiences", [])
    projects = builder.get("projects", [])
    education = builder.get("education", [])

    def bullets_to_html(text: str) -> str:
        lines = [ln.strip("•- \t") for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return ""
        items = "".join([f"<li>{html_escape(ln)}</li>" for ln in lines])
        return f"<ul>{items}</ul>"

    # Build block sections
    if experiences:
        experience_items = []
        for exp in experiences:
            exp_block = (
                f"<p class='sub'>{html_escape(exp.get('position_title',''))}</p>"
                f"<p>{html_escape(exp.get('company_name',''))}{', ' if exp.get('company_name') and (exp.get('city') or exp.get('state')) else ''}{html_escape(', '.join([v for v in [exp.get('city',''), exp.get('state','')] if v]))}</p>"
                f"<p style='color:#666;font-size:12px'>{html_escape(exp.get('start_date',''))} {'to' if exp.get('start_date') or exp.get('end_date') else ''} {html_escape(exp.get('end_date',''))}</p>"
                + bullets_to_html(exp.get('summary',''))
            )
            experience_items.append(exp_block)
        experience_block = "<h2>Professional Experience</h2>" + "".join(experience_items)
    else:
        experience_block = ""

    if projects:
        project_items = []
        for prj in projects:
            prj_block = (
                f"<p class='sub'>{html_escape(prj.get('name',''))}</p>"
                f"<p>{html_escape(prj.get('company',''))}</p>"
                + bullets_to_html(prj.get('summary',''))
            )
            project_items.append(prj_block)
        project_block = "<h2>Personal Project</h2>" + "".join(project_items)
    else:
        project_block = ""

    if education:
        education_items = []
        for ed in education:
            ed_block = (
                f"<p class='sub'>{html_escape(ed.get('university',''))}</p>"
                f"<p>{html_escape(ed.get('degree',''))}{' • ' if ed.get('degree') and ed.get('major') else ''}{html_escape(ed.get('major',''))}</p>"
                f"<p style='color:#666;font-size:12px'>{html_escape(ed.get('start_date',''))} {'to' if ed.get('start_date') or ed.get('end_date') else ''} {html_escape(ed.get('end_date',''))}</p>"
                + (f"<p>Grade: {html_escape(ed.get('grade_value',''))}</p>" if ed.get('grade_value') else '')
                + (f"<p>{html_escape(ed.get('description',''))}</p>" if ed.get('description') else '')
            )
            education_items.append(ed_block)
        education_block = "<h2>Education</h2>" + "".join(education_items)
    else:
        education_block = ""

    html = f"""
    <style>
    .resume-wrap {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
    .resume {{ background:#fff; border:1px solid #eee; padding:28px 34px; }}
    .bar {{ height:8px; background:{color}; margin-bottom:18px; }}
    h1 {{ margin:0; font-size:30px; font-weight:700; text-align:center; }}
    .job {{ text-align:center; color:#444; margin-bottom:2px; }}
    .meta {{ text-align:center; color:#666; font-size:12px; margin-bottom:8px; }}
    .rule {{ height:2px; background:{color}; margin:8px 0 14px; opacity:0.8 }}
    h2 {{ color:{color}; font-size:16px; margin:10px 0 8px; text-transform:uppercase; letter-spacing:0.5px; }}
    p {{ margin:0 0 6px; color:#222; }}
    ul {{ margin:4px 0 8px 18px; }}
    .two-col {{ display:flex; gap:24px; }}
    .left {{ flex:1; }}
    .right {{ width:60%; }}
    .sub {{ font-weight:600; }}
    </style>
    <div class="resume-wrap">
      <div class="resume">
        <div class="bar"></div>
        <h1>{html_escape(full_name or '')}</h1>
        <div class="job">{html_escape(job_title or '')}</div>
        <div class="meta">{html_escape(city_state or '')}</div>
        <div class="meta">{html_escape(contact_line)}</div>
        <div class="rule"></div>

        {('<h2>Summary</h2><p>' + html_escape(summary) + '</p>') if summary else ''}

        {('<h2>Skills</h2><p>' + html_escape('; '.join([s for s in skills if s])) + '</p>') if skills else ''}

        {experience_block}

        {project_block}

        {education_block}

      </div>
    </div>
    """
    return html


def _build_pdf_bytes(builder: Dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title_style", parent=styles["Heading1"], alignment=1, fontSize=18, leading=20, spaceAfter=6)
    job_style = ParagraphStyle("job_style", parent=styles["Normal"], alignment=1, textColor=colors.grey, spaceAfter=4)
    meta_style = ParagraphStyle("meta_style", parent=styles["Normal"], alignment=1, textColor=colors.grey, fontSize=9, spaceAfter=6)
    section_style = ParagraphStyle("section_style", parent=styles["Heading2"], textColor=colors.black, fontSize=12, spaceBefore=6, spaceAfter=4, uppercase=True)
    body_style = ParagraphStyle("body_style", parent=styles["Normal"], fontSize=10, leading=13)

    theme_hex = builder.get("theme_color", "#2ecc71")
    try:
        theme_color = colors.HexColor(theme_hex)
    except Exception:
        theme_color = colors.green

    personal = builder.get("personal", {})
    full_name = f"{personal.get('first_name', '').strip()} {personal.get('last_name', '').strip()}".strip()
    job_title = personal.get("job_title", "")
    city_state = ", ".join([v for v in [personal.get("city", ""), personal.get("state", "")] if v])
    contact_line = " | ".join([v for v in [personal.get("phone", ""), personal.get("email", "")] if v])

    story: List = []
    story.append(Paragraph(full_name or "", title_style))
    if job_title:
        story.append(Paragraph(job_title, job_style))
    if city_state:
        story.append(Paragraph(city_state, meta_style))
    if contact_line:
        story.append(Paragraph(contact_line, meta_style))
    story.append(HRFlowable(width="100%", thickness=2, color=theme_color, spaceBefore=6, spaceAfter=6))

    summary = builder.get("summary", "")
    skills = builder.get("skills", [])
    experiences = builder.get("experiences", [])
    projects = builder.get("projects", [])
    education = builder.get("education", [])

    if summary:
        story.append(Paragraph("Summary", section_style))
        # Convert basic HTML tags from editor to ReportLab-friendly markup
        safe_summary = re.sub(r"<\/?(div|span)[^>]*>", "", summary)
        safe_summary = safe_summary.replace("<strong>", "<b>").replace("</strong>", "</b>")
        safe_summary = safe_summary.replace("<em>", "<i>").replace("</em>", "</i>")
        story.append(Paragraph(safe_summary, body_style))
        story.append(Spacer(1, 6))

    if skills:
        story.append(Paragraph("Skills", section_style))
        story.append(Paragraph("; ".join([s for s in skills if s]), body_style))
        story.append(Spacer(1, 6))

    if experiences:
        story.append(Paragraph("Professional Experience", section_style))
        for exp in experiences:
            position = exp.get("position_title", "")
            company = exp.get("company_name", "")
            city = exp.get("city", "")
            state = exp.get("state", "")
            dates = " ".join([v for v in [exp.get("start_date", ""), "to" if exp.get("start_date") or exp.get("end_date") else "", exp.get("end_date", "")] if v])
            headline = ", ".join([v for v in [company, ", ".join([c for c in [city, state] if c])] if v])
            if position:
                story.append(Paragraph(position, ParagraphStyle("sub", parent=body_style, fontName="Helvetica-Bold")))
            if headline:
                story.append(Paragraph(headline, ParagraphStyle("meta_sub", parent=body_style, textColor=colors.grey)))
            if dates:
                story.append(Paragraph(dates, ParagraphStyle("dates", parent=body_style, textColor=colors.grey, fontSize=9)))
            exp_html = exp.get("summary", "") or ""
            exp_html = re.sub(r"<\/?(div|span)[^>]*>", "\n", exp_html)
            exp_html = exp_html.replace("<strong>", "<b>").replace("</strong>", "</b>")
            exp_html = exp_html.replace("<em>", "<i>").replace("</em>", "</i>")
            bullets = [ln.strip("•- \t") for ln in re.split(r"\r?\n", exp_html) if ln.strip()]
            if bullets:
                story.append(ListFlowable([ListItem(Paragraph(b, body_style)) for b in bullets], bulletType="bullet", leftIndent=14))
            story.append(Spacer(1, 4))

    if projects:
        story.append(Paragraph("Personal Project", section_style))
        for prj in projects:
            name = prj.get("name", "")
            company = prj.get("company", "")
            if name:
                story.append(Paragraph(name, ParagraphStyle("sub2", parent=body_style, fontName="Helvetica-Bold")))
            if company:
                story.append(Paragraph(company, ParagraphStyle("meta2", parent=body_style, textColor=colors.grey)))
            prj_html = prj.get("summary", "") or ""
            prj_html = re.sub(r"<\/?(div|span)[^>]*>", "\n", prj_html)
            prj_html = prj_html.replace("<strong>", "<b>").replace("</strong>", "</b>")
            prj_html = prj_html.replace("<em>", "<i>").replace("</em>", "</i>")
            bullets = [ln.strip("•- \t") for ln in re.split(r"\r?\n", prj_html) if ln.strip()]
            if bullets:
                story.append(ListFlowable([ListItem(Paragraph(b, body_style)) for b in bullets], bulletType="bullet", leftIndent=14))
            story.append(Spacer(1, 4))

    if education:
        story.append(Paragraph("Education", section_style))
        for ed in education:
            uni = ed.get("university", "")
            deg = ed.get("degree", "")
            major = ed.get("major", "")
            degree_line = " • ".join([v for v in [deg, major] if v])
            if uni:
                story.append(Paragraph(uni, ParagraphStyle("sub3", parent=body_style, fontName="Helvetica-Bold")))
            if degree_line:
                story.append(Paragraph(degree_line, body_style))
            dates = " ".join([v for v in [ed.get("start_date", ""), "to" if ed.get("start_date") or ed.get("end_date") else "", ed.get("end_date", "")] if v])
            if dates:
                story.append(Paragraph(dates, ParagraphStyle("dates2", parent=body_style, textColor=colors.grey, fontSize=9)))
            if ed.get("grade_value"):
                story.append(Paragraph(f"Grade: {ed.get('grade_value')}", body_style))
            if ed.get("description"):
                story.append(Paragraph(ed.get("description"), body_style))
            story.append(Spacer(1, 4))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def render_resume_builder() -> None:
    _ensure_builder_state()
    builder = st.session_state["builder"]

    # Restore header to the main page
    header_cols = st.columns([1, 2, 5], gap="large")
    with header_cols[0]:
        if st.button("🆕 New Resume"):
            st.session_state.pop("builder", None)
            st.session_state.pop("builder_ai", None)
            _ensure_builder_state()
            builder = st.session_state["builder"]
    with header_cols[1]:
        _builder_theme_picker()
    with header_cols[2]:
        builder["title"] = st.text_input("Title", value=builder.get("title", ""), placeholder="Ex: Backend Resume")

    # Two-column layout with a stepper experience (Prev/Next)
    # Left = form section for current step; Right = live preview
    if "builder_step" not in st.session_state:
        st.session_state["builder_step"] = 0
    steps = ["Personal Detail", "Summary", "Experience", "Projects", "Education", "Skills"]
    step = st.session_state["builder_step"]

    left, right = st.columns([1.3, 1.7], gap="large")

    with left:
        nav = st.columns([1, 3, 1])
        with nav[0]:
            if st.button("⬅ Prev", key="prev_btn", disabled=step == 0):
                st.session_state["builder_step"] = max(0, step - 1)
                st.rerun()
        with nav[1]:
            st.markdown(f"### {steps[step]}")
        with nav[2]:
            if st.button("Next ➡", key="next_btn", disabled=step == len(steps) - 1):
                st.session_state["builder_step"] = min(len(steps) - 1, step + 1)
                st.rerun()

        if step == 0:
            p = builder["personal"]
            c1, c2 = st.columns(2)
            p["first_name"] = c1.text_input("First Name", value=p.get("first_name", ""))
            p["last_name"] = c2.text_input("Last Name", value=p.get("last_name", ""))
            p["job_title"] = st.text_input("Job Title", value=p.get("job_title", ""))
            p["address"] = st.text_input("Address", value=p.get("address", ""))
            c3, c4 = st.columns(2)
            p["phone"] = c3.text_input("Phone", value=p.get("phone", ""))
            p["email"] = c4.text_input("Email", value=p.get("email", ""))
            if st.button("Save", key="save_personal"):
                st.success("Saved")
        elif step == 1:
            if html_editor:
                builder["summary"] = html_editor("Add Summary", value=builder.get("summary", ""), height=260)
            else:
                builder["summary"] = st.text_area("Add Summary", value=builder.get("summary", ""), height=180)
            # Generate from AI and show suggestions below with Apply buttons
            if st.button("Generate from AI", key="gen_sum_expander"):
                ai = _ai_generate_summary(builder["personal"].get("job_title", ""), builder.get("skills", []))
                st.session_state["builder_ai"]["summary"] = _split_suggestions(ai)
            suggs = st.session_state["builder_ai"].get("summary", [])
            if suggs:
                st.markdown("#### Suggestions")
                for i, s in enumerate(suggs):
                    with st.container():
                        st.write(s)
                        if st.button("Apply", key=f"apply_sum_{i}"):
                            builder["summary"] = s
        elif step == 2:
            if st.button("+ Add Experience"):
                builder["experiences"].append({})
            for idx, exp in enumerate(builder["experiences"]):
                with st.container():
                    c1, c2 = st.columns(2)
                    exp["position_title"] = c1.text_input("Position Title", value=exp.get("position_title", ""), key=f"pos_{idx}")
                    exp["company_name"] = c2.text_input("Company Name", value=exp.get("company_name", ""), key=f"comp_{idx}")
                    c3, c4 = st.columns(2)
                    exp["city"] = c3.text_input("City", value=exp.get("city", ""), key=f"city_{idx}")
                    exp["state"] = c4.text_input("State", value=exp.get("state", ""), key=f"state_{idx}")
                    c5, c6 = st.columns(2)
                    exp["start_date"] = c5.text_input("Start Date", value=exp.get("start_date", ""), key=f"sd_{idx}")
                    exp["end_date"] = c6.text_input("End Date", value=exp.get("end_date", ""), key=f"ed_{idx}")
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        if html_editor:
                            exp["summary"] = html_editor("Summary (bullets)", value=exp.get("summary", ""), height=240, key=f"sum_{idx}")
                        else:
                            exp["summary"] = st.text_area("Summary (bullets)", value=exp.get("summary", ""), key=f"sum_{idx}", height=180)
                    with cols[1]:
                        if st.button("Generate from AI", key=f"gen_exp_{idx}"):
                            if st.session_state.get("gemini_error"):
                                st.error(st.session_state["gemini_error"])
                            else:
                                ai = _ai_generate_experience_bullets(exp.get("position_title", ""), builder.get("skills", []))
                                st.session_state["builder_ai"].setdefault("exp", {})[idx] = _split_suggestions(ai)
                    with cols[2]:
                        if st.button("Delete", key=f"del_exp_{idx}"):
                            builder["experiences"].pop(idx)
                            st.rerun()
                # Suggestions for this experience
                suggs = st.session_state.get("builder_ai", {}).get("exp", {}).get(idx, [])
                if suggs:
                    st.markdown("#### Suggestions")
                    for j, s in enumerate(suggs):
                        with st.container():
                            st.write(s)
                            if st.button("Apply", key=f"apply_exp_{idx}_{j}"):
                                exp["summary"] = _normalize_bullets(s)
            if st.button("Save", key="save_exp"):
                st.success("Saved")
        elif step == 3:
            if st.button("+ Add Project"):
                builder["projects"].append({"name": "", "company": "", "summary": ""})
            for idx, prj in enumerate(builder["projects"]):
                with st.container():
                    prj["name"] = st.text_input("Project Name", value=prj.get("name", ""), key=f"prjn_{idx}")
                    prj["company"] = st.text_input("Tech Stack / Company", value=prj.get("company", ""), key=f"prjc_{idx}")
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        if html_editor:
                            prj["summary"] = html_editor("Summary (bullets)", value=prj.get("summary", ""), height=220, key=f"prjs_{idx}")
                        else:
                            prj["summary"] = st.text_area("Summary (bullets)", value=prj.get("summary", ""), key=f"prjs_{idx}", height=160)
                    with cols[1]:
                        if st.button("Generate from AI", key=f"gen_prj_{idx}"):
                            if st.session_state.get("gemini_error"):
                                st.error(st.session_state["gemini_error"])
                            else:
                                ai = _ai_generate_project_bullets(prj.get("name", ""), builder.get("skills", []))
                                st.session_state["builder_ai"].setdefault("prj", {})[idx] = _split_suggestions(ai)
                    with cols[2]:
                        if st.button("Delete", key=f"del_prj_{idx}"):
                            builder["projects"].pop(idx)
                            st.experimental_rerun()
                # Suggestions for this project
                suggs = st.session_state.get("builder_ai", {}).get("prj", {}).get(idx, [])
                if suggs:
                    st.markdown("#### Suggestions")
                    for j, s in enumerate(suggs):
                        with st.container():
                            st.write(s)
                            if st.button("Apply", key=f"apply_prj_{idx}_{j}"):
                                prj["summary"] = _normalize_bullets(s)
            if st.button("Save", key="save_prj"):
                st.success("Saved")
        elif step == 4:
            if st.button("+ Add Education"):
                builder["education"].append({
                    "university": "",
                    "degree": "",
                    "major": "",
                    "start_date": "",
                    "end_date": "",
                    "grade_type": "CGPA",
                    "grade_value": "",
                    "description": "",
                })
            for idx, ed in enumerate(builder["education"]):
                with st.container():
                    ed["university"] = st.text_input("University Name", value=ed.get("university", ""), key=f"edu_u_{idx}")
                    c1, c2 = st.columns(2)
                    ed["degree"] = c1.text_input("Degree", value=ed.get("degree", ""), key=f"edu_d_{idx}")
                    ed["major"] = c2.text_input("Major", value=ed.get("major", ""), key=f"edu_m_{idx}")
                    c3, c4 = st.columns(2)
                    ed["start_date"] = c3.text_input("Start Date", value=ed.get("start_date", ""), key=f"edu_sd_{idx}")
                    ed["end_date"] = c4.text_input("End Date", value=ed.get("end_date", ""), key=f"edu_ed_{idx}")
                    ed["grade_value"] = st.text_input("Grade (e.g., 8.5/10 or 78%)", value=ed.get("grade_value", ""), key=f"edu_g_{idx}")
                    ed["description"] = st.text_area("Description", value=ed.get("description", ""), key=f"edu_desc_{idx}")
                    if st.button("Delete", key=f"del_edu_{idx}"):
                        builder["education"].pop(idx)
                        st.rerun()
                        st.rerun()
            if st.button("Save", key="save_edu"):
                st.success("Saved")
        else:
            cols = st.columns([1, 1])
            if cols[0].button("+ Add Skill"):
                builder["skills"].append("")
            if cols[1].button("- Remove") and builder["skills"]:
                builder["skills"].pop()
            for i in range(len(builder["skills"])):
                builder["skills"][i] = st.text_input(f"Skill {i+1}", value=builder["skills"][i], key=f"skill_{i}")
            if st.button("Save", key="save_skills"):
                st.success("Saved")

    with right:
        st.header("Live Preview")
        html_str = _builder_preview_html(builder)
        components.html(html_str, height=900, scrolling=True)
        st.download_button("📥 Download PDF", data=_build_pdf_bytes(builder), file_name="resume.pdf", mime="application/pdf")


def main() -> None:
    _init_gemini_once()

    st.title(APP_TITLE)

    tool = st.sidebar.radio(
        "Tools",
        options=("🤖 AI Resume Chatbot", "📊 Bulk Resume → CSV", "📝 Resume Builder"),
    )

    if tool == "🤖 AI Resume Chatbot":
        render_ai_resume_chatbot()
    elif tool == "📊 Bulk Resume → CSV":
        render_bulk_extractor()
    else:
        render_resume_builder()


if __name__ == "__main__":
    main()