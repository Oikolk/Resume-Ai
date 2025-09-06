# ResumeFix AI

Two-tool AI resume analyzer built with Streamlit + Gemini 2.5 Flash.

## Tools
- **🤖 AI Resume Chatbot**: Upload one PDF resume and get a concise review (score, strengths, weaknesses, quick fixes, ATS check).
- **📊 Bulk Resume → CSV**: Upload multiple PDF resumes and export a CSV with columns: Name, Email, Skills, Role, Experience.
- **📝 Resume Builder**: Left-side form with AI suggestions (summary/experience/projects) and right-side live preview with PDF download.

## Setup
1. Ensure Python 3.9+ is installed.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create an environment file:
   - Copy `env.example` to `.env` and set your key.
```bash
# .env
GEMINI_API_KEY=your_key_here
```
   Get an API key: https://aistudio.google.com/app/apikey
4. Run the app:
```bash
streamlit run app.py
```

## Notes
- Only PDF resumes are supported in Chatbot/CSV tools.
- If text extraction fails or the AI call errors, the app shows an error.
- `GEMINI_MODEL` env var can override the model (defaults to `gemini-2.5-flash`).
