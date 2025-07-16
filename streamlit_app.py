# streamlit_app.py (Upgraded)
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

# ----------------------
# Helper Functions
# ----------------------
def extract_text(file):
    if file.name.endswith(".pdf"):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = "\n".join([page.get_text() for page in doc])
    else:
        text = file.read().decode("utf-8")
    return text

def extract_skills(text):
    skills = [
        'python', 'java', 'c++', 'sql', 'machine learning', 'deep learning', 'nlp', 'data analysis', 'tensorflow',
        'pandas', 'numpy', 'scikit-learn', 'keras', 'matplotlib', 'statistics', 'aws', 'docker', 'flask',
        'django', 'git', 'github', 'azure', 'data visualization', 'communication', 'problem solving', 'leadership'
    ]
    text = text.lower()
    found_skills = [skill for skill in skills if skill in text]
    return found_skills

# ----------------------
# Load Predefined JDs
# ----------------------
jd_data = {
    "Data Scientist": open("job_descriptions/data_scientist.txt", encoding='utf-8').read(),
    "ML Engineer": open("job_descriptions/ml_engineer.txt", encoding='utf-8').read(),
    "HR Manager": open("job_descriptions/hr_manager.txt", encoding='utf-8').read(),
    "Software Developer": open("job_descriptions/software_developer.txt", encoding='utf-8').read(),
    "Web Developer": open("job_descriptions/web_developer.txt", encoding='utf-8').read(),
    "Data Analyst": open("job_descriptions/data_analyst.txt", encoding='utf-8').read(),
    "Android Developer": open("job_descriptions/android_developer.txt", encoding='utf-8').read(),
    "UI/UX Designer": open("job_descriptions/ui_ux_designer.txt", encoding='utf-8').read()
}


# ----------------------
# Load BERT model
# ----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Resume Matcher AI", layout="wide")
st.title("🧠 AI-Powered Resume Screener")
st.write("Upload resumes and match them to a job description using semantic + skill-based matching.")

uploaded_files = st.file_uploader("Upload resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
selected_jd_title = st.selectbox("Select a Job Description to Match", list(jd_data.keys()))

if st.button("🔍 Match Resumes"):
    if not uploaded_files:
        st.warning("Please upload at least one resume file.")
    else:
        jd_text = jd_data[selected_jd_title]
        jd_embedding = model.encode([jd_text])[0]
        jd_skills = extract_skills(jd_text)

        results = []

        for uploaded in uploaded_files:
            name = uploaded.name
            try:
                resume_text = extract_text(uploaded)
                resume_embedding = model.encode([resume_text])[0]
                semantic_score = cosine_similarity([jd_embedding], [resume_embedding])[0][0] * 100

                resume_skills = extract_skills(resume_text)
                skill_overlap = len(set(resume_skills) & set(jd_skills))
                skill_score = (skill_overlap / len(jd_skills)) * 100 if jd_skills else 0

                final_score = round((0.6 * semantic_score) + (0.4 * skill_score), 2)
                results.append((name, final_score, ", ".join(resume_skills)))
            except Exception as e:
                results.append((name, f"Error: {e}", "N/A"))

        results = sorted(results, key=lambda x: float(x[1]) if isinstance(x[1], float) else 0, reverse=True)

        st.subheader(f"📄 Matching Results for: {selected_jd_title}")
        df = pd.DataFrame(results, columns=["Resume", "Match %", "Extracted Skills"])
        st.dataframe(df)

        valid_results = [r for r in results if isinstance(r[1], float)]
        if valid_results:
            st.subheader("📊 Visual Match Overview")
            chart_df = pd.DataFrame(valid_results[:10], columns=["Resume", "Match %", "_"])
            st.bar_chart(chart_df.set_index("Resume"))