import glob
import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9+#\s]", " ", text)  # keeps c++, c#


with open("job_description.txt", "r", encoding="utf-8") as f:
    job_desc = clean_text(f.read())

paths = sorted(glob.glob("resumes/*.pdf"))
if not paths:
    raise SystemExit("No PDFs found in resumes/")

resume_texts = [clean_text(extract_text(p)) for p in paths]

# Fit once on JD + all resumes so scores are comparable
vectorizer = TfidfVectorizer(
    stop_words="english",
    token_pattern=r"[a-z0-9+#]+",
    ngram_range=(1, 2),
)
vectors = vectorizer.fit_transform([job_desc] + resume_texts)

scores = cosine_similarity(vectors[0], vectors[1:])[0]
results = sorted(zip(paths, scores), key=lambda x: x[1], reverse=True)

print("\nResume ranking (TF-IDF cosine similarity):\n")
for path, score in results:
    print(f"{path}  ->  {score:.3f}")